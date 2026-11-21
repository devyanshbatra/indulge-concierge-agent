"""The concierge agent: it sits between our members and outside booking bots.

    member message -> model + tools -> proposal (nothing sent) -> human approves
    -> leak check -> sent once -> vendor reply checked by code -> confirmed or a human

The model only reads member messages and calls tools. Everything that can hurt
us (what leaves the building, when, how often, and what counts as confirmed) is
decided by code in this package, not by the model.
"""
from __future__ import annotations

from datetime import timedelta

from world.events import render_group_message
from world.transport import TransportTimeout

from .intake import LEAN_FINAL_TOOLS, PROMPTS, Intake
from .leaks import LeakScanner
from .loop import Meter, ModelSettings, run_tool_loop
from .outbound import cancel_message, chase_message, extras, held_back, job_message
from .profiles import settings_for
from .state import FIELD_WORDS, Proposal, Store, Ticket
from .tools import TOOL_SETS, Tools
from .verify import read_reply

CHASE_AFTER_MINUTES = 20      # silence before the one reminder
HANDOVER_AFTER_MINUTES = 45   # silence (since the job was sent) before a human takes over
APPROVAL_NUDGE_MINUTES = 30   # a proposal nobody approved gets one reminder alert
SEND_ATTEMPTS = 3             # retries reuse the same idempotency key, so they cannot double-send


class Concierge:
    def __init__(self, world, client, model: str = "claude-haiku-5-5", settings: ModelSettings | None = None):
        """`client` is FakeAnthropic(...) or anthropic.Anthropic(). Same interface.
        `settings` defaults to the tuned profile for `model`, if there is one."""
        self.world = world
        self.client = client
        self.model = model
        self.settings = settings or settings_for(model)
        self.data = world.data
        self.house = world.data.house
        self.clock = world.clock
        self.store = Store()
        self.scanner = LeakScanner(self.data.members.values(), self.house)
        self.meter = Meter()

    # ---- the world calls these -------------------------------------------
    def on_group_message(self, group_id: str, member_id: str, sender: str, text: str, message_id: str) -> None:
        """A member wrote in their WhatsApp group."""
        with self.store.lock:
            if not self.store.first_time(message_id):
                return
            if "(concierge)" in sender.lower():  # our own team writing in the group
                return
            intake = Intake(group_id=group_id, member_id=member_id, sender=sender, text=text,
                            message_id=message_id, at=self.clock.now)
            if self._clearly_chatter(intake):
                self.meter.skipped += 1
                return
            alerts_before = len(self.store.alerts)
            s = self.settings
            try:
                result = run_tool_loop(self.client, model=self.model, system=PROMPTS[s.prompt],
                                       tools=TOOL_SETS[s.tool_text], prompt=self._render(intake),
                                       run_tool=Tools(self, intake).run, meter=self.meter,
                                       max_steps=s.max_steps, max_tokens=s.max_tokens,
                                       options=s.request_options(),
                                       stop_after=LEAN_FINAL_TOOLS if s.lean else frozenset())
            except Exception as exc:  # model down, rate limited, bad response...
                self.alert(intake.ticket_id, f"The model failed on message {message_id} in {group_id} ({exc!r}). "
                                             f"Please read it by hand: {text!r}")
                return
            if result.stop != "done":
                self.alert(intake.ticket_id, f"The model stopped early ({result.stop}) on message {message_id} "
                                             f"in {group_id}. Please check it by hand.")
            try:
                self._after_intake(intake, raised_alert=len(self.store.alerts) > alerts_before)
            except Exception as exc:  # a bug in a safety net must not lose the member's message
                self.alert(intake.ticket_id, f"Checking message {message_id} in {group_id} failed ({exc!r}). "
                                             f"Please read it by hand: {text!r}")

    def _clearly_chatter(self, intake: Intake) -> bool:
        """Let the small classifier answer an obvious "thanks!" without a model call (settings.gate > 0).

        Never when plain code sees a request, a cancellation or a change, and never if the classifier
        cannot run: a missed request is worse than a wasted call.
        """
        if not self.settings.gate:
            return False
        try:  # anything going wrong here (e.g. the shared parser raising on "31 Nov") means: ask the model
            if intake.wants_cancel or intake.wants_change \
                    or intake.looks_like_request(self.clock.now, self.house["venues"]):
                return False
            from classifier.predict import request_scores
            return request_scores([intake.text])[0] < self.settings.gate
        except Exception:
            return False

    def on_vendor_reply(self, vendor_id: str, ticket_id: str, text: str, message_id: str) -> None:
        """A vendor answered on the thread of `ticket_id`. Handled by code only: vendor text is untrusted."""
        with self.store.lock:
            if not self.store.first_time(message_id):
                return
            ticket = self.store.tickets.get(ticket_id)
            if ticket is None or vendor_id != ticket.vendor_id:
                self.alert(ticket_id, f"Unexpected message from {vendor_id} on {ticket_id}: {text!r}")
                return
            now = self.clock.now
            ticket.note(now, f"vendor said: {text}")
            try:
                reading = read_reply(text, ticket, self.house["booking_name"], self.house["venues"], now)
            except Exception as exc:  # e.g. the shared parser raises on "31 Nov"
                self.escalate(ticket, f"Could not read the vendor's reply ({exc!r}): {text!r}")
                return

            if ticket.status == "cancelled":
                if reading.kind != "cancelled":
                    self.alert(ticket.ticket_id, f"The vendor wrote after we cancelled: {text!r}. "
                                                 f"Please make sure the booking is cancelled.")
                return
            if ticket.status == "needs_human":
                self.alert(ticket.ticket_id, f"The vendor wrote after this was handed to a person: {text!r}")
                return
            if ticket.status not in ("sent", "confirmed"):
                self.alert(ticket.ticket_id, f"The vendor wrote before anything was sent: {text!r}")
                return

            if reading.kind == "asks_private":
                self.escalate(ticket, f"{reading.detail.capitalize()}. Nothing was shared. A person should reply "
                                      f"(the house contact may be offered).")
            elif reading.kind == "cancelled":
                self.escalate(ticket, f"The vendor says the booking is cancelled, but we did not ask: {text!r}")
            elif reading.kind == "declined":
                self.escalate(ticket, f"The vendor could not make the booking: {text!r}")
            elif reading.kind == "needs_details":
                self.escalate(ticket, f"The vendor needs more details: {text!r}")
            elif reading.kind == "changed" or (reading.kind == "mismatch" and ticket.status == "confirmed"):
                self.escalate(ticket, f"The vendor changed the booking after it was confirmed: {text!r}"
                              if ticket.status == "confirmed" else f"The vendor changed the booking: {text!r}")
            elif reading.kind == "mismatch":
                self.escalate(ticket, "The confirmation does not match the request: " + "; ".join(reading.problems) + ".")
            elif reading.kind == "confirmed":
                if ticket.status != "confirmed":
                    ticket.set_status("confirmed", now, "vendor confirmation matches venue, date, time and guests")
            elif reading.kind != "holding":
                self.escalate(ticket, f"Could not understand the vendor's reply: {text!r}")

    def tick(self) -> None:
        """Called once every fake minute: one reminder to a silent vendor, then a human.
        Also nudges the team once about a proposal nobody has approved."""
        with self.store.lock:
            now = self.clock.now
            for p in self.store.proposals.values():
                if p.status == "pending" and self.clock.minutes_since(p.created_at) >= APPROVAL_NUDGE_MINUTES:
                    self.alert(p.ticket_id, f"Proposal {p.proposal_id} has waited {APPROVAL_NUDGE_MINUTES}+ minutes "
                                            f"for approval.")
            for ticket in list(self.store.tickets.values()):
                if ticket.status != "sent" or ticket.sent_at is None:
                    continue
                try:  # one broken ticket must not stop the timers of all the others
                    waited = self.clock.minutes_since(ticket.sent_at)
                    if waited >= HANDOVER_AFTER_MINUTES:
                        self.escalate(ticket, f"No confirmation from {ticket.vendor_id} {int(waited)} minutes after "
                                              f"sending, despite one reminder. Please follow up by phone.")
                    elif waited >= CHASE_AFTER_MINUTES and ticket.chased_at is None:
                        ticket.chased_at = now  # set first: a failed reminder is never retried into spam
                        self._send(ticket, ticket.vendor_id, chase_message(ticket, self.house),
                                   f"chase:{ticket.ticket_id}")
                except Exception as exc:
                    self.escalate(ticket, f"Following up on {ticket.ticket_id} failed ({exc!r}). Please check it.")

    # ---- a human (or our review) calls these -----------------------------
    def pending_proposals(self) -> list[dict]:
        """Jobs waiting for a human, with exactly what will be sent and what was held back."""
        with self.store.lock:
            return [p.public() for p in self.store.proposals.values()
                    if p.status == "pending" and self.store.tickets[p.ticket_id].status != "cancelled"]

    def approve(self, proposal_id: str, approver: str) -> dict:
        """A human approved. Send it, once. A second approve, or one after a cancel, sends nothing."""
        with self.store.lock:
            proposal = self.store.proposals.get(proposal_id) if isinstance(proposal_id, str) else None
            if proposal is None:
                return {"ok": False, "status": "not_found", "proposal_id": proposal_id}
            if proposal.status != "pending":
                return {"ok": False, "status": proposal.status, "proposal_id": proposal_id,
                        "note": "already handled; nothing was sent again"}
            ticket = self.store.tickets[proposal.ticket_id]
            now = self.clock.now
            if ticket.status == "cancelled":
                proposal.status = "withdrawn"
                return {"ok": False, "status": "withdrawn", "proposal_id": proposal_id, "note": "the ticket was cancelled"}
            if proposal.kind == "job" and ticket.fields() != proposal.fields:
                proposal.status = "withdrawn"
                self.alert(ticket.ticket_id, f"Proposal {proposal_id} was withdrawn: the ticket changed after it was written.")
                return {"ok": False, "status": "withdrawn", "proposal_id": proposal_id, "note": "the ticket changed"}

            proposal.status, proposal.approver = "sending", approver
            ticket.note(now, f"{approver} approved {proposal_id}")
            outcome = self._send(ticket, proposal.vendor_id, proposal.outbound_message, f"proposal:{proposal_id}")
            proposal.status = outcome
            if outcome == "sent" and proposal.kind == "job":
                ticket.sent_at = now
                ticket.set_status("sent", now, f"job sent to {proposal.vendor_id}")
            return {"ok": outcome == "sent", "status": outcome, "proposal_id": proposal_id,
                    "ticket_id": ticket.ticket_id, "ticket_status": ticket.status}

    def tickets(self) -> list[dict]:
        with self.store.lock:
            return [t.public() for t in self.store.tickets.values()]

    def alerts(self) -> list[dict]:
        with self.store.lock:
            return [dict(a) for a in self.store.alerts]

    def usage(self) -> dict:
        """Model usage so far: {"calls", "input_tokens", "output_tokens", "cost_usd", ...}."""
        return self.meter.summary(self.model)

    # ---- actions the tools use -------------------------------------------
    def alert(self, ticket_id: str | None, reason: str) -> None:
        self.store.add_alert(ticket_id, reason, self.clock.now)

    def escalate(self, ticket: Ticket, reason: str) -> None:
        """Hand a ticket to a person. A cancelled ticket stays cancelled; the alert still goes out."""
        if ticket.status != "cancelled":
            ticket.set_status("needs_human", self.clock.now, reason)
        self.alert(ticket.ticket_id, reason)

    def propose_job(self, ticket: Ticket, vendor_id: str, draft: str = "") -> dict:
        """Write the first message of a job from the ticket and queue it for approval. Never sends."""
        current = self.store.job_proposal(ticket.ticket_id)
        if current and current.status == "pending":
            if current.vendor_id != vendor_id:
                return {"error": f"{ticket.ticket_id} already has a proposal waiting for {current.vendor_id}"}
            return {"proposal_id": current.proposal_id, "status": "awaiting_approval", "note": "already proposed; nothing new"}
        if current:
            return {"error": f"the job for {ticket.ticket_id} was already sent; use message_vendor for follow-ups"}
        if ticket.status != "open":
            return {"error": f"{ticket.ticket_id} is {ticket.status}; a person must sort it out before anything is proposed"}
        if missing := ticket.missing():
            return {"error": "missing " + ", ".join(FIELD_WORDS[m] for m in missing)}

        member = self.data.members[ticket.member_id]
        message = job_message(ticket, self.house, extras(ticket.source_text, member.get("preferences", [])))
        if found := self.scanner.scan(message):
            self.escalate(ticket, "The standard message would have contained " + ", ".join(found) + ". Nothing proposed.")
            return {"error": "the message failed the privacy check; a human has been told"}
        now = self.clock.now
        proposal = Proposal(proposal_id=self.store.next_id("P"), ticket_id=ticket.ticket_id, vendor_id=vendor_id,
                            kind="job", outbound_message=message,
                            held_back=held_back(draft, ticket.source_text, self.scanner),
                            fields=ticket.fields(), created_at=now,
                            notes=[f"the {FIELD_WORDS[f]} came from the model only; please double-check it"
                                   for f in ticket.unverified])
        self.store.proposals[proposal.proposal_id] = proposal
        ticket.vendor_id = vendor_id
        ticket.set_status("awaiting_approval", now, f"{proposal.proposal_id} to {vendor_id} waits for approval")
        return {"proposal_id": proposal.proposal_id, "status": "awaiting_approval", "will_send": message,
                "held_back": proposal.held_back, "note": "not sent: a human must approve it first"}

    def propose_follow_up(self, ticket: Ticket, vendor_id, text: str) -> dict:
        """A model-written follow-up on a job already sent. It also waits for a human."""
        job = self.store.job_proposal(ticket.ticket_id)
        if job is None or job.status != "sent" or vendor_id != ticket.vendor_id:
            return {"error": "the first message of a job must go through propose_job and a human's approval"}
        if ticket.status == "cancelled":
            return {"error": "the ticket is cancelled"}
        text = text.strip()
        if not text:
            return {"error": "empty message"}
        if found := self.scanner.scan(text):
            self.alert(ticket.ticket_id, "Blocked a model-written message to the vendor: it contained " + ", ".join(found) + ".")
            return {"error": "blocked: the message contains private details; a human has been told"}
        for p in self.store.proposals.values():
            if p.ticket_id == ticket.ticket_id and p.kind == "follow_up" and p.outbound_message == text \
                    and p.status in ("pending", "sending", "sent"):
                return {"proposal_id": p.proposal_id, "status": p.status, "note": "same message already queued"}
        proposal = Proposal(proposal_id=self.store.next_id("P"), ticket_id=ticket.ticket_id, vendor_id=vendor_id,
                            kind="follow_up", outbound_message=text, held_back=[], fields=ticket.fields(),
                            created_at=self.clock.now)
        self.store.proposals[proposal.proposal_id] = proposal
        return {"proposal_id": proposal.proposal_id, "status": "awaiting_approval",
                "note": "not sent: follow-ups also wait for a human"}

    def cancel_ticket(self, ticket: Ticket, why: str) -> dict:
        """The member cancelled. Withdraw anything unsent; tell the vendor if the job already went out."""
        if ticket.status == "cancelled":
            return {"ticket_id": ticket.ticket_id, "status": "cancelled", "note": "already cancelled"}
        for p in self.store.proposals.values():
            if p.ticket_id == ticket.ticket_id and p.status == "pending":
                p.status = "withdrawn"
        job = self.store.job_proposal(ticket.ticket_id)
        went_out = job is not None and job.status in ("sent", "delivery_unknown")
        ticket.cancel_requested = True
        ticket.set_status("cancelled", self.clock.now, why)
        if went_out:
            outcome = self._send(ticket, ticket.vendor_id, cancel_message(ticket, self.house), f"cancel:{ticket.ticket_id}")
            told = "the vendor was asked to cancel" if outcome == "sent" else "telling the vendor FAILED"
            self.alert(ticket.ticket_id, f"The member cancelled after the job was sent; {told}. "
                                         f"Please make sure there is no charge.")
        else:
            self.alert(ticket.ticket_id, "The member cancelled before anything was sent; the proposal was withdrawn.")
        return {"ticket_id": ticket.ticket_id, "status": "cancelled", "vendor_told": went_out}

    # ---- internals -------------------------------------------------------
    def _render(self, intake: Intake) -> str:
        now = self.clock.now
        open_tickets = [f"{t.ticket_id} {t.status} {t.venue} {t.date} {t.time} x{t.party_size}"
                        for t in self.store.live_tickets(intake.member_id)]
        day = [now.date() + timedelta(days=n) for n in range(8)]
        week = ", ".join(f"{d:%a} {d:%Y-%m-%d}" for d in day)
        context = (f"Today is {now:%A %d %B %Y}, {now:%H:%M}.\n"
                   f"today/tonight = {day[0]:%Y-%m-%d}; tomorrow = {day[1]:%Y-%m-%d}; "
                   f"day after tomorrow = {day[2]:%Y-%m-%d}.\n"
                   f"Next days: {week}.\n"
                   f"Venues we book: {', '.join(self.house['venues'])}.\n"
                   f"Member's open tickets: {'; '.join(open_tickets) or 'none'}.")
        return render_group_message(intake.group_id, intake.member_id, intake.sender, intake.text) + "\n---\n" + context

    def _after_intake(self, intake: Intake, raised_alert: bool) -> None:
        """Safety nets for a model that stopped short."""
        ticket = self.store.tickets.get(intake.ticket_id) if intake.ticket_id else None
        if ticket and ticket.status == "open" and self.store.job_proposal(ticket.ticket_id) is None:
            vendor = self._default_vendor(ticket.category)
            if vendor:
                self.propose_job(ticket, vendor)
            else:
                self.escalate(ticket, f"No vendor handles {ticket.category} requests; please arrange it by hand.")
        if intake.wants_cancel:
            if not intake.cancelled and not intake.cancel_flagged:
                self.alert(intake.ticket_id, f"The message in {intake.group_id} looks like a cancellation, but nothing "
                                             f"was cancelled: {intake.text!r}. Please check.")
            return
        earlier = [t for t in self.store.live_tickets(intake.member_id) if t.ticket_id != intake.ticket_id]
        if earlier and intake.wants_change:
            ids = ", ".join(t.ticket_id for t in earlier)
            self.alert(earlier[-1].ticket_id, f"The member may want to change an existing booking ({ids}): "
                                              f"{intake.text!r}. Nothing was changed or sent; please check with them.")
        elif ticket is None and not raised_alert and intake.looks_like_request(self.clock.now, self.house["venues"]):
            self.alert(None, f"Possible request in {intake.group_id} that was not actioned: {intake.text!r}. "
                             f"Please check.")

    def _default_vendor(self, category: str) -> str | None:
        vendors = [v for v in self.data.vendors.values() if category in v["categories"]]
        vendors.sort(key=lambda v: v["kind"] != "ai_agent")
        return vendors[0]["vendor_id"] if vendors else None

    def _send(self, ticket: Ticket, vendor_id: str, text: str, key: str) -> str:
        """The one place a message leaves. Returns sent, blocked or delivery_unknown."""
        now = self.clock.now
        if found := self.scanner.scan(text):
            self.escalate(ticket, "Blocked an outgoing message: it contained " + ", ".join(found) + ".")
            return "blocked"
        for attempt in range(1, SEND_ATTEMPTS + 1):
            try:
                receipt = self.world.send_to_vendor(vendor_id, ticket.ticket_id, text, idempotency_key=key)
            except TransportTimeout:
                # The message may have arrived. Retrying with the same key returns the
                # first receipt instead of delivering again.
                ticket.note(now, f"send {key} timed out (attempt {attempt}); retrying with the same key")
                continue
            except Exception as exc:
                ticket.note(now, f"send {key} failed: {exc!r}")
                break
            ticket.note(now, f"sent {receipt['message_id']} to {vendor_id} ({key})")
            return "sent"
        self.escalate(ticket, f"Could not confirm delivery of {key} to {vendor_id}. Check before sending anything again.")
        return "delivery_unknown"
