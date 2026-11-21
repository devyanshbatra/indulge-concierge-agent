"""The eight tools the model may call, and the checks behind each one.

The model chooses which tool to call; this code decides what is allowed. Every
result is a JSON object, and a refusal is {"error": ...}.
"""
from __future__ import annotations

from .intake import Intake, check_category, normalise_category, resolve_fields
from .state import FIELD_WORDS, STATUSES, Ticket


def _schema(properties: dict) -> dict:
    return {"type": "object", "properties": properties, "required": list(properties)}


def _text(description: str) -> dict:
    return {"type": "string", "description": description}


def _maybe(kind: str, description: str) -> dict:
    return {"type": [kind, "null"], "description": description}


TOOLS = [
    {"name": "get_member",
     "description": "Look up the member who wrote the message: preferences and open tickets. "
                    "Personal details are withheld by design.",
     "input_schema": _schema({"member_id": _text("The member_id from the event header.")})},
    {"name": "create_ticket",
     "description": "Open a ticket for a member's request. Use null for anything the member did not say; never guess.",
     "input_schema": _schema({
         "member_id": _text("The member_id from the event header."),
         "category": _text("dining, transport, flowers, travel or other."),
         "request_summary": _text("One line describing the request, for our team only."),
         "venue": _maybe("string", "Restaurant name, or null."),
         "date": _maybe("string", "YYYY-MM-DD, or null."),
         "time": _maybe("string", "24-hour HH:MM, or null. 8:30 pm is 20:30; 1 pm is 13:00; "
                                  "'9 baje' for dinner is 21:00."),
         "party_size": _maybe("integer", "Number of guests, or null."),
     })},
    {"name": "list_open_tickets",
     "description": "List the member's tickets that are not cancelled.",
     "input_schema": _schema({"member_id": _text("The member_id from the event header.")})},
    {"name": "find_vendors",
     "description": "List the vendors that handle a category.",
     "input_schema": _schema({"category": _text("dining, transport, flowers, travel or other.")})},
    {"name": "propose_job",
     "description": "Prepare the first message to a vendor for a ticket. It is NOT sent: a human approves it, "
                    "and the system writes the final wording from the ticket.",
     "input_schema": _schema({"ticket_id": _text("Ticket id."), "vendor_id": _text("Vendor id."),
                              "message": _text("Your draft of the request.")})},
    {"name": "message_vendor",
     "description": "Draft a follow-up to the vendor of a job that was already sent. A human approves it before it goes.",
     "input_schema": _schema({"ticket_id": _text("Ticket id."), "vendor_id": _text("Vendor id."),
                              "message": _text("The follow-up text.")})},
    {"name": "update_ticket",
     "description": "Change a ticket's status to cancelled (only when the member asks), needs_info or needs_human. "
                    "Other statuses are set by the system after its own checks.",
     "input_schema": _schema({"ticket_id": _text("Ticket id."),
                              "status": {"type": "string", "enum": list(STATUSES)},
                              "note": _text("Why.")})},
    {"name": "notify_human",
     "description": "Ask a person on our team to look at something.",
     "input_schema": _schema({"ticket_id": _maybe("string", "Ticket id, or null."), "reason": _text("What to look at.")})},
]
TOOL_NAMES = {tool["name"] for tool in TOOLS}


class Tools:
    """Runs tool calls for one member message. `desk` is the Concierge."""

    def __init__(self, desk, intake: Intake):
        self.desk = desk
        self.intake = intake

    def run(self, name: str, args: dict) -> dict:
        if name not in TOOL_NAMES:
            return {"error": f"unknown tool {name!r}"}
        if not isinstance(args, dict):
            return {"error": "arguments must be a JSON object"}
        try:
            return getattr(self, f"_{name}")(args)
        except Exception as exc:  # a bug in one tool must not lose the member's message
            self.desk.alert(self.intake.ticket_id, f"Tool {name} failed ({exc!r}) on message {self.intake.message_id}.")
            return {"error": f"{name} failed inside the system; a human has been told"}

    # ---- helpers -------------------------------------------------------
    def _wrong_member(self, member_id) -> dict | None:
        if member_id != self.intake.member_id:
            return {"error": "you can only act for the member who wrote this message"}
        return None

    def _ticket(self, ticket_id) -> Ticket | None:
        ticket = self.desk.store.tickets.get(ticket_id)
        return ticket if ticket and ticket.member_id == self.intake.member_id else None

    @staticmethod
    def _brief(ticket: Ticket) -> dict:
        return {"ticket_id": ticket.ticket_id, "status": ticket.status, "category": ticket.category, **ticket.fields()}

    # ---- the tools -----------------------------------------------------
    def _get_member(self, args: dict) -> dict:
        if error := self._wrong_member(args.get("member_id")):
            return error
        member = self.desk.data.members[self.intake.member_id]
        return {"member_id": member["member_id"],
                "preferences": member.get("preferences", []),
                "booking_name": self.desk.house["booking_name"],
                "open_tickets": [self._brief(t) for t in self.desk.store.live_tickets(member["member_id"])],
                "withheld": "name, phone, email, address, card and family details are never shared"}

    def _create_ticket(self, args: dict) -> dict:
        if error := self._wrong_member(args.get("member_id")):
            return error
        store, now = self.desk.store, self.desk.clock.now
        if self.intake.ticket_id:
            existing = store.tickets[self.intake.ticket_id]
            return {"ticket_id": existing.ticket_id, "status": existing.status, "note": "this message already has a ticket"}

        fields, problems, unverified = resolve_fields(args, self.intake.text, now, self.desk.house["venues"])
        if all(fields.values()):
            for open_ticket in store.live_tickets(self.intake.member_id):
                if open_ticket.fields() == fields:
                    self.intake.ticket_id = open_ticket.ticket_id
                    open_ticket.note(now, f"same request again in message {self.intake.message_id}; nothing new opened")
                    return {"ticket_id": open_ticket.ticket_id, "status": open_ticket.status, "duplicate": True}

        ticket = Ticket(ticket_id=store.next_id("T"), member_id=self.intake.member_id,
                        category=check_category(normalise_category(args.get("category")), self.intake.text,
                                                self.desk.house["venues"]),
                        summary=str(args.get("request_summary") or "")[:300], source_text=self.intake.text,
                        created_at=now, unverified=unverified, **fields)
        store.tickets[ticket.ticket_id] = ticket
        self.intake.ticket_id = ticket.ticket_id
        ticket.note(now, f"opened from message {self.intake.message_id}")

        if ticket.category != "dining":
            self.desk.escalate(ticket, f"A {ticket.category} request: only dining bookings are prepared "
                                       f"automatically, so a person should arrange it.")
        elif problems:
            self.desk.escalate(ticket, "Could not read the request safely: " + "; ".join(problems) + ".")
        elif missing := ticket.missing():
            words = ", ".join(FIELD_WORDS[m] for m in missing)
            ticket.set_status("needs_info", now, f"missing {words}")
            self.desk.alert(ticket.ticket_id, f"The request is missing the {words}. Please ask the member.")
        return {"ticket_id": ticket.ticket_id, "status": ticket.status, "fields": fields,
                "missing": ticket.missing(), "problems": problems}

    def _list_open_tickets(self, args: dict) -> dict:
        if error := self._wrong_member(args.get("member_id")):
            return error
        return {"tickets": [self._brief(t) for t in self.desk.store.live_tickets(self.intake.member_id)]}

    def _find_vendors(self, args: dict) -> dict:
        category = normalise_category(args.get("category"))
        matching = [v for v in self.desk.data.vendors.values() if category in v["categories"]]
        matching.sort(key=lambda v: v["kind"] != "ai_agent")  # the fast booking agent first
        return {"vendors": [{"vendor_id": v["vendor_id"], "name": v["name"], "kind": v["kind"],
                             "notes": v.get("notes", "")} for v in matching]}

    def _propose_job(self, args: dict) -> dict:
        ticket = self._ticket(args.get("ticket_id"))
        if ticket is None:
            return {"error": "no such ticket for this member"}
        vendor = self.desk.data.vendors.get(args.get("vendor_id"))
        if vendor is None or ticket.category not in vendor["categories"]:
            return {"error": f"that vendor does not handle {ticket.category} requests"}
        return self.desk.propose_job(ticket, vendor["vendor_id"], draft=str(args.get("message") or ""))

    def _message_vendor(self, args: dict) -> dict:
        ticket = self._ticket(args.get("ticket_id"))
        if ticket is None:
            return {"error": "no such ticket for this member"}
        return self.desk.propose_follow_up(ticket, args.get("vendor_id"), str(args.get("message") or ""))

    def _update_ticket(self, args: dict) -> dict:
        ticket = self._ticket(args.get("ticket_id"))
        if ticket is None:
            return {"error": "no such ticket for this member"}
        status, note, now = args.get("status"), str(args.get("note") or "").strip(), self.desk.clock.now
        if status == ticket.status:
            ticket.note(now, f"model note: {note}" if note else "model re-confirmed the status")
            return {"ticket_id": ticket.ticket_id, "status": ticket.status}
        if status == "cancelled":
            return self._cancel(ticket)
        if ticket.status == "cancelled":
            return {"error": "the ticket is cancelled"}
        if status == "needs_human":
            self.desk.escalate(ticket, note or "the model asked for a person")
            return {"ticket_id": ticket.ticket_id, "status": ticket.status}
        if status == "needs_info":
            ticket.set_status("needs_info", now, note or "the model asked for more information")
            self.desk.alert(ticket.ticket_id, note or "More information is needed from the member.")
            return {"ticket_id": ticket.ticket_id, "status": ticket.status}
        return {"error": f"status {status!r} is set by the system after its own checks, not by the model"}

    def _cancel(self, ticket: Ticket) -> dict:
        if not self.intake.wants_cancel:
            self.desk.alert(ticket.ticket_id, "The model tried to cancel this ticket, but the member's message "
                                              "does not ask for that. Nothing was cancelled.")
            return {"error": "the member did not ask to cancel; a human has been told"}
        live = self.desk.store.live_tickets(self.intake.member_id)
        if len(live) > 1 and (not ticket.venue or ticket.venue.lower() not in self.intake.text.lower()):
            self.intake.cancel_flagged = True
            self.desk.alert(ticket.ticket_id, f"The member asked to cancel, but has {len(live)} open bookings and "
                                              f"the message does not say which one. Please ask.")
            return {"error": "more than one open booking; a human will ask the member which one"}
        self.intake.cancelled.append(ticket.ticket_id)
        return self.desk.cancel_ticket(ticket, f"member asked: {self.intake.text!r}")

    def _notify_human(self, args: dict) -> dict:
        ticket_id = args.get("ticket_id")
        if ticket_id and self._ticket(ticket_id) is None:
            return {"error": "no such ticket for this member"}
        reason = str(args.get("reason") or "").strip() or "The model asked for a person to look."
        self.desk.alert(ticket_id or self.intake.ticket_id, f"{reason} (group {self.intake.group_id})")
        return {"ok": True, "note": "a person on the team will look at it"}
