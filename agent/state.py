"""Everything the concierge remembers, kept in memory.

A real deployment would keep this in a database (see NOTES.md). Status changes go
through `Ticket.set_status`, so every change lands in the ticket's history.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime

STATUSES = ("open", "needs_info", "awaiting_approval", "sent", "confirmed", "needs_human", "cancelled")
FIELDS = ("venue", "date", "time", "party_size")
FIELD_WORDS = {"venue": "venue", "date": "date", "time": "time", "party_size": "number of guests"}


@dataclass
class Ticket:
    ticket_id: str
    member_id: str
    category: str
    summary: str          # internal only, never sent to a vendor
    source_text: str      # the member's own words, internal only
    created_at: datetime
    venue: str | None = None
    date: str | None = None       # ISO date, e.g. "2026-11-07"
    time: str | None = None       # "HH:MM", 24 hour
    party_size: int | None = None
    unverified: list[str] = field(default_factory=list)  # fields only the model read, not the parser
    status: str = "open"
    vendor_id: str | None = None
    sent_at: datetime | None = None
    chased_at: datetime | None = None
    cancel_requested: bool = False
    history: list[dict] = field(default_factory=list)

    def note(self, at: datetime, text: str) -> None:
        self.history.append({"at": at.isoformat(timespec="minutes"), "note": text})

    def set_status(self, status: str, at: datetime, why: str) -> None:
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}")
        self.note(at, f"{self.status} -> {status}: {why}" if status != self.status else why)
        self.status = status

    def fields(self) -> dict:
        return {name: getattr(self, name) for name in FIELDS}

    def missing(self) -> list[str]:
        return [name for name in FIELDS if not getattr(self, name)]

    def public(self) -> dict:
        return {"ticket_id": self.ticket_id, "member_id": self.member_id, "status": self.status,
                "category": self.category, **self.fields(), "vendor_id": self.vendor_id,
                "history": [dict(h) for h in self.history]}


@dataclass
class Proposal:
    proposal_id: str
    ticket_id: str
    vendor_id: str
    kind: str                 # "job" (the first message of a job) or "follow_up"
    outbound_message: str     # exactly what will be sent if approved
    held_back: list[str]
    fields: dict              # the ticket fields the message was written from
    created_at: datetime
    notes: list[str] = field(default_factory=list)
    status: str = "pending"   # pending, sending, sent, withdrawn, blocked, delivery_unknown
    approver: str | None = None

    def public(self) -> dict:
        return {"proposal_id": self.proposal_id, "ticket_id": self.ticket_id, "vendor_id": self.vendor_id,
                "outbound_message": self.outbound_message, "held_back": list(self.held_back),
                "kind": self.kind, "notes": list(self.notes)}


class Store:
    def __init__(self):
        self.tickets: dict[str, Ticket] = {}
        self.proposals: dict[str, Proposal] = {}
        self.alerts: list[dict] = []
        self.lock = threading.RLock()  # one event at a time; approve() and tick() share it
        self._seen: set[str] = set()   # inbound message ids already handled
        self._seq: dict[str, int] = {}

    def next_id(self, prefix: str) -> str:
        self._seq[prefix] = self._seq.get(prefix, 0) + 1
        return f"{prefix}-{self._seq[prefix]:04d}"

    def first_time(self, message_id: str) -> bool:
        """True the first time a message id is seen, False for any repeat delivery."""
        if message_id in self._seen:
            return False
        self._seen.add(message_id)
        return True

    def add_alert(self, ticket_id: str | None, reason: str, at: datetime) -> None:
        if any(a["ticket_id"] == ticket_id and a["reason"] == reason for a in self.alerts):
            return
        self.alerts.append({"ticket_id": ticket_id, "reason": reason, "at": at.isoformat(timespec="minutes")})

    def live_tickets(self, member_id: str) -> list[Ticket]:
        return [t for t in self.tickets.values() if t.member_id == member_id and t.status != "cancelled"]

    def job_proposal(self, ticket_id: str) -> Proposal | None:
        """The latest first-message proposal for a ticket that was not withdrawn."""
        jobs = [p for p in self.proposals.values()
                if p.ticket_id == ticket_id and p.kind == "job" and p.status != "withdrawn"]
        return jobs[-1] if jobs else None
