"""Reads a vendor's reply and decides what it means for the ticket.

Vendor text is untrusted, so this is plain code, not the model: a reply cannot
talk its way into a confirmation or into receiving a member's details.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime

from world.parse import fmt_date, fmt_time, parse_date, parse_party_size, parse_time, parse_venue

from .state import FIELD_WORDS, FIELDS

ASKS_PRIVATE = re.compile(
    r"mobile|phone|whats ?app|contact (?:number|details)|\bcell\b|\bcard\b|\bcvv\b|\bcvc\b|expiry|expiration"
    r"|credit|debit|\bupi\b|bank|e-?mail|address|passport|aadha?ar|\bpan\b|id proof|date of birth|\bdob\b"
    r"|\botp\b|full name|guest'?s? (?:own )?name|surname|last name|personal details", re.I)
CANCELLED = re.compile(r"\bcancell?ed\b|\bcancellation\b", re.I)
DECLINED = re.compile(r"fully booked|not available|unavailable|no (?:tables?|availability)|can(?:no|')t|could ?n[o']t"
                      r"|unable|\bsorry\b|declined?|regret", re.I)
NEEDS_DETAILS = re.compile(r"still need|need (?:the|a|more|your)|missing|please (?:provide|tell|share|send)"
                           r"|which (?:date|day|time)|what time|how many", re.I)
CHANGED = re.compile(r"\bmoved\b|\bchanged?\b|reschedul|\binstead\b|\bupdated?\b|\bshifted\b|new (?:time|date|table)", re.I)
CONFIRMED = re.compile(r"\bconfirmed\b|\bbooked\b|\breserved\b|all set|locked in|see you", re.I)
HOLDING = re.compile(r"still working|working on it|will confirm|confirm shortly|checking|get back|one moment", re.I)
BOOKED_NAME = re.compile(r"under the name\s+[\"']?([^.,;\n\"']+)", re.I)


@dataclass
class Reading:
    kind: str     # asks_private, cancelled, declined, needs_details, changed, holding, confirmed, mismatch, unclear
    detail: str = ""
    problems: list[str] = field(default_factory=list)


def read_reply(text: str, ticket, house_name: str, venues: list[str], now: datetime) -> Reading:
    if m := ASKS_PRIVATE.search(text):
        return Reading("asks_private", f"the vendor asked for personal details ({m.group(0)})")
    if CANCELLED.search(text):
        return Reading("cancelled")
    if DECLINED.search(text):
        return Reading("declined")
    if NEEDS_DETAILS.search(text):
        return Reading("needs_details")
    if CHANGED.search(text):
        return Reading("changed")
    if HOLDING.search(text) and not CONFIRMED.search(text):
        return Reading("holding")
    if CONFIRMED.search(text):
        problems = compare(text, ticket, house_name, venues, now)
        return Reading("mismatch", problems=problems) if problems else Reading("confirmed")
    return Reading("unclear")


def compare(text: str, ticket, house_name: str, venues: list[str], now: datetime) -> list[str]:
    """Differences between a vendor's confirmation and what the member asked for."""
    said_date = parse_date(text, now)
    said = {"venue": parse_venue(text, venues), "date": said_date.isoformat() if said_date else None,
            "time": parse_time(text), "party_size": parse_party_size(text)}
    wanted = ticket.fields()
    problems = []
    for name in FIELDS:
        if said[name] is None:
            problems.append(f"the confirmation does not state the {FIELD_WORDS[name]}")
        elif said[name] != wanted[name]:
            problems.append(f"{FIELD_WORDS[name]}: asked for {_show(name, wanted[name])}, vendor says {_show(name, said[name])}")
    if (m := BOOKED_NAME.search(text)) and m.group(1).strip().lower() != house_name.lower():
        problems.append(f"booked under {m.group(1).strip()!r}, not the house name")
    return problems


def _show(name: str, value) -> str:
    if name == "date":
        return fmt_date(date.fromisoformat(value))
    if name == "time":
        return fmt_time(value)
    return str(value)
