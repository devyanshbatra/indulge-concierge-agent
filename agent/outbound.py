"""Writes every message a vendor receives, from ticket fields only.

The model's draft is never forwarded. Vendors get what the house rule allows:
venue, date, time, number of guests, the occasion in general words, seating and
dietary needs, and the house booking name and contacts.
"""
from __future__ import annotations

import re
from datetime import date

from world.parse import fmt_date, fmt_time

# (pattern in the member's words, what the vendor is told)
OCCASIONS = [
    (r"birthday|b'?day", "a birthday"),
    (r"anniversary", "an anniversary"),
    (r"engagement", "an engagement"),
    (r"date night|romantic", "a date night"),
    (r"business|client dinner|client lunch", "a business meal"),
    (r"farewell", "a farewell"),
    (r"graduat", "a graduation"),
    (r"celebrat", "a celebration"),
]
SEATING = [
    (r"corner table", "a corner table"),
    (r"window", "a window table"),
    (r"sea[- ]?view|with a view|\bview\b", "a table with a view"),
    (r"outdoor|outside|al fresco", "outdoor seating"),
    (r"\bquiet\b", "a quiet table"),
    (r"private (?:room|dining)", "a private room"),
    (r"rooftop", "rooftop seating"),
]
DIETARY = [
    (r"\bvegan\b", "vegan options"),
    (r"vegetarian|pure veg\b", "vegetarian options"),
    (r"\bjain\b", "Jain"),
    (r"gluten", "gluten-free"),
    (r"shellfish", "no shellfish"),
    (r"\bnuts?\b|peanut", "nut allergy"),
    (r"halal", "halal"),
    (r"dairy|lactose", "dairy-free"),
]


def _matches(table, text: str) -> list[str]:
    return [label for pattern, label in table if re.search(pattern, text, re.I)]


def extras(request_text: str, preferences: list[str]) -> list[str]:
    """Allowed extra lines: occasion and seating from the request, dietary needs from it and the profile."""
    lines = []
    if occasion := _matches(OCCASIONS, request_text)[:1]:
        lines.append(f"Occasion: {occasion[0]}.")
    if seating := _matches(SEATING, request_text):
        lines.append(f"Seating: {', '.join(seating)}, if possible.")
    dietary = list(dict.fromkeys(_matches(DIETARY, request_text) + _matches(DIETARY, " / ".join(preferences))))
    if dietary:
        lines.append(f"Dietary: {', '.join(dietary)}.")
    return lines


def _booking(ticket, house: dict) -> str:
    when = fmt_date(date.fromisoformat(ticket.date))
    return (f"a table for {ticket.party_size} at {ticket.venue} on {when} at {fmt_time(ticket.time)}, "
            f"under the name {house['booking_name']}")


def job_message(ticket, house: dict, extra_lines: list[str]) -> str:
    lines = [f"Please book {_booking(ticket, house)}.", *extra_lines,
             f"Please confirm on this thread. House contact: {house['house_phone']}, {house['house_email']}."]
    return "\n".join(lines)


def chase_message(ticket, house: dict) -> str:
    return f"Following up on our request for {_booking(ticket, house)}. Could you let us know if it is booked?"


def cancel_message(ticket, house: dict) -> str:
    return f"Please cancel our request for {_booking(ticket, house)}. Kindly reply to confirm the cancellation."


def held_back(draft: str, source_text: str, scanner) -> list[str]:
    """What the vendor will not see, in plain words, for the approver."""
    items = list(dict.fromkeys(scanner.scan(draft) + scanner.scan(source_text)))
    if draft.strip():
        items.append("the model's own wording (vendors only get the standard house message)")
    return items
