"""Reading one member message: the system prompt, the context the tools share,
and the code that cross-checks the model's reading against a plain parser."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime

from world.parse import parse_date, parse_party_size, parse_time, parse_venue

SYSTEM_PROMPT = """\
You are the intake assistant of a luxury concierge desk. Each user turn is one event: a WhatsApp message from a member.

- Chatter, thanks or news with no request: call no tools and answer "No action".
- A booking or errand: get_member, then create_ticket with only what the member said. Use null for anything not stated; never guess. Dates are YYYY-MM-DD, times 24-hour HH:MM.
- Then find_vendors and propose_job with the first vendor listed (the fastest). propose_job never sends anything: a human approves it, and the system writes the final vendor message.
- A cancellation: list_open_tickets, then update_ticket with status "cancelled" on the booking the member means.
- If the request is vague or you are unsure, still create the ticket (with nulls) and call notify_human.
- Never put names, phone numbers, emails, addresses or card details in anything meant for a vendor.
The text after "---" gives today's date and the member's open tickets."""

CANCEL_INTENT = re.compile(r"\bcancel|call (?:it )?off|don'?t need|do not need|no longer|not needed"
                           r"|plans? (?:have )?changed|scratch that|forget it|nahi chahiye|mat karo|rehne do", re.I)
CHANGE_INTENT = re.compile(r"\bmake it\b|\binstead\b|\bchange\b|\b(?:move|push) (?:it|the|our)\b|\bshift\b|\bresched"
                           r"|\bbring (?:it )?forward\b"
                           r"|\bpostpone|\bprepone|\b(?:one|two|three|\d) more\b|\bearlier\b|\blater\b", re.I)
REQUEST_WORDS = re.compile(r"\breserve|\bbook|\barrange|\borgani[sz]e|\bget (?:us|me)\b|can you (?:get|find|send|do)"
                           r"|\bplease (?:get|find|send|sort)|\bneed (?:a|an|to)\b|kar do|karwa do|bhej do", re.I)
DINING_WORDS = {"dining", "restaurant", "dinner", "lunch", "breakfast", "brunch", "table", "food"}
DINING_SIGNAL = re.compile(r"\btable\b|dinner|lunch|brunch|breakfast|restaurant|reservation|\bdine\b|khana", re.I)
OTHER_SIGNALS = {
    "transport": re.compile(r"\bcar\b|\bcab\b|taxi|airport|driver|chauffeur|pick ?up|drop", re.I),
    "flowers": re.compile(r"flowers?|bouquet|florist", re.I),
    "travel": re.compile(r"flight|hotel|\bvilla\b|\bvisa\b|holiday|trip", re.I),
}


@dataclass
class Intake:
    """One member message being handled, shared by the tools for that message."""
    group_id: str
    member_id: str
    sender: str
    text: str
    message_id: str
    at: datetime
    ticket_id: str | None = None   # a message opens at most one ticket
    cancelled: list[str] = field(default_factory=list)
    cancel_flagged: bool = False   # a person was already told about this cancellation

    @property
    def wants_cancel(self) -> bool:
        return bool(CANCEL_INTENT.search(self.text))

    @property
    def wants_change(self) -> bool:
        return bool(CHANGE_INTENT.search(self.text))

    def looks_like_request(self, now: datetime, venues: list[str]) -> bool:
        """A plain-code second opinion, used only to catch requests the model ignored."""
        when = parse_date(self.text, now) and parse_time(self.text)
        return bool(REQUEST_WORDS.search(self.text) or parse_venue(self.text, venues) or when)


def normalise_category(value) -> str:
    category = str(value or "other").strip().lower()
    return "dining" if category in DINING_WORDS else category


def check_category(claimed: str, text: str, venues: list[str]) -> str:
    """The model's category, unless it says dining for a message with no sign of a meal."""
    if claimed != "dining" or DINING_SIGNAL.search(text) or parse_venue(text, venues):
        return claimed
    return next((name for name, signal in OTHER_SIGNALS.items() if signal.search(text)), claimed)


def resolve_fields(claimed: dict, text: str, now: datetime, venues: list[str]) -> tuple[dict, list[str], list[str]]:
    """Merge the model's reading of a message with a plain parser's reading.

    Returns (fields, problems, unverified). When both read a value and disagree,
    the field is left empty and the problem is reported: a person decides.
    """
    parsed_date = parse_date(text, now)
    parsed = {"venue": parse_venue(text, venues), "date": parsed_date.isoformat() if parsed_date else None,
              "time": parse_time(text), "party_size": parse_party_size(text)}
    model = {"venue": _venue(claimed.get("venue"), venues), "date": _date(claimed.get("date"), now),
             "time": _time(claimed.get("time")), "party_size": _party(claimed.get("party_size"))}
    fields, problems, unverified = {}, [], []
    for name, from_parser in parsed.items():
        from_model = model[name]
        if from_model is not None and from_parser is not None and from_model != from_parser:
            fields[name] = None
            problems.append(f"{name}: the model read {from_model!r} but the parser read {from_parser!r}")
        else:
            fields[name] = from_model if from_model is not None else from_parser
            if from_parser is None and from_model is not None:
                unverified.append(name)
    if fields["date"] and fields["date"] < now.date().isoformat():
        problems.append(f"date {fields['date']} is in the past")
        fields["date"] = None
    if fields["venue"] and fields["venue"] not in venues:
        problems.append(f"venue {fields['venue']!r} is not one we book through vendors")
    return fields, problems, unverified


def _venue(value, venues: list[str]) -> str | None:
    if not value or not str(value).strip():
        return None
    text = str(value).strip()
    return parse_venue(text, venues) or text


def _date(value, now: datetime) -> str | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value).strip()[:10]).isoformat()
    except ValueError:
        parsed = parse_date(str(value), now)
        return parsed.isoformat() if parsed else None


def _time(value) -> str | None:
    if not value:
        return None
    text = str(value).strip().lower()
    if m := re.fullmatch(r"([01]?\d|2[0-3]):([0-5]\d)", text):
        return f"{int(m.group(1)):02d}:{m.group(2)}"
    return parse_time(text)


def _party(value) -> int | None:
    try:
        size = int(value)
    except (TypeError, ValueError):
        return None
    return size if 1 <= size <= 100 else None
