"""The leak scanner: private details in many shapes are caught, and the house's
own messages are never flagged (a scanner that blocks everything is useless)."""
import itertools
import random
import re
from datetime import date, timedelta

import pytest

from agent.leaks import LeakScanner
from agent.outbound import cancel_message, chase_message, extras, job_message
from agent.state import Ticket
from world.data import load_data

DATA = load_data()
MEMBERS = list(DATA.members.values())
SCANNER = LeakScanner(MEMBERS, DATA.house)
DIGIT_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def _phone_shapes(phone: str, rng: random.Random) -> list[str]:
    d = re.sub(r"\D", "", phone)[-10:]
    words = [DIGIT_WORDS[int(c)] for c in d]
    mixed = [w if rng.random() < 0.5 else c for w, c in zip(words, d)]
    mixed[0] = words[0]  # at least one spelled-out digit
    return [
        phone, d, f"+91{d}", f"0091 {d[:5]} {d[5:]}", f"{d[:5]}-{d[5:]}", f"{d[:5]} {d[5:]}",
        " ".join(d[i:i + 2] for i in range(0, 10, 2)), f"({d[:3]}) {d[3:6]}-{d[6:]}", f"{d[:3]}.{d[3:6]}.{d[6:]}",
        " ".join(words), " ".join(mixed), f"0{d}",
    ]


@pytest.mark.parametrize("member", MEMBERS, ids=lambda m: m["member_id"])
def test_every_phone_shape_is_caught(member):
    rng = random.Random(member["member_id"])
    for shape in _phone_shapes(member["phone"], rng):
        text = f"Please call {shape} if anything changes."
        assert "phone number" in SCANNER.scan(text), shape


def test_any_unknown_phone_like_number_is_caught():
    rng = random.Random(7)
    for _ in range(200):
        d = str(rng.randrange(6, 10)) + "".join(str(rng.randrange(10)) for _ in range(9))
        shape = rng.choice([d, f"{d[:5]} {d[5:]}", f"+91-{d}", f"{d[:3]} {d[3:6]} {d[6:]}"])
        assert "phone number" in SCANNER.scan(f"reach the guest on {shape}"), shape


def test_spelled_out_unknown_number_is_caught():
    text = "the number is nine eight seven six five four three two one zero"
    assert "phone number" in SCANNER.scan(text)


@pytest.mark.parametrize("member", MEMBERS, ids=lambda m: m["member_id"])
def test_every_name_part_is_caught_in_any_case(member):
    for person in [member["name"], *(f["name"] for f in member["family"])]:
        for part in person.split():
            for shape in (part, part.lower(), part.upper(), f"{part}'s"):
                assert SCANNER.scan(f"Booking for {shape} and friends"), shape


@pytest.mark.parametrize("member", MEMBERS, ids=lambda m: m["member_id"])
def test_email_address_and_card_are_caught(member):
    assert "email address" in SCANNER.scan(f"write to {member['email']}")
    assert "email address" in SCANNER.scan("write to someone.else@gmail.com")
    found = SCANNER.scan(f"deliver to {member['address']}")
    assert any(f.startswith(("address detail", "postcode", "street address")) for f in found), member["address"]
    assert "card digits" in SCANNER.scan(f"card ending {member['card']['last4']}")
    assert "card expiry date" in SCANNER.scan(f"expires {member['card']['expiry']}")


@pytest.mark.parametrize("shape", ["4111 1111 1111 1111", "4111-1111-1111-1111", "4111111111111111",
                                   "3782 822463 10005"])
def test_full_card_numbers_are_caught(shape):
    assert "card number" in SCANNER.scan(f"card {shape}")


@pytest.mark.parametrize("text", ["Flat 1202, Sea Breeze", "14 Carmichael Road", "near Pali Hill",
                                  "B-14 Vasant Vihar", "Mumbai 400050", "3 Altamount Road", "apt 7"])
def test_address_fragments_are_caught(text):
    assert SCANNER.scan(text), text


# ---- no false alarms on what the house really sends ---------------------------
def _all_house_messages():
    start = date(2026, 11, 2)
    venues = DATA.house["venues"]
    times = ["12:00", "13:00", "13:30", "19:00", "20:00", "20:30", "21:00", "22:15", "23:45"]
    for n, (venue, days, hhmm, size) in enumerate(itertools.product(venues, range(0, 60, 3), times, (1, 2, 4, 6, 12))):
        t = Ticket(ticket_id=f"T-{n:04d}", member_id="M001", category="dining", summary="", source_text="",
                   created_at=None, venue=venue, date=(start + timedelta(days=days)).isoformat(), time=hhmm,
                   party_size=size)
        yield job_message(t, DATA.house, extras("birthday dinner, corner table with a view, vegan",
                                                ["no shellfish", "vegetarian wife"]))
        yield chase_message(t, DATA.house)
        yield cancel_message(t, DATA.house)


def test_house_messages_are_never_flagged():
    flagged = [(text, found) for text in _all_house_messages() if (found := SCANNER.scan(text))]
    assert flagged == []


@pytest.mark.parametrize("text", ["Confirmed: Koen, Sat 7 Nov 2026 at 8:00 pm, table for 4. Ref BB-20931.",
                                  "Booking for 2026-11-07 at 20:30", "Tea at 4 pm for one",
                                  "Please call +91 22 4000 0000", "Mail bookings@concierge.example"])
def test_ordinary_text_is_not_flagged(text):
    assert SCANNER.scan(text) == []
