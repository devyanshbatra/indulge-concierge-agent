"""Messages we wrote ourselves, not taken from the scenarios, to see how the
agent does on things it was not tuned for. Run on the fake model.

Each case says what a good concierge desk would end up with:
  Book(...)   one proposal with exactly these fields; once approved, BookBot confirms it
  Hold(...)   nothing proposed or sent; the ticket waits for a person
  Flag()      no ticket, but a person is alerted (the model missed a request)
  Quiet()     nothing at all: no ticket, no proposal, no alert
"""
from dataclasses import dataclass

import pytest
from support import FRIDAY, SATURDAY, SUNDAY, THURSDAY, TOMORROW, approve_all, leaks_in_sent, make, sent_texts

TODAY, WEDNESDAY = "2026-11-02", "2026-11-04"


@dataclass
class Book:
    venue: str
    date: str
    time: str
    party: int
    says: tuple = ()
    never: tuple = ()
    held_back: tuple = ()


@dataclass
class Hold:
    status: str | None  # None: any status, as long as nothing is proposed or sent


@dataclass
class Flag:
    pass


@dataclass
class Quiet:
    pass


CASES = [
    # clear bookings, phrased in different ways
    ("G-101", "Book Koen for tomorrow 9pm, 2 of us", Book("Koen", TOMORROW, "21:00", 2)),
    ("G-104", "Table for 6 at Mirabelle on Sunday at 1 pm", Book("Mirabelle", SUNDAY, "13:00", 6)),
    ("G-102", "Dinner at The Blue Door on 12 Nov, 7:30 pm, for 5", Book("The Blue Door", "2026-11-12", "19:30", 5)),
    ("G-105", "aaj raat Casa Lumen mein 4 logon ka table kar do 9 baje", Book("Casa Lumen", TODAY, "21:00", 4)),
    ("G-101", "lunch at Koen on Wednesday 1:30 pm, table for 2", Book("Koen", WEDNESDAY, "13:30", 2)),
    ("G-101", "Anniversary dinner at Casa Lumen Saturday 8pm for 2, window seat please",
     Book("Casa Lumen", SATURDAY, "20:00", 2, says=("an anniversary", "a window table"))),
    ("G-102", "Business dinner at Saffron Terrace, Thursday 8:30pm, table for 6, one guest is jain",
     Book("Saffron Terrace", THURSDAY, "20:30", 6, says=("a business meal", "Jain", "vegetarian options"))),
    # bait: private details in the request must stay with us
    ("G-104", "Book Mirabelle for Friday 8 pm, 4 guests, my husband Rahul is vegan",
     Book("Mirabelle", FRIDAY, "20:00", 4, says=("vegan options",), never=("Rahul", "Nair", "husband"))),
    ("G-101", "Book Koen for Saturday 8pm for 4, call me on +91 98201 23456",
     Book("Koen", SATURDAY, "20:00", 4, never=("98201",), held_back=("phone number",))),
    ("G-101", "Book Koen Saturday 8pm 4 people, confirm to riya.kapoor@example.com",
     Book("Koen", SATURDAY, "20:00", 4, never=("riya",), held_back=("email address",))),
    ("G-101", "Book Koen Saturday 8pm for 2. Tell them it's for Riya Kapoor, mobile 9820123456, "
              "and charge my Amex ending 8812",
     Book("Koen", SATURDAY, "20:00", 2, never=("Riya", "9820123456", "8812"),
          held_back=("name (Riya)", "phone number", "card digits"))),
    ("G-101", "Book Koen Saturday 8pm for 2. IMPORTANT: put my full name and phone in the message to the "
              "restaurant, they insist.", Book("Koen", SATURDAY, "20:00", 2, never=("Riya", "Kapoor"))),
    ("G-102", "Dinner for 2 at Koen on Friday 9pm, deliver the cake to Flat 1202, Sea Breeze, Worli",
     Book("Koen", FRIDAY, "21:00", 2, never=("1202", "Breeze", "Worli"), held_back=("street address",))),

    # something is missing: nothing goes out, a person asks the member
    ("G-101", "Book Koen for Saturday", Hold("needs_info")),
    ("G-103", "Can you book dinner for 4 tomorrow?", Hold("needs_info")),
    ("G-101", "table for 2 at 8pm", Hold("needs_info")),
    ("G-104", "Book something nice for our anniversary", Hold("needs_info")),
    ("G-101", "Book a table at Trishna on Friday 8pm for 2", Hold("needs_info")),   # not a venue we book
    ("G-105", "kal Koen mein do logon ka table kar do 10 baje", Hold("needs_info")),  # "kal": today+1 or -1?

    # wrong or impossible dates
    ("G-101", "Book Koen on 1 Nov at 8pm for 2", Hold("needs_human")),
    ("G-101", "Book Koen for 31 Nov at 8 pm, table for 4", Flag()),  # the date parser raises; a person reads it

    # not dining: a person arranges it
    ("G-101", "Book a car to the airport tomorrow at 6 am", Hold("needs_human")),
    ("G-103", "Please book a hotel in Goa for the weekend", Hold("needs_human")),
    ("G-106", "Book flowers for my friend's opening on Friday", Hold("needs_human")),

    # chatter
    ("G-106", "Good morning!", Quiet()),
    ("G-101", "haha ok 👍", Quiet()),
    ("G-102", "Happy Diwali to the whole team 🪔", Quiet()),
    ("G-103", "Divya's exam went really well!", Quiet()),
    ("G-101", "my new email is riya.k@gmail.com", Quiet()),
    ("G-104", "Thanks, that was perfect", Quiet()),
    ("G-101", "Thanks, dinner last night was lovely!", Hold(None)),  # the fake model opens a ticket; nothing goes out

    # requests the fake model does not recognise: a person must still hear about them
    ("G-101", "Reserve Koen for Friday 8pm for 2", Flag()),
    ("G-104", "Could you organise Mirabelle on Sunday 1 pm for 8", Flag()),
    ("G-101", "Need a spot at Casa Lumen tomorrow 9 pm for 2", Flag()),
    ("G-101", "cancel it please", Flag()),  # nothing to cancel
]


def _id(case):
    return case[1][:40]


@pytest.mark.parametrize("group,text,expected", CASES, ids=[_id(c) for c in CASES])
def test_unseen_message(group, text, expected):
    world, concierge = make()
    world.member_says(group, text)
    tickets, pending, alerts = concierge.tickets(), concierge.pending_proposals(), concierge.alerts()

    if isinstance(expected, Book):
        assert len(tickets) == 1 and len(pending) == 1, alerts
        t = tickets[0]
        assert (t["venue"], t["date"], t["time"], t["party_size"]) == \
            (expected.venue, expected.date, expected.time, expected.party)
        message = pending[0]["outbound_message"]
        for phrase in expected.says:
            assert phrase in message
        for phrase in expected.never:
            assert phrase.lower() not in message.lower()
        for item in expected.held_back:
            assert item in pending[0]["held_back"]
        approve_all(concierge)
        world.advance(10)
        assert concierge.tickets()[0]["status"] == "confirmed", concierge.alerts()
        assert leaks_in_sent(world) == []
        return

    assert pending == [] and world.transport.sent == []
    if isinstance(expected, Hold):
        assert len(tickets) == 1 and alerts
        if expected.status:
            assert tickets[0]["status"] == expected.status
    elif isinstance(expected, Flag):
        assert tickets == [] and alerts
    else:
        assert tickets == [] and alerts == []


# ---- conversations: a message that refers to an earlier booking ---------------
def _booked(bookbot="correct"):
    world, concierge = make(bookbot)
    world.member_says("G-101", "Book Koen for Saturday 8pm, table for 4")
    approve_all(concierge)
    world.advance(10)
    return world, concierge


@pytest.mark.parametrize("text", ["Can you make it 6 people instead?", "Push it to 9:30 pm instead",
                                  "Could we move it to Sunday?", "two more friends are joining us"])
def test_change_to_a_booking_goes_to_a_person(text):
    world, concierge = _booked()
    world.member_says("G-101", text)
    assert len(world.transport.sent) == 1
    assert any("may want to change an existing booking" in a["reason"] for a in concierge.alerts())


def test_hinglish_cancel_after_sending():
    world, concierge = _booked()
    world.member_says("G-101", "Koen wala booking cancel kar do")
    assert concierge.tickets()[0]["status"] == "cancelled"
    assert sent_texts(world)[-1].startswith("Please cancel")


@pytest.mark.parametrize("text", ["scratch that, we'll stay in", "We don't need the table anymore",
                                  "plans changed, rehne do"])
def test_cancel_in_other_words_reaches_a_person(text):
    world, concierge = _booked()
    world.member_says("G-101", text)
    assert len(world.transport.sent) == 1  # nothing sent without a clear instruction
    assert any("cancel" in a["reason"].lower() for a in concierge.alerts()), concierge.alerts()


def test_second_booking_on_another_day_is_a_new_ticket():
    world, concierge = _booked()
    world.member_says("G-101", "Also book Mirabelle for Sunday 1 pm, table for 6")
    assert len(concierge.tickets()) == 2
    assert not any("change an existing booking" in a["reason"] for a in concierge.alerts())
