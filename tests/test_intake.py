"""The cross-check between the model's reading of a message and a plain parser,
and a check that the tests' own leak oracle really finds leaks."""
from datetime import datetime

import pytest
from support import SATURDAY, independent_leaks

from agent.intake import Intake, check_category, resolve_fields
from world.data import load_data

NOW = datetime(2026, 11, 2, 10, 0)
VENUES = ["Casa Lumen", "Saffron Terrace", "The Blue Door", "Koen", "Mirabelle"]
TEXT = "Book Koen for Saturday 8:30 pm, table for 4"
RIGHT = {"venue": "Koen", "date": SATURDAY, "time": "20:30", "party_size": 4}


def test_agreeing_readings_pass():
    fields, problems, unverified = resolve_fields(RIGHT, TEXT, NOW, VENUES)
    assert (fields, problems, unverified) == (RIGHT, [], [])


@pytest.mark.parametrize("field,wrong", [("date", "2026-11-04"), ("time", "08:30"), ("time", "18:30"),
                                         ("party_size", 3), ("venue", "Mirabelle")])
def test_disagreement_empties_the_field_and_reports_it(field, wrong):
    fields, problems, _ = resolve_fields(dict(RIGHT, **{field: wrong}), TEXT, NOW, VENUES)
    assert fields[field] is None
    assert problems and problems[0].startswith(field)


@pytest.mark.parametrize("model_time", ["8:30 pm", "8:30pm", "20:30", "08:30 PM"])
def test_model_formats_are_normalised(model_time):
    fields, problems, _ = resolve_fields(dict(RIGHT, time=model_time), TEXT, NOW, VENUES)
    assert fields["time"] == "20:30" and problems == []


def test_parser_fills_what_the_model_left_out():
    fields, problems, unverified = resolve_fields({"venue": None, "date": None, "time": None, "party_size": None},
                                                  TEXT, NOW, VENUES)
    assert fields == RIGHT and problems == [] and unverified == []


def test_model_only_values_are_marked_unverified():
    text = "Book Koen this Saturday, half past eight in the evening, four of us"
    fields, problems, unverified = resolve_fields(RIGHT, text, NOW, VENUES)
    assert fields == RIGHT and problems == []
    assert unverified == ["time"]  # "four of us" and "Saturday" the parser can read; the time it cannot


def test_past_date_and_unknown_venue_are_problems():
    _, problems, _ = resolve_fields(dict(RIGHT, date="2026-10-30", venue="Trishna"), "Book Trishna", NOW, VENUES)
    assert any("in the past" in p for p in problems) and any("Trishna" in p for p in problems)


@pytest.mark.parametrize("bad", ["", "not a number", -2, 0, 500, None])
def test_nonsense_party_sizes_are_ignored(bad):
    fields, _, _ = resolve_fields(dict(RIGHT, party_size=bad), "Book Koen Saturday 8:30 pm", NOW, VENUES)
    assert fields["party_size"] is None


@pytest.mark.parametrize("text,claimed,expected", [
    ("Book a car to the airport at 6 am", "dining", "transport"),
    ("Book flowers for Friday", "dining", "flowers"),
    ("Book a villa in Goa", "dining", "travel"),
    ("Book Koen for Saturday", "dining", "dining"),
    ("Book a table for 4", "dining", "dining"),
    ("Book a car to Koen", "transport", "transport"),
])
def test_category_check(text, claimed, expected):
    assert check_category(claimed, text, VENUES) == expected


@pytest.mark.parametrize("text,cancel,change", [
    ("Actually cancel that, plans changed", True, False),
    ("rehne do, we'll stay home", True, False),
    ("Can you make it 8 instead?", False, True),
    ("Push it to 9 pm", False, True),
    ("can we bring it forward to 7", False, True),
    ("Book Koen for Saturday", False, False),
])
def test_intent_words(text, cancel, change):
    intake = Intake("G-101", "M001", "Riya", text, "IN-1", NOW)
    assert (intake.wants_cancel, intake.wants_change) == (cancel, change)


def test_leak_oracle_finds_the_fake_models_careless_draft():
    """The scenario tests say "no leaks"; this shows the oracle would have noticed one."""
    members = list(load_data().members.values())
    draft = ('Guest name: Riya Kapoor. Contact: +91 98201 23456.\n'
             'Client\'s message: "Arjun\'s parents are joining"')
    found = independent_leaks(draft, members)
    assert "M001 name part 'riya'" in found and "M001 phone" in found and "M001 name part 'arjun'" in found
    assert independent_leaks("Card: Amex ending 8812, expiry 09/29", members) == \
        ["M001 card last4", "M001 card expiry"]


@pytest.mark.parametrize("text, expected", [
    ("Dinner at Mirabelle day after tomorrow at 8 for 2", "2026-11-04"),
    ("Koen tomorrow 9 pm for 2", "2026-11-03"),
    ("Koen tonight 9 pm for 2", "2026-11-02"),
])
def test_relative_days_agree_with_the_right_model_reading(text, expected):
    claimed = {"venue": None, "date": expected, "time": None, "party_size": None}
    fields, problems, _ = resolve_fields(claimed, text, NOW, VENUES)
    assert fields["date"] == expected and not problems
