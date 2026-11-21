"""The eleven scenarios from world/scenarios.py, run on the fake model.

`world.transport.sent` is the ground truth: it is everything any vendor received.
"""
import pytest

from world import SCENARIOS

from support import FRIDAY, SATURDAY, SUNDAY, TOMORROW, leaks_in_sent, run_scenario, sent_texts, ticket


# ---- rules every scenario must keep ----------------------------------------
@pytest.mark.parametrize("name", list(SCENARIOS))
def test_nothing_private_reaches_a_vendor(name):
    world, _ = run_scenario(name)
    assert leaks_in_sent(world) == []


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_every_send_is_keyed_and_never_repeated(name):
    world, _ = run_scenario(name)
    keys = [m["idempotency_key"] for m in world.transport.sent]
    assert all(keys), "every send carries an idempotency key"
    assert len(keys) == len(set(keys))
    assert len(world.bookbot.received) == len(world.transport.sent)


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_vendor_messages_use_the_house_name_only(name):
    world, _ = run_scenario(name)
    for text in sent_texts(world):
        assert "under the name Indulge Concierge." in text
        assert "Client's message" not in text and "Guest name" not in text  # the model's draft never goes out


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_usage_is_reported(name):
    _, concierge = run_scenario(name)
    usage = concierge.usage()
    assert usage["calls"] >= 1 and usage["input_tokens"] > 0
    assert usage["cost_usd"] >= 0


# ---- each scenario's own trap ------------------------------------------------
def test_riya_corner_table_is_confirmed_without_family_names():
    world, concierge = run_scenario("riya_corner_table")
    assert len(world.transport.sent) == 1
    text = sent_texts(world)[0]
    assert "table for 4 at Casa Lumen on Sat 7 Nov 2026 at 8:30 pm" in text
    assert "corner table" in text and "no shellfish" in text
    assert "Arjun" not in text and "Kapoor" not in text
    t = ticket(concierge)
    assert (t["status"], t["venue"], t["date"], t["time"], t["party_size"]) == \
        ("confirmed", "Casa Lumen", SATURDAY, "20:30", 4)


def test_mehta_wrong_date_is_not_accepted():
    world, concierge = run_scenario("mehta_birthday_wrong_date")
    text = sent_texts(world)[0]
    assert "Sat 6 Nov" not in text and "Fri 6 Nov 2026" in text
    assert "a birthday" in text and "Anjali" not in text and "99300" not in text
    assert ticket(concierge)["status"] == "needs_human"
    assert any("date" in a["reason"] and "Sat 7 Nov" in a["reason"] for a in concierge.alerts())


def test_kabir_duplicate_propose_sends_once_and_change_goes_to_a_human():
    world, concierge = run_scenario("kabir_hinglish_double")
    assert len(world.transport.sent) == 1
    t = ticket(concierge)
    assert (t["venue"], t["date"], t["time"], t["party_size"]) == ("The Blue Door", SATURDAY, "21:00", 3)
    assert t["status"] == "needs_human"
    assert any("changed the booking after it was confirmed" in a["reason"] for a in concierge.alerts())


def test_sameer_vague_request_sends_nothing_and_asks_a_human():
    world, concierge = run_scenario("sameer_unclear")
    assert world.transport.sent == []
    assert ticket(concierge)["status"] == "needs_info"
    assert concierge.alerts()


def test_priya_cancel_before_approval_sends_nothing():
    world, concierge = run_scenario("priya_cancel_before_approval")
    assert world.transport.sent == []
    assert ticket(concierge)["status"] == "cancelled"
    assert concierge.pending_proposals() == []


def test_priya_cancel_after_send_tells_the_vendor_once():
    world, concierge = run_scenario("priya_cancel_after_send")
    texts = sent_texts(world)
    assert len(texts) == 2
    assert "table for 6 at Mirabelle on Sun 8 Nov 2026 at 1:00 pm" in texts[0]
    assert texts[1].startswith("Please cancel")
    assert ticket(concierge)["status"] == "cancelled"


def test_riya_silent_is_chased_once_then_handed_over():
    world, concierge = run_scenario("riya_silent")
    texts = sent_texts(world)
    assert len(texts) == 2 and texts[1].startswith("Following up")
    chase_at = world.transport.sent[1]["at"]
    assert (chase_at - world.transport.sent[0]["at"]).total_seconds() / 60 == 20
    t = ticket(concierge)
    assert (t["status"], t["date"]) == ("needs_human", TOMORROW)
    assert sum("No confirmation" in a["reason"] for a in concierge.alerts()) == 1


@pytest.mark.parametrize("name", ["riya_asks_phone", "mehta_asks_card"])
def test_vendor_asking_for_private_details_gets_nothing(name):
    world, concierge = run_scenario(name)
    assert len(world.transport.sent) == 1  # only the job itself
    assert ticket(concierge)["status"] == "needs_human"
    assert any("personal details" in a["reason"] for a in concierge.alerts())


def test_riya_asks_phone_books_friday():
    _, concierge = run_scenario("riya_asks_phone")
    assert ticket(concierge)["date"] == FRIDAY


def test_kabir_flaky_network_delivers_exactly_once():
    world, concierge = run_scenario("kabir_flaky_network")
    assert len(world.transport.sent) == 1
    assert len(world.bookbot.received) == 1
    assert ticket(concierge)["status"] == "confirmed"


def test_tara_chatter_creates_nothing():
    world, concierge = run_scenario("tara_chatter")
    assert world.transport.sent == []
    assert concierge.tickets() == []
    assert concierge.pending_proposals() == []
    assert concierge.alerts() == []


def test_priya_sunday_lunch_fields():
    _, concierge = run_scenario("priya_cancel_after_send")
    t = ticket(concierge)
    assert (t["venue"], t["date"], t["time"], t["party_size"]) == ("Mirabelle", SUNDAY, "13:00", 6)
