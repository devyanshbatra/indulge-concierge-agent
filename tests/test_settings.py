"""Model call settings: what reaches the API, the lean one-call flow, and profiles."""
import json

from agent import Concierge, profiles
from agent.intake import LEAN_PROMPT, SYSTEM_PROMPT
from agent.loop import ModelSettings
from agent.tools import TOOLS, TOOLS_SHORT
from world import World

from support import SATURDAY, ScriptedClient, make

TEXT = "Book Koen for Saturday 8pm, table for 4"
BOOKING = {"member_id": "M001", "category": "dining", "request_summary": "Koen Saturday", "venue": "Koen",
           "date": SATURDAY, "time": "20:00", "party_size": 4}


def test_defaults_send_nothing_extra():
    assert ModelSettings().request_options() == {}


def test_options_reach_the_api_call():
    s = ModelSettings(thinking=False, temperature=0.0, top_p=0.9, max_tokens=512, prompt="lean", tool_text="short")
    client = ScriptedClient(["No action"])
    world, _ = make(client=client)
    world.concierge.settings = s
    world.member_says("G-101", "hello")
    request = client.requests[0]
    assert request["thinking"] == {"type": "disabled"}
    assert request["extra_body"] == {"temperature": 0.0, "top_p": 0.9}
    assert request["max_tokens"] == 512 and request["system"] == LEAN_PROMPT and request["tools"] == TOOLS_SHORT


def test_lean_flow_is_one_model_call_and_code_proposes():
    client = ScriptedClient([[("create_ticket", BOOKING)], "should not be asked"])
    world, concierge = make(client=client)
    concierge.settings = ModelSettings(prompt="lean")
    world.member_says("G-101", TEXT)
    assert len(client.requests) == 1
    assert concierge.pending_proposals()[0]["vendor_id"] == "V-BOOKBOT"


def test_lean_flow_lets_the_model_fix_a_failed_call():
    bad = dict(BOOKING, member_id="M002")
    client = ScriptedClient([[("create_ticket", bad)], [("create_ticket", BOOKING)]])
    world, concierge = make(client=client)
    concierge.settings = ModelSettings(prompt="lean")
    world.member_says("G-101", TEXT)
    assert len(client.requests) == 2 and len(concierge.tickets()) == 1


def test_short_tools_keep_names_and_schemas():
    assert [t["name"] for t in TOOLS_SHORT] == [t["name"] for t in TOOLS]
    for short, full in zip(TOOLS_SHORT, TOOLS):
        assert short["input_schema"]["required"] == full["input_schema"]["required"]
        assert len(json.dumps(short)) < len(json.dumps(full))


def test_profiles(tmp_path, monkeypatch):
    path = tmp_path / "model_profiles.json"
    monkeypatch.setattr(profiles, "PROFILES_FILE", path)
    assert profiles.settings_for("qwen3.5:4b") == ModelSettings()  # no file
    path.write_text(json.dumps({"qwen3.5:4b": {"settings": {"thinking": False, "prompt": "lean", "unknown": 1}}}))
    assert profiles.settings_for("qwen3.5:4b") == ModelSettings(thinking=False, prompt="lean")
    assert profiles.settings_for("claude-haiku-5-5") == ModelSettings()
    world = World()
    assert Concierge(world, None, model="qwen3.5:4b").settings.prompt == "lean"


def test_claude_and_the_fake_keep_the_full_flow():
    _, concierge = make()
    assert concierge.settings.prompt == "full" and SYSTEM_PROMPT.startswith("You are the intake assistant")


def _gated(client):
    world, concierge = make(client=client)
    concierge.settings = ModelSettings(prompt="lean", gate=0.10)
    return world, concierge


def test_gate_skips_the_model_for_obvious_chatter():
    client = ScriptedClient([])
    world, concierge = _gated(client)
    for text in ("Thanks so much!", "Good morning!", "haha ok see you"):
        world.member_says("G-101", text)
    assert client.requests == [] and concierge.meter.skipped == 3
    assert concierge.alerts() == [] and concierge.tickets() == []


def test_gate_never_skips_a_request_a_cancellation_or_a_change():
    texts = [TEXT, "cancel it please", "Can you make it 6 people instead?", "The car hasn't turned up yet"]
    client = ScriptedClient(["No action"] * len(texts))
    world, concierge = _gated(client)
    for text in texts:
        world.member_says("G-101", text)
    assert len(client.requests) == len(texts) and concierge.meter.skipped == 0


def test_gate_is_off_by_default():
    client = ScriptedClient(["No action"])
    world, concierge = make(client=client)
    concierge.settings = ModelSettings(prompt="lean")
    world.member_says("G-101", "Thanks so much!")
    assert len(client.requests) == 1 and concierge.meter.skipped == 0
