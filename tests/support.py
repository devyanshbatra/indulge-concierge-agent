"""Shared helpers for the agent tests.

`independent_leaks` is a second leak check written separately from
agent/leaks.py, so a bug in the agent's scanner cannot hide itself in the tests.
`ScriptedClient` is a fake model that makes exactly the tool calls a test asks
for, to play a careless or hostile model.
"""
from __future__ import annotations

import re

from agent import Concierge
from world import SCENARIOS, FakeAnthropic, World
from world.fake_llm import Message, TextBlock, ToolUseBlock, Usage

SATURDAY, FRIDAY, SUNDAY, THURSDAY = "2026-11-07", "2026-11-06", "2026-11-08", "2026-11-05"
TOMORROW = "2026-11-03"
GROUP_OF = {"M001": "G-101", "M002": "G-102", "M003": "G-103", "M004": "G-104", "M005": "G-105", "M006": "G-106"}


def make(bookbot: str = "correct", flaky: bool = False, quirks=None, client=None):
    world = World(bookbot=bookbot, flaky=flaky)
    if client is None:
        client = FakeAnthropic(world.clock, world.data.house["venues"], quirks=quirks)
    concierge = Concierge(world, client)
    world.attach(concierge)
    return world, concierge


def approve_all(concierge, approver: str = "test-human") -> list[dict]:
    return [concierge.approve(p["proposal_id"], approver=approver) for p in concierge.pending_proposals()]


def run_scenario(name: str):
    scenario = SCENARIOS[name]
    world, concierge = make(scenario["bookbot"], scenario.get("flaky", False), scenario.get("quirks"))
    for step in scenario["steps"]:
        if step[0] == "say":
            world.member_says(step[1], step[2])
        elif step[0] == "wait":
            world.advance(step[1])
        elif step[0] == "approve_all":
            approve_all(concierge)
    return world, concierge


def sent_texts(world) -> list[str]:
    return [m["text"] for m in world.transport.sent]


def ticket(concierge, index: int = 0) -> dict:
    return concierge.tickets()[index]


# ---- a leak check that does not share code with agent/leaks.py --------------
def independent_leaks(text: str, members) -> list[str]:
    found = []
    low = text.lower()
    digits = re.sub(r"\D", "", text)
    words = set(re.findall(r"[a-z]+", low))
    for m in members:
        people = [m["name"]] + [f["name"] for f in m.get("family", [])]
        for person in people:
            for part in person.lower().split():
                if part in words:
                    found.append(f"{m['member_id']} name part {part!r}")
        phone = re.sub(r"\D", "", m["phone"])[-10:]
        if phone in digits or phone[-7:] in digits:
            found.append(f"{m['member_id']} phone")
        if m["email"].lower() in low or m["email"].split("@")[0].lower() in low:
            found.append(f"{m['member_id']} email")
        street = m["address"].split(",")[0].lower()
        if street in low:
            found.append(f"{m['member_id']} street")
        postcode = re.search(r"\d{6}", m["address"])
        if postcode and postcode.group(0) in digits:
            found.append(f"{m['member_id']} postcode")
        card = m.get("card", {})
        if card.get("last4") and re.search(rf"(?<!\d){card['last4']}(?!\d)", text):
            found.append(f"{m['member_id']} card last4")
        if card.get("expiry") and card["expiry"] in text:
            found.append(f"{m['member_id']} card expiry")
    return found


def leaks_in_sent(world) -> list[str]:
    members = list(world.data.members.values())
    return [f"{finding} in {text!r}" for text in sent_texts(world) for finding in independent_leaks(text, members)]


# ---- a model that does exactly what the test says ---------------------------
class ScriptedClient:
    """Each call to messages.create plays the next turn of the script.

    A turn is a list of (tool_name, args) pairs (one tool_use block each), a
    string (a final text answer), or an exception instance (raised).
    When the script runs out, the model answers "done".
    """

    def __init__(self, turns=()):
        self.turns = list(turns)
        self.requests: list[dict] = []
        self.messages = self
        self._seq = 0

    def create(self, **kwargs) -> Message:
        self.requests.append(kwargs)
        turn = self.turns.pop(0) if self.turns else "done"
        if isinstance(turn, Exception):
            raise turn
        self._seq += 1
        if isinstance(turn, str):
            content, stop = [TextBlock(turn)], "end_turn"
        else:
            content = [ToolUseBlock(id=f"toolu_s{self._seq}_{i}", name=name, input=args)
                       for i, (name, args) in enumerate(turn)]
            stop = "tool_use"
        return Message(id=f"msg_s{self._seq}", model=kwargs["model"], content=content,
                       stop_reason=stop, usage=Usage(input_tokens=100, output_tokens=20))

    def tool_results(self) -> list[dict]:
        """Every tool_result the agent gave back, in order (from the last request's transcript)."""
        if not self.requests:
            return []
        out = []
        for msg in self.requests[-1]["messages"]:
            if msg["role"] == "user" and isinstance(msg["content"], list):
                out += [b for b in msg["content"] if b.get("type") == "tool_result"]
        return out


class LoopingClient:
    """A model that never stops calling tools."""

    def __init__(self, call=("list_open_tickets", {"member_id": "M001"})):
        self.call = call
        self.calls = 0
        self.messages = self

    def create(self, **kwargs) -> Message:
        self.calls += 1
        return Message(id=f"msg_l{self.calls}", model=kwargs["model"],
                       content=[ToolUseBlock(id=f"toolu_l{self.calls}", name=self.call[0], input=self.call[1])],
                       stop_reason="tool_use", usage=Usage(input_tokens=50, output_tokens=10))
