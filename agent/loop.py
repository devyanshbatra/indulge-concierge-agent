"""Our own model loop: ask the model, run the tools it asks for, give back the
results, repeat until it stops asking or the step limit is hit.

It only uses `client.messages.create`, so it runs unchanged on FakeAnthropic,
anthropic.Anthropic() and Ollama's Anthropic-compatible server.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

from world.pricing import PRICES, cost_usd


@dataclass
class Meter:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    seconds: float = 0.0
    skipped: int = 0              # messages the small classifier answered without a model call

    def add(self, usage, seconds: float) -> None:
        self.calls += 1
        self.input_tokens += getattr(usage, "input_tokens", 0) or 0
        self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.seconds += seconds

    def summary(self, model: str) -> dict:
        priced = model in PRICES  # a local model (e.g. qwen via Ollama) costs nothing per token
        cost = cost_usd(model, self.input_tokens, self.output_tokens, self.cache_read_tokens) if priced else 0.0
        return {"calls": self.calls, "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "cache_read_tokens": self.cache_read_tokens, "cost_usd": round(cost, 6),
                "model_seconds": round(self.seconds, 2), "priced": priced, "skipped_by_classifier": self.skipped}


@dataclass(frozen=True)
class ModelSettings:
    """How we call the model. The defaults are the plain behaviour for Claude and the fake.

    Tuned values for local models live in agent/model_profiles.json (see tuning/).
    """
    max_tokens: int = 4096        # room for a thinking model to reason before its tool call
    max_steps: int = 8
    thinking: bool = True         # False asks the model not to think (a big speed-up on small local models)
    temperature: float | None = None
    top_p: float | None = None
    prompt: str = "full"          # "full": the model proposes; "lean"/"lean_compact": it only reads, code does the rest
    tool_text: str = "full"       # "full" or "short" tool descriptions
    gate: float = 0.0             # skip the model when the classifier's p(request) is below this (0 = off)

    @property
    def lean(self) -> bool:
        return self.prompt.startswith("lean")

    def request_options(self) -> dict:
        options = {}
        if not self.thinking:
            options["thinking"] = {"type": "disabled"}
        sampling = {k: v for k, v in (("temperature", self.temperature), ("top_p", self.top_p)) if v is not None}
        if sampling:  # SDK 2.x has no keyword for these; Ollama reads them from the body
            options["extra_body"] = sampling
        return options


@dataclass
class LoopResult:
    stop: str    # done, max_steps, max_tokens, refusal
    steps: int
    text: str


def run_tool_loop(client, *, model: str, system: str, tools: list, prompt: str, run_tool, meter: Meter,
                  max_steps: int = 8, max_tokens: int = 2048, options: dict | None = None,
                  stop_after: frozenset = frozenset()) -> LoopResult:
    """`stop_after`: tools that end the turn. When every call in a turn is one of these and
    none failed, the loop stops without asking the model for a closing sentence."""
    messages = [{"role": "user", "content": prompt}]
    for step in range(1, max_steps + 1):
        started = time.perf_counter()
        response = client.messages.create(model=model, max_tokens=max_tokens, system=system,
                                          tools=tools, messages=messages, **(options or {}))
        meter.add(response.usage, time.perf_counter() - started)

        blocks = [b for b in (_as_param(block) for block in response.content) if b]
        messages.append({"role": "assistant", "content": blocks})
        calls = [b for b in blocks if b["type"] == "tool_use"]
        if not calls:
            text = " ".join(b["text"] for b in blocks if b["type"] == "text").strip()
            stop = response.stop_reason if response.stop_reason in ("max_tokens", "refusal") else "done"
            return LoopResult(stop, step, text)

        results = []  # every result for this turn goes back in one user message
        for call in calls:
            outcome = run_tool(call["name"], call.get("input") or {})
            result = {"type": "tool_result", "tool_use_id": call["id"], "content": json.dumps(outcome, default=str)}
            if "error" in outcome:
                result["is_error"] = True
            results.append(result)
        messages.append({"role": "user", "content": results})
        if stop_after and all(c["name"] in stop_after for c in calls) and not any(r.get("is_error") for r in results):
            return LoopResult("done", step, "")
    return LoopResult("max_steps", max_steps, "")


def _as_param(block) -> dict | None:
    """A response block as a request block (SDK objects and the fake's dataclasses alike)."""
    kind = getattr(block, "type", None)
    if kind == "text":
        return {"type": "text", "text": block.text} if block.text else None  # the API rejects empty text
    if kind == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    if kind == "thinking":
        return {"type": "thinking", "thinking": block.thinking or "", "signature": getattr(block, "signature", "") or ""}
    if kind == "redacted_thinking":
        return {"type": "redacted_thinking", "data": block.data}
    return block.model_dump() if hasattr(block, "model_dump") else dict(block)
