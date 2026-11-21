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
                "model_seconds": round(self.seconds, 2), "priced": priced}


@dataclass
class LoopResult:
    stop: str    # done, max_steps, max_tokens, refusal
    steps: int
    text: str


def run_tool_loop(client, *, model: str, system: str, tools: list, prompt: str, run_tool, meter: Meter,
                  max_steps: int = 8, max_tokens: int = 2048) -> LoopResult:
    messages = [{"role": "user", "content": prompt}]
    for step in range(1, max_steps + 1):
        started = time.perf_counter()
        response = client.messages.create(model=model, max_tokens=max_tokens, system=system,
                                          tools=tools, messages=messages)
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
