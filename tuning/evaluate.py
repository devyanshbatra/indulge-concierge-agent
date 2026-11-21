"""Run the Concierge over the labelled cases with a real model and score it.

    python -X utf8 -m tuning.evaluate --model qwen3.5:4b --prompt lean --thinking off --split test

Needs a running Ollama (ANTHROPIC_BASE_URL, default http://localhost:11434).
"""
from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

from agent import Concierge
from agent.loop import ModelSettings
from world import FakeAnthropic, World

from .cases import CASES, HINGLISH, Ask, Book, Cancelled, Quiet, split

OLLAMA_URL = os.environ.get("ANTHROPIC_BASE_URL", "http://localhost:11434")


def ollama_client():
    import anthropic
    return anthropic.Anthropic(base_url=OLLAMA_URL, api_key="ollama", max_retries=0, timeout=900)


def with_context(base: str, num_ctx: int | None) -> str:
    """The name of an Ollama model that is `base` with a fixed context length (created once).
    The Anthropic endpoint ignores num_ctx in the request, so it has to live in a Modelfile."""
    if not num_ctx:
        return base
    name = f"{base.replace(':', '-')}-ctx{num_ctx}"
    listed = subprocess.run(["ollama", "list"], capture_output=True, text=True).stdout
    if name not in listed:
        with tempfile.TemporaryDirectory() as tmp:
            modelfile = Path(tmp) / "Modelfile"
            modelfile.write_text(f"FROM {base}\nPARAMETER num_ctx {num_ctx}\n", encoding="utf-8")
            subprocess.run(["ollama", "create", name, "-f", str(modelfile)], check=True, capture_output=True)
    return name


def warm_up(client, model: str) -> None:
    """Load the model before timing anything."""
    client.messages.create(model=model, max_tokens=1, thinking={"type": "disabled"},
                           messages=[{"role": "user", "content": "hi"}])


# ---- one case ------------------------------------------------------------------
def run_case(case, client, model: str, settings: ModelSettings) -> dict:
    world = World(bookbot="correct")
    concierge = Concierge(world, FakeAnthropic(world.clock, world.data.house["venues"]))
    world.attach(concierge)
    for group, text in case.setup:  # earlier bookings, made by the fake model
        world.member_says(group, text)
        for p in concierge.pending_proposals():
            concierge.approve(p["proposal_id"], approver="setup")
    if case.setup:
        world.advance(10)

    n_tickets, n_alerts, n_sent = len(concierge.tickets()), len(concierge.alerts()), len(world.transport.sent)
    old_proposals = set(concierge.store.proposals)
    concierge.client, concierge.model, concierge.settings = client, model, settings
    meter = concierge.meter
    calls0, in0, out0 = meter.calls, meter.input_tokens, meter.output_tokens

    started = time.perf_counter()
    world.member_says(case.group, case.text)
    seconds = time.perf_counter() - started

    tickets = concierge.tickets()
    new_tickets = tickets[n_tickets:]
    new_alerts = concierge.alerts()[n_alerts:]
    new_proposals = [p for pid, p in concierge.store.proposals.items() if pid not in old_proposals and p.kind == "job"]
    new_sent = world.transport.sent[n_sent:]
    setup_status = [t["status"] for t in tickets[:n_tickets]]
    booked = [tuple(concierge.store.tickets[p.ticket_id].fields().values()) for p in new_proposals]

    def outcome_ok(expect) -> bool:
        if isinstance(expect, Book):
            return booked == [(expect.venue, expect.date, expect.time, expect.party)] and not new_sent
        if isinstance(expect, Ask):
            return not new_proposals and not new_sent and bool(new_alerts) and "cancelled" not in setup_status
        if isinstance(expect, Quiet):
            return not (new_tickets or new_alerts or new_proposals or new_sent) and "cancelled" not in setup_status
        if isinstance(expect, Cancelled):
            return [s == "cancelled" for s in setup_status] == [i == expect.index for i in range(len(setup_status))] \
                and not new_proposals
        raise TypeError(expect)

    correct = any(outcome_ok(e) for e in case.expect)
    wanted = {(e.venue, e.date, e.time, e.party) for e in case.expect if isinstance(e, Book)}
    if booked:
        got = "book " + "; ".join(f"{v} {d} {t} x{p}" for v, d, t, p in booked)
    elif "cancelled" in setup_status:
        got = f"cancelled {[i for i, s in enumerate(setup_status) if s == 'cancelled']}"
    elif new_alerts:
        got = f"ask ({new_alerts[0]['reason'][:70]})"
    else:
        got = "quiet" if not new_tickets else f"ticket {new_tickets[0]['status']}, nobody told"
    return {"id": case.id, "kind": case.kind, "correct": correct, "seconds": round(seconds, 2), "got": got,
            "wrong_booking": bool(booked) and not set(booked) <= wanted,
            "model_failed": any("model failed" in a["reason"] or "stopped early" in a["reason"] for a in new_alerts),
            "calls": meter.calls - calls0, "input_tokens": meter.input_tokens - in0,
            "output_tokens": meter.output_tokens - out0}


# ---- a whole set -----------------------------------------------------------------
def evaluate(cases, model: str, settings: ModelSettings, *, budget_seconds: float | None = None,
             client=None, log=print) -> dict:
    client = client or ollama_client()
    warm_up(client, model)
    rows, started = [], time.perf_counter()
    for case in cases:
        if budget_seconds and time.perf_counter() - started > budget_seconds:
            break
        row = run_case(case, client, model, settings)
        rows.append(row)
        log(f"  {'ok ' if row['correct'] else 'BAD'} {row['seconds']:6.1f}s {row['calls']} calls  "
            f"{row['id']:<16} {row['got']}")
    times = [r["seconds"] for r in rows] or [0.0]
    return {
        "model": model, "settings": settings.__dict__, "cases": len(cases), "done": len(rows),
        "aborted": len(rows) < len(cases),
        "accuracy": sum(r["correct"] for r in rows) / len(cases),          # unfinished cases count as wrong
        "accuracy_done": sum(r["correct"] for r in rows) / max(1, len(rows)),
        "mean_seconds": statistics.mean(times), "p90_seconds": sorted(times)[int(0.9 * (len(times) - 1))],
        "calls_per_message": sum(r["calls"] for r in rows) / max(1, len(rows)),
        "input_tokens_per_message": sum(r["input_tokens"] for r in rows) / max(1, len(rows)),
        "output_tokens_per_message": sum(r["output_tokens"] for r in rows) / max(1, len(rows)),
        "wrong_bookings": sum(r["wrong_booking"] for r in rows),
        "model_failures": sum(r["model_failed"] for r in rows),
        "rows": rows,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3.5:4b")
    p.add_argument("--num-ctx", type=int)
    p.add_argument("--split", choices=["dev", "test", "hinglish", "all"], default="test")
    p.add_argument("--prompt", default="full")
    p.add_argument("--tool-text", default="full")
    p.add_argument("--thinking", choices=["on", "off"], default="on")
    p.add_argument("--temperature", type=float)
    p.add_argument("--top-p", type=float)
    p.add_argument("--max-tokens", type=int, default=4096)
    a = p.parse_args()
    dev, test = split()
    cases = {"dev": dev, "test": test, "hinglish": HINGLISH, "all": CASES}[a.split]
    settings = ModelSettings(max_tokens=a.max_tokens, thinking=a.thinking == "on", temperature=a.temperature,
                             top_p=a.top_p, prompt=a.prompt, tool_text=a.tool_text)
    r = evaluate(cases, with_context(a.model, a.num_ctx), settings)
    print(f"accuracy {r['accuracy']:.0%}  mean {r['mean_seconds']:.1f}s  p90 {r['p90_seconds']:.1f}s  "
          f"calls/msg {r['calls_per_message']:.1f}  in/out tokens {r['input_tokens_per_message']:.0f}/"
          f"{r['output_tokens_per_message']:.0f}  wrong bookings {r['wrong_bookings']}")


if __name__ == "__main__":
    main()
