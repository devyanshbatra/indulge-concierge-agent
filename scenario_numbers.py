"""Calls, tokens, cost and time for every scenario (Part 2 "Numbers").

    python -X utf8 scenario_numbers.py                                   # fake model: token counts are estimates
    python -X utf8 scenario_numbers.py --ollama --model qwen3.5-4b-ctx2048          # tuned local model
    python -X utf8 scenario_numbers.py --ollama --model qwen3.5:4b --untuned        # before tuning
    python -X utf8 scenario_numbers.py --live --model claude-haiku-5-5               # real Claude (needs a key)

Each scenario runs through play.py's own runner. A live run is also checked
against the fake model's run of the same scenario: same tickets, same sends.
Results go to results/scenarios_<label>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import agent.concierge
import play
from agent.loop import ModelSettings
from world import SCENARIOS
from world.pricing import cost_usd

RESULTS = Path(__file__).with_name("results")


def outcome(r: dict) -> tuple:
    return r["sent"], tuple(t["status"] for t in r["tickets"]), bool(r["leaks"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="claude-haiku-5-5")
    p.add_argument("--live", action="store_true", help="real Claude through anthropic.Anthropic()")
    p.add_argument("--ollama", action="store_true", help="a local model through Ollama's Anthropic-compatible API")
    p.add_argument("--untuned", action="store_true", help="plain settings (thinking on, full prompt) for a local model")
    p.add_argument("--label")
    p.add_argument("--only", nargs="*", help="run only these scenarios")
    a = p.parse_args()
    live = a.live or a.ollama
    if a.ollama:
        os.environ.setdefault("ANTHROPIC_BASE_URL", "http://localhost:11434")
        os.environ.setdefault("ANTHROPIC_API_KEY", "ollama")
    if a.untuned:
        agent.concierge.settings_for = lambda model: ModelSettings()
    label = a.label or ("fake" if not live else a.model.replace(":", "-") + ("-untuned" if a.untuned else ""))
    names = a.only or list(SCENARIOS)

    rows = []
    print(f"{'scenario':<30}{'calls':>6}{'in tok':>8}{'out tok':>8}{'cost $':>9}{'model s':>9}{'wall s':>8}"
          f"  outcome")
    for name in names:
        reference = outcome(play.run(name, False, a.model, verbose=False)) if live else None
        started = time.perf_counter()
        r = play.run(name, live, a.model, verbose=False)
        wall = time.perf_counter() - started
        u = r["usage"]
        same = None if reference is None else outcome(r) == reference
        rows.append({"scenario": name, "sent": r["sent"], "tickets": [t["status"] for t in r["tickets"]],
                     "alerts": len(r["alerts"]), "leaks": r["leaks"], "wall_seconds": round(wall, 2),
                     "same_outcome_as_fake": same,
                     "haiku_cost_at_these_tokens": round(cost_usd("claude-haiku-5-5", u["input_tokens"],
                                                                  u["output_tokens"]), 6)} | u)
        status = ",".join(rows[-1]["tickets"]) or "-"
        check = "" if same is None else ("  same as fake" if same else "  DIFFERENT from fake")
        print(f"{name:<30}{u['calls']:>6}{u['input_tokens']:>8}{u['output_tokens']:>8}{u['cost_usd']:>9.4f}"
              f"{u['model_seconds']:>9.1f}{wall:>8.1f}  sent={r['sent']} {status} leaks={'YES' if r['leaks'] else 'no'}"
              f"{check}", flush=True)

    total = {k: sum(r[k] for r in rows) for k in ("calls", "input_tokens", "output_tokens", "cost_usd",
                                                    "model_seconds", "wall_seconds", "haiku_cost_at_these_tokens")}
    print(f"{'TOTAL':<30}{total['calls']:>6}{total['input_tokens']:>8}{total['output_tokens']:>8}"
          f"{total['cost_usd']:>9.4f}{total['model_seconds']:>9.1f}{total['wall_seconds']:>8.1f}")
    if live:
        print(f"same outcome as the fake model in {sum(bool(r['same_outcome_as_fake']) for r in rows)}/{len(rows)} "
              f"scenarios; leaks in {sum(bool(r['leaks']) for r in rows)}")
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"scenarios_{label}.json"
    out.write_text(json.dumps({"model": a.model, "live": live, "untuned": a.untuned,
                               "token_counts": "estimated by the fake (chars/4)" if not live else "reported by the API",
                               "total": total, "scenarios": rows}, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
