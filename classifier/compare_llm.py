"""Compare the small classifier with a large language model on 100 unseen messages.

    python -X utf8 -m classifier.compare_llm                 # qwen3.5:4b through a local Ollama
    python -X utf8 -m classifier.compare_llm --model claude-haiku-5-5 --anthropic   # needs ANTHROPIC_API_KEY

The 100 messages are the first 50 requests and first 50 chatter messages of
classifier/unseen.csv (hand-written, not in train.csv). Claude's cost is worked
out from its published price and the token counts measured here; with no key,
the qwen token counts stand in (Claude's tokenizer counts about 30% more).
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

from .predict import predict
from .train import CHATTER, REQUEST

HERE = Path(__file__).parent
OUT = HERE / "llm_comparison.json"
HAIKU_PRICE = (0.10, 0.50)  # USD per million input / output tokens, claude-haiku-5-5 (prompts under 100K tokens)
CLAUDE_TOKENIZER_FACTOR = 1.3

SYSTEM = """\
You sort WhatsApp messages sent to a luxury concierge desk. Answer with exactly one word.
request: the member asks the desk to do, book, find, check, change, cancel or follow up on something (including "any update?" or "it hasn't arrived").
not_request: thanks, greetings, wishes, news, photos, new contact details, or plans the member is handling themselves.
Answer request or not_request."""


def hundred() -> list[tuple[str, str]]:
    with open(HERE / "unseen.csv", encoding="utf-8", newline="") as f:
        rows = [(r["text"], r["label"]) for r in csv.DictReader(f)]
    return [r for r in rows if r[1] == REQUEST][:50] + [r for r in rows if r[1] == CHATTER][:50]


def score(said: list[str], truth: list[str]) -> dict:
    requests = [s for s, t in zip(said, truth) if t == REQUEST]
    return {"accuracy": sum(s == t for s, t in zip(said, truth)) / len(truth),
            "request_recall": requests.count(REQUEST) / len(requests),
            "missed_requests": requests.count(CHATTER),
            "false_alarms": sum(s == REQUEST and t == CHATTER for s, t in zip(said, truth))}


def ask_llm(client, model: str, text: str, local: bool) -> tuple[str, float, int, int]:
    options = {"thinking": {"type": "disabled"}}
    if local:
        options["extra_body"] = {"temperature": 0}
    started = time.perf_counter()
    reply = client.messages.create(model=model, max_tokens=16, system=SYSTEM,
                                   messages=[{"role": "user", "content": text}], **options)
    seconds = time.perf_counter() - started
    answer = "".join(b.text for b in reply.content if b.type == "text").strip().lower()
    label = CHATTER if answer.startswith("not") else REQUEST  # anything unclear counts as a request
    return label, seconds, reply.usage.input_tokens, reply.usage.output_tokens


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3.5:4b")
    p.add_argument("--num-ctx", type=int, default=2048)
    p.add_argument("--anthropic", action="store_true", help="call the real Claude API instead of Ollama")
    a = p.parse_args()
    data = hundred()
    texts, truth = [t for t, _ in data], [label for _, label in data]

    predict(["warm up"])
    started = time.perf_counter()
    small = predict(texts)
    small_batch = time.perf_counter() - started
    started = time.perf_counter()
    for t in texts:
        predict([t])
    small_each = (time.perf_counter() - started) / len(texts)

    if a.anthropic:
        import anthropic
        client, model, local = anthropic.Anthropic(), a.model, False
    else:
        from tuning.evaluate import ollama_client, warm_up, with_context
        client, model, local = ollama_client(), with_context(a.model, a.num_ctx), True
        warm_up(client, model)
    llm, seconds, tokens_in, tokens_out = [], [], [], []
    for text, label in data:
        said, s, i, o = ask_llm(client, model, text, local)
        llm.append(said)
        seconds.append(s)
        tokens_in.append(i)
        tokens_out.append(o)
        print(f"  {'ok ' if said == label else 'BAD'} {s:5.2f}s  {label:<12} {text}", flush=True)

    factor = 1.0 if a.anthropic else CLAUDE_TOKENIZER_FACTOR
    per_1000_in = statistics.mean(tokens_in) * factor * 1000
    per_1000_out = statistics.mean(tokens_out) * factor * 1000
    claude_cost = (per_1000_in * HAIKU_PRICE[0] + per_1000_out * HAIKU_PRICE[1]) / 1e6
    result = {
        "messages": len(texts),
        "small_model": score(small, truth) | {"ms_per_message_one_by_one": round(small_each * 1000, 3),
                                              "ms_per_message_batch_of_100": round(small_batch * 10, 3),
                                              "cost_per_1000_usd": 0.0},
        "llm": score(llm, truth) | {"model": model, "mean_seconds": round(statistics.mean(seconds), 3),
                                    "p90_seconds": round(sorted(seconds)[int(0.9 * len(seconds)) - 1], 3),
                                    "input_tokens_per_message": statistics.mean(tokens_in),
                                    "output_tokens_per_message": statistics.mean(tokens_out),
                                    "cost_per_1000_usd": 0.0 if local else round(claude_cost, 4)},
        "claude_haiku_5_5_estimate": {"cost_per_1000_usd": round(claude_cost, 4),
                                      "basis": "measured" if a.anthropic else
                                      f"qwen token counts x{CLAUDE_TOKENIZER_FACTOR} at ${HAIKU_PRICE[0]}/"
                                      f"${HAIKU_PRICE[1]} per million in/out tokens; not run (no API key)"},
        "llm_wrong": [f"{label}: {t}" for (t, label), s in zip(data, llm) if s != label],
        "small_model_wrong": [f"{label}: {t}" for (t, label), s in zip(data, small) if s != label],
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    s, m = result["small_model"], result["llm"]
    print(f"\n{'':<14}{'accuracy':>9}{'recall':>8}{'missed':>8}{'false':>7}{'time/msg':>11}{'$/1000':>9}")
    print(f"{'small model':<14}{s['accuracy']:>9.0%}{s['request_recall']:>8.0%}{s['missed_requests']:>8}"
          f"{s['false_alarms']:>7}{s['ms_per_message_one_by_one']:>9.2f}ms{0:>9.2f}")
    print(f"{model[:14]:<14}{m['accuracy']:>9.0%}{m['request_recall']:>8.0%}{m['missed_requests']:>8}"
          f"{m['false_alarms']:>7}{m['mean_seconds'] * 1000:>9.0f}ms{m['cost_per_1000_usd']:>9.2f}")
    print(f"Claude Haiku 5.5, estimated: ${claude_cost:.4f} per 1,000 messages ({result['claude_haiku_5_5_estimate']['basis']})")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
