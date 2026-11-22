# Concierge booking agent

An AI agent that sits between a luxury concierge desk's members (WhatsApp) and an outside booking bot. It
reads a member's message, opens a ticket, prepares the job, waits for a human to approve it, sends it exactly
once without any private details, and checks whatever the bot says back. Built for an AI Engineer take-home
assignment; the original brief is in [`README.md`](../README.md), my write-up is in **[`NOTES.md`](../NOTES.md)**.

```
member message -> [small classifier: obvious chatter? stop] -> model + tools (own loop, no frameworks)
  -> ticket -> proposal (nothing sent) -> human approves -> leak check -> sent once
  -> vendor reply read by code -> confirmed, or a human
```

**The model only reads messages. Everything that can hurt (what leaves, when, how often, what counts as
confirmed) is plain code**, so the house rules hold whichever model is plugged in.

## Built with no API key, on a laptop

No Anthropic key, so everything ran on an open-source model (**Qwen 3.5 via Ollama**) on a **4 GB RTX 3050
laptop GPU**. The agent uses the official `anthropic` SDK and runs unchanged on the offline `FakeAnthropic`,
`anthropic.Anthropic()` and Ollama's Anthropic-compatible API.

| | Result |
| --- | --- |
| Agent (qwen3.5:4b, tuned) | **100%** on held-out English cases, 9/10 on fresh ones, **6.8 s per message** (from ~141 s untuned) |
| All 11 scenarios, before -> after tuning | **54 -> 14** model calls, **1,659 s -> 122 s** |
| Safety | **0 leaks, 0 double sends, 0 wrong bookings** in every run |
| Tests | **367** (`pytest -q`), including fuzz tests on every entry point |
| Small model (request vs chatter) | catches **99%** of real requests in unseen phrasings, ~3 ms on a CPU |

## What is inside

| Folder | What |
| --- | --- |
| [`agent/`](../agent) | The `Concierge`: own tool loop, intake and cross-check, tools, leak scanner, vendor-reply reader, tuned model profiles |
| [`classifier/`](../classifier) | TF-IDF + logistic regression request classifier (`predict.py`, `train.py`), label cleaning, LLM comparison, a fine-tuned bge-small experiment |
| [`tuning/`](../tuning) | Optuna search over Qwen 0.8b / 2b / 4b settings, labelled cases, held-out evaluation |
| [`tests/`](../tests) | 367 tests: house rules, messy cases, unseen messages, robustness |
| [`results/`](../results) | Per-scenario calls, tokens, cost and time (fake, untuned and tuned) |
| `world/`, `data/`, `play.py` | The simulated world from the assignment, unchanged |

## Run it

```bash
pip install -r requirements.txt
pytest -q                       # 367 tests, offline
python play.py all              # every scenario on the fake model
python scenario_numbers.py      # calls, tokens, cost, time per scenario
python -m classifier.train      # retrain the classifier (~3 s)
```

On Windows add `-X utf8` (or set `PYTHONUTF8=1`). Running a local model: see "How to run" in
[`NOTES.md`](../NOTES.md).

Everything tried, what was kept and why (thinking off, a one-tool-call flow, Optuna per model size, label
cleaning, frozen and fine-tuned embeddings, int8 ONNX), what breaks at scale and what is unfinished: **[`NOTES.md`](../NOTES.md)**.
