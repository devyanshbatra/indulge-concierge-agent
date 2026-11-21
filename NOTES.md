# Notes

**Setup.** I had no Anthropic API key, so everything ran on my laptop: an open-source model (Qwen 3.5, via
Ollama) on a 4 GB RTX 3050 laptop GPU. That shaped the work. A small local model is slower and less reliable
than Claude, so the agent leans on code: it asks the model one small question per message, checks every
answer, and keeps all sending, approval and verification in code. The same code runs unchanged on
`FakeAnthropic` and `anthropic.Anthropic()`. On that hardware:

- **Agent:** a 4B model handles booking messages at **100% on held-out English cases** (9/10 on fresh ones written
  later), in **6.8 s per message** (down from ~141 s untuned), with **no wrong bookings and no leaks** in any run.
- **Tests:** 367 automated tests, covering every house rule.
- **Small model:** catches **99% of real requests** in phrasings it never saw, in ~3 ms on a CPU, and skips the
  model for obvious chatter.

Everything below is measured, not estimated, unless it says otherwise.

**How to run** (on Windows add `-X utf8`, see the last section):

```bash
pip install -r requirements.txt
pytest -q                                   # 367 tests, offline, ~10 s
python play.py all                          # every scenario on the fake model
python scenario_numbers.py                  # calls, tokens, cost and time per scenario
python -m classifier.train                  # retrain the small model (~3 s) and print each step
python -c "from classifier.predict import predict; print(predict(['Book Koen for Saturday 8pm', 'thanks!']))"

# a local model instead of the fake (Ollama running, `ollama pull qwen3.5:4b`):
python -m tuning.evaluate --model qwen3.5:4b --num-ctx 2048 --profile --split test fresh   # creates qwen3.5-4b-ctx2048
python scenario_numbers.py --ollama --model qwen3.5-4b-ctx2048      # all scenarios on the tuned profile
```

## How the agent is put together

```
member message -> [small classifier: obvious chatter? stop] -> model + tools (our own loop)
  -> ticket -> proposal (nothing sent) -> human approves -> leak check -> sent once
  -> vendor reply read by code -> confirmed, or a human
```

The model only *reads* member messages and calls tools. Everything that can hurt us is plain code:

- **What leaves.** A job's first message is written from ticket fields by a template (`agent/outbound.py`), not
  by the model, and booked under the house name. The model can only suggest a follow-up on a job already sent,
  and that also waits for a human. `LeakScanner` (`agent/leaks.py`) checks every outgoing message against every
  member's names, phones (any format), emails, addresses and cards, and blocks it on a hit. The approval screen
  shows the exact text and what was held back.
- **Exactly once.** Member and vendor messages are de-duplicated by id; each send has an idempotency key that
  retries reuse; approving twice is a no-op; a repeated tool call cannot open a second ticket or proposal.
- **Trust, but verify.** Vendor replies never reach the model (`agent/verify.py`). A confirmation is compared
  with the ticket field by field; a mismatch, a change, a request for personal details or anything unclear goes
  to a human.
- **Silence.** One reminder after 20 minutes, a human after 45. No more messages.
- **When unsure, ask.** The model's reading of a message is cross-checked against a plain parser; if they
  disagree on venue, date, time or party size, the field is left empty and a person is told. Code safety nets
  catch requests, cancellations and changes the model ignored.
- **Own loop** (`agent/loop.py`): `client.messages.create` only, so the same code runs on `FakeAnthropic`,
  `anthropic.Anthropic()` and a local model behind Ollama's Anthropic-compatible API.

## Tests (Part 2)

`pytest -q`: 367 tests, all on the fake model or scripted clients, no network. Besides the five the brief
asks for (a leak, a double send, a wrong confirmation, a silent vendor, a vendor asking for personal details),
they cover duplicate WhatsApp messages, double approval, timeouts and retries with the same idempotency key, a
model that loops, crashes or writes private data, cancellations before and after sending, prompt injection in a
member message, vendor replies that change the booking or are unclear, and messages none of the scenarios use
(`tests/test_unseen_messages.py`). Leak checks use an independent scanner (`tests/support.py`), not the agent's
own, and a test proves it catches a planted leak. `tests/test_robustness.py` (~100 cases) throws odd input at
every entry point: empty, huge or control-character messages, impossible dates, malformed and hostile tool
calls, odd vendor replies, garbage approval ids, eight people approving at the same moment (one send), and a
broken ticket during the timers. Between them, these tests and the live runs found two crashes (a list passed
as an approval id; "31 Nov" reaching the chatter gate), both fixed.

## Numbers (Part 2)

I had no Anthropic API key, so "live" means **qwen3.5:4b on my laptop** (RTX 3050 4 GB, Ryzen 7 4800H) through
Ollama. Its token counts and times are real. The fake model's token counts are estimates (characters / 4).
Cost for qwen is $0; the "Haiku cost" column prices qwen's real token counts at Claude Haiku 5.5 rates.

`python scenario_numbers.py` (all 11 scenarios; per-scenario rows in `results/`):

| run | calls | input tok | output tok | cost | model time | same outcome as fake | leaks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fake model (estimates) | 59 | 93,152 | 4,133 | $0.0114 (Haiku rates) | - | - | 0 |
| qwen3.5:4b **untuned** (before) | 54 | 12,789 | 14,866 | $0 (Haiku rates: $0.0087) | 1,659 s | 9/11 | 0 |
| qwen3.5:4b **tuned** (after) | **14** | 6,264 | 1,412 | $0 (Haiku rates: **$0.0013**) | **122 s** | 10/11 | 0 |

Every live run kept the house rules (0 leaks, nothing sent without approval). Where an outcome differs from the
fake model it is the cautious direction: untuned, `riya_corner_table` and `sameer_unclear` ended with a person
instead of a sent job or a `needs_info` ticket; tuned, `kabir_hinglish_double` misread "9 baje", so nothing was
sent and a person was asked. Untuned output tokens are mostly the model's hidden reasoning.

**The change that cut time: let code do the work, and ask the model one small question.** A small local model
with the default setup (thinking on, full prompt: the model looks up the member, the vendors and writes the
proposal) took ~5 calls and ~141 s per booking message. Now (tuned profile, `agent/model_profiles.json`):
thinking off, a short "lean" prompt where the model makes **one** tool call (create / cancel / escalate) and
code picks the vendor and writes the proposal, plus the classifier gate for chatter. On the 24 held-out
English cases: **141 s -> 6.8 s per message, 4.9 -> 1.0 calls, 0 wrong bookings** (the untuned run finished
only 11 of 24 cases inside 25 minutes). The gate alone skipped the model for 6 of 10 chatter messages (0 s
instead of ~0.8 s) and never skipped a request.

**Tuning, honestly.** Optuna (multi-objective: accuracy up, seconds down; TPE sampler) searched context length,
prompt, tool-description length, temperature, top_p and max_tokens, one study per model size, 12 trials each, on
26 hand-labelled English messages (`tuning/cases.py`, dev half). Results on the 24-message test half it never saw:

| model | test accuracy | mean s / msg | wrong bookings |
| --- | --- | --- | --- |
| qwen3.5:0.8b | 54% | 0.7 | 1 |
| qwen3.5:2b | 62% | 4.8 | 2 |
| qwen3.5:4b | 83% -> **100%** after the date fix below | 9.1 -> **6.8** | 0 |

4b was 100% on dev but 83% on test: every miss was "tomorrow" or "day after tomorrow". The calendar we give the
model did not say which date is tomorrow, and `world/parse.py` reads "day after tomorrow" as tomorrow. After
fixing both I had looked at the test half, so I wrote 10 new messages (`FRESH`): **9/10**, 7.6 s per message,
0 wrong bookings; the miss ("This Wednesday" read as Thursday) was caught by the parser cross-check and sent to a
person. Hinglish (a side check only; English is the main language): 3/4, the miss ("9 baje" read as 09:00) caught
the same way. Final run: `tuning/final_4b_gate.json`. The smaller models are not safe to use: they invent
bookings. Prompt caveat: the lean prompt's example "something nice for the weekend?" is close to the
`sameer_unclear` scenario message.

## The small model (Part 3)

`classifier/` — TF-IDF (word 1-2 grams + character 2-5 grams) and logistic regression; `predict.py` runs offline
in ~3 ms. `python -m classifier.train` reproduces it in ~3 s and prints every step.

- **The data**: 2,000 rows but only 784 templates, so every score is cross-validated **by template**; a random
  split puts the same message on both sides and looks better than it is.
- **Wrong labels**: 27 rows take their template's majority label; 27 more that the cross-validated model gives
  under 25% are left out. All 54 are listed in `classifier/flagged_labels.csv`; read by hand, each is a real
  mislabel ("Please cancel the Mirabelle table" marked chatter).
- **Threshold** 0.48, chosen counting a missed request as 5x a false alarm (my assumption).
- **Unseen phrasings** (`classifier/unseen.csv`, 120 messages I wrote, plus the agent's tuning cases; 184 in all):
  94.6% accurate, 99.1% of requests caught. Errors are mostly "I'll get back to you about Sunday"-type chatter
  flagged as a request, the cheap direction.
- **Tried a transformer** (`classifier/finetune.py`): fine-tuned `BAAI/bge-small-en-v1.5` on the same cleaned
  data and folds, exported to ONNX (fp32 133 MB, int8 34 MB, ~5-7 ms per message on CPU). It understands meaning
  better (1 false alarm vs 9) but missed **4-10** requests on the unseen set against TF-IDF's 1, mostly terse
  bookings ("Koen, this Sat, half past eight, four of us" scored 0.08): the training data has 784 templates, so
  it learns them and grows over-confident. A run with less training and label smoothing, settings fixed before
  running, did not fix it. Averaging it with TF-IDF scored best (98.4%, 2 missed, 1 false alarm) but by one or two
  messages in 184, for a 34-133 MB model plus onnxruntime on your machine. So `predict()` stays TF-IDF.
  Mixed precision vs fp32 training made no difference at this size.

**Against a large model**, 100 unseen messages (`python -m classifier.compare_llm`):

| | accuracy | requests caught | false alarms | time / message | cost / 1,000 |
| --- | --- | --- | --- | --- | --- |
| small model | 94% | 100% | 6 | 3.4 ms (CPU) | $0 |
| qwen3.5:4b | 99% | 100% | 1 | 648 ms (GPU) | $0 local |
| Claude Haiku 5.5 | not run (no key) | | | | ~$0.017, **estimated** from qwen's tokens x1.3 |

I wrote the LLM prompt after seeing the small model's errors, which favours the LLM a little.

**Where it fits:** as a gate in front of the model, never as the last word. A message skips the model only if
the classifier gives p(request) < 0.10 **and** plain code sees no request, cancellation or change; if the
classifier fails, nothing is skipped. No real request in cross-validation or the unseen set scored below 0.21;
the gate skips 34-76% of chatter. **To prove it safe:** run it in shadow mode first (log what it would skip,
let the model decide), sample the skipped messages for human review each week, and only switch it on when
thousands of skips contain no request (0 misses in 3,000 means under ~0.1% at 95% confidence); keep watching
the share of skipped messages for drift.

## Everything we tried, in one place

All on one laptop: RTX 3050 with 4 GB of VRAM (qwen3.5:4b does not fit, so Ollama runs it half on the GPU and
half on the Ryzen 7 CPU), no API key, every Optuna trial capped at 12 minutes.

| what we tried | result | kept? |
| --- | --- | --- |
| Thinking on vs off (qwen 4b) | same tool call, ~19 s -> ~5 s per call | off |
| Full flow (model calls 5 tools) vs lean flow (one tool call, code does the rest) | 4.9 -> 1.0 calls, 141 -> 6.8 s per message, fewer slips | lean |
| First lean prompt ("if unsure, notify_human") | sent chatter like "Thanks!" to a person (85% on dev) | rewritten |
| Optuna, 12 trials x 3 sizes: context, prompt, tool text, temperature, top_p, max_tokens | 4b best; 2b and 0.8b invent bookings | 4b profile |
| Optuna's temperature 0.8 vs 0.1 | same accuracy and speed on dev | 0.1, repeatable |
| Calendar spells out tomorrow; agent reads "day after tomorrow" itself | held-out test 83% -> 100% | kept |
| Classifier: 4 TF-IDF / naive Bayes variants | all ~97% grouped CV; word+char best | word+char LR |
| Label cleaning: template majority + cross-validated flagging | 54 wrong labels found, listed in a file | kept |
| Frozen embeddings (MiniLM, bge-small) + logistic regression | alone: 4-6 missed requests vs TF-IDF's 1; bge averaged with TF-IDF: 0 missed, 8 false alarms | not shipped (download, size) |
| Fine-tuned bge-small, 3 runs (mixed precision, fp32, smoothing + 2 epochs), on the 3050: 5-8 s per fold, under 760 MB of GPU memory | 4-10 missed requests on new wording | not shipped |
| ONNX int8 vs fp32 for that model | 34 vs 133 MB, ~1.5x faster on CPU, but moved some scores by up to 0.5 | - |
| qwen 4b as the classifier vs the small model (100 messages) | 99% vs 94%; 648 ms vs 3.4 ms | small model, as a gate |
| Gate threshold 0.10 | skipped 6 of 10 chatter messages, never a request | kept |
| ~100 fuzz tests on every entry point | found one crash (live runs found another), both fixed | kept |

## At 1,000 members and 50 bookings a day, what breaks first

1. **State lives in memory.** A restart loses tickets, proposals, the de-dup ids and the idempotency keys, so
   it could re-send a job. First change: a database with unique constraints on message id and idempotency key,
   and an outbox table for sends.
2. **One lock around everything, including the model call.** A 7 s model call blocks every other group and
   vendor reply. Move the model call outside the lock, lock per ticket, put messages on a queue with workers.
3. **The approval step is a list in memory.** It needs a real queue with owners, timeouts and audit.
4. **Venues and vendors are a fixed list** in `house.json`; the parser cross-check only knows those names.

## Unfinished, and what I would do with another day

- Never run on real Claude (no key). The full prompt (default for Claude) is tested only against the fake.
- `qwen3.5:4b` misread "This Wednesday" once in 10 fresh cases (caught, sent to a person); Hinglish is weaker.
- The tuning sets are small (26 dev / 24 test / 10 fresh); I would grow them from real traffic.
- Run on Windows with `PYTHONUTF8=1` (or `python -X utf8`): `world/data.py` opens files with the system encoding.
- Time: about 8 hours of continuous work (about 15:00 to 23:00), well over the 4 hours asked for. Most of the
  extra went into running a local model, since I could not use an API key, and into tuning it.

## How I used AI tools

I used an AI coding assistant throughout. I set the direction and made the decisions (open-source models only,
English first, a held-out test set, tuning each Qwen size, trying a fine-tuned transformer for Part 3) and
reviewed the results; it wrote most of the code and tests and ran the experiments. Every number here comes from a
script in this repository.
