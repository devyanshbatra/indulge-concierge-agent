"""Tune how we call each local Qwen size, with Optuna: accuracy up, seconds per message down.

    python -X utf8 -m tuning.tune_qwen search --model qwen3.5:2b --trials 12   # one study per model, resumable
    python -X utf8 -m tuning.tune_qwen report                                  # measure the picks on held-out cases

Each model gets its own multi-objective study (accuracy, mean seconds per message)
on the English dev half, so each ends with its own best settings. Thinking is
fixed off: a probe showed the same tool call at ~5 s instead of ~19 s, and the
untuned baseline in `report` (thinking on) measures what it costs.

`report` picks, per model, the fastest setting whose dev accuracy is within one
case of that model's best, then measures it on the held-out English test half
(and on the Hinglish side set), plus the untuned baseline, so the numbers we
quote were not tuned on.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import optuna

from agent.loop import ModelSettings
from agent.profiles import PROFILES_FILE

from .cases import HINGLISH, split
from .evaluate import evaluate, with_context

HERE = Path(__file__).parent
STORAGE = f"sqlite:///{(HERE / 'optuna.db').as_posix()}"
STUDY_PREFIX = "qwen-intake-v3"
RESULTS = HERE / "results.json"
TRIAL_BUDGET_SECONDS = 12 * 60   # a slower setting is stopped; its unfinished cases count as wrong
BASELINE_BUDGET_SECONDS = 25 * 60
MODELS = ["qwen3.5:0.8b", "qwen3.5:2b", "qwen3.5:4b"]
BASELINE = ("qwen3.5:4b", ModelSettings())  # before tuning: thinking on, full prompt, ~5 calls per booking


def study(model: str) -> optuna.Study:
    return optuna.create_study(study_name=f"{STUDY_PREFIX}-{model}", storage=STORAGE, load_if_exists=True,
                               directions=["maximize", "minimize"],
                               sampler=optuna.samplers.TPESampler(seed=42, multivariate=True, n_startup_trials=5))


def suggest(trial: optuna.Trial) -> tuple[int, ModelSettings]:
    num_ctx = trial.suggest_categorical("num_ctx", [2048, 4096, 8192])
    prompt = trial.suggest_categorical("prompt", ["full", "lean", "lean_compact"])
    tool_text = trial.suggest_categorical("tool_text", ["full", "short"])
    temperature = trial.suggest_float("temperature", 0.0, 1.0, step=0.1)
    top_p = trial.suggest_float("top_p", 0.5, 1.0, step=0.05)
    max_tokens = trial.suggest_categorical("max_tokens", [256, 512, 1024])
    return num_ctx, ModelSettings(max_tokens=max_tokens, thinking=False, temperature=temperature, top_p=top_p,
                                  prompt=prompt, tool_text=tool_text)


def settings_from(params: dict) -> ModelSettings:
    return ModelSettings(max_tokens=params["max_tokens"], thinking=False, temperature=params["temperature"],
                         top_p=params["top_p"], prompt=params["prompt"], tool_text=params["tool_text"])


def search(model: str, trials: int) -> None:
    s = study(model)
    dev, _ = split()

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        num_ctx, settings = suggest(trial)
        print(f"\n[{model} trial {trial.number}] ctx={num_ctx} {settings}", flush=True)
        started = time.perf_counter()
        r = evaluate(dev, with_context(model, num_ctx), settings, budget_seconds=TRIAL_BUDGET_SECONDS,
                     log=lambda line: print(line, flush=True))
        for key in ("accuracy_done", "p90_seconds", "calls_per_message", "input_tokens_per_message",
                    "output_tokens_per_message", "wrong_bookings", "model_failures", "aborted", "done"):
            trial.set_user_attr(key, r[key])
        trial.set_user_attr("misses", [f"{row['id']}: {row['got']}" for row in r["rows"] if not row["correct"]])
        print(f"[{model} trial {trial.number}] accuracy {r['accuracy']:.0%}  mean {r['mean_seconds']:.1f}s  "
              f"({time.perf_counter() - started:.0f}s{', STOPPED' if r['aborted'] else ''})", flush=True)
        return r["accuracy"], r["mean_seconds"]

    if not s.trials:  # start from the setting the smoke tests liked
        s.enqueue_trial({"num_ctx": 4096, "prompt": "lean", "tool_text": "full", "temperature": 0.0,
                         "top_p": 1.0, "max_tokens": 512})
    remaining = trials - sum(t.state == optuna.trial.TrialState.COMPLETE for t in s.trials)
    if remaining > 0:
        s.optimize(objective, n_trials=remaining, catch=(Exception,))
    show_front(model, s)


def show_front(model: str, s: optuna.Study) -> None:
    print(f"\nPareto front on dev, {model}:")
    for t in sorted(s.best_trials, key=lambda t: t.values[1]):
        print(f"  #{t.number:<3} acc {t.values[0]:.0%}  {t.values[1]:5.1f}s  {t.params}")


def pick(s: optuna.Study) -> optuna.trial.FrozenTrial | None:
    """The fastest finished trial whose dev accuracy is within one case of the best, with no wrong booking."""
    done = [t for t in s.trials if t.state == optuna.trial.TrialState.COMPLETE and not t.user_attrs.get("aborted")]
    if not done:
        return None
    best = max(t.values[0] for t in done)
    one_case = 1 / len(split()[0])
    near = [t for t in done if t.values[0] >= best - one_case - 1e-9]
    safe = [t for t in near if t.user_attrs.get("wrong_bookings", 1) == 0]
    return min(safe or near, key=lambda t: t.values[1])


def _summary(r: dict) -> dict:
    keep = ("model", "settings", "cases", "done", "aborted", "accuracy", "accuracy_done", "mean_seconds",
            "p90_seconds", "calls_per_message", "input_tokens_per_message", "output_tokens_per_message",
            "wrong_bookings", "model_failures")
    return {k: r[k] for k in keep} | {"misses": [f"{x['id']}: {x['got']}" for x in r["rows"] if not x["correct"]]}


def report(baseline: bool) -> None:
    _, test = split()
    results, profiles = {}, {}
    for model in MODELS:
        s = study(model)
        chosen = pick(s)
        if chosen is None:
            print(f"\n{model}: no finished trials, skipped")
            continue
        show_front(model, s)
        name = with_context(model, chosen.params["num_ctx"])
        settings = settings_from(chosen.params)
        print(f"\n{model}: trial #{chosen.number} -> {name} {settings}\n  held-out English test half:")
        tuned = evaluate(test, name, settings)
        print("  Hinglish side check (not tuned on):")
        hinglish = evaluate(HINGLISH, name, settings)
        results[model] = {"ollama_model": name, "trial": chosen.number, "params": chosen.params,
                          "dev_accuracy": chosen.values[0], "dev_mean_seconds": chosen.values[1],
                          "test": _summary(tuned), "hinglish": _summary(hinglish)}
        profile = {"settings": settings.__dict__, "tuned_on": "tuning/cases.py English dev half",
                   "test_accuracy": round(tuned["accuracy"], 3), "test_mean_seconds": round(tuned["mean_seconds"], 2)}
        profiles[model] = profiles[name] = profile
    if baseline:
        print("\nBaseline (untuned: 4b, thinking on, full prompt) on the same test half:")
        results["baseline"] = {"test": _summary(evaluate(test, BASELINE[0], BASELINE[1],
                                                         budget_seconds=BASELINE_BUDGET_SECONDS))}
    RESULTS.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    PROFILES_FILE.write_text(json.dumps(profiles, indent=2) + "\n", encoding="utf-8")

    print(f"\n{'':<14}{'test acc':>9}{'mean s':>8}{'p90 s':>7}{'calls':>7}{'wrong':>7}{'hinglish':>10}")
    for model, r in results.items():
        t = r["test"]
        h = f"{r['hinglish']['accuracy']:.0%}" if "hinglish" in r else "-"
        done = "" if not t["aborted"] else f"  (stopped after {t['done']}/{t['cases']}; acc on those {t['accuracy_done']:.0%})"
        print(f"{model:<14}{t['accuracy']:>9.0%}{t['mean_seconds']:>8.1f}{t['p90_seconds']:>7.1f}"
              f"{t['calls_per_message']:>7.1f}{t['wrong_bookings']:>7}{h:>10}{done}")
    print(f"wrote {PROFILES_FILE} and {RESULTS}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("search")
    s.add_argument("--model", required=True, choices=MODELS)
    s.add_argument("--trials", type=int, default=12)
    rep = sub.add_parser("report")
    rep.add_argument("--no-baseline", action="store_true")
    a = p.parse_args()
    if a.command == "search":
        search(a.model, a.trials)
    else:
        report(baseline=not a.no_baseline)


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    main()
