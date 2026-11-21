"""Tune how we call a local Qwen model, with Optuna: accuracy up, seconds per message down.

    python -X utf8 -m tuning.tune_qwen search --trials 40     # tune on the dev half (resumable)
    python -X utf8 -m tuning.tune_qwen report                 # pick a setting, measure it on the test half

The search is multi-objective (accuracy, mean seconds per message), so the result
is a Pareto front, not one number. `report` picks the fastest setting whose dev
accuracy is within one case of the best, then measures it and the untuned
baseline on the held-out test half, so the numbers we quote were not tuned on.
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
STUDY = "qwen-intake-v2"  # v1 was stopped after one trial: the lean prompts told the model to escalate chatter
RESULTS = HERE / "results.json"
TRIAL_BUDGET_SECONDS = 12 * 60  # a slower setting is stopped; its unfinished cases count as wrong
MODELS = ["qwen3.5:0.8b", "qwen3.5:2b", "qwen3.5:4b"]


def suggest(trial: optuna.Trial) -> tuple[str, int, ModelSettings]:
    model = trial.suggest_categorical("model", MODELS)
    num_ctx = trial.suggest_categorical("num_ctx", [2048, 4096, 8192])
    thinking = trial.suggest_categorical("thinking", [False, True])
    prompt = trial.suggest_categorical("prompt", ["full", "lean", "lean_compact"])
    tool_text = trial.suggest_categorical("tool_text", ["full", "short"])
    temperature = trial.suggest_float("temperature", 0.0, 1.0, step=0.1)
    top_p = trial.suggest_float("top_p", 0.5, 1.0, step=0.05)
    # a thinking model needs room to think before its tool call
    max_tokens = 4096 if thinking else trial.suggest_categorical("max_tokens", [256, 512, 1024])
    settings = ModelSettings(max_tokens=max_tokens, thinking=thinking, temperature=temperature, top_p=top_p,
                             prompt=prompt, tool_text=tool_text)
    return model, num_ctx, settings


def objective(trial: optuna.Trial) -> tuple[float, float]:
    model, num_ctx, settings = suggest(trial)
    dev, _ = split()
    print(f"\n[trial {trial.number}] {model} ctx={num_ctx} {settings}", flush=True)
    started = time.perf_counter()
    r = evaluate(dev, with_context(model, num_ctx), settings, budget_seconds=TRIAL_BUDGET_SECONDS,
                 log=lambda line: print(line, flush=True))
    for key in ("accuracy_done", "p90_seconds", "calls_per_message", "input_tokens_per_message",
                "output_tokens_per_message", "wrong_bookings", "model_failures", "aborted", "done"):
        trial.set_user_attr(key, r[key])
    trial.set_user_attr("misses", [f"{row['id']}: {row['got']}" for row in r["rows"] if not row["correct"]])
    print(f"[trial {trial.number}] accuracy {r['accuracy']:.0%}  mean {r['mean_seconds']:.1f}s  "
          f"({time.perf_counter() - started:.0f}s total{', STOPPED' if r['aborted'] else ''})", flush=True)
    return r["accuracy"], r["mean_seconds"]


def study() -> optuna.Study:
    return optuna.create_study(study_name=STUDY, storage=STORAGE, load_if_exists=True,
                               directions=["maximize", "minimize"],
                               sampler=optuna.samplers.TPESampler(seed=42, multivariate=True))


def search(trials: int) -> None:
    s = study()
    if not s.trials:  # start from what the smoke tests suggested, so TPE has good points early
        for model in ("qwen3.5:4b", "qwen3.5:2b"):
            s.enqueue_trial({"model": model, "num_ctx": 4096, "thinking": False, "prompt": "lean",
                             "tool_text": "full", "temperature": 0.0, "top_p": 1.0, "max_tokens": 512})
    s.optimize(objective, n_trials=trials, catch=(Exception,))
    show_front(s)


def show_front(s: optuna.Study) -> None:
    print("\nPareto front (dev):")
    for t in sorted(s.best_trials, key=lambda t: t.values[1]):
        print(f"  #{t.number:<3} acc {t.values[0]:.0%}  {t.values[1]:5.1f}s  {t.params}")


def pick(s: optuna.Study) -> optuna.trial.FrozenTrial:
    """The fastest finished trial whose dev accuracy is within one case of the best."""
    done = [t for t in s.trials if t.state == optuna.trial.TrialState.COMPLETE and not t.user_attrs.get("aborted")]
    best = max(t.values[0] for t in done)
    one_case = 1 / len(split()[0])
    near = [t for t in done if t.values[0] >= best - one_case - 1e-9 and t.user_attrs.get("wrong_bookings", 1) == 0]
    return min(near or done, key=lambda t: t.values[1])


def settings_from(params: dict) -> ModelSettings:
    return ModelSettings(max_tokens=params.get("max_tokens", 4096), thinking=params["thinking"],
                         temperature=params["temperature"], top_p=params["top_p"], prompt=params["prompt"],
                         tool_text=params["tool_text"])


def report(baseline: bool) -> None:
    s = study()
    show_front(s)
    chosen = pick(s)
    model = with_context(chosen.params["model"], chosen.params["num_ctx"])
    settings = settings_from(chosen.params)
    _, test = split()
    print(f"\nChosen: trial #{chosen.number} -> {model} {settings}\nOn the held-out test half:")
    tuned = evaluate(test, model, settings)
    runs = {"tuned": tuned}
    print("\nSide check, Hinglish (not tuned on):")
    runs["tuned_hinglish"] = evaluate(HINGLISH, model, settings)
    if baseline:  # the settings before tuning: 4b, thinking on, full prompt, 5 calls per booking
        print("\nBaseline (untuned) on the same test half:")
        runs["baseline"] = evaluate(test, "qwen3.5:4b", ModelSettings())
    summary = {name: {k: v for k, v in r.items() if k != "rows"} | {"misses": [f"{x['id']}: {x['got']}"
               for x in r["rows"] if not x["correct"]]} for name, r in runs.items()}
    summary["chosen_trial"] = {"number": chosen.number, "params": chosen.params, "dev_accuracy": chosen.values[0],
                               "dev_mean_seconds": chosen.values[1]}
    RESULTS.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    profile = {"settings": settings.__dict__, "tuned_on": "tuning/cases.py dev half",
               "test_accuracy": round(tuned["accuracy"], 3), "test_mean_seconds": round(tuned["mean_seconds"], 2)}
    profiles = {model: profile, chosen.params["model"]: profile}
    PROFILES_FILE.write_text(json.dumps(profiles, indent=2) + "\n", encoding="utf-8")
    for name, r in runs.items():
        print(f"{name:<9} accuracy {r['accuracy']:.0%}  mean {r['mean_seconds']:.1f}s  p90 {r['p90_seconds']:.1f}s  "
              f"calls/msg {r['calls_per_message']:.1f}  wrong bookings {r['wrong_bookings']}")
    print(f"wrote {PROFILES_FILE} and {RESULTS}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("search").add_argument("--trials", type=int, default=40)
    rep = sub.add_parser("report")
    rep.add_argument("--no-baseline", action="store_true")
    a = p.parse_args()
    if a.command == "search":
        search(a.trials)
    else:
        report(baseline=not a.no_baseline)


if __name__ == "__main__":
    main()
