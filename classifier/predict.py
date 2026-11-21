"""Is a member's message a real request, or chatter?

    from classifier.predict import predict
    predict(["Book Koen for Saturday 8pm", "thanks!!"])  # -> ["request", "not_request"]

Word and character TF-IDF with logistic regression (see train.py). Runs offline
on a laptop CPU in well under a millisecond per message. The threshold was set
so that a missed request counts five times a false alarm, so it leans towards
"request" when unsure.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import joblib

try:
    from .train import CHATTER, REQUEST, build, clean
except ImportError:  # run as a plain script from inside classifier/
    from train import CHATTER, REQUEST, build, clean

MODEL_FILE = Path(__file__).with_name("model.joblib")
DEFAULT_THRESHOLD = 0.48  # what train.py picked; only used if the saved model cannot be read

_bundle: dict | None = None


def _load() -> dict:
    global _bundle
    if _bundle is None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # a newer scikit-learn warns when unpickling; we check it works below
                bundle = joblib.load(MODEL_FILE)
                bundle["model"].predict_proba(["ok"])
            _bundle = bundle
        except Exception:  # no file, or a scikit-learn version that cannot read it: retrain (~3 s)
            _bundle = build(DEFAULT_THRESHOLD)
    return _bundle


def request_scores(texts: list[str]) -> list[float]:
    """p(request) for each message."""
    if not texts:
        return []
    bundle = _load()
    model = bundle["model"]
    column = list(model.classes_).index(REQUEST)
    return [float(p) for p in model.predict_proba([clean(str(t or "")) for t in texts])[:, column]]


def predict(texts: list[str]) -> list[str]:
    threshold = _load()["threshold"] if texts else 0.5
    return [REQUEST if s >= threshold else CHATTER for s in request_scores(texts)]
