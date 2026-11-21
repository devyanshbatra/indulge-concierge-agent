"""The small request / not_request model (Part 3)."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from classifier import predict as P
from classifier.predict import predict

UNSEEN = Path(__file__).parents[1] / "classifier" / "unseen.csv"


def test_shape_and_labels():
    out = predict(["Book Koen for Saturday 8pm for 4", "thanks!! perfect", "", "👍"])
    assert out[0] == "request" and out[1] == "not_request"
    assert len(out) == 4 and set(out) <= {"request", "not_request"}
    assert predict([]) == []


@pytest.mark.parametrize("text", [
    "Please cancel the Mirabelle table for Saturday",
    "Any update on the Koen booking?",
    "The car hasn't turned up yet",
    "Can you push dinner to 9?",
])
def test_requests_including_follow_ups_and_complaints(text):
    assert predict([text]) == ["request"]


def test_unseen_phrasings_keep_recall_high():
    """Hand-written messages that are not in train.csv: almost no real request may be missed."""
    with open(UNSEEN, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    said = predict([r["text"] for r in rows])
    requests = [s for s, r in zip(said, rows) if r["label"] == "request"]
    assert requests.count("request") / len(requests) >= 0.97
    assert sum(s == r["label"] for s, r in zip(said, rows)) / len(rows) >= 0.90


def test_retrains_when_the_saved_model_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "MODEL_FILE", tmp_path / "missing.joblib")
    monkeypatch.setattr(P, "_bundle", None)
    assert predict(["Book Koen for Saturday 8pm for 4", "haha ok see you"]) == ["request", "not_request"]
