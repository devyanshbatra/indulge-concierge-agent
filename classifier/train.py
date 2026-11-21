"""Train the request / not_request classifier and write down how well it does.

    python -X utf8 -m classifier.train

Steps, each printed as it runs:
1. Clean the text (Unicode NFKC, spaces) and group messages by a template
   (lowercase, digits -> 0, no punctuation). The 2,000 rows are only ~780
   templates, so every score below is cross-validated by template
   (GroupKFold): the same message never sits on both sides of a split.
2. Fix noisy labels: a text repeated with a clear majority label gets that
   label; then any message the cross-validated model is very sure is
   mislabelled is set aside (written to flagged_labels.csv) and left out.
3. Compare a few small models by template-grouped cross-validation and on
   unseen phrasings (the hand-labelled messages in tuning/cases.py).
4. Pick the decision threshold from out-of-fold scores at the lowest cost,
   counting a missed request MISS_COST times a false alarm.
5. Fit the chosen model on all cleaned data and save it to model.joblib.
"""
from __future__ import annotations

import csv
import json
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline

HERE = Path(__file__).parent
ROOT = HERE.parent
DATA = ROOT / "data" / "classifier" / "train.csv"
MODEL_FILE = HERE / "model.joblib"
REPORT_FILE = HERE / "report.json"
FLAGGED_FILE = HERE / "flagged_labels.csv"
UNSEEN_FILE = HERE / "unseen.csv"

REQUEST, CHATTER = "request", "not_request"
MISS_COST = 5             # a missed request is taken as 5x worse than a false alarm (our assumption)
FLAG_BELOW = 0.25         # a label the cross-validated model gives under 25% is set aside as likely wrong
FOLDS = 5
SEED = 0


def clean(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def template(text: str) -> str:
    t = re.sub(r"\d+", "0", clean(text).lower())
    return " ".join(re.sub(r"[^a-z0<> ]", " ", t).split()) or t


def load(path: Path = DATA) -> list[tuple[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        return [(clean(r["text"]), r["label"].strip()) for r in csv.DictReader(f)]


# ---- models ---------------------------------------------------------------

def _word():
    return TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"(?u)\b\w+\b|[?!]")


def _char():
    return TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)


def _lr():
    return LogisticRegression(C=4.0, class_weight="balanced", max_iter=5000)


CANDIDATES = {
    "word tf-idf + logistic regression": lambda: Pipeline([("tfidf", _word()), ("clf", _lr())]),
    "char tf-idf + logistic regression": lambda: Pipeline([("tfidf", _char()), ("clf", _lr())]),
    "word+char tf-idf + logistic regression": lambda: Pipeline(
        [("tfidf", FeatureUnion([("word", _word()), ("char", _char())])), ("clf", _lr())]),
    "word tf-idf + complement naive Bayes": lambda: Pipeline([("tfidf", _word()), ("clf", ComplementNB(alpha=0.3))]),
}
CHOSEN = "word+char tf-idf + logistic regression"


def p_request(model, texts) -> np.ndarray:
    classes = list(model.classes_)
    return model.predict_proba(list(texts))[:, classes.index(REQUEST)]


def out_of_fold(make, texts: list[str], labels: list[str], groups: list[str], skip=frozenset()) -> np.ndarray:
    """p(request) for every message from a model that never saw its template.
    Rows in `skip` are scored but never trained on."""
    scores = np.zeros(len(texts))
    for train, held in GroupKFold(n_splits=FOLDS).split(texts, labels, groups):
        train = [i for i in train if i not in skip]
        model = make().fit([texts[i] for i in train], [labels[i] for i in train])
        scores[held] = p_request(model, [texts[i] for i in held])
    return scores


# ---- cleaning the labels --------------------------------------------------

def majority_fix(rows: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], list[dict]]:
    """A template seen 3+ times with one label at least 3/4 of the time gets that label everywhere
    ("my new number is 98170 12345" is the same message as the 35 other new numbers)."""
    votes: dict[str, Counter] = defaultdict(Counter)
    for text, label in rows:
        votes[template(text)][label] += 1
    fixed, changes = [], []
    for text, label in rows:
        v = votes[template(text)]
        top, n = v.most_common(1)[0]
        if label != top and sum(v.values()) >= 3 and n / sum(v.values()) >= 0.75:
            changes.append({"text": text, "given": label, "now": top,
                            "why": f"same template is {top} {n}/{sum(v.values())} times"})
            label = top
        fixed.append((text, label))
    return fixed, changes


def flag_mislabelled(make, texts, labels, groups) -> list[tuple[int, float]]:
    """Rows whose label the cross-validated model gives under FLAG_BELOW, repeated until none are left
    (removing the worst noise sharpens the scores of the rest)."""
    keep, flagged = list(range(len(texts))), []
    while True:
        scores = out_of_fold(make, [texts[i] for i in keep], [labels[i] for i in keep], [groups[i] for i in keep])
        p_given = [s if labels[i] == REQUEST else 1 - s for i, s in zip(keep, scores)]
        new = [(i, p) for i, p in zip(keep, p_given) if p < FLAG_BELOW]
        if not new:
            return sorted(flagged, key=lambda x: x[1])
        flagged += new
        dropped = {i for i, _ in new}
        keep = [i for i in keep if i not in dropped]


def unique(rows: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """One row per (text, label): 49 copies of "ok" should not outweigh 49 different requests."""
    return sorted(set(rows))


# ---- scoring --------------------------------------------------------------

def best_threshold(scores: np.ndarray, is_request: np.ndarray) -> float:
    """The threshold with the lowest MISS_COST * missed + false alarms; ties go to the lower threshold."""
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    cost = [MISS_COST * int((~(scores >= t) & is_request).sum()) + int(((scores >= t) & ~is_request).sum())
            for t in grid]
    return float(grid[int(np.argmin(cost))])


def metrics(scores: np.ndarray, is_request: np.ndarray, threshold: float) -> dict:
    said = scores >= threshold
    tp, fp = int((said & is_request).sum()), int((said & ~is_request).sum())
    fn, tn = int((~said & is_request).sum()), int((~said & ~is_request).sum())
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    return {"accuracy": round((tp + tn) / len(said), 4), "request_recall": round(recall, 4),
            "request_precision": round(precision, 4),
            "f1": round(2 * precision * recall / max(precision + recall, 1e-9), 4),
            "missed_requests": fn, "false_alarms": fp, "n": len(said)}


def unseen_phrasings() -> list[tuple[str, str]]:
    """Hand-labelled messages that are not in train.csv: classifier/unseen.csv (written for this, before
    looking at any model output) plus the agent's tuning cases."""
    from tuning.cases import CASES, FRESH, Quiet
    with open(UNSEEN_FILE, encoding="utf-8", newline="") as f:
        written = [(clean(r["text"]), r["label"].strip()) for r in csv.DictReader(f)]
    return written + [(clean(c.text), CHATTER if isinstance(c.expect[0], Quiet) else REQUEST) for c in CASES + FRESH]


def build(threshold: float) -> dict:
    """The shipped model, trained the same way without the report (used if model.joblib cannot be read)."""
    rows = unique(majority_fix(load())[0])
    texts, labels = [t for t, _ in rows], [label for _, label in rows]
    dropped = {i for i, _ in flag_mislabelled(CANDIDATES[CHOSEN], texts, labels, [template(t) for t in texts])}
    model = CANDIDATES[CHOSEN]().fit([t for i, t in enumerate(texts) if i not in dropped],
                                     [label for i, label in enumerate(labels) if i not in dropped])
    return {"model": model, "threshold": threshold, "name": CHOSEN}


def main() -> None:
    started = time.perf_counter()
    raw = load()
    print(f"{len(raw)} rows, {len({t for t, _ in raw})} different texts, "
          f"{len({template(t) for t, _ in raw})} templates; labels {dict(Counter(label for _, label in raw))}")

    rows, majority_changes = majority_fix(raw)
    print(f"\n1. {len(majority_changes)} labels set to the majority label of the same text")
    rows = unique(rows)
    texts, labels = [t for t, _ in rows], [label for _, label in rows]
    groups = [template(t) for t in texts]

    print(f"2. looking for labels the cross-validated model is very sure are wrong ({len(rows)} unique rows)")
    flagged = flag_mislabelled(CANDIDATES[CHOSEN], texts, labels, groups)
    with open(FLAGGED_FILE, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "given_label", "p_given_label", "action"])
        for c in majority_changes:
            w.writerow([c["text"], c["given"], "", f"relabelled {c['now']} ({c['why']})"])
        for i, p in flagged:
            w.writerow([texts[i], labels[i], f"{p:.3f}", "left out of training"])
    for i, p in flagged:
        print(f"   {labels[i]:<12} p={p:.2f}  {texts[i]}")
    dropped = frozenset(i for i, _ in flagged)
    is_request = np.array([label == REQUEST for label in labels])
    clean_rows = np.array([i not in dropped for i in range(len(texts))])

    print(f"\n3. comparing models. Cross-validation is grouped by template; flagged rows are never trained on "
          f"but are still scored against their given label (so label noise shows up as errors).\n"
          f"   Threshold: lowest cost on the clean rows, {MISS_COST}x per missed request, 1x per false alarm.")
    unseen = unseen_phrasings()
    unseen_texts = [t for t, _ in unseen]
    unseen_is_request = np.array([label == REQUEST for _, label in unseen])
    comparison = {}
    for name, make in CANDIDATES.items():
        oof = out_of_fold(make, texts, labels, groups, skip=dropped)
        threshold = best_threshold(oof[clean_rows], is_request[clean_rows])
        model = make().fit([t for i, t in enumerate(texts) if i not in dropped],
                           [label for i, label in enumerate(labels) if i not in dropped])
        comparison[name] = {
            "threshold": threshold, "auc_given_labels": round(roc_auc_score(is_request, oof), 4),
            "cv_given_labels": metrics(oof, is_request, threshold),
            "cv_clean_rows": metrics(oof[clean_rows], is_request[clean_rows], threshold),
            "unseen_phrasings": metrics(p_request(model, unseen_texts), unseen_is_request, threshold)}
        g, c, u = (comparison[name][k] for k in ("cv_given_labels", "cv_clean_rows", "unseen_phrasings"))
        print(f"   {name:<40} given labels: acc {g['accuracy']:.3f} recall {g['request_recall']:.3f} | clean rows: "
              f"acc {c['accuracy']:.3f} | unseen ({u['n']}): acc {u['accuracy']:.3f} recall {u['request_recall']:.3f} "
              f"missed {u['missed_requests']} false alarms {u['false_alarms']}  (threshold {threshold:.2f})")

    texts = [t for i, t in enumerate(texts) if i not in dropped]
    labels = [label for i, label in enumerate(labels) if i not in dropped]
    print(f"\n4-5. fitting '{CHOSEN}' on all {len(texts)} cleaned rows")
    final = CANDIDATES[CHOSEN]().fit(texts, labels)
    threshold = comparison[CHOSEN]["threshold"]
    joblib.dump({"model": final, "threshold": threshold, "name": CHOSEN}, MODEL_FILE)
    misses = [(t, label, round(float(s), 3)) for (t, label), s in zip(unseen, p_request(final, unseen_texts))
              if (s >= threshold) != (label == REQUEST)]
    for t, label, s in misses:
        print(f"   unseen, wrong: {label:<12} p(request)={s:.2f}  {t}")
    REPORT_FILE.write_text(json.dumps({
        "rows": len(raw), "unique_rows_used": len(texts), "relabelled_by_majority": len(majority_changes),
        "left_out_as_likely_mislabelled": len(flagged), "chosen": CHOSEN, "threshold": threshold,
        "miss_cost": MISS_COST, "comparison": comparison, "unseen_wrong": misses,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nsaved {MODEL_FILE.name} (threshold {threshold:.2f}) and {REPORT_FILE.name} "
          f"in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
