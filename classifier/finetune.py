"""Fine-tune BAAI/bge-small-en-v1.5 as a request / not_request classifier and compare it with TF-IDF.

    python -X utf8 -m classifier.finetune                      # mixed precision (fp32 weights, fp16 maths)
    python -X utf8 -m classifier.finetune --precision fp32     # plain fp32 throughout, for comparison

Same data cleaning, same template-grouped folds, same cost-based threshold and
same unseen messages as train.py, so the numbers line up. The model is
exported to ONNX (fp32 and int8); running it needs only onnxruntime and
tokenizers, no torch. Writes classifier/bge/ and classifier/finetune_report.json.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from torch import nn
from transformers import AutoModel, AutoTokenizer

from . import train as T

BASE = "BAAI/bge-small-en-v1.5"
OUT_DIR = T.HERE / "bge"
REPORT_FILE = T.HERE / "finetune_report.json"
MIXED = True   # fp16 autocast on the GPU; --precision fp32 turns it off
LABEL_SMOOTHING = 0.0
FIXED_THRESHOLD: float | None = None   # None: pick the threshold on cross-validated scores, as train.py does
MAX_LEN = 64
EPOCHS = 4
BATCH = 32
LR = 5e-5
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Classifier(nn.Module):
    """bge encoder, [CLS] vector (bge's own pooling), dropout, one linear layer -> 2 logits."""

    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(BASE)
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(self.encoder.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        cls = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.head(self.drop(cls))


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fit(texts: list[str], y: np.ndarray, tok, seed: int = SEED) -> Classifier:
    seed_all(seed)
    model = Classifier().to(DEVICE)
    enc = tok(texts, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
    labels = torch.tensor(y, dtype=torch.long)
    weights = torch.tensor(len(y) / (2 * np.bincount(y, minlength=2)), dtype=torch.float, device=DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)  # "balanced", like TF-IDF
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * ((len(texts) + BATCH - 1) // BATCH)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: min(1.0, (s + 1) / max(1, int(0.1 * steps))) * max(0.0, (steps - s) / steps))
    mixed = MIXED and DEVICE == "cuda"
    scaler = torch.amp.GradScaler(enabled=mixed)
    model.train()
    for _ in range(EPOCHS):
        order = torch.randperm(len(texts))
        for start in range(0, len(texts), BATCH):
            idx = order[start:start + BATCH]
            with torch.autocast(DEVICE, dtype=torch.float16, enabled=mixed):
                logits = model(enc["input_ids"][idx].to(DEVICE), enc["attention_mask"][idx].to(DEVICE))
                loss = loss_fn(logits.float(), labels[idx].to(DEVICE))
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            sched.step()
    return model.eval()


@torch.no_grad()
def scores(model: Classifier, texts: list[str], tok) -> np.ndarray:
    out = []
    for start in range(0, len(texts), 128):
        enc = tok(texts[start:start + 128], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        logits = model(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)).float()
        out.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def export(model: Classifier, tok) -> dict:
    """ONNX fp32 and int8 (dynamic quantisation), plus the tokenizer, for a torch-free predict()."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    OUT_DIR.mkdir(exist_ok=True)
    model = model.float().cpu().eval()
    sample = tok(["Book Koen for Saturday 8pm"], return_tensors="pt")
    fp32, int8 = OUT_DIR / "model_fp32.onnx", OUT_DIR / "model_int8.onnx"
    torch.onnx.export(model, (sample["input_ids"], sample["attention_mask"]), str(fp32),
                      input_names=["input_ids", "attention_mask"], output_names=["logits"],
                      dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                                    "logits": {0: "batch"}}, opset_version=17, dynamo=False)
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    tok.backend_tokenizer.save(str(OUT_DIR / "tokenizer.json"))
    return {"fp32_mb": round(fp32.stat().st_size / 1e6, 1), "int8_mb": round(int8.stat().st_size / 1e6, 1)}


def onnx_scores(path: Path, texts: list[str]) -> tuple[np.ndarray, float, float]:
    """p(request) from an exported model on CPU, with ms per message one by one and in a batch of all."""
    import onnxruntime as ort
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(OUT_DIR / "tokenizer.json"))
    tok.enable_truncation(MAX_LEN)
    tok.enable_padding()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def run(batch):
        enc = tok.encode_batch(batch)
        feed = {"input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64)}
        logits = session.run(None, feed)[0]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return (e / e.sum(axis=1, keepdims=True))[:, 1]

    run(texts[:2])  # warm up
    started = time.perf_counter()
    batch = run(texts)
    batch_ms = (time.perf_counter() - started) * 1000 / len(texts)
    started = time.perf_counter()
    for t in texts:
        run([t])
    each_ms = (time.perf_counter() - started) * 1000 / len(texts)
    return batch, each_ms, batch_ms


def main() -> None:
    global MIXED, OUT_DIR, REPORT_FILE, EPOCHS, LABEL_SMOOTHING, FIXED_THRESHOLD
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--precision", choices=["mixed", "fp32"], default="mixed")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--threshold", type=float, help="use this threshold instead of the cross-validated pick")
    p.add_argument("--name", help="output name (default: the precision)")
    a = p.parse_args()
    MIXED, EPOCHS, LABEL_SMOOTHING, FIXED_THRESHOLD = a.precision == "mixed", a.epochs, a.label_smoothing, a.threshold
    name = a.name or a.precision
    OUT_DIR = T.HERE / f"bge_{name}"
    REPORT_FILE = T.HERE / f"finetune_report_{name}.json"
    started = time.perf_counter()
    print(f"device: {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'cpu'}), "
          f"precision: {a.precision}, epochs {EPOCHS}, label smoothing {LABEL_SMOOTHING}, "
          f"threshold {FIXED_THRESHOLD if FIXED_THRESHOLD is not None else 'picked by cross-validation'}")
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    tok = AutoTokenizer.from_pretrained(BASE)

    rows = T.unique(T.majority_fix(T.load())[0])
    texts, labels = [t for t, _ in rows], [l for _, l in rows]
    groups = [T.template(t) for t in texts]
    dropped = frozenset(i for i, _ in T.flag_mislabelled(T.CANDIDATES[T.CHOSEN], texts, labels, groups))
    y = np.array([l == T.REQUEST for l in labels], dtype=int)
    is_request, clean = y == 1, np.array([i not in dropped for i in range(len(texts))])
    print(f"{len(texts)} unique rows, {len(dropped)} left out as likely mislabelled (same as train.py)")

    print(f"\n{T.FOLDS}-fold cross-validation grouped by template ({EPOCHS} epochs, batch {BATCH}, lr {LR})")
    oof = np.zeros(len(texts))
    fold_seconds = []
    for fold, (tr, held) in enumerate(GroupKFold(n_splits=T.FOLDS).split(texts, labels, groups)):
        t0 = time.perf_counter()
        tr = [i for i in tr if i not in dropped]
        model = fit([texts[i] for i in tr], y[tr], tok)
        oof[held] = scores(model, [texts[i] for i in held], tok)
        del model
        torch.cuda.empty_cache()
        fold_seconds.append(time.perf_counter() - t0)
        print(f"  fold {fold + 1}: {fold_seconds[-1]:.0f}s")
    tfidf_oof = T.out_of_fold(T.CANDIDATES[T.CHOSEN], texts, labels, groups, skip=dropped)

    print("\nfinal model on all clean rows")
    keep = [i for i in range(len(texts)) if i not in dropped]
    t0 = time.perf_counter()
    final = fit([texts[i] for i in keep], y[keep], tok)
    train_seconds = time.perf_counter() - t0
    unseen = T.unseen_phrasings()
    unseen_texts = [t for t, _ in unseen]
    unseen_y = np.array([l == T.REQUEST for _, l in unseen])
    bge_unseen = scores(final, unseen_texts, tok)
    tfidf = T.CANDIDATES[T.CHOSEN]().fit([texts[i] for i in keep], [labels[i] for i in keep])
    tfidf_unseen = T.p_request(tfidf, unseen_texts)

    sizes = export(final, tok)
    print(f"exported ONNX: {sizes}")
    onnx = {}
    for name in ("fp32", "int8"):
        s, each, batch = onnx_scores(OUT_DIR / f"model_{name}.onnx", unseen_texts)
        onnx[name] = {"scores": s, "ms_one_by_one": round(each, 2), "ms_in_batch": round(batch, 2),
                      "max_diff_vs_torch": round(float(np.abs(s - bge_unseen).max()), 4)}

    results = {}
    variants = {"tf-idf (shipped)": (tfidf_oof, tfidf_unseen),
                "bge-small fine-tuned (torch)": (oof, bge_unseen),
                "bge-small fine-tuned, onnx int8": (oof, onnx["int8"]["scores"]),
                "average of tf-idf and fine-tuned bge": ((oof + tfidf_oof) / 2, (bge_unseen + tfidf_unseen) / 2)}
    print(f"\n{'':<40}{'thr':>5} | {'CV given':>9}{'recall':>8} | {'unseen':>7}{'recall':>8}{'missed':>7}"
          f"{'false':>6}{'cost':>6}")
    for name, (o, u) in variants.items():
        th_cv = T.best_threshold(o[clean], is_request[clean])
        # the shipped TF-IDF model keeps its own cross-validated threshold; bge rows use the fixed one if given
        choices = [("", th_cv)] if FIXED_THRESHOLD is None or name.startswith("tf-idf") else \
            [("", FIXED_THRESHOLD), (" [cv threshold]", th_cv)]
        for suffix, th in choices:
            g, un = T.metrics(o, is_request, th), T.metrics(u, unseen_y, th)
            cost = T.MISS_COST * un["missed_requests"] + un["false_alarms"]
            results[name + suffix] = {
                "threshold": th, "cv_given_labels": g, "cv_clean_rows": T.metrics(o[clean], is_request[clean], th),
                "unseen": un, "unseen_cost": cost,
                "unseen_wrong": [f"{l} p={s:.2f}: {t}" for (t, l), s in zip(unseen, u) if (s >= th) != (l == T.REQUEST)]}
            print(f"{(name + suffix)[:40]:<40}{th:>5.2f} | {g['accuracy']:>9.3f}{g['request_recall']:>8.3f} | "
                  f"{un['accuracy']:>7.3f}{un['request_recall']:>8.3f}{un['missed_requests']:>7}"
                  f"{un['false_alarms']:>6}{cost:>6}")
    for name in ("bge-small fine-tuned (torch)", "average of tf-idf and fine-tuned bge"):
        for line in results[name]["unseen_wrong"]:
            print(f"   {name[:24]}: wrong {line}")

    th = results["bge-small fine-tuned, onnx int8"]["threshold"]
    (OUT_DIR / "meta.json").write_text(json.dumps({"base": BASE, "max_len": MAX_LEN, "threshold": th,
                                                    "files": ["model_int8.onnx", "tokenizer.json"]}, indent=2) + "\n")
    peak_mb = round(torch.cuda.max_memory_allocated() / 1e6) if DEVICE == "cuda" else None
    print(f"training: final model {train_seconds:.0f}s, folds {np.mean(fold_seconds):.0f}s each, "
          f"peak GPU memory {peak_mb} MB")
    REPORT_FILE.write_text(json.dumps({
        "base": BASE, "precision": "mixed fp16" if MIXED else "fp32", "epochs": EPOCHS, "batch": BATCH, "lr": LR,
        "label_smoothing": LABEL_SMOOTHING, "fixed_threshold": FIXED_THRESHOLD,
        "max_len": MAX_LEN, "device": DEVICE, "peak_gpu_mb": peak_mb,
        "fold_seconds_mean": round(float(np.mean(fold_seconds)), 1),
        "train_seconds_final": round(train_seconds, 1), "onnx_sizes_mb": sizes,
        "onnx_cpu_speed": {k: {kk: vv for kk, vv in v.items() if kk != "scores"} for k, v in onnx.items()},
        "results": results}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nCPU speed (ms per message): " + ", ".join(
        f"{k}: {v['ms_one_by_one']} one by one, {v['ms_in_batch']} in a batch (max diff vs torch {v['max_diff_vs_torch']})"
        for k, v in onnx.items()))
    print(f"wrote {OUT_DIR} and {REPORT_FILE.name} in {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
