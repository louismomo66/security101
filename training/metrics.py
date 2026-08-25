"""Evaluation metrics.

Accuracy alone is close to meaningless on this task. A dataset that is 80%
`normal` gives 80% accuracy to a model that predicts `normal` unconditionally
and catches zero crimes. Everything here is therefore reported per class, and
model selection uses macro-F1.
"""
from __future__ import annotations

import numpy as np
import torch

from training.labels import CRIME_ACTIONS


@torch.no_grad()
def evaluate(model, dataloader, device: str) -> dict:
    """Run the model over a dataloader and compute per-class statistics."""
    model.eval()
    n_classes = len(CRIME_ACTIONS)
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    losses = []
    criterion = torch.nn.CrossEntropyLoss()

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses.append(criterion(logits, y).item() * y.size(0))
        preds = logits.argmax(1)
        for t, p in zip(y.cpu().numpy(), preds.cpu().numpy()):
            confusion[t, p] += 1

    total = confusion.sum()
    correct = np.trace(confusion)

    tp = np.diag(confusion).astype(np.float64)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp
    support = confusion.sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(precision + recall > 0,
                      2 * precision * recall / (precision + recall), 0.0)

    present = support > 0
    return {
        "accuracy": float(correct / total) if total else 0.0,
        "loss": float(sum(losses) / total) if total else 0.0,
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "per_class": {
            CRIME_ACTIONS[i]: {
                "precision": round(float(precision[i]), 4),
                "recall": round(float(recall[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(support[i]),
            }
            for i in range(n_classes) if support[i] > 0
        },
        "confusion": confusion,
    }


def format_report(stats: dict) -> str:
    """Human-readable classification report plus confusion matrix."""
    lines = [
        f"accuracy {stats['accuracy']:.3f}   macro-F1 {stats['macro_f1']:.3f}   "
        f"loss {stats['loss']:.4f}",
        "",
        f"{'class':18s} {'prec':>7s} {'recall':>7s} {'f1':>7s} {'support':>8s}",
        "-" * 52,
    ]
    for name, m in stats["per_class"].items():
        lines.append(f"{name:18s} {m['precision']:7.3f} {m['recall']:7.3f} "
                     f"{m['f1']:7.3f} {m['support']:8d}")

    conf = stats["confusion"]
    present = [i for i in range(len(CRIME_ACTIONS)) if conf[i].sum() > 0]
    if present:
        lines += ["", "confusion (rows = true, cols = predicted)", ""]
        header = " " * 18 + "".join(f"{CRIME_ACTIONS[j][:6]:>8s}" for j in present)
        lines.append(header)
        for i in present:
            row = "".join(f"{conf[i, j]:8d}" for j in present)
            lines.append(f"{CRIME_ACTIONS[i]:18s}{row}")

    # The number that actually matters operationally.
    crime = [n for n in stats["per_class"] if n not in ("normal", "fall_or_medical")]
    if crime:
        fp_rate = None
        if "normal" in stats["per_class"]:
            normal_idx = CRIME_ACTIONS.index("normal")
            normal_total = conf[normal_idx].sum()
            if normal_total:
                crime_idx = [CRIME_ACTIONS.index(c) for c in crime]
                fp_rate = conf[normal_idx, crime_idx].sum() / normal_total
        if fp_rate is not None:
            lines += ["", f"normal clips misflagged as crime: {fp_rate:.1%}"]
            if fp_rate > 0.05:
                lines.append("  ^ at this rate a busy camera will produce alerts "
                             "faster than anyone can review them.")
    return "\n".join(lines)
