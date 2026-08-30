"""Fit a model of normal, then score this project's labelled spans against it.

The method
----------
One-class, not supervised. We have one confirmed `normal` span and five
confirmed incidents in a single clip — far too few positives to train a
classifier, but the wrong question anyway. Instead:

  1. Build a model of what ordinary footage looks like, from footage confirmed
     to contain no incident. There are hours of it and it needs no labels.
  2. Score any window by how far it sits from that manifold.

Distance is mean cosine distance to the k nearest normal windows. kNN rather
than a Gaussian because normal street footage is multi-modal — a busy junction
and an empty night road are both normal and share no centre — and a single
Gaussian would put its mean in the empty space between them and call
everything anomalous.

What passing looks like
-----------------------
The only question that matters: do the five incident spans score higher than
the confirmed-normal span? Reported as AUC over windows, plus the per-span
medians so a single hot window cannot flatter the result.

This cannot name the crime. It says "these seconds are unlike this camera's
ordinary traffic", and a human decides what that means — which is what
`CLAUDE.md`'s ground rules require of every alert here anyway.

MEASURED 2026-08-29 — and it does not work on this data, for a reason that
matters more than the number.

    window-level AUC 0.004      (0.5 = no signal; below that is inverted)
    incident spans above the normal median: 0/5

A first run scored AUC 0.957, which was leakage: the confirmed-normal span sits
in BOTH `NORMAL_SOURCES` and `annotations.csv`, so it was scored against a bank
containing itself and its nearest neighbours were its own windows. Holding it
out drops the result below chance.

Below chance is the informative part. With the normal span held out, the bank
holds only Kampala traffic (1080p) and the ambulance clip (720p), while every
scored span comes from the 360p police compilation. The distance then measures
**which camera this is**, not what happens in it — and the normal span is
simply the most visually distinct of the six.

And the compilation is six different cameras with one clip each: **no two spans
share a camera**. Per-camera normality cannot be built from this dataset at
all, at any threshold or with any feature extractor.

So this is not a refutation of the method. It is a statement of its input
requirement: **hours of ordinary footage from the same fixed camera**. A real
deployment produces that for free — a camera pointed at a junction records
normal all day. A YouTube compilation of six unrelated cameras never can.

That requirement is worth contrasting with the others measured here: pose,
weapons, VLM and trajectory all need *resolution*, which is expensive and which
this project does not have. This needs *duration on one camera*, which is free.

Usage
-----
    python -m training.anomaly.evaluate
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from training.annotations import load                      # noqa: E402
from training.anomaly.features import span_features        # noqa: E402

# Footage trusted to contain no incident, used to build the model of normal.
# Each entry carries why it is trusted — an unexplained entry here silently
# teaches the model that a crime is ordinary.
#
# Deliberately EXCLUDED: the phone-snatch compilations (they are wall-to-wall
# incidents) and the hit-and-run compilation (collisions are exactly the
# anomalous motion we want flagged).
NORMAL_SOURCES: list[tuple[str, float | None, float | None, str]] = [
    ("videos/Traffic in Kampala in Uganda is very chaotic - Klaas-Jan Gra_fe _1080p_ h264_.mp4",
     None, None,
     "Ordinary Kampala traffic. Same streets, vehicles and light as the "
     "target deployment, and no incident anywhere in it."),
    ("videos/FASTEST AMBULANCE EVER_ - HyperXZ _720p_ h264_.mp4",
     None, None,
     "Ordinary road footage."),
    ("videos/Weird street crimes in Uganda caught on camera by the police - My adventures in Uganda (360p, h264).mp4",
     36.48, 56.72,
     "The operator-confirmed non-incident span, from the same camera set as "
     "the incidents — the most valuable normal we have."),
]



def _overlaps_span(key: str, start: float, end: float) -> bool:
    """Does a bank key's time range overlap [start, end)?

    Keys are "<stem>|<a>|<b>", where a/b may be None meaning the whole clip —
    which overlaps everything from that file.
    """
    try:
        _, a, b = key.rsplit("|", 2)
    except ValueError:
        return True
    if a in ("None", "") or b in ("None", ""):
        return True
    return float(a) < end and float(b) > start


def knn_distance(x: np.ndarray, bank: np.ndarray, k: int) -> np.ndarray:
    """Mean cosine distance from each row of x to its k nearest bank rows."""
    xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    bn = bank / (np.linalg.norm(bank, axis=1, keepdims=True) + 1e-8)
    sim = xn @ bn.T                              # (N, M) cosine similarity
    k = min(k, sim.shape[1])
    top = np.partition(sim, -k, axis=1)[:, -k:]  # k most similar
    return 1.0 - top.mean(axis=1)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Probability a random positive scores above a random negative."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[:len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--annotations", default="training/data/annotations.csv")
    p.add_argument("--stride", type=float, default=1.0,
                   help="seconds between sampled windows")
    p.add_argument("--k", type=int, default=8, help="neighbours in the normal bank")
    p.add_argument("--bank-cap", type=int, default=4000)
    args = p.parse_args()

    # ── 1. the model of normal ────────────────────────────────────────────
    print("building the normal bank")
    bank_parts: dict[str, np.ndarray] = {}
    for rel, a, b, why in NORMAL_SOURCES:
        v = ROOT / rel
        if not v.exists():
            print(f"  -- missing, skipped: {Path(rel).name}")
            continue
        _, feats = span_features(v, a or 0.0, b or 0.0, args.stride)
        # Keyed by source so a span under evaluation can be held out of the
        # bank it is scored against. The confirmed-normal span appears in BOTH
        # NORMAL_SOURCES and annotations.csv: scored against a bank containing
        # itself, its nearest neighbours are itself and it scores ~0 by
        # construction. That alone would produce a flattering AUC and say
        # nothing about whether the method works.
        bank_parts[f"{Path(rel).stem}|{a}|{b}"] = feats
        print(f"  {Path(rel).name[:48]:48} {len(feats):5} windows  — {why[:44]}")
    if not bank_parts:
        raise SystemExit("no normal footage found")
    bank = np.concatenate(list(bank_parts.values()))
    if len(bank) > args.bank_cap:
        idx = np.random.default_rng(0).choice(len(bank), args.bank_cap, replace=False)
        bank = bank[idx]
    print(f"\nnormal bank: {bank.shape[0]} windows x {bank.shape[1]} dims\n")

    # ── 2. score the labelled spans ───────────────────────────────────────
    ann = load(ROOT / args.annotations, root=ROOT)
    pos_all, neg_all, rows = [], [], []
    for span in ann.spans:
        v = ROOT / span.video
        if not v.exists():
            print(f"  -- missing: {span.video}")
            continue
        _, feats = span_features(v, span.start, span.end, args.stride)
        if len(feats) == 0:
            continue
        # Hold out any bank entry drawn from this same span, so nothing is
        # scored against a copy of itself.
        held = [k for k in bank_parts
                if k.startswith(v.stem.split(" (")[0][:30])
                and _overlaps_span(k, span.start, span.end)]
        if held:
            keep = [f for k, f in bank_parts.items() if k not in held]
            scoring_bank = np.concatenate(keep) if keep else bank
            print(f"  (held out of the bank: {len(held)} source(s) overlapping "
                  f"this span — {sum(len(bank_parts[k]) for k in held)} windows)")
        else:
            scoring_bank = bank
        d = knn_distance(feats, scoring_bank, args.k)
        is_incident = span.label.lower() != "normal"
        (pos_all if is_incident else neg_all).append(d)
        rows.append((span.label, span.start, span.end, d))
        print(f"{span.label:13} {span.start:6.1f}-{span.end:6.1f}  "
              f"n={len(d):3}  median {np.median(d):.4f}  max {d.max():.4f}")

    if not pos_all or not neg_all:
        raise SystemExit("\nneed both incident and normal spans to score")
    pos = np.concatenate(pos_all)
    neg = np.concatenate(neg_all)

    print("\n-- SUMMARY --")
    print(f"  incident windows : {len(pos):4}  median {np.median(pos):.4f}")
    print(f"  normal windows   : {len(neg):4}  median {np.median(neg):.4f}")
    print(f"  window-level AUC : {auc(pos, neg):.3f}   (0.5 = no signal)")

    # A per-span view as well: window AUC can be carried by one span, and a
    # reviewer cares whether a *span* surfaces, not whether every window does.
    med = {r[0] + f"@{r[1]:.0f}": float(np.median(r[3])) for r in rows}
    thr = float(np.median(neg))
    hits = sum(1 for r in rows if r[0].lower() != "normal" and np.median(r[3]) > thr)
    n_inc = sum(1 for r in rows if r[0].lower() != "normal")
    print(f"  incident spans above the normal median: {hits}/{n_inc}")
    print("\n  per-span medians:", {k: round(v, 4) for k, v in med.items()})

    print("\nThis names no crime. It says a window is unlike this camera's "
          "ordinary footage, which is a prioritisation aid for a human "
          "reviewer — the only thing CLAUDE.md's ground rules permit an alert "
          "to be.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
