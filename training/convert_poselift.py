"""Convert the PoseLift dataset into this project's clip format.

Why bother
----------
Every model this project has trained has failed for the same measured reason:
the incidents in our own footage are recorded at a resolution where the bodies
are not visible. Pose finds two skeletons in 0 of 23 frames at Naalya, 0 of 21
at Kawempe, 0 of 21 at Kajjansi. The ST-GCN needs two skeletons to see an
interaction between two people, so it has never been given an input containing
the signal it exists to read.

PoseLift (WACV 2025, TeCSAR-UNCC, Apache-2.0) ships **pre-extracted AlphaPose
skeletons** from 1080p retail cameras with frame-level shoplifting labels.
Measured on the download: **21.4% of populated frames contain two or more
people**, against 0% in our own incident spans. Someone else already paid the
resolution cost.

That makes one question answerable today, without new footage:

    Does the ST-GCN architecture learn a two-person interaction at all,
    when the skeletons are clean?

A positive result validates the machinery and confirms the blocker is input
quality alone. A negative one says something is wrong upstream of the cameras,
and better footage would not have saved us.

**It does not tell us the model transfers to a street snatch.** PoseLift is
indoor retail shoplifting: one person concealing an item, not a grab followed
by two bodies separating at speed. `labels.py` maps UCF's Shoplifting to
`pickpocket` on the grounds that the concealment gesture is the shared
signature, and the same reasoning is applied here.

Format notes, both learned the hard way from the files themselves
----------------------------------------------------------------
* The `.pkl` files are `{frame: {person_id: [bbox, keypoints]}}`, where
  keypoints is (17, 3) COCO-17 — matching this project's layout — and bbox is
  XYXY despite the README saying XYWH.
* The confidence column is **NaN throughout**. AlphaPose's real scores live in
  the sibling JSON under a separate `scores` key. NaN confidences would poison
  the normalisation, so they are replaced with 1.0 for any joint that has
  coordinates and 0.0 for one that does not.

Usage
-----
    python -m training.convert_poselift \
        --src training/data/poselift/data --out training/data/clips_poselift
"""
from __future__ import annotations

import argparse
import glob
import pickle
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.labels import resolve  # noqa: E402

# PoseLift is shoplifting. `labels.py` already maps UCF's Shoplifting class to
# pickpocket, for the same reason: the observable body movement is concealment.
ANOMALY_LABEL = "pickpocket"
NORMAL_LABEL = "normal"


def load_pkl(path: Path) -> dict[int, dict[int, np.ndarray]]:
    """{frame: {person_id: (17,3) float32}} with NaN confidences repaired."""
    raw = pickle.load(open(path, "rb"))
    out: dict[int, dict[int, np.ndarray]] = {}
    for frame, people in raw.items():
        if not people:
            continue
        per: dict[int, np.ndarray] = {}
        for pid, rec in people.items():
            kp = np.asarray(rec[1], dtype=np.float32).reshape(-1, 3)
            if kp.shape[0] != 17:
                continue
            # Confidence is NaN in every file. A NaN propagates through the
            # centring and scaling in dataset.py and silently turns a whole
            # clip into NaNs, so it is replaced rather than passed along:
            # present coordinates count as confident, absent ones as missing.
            conf = kp[:, 2]
            bad = ~np.isfinite(conf)
            has_xy = np.isfinite(kp[:, 0]) & np.isfinite(kp[:, 1]) & (
                (kp[:, 0] != 0) | (kp[:, 1] != 0))
            conf[bad] = has_xy[bad].astype(np.float32)
            kp[:, 2] = conf
            kp[~np.isfinite(kp)] = 0.0
            per[int(pid)] = kp
        if per:
            out[int(frame)] = per
    return out


def select_top(people: dict[int, np.ndarray], num_person: int) -> np.ndarray:
    """(M, 17, 3), the M most confident skeletons, zero-padded."""
    ranked = sorted(people.values(), key=lambda k: -float(k[:, 2].sum()))
    sel = ranked[:num_person]
    out = np.zeros((num_person, 17, 3), dtype=np.float32)
    for i, kp in enumerate(sel):
        out[i] = kp
    return out


def spans_from_mask(mask: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Contiguous runs of 1s and of 0s, as (start, end_exclusive)."""
    runs: list[tuple[int, int, int]] = []
    if len(mask) == 0:
        return [], []
    start, cur = 0, int(mask[0])
    for i in range(1, len(mask)):
        if int(mask[i]) != cur:
            runs.append((start, i, cur))
            start, cur = i, int(mask[i])
    runs.append((start, len(mask), cur))
    return ([(a, b) for a, b, v in runs if v == 1],
            [(a, b) for a, b, v in runs if v == 0])


def build_clip(poses, span, clip_len, num_person, min_frames):
    """(T, M, 17, 3) sampled across a frame span, or None if too sparse."""
    a, b = span
    present = [f for f in range(a, b) if f in poses]
    if len(present) < min_frames:
        return None
    idx = (np.linspace(0, len(present) - 1, clip_len).astype(int)
           if len(present) >= clip_len
           else [present[i % len(present)] for i in range(clip_len)])
    frames = ([present[i] for i in idx] if len(present) >= clip_len else idx)
    return np.stack([select_top(poses[f], num_person) for f in frames])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--src", default="training/data/poselift/data")
    p.add_argument("--out", default="training/data/clips_poselift")
    p.add_argument("--clip-len", type=int, default=100)
    p.add_argument("--num-person", type=int, default=2)
    p.add_argument("--min-frames", type=int, default=20,
                   help="skip a span with fewer populated frames than this")
    p.add_argument("--val-frac", type=float, default=0.2)
    args = p.parse_args()

    src = ROOT / args.src if not Path(args.src).is_absolute() else Path(args.src)
    pkls = sorted(glob.glob(str(src / "**" / "*.pkl"), recursive=True))
    masks = {Path(f).stem: f
             for f in glob.glob(str(src / "**" / "test_frame_mask" / "*.npy"),
                                recursive=True)}
    if not pkls:
        raise SystemExit(f"no .pkl pose files under {src}")
    print(f"{len(pkls)} pose files, {len(masks)} label masks")

    out_dir = ROOT / args.out
    for split in ("train", "val"):
        (out_dir / split).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    counts = {"pickpocket": 0, "normal": 0}
    skipped_no_mask = skipped_sparse = 0

    for i, f in enumerate(pkls):
        stem = Path(f).stem                       # e.g. 1_222
        # masks are named 01_0222 — camera and video zero-padded differently
        m = re.match(r"(\d+)_(\d+)$", stem)
        key = f"{int(m.group(1)):02d}_{int(m.group(2)):04d}" if m else stem
        poses = load_pkl(Path(f))
        if key in masks:
            mask = np.load(masks[key])
            anom, norm = spans_from_mask(mask)
        elif "/Train/" in f or "/train/" in f:
            # PoseLift follows the usual anomaly-detection split: the train
            # half is normal by construction and ships no mask. Dropping it
            # for want of a label file would throw away 104 of 151 files —
            # and normals are the ingredient this project has repeatedly
            # measured as the one that buys precision.
            if not poses:
                skipped_sparse += 1
                continue
            anom, norm = [], [(min(poses), max(poses) + 1)]
        else:
            skipped_no_mask += 1
            continue

        for label, spans in ((ANOMALY_LABEL, anom), (NORMAL_LABEL, norm)):
            for span in spans:
                clip = build_clip(poses, span, args.clip_len,
                                  args.num_person, args.min_frames)
                if clip is None:
                    skipped_sparse += 1
                    continue
                # img_shape drives the normalisation that preserves the
                # distance between people, so it must be the real frame size.
                # PoseLift does not ship it; the tightest honest estimate is
                # the extent of the coordinates actually observed.
                xy = clip[..., :2][clip[..., 2] > 0]
                if xy.size == 0:
                    skipped_sparse += 1
                    continue
                w = int(max(clip[..., 0].max(), 1)) + 1
                h = int(max(clip[..., 1].max(), 1)) + 1
                split = "val" if rng.random() < args.val_frac else "train"
                name = f"{key}_{span[0]}_{span[1]}_{label}.npz"
                np.savez_compressed(
                    out_dir / split / name,
                    keypoints=clip.astype(np.float16),
                    label=np.int64(resolve(label)),
                    img_shape=np.array([h, w], dtype=np.int32),
                )
                counts[label] += 1

    print(f"\nwrote {sum(counts.values())} clips to {args.out}")
    for k, v in counts.items():
        print(f"  {k:12} {v}")
    print(f"skipped: {skipped_no_mask} without a label mask, "
          f"{skipped_sparse} too sparse (<{args.min_frames} populated frames)")
    if counts[ANOMALY_LABEL] == 0:
        raise SystemExit("no anomaly clips produced — check mask name matching")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
