"""Generate an annotation CSV from a UCF-Crime style directory tree.

UCF-Crime ships whole videos labelled at the video level, laid out as:

    UCF_Crimes/Videos/Abuse/Abuse001_x264.mp4
    UCF_Crimes/Videos/Normal_Videos_event/Normal_Videos_003_x264.mp4

Video-level labels are weak: an "Abuse" video is mostly ordinary footage with a
few seconds of incident. Training a clip classifier on the whole video teaches
it that normal footage is abuse. This script therefore:

  - takes only a bounded window from each anomaly video (default: the middle
    `--anomaly-window` seconds, where UCF incidents usually sit), and
  - takes generous spans from Normal videos, which really are normal throughout.

Treat the anomaly spans as *provisional* and correct them by hand — the CSV is
plain text and `notes` records that the span was auto-generated.

    python -m training.prepare_ucf_crime \
        --root ~/datasets/UCF_Crimes/Videos \
        --out training/data/ucf_annotations.csv
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.labels import UCF_CRIME_MAP                 # noqa: E402

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov"}


def duration_seconds(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return 0.0
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return frames / fps if fps > 0 else 0.0
    finally:
        cap.release()


def main() -> int:
    p = argparse.ArgumentParser(description="UCF-Crime -> VEREC annotation CSV")
    p.add_argument("--root", required=True, help="directory of per-class folders")
    p.add_argument("--out", default="training/data/ucf_annotations.csv")
    p.add_argument("--anomaly-window", type=float, default=20.0,
                   help="seconds to take from the middle of each anomaly video")
    p.add_argument("--normal-window", type=float, default=40.0,
                   help="seconds to take from each normal video")
    p.add_argument("--max-per-class", type=int, default=250)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    root = Path(args.root).expanduser()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 1

    random.seed(args.seed)
    rows: list[list[str]] = []
    unmapped: set[str] = set()
    per_class: dict[str, int] = {}

    for class_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        ucf_class = class_dir.name
        if ucf_class not in UCF_CRIME_MAP:
            unmapped.add(ucf_class)
            continue
        target = UCF_CRIME_MAP[ucf_class]
        if target is None:
            print(f"skip {ucf_class} (no pose-observable equivalent)")
            continue

        videos = sorted(v for v in class_dir.rglob("*") if v.suffix.lower() in VIDEO_EXTS)
        random.shuffle(videos)
        videos = videos[: args.max_per_class]
        window = args.normal_window if target == "normal" else args.anomaly_window

        for i, video in enumerate(videos):
            dur = duration_seconds(video)
            if dur <= 1.0:
                continue

            if target == "normal":
                start = 0.0
                end = min(dur, window)
            else:
                # Middle window — UCF incidents cluster near the centre.
                mid = dur / 2
                start = max(0.0, mid - window / 2)
                end = min(dur, mid + window / 2)

            r = random.random()
            split = ("val" if r < args.val_fraction
                     else "test" if r < args.val_fraction + args.test_fraction
                     else "train")

            rows.append([str(video), f"{start:.2f}", f"{end:.2f}", target, split,
                         f"auto from UCF {ucf_class} — VERIFY span"])
            per_class[target] = per_class.get(target, 0) + 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["video", "start", "end", "label", "split", "notes"])
        w.writerows(rows)

    print(f"\nWrote {len(rows)} spans to {out}")
    for name, n in sorted(per_class.items()):
        print(f"  {name:16s} {n:5d}")
    if unmapped:
        print(f"\nUnrecognised folders (ignored): {', '.join(sorted(unmapped))}")
    print("\nThe anomaly spans are guesses. Review and tighten them before "
          "extracting poses — span quality dominates final accuracy here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
