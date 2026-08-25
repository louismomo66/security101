"""Stage 1 — turn annotated video spans into skeleton clips on disk.

Pose extraction is the slow part of this pipeline (one YOLO-pose forward pass
per frame), so it runs once and caches. Training then reads .npz files and can
iterate in seconds.

Usage
-----
    python -m training.extract_poses \
        --annotations training/data/annotations.csv \
        --out training/data/clips \
        --pose-model yolo11n-pose.onnx

Output
------
    <out>/<split>/<label>/<video-stem>_<start>-<end>_<n>.npz
        keypoints : float16 (T, M, 17, 3)   raw pixel coords + confidence
        label     : int
        meta      : json blob (source video, time range, fps)

Coordinates are stored **raw**, not normalized — normalization is cheap and
belongs at training time, where augmentation happens first.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow running as a script from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from action.preprocess import select_top_persons          # noqa: E402
from training import annotations as ann                   # noqa: E402
from training.labels import CRIME_ACTIONS                 # noqa: E402


def sample_indices(n_frames: int, clip_len: int, n_clips: int) -> list[list[int]]:
    """Pick `n_clips` windows of `clip_len` frame indices spanning n_frames.

    Short spans are handled by looping the available frames rather than
    zero-padding: a 40-frame snatch padded to 100 zeros teaches the model that
    "crime" means "mostly empty buffer", which it will then happily predict on
    any frame where pose detection drops out.
    """
    if n_frames <= 0:
        return []
    if n_frames <= clip_len:
        base = [i % n_frames for i in range(clip_len)]
        return [base for _ in range(max(1, n_clips))]

    windows = []
    max_start = n_frames - clip_len
    starts = np.linspace(0, max_start, num=max(1, n_clips)).astype(int)
    for s in starts:
        windows.append(list(range(s, s + clip_len)))
    return windows


def extract_span(
    detector,
    span: ann.Span,
    clip_len: int,
    num_person: int,
    clips_per_span: int,
    min_person_frames: float,
) -> tuple[list[np.ndarray], tuple[int, int]]:
    """Decode one annotated span.

    Returns (clips, (height, width)); the frame size travels with the clips so
    training can apply exactly the normalization the runtime will.
    """
    cap = cv2.VideoCapture(str(span.video))
    if not cap.isOpened():
        print(f"  ! cannot open {span.video}", file=sys.stderr)
        return [], (0, 0)

    img_shape = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
                 int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 25.0
    start_f = int(span.start * fps)
    end_f = int(span.end * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    frames: list[np.ndarray] = []
    detected = 0
    for _ in range(max(0, end_f - start_f)):
        ok, bgr = cap.read()
        if not ok or bgr is None:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        _boxes, scores, keypoints = detector.detect(rgb)
        if keypoints is not None and len(keypoints) > 0:
            detected += 1
        frames.append(select_top_persons(keypoints, scores, num_person))
    cap.release()

    if not frames:
        return [], img_shape

    # A span where pose almost never fires carries no learnable signal, and
    # including it mostly teaches the model to predict from absence.
    coverage = detected / len(frames)
    if coverage < min_person_frames:
        print(f"  ~ skipped (pose coverage {coverage:.0%} < "
              f"{min_person_frames:.0%}): {span.video.name} "
              f"{span.start:.1f}-{span.end:.1f}s")
        return [], img_shape

    stacked = np.stack(frames, axis=0)  # (T, M, 17, 3)
    clips = [stacked[idx] for idx in sample_indices(len(frames), clip_len, clips_per_span)]
    return clips, img_shape


def main() -> int:
    p = argparse.ArgumentParser(description="Extract skeleton clips from annotated video")
    p.add_argument("--annotations", required=True, help="annotation CSV")
    p.add_argument("--out", default="training/data/clips", help="output directory")
    p.add_argument("--pose-model", default="yolo11n-pose.onnx")
    p.add_argument("--clip-len", type=int, default=100,
                   help="frames per clip (must match inference clip_len)")
    p.add_argument("--num-person", type=int, default=2)
    p.add_argument("--clips-per-span", type=int, default=3,
                   help="temporal crops sampled from each annotated span")
    p.add_argument("--pose-conf", type=float, default=0.4)
    p.add_argument("--min-pose-coverage", type=float, default=0.25,
                   help="skip spans where pose fires on fewer than this fraction of frames")
    p.add_argument("--limit", type=int, default=0, help="debug: only process N spans")
    args = p.parse_args()

    if not Path(args.pose_model).exists():
        print(f"Pose model not found: {args.pose_model}", file=sys.stderr)
        return 1

    dataset = ann.load(args.annotations)
    print(dataset.summary())
    print()

    from detectors import YOLOPoseDetector
    detector = YOLOPoseDetector(args.pose_model, conf=args.pose_conf)

    out_root = Path(args.out)
    written = 0
    skipped = 0
    per_class = {c: 0 for c in CRIME_ACTIONS}

    spans = dataset.spans[: args.limit] if args.limit else dataset.spans
    for i, span in enumerate(spans, 1):
        print(f"[{i}/{len(spans)}] {span.label:16s} "
              f"{span.video.name} {span.start:.1f}-{span.end:.1f}s")
        if not span.video.exists():
            print(f"  ! missing video: {span.video}", file=sys.stderr)
            skipped += 1
            continue

        clips, img_shape = extract_span(detector, span, args.clip_len,
                                        args.num_person, args.clips_per_span,
                                        args.min_pose_coverage)
        if not clips:
            skipped += 1
            continue

        dest = out_root / span.split / span.label
        dest.mkdir(parents=True, exist_ok=True)
        stem = f"{span.video.stem}_{span.start:.1f}-{span.end:.1f}".replace(" ", "_")

        for n, clip in enumerate(clips):
            np.savez_compressed(
                dest / f"{stem}_{n}.npz",
                keypoints=clip.astype(np.float16),
                label=np.int64(span.label_index),
                img_shape=np.array(img_shape, dtype=np.int32),
                meta=json.dumps({
                    "video": str(span.video),
                    "start": span.start,
                    "end": span.end,
                    "label": span.label,
                    "split": span.split,
                    "img_shape": list(img_shape),
                    "notes": span.notes,
                }),
            )
            written += 1
            per_class[span.label] += 1

    print(f"\nWrote {written} clips to {out_root} ({skipped} spans skipped)")
    print("Clips per class:")
    for name, n in per_class.items():
        flag = ""
        if n and n < 30:
            flag = "  <- thin; expect poor recall on this class"
        print(f"  {name:16s} {n:5d}{flag}")

    if per_class.get("normal", 0) < sum(per_class.values()) * 0.3:
        print("\nWarning: fewer than 30% of clips are 'normal'. A weak negative "
              "set is the main cause of false crime alerts — add more ordinary "
              "footage before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
