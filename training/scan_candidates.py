"""Propose candidate incident spans in a video, to seed hand labelling.

Labelling from scratch means scrubbing a timeline looking for the few seconds
that matter. This narrows the search: it runs pose detection over the video and
ranks windows by signals that correlate with the interactions we care about —
two or more people close together, moving fast, with a sudden change in
separation (approach then flight is the snatch signature).

The output is a **draft annotation CSV with the label column left blank**. It is
a shortlist for a human to watch and label, not a detector. Precision here is
low by design; it is tuned to avoid missing incidents, not to be right.

    python -m training.scan_candidates \
        --video "videos/clip.mp4" \
        --out training/data/candidates.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from action.preprocess import select_top_persons          # noqa: E402
from training.annotations import format_time              # noqa: E402


def scan(video: Path, pose_model: str, sample_every: int, conf: float,
         max_person: int = 4) -> tuple[list[dict], float, tuple[int, int]]:
    """Run pose over sampled frames, returning per-sample measurements."""
    from detectors import YOLOPoseDetector
    detector = YOLOPoseDetector(pose_model, conf=conf)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    shape = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
             int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0))
    diag = float(np.hypot(*shape)) or 1.0

    samples: list[dict] = []
    prev = None
    idx = 0

    while True:
        ok, bgr = cap.read()
        if not ok or bgr is None:
            break
        if idx % sample_every == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            _boxes, scores, kpts = detector.detect(rgb)
            people = select_top_persons(kpts, scores, max_person)

            valid = people[..., 2] > 0.3
            n_people = int(valid.any(axis=-1).sum())

            # Centroid per detected person, for pairwise distance.
            centroids = []
            for m in range(max_person):
                v = valid[m]
                if v.any():
                    centroids.append(people[m, v, :2].mean(axis=0))

            min_gap = np.inf
            if len(centroids) >= 2:
                for a in range(len(centroids)):
                    for b in range(a + 1, len(centroids)):
                        min_gap = min(min_gap,
                                      float(np.linalg.norm(centroids[a] - centroids[b])))
            min_gap = min_gap / diag if np.isfinite(min_gap) else np.nan

            # Motion: mean joint displacement since the previous sample.
            motion = 0.0
            if prev is not None:
                both = valid & (prev[..., 2] > 0.3)
                if both.any():
                    d = np.linalg.norm(people[..., :2] - prev[..., :2], axis=-1)
                    motion = float(d[both].mean() / diag)

            samples.append({
                "t": idx / fps,
                "people": n_people,
                "gap": min_gap,
                "motion": motion,
            })
            prev = people.copy()

            if total and idx % (sample_every * 50) == 0:
                print(f"  {idx}/{total} frames ({idx / fps:5.1f}s)", flush=True)
        idx += 1

    cap.release()
    return samples, fps, shape


def rank_windows(samples: list[dict], window_s: float, stride_s: float) -> list[dict]:
    """Score sliding windows by interaction likelihood."""
    if not samples:
        return []
    times = np.array([s["t"] for s in samples])
    people = np.array([s["people"] for s in samples], dtype=float)
    gap = np.array([s["gap"] if s["gap"] == s["gap"] else 1.0 for s in samples])
    motion = np.array([s["motion"] for s in samples])

    out = []
    t = times[0]
    while t + window_s <= times[-1] + 1e-6:
        sel = (times >= t) & (times < t + window_s)
        if sel.sum() >= 2:
            n = people[sel]
            g = gap[sel]
            mo = motion[sel]

            multi = float((n >= 2).mean())            # fraction with 2+ people
            close = float((g < 0.18).mean())          # fraction with people close
            peak_motion = float(np.percentile(mo, 90))
            gap_swing = float(g.max() - g.min())      # approach-then-separate

            # Weighted toward the combination, not any single signal: a busy
            # street has motion everywhere and proves nothing on its own.
            score = (2.0 * multi * close
                     + 40.0 * peak_motion
                     + 4.0 * gap_swing)
            out.append({
                "start": float(t), "end": float(t + window_s), "score": score,
                "people_max": int(n.max()), "multi": multi, "close": close,
                "peak_motion": peak_motion, "gap_swing": gap_swing,
            })
        t += stride_s

    out.sort(key=lambda w: -w["score"])
    return out


def merge(windows: list[dict], gap_s: float = 0.5,
          max_span: float = 12.0) -> list[dict]:
    """Merge overlapping high-scoring windows so one event yields one row.

    Capped at `max_span`: a span you have to scrub through is no more useful
    than no span at all.
    """
    if not windows:
        return []
    ordered = sorted(windows, key=lambda w: w["start"])
    merged = [dict(ordered[0])]
    for w in ordered[1:]:
        last = merged[-1]
        if (w["start"] <= last["end"] + gap_s
                and w["end"] - last["start"] <= max_span):
            last["end"] = max(last["end"], w["end"])
            last["score"] = max(last["score"], w["score"])
            last["people_max"] = max(last["people_max"], w["people_max"])
        else:
            merged.append(dict(w))
    merged.sort(key=lambda w: -w["score"])
    return merged


def main() -> int:
    p = argparse.ArgumentParser(description="Propose candidate incident spans")
    p.add_argument("--video", required=True)
    p.add_argument("--out", default="training/data/candidates.csv")
    p.add_argument("--pose-model", default="yolo11n-pose.onnx")
    p.add_argument("--sample-every", type=int, default=5, help="frames between samples")
    p.add_argument("--pose-conf", type=float, default=0.35)
    p.add_argument("--window", type=float, default=4.0, help="candidate span seconds")
    p.add_argument("--stride", type=float, default=1.0)
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--percentile", type=float, default=90.0,
                   help="keep windows scoring above this percentile before merging")
    p.add_argument("--max-span", type=float, default=12.0,
                   help="split merged spans longer than this")
    p.add_argument("--normals", type=int, default=8,
                   help="also emit N low-scoring spans pre-labelled 'normal'")
    p.add_argument("--cache", default="", help="pose-scan cache path")
    p.add_argument("--rescan", action="store_true", help="ignore any cached scan")
    args = p.parse_args()

    video = Path(args.video)

    # Pose is the expensive part; cache it so thresholds can be retuned in
    # seconds instead of re-scanning the video each time.
    cache = Path(args.cache) if args.cache else (
        Path("training/data/.scan_cache") / f"{video.stem}_e{args.sample_every}.npz")

    if cache.exists() and not args.rescan:
        with np.load(cache, allow_pickle=False) as d:
            samples = [{"t": float(t), "people": int(p), "gap": float(g),
                        "motion": float(m)}
                       for t, p, g, m in zip(d["t"], d["people"], d["gap"], d["motion"])]
            fps = float(d["fps"])
            shape = tuple(int(v) for v in d["shape"])
        print(f"Loaded cached scan of {video.name} ({cache})")
    else:
        print(f"Scanning {video.name} …")
        samples, fps, shape = scan(video, args.pose_model, args.sample_every,
                                   args.pose_conf)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache,
            t=np.array([s["t"] for s in samples], dtype=np.float32),
            people=np.array([s["people"] for s in samples], dtype=np.int32),
            gap=np.array([s["gap"] for s in samples], dtype=np.float32),
            motion=np.array([s["motion"] for s in samples], dtype=np.float32),
            fps=np.float32(fps), shape=np.array(shape, dtype=np.int32))

    print(f"  {len(samples)} sampled frames, {shape[1]}x{shape[0]} @ {fps:.1f}fps")

    with_people = [s for s in samples if s["people"] > 0]
    multi = [s for s in samples if s["people"] >= 2]
    print(f"  person detected in {len(with_people)}/{len(samples)} samples "
          f"({len(with_people) / max(len(samples), 1):.0%})")
    print(f"  2+ people in {len(multi)}/{len(samples)} samples "
          f"({len(multi) / max(len(samples), 1):.0%})")

    ranked = rank_windows(samples, args.window, args.stride)
    if not ranked:
        print("No scorable windows — pose fired too rarely to propose spans.")
        return 1

    # Keep only the strongest windows *before* merging. Merging first would
    # chain every overlapping window together (stride < window, so they all
    # overlap) and collapse the whole video into a single span.
    scores = np.array([w["score"] for w in ranked])
    cutoff = float(np.percentile(scores, args.percentile))
    strong = [w for w in ranked if w["score"] >= cutoff]
    spans = merge(strong, max_span=args.max_span)
    windows = spans[: args.top]
    print(f"  {len(ranked)} windows scored, {len(strong)} above the "
          f"{args.percentile:.0f}th percentile, merged into {len(spans)} spans")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Quiet stretches that still contain people make the best negatives: same
    # camera, same lighting, same crowd, no incident. Those are exactly the
    # examples that stop the model calling ordinary street life a crime.
    quiet = [w for w in reversed(ranked)
             if w["people_max"] >= 1 and w["score"] < cutoff]
    normals = merge(quiet[: max(args.normals * 4, 0)], max_span=args.max_span)[: args.normals]

    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["video", "start", "end", "label", "split", "notes"])
        for c in windows:
            w.writerow([str(video), f"{c['start']:.2f}", f"{c['end']:.2f}",
                        "", "train",
                        f"CANDIDATE score={c['score']:.2f} "
                        f"people={c['people_max']} — watch and label or delete"])
        for c in normals:
            w.writerow([str(video), f"{c['start']:.2f}", f"{c['end']:.2f}",
                        "normal", "train",
                        f"auto-negative score={c['score']:.2f} — confirm nothing happens here"])

    print(f"\nTop {len(windows)} candidate spans (label column left blank):\n")
    print(f"{'#':>3} {'start':>10} {'end':>10} {'score':>7} {'people':>7}")
    print("-" * 42)
    for i, c in enumerate(windows, 1):
        print(f"{i:3d} {format_time(c['start']):>10} {format_time(c['end']):>10} "
              f"{c['score']:7.2f} {c['people_max']:7d}")

    if normals:
        print(f"\nPlus {len(normals)} low-scoring spans pre-labelled 'normal':")
        for c in normals:
            print(f"    {format_time(c['start'])} - {format_time(c['end'])}")

    print(f"\nWrote {out}")
    print("\nThese are search hints, not detections. Watch each span, fill in the "
          "label column, delete the rows that are nothing, and add `normal` spans "
          "from the quiet stretches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
