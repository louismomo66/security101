"""Mine hard negatives for weapon detection from footage you already have.

Why this script exists
----------------------
On 2026-08-09 a public weapon detector (93.1% claimed mAP@50 on `Gun`) was
measured against this project's clips. It called 58 parking-lot **shrubs**
grenades, a **soda can** a knife, and raised a *critical* firearm alert on
**police officers holding evidence bags**. Full table in
`docs/THREAT_DETECTION.md`.

None of those errors are exotic. They are what happens when a detector trained
on curated weapon photographs meets ordinary video: shrubs, hands, bags, poles,
phones, wing mirrors. A public dataset cannot fix this, because a public
dataset contains almost no pictures of *your* streets not containing weapons.

That is what this script produces: frames from your own footage, in your own
lighting and resolution, labelled as containing nothing. In YOLO format a
negative is an image with an **empty** `.txt` label file, and it is the single
cheapest way to buy precision.

Two details that matter more than they look:

**Deduplicate.** A fixed CCTV camera sampled every half second yields hundreds
of near-identical frames. They cost training time and teach nothing — worse,
they let one unlucky background dominate the negative set. Frames are kept only
when they differ enough from the last one kept.

**Never mine a span you have not confirmed.** A "negative" that actually
contains a weapon teaches the model to suppress exactly what you want it to
find. Spans come from an explicit list, and spans labelled as incidents in
`annotations.csv` are excluded automatically.

Usage
-----
    python -m training.weapons.mine_negatives --out training/data/weapons/negatives

    # preview what would be written, without writing it
    python -m training.weapons.mine_negatives --dry-run
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# Footage confirmed to contain no weapon, with the span to sample.
#
# `None` for the range means the whole clip. Each entry carries the reason it
# is trusted as a negative — an unexplained entry here is a future bug, because
# nobody will remember why it was safe.
NEGATIVE_SOURCES: list[tuple[str, float | None, float | None, str]] = [
    ("videos/Traffic in Kampala in Uganda is very chaotic - Klaas-Jan Gra_fe _1080p_ h264_.mp4",
     None, None,
     "Ordinary Kampala traffic, 1080p. The best domain match available: same "
     "streets, same vehicles, same light as the target deployment."),
    ("videos/California jewelry store targeted in smash-and-grab robbery - NBC News _720p_ h264_.mp4",
     60.0, 110.0,
     "News segment AFTER the raid — reporter interviews, an aerial of a car "
     "park, a studio anchor. This is the exact footage that produced 58 "
     "'grenade' detections (shrubs) and a critical 'knife' on police holding "
     "evidence bags. The highest-value negatives in the collection."),
    ("videos/Weird street crimes in Uganda caught on camera by the police - My adventures in Uganda _360p_ h264__2.mp4",
     36.48, 56.72,
     "Operator-confirmed non-incident span (2026-08-09); see annotations.csv."),
    ("videos/FASTEST AMBULANCE EVER_ - HyperXZ _720p_ h264_.mp4",
     None, None,
     "Ordinary road footage, no weapons."),
    ("videos/Hit and Run Accidents and the Consequences  Cyberabad Traffic Police  Road Safety - Cyberabad Traffic Police _720p_ h264_.mp4",
     None, None,
     "Traffic-safety compilation. Collisions, no weapons — and useful because "
     "it contains sudden motion, which is what a naive model over-fires on."),

    # ── The phone-snatch compilations ────────────────────────────────────
    #
    # These were collected as *positives* for snatch-theft detection. For the
    # weapon detector they are something better: roughly ten minutes of African
    # street scenes containing no weapon at all, dense with the exact objects
    # that produced the false positives — boda bodas, handlebars, fuel tanks,
    # mirrors, and night-time vehicle fronts.
    #
    # Measured on 2026-08-15: of 24 false detections on the then-current
    # negative set, the plurality were motorcycle parts classified as pistols.
    # Nothing in a public weapons dataset teaches a model that a boda fuel tank
    # is not a handgun. This footage does.
    ("videos/Phone Snatching Within 0.2 Seconds (Part 11) - Azuka Bellah (720p, h264).mp4",
     None, None,
     "8 minutes of street snatch-theft footage, 720p. No weapons anywhere in "
     "the clip — the offences are unarmed grabs. Dense in boda bodas, which "
     "are the single largest source of phantom pistol detections."),
    ("videos/Phone Snatching Done Within 0.2 Seconds _Part 21_ - Azuka Bellah _720p_ h264_.mp4",
     None, None,
     "Companion compilation to Part 11. Same rationale: unarmed thefts, so "
     "every frame is a weapon-negative in the target domain."),
    ("videos/The youngest phone snatcher in my life. - AFRICA AVENGERS (720p, h264).mp4",
     None, None,
     "Handheld and in-vehicle street footage, Kampala/Nairobi. Unarmed theft. "
     "Adds in-car and close-range framing absent from the fixed-camera clips."),
]


def load_incident_spans(csv_path: Path) -> dict[str, list[tuple[float, float]]]:
    """Spans labelled as anything other than `normal`, keyed by video stem.

    Matched on filename stem rather than full path: this collection holds
    byte-identical duplicates of the same video under several names, and a
    span confirmed in one of them is equally true of the others.
    """
    out: dict[str, list[tuple[float, float]]] = {}
    if not csv_path.exists():
        return out
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            label = (row.get("label") or "").strip().lower()
            if not label or label == "normal":
                continue
            try:
                start, end = float(row["start"]), float(row["end"])
            except (KeyError, ValueError):
                continue
            stem = Path(row["video"].strip()).stem
            # Duplicate filenames differ only by a trailing suffix; normalise.
            key = stem.split(" (")[0].split("_1")[0].split("_2")[0]
            out.setdefault(key, []).append((start, end))
    return out


def overlaps(t: float, spans: list[tuple[float, float]], pad: float = 1.0) -> bool:
    return any(a - pad <= t <= b + pad for a, b in spans)


def mine(video: Path, start: float | None, end: float | None, out_dir: Path,
         every: float, max_frames: int, dedup: float, incidents: list[tuple[float, float]],
         long_side: int, dry_run: bool) -> tuple[int, int]:
    """Sample one clip. Returns (kept, skipped_as_duplicate)."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"  !! cannot open {video.name}")
        return 0, 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 25.0
    total_s = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
    a = 0.0 if start is None else start
    b = total_s if end is None else min(end, total_s)

    kept = dup = 0
    prev_small: np.ndarray | None = None
    t = a
    while t < b and kept < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        t += every
        if not ok or frame is None:
            continue
        if overlaps(t, incidents):
            continue

        # Dedup on a tiny greyscale thumbnail: cheap, and insensitive to the
        # compression noise that would defeat a pixel-exact comparison.
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                           (64, 36)).astype(np.float32)
        if prev_small is not None and np.abs(small - prev_small).mean() < dedup:
            dup += 1
            continue
        prev_small = small

        h, w = frame.shape[:2]
        if max(h, w) > long_side:
            s = long_side / max(h, w)
            frame = cv2.resize(frame, (int(w * s), int(h * s)))

        stem = f"{video.stem[:40].replace(' ', '_')}_{t:07.2f}"
        if not dry_run:
            cv2.imwrite(str(out_dir / "images" / f"{stem}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            # An empty label file is how YOLO encodes "this image contains
            # none of the classes". The file must exist; a missing file is
            # treated as an unlabelled image and silently ignored.
            (out_dir / "labels" / f"{stem}.txt").write_text("", encoding="utf-8")
        kept += 1
    cap.release()
    return kept, dup


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default="training/data/weapons/negatives")
    p.add_argument("--every", type=float, default=2.0,
                   help="seconds between sampled frames")
    p.add_argument("--max-per-clip", type=int, default=400)
    p.add_argument("--dedup", type=float, default=6.0,
                   help="mean abs thumbnail difference below which a frame is "
                        "considered a near-duplicate of the last kept one")
    p.add_argument("--long-side", type=int, default=1280,
                   help="downscale frames whose longest side exceeds this")
    p.add_argument("--annotations", default="training/data/annotations.csv")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    out_dir = ROOT / args.out
    if not args.dry_run:
        (out_dir / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / "labels").mkdir(parents=True, exist_ok=True)

    incidents = load_incident_spans(ROOT / args.annotations)
    if incidents:
        print(f"excluding labelled incident spans from "
              f"{len(incidents)} video(s) in {args.annotations}\n")

    total = total_dup = 0
    for rel, start, end, why in NEGATIVE_SOURCES:
        video = ROOT / rel
        if not video.exists():
            print(f"  -- missing, skipped: {Path(rel).name}")
            continue
        key = video.stem.split(" (")[0].split("_1")[0].split("_2")[0]
        spans = incidents.get(key, [])
        span_txt = "whole clip" if start is None else f"{start:.1f}-{end:.1f}s"
        print(f"{Path(rel).name[:60]}\n    {span_txt}  — {why[:70]}")
        kept, dup = mine(video, start, end, out_dir, args.every,
                         args.max_per_clip, args.dedup, spans,
                         args.long_side, args.dry_run)
        print(f"    kept {kept}, dropped {dup} near-duplicates"
              f"{', excluded ' + str(len(spans)) + ' incident span(s)' if spans else ''}\n")
        total += kept
        total_dup += dup

    verb = "would write" if args.dry_run else "wrote"
    print(f"{verb} {total} negative frames (dropped {total_dup} duplicates) "
          f"to {args.out}")
    if not args.dry_run and total:
        print("\nEach image has an empty .txt label — that is YOLO for "
              "'nothing here'. Keep them in the TRAIN split; a val split made "
              "only of negatives reports a meaningless mAP.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
