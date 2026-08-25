"""Keep only the images whose objects are the size your cameras actually see.

Public weapon datasets are dominated by close-ups: product photographs, posed
studio shots, hands held up to a webcam. Measured on the Kaggle gun/knife set,
**44% of handgun boxes fill more than half the frame**, and the class median is
38% against the ~0.4% a street camera produces. Train on that and the model
learns size as a feature, then fires on anything large and roughly gun-shaped —
motorcycle fuel tanks, a car front at night, a dark animal.

Filtering is blunt but effective: an image is kept only if *every* annotated
instance in it is below `--max-pct` of frame area. Partial keeps are not
allowed, because dropping one box from an image turns a labelled object into an
unlabelled one, which teaches the detector that guns are background.

    python -m training.weapons.filter_by_scale \
        --src training/data/weapons/kaggle_gunknife \
        --dst training/data/weapons/gunknife_scaled \
        --max-pct 5.0

Run `audit_dataset.py` on the result before training on it.
"""
from __future__ import annotations

import argparse
import collections
import shutil
import sys
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def instance_areas(label_path: Path) -> list[tuple[int, float]]:
    """Return (class_id, area_as_pct_of_frame) for each instance."""
    out: list[tuple[int, float]] = []
    if not label_path.exists():
        return out
    for line in label_path.read_text().splitlines():
        f = line.split()
        if len(f) < 5:
            continue
        try:
            cid = int(float(f[0]))
            vals = [float(x) for x in f[1:]]
        except ValueError:
            continue
        if len(vals) == 4:
            area = vals[2] * vals[3] * 100.0
        elif len(vals) >= 6 and len(vals) % 2 == 0:
            xs, ys = vals[0::2], vals[1::2]
            a = 0.0
            for i in range(len(xs)):
                j = (i + 1) % len(xs)
                a += xs[i] * ys[j] - xs[j] * ys[i]
            area = abs(a) / 2.0 * 100.0
        else:
            continue
        out.append((cid, area))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Filter a YOLO dataset by object scale")
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--max-pct", type=float, default=5.0,
                   help="drop images containing any instance larger than this "
                        "%% of frame area")
    p.add_argument("--min-pct", type=float, default=0.0,
                   help="also drop instances smaller than this (0 = keep all). "
                        "Objects a few pixels across have no learnable shape.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if not src.exists():
        print(f"not found: {src}", file=sys.stderr)
        return 1

    images = [q for q in (src / "images").rglob("*") if q.suffix.lower() in IMG_EXT]
    if not images:
        print(f"no images under {src}/images", file=sys.stderr)
        return 1

    kept: list[tuple[Path, Path, str]] = []
    dropped_big = dropped_small = dropped_empty = 0
    kept_counts: collections.Counter = collections.Counter()
    drop_counts: collections.Counter = collections.Counter()

    for img in images:
        rel = img.relative_to(src / "images")            # e.g. train/1.jpg
        lbl = (src / "labels" / rel).with_suffix(".txt")
        inst = instance_areas(lbl)

        if not inst:
            # Genuine negatives are valuable; keep them.
            dropped_empty += 1
            kept.append((img, lbl, str(rel.parent)))
            continue

        biggest = max(a for _c, a in inst)
        smallest = min(a for _c, a in inst)
        if biggest > args.max_pct:
            dropped_big += 1
            for c, _a in inst:
                drop_counts[c] += 1
            continue
        if args.min_pct and smallest < args.min_pct:
            dropped_small += 1
            continue

        kept.append((img, lbl, str(rel.parent)))
        for c, _a in inst:
            kept_counts[c] += 1

    print(f"source images      {len(images)}")
    print(f"  kept             {len(kept)}")
    print(f"  dropped (too big) {dropped_big}")
    if args.min_pct:
        print(f"  dropped (too small) {dropped_small}")
    print(f"  of kept, unlabelled/negative {dropped_empty}")
    print()
    print(f"  instances kept    {sum(kept_counts.values())}  {dict(kept_counts)}")
    print(f"  instances dropped {sum(drop_counts.values())}  {dict(drop_counts)}")

    if kept_counts:
        lo, hi = min(kept_counts.values()), max(kept_counts.values())
        if hi > 3 * lo:
            print(f"\n  Class imbalance after filtering is {hi/lo:.1f}:1. The "
                  f"close-ups you removed were not evenly distributed across "
                  f"classes, so the rare class needs weighting or more data.")

    if args.dry_run:
        print("\ndry run — nothing written")
        return 0

    for split in {k for _i, _l, k in kept}:
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)

    for img, lbl, split in kept:
        shutil.copy(img, dst / "images" / split / img.name)
        target = dst / "labels" / split / f"{img.stem}.txt"
        target.write_text(lbl.read_text() if lbl.exists() else "")

    # Carry the class names across so the result is self-describing.
    for cand in (src / "data.yaml", src / "dataset.yaml"):
        if cand.exists():
            shutil.copy(cand, dst / "data.yaml")
            break

    print(f"\nwrote {len(kept)} images to {dst}")
    print("Now audit it:")
    print(f"    python -m training.weapons.audit_dataset --root {dst} --target-pct 0.4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
