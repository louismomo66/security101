"""Audit a YOLO dataset before training on it.

Why this exists
---------------
The first weapon model was trained on Sohas, whose `pistol` boxes have a median
area of **43% of the frame**. Guns in this project's footage occupy about
**0.4%**. That is a hundredfold scale gap, and the model that came out of it
learned "large gun-shaped region" rather than "gun". Measured consequence: it
classified motorcycle fuel tanks, a car front at night, an umbrella and a dark
animal as pistols, the last at 0.72 confidence.

No amount of training fixes a dataset whose objects are the wrong size. This
script measures that *before* the GPU time is spent.

It reports, per class:

  - how many instances there are, and in how many images
  - the distribution of box area as a percentage of frame
  - how that compares to the scale your deployment footage actually needs

and it flags datasets that are off by more than an order of magnitude.

Usage
-----
    # any YOLO-format folder (Kaggle download, Roboflow export, merged set)
    python -m training.weapons.audit_dataset --root path/to/dataset

    # compare against a different target scale (default 0.4% of frame)
    python -m training.weapons.audit_dataset --root ds --target-pct 0.4
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}

# Measured on this project's footage: a handgun at street distance in 720p
# occupies roughly 0.4% of the frame area.
DEFAULT_TARGET_PCT = 0.4


def find_pairs(root: Path) -> list[tuple[Path, Path]]:
    """Find (image, label) pairs under a YOLO-layout root.

    Handles the common variants: images/labels siblings, train/val/test splits,
    and Roboflow's <split>/images + <split>/labels layout.
    """
    pairs: list[tuple[Path, Path]] = []
    for img in root.rglob("*"):
        if img.suffix not in IMG_EXT or not img.is_file():
            continue
        # labels/ sits beside images/ at the same depth
        parts = list(img.parts)
        lbl = None
        for i in range(len(parts) - 1, -1, -1):
            if parts[i] == "images":
                cand = Path(*parts[:i], "labels", *parts[i + 1:])
                lbl = cand.with_suffix(".txt")
                break
        if lbl is None:
            lbl = img.with_suffix(".txt")
        pairs.append((img, lbl))
    return pairs


def read_names(root: Path) -> dict[int, str]:
    """Pull class names from a data.yaml if one is present."""
    for y in list(root.rglob("data.yaml")) + list(root.rglob("*.yaml")):
        try:
            text = y.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if "names" not in text:
            continue
        # Deliberately not importing pyyaml — this is a diagnostic, and a
        # malformed yaml should not stop the audit.
        seg = text.split("names", 1)[1]
        inner = seg[seg.find("[") + 1: seg.find("]")] if "[" in seg else ""
        if inner:
            vals = [v.strip().strip("'\"") for v in inner.split(",") if v.strip()]
            if vals:
                return dict(enumerate(vals))
        out: dict[int, str] = {}
        for line in seg.splitlines()[1:]:
            s = line.strip()
            if not s or ":" not in s:
                if out:
                    break
                continue
            k, v = s.split(":", 1)
            try:
                out[int(k.strip().lstrip("-").strip())] = v.strip().strip("'\"")
            except ValueError:
                break
        if out:
            return out
    return {}


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def main() -> int:
    p = argparse.ArgumentParser(description="Audit a YOLO dataset before training")
    p.add_argument("--root", required=True)
    p.add_argument("--target-pct", type=float, default=DEFAULT_TARGET_PCT,
                   help="object area as %% of frame in your deployment footage")
    p.add_argument("--max-images", type=int, default=0,
                   help="sample at most N images (0 = all)")
    args = p.parse_args()

    root = Path(args.root).expanduser()
    if not root.exists():
        print(f"not found: {root}", file=sys.stderr)
        return 1

    pairs = find_pairs(root)
    if args.max_images:
        pairs = pairs[: args.max_images]
    if not pairs:
        print(f"no images found under {root}", file=sys.stderr)
        return 1

    names = read_names(root)

    areas: dict[int, list[float]] = collections.defaultdict(list)
    imgs_with: dict[int, set] = collections.defaultdict(set)
    n_labelled = n_empty = n_missing = 0
    seg_instances = 0
    box_instances = 0

    for img, lbl in pairs:
        if not lbl.exists():
            n_missing += 1
            continue
        try:
            lines = [l for l in lbl.read_text().splitlines() if l.strip()]
        except Exception:
            n_missing += 1
            continue
        if not lines:
            n_empty += 1
            continue
        n_labelled += 1
        for line in lines:
            f = line.split()
            try:
                cid = int(float(f[0]))
            except (ValueError, IndexError):
                continue
            vals = [float(x) for x in f[1:]]
            if len(vals) == 4:
                box_instances += 1
                a = vals[2] * vals[3] * 100.0        # w*h as % of frame
            elif len(vals) >= 6 and len(vals) % 2 == 0:
                # polygon: shoelace area on normalized coords
                seg_instances += 1
                xs, ys = vals[0::2], vals[1::2]
                a = 0.0
                for i in range(len(xs)):
                    j = (i + 1) % len(xs)
                    a += xs[i] * ys[j] - xs[j] * ys[i]
                a = abs(a) / 2.0 * 100.0
            else:
                continue
            areas[cid].append(a)
            imgs_with[cid].add(str(img))

    total_inst = sum(len(v) for v in areas.values())
    print(f"\nDataset: {root}")
    print(f"  images                 {len(pairs)}")
    print(f"  with labels            {n_labelled}")
    print(f"  empty labels (negatives) {n_empty}")
    if n_missing:
        print(f"  MISSING label files    {n_missing}")
    print(f"  instances              {total_inst}  "
          f"({box_instances} boxes, {seg_instances} polygons)")

    if not total_inst:
        print("\nNo annotated instances — nothing to audit.")
        return 1

    print(f"\n  target object scale in your footage: {args.target_pct:.2f}% of frame\n")
    print(f"  {'class':<16} {'inst':>6} {'imgs':>6} "
          f"{'p10':>7} {'median':>8} {'p90':>7}   verdict")
    print("  " + "-" * 72)

    worst = 0.0
    for cid in sorted(areas):
        v = areas[cid]
        med = pct(v, 0.5)
        ratio = med / args.target_pct if args.target_pct > 0 else 0
        worst = max(worst, ratio)
        if ratio > 10:
            verdict = f"{ratio:.0f}x TOO LARGE"
        elif ratio < 0.1:
            verdict = f"{1/ratio:.0f}x too small"
        elif ratio > 3:
            verdict = f"{ratio:.1f}x large"
        else:
            verdict = "ok"
        label = names.get(cid, f"class_{cid}")
        print(f"  {label:<16} {len(v):>6} {len(imgs_with[cid]):>6} "
              f"{pct(v,0.1):>6.2f}% {med:>7.2f}% {pct(v,0.9):>6.2f}%   {verdict}")

    # Class balance
    counts = {names.get(c, f"class_{c}"): len(v) for c, v in areas.items()}
    if len(counts) > 1:
        lo, hi = min(counts.values()), max(counts.values())
        if hi > 3 * lo:
            print(f"\n  Class imbalance {hi/lo:.1f}:1 — the rare class will "
                  f"under-perform unless weighted or resampled.")

    if n_empty == 0:
        print("\n  No negatives in this set. Precision comes from images that "
              "contain nothing; merge your own hard negatives before training.")

    if worst > 10:
        print(f"\n  WARNING: the largest class median is {worst:.0f}x your "
              f"deployment scale.")
        print("  This is the failure mode that produced phantom pistols on "
              "motorcycle fuel tanks.")
        print("  Mitigations, in order of effect:")
        print("    1. Train at a higher --imgsz, or tile large images.")
        print("    2. Aggressive scale augmentation (scale=0.9) so small "
              "objects are actually seen.")
        print("    3. Prefer datasets shot at surveillance distance over posed "
              "close-ups.")
        print("    4. Evaluate on CCTV-framed footage, never on the training "
              "distribution.")
        return 2

    print("\n  Scale looks compatible with the target footage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
