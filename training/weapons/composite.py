"""Composite real guns onto your own footage, at the scale they actually appear.

The problem this solves
-----------------------
Measured on the Sohas dataset (5,859 images, the best public option found):

    pistol box area, % of frame:  p25 10.0%   p50 43.4%   p75 81.3%
    images at <= 0.5% of frame:   9 of 1425

A gun in this project's footage is about **0.4%** of the frame — the Florida
long gun is roughly 190x45 px in 1920x1080. So the public dataset has, in
effect, nine usable examples. Train on it directly and the model learns what a
gun looks like filling a photograph, which is not a question anyone will ask it.

Meanwhile the close-ups that are useless as training images are *ideal as
source material*: high resolution, unoccluded, plain backgrounds that cut out
cleanly. And the one thing no public dataset has — Kampala streets, Florida
shop interiors, your cameras' noise and lighting — you already have 286 frames
of, mined as negatives.

So: cut the guns out of the close-ups, paste them onto your own backgrounds at
physically plausible scale, and get domain-matched positives with exact boxes
for free.

Scale is set relative to a **detected person**, not to the frame. A handgun is
roughly 200mm against a 1700mm person, so ~12% of their height — which
automatically produces a small gun for a distant figure and a large one for a
close figure, without needing to know the camera geometry.

Where the honesty runs out
--------------------------
A composite that looks pasted teaches the model to detect *pasting*. It will
then score beautifully on synthetic validation data and fail on real footage —
the same in-distribution self-deception that made a public model's 93.1% mAP
meaningless here. The mitigations below (blur matching, colour transfer,
grain, recompression) reduce that risk; they do not remove it.

Two rules follow, and neither is optional:

1. **Synthetic images never enter the evaluation set.** `evaluate.py` scores on
   real footage only. That separation is the entire safeguard.
2. **Look at the output.** `--preview` renders a sheet. If you can spot the
   pasted gun instantly, so can a neural network.

Usage
-----
    python -m training.weapons.composite --preview          # inspect first
    python -m training.weapons.composite --n 1500           # then generate
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

SOHAS = ROOT / "training/data/weapons/sohas/obj_train_data"
NEGATIVES = ROOT / "training/data/weapons/negatives/images"

# Physical priors. A handgun is ~200mm long against a ~1700mm person; a long
# gun ~900mm. Expressed as a fraction of the detected person's box height, so
# the result scales correctly with camera distance.
GUN_HEIGHT_FRAC = (0.08, 0.22)


def load_crop_library(min_area: float, limit: int, seed: int) -> list[np.ndarray]:
    """Cut guns out of the large close-ups. Returns BGRA images."""
    rng = random.Random(seed)
    cands: list[tuple[str, list[float]]] = []
    for p in glob.glob(str(SOHAS / "labels/train/*.txt")):
        for line in open(p, encoding="utf-8").read().split("\n"):
            if line.strip().startswith("0 "):          # class 0 == pistol
                f = line.split()
                if float(f[3]) * float(f[4]) >= min_area:
                    cands.append((p, [float(x) for x in f[1:5]]))
    rng.shuffle(cands)

    crops: list[np.ndarray] = []
    for lbl, (cx, cy, bw, bh) in cands:
        if len(crops) >= limit:
            break
        stem = Path(lbl).stem
        img = None
        for ext in (".jpg", ".png", ".jpeg"):
            f = SOHAS / "images/train" / f"{stem}{ext}"
            if f.exists():
                img = cv2.imread(str(f))
                break
        if img is None:
            continue
        h, w = img.shape[:2]
        x1, y1 = max(int((cx - bw / 2) * w), 0), max(int((cy - bh / 2) * h), 0)
        x2, y2 = min(int((cx + bw / 2) * w), w), min(int((cy + bh / 2) * h), h)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0 or min(crop.shape[:2]) < 32:
            continue

        mask = np.zeros(crop.shape[:2], np.uint8)
        rect = (3, 3, max(crop.shape[1] - 6, 1), max(crop.shape[0] - 6, 1))
        try:
            cv2.grabCut(crop, mask, rect, np.zeros((1, 65), np.float64),
                        np.zeros((1, 65), np.float64), 3, cv2.GC_INIT_WITH_RECT)
        except Exception:
            continue
        m = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)

        # Keep only the largest blob: GrabCut leaves specks of background that
        # would paste as floating debris and give the composite away.
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        if n > 1:
            big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            m = np.where(lab == big, 255, 0).astype(np.uint8)
        if m.mean() < 25:                      # cut-out failed, almost nothing left
            continue
        # Tighten to the surviving mask, then feather the edge by a pixel or
        # two so the paste has no razor-sharp boundary to key on.
        ys, xs = np.where(m > 0)
        crop, m = crop[ys.min():ys.max() + 1, xs.min():xs.max() + 1], \
            m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        m = cv2.GaussianBlur(m, (5, 5), 0)
        crops.append(np.dstack([crop, m]))
    return crops


def blur_of(img: np.ndarray) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def harmonise(patch: np.ndarray, bg_patch: np.ndarray) -> np.ndarray:
    """Match the crop's blur and colour statistics to the local background."""
    rgb, alpha = patch[:, :, :3].astype(np.float32), patch[:, :, 3]

    # Colour transfer, per channel, damped: a full match on a small patch
    # overshoots and turns the gun the colour of the pavement.
    vis = alpha > 128
    if vis.sum() > 20:
        for c in range(3):
            src, dst = rgb[:, :, c][vis], bg_patch[:, :, c].astype(np.float32)
            s_std = src.std() or 1.0
            scale = np.clip((dst.std() or 1.0) / s_std, 0.85, 1.2)
            # Damped hard: at 0.5 the gun took on the colour of the pavement
            # and disappeared, which trains the model on an object that is not
            # visibly there. Harmonisation should hide the *seam*, not the gun.
            shift = (dst.mean() - src.mean()) * 0.25
            rgb[:, :, c] = np.clip((rgb[:, :, c] - src.mean()) * scale
                                   + src.mean() + shift, 0, 255)

    # Blur match: a sharp gun on a soft CCTV frame is the single most obvious
    # compositing artefact.
    b_src, b_dst = blur_of(rgb.astype(np.uint8)), blur_of(bg_patch)
    if b_dst > 0 and b_src > b_dst * 2.5:
        # Capped at k=2: a heavier kernel on an already-small crop erases the
        # trigger guard and barrel outline — the shape the model must learn.
        k = int(np.clip(np.sqrt(b_src / max(b_dst, 1e-6)), 1, 2))
        rgb = cv2.GaussianBlur(rgb, (2 * k + 1, 2 * k + 1), 0)
    return np.dstack([rgb, alpha]).astype(np.uint8)


def paste(bg: np.ndarray, patch: np.ndarray, cx: int, cy: int) -> tuple[int, int, int, int] | None:
    ph, pw = patch.shape[:2]
    x1, y1 = int(cx - pw / 2), int(cy - ph / 2)
    x2, y2 = x1 + pw, y1 + ph
    H, W = bg.shape[:2]
    if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
        return None
    a = (patch[:, :, 3:4].astype(np.float32) / 255.0)
    region = bg[y1:y2, x1:x2].astype(np.float32)
    bg[y1:y2, x1:x2] = (patch[:, :, :3].astype(np.float32) * a
                        + region * (1 - a)).astype(np.uint8)
    return x1, y1, x2, y2


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default="training/data/weapons/synthetic")
    p.add_argument("--n", type=int, default=800, help="images to generate")
    p.add_argument("--crops", type=int, default=250, help="gun cut-outs to build")
    p.add_argument("--min-crop-area", type=float, default=0.25,
                   help="only cut guns from images where the box is at least "
                        "this fraction of the frame (i.e. the clean close-ups)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-contrast", type=float, default=18.0,
                   help="reject a placement when the gun's mean intensity is "
                        "within this of its surroundings — it has blended away")
    p.add_argument("--min-gun-px", type=int, default=26,
                   help="skip placements where a physically correct gun would "
                        "be shorter than this; below ~25px there is no shape "
                        "left to learn")
    p.add_argument("--preview", action="store_true",
                   help="render 8 examples to a sheet and stop")
    args = p.parse_args()

    if not SOHAS.exists():
        raise SystemExit(f"Sohas dataset not found at {SOHAS}")
    backgrounds = sorted(NEGATIVES.glob("*.jpg"))
    if not backgrounds:
        raise SystemExit(f"no backgrounds in {NEGATIVES} — run mine_negatives first")

    print(f"cutting out gun crops (>= {args.min_crop_area:.0%} of their frame)…")
    crops = load_crop_library(args.min_crop_area, args.crops, args.seed)
    print(f"  {len(crops)} usable cut-outs from {len(backgrounds)} backgrounds\n")
    if not crops:
        raise SystemExit("no usable cut-outs")

    from backend.models import run_detection
    rng = random.Random(args.seed)
    out = ROOT / args.out
    if not args.preview:
        (out / "images").mkdir(parents=True, exist_ok=True)
        (out / "labels").mkdir(parents=True, exist_ok=True)

    made, skipped, sheet = 0, 0, []
    target = 8 if args.preview else args.n
    attempts = 0
    while made < target and attempts < target * 12:
        attempts += 1
        bg = cv2.imread(str(rng.choice(backgrounds)))
        if bg is None:
            continue
        H, W = bg.shape[:2]
        persons = [o for o in run_detection(bg, conf=0.35)["objects"]
                   if o["class"] == "person"]
        if not persons:
            skipped += 1
            continue
        # Reject frames that are not surveillance-shaped. The negative set
        # legitimately contains news studios and piece-to-camera interviews —
        # they are real weapon-free footage — but a gun pasted beside a
        # presenter's desk is not a scene any camera will ever show, and a
        # head-and-shoulders close-up makes "person height" meaningless: the
        # gun ends up floating in the background at talking-head scale.
        tallest = max(p["box"][3] - p["box"][1] for p in persons)
        if tallest > 0.55 * H:
            skipped += 1
            continue

        usable = [p for p in persons
                  if 60 <= (p["box"][3] - p["box"][1]) <= 0.55 * H
                  # standing figure, not a face: taller than it is wide
                  and (p["box"][3] - p["box"][1]) > 1.4 * (p["box"][2] - p["box"][0])]
        if not usable:
            skipped += 1
            continue
        px1, py1, px2, py2 = rng.choice(usable)["box"]
        person_h = py2 - py1

        crop = crops[rng.randrange(len(crops))].copy()
        if rng.random() < 0.5:
            crop = cv2.flip(crop, 1)
        ang = rng.uniform(-35, 35)
        M = cv2.getRotationMatrix2D((crop.shape[1] / 2, crop.shape[0] / 2), ang, 1.0)
        crop = cv2.warpAffine(crop, M, (crop.shape[1], crop.shape[0]),
                              flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))

        # Size from the person, not the frame.
        want_h = person_h * rng.uniform(*GUN_HEIGHT_FRAC)
        # Physical honesty produces unlearnable data on far-field footage: in
        # the Kampala aerials a person is 40-80px, so a correctly scaled
        # handgun is 4-18px — fewer pixels than the object has parts. Rather
        # than silently inflate the gun (which would teach a wrong size prior),
        # skip the placement. In practice this selects backgrounds where people
        # are large enough for a weapon to be visible at all, which is the same
        # conclusion the resolution measurements reached from the other end.
        if want_h < args.min_gun_px:
            skipped += 1
            continue
        s = want_h / max(crop.shape[0], 1)
        nw, nh = max(int(crop.shape[1] * s), 8), max(int(crop.shape[0] * s), 8)
        crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)

        # Place at hand height, just outside the torso — where a carried
        # weapon actually sits, and where grip geometry will look for it.
        side = rng.choice([-1, 1])
        cx = int((px1 + px2) / 2 + side * (px2 - px1) * rng.uniform(0.25, 0.55))
        cy = int(py1 + person_h * rng.uniform(0.55, 0.80))
        x0, y0 = max(cx - nw, 0), max(cy - nh, 0)
        bgp = bg[y0:min(y0 + nh * 2, H), x0:min(x0 + nw * 2, W)]
        if bgp.size:
            crop = harmonise(crop, bgp)

        # Reject placements where the harmonised gun is no longer visibly
        # distinct from what it sits on — a dark pistol against dark trousers
        # blends to nothing. The box would then point at gun-free pixels, which
        # teaches the model that ordinary clothing is a firearm. Worse than
        # having no example at all, and invisible in any aggregate metric.
        vis_m = crop[:, :, 3] > 128
        if vis_m.sum() > 20:
            gun_px = crop[:, :, :3][vis_m].astype(np.float32)
            y0b, x0b = max(cy - nh, 0), max(cx - nw, 0)
            around = bg[y0b:min(cy + nh, H), x0b:min(cx + nw, W)]
            if around.size:
                contrast = abs(float(gun_px.mean()) - float(around.mean()))
                if contrast < args.min_contrast:
                    skipped += 1
                    continue

        box = paste(bg, crop, cx, cy)
        if box is None:
            skipped += 1
            continue
        x1, y1, x2, y2 = box

        # One JPEG round-trip over the whole frame so the pasted region carries
        # the same compression signature as everything around it.
        ok, enc = cv2.imencode(".jpg", bg, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if ok:
            bg = cv2.imdecode(enc, cv2.IMREAD_COLOR)

        if args.preview:
            # A 480px-wide thumbnail of a 1080p frame shrinks a 30px gun to 13
            # pixels, which makes every composite look convincing. Crop tight
            # at native resolution instead — judge it at the size the model
            # will see it.
            pad = max(nw, nh) * 2
            zx1, zy1 = max(x1 - pad, 0), max(y1 - pad, 0)
            zx2, zy2 = min(x2 + pad, W), min(y2 + pad, H)
            z = bg[zy1:zy2, zx1:zx2].copy()
            cv2.rectangle(z, (x1 - zx1, y1 - zy1), (x2 - zx1, y2 - zy1), (0, 0, 255), 1)
            z = cv2.resize(z, (300, 300), interpolation=cv2.INTER_NEAREST)
            cv2.putText(z, f"{nw}x{nh}px", (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)
            sheet.append(z)
        else:
            name = f"syn_{made:05d}"
            cv2.imwrite(str(out / "images" / f"{name}.jpg"), bg,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            cxn, cyn = ((x1 + x2) / 2) / W, ((y1 + y2) / 2) / H
            wn, hn = (x2 - x1) / W, (y2 - y1) / H
            (out / "labels" / f"{name}.txt").write_text(
                f"0 {cxn:.6f} {cyn:.6f} {wn:.6f} {hn:.6f}\n", encoding="utf-8")
        made += 1

    if args.preview:
        sheet_path = ROOT / "training/data/weapons/preview_composite.jpg"
        rows = [np.hstack(sheet[i:i + 2]) for i in range(0, len(sheet) - 1, 2)]
        cv2.imwrite(str(sheet_path), np.vstack(rows))
        print(f"wrote {sheet_path}")
        print("\nLOOK AT IT. If the pasted gun is instantly obvious to you, the")
        print("model will learn the paste rather than the gun.")
    else:
        print(f"wrote {made} synthetic images to {args.out} "
              f"({skipped} attempts skipped: no usable person)")
        print("\nClass 0 = pistol, matching Sohas order. These go in TRAIN only —")
        print("evaluate.py scores real footage, and that separation is the")
        print("whole safeguard against measuring your own compositing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
