"""Score a weapon detector on *your* footage before you trust it.

The number a model ships with is almost always mAP@50 on its own validation
split — the same distribution it was fitted on. That number told us
`Subh775/Threat-Detection-YOLOv8n` was 93.1% on `Gun`. On this project's clips
the same model called shrubs grenades and police evidence bags knives.

So the only question this script asks is the one that decides whether a model
can be switched on:

    On footage confirmed to contain no weapon, how often does it alert?

That is reported as **false alarms per 100 frames**, swept across confidence
thresholds, because the useful output is not a verdict but a threshold: the
lowest confidence at which the model is quiet on ordinary footage. If no such
threshold exists below the point where the model stops detecting anything at
all, the model is unusable and the honest answer is to say so.

Recall is reported separately, over spans you have labelled as incidents. It is
deliberately secondary: a detector that misses half the weapons is a partial
system, while one that fires on car parks trains its operators to ignore it.

Usage
-----
    python -m training.weapons.evaluate \
        --model checkpoints/threat_yolov8n/weights/best.onnx \
        --classes Gun explosion grenade knife
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

THRESHOLDS = (0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75)

# Spans believed to contain a weapon. Recall here is soft evidence — the box is
# never checked against a ground-truth box, only "did it fire in the window".
POSITIVE_SPANS: list[tuple[str, float, float, str]] = [
    # The best positive available, and the one to judge a model on first:
    # 1920x1080, indoor, well lit, and the weapon is a **long gun** — wooden
    # stock, carried muzzle-down along the right leg — several hundred pixels
    # long rather than the ~20px figures in the Uganda CCTV. If a detector
    # cannot find this, nothing else in the collection will save it.
    #
    # Two spans, skipping the 32-40s mugshot insert (a news graphic, no gun).
    # Confirmed frame by frame at 3, 12, 20, 26, 28, 30, 48, 60, 75, 100s.
    ("videos/_I_m from Chicago_ Bro_ Would-Be Armed Robbery Suspect Backs Off When Florida Clerk Shows Gun - Law_Crime Network _1080p_ h264_.mp4",
     0.0, 32.0, "Florida store, long gun carried at side (1080p) [a]"),
    ("videos/_I_m from Chicago_ Bro_ Would-Be Armed Robbery Suspect Backs Off When Florida Clerk Shows Gun - Law_Crime Network _1080p_ h264_.mp4",
     42.0, 110.0, "Florida store, long gun carried at side (1080p) [b]"),
    ("videos/California jewelry store targeted in smash-and-grab robbery - NBC News _720p_ h264_.mp4",
     0.0, 30.0, "jewellery-store raid (operator reports weapons)"),
    ("videos/Armed Robbers caught on camera at night in Uganda - JUSTICE FREEMAN LIVE _480p_ h264_.mp4",
     0.0, 40.0, "armed robbers, Uganda, night (480p, far field)"),
]


def load_detector(model_path: str, class_names: list[str]):
    from detectors import YOLODetector
    resolved = str(ROOT / model_path) if not Path(model_path).is_absolute() else model_path

    import onnxruntime as ort
    n_out = None
    img_size = 640
    try:
        sess = ort.InferenceSession(resolved)
        out_shape = sess.get_outputs()[0].shape
        n_out = out_shape[1] if isinstance(out_shape[1], int) else None
        in_shape = sess.get_inputs()[0].shape  # [N, C, H, W]
        if isinstance(in_shape[2], int):
            img_size = in_shape[2]
    except Exception:
        pass

    nc = len(class_names)
    # Plain detection export: n_out = 4 + nc.
    # Segmentation export (e.g. yolov8s-seg): n_out = 4 + nc + 32 mask coeffs,
    # with a second model output ([1, 32, H, W] mask prototypes) that this
    # evaluator never reads — only the box+class columns are used.
    is_seg = n_out is not None and n_out - 4 - 32 == nc
    if n_out is not None and not is_seg and n_out - 4 != nc:
        raise SystemExit(
            f"model emits {n_out - 4} classes but {nc} names were "
            f"given ({class_names}). Names must be in the model's own class "
            f"order — a mismatch silently relabels every detection."
        )
    return YOLODetector(resolved, nc=nc if is_seg else None, img_size=img_size)


def sweep_negatives(det, neg_dir: Path) -> dict[float, dict]:
    imgs = sorted((neg_dir / "images").glob("*.jpg"))
    if not imgs:
        raise SystemExit(f"no negative frames in {neg_dir}/images — run "
                         f"`python -m training.weapons.mine_negatives` first")
    out = {t: {"frames_hit": 0, "dets": 0, "by_class": {}} for t in THRESHOLDS}
    for p in imgs:
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        boxes, scores, ids = det.detect(img, conf=min(THRESHOLDS))
        for t in THRESHOLDS:
            keep = [(s, i) for s, i in zip(scores, ids) if s >= t]
            if keep:
                out[t]["frames_hit"] += 1
                out[t]["dets"] += len(keep)
                for s, i in keep:
                    out[t]["by_class"][int(i)] = out[t]["by_class"].get(int(i), 0) + 1
    for t in THRESHOLDS:
        out[t]["n_frames"] = len(imgs)
    return out


def sweep_positives(det, every: float = 0.5) -> dict[str, dict[float, int]]:
    res: dict[str, dict[float, int]] = {}
    for rel, a, b, why in POSITIVE_SPANS:
        v = ROOT / rel
        if not v.exists():
            continue
        cap = cv2.VideoCapture(str(v))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 25.0
        hits = {t: 0 for t in THRESHOLDS}
        n = 0
        for tt in np.arange(a, b, every):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(tt * fps))
            ok, f = cap.read()
            if not ok:
                continue
            n += 1
            _, scores, _ = det.detect(cv2.cvtColor(f, cv2.COLOR_BGR2RGB),
                                      conf=min(THRESHOLDS))
            for t in THRESHOLDS:
                if any(s >= t for s in scores):
                    hits[t] += 1
        cap.release()
        hits["n_frames"] = n  # type: ignore[index]
        res[why] = hits
    return res


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True, help=".onnx in ultralytics export format")
    p.add_argument("--classes", nargs="+", required=True,
                   help="class names IN THE MODEL'S OWN ORDER")
    p.add_argument("--negatives", default="training/data/weapons/negatives")
    p.add_argument("--max-fp-per-100", type=float, default=1.0,
                   help="false alarms per 100 clean frames considered acceptable")
    args = p.parse_args()

    det = load_detector(args.model, args.classes)
    print(f"model   : {args.model}")
    print(f"classes : {args.classes}\n")

    neg = sweep_negatives(det, ROOT / args.negatives)
    n = neg[THRESHOLDS[0]]["n_frames"]
    print(f"── FALSE ALARMS on {n} frames of confirmed weapon-free footage ──")
    print(f"{'conf':>6}  {'frames w/ alert':>16}  {'per 100 frames':>15}  worst class")
    usable = None
    for t in THRESHOLDS:
        d = neg[t]
        rate = 100.0 * d["frames_hit"] / max(n, 1)
        worst = ""
        if d["by_class"]:
            ci, cn = max(d["by_class"].items(), key=lambda kv: kv[1])
            name = args.classes[ci] if ci < len(args.classes) else f"class_{ci}"
            worst = f"{name} x{cn}"
        print(f"{t:>6.2f}  {d['frames_hit']:>16}  {rate:>15.1f}  {worst}")
        if usable is None and rate <= args.max_fp_per_100:
            usable = t

    pos = sweep_positives(det)
    if pos:
        print(f"\n── DETECTIONS on spans reported to contain weapons ──")
        for why, hits in pos.items():
            nf = hits["n_frames"]  # type: ignore[index]
            row = "  ".join(f"{t:.2f}:{hits[t]:>3}/{nf}" for t in THRESHOLDS)
            print(f"  {why}\n     {row}")

    print("\n── VERDICT ──")
    if usable is None:
        print(f"  UNUSABLE at every threshold tried. The model never gets quieter")
        print(f"  than {args.max_fp_per_100} false alarms per 100 clean frames, so")
        print(f"  switching it on would flood a reviewer with alerts on ordinary")
        print(f"  footage. Do not wire it into SENTINEL_WEAPON_MODEL.")
    else:
        still = [why for why, h in pos.items() if h.get(usable, 0) > 0]
        print(f"  Quiet enough at conf >= {usable:.2f} "
              f"(<= {args.max_fp_per_100} false alarms / 100 clean frames).")
        if pos and not still:
            print(f"  BUT at that threshold it detects nothing in any span reported")
            print(f"  to contain a weapon — quiet because it is blind, not because")
            print(f"  it is right.")
        else:
            print(f"  Still fires on: {', '.join(still)}")
            print(f"  Set threat config `weapon_conf` to {usable:.2f} if adopted.")
    print("\n  Reminder: a span 'containing a weapon' here is an operator report,")
    print("  not a verified box. Treat recall as indicative only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
