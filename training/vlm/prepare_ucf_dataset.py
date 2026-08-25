"""Build a fine-tuning set from UCF Crime, targeting Sentinel's own tier-3
prompt format instead of a standalone JSON schema.

Run this in Colab (or anywhere with internet — this project's own sandbox
has huggingface.co blocked, so this script is unverified against the live
dataset; check the printed sample count and a few examples before trusting
the output).

The source dataset (tanzzpatil/ucf-crime-small on HuggingFace) carries one
category label per image, drawn from the 13 UCF Crime anomaly classes plus
Normal. There is no frame-level weapon-holding ground truth and no
free-text caption, so the training TARGET this script builds is templated
from the category, not a genuine per-frame description. That is a real
limitation: the model learns "frames from Robbery videos get labelled
INCIDENT: theft, possible weapon", not "here is exactly what is happening
in this specific frame". Good enough to teach the model the *shape* of a
correct answer and the vocabulary of what counts as an incident; not a
substitute for genuinely captioned data if that becomes available later.

Target format matches backend/threat.py::ThreatEngine.threat_prompt()
exactly, so a checkpoint trained on this can answer that real production
prompt without any parsing changes downstream:

    <one-sentence description>
    INCIDENT: NONE
    -- or --
    <one-sentence description>
    INCIDENT: <a few words naming what is happening>

Usage
-----
    pip install datasets huggingface_hub
    python -m training.vlm.prepare_ucf_dataset --out /content/drive/MyDrive/ucf_prepared
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# UCF Crime category -> (one-sentence description template, INCIDENT: line).
# Description templates are deliberately generic (no genuine per-frame
# caption exists in the source data) — they teach the model the *register*
# of a correct Line 1, not a description of that specific frame's content.
# INCIDENT: vocabulary matches threat_prompt()'s own examples ("a weapon, a
# fight, a theft, forced entry, vandalism, fire, or an injured person") so
# the fine-tune reinforces the exact words ingest_caption() screens for,
# rather than teaching a different vocabulary that would need remapping.
CATEGORY_TARGETS: dict[str, tuple[str, str]] = {
    "Normal":       ("People are going about ordinary activity in the frame.", "NONE"),
    "Abuse":        ("A person appears to be mistreating or threatening another person.", "an assault, an injured person"),
    "Arrest":       ("Officers appear to be detaining or restraining a person.", "a struggle, a detainment"),
    "Arson":        ("A fire appears to have been deliberately started.", "fire, arson"),
    "Assault":      ("Two or more people appear to be in a physical altercation.", "a fight, an assault"),
    "Burglary":     ("A person appears to be forcing entry into a building or vehicle.", "forced entry, a burglary"),
    "Explosion":    ("An explosion or blast appears to be occurring.", "an explosion, fire"),
    "Fighting":     ("Multiple people appear to be physically fighting.", "a fight"),
    "Robbery":      ("A person appears to be taking property from another by force or threat.", "a theft, possible weapon"),
    "Shooting":     ("A person appears to be holding or firing a firearm.", "a weapon, a shooting"),
    "Shoplifting":  ("A person appears to be concealing store merchandise without paying.", "a theft"),
    "Stealing":     ("A person appears to be taking an item that is not theirs.", "a theft"),
    "Vandalism":    ("A person appears to be damaging property.", "vandalism"),
    "RoadAccidents": ("A vehicle collision appears to have occurred.", "a road accident, an injured person"),
}

PROMPT = (
    "You are reviewing a security camera frame.\n"
    "Line 1: describe what the people are doing, in one sentence.\n"
    "Line 2: write 'INCIDENT:' followed by NONE if nothing is wrong, or "
    "else a few words naming what you actually see happening (for example "
    "a weapon, a fight, a theft, forced entry, vandalism, fire, or an "
    "injured person).\n"
    "Do not list things that are absent."
)


# Publishers rename UCF's folders. Map the variants we have actually seen
# rather than letting them fall through to the Normal default.
CATEGORY_ALIASES = {
    "normalvideos": "Normal", "normal_videos": "Normal",
    "normal_videos_event": "Normal", "roadaccident": "RoadAccidents",
    "road_accidents": "RoadAccidents", "fight": "Fighting",
}

_warned: set[str] = set()


def build_target(category: str) -> str:
    key = category
    if key not in CATEGORY_TARGETS:
        key = CATEGORY_ALIASES.get(category.lower().replace(" ", "_"), None)
    if key is None:
        # Falling back to Normal here would label an unrecognised crime class
        # "INCIDENT: NONE" — training the model that the thing you want found
        # is nothing to report. Wrong data is worse than missing data, so this
        # is loud.
        if category not in _warned:
            _warned.add(category)
            print(f"  !! unrecognised category {category!r} — treating as Normal. "
                  f"If that is a crime class, add it to CATEGORY_TARGETS or "
                  f"CATEGORY_ALIASES before trusting this run.", flush=True)
        key = "Normal"
    desc, incident = CATEGORY_TARGETS[key]
    return f"{desc}\nINCIDENT: {incident}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True, help="output directory for prepared JSONL + images")
    p.add_argument("--src", default=None,
                   help="local directory of class-named subfolders of images, e.g. "
                        "/kaggle/input/ucf-crime-dataset/Train. Preferred over --hf-dataset: "
                        "Kaggle attaches datasets with no download, and it works with any "
                        "UCF-Crime copy regardless of who published it.")
    p.add_argument("--hf-dataset", default=None,
                   help="Hugging Face dataset id, as a fallback when --src is not given. "
                        "NOTE: the previous default, tanzzpatil/ucf-crime-small, does not "
                        "exist on the Hub. hibana2077/UCF-Crime-Dataset is a single 11.8 GB "
                        "zip that load_dataset cannot read, and its images are 64x64.")
    p.add_argument("--per-category-cap", type=int, default=1000,
                   help="max images per CRIME category")
    p.add_argument("--normal-cap", type=int, default=None,
                   help="max images for the Normal class specifically. Defaults to "
                        "--per-category-cap, i.e. the balanced recipe. Measured "
                        "2026-08-25: a balanced mix (431 Normal of 5,950 = 7%%) "
                        "produced 88.1%% exact match on a balanced sample and a "
                        "34.8%% FALSE ALARM RATE on ordinary frames — worse than "
                        "the public weapon detector this project rejected. Set "
                        "this to several times --per-category-cap so the training "
                        "mix resembles deployment, where ~99%% of frames are normal.")
    p.add_argument("--holdout-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not args.src and not args.hf_dataset:
        raise SystemExit(
            "give --src (a local directory of class-named subfolders) or --hf-dataset.\n"
            "On Kaggle, attach a UCF-Crime dataset and point --src at it, e.g.\n"
            "  --src /kaggle/input/ucf-crime-dataset/Train\n"
            "That needs no download and no Hugging Face access."
        )

    out_dir = Path(args.out)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    # ── Local directory mode ──────────────────────────────────────────────
    # UCF-Crime copies are almost always laid out as one folder per class:
    #     Train/Abuse/*.png, Train/Robbery/*.png, ...
    # Reading that directly avoids depending on any particular publisher's
    # HF column names, and on Kaggle it costs nothing because the dataset is
    # already mounted read-only under /kaggle/input.
    if args.src:
        src = Path(args.src)
        if not src.is_dir():
            raise SystemExit(f"--src is not a directory: {src}")
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        # Enumeration is bounded, and that matters more than it looks. UCF-Crime
        # copies run to ~1.27M files; on Kaggle's network-backed /kaggle/input a
        # full rglob takes many minutes and produces no output while it runs, so
        # it reads as a hang. We only ever keep `per_category_cap` per class, so
        # scan a multiple of that and sample from it — uniform enough for this,
        # and seconds rather than minutes.
        scan_cap = max(args.per_category_cap * 10, 5000)
        by_category: dict[str, list[Path]] = {}
        for child in sorted(src.iterdir()):
            if not child.is_dir():
                continue
            files: list[Path] = []
            for entry in os.scandir(child):          # one level; UCF is flat
                if entry.is_file() and Path(entry.name).suffix.lower() in exts:
                    files.append(Path(entry.path))
                    if len(files) >= scan_cap:
                        break
            if files:
                by_category[child.name] = files
                print(f"  {child.name}: scanned {len(files)}"
                      f"{' (capped)' if len(files) >= scan_cap else ''}", flush=True)
        if not by_category:
            raise SystemExit(
                f"no class subfolders with images under {src}.\n"
                f"contents: {[p.name for p in list(src.iterdir())[:10]]}"
            )
        print("category counts (raw):", {k: len(v) for k, v in by_category.items()})
        return _write_split(by_category, out_dir, args, loader=Image.open)

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("pip install datasets huggingface_hub first")

    print(f"loading {args.hf_dataset} ...")
    ds = load_dataset(args.hf_dataset, split="train")
    print(f"loaded {len(ds)} rows; columns: {ds.column_names}")

    # Best-effort column detection — HF dataset card conventions vary.
    label_col = next((c for c in ("label", "category", "class") if c in ds.column_names), None)
    image_col = next((c for c in ("image", "img") if c in ds.column_names), None)
    if label_col is None or image_col is None:
        raise SystemExit(
            f"could not find label/image columns in {ds.column_names} — "
            f"open the dataset card and adjust label_col/image_col above by hand."
        )

    by_category: dict[str, list[int]] = {}
    names = ds.features[label_col].names if hasattr(ds.features[label_col], "names") else None
    for i, row in enumerate(ds):
        raw = row[label_col]
        cat = names[raw] if names is not None and isinstance(raw, int) else str(raw)
        by_category.setdefault(cat, []).append(i)

    print("category counts (raw):", {k: len(v) for k, v in by_category.items()})

    return _write_split(by_category, out_dir, args,
                        loader=lambda i: ds[i][image_col])


def _write_split(by_category, out_dir: Path, args, loader) -> int:
    """Sample per category, write images and the train/holdout JSONL.

    `by_category` maps a category name to a list of items, and `loader` turns
    one item into a PIL image — an index into an HF dataset, or a path on disk.
    The two source modes differ only in that.
    """
    random.seed(args.seed)
    records = []
    img_idx = 0
    normal_cap = args.normal_cap if args.normal_cap is not None else args.per_category_cap
    for cat, items in by_category.items():
        items = list(items)
        random.shuffle(items)
        # The Normal class gets its own cap. Training on a balanced mix teaches
        # the model that something is usually wrong, which is true of the
        # training set and false of every camera it will ever see.
        is_normal = build_target(cat).endswith("INCIDENT: NONE")
        items = items[: (normal_cap if is_normal else args.per_category_cap)]
        target = build_target(cat if cat in CATEGORY_TARGETS else "Normal")
        for it in items:
            try:
                img = loader(it)
            except Exception as exc:            # a few UCF frames are truncated
                print(f"  skipped {it}: {exc}")
                continue
            img_path = out_dir / "images" / f"{img_idx:07d}.jpg"
            img.convert("RGB").save(img_path, quality=90)
            records.append({"image": str(img_path.relative_to(out_dir)),
                            "category": cat, "prompt": PROMPT, "response": target})
            img_idx += 1

    if not records:
        raise SystemExit("no images were written — check --src / --hf-dataset")

    random.shuffle(records)
    n_holdout = int(len(records) * args.holdout_frac)
    holdout, train = records[:n_holdout], records[n_holdout:]

    for name, split in (("train", train), ("holdout", holdout)):
        with open(out_dir / f"{name}.jsonl", "w") as f:
            for r in split:
                f.write(json.dumps(r) + "\n")

    n_normal = sum(1 for r in records if r["response"].endswith("INCIDENT: NONE"))
    pct = 100.0 * n_normal / len(records)
    print(f"\nnormal frames: {n_normal}/{len(records)} = {pct:.1f}% of the mix")
    if pct < 30:
        print("  !! A camera sees ~99% normal frames. Training at "
              f"{pct:.0f}% teaches the model that something is usually wrong. "
              "Measured effect at 7%: 34.8% false alarms on ordinary footage. "
              "Raise --normal-cap.")
    print(f"\nwrote {len(train)} train / {len(holdout)} holdout examples to {out_dir}")
    print("sample record:", json.dumps(records[0], indent=2))
    return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
