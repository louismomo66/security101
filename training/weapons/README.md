# Training a weapon detector

COCO has no firearm class. `backend/weapons.py` therefore covers knives, bats
and scissors only, and `SENTINEL_WEAPON_MODEL` is the slot where a real weapon
detector plugs in. This directory is how you fill that slot.

---

## Read this before you start

On 2026-08-09 a public weapon detector was evaluated as a candidate:
`Subh775/Threat-Detection-YOLOv8n`, MIT licence, **93.1% claimed mAP@50** on its
`Gun` class. Measured on this project's own footage:

```
── FALSE ALARMS on 286 frames of confirmed weapon-free footage ──
  conf   frames w/ alert   per 100 frames  worst class
  0.45                57             19.9  explosion x28
  0.75                 3              1.0  grenade x3

── DETECTIONS on spans reported to contain weapons ──
  jewellery-store raid   0.45: 12/60   0.75:  0/60
  armed robbers, Uganda  0.45:  0/60   0.75:  0/60
```

At the app's default `weapon_conf` of 0.45 it alerted on **one in five frames**
of ordinary street footage. The "grenades" were **shrubs in a car park**; a
*critical* `weapon_brandished` fired on **police officers holding evidence
bags**. It only went quiet at 0.75 — by which point it detected nothing at all.

Three things follow, and they shape everything in this directory:

1. **A published mAP predicts almost nothing about your footage.** It is
   measured on the distribution the model was fitted on.
2. **Your bottleneck is precision, not recall.** A detector that misses some
   weapons is a partial system. One that fires on car parks trains its
   operators to ignore it, which is worse than having no detector.
3. **Grip geometry will not save you.** `_rule_weapon` promotes an object at a
   wrist to `weapon_brandished` critical. That defence assumes false boxes fall
   randomly. A model that boxes *people* puts its errors exactly where hands
   are, so layer 3 amplifies the failure instead of filtering it.

---

## The method

```
your footage ──► mine_negatives ──► negatives.zip ─┐
                 (empty labels)                     │
                                                    ├──► Colab: merge + train ──► best.onnx
public weapons dataset ─────────────────────────────┘                                │
                                                                                     ▼
                                                                              evaluate.py
                                                                          (the go/no-go gate)
```

### 1. Mine hard negatives from your own footage

```bash
python -m training.weapons.mine_negatives --every 0.6 --max-per-clip 900
```

Frames from your cameras containing no weapon, each with an **empty** `.txt`
label — YOLO for "nothing here". This is the ingredient no public dataset has,
and the cheapest precision you can buy.

Sources are listed explicitly in `NEGATIVE_SOURCES`, each with the reason it is
trusted. Spans labelled as incidents in `training/data/annotations.csv` are
excluded automatically, because a "negative" containing a weapon teaches the
model to suppress the thing you want found.

Near-duplicate frames are dropped: a fixed camera sampled every half second
yields hundreds of identical images that cost training time and let one
background dominate. Current yield is **286 frames** from 98 duplicates
discarded — including the 79 news-segment frames that produced the public
model's shrub-grenades.

```bash
cd training/data/weapons && zip -r negatives.zip negatives
```

### 1b. The public dataset

> **Sohas and the synthetic composites were deleted on 2026-08-15.** What
> follows replaces them. The history is kept below because the failure is the
> most useful thing this directory has produced.

Current source: a Kaggle set carrying **gun and knife as separate classes**,
imported into Roboflow and annotated with **polygons** rather than boxes.

```
class order (do not reorder): ['gun','knife']
```

Polygons are not decoration. The measured false positives were gun-*shaped
regions* — fuel tanks, a car front, a dark animal. A box head only has to agree
that something roughly rectangular is there; a segmentation head has to produce
a plausible silhouette, which those regions cannot supply.

#### Why Sohas was dropped

| Problem | Measurement |
|---|---|
| Object scale | `pistol` median box **43.4% of frame**; our footage ≈ **0.4%** |
| Composites | 228 guns pasted at arbitrary positions — mid-air, on pavement |
| Result | pistol fired on **7% of weapon-free frames** at conf 0.25 |

Inspecting all 24 false detections on clean footage: the plurality were
motorcycle parts. Uganda is full of boda bodas, and nothing in a Spanish
weapons dataset says a fuel tank is not a handgun.

**Audit any replacement before training it:**

```bash
python -m training.weapons.audit_dataset --root path/to/dataset --target-pct 0.4
```

It reports object area as a percentage of frame, per class, and refuses to
bless a set that is an order of magnitude off your deployment scale. Run it on
the Roboflow export too — an export is not automatically in-distribution just
because you labelled it yourself.

#### What the audit found on the Kaggle gun/knife set (2026-08-15)

`raghavnanjappan/weapon-dataset-for-yolov5`, 4,156 images, classes
`0: Knife, 1: Handgun`:

```
Knife      2268 inst   median  2.31%    5.8x large
Handgun    2383 inst   median 38.02%   95x TOO LARGE
```

**1,057 handgun boxes — 44% of the class — fill half the frame or more.**
Product photographs. Training on this as shipped would have reproduced the
Sohas failure exactly; the medians are barely distinguishable (38.0% vs 43.4%).

Filtering to instances under 5% of frame rescues it:

```bash
python -m training.weapons.filter_by_scale \
    --src training/data/weapons/kaggle_gunknife \
    --dst training/data/weapons/gunknife_scaled --max-pct 5.0
```

```
kept 1,708 / 4,156 images        (dropped 2,448 close-ups)
Knife      1481 inst   median 1.44%   3.6x large
Handgun     476 inst   median 1.56%   3.9x large
→ Scale looks compatible with the target footage
```

95x down to 3.9x, and the annotation job shrinks by 59%.

`filter_by_scale` keeps or drops **whole images**, never individual boxes.
Removing one box from an image would turn an annotated weapon into unannotated
background, which teaches the detector that guns are scenery — worse than the
close-up it was trying to remove.

#### In-domain gun positives

Filtering leaves handguns thin: 476 instances in 380 images, a 3.1:1 imbalance,
because the close-ups removed were overwhelmingly guns. The gap is filled from
our own footage, which contains real weapons at real surveillance distance:

| Source | Frames | Notes |
|---|---|---|
| Florida store, long gun, 1080p | 84 | Gun measured at **0.41% of frame** — the exact target scale |
| Armed robbers, Uganda, 848x480 night | 59 | Figures ~60px tall; a weapon here is a handful of pixels. Annotate what is genuinely visible and let the rest stand as in-domain negatives |

Extracted to `training/data/weapons/armed_frames/images`, deduplicated by
perceptual hash. Unannotated: they go to Roboflow to be labelled.

But measure the pistols before trusting them:

```
pistol box area, % of frame:  p25 10.0%   p50 43.4%   p75 81.3%   p95 93.2%
images at <= 0.5% of frame (the real scale):   9 of 1425
```

The median pistol fills **43% of its image**; a quarter fill over 80%. Your
Florida long gun is **0.41%**. A hundredfold gap — the dataset contains, in
effect, nine examples at the scale that matters, and many images are watermarked
stock photography on white backgrounds. **Do not train on it directly.**

### 1c. Composite guns onto your own footage

```bash
python -m training.weapons.composite --preview     # ALWAYS look first
python -m training.weapons.composite --n 700
```

This inverts the problem. The close-ups that are useless as training images are
*ideal as cut-out sources*: high resolution, unoccluded, plain backgrounds that
GrabCut separates cleanly — and it often keeps the hand gripping the gun, which
is exactly what layer 3 needs to see. Paste those onto your mined negatives and
you get domain-matched positives with exact boxes, free.

Scale comes from a **detected person**, not the frame: a handgun is ~200mm
against a ~1700mm person, so 8–22% of their box height. That adapts to camera
distance automatically.

Four gates, each added after the preview showed it was needed:

| Gate | Why |
|---|---|
| Reject frames where a person exceeds 55% of frame height | The negatives include news studios and piece-to-camera interviews. A gun beside a presenter's desk is not a scene any camera shows. |
| Reject standing-figure aspect < 1.4 | A talking-head close-up makes "person height" meaningless; the gun lands floating in the background. |
| `--min-gun-px 26` | In the Kampala aerials a person is 40–80px, so a *correct* handgun is 4–18px — fewer pixels than the object has parts. Skip rather than inflate, which would teach a wrong size prior. |
| `--min-contrast 18` | A dark pistol on dark trousers harmonises into invisibility. The box then points at gun-free pixels and teaches that ordinary clothing is a firearm — worse than no example, and invisible in any aggregate metric. |

That third gate is the resolution wall arriving from the other direction: on
far-field footage there is no honest way to composite a learnable gun, so the
generator quietly concentrates on backgrounds where people are large enough.

**Yield is the binding constraint, and it is a fact about the footage.** A run
asking for 700 images produced **228**, rejecting 8,172 attempts:

```
backgrounds able to host a >=26px gun:   65 of 286
median tallest person in a negative frame: 117px
```

A gun at the top of the physical range (22% of body height) needs a person
≥118px tall to clear 26px. The median negative frame offers 117px — the pool
sits exactly on the threshold, so most attempts fail by design rather than by
bug. Resulting scale distribution, which is the point of the exercise:

```
synthetic:  p5 0.089%   median 0.150%   p95 0.425%
real Florida gun:                       0.41%
Sohas as shipped:                      43.4%
```

To raise yield, **add closer-range negative footage** — clips where people are
150px+ — rather than lowering `--min-gun-px`. Weakening the gate manufactures
guns too small to have a learnable shape and teaches a false size prior; adding
backgrounds fixes the cause. Any clip added to `NEGATIVE_SOURCES` must first be
confirmed weapon-free, with the reason recorded in the entry.

**Look at `--preview` output at native resolution.** A 480px thumbnail of a
1080p frame shrinks a 30px gun to 13 pixels and makes every composite look
convincing. The preview crops tight instead.

### 2. Train on Colab

Open `colab_train_weapons.ipynb`. It walks through: upload negatives → download
a weapons dataset from Roboflow → merge the negatives into the **train** split
only → train `yolov8s` → export ONNX → print the class list in model order.

Two things it will tell you twice, because both are silent failures:

- **Negatives go in train, never val.** mAP is computed over ground-truth
  boxes; negatives have none, so padding val with them inflates the number
  while measuring less.
- **Record the class order.** `SENTINEL_WEAPON_CLASSES` must match the model's
  own order exactly. A mismatch silently relabels every detection, and
  `weapons.py` refuses to load without names precisely to avoid `class_0`
  events.

### 3. Measure it — the only step that decides anything

```bash
python -m training.weapons.evaluate \
    --model checkpoints/weapons_v1.onnx \
    --classes Gun knife            # model order, from the notebook
```

Sweeps confidence thresholds and reports **false alarms per 100 frames** of
confirmed weapon-free footage, then detections over spans reported to contain
weapons. It prints a verdict:

- **A usable threshold** — quiet on your negatives *and* still firing on the
  weapon spans. Set `weapon_conf` to it in the threat config.
- **UNUSABLE** — never gets quiet, or only by going blind. The script says
  which, because "quiet because it is right" and "quiet because it is blind"
  look identical in a single number.

Recall here is soft evidence: the positive spans are operator reports, not
verified boxes. Precision on the negatives is the hard number.

### 4. If it fails, add negatives — not epochs

The failure will be false positives on some specific thing: shrubs, bollards,
phones, wing mirrors. Add that footage to `NEGATIVE_SOURCES`, re-mine, retrain.
That loop *is* the method. More epochs on the same data will not fix a model
that has never seen a Kampala street without a weapon in it.

---

## Choosing a dataset

Three questions decide whether a public dataset is worth downloading:

- **Are the images CCTV-like, or product photography?** Catalogue photos of
  handguns produce a model that scores beautifully on itself and fails on a
  wide shot of a street. Prefer surveillance-derived data even at lower mAP.
- **What is in its negatives?** Most weapon datasets have almost none, which is
  precisely why step 1 exists.
- **Does it contain long guns, or only pistols?** Most public weapon datasets
  are heavily pistol-weighted. The clearest armed incident in this collection
  (`_I_m from Chicago_…`) is a **rifle or shotgun carried muzzle-down along the
  leg** — a long, thin, near-vertical object against a trouser leg, which
  shares almost no visual signature with a handgun held out at arm's length.
  A pistol-only dataset will not learn it.

## Do not train on the evaluation clips

The clips in `POSITIVE_SPANS` are the held-out test set. Training on them makes
the gate meaningless — it would report memorisation as precision, which is the
same mistake as trusting a published in-distribution mAP.

This matters here because `_I_m from Chicago_…` is tempting training data: it is
1080p, well lit, and the weapon is visible for ~100 seconds, so it would yield
hundreds of easy labelled frames. Resist it, or split it explicitly by time and
record which half went where. Same for the negatives: `mine_negatives.py`
already excludes labelled incident spans, but if you add a clip to
`NEGATIVE_SOURCES` whose span **overlaps** one in `POSITIVE_SPANS`, you have
quietly merged train and test.

The same *file* appearing in both is fine when the spans are disjoint, and one
currently does: the smash-and-grab clip contributes negatives from 60–110s (the
news segment) and positives from 0–30s (the raid). Different scenes, no shared
frames. Check with:

```bash
python -c "
from training.weapons.mine_negatives import NEGATIVE_SOURCES
from training.weapons.evaluate import POSITIVE_SPANS
neg={n[0] for n in NEGATIVE_SOURCES}; pos={p[0] for p in POSITIVE_SPANS}
print('shared clips:', neg & pos or 'none')"
```

### What that clip establishes as a floor

The public model, measured on it:

```
  Florida store, long gun (1080p)  0.45: 8/64   0.75: 0/64
```

1080p, indoor, well lit, weapon several hundred pixels long and visible for
100 seconds — and it fires on ~12% of frames at 0.45, nothing at the threshold
where it stops alerting on car parks. Any model you train should be judged
against this clip first. Failing here means nothing else in the collection will
rescue it.

Roboflow Universe (free account) hosts the widest selection. Open Images V7 is
an alternative with `Handgun`, `Shotgun`, `Rifle`, `Knife` boxes and no API key,
but the images are web photographs rather than surveillance frames.

---

## Honest limits

A gun in a 626×360 far-field CCTV frame is a handful of pixels. This project's
own measurements found figures ~20px tall in the Naalya footage, where pose
finds two skeletons in 0/23 frames. **No weapon detector solves that** — it is a
resolution problem, and the fix is better cameras or closer ones.

Where this pipeline can realistically help is the mid-range case: the
jewellery-store raid, the Gayaza roadside, footage where a person occupies a
meaningful fraction of the frame.

And whatever you train, it stays a prioritisation aid for a human reviewer.
The best-funded vendors in this space route every alert through human analysts
and decline to publish an accuracy number. That is not modesty; it is what the
error rates require.
