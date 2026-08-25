# VEREC — crime detection from video

Perception stack (YOLO detection + YOLO-pose + ST-GCN action recognition +
FastVLM captioning) feeding a rules engine that emits scored incidents for
**human review**. FastAPI backend, Next.js frontend.

The goal driving current work: detect **phone snatching and pickpocketing** in
African street footage (Uganda/Kenya), not just weapons and fighting.

---

## Run it

```bash
conda activate fastvlm            # or: source .venv/bin/activate
export ACTION_MODEL_PATH=checkpoints/stgcn_ntu120_joint.pth
./dev.sh                          # backend :8000, frontend :3000
```

`dev.sh` calls bare `python`. On this machine Homebrew's Python 3.14 shadows
conda on PATH, so `conda activate` changes the prompt but not the interpreter.
If imports fail, use the absolute path: `/opt/anaconda3/envs/fastvlm/bin/python`.
Always `python -m pip install`, never bare `pip`.

Models: `bash get_action_models.sh` (pose ONNX + both ST-GCN checkpoints).

---

## Architecture

```
frame ─┬─ YOLO detection ────────────┐
       ├─ YOLO pose ─► ST-GCN ───────┼─► ThreatEngine.update()      tier 1, every frame
       └─ FastVLM caption ───────────┴─► ThreatEngine.ingest_caption()  tier 2
                                             │
                                             ├─► VLM adjudication    tier 3, opt-in
                                             ▼
                                   scored alerts ─► WS payload / logs / snapshots
```

| Path | Purpose |
|---|---|
| `action/` | ST-GCN model, NTU60/NTU120 labels, shared preprocessing |
| `backend/threat.py` | Rules engine — all alert logic |
| `backend/adjudicate.py` | Tier-3 VLM second opinion |
| `training/` | Fine-tuning pipeline (labels → poses → train → eval) |
| `docs/THREAT_DETECTION.md` | Rule table, action-model docs, tier-3 usage |

### Two invariants worth knowing

**`action/preprocess.py` is imported by both training and inference.** If they
diverge, the model sees a different distribution at test time than it was fitted
on and accuracy collapses silently. Do not fork it.

**Normalization must preserve inter-person distance.** An earlier version
centred each skeleton on its own centroid, which made two people 300px apart and
two people 40px apart produce byte-identical tensors. Every crime here *is* a
relationship between two bodies, so that deleted the entire signal. Measured
effect: stuck at 50% (one class) before the fix, 100% after.

---

## Current state

Working:

- Video-file source with upload, seek, progress, loop (`source=file`)
- NTU120 checkpoint gives `pickpocket` (A57), `snatch_theft` (A109),
  `armed_threat` (A107/A110). Class count auto-detected from the checkpoint head
- Multi-scale temporal inference (100/45/20 frames) with sliding-window search
- Tier-3 VLM adjudication with fine-grained prompts + CLI scoring
- Training pipeline, verified end to end on synthetic data

Measured on real footage:

| Case | Result |
|---|---|
| Motorcycle snatch, full bodies (`Phone Snatching…Part 11`, ~0:03) | **detected**, 0.533 |
| Car-window snatch (`The youngest phone snatcher…`, 1:19) | **missed** — pose sees 1 person, crime classes peak at 0.18 |
| Naalya Nabe Rd motorcycle snatch (`Weird street crimes…`, 2–13s) | **unreachable** — 2 skeletons in 0/22 frames at conf 0.45 |
| Total Gayaza Rd snatch-and-run (`Weird street crimes…`, 64–72s) | **missed** — 0 events at conf 0.40. At 0.15, two *wrong* high-severity events: `vehicle_collision` 0.749 and `crowd_dispersal` 0.6 |
| False positives | ~3 per 15s |

---

## Open problems

**1. Precision is now the bottleneck, not recall.** `action_conf_brief=0.25`
buys recall at a real cost. `shoot at other person with a gun` scored 0.769 on a
window with (probably) no gun. Two candidate fixes, neither implemented:
suppress weapon classes unless the object detector also sees a weapon; or
require a class to persist across two consecutive inferences.

**2. Partial bodies defeat the skeleton model.** At the 1:19 snatch both people
have **0/6 leg keypoints** and one skeleton is 5× the size of the other. NTU was
recorded with two full-body actors at matched scale on a tripod. This is not a
degraded version of that — it is a different distribution. Lowering pose conf
0.30 → 0.10 recovers a second skeleton but not a correct classification.

**3. Tier 3 does not work at 0.5B — measured 2026-08-09.** Full table in
`docs/THREAT_DETECTION.md`. On the one confirmed snatch and two control
windows, the *controls* scored higher (0.820) than the incident (0.690). The
20–24s control is a woman **alone in a car**; FastVLM-0.5B said "yes" to
questions requiring two people and "There are two people in this image", while
its own caption correctly said "a woman seated inside a car". Giraffe/snow
controls got a correct "no". It assents to the premise in the question, and
every tier-3 question carries one.

The predicted failure mode was unusable answers; 0/40 were unparseable. Weak
VLMs fail *confidently and well-formatted*, which nothing downstream can catch.

Fixed by asking every question in both polarities and discarding answers that
affirm a claim and its negation, plus an unscored `two_people` gate. All three
windows now return `insufficient_evidence` — correct on the negatives, honest
on the positive. **Next: the 1.5B checkpoint** (`get_models.sh`), then re-run
the table. Tier 3 stays off by default until it beats these numbers.

**4. Ground truth: seven confirmed incidents.** 1:19 in `The youngest phone
snatcher…`, plus all of `Weird street crimes in Uganda…` labelled by the
operator on 2026-08-09. That clip is a **police compilation that annotates its
own answers** — red circles on the suspect, arrows, zoomed replays — which
makes it the cheapest ground truth available. Scene-cut detection gives the
segment boundaries; `scratchpad` contact sheets made it a confirm-not-scrub job.

| Span | Label | Scene |
|---|---|---|
| 0–11.04 | `phone_snatch` | Naalya Nabe Rd, daytime (replay 11.04–16.04) |
| 16.04–26.24 | `phone_snatch` | Kawempe Tula Round About, night |
| 26.24–36.48 | `phone_snatch` | Kajjansi CPS, daytime |
| 36.48–56.72 | **`normal`** | 2022-09-01 night — confirmed non-incident |
| 64–72 | `phone_snatch` | Total Gayaza Rd, night (segment 56.72–76.68) |
| 76.68–106.60 | `phone_snatch` | Christian Life Church junction (replay to 111.56) |

The `normal` span is deliberate, not a dropped row: `labels.py` calls the
missing negative set the single biggest failure mode of these models, and a
true negative from the same cameras and conditions is worth keeping.

Files are split, because `annotations.load()` is all-or-nothing and a blank
label is a hard error: `training/data/annotations.csv` holds the six confirmed
spans and loads clean; `candidates_uganda.csv` keeps `scan_candidates`' raw
unlabelled proposals, now useful as an eval set for the finder itself.

**Use `phone_snatch`, not `snatch_theft`, in annotation CSVs.** `snatch_theft`
is the ThreatEngine/NTU120 inference name; `training/labels.py::resolve()`
raises on it. Two vocabularies, no warning, easy to mix up.

What the labels exposed:

- **The candidate finder is better than one glance suggests.** Its proposals
  overlap 5/5 incidents, 12/14 touch a real one, and only 2 land in the
  confirmed `normal` span. But five incidents cover ~80% of a 112 s
  compilation, so overlap is nearly free — weak evidence, not validation. It
  needs scoring against footage that is mostly ordinary. (Reading only the
  top-ranked rows gives the opposite and wrong impression: they cluster on the
  busiest junction, which looks like pure crowd-density scoring.)
- **Resolution, not thresholds, is the wall — and it invalidates this clip as
  skeleton training data.** Frames with ≥2 skeletons, which is the minimum for
  a two-person interaction to exist at all:

  | Span | Label | ≥2 skeletons @0.45 | @0.25 | mean persons @0.45 |
  |---|---|---|---|---|
  | A Naalya | `phone_snatch` | 0/23 | 0/23 | 0.17 |
  | C Kawempe | `phone_snatch` | 0/21 | 0/21 | 0.05 |
  | D Kajjansi | `phone_snatch` | 0/21 | 0/21 | **0.00** |
  | E night | `normal` | 0/41 | **9/41** | 0.20 |
  | B Gayaza | `phone_snatch` | 5/16 | 7/16 | 0.88 |
  | F Junction | `phone_snatch` | 5/60 | 12/60 | 0.45 |

  Three of five positives are structurally invisible — at Kajjansi, a busy
  daytime market, pose finds *nobody*. The source is 626×360 and the figures
  run ~20px. Upscaling ×2 and ×3 before pose changes nothing (4/22 at every
  scale on A), so the information is not in the pixels and tiled/sliced
  inference would be building on false hope.

  **The asymmetry is the dangerous part.** The one confirmed `normal` span has
  more two-skeleton frames than any positive. Train on this as-is and the model
  learns "two visible skeletons ⇒ normal" — the exact inversion of the signal.
  These six spans are good for *evaluating* detection and VLM approaches; they
  are not a training set. That needs higher-resolution footage, full stop.

---

**5. Weapons are undetectable, and the obvious fix made it worse.** COCO has no
firearm class, so `_rule_weapon` can only ever see knife/bat/scissors.
`SENTINEL_WEAPON_MODEL` is the designed slot and is unset. The first candidate
tried — `Subh775/Threat-Detection-YOLOv8n`, 93.1% claimed mAP@50 on `Gun` —
called 58 parking-lot **shrubs** grenades, called a **soda can** a knife, raised
`weapon_brandished` **critical** on **police holding evidence bags**, and found
nothing at all in the one clip with real armed robbers. Full table in
`docs/THREAT_DETECTION.md`. Not wired up.

The lesson generalises: **grip geometry does not rescue a bad detector.** Layer
3 assumes false boxes fall randomly, so anchoring to wrists filters them. A
detector that boxes *people* puts its errors exactly where hands are, and layer
3 then promotes them to critical. Evaluate any weapon model on raw detections
over known-clean footage *before* trusting the layers above it.

`training/weapons/` is the pipeline for training a replacement: mine hard
negatives from our own footage (286 frames so far), train on Colab, then gate
adoption on `training.weapons.evaluate`, which reports false alarms per 100
weapon-free frames and refuses to call a model usable when it is merely blind.
**No model is adopted until it passes that gate** — `SENTINEL_WEAPON_MODEL`
stays unset.

---

## Bugs already found — do not reintroduce

- ThreatEngine read `action["action"]`; the recognizer returns
  `{"actions": [...]}`. Action rules **never fired at all**. Use
  `action_candidates()`, which accepts both shapes and reads the whole top-k
  list — on street footage the top-1 is almost always "walking towards/apart",
  with the crime class at rank 2-3.
- Inference gated on the 100-frame buffer filling = 3.3s blind at stream start.
  Gate on the *shortest* scale.
- Sliding the short window by 2 frames samples only one parity of frame index.
  The measured snatch peaked on an odd frame. Shortest scale steps by 1.
- Detection and pose disagree on person count (1 vs 2 in the same frame). The
  two-person guard uses `max()` of both.
- Detection and pose both drew person boxes → every person rendered twice.
- `training/__init__.py` must not `from __future__ import annotations` — it
  shadows the `training.annotations` submodule.
- `init_action()` defaulted to CPU while the VLM used `_resolve_device()`.

---

## Ground rules for this project

Alerts are **prioritisation aids for a human reviewer**, never findings, and
never routed to automated action. The theft classes describe a body movement:
a hand entering another person's pocket, or a grab followed by separation. They
do not observe ownership, consent or intent, and cannot. A parent taking a phone
from a child's hand produces the same skeleton as a thief.

Detection and pose models have documented accuracy differences across skin tone,
body size and clothing; rules built on them inherit that. This system will be
pointed at African street footage, which makes it a live concern rather than a
footnote.

There is no `murder` class and there should not be one — a skeleton shows a
strike, not intent or outcome. Use `assault` and escalate to a human.

Prefer measuring over tuning. Every threshold in `threat.py` is a
precision/recall trade-off, not a fact, and a system nobody has measured is one
that generates false alarms and trains its operators to ignore it.
