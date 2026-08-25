# Training a crime action model

> **Try the NTU-120 checkpoint before you train anything.** NTU RGB+D 120
> already contains *touch other person's pocket* (A57 — pickpocketing), *grab
> other person's stuff* (A109 — snatch theft), *wield knife towards other
> person* (A107) and *shoot at other person with a gun* (A110). The pyskl
> checkpoint is a drop-in for this codebase — same architecture, same COCO-17
> skeleton layout:
>
> ```bash
> bash get_action_models.sh
> export ACTION_MODEL_PATH=checkpoints/stgcn_ntu120_joint.pth
> ```
>
> No labelling, no training, no GPU. Use it as your baseline, then use the
> pipeline below to **fine-tune** it on your own footage — which is where a few
> dozen labelled clips genuinely help, and where training from scratch on the
> same clips would not.

Fine-tunes the ST-GCN skeleton action recogniser on crime-specific classes
(phone snatching, pickpocketing, assault, …) starting from the NTU60 backbone
already in `checkpoints/`.

The model reads **skeletons only** — 17 COCO joints per person, no pixels. That
is a deliberate constraint: it makes the model small enough to run live on a
laptop, and it cannot key on skin tone, clothing, or background, because it
never sees them.

---

## Pipeline

```
annotation CSV ──► extract_poses ──► .npz skeleton clips ──► train ──► checkpoint
   (you label)      (YOLO pose,        (cached, fast)        (GPU)     + labels.json
                     slow, once)                                          │
                                                                          ▼
                                                          ACTION_MODEL_PATH → live app
```

---

## 1. Label your footage

Create `training/data/annotations.csv`:

```csv
video,start,end,label,split,notes
videos/kampala_street.mp4,00:01:12,00:01:16,phone_snatch,train,lady in blue
videos/kampala_street.mp4,00:00:00,00:01:08,normal,train,baseline crowd
videos/market.mp4,142.5,148.0,pickpocket,val,
```

- `start` / `end` accept seconds, `MM:SS`, or `HH:MM:SS.mmm`.
- `split` is `train` / `val` / `test`. Without a `val` split you cannot tell
  learning from memorisation, and the trainer will say so.
- Paths are relative to the CSV's directory, or absolute.

Classes live in `training/labels.py`. Two notes on the taxonomy:

**There is no `murder` class.** A skeleton shows a strike; it does not show
intent or outcome. Training a pose model to output "murder" would teach it to
assert something it cannot observe, and every such prediction would be
unfalsifiable from the input. Use `assault`, and escalate to a human.

**`normal` is the most important class.** Aim for at least a third of your clips
to be ordinary activity, drawn from the same cameras and times of day as the
incidents. A thin negative set is the single biggest cause of false alerts, and
false alerts are what make a system like this useless in practice — reviewers
stop looking.

Rough target: **50–100+ clips per class**, more for anything subtle like
pickpocketing.

### Finding the spans worth labelling

Scrubbing a timeline hunting for the few seconds that matter is the tedious
part. `scan_candidates` narrows it:

```bash
python -m training.scan_candidates \
    --video "videos/clip.mp4" \
    --out training/data/candidates.csv \
    --percentile 70 --normals 8
```

It runs pose over the video and ranks 4-second windows by signals that
correlate with the interactions we care about: two or more people, close
together, moving fast, with a sudden change in separation (approach then flight
is the snatch signature). It writes a draft CSV with the **label column blank**
for the top candidates, plus low-scoring spans pre-labelled `normal`.

This is a search hint, not a detector — precision is deliberately low, tuned to
avoid missing incidents. Watch each span, fill in the label, delete the rows
that are nothing.

The pose pass is cached under `training/data/.scan_cache/`, so retuning
`--percentile` or `--window` re-runs in a second instead of minutes.

## 2. Optional — bootstrap from UCF-Crime

```bash
python -m training.prepare_ucf_crime \
    --root ~/datasets/UCF_Crimes/Videos \
    --out training/data/ucf_annotations.csv
```

UCF-Crime labels whole videos, but a 4-minute "Robbery" video contains maybe 6
seconds of robbery. The script takes a bounded window from the middle of each
anomaly video as a **guess** and marks it `VERIFY` in the notes. Tighten those
spans by hand before extracting — span quality dominates final accuracy, and
loose spans mostly teach the model that normal footage is crime.

Categories with no pose signature (Arson, Explosion, RoadAccidents) are dropped
rather than forced into a class.

## 3. Extract skeletons

```bash
python -m training.extract_poses \
    --annotations training/data/annotations.csv \
    --out training/data/clips \
    --pose-model yolo11n-pose.onnx
```

One YOLO-pose pass per frame, so this is the slow stage — run it once. Spans
where pose fires on under 25% of frames are skipped; they carry no learnable
signal. Output is `.npz` per clip with **raw pixel coordinates** plus the source
frame size, so normalization stays a training-time decision.

## 4. Train

```bash
python -m training.train \
    --clips training/data/clips \
    --init checkpoints/stgcn_ntu60_joint.pth \
    --out checkpoints/stgcn_crime.pth \
    --epochs 40 --batch-size 16
```

The NTU60 backbone transfers; the head is reinitialised because the class count
differs. Useful flags:

| Flag | Why |
|---|---|
| `--balance weights\|sampler\|both` | Crime data is heavily skewed toward `normal`. Unweighted training predicts the majority class and reports flattering accuracy while catching nothing. |
| `--freeze-stages 4` | With few hundred clips, refitting low-level motion features overfits. Freeze early stages. |
| `--early-stop 12` | Stops when macro-F1 plateaus. |
| `--device cuda\|mps\|cpu` | Defaults to auto. |

Model selection uses **macro-F1, not accuracy** — on an 80% `normal` dataset,
accuracy rewards a model that never predicts a crime.

Outputs `checkpoints/stgcn_crime.pth` and `checkpoints/stgcn_crime.pth.labels.json`.

## 5. Evaluate

```bash
python -m training.evaluate \
    --checkpoint checkpoints/stgcn_crime.pth \
    --clips training/data/clips --split test
```

Prints per-class precision/recall/F1, a confusion matrix, sample
misclassifications with source timestamps, and the rate at which `normal` clips
get flagged as crime. Watch that last number: above ~5% a busy camera produces
alerts faster than anyone can review them.

## 6. Use it live

```bash
export ACTION_MODEL_PATH=checkpoints/stgcn_crime.pth
./dev.sh
```

`ActionRecognizer` reads the sidecar `labels.json` automatically, so no code
change is needed. Without the env var the stock NTU60 model loads as before.

---

## Normalization — read this before changing it

`action/preprocess.py` is imported by **both** training and the live recogniser.
They must stay identical; if they drift, the model sees a different distribution
at inference than it was fitted on and accuracy collapses silently.

An earlier version centred each skeleton on its own centroid. That makes two
people 300 px apart and two people 40 px apart produce byte-identical tensors —
and since every crime here *is* a relationship between two bodies, the signal was
being deleted before the network saw it. Measured effect on a two-class
synthetic set: stuck at 50% (predicting one class) before the fix, 100% after.

The current default matches pyskl's `PreNormalize2D` — normalize by frame
dimensions, which preserves both absolute position and inter-person distance.
This requires the frame size, threaded through `run_action(..., img_shape=)`.

---

## Limits worth being honest about

- **Pose-only means context-blind.** `vehicle_theft` is mostly "person at a car
  door", which is object context the detection layer sees and this model does
  not. Expect weak numbers; lean on `ThreatEngine` rules for the object side.
- **Two-person cap.** `num_person=2` by default. In a dense market crowd the
  two highest-confidence detections may not be the two people who matter.
- **Domain shift is severe.** A model trained on UCF-Crime CCTV angles will
  degrade on handheld phone footage, and vice versa. Validate on footage from
  the cameras you will actually deploy on.
- **This is decision support, not evidence.** Output is a ranked prompt for a
  human to review a clip. A confident `phone_snatch` score is a reason to look
  at 4 seconds of video — not a basis for accusing anyone. Keep the human in the
  loop, log false positives, and retrain on them.
