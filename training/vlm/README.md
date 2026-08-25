# Fine-tuning a small VLM for tier-3 adjudication

Replicates [Yashraj Maher's LFM-UCF project](https://yashrajmaher.com/my-words/lfm-ucf):
LoRA fine-tune a small (1.6B) edge VLM on the UCF Crime dataset via Unsloth,
on a free Colab T4. Adapted here to target Sentinel's own tier-3 prompt
format instead of a standalone JSON schema, so a trained checkpoint can drop
straight into `backend/vlm_backend.py` as a third option
(`SENTINEL_VLM_BACKEND=lfm`) alongside the local FastVLM model and the
Claude API backend.

## Why this, not just "wait for FastVLM 1.5B"

`CLAUDE.md` documents FastVLM-0.5B's tier-3 adjudication as measured and
unreliable: it assented to leading questions and scored a clean control
higher than a confirmed incident (2026-08-09). The plan on record was to try
the 1.5B checkpoint next. Fine-tuning is a different, complementary lever —
zero-shot FastVLM has never seen a surveillance frame in training; UCF Crime
is real CCTV footage across 13 crime categories plus normal, exactly the
domain gap a bigger checkpoint alone doesn't close.

## Measured: the training mix sets the false-alarm rate

LFM2.5-VL-1.6B, LoRA r=16, vision frozen (0.57% trainable), on 64x64 UCF
frames from `odins0n/ucf-crime-dataset`. Two runs, identical except for how
much of the training set was the Normal class:

| Normal share | False alarms on ordinary frames | Crime-class match |
|---|---|---|
| 7% (balanced, the blog's recipe) | **34.8%** | 88.1% |
| 49% (`--normal-cap 5000`) | **12.0%** | 43.6% |

Read the first row carefully. 88.1% was measured on a *balanced* sample where
Normal was 1 of 14 classes; a camera makes it ~99% of the input. The same
checkpoint that scored 88.1% produced 34.8 false alarms per 100 ordinary
frames — worse than the public weapon detector this project rejected at 19.9.

Its hallucinations tracked the target frequencies exactly: "a theft" and
"a theft, possible weapon" led, being the most common crime targets. The
model had learned that something is usually wrong, because in its training
set something usually was.

Rebalancing cut false alarms 2.9x and halved crime recall. That is the trade,
and for a system whose alerts land in front of a human reviewer it is the
right side of it — but neither checkpoint is adoptable yet, and the numbers
above are UCF's own normal frames, not Kampala streets.

**The general lesson, now measured twice in this project:** balanced accuracy
is silent about the only question deployment asks. The weapon detector reached
the same conclusion from the object-detection side, where 286 domain negatives
cut false alarms tenfold. Negatives buy precision; nothing else does.

## Honest scope, read before trusting a trained checkpoint

**UCF Crime has video-level category labels, not frame-level weapon-holding
ground truth.** That means this cannot directly retrain the answer to
`backend/weapons.py::VERIFY_PROMPT` ("is this specific person holding a
weapon") — a "Robbery" video doesn't mean every sampled frame from it shows
a weapon in hand. What it CAN train is the broader tier-2 signal:
`ThreatEngine.threat_prompt()`'s two-line description + `INCIDENT:` line,
which asks "does this frame look like something is wrong", a coarser
question UCF Crime's category labels do answer honestly.

**A good score on held-out UCF Crime frames does not mean it works on
Uganda street footage.** Same rule as every other model in this project:
measure on your own footage before trusting it. `evaluate.py` in this
directory runs the trained checkpoint against `training/data/annotations.csv`
(the project's own 6 confirmed incident spans + 1 confirmed normal span) —
that is the number that decides whether this is usable, not the UCF Crime
eval score, exactly the same relationship `training/weapons/evaluate.py` has
to Roboflow's own reported mAP.

## Files

| File | Purpose |
|---|---|
| `prepare_ucf_dataset.py` | Downloads `tanzzpatil/ucf-crime-small`, maps its 14 categories to `threat_prompt()`-format targets, writes a train/held-out split. Run in Colab (needs internet) or anywhere with HF access — this sandbox's network is blocked from huggingface.co, so it has not been run yet. |
| `colab_train_lfm_ucf.ipynb` | LoRA fine-tune via Unsloth. Drive-mounted from the first cell, checkpointed every N steps, resumable — the same survival pattern `colab_train_weapons.ipynb` needed this session, applied from the start instead of learned the hard way. |
| `evaluate.py` | Runs the trained checkpoint against this project's own labelled spans, not UCF Crime. This is the number that matters. |

## Usage

```bash
# 1. In Colab, with internet:
python -m training.vlm.prepare_ucf_dataset --out /content/drive/MyDrive/ucf_prepared

# 2. Open colab_train_lfm_ucf.ipynb, point DATASET_DIR at the output above, run all cells.

# 3. Once trained and exported (LoRA merged + GGUF), locally:
python -m training.vlm.evaluate --model checkpoints/lfm_ucf.gguf
```

Then wire it in: `export SENTINEL_VLM_BACKEND=lfm` and
`export SENTINEL_LFM_MODEL=checkpoints/lfm_ucf.gguf` (see
`backend/lfm_vlm.py`). Same opt-in pattern as `SENTINEL_VLM_BACKEND=claude` —
the local FastVLM model stays the default until this is actually measured
and trusted.
