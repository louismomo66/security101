"""Clip features for low-resolution anomaly detection.

Why this exists alongside the pose pipeline
-------------------------------------------
Four approaches have now been measured out on this project's own footage, all
for the same reason: at ~20px per person the detail they read is not in the
file.

  pose / ST-GCN     two skeletons in 0/23 frames at Naalya
  weapon detector   near-zero recall on far-field footage
  fine-tuned VLM    14% incident detection, 36% false alarms
  trajectory rule   0/5 — median track age 0.00s, identity cannot survive

Each asked "what is this person doing", which needs pixels on the person.
This module asks a different question, and it is the question the project's
own brief actually poses:

    Is this segment unusual for this camera?

That survives bad resolution because it never depended on detail. UCF-Crime is
320x240 — worse than our footage — and the published field gets ~80-85% AUC on
it with exactly this framing. What it gives up is naming the crime and
localising the actor; what it keeps is "these ten seconds are worth watching",
which is what a reviewer needs.

Features come from `r3d_18` pretrained on Kinetics-400: a 3D CNN whose 512-d
penultimate activations summarise motion over 16 frames. It ships with
torchvision, so this adds no dependency. It reads spatio-temporal texture, not
body parts, which is precisely why it degrades gracefully where pose does not.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

CLIP_FRAMES = 16          # r3d_18's native temporal window
INPUT_SIZE = 112          # and its native spatial size
# Kinetics normalisation, as the pretrained weights expect.
MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)

_model = None
_device = None


def _init():
    global _model, _device
    if _model is not None:
        return
    import torch
    from torchvision.models.video import R3D_18_Weights, r3d_18

    _device = ("mps" if getattr(torch.backends, "mps", None)
               and torch.backends.mps.is_available()
               else "cuda" if torch.cuda.is_available() else "cpu")
    m = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
    m.fc = torch.nn.Identity()      # keep the 512-d penultimate activations
    _model = m.to(_device).eval()
    print(f"r3d_18 (Kinetics-400) on {_device}")


def clip_windows(video: Path, start: float, end: float, stride_s: float = 1.0):
    """Yield (t_start, ndarray[16,112,112,3] uint8) windows across a span.

    Windows are sampled at `stride_s` and each spans CLIP_FRAMES frames of the
    source. Overlapping windows are deliberate: a 0.2s snatch falls inside some
    window regardless of where the boundaries land.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
    end = min(end, total) if end else total
    t = start
    while t + CLIP_FRAMES / fps <= end:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        frames = []
        for _ in range(CLIP_FRAMES):
            ok, f = cap.read()
            if not ok or f is None:
                break
            frames.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB),
                                     (INPUT_SIZE, INPUT_SIZE)))
        if len(frames) == CLIP_FRAMES:
            yield t, np.stack(frames)
        t += stride_s
    cap.release()


def embed(windows: list[np.ndarray], batch: int = 8) -> np.ndarray:
    """(N, 16, 112, 112, 3) uint8 -> (N, 512) float32."""
    import torch
    _init()
    out = []
    for i in range(0, len(windows), batch):
        chunk = np.stack(windows[i:i + batch]).astype(np.float32) / 255.0
        chunk = (chunk - MEAN) / STD
        # (B,T,H,W,C) -> (B,C,T,H,W)
        x = torch.from_numpy(chunk).permute(0, 4, 1, 2, 3).to(_device)
        with torch.no_grad():
            out.append(_model(x).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 512), np.float32)


def span_features(video: Path, start: float, end: float,
                  stride_s: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """(times, features) for one span of one video."""
    got = list(clip_windows(video, start, end, stride_s))
    if not got:
        return np.zeros(0), np.zeros((0, 512), np.float32)
    times = np.array([t for t, _ in got], dtype=np.float32)
    return times, embed([w for _, w in got])
