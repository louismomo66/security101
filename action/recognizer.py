"""Real-time action recognizer: accumulate pose sequences → ST-GCN inference.

Usage:
    recognizer = ActionRecognizer("checkpoints/stgcn_ntu60_joint.pth", device="mps")
    # Per frame (called from the WS loop):
    result = recognizer.update(keypoints)
    # result is None while buffer fills, then a dict with predictions.
"""
from __future__ import annotations

import collections
import time
from typing import Any

import numpy as np
import torch

from .ntu60 import NTU60_ACTIONS
from .preprocess import pre_normalize_2d, select_top_persons
from .stgcn import RecognizerGCN


class ActionRecognizer:
    """Wraps ST-GCN with a sliding-window pose buffer for real-time use.

    Parameters
    ----------
    checkpoint : str       Path to pyskl-format .pth checkpoint.
    device     : str       'cpu', 'mps', or 'cuda'.
    clip_len   : int       Temporal window size (default 100 to match pyskl config).
    stride     : int       Run inference every `stride` frames (default 30).
    num_person : int       Max persons per frame (top-N by confidence; default 2).
    top_k      : int       Return top-k action predictions (default 3).
    """

    def __init__(self, checkpoint: str, device: str = "cpu",
                 clip_len: int = 100, stride: int = 30,
                 num_person: int = 2, top_k: int = 5,
                 labels: list[str] | None = None,
                 img_shape: tuple[int, int] | None = None,
                 scales: tuple[int, ...] | None = None):
        self.device = device
        self.clip_len = clip_len
        self.stride = stride
        # Temporal scales evaluated per inference. See _infer() for why more
        # than one is necessary; ~0.2 s events vanish in a 3.3 s window.
        self.scales = tuple(sorted({clip_len, *(scales or (45, 20))},
                                   reverse=True))
        # Caps the sliding-window search so inference cost stays predictable.
        self.max_windows_per_scale = 40
        self.num_person = num_person
        self.top_k = top_k
        # Frame dimensions for pyskl-compatible normalization; learned from the
        # first frame if not supplied.
        self.img_shape = img_shape
        self.labels = labels or self._discover_labels(checkpoint)

        self.model = RecognizerGCN.from_checkpoint(
            checkpoint, device=device, num_classes=len(self.labels))

        # Circular buffer: list of frames, each (M, 17, 3) — (persons, joints, xyc)
        self._buffer: collections.deque[np.ndarray] = collections.deque(maxlen=clip_len)
        self._frame_count = 0
        self._last_result: dict[str, Any] | None = None

    @staticmethod
    def _discover_labels(checkpoint: str) -> list[str]:
        """Work out the class names for a checkpoint, in order of authority.

        1. A sidecar `<checkpoint>.labels.json` (written by training/train.py),
           or `labels.json` beside the weights.
        2. A `classes` list stored inside the checkpoint itself.
        3. The head's output width: 60 -> NTU-60, 120 -> NTU-120.

        Step 3 is what makes the pyskl NTU-120 checkpoint a genuine drop-in:
        the file carries no metadata, but its head shape is unambiguous.
        """
        import json
        import os

        for cand in (f"{checkpoint}.labels.json",
                     os.path.join(os.path.dirname(checkpoint), "labels.json")):
            if os.path.exists(cand):
                try:
                    with open(cand, encoding="utf-8") as fh:
                        data = json.load(fh)
                    labels = data["classes"] if isinstance(data, dict) else data
                    if isinstance(labels, list) and labels:
                        return [str(x) for x in labels]
                except Exception:
                    pass

        n = ActionRecognizer._checkpoint_num_classes(checkpoint)
        if n == 120:
            from .ntu120 import NTU120_ACTIONS
            return NTU120_ACTIONS
        if n and n != len(NTU60_ACTIONS):
            # Unknown taxonomy — better a truthful placeholder than silently
            # attaching the wrong names to real class indices.
            return [f"class_{i}" for i in range(n)]
        return NTU60_ACTIONS

    @staticmethod
    def _checkpoint_num_classes(checkpoint: str) -> int | None:
        """Read the classifier head width straight from the weights file."""
        try:
            raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except Exception:
            return None

        if isinstance(raw, dict):
            classes = raw.get("classes")
            if isinstance(classes, list) and classes:
                return len(classes)

        sd = raw.get("state_dict", raw) if isinstance(raw, dict) else {}
        for key in ("cls_head.fc.weight", "cls_head.fc_cls.weight"):
            w = sd.get(key)
            if w is not None and hasattr(w, "shape"):
                return int(w.shape[0])
        return None

    # ── public API ────────────────────────────────────────────────────────

    def update(self, keypoints: np.ndarray | None,
               scores: np.ndarray | None = None,
               img_shape: tuple[int, int] | None = None) -> dict[str, Any] | None:
        """Feed one frame of keypoints and optionally run inference.

        Parameters
        ----------
        keypoints : ndarray[P, 17, 3] or None
            Detected keypoints for P persons (from YOLOPoseDetector).
            Pass None if no persons detected (zero-padded frame is appended).
        scores : ndarray[P] or None
            Person confidence scores (used to pick top-M persons).

        Returns
        -------
        dict or None
            Returns prediction dict every `stride` frames once buffer is full,
            or the last cached prediction otherwise.
        """
        self._frame_count += 1
        if img_shape is not None:
            self.img_shape = (int(img_shape[0]), int(img_shape[1]))

        # Pick top-M persons by confidence, zero-pad if fewer
        self._buffer.append(
            select_top_persons(keypoints, scores, self.num_person)
        )

        # Run inference as soon as the *shortest* scale is satisfiable, not the
        # longest. Waiting for the full 100-frame buffer blinds the model for
        # the first 3.3 s of every stream — and on recorded footage that is
        # often exactly where the incident is. Scales that do not yet fit are
        # skipped inside _infer().
        min_frames = min(self.scales)
        if len(self._buffer) >= min_frames and (
                self._last_result is None
                or self._frame_count % self.stride == 0):
            self._last_result = self._infer()

        return self._last_result

    # ── internals ─────────────────────────────────────────────────────────

    def _pre_normalize(self, data: np.ndarray) -> np.ndarray:
        """PreNormalize2D — see action/preprocess.py (shared with training)."""
        return pre_normalize_2d(data, img_shape=self.img_shape)

    def _forward(self, data: np.ndarray) -> np.ndarray:
        """One forward pass over a (T, M, V, C) window → class probabilities."""
        data = self._pre_normalize(data.astype(np.float32).copy())
        x = torch.from_numpy(data).unsqueeze(0)  # (1, T, M, V, C)
        x = x.permute(0, 2, 1, 3, 4)             # (1, M, T, V, C)
        x = x.to(self.device)
        with torch.inference_mode():
            logits = self.model(x)
        return torch.softmax(logits, dim=-1)[0].cpu().numpy()

    def _infer(self) -> dict[str, Any]:
        """Run ST-GCN over the buffer at several temporal scales.

        ST-GCN average-pools over time, so a brief event is diluted by whatever
        else is in the window. A phone snatch lasts roughly 0.2 s; in a 100
        frame (3.3 s) buffer that is 6% of the input, and the classifier
        reports the *ambient* motion instead — usually "walking towards/apart
        from each other", which is precisely what a snatch looks like when
        averaged over three seconds.

        Measured on real footage (720p street video, YOLO-pose skeletons), the
        best crime-class probability across a 12 s segment was:

            clip_len=100 (3.3 s)   p = 0.026
            clip_len= 45 (1.5 s)   p = 0.341     <- 13x higher
            clip_len= 20 (0.7 s)   p = 0.257

        So we evaluate the long window (good for sustained actions like
        fighting or falling) *and* shorter ones (good for brief events), then
        keep the strongest evidence per class. `scale` records which window
        produced each result, because a detection that only appears at 0.7 s is
        weaker evidence than one visible across all three.
        """
        buf = np.stack(list(self._buffer), axis=0).astype(np.float32)
        n = len(buf)

        scales = [s for s in self.scales if s <= n] or [n]
        t0 = time.perf_counter()

        best = np.zeros(len(self.labels), dtype=np.float32)
        best_scale = np.zeros(len(self.labels), dtype=np.int32)
        per_scale: dict[int, str] = {}
        passes = 0

        for s in scales:
            # Short windows are highly alignment-sensitive: measured on a real
            # snatch, a 20-frame window ending on frame 83 scored 0.533 for
            # "grab other person's stuff", while the same window shifted four
            # frames scored near zero. Evaluating only the window ending at
            # "now" therefore misses most brief events. Slide the window back
            # in half-window steps so consecutive inferences tile the timeline
            # with 50% overlap and no gaps.
            # Step is a fraction of the window, not half of it. Measured on a
            # real snatch, the response is one frame wide at its peak (0.533)
            # with a 0.11-0.29 shoulder over the next four frames, so coarse
            # steps skip straight over the event. Short windows are cheap
            # (8 ms at scale 20 vs 64 ms at scale 100), which is what makes a
            # dense search affordable.
            # The shortest scale is searched at every single alignment. Any
            # step > 1 samples only one parity of frame index, and the peak in
            # the measured snatch sat on an odd frame while a step of 2 tested
            # only even ones — a miss caused purely by arithmetic. Because
            # `max_back == stride`, consecutive inferences tile every possible
            # end position exactly once, and this scale is the cheap one (8 ms
            # vs 64 ms at scale 100).
            step = 1 if s == min(scales) else max(1, s // 6)
            max_back = min(self.stride, n - s)
            offsets = list(range(0, max_back + 1, step)) or [0]
            if len(offsets) > self.max_windows_per_scale:
                offsets = offsets[: self.max_windows_per_scale]

            scale_best = None
            for back in offsets:
                end = n - back
                probs = self._forward(buf[end - s:end])
                passes += 1
                if scale_best is None or probs.max() > scale_best.max():
                    scale_best = probs
                better = probs > best
                best_scale[better] = s
                best[better] = probs[better]

            if scale_best is not None:
                per_scale[s] = self.labels[int(np.argmax(scale_best))]

        dt = time.perf_counter() - t0
        top_idx = np.argsort(-best)[: self.top_k]

        actions = [
            {
                "label": self.labels[i],
                "confidence": round(float(best[i]), 3),
                "scale": int(best_scale[i]),
            }
            for i in top_idx
        ]
        return {
            "actions": actions,
            "scales": {str(k): v for k, v in per_scale.items()},
            "windows": passes,
            "time_ms": round(dt * 1000, 1),
        }
