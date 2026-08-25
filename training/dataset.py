"""Skeleton clip dataset with pose-space augmentation.

Augmentation matters more here than in most vision tasks: labelled crime clips
are scarce, and a skeleton has few enough degrees of freedom that a model will
memorize individual incidents within a few epochs otherwise.

Every augmentation is applied to *raw pixel coordinates*, before normalization,
so that the normalization the model sees at training time is identical in form
to the one applied at inference.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from action.preprocess import pre_normalize_2d
from training.labels import CRIME_ACTIONS


class SkeletonClipDataset(Dataset):
    """Loads .npz clips produced by training/extract_poses.py.

    Parameters
    ----------
    root : str | Path        Directory containing <split>/<label>/*.npz
    split : str              'train' | 'val' | 'test'
    clip_len : int           Frames per sample (must match inference).
    augment : bool           Enable train-time augmentation.
    """

    def __init__(self, root: str | Path, split: str = "train",
                 clip_len: int = 100, augment: bool = False,
                 flip_prob: float = 0.5, rotate_deg: float = 12.0,
                 scale_jitter: float = 0.15, drop_joint_prob: float = 0.05,
                 temporal_jitter: bool = True):
        self.root = Path(root) / split
        self.clip_len = clip_len
        self.augment = augment
        self.flip_prob = flip_prob
        self.rotate_deg = rotate_deg
        self.scale_jitter = scale_jitter
        self.drop_joint_prob = drop_joint_prob
        self.temporal_jitter = temporal_jitter

        self.files: list[Path] = sorted(self.root.rglob("*.npz"))
        if not self.files:
            raise FileNotFoundError(
                f"No clips found under {self.root}. Run training/extract_poses.py first."
            )

        # Cache labels so class weights don't require reading every array.
        self.labels: list[int] = []
        for f in self.files:
            with np.load(f, allow_pickle=False) as d:
                self.labels.append(int(d["label"]))

    # ── stats ────────────────────────────────────────────────────────────

    def class_counts(self) -> np.ndarray:
        counts = np.zeros(len(CRIME_ACTIONS), dtype=np.int64)
        for y in self.labels:
            counts[y] += 1
        return counts

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights, normalized to mean 1.

        Crime datasets are heavily skewed toward `normal`; unweighted training
        converges to predicting the majority class and reports a flattering
        accuracy while catching nothing.
        """
        counts = self.class_counts().astype(np.float64)
        present = counts > 0
        w = np.ones_like(counts)
        w[present] = counts[present].sum() / (present.sum() * counts[present])
        return torch.tensor(w, dtype=torch.float32)

    def sample_weights(self) -> np.ndarray:
        """Per-sample weights for a WeightedRandomSampler."""
        cw = self.class_weights().numpy()
        return np.array([cw[y] for y in self.labels], dtype=np.float64)

    # ── augmentation ─────────────────────────────────────────────────────

    # COCO joint index pairs swapped by a horizontal mirror.
    _FLIP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10),
                   (11, 12), (13, 14), (15, 16)]

    def _augment(self, clip: np.ndarray,
                 img_shape: tuple[int, int] | None) -> np.ndarray:
        """clip: (T, M, V, 3) raw pixel coordinates, modified in place.

        All geometric transforms happen about the **frame centre**, not the
        subject centroid. Transforming about the subject would move people
        around the frame, and since normalization is frame-relative that would
        inject a spurious position change the label does not account for.
        """
        rng = np.random
        xy = clip[..., :2]
        conf = clip[..., 2]
        valid = conf > 0
        if not valid.any():
            return clip

        if img_shape and img_shape[0] and img_shape[1]:
            centre = np.array([img_shape[1] / 2.0, img_shape[0] / 2.0])
        else:
            centre = xy[valid].mean(axis=0)

        # Horizontal mirror — a left-handed snatch is the same event. Joint
        # indices must swap too, or left/right limbs end up crossed.
        if rng.random() < self.flip_prob:
            xy[..., 0] = 2 * centre[0] - xy[..., 0]
            for a, b in self._FLIP_PAIRS:
                clip[:, :, [a, b]] = clip[:, :, [b, a]]
            xy = clip[..., :2]
            conf = clip[..., 2]
            valid = conf > 0

        # Small rotation — camera tilt varies between installations.
        if self.rotate_deg > 0:
            theta = np.deg2rad(rng.uniform(-self.rotate_deg, self.rotate_deg))
            c, s = np.cos(theta), np.sin(theta)
            shifted = xy - centre
            xy[...] = np.stack([
                shifted[..., 0] * c - shifted[..., 1] * s,
                shifted[..., 0] * s + shifted[..., 1] * c,
            ], axis=-1) + centre

        # Scale jitter — subject distance from camera.
        if self.scale_jitter > 0:
            f = 1.0 + rng.uniform(-self.scale_jitter, self.scale_jitter)
            xy[...] = (xy - centre) * f + centre

        clip[..., :2] = xy * valid[..., None]

        # Random joint dropout — simulates occlusion and detector misses, which
        # are constant in real CCTV and rare in curated training clips.
        if self.drop_joint_prob > 0:
            mask = rng.random(conf.shape) < self.drop_joint_prob
            clip[mask] = 0

        return clip

    def _fit_length(self, clip: np.ndarray) -> np.ndarray:
        """Crop or loop a clip to exactly clip_len frames."""
        T = clip.shape[0]
        if T == self.clip_len:
            return clip
        if T > self.clip_len:
            start = (np.random.randint(0, T - self.clip_len + 1)
                     if (self.augment and self.temporal_jitter)
                     else (T - self.clip_len) // 2)
            return clip[start:start + self.clip_len]
        idx = [i % T for i in range(self.clip_len)]
        return clip[idx]

    # ── Dataset protocol ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        with np.load(self.files[i], allow_pickle=False) as d:
            clip = d["keypoints"].astype(np.float32)   # (T, M, V, 3)
            label = int(d["label"])
            # Older clips predate img_shape; fall back to centroid mode.
            img_shape = tuple(int(v) for v in d["img_shape"]) if "img_shape" in d else None

        clip = self._fit_length(clip).copy()
        if self.augment:
            clip = self._augment(clip, img_shape)

        clip = pre_normalize_2d(clip, img_shape=img_shape)

        # Model expects (M, T, V, C)
        x = torch.from_numpy(np.ascontiguousarray(clip.transpose(1, 0, 2, 3)))
        return x, label

    def meta(self, i: int) -> dict:
        with np.load(self.files[i], allow_pickle=False) as d:
            raw = d["meta"]
        return json.loads(str(raw))
