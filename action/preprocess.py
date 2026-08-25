"""Skeleton preprocessing shared by training and runtime inference.

Training and inference **must** normalize poses identically or the model sees a
different distribution at test time than it was fitted on. Both paths import
from here so the two can never drift apart.

Layout convention throughout: arrays are (T, M, V, C) where
    T = frames, M = persons, V = 17 COCO joints, C = 3 (x, y, confidence).
"""
from __future__ import annotations

import numpy as np

# Keypoints below this confidence are treated as missing.
CONF_THRESHOLD = 0.3


def pre_normalize_2d(
    data: np.ndarray,
    img_shape: tuple[int, int] | None = None,
    conf_threshold: float = CONF_THRESHOLD,
) -> np.ndarray:
    """Map keypoints into roughly [-1, 1], preserving inter-person geometry.

    Two modes:

    ``img_shape=(h, w)`` — frame normalization, matching pyskl's PreNormalize2D
    (``x = (x - w/2) / (w/2)``). This is what the pretrained NTU60 checkpoint
    was fitted on, and it is the mode you want. Absolute position and the
    distance *between* people both survive.

    ``img_shape=None`` — fallback for when the frame size is unknown. Centres on
    a centroid shared by **all** persons in the clip and applies one global
    scale.

    The distinction is not cosmetic. An earlier version of this function centred
    each person on their own centroid, which makes two people 300px apart and
    two people 40px apart produce byte-identical tensors. Every crime this model
    is meant to catch — snatching, pickpocketing, assault — *is* a relationship
    between two bodies, so that normalization removed the entire signal before
    the network saw it. Keep inter-person geometry intact.

    Parameters
    ----------
    data : ndarray (T, M, V, C)
        Raw keypoints in pixel coordinates. Modified in place and returned.
    img_shape : (height, width) or None
        Source frame dimensions.
    conf_threshold : float
        Keypoints at or below this confidence are treated as missing.

    Returns
    -------
    ndarray (T, M, V, C)
    """
    if data.size == 0:
        return data

    xy = data[..., :2]                      # (T, M, V, 2)
    conf = data[..., 2]                     # (T, M, V)
    valid = conf > conf_threshold           # (T, M, V)

    # Zero-padding marks "no person detected". It must stay exactly zero, or the
    # model learns to read absence as a position.
    present = valid.any(axis=-1)            # (T, M)

    if img_shape is not None:
        h, w = float(img_shape[0]), float(img_shape[1])
        if w > 0 and h > 0:
            norm = np.empty_like(xy)
            norm[..., 0] = (xy[..., 0] - w / 2) / (w / 2)
            norm[..., 1] = (xy[..., 1] - h / 2) / (h / 2)
            xy[...] = norm * present[..., None, None]
            data[..., :2] = xy
            return data

    # Fallback: one centroid for the whole clip, across all persons.
    if valid.any():
        centroid = xy[valid].mean(axis=0)                    # (2,)
        xy -= centroid * present[..., None, None]
        max_val = np.abs(xy[valid]).max()
        if max_val > 1e-6:
            xy /= max_val
        xy *= present[..., None, None]

    data[..., :2] = xy
    return data


def select_top_persons(
    keypoints: np.ndarray | None,
    scores: np.ndarray | None,
    num_person: int,
) -> np.ndarray:
    """Reduce a frame's detections to a fixed (num_person, V, 3) array.

    Persons are ranked by detection confidence and zero-padded when fewer than
    `num_person` are present.
    """
    out = np.zeros((num_person, 17, 3), dtype=np.float32)
    if keypoints is None or len(keypoints) == 0:
        return out
    if scores is not None and len(scores) == len(keypoints):
        keypoints = keypoints[np.argsort(-np.asarray(scores))]
    n = min(len(keypoints), num_person)
    out[:n] = keypoints[:n]
    return out
