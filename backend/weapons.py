"""Crime-tool recognition — what a person is carrying, and whether they hold it.

Three layers, in increasing cost and decreasing determinism:

  1. **Class vocabulary** over the stock COCO detector. COCO contains no firearm
     class, so this layer covers knives, bats, scissors and a handful of
     improvised implements only. See CRIME_TOOL_CLASSES.

  2. **A pluggable weapon detector.** Point ``SENTINEL_WEAPON_MODEL`` at a
     YOLO ONNX export trained on weapons and its classes join the same
     vocabulary. Nothing else in the pipeline changes. This is the only way to
     get real firearm coverage — see `load_weapon_model`.

  3. **Grip geometry.** An object *near* a person is weak evidence; an object at
     the end of someone's arm is strong evidence. `held_objects` anchors object
     boxes to wrist keypoints from the pose model, which the rest of the stack
     already computes every frame.

  4. **VLM verification.** For the ambiguous cases, `verify_held_object` crops
     the person, upscales, and asks the VLM a closed question about what is in
     their hands. Slow (hundreds of ms), so callers should reserve it for
     candidates that already passed layers 1–3.

Nothing here decides that a crime occurred. It decides which frames a human
should look at, and in what order.
"""
from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass
from typing import Any, Iterable

import cv2
import numpy as np

# ── Vocabulary ────────────────────────────────────────────────────────────

# Objects whose presence in a hand is itself the signal. `weight` is the prior
# on "this really is a weapon" — it multiplies detector confidence, so a class
# YOLO gets wrong a lot (scissors) cannot on its own reach a high score.
#
# `category`:
#   weapon      — designed to injure; brandishing is the incident
#   improvised  — ordinary object used as a weapon; needs corroboration
#   burglary    — entry/defeat tools; interesting near doors, windows, vehicles
#   concealment — face/identity concealment, an intent signal in context
WEAPON_CLASSES: dict[str, float] = {
    "knife": 0.85,
    "baseball bat": 0.55,
    "scissors": 0.45,
}

CRIME_TOOL_CLASSES: dict[str, dict[str, Any]] = {
    # ── from stock COCO ──
    "knife":         {"weight": 0.85, "category": "weapon",     "label": "Knife"},
    "baseball bat":  {"weight": 0.55, "category": "weapon",     "label": "Bat or club"},
    "scissors":      {"weight": 0.45, "category": "weapon",     "label": "Scissors"},
    "bottle":        {"weight": 0.20, "category": "improvised", "label": "Bottle"},
    # ── typical classes from weapon-trained YOLO models; harmless if the
    #    loaded model never emits them ──
    "gun":           {"weight": 0.95, "category": "weapon", "label": "Firearm"},
    "pistol":        {"weight": 0.95, "category": "weapon", "label": "Handgun"},
    "handgun":       {"weight": 0.95, "category": "weapon", "label": "Handgun"},
    "revolver":      {"weight": 0.95, "category": "weapon", "label": "Handgun"},
    "rifle":         {"weight": 0.95, "category": "weapon", "label": "Rifle"},
    "shotgun":       {"weight": 0.95, "category": "weapon", "label": "Shotgun"},
    "firearm":       {"weight": 0.95, "category": "weapon", "label": "Firearm"},
    "machete":       {"weight": 0.90, "category": "weapon", "label": "Machete"},
    "axe":           {"weight": 0.80, "category": "weapon", "label": "Axe"},
    "hammer":        {"weight": 0.50, "category": "improvised", "label": "Hammer"},
    "crowbar":       {"weight": 0.70, "category": "burglary",   "label": "Crowbar"},
    "bolt cutter":   {"weight": 0.75, "category": "burglary",   "label": "Bolt cutters"},
    "angle grinder": {"weight": 0.70, "category": "burglary",   "label": "Angle grinder"},
    "balaclava":     {"weight": 0.60, "category": "concealment", "label": "Face covering"},
    "mask":          {"weight": 0.25, "category": "concealment", "label": "Face covering"},
}

# Severity floor per category when the item is confirmed in a hand.
CATEGORY_SEVERITY = {
    "weapon": "critical",
    "improvised": "high",
    "burglary": "high",
    "concealment": "medium",
}

# COCO keypoint indices from detectors/yolo_pose_detector.py.
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10
KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW = 8
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6


def tool_info(cls: str) -> dict[str, Any] | None:
    """Vocabulary lookup, case- and separator-insensitive."""
    if not cls:
        return None
    key = cls.strip().lower().replace("_", " ").replace("-", " ")
    return CRIME_TOOL_CLASSES.get(key)


# ── Pluggable weapon detector ─────────────────────────────────────────────

@dataclass
class WeaponModel:
    """A second YOLO head run alongside the COCO detector.

    The stock detector cannot see firearms — COCO has no such class — so the
    only route to real weapon coverage is a purpose-trained model. Any
    ultralytics-style ONNX export works, because `YOLODetector.detect` derives
    the class count from the output tensor rather than assuming 80.
    """

    detector: Any
    class_names: list[str]
    path: str
    # 0.45 matches threat.py's weapon_conf default and the threshold
    # `training.weapons.evaluate` measured as clean (0 false alarms / 100
    # held-out frames) for weapons_v2.onnx on 2026-08-21. Lower thresholds
    # were not clean, do not drop this without rerunning that evaluation.
    conf: float = 0.45

    def detect(self, frame_rgb: np.ndarray, conf: float | None = None,
               iou: float = 0.45) -> list[dict]:
        boxes, scores, class_ids = self.detector.detect(
            frame_rgb, conf=conf if conf is not None else self.conf, iou=iou
        )
        out = []
        for b, s, c in zip(boxes, scores, class_ids):
            idx = int(c)
            name = self.class_names[idx] if idx < len(self.class_names) else f"class_{idx}"
            out.append({
                "class": name.strip().lower(),
                "confidence": round(float(s), 3),
                "box": [int(v) for v in b],
                "source": "weapon_model",
            })
        return out


class SmoothedWeaponDetector:
    """Wraps a WeaponModel with two independent recall techniques.

    Both exist because a straight confidence-threshold sweep already showed
    the limit of that one knob: 0.45 is the lowest single-frame threshold
    that stayed at 0 false alarms / 100 held-out frames (see dev.sh), and a
    real hit on the Florida-store gun clip scored 0.463-0.514 there — right
    at that edge, and inconsistent frame to frame as a result. Lowering the
    threshold catches more of that, but also reopens the false-alarm gate;
    neither technique here does that, because neither one lowers the bar for
    a *single* frame.

    1. Person-cropped inference. A gun a few hundred pixels long inside a
       1920x1080 frame is a small fraction of what the model actually sees
       once letterboxed to its input size. Also running detection on a
       padded, upscaled crop around each known person box gives the model
       more pixels on the actual object, which usually moves a borderline
       score meaningfully rather than marginally.

    2. Temporal persistence. An obvious single-frame hit (>= `confirm_conf`)
       fires immediately, unchanged from before. A weaker signal
       (>= `sustain_conf`) that holds for `sustain_frames` consecutive
       processed frames also fires — a real gun held at someone's side stays
       roughly the same shape and score for several frames in a row; a
       stray high-confidence blip on a shrub or a handlebar generally does
       not repeat.

    Caveat carried over from every threshold in this file: this has NOT been
    re-run through `training.weapons.evaluate`, which only measures raw
    single-frame confidence, not this wrapper's behaviour. Re-measure the
    false-alarm rate of the *combined* system before trusting these
    defaults the way weapon_conf=0.45 alone was trusted.
    """

    def __init__(self, weapon_model: WeaponModel,
                sustain_conf: float | None = None,
                sustain_frames: int | None = None,
                confirm_conf: float | None = None,
                crop_pad: float | None = None):
        self.wm = weapon_model
        self.sustain_conf = sustain_conf if sustain_conf is not None else float(
            os.environ.get("SENTINEL_WEAPON_SUSTAIN_CONF", "0.30"))
        self.sustain_frames = sustain_frames if sustain_frames is not None else int(
            os.environ.get("SENTINEL_WEAPON_SUSTAIN_FRAMES", "3"))
        self.confirm_conf = confirm_conf if confirm_conf is not None else weapon_model.conf
        self.crop_pad = crop_pad if crop_pad is not None else float(
            os.environ.get("SENTINEL_WEAPON_CROP_PAD", "0.25"))
        self._history: dict[str, "deque[dict | None]"] = {}

    def detect(self, frame_rgb: np.ndarray, person_boxes: Iterable[Iterable[float]] | None = None,
               iou: float = 0.45) -> list[dict]:
        raw = self._raw_detect(frame_rgb, person_boxes, iou)
        return self._apply_persistence(raw)

    # -- person-cropped inference --------------------------------------

    def _raw_detect(self, frame_rgb: np.ndarray, person_boxes, iou: float) -> list[dict]:
        dets = list(self.wm.detect(frame_rgb, conf=self.sustain_conf, iou=iou))
        h, w = frame_rgb.shape[:2]
        for pbox in (person_boxes or []):
            if not pbox or len(pbox) != 4:
                continue
            x1, y1, x2, y2 = pbox
            bw, bh = x2 - x1, y2 - y1
            if bw <= 1 or bh <= 1:
                continue
            px1 = max(0, int(x1 - bw * self.crop_pad))
            py1 = max(0, int(y1 - bh * self.crop_pad))
            px2 = min(w, int(x2 + bw * self.crop_pad))
            py2 = min(h, int(y2 + bh * self.crop_pad))
            if px2 - px1 < 10 or py2 - py1 < 10:
                continue
            crop = frame_rgb[py1:py2, px1:px2]
            for d in self.wm.detect(crop, conf=self.sustain_conf, iou=iou):
                bx1, by1, bx2, by2 = d["box"]
                d = dict(d)
                d["box"] = [bx1 + px1, by1 + py1, bx2 + px1, by2 + py1]
                d["source"] = "weapon_model_crop"
                dets.append(d)
        return self._dedup(dets)

    @staticmethod
    def _iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        aarea = max(1.0, (ax2 - ax1) * (ay2 - ay1))
        barea = max(1.0, (bx2 - bx1) * (by2 - by1))
        return inter / (aarea + barea - inter)

    def _dedup(self, dets: list[dict], iou_thresh: float = 0.5) -> list[dict]:
        """The same gun can be found by both the full-frame pass and a
        person-crop pass; keep the higher-confidence box, not both."""
        dets = sorted(dets, key=lambda d: -d["confidence"])
        kept: list[dict] = []
        for d in dets:
            if any(d["class"] == k["class"] and self._iou(d["box"], k["box"]) > iou_thresh
                  for k in kept):
                continue
            kept.append(d)
        return kept

    # -- temporal persistence --------------------------------------------

    def _apply_persistence(self, raw: list[dict]) -> list[dict]:
        import collections

        best_by_class: dict[str, dict] = {}
        for d in raw:
            cur = best_by_class.get(d["class"])
            if cur is None or d["confidence"] > cur["confidence"]:
                best_by_class[d["class"]] = d

        out: list[dict] = []
        for cls in set(best_by_class) | set(self._history):
            hist = self._history.setdefault(cls, collections.deque(maxlen=self.sustain_frames))
            best = best_by_class.get(cls)
            hist.append(best if (best and best["confidence"] >= self.sustain_conf) else None)

            if best and best["confidence"] >= self.confirm_conf:
                out.append(best)  # obvious single-frame hit — unchanged from before
                continue

            if len(hist) == self.sustain_frames and all(h is not None for h in hist):
                peak = max(hist, key=lambda h: h["confidence"])
                d = dict(peak)
                d["source"] = d.get("source", "weapon_model") + "_sustained"
                out.append(d)
        return out


_weapon_model: WeaponModel | None = None
_weapon_model_tried = False
_weapon_lock = threading.Lock()


def load_weapon_model(path: str | None = None,
                      class_names: list[str] | None = None) -> WeaponModel | None:
    """Load the optional weapon detector. Returns None when unconfigured.

    Configuration, in precedence order:

      * explicit `path` / `class_names` arguments
      * ``SENTINEL_WEAPON_MODEL``   — path to a .onnx export
      * ``SENTINEL_WEAPON_CLASSES`` — either a JSON array of names, or a path to a
        file containing one class name per line, in the model's own class order

    Without class names the model's outputs cannot be mapped to the vocabulary,
    so loading is refused rather than silently producing `class_0` events.
    """
    global _weapon_model, _weapon_model_tried

    with _weapon_lock:
        if _weapon_model is not None:
            return _weapon_model
        if _weapon_model_tried and path is None:
            return None
        _weapon_model_tried = True

        model_path = path or os.environ.get("SENTINEL_WEAPON_MODEL", "").strip()
        if not model_path or not os.path.exists(model_path):
            return None

        names = class_names
        if names is None:
            raw = os.environ.get("SENTINEL_WEAPON_CLASSES", "").strip()
            if raw.startswith("["):
                try:
                    names = list(json.loads(raw))
                except Exception:
                    names = None
            elif raw and os.path.exists(raw):
                names = [ln.strip() for ln in open(raw, encoding="utf-8") if ln.strip()]
        if not names:
            import logging
            logging.warning(
                "SENTINEL_WEAPON_MODEL is set but class names are not; refusing to "
                "load %s. Set SENTINEL_WEAPON_CLASSES to a JSON array or a file of "
                "names in the model's class order.", model_path,
            )
            return None

        try:
            from detectors import YOLODetector

            # Auto-detect the export's input size and whether it is a
            # segmentation model (box + class + 32 mask-coefficient columns)
            # vs a plain detection model (box + class only). Defaulting to
            # img_size=640/nc=None here silently breaks any model exported
            # at a different imgsz or as instance-segmentation: the ONNX
            # session throws a shape-mismatch error on every single
            # inference call, which looks from the outside like "the
            # detector loaded but never detects anything".
            nc = len(names)
            img_size = 640
            try:
                import onnxruntime as ort
                # CPU on purpose: this session only reads tensor shapes and
                # never runs inference, and compiling the graph for CoreML
                # to throw it away costs seconds of startup.
                probe = ort.InferenceSession(
                    model_path, providers=["CPUExecutionProvider"])
                out_shape = probe.get_outputs()[0].shape
                n_out = out_shape[1] if isinstance(out_shape[1], int) else None
                in_shape = probe.get_inputs()[0].shape  # [N, C, H, W]
                if isinstance(in_shape[2], int):
                    img_size = in_shape[2]
                is_seg = n_out is not None and n_out - 4 - 32 == nc
            except Exception:
                is_seg = False

            _weapon_model = WeaponModel(
                detector=YOLODetector(model_path, img_size=img_size,
                                      nc=nc if is_seg else None),
                class_names=[str(n) for n in names],
                path=model_path,
            )
            import logging
            logging.info("Weapon model loaded: %s (%d classes)", model_path, len(names))
            return _weapon_model
        except Exception as exc:
            import logging
            logging.warning("Failed to load weapon model %s: %s", model_path, exc)
            return None


def weapon_model_status() -> dict:
    m = _weapon_model
    return {
        "loaded": m is not None,
        "path": m.path if m else os.environ.get("SENTINEL_WEAPON_MODEL", "") or None,
        "classes": m.class_names if m else [],
        "note": (
            "Firearms are undetectable without this model — COCO has no gun class."
            if m is None else None
        ),
    }


# ── Grip geometry ─────────────────────────────────────────────────────────

def _box_distance(point: tuple[float, float], box: Iterable[float]) -> float:
    """Euclidean distance from a point to a box; 0 when the point is inside."""
    x, y = point
    x1, y1, x2, y2 = box
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return float(np.hypot(dx, dy))


def _person_scale(box: Iterable[float]) -> float:
    """A length that scales with the person's apparent size.

    Shoulder-to-wrist would be ideal but is unreliable when an arm is occluded,
    so this uses the box diagonal, which degrades gracefully.
    """
    x1, y1, x2, y2 = box
    return float(np.hypot(max(1.0, x2 - x1), max(1.0, y2 - y1)))


def hand_points(keypoints: Iterable[Iterable[float]] | None,
                kp_conf: float = 0.3) -> list[tuple[str, tuple[float, float]]]:
    """Confident wrist positions as (hand, (x, y)).

    Falls back to the elbow when a wrist is occluded but its elbow is visible:
    an object at the elbow's reach is still plausibly held, and dropping the
    person entirely would miss the exact frames that matter (an arm extended
    toward another person is often the frame where the wrist blurs out).
    """
    if keypoints is None:
        return []
    kp = np.asarray(keypoints, dtype=float)
    if kp.ndim != 2 or kp.shape[0] < 17:
        return []

    out: list[tuple[str, tuple[float, float]]] = []
    for hand, wrist_i, elbow_i in (
        ("left", KP_LEFT_WRIST, KP_LEFT_ELBOW),
        ("right", KP_RIGHT_WRIST, KP_RIGHT_ELBOW),
    ):
        conf_w = kp[wrist_i][2] if kp.shape[1] > 2 else 1.0
        if conf_w >= kp_conf:
            out.append((hand, (float(kp[wrist_i][0]), float(kp[wrist_i][1]))))
            continue
        conf_e = kp[elbow_i][2] if kp.shape[1] > 2 else 0.0
        if conf_e >= kp_conf:
            out.append((f"{hand} (elbow)", (float(kp[elbow_i][0]), float(kp[elbow_i][1]))))
    return out


def held_objects(objects: list[dict],
                 pose_persons: list[dict] | None,
                 grip_tolerance: float = 0.18,
                 kp_conf: float = 0.3) -> list[dict]:
    """Match detected objects to the hand holding them.

    `grip_tolerance` is expressed as a fraction of the *person's* box diagonal,
    not the frame's, so the same threshold works for someone close to the
    camera and someone at the far end of a car park.

    Returns one entry per (object, person) match:

        {"object": {...}, "person": {...}, "person_index": int,
         "hand": "right", "distance_px": float, "grip": 0.0–1.0}

    `grip` is a confidence in the *holding*, separate from the detector's
    confidence in the *object*. 1.0 means the wrist is inside the object box.
    """
    if not objects or not pose_persons:
        return []

    matches: list[dict] = []
    for obj in objects:
        box = obj.get("box")
        if not box or len(box) != 4:
            continue
        best: dict | None = None
        for idx, person in enumerate(pose_persons):
            pbox = person.get("box")
            if not pbox or len(pbox) != 4:
                continue
            scale = _person_scale(pbox)
            limit = grip_tolerance * scale
            for hand, pt in hand_points(person.get("keypoints"), kp_conf=kp_conf):
                dist = _box_distance(pt, box)
                if dist > limit:
                    continue
                grip = 1.0 - (dist / limit if limit > 0 else 0.0)
                if best is None or grip > best["grip"]:
                    best = {
                        "object": obj,
                        "person": person,
                        "person_index": idx,
                        "hand": hand,
                        "distance_px": round(dist, 1),
                        "grip": round(float(grip), 3),
                    }
        if best is not None:
            matches.append(best)
    return matches


def carried_objects(objects: list[dict], persons: list[dict],
                    overlap: float = 0.55) -> list[dict]:
    """Objects mostly inside a person's box but not matched to a hand.

    Weaker than a grip match — it covers items tucked under an arm, in a
    waistband, or held below frame — so callers should score it lower.
    """
    out = []
    for obj in objects:
        ob = obj.get("box")
        if not ob or len(ob) != 4:
            continue
        ox1, oy1, ox2, oy2 = ob
        oarea = max(1.0, (ox2 - ox1) * (oy2 - oy1))
        for idx, p in enumerate(persons):
            pb = p.get("box")
            if not pb or len(pb) != 4:
                continue
            px1, py1, px2, py2 = pb
            ix1, iy1 = max(ox1, px1), max(oy1, py1)
            ix2, iy2 = min(ox2, px2), min(oy2, py2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            frac = inter / oarea
            if frac >= overlap:
                out.append({"object": obj, "person": p, "person_index": idx,
                            "containment": round(float(frac), 3)})
                break
    return out


# ── VLM verification ──────────────────────────────────────────────────────

VERIFY_PROMPT = (
    "Look at this person's hands. Are they holding a weapon or a tool "
    "(for example a gun, knife, machete, club, crowbar or hammer)? "
    "Answer in this exact form: 'YES: <item>' or 'NO'. "
    "If you cannot see their hands clearly, answer 'UNCLEAR'."
)

_VERIFY_YES = re.compile(r"^\s*yes\b[:\-\s]*(.*)", re.IGNORECASE)
_VERIFY_NO = re.compile(r"^\s*no\b", re.IGNORECASE)
_VERIFY_UNCLEAR = re.compile(r"\bunclear\b|\bcannot see\b|\bcan't see\b|\bnot visible\b",
                             re.IGNORECASE)

# Items the VLM may name that count as a weapon confirmation. Deliberately
# narrower than the caption lexicon: this question is closed, so a vague answer
# ("something", "an object") must not count as a confirmation.
_VERIFY_ITEMS = re.compile(
    r"\b(gun|firearm|pistol|handgun|revolver|rifle|shotgun|weapon|knife|blade|"
    r"machete|panga|axe|hatchet|club|bat|baton|crowbar|hammer|screwdriver|"
    r"bolt ?cutter|grinder|stick)\b",
    re.IGNORECASE,
)

# Small VLMs frequently answer a closed question by restating it — "To determine
# whether the person is holding a weapon, we need to examine…". That echo
# contains the question's own vocabulary, so a naive item search scores it as a
# confirmation. Observed on FastVLM-0.5B: every sampled frame of a night clip
# came back "yes" on the strength of the word "weapon" in the restatement.
_VERIFY_META = re.compile(
    r"\b(to determine|to answer|in order to|we (need|would need|must|can)|"
    r"let'?s|first,|the (question|task)|it (is )?(not )?possible to tell|"
    r"i (cannot|can't|am unable)|based on the (image|description))\b",
    re.IGNORECASE,
)

# An answer to a closed question is short. Anything longer is commentary, and
# commentary is not a confirmation.
_VERIFY_MAX_WORDS = 12


def crop_person(frame: np.ndarray, box: Iterable[float], pad: float = 0.15,
                min_size: int = 224) -> np.ndarray | None:
    """Crop a person with padding, upscaled if small.

    A 40-pixel-tall person carries no readable detail; upscaling does not add
    information but it does stop the VLM's preprocessing from discarding what
    little there is.
    """
    if frame is None or box is None or len(box) != 4:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    if bw <= 1 or bh <= 1:
        return None
    x1 = max(0, int(x1 - bw * pad))
    y1 = max(0, int(y1 - bh * pad))
    x2 = min(w, int(x2 + bw * pad))
    y2 = min(h, int(y2 + bh * pad))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None

    crop = frame[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    if max(ch, cw) < min_size:
        scale = min_size / max(ch, cw)
        crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)),
                          interpolation=cv2.INTER_CUBIC)
    return crop


def verify_held_object(frame: np.ndarray, person_box: Iterable[float],
                       vlm_fn: Any = None, max_tokens: int = 24) -> dict:
    """Ask the VLM a closed question about what a person is holding.

    Returns ``{"verdict": "yes"|"no"|"unclear"|"unavailable", "item": str|None,
    "text": str}``. `verdict == "yes"` still only means the VLM said so — a
    0.5B model is confidently wrong often enough that this should adjust a
    score, never create an incident on its own.
    """
    crop = crop_person(frame, person_box)
    if crop is None:
        return {"verdict": "unavailable", "item": None, "text": "", "reason": "bad crop"}

    if vlm_fn is None:
        try:
            from backend.models import run_vlm as vlm_fn  # type: ignore
        except Exception as exc:
            return {"verdict": "unavailable", "item": None, "text": "",
                    "reason": f"VLM unavailable: {exc}"}

    try:
        result = vlm_fn(crop, prompt=VERIFY_PROMPT, max_tokens=max_tokens)
    except Exception as exc:
        return {"verdict": "unavailable", "item": None, "text": "",
                "reason": f"VLM error: {exc}"}

    text = (result or {}).get("text", "").strip()
    if not text:
        return {"verdict": "unavailable", "item": None, "text": "",
                "reason": "empty response"}

    if _VERIFY_UNCLEAR.search(text):
        return {"verdict": "unclear", "item": None, "text": text}

    m = _VERIFY_YES.match(text)
    if m:
        item_m = _VERIFY_ITEMS.search(m.group(1) or text)
        # "YES" with no nameable item is not a confirmation — small VLMs
        # agree with leading questions readily.
        if item_m:
            return {"verdict": "yes", "item": item_m.group(0).lower(), "text": text}
        return {"verdict": "unclear", "item": None, "text": text}

    if _VERIFY_NO.match(text):
        return {"verdict": "no", "item": None, "text": text}

    # Neither format was followed. Before falling back to a keyword search,
    # reject restatements of the question and anything long enough to be
    # commentary — both contain the question's vocabulary without answering it.
    if _VERIFY_META.search(text) or len(text.split()) > _VERIFY_MAX_WORDS:
        return {"verdict": "unclear", "item": None, "text": text,
                "reason": "model did not answer the question"}

    item_m = _VERIFY_ITEMS.search(text)
    if item_m:
        return {"verdict": "yes", "item": item_m.group(0).lower(), "text": text}
    return {"verdict": "unclear", "item": None, "text": text}
