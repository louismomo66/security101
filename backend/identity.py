"""Identity registry — remembering vehicles and persons across time.

The tracker in `threat.py` maintains identity *within* a continuous shot: it
breaks the moment an object leaves frame, is occluded for a second, or the clip
cuts. This module is what survives that. It keeps a persistent record of the
vehicles and persons that were involved in an incident, so that when one of them
appears again — later in the same video, or in footage analysed next week — the
system can say "this is the vehicle from incident #3".

## How matching works

Each sighting produces an **appearance signature**: colour histograms computed
over three horizontal bands of the object's crop, plus its aspect ratio. Two
signatures are compared by histogram intersection, averaged per block. An
entity's stored signature is a running mean over its sightings.

## What this is and is not

This is appearance re-identification, and its limits are structural:

  * It is **colour-dominated**. Two silver saloons of different makes will match
    each other. A single vehicle will *fail* to match itself between daylight
    and sodium-lit night footage.
  * It is **not** a licence plate read, and it is **not** face recognition.
    Nothing here identifies a person by name, and the person signature is
    clothing colour — it survives a walk across a car park, not a change of
    jacket.
  * A match is therefore an **investigative lead for a human to confirm**, and
    every match is reported with its similarity score so that a reviewer can
    weigh it. Never treat `matched=True` as an identification.

Records written here are personal data in most jurisdictions. `forget()` exists
so deletion requests can be honoured, and `prune older_than_days` so retention
limits can be enforced.
"""
from __future__ import annotations

import json
import math
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

# ── Class families ────────────────────────────────────────────────────────

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "train", "boat"}
PERSON_CLASSES = {"person"}

# Vehicle classes that a detector routinely confuses with one another. A match
# across classes inside a group is allowed (at a penalty); across groups it is
# not — a bus is not a motorcycle no matter how similar the colour.
_CLASS_GROUPS = [
    {"car", "truck", "bus"},
    {"motorcycle", "bicycle"},
]


def kind_of(cls: str) -> str | None:
    """Map a detection class to a registry kind, or None if not tracked."""
    c = (cls or "").strip().lower()
    if c in VEHICLE_CLASSES:
        return "vehicle"
    if c in PERSON_CLASSES:
        return "person"
    return None


def _classes_compatible(a: str, b: str) -> float:
    """1.0 for identical classes, 0.85 for a known confusion pair, else 0."""
    a, b = (a or "").lower(), (b or "").lower()
    if a == b:
        return 1.0
    for group in _CLASS_GROUPS:
        if a in group and b in group:
            return 0.85
    return 0.0


# ── Appearance signature ──────────────────────────────────────────────────

# 3 horizontal bands × 3 channel histograms (H=16, S=8, V=8) = 96 values.
_BANDS = 3
_H_BINS, _S_BINS, _V_BINS = 16, 8, 8
_BLOCK = _H_BINS + _S_BINS + _V_BINS          # 32 per band
SIGNATURE_DIM = _BANDS * _BLOCK               # 96

# Pixels darker or less saturated than this carry no reliable hue. Excluding
# them stops every object in night footage from signing as "dark grey".
_MIN_V = 25
_MIN_S = 20


def appearance_signature(frame_rgb: np.ndarray,
                         box: Iterable[float]) -> list[float] | None:
    """Colour signature of the object inside `box`. None if the crop is unusable.

    Bands run top-to-bottom, so for a person they roughly separate head, torso
    and legs, and for a vehicle roof, windows and body. That ordering is what
    makes the signature more than a single average colour.
    """
    if frame_rgb is None or box is None:
        return None
    box = [float(v) for v in box]
    if len(box) != 4:
        return None

    h, w = frame_rgb.shape[:2]
    x1 = max(0, int(box[0])); y1 = max(0, int(box[1]))
    x2 = min(w, int(box[2])); y2 = min(h, int(box[3]))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None

    crop = frame_rgb[y1:y2, x1:x2]
    crop = cv2.resize(crop, (64, 96), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)

    band_h = hsv.shape[0] // _BANDS
    sig: list[float] = []
    for b in range(_BANDS):
        band = hsv[b * band_h:(b + 1) * band_h]
        hue, sat, val = band[..., 0], band[..., 1], band[..., 2]
        chroma = (val >= _MIN_V) & (sat >= _MIN_S)

        # Hue only from pixels that actually have one.
        if chroma.sum() >= 16:
            hh = np.histogram(hue[chroma], bins=_H_BINS, range=(0, 180))[0].astype(float)
        else:
            hh = np.zeros(_H_BINS, dtype=float)
        sh = np.histogram(sat, bins=_S_BINS, range=(0, 256))[0].astype(float)
        vh = np.histogram(val, bins=_V_BINS, range=(0, 256))[0].astype(float)

        for block in (hh, sh, vh):
            total = block.sum()
            sig.extend((block / total).tolist() if total > 0
                       else np.zeros_like(block).tolist())
    return [round(float(v), 6) for v in sig]


# Hue carries almost all the discriminative power: a red car and a blue car have
# near-identical saturation and value histograms, so equal weighting would score
# them ~0.67 similar. Value is weighted lowest because it moves with lighting,
# which is precisely the thing that should not break a match.
_BLOCK_WEIGHTS = {"hue": 2.0, "sat": 1.0, "val": 0.5}


def signature_similarity(a: Iterable[float] | None,
                         b: Iterable[float] | None) -> float:
    """Weighted per-block histogram intersection, 0.0–1.0.

    Blocks are averaged rather than concatenated so a band with more pixel mass
    cannot dominate, and an all-zero hue block (a grey object) drops out
    entirely instead of scoring a false 1.0 against another grey object.
    """
    if a is None or b is None:
        return 0.0
    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if va.shape != vb.shape or va.size != SIGNATURE_DIM:
        return 0.0

    total = 0.0
    weight_sum = 0.0
    offset = 0
    for _ in range(_BANDS):
        for size, channel in ((_H_BINS, "hue"), (_S_BINS, "sat"), (_V_BINS, "val")):
            ba, bb = va[offset:offset + size], vb[offset:offset + size]
            offset += size
            if ba.sum() <= 0 and bb.sum() <= 0:
                continue          # both uninformative — neither agree nor disagree
            w = _BLOCK_WEIGHTS[channel]
            total += w * float(np.minimum(ba, bb).sum())
            weight_sum += w
    return round(total / weight_sum, 4) if weight_sum > 0 else 0.0


# Hue ranges in OpenCV's 0–179 scale, for human-readable descriptions.
_HUE_NAMES = [
    (0, 10, "red"), (10, 22, "orange"), (22, 33, "yellow"),
    (33, 78, "green"), (78, 100, "cyan"), (100, 130, "blue"),
    (130, 160, "purple"), (160, 180, "red"),
]


def describe_colour(frame_rgb: np.ndarray, box: Iterable[float]) -> str:
    """A short human-readable colour, e.g. "dark blue" or "silver/grey".

    Reports are read by people, and "vehicle #a41f" means nothing to a reviewer
    scanning a timeline while "white van" does.
    """
    if frame_rgb is None or box is None:
        return "unknown"
    h, w = frame_rgb.shape[:2]
    x1 = max(0, int(box[0])); y1 = max(0, int(box[1]))
    x2 = min(w, int(box[2])); y2 = min(h, int(box[3]))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return "unknown"

    hsv = cv2.cvtColor(frame_rgb[y1:y2, x1:x2], cv2.COLOR_RGB2HSV)
    # Sample the middle half of the crop — edges are mostly background.
    ih, iw = hsv.shape[:2]
    hsv = hsv[ih // 4: ih * 3 // 4, iw // 4: iw * 3 // 4]
    if hsv.size == 0:
        return "unknown"

    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mean_v = float(val.mean())
    chroma = (sat >= 60) & (val >= 40)

    if chroma.mean() < 0.18:
        # Achromatic: describe by lightness instead of hue.
        if mean_v < 60:
            return "black/dark"
        if mean_v < 120:
            return "dark grey"
        if mean_v < 190:
            return "silver/grey"
        return "white"

    dominant = int(np.bincount(hue[chroma].astype(int), minlength=180).argmax())
    name = next((n for lo, hi, n in _HUE_NAMES if lo <= dominant < hi), "unknown")
    if mean_v < 80:
        return f"dark {name}"
    if mean_v > 200 and float(sat[chroma].mean()) < 120:
        return f"light {name}"
    return name


# ── Entities ──────────────────────────────────────────────────────────────

@dataclass
class Entity:
    """A vehicle or person the system has committed to memory."""

    entity_id: str
    kind: str                       # "vehicle" | "person"
    cls: str                        # detector class at first sighting
    label: str                      # human-readable, e.g. "white car"
    signature: list[float]
    aspect: float                   # width / height
    samples: int = 1
    first_seen: str = ""
    last_seen: str = ""
    sightings: list[dict] = field(default_factory=list)
    incidents: list[dict] = field(default_factory=list)
    plate: str | None = None        # reserved: populated by a plate reader
    plate_confidence: float | None = None
    notes: list[str] = field(default_factory=list)

    MAX_SIGHTINGS = 200

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("MAX_SIGHTINGS", None)
        d["incident_count"] = len(self.incidents)
        d["sighting_count"] = len(self.sightings)
        return d

    def summary(self) -> dict:
        """Compact form for report and list views — omits the raw signature."""
        return {
            "entity_id": self.entity_id,
            "kind": self.kind,
            "cls": self.cls,
            "label": self.label,
            "plate": self.plate,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "sighting_count": len(self.sightings),
            "incident_count": len(self.incidents),
            "incidents": self.incidents[-10:],
            "sightings": [
                {k: s.get(k) for k in
                 ("source", "video_time_s", "timecode", "frame", "snapshot")}
                for s in self.sightings[-20:]
            ],
            "notes": self.notes,
        }

    def merge_signature(self, sig: list[float], weight_cap: int = 20):
        """Fold a new observation into the running mean.

        Capping the effective sample count keeps the signature responsive to
        genuine appearance change (a vehicle driving into shade) instead of
        freezing on the first sighting.
        """
        n = min(self.samples, weight_cap)
        cur = np.asarray(self.signature, dtype=float)
        new = np.asarray(sig, dtype=float)
        if cur.shape != new.shape:
            self.signature = [round(float(v), 6) for v in new]
        else:
            blended = (cur * n + new) / (n + 1)
            self.signature = [round(float(v), 6) for v in blended]
        self.samples += 1


# ── Registry ──────────────────────────────────────────────────────────────

class IdentityRegistry:
    """Persistent store of remembered vehicles and persons.

    Thread-safe: the analysis worker and the API read and write concurrently.
    Persistence is a single JSON file, rewritten on change — adequate for the
    hundreds-of-entities scale this operates at, and trivially inspectable.
    """

    def __init__(self, path: str | Path | None = None,
                 match_threshold: float = 0.62,
                 autosave: bool = True):
        self.path = Path(path) if path else None
        self.match_threshold = match_threshold
        self.autosave = autosave
        self._entities: dict[str, Entity] = {}
        self._lock = threading.RLock()
        if self.path and self.path.exists():
            self.load()

    # ── persistence ───────────────────────────────────────────────────────

    def load(self) -> int:
        if not self.path or not self.path.exists():
            return 0
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return 0
        with self._lock:
            self._entities.clear()
            for item in raw.get("entities", []):
                item.pop("incident_count", None)
                item.pop("sighting_count", None)
                try:
                    self._entities[item["entity_id"]] = Entity(**item)
                except (TypeError, KeyError):
                    continue
            return len(self._entities)

    def save(self) -> bool:
        if not self.path:
            return False
        with self._lock:
            payload = {
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "match_threshold": self.match_threshold,
                "entities": [e.to_dict() for e in self._entities.values()],
            }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(self.path)          # atomic — a crash mid-write keeps the old file
            return True
        except Exception:
            return False

    def _touch(self):
        if self.autosave:
            self.save()

    # ── lookup ────────────────────────────────────────────────────────────

    def get(self, entity_id: str) -> Entity | None:
        with self._lock:
            return self._entities.get(entity_id)

    def all(self, kind: str | None = None,
            with_incidents_only: bool = False) -> list[Entity]:
        with self._lock:
            items = list(self._entities.values())
        if kind:
            items = [e for e in items if e.kind == kind]
        if with_incidents_only:
            items = [e for e in items if e.incidents]
        return sorted(items, key=lambda e: e.last_seen or "", reverse=True)

    def best_match(self, kind: str, cls: str, signature: list[float],
                   aspect: float) -> tuple[Entity | None, float]:
        """Closest stored entity and its similarity score.

        Score combines appearance (dominant), class agreement and aspect ratio.
        Aspect matters for vehicles: a bus and a hatchback in the same white
        share a colour histogram but not a silhouette.
        """
        best: Entity | None = None
        best_score = 0.0
        with self._lock:
            candidates = [e for e in self._entities.values() if e.kind == kind]

        for ent in candidates:
            class_factor = _classes_compatible(ent.cls, cls)
            if class_factor <= 0:
                continue
            appearance = signature_similarity(ent.signature, signature)
            if appearance <= 0:
                continue
            if ent.aspect > 0 and aspect > 0:
                ratio = min(ent.aspect, aspect) / max(ent.aspect, aspect)
            else:
                ratio = 1.0
            # Appearance carries the decision; shape and class only modulate it.
            score = appearance * (0.80 + 0.20 * ratio) * class_factor
            if score > best_score:
                best, best_score = ent, score
        return best, round(best_score, 4)

    # ── writes ────────────────────────────────────────────────────────────

    def observe(self, *, kind: str, cls: str, signature: list[float],
                aspect: float, label: str = "", sighting: dict | None = None,
                threshold: float | None = None) -> tuple[Entity, float, bool]:
        """Match a sighting to a known entity, or enrol a new one.

        Returns ``(entity, similarity, is_new)``. `similarity` is 1.0 for a new
        entity by convention; callers that care should check `is_new`.
        """
        threshold = self.match_threshold if threshold is None else threshold
        now = datetime.now(timezone.utc).isoformat()

        match, score = self.best_match(kind, cls, signature, aspect)
        with self._lock:
            if match is not None and score >= threshold:
                match.merge_signature(signature)
                match.last_seen = now
                if sighting:
                    match.sightings.append({**sighting, "similarity": score})
                    if len(match.sightings) > Entity.MAX_SIGHTINGS:
                        del match.sightings[:len(match.sightings) - Entity.MAX_SIGHTINGS]
                self._touch()
                return match, score, False

            ent = Entity(
                entity_id=f"{kind[:3]}_{uuid.uuid4().hex[:8]}",
                kind=kind, cls=cls,
                label=label or f"{cls}",
                signature=list(signature), aspect=float(aspect),
                first_seen=now, last_seen=now,
                sightings=[sighting] if sighting else [],
            )
            self._entities[ent.entity_id] = ent
            self._touch()
            return ent, 1.0, True

    def link_incident(self, entity_id: str, event: dict,
                      role: str = "involved") -> bool:
        """Attach an incident to an entity — this is the "remember" step.

        Keeps only the fields a reviewer needs, so the registry does not become
        a second copy of the event log.
        """
        with self._lock:
            ent = self._entities.get(entity_id)
            if ent is None:
                return False
            ent.incidents.append({
                "event_id": event.get("id"),
                "type": event.get("type"),
                "label": event.get("label"),
                "severity": event.get("severity"),
                "score": event.get("score"),
                "role": role,
                "source": event.get("source"),
                "video_time_s": event.get("video_time_s"),
                "timecode": event.get("timecode"),
                "timestamp": event.get("timestamp"),
                "snapshot": event.get("snapshot"),
            })
            ent.last_seen = datetime.now(timezone.utc).isoformat()
            self._touch()
            return True

    def add_note(self, entity_id: str, note: str) -> bool:
        with self._lock:
            ent = self._entities.get(entity_id)
            if ent is None:
                return False
            ent.notes.append(note)
            self._touch()
            return True

    def set_plate(self, entity_id: str, plate: str,
                  confidence: float | None = None) -> bool:
        """Record a plate read. No reader ships by default — see module docstring."""
        with self._lock:
            ent = self._entities.get(entity_id)
            if ent is None:
                return False
            ent.plate = plate
            ent.plate_confidence = confidence
            self._touch()
            return True

    def forget(self, entity_id: str) -> bool:
        """Delete an entity outright, for erasure requests."""
        with self._lock:
            existed = self._entities.pop(entity_id, None) is not None
            if existed:
                self._touch()
            return existed

    def prune(self, older_than_days: float, keep_with_incidents: bool = True) -> int:
        """Drop entities not seen recently. Enforces a retention policy."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
        removed = 0
        with self._lock:
            for eid, ent in list(self._entities.items()):
                if keep_with_incidents and ent.incidents:
                    continue
                try:
                    last = datetime.fromisoformat(ent.last_seen)
                except (ValueError, TypeError):
                    continue
                if last.tzinfo is None:
                    last = last.replace(tzinfo=timezone.utc)
                if last < cutoff:
                    del self._entities[eid]
                    removed += 1
            if removed:
                self._touch()
        return removed

    def clear(self):
        with self._lock:
            self._entities.clear()
            self._touch()

    def stats(self) -> dict:
        with self._lock:
            items = list(self._entities.values())
        return {
            "total": len(items),
            "vehicles": sum(1 for e in items if e.kind == "vehicle"),
            "persons": sum(1 for e in items if e.kind == "person"),
            "with_incidents": sum(1 for e in items if e.incidents),
            "match_threshold": self.match_threshold,
            "path": str(self.path) if self.path else None,
        }


def aspect_of(box: Iterable[float]) -> float:
    """Width / height of a box, guarded against degenerate values."""
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except (TypeError, ValueError):
        return 0.0
    h = y2 - y1
    return round((x2 - x1) / h, 4) if h > 1 else 0.0


def format_timecode(seconds: float | None) -> str:
    """Seconds → HH:MM:SS.mmm, the form a reviewer scrubs to."""
    if seconds is None or (isinstance(seconds, float) and math.isnan(seconds)):
        return "--:--:--"
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"
