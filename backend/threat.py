"""Sentinel threat engine — turns raw perception signals into scored incidents.

The perception stack (YOLO detection, YOLO-pose + ST-GCN action recognition,
FastVLM captioning) tells you *what is in the frame*. This module decides
*whether what is in the frame is worth a human looking at*.

Design: two tiers.

  Tier 1 — deterministic rules over structured signals (object classes, pose
  actions, spatial relationships, dwell time, zones). Cheap, explainable,
  runs every frame, low false-negative rate on the things it covers.

  Tier 2 — open-vocabulary screening of the VLM caption. Catches categories no
  rule was written for, at the cost of precision. This is what gives coverage
  of "any crime" rather than a fixed list.

Nothing here determines that a crime occurred. It ranks frames for review.
See CALIBRATION notes at the bottom of this file before deploying.

Two capabilities sit on top of that base:

  * **Grip reasoning** (`backend.weapons`) — an object at the end of someone's
    arm is a materially different signal from an object merely near them, and
    the pose model already gives us wrist positions to test it with.

  * **Vehicle dynamics** — tracks carry velocity, so collisions can be detected
    from contact plus abrupt deceleration, and a vehicle that leaves straight
    after one can be flagged. Events name their subject object so the caller can
    commit it to the identity registry (`backend.identity`).
"""
from __future__ import annotations

import math
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np

from backend.weapons import (
    CATEGORY_SEVERITY,
    CRIME_TOOL_CLASSES,
    WEAPON_CLASSES,
    carried_objects,
    held_objects,
    tool_info,
)

# ── Severity ──────────────────────────────────────────────────────────────

SEVERITY_ORDER = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


def severity_at_least(a: str, b: str) -> bool:
    return SEVERITY_ORDER.get(a, 0) >= SEVERITY_ORDER.get(b, 0)


def action_candidates(action: dict | None) -> list[tuple[str, float]]:
    """Extract every (label, confidence) pair from an action payload.

    Two things this fixes:

    1. **Shape.** ActionRecognizer returns ``{"actions": [{"label", "confidence"}...]}``
       but the rules used to read ``action["action"]``, which is never present.
       The result was that action rules silently never fired, no matter how
       confident the model was. Both shapes are accepted here.

    2. **Rank.** Only the top-1 label used to be considered. On real street
       footage the top-1 is almost always "walking towards/apart from each
       other" — which is what a snatch looks like in aggregate — while the
       theft class sits at rank 2 or 3. Checking the whole top-k list is what
       makes those reachable.
    """
    if not action:
        return []

    out: list[tuple[str, float]] = []

    def _conf(d: dict) -> float:
        try:
            return float(d.get("confidence", d.get("score", 0)) or 0)
        except (TypeError, ValueError):
            return 0.0

    items = action.get("actions")
    if isinstance(items, list):
        for a in items:
            if isinstance(a, dict):
                lbl = a.get("label") or a.get("action")
                if lbl:
                    out.append((str(lbl), _conf(a)))

    lbl = action.get("action") or action.get("label")
    if lbl:
        out.append((str(lbl), _conf(action)))

    return out


_SEVERITY_NAMES = [k for k, _ in sorted(SEVERITY_ORDER.items(), key=lambda kv: kv[1])]


def _severity_step(severity: str, delta: int) -> str:
    """Move a severity up or down the scale, clamped at both ends."""
    idx = SEVERITY_ORDER.get(severity, 0) + delta
    return _SEVERITY_NAMES[max(0, min(len(_SEVERITY_NAMES) - 1, idx))]


# ── Signal vocabularies ───────────────────────────────────────────────────

# WEAPON_CLASSES and the wider CRIME_TOOL_CLASSES vocabulary live in
# `backend.weapons`, alongside the grip geometry and the pluggable
# weapon-detector slot. They are re-exported here because the rules and the
# caption-corroboration logic below both consult them.

# Classes the vehicle rules apply to. `bicycle` is included because a struck
# cyclist is the incident even though the bicycle is not the striking vehicle.
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "train"}

# Vehicles heavy and fast enough for contact to constitute a collision rather
# than a parking nudge. Bicycles are excluded as *strikers* for that reason.
STRIKING_VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "train"}

# Crime / violence classes, keyed by action label. Covers both the NTU-60 and
# NTU-120 spellings — see action/ntu120.py for the mapping and the reasoning
# behind each score.
from action.ntu120 import (  # noqa: E402
    CRIME_ACTION_MAP, CRIME_SEVERITY, CRIME_LABELS,
    DISTRESS_ACTIONS_120 as DISTRESS_ACTIONS,
)

# Retained for the caption-corroboration logic below, which asks "did the
# skeleton model see violence?" rather than "which crime was it?".
VIOLENT_ACTIONS: dict[str, float] = {
    label: base
    for label, (etype, _cat, base, _two) in CRIME_ACTION_MAP.items()
    if etype in ("violence", "armed_threat")
}

# Event types that describe a brief, rare act rather than a sustained state.
# These use `action_conf_brief` — see ThreatConfig for the reasoning.
BRIEF_EVENT_TYPES = {"pickpocket", "snatch_theft", "armed_threat"}

# Theft classes, used the same way for caption corroboration.
THEFT_ACTIONS: dict[str, float] = {
    label: base
    for label, (etype, _cat, base, _two) in CRIME_ACTION_MAP.items()
    if etype in ("pickpocket", "snatch_theft")
}

# Concealment / handoff behaviour. Weak on their own — only escalate when
# combined with other signals.
#
# Note: "touch pocket" is deliberately NOT here. In both NTU taxonomies that
# class is A57, "touch *other person's* pocket", i.e. pickpocketing — it is
# handled as a theft class in CRIME_ACTION_MAP. Listing it here as well would
# double-count the same prediction. "reach into pocket" (A25) is the
# self-directed one and belongs here.
FURTIVE_ACTIONS: dict[str, float] = {
    "reach into pocket": 0.35,
    "giving something": 0.30,
    "giving something to other person": 0.30,
    "exchange things with other person": 0.30,
    "put something into a bag": 0.25,
    "take something out of a bag": 0.25,
}

# Objects that become interesting when left alone.
ABANDONABLE_CLASSES = {"backpack", "handbag", "suitcase"}

# Tier-2 open-vocabulary lexicon applied to VLM captions. Each entry maps a
# regex to (event_type, severity, base_score). Ordered most-specific first.
CAPTION_LEXICON: list[tuple[str, str, str, float]] = [
    (r"\b(gun|firearm|pistol|rifle|shotgun|handgun|armed)\b",
     "weapon_reported", "critical", 0.85),
    # `panga` is the East African term for a machete and is what a caption of
    # footage from the region is most likely to use; `bush ?knife` and `cutlass`
    # cover the same implement elsewhere. Omitting regional vocabulary silently
    # blinds the screen on exactly the footage it is pointed at.
    (r"\b(knife|blade|machete|panga|cutlass|bush ?knife|matchet|"
     r"stabb?(ing|ed)?)\b",
     "weapon_reported", "critical", 0.80),
    (r"\b(shoot(ing|s)?|shot|gunfire)\b",
     "shooting_reported", "critical", 0.85),
    (r"\b(fight(ing)?|brawl|assault(ing|ed)?|attack(ing|ed)?|punch(ing|ed)?|beating)\b",
     "violence_reported", "high", 0.75),
    (r"\b(rob(bing|bery)|mugg(ing|ed)|hold[- ]?up)\b",
     "robbery_reported", "critical", 0.80),
    (r"\b(steal(ing)?|stole|theft|shoplift(ing)?|pickpocket(ing)?|snatch(ing|ed)?)\b",
     "theft_reported", "high", 0.65),
    (r"\b(break(ing)? in|burglar(y|ising|izing)?|forc(ing|ed) (the )?(door|window)|prying)\b",
     "burglary_reported", "high", 0.70),
    # Entry/defeat tools. `weapons.py::CRIME_TOOL_CLASSES` already scores these
    # as burglary/improvised for the object detector, but the caption lexicon
    # was silent on every one of them, so a caption reading "smashing the
    # display case with a hammer" matched only `smash` → vandalism at *medium*.
    # A smash-and-grab raid is not vandalism. Kept above the vandalism entry for
    # readability; the loop accumulates rather than stopping at the first hit,
    # so both may fire on the same caption.
    (r"\b(sledge ?hammers?|hammers?|crowbars?|bolt ?cutters?|angle grinders?|"
     r"pry ?bars?|smash[- ]and[- ]grab)\b",
     "burglary_reported", "high", 0.70),
    (r"\b(vandali[sz](ing|ed|sm)|graffiti|smash(ing|ed)?|destroy(ing)?)\b",
     "vandalism_reported", "medium", 0.55),
    (r"\b(arson|set(ting)? (it |the )?(on )?fire|fire|flames|smoke|burning)\b",
     "fire_reported", "high", 0.60),
    (r"\b(climb(ing)? (over|the) (fence|wall|gate)|trespass(ing)?|forced entry)\b",
     "intrusion_reported", "medium", 0.55),
    (r"\b(unconscious|collaps(ing|ed)|lying motionless|body on the ground|injured|bleeding)\b",
     "person_down_reported", "high", 0.65),
    (r"\b(crash(ing|ed|es)?|collision|collid(ing|ed)|accident|"
     r"(ran|run|knocked) (over|down|into)|rear[- ]ended|hit[- ]and[- ]run)\b",
     "collision_reported", "high", 0.70),
    (r"\b(chas(ing|ed)|fleeing|running away|pursu(ing|it))\b",
     "pursuit_reported", "medium", 0.45),
    (r"\b(crowd (surge|panic|stampede)|people (fleeing|scattering|running))\b",
     "crowd_panic_reported", "high", 0.60),
    (r"\b(drug|deal(ing)?|syringe|needle)\b",
     "narcotics_reported", "medium", 0.45),
]

# ── Negation handling ─────────────────────────────────────────────────────
#
# Captioning models routinely answer an incident prompt by *denying the whole
# list*: "There is no visible weapon, fighting, theft, forced entry, vandalism,
# fire, or a person who appears injured." A fixed-width lookbehind cannot cope
# with that — by the fourth item the negation cue is 60+ characters back — and
# the result is that the cleanest possible caption fires the most alerts.
#
# So negation is scoped to the sentence instead: a lexicon hit is suppressed if
# a negation cue appears earlier in the same sentence, unless a contrastive
# conjunction intervenes ("no fence, but a man is stabbing someone").

NEGATION_CUES = re.compile(
    r"\b(no|not|n't|without|nothing|none|nobody|no one|neither|nor|"
    r"absence of|free of|clear of|devoid of|unable to see)\b"
)

CONTRAST_CUES = re.compile(r"\b(but|however|although|though|except|yet|whereas)\b")

_SENTENCE_END = re.compile(r"[.!?;\n]")

# Retained for backwards compatibility with anything importing it.
NEGATION_RE = NEGATION_CUES


def _is_negated(text: str, match_start: int) -> bool:
    """Does a negation cue govern the hit at `match_start`?"""
    starts = [m.end() for m in _SENTENCE_END.finditer(text, 0, match_start)]
    scope = text[(starts[-1] if starts else 0):match_start]

    # A contrastive conjunction ends the negation's reach.
    contrasts = list(CONTRAST_CUES.finditer(scope))
    if contrasts:
        scope = scope[contrasts[-1].end():]
    return bool(NEGATION_CUES.search(scope))


# The VLM's answer to `threat_prompt()`. Screening only this line — rather than
# the whole free-text description — is what keeps a scene description of an
# ordinary street from tripping the lexicon.
INCIDENT_LINE_RE = re.compile(r"incident\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)
NO_INCIDENT_RE = re.compile(
    r"^\s*(none|no\b|n/?a|nothing|nil)", re.IGNORECASE
)


# ── Configuration ─────────────────────────────────────────────────────────

@dataclass
class ThreatConfig:
    """Runtime-tunable thresholds. Every number here is a policy choice."""

    # Global
    enabled: bool = True
    min_severity: str = "low"          # events below this are dropped
    cooldown_s: float = 12.0           # per event-type debounce
    min_score: float = 0.40            # events below this score are dropped

    # Detection-based rules
    weapon_conf: float = 0.45
    weapon_person_proximity: float = 0.25   # frac of frame diagonal

    # Grip reasoning — an object at the end of an arm vs. one merely nearby.
    # Tolerance is a fraction of the *person's* box diagonal, so it holds at any
    # distance from the camera.
    grip_enabled: bool = True
    grip_tolerance: float = 0.18
    grip_kp_conf: float = 0.30              # min keypoint confidence for a wrist
    carried_score_factor: float = 0.75      # penalty for "in the box, not in a hand"
    crime_tool_conf: float = 0.40           # detector floor for non-weapon tools

    # Vehicle dynamics. Speeds are fractions of the frame diagonal per second,
    # so they survive a change of resolution — but NOT a change of camera angle,
    # which is why these need tuning per site.
    vehicle_rules: bool = True
    collision_iou: float = 0.08             # box overlap that counts as contact
    collision_min_speed: float = 0.05       # closing speed before impact
    collision_decel_ratio: float = 0.45     # post-impact speed ≤ this × pre-impact
    collision_contact_frames: int = 2       # sustained contact before firing
    collision_settle_s: float = 1.0         # footage to watch before judging a contact
    pedestrian_impact_speed: float = 0.03   # lower bar: any moving vehicle
    hit_and_run_window_s: float = 25.0      # departure after impact still counts
    hit_and_run_min_speed: float = 0.04     # departure must be under power

    # Trajectory snatch. The only rule here that works at the resolution our
    # own footage actually has. Pose finds two skeletons in 0/23 frames at
    # Naalya because the figures are ~20px; a *box centre over time* survives
    # that, and detection does find the motorcycle (5-12 detections across the
    # same span). So this reads the shape of the event rather than the bodies:
    #
    #   a vehicle closes on a pedestrian, they are briefly co-located, and the
    #   vehicle leaves *faster than it arrived* instead of slowing down.
    #
    # That last clause is what separates a snatch from a collision and from a
    # boda simply passing someone. `_rule_vehicles` already looks for impact
    # deceleration; this looks for its opposite.
    #
    # It cannot detect pickpocketing — that genuinely needs the hand — so this
    # narrows the system to snatch-from-vehicle, which is where five of our six
    # confirmed spans sit anyway.
    # MEASURED 2026-08-29: fires 0/5 on our confirmed spans and 0/1 on the
    # normal span. Not a threshold problem — the tracks do not exist. Over an
    # 11s span the tracker creates 28 IDs with a MEDIAN AGE OF 0.00s and a
    # median history of one point, so speed() (which needs two) returns 0 and
    # the approach gate rejects everything. Detections at 360p flicker in and
    # out frame to frame and identity cannot survive the gaps.
    #
    # So low resolution defeats tracking as well as pose. Left enabled because
    # the rule is sound and costs nothing when tracks are empty; it should
    # start working on footage where detection is stable. Re-measure before
    # trusting it, and consider a Kalman/ByteTrack-style tracker first —
    # SimpleTracker does nearest-box matching with no motion prediction, so it
    # cannot bridge a missed frame.
    snatch_trajectory: bool = True
    snatch_approach_speed: float = 0.03     # vehicle must actually be moving
    snatch_contact_frac: float = 0.10       # centres within this × frame diagonal
    snatch_max_contact_s: float = 2.5       # a snatch is brief; a pickup is not
    snatch_departure_ratio: float = 1.05    # leaves at >= this × approach speed
    snatch_departure_s: float = 2.0         # window watched after separation
    snatch_min_score: float = 0.45

    # Action-based rules
    action_conf: float = 0.40
    # Brief, rare classes get a lower bar. A snatch lasts ~0.2 s, so even the
    # best-aligned window contains mostly context; measured peak confidence on
    # a real snatch was 0.53, with the surrounding windows at 0.11-0.29. A flat
    # 0.40 gate therefore turns a correct detection into a miss whenever the
    # sampling is a frame or two off. This is an explicit recall-over-precision
    # choice for theft and weapon classes: expect more false positives, and
    # keep a human reviewing them.
    action_conf_brief: float = 0.25
    violence_requires_two_people: bool = True

    # Proximity / grouping
    crowd_person_threshold: int = 8
    crowd_surge_delta: int = 5          # persons appearing within surge_window
    crowd_surge_window_s: float = 4.0
    crowd_dispersal_delta: int = 5      # persons vanishing within window

    # Dwell / loitering
    loiter_seconds: float = 90.0
    loiter_motion_tolerance: float = 0.06   # frac of frame diagonal

    # Unattended objects
    unattended_seconds: float = 45.0
    unattended_owner_distance: float = 0.22  # frac of frame diagonal

    # Restricted zones: list of {"name": str, "polygon": [[x,y], ...]}
    # Coordinates normalised 0-1 relative to frame width/height.
    zones: list[dict] = field(default_factory=list)
    zone_severity: str = "medium"

    # After-hours: persons detected outside this window escalate.
    after_hours_enabled: bool = False
    after_hours_start: int = 22         # local hour, inclusive
    after_hours_end: int = 6            # local hour, exclusive

    # Tier 2 (caption screening)
    caption_screening: bool = True
    caption_min_score: float = 0.40

    # Evidence
    save_snapshots: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    def update(self, patch: dict) -> "ThreatConfig":
        for k, v in patch.items():
            if hasattr(self, k) and v is not None:
                setattr(self, k, v)
        return self


# ── Geometry helpers ──────────────────────────────────────────────────────

def _centroid(box: Iterable[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _iou(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _dist(p: tuple[float, float], q: tuple[float, float]) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def _format_timecode(seconds: float | None) -> str:
    """Seconds into the footage → HH:MM:SS.mmm, the form a reviewer scrubs to."""
    if seconds is None:
        return "--:--:--"
    seconds = max(0.0, float(seconds))
    return (f"{int(seconds // 3600):02d}:"
            f"{int((seconds % 3600) // 60):02d}:"
            f"{seconds % 60:06.3f}")


def _point_in_polygon(pt: tuple[float, float], poly: list[list[float]]) -> bool:
    """Ray casting. `poly` is a list of [x, y] in the same units as `pt`."""
    x, y = pt
    inside = False
    n = len(poly)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i][0], poly[i][1]
        xj, yj = poly[j][0], poly[j][1]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi):
            inside = not inside
        j = i
    return inside


# ── Lightweight tracker ───────────────────────────────────────────────────

@dataclass
class Track:
    track_id: int
    cls: str
    box: list[float]
    centroid: tuple[float, float]
    first_seen: float
    last_seen: float
    anchor: tuple[float, float]        # position used for dwell measurement
    anchor_time: float
    misses: int = 0
    history: deque = field(default_factory=lambda: deque(maxlen=60))
    fired: set = field(default_factory=set)   # rules already fired for this track
    speed_history: deque = field(default_factory=lambda: deque(maxlen=120))

    @property
    def age(self) -> float:
        return self.last_seen - self.first_seen

    @property
    def dwell(self) -> float:
        return self.last_seen - self.anchor_time

    def speed(self, window: float = 0.5) -> float:
        """Mean speed in pixels/second over roughly the last `window` seconds.

        Measured over a window rather than frame-to-frame because box jitter on
        a stationary object produces a large instantaneous speed that would
        swamp any deceleration test.
        """
        if len(self.history) < 2:
            return 0.0
        t_end, c_end = self.history[-1]
        t_start, c_start = self.history[0]
        for t, c in reversed(self.history):
            if t_end - t >= window:
                t_start, c_start = t, c
                break
        dt = t_end - t_start
        return _dist(c_end, c_start) / dt if dt > 1e-6 else 0.0

    def peak_speed(self, since: float, until: float | None = None) -> float:
        """Highest recorded speed in a time range — the pre-impact approach speed.

        Peak rather than mean: a vehicle that brakes hard just before contact
        still arrived fast, and averaging would hide exactly the event we want.
        """
        until = until if until is not None else float("inf")
        vals = [s for t, s in self.speed_history if since <= t <= until]
        return max(vals) if vals else 0.0

    def heading(self, window: float = 0.5) -> tuple[float, float] | None:
        """Unit direction of travel, or None when effectively stationary."""
        if len(self.history) < 2:
            return None
        t_end, c_end = self.history[-1]
        t_start, c_start = self.history[0]
        for t, c in reversed(self.history):
            if t_end - t >= window:
                t_start, c_start = t, c
                break
        dx, dy = c_end[0] - c_start[0], c_end[1] - c_start[1]
        mag = math.hypot(dx, dy)
        return (dx / mag, dy / mag) if mag > 1e-6 else None


class SimpleTracker:
    """Greedy IoU tracker. Adequate for dwell/loitering, not for re-ID.

    Cross-shot identity is `backend.identity`'s job; this only has to survive a
    few frames of occlusion.
    """

    def __init__(self, iou_threshold: float = 0.3, max_misses: int = 15):
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self.tracks: dict[int, Track] = {}
        self.lost: list[Track] = []     # dropped on the most recent update
        self._next_id = 1

    def update(self, objects: list[dict], now: float, diag: float) -> dict[int, Track]:
        unmatched = set(self.tracks.keys())
        used: set[int] = set()
        self.lost = []

        for obj in objects:
            box = [float(v) for v in obj["box"]]
            cls = obj.get("class", "unknown")
            best_id, best_iou = None, self.iou_threshold
            for tid in unmatched:
                t = self.tracks[tid]
                if t.cls != cls or tid in used:
                    continue
                score = _iou(t.box, box)
                if score > best_iou:
                    best_id, best_iou = tid, score

            c = _centroid(box)
            if best_id is None:
                tid = self._next_id
                self._next_id += 1
                self.tracks[tid] = Track(
                    track_id=tid, cls=cls, box=box, centroid=c,
                    first_seen=now, last_seen=now, anchor=c, anchor_time=now,
                )
                self.tracks[tid].history.append((now, c))
                used.add(tid)
            else:
                t = self.tracks[best_id]
                t.box = box
                t.centroid = c
                t.last_seen = now
                t.misses = 0
                t.history.append((now, c))
                t.speed_history.append((now, t.speed()))
                # Reset the dwell anchor if the object has actually moved.
                if diag > 0 and _dist(c, t.anchor) / diag > 0.06:
                    t.anchor = c
                    t.anchor_time = now
                    t.fired.discard("loiter")
                    t.fired.discard("unattended")
                unmatched.discard(best_id)
                used.add(best_id)

        for tid in list(unmatched):
            t = self.tracks[tid]
            t.misses += 1
            if t.misses > self.max_misses:
                self.lost.append(t)
                del self.tracks[tid]

        return self.tracks

    def by_class(self, cls: str) -> list[Track]:
        return [t for t in self.tracks.values() if t.cls == cls and t.misses == 0]


# ── Engine ────────────────────────────────────────────────────────────────

class ThreatEngine:
    """Consumes per-frame perception output, emits incident events.

    Usage per frame:

        events = engine.update(
            detections=payload["detection"]["objects"],
            action=payload.get("action"),
            frame_shape=frame.shape,
        )

    And whenever a fresh caption arrives (VLM cadence, not every frame):

        events += engine.ingest_caption(caption_text)
    """

    def __init__(self, config: ThreatConfig | None = None):
        self.config = config or ThreatConfig()
        self.tracker = SimpleTracker()
        self._last_fired: dict[str, float] = {}
        self._person_history: deque[tuple[float, int]] = deque(maxlen=300)
        self._frame_diag: float = 1.0
        self._last_signals: dict[str, Any] = {}
        self.events: list[dict] = []
        self.MAX_EVENTS = 500
        # Video clock — set per frame so events can name a position in the clip
        # rather than only a wall-clock instant. See set_clock().
        self._video_time_s: float | None = None
        self._frame_index: int | None = None
        self._source: str | None = None
        # Collision bookkeeping
        self._contacts: dict[tuple[int, int], dict] = {}
        self._recent_collisions: dict[int, dict] = {}   # track_id → collision record
        self._snatch_contacts: dict[str, dict] = {}     # vehicle:person → encounter

    # ── public API ────────────────────────────────────────────────────────

    def set_clock(self, video_time_s: float | None = None,
                  frame: int | None = None,
                  source: str | None = None):
        """Tell the engine where in the footage it currently is.

        Every event minted afterwards carries `video_time_s`, `frame` and a
        `timecode`, which is what lets a report say "weapon at 00:01:47.320"
        instead of only naming the instant the analysis happened to run.
        """
        if video_time_s is not None:
            self._video_time_s = float(video_time_s)
        if frame is not None:
            self._frame_index = int(frame)
        if source is not None:
            self._source = source

    def update(self,
               detections: list[dict] | None = None,
               action: dict | None = None,
               pose_persons: list[dict] | None = None,
               frame_shape: tuple[int, ...] | None = None,
               now: float | None = None) -> list[dict]:
        cfg = self.config
        if not cfg.enabled:
            return []

        now = now if now is not None else time.time()
        detections = detections or []

        if frame_shape is not None and len(frame_shape) >= 2:
            h, w = float(frame_shape[0]), float(frame_shape[1])
            self._frame_diag = math.hypot(w, h) or 1.0
            self._frame_wh = (w, h)
        else:
            self._frame_wh = getattr(self, "_frame_wh", (1.0, 1.0))

        persons = [d for d in detections if d.get("class") == "person"]
        self._person_history.append((now, len(persons)))
        self.tracker.update(detections, now, self._frame_diag)

        # Detection and pose are separate networks with separate thresholds and
        # they routinely disagree — measured on real footage, the detector found
        # one person in a frame where pose found two. The action model is fed by
        # *pose*, so gating its rules on the *detector's* count can veto a
        # two-person interaction the skeleton model saw perfectly well. Take the
        # larger of the two: for a guard whose job is "was anyone else there?",
        # either network seeing a second body is sufficient evidence.
        people_count = max(len(persons), len(pose_persons or []))

        self._last_signals = {
            "person_count": len(persons),
            "detections": [d.get("class") for d in detections],
            "actions": [lbl for lbl, _c in action_candidates(action)],
        }

        out: list[dict] = []
        out += self._rule_weapon(detections, persons, pose_persons, now)
        out += self._rule_vehicles(detections, persons, now)
        out += self._rule_snatch_trajectory(now)
        out += self._rule_violent_action(action, persons, now,
                                         people_count=people_count)
        out += self._rule_distress(action, now)
        out += self._rule_crowd(now)
        out += self._rule_zones(persons, now)
        out += self._rule_loitering(now)
        out += self._rule_unattended(now)
        out += self._rule_after_hours(persons, now)

        return self._finalise(out, now)

    def ingest_caption(self, caption: str, now: float | None = None) -> list[dict]:
        """Tier 2: open-vocabulary screening of a VLM caption."""
        cfg = self.config
        if not cfg.enabled or not cfg.caption_screening or not caption:
            return []
        now = now if now is not None else time.time()

        # `threat_prompt` asks for a dedicated INCIDENT: line. When the model
        # complies, screen only that line — the surrounding scene description
        # is prose about an ordinary street and only generates noise.
        screened = caption
        marker = INCIDENT_LINE_RE.search(caption)
        if marker:
            answer = marker.group(1).strip()
            if NO_INCIDENT_RE.match(answer):
                return []
            screened = answer
        text = screened.lower()

        out: list[dict] = []
        for pattern, etype, severity, base in CAPTION_LEXICON:
            m = re.search(pattern, text)
            if not m:
                continue
            if _is_negated(text, m.start()):
                continue
            score = base
            # Corroboration bonus: a caption hit backed by a structured signal
            # is much more trustworthy than a caption hit alone.
            if self._corroborates(etype):
                score = min(1.0, score + 0.15)
            else:
                score -= 0.10
            if score < cfg.caption_min_score:
                continue
            out.append(self._make_event(
                etype=etype,
                label=etype.replace("_", " ").replace(" reported", "").strip().title(),
                category="caption",
                severity=severity,
                score=score,
                rule="caption_lexicon",
                detail=f"VLM caption matched /{pattern}/",
                evidence={"caption": caption, "match": m.group(0)},
                now=now,
            ))
        return self._finalise(out, now)

    def threat_prompt(self) -> str:
        """Prompt for the VLM that biases it toward reporting incidents.

        Replaces the generic 'describe the scene' prompt when screening is on.
        Kept short: FastVLM-0.5B degrades with long instructions.

        The two-line structure is load-bearing. Asking the model to describe a
        frame *and* consider a list of incident types in one breath makes it
        answer by denying the list item by item — "no weapon, no fighting, no
        theft…" — which reads to a keyword screen as a hit on every one of them.
        Confining the verdict to its own INCIDENT: line separates the answer
        from the description, and `ingest_caption` screens only the answer.
        """
        return (
            "You are reviewing a security camera frame.\n"
            "Line 1: describe what the people are doing, in one sentence.\n"
            "Line 2: write 'INCIDENT:' followed by NONE if nothing is wrong, or "
            "else a few words naming what you actually see happening (for example "
            "a weapon, a fight, a theft, forced entry, vandalism, fire, or an "
            "injured person).\n"
            "Do not list things that are absent."
        )

    def recent(self, limit: int = 50, min_severity: str | None = None) -> list[dict]:
        items = self.events[-limit:][::-1]
        if min_severity:
            items = [e for e in items if severity_at_least(e["severity"], min_severity)]
        return items

    def stats(self) -> dict:
        by_sev: dict[str, int] = {}
        by_type: dict[str, int] = {}
        for e in self.events:
            by_sev[e["severity"]] = by_sev.get(e["severity"], 0) + 1
            by_type[e["type"]] = by_type.get(e["type"], 0) + 1
        return {
            "total": len(self.events),
            "by_severity": by_sev,
            "by_type": by_type,
            "active_tracks": len(self.tracker.tracks),
        }

    def reset(self):
        self.events.clear()
        self._last_fired.clear()
        self.tracker = SimpleTracker()
        self._person_history.clear()
        self._contacts.clear()
        self._recent_collisions.clear()
        self._snatch_contacts.clear()
        self._video_time_s = None
        self._frame_index = None

    # ── rules ─────────────────────────────────────────────────────────────

    def _rule_weapon(self, detections: list[dict], persons: list[dict],
                     pose_persons: list[dict] | None, now: float) -> list[dict]:
        """Crime tools, ranked by how strongly they are attached to a person.

        Three tiers of evidence, strongest first:

          held      — the object sits at a wrist keypoint. This is the signal
                      that distinguishes "brandishing a knife" from "standing in
                      a kitchen", and it is the only one worth waking someone for.
          carried   — the object is inside a person's box but not at a hand:
                      under an arm, in a waistband, or hand occluded.
          proximate — the object is merely near a person. Retained as a fallback
                      for when pose is unavailable, and reported as such.
        """
        cfg = self.config
        out = []

        tools = [d for d in detections
                 if d.get("box") and tool_info(d.get("class")) is not None]
        if not tools:
            return out

        # Object dicts flow through by reference, so identity keys the lookup.
        grips: dict[int, dict] = {}
        carried: dict[int, dict] = {}
        have_pose = bool(cfg.grip_enabled and pose_persons)
        if have_pose:
            for m in held_objects(tools, pose_persons,
                                  grip_tolerance=cfg.grip_tolerance,
                                  kp_conf=cfg.grip_kp_conf):
                grips[id(m["object"])] = m
            for m in carried_objects([t for t in tools if id(t) not in grips], persons):
                carried[id(m["object"])] = m

        for d in tools:
            cls = (d.get("class") or "").lower()
            info = tool_info(cls) or {}
            category = info.get("category", "weapon")
            weight = float(info.get("weight", 0.5))
            label_name = info.get("label", cls.title())

            conf = float(d.get("confidence", 0))
            floor = cfg.weapon_conf if category == "weapon" else cfg.crime_tool_conf
            if conf < floor:
                continue

            base = weight * conf
            grip = grips.get(id(d))
            carry = carried.get(id(d))

            if grip is not None:
                score = min(1.0, base + 0.10 + 0.20 * grip["grip"])
                severity = CATEGORY_SEVERITY.get(category, "high")
                etype = "weapon_brandished" if category == "weapon" else "crime_tool_held"
                detail = (f"{label_name} at a person's {grip['hand']} hand "
                          f"({grip['distance_px']:.0f}px from the wrist)")
                evidence = {"class": cls, "confidence": conf, "box": d["box"],
                            "attachment": "held", "hand": grip["hand"],
                            "grip_confidence": grip["grip"],
                            "person_box": grip["person"].get("box")}
                subject = {"class": "person", "box": grip["person"].get("box")}
            elif carry is not None:
                score = min(1.0, base * cfg.carried_score_factor + 0.05)
                severity = _severity_step(CATEGORY_SEVERITY.get(category, "high"), -1)
                etype = "weapon_carried" if category == "weapon" else "crime_tool_carried"
                detail = (f"{label_name} on a person but not at a hand "
                          f"({carry['containment']:.0%} inside their outline)")
                evidence = {"class": cls, "confidence": conf, "box": d["box"],
                            "attachment": "carried",
                            "containment": carry["containment"],
                            "person_box": carry["person"].get("box")}
                subject = {"class": "person", "box": carry["person"].get("box")}
            else:
                # Fallback: centroid proximity. Weaker, and weaker still when
                # pose was available and simply found no hand holding it.
                wc = _centroid(d["box"])
                near = None
                for p in persons:
                    dist = _dist(wc, _centroid(p["box"])) / self._frame_diag
                    if dist <= cfg.weapon_person_proximity and (near is None or dist < near):
                        near = dist

                # Non-weapon tools with nobody holding them are scene furniture.
                if category != "weapon" and near is None:
                    continue

                if near is not None and not have_pose:
                    score = min(1.0, base + 0.15)
                    severity = "critical"
                    etype = "weapon_brandished"
                    detail = (f"{label_name} within {near:.0%} of frame diagonal of a "
                              f"person (no pose data — proximity only)")
                elif near is not None:
                    score = base * 0.85
                    severity = "high"
                    etype = "weapon_near_person"
                    detail = (f"{label_name} near a person but not matched to a hand "
                              f"({near:.0%} of frame diagonal away)")
                else:
                    score = base * 0.8
                    severity = "high"
                    etype = "weapon_visible"
                    detail = f"{label_name} detected, no person in close proximity"

                evidence = {"class": cls, "confidence": conf, "box": d["box"],
                            "attachment": "proximate" if near is not None else "none",
                            "person_distance": near}
                subject = None

            out.append(self._make_event(
                etype=etype, label=f"{label_name} detected",
                category="weapon" if category == "weapon" else "crime_tool",
                severity=severity, score=score, rule="crime_tool_detection",
                detail=detail, evidence=evidence, subject=subject,
                now=now,
            ))
        return out

    # ── vehicles ──────────────────────────────────────────────────────────

    def _rule_snatch_trajectory(self, now: float) -> list[dict]:
        """A snatch read from box trajectories, not from bodies.

        Fires when a vehicle closes on a pedestrian, is briefly beside them,
        and then leaves *faster than it arrived*. The departure clause is the
        whole rule: a collision decelerates, a passing boda holds its speed, a
        pickup lingers. Only a snatch accelerates away from a momentary contact.

        Deliberately blind to what happened between the two boxes — at 20px
        there is nothing to see there. It reads the shape of the encounter.
        """
        cfg = self.config
        if not cfg.snatch_trajectory:
            return []
        diag = self._frame_diag or 1.0
        live = [t for t in self.tracker.tracks.values() if t.misses == 0]
        vehicles = [t for t in live if t.cls in VEHICLE_CLASSES]
        people = [t for t in live if t.cls == "person"]
        out: list[dict] = []

        for v in vehicles:
            for p in people:
                key = f"snatch:{v.track_id}:{p.track_id}"
                rec = self._snatch_contacts.get(key)
                close = _dist(v.centroid, p.centroid) / diag <= cfg.snatch_contact_frac

                if close:
                    if rec is None:
                        # Record the approach speed *now*, before contact: after
                        # separation the speed history is dominated by the
                        # departure and the arrival is no longer recoverable.
                        approach = v.peak_speed(now - 2.0, now) / diag
                        if approach < cfg.snatch_approach_speed:
                            continue
                        self._snatch_contacts[key] = {
                            "start": now, "approach": approach,
                            "vid": v.track_id, "pid": p.track_id, "fired": False}
                    continue

                if rec is None or rec["fired"]:
                    continue

                contact_s = now - rec["start"]
                since_sep = now - (rec.get("separated") or now)
                rec.setdefault("separated", now)
                if contact_s > cfg.snatch_max_contact_s:
                    # Too long beside each other to be a snatch — a drop-off, a
                    # conversation, traffic. Drop it rather than let it ripen.
                    self._snatch_contacts.pop(key, None)
                    continue
                if since_sep < cfg.snatch_departure_s:
                    continue                      # not enough departure to judge

                departure = v.peak_speed(rec["separated"], now) / diag
                ratio = departure / rec["approach"] if rec["approach"] > 1e-6 else 0.0
                if ratio < cfg.snatch_departure_ratio:
                    self._snatch_contacts.pop(key, None)
                    continue

                # Brief contact and a faster exit. Score on how much faster and
                # how brief — both are what distinguishes this from traffic.
                score = min(1.0, 0.40 + 0.25 * min(ratio - 1.0, 1.0)
                            + 0.25 * (1.0 - contact_s / cfg.snatch_max_contact_s))
                rec["fired"] = True
                if score < cfg.snatch_min_score:
                    continue
                out.append(self._make_event(
                    etype="snatch_theft", label="Possible snatch from vehicle",
                    category="trajectory", severity="high", score=round(score, 3),
                    rule="snatch_trajectory",
                    detail=(f"{v.cls} closed on a pedestrian, {contact_s:.1f}s "
                            f"alongside, then left at {ratio:.2f}x its approach speed"),
                    evidence={"vehicle_track": v.track_id, "person_track": p.track_id,
                              "contact_s": round(contact_s, 2),
                              "approach_speed": round(rec["approach"], 4),
                              "departure_speed": round(departure, 4),
                              "speed_ratio": round(ratio, 2)},
                    now=now, subject={"class": v.cls, "box": v.box},
                    cooldown_key=key))
        return out

    def _rule_vehicles(self, detections: list[dict], persons: list[dict],
                       now: float) -> list[dict]:
        """Collisions, and vehicles that leave straight afterwards.

        Box overlap alone is worthless from a fixed camera: a car passing behind
        another overlaps in projection without touching in the world. What makes
        contact a *collision* is what the motion does immediately afterwards —
        someone stops hard, or something that was still gets shoved. So contacts
        are recorded, then judged after `collision_settle_s` of footage.
        """
        cfg = self.config
        if not cfg.vehicle_rules:
            return []

        diag = self._frame_diag or 1.0
        live = [t for t in self.tracker.tracks.values() if t.misses == 0]
        vehicles = [t for t in live if t.cls in VEHICLE_CLASSES]
        person_tracks = [t for t in live if t.cls == "person"]
        out: list[dict] = []

        # ── 1. record contacts ────────────────────────────────────────────
        for i, a in enumerate(vehicles):
            for b in vehicles[i + 1:]:
                if _iou(a.box, b.box) >= cfg.collision_iou:
                    self._note_contact(a, b, "vehicle", now)
        for v in vehicles:
            if v.cls not in STRIKING_VEHICLE_CLASSES:
                continue
            for p in person_tracks:
                if _iou(v.box, p.box) >= cfg.collision_iou:
                    self._note_contact(v, p, "pedestrian", now)

        # ── 2. judge settled contacts ─────────────────────────────────────
        for key, rec in list(self._contacts.items()):
            if rec["resolved"]:
                if now - rec["last_contact"] > 5.0:
                    del self._contacts[key]
                continue
            if now - rec["start"] < cfg.collision_settle_s:
                continue

            rec["resolved"] = True
            if rec["frames"] < cfg.collision_contact_frames:
                continue

            a = self.tracker.tracks.get(rec["a_id"])
            b = self.tracker.tracks.get(rec["b_id"])
            verdict = self._judge_contact(rec, a, b, diag, now)
            if verdict is None:
                continue

            score, severity, detail, etype, label = verdict
            striker_id = rec["striker_id"]
            striker = self.tracker.tracks.get(striker_id)
            other_id = rec["b_id"] if striker_id == rec["a_id"] else rec["a_id"]
            other = self.tracker.tracks.get(other_id)

            event = self._make_event(
                etype=etype, label=label, category="traffic",
                severity=severity, score=score, rule="collision_dynamics",
                detail=detail,
                evidence={
                    "kind": rec["kind"],
                    "striker": {"track_id": striker_id,
                                "class": rec["striker_cls"],
                                "box": striker.box if striker else rec["striker_box"],
                                "approach_speed": round(rec["pre_striker"] / diag, 4)},
                    "other": {"track_id": other_id,
                              "class": rec["other_cls"],
                              "box": other.box if other else rec["other_box"]},
                    "contact_frames": rec["frames"],
                },
                # The striking vehicle is what gets committed to memory.
                subject={"class": rec["striker_cls"],
                         "box": striker.box if striker else rec["striker_box"],
                         "track_id": striker_id},
                cooldown_key=f"{etype}:{key[0]}:{key[1]}",
                now=now,
            )
            out.append(event)

            # Arm the departure watch for everything involved.
            for tid, cls in ((striker_id, rec["striker_cls"]),
                             (other_id, rec["other_cls"])):
                self._recent_collisions[tid] = {
                    "time": now, "event_id": event["id"], "cls": cls,
                    "role": "striker" if tid == striker_id else "other",
                    "box": (self.tracker.tracks[tid].box
                            if tid in self.tracker.tracks else None),
                }

        # ── 3. departures after a collision ───────────────────────────────
        out += self._rule_left_scene(diag, now)
        return out

    def _note_contact(self, a: Track, b: Track, kind: str, now: float):
        """Open or extend a contact record between two tracks."""
        cfg = self.config
        key = (min(a.track_id, b.track_id), max(a.track_id, b.track_id))
        rec = self._contacts.get(key)
        if rec is None:
            # Approach speeds must be sampled from *before* contact, so this
            # window deliberately ends at the moment contact was first seen.
            pre_a = a.peak_speed(now - 2.0, now)
            pre_b = b.peak_speed(now - 2.0, now)
            striker_is_a = (pre_a >= pre_b) if kind == "vehicle" else True
            striker, other = (a, b) if striker_is_a else (b, a)
            self._contacts[key] = {
                "a_id": a.track_id, "b_id": b.track_id, "kind": kind,
                "start": now, "last_contact": now, "frames": 1, "resolved": False,
                "pre_a": pre_a, "pre_b": pre_b,
                "pre_striker": max(pre_a, pre_b) if kind == "vehicle" else pre_a,
                "pre_other": min(pre_a, pre_b) if kind == "vehicle" else pre_b,
                "striker_id": striker.track_id, "striker_cls": striker.cls,
                "other_cls": other.cls,
                "striker_box": list(striker.box), "other_box": list(other.box),
            }
        else:
            rec["frames"] += 1
            rec["last_contact"] = now

    def _judge_contact(self, rec: dict, a: Track | None, b: Track | None,
                       diag: float, now: float):
        """Decide whether a settled contact looks like an impact.

        Returns ``(score, severity, detail, etype, label)`` or None. The three
        accepted signatures are: the striker stops hard; the struck party stops
        hard; or something that was stationary is suddenly displaced.
        """
        cfg = self.config
        striker = a if (a and a.track_id == rec["striker_id"]) else b
        other = b if striker is a else a

        pre_striker = rec["pre_striker"]
        min_speed = (cfg.pedestrian_impact_speed if rec["kind"] == "pedestrian"
                     else cfg.collision_min_speed)
        if pre_striker / diag < min_speed:
            return None                     # nothing was moving fast enough

        post_striker = striker.speed() if striker else 0.0
        post_other = other.speed() if other else 0.0
        pre_other = rec["pre_other"]

        reasons = []
        if post_striker <= cfg.collision_decel_ratio * pre_striker:
            reasons.append(
                f"striking {rec['striker_cls']} decelerated "
                f"{pre_striker / diag:.3f} → {post_striker / diag:.3f} diag/s"
            )
        if pre_other / diag >= min_speed and post_other <= cfg.collision_decel_ratio * pre_other:
            reasons.append(f"{rec['other_cls']} decelerated sharply")
        if pre_other / diag < 0.005 and post_other / diag > 0.02:
            reasons.append(f"stationary {rec['other_cls']} was displaced")
        if other is None and rec["kind"] == "pedestrian":
            # The pedestrian track vanished during contact — knocked down,
            # occluded by the vehicle, or both. Ambiguous, but not ignorable.
            reasons.append("pedestrian track lost at the moment of contact")

        if not reasons:
            return None                     # overlap without impact dynamics

        if rec["kind"] == "pedestrian":
            score = min(1.0, 0.55 + 0.35 * min(1.0, pre_striker / diag / 0.12))
            return (score, "critical",
                    f"{rec['striker_cls']} contacted a pedestrian; "
                    + "; ".join(reasons),
                    "vehicle_pedestrian_collision", "Vehicle struck a pedestrian")

        score = min(1.0, 0.50 + 0.35 * min(1.0, pre_striker / diag / 0.15))
        severity = "critical" if score >= 0.75 else "high"
        return (score, severity,
                f"{rec['striker_cls']} and {rec['other_cls']} in contact; "
                + "; ".join(reasons),
                "vehicle_collision", "Vehicle collision")

    def _rule_left_scene(self, diag: float, now: float) -> list[dict]:
        """A vehicle involved in a collision that then drives out of frame.

        Not a finding of hit-and-run — the camera's field of view is not the
        scene, and a vehicle may legitimately pull over out of shot. It is a
        prompt to check whether anyone stopped.
        """
        cfg = self.config
        out = []
        for lost in self.tracker.lost:
            rec = self._recent_collisions.get(lost.track_id)
            if rec is None:
                continue
            if now - rec["time"] > cfg.hit_and_run_window_s:
                del self._recent_collisions[lost.track_id]
                continue
            if lost.cls not in STRIKING_VEHICLE_CLASSES:
                continue
            departure = lost.peak_speed(rec["time"], now)
            if departure / diag < cfg.hit_and_run_min_speed:
                continue                    # left the frame slowly, or stopped

            del self._recent_collisions[lost.track_id]
            out.append(self._make_event(
                etype="vehicle_left_scene",
                label="Vehicle left after a collision",
                category="traffic", severity="high",
                score=0.70, rule="post_collision_departure",
                detail=(f"{lost.cls} involved in a collision "
                        f"{now - rec['time']:.0f}s earlier left the frame at "
                        f"{departure / diag:.3f} diag/s"),
                evidence={"track_id": lost.track_id, "class": lost.cls,
                          "role": rec["role"], "collision_event": rec["event_id"],
                          "seconds_after_impact": round(now - rec["time"], 1)},
                subject={"class": lost.cls, "box": rec["box"] or list(lost.box),
                         "track_id": lost.track_id},
                cooldown_key=f"vehicle_left_scene:{lost.track_id}",
                now=now,
            ))

        # Expire stale watches so the dict cannot grow without bound.
        for tid, rec in list(self._recent_collisions.items()):
            if now - rec["time"] > cfg.hit_and_run_window_s:
                del self._recent_collisions[tid]
        return out

    def _rule_violent_action(self, action: dict | None, persons: list[dict],
                             now: float,
                             people_count: int | None = None) -> list[dict]:
        """Crime and violence rules over the action classifier.

        Covers both taxonomies. NTU-60 gives interpersonal violence plus
        pickpocketing (A57); NTU-120 adds snatch theft (A109), knife (A107),
        gun (A110) and hit-with-object (A106) — the classes that make street
        theft detectable at all.
        """
        cfg = self.config
        # Consider every candidate the model returned, not just its top-1.
        best: tuple[float, str, tuple] | None = None
        for label, conf in action_candidates(action):
            mapping = CRIME_ACTION_MAP.get(label)
            if mapping is None:
                continue
            threshold = (cfg.action_conf_brief
                         if mapping[0] in BRIEF_EVENT_TYPES
                         else cfg.action_conf)
            if conf < threshold:
                continue
            weighted = mapping[2] * conf
            if best is None or weighted > best[0]:
                best = (weighted, label, mapping)

        if best is None:
            return []

        _w, label, mapping = best
        conf = dict(action_candidates(action)).get(label, 0.0)
        etype, category, base, needs_two = mapping

        # Mutual classes are meaningless with one person in frame: a lone
        # person shadow-boxing trips 'punching', and someone reaching into
        # their own jacket trips 'touch other person's pocket'.
        n_people = people_count if people_count is not None else len(persons)
        if needs_two and cfg.violence_requires_two_people and n_people < 2:
            return []

        score = min(1.0, base * (0.5 + conf / 2))
        severity = CRIME_SEVERITY.get(etype, "medium")
        # A low-confidence critical class is still worth surfacing, but not as
        # a critical.
        if severity == "critical" and score < 0.7:
            severity = "high"

        return [self._make_event(
            etype=etype,
            label=CRIME_LABELS.get(etype, "Suspicious action"),
            category=category,
            severity=severity,
            score=score, rule="action_recognition",
            detail=(f"Action '{label}' at {conf:.0%} confidence with "
                    f"{n_people} people in frame"),
            evidence={"action": label, "confidence": conf,
                      "person_count": n_people,
                      "detector_person_count": len(persons)},
            now=now,
        )]

    def _rule_distress(self, action: dict | None, now: float) -> list[dict]:
        cfg = self.config
        best: tuple[float, str] | None = None
        for lbl, conf in action_candidates(action):
            if lbl in DISTRESS_ACTIONS and conf >= cfg.action_conf:
                if best is None or conf > best[0]:
                    best = (conf, lbl)
        if best is None:
            return []
        conf, label = best
        score = min(1.0, DISTRESS_ACTIONS[label] * (0.5 + conf / 2))
        return [self._make_event(
            etype="person_down", label="Person in distress", category="medical",
            severity="high" if label in ("falling", "staggering") else "medium",
            score=score, rule="action_recognition",
            detail=f"Action '{label}' at {conf:.0%} confidence",
            evidence={"action": label, "confidence": conf},
            now=now,
        )]

    def _rule_crowd(self, now: float) -> list[dict]:
        cfg = self.config
        if len(self._person_history) < 5:
            return []
        window = [(t, n) for t, n in self._person_history
                  if now - t <= cfg.crowd_surge_window_s]
        if len(window) < 3:
            return []
        counts = [n for _, n in window]
        first, last, peak = counts[0], counts[-1], max(counts)
        out = []

        if last - first >= cfg.crowd_surge_delta:
            out.append(self._make_event(
                etype="crowd_surge", label="Sudden crowd build-up", category="crowd",
                severity="medium", score=0.55, rule="crowd_dynamics",
                detail=f"Person count rose {first} → {last} in {cfg.crowd_surge_window_s:.0f}s",
                evidence={"from": first, "to": last}, now=now,
            ))
        if peak - last >= cfg.crowd_dispersal_delta and peak >= cfg.crowd_surge_delta:
            # People leaving a scene fast is a classic post-incident signature.
            out.append(self._make_event(
                etype="crowd_dispersal", label="Sudden dispersal", category="crowd",
                severity="high", score=0.60, rule="crowd_dynamics",
                detail=f"Person count fell {peak} → {last} in {cfg.crowd_surge_window_s:.0f}s",
                evidence={"from": peak, "to": last}, now=now,
            ))
        if last >= cfg.crowd_person_threshold:
            out.append(self._make_event(
                etype="crowd_density", label="High crowd density", category="crowd",
                severity="low", score=0.35, rule="crowd_dynamics",
                detail=f"{last} people in frame",
                evidence={"count": last}, now=now,
            ))
        return out

    def _rule_zones(self, persons: list[dict], now: float) -> list[dict]:
        cfg = self.config
        if not cfg.zones or not persons:
            return []
        w, h = self._frame_wh
        out = []
        for zone in cfg.zones:
            poly = zone.get("polygon") or []
            if len(poly) < 3:
                continue
            abs_poly = [[p[0] * w, p[1] * h] for p in poly]
            for p in persons:
                x1, y1, x2, y2 = p["box"]
                foot = ((x1 + x2) / 2.0, y2)   # feet, not centre — people stand on the ground
                if _point_in_polygon(foot, abs_poly):
                    out.append(self._make_event(
                        etype="zone_intrusion",
                        label=f"Intrusion: {zone.get('name', 'restricted zone')}",
                        category="perimeter",
                        severity=zone.get("severity", cfg.zone_severity),
                        score=0.65, rule="zone_intrusion",
                        detail=f"Person inside '{zone.get('name', 'restricted zone')}'",
                        evidence={"zone": zone.get("name"), "box": p["box"]},
                        now=now,
                    ))
                    break
        return out

    def _rule_loitering(self, now: float) -> list[dict]:
        cfg = self.config
        out = []
        for t in self.tracker.by_class("person"):
            if "loiter" in t.fired or t.dwell < cfg.loiter_seconds:
                continue
            if not t.history:
                continue
            spread = max(
                (_dist(c, t.anchor) for _, c in t.history), default=0.0
            ) / self._frame_diag
            if spread > cfg.loiter_motion_tolerance:
                continue
            t.fired.add("loiter")
            out.append(self._make_event(
                etype="loitering", label="Loitering", category="behaviour",
                severity="low", score=0.45, rule="dwell_time",
                detail=f"Person stationary for {t.dwell:.0f}s",
                evidence={"track_id": t.track_id, "dwell_s": round(t.dwell, 1)},
                now=now,
            ))
        return out

    def _rule_unattended(self, now: float) -> list[dict]:
        cfg = self.config
        persons = self.tracker.by_class("person")
        out = []
        for cls in ABANDONABLE_CLASSES:
            for t in self.tracker.by_class(cls):
                if "unattended" in t.fired or t.dwell < cfg.unattended_seconds:
                    continue
                nearest = min(
                    (_dist(t.centroid, p.centroid) / self._frame_diag for p in persons),
                    default=float("inf"),
                )
                if nearest <= cfg.unattended_owner_distance:
                    continue
                t.fired.add("unattended")
                out.append(self._make_event(
                    etype="unattended_object", label=f"Unattended {cls}",
                    category="object", severity="medium", score=0.55,
                    rule="unattended_object",
                    detail=f"{cls} stationary {t.dwell:.0f}s with no person within "
                           f"{cfg.unattended_owner_distance:.0%} of frame diagonal",
                    evidence={"track_id": t.track_id, "class": cls,
                              "dwell_s": round(t.dwell, 1)},
                    now=now,
                ))
        return out

    def _rule_after_hours(self, persons: list[dict], now: float) -> list[dict]:
        cfg = self.config
        if not cfg.after_hours_enabled or not persons:
            return []
        hour = datetime.now().hour
        start, end = cfg.after_hours_start, cfg.after_hours_end
        active = (hour >= start or hour < end) if start > end else (start <= hour < end)
        if not active:
            return []
        return [self._make_event(
            etype="after_hours_presence", label="After-hours presence",
            category="perimeter", severity="medium", score=0.50,
            rule="after_hours",
            detail=f"{len(persons)} person(s) detected at {hour:02d}:00",
            evidence={"count": len(persons), "hour": hour}, now=now,
        )]

    # ── plumbing ──────────────────────────────────────────────────────────

    def _corroborates(self, etype: str) -> bool:
        """Does a structured signal support this caption-derived event type?"""
        sig = self._last_signals
        dets = set(sig.get("detections") or [])
        actions = set(sig.get("actions") or [])
        people = sig.get("person_count", 0)

        if etype in ("weapon_reported", "shooting_reported"):
            return bool(dets & set(WEAPON_CLASSES))
        if etype in ("violence_reported", "robbery_reported"):
            return bool(actions & set(VIOLENT_ACTIONS)) or people >= 2
        if etype == "person_down_reported":
            return bool(actions & set(DISTRESS_ACTIONS))
        if etype in ("theft_reported", "burglary_reported"):
            # A skeleton-level theft class is far stronger corroboration than
            # "two people are present", so check it first.
            if actions & set(THEFT_ACTIONS):
                return True
            return people >= 1 and bool(dets & ABANDONABLE_CLASSES or people >= 2)
        if etype in ("crowd_panic_reported", "pursuit_reported"):
            return people >= 3
        if etype == "collision_reported":
            return bool(dets & VEHICLE_CLASSES)
        return people >= 1

    def _make_event(self, *, etype: str, label: str, category: str, severity: str,
                    score: float, rule: str, detail: str, evidence: dict,
                    now: float, subject: dict | None = None,
                    cooldown_key: str | None = None) -> dict:
        """Mint an event, stamped with both wall-clock and video position.

        `subject` names the object the event is *about* — the vehicle that
        struck something, the person holding a weapon — so the caller can
        commit it to the identity registry without re-deriving which box
        mattered. `cooldown_key` overrides per-type debounce for rules where
        two simultaneous incidents are genuinely distinct (two separate
        collisions must not silence each other).
        """
        return {
            "id": uuid.uuid4().hex[:12],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "monotonic": now,
            "video_time_s": (round(self._video_time_s, 3)
                             if self._video_time_s is not None else None),
            "timecode": _format_timecode(self._video_time_s),
            "frame": self._frame_index,
            "source": self._source,
            "type": etype,
            "label": label,
            "category": category,
            "severity": severity,
            "score": round(float(max(0.0, min(1.0, score))), 3),
            "rule": rule,
            "detail": detail,
            "evidence": evidence,
            "subject": subject,
            "snapshot": None,
            "acknowledged": False,
            "_cooldown_key": cooldown_key or etype,
        }

    def _finalise(self, events: list[dict], now: float) -> list[dict]:
        """Apply score floor, severity floor and per-type cooldown."""
        cfg = self.config
        kept = []
        for e in events:
            if e["score"] < cfg.min_score:
                continue
            if not severity_at_least(e["severity"], cfg.min_severity):
                continue
            key = e.pop("_cooldown_key", e["type"])
            last = self._last_fired.get(key, -1e9)
            if now - last < cfg.cooldown_s:
                continue
            self._last_fired[key] = now
            kept.append(e)

        if kept:
            self.events.extend(kept)
            if len(self.events) > self.MAX_EVENTS:
                del self.events[: len(self.events) - self.MAX_EVENTS]
        return kept


# ── CALIBRATION ───────────────────────────────────────────────────────────
#
# Read this before trusting any output.
#
# 1. Firearms are not detectable with the stock model. COCO has no gun class.
#    `weapon_visible` covers knife / bat / scissors only, and those three are
#    among YOLO's noisiest classes at distance. For real weapon detection,
#    train or source a weapon-specific YOLO model and point SENTINEL_WEAPON_MODEL
#    at it (see backend/weapons.py). Until then Tier 2 (caption) is your only
#    gun signal, and a 0.5B VLM is not reliable at it.
#
# 2. NTU was recorded indoors, close range, with actors facing the camera and
#    performing on cue. Its classes transfer imperfectly to overhead CCTV or
#    handheld street footage. Expect both misses and false alarms;
#    `violence_requires_two_people` removes the worst of the false alarms but
#    not all. Fine-tuning on your own footage (see training/README.md) is what
#    closes this gap — the stock checkpoint is a starting point, not a finish.
#
# 3. The theft classes (`pickpocket`, `snatch_theft`) come from NTU A57 and
#    A109. They describe a *body movement* — a hand entering another person's
#    pocket, or a grab followed by separation. They do not observe ownership,
#    consent, or intent, and cannot. A parent taking a phone from a child's
#    hand and a thief snatching one produce the same skeleton. NTU itself
#    separates A57 from A109 largely by speed.
#
#    So: these fire on a gesture that is *consistent with* theft. That is a
#    reason to review four seconds of video. It is not a finding, and must
#    never be presented to anyone as evidence that a crime occurred.
#
# 4. Every threshold above is a precision/recall trade-off, not a fact. Tune
#    them against recorded footage from your own camera, and measure. A system
#    nobody has measured is a system that mostly generates false alarms and
#    trains its operators to ignore it.
#
# 5. Demographic bias: detection and pose models have documented accuracy
#    differences across skin tone, body size and clothing. Rules built on top
#    inherit that. Do not route these alerts to automated action; they are a
#    prioritisation aid for a human reviewer.
