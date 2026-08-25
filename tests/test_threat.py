"""Synthetic tests for the Sentinel threat engine.

Run:  python -m pytest tests/test_threat.py -v
      (or)  python tests/test_threat.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.threat import ThreatEngine, ThreatConfig, severity_at_least  # noqa: E402

SHAPE = (720, 1280, 3)


def person(x=100, y=100, w=80, h=200, conf=0.9):
    return {"class": "person", "confidence": conf, "box": [x, y, x + w, y + h]}


def obj(cls, x=100, y=100, w=40, h=40, conf=0.8):
    return {"class": cls, "confidence": conf, "box": [x, y, x + w, y + h]}


def engine(**cfg):
    return ThreatEngine(ThreatConfig(**cfg))


# ── Weapon rules ──────────────────────────────────────────────────────────

def test_knife_near_person_is_critical():
    e = engine()
    events = e.update(detections=[person(100, 100), obj("knife", 150, 150)],
                      frame_shape=SHAPE, now=1000.0)
    assert len(events) == 1
    assert events[0]["type"] == "weapon_brandished"
    assert events[0]["severity"] == "critical"


def test_knife_far_from_person_is_lower_severity():
    e = engine()
    events = e.update(detections=[person(50, 50), obj("knife", 1200, 650)],
                      frame_shape=SHAPE, now=1000.0)
    assert len(events) == 1
    assert events[0]["type"] == "weapon_visible"
    assert events[0]["severity"] == "high"


def test_low_confidence_weapon_is_ignored():
    e = engine(weapon_conf=0.6)
    events = e.update(detections=[person(100, 100), obj("knife", 150, 150, conf=0.3)],
                      frame_shape=SHAPE, now=1000.0)
    assert events == []


# ── Action rules ──────────────────────────────────────────────────────────

def test_violence_requires_two_people():
    e = engine()
    action = {"action": "punching/slapping", "confidence": 0.9}
    solo = e.update(detections=[person()], action=action, frame_shape=SHAPE, now=1000.0)
    assert solo == [], "single person should not trigger interpersonal violence"

    pair = e.update(detections=[person(100), person(400)], action=action,
                    frame_shape=SHAPE, now=1100.0)
    assert any(ev["type"] == "violence" for ev in pair)


def test_falling_raises_person_down():
    e = engine()
    events = e.update(detections=[person()],
                      action={"action": "falling", "confidence": 0.8},
                      frame_shape=SHAPE, now=1000.0)
    assert any(ev["type"] == "person_down" and ev["severity"] == "high"
               for ev in events)


def test_benign_action_raises_nothing():
    e = engine()
    events = e.update(detections=[person(), person(400)],
                      action={"action": "hand waving", "confidence": 0.95},
                      frame_shape=SHAPE, now=1000.0)
    assert events == []


# ── Cooldown / debounce ───────────────────────────────────────────────────

def test_cooldown_suppresses_repeat_alerts():
    e = engine(cooldown_s=10.0)
    dets = [person(100, 100), obj("knife", 150, 150)]
    first = e.update(detections=dets, frame_shape=SHAPE, now=1000.0)
    second = e.update(detections=dets, frame_shape=SHAPE, now=1002.0)
    third = e.update(detections=dets, frame_shape=SHAPE, now=1015.0)
    assert len(first) == 1 and second == [] and len(third) == 1


# ── Zones ─────────────────────────────────────────────────────────────────

def test_zone_intrusion_uses_feet_not_centre():
    zone = {"name": "platform edge",
            "polygon": [[0.0, 0.8], [1.0, 0.8], [1.0, 1.0], [0.0, 1.0]]}
    e = engine(zones=[zone])
    # Person whose centre is above the zone but whose feet are inside it.
    inside = e.update(detections=[person(600, 400, 80, 250)],
                      frame_shape=SHAPE, now=1000.0)
    assert any(ev["type"] == "zone_intrusion" for ev in inside)

    e2 = engine(zones=[zone])
    outside = e2.update(detections=[person(600, 50, 80, 200)],
                        frame_shape=SHAPE, now=1000.0)
    assert outside == []


# ── Dwell-based rules ─────────────────────────────────────────────────────

def test_loitering_fires_after_threshold_and_only_once():
    e = engine(loiter_seconds=30.0, cooldown_s=0.0)
    fired = []
    for i in range(80):
        t = 1000.0 + i
        fired += e.update(detections=[person(300, 300)], frame_shape=SHAPE, now=t)
    loiters = [f for f in fired if f["type"] == "loitering"]
    assert len(loiters) == 1, f"expected one loitering event, got {len(loiters)}"


def test_moving_person_does_not_loiter():
    e = engine(loiter_seconds=30.0, cooldown_s=0.0)
    fired = []
    for i in range(80):
        fired += e.update(detections=[person(100 + i * 12, 300)],
                          frame_shape=SHAPE, now=1000.0 + i)
    assert not any(f["type"] == "loitering" for f in fired)


def test_unattended_bag_requires_no_nearby_person():
    e = engine(unattended_seconds=20.0, cooldown_s=0.0)
    fired = []
    # Bag alone in frame for 40s.
    for i in range(40):
        fired += e.update(detections=[obj("backpack", 800, 500)],
                          frame_shape=SHAPE, now=1000.0 + i)
    assert any(f["type"] == "unattended_object" for f in fired)

    e2 = engine(unattended_seconds=20.0, cooldown_s=0.0)
    fired2 = []
    for i in range(40):
        fired2 += e2.update(
            detections=[obj("backpack", 800, 500), person(790, 480)],
            frame_shape=SHAPE, now=1000.0 + i)
    assert not any(f["type"] == "unattended_object" for f in fired2), \
        "bag with owner standing next to it is not unattended"


# ── Crowd dynamics ────────────────────────────────────────────────────────

def test_crowd_surge_and_dispersal():
    e = engine(crowd_surge_delta=4, crowd_dispersal_delta=4, cooldown_s=0.0)
    for i in range(4):
        e.update(detections=[person(i * 100)], frame_shape=SHAPE, now=1000.0 + i * 0.5)
    surge = e.update(detections=[person(i * 90) for i in range(9)],
                     frame_shape=SHAPE, now=1002.5)
    assert any(ev["type"] == "crowd_surge" for ev in surge)

    disperse = e.update(detections=[person(10)], frame_shape=SHAPE, now=1003.5)
    assert any(ev["type"] == "crowd_dispersal" for ev in disperse)


# ── Tier 2: caption screening ─────────────────────────────────────────────

def test_caption_flags_open_vocabulary_crime():
    e = engine()
    e.update(detections=[person(), person(400)], frame_shape=SHAPE, now=1000.0)
    events = e.ingest_caption("Two men are fighting near the entrance.", now=1000.0)
    assert any(ev["type"] == "violence_reported" for ev in events)


def test_caption_flags_burglary_tools():
    """A smash-and-grab is not vandalism.

    `weapons.py::CRIME_TOOL_CLASSES` scores hammers and crowbars as
    improvised/burglary tools for the object detector, but the caption lexicon
    had no entry for any of them, so this text matched only `smash` and came
    out as vandalism at medium.
    """
    for caption in ("Three men with sledgehammers at the jewelry counter.",
                    "A group smashing the display case with a hammer.",
                    "A smash-and-grab robbery in progress.",
                    "A man using a crowbar on the door."):
        e = engine()
        e.update(detections=[person(), person(400)], frame_shape=SHAPE, now=1000.0)
        events = e.ingest_caption(caption, now=1000.0)
        assert any(ev["type"] == "burglary_reported" and ev["severity"] == "high"
                   for ev in events), caption


def test_caption_burglary_tools_respect_negation():
    e = engine()
    e.update(detections=[person()], frame_shape=SHAPE, now=1000.0)
    assert not e.ingest_caption("There is no hammer, crowbar or weapon visible.",
                                now=1000.0)


def test_caption_negation_is_suppressed():
    e = engine()
    e.update(detections=[person()], frame_shape=SHAPE, now=1000.0)
    events = e.ingest_caption("No incident observed. There is no weapon visible.",
                              now=1000.0)
    assert events == [], f"negated caption should not alert, got {events}"


def test_caption_corroboration_raises_score():
    with_weapon = engine()
    with_weapon.update(detections=[person(100, 100), obj("knife", 150, 150)],
                       frame_shape=SHAPE, now=1000.0)
    a = with_weapon.ingest_caption("A man is holding a knife.", now=1000.0)

    without = engine()
    without.update(detections=[person()], frame_shape=SHAPE, now=1000.0)
    b = without.ingest_caption("A man is holding a knife.", now=1000.0)

    assert a and b
    assert a[0]["score"] > b[0]["score"], \
        "caption backed by a detection should score higher"


def test_clean_caption_is_silent():
    e = engine()
    e.update(detections=[person()], frame_shape=SHAPE, now=1000.0)
    assert e.ingest_caption("A person walks along the pavement carrying a bag.",
                            now=1000.0) == []


# ── Config / filtering ────────────────────────────────────────────────────

def test_min_severity_filter():
    e = engine(min_severity="high", cooldown_s=0.0)
    fired = []
    for i in range(80):
        fired += e.update(detections=[person(300, 300)], frame_shape=SHAPE,
                          now=1000.0 + i)
    assert not any(f["type"] == "loitering" for f in fired), \
        "low-severity loitering should be filtered out at min_severity=high"


def test_disabled_engine_is_silent():
    e = engine(enabled=False)
    assert e.update(detections=[person(100, 100), obj("knife", 150, 150)],
                    frame_shape=SHAPE, now=1000.0) == []


def test_config_update_is_live():
    e = engine(min_severity="critical")
    e.config.update({"min_severity": "low"})
    assert e.config.min_severity == "low"


def test_severity_ordering():
    assert severity_at_least("critical", "high")
    assert not severity_at_least("low", "high")


def test_stats_shape():
    e = engine()
    e.update(detections=[person(100, 100), obj("knife", 150, 150)],
             frame_shape=SHAPE, now=1000.0)
    s = e.stats()
    assert s["total"] == 1
    assert s["by_severity"]["critical"] == 1


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    passed = failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception:
            print(f"  FAIL  {name}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
