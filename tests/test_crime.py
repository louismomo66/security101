"""Synthetic tests for the crime-analysis layer.

Covers grip reasoning, collision dynamics, video timecodes, appearance
re-identification and report construction. No models, no video files — every
signal is fabricated so the rules are tested rather than the detector.

Run:  python -m pytest tests/test_crime.py -v
      (or)  python tests/test_crime.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.threat import ThreatEngine, ThreatConfig  # noqa: E402
from backend.weapons import (  # noqa: E402
    held_objects, carried_objects, hand_points, tool_info, crop_person,
    verify_held_object,
)
from backend.identity import (  # noqa: E402
    IdentityRegistry, appearance_signature, aspect_of, describe_colour,
    format_timecode, kind_of, signature_similarity,
)
from backend.report import build_report, cluster_events, render_markdown  # noqa: E402

SHAPE = (720, 1280, 3)
DIAG = float(np.hypot(1280, 720))


# ── fixtures ──────────────────────────────────────────────────────────────

def person(x=100, y=100, w=80, h=200, conf=0.9):
    return {"class": "person", "confidence": conf, "box": [x, y, x + w, y + h]}


def obj(cls, x=100, y=100, w=40, h=40, conf=0.8):
    return {"class": cls, "confidence": conf, "box": [x, y, x + w, y + h]}


def vehicle(cls="car", x=100, y=300, w=140, h=90, conf=0.9):
    return {"class": cls, "confidence": conf, "box": [x, y, x + w, y + h]}


def pose_person(box, left_wrist=None, right_wrist=None, kp_conf=0.9):
    """A pose entry with every keypoint low-confidence except the given wrists."""
    kps = [[0.0, 0.0, 0.0] for _ in range(17)]
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    kps[5] = [cx - 20, cy - 40, kp_conf]     # shoulders, so the skeleton is plausible
    kps[6] = [cx + 20, cy - 40, kp_conf]
    if left_wrist:
        kps[9] = [left_wrist[0], left_wrist[1], kp_conf]
    if right_wrist:
        kps[10] = [right_wrist[0], right_wrist[1], kp_conf]
    return {"confidence": 0.9, "box": list(box), "keypoints": kps}


def engine(**cfg):
    return ThreatEngine(ThreatConfig(**cfg))


def solid(colour, w=120, h=160):
    """A flat RGB patch — a deterministic stand-in for an object crop."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = colour
    return img


def scene_with(colour, box, size=(720, 1280)):
    frame = np.full((size[0], size[1], 3), 20, dtype=np.uint8)
    x1, y1, x2, y2 = [int(v) for v in box]
    frame[y1:y2, x1:x2] = colour
    return frame


# ── grip geometry ─────────────────────────────────────────────────────────

def test_hand_points_prefers_wrist_over_elbow():
    box = [100, 100, 180, 300]
    p = pose_person(box, right_wrist=(190, 220))
    hands = hand_points(p["keypoints"])
    assert len(hands) == 1
    assert hands[0][0] == "right"
    assert hands[0][1] == (190.0, 220.0)


def test_hand_points_falls_back_to_elbow():
    kps = [[0.0, 0.0, 0.0] for _ in range(17)]
    kps[8] = [150.0, 200.0, 0.8]          # right elbow visible, wrist not
    hands = hand_points(kps)
    assert hands and hands[0][0] == "right (elbow)"


def test_object_at_wrist_is_held():
    pbox = [100, 100, 180, 300]
    p = pose_person(pbox, right_wrist=(190, 220))
    knife = obj("knife", 185, 215, w=20, h=20)
    matches = held_objects([knife], [p])
    assert len(matches) == 1
    assert matches[0]["hand"] == "right"
    assert matches[0]["grip"] > 0.9        # wrist is inside the object box


def test_object_across_the_room_is_not_held():
    pbox = [100, 100, 180, 300]
    p = pose_person(pbox, right_wrist=(190, 220))
    knife = obj("knife", 900, 600, w=20, h=20)
    assert held_objects([knife], [p]) == []


def test_grip_tolerance_scales_with_person_size():
    """The same pixel gap is a grip up close and not a grip far away."""
    gap = 30
    near_box = [100, 100, 300, 600]        # large person (close to camera)
    far_box = [100, 100, 130, 180]         # small person (far away)
    near = pose_person(near_box, right_wrist=(310, 300))
    far = pose_person(far_box, right_wrist=(135, 140))
    near_obj = obj("knife", 310 + gap, 300, w=10, h=10)
    far_obj = obj("knife", 135 + gap, 140, w=10, h=10)
    assert held_objects([near_obj], [near]), "should hold at close range"
    assert held_objects([far_obj], [far]) == [], "same gap is too far at distance"


def test_carried_object_is_inside_the_box_but_not_at_a_hand():
    pbox = [100, 100, 180, 300]
    tucked = obj("knife", 120, 200, w=20, h=20)     # mid-torso, no wrist there
    matches = carried_objects([tucked], [{"box": pbox}])
    assert len(matches) == 1
    assert matches[0]["containment"] == 1.0


def test_vocabulary_is_separator_insensitive():
    assert tool_info("Baseball Bat")["category"] == "weapon"
    assert tool_info("bolt_cutter")["category"] == "burglary"
    assert tool_info("teapot") is None


# ── weapon rules ──────────────────────────────────────────────────────────

def test_held_weapon_outranks_a_merely_nearby_one():
    pbox = [100, 100, 180, 300]
    p = pose_person(pbox, right_wrist=(190, 220))
    holding = engine().update(
        detections=[person(100, 100, 80, 200), obj("knife", 185, 215, 20, 20)],
        pose_persons=[p], frame_shape=SHAPE, now=1000.0,
    )
    nearby = engine().update(
        detections=[person(100, 100, 80, 200), obj("knife", 240, 260, 20, 20)],
        pose_persons=[p], frame_shape=SHAPE, now=1000.0,
    )
    assert holding[0]["type"] == "weapon_brandished"
    assert holding[0]["severity"] == "critical"
    assert nearby[0]["type"] == "weapon_near_person"
    assert holding[0]["score"] > nearby[0]["score"]


def test_held_weapon_names_the_hand_and_the_person():
    pbox = [100, 100, 180, 300]
    p = pose_person(pbox, right_wrist=(190, 220))
    ev = engine().update(
        detections=[person(100, 100, 80, 200), obj("knife", 185, 215, 20, 20)],
        pose_persons=[p], frame_shape=SHAPE, now=1000.0,
    )[0]
    assert ev["evidence"]["attachment"] == "held"
    assert ev["evidence"]["hand"] == "right"
    assert ev["subject"]["box"] == pbox          # the person, for the registry


def test_burglary_tool_only_fires_when_attached_to_a_person():
    loose = engine().update(detections=[obj("crowbar", 500, 500)],
                            frame_shape=SHAPE, now=1000.0)
    assert loose == [], "a crowbar lying around is scene furniture"

    pbox = [100, 100, 180, 300]
    p = pose_person(pbox, right_wrist=(190, 220))
    held = engine().update(
        detections=[person(100, 100, 80, 200), obj("crowbar", 185, 215, 20, 20)],
        pose_persons=[p], frame_shape=SHAPE, now=1000.0,
    )
    assert [e["type"] for e in held] == ["crime_tool_held"]
    assert held[0]["severity"] == "high"


def test_proximity_fallback_survives_without_pose():
    """The original behaviour must hold when the pose model is unavailable."""
    ev = engine().update(detections=[person(100, 100), obj("knife", 150, 150)],
                         frame_shape=SHAPE, now=1000.0)
    assert ev[0]["type"] == "weapon_brandished"
    assert "no pose data" in ev[0]["detail"]


# ── video clock ───────────────────────────────────────────────────────────

def test_events_carry_a_video_timecode():
    e = engine()
    e.set_clock(video_time_s=107.32, frame=2683, source="clip.mp4")
    ev = e.update(detections=[person(100, 100), obj("knife", 150, 150)],
                  frame_shape=SHAPE, now=107.32)[0]
    assert ev["video_time_s"] == 107.32
    assert ev["timecode"] == "00:01:47.320"
    assert ev["frame"] == 2683
    assert ev["source"] == "clip.mp4"


def test_timecode_formats_hours():
    assert format_timecode(0) == "00:00:00.000"
    assert format_timecode(3661.5) == "01:01:01.500"
    assert format_timecode(None) == "--:--:--"


def test_events_without_a_clock_have_no_timecode():
    ev = engine().update(detections=[person(100, 100), obj("knife", 150, 150)],
                         frame_shape=SHAPE, now=5.0)[0]
    assert ev["video_time_s"] is None
    assert ev["timecode"] == "--:--:--"


# ── collisions ────────────────────────────────────────────────────────────

def _drive(e, frames, dt=0.1, t0=0.0):
    """Feed a scripted sequence of detection lists; return everything raised."""
    events = []
    for i, dets in enumerate(frames):
        events += e.update(detections=dets, frame_shape=SHAPE, now=t0 + i * dt)
    return events


# Boxes are 140 wide, so two vehicles at the same y overlap at IoU >= 0.08 once
# they share ~21px. Approaches therefore end 40px into the other vehicle's box.
_IMPACT_OVERLAP = 40


def _approach_then_stop(striker_cls="car", other_cls="car", steps_before=12,
                        steps_after=15, speed_px=15.0, other_box=(600, 300)):
    """Vehicle A closes on stationary B at `speed_px` per frame, then stops dead."""
    ox, oy = other_box
    other = vehicle(other_cls, ox, oy)
    impact_x = ox - 140 + _IMPACT_OVERLAP
    frames = []
    x = impact_x - (steps_before - 1) * speed_px
    for _ in range(steps_before):
        frames.append([vehicle(striker_cls, int(round(x)), oy), other])
        x += speed_px
    for _ in range(steps_after):
        frames.append([vehicle(striker_cls, impact_x, oy), other])
    return frames


def test_collision_fires_on_contact_plus_deceleration():
    e = engine()
    events = _drive(e, _approach_then_stop())
    kinds = [ev["type"] for ev in events]
    assert "vehicle_collision" in kinds
    ev = next(ev for ev in events if ev["type"] == "vehicle_collision")
    assert ev["severity"] in ("high", "critical")
    assert ev["subject"]["class"] == "car"
    assert "decelerated" in ev["detail"]


def test_overlap_without_deceleration_is_not_a_collision():
    """A vehicle passing behind another overlaps in projection only."""
    e = engine()
    frames = []
    other = vehicle("car", 600, 300)
    x = 200.0
    for _ in range(30):                      # drives straight through and onward
        frames.append([vehicle("car", int(x), 300), other])
        x += 15.0
    events = _drive(e, frames)
    assert not [ev for ev in events if ev["type"] == "vehicle_collision"]


def test_stationary_vehicles_touching_are_not_a_collision():
    e = engine()
    frames = [[vehicle("car", 600, 300), vehicle("car", 700, 300)] for _ in range(30)]
    events = _drive(e, frames)
    assert not [ev for ev in events if ev["type"].startswith("vehicle_")]


def test_vehicle_pedestrian_collision_is_critical():
    e = engine()
    ped = person(640, 300, w=60, h=180)
    impact_x = 560                          # 45px of the car's box over the person
    frames = []
    x = impact_x - 11 * 15.0
    for _ in range(12):
        frames.append([vehicle("car", int(round(x)), 300), ped])
        x += 15.0
    for _ in range(15):
        frames.append([vehicle("car", impact_x, 300), ped])
    events = _drive(e, frames)
    hit = [ev for ev in events if ev["type"] == "vehicle_pedestrian_collision"]
    assert hit, [ev["type"] for ev in events]
    assert hit[0]["severity"] == "critical"


def test_vehicle_leaving_after_a_collision_is_flagged():
    e = engine()
    frames = _approach_then_stop(steps_after=15)
    other = vehicle("car", 600, 300)

    # Reverse away from the impact, then leave the frame entirely.
    x = float(600 - 140 + _IMPACT_OVERLAP)
    for _ in range(8):
        x -= 22.0
        frames.append([vehicle("car", int(round(x)), 300), other])
    for _ in range(25):                      # gone; the tracker ages it out
        frames.append([other])

    events = _drive(e, frames)
    assert [ev for ev in events if ev["type"] == "vehicle_collision"], \
        "precondition: the collision itself must fire"
    left = [ev for ev in events if ev["type"] == "vehicle_left_scene"]
    assert left, [ev["type"] for ev in events]
    assert left[0]["severity"] == "high"
    assert left[0]["subject"]["class"] == "car"


def test_vehicle_rules_can_be_disabled():
    e = engine(vehicle_rules=False)
    events = _drive(e, _approach_then_stop())
    assert not [ev for ev in events if ev["type"].startswith("vehicle_")]


def test_two_collisions_are_not_debounced_into_one():
    """Per-pair cooldown keys: separate incidents must both be reported."""
    e = engine()
    a_frames = _approach_then_stop(other_box=(600, 200))
    b_frames = _approach_then_stop(other_box=(600, 500))
    merged = [a + b for a, b in zip(a_frames, b_frames)]
    events = _drive(e, merged)
    collisions = [ev for ev in events if ev["type"] == "vehicle_collision"]
    assert len(collisions) == 2, [c["detail"] for c in collisions]


# ── appearance re-identification ──────────────────────────────────────────

def test_signature_matches_the_same_colour_and_rejects_another():
    red = scene_with((200, 30, 30), (100, 100, 260, 300))
    red2 = scene_with((205, 35, 28), (400, 100, 560, 300))
    blue = scene_with((30, 40, 200), (100, 100, 260, 300))

    s_red = appearance_signature(red, [100, 100, 260, 300])
    s_red2 = appearance_signature(red2, [400, 100, 560, 300])
    s_blue = appearance_signature(blue, [100, 100, 260, 300])

    assert signature_similarity(s_red, s_red2) > 0.9
    assert signature_similarity(s_red, s_blue) < 0.5


def test_signature_rejects_a_degenerate_crop():
    frame = scene_with((200, 30, 30), (100, 100, 260, 300))
    assert appearance_signature(frame, [100, 100, 102, 102]) is None


def test_registry_recognises_a_returning_vehicle():
    reg = IdentityRegistry(path=None, autosave=False)
    frame = scene_with((200, 30, 30), (100, 300, 340, 440))
    box = [100, 300, 340, 440]
    sig = appearance_signature(frame, box)

    ent, _, is_new = reg.observe(kind="vehicle", cls="car", signature=sig,
                                 aspect=aspect_of(box), label="red car")
    assert is_new

    later = scene_with((198, 33, 34), (700, 300, 940, 440))
    box2 = [700, 300, 940, 440]
    same, score, is_new2 = reg.observe(
        kind="vehicle", cls="car", signature=appearance_signature(later, box2),
        aspect=aspect_of(box2), label="red car",
    )
    assert not is_new2
    assert same.entity_id == ent.entity_id
    assert score > reg.match_threshold


def test_registry_keeps_a_different_vehicle_separate():
    reg = IdentityRegistry(path=None, autosave=False)
    box = [100, 300, 340, 440]
    red = appearance_signature(scene_with((200, 30, 30), box), box)
    blue = appearance_signature(scene_with((30, 40, 200), box), box)

    a, _, _ = reg.observe(kind="vehicle", cls="car", signature=red, aspect=aspect_of(box))
    b, _, is_new = reg.observe(kind="vehicle", cls="car", signature=blue, aspect=aspect_of(box))
    assert is_new and a.entity_id != b.entity_id


def test_registry_never_matches_across_kinds():
    reg = IdentityRegistry(path=None, autosave=False)
    box = [100, 300, 340, 440]
    sig = appearance_signature(scene_with((200, 30, 30), box), box)
    reg.observe(kind="vehicle", cls="car", signature=sig, aspect=aspect_of(box))
    _, _, is_new = reg.observe(kind="person", cls="person", signature=sig,
                               aspect=aspect_of(box))
    assert is_new


def test_incident_link_and_persistence_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "registry.json")
        reg = IdentityRegistry(path=path)
        box = [100, 300, 340, 440]
        sig = appearance_signature(scene_with((200, 30, 30), box), box)
        ent, _, _ = reg.observe(kind="vehicle", cls="car", signature=sig,
                                aspect=aspect_of(box), label="red car")
        reg.link_incident(ent.entity_id, {
            "id": "ev1", "type": "vehicle_collision", "label": "Vehicle collision",
            "severity": "high", "score": 0.7, "timecode": "00:00:12.000",
            "video_time_s": 12.0, "source": "clip.mp4",
        }, role="striker")

        reloaded = IdentityRegistry(path=path)
        got = reloaded.get(ent.entity_id)
        assert got is not None
        assert got.incidents[0]["type"] == "vehicle_collision"
        assert got.incidents[0]["timecode"] == "00:00:12.000"
        assert got.incidents[0]["role"] == "striker"


def test_forget_erases_a_subject():
    reg = IdentityRegistry(path=None, autosave=False)
    box = [100, 300, 340, 440]
    sig = appearance_signature(scene_with((200, 30, 30), box), box)
    ent, _, _ = reg.observe(kind="vehicle", cls="car", signature=sig,
                            aspect=aspect_of(box))
    assert reg.forget(ent.entity_id)
    assert reg.get(ent.entity_id) is None
    assert not reg.forget(ent.entity_id)


def test_kind_and_colour_helpers():
    assert kind_of("truck") == "vehicle"
    assert kind_of("person") == "person"
    assert kind_of("banana") is None
    frame = scene_with((240, 240, 240), (100, 100, 300, 300))
    assert describe_colour(frame, [100, 100, 300, 300]) == "white"


# ── VLM verification parsing ──────────────────────────────────────────────

def test_verification_accepts_a_named_item():
    frame = scene_with((120, 120, 120), (100, 100, 300, 400))
    out = verify_held_object(frame, [100, 100, 300, 400],
                             vlm_fn=lambda *a, **k: {"text": "YES: a handgun"})
    assert out["verdict"] == "yes" and out["item"] == "handgun"


def test_verification_rejects_a_yes_with_no_item():
    """Small VLMs agree with leading questions; a bare 'yes' proves nothing."""
    frame = scene_with((120, 120, 120), (100, 100, 300, 400))
    out = verify_held_object(frame, [100, 100, 300, 400],
                             vlm_fn=lambda *a, **k: {"text": "YES: something"})
    assert out["verdict"] == "unclear"


def test_verification_handles_no_and_unclear():
    frame = scene_with((120, 120, 120), (100, 100, 300, 400))
    no = verify_held_object(frame, [100, 100, 300, 400],
                            vlm_fn=lambda *a, **k: {"text": "NO"})
    unclear = verify_held_object(frame, [100, 100, 300, 400],
                                 vlm_fn=lambda *a, **k: {"text": "UNCLEAR"})
    assert no["verdict"] == "no"
    assert unclear["verdict"] == "unclear"


def test_verification_rejects_an_echo_of_the_question():
    """FastVLM-0.5B restates closed questions; the echo contains 'weapon'."""
    frame = scene_with((120, 120, 120), (100, 100, 300, 400))
    echo = ("To determine whether the person in the image is holding a weapon, "
            "we need to examine their hands closely.")
    out = verify_held_object(frame, [100, 100, 300, 400],
                             vlm_fn=lambda *a, **k: {"text": echo})
    assert out["verdict"] == "unclear", out


def test_verification_rejects_long_commentary():
    frame = scene_with((120, 120, 120), (100, 100, 300, 400))
    rambling = ("The person in the image appears to be walking away from the "
                "camera and there might possibly be some kind of stick or "
                "similar object visible near their side.")
    out = verify_held_object(frame, [100, 100, 300, 400],
                             vlm_fn=lambda *a, **k: {"text": rambling})
    assert out["verdict"] == "unclear"


def test_verification_still_accepts_a_terse_unformatted_answer():
    frame = scene_with((120, 120, 120), (100, 100, 300, 400))
    out = verify_held_object(frame, [100, 100, 300, 400],
                             vlm_fn=lambda *a, **k: {"text": "a machete"})
    assert out["verdict"] == "yes" and out["item"] == "machete"


def test_panga_is_in_the_caption_lexicon():
    """Regional vocabulary: 'panga' is what East African footage gets captioned."""
    e = engine()
    e.update(detections=[person()], frame_shape=SHAPE, now=1000.0)
    events = e.ingest_caption("A man is carrying a panga in his right hand.",
                              now=1000.0)
    assert [ev["type"] for ev in events] == ["weapon_reported"]
    assert events[0]["severity"] == "critical"


def test_verification_survives_a_broken_vlm():
    frame = scene_with((120, 120, 120), (100, 100, 300, 400))

    def boom(*a, **k):
        raise RuntimeError("model not loaded")

    out = verify_held_object(frame, [100, 100, 300, 400], vlm_fn=boom)
    assert out["verdict"] == "unavailable"


def test_crop_person_upscales_small_subjects():
    frame = scene_with((120, 120, 120), (100, 100, 140, 160))
    crop = crop_person(frame, [100, 100, 140, 160])
    assert crop is not None and max(crop.shape[:2]) >= 224


# ── caption screening ─────────────────────────────────────────────────────

# The exact caption FastVLM-0.5B produced on the sample night footage. Before
# the sentence-scoped negation fix this one sentence raised four alerts.
_DENIAL_LIST = (
    "The image depicts a security camera frame capturing a residential area at "
    "night. There is no visible weapon, fighting, theft, forced entry, "
    "vandalism, fire, or a person who appears injured or unconscious."
)


def test_denial_list_raises_nothing():
    e = engine()
    e.update(detections=[person()], frame_shape=SHAPE, now=1000.0)
    events = e.ingest_caption(_DENIAL_LIST, now=1000.0)
    assert events == [], [ev["type"] for ev in events]


def test_negation_reaches_past_the_start_of_a_long_list():
    """The item furthest from the cue is the one a fixed lookbehind misses."""
    e = engine()
    e.update(detections=[person()], frame_shape=SHAPE, now=1000.0)
    text = ("There is no weapon, fighting, theft, forced entry, vandalism, or "
            "fire in this scene.")
    assert e.ingest_caption(text, now=1000.0) == []


def test_contrast_conjunction_ends_the_negation():
    e = engine()
    e.update(detections=[person(100, 100), person(300, 100)],
             frame_shape=SHAPE, now=1000.0)
    events = e.ingest_caption(
        "There is no fence or gate, but two men are fighting in the street.",
        now=1000.0,
    )
    assert [ev["type"] for ev in events] == ["violence_reported"]


def test_negation_does_not_leak_across_sentences():
    e = engine()
    e.update(detections=[person(100, 100), person(300, 100)],
             frame_shape=SHAPE, now=1000.0)
    events = e.ingest_caption("There is no car. A man is punching another man.",
                              now=1000.0)
    assert [ev["type"] for ev in events] == ["violence_reported"]


def test_incident_line_none_short_circuits():
    e = engine()
    e.update(detections=[person()], frame_shape=SHAPE, now=1000.0)
    caption = ("Two people walk past a parked car.\nINCIDENT: NONE")
    assert e.ingest_caption(caption, now=1000.0) == []


def test_only_the_incident_line_is_screened():
    """A description mentioning a knife block must not fire; the verdict rules."""
    e = engine()
    e.update(detections=[person()], frame_shape=SHAPE, now=1000.0)
    caption = ("A man stands in a kitchen beside a knife block.\n"
               "INCIDENT: NONE")
    assert e.ingest_caption(caption, now=1000.0) == []


def test_incident_line_reports_a_real_finding():
    e = engine()
    e.update(detections=[person(100, 100), person(300, 100)],
             frame_shape=SHAPE, now=1000.0)
    caption = ("Two men face each other on the pavement.\n"
               "INCIDENT: one man is punching the other")
    events = e.ingest_caption(caption, now=1000.0)
    assert [ev["type"] for ev in events] == ["violence_reported"]


def test_threat_prompt_asks_for_a_verdict_line():
    assert "INCIDENT:" in engine().threat_prompt()


# ── report ────────────────────────────────────────────────────────────────

def _event(etype, t, severity="high", score=0.7, label=None, **extra):
    return {
        "id": f"e{int(t * 1000)}", "type": etype, "label": label or etype,
        "category": "test", "severity": severity, "score": score,
        "rule": "test", "detail": f"{etype} at {t}", "evidence": {},
        "video_time_s": t, "timecode": format_timecode(t), "frame": int(t * 25),
        "monotonic": t, "snapshot": None, "timestamp": "2026-01-01T00:00:00Z",
        **extra,
    }


def test_repeats_collapse_into_one_incident():
    events = [_event("loitering", t, severity="low", score=0.45)
              for t in (10.0, 15.0, 20.0, 25.0)]
    clusters = cluster_events(events, gap_s=20.0)
    assert len(clusters) == 1
    assert clusters[0]["count"] == 4
    assert clusters[0]["first_timecode"] == "00:00:10.000"
    assert clusters[0]["last_timecode"] == "00:00:25.000"


def test_a_long_gap_starts_a_new_incident():
    events = [_event("loitering", 10.0), _event("loitering", 400.0)]
    assert len(cluster_events(events, gap_s=20.0)) == 2


def test_clusters_take_the_highest_severity_seen():
    events = [_event("violence", 10.0, severity="high"),
              _event("violence", 12.0, severity="critical")]
    clusters = cluster_events(events, gap_s=20.0)
    assert len(clusters) == 1 and clusters[0]["severity"] == "critical"


class _FakeJob:
    job_id = "job_test"
    video_name = "clip.mp4"
    video_path = "/tmp/clip.mp4"
    status = "completed"
    options = None
    video = {"duration_s": 300.0, "fps": 25.0, "width": 1280, "height": 720}
    progress = {"frames_processed": 2500}
    captions = [{"video_time_s": 12.0, "timecode": "00:00:12.000",
                 "text": "Two people are arguing near a car."}]
    entities = [{
        "entity_id": "veh_abc123", "kind": "vehicle", "cls": "car",
        "label": "white car", "plate": None,
        "first_seen": "", "last_seen": "", "sighting_count": 4,
        "incident_count": 2, "notes": [],
        "sightings": [{"source": "clip.mp4", "timecode": "00:01:47.320",
                       "video_time_s": 107.32, "frame": 2683, "snapshot": None}],
        "incidents": [
            {"type": "vehicle_collision", "timecode": "00:01:47.320",
             "severity": "high", "role": "striker", "source": "clip.mp4"},
            {"type": "vehicle_left_scene", "timecode": "00:01:52.000",
             "severity": "high", "role": "striker", "source": "clip.mp4"},
        ],
    }]
    events = [
        _event("vehicle_collision", 107.32, severity="high", score=0.74,
               label="Vehicle collision",
               entities=[{"entity_id": "veh_abc123", "label": "white car",
                          "kind": "vehicle", "role": "involved",
                          "recognised": False, "similarity": None,
                          "prior_incidents": 0}]),
        _event("vehicle_left_scene", 112.0, severity="high", score=0.7,
               label="Vehicle left after a collision"),
        _event("weapon_brandished", 45.0, severity="critical", score=0.88,
               label="Knife detected"),
        _event("theft_reported", 60.0, severity="high", score=0.65,
               label="Theft"),
    ]


def test_report_has_timecoded_incidents_and_sections():
    report = build_report(_FakeJob(), narrative=False)
    assert report["counts"]["events"] == 4
    assert report["source"]["video"] == "clip.mp4"

    types = {i["type"]: i for i in report["incidents"]}
    assert types["weapon_brandished"]["first_timecode"] == "00:00:45.000"
    assert types["vehicle_collision"]["first_timecode"] == "00:01:47.320"

    # Highest severity first.
    assert report["incidents"][0]["severity"] == "critical"

    veh = report["vehicles"][0]
    assert veh["involved_in_collision"] and veh["left_scene"]
    assert veh["description"] == "white car"


def test_report_separates_caption_only_signals():
    report = build_report(_FakeJob(), narrative=False)
    tiers = {i["type"]: i["evidence_tier"] for i in report["incidents"]}
    assert tiers["theft_reported"] == "caption"
    assert tiers["weapon_brandished"] == "detector"
    assert any("keyword match" in c for c in report["caveats"])


def test_report_warns_when_no_weapon_model_is_loaded():
    report = build_report(_FakeJob(), narrative=False)
    assert any("firearms cannot be detected" in c.lower() or
               "no weapon-trained detector" in c.lower()
               for c in report["caveats"])


def test_recommended_actions_cite_timecodes():
    report = build_report(_FakeJob(), narrative=False)
    actions = report["recommended_actions"]
    assert actions[0]["priority"] == "critical"
    assert "00:00:45.000" in actions[0]["action"]
    assert all(a["timecode"] for a in actions)


def test_empty_report_is_still_valid():
    class Empty(_FakeJob):
        events: list = []
        entities: list = []
        captions: list = []

    report = build_report(Empty(), narrative=False)
    assert report["incidents"] == []
    assert "No incidents were raised" in report["summary"]["headline"]
    assert render_markdown(report)


def test_markdown_renders_the_timeline():
    md = render_markdown(build_report(_FakeJob(), narrative=False))
    assert "# Incident Report — clip.mp4" in md
    assert "## Incident timeline" in md
    assert "00:01:47.320" in md
    assert "white car" in md
    assert "Confidence and caveats" in md


def test_markdown_survives_a_pipe_in_a_detail_string():
    class Piped(_FakeJob):
        events = [_event("violence", 10.0, severity="high")]

    ev = Piped.events[0]
    ev["detail"] = "a | b | c"
    md = render_markdown(build_report(Piped(), narrative=False))
    assert "a \\| b \\| c" in md


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
