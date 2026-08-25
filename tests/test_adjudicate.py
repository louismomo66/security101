"""Tests for tier-3 VLM adjudication.

The scenarios here are taken from a measured run of FastVLM-0.5B against
`videos/The youngest phone snatcher in my life…mp4`, not invented. The one that
matters is `test_sycophantic_model_yields_no_verdict`: on a frame containing a
single woman alone in a car, the real model answered "yes" to questions that
presuppose a second person, and the pipeline reported pickpocket at 0.84.
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.adjudicate import (
    QUESTIONS, Question, adjudicate_frames, parse_yes_no, sample_indices,
)

FRAMES = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(5)]


def mock(answer_for):
    """Build an `ask` from a callable mapping prompt → answer text."""
    def ask(image, prompt="", max_tokens=12):
        return {"text": answer_for(prompt)}
    return ask


ALWAYS_YES = mock(lambda p: "yes")
ALWAYS_NO = mock(lambda p: "no")


# ── answer parsing ────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("yes", 1.0), ("Yes.", 1.0), ("yes, a phone is visible", 1.0),
    ("no", 0.0), ("No.", 0.0), ("nope", 0.0),
    ("Answer: yes", 1.0),
    ("", None), ("banana", None),
    ("I'm not sure", None),          # hedge, not a "no" despite "not"
    ("it is unclear", None),
    ("yes, but no weapon is visible", None),   # contradictory
    ("maybe", None), ("hard to tell", None),
])
def test_parse_yes_no(text, expected):
    assert parse_yes_no(text) == expected


def test_sample_indices_includes_ends():
    assert sample_indices(10, 3) == [0, 4, 9]
    assert sample_indices(2, 5) == [0, 1]
    assert sample_indices(0, 5) == []


# ── the measured failure mode ─────────────────────────────────────────────

def test_sycophantic_model_yields_no_verdict():
    """A model that says yes to everything must produce no verdict.

    This is the FastVLM-0.5B behaviour measured on real footage. Before
    polarity verification it scored 0.84 pickpocket on a woman sitting alone
    in a car.
    """
    r = adjudicate_frames(FRAMES, ALWAYS_YES)
    assert r["verdict"] is None
    assert r["insufficient_evidence"] is True
    assert r["inconsistent_answers"] > 0


def test_credulous_mode_reproduces_the_false_verdict():
    """The old behaviour is still reachable, and still wrong — that is the point."""
    r = adjudicate_frames(FRAMES, ALWAYS_YES, verify_polarity=False)
    assert r["verdict"] is not None
    assert r["score"] > 0.5


def test_always_no_is_consistent_but_finds_nothing():
    r = adjudicate_frames(FRAMES, ALWAYS_NO)
    assert r["verdict"] is None
    # "no" to a claim and "no" to its negation is equally self-contradictory.
    assert r["inconsistent_answers"] > 0


# ── a model that actually reads the image ─────────────────────────────────

def _consistent(yes_keys: set[str]):
    """An `ask` that answers each question truthfully in both polarities."""
    def answer_for(prompt: str) -> str:
        for q in QUESTIONS:
            truth = q.key in yes_keys
            if prompt == q.prompt:
                return "yes" if truth else "no"
            if q.negation and prompt == q.negation:
                return "no" if truth else "yes"
        raise AssertionError(f"unexpected prompt: {prompt}")
    return mock(answer_for)


def test_consistent_snatch_scores():
    r = adjudicate_frames(FRAMES, _consistent(
        {"two_people", "hand_to_object", "phone_visible", "fleeing"}))
    assert r["insufficient_evidence"] is False
    assert r["verdict"] == "snatch_theft"
    assert r["scores"]["snatch_theft"] == pytest.approx(0.95, abs=1e-6)
    assert r["inconsistent_answers"] == 0


def test_amicable_suppresses_the_verdict():
    with_ = adjudicate_frames(FRAMES, _consistent(
        {"two_people", "hand_to_object", "phone_visible"}))
    without = adjudicate_frames(FRAMES, _consistent(
        {"two_people", "hand_to_object", "phone_visible", "amicable"}))
    assert without["scores"]["snatch_theft"] < with_["scores"]["snatch_theft"]


def test_single_person_gates_two_person_questions():
    """Without a confirmed second person, "hand in another's pocket" is moot."""
    r = adjudicate_frames(FRAMES, _consistent({"reaching_into", "phone_visible"}))
    assert "reaching_into" in r["gated_questions"]
    assert r["observations"]["reaching_into"] is None
    # phone_visible still counts — a phone needs only one person to hold it —
    # but reaching_into's 0.60 is excluded, leaving nothing near a threshold.
    assert r["scores"]["pickpocket"] == pytest.approx(0.10)


def test_gate_does_not_fire_when_two_people_present():
    r = adjudicate_frames(FRAMES, _consistent({"two_people", "reaching_into"}))
    assert r["gated_questions"] == []
    assert r["scores"]["pickpocket"] > 0


# ── failure handling ──────────────────────────────────────────────────────

def test_vlm_exception_is_reported_not_swallowed():
    def boom(image, prompt="", max_tokens=12):
        raise RuntimeError("cuda oom")
    r = adjudicate_frames(FRAMES, boom)
    assert "error" in r and "cuda oom" in r["error"]


def test_unconfigured_vlm_is_an_error_not_a_null_verdict():
    """`run_vlm` reports configuration failure in its return value."""
    ask = lambda image, prompt="", max_tokens=12: {
        "text": "", "error": "VLM not configured"}
    r = adjudicate_frames(FRAMES, ask)
    assert "error" in r
    assert "not configured" in r["error"]
    assert "verdict" not in r


def test_no_frames():
    assert "error" in adjudicate_frames([], ALWAYS_YES)


def test_questions_needing_two_people_have_negations():
    for q in QUESTIONS:
        assert q.negation, f"{q.key} has no opposite-polarity control"
