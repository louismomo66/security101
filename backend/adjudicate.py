"""VLM adjudication — a second opinion on candidate incidents.

The architecture the video-anomaly literature converges on is two-stage: cheap
detectors *propose* moments, a vision-language model *adjudicates* them. Recent
work (ASK-HINT and related fine-grained-prompting methods) reports ~90% AUC on
UCF-Crime with a **frozen** VLM and no training at all, beating fine-tuned
specialist models. The gain comes from the prompting, not the weights.

Two design rules follow from that literature, and both matter here:

1. **Ask many narrow questions, not one broad one.** "Is this a crime?" invites
   a guess. "Is one person's hand touching an object held by another person?"
   is answerable from pixels. Fine-grained decomposition is what unlocks the
   frozen model.

2. **Ask only about what is visible.** Every question below is observational.
   None asks about ownership, permission, or intent, because none of those are
   in the image — a person taking a phone from another person's hand looks
   identical whether they stole it or were handed it. The adjudicator reports
   observations; a human decides what they mean.

The verdict this produces is evidence for review, never a finding.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

# A callable that takes (image, prompt) and returns {"text": ...}.
AskFn = Callable[..., dict]


@dataclass(frozen=True)
class Question:
    key: str
    prompt: str
    # Event types this question supports, and how much it contributes.
    supports: dict[str, float] = field(default_factory=dict)
    # Questions that *lower* confidence in an event when answered yes.
    contradicts: dict[str, float] = field(default_factory=dict)
    # The same question with the opposite polarity. Asked as a control: a model
    # that answers "yes" to both a proposition and its negation is agreeing
    # with the question rather than reading the image, and its answer to the
    # positive form carries no information. See `adjudicate_frames`.
    negation: str = ""


_YESNO = " Answer with one word: yes or no."

QUESTIONS: list[Question] = [
    Question(
        "two_people",
        "Are there two or more people in this image?" + _YESNO,
        # Not scored. It gates the questions below, all of which presuppose a
        # second person — the presupposition a weak model will simply grant.
        negation="Is there exactly one person, or nobody, in this image?"
                 + _YESNO,
    ),
    Question(
        "hand_to_object",
        "Is one person's hand touching, pulling, or taking an object that "
        "another person is holding?" + _YESNO,
        supports={"snatch_theft": 0.55, "pickpocket": 0.25},
        negation="Is every person in this image keeping their hands to "
                 "themselves, not touching anything held by anyone else?"
                 + _YESNO,
    ),
    Question(
        "phone_visible",
        "Is a mobile phone visible in this image?" + _YESNO,
        supports={"snatch_theft": 0.20, "pickpocket": 0.10},
        negation="Is this image free of any mobile phone?" + _YESNO,
    ),
    Question(
        "reaching_into",
        "Is a person's hand reaching into another person's pocket, bag, or "
        "clothing?" + _YESNO,
        supports={"pickpocket": 0.60},
        negation="Are all hands in this image clear of other people's "
                 "pockets, bags and clothing?" + _YESNO,
    ),
    Question(
        "arm_extended_toward",
        "Is one person's arm extended toward another person's body or hands?"
        + _YESNO,
        supports={"snatch_theft": 0.15, "pickpocket": 0.10, "violence": 0.10},
        negation="Is everyone in this image keeping their arms close to their "
                 "own body?" + _YESNO,
    ),
    Question(
        "fleeing",
        "Is anyone running or moving away quickly?" + _YESNO,
        supports={"snatch_theft": 0.20},
        negation="Is everyone in this image standing still or moving calmly?"
                 + _YESNO,
    ),
    Question(
        "weapon",
        "Is anyone holding a weapon such as a knife or a gun?" + _YESNO,
        supports={"armed_threat": 0.85},
        negation="Are the hands of everyone in this image empty of weapons?"
                 + _YESNO,
    ),
    Question(
        "striking",
        "Is one person hitting, kicking, or pushing another person?" + _YESNO,
        supports={"violence": 0.80},
        negation="Are the people in this image refraining from any violent "
                 "physical contact?" + _YESNO,
    ),
    Question(
        "amicable",
        "Are the people interacting in a friendly or calm way, such as "
        "shaking hands, greeting, or talking?" + _YESNO,
        # Friendliness is the single most useful negative signal: it is the
        # main thing that separates a snatch from a handover, and the skeleton
        # model is blind to it.
        contradicts={"snatch_theft": 0.35, "pickpocket": 0.35, "violence": 0.35},
        negation="Do the people in this image appear tense, hostile, or in "
                 "conflict?" + _YESNO,
    ),
]

# Questions that presuppose a second person. If the model cannot confirm two
# people are present, these cannot be answered about anything real.
_NEEDS_TWO = {"hand_to_object", "reaching_into", "arm_extended_toward",
              "striking", "amicable"}
_GATE = "two_people"

EVENT_TYPES = sorted({e for q in QUESTIONS for e in {**q.supports, **q.contradicts}})

# Answer parsing ──────────────────────────────────────────────────────────

_YES = re.compile(r"\b(yes|yeah|correct|true|affirmative)\b", re.I)
_NO = re.compile(r"\b(no|nope|not|none|negative|isn'?t|aren'?t|cannot|can'?t)\b", re.I)
_HEDGE = re.compile(r"(\bmaybe\b|\bpossibly\b|\bunclear\b|\buncertain\b|"
                    r"\bnot sure\b|\bnot certain\b|\bno idea\b|hard to tell|"
                    r"difficult to tell|can'?t tell|cannot tell|\bblurry\b|"
                    r"\bmight\b|\bcould be\b|\bperhaps\b)", re.I)


def parse_yes_no(text: str) -> float | None:
    """Map a free-text answer to 1.0 (yes), 0.0 (no) or None (unusable).

    Small VLMs ignore "answer yes or no" often enough that naive
    `text.startswith("yes")` throws away a large share of valid answers. This
    checks the leading token first, then falls back to whole-string matching,
    and returns None rather than guessing when the answer is genuinely
    ambiguous — an unusable answer must not be silently counted as "no".
    """
    if not text:
        return None
    t = text.strip()

    has_yes, has_no = bool(_YES.search(t)), bool(_NO.search(t))

    # A contradictory answer ("yes and no", "yes, but no weapon is visible") is
    # unusable, and must be reported as such. Trusting the leading token here
    # would silently convert hedging into a confident yes — the worst possible
    # failure for a signal that ends up in front of a human as an accusation.
    if has_yes and has_no:
        return None

    head = re.match(r"^\W*(\w+)", t)
    if head:
        w = head.group(1).lower()
        if w in ("yes", "yeah", "yep", "true", "correct"):
            return 1.0
        if w in ("no", "nope", "false", "none"):
            return 0.0

    # Hedges are checked before the negation fallback: "I'm not sure" contains
    # "not" and would otherwise be scored as a confident "no", turning the
    # model's uncertainty into evidence of innocence — or, on the supporting
    # questions, quietly suppressing a real detection.
    if _HEDGE.search(t):
        return None

    if has_yes:
        return 1.0
    if has_no:
        return 0.0
    return None


def sample_indices(n_frames: int, k: int) -> list[int]:
    """Pick k frame indices spread across a window, always including the ends."""
    if n_frames <= 0:
        return []
    if n_frames <= k:
        return list(range(n_frames))
    return list(np.linspace(0, n_frames - 1, num=k).astype(int))


def adjudicate_frames(
    frames: Sequence[np.ndarray],
    ask: AskFn,
    max_frames: int = 5,
    max_tokens: int = 12,
    questions: Sequence[Question] = tuple(QUESTIONS),
    verify_polarity: bool = True,
    min_usable_fraction: float = 0.5,
) -> dict:
    """Ask each question about several frames and aggregate into a verdict.

    Aggregation is `any-frame` (max over frames) for supporting questions: a
    0.2 s snatch is visible in one or two frames out of five, so averaging
    would wash it out — the same dilution problem the skeleton model has in the
    temporal dimension. Contradicting questions aggregate by *mean*, because
    "these people look friendly" is only meaningful if it holds broadly.

    **Polarity verification.** Measured against FastVLM-0.5B on real footage,
    a narrow yes/no question is not answered from the image — it is answered
    from the question. On a frame containing one woman alone in a car the model
    answered "yes" to *"is one person's hand taking an object another person is
    holding"*, "yes" to *"is anyone running"*, and "There are two people in
    this image", while its own free-text caption correctly said "a woman
    seated inside a car". Implausible controls ("is there a giraffe") got a
    correct "no", so this is assent to a plausible premise, not blindness.

    Any-frame aggregation then turns one such yes into full weight, which is
    how two unrelated control windows both scored 0.820 snatch_theft — higher
    than the one confirmed snatch in the same clip.

    So each question is also asked in the opposite polarity. An answer counts
    only if the two disagree; a model that affirms both a claim and its
    negation has told us nothing, and that is recorded as unusable rather than
    resolved in either direction. When too few answers survive, the verdict is
    None with `insufficient_evidence` set — the honest output, and far safer
    than a confident score in front of a human reviewer.

    Set `verify_polarity=False` for the older, cheaper, credulous behaviour.
    """
    idx = sample_indices(len(frames), max_frames)
    if not idx:
        return {"error": "no frames"}

    answers: dict[str, list[float]] = {q.key: [] for q in questions}
    raw: dict[str, list[str]] = {q.key: [] for q in questions}
    unusable = 0
    inconsistent = 0
    asked = 0

    def _ask(frame, prompt) -> tuple[str, dict | None]:
        try:
            out = ask(frame, prompt=prompt, max_tokens=max_tokens) or {}
        except Exception as exc:  # a dead VLM must not kill the pipeline
            return "", {"error": f"VLM call failed: {exc}"}
        # A configuration failure reports itself through the return value, not
        # an exception. Left alone it becomes an empty string, then an unusable
        # answer, then a null verdict indistinguishable from "the VLM looked
        # and saw nothing" — so surface it as the error it is.
        if out.get("error") and not out.get("text"):
            return "", {"error": f"VLM call failed: {out['error']}"}
        return out.get("text", ""), None

    for i in idx:
        for q in questions:
            text, err = _ask(frames[i], q.prompt)
            if err:
                return err
            asked += 1
            raw[q.key].append(text)
            v = parse_yes_no(text)

            if v is not None and verify_polarity and q.negation:
                neg_text, err = _ask(frames[i], q.negation)
                if err:
                    return err
                asked += 1
                raw[q.key].append(f"[neg] {neg_text}")
                nv = parse_yes_no(neg_text)
                # Consistent means the two polarities disagree. Equal answers
                # (yes/yes or no/no) are self-contradictory, and an unusable
                # control leaves the positive answer unverified.
                if nv is None or nv == v:
                    inconsistent += 1
                    v = None

            if v is None:
                unusable += 1
            else:
                answers[q.key].append(v)

    # The gate question is evidence about the scene, not about an event: if we
    # cannot establish that a second person is present, every question that
    # presupposes one is unanswerable regardless of what the model said.
    gated: list[str] = []
    gate_vals = answers.get(_GATE)
    if gate_vals is not None and any(q.key == _GATE for q in questions):
        if not gate_vals or max(gate_vals) < 1.0:
            for q in questions:
                if q.key in _NEEDS_TWO and answers[q.key]:
                    gated.append(q.key)
                    answers[q.key] = []

    scores: dict[str, float] = {e: 0.0 for e in EVENT_TYPES}
    for q in questions:
        vals = answers[q.key]
        if not vals:
            continue
        for ev, w in q.supports.items():
            scores[ev] += w * max(vals)
        for ev, w in q.contradicts.items():
            scores[ev] -= w * (sum(vals) / len(vals))

    for e in scores:
        scores[e] = round(max(0.0, min(1.0, scores[e])), 3)

    # Measured over *decisions* (one per question per frame), not over VLM
    # calls: polarity verification doubles the calls without adding evidence,
    # so counting calls would let a model that fails every single question
    # still clear a 50% bar.
    decisions = len(idx) * len(questions)
    usable = decisions - unusable
    enough = decisions > 0 and (usable / decisions) >= min_usable_fraction
    top = max(scores, key=lambda e: scores[e]) if scores else None
    verdict = top if (enough and top and scores[top] > 0) else None

    return {
        "verdict": verdict,
        "score": scores.get(top, 0.0) if verdict else 0.0,
        "scores": scores,
        "observations": {
            q.key: (max(answers[q.key]) if answers[q.key] else None)
            for q in questions
        },
        "frames_used": len(idx),
        "questions_asked": decisions,
        "vlm_calls": asked,
        "unusable_answers": unusable,
        "inconsistent_answers": inconsistent,
        "gated_questions": gated,
        "insufficient_evidence": not enough,
        "raw": raw,
    }


def adjudicate_video(
    path: str,
    start: float,
    end: float,
    ask: AskFn,
    max_frames: int = 5,
    **kwargs,
) -> dict:
    """Decode a time range from a video file and adjudicate it."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"error": f"cannot open {path}"}
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 25.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
        n = max(1, int((end - start) * fps))
        frames = []
        for _ in range(n):
            ok, bgr = cap.read()
            if not ok or bgr is None:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()

    if not frames:
        return {"error": "no frames decoded"}
    out = adjudicate_frames(frames, ask, max_frames=max_frames, **kwargs)
    out["window"] = {"start": start, "end": end, "fps": fps,
                     "frames_decoded": len(frames)}
    return out
