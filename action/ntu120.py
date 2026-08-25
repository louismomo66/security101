"""NTU RGB+D 120 action class labels (0-indexed) and their crime mapping.

Why this matters for VEREC: NTU-120 extends NTU-60 with mutual (two-person)
actions that map directly onto street crime. Three in particular:

    A57  touch other person's pocket   -> pickpocketing
    A109 grab other person's stuff     -> snatching
    A107 wield knife towards other person

The NTU-120 paper notes that A57 and A109 share almost identical posture and
are separated mainly by *speed* — slow is theft, fast is a snatch. That is the
distinction a per-frame object detector structurally cannot make and a temporal
skeleton model can.

Order is authoritative (A1..A120 from the official dataset listing) and must not
be reordered; the indices are the model's output classes.

Source: https://github.com/shahroudy/NTURGB-D
"""
from __future__ import annotations

NTU120_ACTIONS: list[str] = [
    # ── A1–A60: identical to NTU-60 ──────────────────────────────────────
    "drink water",
    "eat meal/snack",
    "brushing teeth",
    "brushing hair",
    "drop",
    "pickup",
    "throw",
    "sitting down",
    "standing up",
    "clapping",
    "reading",
    "writing",
    "tear up paper",
    "wear jacket",
    "take off jacket",
    "wear a shoe",
    "take off a shoe",
    "wear on glasses",
    "take off glasses",
    "put on a hat/cap",
    "take off a hat/cap",
    "cheer up",
    "hand waving",
    "kicking something",
    "reach into pocket",
    "hopping",
    "jump up",
    "make a phone call/answer phone",
    "playing with phone/tablet",
    "typing on a keyboard",
    "pointing to something with finger",
    "taking a selfie",
    "check time (from watch)",
    "rub two hands together",
    "nod head/bow",
    "shake head",
    "wipe face",
    "salute",
    "put the palms together",
    "cross hands in front (say stop)",
    "sneeze/cough",
    "staggering",
    "falling",
    "touch head (headache)",
    "touch chest (stomachache/heart pain)",
    "touch back (backache)",
    "touch neck (neckache)",
    "nausea or vomiting condition",
    "use a fan/feeling warm",
    "punching/slapping other person",
    "kicking other person",
    "pushing other person",
    "pat on back of other person",
    "point finger at the other person",
    "hugging other person",
    "giving something to other person",
    "touch other person's pocket",
    "handshaking",
    "walking towards each other",
    "walking apart from each other",
    # ── A61–A120: NTU-120 extension ──────────────────────────────────────
    "put on headphone",
    "take off headphone",
    "shoot at the basket",
    "bounce ball",
    "tennis bat swing",
    "juggling table tennis balls",
    "hush (quiet)",
    "flick hair",
    "thumb up",
    "thumb down",
    "make ok sign",
    "make victory sign",
    "staple book",
    "counting money",
    "cutting nails",
    "cutting paper (using scissors)",
    "snapping fingers",
    "open bottle",
    "sniff (smell)",
    "squat down",
    "toss a coin",
    "fold paper",
    "ball up paper",
    "play magic cube",
    "apply cream on face",
    "apply cream on hand back",
    "put on bag",
    "take off bag",
    "put something into a bag",
    "take something out of a bag",
    "open a box",
    "move heavy objects",
    "shake fist",
    "throw up cap/hat",
    "hands up (both hands)",
    "cross arms",
    "arm circles",
    "arm swings",
    "running on the spot",
    "butt kicks",
    "cross toe touch",
    "side kick",
    "yawn",
    "stretch oneself",
    "blow nose",
    "hit other person with something",
    "wield knife towards other person",
    "knock over other person",
    "grab other person's stuff",
    "shoot at other person with a gun",
    "step on foot",
    "high-five",
    "cheers and drink",
    "carry something with other person",
    "take a photo of other person",
    "follow other person",
    "whisper in other person's ear",
    "exchange things with other person",
    "support somebody with hand",
    "finger-guessing game",
]

assert len(NTU120_ACTIONS) == 120, f"expected 120 classes, got {len(NTU120_ACTIONS)}"


# ── Crime mapping ─────────────────────────────────────────────────────────
#
# Maps NTU-120 labels onto ThreatEngine event types. Each entry is:
#   label -> (event_type, category, base_score, requires_two_people)
#
# `base_score` is a prior on how much the class deserves attention when the
# model is confident, not a probability that a crime occurred. Scores are
# deliberately below 1.0 even for the worst classes: NTU is acted, indoors, at
# close range with subjects facing the camera, and it transfers imperfectly to
# street CCTV. Treat every one of these as "a human should look at this clip".
#
# `requires_two_people` guards the mutual classes. A lone person shadow-boxing
# trips "punching" constantly, and a lone person adjusting their own jacket
# trips "touch other person's pocket".

CRIME_ACTION_MAP: dict[str, tuple[str, str, float, bool]] = {
    # Theft — the classes this whole exercise was for.
    "touch other person's pocket":
        ("pickpocket", "theft", 0.80, True),
    "grab other person's stuff":
        ("snatch_theft", "theft", 0.85, True),

    # Armed threat.
    "wield knife towards other person":
        ("armed_threat", "weapon", 0.95, True),
    "shoot at other person with a gun":
        ("armed_threat", "weapon", 0.95, True),

    # Violence.
    "punching/slapping other person":
        ("violence", "violence", 0.90, True),
    "kicking other person":
        ("violence", "violence", 0.90, True),
    "pushing other person":
        ("violence", "violence", 0.75, True),
    "hit other person with something":
        ("violence", "violence", 0.90, True),
    "knock over other person":
        ("violence", "violence", 0.85, True),

    # Weak signals — real but ambiguous. Low scores so they surface only when
    # something else corroborates.
    "follow other person":
        ("following", "suspicious", 0.35, True),
    "step on foot":
        ("altercation", "suspicious", 0.25, True),
    "shake fist":
        ("altercation", "suspicious", 0.30, False),
    "hands up (both hands)":
        ("possible_surrender", "suspicious", 0.30, True),

    # ── NTU-60 aliases ───────────────────────────────────────────────────
    # action/ntu60.py uses abbreviated names for the same underlying classes,
    # so rules keyed on one taxonomy would silently miss the other. Note that
    # NTU-60 already contains A57, meaning the stock model can flag
    # pickpocketing; what NTU-120 adds is snatching, knife, gun and
    # hit-with-object.
    "touch pocket":
        ("pickpocket", "theft", 0.80, True),
    "punching/slapping":
        ("violence", "violence", 0.90, True),
}

# Severity by event type.
CRIME_SEVERITY: dict[str, str] = {
    "pickpocket": "high",
    "snatch_theft": "high",
    "armed_threat": "critical",
    "violence": "critical",
    "following": "low",
    "altercation": "low",
    "possible_surrender": "medium",
}

# Human-readable names for alerts.
CRIME_LABELS: dict[str, str] = {
    "pickpocket": "Possible pickpocketing",
    "snatch_theft": "Possible snatch theft",
    "armed_threat": "Weapon directed at a person",
    "violence": "Physical altercation",
    "following": "One person following another",
    "altercation": "Possible altercation",
    "possible_surrender": "Hands raised — possible robbery",
}

# Distress classes (medical, not crime) — kept separate so they never raise a
# crime alert.
DISTRESS_ACTIONS_120: dict[str, float] = {
    "falling": 0.70,
    "staggering": 0.55,
    "touch chest (stomachache/heart pain)": 0.50,
    "touch back (backache)": 0.45,
    "touch head (headache)": 0.40,
    "touch neck (neckache)": 0.40,
    "nausea or vomiting condition": 0.45,
    # NTU-60 spellings of the same classes.
    "chest pain": 0.50,
    "back pain": 0.45,
    "headache": 0.40,
    "neck pain": 0.40,
    "vomiting": 0.45,
}


def is_ntu120(labels: list[str] | None) -> bool:
    """True when a label list looks like the NTU-120 taxonomy."""
    return bool(labels) and len(labels) == 120
