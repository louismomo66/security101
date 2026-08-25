"""Crime action taxonomy for the fine-tuned ST-GCN.

Design notes
------------
The class list is deliberately short and skeleton-observable. A pose-only model
sees joint trajectories — nothing else. So classes must be *distinguishable from
body motion alone*:

  - "phone_snatch" works: a fast reach toward another person's hand/torso
    followed by divergent motion (one person flees) has a real skeletal
    signature.
  - "murder" does not work as a class, and is deliberately absent. Intent and
    outcome are not visible in a skeleton; what is visible is assault. Labelling
    a pose sequence "murder" would train the model to assert something it cannot
    observe. Use `assault` and escalate to human review.
  - "car_theft" is mostly object-context (a person at a vehicle door), which is
    the detection layer's job, not the skeleton model's. It is kept as a coarse
    class but expect weak performance without the vehicle box; the existing
    ThreatEngine rules handle the object side.

`normal` is a real class, not an afterthought. Without a large, diverse negative
set the model will label ordinary street motion as crime — this is the single
biggest failure mode of action-recognition security models.
"""
from __future__ import annotations

# Order defines the class indices used by the trained model. Append only —
# inserting in the middle silently invalidates existing checkpoints.
CRIME_ACTIONS: list[str] = [
    "normal",           # ordinary walking, standing, talking, commerce
    "phone_snatch",     # grabbing a handheld item and departing at speed
    "pickpocket",       # hand entering another person's pocket/bag, no flight
    "assault",          # strike, kick, push, group beating
    "robbery_threat",   # cornering/restraining a person, hands raised in surrender
    "vehicle_theft",    # forcing a vehicle door / window, riding off
    "bag_snatch",       # grabbing a carried bag or backpack
    "fall_or_medical",  # collapse without an assailant — excluded from alerts
]

NUM_CLASSES = len(CRIME_ACTIONS)
LABEL_TO_INDEX = {name: i for i, name in enumerate(CRIME_ACTIONS)}

# Classes that should never raise a crime alert on their own.
NON_CRIME = {"normal", "fall_or_medical"}

# Severity feeding ThreatEngine when one of these fires.
SEVERITY: dict[str, str] = {
    "normal": "none",
    "phone_snatch": "high",
    "pickpocket": "high",
    "assault": "critical",
    "robbery_threat": "critical",
    "vehicle_theft": "high",
    "bag_snatch": "high",
    "fall_or_medical": "medium",
}

# ── UCF-Crime mapping ────────────────────────────────────────────────────
#
# UCF-Crime ships 13 anomaly categories plus Normal. Several do not map onto a
# pose-observable class at all (Explosion, RoadAccidents, Arson) and are dropped
# rather than forced into a bucket — training on mislabeled data is worse than
# training on less data.
UCF_CRIME_MAP: dict[str, str | None] = {
    "Normal": "normal",
    "Normal_Videos_event": "normal",
    "Abuse": "assault",
    "Arrest": None,          # police restraint, not a crime by the subject
    "Arson": None,           # no skeletal signature
    "Assault": "assault",
    "Burglary": None,        # mostly object/scene context
    "Explosion": None,
    "Fighting": "assault",
    "RoadAccidents": None,
    "Robbery": "robbery_threat",
    "Shooting": "robbery_threat",
    "Shoplifting": "pickpocket",   # concealment gesture is the shared signature
    "Stealing": "bag_snatch",
    "Vandalism": None,
}


def resolve(label: str) -> int:
    """Map a label string to its class index, raising a helpful error if unknown."""
    key = label.strip().lower()
    if key not in LABEL_TO_INDEX:
        raise ValueError(
            f"Unknown label {label!r}. Known classes: {', '.join(CRIME_ACTIONS)}"
        )
    return LABEL_TO_INDEX[key]
