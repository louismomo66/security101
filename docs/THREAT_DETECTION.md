# Sentinel Threat Detection

Turns the perception stack (YOLO detection, pose + ST-GCN action recognition,
FastVLM captioning) into scored incidents for human review.

## Architecture

```
frame ─┬─ YOLO detection ──────┐
       ├─ YOLO pose → ST-GCN ──┼──► ThreatEngine.update()   ← tier 1, every frame
       └─ FastVLM caption ─────┴──► ThreatEngine.ingest_caption()  ← tier 2, every vlm_interval
                                          │
                                          ▼
                              scored events → cooldown → alerts
                                          │
                          ┌───────────────┼───────────────┐
                          ▼               ▼               ▼
                    WS payload      logs/alerts_*.json  snapshot JPEG
```

## Action model

Two checkpoints work interchangeably; the runtime reads the class count off the
checkpoint head and picks the label set automatically.

| Checkpoint | Classes | Adds |
|---|---|---|
| `stgcn_ntu60_joint.pth` | 60 | violence, distress, pickpocketing (A57) |
| `stgcn_ntu120_joint.pth` | 120 | **snatch theft (A109)**, knife (A107), gun (A110), hit-with-object (A106), following (A116) |

```bash
bash get_action_models.sh                              # downloads both
export ACTION_MODEL_PATH=checkpoints/stgcn_ntu120_joint.pth
```

NTU-120 is the recommended default: A109 (*grab other person's stuff*) is the
snatch-theft class, and A57 vs A109 is separated mainly by speed — slow is
theft, fast is a snatch. See [action/ntu120.py](../action/ntu120.py) for the
full class-to-alert mapping.

**Tier 1 — rules** over structured signals. Deterministic, explainable, cheap.
Covers what can be measured: object classes, spatial relationships, dwell time,
zone occupancy, crowd dynamics, action labels.

**Tier 2 — caption screening.** Regex lexicon over the VLM caption, which gives
open-vocabulary coverage of categories no rule was written for. Lower precision;
scores are adjusted up or down depending on whether a tier-1 signal corroborates
the caption.

## Rules

| Event | Signal | Default severity |
|---|---|---|
| `weapon_brandished` | crime tool matched to a **wrist keypoint** (or, without pose, within 25% of frame diagonal of a person) | critical |
| `weapon_carried` | inside a person's box but not at a hand | high |
| `weapon_near_person` | near a person, pose found no hand holding it | high |
| `weapon_visible` | same object, no person nearby | high |
| `crime_tool_held` / `crime_tool_carried` | burglary or improvised tool attached to a person | high / medium |
| `vehicle_collision` | vehicle contact plus abrupt deceleration or displacement | high / critical |
| `vehicle_pedestrian_collision` | moving vehicle contacts a person | critical |
| `vehicle_left_scene` | a collision-involved vehicle leaves frame under power | high |
| `pickpocket` | NTU A57 *touch other person's pocket* **and** ≥2 people | high |
| `snatch_theft` | NTU120 A109 *grab other person's stuff* **and** ≥2 people | high |
| `armed_threat` | NTU120 A107 *wield knife toward person* / A110 *shoot at person* | critical |
| `violence` | NTU punch / kick / push / hit-with-object / knock-over **and** ≥2 people | critical / high |
| `following` | NTU120 A116 *follow other person* | low |
| `person_down` | NTU falling, staggering, chest pain | high / medium |
| `zone_intrusion` | person's **feet** inside a configured polygon | medium |
| `loitering` | person track stationary > 90 s | low |
| `unattended_object` | bag/backpack/suitcase stationary > 45 s, no person within 22% of diagonal | medium |
| `crowd_surge` | person count +5 within 4 s | medium |
| `crowd_dispersal` | person count −5 within 4 s (post-incident signature) | high |
| `crowd_density` | ≥8 people in frame | low |
| `after_hours_presence` | any person during configured hours | medium |
| `*_reported` | VLM caption lexicon hit (weapon, fight, robbery, theft, burglary, vandalism, fire, intrusion, pursuit, crowd panic, narcotics) | varies |

Every event carries `score`, `severity`, `rule`, `detail`, `evidence`, `subject`
and a snapshot path, plus `video_time_s` / `frame` / `timecode` when a clock has
been set via `ThreatEngine.set_clock()`. Per-type cooldown (default 12 s)
prevents alert storms; rules where two simultaneous incidents are genuinely
distinct (collisions) use a per-pair key so they cannot silence each other.

For grip reasoning, collision dynamics, subject re-identification and timecoded
reports over recorded footage, see [CRIME_ANALYSIS.md](CRIME_ANALYSIS.md).

## Tier 3 — VLM adjudication

The skeleton model proposes; the VLM checks. This is the two-stage architecture
the video-anomaly literature converges on, and recent work reports ~90% AUC on
UCF-Crime with a *frozen* VLM and structured fine-grained prompting — no
training. The gain comes from asking many narrow questions instead of one broad
one.

```bash
# measure it against footage you have labelled
python -m backend.adjudicate_cli --video "videos/clip.mp4" --start 78 --end 82

# sweep a whole clip and score against known incidents
python -m backend.adjudicate_cli --video "videos/clip.mp4" --sweep \
    --truth 79.0 --tolerance 2.0
```

Live, it is **off by default** because it costs several VLM calls per alert:

```
ws://…/ws/feed?enable_adjudication=true
```

Only `snatch_theft`, `pickpocket`, `armed_threat` and `violence` are
adjudicated — the brief, geometrically ambiguous events. A snatch and a
friendly handover are the same skeleton and completely different pictures, so
the `amicable` question does most of the work in suppressing false positives.

Every question is **observational** — what is visible, never intent or
ownership. A person taking a phone from another person's hand looks the same
whether they stole it or were handed it, so the adjudicator reports what it
sees and a human decides what it means.

Answers that are hedged, contradictory or unparseable are counted as *unusable*
rather than as "no". Reported in `unusable_answers`; if that number is high the
verdict is not trustworthy.

### Measured: `Subh775/Threat-Detection-YOLOv8n` is unusable — do not plug it in

Tested 2026-08-09 as a candidate for `SENTINEL_WEAPON_MODEL`. Its card claims
93.1% mAP@50 on `Gun`, MIT licence, classes `[Gun, explosion, grenade, knife]`.
It also carries its own disclaimer against real-world security use. Believe the
disclaimer.

Raw detections at conf 0.25:

| Clip | Result |
|---|---|
| Smash-and-grab raid, 0–30s | Gun ×30 (max 0.59) |
| Same video, **news segment** 60–110s | **grenade ×58 (max 0.79)**, knife ×25, Gun ×6 |
| Uganda `normal` span (confirmed non-incident) | Gun ×1 at 0.57 |
| **Uganda armed-robbers clip** | **nothing at all** |

The 58 "grenades" are **shrubs in a parking lot** in an aerial drone shot. The
one clip that actually contains armed robbers produced zero detections.

Through the full pipeline it is worse, because **grip geometry promotes the
hallucinations instead of filtering them** — a false box near a hand is
indistinguishable from a grip:

| Alert | Ground truth |
|---|---|
| `weapon_brandished` **critical** 0.778 @4.6s | a person in dark clothing, no gun |
| `weapon_brandished` **critical** 0.868 @19.2s | a person at the counter, no gun |
| `weapon_brandished` **critical** 0.872 @96.2s | **police officers holding evidence bags** |
| `knife` 0.43 @79.5s | a man's hand holding a **soda can** |
| `weapon_visible` high @36.5s | the confirmed `normal` span |

Every alert checked was false. The model boxes *people* and labels them
firearms. Layer 3 was the designed defence against a weak layer 2; it does not
survive a detector this bad, because it assumes false boxes are randomly
placed and these cluster on people.

**Do not "fix" the unmapped-class gap.** `grenade` and `explosion` are absent
from `CRIME_TOOL_CLASSES`, so `_rule_weapon` drops them — which is the only
reason 58 shrub-grenades never became alerts. Adding those classes to the
vocabulary would release them.

Nothing was wired up: `SENTINEL_WEAPON_MODEL` remains unset everywhere. The
ONNX export is kept at `checkpoints/threat_yolov8n/` as a measured negative.

### Measured: FastVLM-0.5B cannot answer these questions

Run on 2026-08-09 against `videos/The youngest phone snatcher in my life…mp4`,
whose one confirmed snatch is at 1:19. Three 4-second windows, 5 frames each:

| Window | Before verification | After |
|---|---|---|
| 78–82s — **the confirmed snatch** | pickpocket **0.840** | none, insufficient evidence |
| 20–24s — control | snatch_theft **0.820** | none, insufficient evidence |
| 45–49s — control | snatch_theft **0.820** | none, insufficient evidence |

The controls outscored the real incident. The 20–24s window is a woman sitting
**alone in a car**, and the model answered "yes" to *"is one person's hand
taking an object another person is holding"*, "yes" to *"is anyone running"*,
and "There are two people in this image" — while its own free-text caption
said, correctly, "a woman seated inside a car". Asked whether the image
contained a giraffe, or snow, it correctly said no.

So this is not a blind model and not a threshold that needs tuning. The model
assents to whatever premise the question carries, and every tier-3 question
carries one. Parsing was never the problem: 0/40 answers were unparseable.

Two changes follow, both in `adjudicate.py`:

- **Polarity verification.** Each question is also asked inverted. An answer
  counts only if the two disagree; affirming both a claim and its negation is
  recorded as `inconsistent_answers`, not resolved in either direction. Below
  `min_usable_fraction` surviving answers the verdict is `None` with
  `insufficient_evidence` set. Doubles the VLM calls; `verify_polarity=False`
  (CLI `--no-verify`) restores the old behaviour.
- **A gate question.** `two_people` is unscored and gates the five questions
  that presuppose a second person. It fired correctly on the 20–24s control.

After this, all three windows return no verdict — right on the negatives, and
honest on the positive. **Tier 3 is not usable at 0.5B**; the next step is the
1.5B checkpoint (`get_models.sh`), re-running the table above.

Any-frame `max()` aggregation is what turned one spurious "yes" in five frames
into full weight, and is why two unrelated controls scored identically. It is
left in place — it is correct for a 0.2 s event — but it is only safe on top of
verified answers.

## API

```
GET  /api/alerts?limit=100&min_severity=high&unacknowledged_only=true
POST /api/alerts/{id}/ack
POST /api/alerts/clear
GET  /api/alerts/snapshot/{name}
GET  /api/threat/config
POST /api/threat/config      { "loiter_seconds": 45, "min_severity": "medium" }
GET  /api/export/alerts
POST /api/report             now includes an Incidents section
```

WebSocket `/ws/feed` accepts `enable_threat=true|false` and live control
messages: `{"enable_threat": false}` or `{"threat_config": {...}}`.
Frames carry `alerts: [...]` and `threat: {stats}` when events fire.

## Configuring zones

Zone polygons use normalised coordinates (0–1), so they survive resolution
changes:

```bash
curl -X POST http://localhost:8000/api/threat/config \
  -H 'Content-Type: application/json' \
  -d '{"zones": [{"name": "loading bay",
                  "polygon": [[0.1,0.6],[0.9,0.6],[0.9,1.0],[0.1,1.0]],
                  "severity": "high"}]}'
```

Occupancy is tested against the midpoint of the box's bottom edge — where the
person is standing — not the box centre.

## Tuning

Start permissive, measure, then tighten. The knobs that matter most:

- `min_severity` — raise to `high` to silence loitering/crowd noise
- `cooldown_s` — raise on busy scenes
- `weapon_conf` — the single biggest false-positive lever
- `caption_screening` — turn off if the 0.5B VLM is too noisy on your footage
- `loiter_seconds`, `unattended_seconds` — scene-dependent, no universal default

## Tests

```bash
python tests/test_threat.py        # or: python -m pytest tests/test_threat.py -v
python tests/test_crime.py         # grip, collisions, re-ID, captions, reports
```

21 synthetic tests cover each rule's firing and non-firing paths, cooldown,
severity filtering, and caption negation handling. A further 52 in
`tests/test_crime.py` cover the crime-analysis layer.

## Known limits

Read these before relying on output.

1. **No firearm detection out of the box.** COCO has no gun class, so tier 1
   covers knife, bat and scissors only — the latter two are noisy. Point
   `SENTINEL_WEAPON_MODEL` at a weapon-trained YOLO ONNX export (with
   `SENTINEL_WEAPON_CLASSES`) and its classes join the vocabulary automatically;
   check `GET /api/weapons/status` to confirm what loaded.
2. **NTU60 transfers poorly to CCTV.** It was recorded indoors, close range,
   actors facing camera. Expect both misses and false alarms on overhead views.
3. **Intent is not visible.** Nothing here detects theft. `theft_reported` is a
   caption keyword match — a prompt to look, never a finding.
4. **Thresholds are policy, not fact.** Tune against your own recorded footage
   and measure precision/recall. An unmeasured system trains its operators to
   ignore it.
5. **Bias is inherited.** Detection and pose models have documented accuracy
   differences across skin tone, body size and clothing; rules built on top
   inherit those. Route alerts to a human, never to automated action.
6. **Legal.** Recording and processing identifiable people is regulated in most
   jurisdictions (UK/EU GDPR, US state biometric laws, and others). Signage,
   retention limits, DPIA and lawful basis are your responsibility — the
   snapshots this system writes to `logs/alerts/` are personal data.
