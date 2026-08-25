# Sentinel Crime Analysis

Analyse a recorded video end to end and get back a timecoded incident report:
what was seen, when in the clip, which people and vehicles were involved, and
which of them the system has seen before.

This document covers the crime-analysis layer. For the underlying scoring
engine and its rules, see [THREAT_DETECTION.md](THREAT_DETECTION.md).

```
video file ──► frame sampler (stride) ──► detection ──┐
                    │                     pose ───────┤
                    │                     action ─────┼──► ThreatEngine ──► events
                    │                     VLM caption ┘         │        (timecoded)
                    │                                           ▼
                    │                              ┌────────────┴────────────┐
                    │                              ▼                         ▼
                    │                     identity registry          evidence snapshot
                    │                    (vehicles, persons)          logs/alerts/*.jpg
                    ▼                              │
              progress / cancel                    ▼
                                            report builder ──► JSON + Markdown
```

## Quick start

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{"video": "forecourt.mp4", "stride": 3, "vlm_interval_s": 4}'
```

Returns a job immediately. Poll it, then fetch the report:

```bash
curl http://localhost:8000/api/analyze/job_a1b2c3d4e5
curl "http://localhost:8000/api/analyze/job_a1b2c3d4e5/report?format=markdown"
```

## What it detects

### Crime tools, and who is holding them

An object *near* a person and an object *in their hand* are different signals,
and only the second one means much. The pose model already produces wrist
keypoints every frame, so `backend/weapons.py` anchors object boxes to them.

| Attachment | How it's decided | Event |
|---|---|---|
| **held** | object box within `grip_tolerance` of a wrist keypoint | `weapon_brandished` / `crime_tool_held` |
| **carried** | object ≥55% inside a person's box, no wrist match | `weapon_carried` / `crime_tool_carried` |
| **proximate** | centroid within `weapon_person_proximity` of a person | `weapon_near_person` |
| **loose** | neither | `weapon_visible` |

`grip_tolerance` is a fraction of the **person's** box diagonal, not the
frame's, so one threshold works for someone at the camera and someone at the
far end of a car park.

When pose is unavailable the old proximity behaviour is used unchanged, and the
event says so in its `detail`.

The vocabulary spans four categories — `weapon`, `improvised`, `burglary`,
`concealment`. Non-weapon tools only fire when attached to a person: a crowbar
lying in a yard is scene furniture.

**Firearms need a model you must supply.** COCO has no gun class, so tier 1
cannot see one. Point Sentinel at a weapon-trained YOLO ONNX export and its classes
join the same vocabulary with no other changes:

```bash
export SENTINEL_WEAPON_MODEL=/path/to/weapons.onnx
export SENTINEL_WEAPON_CLASSES='["pistol","rifle","knife","machete"]'
```

Class names are mandatory — without them the outputs cannot be mapped to the
vocabulary, and Sentinel refuses to load the model rather than emit `class_0`
alerts. Check what loaded with `GET /api/weapons/status`.

#### Measured: what "no weapon model" actually costs

This is the single largest gap in the system, so it is worth stating concretely
rather than as a disclaimer. Tested against 30 s of 848×480 night CCTV in which
a subject carries a panga (machete):

| Tier | Result |
|---|---|
| Detector @ conf 0.40 | `car`, `person` only |
| Detector @ conf **0.15** | `car`, `person`, `chair` — no weapon class at any threshold |
| VLM, whole frame, direct question | "NO" at every sampled point |
| VLM, upscaled person crop | detects an object in the right hand on every sample, and identifies it as *"a black bag"*, *"a wallet"*, *"a smartphone"* |

Two distinct failures, and only one is fixable by tuning:

1. **The detector cannot be wrong about a panga, because it cannot say
   "panga".** COCO has 80 classes and none of them is a machete-like implement.
   Lowering thresholds cannot produce a class the network has no output for.
2. **The 0.5B VLM resolves *that* something is held, never *what*.** At this
   resolution and light level that is its ceiling; a larger checkpoint may or
   may not clear it.

The practical consequence: **an empty weapon section in a report is not evidence
that no weapon was present.** The generated caveats say so explicitly on every
run where no weapon model is loaded.

#### Regional vocabulary

The caption lexicon covers `panga`, `cutlass`, `bush knife` and `matchet`
alongside `machete`. This matters more than it looks: a caption of East African
footage will say *panga*, and a lexicon that only knows *machete* is silently
blind on exactly the footage it is pointed at. If you deploy in a region with
its own terms for common implements, add them to `CAPTION_LEXICON`.

#### Verification is not a rubber stamp

`verify_held_object` rejects answers that do not answer. Small VLMs frequently
restate a closed question — *"To determine whether the person is holding a
weapon, we need to…"* — and that restatement contains the question's own
vocabulary. A keyword search over it scores a confirmation on every frame.
Observed on FastVLM-0.5B: 14 of 14 sampled frames returned a false `yes` before
this was fixed. Question-restatement patterns and responses longer than 12 words
are now returned as `unclear`.

### VLM verification

Weapon candidates get a second opinion: the person is cropped, upscaled, and the
VLM is asked a closed question — *"are they holding a weapon? answer YES: <item>,
NO, or UNCLEAR"*. A bare "YES" with no nameable item is treated as `unclear`,
because small VLMs agree with leading questions readily.

The verdict **adjusts the score** (±0.12/0.15) and never creates or deletes an
event. Capped at `max_verifications` per run.

### Collisions, and vehicles that leave

Box overlap alone is worthless from a fixed camera: a car passing behind another
overlaps in projection without touching in the world. What makes contact a
collision is what the motion does next.

So contacts are recorded, then judged after `collision_settle_s` of footage.
An impact is accepted when any of these holds:

- the striking vehicle decelerates to ≤ `collision_decel_ratio` × its approach speed
- the struck party decelerates sharply
- something that was stationary is suddenly displaced
- (pedestrians) the person's track vanishes at the moment of contact

| Event | Severity |
|---|---|
| `vehicle_collision` | high / critical by approach speed |
| `vehicle_pedestrian_collision` | critical |
| `vehicle_left_scene` | high |

Speeds are fractions of the frame diagonal per second, so they survive a
resolution change — but **not** a change of camera angle. These need tuning per
site.

`vehicle_left_scene` fires when a vehicle involved in a collision leaves the
frame under power within `hit_and_run_window_s`. It is not a finding of
hit-and-run: the field of view is not the scene, and a driver may pull over out
of shot. It is a prompt to check whether anyone stopped.

### Remembering vehicles and suspects

Every event names its **subject** — the striking vehicle, the person holding the
weapon — and that subject is committed to the identity registry
(`logs/registry.json`).

An entity's signature is a colour histogram over three horizontal bands of its
crop, plus aspect ratio. Bands roughly separate roof/window/body on a vehicle
and head/torso/legs on a person, which is what makes the signature more than an
average colour. Hue is weighted 4× value, because value moves with lighting and
that is exactly what should not break a match.

When a subject re-appears — later in the clip or in footage analysed next month
— it matches its stored record, the incident is appended to its history, and the
event gains `repeat_subject: true`.

**This is not identification.** It is colour-based re-identification, and its
limits are structural:

- two silver saloons of different makes will match each other
- one vehicle will fail to match itself between daylight and sodium-lit night
- it is not a plate read, and it is not face recognition
- a person's signature is their clothing — it survives a walk across a car park,
  not a change of jacket

Every match carries a similarity score. Treat it as a lead for a human to
confirm.

Plate fields (`plate`, `plate_confidence`) exist on every entity but no reader
ships by default. A reviewer who reads a plate off a snapshot can record it:

```bash
curl -X POST http://localhost:8000/api/registry/veh_a1b2c3d4/note \
  -H 'Content-Type: application/json' \
  -d '{"note": "plate read from 00:01:47 snapshot", "plate": "UAX 123K"}'
```

### Caption screening

Tier 2 gives open-vocabulary coverage — theft, robbery, arson, anything without
a rule — by matching a lexicon against the VLM's caption.

The prompt asks for two lines: a description, then `INCIDENT:` followed by NONE
or a few words. Only the verdict line is screened. This structure is
load-bearing. Asked to describe a frame *and* weigh a list of incident types at
once, the model answers by denying the list — *"there is no visible weapon,
fighting, theft, forced entry, vandalism, fire…"* — which reads to a keyword
screen as a hit on every item. On sample night footage that single sentence
raised four high-severity alerts.

Two defences, both tested:

1. Only the `INCIDENT:` line is screened when the model emits one.
2. Negation is scoped to the **sentence**, not a fixed lookbehind, so a denial
   list is suppressed however long it runs. A contrastive conjunction ends the
   negation's reach: *"no fence, but a man is stabbing someone"* still fires.

## Reports

A report is built deterministically from the event log. An LLM narrative is
added when Ollama is reachable, but **every fact comes from the structured
data** — a report is complete and correct with nothing running.

Sections:

- **Summary** — incident count, highest-priority event and its timecode
- **Incident timeline** — clustered, with `HH:MM:SS.mmm` spans, severity, score,
  occurrence count, and evidence tier
- **Persons of interest** — appearance-grouped, separating *involved* (a rule
  named their box) from *present* (in frame when a caption signal fired)
- **Vehicles** — collision involvement, departure, appearance timecodes
- **Repeat subjects** — matches against previously recorded entities
- **Scene descriptions** — sampled captions, marked unverified
- **Recommended actions** — verification steps, ordered by severity
- **Confidence and caveats** — generated per run, see below

Repeats collapse: forty `loitering` events over three minutes become one
incident with a span and a count, not forty rows.

Every incident states its `evidence_tier`. A `weapon_brandished` grounded in a
detector box, a wrist keypoint and a VLM confirmation is a different object from
a `theft_reported` that is one regex hit, and the report never lets them look
alike.

### Caveats are computed, not boilerplate

The caveats section reacts to how the run actually went. It will tell you when:

- no weapon model is loaded, so firearms were undetectable
- caption screening was enabled but produced **no captions** — the most
  dangerous failure mode, because an empty timeline otherwise looks like a
  clean scan
- the VLM failed on some frames, so coverage is patchier than the interval implies
- pose was off, so grip reasoning degraded to proximity
- stride was coarse enough to miss short events

## API

```
POST   /api/analyze                      { "video": "clip.mp4", ...options }
GET    /api/analyze/jobs
GET    /api/analyze/{job_id}             [?include_events=false]
GET    /api/analyze/{job_id}/events      [?min_severity=high&limit=500]
GET    /api/analyze/{job_id}/report      [?format=json|markdown&narrative=true]
POST   /api/analyze/{job_id}/cancel

POST   /api/session/report               deterministic report over the live session

GET    /api/registry                     [?kind=vehicle|person&with_incidents_only=true]
GET    /api/registry/{entity_id}
POST   /api/registry/{entity_id}/note    { "note": "...", "plate": "..." }
DELETE /api/registry/{entity_id}         erasure
POST   /api/registry/prune               { "older_than_days": 30 }

GET    /api/weapons/status
```

Jobs run one at a time — the models are shared singletons holding non-reentrant
ONNX sessions and a single GPU, so parallel runs would contend rather than scale.

### Options

| Option | Default | Notes |
|---|---|---|
| `stride` | 3 | process every Nth frame; skipped frames are never decoded |
| `conf` / `iou` | 0.40 / 0.45 | detector thresholds |
| `enable_pose` | true | required for grip reasoning and action recognition |
| `enable_vlm` | true | tier-2 caption screening |
| `vlm_interval_s` | 4.0 | **video** seconds between captions |
| `verify_weapons` | true | targeted VLM re-check of weapon candidates |
| `max_verifications` | 30 | cap on those re-checks |
| `register_identities` | true | commit subjects to the registry |
| `save_snapshots` | true | annotated evidence stills |
| `start_s` / `max_duration_s` | 0 / null | analyse a segment |
| `threat_config` | {} | any `ThreatConfig` field, per job |

### The clock

Every duration rule — loitering, dwell, collision settling, cooldown — is fed
`video_time_s` as its `now`. A clip analysed in 40 seconds of real time still
reports a 90-second loiter correctly. Every event carries `video_time_s`,
`frame`, `timecode` and `source`.

This applies to live playback of a file too: the WebSocket feed drives the
engine from the clip's own clock rather than the wall's.

## Tuning

Start permissive, measure against your own footage, then tighten.

| Knob | Effect |
|---|---|
| `grip_tolerance` | 0.18 of the person's diagonal. Raise for low frame rates where hands blur. |
| `weapon_conf` | the single biggest false-positive lever on weapons |
| `collision_iou` | raise if your camera angle stacks vehicles in projection |
| `collision_min_speed` | raise to ignore car-park manoeuvring |
| `collision_settle_s` | how much footage to watch before judging a contact |
| `match_threshold` | registry, default 0.62. Raise to reduce false re-identifications. |
| `stride` | temporal resolution vs. runtime |

## Limits

Read these before relying on output.

1. **Nothing here establishes that a crime occurred.** Every output is a signal
   for a human reviewer. Route alerts to a person, never to automated action.
2. **Intent is not visible.** No rule detects theft. `theft_reported` is a
   keyword hit on a caption — a prompt to look, never a finding.
3. **Weapons outside `knife` / `baseball bat` / `scissors` are invisible without
   a weapon-trained model** — firearms, machetes and pangas, clubs, crowbars.
   This is a missing class, not a low score, so no threshold change reaches it.
   Measured results above.
4. **NTU-60 transfers poorly to CCTV.** It was recorded indoors, close range,
   actors facing camera. Expect misses and false alarms on overhead views.
5. **Collision detection is 2D.** It infers impact from image-plane overlap and
   speed change. Camera geometry determines how well that works.
6. **Re-identification is colour.** See the honest limits above.
7. **Bias is inherited.** Detection and pose models have documented accuracy
   differences across skin tone, body size and clothing. Rules built on top
   inherit them, and the appearance signature is literally a function of
   clothing and skin colour.
8. **Legal.** Snapshots in `logs/alerts/` and the identity records in
   `logs/registry.json` are personal data. Signage, retention limits, DPIA and
   lawful basis are your responsibility. `DELETE /api/registry/{id}` and
   `POST /api/registry/prune` exist so erasure and retention can be honoured —
   they are not honoured automatically.

## Tests

```bash
python tests/test_crime.py       # 52 tests — grip, collisions, re-ID, captions, reports
python tests/test_threat.py      # 21 tests — the underlying rule engine
```

All synthetic: no models, no video files, every signal fabricated so the rules
are tested rather than the detector.
