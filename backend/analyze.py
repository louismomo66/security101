"""Offline video analysis — walk a recorded clip end to end and produce a case file.

The WebSocket feed in `server.py` exists to show a live scene. This module
exists to answer a different question: *given this recording, what happened in
it and when?* That difference drives three design choices.

**The clock is the video's, not the wall's.** Every rule that measures duration
— loitering, dwell, collision settling, cooldown — is fed `video_time_s` as its
`now`. A clip analysed in 40 seconds of real time still reports a 90-second
loiter correctly, and every event carries the timecode a reviewer scrubs to.

**Frames are sampled, not streamed.** `stride` skips frames with `grab()`, which
decodes nothing. Analysis at stride 3 on 25fps footage still gives ~8Hz temporal
resolution, which is finer than any rule here needs.

**Identity outlives the shot.** Every event that names a subject gets that
subject committed to `backend.identity`, so a vehicle that caused a collision is
recognisable when it reappears — later in the clip, or in a different clip
analysed next month.

Jobs run one at a time. The models are shared singletons holding non-reentrant
ONNX sessions and a single GPU, so parallel jobs would contend rather than
scale.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from backend.identity import (
    IdentityRegistry, appearance_signature, aspect_of, describe_colour,
    format_timecode, kind_of,
)
from backend.threat import ThreatConfig, ThreatEngine, VEHICLE_CLASSES
from backend.weapons import load_weapon_model, verify_held_object, SmoothedWeaponDetector

# Only one analysis at a time — see module docstring.
_run_lock = threading.Lock()

# Caption-tier events name no box — the lexicon matched a sentence, not an
# object. For those, the subjects are inferred from what is in frame at that
# moment and recorded with role "present", never "involved". That distinction
# is carried into the report: being visible while a signal fired is not
# participation in anything.
PERSON_SUBJECT_TYPES = {
    "violence", "violence_reported", "robbery_reported", "theft_reported",
    "burglary_reported", "weapon_reported", "shooting_reported",
    "vandalism_reported", "narcotics_reported", "pursuit_reported",
    "intrusion_reported", "person_down", "person_down_reported",
    "zone_intrusion", "after_hours_presence",
}
VEHICLE_SUBJECT_TYPES = {"collision_reported", "pursuit_reported"}

MAX_INFERRED_SUBJECTS = 4


# ── Options ───────────────────────────────────────────────────────────────

@dataclass
class AnalysisOptions:
    """Everything tunable about a run. All fields are safe to leave at default."""

    stride: int = 3                     # process every Nth frame
    conf: float = 0.40
    iou: float = 0.45
    enable_pose: bool = True
    enable_vlm: bool = True
    vlm_interval_s: float = 4.0         # video seconds between captions
    vlm_max_tokens: int = 80

    # Targeted VLM re-check of weapon candidates. Expensive, so capped.
    verify_weapons: bool = True
    max_verifications: int = 30

    register_identities: bool = True
    save_snapshots: bool = True

    start_s: float = 0.0                # skip the first N seconds
    max_duration_s: float | None = None  # analyse at most N seconds of footage

    threat_config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ── Job ───────────────────────────────────────────────────────────────────

@dataclass
class AnalysisJob:
    job_id: str
    video_path: str
    video_name: str
    options: AnalysisOptions
    status: str = "queued"              # queued|running|completed|failed|cancelled
    created_at: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None

    video: dict = field(default_factory=dict)
    progress: dict = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    captions: list[dict] = field(default_factory=list)
    entities: list[dict] = field(default_factory=list)
    report: dict | None = None
    counters: dict = field(default_factory=dict)

    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    def cancel(self):
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def summary(self) -> dict:
        """Status view — omits the event bodies, which can be large."""
        return {
            "job_id": self.job_id,
            "video_name": self.video_name,
            "video_path": self.video_path,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "video": self.video,
            "progress": self.progress,
            "counters": self.counters,
            "event_count": len(self.events),
            "entity_count": len(self.entities),
            "has_report": self.report is not None,
            "options": self.options.to_dict(),
        }

    def to_dict(self, include_events: bool = True) -> dict:
        d = self.summary()
        if include_events:
            d["events"] = self.events
            d["captions"] = self.captions
            d["entities"] = self.entities
            d["report"] = self.report
        return d


class JobStore:
    """In-memory job registry with a bounded history."""

    def __init__(self, max_jobs: int = 25):
        self._jobs: dict[str, AnalysisJob] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self.max_jobs = max_jobs

    def add(self, job: AnalysisJob):
        with self._lock:
            self._jobs[job.job_id] = job
            self._order.append(job.job_id)
            while len(self._order) > self.max_jobs:
                old = self._order.pop(0)
                # Never evict a job that is still doing work.
                if self._jobs.get(old) and self._jobs[old].status in ("running", "queued"):
                    self._order.append(old)
                    break
                self._jobs.pop(old, None)

    def get(self, job_id: str) -> AnalysisJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[AnalysisJob]:
        with self._lock:
            return [self._jobs[j] for j in reversed(self._order) if j in self._jobs]


# ── Frame annotation ──────────────────────────────────────────────────────

def _box_area(det: dict) -> float:
    box = det.get("box") or [0, 0, 0, 0]
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _annotate(frame_rgb: np.ndarray, det_result: dict | None,
              highlight: list[float] | None = None,
              caption: str | None = None) -> np.ndarray:
    """Evidence still: detection overlay, the subject boxed, a timecode burn-in.

    A snapshot with no context is hard to act on — the reviewer needs to see
    which of six people in frame the alert is about.
    """
    out = frame_rgb
    if det_result is not None and len(det_result.get("boxes", [])) > 0:
        try:
            from detectors import YOLODetector
            out = YOLODetector.draw(frame_rgb, det_result["boxes"],
                                    det_result["scores"], det_result["class_ids"])
        except Exception:
            out = frame_rgb.copy()
    else:
        out = frame_rgb.copy()

    if highlight and len(highlight) == 4:
        x1, y1, x2, y2 = [int(v) for v in highlight]
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 32, 32), 3)

    if caption:
        h, w = out.shape[:2]
        scale = max(0.4, min(h, w) * 0.0009)
        thick = max(1, int(min(h, w) * 0.0018))
        (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.rectangle(out, (0, 0), (min(w, tw + 12), th + 14), (0, 0, 0), -1)
        cv2.putText(out, caption, (6, th + 6), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (255, 255, 255), thick, cv2.LINE_AA)
    return out


# ── Analyzer ──────────────────────────────────────────────────────────────

class VideoAnalyzer:
    """Runs one job. Instantiated per job so its engine state is isolated."""

    def __init__(self, job: AnalysisJob, registry: IdentityRegistry,
                 snapshot_dir: Path,
                 on_event: Callable[[dict], None] | None = None):
        self.job = job
        self.registry = registry
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.on_event = on_event

        cfg = ThreatConfig(**{k: v for k, v in (job.options.threat_config or {}).items()
                              if hasattr(ThreatConfig, k)})
        # Snapshots are written here, with the subject highlighted, rather than
        # by the engine — it never sees pixels.
        cfg.save_snapshots = False
        self.engine = ThreatEngine(cfg)
        self._verifications = 0
        self._last_vlm = -1e9

    # ── main loop ─────────────────────────────────────────────────────────

    def run(self) -> AnalysisJob:
        job = self.job
        opts = job.options
        job.status = "running"
        job.started_at = datetime.now(timezone.utc).isoformat()

        cap = cv2.VideoCapture(job.video_path)
        if not cap.isOpened():
            job.status = "failed"
            job.error = f"Could not open video: {job.video_path}"
            job.finished_at = datetime.now(timezone.utc).isoformat()
            return job

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        job.video = {
            "name": job.video_name,
            "fps": round(fps, 3),
            "frames": total,
            "duration_s": round(total / fps, 2) if total else 0.0,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        }

        if opts.start_s > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(opts.start_s * fps))

        stop_at = None
        if opts.max_duration_s:
            stop_at = opts.start_s + opts.max_duration_s

        # Lazy: importing models pulls in torch and the VLM stack, which we do
        # not want to pay for merely to construct a job.
        from backend.models import run_detection, run_pose, run_action
        # See backend/vlm_backend.py: SENTINEL_VLM_BACKEND=claude swaps this
        # for the Claude API instead of the local offline FastVLM model.
        from backend.vlm_backend import get_vlm_fn
        run_vlm = get_vlm_fn()

        # Wrapped for person-cropped inference + temporal persistence — see
        # backend/weapons.py:SmoothedWeaponDetector and the note in
        # backend/server.py's ws_feed for why. One instance per job.
        _base_weapon_model = load_weapon_model()
        weapon_model = (SmoothedWeaponDetector(_base_weapon_model)
                        if _base_weapon_model is not None else None)
        frame_pos = int(opts.start_s * fps)
        processed = 0
        last_action: dict | None = None
        wall_start = time.perf_counter()

        try:
            while not job.cancelled:
                if not cap.grab():
                    break
                frame_pos += 1
                video_time = frame_pos / fps

                if stop_at is not None and video_time > stop_at:
                    break
                if (frame_pos % max(1, opts.stride)) != 0:
                    continue

                ok, bgr = cap.retrieve()
                if not ok or bgr is None:
                    continue
                frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                processed += 1

                self.engine.set_clock(video_time_s=video_time, frame=frame_pos,
                                      source=job.video_name)
                result = self._process_frame(
                    frame, video_time, run_detection, run_pose, run_action,
                    run_vlm, weapon_model, opts, last_action,
                )
                last_action = result["action"]
                job.events.extend(result["events"])

                elapsed = time.perf_counter() - wall_start
                job.progress = {
                    "frame": frame_pos,
                    "total": total,
                    "percent": round(100.0 * frame_pos / total, 2) if total else 0.0,
                    "video_time_s": round(video_time, 2),
                    "timecode": format_timecode(video_time),
                    "duration_s": job.video["duration_s"],
                    "frames_processed": processed,
                    "elapsed_s": round(elapsed, 1),
                    "processing_fps": round(processed / elapsed, 2) if elapsed > 0 else 0.0,
                    "eta_s": self._eta(frame_pos, total, elapsed),
                }
        except Exception as exc:  # a bad frame should not lose the whole run
            job.error = f"{type(exc).__name__}: {exc}"
            job.status = "failed"
        finally:
            cap.release()

        if job.cancelled and job.status == "running":
            job.status = "cancelled"
        elif job.status == "running":
            job.status = "completed"
            job.progress["percent"] = 100.0

        self._finalise()
        job.finished_at = datetime.now(timezone.utc).isoformat()
        return job

    def _eta(self, frame_pos: int, total: int, elapsed: float) -> float | None:
        if not total or frame_pos <= 0 or elapsed <= 0:
            return None
        return round(elapsed * (total - frame_pos) / frame_pos, 1)

    # ── per frame ─────────────────────────────────────────────────────────

    def _process_frame(self, frame, video_time, run_detection, run_pose,
                       run_action, run_vlm, weapon_model, opts,
                       last_action) -> dict:
        job = self.job

        det_result = run_detection(frame, conf=opts.conf, iou=opts.iou)
        detections = list(det_result["objects"])

        # The optional weapon head runs alongside COCO and merges into the same
        # detection list, so every downstream rule sees its classes for free.
        if weapon_model is not None:
            try:
                person_boxes = [o["box"] for o in detections if o.get("class") == "person"]
                detections.extend(weapon_model.detect(frame, person_boxes=person_boxes,
                                                       iou=opts.iou))
            except Exception:
                pass

        pose_persons = None
        if opts.enable_pose:
            pose_result = run_pose(frame, conf=opts.conf, iou=opts.iou)
            if pose_result is not None:
                pose_persons = pose_result["persons"]
                action = run_action(pose_result["keypoints"], pose_result["scores"])
                if action is not None:
                    last_action = action

        # Video time is the clock, so duration-based rules measure the footage.
        events = self.engine.update(
            detections=detections, action=last_action,
            pose_persons=pose_persons, frame_shape=frame.shape,
            now=video_time,
        )

        if opts.enable_vlm and (video_time - self._last_vlm >= opts.vlm_interval_s):
            self._last_vlm = video_time
            try:
                vlm = run_vlm(frame, prompt=self.engine.threat_prompt(),
                              max_tokens=opts.vlm_max_tokens)
                text = (vlm or {}).get("text", "").strip()
                if text:
                    job.captions.append({
                        "video_time_s": round(video_time, 2),
                        "timecode": format_timecode(video_time),
                        "text": text,
                    })
                    events += self.engine.ingest_caption(text, now=video_time)
                else:
                    # run_vlm reports an unloaded model by returning an error
                    # rather than raising, so without this the whole tier-2
                    # screen would be silently absent from the report.
                    job.counters["vlm_empty"] = job.counters.get("vlm_empty", 0) + 1
                    if (vlm or {}).get("error"):
                        job.counters["last_vlm_error"] = vlm["error"]
            except Exception as exc:
                job.counters["vlm_errors"] = job.counters.get("vlm_errors", 0) + 1
                job.counters["last_vlm_error"] = f"{type(exc).__name__}: {exc}"

        for ev in events:
            self._enrich_event(ev, frame, det_result, detections, pose_persons,
                               video_time, opts, run_vlm)
            if self.on_event:
                try:
                    self.on_event(ev)
                except Exception:
                    pass

        return {"events": events, "action": last_action}

    # ── event enrichment ──────────────────────────────────────────────────

    def _enrich_event(self, ev: dict, frame, det_result, detections,
                      pose_persons, video_time, opts, run_vlm):
        """Attach evidence, a VLM second opinion, and identity links."""
        job = self.job

        # 1. VLM verification of weapon claims. The detector saying "knife" at
        #    0.5 confidence on a 30-pixel object is exactly the case where a
        #    look at the crop is worth the latency.
        if (opts.verify_weapons and opts.enable_vlm
                and ev["type"] in ("weapon_brandished", "weapon_carried",
                                   "crime_tool_held", "weapon_near_person")
                and self._verifications < opts.max_verifications):
            person_box = (ev.get("evidence", {}).get("person_box")
                          or (ev.get("subject") or {}).get("box")
                          or ev.get("evidence", {}).get("box"))
            if person_box:
                self._verifications += 1
                verdict = verify_held_object(frame, person_box, vlm_fn=run_vlm)
                ev["verification"] = verdict
                # Adjust, never decide: a 0.5B VLM is not an oracle either way.
                if verdict["verdict"] == "yes":
                    ev["score"] = round(min(1.0, ev["score"] + 0.12), 3)
                    ev["detail"] += f" — VLM confirms: {verdict['item']}"
                elif verdict["verdict"] == "no":
                    ev["score"] = round(max(0.0, ev["score"] - 0.15), 3)
                    ev["detail"] += " — VLM saw nothing in their hands"

        # 2. Evidence still.
        if opts.save_snapshots:
            highlight = ((ev.get("subject") or {}).get("box")
                         or ev.get("evidence", {}).get("box"))
            burn = f"{ev['timecode']}  {ev['severity'].upper()}  {ev['label']}"
            annotated = _annotate(frame, det_result, highlight, burn)
            ev["snapshot"] = self._save_snapshot(annotated, ev)

        # 3. Identity — the "remember this vehicle" step.
        if opts.register_identities:
            self._register_subjects(ev, frame, detections, video_time)

    def _save_snapshot(self, frame_rgb: np.ndarray, ev: dict) -> str | None:
        try:
            name = f"{self.job.job_id}_{ev['type']}_{ev['id']}.jpg"
            bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                return None
            (self.snapshot_dir / name).write_bytes(buf.tobytes())
            return f"alerts/{name}"
        except Exception:
            return None

    def _register_subjects(self, ev: dict, frame, detections, video_time):
        """Commit the objects an event is about to the identity registry.

        Which objects those are depends on the event: a collision is about the
        striking vehicle; a weapon event is about the person holding it; a
        violent altercation has no single subject, so everyone present is
        recorded as present — explicitly labelled as such, because "in frame
        during an incident" is not "involved in it".
        """
        subjects: list[tuple[dict, str]] = []

        subject = ev.get("subject")
        if subject and subject.get("box"):
            subjects.append((subject, "involved"))

        if ev["type"] in PERSON_SUBJECT_TYPES:
            # Largest boxes first: the people the camera actually resolves are
            # the ones whose appearance signature is worth anything.
            persons = sorted(
                (d for d in detections if d.get("class") == "person"),
                key=_box_area, reverse=True,
            )[:MAX_INFERRED_SUBJECTS]
            subjects += [({"class": "person", "box": p["box"]}, "present")
                         for p in persons]

        if ev["type"] in VEHICLE_SUBJECT_TYPES:
            vehicles = sorted(
                (d for d in detections if d.get("class") in VEHICLE_CLASSES),
                key=_box_area, reverse=True,
            )[:MAX_INFERRED_SUBJECTS]
            subjects += [({"class": v["class"], "box": v["box"]}, "present")
                         for v in vehicles]

        linked = []
        for subj, role in subjects:
            kind = kind_of(subj.get("class", ""))
            if kind is None:
                continue
            sig = appearance_signature(frame, subj["box"])
            if sig is None:
                continue
            colour = describe_colour(frame, subj["box"])
            entity, similarity, is_new = self.registry.observe(
                kind=kind, cls=subj["class"], signature=sig,
                aspect=aspect_of(subj["box"]),
                label=f"{colour} {subj['class']}",
                sighting={
                    "source": self.job.video_name,
                    "job_id": self.job.job_id,
                    "video_time_s": round(video_time, 2),
                    "timecode": ev["timecode"],
                    "frame": ev["frame"],
                    "box": subj["box"],
                    "snapshot": ev.get("snapshot"),
                    "event_type": ev["type"],
                },
            )
            self.registry.link_incident(entity.entity_id, ev, role=role)
            linked.append({
                "entity_id": entity.entity_id,
                "label": entity.label,
                "kind": entity.kind,
                "role": role,
                "recognised": not is_new,
                "similarity": similarity if not is_new else None,
                "prior_incidents": max(0, len(entity.incidents) - 1),
            })

        if linked:
            ev["entities"] = linked
            # A previously-seen subject is the headline fact in a report, so it
            # is surfaced on the event rather than buried in the registry.
            repeats = [l for l in linked if l["recognised"] and l["prior_incidents"] > 0]
            if repeats:
                names = ", ".join(
                    f"{r['label']} ({r['entity_id']}, {r['prior_incidents']} prior)"
                    for r in repeats
                )
                ev["detail"] += f" — previously recorded: {names}"
                ev["repeat_subject"] = True

    # ── wrap-up ───────────────────────────────────────────────────────────

    def _finalise(self):
        job = self.job
        seen: dict[str, dict] = {}
        for ev in job.events:
            for link in ev.get("entities", []):
                ent = self.registry.get(link["entity_id"])
                if ent is not None:
                    seen[ent.entity_id] = ent.summary()
        job.entities = list(seen.values())

        by_sev: dict[str, int] = {}
        by_type: dict[str, int] = {}
        for ev in job.events:
            by_sev[ev["severity"]] = by_sev.get(ev["severity"], 0) + 1
            by_type[ev["type"]] = by_type.get(ev["type"], 0) + 1
        job.counters.update({
            "events": len(job.events),
            "by_severity": by_sev,
            "by_type": by_type,
            "captions": len(job.captions),
            "entities": len(job.entities),
            "vehicles": sum(1 for e in job.entities if e["kind"] == "vehicle"),
            "persons": sum(1 for e in job.entities if e["kind"] == "person"),
            "vlm_verifications": self._verifications,
        })
        self.registry.save()


# ── Runner ────────────────────────────────────────────────────────────────

def start_analysis(video_path: str, video_name: str, options: AnalysisOptions,
                   store: JobStore, registry: IdentityRegistry,
                   snapshot_dir: Path,
                   build_report: Callable[[AnalysisJob], dict] | None = None,
                   ) -> AnalysisJob:
    """Queue a video for analysis and return immediately with the job."""
    job = AnalysisJob(
        job_id=f"job_{uuid.uuid4().hex[:10]}",
        video_path=str(video_path),
        video_name=video_name,
        options=options,
        created_at=datetime.now(timezone.utc).isoformat(),
        progress={"percent": 0.0, "frame": 0, "total": 0},
    )
    store.add(job)

    def _worker():
        # Serialised: the model singletons are not safe to share across runs.
        with _run_lock:
            if job.cancelled:
                job.status = "cancelled"
                job.finished_at = datetime.now(timezone.utc).isoformat()
                return
            analyzer = VideoAnalyzer(job, registry, snapshot_dir)
            analyzer.run()
        if build_report and job.status == "completed":
            try:
                job.report = build_report(job)
            except Exception as exc:
                job.counters["report_error"] = f"{type(exc).__name__}: {exc}"

    threading.Thread(target=_worker, name=f"analyze-{job.job_id}", daemon=True).start()
    return job
