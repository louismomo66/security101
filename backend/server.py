"""Sentinel FastAPI backend — REST + WebSocket API for the Next.js frontend.

Run:
    python -m backend.server --model-path checkpoints/llava-fastvithd_0.5b_stage3
"""
from __future__ import annotations

import asyncio
import base64
import dataclasses
import io
import json
import os
import platform
import sys
import time
import argparse
import collections
import re
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

# Add project root to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from backend.models import (
    init_yolo, init_camera, configure_vlm, schedule_vlm_warmup,
    init_yolo_pose, init_action,
    get_camera_frame, run_detection,
    run_pose, run_action,
    resolve_stream_url, last_stream_error, IP_CAMERA_PRESETS, COCO_CLASSES,
)
import backend.models as _models

# run_vlm is selected here rather than imported directly from backend.models
# so SENTINEL_VLM_BACKEND=claude can swap every VLM call in this file (tier-2
# captioning, tier-3 adjudication) over to the Claude API — see
# backend/vlm_backend.py for what that trade actually costs before enabling
# it. Every `run_vlm(...)` call below still works unchanged either way.
from backend.vlm_backend import get_vlm_fn
run_vlm = get_vlm_fn()
from backend.threat import ThreatEngine, ThreatConfig, SEVERITY_ORDER
from backend.adjudicate import adjudicate_frames

# Event types worth a VLM second opinion. Deliberately narrow: adjudication
# costs several VLM calls per alert, and only the brief, geometrically
# ambiguous events benefit — a snatch and a friendly handover look identical
# to a skeleton, and completely different to a VLM.
ADJUDICATED_TYPES = {"snatch_theft", "pickpocket", "armed_threat", "violence"}


def _adjudicate(frames: list, max_frames: int = 4) -> dict:
    """Run VLM adjudication over a short buffer of recent frames."""
    return adjudicate_frames(frames, run_vlm, max_frames=max_frames)
from backend.analyze import AnalysisOptions, JobStore, start_analysis
from backend.identity import IdentityRegistry
from backend.report import build_report, render_markdown
from backend.weapons import load_weapon_model, weapon_model_status, SmoothedWeaponDetector

app = FastAPI(title="Sentinel API", version="1.0.0")

# ── Default model path (used when running via uvicorn --reload / dev mode) ──
_DEFAULT_MODEL_PATH = os.environ.get(
    "VLM_MODEL_PATH",
    os.path.join(os.path.dirname(os.path.dirname(__file__)),
                 "checkpoints", "llava-fastvithd_0.5b_stage3"),
)

@app.on_event("startup")
def _auto_init():
    """Initialise models when uvicorn imports the app (e.g. --reload mode).

    When run via `python -m backend.server`, main() calls these first and
    this is a harmless no-op because init functions are idempotent.
    """
    init_yolo()
    # Optional weapon-trained detector; a no-op unless SENTINEL_WEAPON_MODEL is set.
    load_weapon_model()
    if _models._vlm_config is None:
        configure_vlm(_DEFAULT_MODEL_PATH)
    schedule_vlm_warmup(delay=0)

_ALLOWED_ORIGINS = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,http://frontend:3000",
).split(",")

# The header shows the machine's LAN address so the UI can be opened from
# another device on the same network — but that origin was not allowed, so
# on 192.168.x.x every write failed while reads appeared to work. Saving a
# recipient is a PUT carrying JSON, which needs a preflight; the ack POST
# sends no content-type and needs none. That asymmetry is why "the app
# works, but nothing I change sticks".
_PRIVATE_ORIGIN_RE = (
    r"https?://("
    r"localhost|127\.0\.0\.1|\[::1\]|"
    r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
    r"192\.168\.\d{1,3}\.\d{1,3}|"
    r"172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|"
    r"[A-Za-z0-9-]+\.local"
    r")(:\d+)?"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_origin_regex=_PRIVATE_ORIGIN_RE,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Data store — rolling logs + JSON disk persistence ─────────────────────

_PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(__file__)))
_LOG_DIR = _PROJECT_ROOT / "logs"
_LOG_DIR.mkdir(exist_ok=True)

_export_lock = threading.Lock()
_detection_log: list[dict] = []
_vlm_log: list[dict] = []
_reports: list[dict] = []
_alerts: list[dict] = []
_object_counts: dict[str, int] = {}  # cumulative class → count
MAX_LOG = 500

# ── Threat engine — shared across connections ─────────────────────────────

_SNAPSHOT_DIR = _LOG_DIR / "alerts"
_SNAPSHOT_DIR.mkdir(exist_ok=True)

# ── Local video files ─────────────────────────────────────────────────────

_VIDEO_DIR = _PROJECT_ROOT / "videos"
_VIDEO_DIR.mkdir(exist_ok=True)

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv"}


def _safe_video_name(name: str) -> str:
    """Strip any path components and unsafe characters from an upload name."""
    base = os.path.basename(name or "").strip()
    base = re.sub(r"[^A-Za-z0-9._ -]", "_", base)
    return base or "video.mp4"


def _video_meta(path: Path) -> dict:
    """Probe a video file for fps / frame count / duration."""
    meta = {
        "name": path.name,
        "path": str(path),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "fps": 0.0,
        "frames": 0,
        "duration_s": 0.0,
        "width": 0,
        "height": 0,
    }
    cap = cv2.VideoCapture(str(path))
    try:
        if cap.isOpened():
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            meta["fps"] = round(fps, 2)
            meta["frames"] = max(frames, 0)
            meta["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            meta["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if fps > 0 and frames > 0:
                meta["duration_s"] = round(frames / fps, 2)
    finally:
        cap.release()
    return meta


def _resolve_video_path(ref: str) -> Path:
    """Resolve a video reference to a real file.

    Accepts either a bare filename living in `videos/` (the upload case) or an
    absolute path to a file anywhere on disk (so you can point at footage
    without copying it in first).
    """
    ref = (ref or "").strip()
    if not ref:
        raise HTTPException(status_code=400, detail="No video specified")

    # Tolerate file:// URLs pasted from Finder
    if ref.startswith("file://"):
        ref = ref[len("file://"):]

    candidate = Path(ref).expanduser()
    if not candidate.is_absolute():
        candidate = _VIDEO_DIR / _safe_video_name(ref)

    candidate = candidate.resolve()
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"Video not found: {ref}")
    return candidate

threat_engine = ThreatEngine(ThreatConfig())

# ── Analysis jobs + identity memory ───────────────────────────────────────
#
# The registry is what makes "remember this vehicle" outlive a single run: it
# is loaded from disk at import and written back after every analysis.

_job_store = JobStore()
_registry = IdentityRegistry(_LOG_DIR / "registry.json")

# Session timestamp for file names
_session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _flush_json(filename: str, data: list[dict]):
    """Write a log list to disk as JSON."""
    path = _LOG_DIR / filename
    try:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def _append_detection(entry: dict):
    with _export_lock:
        _detection_log.append(entry)
        if len(_detection_log) > MAX_LOG:
            del _detection_log[: len(_detection_log) - MAX_LOG]
        # Tally object class frequencies
        for obj in entry.get("objects", []):
            cls = obj.get("class", "unknown")
            _object_counts[cls] = _object_counts.get(cls, 0) + 1
        _flush_json(f"detections_{_session_id}.json", _detection_log)


def _append_vlm(entry: dict):
    with _export_lock:
        _vlm_log.append(entry)
        if len(_vlm_log) > MAX_LOG:
            del _vlm_log[: len(_vlm_log) - MAX_LOG]
        _flush_json(f"captions_{_session_id}.json", _vlm_log)


def _append_report(entry: dict):
    with _export_lock:
        _reports.append(entry)
        _flush_json(f"reports_{_session_id}.json", _reports)


def _append_alerts(entries: list[dict]):
    if not entries:
        return
    with _export_lock:
        _alerts.extend(entries)
        if len(_alerts) > MAX_LOG:
            del _alerts[: len(_alerts) - MAX_LOG]
        _flush_json(f"alerts_{_session_id}.json", _alerts)


def _save_snapshot(frame: np.ndarray, event: dict) -> str | None:
    """Persist an evidence still for an alert. Returns a relative path."""
    try:
        name = f"{_session_id}_{event['type']}_{event['id']}.jpg"
        (_SNAPSHOT_DIR / name).write_bytes(_frame_to_jpeg(frame, quality=85))
        return f"alerts/{name}"
    except Exception:
        return None


def _frame_to_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    img = Image.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _frame_to_b64(frame: np.ndarray, quality: int = 80) -> str:
    return base64.b64encode(_frame_to_jpeg(frame, quality)).decode()


# ── REST endpoints ────────────────────────────────────────────────────────

_start_time = time.time()


@app.get("/api/health")
def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


def _get_local_ip() -> str:
    """Best-effort local LAN IP."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


@app.get("/api/system")
def system_info():
    """Return system details for the UI status bar."""
    import torch
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    with _export_lock:
        det_count = len(_detection_log)
        vlm_count = len(_vlm_log)
        report_count = len(_reports)
        alert_count = len(_alerts)
    return {
        "platform": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "device": device,
        "local_ip": _get_local_ip(),
        "models": {
            "detector": "YOLOv11n (ONNX)",
            "pose": "YOLOv11n-Pose (ONNX)",
            "action": "ST-GCN (NTU-60)",
            "vlm": "FastVLM 0.5B",
            "llm": "DeepSeek-R1:1.5B (Ollama)",
        },
        "uptime_s": round(time.time() - _start_time),
        "log_counts": {
            "detections": det_count,
            "captions": vlm_count,
            "reports": report_count,
            "alerts": alert_count,
        },
        "threat": threat_engine.stats(),
        "registry": _registry.stats(),
        "weapon_model": weapon_model_status(),
        "analysis_jobs": len(_job_store.list()),
        "log_dir": str(_LOG_DIR),
    }


@app.get("/api/presets")
def presets():
    return IP_CAMERA_PRESETS


@app.get("/api/videos")
def list_videos():
    """List video files available in the `videos/` directory."""
    items = []
    for p in sorted(_VIDEO_DIR.iterdir()):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            items.append(_video_meta(p))
    return {"videos": items, "dir": str(_VIDEO_DIR)}


@app.post("/api/videos/upload")
async def upload_video(file: UploadFile = File(...)):
    """Accept a video upload and store it in `videos/`."""
    name = _safe_video_name(file.filename or "")
    if Path(name).suffix.lower() not in VIDEO_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(sorted(VIDEO_EXTS))}",
        )

    dest = _VIDEO_DIR / name
    stem, suffix = dest.stem, dest.suffix
    n = 1
    while dest.exists():
        dest = _VIDEO_DIR / f"{stem}_{n}{suffix}"
        n += 1

    try:
        with dest.open("wb") as out:
            shutil.copyfileobj(file.file, out, length=1024 * 1024)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}")
    finally:
        await file.close()

    meta = _video_meta(dest)
    if meta["frames"] == 0 and meta["fps"] == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="File is not a readable video")
    return meta


@app.delete("/api/videos/{name}")
def delete_video(name: str):
    path = _VIDEO_DIR / _safe_video_name(name)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Video not found")
    path.unlink()
    return {"deleted": path.name}


@app.get("/api/videos/probe")
def probe_video(path: str):
    """Probe an arbitrary on-disk video path without copying it in."""
    return _video_meta(_resolve_video_path(path))


@app.get("/api/frame")
def get_frame(source: str = "local", url: str = ""):
    """Get a single frame as JPEG."""
    if source == "local":
        frame = get_camera_frame()
    else:
        resolved = str(_resolve_video_path(url)) if source == "file" else resolve_stream_url(url)
        cap = cv2.VideoCapture(resolved, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000)
        try:
            ok, bgr = cap.read()
            frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok and bgr is not None else None
        finally:
            cap.release()

    if frame is None:
        return JSONResponse({"error": "No frame available"}, status_code=503)

    jpeg = _frame_to_jpeg(frame)
    return StreamingResponse(io.BytesIO(jpeg), media_type="image/jpeg")


@app.post("/api/detect")
def detect(conf: float = 0.45, iou: float = 0.45, source: str = "local", url: str = ""):
    frame = _get_source_frame(source, url)
    if frame is None:
        return JSONResponse({"error": "No frame"}, status_code=503)
    result = run_detection(frame, conf=conf, iou=iou)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "objects": result["objects"],
        "count": result["count"],
        "time_ms": result["time_ms"],
    }
    _append_detection(entry)
    return {**entry, "fps": result["fps"]}


@app.post("/api/vlm")
def vlm_caption(prompt: str = "Describe what is happening in one sentence.",
                temperature: float = 0.0, max_tokens: int = 64,
                source: str = "local", url: str = ""):
    frame = _get_source_frame(source, url)
    if frame is None:
        return JSONResponse({"error": "No frame"}, status_code=503)
    result = run_vlm(frame, prompt=prompt, temperature=temperature, max_tokens=max_tokens)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **result,
    }
    _append_vlm(entry)
    return entry


@app.post("/api/report")
def generate_report(
    model: str = "deepseek-r1:1.5b",
    ollama_url: str = os.environ.get("OLLAMA_URL", "http://localhost:11434"),
):
    """Generate an LLM report from collected detection + VLM data."""
    from openai import OpenAI

    with _export_lock:
        det_summary = _detection_log[-20:]
        vlm_summary = _vlm_log[-10:]
        alert_summary = _alerts[-15:]

    def _fmt_det(d: dict) -> str:
        objs = ", ".join(
            f"{o['class']} ({o.get('confidence', '?')})" for o in d["objects"]
        )
        return f"[{d['timestamp']}] {objs} — {d['count']} total"

    det_text = "\n".join(_fmt_det(d) for d in det_summary) or "(no detections)"
    vlm_text = "\n".join(
        f"- [{v['timestamp']}] {v.get('text', '')}" for v in vlm_summary
    ) or "(no captions)"
    alert_text = "\n".join(
        f"- [{a['timestamp']}] {a['severity'].upper()} · {a['label']} "
        f"(score {a['score']}, rule {a['rule']}) — {a['detail']}"
        for a in alert_summary
    ) or "(no alerts raised)"

    prompt = (
        "You are Sentinel, a video recognition and incident-reporting system. "
        "Given the following alert log, detection log and scene captions from a live "
        "camera feed, write a concise incident report.\n\n"
        "Your report MUST include:\n"
        "1. **Summary** (2-3 sentences describing the scene)\n"
        "2. **Incidents** (each alert: what fired, when, and how much weight it deserves)\n"
        "3. **Key Observations** (bullet points of notable detections)\n"
        "4. **Recommended Actions** (specific actionable items, "
        "e.g. \"Review 14:32 clip for weapon alert\", \"Dispatch to entrance\")\n"
        "5. **Confidence & Caveats** (state which conclusions rest on weak signals)\n\n"
        "Rules: describe only what the logs support. Do not assert that a crime "
        "occurred — these are automated signals for human review, and both the "
        "detector and the captioner make mistakes. Flag uncertainty explicitly.\n\n"
        f"## Alerts\n{alert_text}\n\n"
        f"## Detection Log\n```\n{det_text}\n```\n\n"
        f"## Scene Captions\n{vlm_text}\n\n"
        "## Report"
    )
    try:
        client = OpenAI(base_url=f"{ollama_url.rstrip('/')}/v1", api_key="ollama")
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], max_tokens=512,
        )
        report_text = resp.choices[0].message.content or ""
    except Exception as e:
        report_text = f"Report generation failed: {e}"

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "report": report_text,
        "detection_count": len(det_summary),
        "caption_count": len(vlm_summary),
        "alert_count": len(alert_summary),
    }
    _append_report(entry)
    return entry


# ── JSON export endpoints ─────────────────────────────────────────────────

@app.get("/api/export/detections")
def export_detections():
    with _export_lock:
        data = list(_detection_log)
    return {"count": len(data), "detections": data}


@app.get("/api/export/vlm")
def export_vlm():
    with _export_lock:
        data = list(_vlm_log)
    return {"count": len(data), "captions": data}


@app.get("/api/export/reports")
def export_reports():
    with _export_lock:
        data = list(_reports)
    return {"count": len(data), "reports": data}


@app.get("/api/export/all")
def export_all():
    with _export_lock:
        return {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "detections": list(_detection_log),
            "vlm_captions": list(_vlm_log),
            "reports": list(_reports),
            "alerts": list(_alerts),
        }


# ── Threat / alert endpoints ──────────────────────────────────────────────

@app.get("/api/alerts")
def list_alerts(limit: int = 100, min_severity: str | None = None,
                unacknowledged_only: bool = False):
    """Most recent alerts, newest first."""
    with _export_lock:
        data = list(_alerts)
    if min_severity:
        floor = SEVERITY_ORDER.get(min_severity, 0)
        data = [a for a in data if SEVERITY_ORDER.get(a["severity"], 0) >= floor]
    if unacknowledged_only:
        data = [a for a in data if not a.get("acknowledged")]
    data = data[-limit:][::-1]
    return {"count": len(data), "alerts": data, "stats": threat_engine.stats()}


@app.post("/api/alerts/{alert_id}/ack")
def acknowledge_alert(alert_id: str):
    with _export_lock:
        for a in _alerts:
            if a["id"] == alert_id:
                a["acknowledged"] = True
                a["acknowledged_at"] = datetime.now(timezone.utc).isoformat()
                _flush_json(f"alerts_{_session_id}.json", _alerts)
                return {"ok": True, "alert": a}
    return JSONResponse({"ok": False, "error": "not found"}, status_code=404)


@app.get("/api/cameras")
def list_cameras():
    """Registered cameras and where they are."""
    from backend.dispatch import cameras
    return cameras()


@app.get("/api/recipients")
def list_recipients():
    """Officers on call, and how they are reached."""
    from backend.dispatch import recipients
    return {"recipients": recipients(),
            "mode": os.environ.get("SENTINEL_DISPATCH_MODE", "manual").lower(),
            "email_ready": bool(os.environ.get("SMTP_HOST", "").strip()),
            "whatsapp_ready": bool(os.environ.get("TWILIO_ACCOUNT_SID", "").strip()
                                   and os.environ.get("TWILIO_WHATSAPP_FROM", "").strip())}


@app.put("/api/recipients")
def put_recipients(body: dict):
    """Replace the recipient list. Writes config/recipients.json."""
    from backend.dispatch import save_recipients
    return save_recipients(body.get("recipients", []))


@app.get("/api/alerts/{alert_id}/dispatch")
def preview_dispatch(alert_id: str, camera_id: str = ""):
    """What WOULD be sent, and to whom. Sends nothing."""
    from backend.dispatch import dispatch
    for a in _alerts:
        if a["id"] == alert_id:
            return dispatch(a, camera_id or a.get("source", ""))
    return JSONResponse({"ok": False, "error": "not found"}, status_code=404)


@app.post("/api/alerts/{alert_id}/dispatch")
def send_dispatch(alert_id: str, body: dict | None = None):
    """Send an alert to the officers on call for that camera.

    This is the human-in-the-loop step. `force` is set by a person pressing
    send in the UI, and without it nothing leaves the building unless
    SENTINEL_DISPATCH_MODE is explicitly set to `auto`.
    """
    from backend.dispatch import dispatch
    body = body or {}
    for a in _alerts:
        if a["id"] == alert_id:
            res = dispatch(a, body.get("camera_id") or a.get("source", ""),
                           confirmed_by=body.get("confirmed_by"),
                           force=bool(body.get("force", True)))
            if res.get("sent"):
                a["dispatched_at"] = datetime.now(timezone.utc).isoformat()
                a["dispatched_to"] = [r["name"] for r in res.get("recipients", [])]
                with _export_lock:
                    _flush_json(f"alerts_{_session_id}.json", _alerts)
            return res
    return JSONResponse({"ok": False, "error": "not found"}, status_code=404)


@app.post("/api/alerts/clear")
def clear_alerts():
    with _export_lock:
        _alerts.clear()
        _flush_json(f"alerts_{_session_id}.json", _alerts)
    threat_engine.reset()
    return {"ok": True}


@app.get("/api/alerts/snapshot/{name}")
def alert_snapshot(name: str):
    """Serve a stored evidence still."""
    path = (_SNAPSHOT_DIR / name).resolve()
    if not str(path).startswith(str(_SNAPSHOT_DIR.resolve())) or not path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return StreamingResponse(io.BytesIO(path.read_bytes()), media_type="image/jpeg")


@app.get("/api/threat/config")
def get_threat_config():
    return threat_engine.config.to_dict()


@app.post("/api/threat/config")
def set_threat_config(patch: dict):
    threat_engine.config.update(patch)
    return threat_engine.config.to_dict()


@app.get("/api/export/alerts")
def export_alerts():
    with _export_lock:
        data = list(_alerts)
    return {"count": len(data), "alerts": data}


# ── Offline video analysis ────────────────────────────────────────────────

def _report_for(job) -> dict:
    """Report builder handed to the analysis worker."""
    return build_report(
        job,
        ollama_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
    )


@app.post("/api/analyze")
def analyze_video(body: dict):
    """Queue a full pass over a recorded video.

    Body: ``{"video": "<filename in videos/ or absolute path>", ...options}``.
    Returns immediately with a job; poll ``/api/analyze/{job_id}`` for progress
    and the finished report.
    """
    path = _resolve_video_path(str(body.get("video") or body.get("path") or ""))

    known = {f.name for f in dataclasses.fields(AnalysisOptions)}
    opts = AnalysisOptions(**{k: v for k, v in body.items()
                              if k in known and v is not None})
    if opts.stride < 1:
        raise HTTPException(status_code=400, detail="stride must be >= 1")

    job = start_analysis(
        video_path=str(path), video_name=path.name, options=opts,
        store=_job_store, registry=_registry, snapshot_dir=_SNAPSHOT_DIR,
        build_report=_report_for,
    )
    return job.summary()


@app.get("/api/analyze/jobs")
def list_jobs():
    return {"jobs": [j.summary() for j in _job_store.list()]}


@app.get("/api/analyze/{job_id}")
def get_job(job_id: str, include_events: bool = True):
    job = _job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict(include_events=include_events)


@app.post("/api/analyze/{job_id}/cancel")
def cancel_job(job_id: str):
    job = _job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    job.cancel()
    return {"ok": True, "status": job.status}


@app.get("/api/analyze/{job_id}/report")
def get_job_report(job_id: str, format: str = "json", narrative: bool = True):
    """The incident report for a finished job, as JSON or Markdown."""
    job = _job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status not in ("completed", "cancelled"):
        raise HTTPException(
            status_code=409,
            detail=f"Job is {job.status}; a report is available once it finishes",
        )

    if job.report is None:
        job.report = build_report(job, narrative=narrative)
    report = job.report

    if format in ("markdown", "md"):
        text = render_markdown(report)
        _append_report({"timestamp": datetime.now(timezone.utc).isoformat(),
                        "job_id": job_id, "video": job.video_name,
                        "format": "markdown", "report": text})
        return StreamingResponse(
            io.BytesIO(text.encode("utf-8")), media_type="text/markdown",
            headers={"Content-Disposition":
                     f'inline; filename="report_{job_id}.md"'},
        )
    return report


@app.get("/api/analyze/{job_id}/events")
def get_job_events(job_id: str, min_severity: str | None = None,
                   limit: int = 500):
    """Timecoded event log for a job — the raw rows behind the report."""
    job = _job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    events = job.events
    if min_severity:
        floor = SEVERITY_ORDER.get(min_severity, 0)
        events = [e for e in events if SEVERITY_ORDER.get(e["severity"], 0) >= floor]
    return {"count": len(events), "events": events[:limit]}


# ── Live-session report ───────────────────────────────────────────────────

class _SessionJob:
    """Adapter so a live session can use the same report builder as a job.

    Times are seconds since the session started rather than positions in a
    file, which is the honest reading of "when" for a live feed.
    """

    def __init__(self, alerts: list[dict], captions: list[dict]):
        self.job_id = f"session_{_session_id}"
        self.video_name = "live session"
        self.video_path = None
        self.status = "completed"
        self.options = None
        self.video = {"duration_s": round(time.time() - _start_time, 2)}
        self.progress = {}
        self.events = alerts
        self.captions = captions
        self.entities = []
        self.report = None


@app.post("/api/session/report")
def session_report(format: str = "json", narrative: bool = True):
    """Deterministic report over the current live session's alerts."""
    with _export_lock:
        alerts = list(_alerts)
        captions = [
            {"video_time_s": None, "timecode": v.get("timestamp", ""),
             "text": v.get("text", "")}
            for v in _vlm_log[-40:] if v.get("text")
        ]
    report = build_report(_SessionJob(alerts, captions), narrative=narrative)
    if format in ("markdown", "md"):
        text = render_markdown(report)
        return StreamingResponse(io.BytesIO(text.encode("utf-8")),
                                 media_type="text/markdown")
    _append_report({"timestamp": datetime.now(timezone.utc).isoformat(),
                    "kind": "session", "report": report})
    return report


# ── Identity registry ─────────────────────────────────────────────────────

@app.get("/api/registry")
def list_registry(kind: str | None = None, with_incidents_only: bool = False,
                  limit: int = 200):
    """Remembered vehicles and persons, most recently seen first."""
    items = _registry.all(kind=kind, with_incidents_only=with_incidents_only)
    return {
        "count": len(items),
        "entities": [e.summary() for e in items[:limit]],
        "stats": _registry.stats(),
    }


@app.get("/api/registry/{entity_id}")
def get_entity(entity_id: str):
    ent = _registry.get(entity_id)
    if ent is None:
        raise HTTPException(status_code=404, detail="Entity not found")
    return ent.summary()


@app.post("/api/registry/{entity_id}/note")
def annotate_entity(entity_id: str, body: dict):
    """Attach a reviewer's note — e.g. a plate read off the snapshot by eye."""
    note = str(body.get("note") or "").strip()
    if not note:
        raise HTTPException(status_code=400, detail="note is required")
    if not _registry.add_note(entity_id, note):
        raise HTTPException(status_code=404, detail="Entity not found")
    if body.get("plate"):
        _registry.set_plate(entity_id, str(body["plate"]),
                            confidence=body.get("plate_confidence"))
    return _registry.get(entity_id).summary()


@app.delete("/api/registry/{entity_id}")
def forget_entity(entity_id: str):
    """Erase a stored subject. These records are personal data."""
    if not _registry.forget(entity_id):
        raise HTTPException(status_code=404, detail="Entity not found")
    return {"ok": True, "forgotten": entity_id}


@app.post("/api/registry/prune")
def prune_registry(body: dict | None = None):
    """Enforce a retention limit: drop subjects not seen for N days."""
    body = body or {}
    days = float(body.get("older_than_days", 30))
    keep = bool(body.get("keep_with_incidents", True))
    removed = _registry.prune(days, keep_with_incidents=keep)
    return {"ok": True, "removed": removed, "stats": _registry.stats()}


@app.get("/api/weapons/status")
def weapons_status():
    """Whether a weapon-trained detector is loaded, and what it can see."""
    return weapon_model_status()


# ── WebSocket — live feed with AI ─────────────────────────────────────────

@app.websocket("/ws/feed")
async def ws_feed(
    websocket: WebSocket,
    source: str = "local",
    url: str = "",
    conf: float = 0.45,
    iou: float = 0.45,
    vlm_interval: int = 5,
    enable_det: bool = True,
    enable_vlm: bool = True,
    enable_pose: bool = True,
    enable_threat: bool = True,
    loop: bool = False,
    enable_adjudication: bool = False,
    adjudicate_frames_n: int = 4,
):
    await websocket.accept()
    cap = None
    use_local = source == "local"
    use_file = source == "file"

    # File playback bookkeeping
    total_frames = 0
    video_fps = 0.0
    frame_idx = 0

    if not use_local:
        if use_file:
            try:
                resolved = str(_resolve_video_path(url))
            except HTTPException as exc:
                await websocket.send_json({"error": str(exc.detail)})
                await websocket.close()
                return
            cap = cv2.VideoCapture(resolved)
            if not cap.isOpened():
                await websocket.send_json({"error": f"Failed to open video file: {url}"})
                await websocket.close()
                return
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
            await websocket.send_json({
                "event": "video_opened",
                "video": {
                    "name": os.path.basename(resolved),
                    "path": resolved,
                    "frames": total_frames,
                    "fps": round(video_fps, 2),
                    "duration_s": round(total_frames / video_fps, 2) if video_fps > 0 else 0,
                },
            })
        else:
            resolved = resolve_stream_url(url)
            cap = cv2.VideoCapture(resolved, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
            if not cap.isOpened():
                # Carry the resolver's reason through. "Failed to open
                # stream" is the same message for a dead link, a private
                # video, a geo-block and a missing yt-dlp, and only one of
                # those is fixed by trying again.
                why = last_stream_error()
                await websocket.send_json({
                    "error": f"Failed to open stream — {why}" if why
                             else "Failed to open stream"})
                await websocket.close()
                return

    last_vlm_time = 0.0
    last_vlm_frame = -10**9
    last_caption = ""
    last_action: dict | None = None
    # Short rolling buffer of raw frames so an alert can be adjudicated against
    # the moment it fired rather than a single still.
    recent_frames: collections.deque = collections.deque(maxlen=24)
    # The optional weapon-trained detector. load_weapon_model() is a cached
    # singleton — cheap to call again — but this loop must actually call
    # .detect() on it, which is the piece that was previously missing here:
    # the model loaded fine and reported "loaded": true from /api/weapons/
    # status, yet no frame in this websocket loop ever ran it. Only
    # backend/analyze.py's offline batch-job path did.
    #
    # Wrapped in SmoothedWeaponDetector for person-cropped inference and
    # temporal persistence (see backend/weapons.py) — real hits on small or
    # distant guns often sit just below the single-frame confirm threshold
    # (measured: 0.463-0.514 on the Florida-store clip against a 0.45
    # floor), which is exactly what made this flicker on/off between runs
    # of the same footage. One instance per connection/video, not a module
    # global — its history state is specific to this one playback session.
    _base_weapon_model = load_weapon_model()
    weapon_model = (SmoothedWeaponDetector(_base_weapon_model)
                    if _base_weapon_model is not None else None)

    try:
        while True:
            # Check for control messages (non-blocking)
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                ctrl = json.loads(msg)
                if ctrl.get("action") == "stop":
                    break
                # Allow live toggle of settings
                enable_det = ctrl.get("enable_det", enable_det)
                enable_vlm = ctrl.get("enable_vlm", enable_vlm)
                enable_pose = ctrl.get("enable_pose", enable_pose)
                enable_threat = ctrl.get("enable_threat", enable_threat)
                if isinstance(ctrl.get("threat_config"), dict):
                    threat_engine.config.update(ctrl["threat_config"])
                conf = ctrl.get("conf", conf)
                iou = ctrl.get("iou", iou)
                vlm_interval = ctrl.get("vlm_interval", vlm_interval)
                loop = ctrl.get("loop", loop)
                # Seek within a video file (0.0–1.0 of total duration)
                if use_file and cap is not None and ctrl.get("action") == "seek":
                    frac = max(0.0, min(1.0, float(ctrl.get("position", 0))))
                    if total_frames > 0:
                        frame_idx = int(total_frames * frac)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            except (asyncio.TimeoutError, Exception):
                pass

            # --- frame processing wrapped so one bad frame won't kill the socket ---
            try:
                # Get frame
                if use_local:
                    frame = get_camera_frame()
                else:
                    ok, bgr = cap.read()
                    frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok and bgr is not None else None

                    # End of a video file: loop back or finish cleanly.
                    if use_file and frame is None:
                        if loop:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            frame_idx = 0
                            await websocket.send_json({"event": "loop"})
                            await asyncio.sleep(0.01)
                            continue
                        await websocket.send_json({
                            "event": "eof",
                            "progress": {
                                "frame": frame_idx,
                                "total": total_frames,
                                "percent": 100.0,
                            },
                        })
                        break

                    if use_file:
                        frame_idx += 1

                if frame is None:
                    await asyncio.sleep(0.05)
                    continue

                recent_frames.append(frame)

                now = time.perf_counter()
                ts = datetime.now(timezone.utc).isoformat()
                payload: dict[str, Any] = {"timestamp": ts}

                if use_file:
                    payload["progress"] = {
                        "frame": frame_idx,
                        "total": total_frames,
                        "percent": round(100.0 * frame_idx / total_frames, 2) if total_frames > 0 else 0.0,
                        "time_s": round(frame_idx / video_fps, 2) if video_fps > 0 else 0.0,
                        "duration_s": round(total_frames / video_fps, 2) if video_fps > 0 and total_frames > 0 else 0.0,
                    }

                # Detection
                det_objects = []
                if enable_det:
                    det_result = run_detection(frame, conf=conf, iou=iou)
                    det_objects = det_result["objects"]
                    # The COCO detector has no gun class at all and no
                    # knife-in-a-crowded-scene reliability either — this is
                    # the only path to real firearm coverage. Merges into
                    # the same list every downstream rule (threat engine,
                    # snapshots, /api/detect) already reads.
                    if weapon_model is not None:
                        try:
                            person_boxes = [o["box"] for o in det_objects
                                           if o.get("class") == "person"]
                            det_objects = det_objects + weapon_model.detect(
                                frame, person_boxes=person_boxes, iou=iou)
                        except Exception:
                            pass
                    payload["detection"] = {
                        "objects": det_objects,
                        "count": len(det_objects),
                        "time_ms": det_result["time_ms"],
                        "fps": det_result["fps"],
                    }
                    _append_detection({"timestamp": ts, "objects": det_objects,
                                       "count": len(det_objects), "time_ms": det_result["time_ms"]})
                    # Include cumulative object frequency counts
                    with _export_lock:
                        payload["object_counts"] = dict(_object_counts)

                # Pose estimation + action recognition
                pose_keypoints = None
                pose_scores = None
                if enable_pose:
                    pose_result = run_pose(frame, conf=conf, iou=iou)
                    if pose_result is not None:
                        payload["pose"] = {
                            "persons": pose_result["persons"],
                            "count": pose_result["count"],
                            "time_ms": pose_result["time_ms"],
                            "fps": pose_result["fps"],
                        }
                        pose_keypoints = pose_result["keypoints"]
                        pose_scores = pose_result["scores"]

                        # Feed to ST-GCN action recogniser
                        action_result = run_action(
                            pose_keypoints, pose_scores,
                            img_shape=frame.shape[:2],
                        )
                        if action_result is not None:
                            last_action = action_result
                        if last_action is not None:
                            payload["action"] = last_action

                # ── Threat engine (tier 1: structured signals) ───────────
                frame_alerts: list[dict] = []
                if enable_threat:
                    # Stamp events with a position in the footage so alerts from
                    # a file are scrubbable, not just timestamped.
                    if use_file:
                        threat_engine.set_clock(
                            video_time_s=(frame_idx / video_fps) if video_fps > 0 else None,
                            frame=frame_idx,
                            source=os.path.basename(resolved),
                        )
                    else:
                        threat_engine.set_clock(
                            video_time_s=time.time() - _start_time,
                            source=("camera" if use_local else url),
                        )
                    # A file plays back as fast as the models allow, so duration
                    # rules (loitering, dwell, collision settling) must be driven
                    # by the clip's own clock or they never fire correctly.
                    engine_now = ((frame_idx / video_fps)
                                  if use_file and video_fps > 0 else None)
                    try:
                        frame_alerts += threat_engine.update(
                            detections=det_objects,
                            action=last_action,
                            pose_persons=(payload.get("pose") or {}).get("persons"),
                            frame_shape=frame.shape,
                            now=engine_now,
                        )
                    except Exception as te:
                        import logging
                        logging.warning("Threat engine error: %s", te)

                # Build AI frame with detection overlay
                ai_frame = frame.copy()
                if enable_det:
                    from detectors import YOLODetector
                    ai_frame = YOLODetector.draw(
                        ai_frame, det_result["boxes"], det_result["scores"], det_result["class_ids"]
                    )

                # Overlay pose skeleton on AI frame
                if enable_pose and pose_keypoints is not None and len(pose_keypoints) > 0:
                    from detectors import YOLOPoseDetector
                    pose_r = pose_result  # type: ignore[possibly-undefined]
                    ai_frame = YOLOPoseDetector.draw(
                        ai_frame,
                        np.array([p["box"] for p in pose_r["persons"]]),
                        pose_scores, pose_keypoints,
                        # The detector already boxed these people; drawing the
                        # pose model's own boxes on top renders each person
                        # twice, offset by a few pixels.
                        draw_boxes=not enable_det,
                    )

                # VLM (throttled) — enriched with detection context
                # For video files the loop runs as fast as the models allow, so a
                # wall-clock interval would sample only the first few seconds of
                # footage. Throttle by *video* time instead: one caption every
                # `vlm_interval` seconds of the clip's own timeline.
                if use_file and video_fps > 0:
                    vlm_due = frame_idx - last_vlm_frame >= max(1, int(vlm_interval * video_fps))
                else:
                    vlm_due = now - last_vlm_time >= vlm_interval

                if enable_vlm and vlm_due:
                    last_vlm_time = now
                    last_vlm_frame = frame_idx
                    if enable_threat and threat_engine.config.caption_screening:
                        # Incident-biased prompt: the generic "describe the scene"
                        # prompt rarely names what matters for review.
                        vlm_prompt = threat_engine.threat_prompt()
                    elif det_objects:
                        obj_detail = "; ".join(
                            f"{o['class']} ({o['confidence']:.0%})" for o in det_objects
                        )
                        vlm_prompt = (
                            f"Objects detected: {obj_detail}. "
                            f"Total: {len(det_objects)} object(s). "
                            "Describe the scene and any notable activity in one sentence."
                        )
                    else:
                        vlm_prompt = "Describe what is happening in this scene in one sentence."
                    vlm_result = run_vlm(
                        frame,
                        prompt=vlm_prompt,
                        max_tokens=80,
                    )
                    last_caption = vlm_result.get("text", "")
                    payload["vlm"] = vlm_result
                    _append_vlm({"timestamp": ts, **vlm_result})

                    # ── Threat engine (tier 2: open-vocabulary caption) ──
                    if enable_threat and last_caption:
                        try:
                            frame_alerts += threat_engine.ingest_caption(
                                last_caption,
                                now=((frame_idx / video_fps)
                                     if use_file and video_fps > 0 else None),
                            )
                        except Exception as te:
                            import logging
                            logging.warning("Caption screening error: %s", te)

                # Persist evidence + surface alerts to the client
                if frame_alerts:
                    # VLM adjudication: the skeleton model proposes, the VLM
                    # checks. Only brief, easily-confused events are worth the
                    # latency — a snatch and a friendly handover are the same
                    # geometry, and only the VLM can see the difference.
                    if enable_adjudication and enable_vlm:
                        for ev in frame_alerts:
                            if ev.get("type") not in ADJUDICATED_TYPES:
                                continue
                            try:
                                ev["adjudication"] = _adjudicate(
                                    list(recent_frames), max_frames=adjudicate_frames_n)
                            except Exception as exc:
                                import logging
                                logging.warning("Adjudication failed: %s", exc)

                    for ev in frame_alerts:
                        if threat_engine.config.save_snapshots:
                            ev["snapshot"] = _save_snapshot(ai_frame, ev)
                    _append_alerts(frame_alerts)
                    payload["alerts"] = frame_alerts
                if enable_threat:
                    payload["threat"] = threat_engine.stats()

                # Encode frames
                raw_b64 = _frame_to_b64(frame, quality=70)
                ai_b64 = _frame_to_b64(ai_frame, quality=70)
                payload["raw_frame"] = raw_b64
                payload["ai_frame"] = ai_b64
                payload["frame_bytes"] = len(raw_b64) + len(ai_b64)  # approx payload size
                if last_caption:
                    payload["caption"] = last_caption

                await websocket.send_json(payload)
            except Exception as frame_exc:
                # Log but don't kill the connection for a single frame failure
                import logging
                logging.warning("Frame processing error: %s", frame_exc)
                await asyncio.sleep(0.1)
                continue
            # Files are processed as fast as the models allow; yield only
            # enough to keep the event loop responsive to control messages.
            await asyncio.sleep(0 if use_file else 0.01)

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        # Send a meaningful error to the client before closing
        try:
            await websocket.send_json({"error": str(exc)})
            await websocket.close(code=1011)
        except Exception:
            pass
    finally:
        if cap is not None:
            cap.release()


def _get_source_frame(source: str, url: str) -> np.ndarray | None:
    if source == "local":
        return get_camera_frame()
    resolved = str(_resolve_video_path(url)) if source == "file" else resolve_stream_url(url)
    cap = cv2.VideoCapture(resolved, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000)
    try:
        ok, bgr = cap.read()
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok and bgr is not None else None
    finally:
        cap.release()


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description="Sentinel API Server")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print("Loading detector…")
    init_yolo()
    # VLM, camera, pose, and action models are all lazy-loaded on first
    # use to keep startup fast and avoid memory spikes.
    configure_vlm(args.model_path, args.model_base)
    # Warm up VLM in a background thread after 10s so the first
    # caption request doesn't stall.
    schedule_vlm_warmup(delay=0)  # load immediately
    print("Ready (VLM loading in background; camera/pose/action load on first use). Starting API server…")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
