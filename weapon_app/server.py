"""Standalone weapon-detector tester — its own server, its own frontend.

Deliberately isolated from the rest of Sentinel: no pose model, no action
recognition, no VLM, no threat engine. It loads exactly one thing,
SENTINEL_WEAPON_MODEL (or --model), and answers exactly one question per
upload: where (if anywhere) does the weapon detector fire, and at what
confidence.

This exists to let you test the detector in isolation, on a single image or
a video, without the rest of the app in the way. It is not a replacement
for `training.weapons.evaluate` (that script is the honest false-alarm
measurement); this is a manual point-and-shoot tool for spot-checking one
file at a time.

Usage
-----
    export SENTINEL_WEAPON_MODEL=checkpoints/weapons_v2.onnx
    export SENTINEL_WEAPON_CLASSES='["gun","knife"]'
    python -m weapon_app.server            # http://localhost:8010

    # or point at a different model without touching env vars:
    python -m weapon_app.server --model checkpoints/weapons_v2.onnx \
                                 --classes gun knife --port 8010
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detectors import YOLODetector  # noqa: E402

app = FastAPI(title="Sentinel — weapon detector tester")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_detector: YOLODetector | None = None
_class_names: list[str] = []
_model_path: str = ""


def _resolve_input_specs(model_path: str, nc: int) -> tuple[int, bool]:
    """Auto-detect input size and whether the export is instance-segmentation.

    Segmentation exports (e.g. yolov8s-seg) carry 32 extra mask-coefficient
    columns after the class scores. Feeding those into a plain-detection
    reader silently misreads them as extra phantom classes — the same bug
    fixed in backend/weapons.py and training/weapons/evaluate.py. Reusing
    the same detection here rather than re-copying it with different bugs.
    """
    img_size, is_seg = 640, False
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(model_path)
        out_shape = sess.get_outputs()[0].shape
        n_out = out_shape[1] if isinstance(out_shape[1], int) else None
        in_shape = sess.get_inputs()[0].shape
        if isinstance(in_shape[2], int):
            img_size = in_shape[2]
        is_seg = n_out is not None and n_out - 4 - 32 == nc
    except Exception:
        pass
    return img_size, is_seg


def load(model_path: str, class_names: list[str]) -> None:
    global _detector, _class_names, _model_path
    img_size, is_seg = _resolve_input_specs(model_path, len(class_names))
    _detector = YOLODetector(model_path, img_size=img_size,
                             nc=len(class_names) if is_seg else None)
    _class_names = class_names
    _model_path = model_path
    print(f"loaded {model_path} ({'segmentation' if is_seg else 'detection'} "
          f"export, img_size={img_size}, classes={class_names})")


def _draw(frame_bgr: np.ndarray, boxes, scores, class_ids) -> np.ndarray:
    out = frame_bgr.copy()
    for box, score, cid in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = [int(v) for v in box]
        color = (0, 0, 255) if _class_names[int(cid)] == "gun" else (0, 165, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        label = f"{_class_names[int(cid)]} {score:.2f}"
        cv2.rectangle(out, (x1, y1 - 22), (x1 + 9 * len(label), y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _b64_jpg(frame_bgr: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("ascii") if ok else ""


@app.get("/")
def index():
    return FileResponse(str(Path(__file__).parent / "index.html"))


@app.get("/api/status")
def status():
    return {
        "loaded": _detector is not None,
        "model": _model_path or None,
        "classes": _class_names,
    }


# ── Live view ────────────────────────────────────────────────────────────
#
# "Live" is a stretch on CPU: at imgsz=960 a single inference takes roughly
# 0.3-0.5s (see the box_to_poly/eval timing notes in training/weapons/
# evaluate.py), so this streams frames as fast as they can be processed
# rather than at the source video's real frame rate. Expect noticeably
# slower-than-real-time playback, not a live camera feed. It is still the
# most direct way to *watch* where the detector fires as a video plays,
# rather than reading a table of timestamps after the fact.

_LIVE_UPLOADS: dict[str, str] = {}  # id -> temp file path


@app.get("/api/local_videos")
def local_videos():
    vids = ROOT / "videos"
    if not vids.exists():
        return {"videos": []}
    return {"videos": sorted(p.name for p in vids.glob("*.mp4"))}


@app.post("/api/live/upload")
async def live_upload(file: UploadFile = File(...)):
    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        vid_id = Path(tmp.name).stem
        _LIVE_UPLOADS[vid_id] = tmp.name
    return {"id": vid_id}


def _live_frames(video_path: str, conf: float, stride: int, hold_frames: int):
    if _detector is None:
        return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    last_boxes: tuple = ((), (), ())
    hold = 0
    idx = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            idx += 1
            if idx % max(1, stride) == 0:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                boxes, scores, class_ids = _detector.detect(rgb, conf=conf, iou=0.45)
                if len(boxes):
                    last_boxes = (boxes, scores, class_ids)
                    hold = hold_frames
                elif hold > 0:
                    hold -= 1
                else:
                    last_boxes = ((), (), ())
            boxes, scores, class_ids = last_boxes
            annotated = _draw(frame_bgr, boxes, scores, class_ids) if len(boxes) else frame_bgr
            ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ok:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
    finally:
        cap.release()


@app.get("/api/live/stream_local")
def live_stream_local(name: str, conf: float = 0.45, stride: int = 3, hold_frames: int = 6):
    path = ROOT / "videos" / name
    if not path.exists():
        return JSONResponse({"error": "video not found"}, status_code=404)
    return StreamingResponse(
        _live_frames(str(path), conf, stride, hold_frames),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/live/stream/{vid_id}")
def live_stream_upload(vid_id: str, conf: float = 0.45, stride: int = 3, hold_frames: int = 6):
    path = _LIVE_UPLOADS.get(vid_id)
    if not path or not Path(path).exists():
        return JSONResponse({"error": "unknown or expired upload id"}, status_code=404)
    return StreamingResponse(
        _live_frames(path, conf, stride, hold_frames),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/api/detect/image")
async def detect_image(file: UploadFile = File(...), conf: float = Form(0.45)):
    if _detector is None:
        return JSONResponse({"error": "no model loaded"}, status_code=503)
    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        return JSONResponse({"error": "could not decode image"}, status_code=400)

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    boxes, scores, class_ids = _detector.detect(rgb, conf=conf, iou=0.45)
    annotated = _draw(frame_bgr, boxes, scores, class_ids)

    detections = [
        {"class": _class_names[int(c)] if int(c) < len(_class_names) else f"class_{c}",
         "confidence": round(float(s), 3), "box": [int(v) for v in b]}
        for b, s, c in zip(boxes, scores, class_ids)
    ]
    return {"detections": detections, "annotated_jpg_b64": _b64_jpg(annotated)}


@app.post("/api/detect/video")
async def detect_video(file: UploadFile = File(...), conf: float = Form(0.45),
                       every_s: float = Form(1.0), max_hits: int = Form(24)):
    if _detector is None:
        return JSONResponse({"error": "no model loaded"}, status_code=503)

    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return JSONResponse({"error": "could not open video"}, status_code=400)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_s = total_frames / fps if fps else 0.0

        t0 = time.perf_counter()
        n_sampled = 0
        hits = []
        t = 0.0
        while t < duration_s:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, frame_bgr = cap.read()
            t += every_s
            if not ok or frame_bgr is None:
                continue
            n_sampled += 1
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            boxes, scores, class_ids = _detector.detect(rgb, conf=conf, iou=0.45)
            if len(boxes) and len(hits) < max_hits:
                annotated = _draw(frame_bgr, boxes, scores, class_ids)
                hits.append({
                    "t_s": round(t - every_s, 2),
                    "detections": [
                        {"class": _class_names[int(c)] if int(c) < len(_class_names) else f"class_{c}",
                         "confidence": round(float(s), 3), "box": [int(v) for v in b]}
                        for b, s, c in zip(boxes, scores, class_ids)
                    ],
                    "thumb_jpg_b64": _b64_jpg(annotated, quality=70),
                })
        cap.release()
        elapsed = time.perf_counter() - t0
        return {
            "duration_s": round(duration_s, 1),
            "frames_sampled": n_sampled,
            "frames_with_hit": len(hits),
            "elapsed_s": round(elapsed, 1),
            "hits": hits,
            "truncated": len(hits) >= max_hits,
        }
    finally:
        os.unlink(tmp_path)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default=os.environ.get("SENTINEL_WEAPON_MODEL", ""))
    p.add_argument("--classes", nargs="+", default=None,
                   help="overrides SENTINEL_WEAPON_CLASSES if given")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8010)
    args = p.parse_args()

    model_path = args.model
    if not model_path:
        raise SystemExit("no model: pass --model or set SENTINEL_WEAPON_MODEL")
    if not Path(model_path).is_absolute():
        model_path = str(ROOT / model_path)
    if not Path(model_path).exists():
        raise SystemExit(f"model not found: {model_path}")

    names = args.classes
    if names is None:
        raw = os.environ.get("SENTINEL_WEAPON_CLASSES", "").strip()
        if raw.startswith("["):
            names = list(json.loads(raw))
        elif raw and Path(raw).exists():
            names = [ln.strip() for ln in open(raw) if ln.strip()]
    if not names:
        raise SystemExit("no class names: pass --classes or set SENTINEL_WEAPON_CLASSES")

    load(model_path, [str(n) for n in names])

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
