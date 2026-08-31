"""Sentinel backend — shared model singletons and inference helpers.

All heavy models (YOLO, FastVLM, Pose, ST-GCN) are loaded once and reused.
"""
from __future__ import annotations

import os
import re
import time
import threading
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from camera import OpenCVCamera
from detectors import COCO_CLASSES, YOLODetector, YOLOPoseDetector
from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
)

_lock = threading.Lock()


def _resolve_device() -> str:
    """Pick the best available torch device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# ── Singletons ────────────────────────────────────────────────────────────

_yolo: YOLODetector | None = None
_yolo_pose: YOLOPoseDetector | None = None
_camera: OpenCVCamera | None = None
_action_recognizer = None  # ActionRecognizer | None

# VLM globals (set by init_vlm)
tokenizer: Any = None
vlm_model: Any = None
image_processor: Any = None
_vlm_config: dict[str, Any] | None = None  # stored for lazy init


def init_yolo() -> YOLODetector:
    global _yolo
    if _yolo is None:
        p = os.path.join(os.path.dirname(os.path.dirname(__file__)), "yolo11n.onnx")
        _yolo = YOLODetector(p)
    return _yolo




def init_yolo_pose() -> YOLOPoseDetector | None:
    """Initialise YOLO11n-pose. Returns None if ONNX file not found."""
    global _yolo_pose
    if _yolo_pose is None:
        p = os.path.join(os.path.dirname(os.path.dirname(__file__)), "yolo11n-pose.onnx")
        if not os.path.exists(p):
            return None
        _yolo_pose = YOLOPoseDetector(p)
    return _yolo_pose


def init_action(device: str | None = None):
    """Initialise ST-GCN action recogniser. Returns None if checkpoint not found.

    Defaults to the best available device. This used to hard-default to CPU
    while the VLM used `_resolve_device()`, so the action model quietly ran on
    CPU even on Apple Silicon.
    """
    global _action_recognizer
    if device is None:
        device = _resolve_device()
    if _action_recognizer is None:
        root = os.path.dirname(os.path.dirname(__file__))
        # ACTION_MODEL_PATH lets a fine-tuned crime model replace the stock
        # NTU60 one without a code change. Class names come from the sidecar
        # <checkpoint>.labels.json written by training/train.py.
        ckpt = os.environ.get("ACTION_MODEL_PATH") or os.path.join(
            root, "checkpoints", "stgcn_ntu60_joint.pth")
        if not os.path.isabs(ckpt):
            ckpt = os.path.join(root, ckpt)
        if not os.path.exists(ckpt):
            return None
        from action import ActionRecognizer
        try:
            _action_recognizer = ActionRecognizer(ckpt, device=device)
        except Exception as exc:
            # Not every ST-GCN op is implemented on every MPS build; falling
            # back is far better than losing action recognition entirely.
            if device == "cpu":
                raise
            import logging
            logging.warning("Action model failed on %s (%s) — falling back to CPU",
                            device, exc)
            _action_recognizer = ActionRecognizer(ckpt, device="cpu")
    return _action_recognizer


def init_camera() -> OpenCVCamera:
    global _camera
    if _camera is None:
        _camera = OpenCVCamera(device_id=0, width=640, height=480)
    return _camera


def configure_vlm(model_path: str, model_base: str | None = None):
    """Store VLM config for lazy loading. Does NOT load the model."""
    global _vlm_config
    _vlm_config = {"model_path": model_path, "model_base": model_base}


_vlm_warmup_scheduled = False

def schedule_vlm_warmup(delay: float = 10.0):
    """Start a background thread that loads the VLM after *delay* seconds.

    This keeps startup instant while ensuring the model is warm
    by the time the user actually needs a caption.
    Idempotent: only the first call schedules a thread.
    """
    global _vlm_warmup_scheduled
    if _vlm_warmup_scheduled:
        return
    _vlm_warmup_scheduled = True

    def _warmup():
        import time as _time
        _time.sleep(delay)
        print(f"[warmup] Loading VLM in background after {delay}s delay…")
        init_vlm()
    t = threading.Thread(target=_warmup, daemon=True)
    t.start()


def init_vlm():
    """Lazy-load VLM on first use. Returns True if model is ready."""
    global tokenizer, vlm_model, image_processor
    if vlm_model is not None:
        return True
    if _vlm_config is None:
        return False
    model_path = os.path.expanduser(_vlm_config["model_path"])
    model_base = _vlm_config["model_base"]
    gen_cfg = os.path.join(model_path, "generation_config.json")
    gen_cfg_hidden = os.path.join(model_path, ".generation_config.json")
    renamed = False
    if os.path.exists(gen_cfg):
        try:
            os.rename(gen_cfg, gen_cfg_hidden)
            renamed = True
        except OSError:
            pass  # read-only filesystem (e.g. Docker :ro volume) — skip rename

    print("Loading VLM (first use)…")
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    _device = _resolve_device()
    tokenizer, vlm_model, image_processor, _ = load_pretrained_model(
        model_path, model_base, model_name, device=_device
    )
    vlm_model.generation_config.pad_token_id = tokenizer.pad_token_id

    if renamed:
        os.rename(gen_cfg_hidden, gen_cfg)
    print("VLM loaded.")
    return True


# ── Inference helpers ─────────────────────────────────────────────────────

def get_camera_frame() -> np.ndarray | None:
    cam = init_camera()
    return cam.read() if cam.is_open else None


def run_detection(frame: np.ndarray, conf: float = 0.45, iou: float = 0.45) -> dict:
    yolo = init_yolo()
    t0 = time.perf_counter()
    boxes, scores, class_ids = yolo.detect(frame, conf=conf, iou=iou)
    dt = time.perf_counter() - t0
    objects = [
        {"class": COCO_CLASSES[c], "confidence": round(float(s), 3),
         "box": [int(x) for x in b]}
        for b, s, c in zip(boxes, scores, class_ids)
    ]
    annotated = YOLODetector.draw(frame, boxes, scores, class_ids)
    return {
        "objects": objects,
        "count": len(objects),
        "time_ms": round(dt * 1000, 1),
        "fps": round(1 / dt, 1) if dt > 0 else 0,
        "annotated": annotated,
        "boxes": boxes,
        "scores": scores,
        "class_ids": class_ids,
    }




def run_pose(frame: np.ndarray, conf: float = 0.45, iou: float = 0.45) -> dict | None:
    """Run YOLO Pose detection. Returns None if model not available."""
    pose = init_yolo_pose()
    if pose is None:
        return None
    t0 = time.perf_counter()
    boxes, scores, keypoints = pose.detect(frame, conf=conf, iou=iou)
    dt = time.perf_counter() - t0
    annotated = YOLOPoseDetector.draw(frame, boxes, scores, keypoints)
    persons = [
        {"confidence": round(float(s), 3), "box": [int(x) for x in b],
         "keypoints": kpts.tolist()}
        for b, s, kpts in zip(boxes, scores, keypoints)
    ]
    return {
        "persons": persons,
        "count": len(persons),
        "time_ms": round(dt * 1000, 1),
        "fps": round(1 / dt, 1) if dt > 0 else 0,
        "annotated": annotated,
        "keypoints": keypoints,   # raw ndarray for action recogniser
        "scores": scores,
    }


def run_action(keypoints: np.ndarray | None,
               scores: np.ndarray | None = None,
               img_shape: tuple[int, int] | None = None) -> dict | None:
    """Feed one frame of keypoints to ST-GCN action recogniser.

    `img_shape` is the source frame's (height, width). Passing it lets the
    recognizer normalize keypoints the same way the model was trained, which
    preserves the distance between people — the signal that distinguishes an
    interaction from two people simply standing near each other.

    Returns None if model not available or buffer not yet full.
    """
    rec = init_action()
    if rec is None:
        return None
    return rec.update(keypoints, scores, img_shape=img_shape)


def run_vlm(image: Image.Image | np.ndarray, prompt: str = "",
            temperature: float = 0.0, max_tokens: int = 64) -> dict:
    if vlm_model is None:
        if not init_vlm():
            return {"text": "", "error": "VLM not configured"}
    if not prompt:
        prompt = "Briefly describe what is happening."

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGB")

    qs = prompt
    if vlm_model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates["qwen_2"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    _device = _resolve_device()
    input_ids = torch.as_tensor(
        tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    ).unsqueeze(0).to(torch.device(_device))

    image_tensor = process_images([image], image_processor, vlm_model.config)[0]

    with _lock, torch.inference_mode():
        t0 = time.perf_counter()
        output_ids = vlm_model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half(),
            image_sizes=[image.size],
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            max_new_tokens=int(max_tokens),
            use_cache=True,
        )
        dt = time.perf_counter() - t0

    text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    n_tokens = output_ids.shape[1] - input_ids.shape[1]
    tps = n_tokens / dt if dt > 0 else 0
    return {
        "text": text,
        "time_s": round(dt, 3),
        "tokens": n_tokens,
        "tokens_per_s": round(tps, 1),
    }


def resolve_stream_url(url: str) -> str:
    if not url:
        return url
    if re.match(r"https?://(www\.)?(youtube\.com|youtu\.be)/", url):
        try:
            import yt_dlp
            # No `format` filter. A selector like "best[ext=mp4][height<=720]"
            # raises "Requested format is not available" on live streams,
            # because YouTube serves those as HLS — every one of Shibuya's
            # eight formats is m3u8 and none is a progressive mp4. Enumerate
            # instead and choose, so a live stream never fails at selection.
            ydl_opts = {
                "quiet": True, "no_warnings": True,
                # Pin the player client. yt-dlp's current default (ANDROID_VR)
                # returns URLs that answer 403 Forbidden to any request that
                # is not its own player, so resolution "succeeds" and OpenCV
                # then fails to open a link that looks perfectly valid. The
                # android client serves a plain progressive mp4 that ffmpeg
                # fetches with no special headers — which is what OpenCV needs.
                "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)

                usable = [
                    f for f in (info.get("formats") or [])
                    if f.get("url")
                    and f.get("vcodec") not in (None, "none")
                    and (f.get("height") or 0) <= 720
                ]
                # Order by what OpenCV can actually open, not by pixels.
                # HLS first: a live stream has nothing else, and it carries
                # audio and video in one url. Then progressive (audio+video
                # muxed), because YouTube serves those unrestricted, where
                # the higher-resolution video-only formats are client-bound
                # and 403. A working 360p beats a 720p that never opens.
                def _rank(f: dict) -> tuple:
                    return (
                        2 if "m3u8" in (f.get("protocol") or "") else 0,
                        1 if f.get("acodec") not in (None, "none") else 0,
                        f.get("height") or 0,
                        f.get("tbr") or 0,
                    )

                for f in sorted(usable, key=_rank, reverse=True):
                    if _url_is_fetchable(f["url"]):
                        return f["url"]

                for key in ("url", "manifest_url"):
                    if info.get(key) and _url_is_fetchable(info[key]):
                        return info[key]
                _stream_error(
                    url,
                    f"yt-dlp found {len(usable)} video format(s) but none could "
                    f"be fetched — YouTube may have changed its player, or the "
                    f"video is private, geo-blocked or age-restricted")
        except ImportError:
            # The one failure that looks nothing like its cause: without
            # yt-dlp the raw watch-page URL is handed to OpenCV, which cannot
            # open HTML and reports "Failed to open stream" — a missing
            # dependency wearing a network error's clothes.
            _stream_error(url, "yt-dlp is not installed "
                               "(python -m pip install yt-dlp)")
        except Exception as exc:
            # Dead or geo-blocked links are routine; the reason matters and
            # was previously discarded, leaving the same generic stream error
            # for an expired link, a private video and a bad network.
            _stream_error(url, f"{type(exc).__name__}: {exc}")
    return url


def _url_is_fetchable(url: str, timeout: float = 6.0) -> bool:
    """Does this url actually serve bytes?

    Worth the ~200ms. A YouTube media url that 403s is indistinguishable
    from a good one by inspection, and handing it to OpenCV turns a
    diagnosable "403 Forbidden" into a bare "Failed to open stream". Asking
    for the first byte here means a format that cannot be played is skipped
    while alternatives remain, instead of failing the whole request.
    """
    import urllib.error
    import urllib.request
    try:
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status < 400
    except urllib.error.HTTPError as exc:
        # 416 means the server rejected the range but the url is live.
        return exc.code == 416
    except Exception:
        return False


_LAST_STREAM_ERROR: dict[str, str] = {}


def _stream_error(url: str, reason: str) -> None:
    _LAST_STREAM_ERROR["reason"] = reason
    _LAST_STREAM_ERROR["url"] = url
    print(f"[stream] could not resolve {url}: {reason}", flush=True)


def last_stream_error() -> str:
    """Why the most recent stream resolution failed, for the API to report."""
    return _LAST_STREAM_ERROR.get("reason", "")


IP_CAMERA_PRESETS = {
    "Walworth Road, London": "https://www.youtube.com/live/8JCk5M_xrBs",
    "Times Square, New York": "https://www.youtube.com/live/eJ7ZkQ5TC08",
    "Shibuya Crossing, Tokyo": "https://www.youtube.com/live/DjdUEyjx8GM",
    "Jackson Hole Town Square": "https://www.youtube.com/live/psfFJR3vZ78",
    "Big Buck Bunny (HLS test)": "https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8",
    "Tears of Steel (HLS test)": "https://demo.unified-streaming.com/k8s/features/stable/video/tears-of-steel/tears-of-steel.ism/.m3u8",
}
