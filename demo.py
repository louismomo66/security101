"""
FastVLM Speed Test — Local camera + upload UI.

Uses OpenCV for local camera access (auto-starts, no browser permission needed).

Tabs:
  • Camera — live OpenCV feed, capture & analyse with VLM
  • Upload — drag-and-drop an image, then analyse
  • IP Camera — connect to YouTube Live / RTSP / MJPEG / HLS streams
  • Object Detection — YOLO11n via ONNX Runtime (live OpenCV stream)
  • Live Analysis — detect + pose + VLM caption
  • AI Chat — OpenAI gpt-4o-mini (requires OPENAI_API_KEY in .env)
  • Reasoning (Offline) — DeepSeek-R1 via Ollama (local, no API key)

Usage:
    python demo.py --model-path checkpoints/llava-fastvithd_0.5b_stage3
"""
from __future__ import annotations

import os
import time
import argparse
import threading

from dotenv import load_dotenv
load_dotenv()  # load .env before anything reads os.environ

import cv2
import numpy as np
import torch
import gradio as gr
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from PIL import Image

from camera import OpenCVCamera
from detectors import COCO_CLASSES, YOLODetector
from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
)

_lock = threading.Lock()

# Shared frame buffer for decoupled raw feed
_live_frame_buf: np.ndarray | None = None
_live_frame_lock = threading.Lock()


# ── YOLO singletons ──────────────────────────────────────────────────────

_yolo = None


def _ensure_yolo():
    global _yolo
    if _yolo is None:
        onnx_path = os.path.join(os.path.dirname(__file__), 'yolo11n.onnx')
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(
                f'yolo11n.onnx not found at {onnx_path}. '
                'Export it with: python -c "from ultralytics import YOLO; '
                "YOLO('yolo11n.pt').export(format='onnx',imgsz=640)\"")
        _yolo = YOLODetector(onnx_path)
        print(f'YOLO loaded from {onnx_path}')
    return _yolo




# ── Local camera singleton ────────────────────────────────────────────────

_camera: OpenCVCamera | None = None


def _ensure_camera() -> OpenCVCamera:
    global _camera
    if _camera is None:
        _camera = OpenCVCamera(device_id=0, width=640, height=480)
    return _camera


def get_live_frame():
    """Return the latest OpenCV camera frame (numpy RGB), or None."""
    cam = _ensure_camera()
    return cam.read() if cam.is_open else None


# ── Model helpers ─────────────────────────────────────────────────────────

def load_model(model_path, model_base=None):
    model_path = os.path.expanduser(model_path)
    gen_cfg = os.path.join(model_path, "generation_config.json")
    gen_cfg_hidden = os.path.join(model_path, ".generation_config.json")
    renamed = False
    if os.path.exists(gen_cfg):
        os.rename(gen_cfg, gen_cfg_hidden)
        renamed = True

    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name, device="mps"
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    if renamed:
        os.rename(gen_cfg_hidden, gen_cfg)
    return tokenizer, model, image_processor


def run_inference(image, prompt, temperature, max_tokens):
    """Run model on a PIL Image.  Returns (text, stats_str, status_str)."""
    if image is None:
        return "", "", "No image provided"
    if not prompt or not prompt.strip():
        prompt = "Briefly describe what is happening."

    try:
        qs = prompt
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates["qwen_2"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        input_ids = torch.as_tensor(
            tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        ).unsqueeze(0).to(torch.device("mps"))

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        image_tensor = process_images([image], image_processor, model.config)[0]

        with _lock, torch.inference_mode():
            t0 = time.perf_counter()
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half(),
                image_sizes=[image.size],
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                max_new_tokens=int(max_tokens),
                use_cache=True,
            )
            t1 = time.perf_counter()

        text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        dt = t1 - t0
        n = output_ids.shape[1] - input_ids.shape[1]
        tps = n / dt if dt > 0 else 0
        stats = f"Time: {dt:.2f}s  |  Tokens: {n}  |  Speed: {tps:.1f} tok/s"
        return text, stats, "Done"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return "", "", f"Error: {e}"


def upload_analyse(image, prompt, temp, tokens):
    """Analyse an uploaded / webcam-captured image."""
    if image is None:
        yield "", "", "No image provided"
        return
    yield "", "", "Processing…"
    text, stats, status = run_inference(image, prompt, temp, tokens)
    yield text, stats, status


# ── YOLO helpers ──────────────────────────────────────────────────────────

def _run_yolo(frame, conf, iou_thresh):
    """Run YOLO detection on one RGB frame."""
    yolo = _ensure_yolo()
    t0 = time.perf_counter()
    boxes, scores, class_ids = yolo.detect(frame, conf=conf, iou=iou_thresh)
    t1 = time.perf_counter()
    annotated = YOLODetector.draw(frame, boxes, scores, class_ids)
    dt = t1 - t0
    n = len(boxes)
    stats = f"Time: {dt*1000:.0f}ms  |  Objects: {n}  |  FPS: {1/dt:.1f}"
    labels = ", ".join(
        f"{COCO_CLASSES[c]} ({s:.0%})" for c, s in zip(class_ids, scores)
    ) if n else "No objects detected"
    return annotated, stats, labels



# ── IP Camera helper ──────────────────────────────────────────────────────

# Public MJPEG / RTSP demo streams (no auth required)
IP_CAMERA_PRESETS = {
    "Walworth Road, London": "https://www.youtube.com/live/8JCk5M_xrBs",
    "Times Square, New York": "https://www.youtube.com/live/eJ7ZkQ5TC08",
    "Shibuya Crossing, Tokyo": "https://www.youtube.com/live/DjdUEyjx8GM",
    "Jackson Hole Town Square": "https://www.youtube.com/live/psfFJR3vZ78",
    "Big Buck Bunny (HLS test)": "https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8",
    "Tears of Steel (HLS test)": "https://demo.unified-streaming.com/k8s/features/stable/video/tears-of-steel/tears-of-steel.ism/.m3u8",
    "Custom URL": "",
}


def _resolve_stream_url(url: str) -> str:
    """If *url* is a YouTube link, use yt-dlp to extract the real stream URL.
    For everything else just return the URL unchanged."""
    if not url:
        return url
    import re
    if re.match(r"https?://(www\.)?(youtube\.com|youtu\.be)/", url):
        try:
            import yt_dlp
            ydl_opts = {
                "format": "best[ext=mp4][height<=720]/best[height<=720]/best",
                "quiet": True,
                "no_warnings": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                resolved = info.get("url", "")
                if resolved:
                    print(f"[yt-dlp] resolved → {info.get('title', 'unknown')}")
                    return resolved
        except Exception as exc:
            print(f"[yt-dlp] failed: {exc}")
    return url


def _grab_ip_frame(url: str):
    """Open a network stream, grab one frame, release immediately.
    Returns an RGB numpy array or None."""
    if not url or not url.strip():
        return None
    url = _resolve_stream_url(url)
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000)
    try:
        if not cap.isOpened():
            return None
        ok, bgr = cap.read()
        if not ok or bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def _stream_ip_frames(url: str, interval: float = 0.2):
    """Generator: continuously grab frames from a network stream."""
    if not url or not url.strip():
        return
    url = _resolve_stream_url(url)
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000)
    try:
        while cap.isOpened():
            ok, bgr = cap.read()
            if not ok or bgr is None:
                break
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            time.sleep(interval)
    finally:
        cap.release()


# ── LLM Report helper ─────────────────────────────────────────────────────

def _generate_report(log_text: str, captions: list[str], model_name: str, base_url: str) -> str:
    """Ask the LLM to generate a short surveillance-style report from logs."""
    if not log_text.strip() and not captions:
        return "No data collected yet — start the live feed first."

    caption_block = "\n".join(f"- {c}" for c in captions[-10:]) if captions else "(none yet)"
    prompt = (
        "You are Sentinel, a video recognition and reporting system. "
        "Given the following detection log and scene captions from a live camera feed, "
        "write a concise surveillance report (3-6 sentences). "
        "Mention key objects, activity patterns, and any notable events.\n\n"
        f"## Detection Log (latest entries)\n```\n{log_text}\n```\n\n"
        f"## Scene Captions\n{caption_block}\n\n"
        "## Report"
    )
    try:
        client = OpenAI(
            base_url=f"{base_url.rstrip('/')}/v1",
            api_key="ollama",
        )
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        )
        return resp.choices[0].message.content or "(empty response)"
    except Exception as e:
        return f"Report generation failed: {e}"


# ── Gradio UI ─────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(
        title="Sentinel — Video Recognition",
        theme=gr.themes.Base(primary_hue="orange", neutral_hue="stone"),
        css="""
            .sentinel-title { font-size: 2.2em !important; font-weight: 800 !important; margin-bottom: 0 !important; }
            .sentinel-sub   { opacity: 0.7; margin-top: 0 !important; }
        """,
    ) as demo:

        # ── Header ────────────────────────────────────────────────────
        gr.Markdown("# Sentinel", elem_classes="sentinel-title")
        gr.Markdown(
            "**Video Recognition & Reporting** — "
            "real-time object detection, pose estimation, VLM captioning & LLM reports",
            elem_classes="sentinel-sub",
        )

        with gr.Tabs():

            # ══════════════════════════════════════════════════════════
            #  MAIN TAB — Live Feed
            # ══════════════════════════════════════════════════════════
            with gr.Tab("Live Feed", id="main"):

                with gr.Row():
                    # ── Left: big AI annotated feed ───────────────────
                    with gr.Column(scale=3):
                        la_ai_frame = gr.Image(
                            label="AI View",
                            interactive=False,
                            height=520,
                        )
                        with gr.Row():
                            la_status = gr.Textbox(label="Status", lines=1, value="Idle", scale=2)
                            la_det_stats = gr.Textbox(label="Detection", lines=1, scale=3)

                    # ── Right: small raw feed + controls ──────────────
                    with gr.Column(scale=2):
                        la_raw_frame = gr.Image(
                            label="Raw Feed",
                            interactive=False,
                            height=200,
                        )
                        raw_timer = gr.Timer(value=0.1, active=False)

                        # -- Source settings (accordion) --
                        with gr.Accordion("Source", open=True):
                            la_source = gr.Radio(
                                choices=["Local Camera", "IP Camera"],
                                value="Local Camera", label="Feed Source",
                            )
                            la_preset = gr.Dropdown(
                                choices=list(IP_CAMERA_PRESETS.keys()),
                                value="Walworth Road, London",
                                label="IP Preset", visible=False,
                            )
                            la_url = gr.Textbox(
                                value="", label="Stream URL",
                                placeholder="YouTube / RTSP / MJPEG URL",
                                visible=False,
                            )
                            with gr.Row():
                                la_start = gr.Button("▶ Start", variant="primary")
                                la_stop = gr.Button("⏹ Stop", variant="stop")

                        # -- Detection settings (accordion) --
                        with gr.Accordion("Detection Settings", open=False):
                            gr.Markdown("**Models**")
                            with gr.Row():
                                la_tog_det = gr.Checkbox(value=True, label="Detection")
                                la_tog_vlm = gr.Checkbox(value=True, label="VLM")
                            la_conf = gr.Slider(0.1, 1.0, value=0.45, step=0.05, label="Confidence")
                            la_iou = gr.Slider(0.1, 1.0, value=0.45, step=0.05, label="IoU / NMS")
                            la_vlm_interval = gr.Slider(3, 15, value=5, step=1, label="VLM Interval (s)")

                        # -- VLM Caption --
                        la_caption = gr.Textbox(label="Scene Caption (VLM)", lines=2)
                        la_vlm_stats = gr.Textbox(label="VLM Speed", lines=1)

                # ── Detection Log ─────────────────────────────────────
                with gr.Row():
                    la_log = gr.Textbox(
                        label="Detection Log (rolling)",
                        lines=8, max_lines=8,
                        interactive=False,
                        scale=3,
                    )

                    # ── Report panel ──────────────────────────────────
                    with gr.Column(scale=2):
                        with gr.Accordion("LLM Report", open=True):
                            rpt_model = gr.Dropdown(
                                choices=["deepseek-r1:1.5b"],
                                value="deepseek-r1:1.5b", label="Report Model",
                            )
                            rpt_url = gr.Textbox(
                                value=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
                                label="Ollama URL",
                            )
                            rpt_btn = gr.Button("📝 Generate Report", variant="primary")
                            rpt_output = gr.Textbox(label="Report", lines=8, interactive=False)

                # ── Hidden state for report data ──────────────────────
                rpt_captions = gr.State([])

                # ── Wiring ────────────────────────────────────────────

                def _on_source_change(source):
                    show_ip = source == "IP Camera"
                    return gr.update(visible=show_ip), gr.update(visible=show_ip)

                la_source.change(
                    fn=_on_source_change,
                    inputs=[la_source],
                    outputs=[la_preset, la_url],
                )

                def _on_la_preset(preset):
                    return IP_CAMERA_PRESETS.get(preset, "")

                la_preset.change(fn=_on_la_preset, inputs=[la_preset], outputs=[la_url])

                # -- Independent raw feed via Timer --
                def _poll_raw_frame():
                    with _live_frame_lock:
                        f = _live_frame_buf
                    return f

                raw_timer.tick(fn=_poll_raw_frame, outputs=[la_raw_frame])

                def live_analysis_stream(source, url, conf, iou_thresh, vlm_interval, en_det, en_vlm):
                    """Generator: AI overlay (detection every frame, VLM every Ns)."""
                    global _live_frame_buf
                    cap = None
                    use_local = source == "Local Camera"
                    _empty = None, "", "", "", "", []

                    if not use_local:
                        if not url or not url.strip():
                            yield *_empty, "Enter a stream URL first"
                            return
                        yield *_empty, "Resolving stream…"
                        resolved = _resolve_stream_url(url)
                        cap = cv2.VideoCapture(resolved, cv2.CAP_FFMPEG)
                        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
                        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
                        if not cap.isOpened():
                            yield *_empty, "Failed to open stream"
                            return

                    yield *_empty, "Running…"

                    log_lines: list[str] = []
                    captions: list[str] = []
                    last_vlm_time = 0.0
                    last_caption = ""
                    last_vlm_stats = ""

                    try:
                        while True:
                            if use_local:
                                frame = get_live_frame()
                            else:
                                assert cap is not None
                                ok, bgr = cap.read()
                                if not ok or bgr is None:
                                    break
                                frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                            if frame is None:
                                time.sleep(0.05)
                                continue

                            # -- Push raw frame to shared buffer (non-blocking) --
                            with _live_frame_lock:
                                _live_frame_buf = frame

                            now = time.perf_counter()

                            # -- YOLO detection (every frame — fast) --
                            boxes, scores, class_ids = [], [], []
                            t_det = 0.0
                            if en_det:
                                yolo = _ensure_yolo()
                                t0 = time.perf_counter()
                                boxes, scores, class_ids = yolo.detect(frame, conf=conf, iou=iou_thresh)
                                t_det = time.perf_counter() - t0
                            n_det = len(boxes)
                            det_labels = [
                                f"{COCO_CLASSES[c]} ({s:.0%})"
                                for c, s in zip(class_ids, scores)
                            ]

                            # -- Build AI annotated frame --
                            ai_frame = frame.copy()
                            if en_det:
                                ai_frame = YOLODetector.draw(ai_frame, boxes, scores, class_ids)

                            active = [m for m, on in [("Det", en_det), ("VLM", en_vlm)] if on]
                            fps = 1 / t_det if t_det > 0 else 0
                            det_stats = (f"Det: {t_det*1000:.0f}ms | Objects: {n_det} | FPS: {fps:.0f}" if en_det else "") + f"  [{'+'.join(active) or 'none'}]"

                            # -- log --
                            ts = time.strftime("%H:%M:%S")
                            line = f"[{ts}] {', '.join(det_labels)}" if det_labels else f"[{ts}] (nothing)"
                            log_lines.append(line)
                            if len(log_lines) > 60:
                                log_lines = log_lines[-60:]
                            log_text = "\n".join(log_lines[-8:])

                            # -- VLM captioning (throttled) --
                            if en_vlm and now - last_vlm_time >= vlm_interval:
                                last_vlm_time = now
                                obj_hint = ", ".join(
                                    COCO_CLASSES[c] for c in class_ids
                                ) if n_det else "nothing specific"
                                vlm_prompt = (
                                    f"Objects detected: {obj_hint}. "
                                    "Describe what is happening in one sentence."
                                )
                                image_pil = Image.fromarray(frame)
                                text, stats, _ = run_inference(image_pil, vlm_prompt, 0.0, 64)
                                last_caption = text
                                last_vlm_stats = stats
                                if text:
                                    captions.append(f"[{ts}] {text}")

                            yield (
                                ai_frame, det_stats, last_caption,
                                last_vlm_stats, log_text, captions, "Running…",
                            )
                            time.sleep(0.01)

                    finally:
                        if cap is not None:
                            cap.release()
                        with _live_frame_lock:
                            _live_frame_buf = None

                    yield (
                        None, "", last_caption, last_vlm_stats,
                        "\n".join(log_lines[-8:]), captions, "Stream ended",
                    )

                def _start_feed(source, url, conf, iou_thresh, vlm_interval):
                    """Activate the raw-feed timer, then delegate to the AI generator."""
                    return gr.update(active=True)

                def _stop_feed():
                    global _live_frame_buf
                    with _live_frame_lock:
                        _live_frame_buf = None
                    return gr.update(active=False)

                la_start.click(fn=_start_feed, inputs=[la_source, la_url, la_conf, la_iou, la_vlm_interval], outputs=[raw_timer])
                la_event = la_start.click(
                    fn=live_analysis_stream,
                    inputs=[la_source, la_url, la_conf, la_iou, la_vlm_interval, la_tog_det, la_tog_vlm],
                    outputs=[la_ai_frame, la_det_stats, la_caption, la_vlm_stats, la_log, rpt_captions, la_status],
                )
                la_stop.click(fn=_stop_feed, cancels=[la_event], outputs=[raw_timer])

                def _gen_report(log_text, captions_state, model_name, base_url):
                    yield "Generating report…"
                    report = _generate_report(log_text, captions_state, model_name, base_url)
                    yield report

                rpt_btn.click(
                    fn=_gen_report,
                    inputs=[la_log, rpt_captions, rpt_model, rpt_url],
                    outputs=[rpt_output],
                )

            # ══════════════════════════════════════════════════════════
            #  DEBUG TABS
            # ══════════════════════════════════════════════════════════
            with gr.Tab("Debug Tools"):
                gr.Markdown("*Individual model testing — use these to verify each component works correctly.*")

                with gr.Tabs():

                    # -- Camera debug --
                    with gr.Tab("Camera"):
                        gr.Markdown("**Local camera** — capture a frame and run VLM.")
                        with gr.Row():
                            with gr.Column(scale=1):
                                dbg_cam_img = gr.Image(label="Live Camera", type="numpy", interactive=False)
                                dbg_cam_timer = gr.Timer(value=0.1)
                                dbg_cam_timer.tick(fn=get_live_frame, outputs=[dbg_cam_img])
                                dbg_cam_prompt = gr.Textbox(value="What is happening? Answer in one short sentence.", label="Prompt", lines=2)
                                with gr.Row():
                                    dbg_cam_temp = gr.Slider(0.0, 1.0, value=0.0, step=0.1, label="Temp")
                                    dbg_cam_tokens = gr.Slider(8, 128, value=32, step=8, label="Tokens")
                                dbg_cam_btn = gr.Button("📷 Capture & Analyse", variant="primary")
                            with gr.Column(scale=1):
                                dbg_cam_status = gr.Textbox(label="Status", lines=1, value="Idle")
                                dbg_cam_out_img = gr.Image(label="Analysed Frame", interactive=False)
                                dbg_cam_output = gr.Textbox(label="Output", lines=5)
                                dbg_cam_stats = gr.Textbox(label="Speed", lines=1)

                        def dbg_cam_analyse(frame, prompt, temp, tokens):
                            if frame is None:
                                yield None, "", "", "No frame"
                                return
                            yield frame, "", "", "Processing…"
                            image = Image.fromarray(frame)
                            text, stats, status = run_inference(image, prompt, temp, tokens)
                            yield frame, text, stats, status

                        dbg_cam_btn.click(
                            fn=dbg_cam_analyse,
                            inputs=[dbg_cam_img, dbg_cam_prompt, dbg_cam_temp, dbg_cam_tokens],
                            outputs=[dbg_cam_out_img, dbg_cam_output, dbg_cam_stats, dbg_cam_status],
                        )

                    # -- Upload debug --
                    with gr.Tab("Upload"):
                        gr.Markdown("Upload an image and click **Analyse**.")
                        with gr.Row():
                            with gr.Column(scale=1):
                                dbg_up_img = gr.Image(sources=["upload", "webcam"], type="pil", label="Image")
                                dbg_up_prompt = gr.Textbox(value="Describe the image.", label="Prompt", lines=2)
                                with gr.Row():
                                    dbg_up_temp = gr.Slider(0.0, 1.0, value=0.2, step=0.1, label="Temp")
                                    dbg_up_tokens = gr.Slider(16, 512, value=128, step=16, label="Tokens")
                                dbg_up_btn = gr.Button("Analyse", variant="primary")
                            with gr.Column(scale=1):
                                dbg_up_status = gr.Textbox(label="Status", lines=1, value="Ready")
                                dbg_up_output = gr.Textbox(label="Output", lines=10)
                                dbg_up_stats = gr.Textbox(label="Speed", lines=1)

                        dbg_up_btn.click(
                            fn=upload_analyse,
                            inputs=[dbg_up_img, dbg_up_prompt, dbg_up_temp, dbg_up_tokens],
                            outputs=[dbg_up_output, dbg_up_stats, dbg_up_status],
                        )

                    # -- IP Camera debug --
                    with gr.Tab("IP Camera"):
                        gr.Markdown("**IP Camera** — YouTube / RTSP / MJPEG / HLS.")
                        with gr.Row():
                            with gr.Column(scale=1):
                                dbg_ip_preset = gr.Dropdown(choices=list(IP_CAMERA_PRESETS.keys()), value="Custom URL", label="Preset")
                                dbg_ip_url = gr.Textbox(value="", label="Stream URL", placeholder="rtsp://... or YouTube URL")
                                dbg_ip_prompt = gr.Textbox(value="What is happening? Answer in one short sentence.", label="Prompt", lines=2)
                                with gr.Row():
                                    dbg_ip_temp = gr.Slider(0.0, 1.0, value=0.0, step=0.1, label="Temp")
                                    dbg_ip_tokens = gr.Slider(8, 256, value=64, step=8, label="Tokens")
                                with gr.Row():
                                    dbg_ip_grab = gr.Button("📷 Grab Frame", variant="primary")
                                    dbg_ip_stream = gr.Button("▶ Continuous", variant="secondary")
                                    dbg_ip_stop = gr.Button("⏹ Stop", variant="stop")
                            with gr.Column(scale=1):
                                dbg_ip_status = gr.Textbox(label="Status", lines=1, value="Idle")
                                dbg_ip_frame = gr.Image(label="Frame", interactive=False)
                                dbg_ip_output = gr.Textbox(label="Output", lines=5)
                                dbg_ip_stats = gr.Textbox(label="Speed", lines=1)

                        dbg_ip_preset.change(fn=lambda p: IP_CAMERA_PRESETS.get(p, ""), inputs=[dbg_ip_preset], outputs=[dbg_ip_url])

                        def dbg_ip_grab_fn(url, prompt, temp, tokens):
                            if not url or not url.strip():
                                yield None, "", "", "Enter a stream URL"
                                return
                            yield None, "", "", "Connecting…"
                            frame = _grab_ip_frame(url)
                            if frame is None:
                                yield None, "", "", "Failed to grab frame"
                                return
                            yield frame, "", "", "Processing…"
                            image = Image.fromarray(frame)
                            text, stats, status = run_inference(image, prompt, temp, tokens)
                            yield frame, text, stats, status

                        def dbg_ip_stream_fn(url, prompt, temp, tokens):
                            if not url or not url.strip():
                                yield None, "", "", "Enter a stream URL"
                                return
                            yield None, "", "", "Connecting…"
                            for frame in _stream_ip_frames(url, interval=0.05):
                                image = Image.fromarray(frame)
                                text, stats, status = run_inference(image, prompt, temp, tokens)
                                yield frame, text, stats, status
                            yield None, "", "", "Stream ended"

                        dbg_ip_grab.click(fn=dbg_ip_grab_fn, inputs=[dbg_ip_url, dbg_ip_prompt, dbg_ip_temp, dbg_ip_tokens], outputs=[dbg_ip_frame, dbg_ip_output, dbg_ip_stats, dbg_ip_status])
                        dbg_ip_cont_ev = dbg_ip_stream.click(fn=dbg_ip_stream_fn, inputs=[dbg_ip_url, dbg_ip_prompt, dbg_ip_temp, dbg_ip_tokens], outputs=[dbg_ip_frame, dbg_ip_output, dbg_ip_stats, dbg_ip_status])
                        dbg_ip_stop.click(fn=None, cancels=[dbg_ip_cont_ev])

                    # -- Object Detection debug --
                    with gr.Tab("Object Detection"):
                        gr.Markdown("**YOLO11n** — live detection from local camera.")
                        with gr.Row():
                            with gr.Column(scale=1):
                                dbg_det_live = gr.Image(label="Camera", type="numpy", interactive=False)
                                dbg_det_timer = gr.Timer(value=0.1)
                                with gr.Row():
                                    dbg_det_conf = gr.Slider(0.1, 1.0, value=0.5, step=0.05, label="Confidence")
                                    dbg_det_iou = gr.Slider(0.1, 1.0, value=0.45, step=0.05, label="IoU")
                                dbg_det_upload = gr.Image(sources=["upload"], type="numpy", label="Or upload")
                                dbg_det_up_btn = gr.Button("Detect", variant="secondary")
                            with gr.Column(scale=1):
                                dbg_det_status = gr.Textbox(label="Status", lines=1, value="Idle")
                                dbg_det_output = gr.Image(label="Detections", interactive=False)
                                dbg_det_stats = gr.Textbox(label="Stats", lines=1)
                                dbg_det_labels = gr.Textbox(label="Objects", lines=5)

                        def dbg_det_tick(conf, iou_t):
                            frame = get_live_frame()
                            if frame is None:
                                return None, None, "", "", "Waiting…"
                            try:
                                ann, stats, labels = _run_yolo(frame, conf, iou_t)
                                return frame, ann, stats, labels, "Detecting…"
                            except Exception as e:
                                return frame, None, "", "", f"Error: {e}"

                        dbg_det_timer.tick(fn=dbg_det_tick, inputs=[dbg_det_conf, dbg_det_iou], outputs=[dbg_det_live, dbg_det_output, dbg_det_stats, dbg_det_labels, dbg_det_status])

                        def dbg_det_up_fn(image, conf, iou_t):
                            if image is None:
                                return None, "", "", "No image"
                            try:
                                ann, stats, labels = _run_yolo(image, conf, iou_t)
                                return ann, stats, labels, "Done"
                            except Exception as e:
                                return None, "", "", f"Error: {e}"

                        dbg_det_up_btn.click(fn=dbg_det_up_fn, inputs=[dbg_det_upload, dbg_det_conf, dbg_det_iou], outputs=[dbg_det_output, dbg_det_stats, dbg_det_labels, dbg_det_status])

                    # -- AI Chat debug --
                    with gr.Tab("AI Chat"):
                        gr.Markdown(
                            "**AI Chat** powered by OpenAI `gpt-4o-mini`.  "
                            "Set `OPENAI_API_KEY` in your `.env` file."
                        )
                        chat_model = gr.Dropdown(
                            choices=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
                            value="gpt-4o-mini", label="Model",
                        )
                        chatbot = gr.Chatbot(label="Conversation", height=400, autoscroll=True)
                        with gr.Row():
                            chat_input = gr.Textbox(label="Message", placeholder="Type a message…", scale=4, lines=1)
                            chat_send = gr.Button("Send", variant="primary", scale=1)
                        chat_clear = gr.Button("🗑 Clear Chat")

                        def chat_respond(message, history, model_name):
                            api_key = os.environ.get("OPENAI_API_KEY", "")
                            if not api_key:
                                history = history + [
                                    {"role": "user", "content": message},
                                    {"role": "assistant", "content": "⚠️ Set OPENAI_API_KEY in your .env file and restart."},
                                ]
                                return history, ""
                            if not message or not message.strip():
                                return history, ""
                            history = history + [{"role": "user", "content": message}]
                            try:
                                client = OpenAI(api_key=api_key)
                                messages: list[ChatCompletionMessageParam] = [
                                    {"role": "system", "content": "You are a helpful assistant."},
                                ] + [
                                    {"role": m["role"], "content": m["content"]}  # type: ignore[typeddict-item]
                                    for m in history
                                ]
                                resp = client.chat.completions.create(model=model_name, messages=messages, max_tokens=1024)
                                reply = resp.choices[0].message.content
                            except Exception as e:
                                reply = f"⚠️ Error: {e}"
                            history = history + [{"role": "assistant", "content": reply}]
                            return history, ""

                        chat_send.click(fn=chat_respond, inputs=[chat_input, chatbot, chat_model], outputs=[chatbot, chat_input])
                        chat_input.submit(fn=chat_respond, inputs=[chat_input, chatbot, chat_model], outputs=[chatbot, chat_input])
                        chat_clear.click(fn=lambda: ([], ""), outputs=[chatbot, chat_input])

                    # -- Reasoning debug --
                    with gr.Tab("Reasoning (Offline)"):
                        gr.Markdown(
                            "**DeepSeek-R1** — local reasoning via Ollama (no API key)."
                        )
                        with gr.Row():
                            r1_model = gr.Dropdown(choices=["deepseek-r1:1.5b"], value="deepseek-r1:1.5b", label="Model", scale=1)
                            r1_url = gr.Textbox(value=os.environ.get("OLLAMA_URL", "http://localhost:11434"), label="Ollama URL", scale=2)
                        r1_chatbot = gr.Chatbot(label="Conversation", height=400, autoscroll=True)
                        with gr.Row():
                            r1_input = gr.Textbox(label="Message", placeholder="Ask the reasoning model…", scale=4, lines=1)
                            r1_send = gr.Button("Send", variant="primary", scale=1)
                        r1_clear = gr.Button("🗑 Clear Chat")
                        r1_status = gr.Textbox(label="Status", lines=1, value="Ready")

                        def _parse_r1_response(text):
                            import re
                            think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
                            if think_match:
                                thinking = think_match.group(1).strip()
                                answer = text[think_match.end():].strip()
                                return thinking, answer
                            return "", text.strip()

                        def r1_respond(message, history, model_name, base_url):
                            if not message or not message.strip():
                                return history, "", "Ready"
                            history = history + [{"role": "user", "content": message}]
                            try:
                                client = OpenAI(base_url=f"{base_url.rstrip('/')}/v1", api_key="ollama")
                                api_messages: list[ChatCompletionMessageParam] = [
                                    {"role": "system", "content": "You are a helpful reasoning assistant. Think step by step."},
                                ]
                                for m in history:
                                    api_messages.append({"role": m["role"], "content": m["content"]})  # type: ignore[typeddict-item]
                                resp = client.chat.completions.create(model=model_name, messages=api_messages, max_tokens=2048)
                                raw = resp.choices[0].message.content
                                thinking, answer = _parse_r1_response(raw)
                                reply = f"💭 **Thinking:**\n{thinking}\n\n---\n\n**Answer:**\n{answer}" if thinking else answer
                                status = "Done"
                            except Exception as e:
                                err = str(e)
                                if "Connection refused" in err or "ConnectError" in err:
                                    reply = "⚠️ Cannot connect to Ollama. Run `ollama serve` then `ollama pull deepseek-r1:1.5b`."
                                else:
                                    reply = f"⚠️ Error: {e}"
                                status = "Error"
                            history = history + [{"role": "assistant", "content": reply}]
                            return history, "", status

                        r1_send.click(fn=r1_respond, inputs=[r1_input, r1_chatbot, r1_model, r1_url], outputs=[r1_chatbot, r1_input, r1_status])
                        r1_input.submit(fn=r1_respond, inputs=[r1_input, r1_chatbot, r1_model, r1_url], outputs=[r1_chatbot, r1_input, r1_status])
                        r1_clear.click(fn=lambda: ([], "", "Ready"), outputs=[r1_chatbot, r1_input, r1_status])

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    print("Loading model…")
    tokenizer, model, image_processor = load_model(args.model_path, args.model_base)
    print("Model loaded.  Starting UI…")

    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=True)
