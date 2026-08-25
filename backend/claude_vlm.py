"""Claude (Anthropic API) as an optional VLM backend.

Opt-in only, via SENTINEL_VLM_BACKEND=claude (see backend/vlm_backend.py).
The default remains the local FastVLM model (backend.models.run_vlm), which
runs fully offline on-device.

Routing through Claude instead requires:

  * ANTHROPIC_API_KEY set in your own environment — get one from
    console.anthropic.com. Never hardcode it here or commit it; this file
    only reads it via the Anthropic SDK's normal env-var lookup.
  * Internet access at inference time, for every single call.
  * `pip install anthropic`.

This is a real architectural trade, not a free upgrade. CLAUDE.md documents
the offline/edge-first design as deliberate: cameras in this project are
expected to keep working through Uganda's load-shedding, which is exactly
what a cloud API dependency breaks. It also costs money per frame verified,
where the local model costs only latency. Use this where the trade is
acceptable — a monitoring station with reliable power and internet, or
research/testing against hard cases — not as a blanket replacement for
every camera in the field.

Interface matches backend.models.run_vlm exactly (same signature, same
{"text": str} return shape) so it drops in anywhere run_vlm is used,
including verify_held_object's vlm_fn parameter and the tier-2/tier-3
captioning and adjudication paths.
"""
from __future__ import annotations

import base64
import io
import os

import numpy as np
from PIL import Image

_client = None


def _get_client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    return _client


def run_vlm_claude(image, prompt: str = "", temperature: float = 0.0,
                   max_tokens: int = 64) -> dict:
    """Same contract as backend.models.run_vlm: returns {"text": str[, "error": str]}."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGB")

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    if not prompt:
        prompt = "Briefly describe what is happening."
    model = os.environ.get("SENTINEL_CLAUDE_MODEL", "claude-haiku-4-5-20251001")

    try:
        client = _get_client()
        resp = client.messages.create(
            model=model,
            max_tokens=int(max_tokens),
            temperature=temperature,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64",
                                                 "media_type": "image/jpeg", "data": b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        return {"text": text.strip()}
    except Exception as exc:
        return {"text": "", "error": f"{type(exc).__name__}: {exc}"}
