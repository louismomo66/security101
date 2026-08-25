"""Run a LoRA-fine-tuned LFM2.5-VL adapter through transformers + PEFT.

Why this exists alongside `backend/lfm_vlm.py`
---------------------------------------------
`lfm_vlm.py` loads a GGUF export through `llama_cpp`, which is the right shape
for a deployed edge device. It is the wrong shape for *evaluating* a checkpoint
you just trained:

  - The training output is a LoRA adapter directory (~40 MB). Turning that into
    a GGUF means merging it into the base model (3.2 GB) and converting
    (another 1.5 GB) — 4.7 GB of intermediates to test 40 MB of weights.
  - `llama-cpp-python` builds from source on Apple Silicon.
  - Its `Llava15ChatHandler` targets the LLaVA-1.5 vision stack. `lfm2_vl` is a
    different architecture, and Unsloth's own usage note points at
    `llama-mtmd-cli` rather than that handler.

This module loads the adapter directly on top of the base model. It is slower
per frame than GGUF and that is fine — evaluation runs over a few hundred
frames, not a live stream.

Configuration
-------------
  * ``SENTINEL_LFM_ADAPTER`` — path to the adapter directory (the one holding
    ``adapter_model.safetensors``).
  * ``SENTINEL_LFM_BASE`` — base model id, default ``LiquidAI/LFM2.5-VL-1.6B``.
    Must match what the adapter was trained on; PEFT will load a mismatched
    pair without complaint and produce nonsense.
"""
from __future__ import annotations

import os
import threading

import numpy as np

_model = None
_processor = None
_lock = threading.Lock()

DEFAULT_BASE = "LiquidAI/LFM2.5-VL-1.6B"


def _resolve_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def init_lfm_peft():
    """Load base + adapter once. Returns True when ready."""
    global _model, _processor
    if _model is not None:
        return True

    adapter = os.environ.get("SENTINEL_LFM_ADAPTER", "").strip()
    if not adapter:
        raise RuntimeError("SENTINEL_LFM_ADAPTER is not set")
    if not os.path.isdir(adapter):
        raise RuntimeError(f"adapter directory not found: {adapter}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    base_id = os.environ.get("SENTINEL_LFM_BASE", "").strip() or DEFAULT_BASE
    device = _resolve_device()
    # No bitsandbytes: it is CUDA-only, and this path exists to run on a laptop.
    # fp16 of a 1.6B model is ~3.2 GB, which fits in unified memory.
    dtype = torch.float16 if device != "cpu" else torch.float32

    print(f"Loading {base_id} on {device} ({dtype})…")
    model = AutoModelForImageTextToText.from_pretrained(
        base_id, dtype=dtype, trust_remote_code=True)
    print(f"Applying adapter {adapter}…")
    model = PeftModel.from_pretrained(model, adapter)
    model.to(device).eval()

    _processor = AutoProcessor.from_pretrained(base_id, trust_remote_code=True)
    _model = model
    print("LFM adapter ready.")
    return True


def run_vlm_lfm_peft(image, prompt: str = "", max_tokens: int = 96,
                     temperature: float = 0.0) -> dict:
    """Same call signature as `backend.lfm_vlm.run_vlm_lfm`.

    Accepts a PIL image or an RGB ndarray, returns {"text": ...}.
    """
    import torch
    from PIL import Image

    if _model is None:
        init_lfm_peft()

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGB")

    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": prompt},
    ]}]
    text = _processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = _processor(image, text, add_special_tokens=False,
                        return_tensors="pt").to(_model.device)

    # Serialised: one model instance, and callers may be threaded.
    with _lock, torch.no_grad():
        out = _model.generate(**inputs, max_new_tokens=int(max_tokens),
                              do_sample=temperature > 0,
                              temperature=temperature if temperature > 0 else None,
                              use_cache=True)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return {"text": _processor.decode(gen, skip_special_tokens=True).strip()}
