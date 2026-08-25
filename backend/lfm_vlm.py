"""LFM2.5-VL-1.6B (fine-tuned on UCF Crime) as an optional local VLM backend.

Opt-in via SENTINEL_VLM_BACKEND=lfm (see backend/vlm_backend.py). Unlike
backend/claude_vlm.py, this stays fully offline — it's a local GGUF
checkpoint run through llama.cpp, not a cloud API — so it doesn't trade away
the project's offline/edge-first design the way the Claude backend does.

Requires:
  * pip install llama-cpp-python
  * SENTINEL_LFM_MODEL pointing at a GGUF export (see
    training/vlm/colab_train_lfm_ucf.ipynb for how to produce one)
  * SENTINEL_LFM_MMPROJ pointing at the matching mmproj (vision projector)
    GGUF file, if the export produced one separately from the language
    model file — check what your specific export produced; Unsloth's GGUF
    export path for vision models was not verified end-to-end here (see
    training/vlm/README.md).

Do not trust this checkpoint's answers over FastVLM's without first running
`python -m training.vlm.evaluate` against this project's own labelled
footage. A UCF Crime holdout score does not transfer automatically —
that's the same rule every other model in this project has had to earn its
way past.
"""
from __future__ import annotations

import base64
import io
import os

import numpy as np
from PIL import Image

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Llava15ChatHandler

        model_path = os.environ.get("SENTINEL_LFM_MODEL", "").strip()
        mmproj_path = os.environ.get("SENTINEL_LFM_MMPROJ", "").strip() or None
        if not model_path:
            raise RuntimeError("SENTINEL_LFM_MODEL is not set")

        chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path) if mmproj_path else None
        _llm = Llama(model_path=model_path, chat_handler=chat_handler,
                    n_ctx=2048, n_threads=os.cpu_count(), verbose=False)
    return _llm


def run_vlm_lfm(image, prompt: str = "", temperature: float = 0.0,
                max_tokens: int = 64) -> dict:
    """Same contract as backend.models.run_vlm: returns {"text": str[, "error": str]}."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGB")

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    data_uri = f"data:image/jpeg;base64,{b64}"

    if not prompt:
        prompt = "Briefly describe what is happening."

    try:
        llm = _get_llm()
        resp = llm.create_chat_completion(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=int(max_tokens),
            temperature=temperature,
        )
        text = resp["choices"][0]["message"]["content"]
        return {"text": (text or "").strip()}
    except Exception as exc:
        return {"text": "", "error": f"{type(exc).__name__}: {exc}"}
