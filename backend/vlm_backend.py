"""Select which VLM backend answers run_vlm() calls across the pipeline.

Default (no env var set): the local, offline FastVLM model
(backend.models.run_vlm). This is the project's baseline — no network
access or API key required, and what CLAUDE.md's offline/edge-first design
assumes.

Two opt-in alternatives:

  SENTINEL_VLM_BACKEND=claude — route every call (tier-2 captioning, tier-3
  adjudication, weapon-hold verification) through the Claude API instead
  (backend.claude_vlm.run_vlm_claude). Trades offline operation and per-call
  cost for general-purpose reasoning quality — see backend/claude_vlm.py.

  SENTINEL_VLM_BACKEND=lfm — route the same calls through a local
  LFM2.5-VL-1.6B checkpoint fine-tuned on UCF Crime
  (backend.lfm_vlm.run_vlm_lfm). Stays fully offline, unlike the Claude
  option, but is an unproven checkpoint until it's been run through
  training/vlm/evaluate.py against this project's own footage — see
  training/vlm/README.md before trusting it over FastVLM.
"""
from __future__ import annotations

import os


def get_vlm_fn():
    backend = os.environ.get("SENTINEL_VLM_BACKEND", "local").strip().lower()
    if backend == "claude":
        from backend.claude_vlm import run_vlm_claude
        return run_vlm_claude
    if backend == "lfm":
        from backend.lfm_vlm import run_vlm_lfm
        return run_vlm_lfm
    from backend.models import run_vlm
    return run_vlm
