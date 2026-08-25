#!/usr/bin/env bash
# ── Sentinel native dev mode ─────────────────────────────────────────────────
# Runs backend (with MPS GPU + camera) and frontend (Next.js dev server)
# side by side. Use this instead of Docker for local Mac development.
#
# Usage:  ./dev.sh
# Stop:   Ctrl-C (kills both processes)
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"

# ── Weapon detector ───────────────────────────────────────────────────────
# weapons_v1.onnx (Sohas, 2026-08-10) is retired, see training/weapons/README.md.
# It never passed `training.weapons.evaluate`: at every threshold quiet enough
# for ordinary streets it also detected nothing in any clip known to contain
# a weapon. Quiet because blind. Measured 2026-08-15 at conf 0.25 on
# weapon-free footage: pistol fired on 7% of frames (mostly motorcycle fuel
# tanks and handlebars). Root cause was the training data: Sohas pistols have
# a median box of 43% of frame against ~0.4% in our footage.
#
# weapons_v2.onnx (yolov8s-seg, gun/knife, polygon labels, 493-image Roboflow
# set + 1,100 hard negatives) passed the gate on 2026-08-21, measured with
# `training.weapons.evaluate` against 158 held-out frames from clips NOT used
# as training negatives (different timestamps, same source videos):
#   conf 0.15  1.9 false alarms / 100 frames (gun)
#   conf 0.25  1.3 / 100
#   conf 0.35  1.3 / 100
#   conf 0.45  0.0 / 100   <- quiet enough, matches threat.py's weapon_conf default
# At conf 0.45 it still fires on the clearest confirmed-gun footage (Florida
# store, long gun, 1080p: 3/16 and 1/34 sampled frames) but misses the
# jewellery-store raid and far-field night footage entirely. Recall is weak,
# especially on `gun` (28 training instances vs 83 for `knife`) — this clears
# the false-alarm bar, it is not a strong detector. Re-evaluate before raising
# confidence in it; do not lower weapon_conf below 0.45 without rerunning
# `training.weapons.evaluate`, thresholds below that were not clean.

export SENTINEL_WEAPON_MODEL="$ROOT/checkpoints/weapons_v2.onnx"
export SENTINEL_WEAPON_CLASSES='["gun","knife"]'

# ── VLM backend (tier-2 captioning, tier-3 adjudication, weapon-hold verify) ─
# Default (unset): the local FastVLM model, fully offline. This is the
# design this project is built around — see the top of this file's header
# comment and CLAUDE.md: cameras are expected to keep working through
# Uganda's load-shedding, which an internet-dependent VLM cannot do.
#
# export SENTINEL_VLM_BACKEND="claude"   # opt-in: route the same calls
# export ANTHROPIC_API_KEY="..."         # through the Claude API instead.
# export SENTINEL_CLAUDE_MODEL="claude-haiku-4-5-20251001"  # optional override
#
# Requires internet + an API key + pip-installing `anthropic`, and costs
# money per frame verified where the local model costs only latency. See
# backend/claude_vlm.py for the full trade before turning this on for
# anything other than a monitoring station with reliable power/internet, or
# testing against hard cases the local 0.5B model gets wrong.

# ── Backend ───────────────────────────────────────────────────────────────
echo "▶ Starting backend (MPS GPU + camera)…"
cd "$ROOT"
python -m backend.server \
  --model-path checkpoints/llava-fastvithd_0.5b_stage3 \
  --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# ── Frontend ──────────────────────────────────────────────────────────────
echo "▶ Starting frontend dev server…"
cd "$ROOT/frontend"
npm run dev &
FRONTEND_PID=$!

# ── Cleanup on exit ──────────────────────────────────────────────────────
cleanup() {
  echo ""
  echo "Shutting down…"
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null
  wait "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null
}
trap cleanup EXIT INT TERM

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Backend:  http://localhost:8000"
echo "  Frontend: http://localhost:3000"
echo "  GPU:      MPS (Apple Silicon)"
echo "  Camera:   native access"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

wait
