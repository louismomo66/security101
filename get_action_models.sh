#!/usr/bin/env bash
# Download YOLO Pose + ST-GCN action recognition models for Sentinel.
#
# Usage:
#   bash get_action_models.sh
#
# Models downloaded:
#   1. yolo11n-pose.onnx  — YOLO11n pose estimation (skeleton extraction)
#   2. checkpoints/stgcn_ntu60_joint.pth — ST-GCN (PYSKL, NTU60, 2D COCO skeleton, joint modality)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── 1. YOLO11n-pose ONNX ──────────────────────────────────────────────────

POSE_ONNX="yolo11n-pose.onnx"

if [ -f "$POSE_ONNX" ]; then
    echo "✓ $POSE_ONNX already exists, skipping."
else
    echo "→ Exporting $POSE_ONNX from ultralytics..."
    # Install ultralytics temporarily if needed, export, then the .onnx stays
    python3 -c "
from ultralytics import YOLO
model = YOLO('yolo11n-pose.pt')
model.export(format='onnx', imgsz=640, simplify=True)
print('Export complete.')
" 2>&1
    if [ -f "$POSE_ONNX" ]; then
        echo "✓ $POSE_ONNX exported successfully."
    else
        echo "✗ Failed to export $POSE_ONNX. Install ultralytics: pip install ultralytics"
        exit 1
    fi
fi

# ── 2. ST-GCN checkpoint (pyskl, NTU60, HRNet 2D, Joint) ─────────────────

mkdir -p checkpoints
STGCN_CKPT="checkpoints/stgcn_ntu60_joint.pth"
STGCN_URL="http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_pyskl_ntu60_xsub_hrnet/j.pth"

if [ -f "$STGCN_CKPT" ]; then
    echo "✓ $STGCN_CKPT already exists, skipping."
else
    echo "→ Downloading ST-GCN checkpoint from pyskl model zoo..."
    curl -L -o "$STGCN_CKPT" "$STGCN_URL"
    if [ -f "$STGCN_CKPT" ]; then
        echo "✓ $STGCN_CKPT downloaded successfully."
    else
        echo "✗ Failed to download ST-GCN checkpoint."
        exit 1
    fi
fi

# ── 3. ST-GCN NTU120 checkpoint — adds the crime classes ─────────────────
#
# Same architecture, same COCO-17 2D skeleton layout, 120 classes instead of
# 60. The extra classes are what make street theft detectable:
#
#   A57  touch other person's pocket   (also in NTU60)  -> pickpocketing
#   A109 grab other person's stuff                      -> snatch theft
#   A107 wield knife towards other person
#   A110 shoot at other person with a gun
#   A106 hit other person with something
#
# The runtime reads the class count off the checkpoint head, so switching is
# just a matter of pointing ACTION_MODEL_PATH at this file.

STGCN120_CKPT="checkpoints/stgcn_ntu120_joint.pth"
STGCN120_URL="http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_pyskl_ntu120_xsub_hrnet/j.pth"

if [ -f "$STGCN120_CKPT" ]; then
    echo "✓ $STGCN120_CKPT already exists, skipping."
else
    echo "→ Downloading ST-GCN NTU120 checkpoint (adds theft/weapon classes)..."
    curl -L -o "$STGCN120_CKPT" "$STGCN120_URL"
    if [ -f "$STGCN120_CKPT" ]; then
        echo "✓ $STGCN120_CKPT downloaded successfully."
    else
        echo "✗ Failed to download NTU120 checkpoint."
        exit 1
    fi
fi

echo ""
echo "All action recognition models are ready!"
echo "  • Pose:        $POSE_ONNX"
echo "  • Action (60):  $STGCN_CKPT"
echo "  • Action (120): $STGCN120_CKPT   ← recommended"
echo ""
echo "To use the NTU120 model (pickpocketing, snatch theft, knife, gun):"
echo "    export ACTION_MODEL_PATH=$STGCN120_CKPT"
echo "    ./dev.sh"
