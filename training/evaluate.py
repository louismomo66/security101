"""Stage 3 — evaluate a trained checkpoint on a held-out split.

    python -m training.evaluate \
        --checkpoint checkpoints/stgcn_crime.pth \
        --clips training/data/clips --split test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from action.stgcn import RecognizerGCN                    # noqa: E402
from training.dataset import SkeletonClipDataset          # noqa: E402
from training.labels import NUM_CLASSES                   # noqa: E402
from training.metrics import evaluate, format_report      # noqa: E402
from training.train import pick_device                    # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate a trained crime ST-GCN")
    p.add_argument("--checkpoint", default="checkpoints/stgcn_crime.pth")
    p.add_argument("--clips", default="training/data/clips")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--clip-len", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="auto")
    p.add_argument("--show-errors", type=int, default=10,
                   help="print N misclassified clips with their source video")
    args = p.parse_args()

    device = pick_device(args.device)
    ds = SkeletonClipDataset(args.clips, args.split, args.clip_len, augment=False)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = RecognizerGCN(num_classes=NUM_CLASSES)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state.get("state_dict", state))
    model.to(device).eval()

    stats = evaluate(model, dl, device)
    print(f"{args.split} split — {len(ds)} clips\n")
    print(format_report(stats))

    if args.show_errors:
        print(f"\nSample misclassifications (up to {args.show_errors}):")
        shown = 0
        with torch.no_grad():
            for i in range(len(ds)):
                if shown >= args.show_errors:
                    break
                x, y = ds[i]
                pred = model(x.unsqueeze(0).to(device)).argmax(1).item()
                if pred != y:
                    from training.labels import CRIME_ACTIONS
                    m = ds.meta(i)
                    print(f"  {Path(m['video']).name} "
                          f"{m['start']:.1f}-{m['end']:.1f}s  "
                          f"true={CRIME_ACTIONS[y]} pred={CRIME_ACTIONS[pred]}")
                    shown += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
