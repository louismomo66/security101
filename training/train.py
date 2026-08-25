"""Stage 2 — fine-tune ST-GCN on the crime taxonomy.

Transfers the NTU60 backbone (which already knows human motion) and fits a new
classification head. With a few hundred clips per class this converges in
minutes on a single GPU.

Usage
-----
    python -m training.train \
        --clips training/data/clips \
        --init checkpoints/stgcn_ntu60_joint.pth \
        --out checkpoints/stgcn_crime.pth \
        --epochs 40

Outputs `--out` plus `<out>.labels.json`, which the runtime picks up
automatically (see action/recognizer.py::_discover_labels).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from action.stgcn import RecognizerGCN                    # noqa: E402
from training.dataset import SkeletonClipDataset          # noqa: E402
from training.labels import CRIME_ACTIONS, NUM_CLASSES    # noqa: E402
from training.metrics import evaluate, format_report      # noqa: E402


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_model(init_checkpoint: str | None, device: str,
                dropout: float, freeze_stages: int) -> RecognizerGCN:
    """Create the model, transferring backbone weights when available.

    The NTU60 head has 60 outputs and ours has ~8, so the head is deliberately
    left at its random initialisation; only backbone tensors are transferred.
    """
    model = RecognizerGCN(num_classes=NUM_CLASSES, dropout=dropout)

    if init_checkpoint:
        raw = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
        sd = raw.get("state_dict", raw)
        sd = {k.replace("cls_head.fc_cls.", "cls_head.fc."): v for k, v in sd.items()}

        own = model.state_dict()
        transferred, skipped = {}, []
        for k, v in sd.items():
            if k in own and own[k].shape == v.shape:
                transferred[k] = v
            else:
                skipped.append(k)
        model.load_state_dict(transferred, strict=False)
        print(f"Transferred {len(transferred)} tensors from {init_checkpoint}")
        head_skipped = [k for k in skipped if k.startswith("cls_head")]
        if head_skipped:
            print(f"  head reinitialised ({len(head_skipped)} tensors) — expected, "
                  f"class count differs")
        other = [k for k in skipped if not k.startswith("cls_head")]
        if other:
            print(f"  warning: {len(other)} backbone tensors did not match: "
                  f"{other[:4]}{'…' if len(other) > 4 else ''}")

    # Freezing early stages helps when the labelled set is small: the low-level
    # motion features transfer, and there isn't enough data to re-fit them.
    if freeze_stages > 0:
        frozen = 0
        for i, block in enumerate(model.backbone.gcn):
            if i < freeze_stages:
                for prm in block.parameters():
                    prm.requires_grad = False
                    frozen += 1
        print(f"Froze the first {freeze_stages} backbone stage(s)")

    return model.to(device)


def main() -> int:
    p = argparse.ArgumentParser(description="Fine-tune ST-GCN for crime actions")
    p.add_argument("--clips", default="training/data/clips")
    p.add_argument("--init", default="checkpoints/stgcn_ntu60_joint.pth",
                   help="NTU60 checkpoint to transfer from ('' to train from scratch)")
    p.add_argument("--out", default="checkpoints/stgcn_crime.pth")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--clip-len", type=int, default=100)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--freeze-stages", type=int, default=0)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--balance", choices=["weights", "sampler", "both", "none"],
                   default="weights")
    p.add_argument("--early-stop", type=int, default=12,
                   help="stop after N epochs without macro-F1 improvement")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device(args.device)
    print(f"Device: {device}")

    train_ds = SkeletonClipDataset(args.clips, "train", args.clip_len, augment=True)
    try:
        val_ds = SkeletonClipDataset(args.clips, "val", args.clip_len, augment=False)
    except FileNotFoundError:
        print("No val split found — training without validation. Add val rows to "
              "your annotation CSV; without them you cannot tell overfitting from "
              "learning.")
        val_ds = None

    counts = train_ds.class_counts()
    print(f"\nTrain clips: {len(train_ds)}")
    for name, n in zip(CRIME_ACTIONS, counts):
        if n:
            print(f"  {name:16s} {n:5d}")
    if val_ds:
        print(f"Val clips:   {len(val_ds)}")

    empty = [c for c, n in zip(CRIME_ACTIONS, counts) if n == 0]
    if empty:
        print(f"\nNote: no training data for {', '.join(empty)}. These classes "
              f"can never be predicted correctly; the model will still emit "
              f"logits for them.")

    sampler = None
    shuffle = True
    if args.balance in ("sampler", "both"):
        w = train_ds.sample_weights()
        sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
        shuffle = False

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=shuffle,
                          sampler=sampler, num_workers=args.workers,
                          pin_memory=(device == "cuda"), drop_last=False)
    val_dl = (DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.workers) if val_ds else None)

    model = build_model(args.init or None, device, args.dropout, args.freeze_stages)

    weights = None
    if args.balance in ("weights", "both"):
        weights = train_ds.class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights,
                                    label_smoothing=args.label_smoothing)

    params = [q for q in model.parameters() if q.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    best_epoch = -1
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        t0 = time.perf_counter()

        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()

            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        scheduler.step()
        train_loss = loss_sum / max(total, 1)
        train_acc = correct / max(total, 1)
        dt = time.perf_counter() - t0

        line = (f"epoch {epoch:3d}/{args.epochs}  loss {train_loss:.4f}  "
                f"acc {train_acc:.3f}  ({dt:.1f}s)")

        if val_dl:
            stats = evaluate(model, val_dl, device)
            line += f"  | val acc {stats['accuracy']:.3f}  macro-F1 {stats['macro_f1']:.3f}"
            history.append({"epoch": epoch, "train_loss": train_loss,
                            "train_acc": train_acc, **{k: v for k, v in stats.items()
                                                       if k != "confusion"}})
            # Macro-F1, not accuracy: with a dominant `normal` class, accuracy
            # rewards a model that never predicts a crime.
            if stats["macro_f1"] > best_f1:
                best_f1 = stats["macro_f1"]
                best_epoch = epoch
                save(model, out_path, args, epoch, stats)
                line += "  *saved"
        else:
            history.append({"epoch": epoch, "train_loss": train_loss,
                            "train_acc": train_acc})
            save(model, out_path, args, epoch, None)

        print(line)

        if val_dl and args.early_stop and epoch - best_epoch >= args.early_stop:
            print(f"\nNo macro-F1 improvement in {args.early_stop} epochs — stopping.")
            break

    # Final report on the best checkpoint.
    if val_dl:
        print(f"\nBest macro-F1 {best_f1:.3f} at epoch {best_epoch}")
        state = torch.load(out_path, map_location=device, weights_only=False)
        model.load_state_dict(state["state_dict"])
        stats = evaluate(model, val_dl, device)
        print()
        print(format_report(stats))

    (out_path.parent / f"{out_path.stem}_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8")

    print(f"\nCheckpoint: {out_path}")
    print(f"Labels:     {out_path}.labels.json")
    print("\nTo use it in the app, point VEREC at the new checkpoint:")
    print(f"    export ACTION_MODEL_PATH={out_path}")
    return 0


def save(model, path: Path, args, epoch: int, stats: dict | None) -> None:
    """Write a checkpoint plus the sidecar label file the runtime looks for."""
    torch.save({
        "state_dict": model.state_dict(),
        "classes": CRIME_ACTIONS,
        "epoch": epoch,
        "clip_len": args.clip_len,
        "num_person": 2,
        "metrics": {k: v for k, v in (stats or {}).items() if k != "confusion"},
    }, path)

    Path(f"{path}.labels.json").write_text(
        json.dumps({"classes": CRIME_ACTIONS,
                    "clip_len": args.clip_len,
                    "trained_epoch": epoch}, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
