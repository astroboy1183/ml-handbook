"""Level 7 capstone: train a FashionMNIST classifier on a CPU.

Usage (from this directory):
    python train.py                         # SmallResNet, 8 epochs, writes runs/resnet/
    python train.py --model linear --epochs 3 --no-augment   # the baseline, writes runs/linear/
    python train.py --data-dir /path/to/data --epochs 12 --width 24

Outputs in runs/<run-name>/:
    best.pt        checkpoint with the best validation accuracy (model + optimizer state, config, epoch)
    history.csv    one row per epoch: train/val loss and accuracy, learning rate, seconds
    tb/            TensorBoard event files, if the tensorboard package is installed
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import load_splits, make_loaders
from model import build_model, count_params


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, optimizer=None, scheduler=None):
    """One pass over loader. Trains if an optimizer is given, otherwise evaluates. Returns (loss, acc)."""
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            n += len(yb)
    return total_loss / n, correct / n


def make_logger(out_dir):
    """TensorBoard if available; the CSV history is always written."""
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(out_dir / "tb")
    except ImportError:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--model", choices=["resnet", "linear"], default="resnet")
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=4e-3, help="peak learning rate of the one-cycle schedule")
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    out = Path(args.out or f"runs/{args.model}")
    out.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds = load_splits(args.data_dir, seed=args.seed)
    if args.no_augment:
        train_ds.augment = None
    train_dl, val_dl, _ = make_loaders(train_ds, val_ds, test_ds, args.batch_size, seed=args.seed)
    model = build_model(args.model, args.width)
    print(f"data: train {len(train_ds):,}, val {len(val_ds):,}, test {len(test_ds):,} "
          f"(normalize mean {train_ds.mean:.4f}, std {train_ds.std:.4f}); augment={not args.no_augment}")
    print(f"model: {args.model}, {count_params(model):,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs,
                                                    steps_per_epoch=len(train_dl))
    writer = make_logger(out)
    config = vars(args) | {"mean": train_ds.mean, "std": train_ds.std}
    (out / "config.json").write_text(json.dumps(config, indent=2))

    fields = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "seconds"]
    best_acc = -1.0
    with open(out / "history.csv", "w", newline="") as f:
        log = csv.DictWriter(f, fieldnames=fields)
        log.writeheader()
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            lr = optimizer.param_groups[0]["lr"]
            tr_loss, tr_acc = run_epoch(model, train_dl, optimizer, scheduler)
            va_loss, va_acc = run_epoch(model, val_dl)
            row = dict(epoch=epoch, train_loss=round(tr_loss, 4), train_acc=round(tr_acc, 4),
                       val_loss=round(va_loss, 4), val_acc=round(va_acc, 4), lr=round(lr, 6),
                       seconds=round(time.time() - t0, 1))
            log.writerow(row)
            f.flush()
            if writer is not None:
                for k in ["train_loss", "train_acc", "val_loss", "val_acc", "lr"]:
                    writer.add_scalar(k, row[k], epoch)
            flag = ""
            if va_acc > best_acc:
                best_acc = va_acc
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "epoch": epoch, "val_acc": va_acc, "config": config}, out / "best.pt")
                flag = "  * saved best.pt"
            print(f"epoch {epoch:2d}  train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                  f"val loss {va_loss:.4f} acc {va_acc:.4f}{flag}")
    if writer is not None:
        writer.close()
    print(f"best val acc {best_acc:.4f}; checkpoint {out / 'best.pt'}")


if __name__ == "__main__":
    main()
