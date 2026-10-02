"""Level 7 capstone: evaluate the best checkpoint on the test set, once, and analyze its errors.

Usage (from this directory, after train.py):
    python evaluate.py                       # uses runs/resnet/
    python evaluate.py --run runs/resnet --data-dir /path/to/data

Prints test accuracy, per-class accuracy, and the most-confused pairs. Saves to the run folder:
    test_metrics.json, curves.png, confusion.png, misclassified.png
The plotting functions return matplotlib figures, so a notebook can import and show them.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from data import CLASSES, load_splits, make_loaders
from model import build_model


@torch.no_grad()
def predict(model, loader):
    model.eval()
    probs, labels = [], []
    for xb, yb in loader:
        probs.append(torch.softmax(model(xb), 1))
        labels.append(yb)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def read_history(run_dir):
    with open(Path(run_dir) / "history.csv") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def plot_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    e = history["epoch"]
    axes[0].plot(e, history["train_loss"], "o-", label="train (augmented)")
    axes[0].plot(e, history["val_loss"], "o-", label="validation")
    axes[0].set(xlabel="epoch", ylabel="cross-entropy loss", title="Loss")
    axes[1].plot(e, history["train_acc"], "o-", label="train (augmented)")
    axes[1].plot(e, history["val_acc"], "o-", label="validation")
    axes[1].axhline(0.9, color="gray", ls="--", lw=1, label="90% target")
    axes[1].set(xlabel="epoch", ylabel="accuracy", title="Accuracy")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    return fig


def plot_confusion(cm):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    norm = cm / cm.sum(1, keepdims=True)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.6 else "black")
    ax.set_xticks(range(10), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(10), CLASSES)
    ax.set(xlabel="predicted", ylabel="true", title="Test confusion matrix (counts; color = row fraction)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def plot_misclassified(images, labels, probs, n=24):
    """The n misclassified test images the model was most confident about."""
    pred = probs.argmax(1)
    wrong = np.where(pred != labels)[0]
    wrong = wrong[np.argsort(-probs[wrong, pred[wrong]])][:n]
    cols = 8
    fig, axes = plt.subplots(int(np.ceil(len(wrong) / cols)), cols, figsize=(13, 6.6))
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, wrong):
        ax.imshow(images[i], cmap="gray_r")
        ax.set_title(f"true: {CLASSES[labels[i]]}\npred: {CLASSES[pred[i]]} ({probs[i, pred[i]]:.2f})",
                     fontsize=7.5)
    fig.suptitle("Most confident mistakes on the test set")
    fig.tight_layout()
    return fig


def confused_pairs(cm, k=5):
    off = cm.copy()
    np.fill_diagonal(off, 0)
    order = np.dstack(np.unravel_index(np.argsort(-off.ravel()), off.shape))[0][:k]
    return [(CLASSES[i], CLASSES[j], int(off[i, j])) for i, j in order]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/resnet")
    p.add_argument("--data-dir", default="data")
    args = p.parse_args()
    torch.set_num_threads(4)
    run = Path(args.run)

    ckpt = torch.load(run / "best.pt", map_location="cpu")
    cfg = ckpt["config"]
    model = build_model(cfg["model"], cfg["width"])
    model.load_state_dict(ckpt["model"])
    train_ds, val_ds, test_ds = load_splits(args.data_dir, seed=cfg["seed"])
    _, _, test_dl = make_loaders(train_ds, val_ds, test_ds)

    probs, labels = predict(model, test_dl)
    pred = probs.argmax(1)
    acc = (pred == labels).mean()
    cm = confusion_matrix(labels, pred)
    print(f"checkpoint: epoch {ckpt['epoch']}, val acc {ckpt['val_acc']:.4f}")
    print(f"TEST accuracy: {acc:.4f}  ({(pred != labels).sum()} errors out of {len(labels):,})")
    print("per-class accuracy:")
    for c, name in enumerate(CLASSES):
        print(f"  {name:12s} {cm[c, c] / cm[c].sum():.3f}")
    print("most-confused pairs (true -> predicted: count):")
    pairs = confused_pairs(cm)
    for t, pr, n in pairs:
        print(f"  {t:12s} -> {pr:12s} {n}")

    metrics = {"test_accuracy": round(float(acc), 4), "best_epoch": ckpt["epoch"],
               "val_accuracy": round(float(ckpt["val_acc"]), 4),
               "per_class": {n: round(float(cm[c, c] / cm[c].sum()), 4) for c, n in enumerate(CLASSES)},
               "confused_pairs": pairs}
    (run / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    np.save(run / "test_probs.npy", probs)
    plot_curves(read_history(run)).savefig(run / "curves.png", dpi=110)
    plot_confusion(cm).savefig(run / "confusion.png", dpi=110)
    plot_misclassified(test_ds.images.numpy(), labels, probs).savefig(run / "misclassified.png", dpi=110)
    print(f"saved test_metrics.json, test_probs.npy, curves.png, confusion.png, misclassified.png to {run}/")


if __name__ == "__main__":
    main()
