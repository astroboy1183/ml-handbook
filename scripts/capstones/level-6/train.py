"""Train a tinydl MLP on scikit-learn's handwritten digits and report test accuracy.

Usage:
    python train.py                         # Adam, 30 epochs
    python train.py --optimizer sgd --lr 0.2
    python train.py --hidden 256 --epochs 40

The data split matches the Level 6 chapters: 60% train, 20% validation, 20% test,
stratified, random_state=0. The model from the epoch with the best validation
accuracy is restored before the single test-set evaluation.
"""

import argparse

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import tinydl as td

TARGET_ACCURACY = 0.95


def load_data():
    digits = load_digits()
    X, y = digits.data / 16.0, digits.target  # pixel intensities 0..16 -> 0..1
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp,
                                                      random_state=0)
    return X_train, y_train, X_val, y_val, X_test, y_test


def accuracy(model, X, y):
    with td.no_grad():
        return float(np.mean(model(td.Tensor(X)).data.argmax(axis=1) == y))


def make_optimizer(name, params, lr):
    if name == "adam":
        return td.Adam(params, lr=lr or 3e-3)
    if name == "adamw":
        return td.AdamW(params, lr=lr or 3e-3, weight_decay=0.01)
    if name == "sgd":
        return td.SGD(params, lr=lr or 0.1, momentum=0.9)
    raise ValueError(f"unknown optimizer {name!r}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], default="adam")
    parser.add_argument("--lr", type=float, default=None, help="default: 3e-3 for Adam(W), 0.1 for SGD")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=5)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    td.manual_seed(args.seed)
    model = td.Sequential(td.Linear(64, args.hidden), td.ReLU(), td.Linear(args.hidden, 10))
    params = list(model.parameters())
    opt = make_optimizer(args.optimizer, params, args.lr)
    rng = np.random.default_rng(args.seed)
    print(f"model: 64-{args.hidden}-10 ReLU MLP, {sum(p.size for p in params):,} parameters; "
          f"optimizer {args.optimizer} (lr {opt.lr}), batch size {args.batch_size}")

    with td.no_grad():
        initial = td.cross_entropy(model(td.Tensor(X_train)), y_train).item()
    print(f"initial training loss {initial:.3f} (log 10 = {np.log(10):.3f})")

    history = {"loss": [], "val_acc": []}
    best_acc, best_epoch, best_params = -1.0, 0, None
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(X_train))
        running = 0.0
        for i in range(0, len(order), args.batch_size):
            idx = order[i:i + args.batch_size]
            loss = td.cross_entropy(model(td.Tensor(X_train[idx])), y_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * len(idx)
        history["loss"].append(running / len(order))
        history["val_acc"].append(accuracy(model, X_val, y_val))
        if history["val_acc"][-1] > best_acc:
            best_acc, best_epoch = history["val_acc"][-1], epoch
            best_params = [p.data.copy() for p in params]
        if epoch % args.log_every == 0 or epoch == 1:
            print(f"epoch {epoch:>3}: train loss {history['loss'][-1]:.4f}, val acc {history['val_acc'][-1]:.3f}")

    for p, saved in zip(params, best_params):  # restore the best checkpoint, in place
        p.data[...] = saved
    test_acc = accuracy(model, X_test, y_test)
    print(f"best val acc {best_acc:.3f} at epoch {best_epoch} (restored)")
    print(f"test accuracy: {test_acc:.3f} -> {'PASS' if test_acc > TARGET_ACCURACY else 'FAIL'} "
          f"(target > {TARGET_ACCURACY:.0%})")
    return {"history": history, "best_epoch": best_epoch, "val_acc": best_acc, "test_acc": test_acc,
            "model": model}


if __name__ == "__main__":
    main()
