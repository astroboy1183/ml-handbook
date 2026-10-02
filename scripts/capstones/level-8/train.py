"""Level 8 capstone: train a small GPT on a text file.

Examples (from this directory, with alice.txt present):
    python train.py --out runs/base
    python train.py --out runs/nopos --no-pos
    python train.py --out runs/small --n-layer 2 --n-embd 32
    python train.py --out runs/bpe --tokenizer bpe

Writes <out>/ckpt.pt (the best checkpoint by validation loss) and <out>/history.json.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch

from data import get_batch, load_text, make_splits
from model import GPT, GPTConfig


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="alice.txt")
    ap.add_argument("--out", default="runs/base")
    ap.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
    ap.add_argument("--bpe-merges", type=int, default=200)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--n-embd", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--no-pos", action="store_true", help="disable position embeddings (ablation)")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-batches", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    return ap.parse_args()


def lr_at(step, args):
    """Linear warmup, then cosine decay to 10% of the peak learning rate."""
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    progress = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def estimate_loss(model, splits, args):
    """Mean loss on the same random batches every time, so evaluations are comparable."""
    model.eval()
    out = {}
    for name, ids in splits.items():
        g = torch.Generator().manual_seed(1234)
        losses = [model(*get_batch(ids, args.batch_size, args.block_size, g))[1].item()
                  for _ in range(args.eval_batches)]
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


@torch.no_grad()
def full_eval(model, ids, block_size):
    """Exact mean loss over all of `ids`, in non-overlapping windows. Returns (mean nats per token, n tokens)."""
    model.eval()
    total, count = 0.0, 0
    for start in range(0, len(ids) - 1, block_size):
        x = ids[start:start + block_size]
        y = ids[start + 1:start + block_size + 1]
        x = x[:len(y)]
        logits, _ = model(x.unsqueeze(0))
        total += torch.nn.functional.cross_entropy(logits[0], y, reduction="sum").item()
        count += len(y)
    return total / count, count


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    text = load_text(args.data)
    tok, train_ids, val_ids, n_train_chars, n_val_chars = make_splits(text, args.tokenizer, n_merges=args.bpe_merges)
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=args.block_size, n_layer=args.n_layer,
                    n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout, pos_emb=not args.no_pos)
    model = GPT(cfg)
    print(f"tokenizer {args.tokenizer}: vocab {tok.vocab_size}, train {len(train_ids):,} tokens, "
          f"val {len(val_ids):,} tokens")
    print(f"model: {model.n_params():,} parameters, config {cfg.to_dict()}")

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95))

    splits = {"train": train_ids, "val": val_ids}
    history = {"step": [], "train": [], "val": [], "lr": []}
    best_val, best_step = float("inf"), -1
    t0 = time.time()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            losses = estimate_loss(model, splits, args)
            for k in ["train", "val"]:
                history[k].append(round(losses[k], 4))
            history["step"].append(step)
            history["lr"].append(lr_at(min(step, args.steps - 1), args))
            flag = ""
            if losses["val"] < best_val:
                best_val, best_step, flag = losses["val"], step, " *"
                torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "tokenizer": tok.state(),
                            "step": step}, out / "ckpt.pt")
            print(f"step {step:5d} | train {losses['train']:.3f} | val {losses['val']:.3f}{flag}", flush=True)
        if step == args.steps:
            break
        for group in opt.param_groups:
            group["lr"] = lr_at(step, args)
        x, y = get_batch(train_ids, args.batch_size, args.block_size)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    train_time = time.time() - t0

    # Final, exact evaluation of the best checkpoint on the whole validation text.
    model.load_state_dict(torch.load(out / "ckpt.pt")["model"])
    val_nll, n_val_tokens = full_eval(model, val_ids, args.block_size)
    bits_per_char = val_nll * n_val_tokens / n_val_chars / math.log(2)
    tokens_seen = args.steps * args.batch_size * args.block_size
    summary = {
        "best_step": best_step, "val_loss": round(val_nll, 4), "val_perplexity": round(math.exp(val_nll), 3),
        "val_bits_per_char": round(bits_per_char, 4), "n_params": model.n_params(),
        "tokens_seen": tokens_seen, "train_flops_6ND": 6 * model.n_params() * tokens_seen,
        "train_seconds": round(train_time, 1),
    }
    history.update(summary=summary, config=cfg.to_dict(), args=vars(args))
    (out / "history.json").write_text(json.dumps(history, indent=1))
    print(f"best checkpoint at step {best_step}; full validation: loss {val_nll:.4f} nats/token, "
          f"perplexity {math.exp(val_nll):.2f}, {bits_per_char:.3f} bits/char")
    print(f"trained {tokens_seen:,} tokens in {train_time:.0f} s")


if __name__ == "__main__":
    main()
