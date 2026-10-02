"""Level 8 capstone: generate text from a trained checkpoint.

    python sample.py --ckpt runs/base/ckpt.pt --prompt "Alice " --temperature 0.8 --top-k 10
"""

import argparse

import torch

from data import tokenizer_from_state
from model import GPT, GPTConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/base/ckpt.pt")
    ap.add_argument("--prompt", default="Alice ")
    ap.add_argument("--n", type=int, default=300, help="number of new tokens")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.set_num_threads(4)
    ckpt = torch.load(args.ckpt)
    tok = tokenizer_from_state(ckpt["tokenizer"])
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    idx = torch.tensor([tok.encode(args.prompt)], dtype=torch.long)
    g = torch.Generator().manual_seed(args.seed)
    out = model.generate(idx, args.n, temperature=args.temperature, top_k=args.top_k, generator=g)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
