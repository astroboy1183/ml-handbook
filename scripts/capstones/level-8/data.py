"""Level 8 capstone: text loading, tokenization, and batching.

The corpus is not part of the repository. Download it once with
    python data.py --download
which saves Project Gutenberg eBook #11 (Alice's Adventures in Wonderland,
public domain) to alice.txt with the Gutenberg header and license removed.
"""

import argparse
import re
import urllib.request
from collections import Counter
from pathlib import Path

import torch

GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/11/pg11.txt"


def download_alice(path="alice.txt"):
    raw = urllib.request.urlopen(GUTENBERG_URL).read().decode("utf-8")
    start = raw.index("\n", raw.index("*** START")) + 1
    end = raw.index("*** END")
    Path(path).write_text(raw[start:end].strip())
    print(f"saved {path}")


def load_text(path="alice.txt"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python data.py --download` first.")
    return path.read_text()


class CharTokenizer:
    """One token per character. The alphabet is a fixed property of the corpus, so it's built from the whole text."""

    kind = "char"

    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)

    def state(self):
        return {"kind": self.kind, "chars": self.chars}


PRETOKENIZE = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^A-Za-z0-9\s]+|\s+")


class BPETokenizer:
    """A small byte-pair-encoding tokenizer over characters (see the Language models chapter)."""

    kind = "bpe"

    def __init__(self, text=None, n_merges=200, alphabet=None, state=None):
        if state is not None:
            self.base, self.merges = state["base"], [tuple(m) for m in state["merges"]]
        else:
            self.base = sorted(set(text) | set(alphabet or ""))
            self.merges = self._train(text, n_merges)
        self.vocab = self.base + ["".join(m) for m in self.merges]
        self.stoi = {t: i for i, t in enumerate(self.vocab)}
        self.rank = {m: r for r, m in enumerate(self.merges)}

    @staticmethod
    def _train(text, n_merges):
        words = Counter(PRETOKENIZE.findall(text))
        splits = {w: list(w) for w in words}
        merges = []
        for _ in range(n_merges):
            pairs = Counter()
            for w, n in words.items():
                s = splits[w]
                for a, b in zip(s, s[1:]):
                    pairs[a, b] += n
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            merges.append(best)
            for w in words:
                s, out, i = splits[w], [], 0
                while i < len(s):
                    if i < len(s) - 1 and (s[i], s[i + 1]) == best:
                        out.append(s[i] + s[i + 1])
                        i += 2
                    else:
                        out.append(s[i])
                        i += 1
                splits[w] = out
        return merges

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, s):
        ids = []
        for chunk in PRETOKENIZE.findall(s):
            sym = [c for c in chunk if c in self.stoi]
            while len(sym) > 1:
                r, i = min((self.rank.get((a, b), float("inf")), i) for i, (a, b) in enumerate(zip(sym, sym[1:])))
                if r == float("inf"):
                    break
                sym[i:i + 2] = [sym[i] + sym[i + 1]]
            ids.extend(self.stoi[t] for t in sym)
        return ids

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids)

    def state(self):
        return {"kind": self.kind, "base": self.base, "merges": [list(m) for m in self.merges]}


def tokenizer_from_state(state):
    if state["kind"] == "char":
        tok = CharTokenizer("")
        tok.chars = state["chars"]
        tok.stoi = {c: i for i, c in enumerate(tok.chars)}
        return tok
    return BPETokenizer(state=state)


def make_splits(text, tokenizer_kind="char", val_frac=0.1, n_merges=200):
    """Split the text first (the last val_frac is validation), then build the tokenizer.

    BPE merges are learned from the training text only, so validation text is never used for fitting.
    The character alphabet comes from the whole text so that every validation character can be encoded.
    """
    cut = int(len(text) * (1 - val_frac))
    train_text, val_text = text[:cut], text[cut:]
    alphabet = sorted(set(text))
    tok = CharTokenizer(text) if tokenizer_kind == "char" else BPETokenizer(train_text, n_merges, alphabet)
    train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tok.encode(val_text), dtype=torch.long)
    return tok, train_ids, val_ids, len(train_text), len(val_text)


def get_batch(ids, batch_size, block_size, generator=None):
    ix = torch.randint(len(ids) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([ids[i:i + block_size] for i in ix])
    y = torch.stack([ids[i + 1:i + block_size + 1] for i in ix])
    return x, y


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="download alice.txt from Project Gutenberg")
    ap.add_argument("--data", default="alice.txt")
    args = ap.parse_args()
    if args.download:
        download_alice(args.data)
    text = load_text(args.data)
    for kind in ["char", "bpe"]:
        tok, tr, va, n_tr, n_va = make_splits(text, kind)
        print(f"{kind}: vocab {tok.vocab_size}, train {len(tr):,} tokens ({n_tr:,} chars), "
              f"val {len(va):,} tokens ({n_va:,} chars), {n_tr / len(tr):.2f} chars/token")
