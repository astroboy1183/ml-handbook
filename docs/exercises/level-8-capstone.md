# Level 8 capstone: train a small GPT from scratch and write up what you learned

> **Level 8 · Capstone** · ⏱️ 10–20 h · Prerequisites: all of [Level 8](../chapters/08-modern-deep-learning/index.md), especially [Attention and transformers](../chapters/08-modern-deep-learning/01-attention-and-transformers.md) and [Language models](../chapters/08-modern-deep-learning/02-language-models.md)

This capstone puts the whole language-modeling stack in your hands: a tokenizer, a GPT you can resize from the command line, an honest train/validation split, a training loop with a learning-rate schedule and checkpointing, sampling with temperature and top-k, a perplexity report, and an ablation that tests one design decision. Then you write up what you found, like a short research note. It's the last capstone in the handbook. Do it without notes, and you'll have shown you understand how the models at the center of modern AI are built and trained.

## The scenario

You've joined a small research group that wants a minimal, fully understood language-model codebase for teaching and quick experiments. Nobody on the team trusts code they can't read in one sitting, so the brief from the group lead, Wei, is:

1. "Build a GPT we can read end to end: a tokenizer, a model, a training script, and a sampling script. Every hyperparameter should be a command-line flag."
2. "Train it on a public-domain book. Show me the loss curves, and tell me honestly how good it is on text it hasn't seen."
3. "Pick one design decision and test it properly. I want to know whether it matters at this scale, and why."
4. "Write it up in a page. What worked, what didn't, what surprised you, and what you'd do with 100 times the compute."

## The data

Use *Alice's Adventures in Wonderland* by Lewis Carroll, Project Gutenberg eBook #11, which is in the public domain. Download it yourself and keep it out of version control (add `alice.txt` to your `.gitignore`). This snippet downloads it and strips Project Gutenberg's header and license text:

<!-- skip-run -->
```python
import urllib.request
from pathlib import Path

path = Path("alice.txt")
if not path.exists():
    raw = urllib.request.urlopen("https://www.gutenberg.org/cache/epub/11/pg11.txt").read().decode("utf-8")
    start = raw.index("\n", raw.index("*** START")) + 1
    end = raw.index("*** END")
    path.write_text(raw[start:end].strip())
print(f"{len(path.read_text()):,} characters")
```

Any other plain-text corpus of 100 KB to a few MB works too, such as another public-domain novel or a collection of your own notes. A smaller corpus overfits sooner, which makes the validation curve more interesting, not less.

## What to do

Build a small project folder with four files: `data.py`, `model.py`, `train.py`, and `sample.py`. Fix the random seed everywhere.

### Part A: data and tokenization

1. Load the text and split it into training (the first 90%) and validation (the last 10%) **before** doing anything else. The split must be contiguous, not random windows, so that validation text is genuinely unseen.
2. Implement a character-level tokenizer with `encode` and `decode`, and test that `decode(encode(s)) == s`.
3. (Optional) Implement a small BPE tokenizer whose merges are learned from the training text only, and report characters per token on the validation text.
4. Write `get_batch(ids, batch_size, block_size)` that returns inputs and next-token targets.

### Part B: the model

1. Implement a GPT in `model.py` with a `GPTConfig` dataclass: vocabulary size, block size, number of layers, heads, embedding size, dropout, and a switch to turn position embeddings off.
2. Use pre-LN blocks with causal multi-head self-attention, a final LayerNorm, and a weight-tied output head. Initialize weights with a small standard deviation (0.02 is standard).
3. Count the parameters with the formula from the [Language models](../chapters/08-modern-deep-learning/02-language-models.md) chapter and check it against `sum(p.numel() for p in model.parameters())`.
4. Write a causality test: changing token $t$ must not change the logits at positions before $t$.

### Part C: training

1. Write `train.py` with command-line flags for every hyperparameter. Use AdamW, gradient clipping, and a learning-rate schedule with warmup and cosine decay.
2. Every few hundred steps, estimate training and validation loss on fixed batches, and save a checkpoint whenever validation loss improves (keep the best one, not the last one).
3. Save the loss history to a JSON file, and plot the train and validation loss curves.
4. Keep a training run to a few minutes on a CPU. Check that the loss at step 0 is close to $\ln V$.

### Part D: sampling and evaluation

1. Write `sample.py`: load a checkpoint and generate text from a prompt, with temperature and top-k. Show samples at two or three settings, including greedy decoding.
2. Evaluate the best checkpoint on the **whole** validation text and report the loss in nats per token, the perplexity, and bits per character. Compare with two baselines: uniform guessing and a unigram (character-frequency) model.

### Part E: an ablation

Choose at least one controlled experiment, change only that one thing, and keep the seed, data, and number of steps fixed:

- with and without position embeddings;
- two model sizes (for example, 4 layers with 64 dimensions versus 2 layers with 32);
- character-level versus BPE tokenization (compare in bits per character, not per-token perplexity);
- with and without dropout, or two context lengths.

Report the best validation loss of each variant in a table, plot the validation curves together, and explain the result.

### Part F: the write-up

Write one page (in a `REPORT.md` or a notebook's final cell) covering: the setup (data, split, model size, training budget, in tokens and in FLOPs using $6ND$), the results (curves, the perplexity table, samples), the ablation and your explanation of it, what surprised you, the limitations, and what you'd change with 100 times more data or compute.

## Deliverables

- A folder with `data.py`, `model.py`, `train.py`, and `sample.py` that runs from the command line.
- The console logs of your main run and your ablation runs, and their `history.json` files.
- A loss-curve figure and an ablation figure.
- Samples at two or more decoding settings.
- The one-page write-up.

## Acceptance checklist

- [ ] The text is split into contiguous train and validation parts before any tokenizer fitting, and BPE merges (if used) are learned from the training part only.
- [ ] `decode(encode(s)) == s` holds on the validation text.
- [ ] The model's size is set from the command line, and the parameter count matches the formula exactly.
- [ ] A causality test passes.
- [ ] The step-0 loss is within about 0.1 of $\ln V$.
- [ ] Training uses AdamW, gradient clipping, warmup and cosine decay, and saves the checkpoint with the best validation loss.
- [ ] A plot shows train and validation loss over training, and you can point to where overfitting starts (if it does).
- [ ] `sample.py` supports temperature and top-k, and greedy decoding (`temperature 0`).
- [ ] The report gives validation loss, perplexity, and bits per character on the full validation text, plus uniform and unigram baselines.
- [ ] The ablation changes exactly one thing, and its result is explained, not just reported.
- [ ] Every number in the write-up comes from a run you can reproduce with the same seed.

## Hints

??? tip "Hint for Part A"
    Split the *text*, not the token ids, so the split is the same whatever tokenizer you use. For a character tokenizer, building the alphabet from the whole text is fine (it's a fixed property of the corpus, like the set of letters in English). For BPE, learn merges from the training text only, or validation text will influence the vocabulary.

??? tip "Hint for Part B"
    Start from the GPT in [Language models](../chapters/08-modern-deep-learning/02-language-models.md). The parameter formula with a tied head is $Vd + T_{\max}d + L(12d^2 + 13d) + 2d$; drop the $T_{\max}d$ term when position embeddings are off. For the causality test, perturb one token in a copy of a batch and compare logits with `torch.allclose` on the earlier positions.

??? tip "Hint for Part C"
    Evaluate on the same batches every time (use a `torch.Generator` with a fixed seed), or the curves will be noisy and the "best" checkpoint will be partly luck. A schedule of 100 warmup steps then cosine decay to 10% of the peak works well. On a small book, a model with a few hundred thousand parameters will overfit within a couple of thousand steps; that's expected and is what checkpointing on validation loss is for.

??? tip "Hint for Part D"
    Evaluate the full validation text in non-overlapping windows of `block_size` tokens and sum the losses with `reduction="sum"`. Bits per character is total nats divided by the number of validation *characters*, divided by $\ln 2$. For the unigram baseline, count training-set character frequencies (with add-one smoothing) and average $-\ln p(c)$ over validation characters.

??? tip "Hint for Part E"
    A causal transformer without position embeddings isn't as helpless as an encoder without them. Think about what the causal mask tells a position about how many tokens precede it. Before you run the ablation, write down what you expect, then compare.

## Solution

A complete worked solution, with all four files, real training logs, curves, samples, the ablations, and an example write-up, is in the [Level 8 capstone solution](solutions/level-8-capstone.md). Build your own first: the understanding comes from the bugs you fix along the way.
