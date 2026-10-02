# Level 6 capstone: a tiny deep learning library

> **Level 6 · Capstone** · ⏱️ 8–14 h · Prerequisites: all of [Level 6](../chapters/06-neural-networks/index.md)

Build a small deep learning library from scratch in NumPy: a tensor class with reverse-mode autograd, layers, a loss, and two optimizers, all verified with gradient checks. Then use it, and nothing else, to train a classifier to over 95% test accuracy on handwritten digits. This is the gate out of Level 6. Finish it without notes before moving on to [Level 7](../chapters/07-deep-learning-pytorch/index.md), where you'll use PyTorch, which will then look very familiar.

## The scenario

You've joined Inkwell, a (fictional) company that makes rugged handheld scanners for warehouses. The newest model reads handwritten digits on delivery slips: a quantity box, a bay number, a check digit. The device runs a stripped-down Linux with Python and NumPy, and nothing else: no PyTorch, no TensorFlow, no compiler toolchain, and the security team won't approve new packages for this hardware generation.

The current recognizer is a template matcher that misreads about one digit in eight. A colleague trained a neural network on a laptop that does far better, but the device can't run the framework it was built with. Your lead, Priya, gives you the brief:

1. "Write us a tiny training-and-inference library in pure NumPy. It has to be small enough that anyone on the team can read all of it in an afternoon."
2. "I don't want hand-written backward passes for every layer. When we add a layer next year, I want to write only its forward pass. So: autograd."
3. "Prove the gradients are right. A model that trains with a subtly wrong gradient still learns *something*, and we'd never know."
4. "Train a digit classifier with it, and show me it beats 95% on data it has never seen."

## The data

Use scikit-learn's `load_digits` ($8 \times 8$ grayscale images, intensities 0 to 16, 1,797 examples). It ships with scikit-learn, so there's nothing to download. It's a stand-in for the scanner's real digit crops, which are similar in size.

Use the same split as the Level 6 chapters, so your numbers are comparable with theirs: 20% test, then 25% of the rest for validation (60/20/20 overall), stratified by class, `random_state=0`, and pixels divided by 16.

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X, y = digits.data / 16.0, digits.target
X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=0)
print(X_train.shape, X_val.shape, X_test.shape)
```

```text
(1077, 64) (360, 64) (360, 64)
```

The test set is touched exactly once, at the very end.

## What to do

You may use NumPy, and scikit-learn only for loading and splitting the data. You may use PyTorch *only* in tests, to compare your gradients with a reference. Everything else is yours.

### Part A: a tensor with autograd

Write a `Tensor` class that wraps a float64 NumPy array and records how it was computed. It needs:

- `data`, `grad`, and `requires_grad`. Leaves you create (parameters) have `requires_grad=True`; the result of an operation requires gradients if any input does.
- Operations, each with its own backward rule: `+`, `-` (binary and unary), `*`, `/`, `**` with a constant exponent, `@` (2-D matrix multiplication), `sum` and `mean` (with `axis` and `keepdims`), `relu`, `exp`, and `log`. Reverse versions (`2 * t`, `1 - t`) should work too.
- `backward()`: seed the output's gradient with 1 (scalars only, unless a gradient is passed in), order the graph topologically, and apply each node's backward rule once.
- PyTorch's semantics for leaves: gradients **accumulate** into a leaf's `.grad` across `backward()` calls, until you zero them.
- A `no_grad()` context manager that turns graph recording off, for evaluation and updates.

### Part B: broadcasting

Make every element-wise operation work with NumPy broadcasting, in both directions: a `(32, 10)` tensor plus a `(10,)` bias, a `(4, 1)` column times a `(1, 6)` row. The gradient of each operand must have that operand's shape. Write the helper that sums a gradient back down to a shape, and explain in a comment why summing is correct.

### Part C: the loss

Write `cross_entropy(logits, targets)`: mean softmax cross-entropy over a batch, from a Tensor of logits of shape `(n, K)` and integer labels. Make it a single fused operation, stable for logits like $\pm 1000$, whose backward rule is $\frac{1}{n}(\mathbf{P} - \mathbf{Y})$.

### Part D: layers

Write a small layer API:

- A `Module` base class whose `parameters()` finds every parameter tensor in a module and its sub-modules (including modules stored in lists), without duplicates, and a `zero_grad()`.
- `Linear(d_in, d_out)` computing $\mathbf{x}\mathbf{W} + \mathbf{b}$ with He initialization, `ReLU()`, and `Sequential(*layers)`.
- A seeding function, so that a model built twice with the same seed has identical weights.

None of these layers may contain a backward pass. They're compositions of `Tensor` operations.

### Part E: optimizers

Write `SGD` (with optional momentum) and `Adam` (with bias correction), each constructed from a list of parameters and a learning rate, with `step()` and `zero_grad()`. They update the parameters' data in place and skip parameters whose gradient is `None`.

### Part F: prove the gradients are right

Write a `gradcheck(fn, inputs)` function that compares autograd gradients of a scalar function with central differences and returns a relative error per input. Then write a test file that checks, in float64:

1. Every operation from Part A, including at least three different broadcasting patterns and a tensor used twice in one expression.
2. `sum` and `mean` along an axis, with and without `keepdims`.
3. `cross_entropy`, including that its value matches a direct log-softmax computation and that it stays finite at extreme logits.
4. A full multi-layer network, through every parameter at once.
5. Behaviors, not just gradients: accumulation across two `backward()` calls, `zero_grad()`, `no_grad()`, SGD with momentum converging on a quadratic, and Adam's first step having size $\eta$ in every coordinate regardless of gradient scale.

Use the thresholds from [Forward and backpropagation](../chapters/06-neural-networks/02-forward-and-backpropagation.md#gradient-checking): every relative error should be below $10^{-7}$.

### Part G: train it

Write `train.py`, which builds a `Sequential` MLP for the 64-pixel inputs and 10 classes, trains it with mini-batches and your optimizer, logs the training loss and validation accuracy per epoch, keeps the parameters from the epoch with the best validation accuracy, restores them, and only then evaluates on the test set, once. It must reach **over 95% test accuracy** with a fixed seed, and it should run in under a minute on a laptop CPU. Make the optimizer, learning rate, epochs, hidden size, and seed command-line options.

Also train with the other optimizer, and report both.

### Part H: the write-up

Write a short report (a README, or a markdown cell at the end of a notebook) for Priya, under a page:

- What's in the library, and the one design decision you'd defend hardest.
- The gradient-check table.
- Learning curves for both optimizers, and the final test accuracy of the model you'd ship.
- One limitation of the library that would matter on the device, and what you'd do about it.

### Stretch goals (optional)

- Add `tanh` and `sigmoid` operations, `AdamW`, and a learning-rate schedule with warmup and cosine decay.
- Add `Dropout` with training and evaluation modes, and a `LayerNorm` written purely from `Tensor` operations (no backward code), and gradient-check it.
- Compare your gradients for a whole network with PyTorch's, to the last bit of float64.
- Measure how much faster training is in float32, and whether your gradient checks still pass in float32 (they shouldn't, and you should be able to explain why).
- Replace the recursive topological sort with an iterative one, and show the recursive version fails on a graph a few thousand nodes deep.

## Deliverables

1. `tinydl.py`: the library (Tensor, no_grad, cross_entropy, Module, Linear, ReLU, Sequential, SGD, Adam, gradcheck, seeding).
2. `test_tinydl.py`: the gradient checks and behavior tests, runnable with `python test_tinydl.py` (exit code 1 on failure) or `pytest`.
3. `train.py`: the training script with command-line options, printing its log and the final test accuracy.
4. The write-up from Part H.

## Acceptance checklist

- [ ] No layer contains a hand-written backward pass; every gradient comes from the `Tensor` operations' rules.
- [ ] `backward()` visits nodes in reverse topological order, each exactly once.
- [ ] Gradients accumulate into leaves across `backward()` calls, `zero_grad()` clears them, and `no_grad()` stops recording.
- [ ] Every operand's gradient has that operand's shape, for every broadcasting pattern tested.
- [ ] `cross_entropy` is fused, finite at logits of $\pm 1000$, and has the gradient $\frac{1}{n}(\mathbf{P} - \mathbf{Y})$.
- [ ] Every gradient check passes with relative error below $10^{-7}$ in float64, including a whole multi-layer network.
- [ ] Adam implements bias correction, and a test shows its first step has size $\eta$ per coordinate.
- [ ] The data split matches the one above; the test set is used once, after choosing the checkpoint on validation data.
- [ ] Test accuracy is above 95% with a fixed seed, and running `train.py` twice gives the same result.
- [ ] The write-up states the test accuracy, shows the gradient-check results and learning curves, and names a real limitation.

## Hints

??? tip "Hint for Part A: what each node stores"
    Each operation's output can store its parents and a function that maps the output's gradient to a tuple of gradients for the parents: a vector-Jacobian product. `backward()` then only needs a topological order and a dictionary from node to accumulated gradient. Keep intermediate gradients in that dictionary rather than on the nodes, and write only into leaves' `.grad`: that gives you PyTorch's accumulation semantics without double counting if `backward()` is called twice. [Autograd from scratch](../chapters/06-neural-networks/06-autograd-from-scratch.md#a-tensor-autograd-engine) has the closure-based version to start from.

??? tip "Hint for Part B: unbroadcast"
    Two steps: while the gradient has more dimensions than the target shape, sum over axis 0; then, for each axis where the target has size 1 but the gradient doesn't, sum over that axis with `keepdims=True`. Test it on `np.ones` gradients: each entry of the result should equal the number of output positions that entry was copied into.

??? tip "Hint for Part C: stability"
    Subtract each row's maximum logit before exponentiating, compute log-probabilities as `z - log(sum(exp(z)))`, and never take the log of a probability. Save the probabilities (or log-probabilities) for the backward rule.

??? tip "Hint for Part D: finding parameters"
    Loop over `vars(self).values()`: yield tensors with `requires_grad=True`, recurse into `Module` values, and into lists or tuples of modules (that's where `Sequential` keeps its layers). Track `id()`s in a set to avoid yielding a shared parameter twice.

??? tip "Hint for Part F: gradcheck"
    Compute the analytic gradients first with one `backward()`. Then, inside `no_grad()`, perturb each entry of each input's `data` by $\pm h$ (with $h$ around $10^{-6}$), re-evaluate the function, and restore the entry. Compare with $\lVert a - b\rVert / (\lVert a\rVert + \lVert b\rVert)$. If a check fails only occasionally, look for a ReLU input within $h$ of zero.

??? tip "Hint for Part G: getting past 95%"
    One hidden layer of 128 ReLU units is enough. With Adam, start at a learning rate around $3 \times 10^{-3}$; with SGD and momentum 0.9, around 0.1. Use batches of 32 and 20 to 40 epochs. If the first epoch's loss doesn't start near $\log 10 \approx 2.3$, or doesn't fall, go back to [A neural network in NumPy](../chapters/06-neural-networks/03-neural-network-in-numpy.md#debugging-a-network-that-wont-learn) and work through the checklist: initial loss, tiny-batch overfit, learning-rate sweep.

## Solution

When you're done, or truly stuck, compare with the [worked solution](solutions/level-6-capstone.md). It includes the complete library, tests, and training script, their outputs, and a sample write-up.

## Next

Congratulations on finishing Level 6. You've built, by hand, everything a deep learning framework does for you. [Level 7: Deep learning with PyTorch](../chapters/07-deep-learning-pytorch/index.md) switches to the real thing, and you'll recognize every piece: tensors with `requires_grad`, `backward()`, `nn.Module`, and `torch.optim`.
