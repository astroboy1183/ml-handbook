# Level 6 capstone solution: a tiny deep learning library

> **Level 6 · Capstone solution** · Back to the [capstone brief](../level-6-capstone.md)

This is one complete, worked solution. Yours can differ in structure and naming and still be right: compare it against the [acceptance checklist](../level-6-capstone.md#acceptance-checklist), not line by line. The project is saved in the repository under `scripts/capstones/level-6/`. To reproduce everything below:

<!-- skip-run -->
```bash
cd scripts/capstones/level-6
python test_tinydl.py                   # gradient checks and behavior tests (a few seconds)
python train.py                         # Adam, 30 epochs (about a second)
python train.py --optimizer sgd         # SGD with momentum
```

The outputs on this page come from running exactly these files with their default seeds.

## The project

```text
scripts/capstones/level-6/
├── tinydl.py        the library: Tensor + autograd, no_grad, cross_entropy, Module, Linear,
│                    ReLU, Tanh, Sequential, SGD, Adam, AdamW, gradcheck, manual_seed
├── test_tinydl.py   gradient checks for every operation, the loss, and a whole MLP;
│                    behavior tests for accumulation, no_grad, and the optimizers
└── train.py         trains a 64-128-10 MLP on load_digits and reports test accuracy
```

The library is about 470 lines including comments and blank lines (about 330 without them). It follows the design of [Autograd from scratch](../../chapters/06-neural-networks/06-autograd-from-scratch.md), with three changes that make it behave like a real framework:

1. **Each operation stores a vector-Jacobian product, not a closure that mutates its parents.** An operation's output keeps its parents and a function `_vjp(g)` that returns one gradient per parent. `backward()` walks the graph in reverse topological order and keeps intermediate gradients in a local dictionary. Only **leaves** get a `.grad`, and they accumulate into it. That's PyTorch's semantics: calling `backward()` twice doubles the leaves' gradients, but never double-counts inside the graph, and intermediate results don't hold on to gradient arrays.
2. **Graphs are only recorded when needed.** An operation records its parents only if some input requires gradients and recording is enabled. Inside `no_grad()`, or on plain data tensors, operations produce ordinary arrays with no graph attached, so evaluation and parameter updates cost no extra memory.
3. **The topological sort is iterative.** A recursive depth-first search hits Python's recursion limit (about 1,000 frames) on long graphs, such as a long chain of operations. An explicit stack has no such limit.

The cell below makes the project importable from a notebook started at the repository root.

```python
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT = Path(os.environ.get("CAPSTONE_DIR", "scripts/capstones/level-6")).resolve()
sys.path.insert(0, str(PROJECT))
import tinydl as td

print("tinydl loaded from", Path(td.__file__).relative_to(PROJECT.parent.parent.parent))
```

```text
tinydl loaded from scripts/capstones/level-6/tinydl.py
```

## Parts A to E: the library

Here's the whole library. Read it top to bottom: global state, the `Tensor` and its operations, `backward()`, the loss, the layers, the optimizers, and `gradcheck`.

<!-- skip-run -->
```python
"""tinydl: a tiny deep learning library with reverse-mode autograd, in NumPy.

Level 6 capstone solution. The pieces, in order:

    Tensor          an n-dimensional array that records the operations applied to it
    no_grad         a context manager that turns graph recording off
    cross_entropy   fused softmax + cross-entropy loss (stable, gradient (P - Y) / n)
    Module, Linear, ReLU, Tanh, Sequential    a small neural-network layer API
    SGD, Adam, AdamW                          optimizers that update parameters in place
    gradcheck       finite-difference check of any function of Tensors
    manual_seed     seeds the generator used for weight initialization

Everything is float64, CPU-only, and written for clarity rather than speed.
"""

import numpy as np

# ----------------------------------------------------------------------------
# Global state: random generator for initialization, and the grad-recording flag
# ----------------------------------------------------------------------------

_rng = np.random.default_rng(0)
_grad_enabled = True


def manual_seed(seed):
    """Seed the generator used to initialize layer weights."""
    global _rng
    _rng = np.random.default_rng(seed)


class no_grad:
    """Context manager: operations inside it are not recorded for backward."""

    def __enter__(self):
        global _grad_enabled
        self._previous, _grad_enabled = _grad_enabled, False

    def __exit__(self, *exc):
        global _grad_enabled
        _grad_enabled = self._previous
        return False


# ----------------------------------------------------------------------------
# Tensor
# ----------------------------------------------------------------------------


def unbroadcast(grad, shape):
    """Sum `grad` down to `shape`, undoing NumPy broadcasting.

    Broadcasting copies a value into many positions, so its gradient is the sum
    of the gradients of all the copies.
    """
    while grad.ndim > len(shape):  # leading axes added by broadcasting
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):  # size-1 axes that were stretched
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


def _as_tensor(x):
    return x if isinstance(x, Tensor) else Tensor(x)


class Tensor:
    """A NumPy array plus the information needed to differentiate through it.

    Leaves are tensors you create (requires_grad=True for parameters).
    Every operation on tensors that require gradients returns a new tensor
    that remembers its parents and a vector-Jacobian product function
    `_vjp(g) -> tuple of parent gradients`.
    """

    def __init__(self, data, requires_grad=False):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = bool(requires_grad)
        self.grad = None  # filled in for leaves by backward()
        self._parents = ()
        self._vjp = None
        self._op = ""

    # -- basic properties ----------------------------------------------------
    shape = property(lambda self: self.data.shape)
    ndim = property(lambda self: self.data.ndim)
    size = property(lambda self: self.data.size)
    is_leaf = property(lambda self: not self._parents)

    def __repr__(self):
        flag = ", requires_grad=True" if self.requires_grad else ""
        op = f", op={self._op!r}" if self._op else ""
        return f"Tensor({np.array2string(self.data, precision=4, threshold=8)}{flag}{op})"

    def numpy(self):
        return self.data

    def item(self):
        return float(self.data)

    def detach(self):
        return Tensor(self.data)

    def zero_grad(self):
        self.grad = None

    # -- graph construction ----------------------------------------------------
    @staticmethod
    def _result(data, parents, op, vjp):
        """Create an op's output; record the graph only if a parent needs gradients."""
        out = Tensor(data)
        if _grad_enabled and any(p.requires_grad for p in parents):
            out.requires_grad = True
            out._parents = parents
            out._vjp = vjp
            out._op = op
        return out

    # -- element-wise arithmetic (with broadcasting) -----------------------------
    def __add__(self, other):
        other = _as_tensor(other)
        a, b = self, other
        return Tensor._result(a.data + b.data, (a, b), "add",
                              lambda g: (unbroadcast(g, a.shape), unbroadcast(g, b.shape)))

    def __mul__(self, other):
        other = _as_tensor(other)
        a, b = self, other
        return Tensor._result(a.data * b.data, (a, b), "mul",
                              lambda g: (unbroadcast(g * b.data, a.shape), unbroadcast(g * a.data, b.shape)))

    def __truediv__(self, other):
        other = _as_tensor(other)
        a, b = self, other
        return Tensor._result(a.data / b.data, (a, b), "div",
                              lambda g: (unbroadcast(g / b.data, a.shape),
                                         unbroadcast(-g * a.data / b.data ** 2, b.shape)))

    def __pow__(self, k):
        if isinstance(k, Tensor):
            raise TypeError("only constant exponents are supported")
        a = self
        return Tensor._result(a.data ** k, (a,), f"pow{k}", lambda g: (g * k * a.data ** (k - 1),))

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-_as_tensor(other))

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return _as_tensor(other) - self

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return _as_tensor(other) / self

    # -- linear algebra ----------------------------------------------------------
    def __matmul__(self, other):
        a, b = self, _as_tensor(other)
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError(f"matmul supports 2-D tensors only, got {a.shape} @ {b.shape}")
        return Tensor._result(a.data @ b.data, (a, b), "matmul",
                              lambda g: (g @ b.data.T, a.data.T @ g))

    @property
    def T(self):
        a = self
        return Tensor._result(a.data.T, (a,), "transpose", lambda g: (g.T,))

    def reshape(self, *shape):
        a = self
        return Tensor._result(a.data.reshape(*shape), (a,), "reshape", lambda g: (g.reshape(a.shape),))

    # -- reductions ----------------------------------------------------------------
    def sum(self, axis=None, keepdims=False):
        a = self

        def vjp(g):
            if axis is not None and not keepdims:  # put the reduced axes back
                g = np.expand_dims(g, axis)
            return (np.broadcast_to(g, a.shape).copy(),)

        return Tensor._result(a.data.sum(axis=axis, keepdims=keepdims), (a,), "sum", vjp)

    def mean(self, axis=None, keepdims=False):
        count = self.data.size if axis is None else np.prod([self.shape[i] for i in np.atleast_1d(axis)])
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / count)

    # -- element-wise functions -----------------------------------------------------
    def relu(self):
        a = self
        return Tensor._result(np.maximum(a.data, 0.0), (a,), "relu", lambda g: (g * (a.data > 0),))

    def tanh(self):
        a, t = self, np.tanh(self.data)
        return Tensor._result(t, (a,), "tanh", lambda g: (g * (1.0 - t * t),))

    def sigmoid(self):
        a = self
        s = np.exp(-np.logaddexp(0.0, -a.data))  # stable 1 / (1 + e^-x)
        return Tensor._result(s, (a,), "sigmoid", lambda g: (g * s * (1.0 - s),))

    def exp(self):
        a, e = self, np.exp(self.data)
        return Tensor._result(e, (a,), "exp", lambda g: (g * e,))

    def log(self):
        a = self
        return Tensor._result(np.log(a.data), (a,), "log", lambda g: (g / a.data,))

    # -- backward ----------------------------------------------------------------
    def _topological_order(self):
        """Nodes reachable from self, parents before children (iterative DFS)."""
        order, visited, stack = [], set(), [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            stack.extend((p, False) for p in node._parents if id(p) not in visited)
        return order

    def backward(self, grad=None):
        """Reverse-mode autodiff from this tensor.

        Gradients flow through intermediate nodes in a local table and are
        *accumulated* (+=) into the .grad of leaf tensors that require grad,
        exactly like PyTorch. Call zero_grad() (or optimizer.zero_grad()) between steps.
        """
        if not self.requires_grad:
            raise RuntimeError("backward() on a tensor that doesn't require grad")
        if grad is None:
            if self.size != 1:
                raise RuntimeError("backward() needs a gradient argument for non-scalar tensors")
            grad = np.ones_like(self.data)
        grads = {id(self): np.asarray(grad, dtype=np.float64)}
        for node in reversed(self._topological_order()):
            g = grads.pop(id(node), None)
            if g is None:
                continue
            if node.is_leaf:
                node.grad = g.copy() if node.grad is None else node.grad + g
                continue
            for parent, pg in zip(node._parents, node._vjp(g)):
                if parent.requires_grad:
                    grads[id(parent)] = pg if id(parent) not in grads else grads[id(parent)] + pg


# ----------------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------------


def cross_entropy(logits, targets):
    """Mean softmax cross-entropy. logits: Tensor (n, K); targets: integer array (n,)."""
    targets = np.asarray(targets, dtype=int)
    n = len(targets)
    z = logits.data - logits.data.max(axis=1, keepdims=True)
    log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    loss = -log_probs[np.arange(n), targets].mean()

    def vjp(g):
        d = np.exp(log_probs)
        d[np.arange(n), targets] -= 1.0
        return (g * d / n,)

    return Tensor._result(loss, (logits,), "cross_entropy", vjp)


def mse_loss(pred, target):
    """Mean squared error, composed from primitives."""
    return ((pred - _as_tensor(target)) ** 2).mean()


# ----------------------------------------------------------------------------
# Layers
# ----------------------------------------------------------------------------


class Module:
    """Base class: finds parameters (Tensors with requires_grad) in attributes, recursively."""

    training = True

    def __call__(self, *args):
        return self.forward(*args)

    def children(self):
        for value in vars(self).values():
            if isinstance(value, Module):
                yield value
            elif isinstance(value, (list, tuple)):
                yield from (v for v in value if isinstance(v, Module))

    def parameters(self):
        seen = set()
        for value in vars(self).values():
            if isinstance(value, Tensor) and value.requires_grad and id(value) not in seen:
                seen.add(id(value))
                yield value
        for child in self.children():
            for p in child.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    yield p

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def train(self, mode=True):
        self.training = mode
        for child in self.children():
            child.train(mode)
        return self

    def eval(self):
        return self.train(False)


class Linear(Module):
    """y = x @ W + b, with W of shape (d_in, d_out). He initialization by default."""

    def __init__(self, d_in, d_out, init="he"):
        std = np.sqrt(2.0 / d_in) if init == "he" else np.sqrt(2.0 / (d_in + d_out))
        self.W = Tensor(_rng.normal(0.0, std, size=(d_in, d_out)), requires_grad=True)
        self.b = Tensor(np.zeros(d_out), requires_grad=True)

    def forward(self, x):
        return _as_tensor(x) @ self.W + self.b


class ReLU(Module):
    def forward(self, x):
        return x.relu()


class Tanh(Module):
    def forward(self, x):
        return x.tanh()


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ----------------------------------------------------------------------------
# Optimizers
# ----------------------------------------------------------------------------


class Optimizer:
    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is not None:
                self._update(i, p, p.grad)

    def _update(self, i, p, g):
        raise NotImplementedError


class SGD(Optimizer):
    """SGD with optional momentum, Nesterov momentum, and (coupled, L2) weight decay."""

    def __init__(self, params, lr, momentum=0.0, nesterov=False, weight_decay=0.0):
        super().__init__(params, lr)
        self.momentum, self.nesterov, self.weight_decay = momentum, nesterov, weight_decay
        self.velocity = [np.zeros_like(p.data) for p in self.params]

    def _update(self, i, p, g):
        if self.weight_decay:
            g = g + self.weight_decay * p.data
        if self.momentum:
            v = self.velocity[i]
            v *= self.momentum
            v += g
            g = g + self.momentum * v if self.nesterov else v
        p.data -= self.lr * g


class Adam(Optimizer):
    """Adam with bias correction. weight_decay is coupled L2 unless decoupled=True (AdamW)."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, decoupled=False):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps, self.weight_decay, self.decoupled = eps, weight_decay, decoupled
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def _update(self, i, p, g):
        if self.weight_decay and not self.decoupled:
            g = g + self.weight_decay * p.data
        m, v = self.m[i], self.v[i]
        m *= self.beta1
        m += (1 - self.beta1) * g
        v *= self.beta2
        v += (1 - self.beta2) * g * g
        m_hat = m / (1 - self.beta1 ** self.t)
        v_hat = v / (1 - self.beta2 ** self.t)
        if self.weight_decay and self.decoupled:
            p.data -= self.lr * self.weight_decay * p.data
        p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params, lr, betas, eps, weight_decay, decoupled=True)


# ----------------------------------------------------------------------------
# Gradient checking
# ----------------------------------------------------------------------------


def gradcheck(fn, inputs, h=1e-6):
    """Compare autograd gradients of the scalar fn(*inputs) with central differences.

    inputs: Tensors with requires_grad=True (float64). Returns a list with the
    relative error for each input: below ~1e-7 is correct, above ~1e-4 is a bug.
    """
    for x in inputs:
        x.grad = None
    out = fn(*inputs)
    out.backward()
    errors = []
    for x in inputs:
        analytic = np.zeros_like(x.data) if x.grad is None else x.grad.copy()
        numeric = np.zeros_like(x.data)
        with no_grad():
            for idx in np.ndindex(x.shape):
                old = x.data[idx]
                x.data[idx] = old + h
                f_plus = fn(*inputs).item()
                x.data[idx] = old - h
                f_minus = fn(*inputs).item()
                x.data[idx] = old
                numeric[idx] = (f_plus - f_minus) / (2 * h)
        denom = max(np.linalg.norm(analytic) + np.linalg.norm(numeric), 1e-12)
        errors.append(float(np.linalg.norm(analytic - numeric) / denom))
    return errors
```

A few design notes, part by part.

**Part A, the tensor.** Every operation is three lines: compute the forward value with NumPy, list the parents, and give the VJP. For example, `__mul__`'s VJP returns `(g * b.data, g * a.data)`, each passed through `unbroadcast`. `_result` decides whether to record anything. `backward()` refuses non-scalar outputs without an explicit gradient, as PyTorch does, because "the gradient of a vector" is ambiguous.

**Part B, broadcasting.** `unbroadcast` sums over the leading axes that broadcasting added, then over stretched size-1 axes with `keepdims=True`. The docstring states the reason: broadcasting copies a value into many positions, and a value used in many places gets the sum of their gradients. The `sum` VJP is the mirror image: it broadcasts the gradient back to the input's shape.

**Part C, the loss.** `cross_entropy` computes log-probabilities with the max-shift and never takes the log of a probability, so it's finite for any logits. Its VJP is $\frac{1}{n}(\mathbf{P} - \mathbf{Y})$, scaled by the incoming gradient $g$ (which is 1 when the loss is the output, but needn't be, for example if you add a regularization term to the loss).

**Part D, layers.** `Module.parameters()` discovers parameters by looking at the module's attributes, recursing into sub-modules and lists of modules, and de-duplicating by `id`. So `Linear` only has to store `W` and `b` as attributes, and `Sequential` only has to store its list. No layer has a backward pass. `Linear` draws from the module-level generator that `manual_seed` resets, so a model built twice with the same seed is identical.

**Part E, optimizers.** Both work on `p.data` in place, skip parameters with no gradient, and keep their state (velocities, moments) in lists aligned with the parameters. `Adam` has the bias correction, and `AdamW` is `Adam` with decoupled weight decay, as in [Optimizers](../../chapters/06-neural-networks/04-optimizers.md#weight-decay-and-adamw).

A quick tour of the behavior, in a notebook:

```python
td.manual_seed(0)
x = td.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)   # (2, 3)
b = td.Tensor([0.5, -1.0, 2.0], requires_grad=True)                     # (3,), broadcast over rows
loss = ((x * b + x).relu() * x).sum()                                   # x is used three times
loss.backward()
print("loss:", loss.item(), " op:", loss._op)
print("x.grad:\n", x.grad)
print("b.grad:", b.grad, " shape", b.grad.shape)

loss = ((x * b + x).relu() * x).sum()
loss.backward()                                       # a second backward accumulates
print("b.grad after a second backward:", b.grad)
with td.no_grad():
    y = x * 2.0
print("inside no_grad: requires_grad =", y.requires_grad, ", leaf =", y.is_leaf)
```

```text
loss: 160.5  op: sum
x.grad:
 [[ 3.  0. 18.]
 [12.  0. 36.]]
b.grad: [17.  0. 45.]  shape (3,)
b.grad after a second backward: [34.  0. 90.]
inside no_grad: requires_grad = False , leaf = True
```

Check one entry by hand. With $z = x(b + 1)$ passed through a ReLU and multiplied by $x$, an active entry contributes $x^2(b + 1)$ to the loss, whose derivative with respect to $x$ is $2x(b + 1)$: for $x = 1$, $b = 0.5$ that's $3$, matching `x.grad[0, 0]`. The middle column has $b + 1 = 0$, so the ReLU blocks it and its gradient is 0. The bias gradient sums over the two rows (each entry is $\sum_i x_i^2$ over an active column), and its shape is `(3,)`, not `(2, 3)`.

## Part F: the gradient checks

<!-- skip-run -->
```python
"""Gradient checks and behavior tests for tinydl.

Run with `python test_tinydl.py` (prints a report, exit code 1 on failure),
or with pytest, which collects the test_* functions.
"""

import sys

import numpy as np

import tinydl as td

TOL = 1e-7  # relative error threshold for float64 central differences


def _t(rng, *shape):
    return td.Tensor(rng.normal(size=shape), requires_grad=True)


def _check(name, fn, inputs):
    errors = td.gradcheck(fn, inputs)
    worst = max(errors)
    assert worst < TOL, f"{name}: relative error {worst:.1e}"
    return worst


def test_elementwise_ops_with_broadcasting():
    rng = np.random.default_rng(0)
    cases = {
        "add (3,4)+(4,)": (lambda a, b: (a + b).sum(), [_t(rng, 3, 4), _t(rng, 4)]),
        "sub (3,1)-(1,4)": (lambda a, b: ((a - b) ** 2).sum(), [_t(rng, 3, 1), _t(rng, 1, 4)]),
        "mul (2,3,4)*(3,1)": (lambda a, b: (a * b).sum(), [_t(rng, 2, 3, 4), _t(rng, 3, 1)]),
        "div (3,4)/(4,)": (lambda a, b: (a / (b * b + 1.0)).sum(), [_t(rng, 3, 4), _t(rng, 4)]),
        "pow, neg, rsub": (lambda a: (1.0 - (-a) ** 3).sum(), [_t(rng, 5)]),
        "relu": (lambda a: (a.relu() * a).sum(), [_t(rng, 4, 5)]),
        "tanh": (lambda a: a.tanh().sum(), [_t(rng, 4, 5)]),
        "sigmoid": (lambda a: a.sigmoid().sum(), [_t(rng, 4, 5)]),
        "exp, log": (lambda a: (a.exp() + 1.0).log().sum(), [_t(rng, 4, 5)]),
        "value used twice": (lambda a: (a * a + a).sum(), [_t(rng, 3, 3)]),
    }
    return {name: _check(name, fn, inputs) for name, (fn, inputs) in cases.items()}


def test_matmul_reductions_and_shapes():
    rng = np.random.default_rng(1)
    cases = {
        "matmul": (lambda a, b: (a @ b).tanh().sum(), [_t(rng, 3, 4), _t(rng, 4, 2)]),
        "sum axis=0": (lambda a: (a.sum(axis=0) ** 2).sum(), [_t(rng, 3, 4)]),
        "sum axis=1 keepdims": (lambda a: (a * a.sum(axis=1, keepdims=True)).sum(), [_t(rng, 3, 4)]),
        "mean": (lambda a: (a.mean(axis=0) * a.mean()).sum(), [_t(rng, 3, 4)]),
        "reshape, transpose": (lambda a: (a.reshape(4, 3).T * a).sum(), [_t(rng, 3, 4)]),
    }
    return {name: _check(name, fn, inputs) for name, (fn, inputs) in cases.items()}


def test_cross_entropy():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 5, size=6)
    logits = _t(rng, 6, 5)
    err = _check("cross_entropy", lambda z: td.cross_entropy(z, y), [logits])
    # value matches a direct log-softmax computation, and the gradient is (P - Y) / n
    z = logits.data
    p = np.exp(z - z.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    expected = -np.mean(np.log(p[np.arange(6), y]))
    logits.grad = None
    loss = td.cross_entropy(logits, y)
    loss.backward()
    assert abs(loss.item() - expected) < 1e-12
    assert np.allclose(logits.grad, (p - np.eye(5)[y]) / 6)
    # stable for extreme logits
    big = td.Tensor(np.array([[1000.0, -1000.0]]), requires_grad=True)
    assert np.isfinite(td.cross_entropy(big, [1]).item())
    return {"cross_entropy": err}


def test_mlp_end_to_end():
    td.manual_seed(0)
    model = td.Sequential(td.Linear(4, 6), td.ReLU(), td.Linear(6, 5), td.Tanh(), td.Linear(5, 3))
    rng = np.random.default_rng(3)
    X, y = rng.normal(size=(7, 4)), rng.integers(0, 3, size=7)
    params = list(model.parameters())
    assert len(params) == 6
    errors = td.gradcheck(lambda *ps: td.cross_entropy(model(td.Tensor(X)), y), params)
    worst = max(errors)
    assert worst < TOL, f"MLP: relative error {worst:.1e}"
    return {"MLP (6 parameter tensors)": worst}


def test_accumulation_and_no_grad():
    w = td.Tensor([1.0, -2.0], requires_grad=True)
    for _ in range(3):
        (w * w).sum().backward()  # gradient 2w each time
    assert np.allclose(w.grad, 3 * 2 * w.data), "backward() must accumulate into leaves"
    w.zero_grad()
    assert w.grad is None
    with td.no_grad():
        out = w * 2.0
    assert not out.requires_grad and out.is_leaf
    x = td.Tensor([3.0])  # does not require grad
    assert not (x * 2.0).requires_grad
    return {}


def test_optimizers():
    # SGD with momentum minimizes a quadratic
    w = td.Tensor([5.0, -3.0], requires_grad=True)
    opt = td.SGD([w], lr=0.05, momentum=0.9)
    for _ in range(300):
        opt.zero_grad()
        ((w - 1.0) ** 2).sum().backward()
        opt.step()
    assert np.allclose(w.data, 1.0, atol=1e-5), w.data
    # Adam's first step has size lr in every coordinate, whatever the gradient's scale
    w = td.Tensor([1.0, 1.0], requires_grad=True)
    opt = td.Adam([w], lr=0.01)
    (w * td.Tensor([1000.0, 0.001])).sum().backward()
    opt.step()
    assert np.allclose(1.0 - w.data, 0.01, rtol=1e-4), w.data
    # AdamW with zero gradient decays weights by exactly (1 - lr * wd) per step
    w = td.Tensor([2.0], requires_grad=True)
    opt = td.AdamW([w], lr=0.1, weight_decay=0.5)
    for _ in range(10):
        w.grad = np.zeros(1)
        opt.step()
    assert np.allclose(w.data, 2.0 * (1 - 0.05) ** 10)
    return {}


def test_against_pytorch():
    try:
        import torch
    except ImportError:
        return {"(PyTorch not installed, skipped)": 0.0}
    torch.set_num_threads(4)
    td.manual_seed(1)
    model = td.Sequential(td.Linear(8, 16), td.ReLU(), td.Linear(16, 4))
    rng = np.random.default_rng(4)
    X, y = rng.normal(size=(10, 8)), rng.integers(0, 4, size=10)
    loss = td.cross_entropy(model(td.Tensor(X)), y)
    loss.backward()
    W1, b1, W2, b2 = (torch.tensor(p.data, requires_grad=True) for p in model.parameters())
    logits = torch.relu(torch.tensor(X) @ W1 + b1) @ W2 + b2
    tloss = torch.nn.functional.cross_entropy(logits, torch.tensor(y))
    tloss.backward()
    diff = max(np.max(np.abs(p.grad - t.grad.numpy())) for p, t in zip(model.parameters(), (W1, b1, W2, b2)))
    assert abs(loss.item() - tloss.item()) < 1e-12 and diff < 1e-12, diff
    return {"max |grad - torch grad|": diff}


def run_all():
    tests = [test_elementwise_ops_with_broadcasting, test_matmul_reductions_and_shapes, test_cross_entropy,
             test_mlp_end_to_end, test_accumulation_and_no_grad, test_optimizers, test_against_pytorch]
    ok = True
    for test in tests:
        try:
            details = test()
            print(f"PASS  {test.__name__}")
            for name, value in details.items():
                print(f"        {name:<28} {value:.1e}")
        except AssertionError as exc:
            ok = False
            print(f"FAIL  {test.__name__}: {exc}")
    print("all tests passed" if ok else "SOME TESTS FAILED")
    return ok


if __name__ == "__main__":
    sys.exit(0 if run_all() else 1)
```

Run the whole suite:

```python
import test_tinydl
ok = test_tinydl.run_all()
```

```text
PASS  test_elementwise_ops_with_broadcasting
        add (3,4)+(4,)               3.4e-10
        sub (3,1)-(1,4)              6.4e-11
        mul (2,3,4)*(3,1)            8.1e-11
        div (3,4)/(4,)               8.5e-11
        pow, neg, rsub               6.3e-11
        relu                         2.1e-10
        tanh                         7.0e-11
        sigmoid                      1.9e-09
        exp, log                     7.0e-10
        value used twice             6.8e-11
PASS  test_matmul_reductions_and_shapes
        matmul                       6.0e-11
        sum axis=0                   1.4e-10
        sum axis=1 keepdims          1.0e-10
        mean                         5.5e-11
        reshape, transpose           9.6e-11
PASS  test_cross_entropy
        cross_entropy                7.5e-10
PASS  test_mlp_end_to_end
        MLP (6 parameter tensors)    6.9e-10
PASS  test_accumulation_and_no_grad
PASS  test_optimizers
PASS  test_against_pytorch
        max |grad - torch grad|      1.1e-16
all tests passed
```

Every relative error is between about $10^{-11}$ and $2 \times 10^{-9}$, at least 50 times below the $10^{-7}$ threshold, including a three-layer network checked through all six parameter tensors at once, and the gradients agree with PyTorch's to rounding error. The behavior tests cover what gradient checks can't see: accumulation across `backward()` calls, `no_grad()`, convergence of SGD with momentum, Adam's first-step size of exactly $\eta$ for gradients of 1,000 and 0.001, and AdamW's decay factor.

## Part G: training on digits

<!-- skip-run -->
```python
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
```

`main()` takes an argument list, so it can be called from a notebook as well as from the command line. Train with Adam (the default) and with SGD and momentum:

```python
import train
adam = train.main([])
print()
sgd = train.main(["--optimizer", "sgd"])
```

```text
model: 64-128-10 ReLU MLP, 9,610 parameters; optimizer adam (lr 0.003), batch size 32
initial training loss 2.498 (log 10 = 2.303)
epoch   1: train loss 1.5414, val acc 0.889
epoch   5: train loss 0.1638, val acc 0.958
epoch  10: train loss 0.0776, val acc 0.967
epoch  15: train loss 0.0363, val acc 0.961
epoch  20: train loss 0.0249, val acc 0.978
epoch  25: train loss 0.0142, val acc 0.969
epoch  30: train loss 0.0090, val acc 0.972
best val acc 0.978 at epoch 20 (restored)
test accuracy: 0.969 -> PASS (target > 95%)

model: 64-128-10 ReLU MLP, 9,610 parameters; optimizer sgd (lr 0.1), batch size 32
initial training loss 2.498 (log 10 = 2.303)
epoch   1: train loss 1.0216, val acc 0.908
epoch   5: train loss 0.0611, val acc 0.967
epoch  10: train loss 0.0268, val acc 0.969
epoch  15: train loss 0.0077, val acc 0.958
epoch  20: train loss 0.0058, val acc 0.972
epoch  25: train loss 0.0039, val acc 0.969
epoch  30: train loss 0.0029, val acc 0.969
best val acc 0.975 at epoch 9 (restored)
test accuracy: 0.964 -> PASS (target > 95%)
```

Both pass the 95% bar on the test set. The initial loss of about 2.5 is close to $\log 10$, as it should be for a freshly initialized 10-class network. The learning curves:

```python
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for name, run in [("Adam, lr 3e-3", adam), ("SGD + momentum, lr 0.1", sgd)]:
    epochs = np.arange(1, len(run["history"]["loss"]) + 1)
    axes[0].semilogy(epochs, run["history"]["loss"], marker="o", ms=3, label=name)
    axes[1].plot(epochs, run["history"]["val_acc"], marker="o", ms=3, label=name)
    axes[1].plot(run["best_epoch"], run["val_acc"], "k*", ms=10)
axes[0].set_xlabel("epoch"); axes[0].set_title("training loss (log scale)"); axes[0].legend()
axes[1].set_xlabel("epoch"); axes[1].set_title("validation accuracy (star: restored checkpoint)")
axes[1].set_ylim(0.85, 1.0)
plt.tight_layout()
plt.show()
```

![Left: training loss on a log scale for Adam and SGD with momentum over 30 epochs, both falling steadily, SGD faster. Right: validation accuracy for both, rising above 0.95 within five epochs and fluctuating between about 0.96 and 0.98, with stars at the best epochs](../../assets/figures/solutions/level-6-capstone-fig1.png)

*Learning curves for the two optimizers. Both reach the 95% region within a few epochs; after that the validation accuracy fluctuates by about one image either way, while the training loss keeps falling.*

Is the result robust, or a lucky seed? Train with five seeds for each optimizer:

```python
import contextlib
import io

for opt in ["adam", "sgd"]:
    accs = []
    for seed in range(5):
        with contextlib.redirect_stdout(io.StringIO()):          # silence the training logs
            accs.append(train.main(["--optimizer", opt, "--seed", str(seed)])["test_acc"])
    print(f"{opt:<5} test accuracy over 5 seeds: mean {np.mean(accs):.3f}, min {np.min(accs):.3f}, "
          f"max {np.max(accs):.3f}")
```

```text
adam  test accuracy over 5 seeds: mean 0.971, min 0.969, max 0.972
sgd   test accuracy over 5 seeds: mean 0.967, min 0.964, max 0.972
```

Every seed clears 95%, so the result doesn't depend on a lucky initialization. The two optimizers are within noise of each other on this problem, consistent with the comparison in [Optimizers](../../chapters/06-neural-networks/04-optimizers.md#optimizers-on-the-digits-network).

## Part H: the write-up

> **To:** Priya · **Re:** a NumPy-only digit recognizer for the scanners
>
> **Result.** `tinydl` trains a 64-128-10 network on handwritten digits to **96.9% test accuracy** (Adam, seed 0), on 360 images it never saw during training or model selection. Across five seeds it ranges from 96.9% to 97.2%. The current template matcher misreads about one digit in eight; this network misreads about one in thirty.
>
> **What's in it.** One file, about 330 lines of code, depending only on NumPy: a `Tensor` with reverse-mode autograd (13 primitive operations, plus composed ones such as `-` and `mean`), a fused cross-entropy loss, `Linear`/`ReLU`/`Tanh`/`Sequential` layers, SGD and Adam(W), and a gradient checker. Training takes about a second on a laptop.
>
> **The decision I'd defend.** Layers contain no backward code. Each primitive operation has one small, tested gradient rule, and everything else is composed from them, so a new layer next year is a forward function and nothing else, and it inherits correct gradients.
>
> **Evidence the gradients are right.** Every operation, the loss, and a whole three-layer network pass finite-difference checks with relative errors of $10^{-9}$ or better (the bug threshold is $10^{-4}$), and the network's gradients match PyTorch's to $10^{-16}$.
>
> **Limitation that matters on the device.** Everything is float64 and single-threaded, and inference re-uses the training code path. For deployment, I'd export the trained weights and run inference as two NumPy matrix multiplies in float32 (about 10,000 parameters, 80 KB), which needs none of the autograd machinery. The real risk is the data, not the code: these are scikit-learn's clean $8 \times 8$ digits. Before shipping, we need a labeled sample of real slip crops to measure accuracy on the device's actual images.

### How this solution meets the checklist

- **No hand-written layer backward passes:** `Linear`, `ReLU`, `Tanh`, and `Sequential` only define `forward`.
- **Topological order, each node once:** `_topological_order` is an iterative depth-first search with a visited set; `backward()` processes each node once, after all its consumers, and pops its gradient from the table.
- **Accumulation, `zero_grad`, `no_grad`:** tested in `test_accumulation_and_no_grad`, and demonstrated in the tour above.
- **Broadcast gradient shapes:** three broadcasting patterns in `test_elementwise_ops_with_broadcasting`, and the `(3,)` bias gradient in the tour.
- **Fused, stable cross-entropy with gradient $(\mathbf{P} - \mathbf{Y})/n$:** tested in `test_cross_entropy`, including logits of $\pm 1000$.
- **Gradient checks below $10^{-7}$, including a full network:** see the test output; the worst is under $10^{-8}$.
- **Adam bias correction:** `test_optimizers` checks that the first step is exactly $\eta$ per coordinate.
- **Split and single test evaluation:** `load_data()` reproduces the brief's split; `main()` restores the best-validation checkpoint before its only call on the test set.
- **Above 95%, reproducibly:** 96.9% with Adam and 96.4% with SGD at seed 0, at least 96.4% across ten runs; the same command gives the same output.
- **Write-up:** above, with the result first, the gradient evidence, and a concrete limitation.
