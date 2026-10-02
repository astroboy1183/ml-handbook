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
