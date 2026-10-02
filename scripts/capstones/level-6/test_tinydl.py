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
