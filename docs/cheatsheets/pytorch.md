# PyTorch Cheat Sheet

A dense reference for everyday PyTorch 2: tensors, autograd, modules, layers, losses, optimizers, data loading, the training loop, checkpoints, devices and mixed precision, and debugging.

Related chapters: [PyTorch fundamentals](../chapters/07-deep-learning-pytorch/01-pytorch-fundamentals.md) · [The training loop](../chapters/07-deep-learning-pytorch/02-the-training-loop.md) · [CNNs](../chapters/07-deep-learning-pytorch/03-convolutional-networks.md) · [Sequence models](../chapters/07-deep-learning-pytorch/04-sequence-models.md) · [Embeddings](../chapters/07-deep-learning-pytorch/05-embeddings-and-nlp.md) · [Transfer learning](../chapters/07-deep-learning-pytorch/06-transfer-learning.md)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

torch.manual_seed(0)
```

## Tensors

| Code | Result / note |
|---|---|
| `torch.tensor([[1., 2.], [3., 4.]])` | copies data; dtype inferred (`float32` for floats, `int64` for ints) |
| `torch.zeros(2, 3)`, `torch.ones(3)`, `torch.full((2, 2), 7.)` | constant tensors |
| `torch.arange(0, 10, 2)`, `torch.linspace(0, 1, 5)` | ranges (stop excluded / included) |
| `torch.rand(2, 3)`, `torch.randn(2, 3)`, `torch.randint(0, 10, (5,))` | uniform $[0, 1)$, standard normal, integers |
| `torch.zeros_like(x)`, `torch.randn_like(x)` | same shape, dtype, device as `x` |
| `x.shape`, `x.dtype`, `x.device`, `x.ndim`, `x.numel()` | attributes |
| `x.float()`, `x.long()`, `x.to(torch.float16)` | change dtype |
| `x.reshape(3, -1)`, `x.view(-1)` | reshape (`view` needs compatible memory; `reshape` copies if needed) |
| `x.unsqueeze(0)`, `x.squeeze(1)`, `x.flatten(1)` | add / remove size-1 dims; flatten all but the batch dim |
| `x.permute(0, 2, 3, 1)`, `x.transpose(0, 1)`, `x.T` | reorder dims (views, not contiguous) |
| `torch.cat([a, b], dim=0)`, `torch.stack([a, b])` | join along an existing / a new dim |
| `x.chunk(4, dim=1)`, `x.split(16, dim=1)` | split into pieces |
| `x[:, 0]`, `x[1:, ::2]` | basic slicing: **views** that share memory |
| `x[[0, 2]]`, `x[x > 0]` | fancy / boolean indexing: **copies** |
| `x.clone()`, `x.contiguous()` | explicit copy; compact copy |
| `x.add_(1)`, `x.zero_()` | in-place ops end in `_` (avoid on tensors autograd needs) |
| `x.sum(dim=1, keepdim=True)`, `x.mean(0)`, `x.max(1).values`, `x.argmax(1)` | reductions (`dim` = NumPy's `axis`) |
| `a @ b`, `torch.einsum("bij,bjk->bik", a, b)` | matrix product, batched via einsum |
| `torch.where(c, a, b)`, `x.clamp(0, 1)` | elementwise select, clip |

**Dtypes:** `float32` (default), `float64` (gradient checks), `float16`/`bfloat16` (mixed precision), `int64` (labels, indices), `uint8` (raw pixels), `bool` (masks). Python scalars don't promote a tensor's float dtype.

**Broadcasting:** NumPy's rules. Align shapes from the right; sizes must match or be 1. Watch `(n, 1) - (n,)`, which silently becomes `(n, n)`.

## NumPy interop

| Code | Shares memory? |
|---|---|
| `torch.from_numpy(a)` | yes (keeps `float64`; add `.float()`) |
| `torch.as_tensor(a, dtype=torch.float32)` | when possible |
| `torch.tensor(a)` | no, always copies |
| `t.numpy()` | yes (CPU, no grad only) |
| `t.detach().cpu().numpy()` | the universal way back |
| `t.item()` | a one-element tensor as a Python number |

## Autograd

| Code | Meaning |
|---|---|
| `w = torch.randn(3, requires_grad=True)` | a leaf you want gradients for |
| `loss.backward()` | fills `.grad` of leaves; **adds** to existing grads; frees the graph |
| `w.grad`, `w.grad = None` | the gradient; clear it |
| `y.backward(gradient=v)` | non-scalar output: computes $J^\top\mathbf{v}$ |
| `torch.autograd.grad(loss, [w])` | returns grads without touching `.grad` |
| `x.grad_fn`, `x.is_leaf` | the backward node; is it a leaf |
| `with torch.no_grad():` | don't record (evaluation, manual updates) |
| `with torch.inference_mode():` | stricter, faster `no_grad` for inference |
| `t.detach()` | same data, cut from the graph |
| `p.requires_grad_(False)` | freeze a parameter |
| `h.retain_grad()` | keep `.grad` for a non-leaf |
| `torch.autograd.gradcheck(f, (x64,))` | compare with finite differences (use float64) |

## Modules

```python
class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, p=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Dropout(p), nn.Linear(d_hidden, d_out))

    def forward(self, x):
        return self.net(x)

model = MLP(20, 64, 3)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
```

| Code | Use |
|---|---|
| `model(x)` | call (runs hooks); never `model.forward(x)` |
| `model.parameters()`, `model.named_parameters()` | learnable tensors (registered automatically) |
| `model.named_buffers()` | state that isn't learned (BN running stats) |
| `self.register_buffer("mask", t)` | add a buffer |
| `nn.ModuleList([...])`, `nn.ModuleDict({...})` | containers that register submodules (a plain list doesn't) |
| `model.train()`, `model.eval()` | dropout / batch norm behavior (not gradients) |
| `model.to(device)` | move all parameters and buffers |
| `model.apply(init_fn)` | apply a function to every submodule |
| `nn.init.kaiming_normal_(m.weight, nonlinearity="relu")`, `nn.init.xavier_uniform_`, `nn.init.zeros_` | initialization |

## Common layers

| Layer | Input → output shape | Parameters |
|---|---|---|
| `nn.Linear(d_in, d_out)` | `(*, d_in)` → `(*, d_out)` | $d_{\text{out}}(d_{\text{in}} + 1)$; weight is `(d_out, d_in)` |
| `nn.Conv2d(c_in, c_out, k, stride, padding)` | `(N, c_in, H, W)` → `(N, c_out, H', W')` | $c_{\text{out}}(c_{\text{in}}k^2 + 1)$ |
| `nn.MaxPool2d(2)`, `nn.AvgPool2d(2)` | halves `H, W` | 0 |
| `nn.AdaptiveAvgPool2d(1)` | `(N, C, H, W)` → `(N, C, 1, 1)` | 0 |
| `nn.ConvTranspose2d(c_in, c_out, 2, stride=2)` | doubles `H, W` | $c_{\text{in}}c_{\text{out}}k^2 + c_{\text{out}}$ |
| `nn.BatchNorm1d(d)`, `nn.BatchNorm2d(c)` | same shape | $2d$ (+ $2d$ buffers) |
| `nn.LayerNorm(d)` | same shape | $2d$ |
| `nn.Dropout(p)` | same shape | 0 |
| `nn.Embedding(V, d)` | `(*)` int64 → `(*, d)` | $Vd$ |
| `nn.EmbeddingBag(V, d, mode="mean")` | flat ids + offsets → `(N, d)` | $Vd$ |
| `nn.RNN/GRU/LSTM(d_in, h, batch_first=True)` | `(N, T, d_in)` → `(N, T, h)`, `h_n` `(layers·dirs, N, h)` | $\{1,3,4\} \times (h\,d_{\text{in}} + h^2 + 2h)$ |
| `nn.MultiheadAttention(d, heads, batch_first=True)` | `(N, T, d)` → `(N, T, d)` | $4d^2 + 4d$ |
| `nn.Flatten()`, `nn.Identity()` | flatten from dim 1; pass-through | 0 |

Conv/pool output size: $\lfloor (I + 2P - K)/S \rfloor + 1$. Activations: `nn.ReLU`, `nn.GELU`, `nn.SiLU`, `nn.Tanh`, `nn.Sigmoid`, `nn.LeakyReLU(0.01)`, `nn.Softmax(dim=-1)`.

## Losses

| Task | Loss | Model output | Target |
|---|---|---|---|
| Multi-class | `F.cross_entropy(logits, y)` | raw logits `(N, C)` | int64 `(N,)` (or probabilities `(N, C)`) |
| Multi-class, imbalanced | `F.cross_entropy(logits, y, weight=w, label_smoothing=0.1)` | logits | int64 |
| Binary / multi-label | `F.binary_cross_entropy_with_logits(z, y, pos_weight=pw)` | raw logits | float, same shape |
| Regression | `F.mse_loss(pred, y)`, `F.l1_loss`, `F.smooth_l1_loss` | values | float, **same shape** |
| Sequences with padding | `F.cross_entropy(logits.reshape(-1, C), y.reshape(-1), ignore_index=PAD)` | logits | int64 |
| Distributions | `F.kl_div(log_q, p, reduction="batchmean")` | log-probabilities | probabilities |

Never apply softmax or sigmoid before the `..._with_logits` / `cross_entropy` losses. `reduction="sum"` or `"none"` for custom weighting.

## Optimizers and schedulers

| Code | Notes |
|---|---|
| `torch.optim.SGD(params, lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)` | classic for CNNs |
| `torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)` | the default choice |
| `torch.optim.Adam(params, lr=1e-3)` | L2 inside the gradient (prefer AdamW) |
| `[{"params": backbone.parameters(), "lr": 1e-4}, {"params": head.parameters(), "lr": 1e-3}]` | parameter groups (discriminative LRs) |
| `lr_scheduler.OneCycleLR(opt, max_lr, epochs=E, steps_per_epoch=S)` | step **per batch** |
| `lr_scheduler.CosineAnnealingLR(opt, T_max=E)` | step per epoch |
| `lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)` | step per epoch |
| `lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)` | `sched.step(val_loss)` |
| `lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warmup))` | linear warmup |
| `nn.utils.clip_grad_norm_(model.parameters(), 1.0)` | after `backward`, before `step`; returns the norm |
| `opt.param_groups[0]["lr"]` | read the current learning rate |

## Data loading

| Code | Use |
|---|---|
| `TensorDataset(X, y)` | in-memory tensors |
| custom `Dataset` with `__len__` and `__getitem__(i)` | anything else; apply transforms in `__getitem__` |
| `DataLoader(ds, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(0))` | reproducible shuffling |
| `num_workers=4, pin_memory=True, persistent_workers=True` | parallel loading for GPU training |
| `drop_last=True` | avoid a tiny last batch (batch norm) |
| `collate_fn=my_collate` | custom batching (padding, offsets) |
| `torch.utils.data.Subset(ds, idx)`, `random_split(ds, [0.9, 0.1], generator=g)` | splits |
| `nn.utils.rnn.pad_sequence(seqs, batch_first=True)` | pad variable-length tensors |
| `pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)` | RNNs skip padding |
| `torchvision.transforms.v2.Compose([v2.RandomCrop(32, padding=4), v2.RandomHorizontalFlip()])` | augmentation (training only) |

## The training-loop template

```python
X = torch.randn(1000, 20)
y = (X[:, 0] + X[:, 1] > 0).long() + (X[:, 2] > 1).long()          # 3 classes
train_dl = DataLoader(TensorDataset(X[:800], y[:800]), batch_size=64, shuffle=True,
                      generator=torch.Generator().manual_seed(0))
val_dl = DataLoader(TensorDataset(X[800:], y[800:]), batch_size=256)

def train_one_epoch(model, loader, opt, scheduler=None, clip=1.0):
    model.train()
    total, n = 0.0, 0
    for xb, yb in loader:
        loss = F.cross_entropy(model(xb), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        if scheduler is not None:
            scheduler.step()                   # per-batch schedulers
        total, n = total + loss.item() * len(yb), n + len(yb)
    return total / n

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    loss, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        logits = model(xb)
        loss += F.cross_entropy(logits, yb, reduction="sum").item()
        correct += (logits.argmax(1) == yb).sum().item()
        n += len(yb)
    return loss / n, correct / n

model = MLP(20, 64, 3)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
best, patience, bad = float("inf"), 5, 0
for epoch in range(1, 51):
    train_loss = train_one_epoch(model, train_dl, opt)
    val_loss, val_acc = evaluate(model, val_dl)
    if val_loss < best - 1e-4:
        best, bad = val_loss, 0
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict()}, "best.pt")
    else:
        bad += 1
        if bad >= patience:
            break
model.load_state_dict(torch.load("best.pt")["model"])
```

Before a long run: loss at init ≈ $\ln C$; overfit one batch to ~0 loss; LR range test; then train with validation, early stopping, and checkpoints; evaluate on test once.

## Saving and loading

| Code | Use |
|---|---|
| `torch.save(model.state_dict(), "model.pt")` | weights only (inference) |
| `model = MLP(...); model.load_state_dict(torch.load("model.pt"))` | rebuild from code, then load |
| `torch.save({"model": ..., "optimizer": ..., "scheduler": ..., "epoch": e, "torch_rng": torch.get_rng_state()}, p)` | resumable checkpoint |
| `torch.load(p, map_location="cpu")` | load a GPU checkpoint on a CPU |
| `torch.load(p)` (`weights_only=True` default) | safe: tensors and plain containers only |
| `model.load_state_dict(sd, strict=False)` | partial loads (returns missing / unexpected keys) |
| `torch.jit.script(model)`, `torch.export.export(model, (x,))` | export for deployment without Python class code |

Never `torch.save(model)` for long-term storage (it pickles the class by reference), and never load pickles you didn't create with `weights_only=False`.

## Devices and mixed precision

```python
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)
xb = X[:8].to(device, non_blocking=True)

with torch.autocast(device_type="cpu", dtype=torch.bfloat16):   # bf16: no scaler needed
    logits = model(xb.cpu() if device != "cpu" else xb)
```

<!-- skip-run -->
```python
scaler = torch.amp.GradScaler("cuda")                 # fp16 on CUDA needs loss scaling
for xb, yb in train_dl:
    xb, yb = xb.to(device), yb.to(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = F.cross_entropy(model(xb), yb)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)                              # before clipping
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)                                  # skips the step on inf/NaN
    scaler.update()
```

| Fact | Detail |
|---|---|
| All operands on one device | no implicit transfers; move batches to the model's device |
| GPU calls are async | `.item()`, `print`, `.cpu()` synchronize; time with `torch.cuda.synchronize()` |
| Memory | params + grads + optimizer state (2× for Adam) + activations; shrink batch, use AMP |
| Autocast | matmul/conv in 16-bit; softmax, losses, norms in fp32; weights stay fp32 |
| fp16 vs bf16 | fp16: 5 exponent bits, needs `GradScaler`; bf16: fp32's range, less precision |
| Reproducibility | `torch.manual_seed`, seeded loader generator; on GPU `torch.use_deterministic_algorithms(True)` |
| Threads | `torch.set_num_threads(n)` on shared CPUs |

## Debugging

| Symptom | Likely causes |
|---|---|
| Loss doesn't move | optimizer got the wrong params; LR ~0; `no_grad`/`detach` in the forward; frozen layers |
| Loss stuck above 0 on one batch | softmax before `cross_entropy`; labels misaligned; bottleneck |
| Loss `NaN`/`inf` | LR too high; `log(0)`; division by zero; fp16 overflow (use a scaler or bf16) |
| Initial loss far from $\ln C$ | unnormalized inputs; bad init; leakage if much lower |
| Great train, bad val | overfitting: augmentation, weight decay, dropout, early stopping, more data |
| Val worse than expected, noisy | forgot `model.eval()`; augmentation applied to val |
| Training worse after eval | forgot `model.train()` |
| `expected ... same device` | move inputs (and new tensors) to `device` |
| `mat1 and mat2 must have the same dtype` | float64 from NumPy: `.float()` |
| `expected scalar type Long` | class labels must be int64 |
| `view size is not compatible` | use `reshape` or `.contiguous()` |
| `Trying to backward through the graph a second time` | a fresh forward per step; or `detach` carried state (RNN hidden) |
| `modified by an inplace operation` | replace `x += ...` / `relu_` with out-of-place versions |
| Gradient is `None` | not a leaf (`retain_grad`), not used in the loss, or `requires_grad=False` |
| Shapes wrong at `Linear` after convs | compute $C \cdot H \cdot W$, or use `AdaptiveAvgPool2d(1)` |

Useful checks: `print(x.shape)` everywhere; `torch.autograd.set_detect_anomaly(True)` (slow) to find the op that made a `NaN`; log `clip_grad_norm_` return values; `torch.isfinite(t).all()`; compare against a tiny hand-computed case.
