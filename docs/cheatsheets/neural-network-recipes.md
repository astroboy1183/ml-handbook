# Neural Network Recipes Cheat Sheet

Defaults and rules of thumb for building and training neural networks: which architecture, loss, activation, initialization, optimizer, and regularizer to start with, the shapes and formulas you'll need, and a checklist for when training goes wrong. These are starting points, not laws: tune on validation data.

Related chapters: [From neurons to networks](../chapters/06-neural-networks/01-from-neurons-to-networks.md) · [Forward and backpropagation](../chapters/06-neural-networks/02-forward-and-backpropagation.md) · [A neural network in NumPy](../chapters/06-neural-networks/03-neural-network-in-numpy.md) · [Optimizers](../chapters/06-neural-networks/04-optimizers.md) · [Training deep networks](../chapters/06-neural-networks/05-training-deep-networks.md) · [Autograd from scratch](../chapters/06-neural-networks/06-autograd-from-scratch.md)

Notation: $n$ batch size, $d_{\text{in}}, d_{\text{out}}$ a layer's input and output widths, $K$ classes, $\eta$ learning rate, $\lambda$ weight decay, $\mathbf{Z}$ logits or pre-activations, $\mathbf{P}$ predicted probabilities, $\mathbf{Y}$ one-hot labels.

## Architectures per task

| Data | Start with | Then try | Notes |
|---|---|---|---|
| Tabular (rows and columns) | gradient-boosted trees | MLP: 2–4 layers of 64–512 units, ReLU, dropout 0.1–0.3 | Trees usually win on small and medium tabular data; standardize inputs and embed categoricals for an MLP |
| Images | pretrained CNN (ResNet) or vision transformer, fine-tuned | train a ResNet from scratch if data is large | Use data augmentation; [Transfer learning](../chapters/07-deep-learning-pytorch/06-transfer-learning.md) |
| Text | pretrained transformer (fine-tune or embed) | small transformer or LSTM from scratch | Tokenization matters as much as architecture |
| Sequences and time series | gradient-boosted trees on lag features, or a small transformer | LSTM/GRU, temporal CNN | Respect time order in validation |
| Recommendations, IDs | embeddings + MLP | two-tower models | Embedding size around 16–256 |
| Small images, toy problems (like `load_digits`) | MLP, 1–2 hidden layers | small CNN | Always compare with a linear baseline |

**Depth and width.** Start shallow (1–3 hidden layers) and widen before you deepen. Go deeper only with residual connections and normalization.

## Output layers and losses per task

Always output **logits** from the model and use a loss that takes logits.

| Task | Output units | Output activation (for predictions) | Loss | PyTorch |
|---|---|---|---|---|
| Regression | 1 per target | identity | MSE (or MAE, Huber for outliers) | `nn.MSELoss`, `nn.L1Loss`, `nn.HuberLoss` |
| Binary classification | 1 | sigmoid | binary cross-entropy | `nn.BCEWithLogitsLoss` |
| Multiclass (one label) | $K$ | softmax | categorical cross-entropy | `nn.CrossEntropyLoss` (integer labels) |
| Multi-label | $K$ | sigmoid per unit | sum of $K$ binary cross-entropies | `nn.BCEWithLogitsLoss` |
| Counts | 1 | exp | Poisson negative log-likelihood | `nn.PoissonNLLLoss(log_input=True)` |
| Imbalanced classes | as above | as above | weighted cross-entropy, or focal loss | `weight=` / `pos_weight=` arguments |

With these canonical pairings, the gradient at the logits is $\frac{1}{n}(\hat{\mathbf{Y}} - \mathbf{Y})$. Optional for classification: **label smoothing** of 0.1 (`label_smoothing=0.1`) often improves calibration and accuracy slightly.

## Activations

| Activation | $\phi(z)$ | $\phi'(z)$ | Use |
|---|---|---|---|
| ReLU | $\max(0, z)$ | 1 if $z > 0$, else 0 | default for MLPs and CNNs |
| Leaky ReLU | $\max(\alpha z, z)$, $\alpha \approx 0.01$ | 1 or $\alpha$ | when many units die |
| GELU | $z\,\Phi(z)$ | $\Phi(z) + z\,\varphi(z)$ | transformers |
| SiLU / Swish | $z\,\sigma(z)$ | $\sigma(z)(1 + z(1 - \sigma(z)))$ | modern CNNs and LLMs |
| Tanh | $\tanh z$ | $1 - \tanh^2 z$ | RNN states; small networks |
| Sigmoid | $1/(1 + e^{-z})$ | $\sigma(z)(1 - \sigma(z)) \leq 0.25$ | outputs and gates only, not hidden layers |

Without a nonlinearity between them, stacked linear layers collapse to one linear layer.

## Initialization

| Activation | Scheme | Weight variance | PyTorch |
|---|---|---|---|
| ReLU family | He / Kaiming | $2/d_{\text{in}}$ | `nn.init.kaiming_normal_(W, nonlinearity="relu")` |
| Tanh, sigmoid, linear | Xavier / Glorot | $2/(d_{\text{in}} + d_{\text{out}})$; uniform bound $\sqrt{6/(d_{\text{in}} + d_{\text{out}})}$ | `nn.init.xavier_uniform_(W)` |
| SELU | LeCun | $1/d_{\text{in}}$ | `nn.init.kaiming_normal_(W, nonlinearity="linear")` |

- Biases start at 0. Never initialize all weights to the same value (symmetry never breaks).
- Residual branches: initialize the last layer (or the last normalization's $\gamma$) of each branch to 0, so each block starts as the identity.
- PyTorch's `nn.Linear` default (uniform, variance $\frac{1}{3d_{\text{in}}}$) is fine for shallow nets; set He explicitly for deep ReLU nets.
- Note: PyTorch stores `nn.Linear.weight` as $d_{\text{out}} \times d_{\text{in}}$ and computes $\mathbf{x}\mathbf{W}^\top + \mathbf{b}$.

## Optimizers and their defaults

| Optimizer | Update (per parameter) | Typical learning rate | Other defaults |
|---|---|---|---|
| SGD | $\theta \mathrel{-}= \eta\,\mathbf{g}$ | 0.01–0.1 | needs a schedule |
| SGD + momentum | $\mathbf{v} = \beta\mathbf{v} + \mathbf{g}$; $\theta \mathrel{-}= \eta\mathbf{v}$ | 0.01–0.1 (vision: 0.1 at batch 256) | $\beta = 0.9$; weight decay $10^{-4}$–$5 \times 10^{-4}$ |
| Nesterov | $\theta \mathrel{-}= \eta(\mathbf{g} + \beta\mathbf{v})$ | as momentum | `nesterov=True` |
| AdaGrad | $\mathbf{G} \mathrel{+}= \mathbf{g}^2$; $\theta \mathrel{-}= \eta\,\mathbf{g}/(\sqrt{\mathbf{G}} + \epsilon)$ | 0.01–0.1 | step size decays forever; good for sparse features |
| RMSProp | $\mathbf{s} = \rho\mathbf{s} + (1 - \rho)\mathbf{g}^2$; $\theta \mathrel{-}= \eta\,\mathbf{g}/(\sqrt{\mathbf{s}} + \epsilon)$ | $10^{-3}$ | $\rho = 0.9$ (PyTorch `alpha=0.99`) |
| Adam | moments $\mathbf{m}, \mathbf{v}$ with bias correction; $\theta \mathrel{-}= \eta\,\hat{\mathbf{m}}/(\sqrt{\hat{\mathbf{v}}} + \epsilon)$ | $10^{-3}$ (range $10^{-4}$–$3 \times 10^{-3}$) | $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$ |
| AdamW | Adam, plus $\theta \mathrel{-}= \eta\lambda\theta$ separately | $10^{-4}$–$10^{-3}$ (transformers: $3 \times 10^{-4}$ or less) | $\lambda = 0.01$–$0.1$; $\beta_2 = 0.95$–$0.999$ |

- Momentum multiplies the effective step along consistent directions by $\frac{1}{1 - \beta}$ (10× for 0.9).
- Adam's $\eta$ is roughly a per-parameter step size; SGD's $\eta$ multiplies the gradient. **Never copy a learning rate between optimizers.**
- Use AdamW, not Adam with `weight_decay` (that's coupled L2). Exclude biases and normalization parameters from weight decay.
- Tune the learning rate first, on a log grid (factors of 3), for each optimizer separately.

### Learning-rate schedules

| Schedule | Formula | When |
|---|---|---|
| Constant | $\eta_t = \eta$ | quick experiments only |
| Step decay | $\eta_t = \eta\,\gamma^{\lfloor t/s \rfloor}$, $\gamma = 0.1$ at 50% and 75% of training | classic vision recipes |
| Cosine | $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\pi t/T))$ | strong default |
| Linear warmup | $\eta_t = \eta_{\max}\,t/T_w$ for $t \leq T_w$ | first 1–10% of steps, for Adam, transformers, large batches |
| Warmup + cosine or linear decay | warmup, then decay to ~0 | transformers, fine-tuning |

Scale the learning rate roughly with the batch size when you change the batch size a lot, and add warmup when you do.

## Normalization and regularization

| Tool | What it does | Defaults | Watch out for |
|---|---|---|---|
| Batch norm | standardizes each feature over the batch, then $\gamma\hat{x} + \beta$ | momentum 0.1, $\epsilon = 10^{-5}$; after linear/conv, before activation | train vs eval mode; batches under ~16; drop the preceding layer's bias |
| Layer norm | standardizes each example over its features | $\epsilon = 10^{-5}$ | none of batch norm's batch issues; the transformer default (pre-norm) |
| RMSNorm | divides by the root mean square, no centering | | cheaper layer norm, common in LLMs |
| Dropout (inverted) | zeroes units with probability $p$, scales survivors by $\frac{1}{1-p}$; identity at inference | $p = 0.1$–$0.5$ in MLPs; 0.1 in transformers | turn off for gradient checks; interacts poorly with batch norm |
| Weight decay | shrinks weights by $(1 - \eta\lambda)$ per step | SGD $10^{-4}$–$5 \times 10^{-4}$; AdamW 0.01–0.1 | not on biases, $\gamma$, $\beta$ |
| Gradient clipping (global norm) | $\mathbf{g} \leftarrow \mathbf{g}\min(1, c/\lVert\mathbf{g}\rVert)$ | $c = 1.0$ | if most steps clip, lower the learning rate instead |
| Residual connections | $\mathbf{x} + F(\mathbf{x})$ | zero-init branch output, or normalize in the branch | standard init in every branch explodes with depth |
| Early stopping | keep the best-validation checkpoint | patience 5–20 epochs | pick on validation, never on test |
| Data augmentation | random transformations of inputs | flips, crops, color jitter for images | must preserve the label |

## Shape cheats

Rows are examples. For a linear layer $\mathbf{Z} = \mathbf{A}\mathbf{W} + \mathbf{b}$:

| Quantity | Shape | Backward |
|---|---|---|
| $\mathbf{A}$ (input) | $n \times d_{\text{in}}$ | $\mathrm{d}\mathbf{A} = \mathrm{d}\mathbf{Z}\,\mathbf{W}^\top$ |
| $\mathbf{W}$ | $d_{\text{in}} \times d_{\text{out}}$ | $\mathrm{d}\mathbf{W} = \mathbf{A}^\top\mathrm{d}\mathbf{Z}$ |
| $\mathbf{b}$ | $d_{\text{out}}$ | $\mathrm{d}\mathbf{b} = \mathbf{1}^\top\mathrm{d}\mathbf{Z}$ (column sums) |
| $\mathbf{Z}$ (output) | $n \times d_{\text{out}}$ | from the layer above |
| Element-wise $\mathbf{A} = \phi(\mathbf{Z})$ | same shape | $\mathrm{d}\mathbf{Z} = \mathrm{d}\mathbf{A} \odot \phi'(\mathbf{Z})$ |
| Softmax + cross-entropy | logits $n \times K$, labels $n$ | $\mathrm{d}\mathbf{Z} = \frac{1}{n}(\mathbf{P} - \mathbf{Y})$ |

- **Every gradient has the shape of the thing it's the gradient of.** If shapes don't match, the formula is wrong.
- **Broadcast operands get summed gradients:** sum over added leading axes, and over stretched size-1 axes with `keepdims=True`.
- **Parameter count** of an MLP: $\sum_l (d_{l-1}d_l + d_l)$.
- **Backward cost** is about 2× the forward pass; a training step is about 3×. Training memory grows with batch size × width × depth (stored activations).
- **PyTorch layouts:** images `(N, C, H, W)`; sequences `(N, L, D)` with `batch_first=True`; `nn.CrossEntropyLoss` takes logits `(N, K)` and integer targets `(N,)`.
- **Conv output size:** $\left\lfloor\frac{W - K + 2P}{S}\right\rfloor + 1$ for input width $W$, kernel $K$, padding $P$, stride $S$.

## Gradient checking

- Central differences, $h \approx 10^{-5}$ to $10^{-6}$, in **float64**.
- Relative error $\frac{\lVert a - b\rVert}{\lVert a\rVert + \lVert b\rVert}$: below $10^{-7}$ correct; $10^{-7}$ to $10^{-4}$ suspicious (or a ReLU kink); above $10^{-4}$ a bug.
- Tiny network, a few examples, dropout off, fixed seeds, every parameter array checked separately. The first layer whose error is large (going from the top down) locates the bug.

## Training debugging checklist

**Before the first real run**

1. Look at a batch of inputs and labels. Are they aligned, scaled, and of the right dtype and shape?
2. Initial loss $\approx \log K$ for $K$ classes (2.30 for 10). Much larger: inputs unscaled or initialization too large.
3. Gradient-check any custom layer or loss.
4. Overfit one small batch (10–50 examples) to near-zero loss with regularization off. If you can't, the bug is in the model, loss, gradients, or update, not in the data size.
5. Train a linear baseline, so you know what the network adds.

**Reading the symptoms**

| Symptom | Likely causes | First fixes |
|---|---|---|
| Loss stuck at $\log K$, accuracy $1/K$ | labels misaligned; zero/constant init; all ReLUs dead; update not applied | check data pipeline; check init; check `optimizer.step()` and `zero_grad()` |
| Loss is `nan` or `inf` | learning rate too high; unstable loss; exploding gradients | lower $\eta$; fused losses; warmup; gradient clipping |
| Loss spikes, then recovers or diverges | learning rate too high; bad batch | lower $\eta$; clipping; warmup |
| Loss falls very slowly | $\eta$ too low; vanishing gradients; saturated units | learning-rate sweep; He init; ReLU; normalization |
| Deep network worse than shallow one, on training data | degradation | residual connections (zero-init branches) + normalization |
| Training accuracy high, validation near chance | labels shuffled relative to inputs; different preprocessing for validation | fix the pipeline |
| Validation much worse than training | overfitting | weight decay, dropout, augmentation, early stopping, more data |
| Results change with batch composition or batch size at evaluation | batch norm in training mode | `model.eval()` |
| Accuracy fine, gradients `None` for some parameters | graph broken (`.detach()`, `.numpy()`, `.item()`, new tensor) | keep computations in tensor ops |
| Gradients grow every step in PyTorch | missing `zero_grad()` | zero gradients before each `backward()` |

**Log every run:** training loss, validation metric, learning rate, global gradient norm (before clipping), and, when something's off, per-layer activation and gradient statistics and the fraction of dead ReLUs. Healthy update-to-weight ratios $\eta\lVert\mathrm{d}\mathbf{W}\rVert/\lVert\mathbf{W}\rVert$ are around $10^{-3}$.

## A default first run

For a new classification problem, a reasonable first configuration:

- Inputs standardized (statistics from the training set only).
- MLP with 2 hidden layers of 128–256 ReLU units, He init (or a pretrained model for images and text).
- Logits output; `CrossEntropyLoss`.
- AdamW, $\eta = 10^{-3}$, $\lambda = 0.01$, batch size 32–128, cosine decay with 5% warmup.
- Gradient clipping at 1.0, logged.
- Early stopping on validation; test set evaluated once.

Then change one thing at a time, and re-sweep the learning rate after any architectural change.
