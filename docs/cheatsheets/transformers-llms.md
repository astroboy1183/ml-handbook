# Transformers and LLMs Cheat Sheet

A dense reference for attention, the transformer block, parameter, FLOP, and memory counting, decoding, fine-tuning methods, retrieval-augmented generation, and evaluation. Derivations and runnable implementations are in the chapters.

Related chapters: [Attention and transformers](../chapters/08-modern-deep-learning/01-attention-and-transformers.md) · [Language models](../chapters/08-modern-deep-learning/02-language-models.md) · [LLMs in practice](../chapters/08-modern-deep-learning/03-llms-in-practice.md) · [Scaling and efficiency](../chapters/08-modern-deep-learning/06-scaling-and-efficiency.md) · [Reinforcement learning](../chapters/08-modern-deep-learning/07-reinforcement-learning.md)

Notation: $T$ sequence length, $d$ model width, $h$ heads, $d_k = d/h$ head width, $L$ layers, $V$ vocabulary size, $N$ parameters, $D$ training tokens, $B$ batch size.

## Attention

| Item | Formula or rule |
|---|---|
| Scaled dot-product attention | $\operatorname{softmax}\!\left(QK^\top/\sqrt{d_k} + M\right)V$, softmax over each row |
| Why $\sqrt{d_k}$ | For unit-variance components, $\operatorname{Var}(\mathbf{q}^\top\mathbf{k}) = d_k$; dividing restores variance 1 and keeps the softmax out of saturation |
| Mask $M$ | $0$ where allowed, $-\infty$ where blocked; causal mask blocks $j > i$ |
| Self- vs cross-attention | Self: $Q, K, V$ from the same sequence. Cross: $Q$ from the decoder, $K, V$ from the encoder |
| Multi-head | $\operatorname{Concat}(\text{head}_1, \ldots, \text{head}_h)W_O$, each head attends in a $d/h$ subspace |
| MHA parameters | $4d^2 + 4d$ with biases, independent of $h$ |
| Cost | $O(T^2 d)$ time, $O(T^2)$ memory for scores (FlashAttention avoids storing them) |
| Grouped-query attention | Several query heads share one key-value head: smaller KV cache |

```python
import math
import torch
import torch.nn.functional as F

def attention(Q, K, V, mask=None):                  # mask: True = may attend
    scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return scores.softmax(-1) @ V

Q = K = V = torch.randn(2, 4, 5, 8)                 # (batch, heads, T, d_k)
causal = torch.tril(torch.ones(5, 5, dtype=torch.bool))
assert torch.allclose(attention(Q, K, V, causal), F.scaled_dot_product_attention(Q, K, V, is_causal=True), atol=1e-6)
```

!!! warning "Opposite mask conventions"
    `F.scaled_dot_product_attention`: boolean `True` = **may attend**. `nn.MultiheadAttention` and `nn.Transformer`: boolean `True` = **blocked**.

## Positional encodings

| Method | How | Notes |
|---|---|---|
| Sinusoidal | Add $\sin(p/10000^{2i/d})$, $\cos(\cdot)$ to embeddings | No parameters; $\mathrm{PE}(p + k)$ is a rotation of $\mathrm{PE}(p)$ |
| Learned absolute | Add row $p$ of `nn.Embedding(max_len, d)` | GPT-2, BERT; no positions beyond `max_len` |
| RoPE | Rotate each $(q_{2i}, q_{2i+1})$ pair, and the same for keys, by angle $p\,\theta_i$ | Score depends on content and offset $m - n$ only; Llama-family default |
| ALiBi | Add $-\text{slope} \cdot \lvert m - n\rvert$ to scores | Relative, no embeddings |
| None | Nothing | Encoders fail (permutation-equivariant); causal decoders still recover some position information |

## Block anatomy (pre-LN)

```text
x ─┬─ LayerNorm ─ MultiHeadAttention ─(+)─┬─ LayerNorm ─ FFN (d → 4d → d, GELU) ─(+)─→
   └──────────────────────────────────────┘└───────────────────────────────────────┘
```

- $x \leftarrow x + \operatorname{MHA}(\operatorname{LN}(x))$, then $x \leftarrow x + \operatorname{FFN}(\operatorname{LN}(x))$; a final LayerNorm before the output head.
- Attention moves information between positions; the FFN processes each position.
- Pre-LN trains stably without long warmup; post-LN, $\operatorname{LN}(x + f(x))$, was the original.
- Common variants: RMSNorm instead of LayerNorm, SwiGLU FFN (three matrices, hidden size about $\frac{8}{3}d$), no biases.

| Family | Mask | Typical use | Examples |
|---|---|---|---|
| Encoder-only | None (bidirectional) | Classification, embeddings, retrieval | BERT, ViT |
| Decoder-only | Causal | Generation, general LLMs | GPT, Llama |
| Encoder-decoder | Encoder none; decoder causal + cross-attention | Translation, speech-to-text | T5, Whisper |

## Counting parameters and FLOPs

| Quantity | Rule |
|---|---|
| One block (4× FFN, biases, 2 LayerNorms) | $12d^2 + 13d \approx 12d^2$ |
| GPT, tied head, learned positions | $N = Vd + T_{\max}d + L(12d^2 + 13d) + 2d$ |
| Large-model shortcut | $N \approx 12Ld^2$ (plus $Vd$ for embeddings) |
| GPT-2 small check | $V = 50{,}257$, $T_{\max} = 1024$, $d = 768$, $L = 12$ → 124.4M |
| Forward FLOPs per token | $\approx 2N$ (plus attention $\approx 4LTd$) |
| Training FLOPs | $C \approx 6ND$ (forward $2N$ + backward $4N$ per token) |
| Compute-optimal (Chinchilla) | $D \approx 20N$; so $N \approx \sqrt{C/120}$ |
| Matrix multiply $m \times k$ by $k \times n$ | $2mkn$ FLOPs |

```python
def gpt_params(V, T_max, d, L):
    return V * d + T_max * d + L * (12 * d**2 + 13 * d) + 2 * d

assert gpt_params(50257, 1024, 768, 12) == 124_439_808
```

## Memory rules of thumb

| Setting | Bytes per parameter | 7B model |
|---|---|---|
| Inference, fp32 | 4 | 28 GB |
| Inference, bf16/fp16 | 2 | 14 GB |
| Inference, int8 | 1 | 7 GB |
| Inference, 4-bit | 0.5 | 3.5 GB |
| Training, mixed-precision Adam (bf16 weights + grads, fp32 master + $m$ + $v$) | 16 | 112 GB + activations |
| LoRA, bf16 frozen base | ~2 | ~14 GB + adapters + activations |
| QLoRA, 4-bit frozen base | ~0.5 | ~4 GB + activations |
| FSDP / ZeRO-3 over $n$ GPUs | $16/n$ | 14 GB each on 8 GPUs |

- **KV cache**: $2 \times L \times T \times d_{kv} \times$ bytes, per sequence ($d_{kv} = d$ without GQA). $L = 32$, $d = 4096$, $T = 4096$, bf16 → about 2.1 GB per sequence.
- **Activations** (GPT layer, 16-bit, no recomputation, Korthikanti et al.): $\approx sbh(34 + 5as/h)$ bytes per layer. Checkpointing each layer keeps about $2sbh$.
- Add 10% to 20% for buffers and fragmentation.

| Technique | Saves | Costs |
|---|---|---|
| Mixed precision (bf16) | Memory, time (tensor cores) | Keep fp32 master weights and reductions |
| fp16 | Same | Needs loss scaling (`GradScaler`) |
| Gradient accumulation | Memory (smaller micro-batches) | Time; divide each loss by $k$ |
| Activation checkpointing | Activation memory | ~33% more compute |
| Data parallel (DDP) | Time | Full model per GPU; all-reduce gradients |
| Tensor parallel | Memory and time per layer | Communication in every layer (within a node) |
| Pipeline parallel | Memory | Bubble $\approx (p - 1)/(m + p - 1)$ |
| FSDP / ZeRO | Model-state memory, $\propto 1/n$ | About 1.5× DDP's communication |

## Decoding

| Method | Rule | Use |
|---|---|---|
| Greedy | $\arg\max$ each step | Deterministic; loops on open-ended text |
| Beam search | Keep the $b$ best partial sequences | Translation, short exact outputs |
| Temperature $\tau$ | $\operatorname{softmax}(z/\tau)$; $\tau < 1$ sharper, $\tau > 1$ flatter | Creativity knob |
| Top-k | Keep the $k$ most likely tokens | Cuts the long tail; fixed size |
| Top-p (nucleus) | Keep the smallest set with total probability $\ge p$ | Adapts to confidence; common default ($p \approx 0.9$) |
| Min-p | Keep tokens with $p \ge p_{\min} \cdot p_{\max}$ | Stable at high temperature |
| Speculative decoding | Draft $k$ tokens with a small model; accept with $\min(1, p/q)$, else resample from $\max(0, p - q)$ | Faster, exactly the target distribution |

**Perplexity** $= \exp(\text{mean NLL per token})$, the effective number of choices. Compare across tokenizers with **bits per character** $= \text{NLL}_{\text{token}} / (\text{chars per token} \cdot \ln 2)$. A uniform model over $V$ tokens has perplexity $V$; check that step-0 loss $\approx \ln V$.

## Fine-tuning methods

| Method | Trains | Data | Key formula or idea |
|---|---|---|---|
| Prompting / in-context learning | Nothing | A few examples in the prompt | Format and examples steer behavior |
| Full fine-tuning / SFT | All weights | (instruction, response) pairs | Cross-entropy on response tokens only (labels `-100` on the prompt) |
| LoRA | Low-rank $A, B$ per adapted matrix | Same as SFT | $h = W_0x + \frac{\alpha}{r}BAx$; $B = 0$ at start; $r(d_{\text{in}} + d_{\text{out}})$ parameters; mergeable |
| QLoRA | LoRA adapters | Same | 4-bit NF4 frozen base, 16-bit adapters |
| Reward model (RLHF step 1) | Scalar head | Preference pairs $(x, y_w, y_l)$ | $-\log\sigma(r(x, y_w) - r(x, y_l))$ (Bradley-Terry) |
| RLHF with PPO | Policy (+ critic) | Prompts + reward model | $\max\mathbb{E}[r] - \beta\operatorname{KL}(\pi\,\Vert\,\pi_{\text{ref}})$, clipped ratio objective |
| DPO | Policy | Preference pairs | $-\log\sigma\!\left(\beta\log\frac{\pi(y_w)}{\pi_{\text{ref}}(y_w)} - \beta\log\frac{\pi(y_l)}{\pi_{\text{ref}}(y_l)}\right)$ |
| RL with verifiable rewards (e.g. GRPO) | Policy | Prompts + a checker | Group-relative advantage $(r_i - \bar{r})/\operatorname{std}(r)$, no critic |

```python
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base, self.scale = base, alpha / r
        base.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features) / base.in_features ** 0.5)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale

def dpo_loss(logp_w, logp_l, ref_logp_w, ref_logp_l, beta=0.1):   # summed token log-probs per response
    return -F.logsigmoid(beta * ((logp_w - ref_logp_w) - (logp_l - ref_logp_l))).mean()
```

Choosing: try prompting first; fine-tune for format, style, and task behavior; use LoRA or QLoRA when memory is tight or you need many task variants; use RAG, not fine-tuning, for facts that change or must be cited; use DPO or RLHF to align with preferences.

## RAG components

| Component | Choices and rules of thumb |
|---|---|
| Chunking | A few hundred tokens with 10% to 20% overlap; split on structure (headings, paragraphs) when possible |
| Embeddings | Sparse (TF-IDF, BM25): exact terms, names, codes. Dense (contrastive encoders): meaning, synonyms. Hybrid: both |
| Similarity | Cosine; normalize vectors so it's a dot product |
| Index | Exact search for up to ~$10^5$ chunks; approximate (HNSW, IVF) beyond |
| Reranking | Cross-encoder rescoring of the top 20 to 100 |
| Prompt | Instructions (use only the context, cite ids, say when unknown), chunks with ids, the question, within the context budget |
| Security | Retrieved text is untrusted: prompt injection; enforce access control at retrieval time |
| Evaluation | Retrieval: hit rate / recall@k on labeled questions. Generation: correctness and faithfulness to the context |

## Evaluation

| Approach | Measures | Watch out for |
|---|---|---|
| Perplexity / bits per character | Fit to a text distribution | Tokenizer-dependent; weak proxy for usefulness |
| Multiple-choice benchmarks (e.g. MMLU) | Knowledge, reasoning | Contamination, prompt format sensitivity |
| Code with tests: pass@k | Functional correctness | Report $k$; unbiased estimator $1 - \binom{n-c}{k}/\binom{n}{k}$ |
| Human pairwise preference | Overall quality | Cost, rater bias toward length |
| LLM-as-a-judge | Scalable quality ratings | Position, verbosity, and self-preference bias; validate against humans |
| Task-specific eval set | What you actually need | Build it early: a few hundred real inputs with references or rubrics |
| Retrieval metrics | Whether RAG finds the evidence | Measure separately from generation |
| Safety and red-teaming | Harmful outputs, jailbreaks, injection | Layer defenses; least-privilege tools |

**Hallucination**: plausible but unsupported output. Mitigate with grounding and citations, allowing abstention, automatic verification, and human review for high stakes. Temperature 0 makes outputs repeatable, not true.

## Common pitfalls

- **Missing $\sqrt{d_k}$** or a wrong mask convention: training stalls, or the loss is suspiciously low because the model sees the future. Test causality by perturbing a future token and checking that earlier outputs don't change.
- **Initial loss far from $\ln V$**: bad initialization or a label-shift bug. Fix it before tuning anything else.
- **Comparing perplexities across tokenizers**: convert to bits per character or byte first.
- **Forgetting the position offset with a KV cache**: every generated token gets position 0. Compare cached and uncached greedy outputs.
- **Sampling with greedy decoding on open-ended text**: repetition loops. Use temperature with top-p or top-k.
- **SFT without loss masking or without an end-of-turn token**: the model learns to write user turns, or never stops.
- **LoRA with an unfrozen base**: print the trainable parameter count before training.
- **16-bit weights without fp32 master copies**: small updates round to nothing and learning silently stops.
- **Gradient accumulation without dividing the loss by $k$**: an effective learning rate $k$ times too large.
- **Quantizing without evaluating**: check task accuracy and per-layer error; keep embeddings, the output head, and norms in higher precision.
- **Evaluating RAG only end to end**: measure retrieval separately; most failures start there.
- **Optimizing a reward model without a KL penalty**: reward hacking and collapsed diversity.
