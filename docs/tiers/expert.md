# 🚀 Expert Tier: Deep Learning

**Levels 6–8 · Goal: build and understand modern deep learning** · ~14–20 weeks at about an hour a day

You know classical ML. Now you'll go deep. First you'll build neural networks, and the automatic differentiation that trains them, from nothing but NumPy, so no part of deep learning is magic. Then you'll train real models in PyTorch, and finally study the architectures behind modern AI: transformers, large language models, diffusion models, and reinforcement learning.

## What you'll be able to do

- Derive backpropagation in matrix form and implement a neural network and an autograd engine from scratch.
- Explain and implement SGD, momentum, Adam, initialization schemes, normalization, and dropout, and know when each matters.
- Write clean PyTorch training loops, and train and debug CNNs, RNNs, and transformers.
- Fine-tune pretrained models, and know when transfer learning is the right call.
- Implement attention and a GPT-style language model from scratch, and explain LoRA, RLHF, and RAG.
- Explain how VAEs, GANs, diffusion models, CLIP, and policy-gradient RL work, and reason about training at scale.

## The levels

<div class="grid cards" markdown>

-   :material-numeric-6-circle:{ .lg .middle } **Level 6: Neural networks from scratch**

    ---

    Neurons to networks, backpropagation, a NumPy network, optimizers, training deep networks, and autograd from scratch.

    [:octicons-arrow-right-24: Start Level 6](../chapters/06-neural-networks/index.md)

-   :material-numeric-7-circle:{ .lg .middle } **Level 7: Deep learning with PyTorch**

    ---

    PyTorch fundamentals, the training loop, CNNs, sequence models, embeddings, and transfer learning.

    [:octicons-arrow-right-24: Start Level 7](../chapters/07-deep-learning-pytorch/index.md)

-   :material-numeric-8-circle:{ .lg .middle } **Level 8: Modern deep learning**

    ---

    Transformers, language models, LLMs in practice, generative models, self-supervised learning, scaling, and RL.

    [:octicons-arrow-right-24: Start Level 8](../chapters/08-modern-deep-learning/index.md)

</div>

## Tier checkpoint

You've mastered this handbook when you can do all of these without notes:

- [ ] Derive the gradient of softmax plus cross-entropy, and explain why it's so simple.
- [ ] Implement a working autograd engine and train a small network with it.
- [ ] Diagnose a network that won't train, using loss curves, gradient norms, and the overfit-one-batch test.
- [ ] Write scaled dot-product attention from memory, and explain every term, including the $\sqrt{d_k}$.
- [ ] Count the parameters of a transformer from its configuration.
- [ ] Explain the diffusion training objective, and how sampling reverses the noise.
- [ ] Explain how RLHF uses policy gradients to fine-tune a language model.

See the [full roadmap](../roadmap.md) for every concept in this tier. For what to study next, see "Where to go after this handbook" in the [Level 8 overview](../chapters/08-modern-deep-learning/index.md).
