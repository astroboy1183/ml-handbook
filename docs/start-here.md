# How to Use This Handbook

This handbook is long on purpose. Data science, machine learning, and deep learning are deep fields, and the people who are genuinely good at them understand *why* things work, not just which function to call. This page explains how the handbook is organized and how to study it so the material sticks.

## The three tiers

| Tier | Levels | Goal | You're ready to move on when… |
|---|---|---|---|
| 🌱 [**Beginner: Data Science Foundations**](tiers/beginner.md) | 0–2 | Fluency with data and the math underneath | You can take a messy dataset from question to a defensible, well-presented answer |
| 🔧 [**Intermediate: Machine Learning**](tiers/intermediate.md) | 3–5 | Understand, build, and ship classical ML | You can implement the core algorithms from scratch, choose and validate models rigorously, and deploy one |
| 🚀 [**Expert: Deep Learning**](tiers/expert.md) | 6–8 | Build and understand modern deep learning | You can derive backprop, train deep models in PyTorch, and explain how transformers and diffusion models work |

Each tier page ends with a **checkpoint**: a short list of things you should be able to do without notes before moving on. The [full roadmap](roadmap.md) lists every concept in every chapter.

## Where to start

| If you… | Start at |
|---|---|
| Know a little Python but have never used NumPy or pandas | [Level 0](chapters/00-python-for-data/index.md), from the beginning |
| Use pandas comfortably, but your math is rusty | [Level 1: Math foundations](chapters/01-math-foundations/index.md). Try the [Level 0 capstone](exercises/level-0-capstone.md) first as a placement test. |
| Are comfortable with linear algebra, calculus, and statistics | Try the [Level 1](exercises/level-1-capstone.md) and [Level 2](exercises/level-2-capstone.md) capstones. If they're easy, start the [Intermediate tier](tiers/intermediate.md). |
| Already train scikit-learn models at work | Skim Level 3, and do the [Level 3 capstone](exercises/level-3-capstone.md) from scratch. Most people who skip the from-scratch work have gaps there. |
| Want deep learning only | You can, but you'll need the math in Level 1 and the fundamentals in Level 3 (loss, gradient descent, generalization, evaluation). Do those first. |

!!! warning "Don't skip the math"
    It's tempting to jump straight to neural networks. Resist it. Almost every confusing thing in deep learning (why a model won't train, why a loss explodes, why attention is scaled by $\sqrt{d_k}$) becomes obvious once you're fluent in gradients, probability, and matrix shapes. Level 1 is the highest-leverage level in the handbook.

## How each chapter works

Every chapter has the same structure, so you always know where to look:

1. **Why it matters.** A short real story about what goes wrong without this knowledge.
2. **Concepts.** Each idea in three passes: **intuition** (plain language and pictures), **math** (precise, derived, every symbol defined), and **code** (from scratch in NumPy, then the library version).
3. **In practice.** Complete, runnable code with real output, plus the common mistakes.
4. **Exercises.** 4–6 tasks: derivations with pen and paper, coding, and reasoning. Solutions are hidden in collapsible boxes.
5. **Check yourself.** Questions to answer without notes.
6. **Key takeaways, further reading, and next.**

## How to study

**Type the code, don't paste it.** Typing forces you to read every line. Run it, then change something and predict the result before you run it again. Prediction is where the learning happens.

**Do the math with pen and paper.** When a chapter derives a gradient, derive it yourself on paper, then check your answer numerically in NumPy. The handbook shows you how in Level 1. Being able to check math with code is the most valuable habit in this field.

**Do the exercises before opening the solutions.** Struggle for at least 20 minutes. If you're still stuck, open the solution, understand it, close it, and solve the exercise again from a blank page the next day.

**Treat capstones as gates.** Every level ends with a capstone project. Don't move on until you can complete it without notes. If you can't, the gap shows you exactly which chapter to revisit.

**Keep a "mistakes I made" log.** Write down every bug, leak, and wrong conclusion: what you did, what happened, and what you learned. Data leakage, wrong shapes, and misread metrics are where most real-world ML failures come from, and the log turns each one into a lesson you won't repeat.

**A short session most days beats a long one once a week.** About an hour a day works well. Spaced practice is much more effective than cramming.

## Conventions used in this handbook

| You'll see | It means |
|---|---|
| A `python` code block followed by a `text` block | The code, then its real output. The copy button copies only the code. |
| $\hat{y} = \mathbf{w}^\top \mathbf{x} + b$ | Math. The [math notation cheat sheet](cheatsheets/math-notation.md) explains every symbol. |
| `=== "From scratch"` / `=== "scikit-learn"` tabs | Two implementations of the same thing. Read both. |
| `!!! warning "Common mistake"` | A mistake that real practitioners make all the time. |
| "(your numbers will vary slightly)" | Output from a dataset you download, or from randomness that differs by platform. |

All randomness in the handbook is seeded, so where no such note appears you should get exactly the output shown.

## What you need

- A computer with Python 3.12 or newer. Any OS works.
- No GPU. Every example trains on a laptop CPU in minutes. Where a GPU would help in Levels 7–8, the chapter says so and points to free options like Google Colab.
- About an hour a day, and some patience with the math.

Next: [set up your environment](setup.md).
