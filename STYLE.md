# Handbook Style Guide

Every page in `docs/` follows these rules. `docs/roadmap.md` is the syllabus:
each chapter must cover every concept listed for it there.

## Audience

Beginners who want to master data science, machine learning, and deep
learning. Assume **basic Python** (variables, loops, functions) and high-school
math, and nothing else beyond earlier chapters. Readers are smart and
motivated, and many are software or data engineers. Explain the *why* and the
*how it works underneath*, not just the *what*. Go deep: this handbook is for
mastering the field, not a quick tutorial.

**The teaching pattern for every important idea is intuition → math → code:**

1. **Intuition.** A plain-language explanation, an analogy, or a picture.
2. **Math.** The precise formulation, with every symbol defined. Derive results rather than stating them, unless a derivation would take pages; in that case say so and give the key idea.
3. **Code.** A from-scratch NumPy implementation where feasible, then the library version (scikit-learn or PyTorch), with real output.

Reference environment: **Python 3.12+, NumPy 2, pandas 2+, scikit-learn 1.5+,
PyTorch 2 (CPU is fine), matplotlib, and seaborn.** All code must run there
as written. Deep learning examples must train on a laptop CPU in minutes; say
so when a GPU would help.

## Chapter format (mandatory)

```markdown
# <Chapter title>

> **Level N · Chapter M** · ⏱️ ~XX min read · Prerequisites: [link](...) or "None"

<one or two sentence intro of what this chapter covers>

## Why it matters

A concrete, real situation where this knowledge matters: a model that
silently failed, a wrong business decision, or a bug that cost days. Tell it
as a short story.

## Concepts

The ideas built up step by step, following intuition → math → code. Use ###
subsections. Use math (see below), mermaid diagrams, and figures where they
help. Define every new term in **bold** the first time it appears.

## In practice

Hands-on code, grouped by ### subsections: complete, runnable examples with
real output. Show the from-scratch version, then the library version. Cover
common pitfalls in `!!! warning "Common mistake"` boxes.

## Exercises

4–6 tasks from easy to hard. Mix the three kinds: derivations or by-hand
math, coding tasks, and conceptual reasoning. Each has a collapsible solution:

### Exercise 1: <name> (easy)

<task>

??? success "Solution"

    <full solution: code, output, and explanation, indented 4 spaces>

## Check yourself

6–8 questions to answer without notes, each with a collapsible answer:

1. Question?

    ??? note "Answer"

        Answer text.

## Key takeaways

- 4–7 bullets.

## Further reading

2–5 high-quality, real, stable references (classic textbooks, original
papers, official docs). Never invent a URL. If unsure of a URL, cite by title
and author only.

## Next

Link to the next chapter (or the level capstone).
```

Length: aim for **3,500–6,000 words of prose** per chapter (code excluded).
Depth beats breadth, but cover every concept the roadmap lists.

## Math

The site renders LaTeX with KaTeX.

- Inline math: `$\hat{y} = w^\top x + b$`. Display math goes on its own lines, with blank lines around it:

  ```markdown
  $$
  \mathcal{L}(w) = \frac{1}{n} \sum_{i=1}^{n} \left(y_i - w^\top x_i\right)^2
  $$
  ```

- Inside admonitions and solutions, indent the `$$` block like the rest of the content.
- Use KaTeX-supported commands only. Avoid `\begin{align}`; use `\begin{aligned} ... \end{aligned}` inside `$$`.
- **Dollar signs that aren't math** (money) must be written `USD 5` or `\$5`, or they will break rendering.
- Notation, used consistently: scalars $x$, vectors $\mathbf{x}$, matrices $\mathbf{X}$ (or plain $X$ if already clear), $n$ samples, $d$ features, $\theta$ or $\mathbf{w}, b$ for parameters, $\hat{y}$ for predictions, $\mathcal{L}$ for loss, $\eta$ for learning rate, $\sigma$ for sigmoid. Define every symbol on first use in each chapter.

## Code

- Every code block has a language: `python`, `bash`, `sql`, `text` (for output), and so on.
- Show code, then its output in a separate `text` block, so the copy button copies only code. Output must come from **actually running the code** in the verification environment. Trim with `...` if long.
- Seed all randomness (`np.random.default_rng(0)`, `torch.manual_seed(0)`, `random_state=0`) so outputs are reproducible.
- Prefer built-in or synthetic datasets that need no download: `sklearn.datasets.load_*` (iris, wine, breast_cancer, digits, diabetes), `make_classification`, `make_regression`, `make_moons`, `make_blobs`, or data generated with NumPy. Where a real downloadable dataset is better for readers (Titanic, California housing, MNIST/FashionMNIST, CIFAR-10, Tiny Shakespeare), you may use it in exercises and capstones with the download code. You won't have network access while verifying, though, so outputs shown for downloaded data are illustrative and must be labeled "(your numbers will vary slightly)".
- Keep examples self-contained: imports at the top of the first block in a section.

## Figures and verification

- Mermaid diagrams (`flowchart`, `sequenceDiagram`, `graph`) for architectures, pipelines, and workflows.
- `scripts/run_chapter.py <chapter.md> --fix` runs every `python` block in order (like a notebook), replaces each following `text` block with the real output, and saves each `plt.show()` as `docs/assets/figures/<level-dir>/<chapter-stem>-fig<N>.png`.
- Reference figures after the plotting code: `![Alt text](../../assets/figures/<level-dir>/<chapter-stem>-fig1.png)`, followed by a one-line italic caption.
- Put `<!-- skip-run -->` on the line before a block that must not run (downloads, GPU-only code).

## MkDocs Material features

- Admonitions: `!!! note`, `!!! tip`, `!!! warning`, `!!! danger`, `!!! info`, `!!! example`. Collapsible: `??? note "Title"`. Content is indented 4 spaces.
- Content tabs (`=== "From scratch"` / `=== "scikit-learn"`) for alternative implementations.
- Tables for comparisons. Inside tables, don't escape pipes in code spans: `a|b` works as is, while `\|` shows a literal backslash.
- Internal links are **relative** to the current file and point only to pages in `mkdocs.yml` nav.

## Writing style

- Plain, direct, and friendly. Second person ("you").
- Short paragraphs, one idea per sentence.
- No filler, hype, or "In this chapter we will..." padding beyond the intro line.
- Bold a new term the first time it appears and define it right there.
- Use realistic scenarios: churn, fraud, pricing, medical tests, recommendations, text, and images.
- Call out common mistakes, especially data leakage, train/test contamination, and misread metrics.
- In stories, use neutral names (Alex, Sam, Priya, Wei) and they/them pronouns.

## Accuracy

- Accuracy is the top priority. Run every code example, and make math and code agree with each other.
- Don't overstate. Mark open questions and rules of thumb as such.
- Don't invent citations, benchmark numbers, or URLs.
