# The ML Handbook

A hands-on handbook for learning data science, machine learning, and deep
learning, from your first DataFrame to transformers and diffusion models. It
assumes basic Python and high-school math. It's written for beginners who
want to truly master the field, not just copy notebooks.

📖 **Read it online:** <https://jayanthappalla.com/ml-handbook/>

Every idea is taught three ways:

1. **Intuition.** A plain-language explanation, an analogy, or a picture.
2. **Math.** The precise formulation, with every symbol defined and results derived.
3. **Code.** A from-scratch NumPy implementation, then the scikit-learn or PyTorch version.

## Roadmap

Three tiers, nine levels, and 58 chapters. Each level ends with a capstone
project. The full syllabus, with the concepts taught in every chapter, is in
[`docs/roadmap.md`](docs/roadmap.md).

| Tier | Levels | Goal | Est. time |
|---|---|---|---|
| 🌱 Beginner: Data Science Foundations | 0–2 | Python for data, math foundations, the data science workflow | 10–14 weeks |
| 🔧 Intermediate: Machine Learning | 3–5 | ML fundamentals, algorithms in depth, applied ML and MLOps | 12–16 weeks |
| 🚀 Expert: Deep Learning | 6–8 | Neural networks from scratch, PyTorch, transformers, LLMs, generative models, RL | 14–20 weeks |

### 🌱 Beginner

**Level 0: Python for data.** The data landscape · environments · Python
essentials · NumPy · pandas · visualization.
**Capstone:** analyze a real dataset end to end in a notebook.

**Level 1: Math foundations.** Linear algebra · matrix decompositions (eigen,
SVD) · calculus and gradients · probability · statistics · information theory
and optimization.
**Capstone:** linear regression three ways, from scratch.

**Level 2: The data science workflow.** SQL · data cleaning · EDA · feature
engineering · experimentation and A/B testing · communicating results.
**Capstone:** a full investigation, from SQL to a written report.

### 🔧 Intermediate

**Level 3: ML fundamentals.** What ML is · linear regression · logistic
regression · generalization and bias-variance · regularization · evaluation
metrics · gradient descent in depth.
**Capstone:** a churn classifier from scratch in NumPy, matched with scikit-learn.

**Level 4: ML algorithms in depth.** kNN and naive Bayes · decision trees ·
random forests and gradient boosting · SVMs · clustering · dimensionality
reduction · time series · anomaly detection and recommenders.
**Capstone:** a rigorous model bake-off with a written recommendation.

**Level 5: Applied ML and MLOps.** Pipelines · hyperparameter tuning ·
imbalanced data · interpretability (SHAP) · ML in production (FastAPI, MLflow,
Docker, monitoring) · responsible ML.
**Capstone:** ship a production-ready model service.

### 🚀 Expert

**Level 6: Neural networks from scratch.** Neurons to networks ·
backpropagation · a NumPy network · optimizers · training deep networks ·
autograd from scratch.
**Capstone:** a tiny deep learning library that hits over 95% on digits.

**Level 7: Deep learning with PyTorch.** PyTorch fundamentals · the training
loop · CNNs · sequence models · embeddings and NLP · transfer learning.
**Capstone:** an image classifier above 90% accuracy, with an error analysis.

**Level 8: Modern deep learning.** Attention and transformers · language
models · LLMs in practice (LoRA, RLHF, RAG) · generative models (VAE, GAN,
diffusion) · self-supervised and multimodal learning · scaling and efficiency
· reinforcement learning.
**Capstone:** train a small GPT from scratch.

## Structure

```
ml-handbook/
  README.md            this file
  STYLE.md             the writing and formatting rules every page follows
  mkdocs.yml           website configuration and navigation
  requirements.txt     pinned site-build dependencies
  docs/                everything published on the website
    roadmap.md         the full syllabus
    tiers/             beginner / intermediate / expert overviews
    chapters/          one Markdown file per topic, grouped by level
    exercises/         capstones, with solutions kept separate in solutions/
    cheatsheets/       one-page quick references and a glossary
    assets/figures/    generated figures
  scripts/
    capstones/         reference solutions for each capstone
    figures/           scripts that generate the figures in docs/assets/figures
  notes/               personal notes and a "mistakes I made" log (not published)
```

## Working on the site

The site is built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/),
with math rendered by KaTeX.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt   # pinned: MkDocs 2.0 is incompatible
.venv/bin/mkdocs serve        # live preview at http://127.0.0.1:8000
.venv/bin/mkdocs build --strict
```

Every push to `main` deploys the site to GitHub Pages through
`.github/workflows/deploy.yml`.
