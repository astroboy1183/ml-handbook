# Set Up Your Environment

This page gets you a clean, reproducible Python environment that runs every example in the handbook. It takes about 15 minutes. [Level 0, Chapter 2](chapters/00-python-for-data/02-environment-setup.md) explains *why* each piece exists and how environments work underneath. This page is just the setup steps.

## 1. Install Python and uv

You need Python 3.12 or newer. We use **uv**, a fast tool that installs Python versions, creates virtual environments, and installs packages.

=== "Linux / macOS"

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "Windows (PowerShell)"

    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

Open a new terminal, then check it worked:

```bash
uv --version
```

!!! tip "Prefer conda or plain pip?"
    That's fine. Any tool that gives you an isolated environment with the packages below works. With plain Python, use `python3 -m venv .venv` and `pip install ...` instead of the uv commands.

## 2. Create a project and environment

```bash
mkdir ml-handbook-work && cd ml-handbook-work
uv venv --python 3.12
```

Activate the environment. You need to do this in every new terminal:

=== "Linux / macOS"

    ```bash
    source .venv/bin/activate
    ```

=== "Windows"

    ```powershell
    .venv\Scripts\activate
    ```

## 3. Install the packages

Install packages tier by tier, as you need them.

**Beginner tier (Levels 0–2):**

```bash
uv pip install numpy pandas scipy matplotlib seaborn statsmodels jupyterlab duckdb pyarrow plotly
```

**Intermediate tier (Levels 3–5):**

```bash
uv pip install scikit-learn xgboost lightgbm optuna imbalanced-learn shap mlflow fastapi uvicorn httpx
```

**Expert tier (Levels 6–8):** install PyTorch. The CPU build is all you need for this handbook:

=== "CPU (any OS)"

    ```bash
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    ```

=== "NVIDIA GPU (Linux / Windows)"

    Use the selector at [pytorch.org/get-started](https://pytorch.org/get-started/locally/) to get the command for your CUDA version.

=== "Apple Silicon"

    ```bash
    uv pip install torch torchvision
    ```

    PyTorch uses the Apple GPU through the `mps` device.

Optional, for the Level 7–8 examples that use pretrained models:

```bash
uv pip install transformers datasets
```

## 4. Check everything works

Save this as `check.py` and run it with `python check.py`:

```python
import importlib

for name in ["numpy", "pandas", "scipy", "matplotlib", "seaborn", "sklearn", "torch"]:
    try:
        module = importlib.import_module(name)
        print(f"{name:12s} {module.__version__}")
    except ImportError:
        print(f"{name:12s} not installed (install it when you reach that tier)")
```

```text
numpy        2.5.3
pandas       3.0.6
scipy        1.18.1
matplotlib   3.11.2
seaborn      0.13.2
sklearn      1.9.1
torch        2.14.1+cpu
```

Your version numbers will differ. Any recent version works (Python 3.12+, NumPy 2, PyTorch 2).

## 5. Start JupyterLab

```bash
jupyter lab
```

This opens in your browser. Create a notebook and run `import numpy as np; np.__version__` to confirm the notebook is using your environment. Many people prefer VS Code with the Python and Jupyter extensions instead. Both work.

!!! warning "Common mistake: the wrong kernel"
    If a package is "not found" in a notebook but works in your terminal, the notebook is running a different Python. Start `jupyter lab` from inside the activated environment, or select the `.venv` kernel in VS Code.

## No setup at all: Google Colab

[Google Colab](https://colab.research.google.com/) gives you free notebooks in the browser, with NumPy, pandas, scikit-learn, and PyTorch preinstalled, plus a free (time-limited) GPU. It's great for Levels 7–8 if your laptop is slow. Its downsides are that sessions reset and files disappear unless you save them to Google Drive.

## Organizing your work

A simple layout that scales from Level 0 to Level 8:

```text
ml-handbook-work/
├── .venv/                 # the environment (never commit this)
├── notebooks/             # exploration, one per chapter
├── src/                   # reusable code you write (your own mini-library)
├── data/                  # datasets (never commit large files)
├── capstones/             # one folder per capstone
└── mistakes.md            # your "mistakes I made" log
```

Put it under git from day one (`git init`), and add `.venv/` and `data/` to `.gitignore`.

Next: start [Level 0](chapters/00-python-for-data/index.md), or tick off your progress on the [progress checklist](progress.md).
