# Fairness CFS

This folder contains setup notes for the `fairness_cfs` project.

## Setup

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade packaging tools:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Install project dependencies if a requirements file is available:

```bash
pip install -r requirements.txt
```

If the project uses notebooks, install Jupyter:

```bash
pip install jupyter
```

## Suggested Project Layout

```text
fbk/fairness_cfs/
  README.md
  src/
  process_models/
  plot_scripts/
  jupyter/
  plots/
```

Keep generated outputs and experiment results out of version control. Files and folders with `results` in their name are ignored by the repository-level `.gitignore`.
