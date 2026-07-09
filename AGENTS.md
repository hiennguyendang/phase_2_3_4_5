# Repository Guidelines

## Project Structure & Module Organization

This repository is organized as a phase-based VERA research pipeline. `phase_1/` extracts frozen BioViL-T features, `phase_2/` builds the 29-region detector and scene-graph parser, `phase_3/` trains/evaluates per-region concept and disease heads, `phase_4/` models temporal progression, and `phase_5/` assembles faithful reports. Each phase generally uses `src/` for importable libraries, `scripts/` for numbered run-order entry points, and `notebooks/` for Kaggle or audit workflows. Shared or generated inputs live under `data/`; experiment checkpoints and logs live under `RUN/`, `artifacts/`, and `LOGS/`.

## Build, Test, and Development Commands

Use a Python virtual environment and install phase-specific dependencies as needed:

```bash
python -m venv .venv
pip install -r phase_2/requirements.txt
```

Common entry points:

```bash
python phase_5/demo.py
bash phase_3/run_experiments.sh
bash phase_4/run_experiments.sh
python phase_3/scripts/5-eval.py --ckpt <run>/best.pt --split test
python phase_3/scripts/6-faithfulness.py --ckpt <run>/best.pt --split val
```

Prefer running scripts from the repository root unless a phase README says otherwise.

## Coding Style & Naming Conventions

Python code uses flat phase-local imports: scripts add the corresponding `src/` directory to `sys.path`, then import clean module names such as `config`, `dataset`, or `eval`. Keep importable library files unnumbered. Use numeric prefixes only for executable pipeline steps, for example `scripts/4-train.py`. Follow the existing style: 4-space indentation, `snake_case` functions and variables, uppercase constants, and concise type/shape comments where tensor contracts are non-obvious.

## Testing Guidelines

There is no central pytest suite in the current tree. Validate changes with the smallest relevant phase command: smoke-test deterministic code with `python phase_5/demo.py`, evaluate trained checkpoints with phase-specific `*-eval.py`, and run faithfulness checks for M3 changes. For data-prep edits, run the numbered script on a small subset or `--limit` option when available, then inspect generated manifests, JSONL, or overlays.

## Commit & Pull Request Guidelines

Recent commits are short, imperative summaries such as `adjust frozen`, `add worker`, or `fix bug phase 4`. Keep commits focused on one phase or behavior. Pull requests should include the affected phase, commands run, key metrics or faithfulness results, and any data/checkpoint assumptions. Include screenshots or notebook output only when visual alignment, detector boxes, or report text changed.

## Security & Configuration Tips

Do not commit private datasets, Drive tokens, Kaggle secrets, or large derived caches unless they are intentionally tracked artifacts. Preserve contracts documented in phase READMEs, especially BioViL-T feature shape `[197, 512]`, M3 label values `1/0/-100`, and detector-vs-GT box source choices.
