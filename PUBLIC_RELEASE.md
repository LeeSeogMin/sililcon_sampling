# Public Release Notes (GitHub)

This repository is intended to be published as a **reproducibility code package**.

## What is included

- `code/`: scripts to generate personas, run LLM survey simulation, and analyze outputs
- `data/kgss_benchmarks_2023.json`: KGSS 2023 **aggregate** benchmark distributions (no microdata)
- `config/`: variable definitions used for prompting (question text + valid response ranges)
- `requirements.txt`, `README.md`, `LICENSE`, `CONTRIBUTING.md`

## What is excluded (local-only)

These are intentionally ignored by `.gitignore` and should not be pushed to GitHub:

- `outputs/`, `output/`: generated run artifacts (CSV/JSON/plots)
- `results/`: internal results, plots, draft reports
- `papers/`: paper drafts and manuscript materials
- `.env*`: API keys and local secrets

## Pre-publication checklist

- Ensure `.env` is not present in the repository root (use `.env.example` only).
- Confirm `outputs/`, `results/`, `papers/` contain no required public inputs.
- Run a clean reproduction from scratch using only:
  - `data/kgss_benchmarks_2023.json`
  - `config/`
  - `code/`

