# SNA Twitch Influencer Project (MAPR2026 v3)

This repository studies influence approximation on the Twitch Gamers graph, with
the current execution path centered on `src/mapr2026_v3`.

## Current Scope

Main workflow in this repo:

1. Build graph artifacts and IC labels (A0 / optional alternatives).
2. Build diffusion proxies and typology diagnostics.
3. Train and evaluate baselines + graph-aware surrogates.
4. Export paper-facing artifacts in `outputs/mapr2026_v3_results`.

Paper draft lives in `paper main.md`.

## Python and Environment

- Required Python: `3.10` to `3.12` (64-bit).
- Python `3.13` is not supported by pinned dependencies.

Recommended setup (Windows + Conda):

```powershell
conda env create -f environment.yml
conda activate sna_group9_cbow_py312
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Repository Layout

- `src/mapr2026_v3/`: active MAPR2026 v3 scripts and shared contracts.
- `src/data`, `src/graph`, `src/sis`: legacy stage-0..3 pipeline pieces still used for base artifacts.
- `data/processed/`: canonical inputs/outputs consumed across tracks.
- `outputs/day1_benchmark/`: day-1 benchmark, stability, uncertainty, and quality-gate artifacts.
- `outputs/mapr2026_v3_results/`: main experiment outputs used in analysis/paper.
- `tests/`: unit tests for MAPR v3 contracts and determinism.
- `docs/`: execution plans, runbooks, and decision logs.
- `evaluate/test_repos/`: external benchmark corpora (not core project logic).

## Quickstart (Current MAPR v3 Path)

Run from repository root.

### 1) Build base graph artifacts (if missing)

```powershell
python run_all.py --stage 0
python run_all.py --stage 1
python run_all.py --stage 2
python run_all.py --stage 3
```

### 2) MAPR v3 core artifacts

```powershell
python src/mapr2026_v3/export_csr.py --run
python src/mapr2026_v3/day1_benchmark.py --n-jobs -1
python src/mapr2026_v3/ic_labels_primary.py --n-runs 200 --n-sample 5000 --n-jobs -1
python src/mapr2026_v3/ic_label_uncertainty.py
python src/mapr2026_v3/diffusion_proxies.py
python src/mapr2026_v3/typology_ic_views.py
```

### 3) Baselines and surrogates

```powershell
python src/mapr2026_v3/run_baselines.py --targets-path data/processed/regression_targets_a0.parquet --label-regime a0
python src/mapr2026_v3/run_surrogates.py --targets-path data/processed/regression_targets_a0.parquet --label-regime a0
```

### 4) Preflight check

```powershell
python src/mapr2026_v3/preflight_person2.py
```

## Key Artifacts

Core contracts:

- `data/processed/graph_csr.npz`
- `data/processed/ic_scores_primary.parquet`
- `data/processed/regression_targets.parquet`
- `data/processed/classification_labels.parquet`
- `data/processed/split_masks.parquet`

Main result files:

- `outputs/mapr2026_v3_results/baseline_ranking_metrics.csv`
- `outputs/mapr2026_v3_results/surrogate_ranking_metrics.csv`
- `outputs/mapr2026_v3_results/runtime_breakdown.csv`

## Tests

```powershell
pytest -q
```

## Important Notes

- `src/mapr2026_v3` is the active path; some legacy stage-4..6 files under `src/simulation`, `src/ml`, and `src/evaluation` are placeholders.
- Several old PowerShell runners (`scripts/run_stage4_single_seed_ic.ps1`, `scripts/run_stage5_multi_seed_ic.ps1`, `scripts/run_stage6_ml.ps1`, `scripts/run_all.ps1`) are empty and should not be used as primary entrypoints.
- Keep `data/raw` immutable and write generated artifacts to `data/processed` or `outputs`.
