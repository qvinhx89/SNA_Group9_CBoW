# Operationalization Matters: IC-Based Influence Approximation on Twitch Gamers

This repository studies influence approximation on the Twitch Gamers graph, with
the current execution path centered on `src/mapr2026_v3`.

## Paper and Citation

This repository contains the lightweight source code and small verification
artifacts for the accepted MAPR 2026 paper:

**Operationalization Matters: When Graph-Aware Learning Adds Value for IC-Based
Influence Approximation**<br>
Dinh-Duy Tran, Quoc-Vinh Pham, Quoc-Hai Tran, Hung-Vi Tran, and Hung-Nghiep Tran<br>
University of Information Technology, Ho Chi Minh City, Vietnam<br>
Vietnam National University, Ho Chi Minh City, Vietnam

The official paper link will be added here when the proceedings page or DOI is
available. Large reproducibility data are distributed separately via a GitHub
Release or Zenodo archive.

## How to Cite

If you found this work useful, please cite:

Dinh-Duy Tran, Quoc-Vinh Pham, Quoc-Hai Tran, Hung-Vi Tran, and Hung-Nghiep Tran.
"Operationalization Matters: When Graph-Aware Learning Adds Value for IC-Based
Influence Approximation." MAPR, 2026. DOI/link to be added when available.

```bibtex
@inproceedings{tran2026operationalization,
  title     = {Operationalization Matters: When Graph-Aware Learning Adds Value for IC-Based Influence Approximation},
  author    = {Tran, Dinh-Duy and Pham, Quoc-Vinh and Tran, Quoc-Hai and Tran, Hung-Vi and Tran, Hung-Nghiep},
  booktitle = {Proceedings of the International Conference on Multimedia Analysis and Pattern Recognition (MAPR)},
  year      = {2026},
  note      = {Accepted. DOI/link to be added when available}
}
```

Corresponding author: Hung-Nghiep Tran (`nghiepth@uit.edu.vn`).

## Data Availability

This GitHub repository is intentionally kept lightweight. The data protocol
bundle used to reproduce the paper tables is distributed separately as a release
asset or Zenodo archive:

- GitHub Release: to be added
- Zenodo DOI: to be added

After downloading the archive, extract its contents into `data/processed/` so
the default script paths resolve, for example:

```powershell
New-Item -ItemType Directory -Force data/processed
Copy-Item "data_protocol/*" data/processed/ -Recurse -Force
```

The release archive should contain the frozen graph, split, target, prediction,
and diagnostic files described in the paper's data protocol. When publishing a
release, include a checksum or manifest file for the archive.

## Current Scope

Main workflow in this repo:

1. Build graph artifacts and IC labels (A0 / HSCC).
2. Build diffusion proxies and typology diagnostics.
3. Train and evaluate baselines + graph-aware surrogates.
4. Export paper-facing artifacts in `outputs/mapr2026_v3_results`.

Operationalization labels used throughout the code and artifacts:

- `A0`: Operationalization 1, the structural weighted-cascade regime.
- `HSCC`: Operationalization 2, the source-community operationalization.

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
- `data/processed/`: local extraction target for the external data protocol bundle.
- `Artifacts - frozen results/`: small paper-facing metric and bootstrap artifacts.
- `Artifacts - feasibility stability/`: small stability and feasibility diagnostics.
- `outputs/`: generated local outputs; not tracked in the lightweight public repo.
- `tests/`: unit tests for MAPR v3 contracts and determinism.

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
# A0: Operationalization 1, structural weighted-cascade regime.
python src/mapr2026_v3/run_baselines.py --targets-path data/processed/regression_targets_a0.parquet --label-regime a0
python src/mapr2026_v3/run_surrogates.py --targets-path data/processed/regression_targets_a0.parquet --label-regime a0

# HSCC: Operationalization 2, source-community operationalization.
python src/mapr2026_v3/run_baselines.py --targets-path data/processed/regression_targets_hscc_refined.parquet --label-regime hscc
python src/mapr2026_v3/run_surrogates.py --targets-path data/processed/regression_targets_hscc_refined.parquet --label-regime hscc
```

### 4) Preflight check

```powershell
python src/mapr2026_v3/preflight_person2.py
```

## Key Artifacts

The public repo keeps small frozen result artifacts under:

- `Artifacts - frozen results/`
- `Artifacts - feasibility stability/`

The external data protocol bundle supplies the larger core contracts:

- `data/processed/graph_csr.npz`
- `data/processed/ic_scores_primary.parquet`
- `data/processed/regression_targets_a0.parquet`
- `data/processed/regression_targets_hscc_refined.parquet`
- `data/processed/split_masks.parquet`

Generated local result files are written to:

- `outputs/mapr2026_v3_results/baseline_ranking_metrics.csv`
- `outputs/mapr2026_v3_results/surrogate_ranking_metrics.csv`
- `outputs/mapr2026_v3_results/runtime_breakdown.csv`

## Tests

```powershell
pytest -q
```

## Important Notes

- `src/mapr2026_v3` is the active path; legacy stage-0..3 modules are retained only where they build base graph artifacts.
- Keep large raw, intermediate, processed, and generated result files outside git. Use `data/processed/` as the local extraction target for the external data protocol bundle.
