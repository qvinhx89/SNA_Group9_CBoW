# Operationalization Matters: IC-Based Influence Approximation on Twitch Gamers

This repository studies influence approximation on the Twitch Gamers graph, with
the current execution path centered on `src/mapr2026_v3`.

## Paper and Citation

This repository contains the source code and reproducibility artifacts for the
accepted MAPR 2026 paper:

**Operationalization Matters: When Graph-Aware Learning Adds Value for IC-Based
Influence Approximation**<br>
Dinh-Duy Tran, Quoc-Vinh Pham, Quoc-Hai Tran, Hung-Vi Tran, and Hung-Nghiep Tran<br>
University of Information Technology, Ho Chi Minh City, Vietnam<br>
Vietnam National University, Ho Chi Minh City, Vietnam

The official paper link will be added here when the proceedings page or DOI is
available. Processed experiment artifacts are generated locally from the public
dataset by the pipeline below.

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

This project uses the public Twitch Gamers dataset. The raw dataset is available
from SNAP and the original project repository; see the Dataset section below.
The repository does not redistribute the raw dataset or precomputed generated
experiment outputs.

All processed artifacts used by this project, including graph CSR files, split
masks, IC simulation scores, A0/HSCC regression targets, baseline metrics, and
surrogate metrics, can be regenerated from the public dataset by running the
pipeline described below.

Generated files are written to `data/processed/` and `outputs/`. Because IC
labels are Monte Carlo simulation outputs, exact bitwise reproduction may depend
on random seeds, software versions, and hardware. Frozen summary artifacts are
included for checking the expected numerical range of the reported results.

## Dataset

This project uses the Twitch Gamers social network dataset.

- Dataset page: https://snap.stanford.edu/data/twitch_gamers.html
- Original project repository: https://github.com/benedekrozemberczki/datasets#twitch-gamers
- Dataset paper: https://arxiv.org/abs/2101.03091

Please cite the original Twitch Gamers dataset paper when using the underlying
graph data:

Benedek Rozemberczki and Rik Sarkar. "Twitch Gamers: a Dataset for Evaluating
Proximity Preserving and Structural Role-based Node Embeddings." arXiv:2101.03091,
2021.

```bibtex
@misc{rozemberczki2021twitch,
  title         = {Twitch Gamers: A Dataset for Evaluating Proximity Preserving and Structural Role-based Node Embeddings},
  author        = {Rozemberczki, Benedek and Sarkar, Rik},
  year          = {2021},
  eprint        = {2101.03091},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SI}
}
```

## Raw Data Setup

Download `twitch_gamers.zip` from the SNAP dataset page and extract the raw CSV
files directly under `data/raw/`. The pipeline expects the following files:

- `data/raw/large_twitch_edges.csv`
- `data/raw/large_twitch_features.csv`

On Windows PowerShell, one possible setup is:

```powershell
New-Item -ItemType Directory -Force data/raw
Invoke-WebRequest -Uri https://snap.stanford.edu/data/twitch_gamers.zip -OutFile data/raw/twitch_gamers.zip
Expand-Archive data/raw/twitch_gamers.zip -DestinationPath data/raw -Force
```

If the archive extracts into a nested folder, move `large_twitch_edges.csv` and
`large_twitch_features.csv` so they are directly inside `data/raw/`.

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
- `data/processed/`: generated processed artifacts.
- `Artifacts - frozen results/`: small paper-facing metric and bootstrap artifacts.
- `Artifacts - feasibility stability/`: small stability and feasibility diagnostics.
- `outputs/`: generated experiment outputs.
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
python src/mapr2026_v3/ic_labels_primary.py --n-runs 200 --n-sample 5000 --n-jobs -1 --out-reg data/processed/regression_targets_a0.parquet
python src/mapr2026_v3/ic_labels_hscc_refined.py --n-jobs -1
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

Frozen result artifacts are included under:

- `Artifacts - frozen results/`
- `Artifacts - feasibility stability/`

The pipeline regenerates the larger core contracts locally:

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
- Keep large raw, intermediate, processed, and generated result files outside git. Use `data/processed/` for regenerated processed artifacts.
