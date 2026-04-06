# SNA Twitch Influencer Project

Implementation scaffold for the project: **Finding the Most Influential Users in Online Communities**.

## Project Goal

Build an end-to-end, reproducible pipeline to:

- Compute structural influence signals (Degree, PageRank, Betweenness, k-shell).
- Construct SIS and 2x2 typology (True, Hidden, Overrated, Non-influencer).
- Validate with single-seed and multi-seed IC simulations.
- Test detectability using surface-metric ML baselines.

## Folder Structure

```text
.
|-- data/
|   |-- raw/
|   |-- interim/
|   `-- processed/
|-- notebooks/
|-- src/
|   |-- config/
|   |-- data/
|   |-- graph/
|   |-- sis/
|   |-- simulation/
|   |-- ml/
|   |-- evaluation/
|   `-- utils/
|-- scripts/
|-- reports/
|   |-- figures/
|   |-- tables/
|   `-- drafts/
|-- outputs/
|   |-- stage1/
|   |-- stage2/
|   |-- stage3/
|   |-- stage4_single_seed/
|   |-- stage5_multi_seed/
|   `-- stage6_ml/
|-- logs/
|   |-- run_history/
|   |-- timing/
|   `-- errors/
|-- tests/
|-- docs/
|-- requirements.txt
`-- README.md
```

## Quick Start

### Prerequisites

- **Python: 3.10–3.12 (64-bit)**
  - The pinned dependencies in `requirements.txt` are not compatible with Python 3.13.

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

#### Recommended (Windows + Anaconda): create a Python 3.12 conda env

```powershell
conda env create -f environment.yml
conda activate sna_group9_cbow_py312
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. Put source dataset files into `data/raw/`.
4. Run stage scripts in order (PowerShell):

```powershell
./scripts/run_stage1_centrality.ps1
./scripts/run_stage2_structure.ps1
./scripts/run_stage3_sis.ps1
./scripts/run_stage4_single_seed_ic.ps1
./scripts/run_stage5_multi_seed_ic.ps1
./scripts/run_stage6_ml.ps1
```

Or run full pipeline:

```powershell
./scripts/run_all.ps1
```

## Reproducibility Rules

- Keep raw data immutable in `data/raw/`.
- Save all stage outputs under `outputs/`.
- Log params and random seeds for every experiment.
- Keep figure/table links in `reports/` aligned with outputs.

## Git Notes

- `.gitkeep` files are included so empty directories are tracked.
- Do not commit large raw datasets unless your GitHub repo supports LFS.
