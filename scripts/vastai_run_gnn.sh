#!/usr/bin/env bash
set -euo pipefail

# Minimal runner for vast.ai (Linux).
# Assumes you already activated a Python env with torch + torch_geometric installed.

OUT_DIR="outputs/mapr2026_v3_results"
mkdir -p "${OUT_DIR}"

# 1) Edge-only (graph-only strict, x=1)
python src/mapr2026_v3/run_surrogates.py \
  --only-edge-only \
  --early-stop --patience 20 \
  --out-csv "${OUT_DIR}/surrogate_edge_only.csv"

# 2) C2 architecture comparison on A0 targets
python src/mapr2026_v3/run_surrogates.py \
  --include-c2-arch \
  --early-stop --patience 20 \
  --out-csv "${OUT_DIR}/surrogate_c2_raw_attr.csv"

# 3) C2 on A2 targets (if present)
if [[ -f data/processed/regression_targets_a2.parquet ]]; then
  python src/mapr2026_v3/run_surrogates.py \
    --targets-path data/processed/regression_targets_a2.parquet \
    --include-c2-arch \
    --early-stop --patience 20 \
    --out-csv "${OUT_DIR}/surrogate_c2_a2_raw_attr.csv"
fi

echo "[OK] vastai_run_gnn.sh completed"
ls -lah "${OUT_DIR}" | sed -n '1,200p'
