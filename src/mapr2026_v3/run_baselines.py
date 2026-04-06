"""MAPR2026 v3 — Run analytical baselines (Group 1–3) and write ranking metrics.

Owner: Person 3

Inputs (expected, depending on baseline)
--------------------------------------
- data/processed/regression_targets.parquet
- data/processed/node_attributes.parquet
- data/processed/centrality_table.parquet
- data/processed/kshell_table.parquet
- data/processed/diffusion_proxies.parquet

Output (contract)
---------------
- outputs/mapr2026_v3_results/baseline_ranking_metrics.csv

Scaffold behavior
-----------------
- --dry-run writes an empty CSV with headers only.
- Real mode should implement: load y targets, compute y_pred per baseline, compute metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _shared import PATHS, ensure_dir, now_iso


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MAPR2026 v3 baseline runner scaffold")
    p.add_argument("--out-dir", default=PATHS.results_dir)
    p.add_argument("--out-csv", default=str(Path(PATHS.results_dir) / "baseline_ranking_metrics.csv"))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    cols = ["model_name", "spearman_rho", "ndcg_at_10pct", "precision_at_10pct", "runtime_sec"]

    if args.dry_run:
        pd.DataFrame(columns=cols).to_csv(out_csv, index=False)
        print(f"[OK] Wrote dry-run baseline metrics header: {out_csv} (timestamp={now_iso()})")
        return

    raise NotImplementedError(
        "Implement baseline predictions + metric computation. Run with --dry-run for header-only output."
    )


if __name__ == "__main__":
    main()
