"""MAPR2026 v3 — Run surrogate models (Node2Vec+LR / MLP / optional GNN) scaffold.

Owner: Person 3

Inputs
------
- data/processed/regression_targets.parquet
- data/processed/node_attributes.parquet
- (optional) graph artifacts for embeddings/GNN

Outputs
-------
- outputs/mapr2026_v3_results/surrogate_ranking_metrics.csv (suggested)
- (optional) model checkpoints under outputs/mapr2026_v3_results/

Scaffold behavior
-----------------
- --dry-run writes a header-only CSV.
- Real mode should implement training (5 seeds) + aggregation per MAPR2026 v3.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _shared import PATHS, ensure_dir, now_iso


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MAPR2026 v3 surrogate runner scaffold")
    p.add_argument("--out-dir", default=PATHS.results_dir)
    p.add_argument("--out-csv", default=str(Path(PATHS.results_dir) / "surrogate_ranking_metrics.csv"))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    cols = ["model_name", "spearman_rho_mean", "spearman_rho_std", "ndcg_mean", "ndcg_std", "precision_mean", "precision_std", "runtime_sec"]

    if args.dry_run:
        pd.DataFrame(columns=cols).to_csv(out_csv, index=False)
        print(f"[OK] Wrote dry-run surrogate metrics header: {out_csv} (timestamp={now_iso()})")
        return

    raise NotImplementedError(
        "Implement surrogate training + evaluation (5 seeds) per MAPR2026 v3. Run with --dry-run for header-only."
    )


if __name__ == "__main__":
    main()
