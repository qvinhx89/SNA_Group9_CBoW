"""MAPR2026 v3 — Null model typology comparison (scaffold).

Owner: Person 2

Inputs
------
- data/processed/typology_labels_ic_views.parquet
- data/processed/graph_active.edgelist (or CSR)

Outputs
-------
- outputs/mapr2026_v3_results/null_model_typology_summary.json

Scaffold behavior
-----------------
- --dry-run writes a placeholder JSON with required metadata.
- Real implementation should follow plan v3: configuration model realizations, compare quadrant profiles.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _shared import PATHS, ensure_dir, now_iso, require_columns, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MAPR2026 v3 null model typology scaffold")
    p.add_argument("--typology", default=PATHS.typology)
    p.add_argument("--out-dir", default=PATHS.results_dir)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)

    typology_path = Path(args.typology)
    out_path = out_dir / "null_model_typology_summary.json"

    if args.dry_run:
        # Dry-run: emit placeholder without requiring typology file.
        n_nodes = 0
        if typology_path.exists():
            df = pd.read_parquet(typology_path)
            n_nodes = int(df.shape[0])
        else:
            print(f"[dry-run] Typology file not found ({typology_path}); n_nodes will be 0 in placeholder.")
        write_json(
            out_path,
            {
                "timestamp": now_iso(),
                "dry_run": True,
                "n_nodes": n_nodes,
                "note": "Scaffold placeholder. Implement configuration-model comparisons per MAPR2026 v3.",
            },
        )
        print(f"[OK] Wrote dry-run placeholder: {out_path}")
        return

    if not typology_path.exists():
        raise FileNotFoundError(f"Missing typology labels: {typology_path}")

    df = pd.read_parquet(typology_path)
    require_columns(df, ["node_id", "typology_label"], "typology")

    raise NotImplementedError(
        "Implement null model typology comparison. Run with --dry-run to emit placeholder output."
    )


if __name__ == "__main__":
    main()
