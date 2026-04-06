"""MAPR2026 v3 — Cheap diffusion proxies (one-hop, two-hop).

Owner: Person 2 (typology/proxies)

Inputs
------
- data/processed/graph_csr.npz

Output (contract)
---------------
- data/processed/diffusion_proxies.parquet
  columns: node_id, one_hop_spread, two_hop_spread

Scope rule (M0-locked)
----------------------
This artifact MUST cover ALL active nodes (not just the labeled IC subset).
Reason: runtime_sec in baseline_ranking_metrics.csv must measure full-graph
inference so the comparison with GNN inference (168k nodes) is fair.

Person 3's eval harness applies the test mask from split_masks.parquet when
computing ranking metrics — Person 2 does NOT need to filter here.

Log total_inference_sec for the full graph to
outputs/mapr2026_v3_results/runtime_breakdown.csv.

Scaffold behavior
-----------------
- Default mode raises NotImplementedError.
- Use --dry-run to emit a schema-correct parquet with placeholder values.
  Note: dry-run samples --n-sample nodes for speed; real mode uses full graph.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _shared import PATHS, ensure_parent, load_csr_npz, now_iso, require_columns


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MAPR2026 v3 diffusion proxies scaffold")
    p.add_argument("--csr", default=PATHS.csr_npz)
    p.add_argument("--out", default=PATHS.proxies)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-sample", type=int, default=2000, help="Dry-run sampling only")
    return p.parse_args()


def _resolve_io_path(path_like: str | Path) -> Path:
    """Resolve relative I/O paths for both run contexts.

    Supports running from either:
    - repository root
    - src/mapr2026_v3
    """
    p = Path(path_like)
    if p.is_absolute():
        return p

    # Prefer current working directory when it already points to a valid location.
    if p.exists() or p.parent.exists():
        return p

    # Fallback to repository root relative to this file.
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / p


def main() -> None:
    args = parse_args()
    csr_path = _resolve_io_path(args.csr)
    out_path = _resolve_io_path(args.out)
    node_ids: np.ndarray

    if csr_path.exists():
        csr = load_csr_npz(csr_path)
        node_ids = csr["node_ids"]
    else:
        if not args.dry_run:
            raise FileNotFoundError(
                f"Missing CSR artifact: {csr_path}. Run export_csr.py first (or use --dry-run)."
            )

        # Dry-run fallback: derive node_ids from node attributes or SIS.
        attrs_path = _resolve_io_path(PATHS.node_attributes)
        sis_path = _resolve_io_path(PATHS.sis_table)

        if attrs_path.exists():
            df_attrs = pd.read_parquet(attrs_path)
            if "node_id" not in df_attrs.columns:
                raise ValueError("node_attributes.parquet must contain node_id for dry-run fallback")
            node_ids = df_attrs["node_id"].astype(str).unique()
        elif sis_path.exists():
            df_sis = pd.read_parquet(sis_path)
            if "node_id" not in df_sis.columns:
                raise ValueError("sis_table.parquet must contain node_id for dry-run fallback")
            node_ids = df_sis["node_id"].astype(str).unique()
        else:
            raise FileNotFoundError(
                "Dry-run fallback requires node_attributes.parquet or sis_table.parquet when CSR is missing."
            )

    if not args.dry_run:
        raise NotImplementedError(
            "Implement one-hop and two-hop expected spread proxies on CSR. "
            "Run with --dry-run to emit placeholder artifacts."
        )

    rng = np.random.default_rng(args.seed)
    n = len(node_ids)
    sample_n = min(int(args.n_sample), n)
    sample_idx = rng.choice(n, size=sample_n, replace=False)

    df = pd.DataFrame(
        {
            "node_id": node_ids[sample_idx],
            "one_hop_spread": np.full(sample_n, np.nan, dtype=float),
            "two_hop_spread": np.full(sample_n, np.nan, dtype=float),
        }
    )
    require_columns(df, ["node_id", "one_hop_spread", "two_hop_spread"], "diffusion_proxies")

    ensure_parent(out_path)
    df.to_parquet(out_path, index=False)
    print(f"[OK] Wrote dry-run proxies placeholder: {out_path} (timestamp={now_iso()})")


if __name__ == "__main__":
    main()
