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
- Use --dry-run to emit a header-only parquet (0 rows) and a status JSON marker.
    This prevents accidental consumption in evaluation/runtime.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _shared import PATHS, ensure_parent, load_csr_npz, now_iso, require_columns, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MAPR2026 v3 diffusion proxies scaffold")
    p.add_argument("--csr", default=PATHS.csr_npz)
    p.add_argument("--out", default=PATHS.proxies)
    p.add_argument("--status-json", default=PATHS.proxies_status)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--n-sample",
        type=int,
        default=2000,
        help="Deprecated for dry-run header-only mode (kept for CLI compatibility)",
    )
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
    status_path = _resolve_io_path(args.status_json)

    csr_exists = csr_path.exists()
    if csr_exists:
        _ = load_csr_npz(csr_path)
    elif not args.dry_run:
        raise FileNotFoundError(
            f"Missing CSR artifact: {csr_path}. Run export_csr.py first (or use --dry-run)."
        )

    if not args.dry_run:
        raise NotImplementedError(
            "Implement one-hop and two-hop expected spread proxies on CSR. "
            "Run with --dry-run to emit placeholder artifacts."
        )

    # Header-only schema artifact for M1 dry-run unblock.
    df = pd.DataFrame(
        {
            "node_id": pd.Series(dtype="string"),
            "one_hop_spread": pd.Series(dtype="float64"),
            "two_hop_spread": pd.Series(dtype="float64"),
        }
    )
    require_columns(df, ["node_id", "one_hop_spread", "two_hop_spread"], "diffusion_proxies")

    ensure_parent(out_path)
    df.to_parquet(out_path, index=False)

    write_json(
        status_path,
        {
            "timestamp": now_iso(),
            "mode": "dry_run_header_only",
            "proxies_path": str(out_path),
            "rows": 0,
            "ready_for_eval_runtime": False,
            "guard_message": (
                "This artifact is dry-run header only. "
                "Do NOT use for evaluation/runtime until real mode is implemented and executed."
            ),
            "csr_detected": bool(csr_exists),
        },
    )

    print(f"[OK] Wrote dry-run proxies header: {out_path} (timestamp={now_iso()})")
    print(f"[OK] Wrote status marker: {status_path}")


if __name__ == "__main__":
    main()
