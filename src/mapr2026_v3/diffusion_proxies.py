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
- --dry-run emits a header-only parquet (0 rows) and a status JSON marker.
    This prevents accidental consumption in evaluation/runtime.
- Default mode computes real full-graph one-hop/two-hop proxies from CSR.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from _shared import PATHS, ensure_parent, load_csr_npz, now_iso, require_columns, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MAPR2026 v3 diffusion proxies scaffold")
    p.add_argument("--csr", default=PATHS.csr_npz)
    p.add_argument("--out", default=PATHS.proxies)
    p.add_argument("--status-json", default=PATHS.proxies_status)
    p.add_argument("--runtime-csv", default=getattr(PATHS, "runtime_csv", "outputs/mapr2026_v3_results/runtime_breakdown.csv"))
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


def _compute_one_hop(indptr: np.ndarray, indices: np.ndarray, inv_deg: np.ndarray) -> np.ndarray:
    n_nodes = inv_deg.shape[0]
    one_hop = np.zeros(n_nodes, dtype=np.float64)
    for u in range(n_nodes):
        start, end = int(indptr[u]), int(indptr[u + 1])
        nbrs = indices[start:end]
        if nbrs.size:
            one_hop[u] = float(inv_deg[nbrs].sum())
    return one_hop


def _assert_csr_bidirectional(indptr: np.ndarray, indices: np.ndarray) -> None:
    """Fail fast when CSR violates undirected-bidirectional storage contract.

    Contract requirement: for every directed edge u->v in CSR indices, v->u must
    also exist exactly the same number of times.
    """
    n_nodes = int(indptr.shape[0] - 1)
    rows = np.repeat(np.arange(n_nodes, dtype=np.int64), np.diff(indptr).astype(np.int64, copy=False))
    cols = indices.astype(np.int64, copy=False)

    if rows.shape[0] != cols.shape[0]:
        raise ValueError("CSR symmetry check failed: rows/cols edge array length mismatch.")

    edge_codes = rows * n_nodes + cols
    reverse_codes = cols * n_nodes + rows

    edge_codes_sorted = np.sort(edge_codes)
    reverse_codes_sorted = np.sort(reverse_codes)

    if np.array_equal(edge_codes_sorted, reverse_codes_sorted):
        return

    mismatch_idx = np.flatnonzero(edge_codes_sorted != reverse_codes_sorted)
    if mismatch_idx.size > 0:
        sample_code = int(edge_codes_sorted[int(mismatch_idx[0])])
        u = sample_code // n_nodes
        v = sample_code % n_nodes
        detail = f" Example unmatched directed edge: row_index {u}->{v}."
    else:
        detail = ""

    raise ValueError(
        "CSR symmetry check failed: graph_csr.npz must store undirected edges in both directions (u->v and v->u)."
        + detail
    )


def _compute_two_hop(
    indptr: np.ndarray,
    indices: np.ndarray,
    inv_deg: np.ndarray,
    one_hop: np.ndarray,
) -> np.ndarray:
    """Compute two-hop proxy with O(E) aggregation under undirected CSR contract.

    For each node u:
      two_hop(u) = sum_{v in N(u)} sum_{w in N(v), w != u} 1/deg(w)

    The CSR contract for this pipeline stores undirected edges in both directions,
    so each neighbor v contributes one back-edge term to subtract.
    """
    n_nodes = inv_deg.shape[0]
    two_hop = np.zeros(n_nodes, dtype=np.float64)
    for u in range(n_nodes):
        start, end = int(indptr[u]), int(indptr[u + 1])
        nbrs = indices[start:end]
        if not nbrs.size:
            continue
        two_hop[u] = float(one_hop[nbrs].sum() - (nbrs.size * inv_deg[u]))
    return two_hop


def _upsert_runtime_row(runtime_csv_path: Path, inference_sec_full_graph: float) -> None:
    cols = ["model_name", "inference_sec_full_graph", "train_sec"]
    ensure_parent(runtime_csv_path)

    if runtime_csv_path.exists():
        runtime_df = pd.read_csv(runtime_csv_path)
    else:
        runtime_df = pd.DataFrame(columns=cols)

    for col in cols:
        if col not in runtime_df.columns:
            runtime_df[col] = np.nan

    runtime_df = runtime_df[cols].copy()
    runtime_df = runtime_df[runtime_df["model_name"] != "diffusion_proxies"].copy()
    runtime_df.loc[len(runtime_df)] = {
        "model_name": "diffusion_proxies",
        "inference_sec_full_graph": float(inference_sec_full_graph),
        "train_sec": np.nan,
    }
    runtime_df.to_csv(runtime_csv_path, index=False)


def main() -> None:
    args = parse_args()
    csr_path = _resolve_io_path(args.csr)
    out_path = _resolve_io_path(args.out)
    status_path = _resolve_io_path(args.status_json)
    runtime_csv_path = _resolve_io_path(args.runtime_csv)

    csr: dict[str, np.ndarray] | None = None
    csr_exists = csr_path.exists()
    if csr_exists:
        csr = load_csr_npz(csr_path)
    elif not args.dry_run:
        raise FileNotFoundError(
            f"Missing CSR artifact: {csr_path}. Run export_csr.py first (or use --dry-run)."
        )

    if args.dry_run:
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
        return

    if csr is None:
        raise RuntimeError("CSR data not loaded in real mode")

    indptr = csr["indptr"]
    indices = csr["indices"]
    degrees = csr["degrees"]
    node_ids = csr["node_ids"]

    inv_deg = 1.0 / np.maximum(degrees.astype(np.float64, copy=False), 1.0)

    # Two-hop optimization assumes undirected CSR is stored bidirectionally.
    _assert_csr_bidirectional(indptr, indices)

    t0 = time.time()
    one_hop = _compute_one_hop(indptr, indices, inv_deg)
    two_hop = _compute_two_hop(indptr, indices, inv_deg, one_hop)
    inference_sec = time.time() - t0

    df = pd.DataFrame(
        {
            "node_id": pd.Series(node_ids, dtype="string"),
            "one_hop_spread": one_hop,
            "two_hop_spread": two_hop,
        }
    )
    require_columns(df, ["node_id", "one_hop_spread", "two_hop_spread"], "diffusion_proxies")

    if int(df["node_id"].nunique()) != int(df.shape[0]):
        raise ValueError("diffusion_proxies contains duplicate node_id rows")
    if int(df.shape[0]) != int(node_ids.shape[0]):
        raise ValueError(
            "diffusion_proxies does not cover full active graph: "
            f"got {df.shape[0]}, expected {node_ids.shape[0]}"
        )
    if df[["one_hop_spread", "two_hop_spread"]].isna().any().any():
        raise ValueError("NaN detected in computed diffusion proxies")

    ensure_parent(out_path)
    df.to_parquet(out_path, index=False)
    _upsert_runtime_row(runtime_csv_path, inference_sec_full_graph=inference_sec)

    write_json(
        status_path,
        {
            "timestamp": now_iso(),
            "mode": "real_full_graph",
            "proxies_path": str(out_path),
            "rows": int(df.shape[0]),
            "ready_for_eval_runtime": True,
            "runtime_csv_path": str(runtime_csv_path),
            "inference_sec_full_graph": float(inference_sec),
            "csr_detected": True,
            "n_nodes": int(node_ids.shape[0]),
        },
    )

    print(
        "[OK] Wrote real diffusion proxies: "
        f"{out_path} rows={df.shape[0]} inference_sec_full_graph={inference_sec:.4f}"
    )
    print(f"[OK] Updated runtime breakdown: {runtime_csv_path}")
    print(f"[OK] Updated status marker: {status_path}")


if __name__ == "__main__":
    main()
