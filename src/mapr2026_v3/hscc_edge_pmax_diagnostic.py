"""HSCC — fraction of directed CSR edges where p_unc exceeds the *effective* upper cap.

Matches the per-edge formula in ``ic_labels_hscc_refined`` (CPU + CUDA precompute):
  p_unc(u->v) = base_sender[u] * (1 + gamma)   if c_u != c_v
                base_sender[u]                 otherwise
where base_sender[u] = lambda * phi(u) / deg(u), deg=0 => 0.

Labeling applies ``min(p_unc, p_max, 1.0)`` (sequential clip to ``p_max`` then to ``1``).
This scan uses threshold ``min(p_max, 1.0)``, so it matches both steps when ``p_max > 1``.

No Monte Carlo; O(|E|) scan. Use for paper saturation / clipping sentence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAP_DIR = Path(__file__).resolve().parent
if str(MAP_DIR) not in sys.path:
    sys.path.insert(0, str(MAP_DIR))

from _shared import PATHS, load_csr_npz, now_iso  # noqa: E402
from ic_labels_hscc_refined import _load_community_ids, _load_source_strength  # noqa: E402


def resolve_project_path(path_like: str | Path) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _scan_edges(
    indptr: np.ndarray,
    indices: np.ndarray,
    base_sender: np.ndarray,
    comm_vec: np.ndarray,
    gamma: float,
    effective_upper: float,
) -> dict[str, Any]:
    n_gt_cap_all = 0
    n_gt_cap_cross = 0
    n_gt_cap_within = 0
    n_edges_active = 0  # edges from u with base_sender[u] > 0
    n_edges_cross = 0
    n_edges_within = 0

    thr = float(effective_upper)

    for u in range(int(indptr.shape[0] - 1)):
        bu = float(base_sender[u])
        if bu <= 0.0:
            continue
        start = int(indptr[u])
        end = int(indptr[u + 1])
        if end <= start:
            continue
        cu = int(comm_vec[u])
        nbrs = indices[start:end]
        cross = comm_vec[nbrs] != cu
        mult = np.where(cross, 1.0 + float(gamma), 1.0)
        p_unc = bu * mult

        n_edges_active += int(p_unc.size)
        n_edges_cross += int(np.sum(cross))
        n_edges_within += int(np.sum(~cross))

        gt = p_unc > thr
        n_gt_cap_all += int(np.sum(gt))
        n_gt_cap_cross += int(np.sum(gt & cross))
        n_gt_cap_within += int(np.sum(gt & ~cross))

    def _pct(num: int, den: int) -> float:
        return float(100.0 * num / den) if den > 0 else float("nan")

    return {
        "n_directed_edges_from_positive_base": int(n_edges_active),
        "n_cross_edges": int(n_edges_cross),
        "n_within_edges": int(n_edges_within),
        "n_gt_effective_upper_all": int(n_gt_cap_all),
        "n_gt_effective_upper_cross": int(n_gt_cap_cross),
        "n_gt_effective_upper_within": int(n_gt_cap_within),
        "pct_gt_effective_upper_all": _pct(n_gt_cap_all, n_edges_active),
        "pct_gt_effective_upper_cross": _pct(n_gt_cap_cross, n_edges_cross),
        "pct_gt_effective_upper_within": _pct(n_gt_cap_within, n_edges_within),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csr", default=PATHS.csr_npz, help="graph_csr.npz")
    p.add_argument("--node-attrs", default=PATHS.node_attributes)
    p.add_argument("--community", default="data/processed/community_labels.parquet")
    p.add_argument("--lambda-coef", type=float, default=1.0)
    p.add_argument("--p-max", type=float, default=1.0)
    p.add_argument(
        "--gammas",
        default="0,0.5,1.0",
        help="Comma-separated gamma values (same semantics as HSCC labeling).",
    )
    p.add_argument(
        "--out-json",
        default="",
        help="Optional path to write full payload JSON (relative to repo root if not absolute).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    csr_path = resolve_project_path(args.csr)
    attrs_path = resolve_project_path(args.node_attrs)
    comm_path = resolve_project_path(args.community)

    csr = load_csr_npz(csr_path)
    indptr = csr["indptr"]
    indices = csr["indices"]
    degrees = csr["degrees"].astype(np.int64, copy=False)
    node_ids_csr = csr["node_ids"].astype(str)
    node_ids_index = pd.Index(node_ids_csr)

    phi_df = _load_source_strength(attrs_path)
    phi_rows = node_ids_index.get_indexer(phi_df["node_id"].astype(str).to_numpy())
    valid_phi = phi_rows >= 0
    phi_vec = np.zeros(len(node_ids_csr), dtype=float)
    phi_vec[phi_rows[valid_phi]] = phi_df.loc[valid_phi, "phi"].to_numpy(dtype=float)

    comm_df = _load_community_ids(comm_path)
    comm_rows = node_ids_index.get_indexer(comm_df["node_id"].astype(str).to_numpy())
    valid_comm = comm_rows >= 0
    comm_vec = np.full(len(node_ids_csr), -1, dtype=np.int64)
    comm_vec[comm_rows[valid_comm]] = comm_df.loc[valid_comm, "community_id"].to_numpy(dtype=np.int64)

    csr_id_set = set(node_ids_csr.tolist())
    phi_id_set = set(phi_df["node_id"].astype(str).tolist())
    comm_id_set = set(comm_df["node_id"].astype(str).tolist())
    n_csr_missing_attrs = len(csr_id_set - phi_id_set)
    n_csr_missing_comm = len(csr_id_set - comm_id_set)
    if n_csr_missing_attrs > 0:
        print(f"[WARN] {n_csr_missing_attrs:,} CSR node_ids have no row in node_attributes (phi forced to 0).")
    if n_csr_missing_comm > 0:
        print(f"[WARN] {n_csr_missing_comm:,} CSR node_ids have no row in community parquet (community_id forced to -1).")

    base_sender = np.zeros(len(node_ids_csr), dtype=float)
    deg_mask = degrees > 0
    base_sender[deg_mask] = float(args.lambda_coef) * (
        phi_vec[deg_mask] / degrees[deg_mask].astype(float)
    )

    gammas = [float(x.strip()) for x in str(args.gammas).split(",") if x.strip() != ""]
    effective_upper = min(float(args.p_max), 1.0)
    payload: dict[str, Any] = {
        "timestamp": now_iso(),
        "csr": str(csr_path),
        "node_attrs": str(attrs_path),
        "community": str(comm_path),
        "lambda_coef": float(args.lambda_coef),
        "p_max": float(args.p_max),
        "effective_upper_cap": float(effective_upper),
        "note": "Counts p_unc > min(p_max,1) on directed edges from u with base_sender[u]>0; matches min(p_unc,p_max,1) tightening.",
        "csr_nodes_missing_node_attributes_row": int(n_csr_missing_attrs),
        "csr_nodes_missing_community_parquet_row": int(n_csr_missing_comm),
        "n_nodes_csr": int(len(node_ids_csr)),
        "n_directed_edges_csr": int(indices.shape[0]),
        "per_gamma": [],
    }

    for gamma in gammas:
        stats = _scan_edges(indptr, indices, base_sender, comm_vec, gamma, effective_upper)
        row = {"gamma": float(gamma), **stats}
        payload["per_gamma"].append(row)
        print(
            f"gamma={gamma:g} | edges(from base>0)={stats['n_directed_edges_from_positive_base']:,} | "
            f"p_unc>{effective_upper:g} (effective upper): {stats['pct_gt_effective_upper_all']:.4f}% all | "
            f"{stats['pct_gt_effective_upper_cross']:.4f}% of cross | "
            f"{stats['pct_gt_effective_upper_within']:.4f}% of within"
        )

    if str(args.out_json).strip():
        out_p = resolve_project_path(str(args.out_json).strip())
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[OK] wrote {out_p}")


if __name__ == "__main__":
    main()
