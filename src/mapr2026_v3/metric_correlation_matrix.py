"""MAPR2026 v3 — Metric correlation matrix (8×8).

Owner: Person 2

Purpose
-------
Generate the MAPR-MUST artifact:
- outputs/mapr2026_v3_results/metric_correlation_matrix.json

This is intentionally standalone so it does NOT depend on community detection
(Track B BOOST). It operates on the labeled IC subset only (same node set as
ic_scores_primary.parquet), but pulls other features from full-graph tables.

Contract (validated by preflight_person2.py)
-------------------------------------------
Top-level keys:
- timestamp
- n_rows
- n_rows_expected
- coverage_ok
- metrics
- rho_matrix
- p_matrix_corrected
- column_mapping

Metrics order (locked)
----------------------
[ic_score_mean, views, degree, pagerank, kshell, betweenness_approx,
 one_hop_spread, two_hop_spread]

Notes
-----
- Correlations are Spearman's rho.
- p-values are corrected using BH-FDR over all off-diagonal pairs.
- Diagonal of p_matrix_corrected is set to 1.0 by convention.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

from _shared import PATHS, ensure_parent, now_iso, require_columns, write_json


EXPECTED_METRICS: list[str] = [
    "ic_score_mean",
    "views",
    "degree",
    "pagerank",
    "kshell",
    "betweenness_approx",
    "one_hop_spread",
    "two_hop_spread",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MAPR2026 v3 metric correlation matrix (8×8)")
    p.add_argument("--ic", default=PATHS.ic_scores)
    p.add_argument("--node-attrs", default=PATHS.node_attributes)
    p.add_argument("--centrality", default="data/processed/centrality_table.parquet")
    p.add_argument("--kshell", default="data/processed/kshell_table.parquet")
    p.add_argument("--proxies", default=PATHS.proxies)
    p.add_argument("--out", default="outputs/mapr2026_v3_results/metric_correlation_matrix.json")
    p.add_argument(
        "--include-rho-by-degree-quintile",
        action="store_true",
        help="[IF TIME] Add rho_by_degree_quintile to output JSON.",
    )
    p.add_argument("--min-quintile-n", type=int, default=30)
    return p.parse_args()


def _ensure_node_id_str(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["node_id"] = df["node_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    return df


def _load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, str]]:
    ic_path = Path(args.ic)
    attrs_path = Path(args.node_attrs)
    centrality_path = Path(args.centrality)
    kshell_path = Path(args.kshell)
    proxies_path = Path(args.proxies)

    if not ic_path.exists():
        raise FileNotFoundError(f"Missing IC scores: {ic_path}")
    if not attrs_path.exists():
        raise FileNotFoundError(f"Missing node attributes: {attrs_path}")
    if not centrality_path.exists():
        raise FileNotFoundError(f"Missing centrality table: {centrality_path}")
    if not kshell_path.exists():
        raise FileNotFoundError(f"Missing k-shell table: {kshell_path}")
    if not proxies_path.exists():
        raise FileNotFoundError(f"Missing diffusion proxies: {proxies_path}")

    df_ic = _ensure_node_id_str(pd.read_parquet(ic_path))
    require_columns(df_ic, ["node_id", "ic_score_mean"], "ic_scores_primary")

    df_attrs = _ensure_node_id_str(pd.read_parquet(attrs_path))
    require_columns(df_attrs, ["node_id", "views", "degree"], "node_attributes")

    df_c = _ensure_node_id_str(pd.read_parquet(centrality_path))
    require_columns(df_c, ["node_id", "pagerank"], "centrality_table")

    df_k = _ensure_node_id_str(pd.read_parquet(kshell_path))
    require_columns(df_k, ["node_id", "kshell"], "kshell_table")

    df_p = _ensure_node_id_str(pd.read_parquet(proxies_path))
    require_columns(df_p, ["node_id", "one_hop_spread", "two_hop_spread"], "diffusion_proxies")

    # Canonical mapping for preflight + provenance.
    # Centrality table uses 'betweenness' (not betweenness_approx).
    bet_col = "betweenness_approx" if "betweenness_approx" in df_c.columns else "betweenness"
    if bet_col not in df_c.columns:
        raise ValueError(
            "centrality_table missing betweenness column: expected 'betweenness_approx' or 'betweenness'"
        )

    column_mapping = {
        "ic_score_mean": "ic_score_mean",
        "views": "views",
        "degree": "degree",
        "pagerank": "pagerank",
        "kshell": "kshell",
        "betweenness_approx": bet_col,
        "one_hop_spread": "one_hop_spread",
        "two_hop_spread": "two_hop_spread",
    }

    # Restrict to labeled IC node set; preserve expected row count.
    node_ids = df_ic[["node_id"]].drop_duplicates().copy()
    if len(node_ids) != int(df_ic["node_id"].nunique()):
        raise ValueError("ic_scores_primary contains duplicate node_id rows")

    work = (
        node_ids
        .merge(df_ic[["node_id", "ic_score_mean"]], on="node_id", how="left")
        .merge(df_attrs[["node_id", "views", "degree"]], on="node_id", how="left")
        .merge(df_c[["node_id", "pagerank", bet_col]], on="node_id", how="left")
        .merge(df_k[["node_id", "kshell"]], on="node_id", how="left")
        .merge(df_p[["node_id", "one_hop_spread", "two_hop_spread"]], on="node_id", how="left")
    )

    # Normalize types.
    for col in [
        "ic_score_mean",
        "views",
        "degree",
        "pagerank",
        bet_col,
        "kshell",
        "one_hop_spread",
        "two_hop_spread",
    ]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    if work.isna().any().any():
        na_counts = work[EXPECTED_METRICS + [bet_col]].isna().sum().to_dict()
        # Keep this strict: preflight requires full coverage across labeled nodes.
        raise ValueError(f"Missing metric values after merge (must be zero-missing): {na_counts}")

    # Map betweenness into canonical 'betweenness_approx' slot.
    work["betweenness_approx"] = work[bet_col].astype(float)

    # Keep only canonical metric columns.
    final = work[["node_id"] + EXPECTED_METRICS].copy()
    return final, column_mapping


def _spearman_matrix(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute Spearman rho and raw p-value matrices for a (n×m) numeric array."""
    n_metrics = values.shape[1]
    rho = np.eye(n_metrics, dtype=float)
    p = np.ones((n_metrics, n_metrics), dtype=float)

    for i in range(n_metrics):
        for j in range(i + 1, n_metrics):
            res = spearmanr(values[:, i], values[:, j])
            r = getattr(res, "correlation", res[0])
            pv = getattr(res, "pvalue", res[1])

            r_scalar = float(np.asarray(r).reshape(-1)[0])
            pv_scalar = float(np.asarray(pv).reshape(-1)[0])

            rho[i, j] = r_scalar
            rho[j, i] = r_scalar
            p[i, j] = pv_scalar
            p[j, i] = pv_scalar

    return rho, p


def _bh_fdr_correct(p_raw: np.ndarray) -> np.ndarray:
    n = int(p_raw.shape[0])
    if p_raw.shape != (n, n):
        raise ValueError("p_raw must be square")

    iu = np.triu_indices(n, k=1)
    pvals = p_raw[iu]

    # BH-FDR over all pairwise tests.
    reject, p_corr_flat, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
    _ = reject  # reject vector not needed in artifact contract

    p_corr = np.ones((n, n), dtype=float)
    p_corr[iu] = p_corr_flat
    p_corr[(iu[1], iu[0])] = p_corr_flat

    np.fill_diagonal(p_corr, 1.0)
    return p_corr


def _rho_by_degree_quintile(df: pd.DataFrame, min_n: int) -> dict[str, Any]:
    deg = pd.to_numeric(df["degree"], errors="coerce")
    if deg.isna().any():
        raise ValueError("degree contains NaN unexpectedly")

    q = pd.qcut(deg, q=5, labels=False, duplicates="drop")
    out: dict[str, Any] = {}

    for qi in sorted(pd.Series(q).dropna().unique().tolist()):
        mask = (q == qi)
        sub = df.loc[mask, EXPECTED_METRICS]
        n_rows = int(len(sub))
        label = f"q{int(qi) + 1}"
        if n_rows < int(min_n):
            out[label] = {
                "n_rows": n_rows,
                "skipped": True,
                "reason": f"n_rows < min_quintile_n ({n_rows} < {int(min_n)})",
            }
            continue

        values = sub.to_numpy(dtype=float)
        rho, _ = _spearman_matrix(values)
        out[label] = {
            "n_rows": n_rows,
            "skipped": False,
            "rho_matrix": rho.tolist(),
        }

    return out


def main() -> None:
    args = parse_args()
    df, column_mapping = _load_inputs(args)

    n_rows_expected = int(df["node_id"].nunique())
    values = df[EXPECTED_METRICS].to_numpy(dtype=float)

    rho, p_raw = _spearman_matrix(values)
    p_corr = _bh_fdr_correct(p_raw)

    payload: dict[str, Any] = {
        "timestamp": now_iso(),
        "n_rows": int(len(df)),
        "n_rows_expected": int(n_rows_expected),
        "coverage_ok": bool(len(df) == n_rows_expected),
        "metrics": EXPECTED_METRICS,
        "rho_matrix": rho.tolist(),
        "p_matrix_corrected": p_corr.tolist(),
        "column_mapping": column_mapping,
    }

    if args.include_rho_by_degree_quintile:
        payload["rho_by_degree_quintile"] = _rho_by_degree_quintile(df, min_n=int(args.min_quintile_n))

    out_path = Path(args.out)
    ensure_parent(out_path)
    write_json(out_path, payload)
    print(f"[OK] Wrote metric correlation matrix: {out_path} (n_rows={len(df)})")


if __name__ == "__main__":
    main()
