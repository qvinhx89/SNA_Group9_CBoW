"""MAPR2026 v3.1 — I-A pilot gate diagnostics.

Reviewer-driven policy (R1+R2)
-----------------------------
- I-A pilot gate is MUST (unconditional): cheap (~20 minutes) and high leverage.
- Full I-A labeling is conditional-MUST only if the pilot passes.

I-A (Attribute-Informed) rule
-----------------------------
    p(u,v) = w(v) / sum_{x in N(u)} w(x),  where w(v) = log1p(max(views(v), 0))

Pilot defaults
--------------
- 200 nodes × 50 MC runs

Outputs
-------
- outputs/mapr2026_v3_results/ia_pilot_diagnostics.json

Notes
-----
This script is intentionally narrow: it runs the pilot and reports the fixed
PASS criteria used to decide whether to activate full I-A.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr

from _shared import PATHS, load_csr_npz, now_iso, require_columns, write_json
from ic_labels_primary import _sample_labeled_indices


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="I-A pilot gate diagnostics (MAPR2026 v3.1)")
    p.add_argument("--csr", default=PATHS.csr_npz)
    p.add_argument("--node-attributes", default=PATHS.node_attributes)
    p.add_argument(
        "--out",
        default=str(Path(PATHS.results_dir) / "ia_pilot_diagnostics.json"),
        help="Output JSON path",
    )
    p.add_argument("--n-pilot-nodes", type=int, default=200)
    p.add_argument("--n-pilot-runs", type=int, default=50)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)

    # Fixed thresholds for the decision tree (documented in plans).
    p.add_argument("--thresh-cv", type=float, default=0.30)
    p.add_argument("--thresh-abs-rho-degree", type=float, default=0.75)
    p.add_argument("--thresh-abs-rho-proxy", type=float, default=0.85)
    return p.parse_args()


def _safe_float(x: float | np.floating) -> float:
    return float(x) if np.isfinite(x) else float("nan")


def _simulate_ic_once_ia(
    source: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    w: np.ndarray,
    neigh_w_sum: np.ndarray,
    rng: np.random.Generator,
) -> int:
    activated = {int(source)}
    frontier = [int(source)]

    while frontier:
        next_frontier: list[int] = []
        for u in frontier:
            denom = float(neigh_w_sum[u])
            if denom <= 0.0:
                continue
            start_idx = int(indptr[u])
            end_idx = int(indptr[u + 1])
            for nb_raw in indices[start_idx:end_idx]:
                v = int(nb_raw)
                if v in activated:
                    continue
                p = float(w[v]) / denom
                if p <= 0.0:
                    continue
                if rng.random() < p:
                    activated.add(v)
                    next_frontier.append(v)
        frontier = next_frontier

    return int(len(activated))


def _simulate_node_summary_ia(
    source: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    w: np.ndarray,
    neigh_w_sum: np.ndarray,
    n_runs: int,
    worker_seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(int(worker_seed))
    runs = np.empty(int(n_runs), dtype=np.int32)
    for i in range(int(n_runs)):
        runs[i] = _simulate_ic_once_ia(
            source=source,
            indptr=indptr,
            indices=indices,
            w=w,
            neigh_w_sum=neigh_w_sum,
            rng=rng,
        )
    return float(runs.mean()), float(runs.std(ddof=0))


def main() -> None:
    args = parse_args()

    csr = load_csr_npz(args.csr)
    indptr = csr["indptr"]
    indices = csr["indices"]
    degrees = csr["degrees"]
    node_ids = csr["node_ids"].astype(str)

    # Stratified-by-degree sampling (same philosophy as A0 pilot).
    pilot_rows = _sample_labeled_indices(
        degrees=degrees,
        n_sample=int(args.n_pilot_nodes),
        seed=int(args.seed),
    ).astype(np.int64)

    # Load views and align to CSR order.
    df_attr = pd.read_parquet(Path(args.node_attributes))
    df_attr["node_id"] = df_attr["node_id"].astype(str)
    require_columns(df_attr, ["node_id", "views"], "node_attributes")

    views_series = pd.to_numeric(df_attr.set_index("node_id")["views"], errors="coerce").fillna(0.0)
    views_aligned = views_series.reindex(pd.Index(node_ids), fill_value=0.0).to_numpy(dtype=np.float64)

    w = np.log1p(np.maximum(0.0, views_aligned)).astype(np.float64, copy=False)

    # Precompute denom per source node u: sum_{v in N(u)} w[v]
    neigh_w_sum = np.add.reduceat(w[indices], indptr[:-1]).astype(np.float64, copy=False)
    neigh_w_sum[degrees <= 0] = 0.0

    # Proxy: mean w over neighborhood (1-hop).
    nbr_views_mean_proxy = np.zeros_like(neigh_w_sum)
    mask = degrees > 0
    nbr_views_mean_proxy[mask] = neigh_w_sum[mask] / degrees[mask].astype(np.float64)

    def _worker(row: int) -> tuple[float, float]:
        return _simulate_node_summary_ia(
            source=int(row),
            indptr=indptr,
            indices=indices,
            w=w,
            neigh_w_sum=neigh_w_sum,
            n_runs=int(args.n_pilot_runs),
            worker_seed=int(args.seed) + int(row),
        )

    t0 = time.time()
    stats = Parallel(n_jobs=int(args.n_jobs), backend="loky")(delayed(_worker)(int(r)) for r in pilot_rows)
    elapsed = float(time.time() - t0)

    ic_means = np.array([m for m, _ in stats], dtype=np.float64)
    ic_stds = np.array([s for _, s in stats], dtype=np.float64)

    # Summary stats
    mean_reach = float(np.mean(ic_means))
    median_reach = float(np.median(ic_means))
    iqr_reach = float(np.percentile(ic_means, 75) - np.percentile(ic_means, 25))
    top10_thresh = float(np.quantile(ic_means, 0.90))
    top10 = ic_means[ic_means >= top10_thresh]
    top10_to_median_ratio = float(np.mean(top10) / (median_reach + 1e-9)) if len(top10) else float("nan")

    cv_across_nodes = float(ic_means.std(ddof=0) / (mean_reach + 1e-9))

    rho_deg, _ = spearmanr(ic_means, degrees[pilot_rows].astype(np.float64))
    rho_proxy, _ = spearmanr(ic_means, nbr_views_mean_proxy[pilot_rows].astype(np.float64))

    rho_deg_f = _safe_float(float(rho_deg))
    rho_proxy_f = _safe_float(float(rho_proxy))

    pass_cv = bool(cv_across_nodes > float(args.thresh_cv))
    pass_deg = bool(abs(rho_deg_f) < float(args.thresh_abs_rho_degree))
    pass_proxy = bool(abs(rho_proxy_f) < float(args.thresh_abs_rho_proxy))
    pilot_pass = bool(pass_cv and pass_deg and pass_proxy)

    payload = {
        "timestamp": now_iso(),
        "config": {
            "p_model": "ia_row_norm_views",
            "w_transform": "log1p(max(views,0))",
            "n_pilot_nodes": int(len(pilot_rows)),
            "n_pilot_runs": int(args.n_pilot_runs),
            "seed": int(args.seed),
            "worker_seed_rule": "seed + csr_row",
            "thresholds": {
                "cv_gt": float(args.thresh_cv),
                "abs_rho_deg_lt": float(args.thresh_abs_rho_degree),
                "abs_rho_proxy_lt": float(args.thresh_abs_rho_proxy),
            },
        },
        "summary": {
            "mean_reach": mean_reach,
            "median_reach": median_reach,
            "iqr_reach": iqr_reach,
            "top10_to_median_ratio": top10_to_median_ratio,
            "cv_across_nodes": cv_across_nodes,
            "spearman_ic_vs_degree": rho_deg_f,
            "spearman_ic_vs_nbr_views_mean_proxy": rho_proxy_f,
            "pass": pilot_pass,
            "pass_checks": {
                "cv": pass_cv,
                "degree": pass_deg,
                "proxy": pass_proxy,
            },
            "elapsed_sec": elapsed,
        },
    }

    write_json(args.out, payload)

    decision = "PASS (activate full I-A)" if pilot_pass else "FAIL (skip full I-A; commit A0-only)"
    print(
        "[OK] I-A pilot done "
        f"(n_nodes={len(pilot_rows)}, n_runs={int(args.n_pilot_runs)}, elapsed_sec={elapsed:.2f}).\n"
        f" - cv_across_nodes={cv_across_nodes:.3f}\n"
        f" - rho(ic,degree)={rho_deg_f:.3f}\n"
        f" - rho(ic,nbr_views_mean_proxy)={rho_proxy_f:.3f}\n"
        f" - decision={decision}\n"
        f" - out={Path(args.out)}"
    )


if __name__ == "__main__":
    main()
