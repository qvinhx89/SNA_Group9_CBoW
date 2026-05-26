"""MAPR2026 v3 — IC pilot diagnostics (Stage 4).

Owner: Person 1

Purpose
-------
Generate the missing Stage-4 pilot diagnostics artifact with reproducible
sampling/simulation and schema-aligned metrics used for quality gating.

Output
------
- outputs/day1_benchmark/ic_pilot_diagnostics.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import ks_2samp, spearmanr

from _shared import PATHS, load_csr_npz, now_iso, require_columns, write_json
from ic_labels_primary import _sample_labeled_indices, _simulate_ic_once


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate IC pilot diagnostics artifact")
    p.add_argument("--csr", default=PATHS.csr_npz)
    p.add_argument("--centrality", default="data/processed/centrality_table.parquet")
    p.add_argument("--out", default=f"{PATHS.day1_dir}/ic_pilot_diagnostics.json")
    p.add_argument("--n-pilot-nodes", type=int, default=200)
    p.add_argument("--n-pilot-runs", type=int, default=50)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stability-seed-base", type=int, default=10000)
    p.add_argument("--cv-noise-threshold", type=float, default=0.50)
    p.add_argument("--ks-threshold", type=float, default=0.10)
    p.add_argument("--top-pct", type=float, default=0.10)
    return p.parse_args()


def _simulate_runs_for_node(
    source_row: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    inv_degrees: np.ndarray,
    n_runs: int,
    worker_seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(worker_seed)
    runs = np.empty(n_runs, dtype=np.int32)
    for i in range(n_runs):
        runs[i] = _simulate_ic_once(
            source=source_row,
            indptr=indptr,
            indices=indices,
            inv_degrees=inv_degrees,
            rng=rng,
        )
    return runs


def _jaccard(a: set[int], b: set[int]) -> float:
    union = a | b
    if not union:
        return 1.0
    return float(len(a & b) / len(union))


def _safe_float(x: float | np.floating) -> float:
    return float(x) if np.isfinite(x) else float("nan")


def _ks_results(
    centrality_path: str | Path,
    pilot_node_ids: np.ndarray,
    ks_threshold: float,
) -> dict[str, dict[str, float | bool | int | None]]:
    df_c = pd.read_parquet(centrality_path)
    require_columns(df_c, ["node_id", "degree", "kshell", "pagerank"], "centrality_table")
    df_c = df_c[["node_id", "degree", "kshell", "pagerank"]].copy()
    df_c["node_id"] = df_c["node_id"].astype(str)

    pilot_set = set(pilot_node_ids.astype(str).tolist())
    pilot = df_c[df_c["node_id"].isin(pilot_set)]

    out: dict[str, dict[str, float | bool | int | None]] = {}
    for feat in ["degree", "kshell", "pagerank"]:
        full_vals = pd.to_numeric(df_c[feat], errors="coerce").dropna().to_numpy(dtype=float)
        pilot_vals = pd.to_numeric(pilot[feat], errors="coerce").dropna().to_numpy(dtype=float)
        if len(full_vals) == 0 or len(pilot_vals) == 0:
            out[feat] = {
                "ks_stat": None,
                "p_value": None,
                "warn": True,
                "n_pilot": int(len(pilot_vals)),
                "n_full": int(len(full_vals)),
            }
            continue

        stat, pval = ks_2samp(pilot_vals, full_vals)
        out[feat] = {
            "ks_stat": float(stat),
            "p_value": float(pval),
            "warn": bool(float(stat) > float(ks_threshold)),
            "n_pilot": int(len(pilot_vals)),
            "n_full": int(len(full_vals)),
        }
    return out


def _per_quintile_cv_table(
    sampled_degrees: np.ndarray,
    per_node_cv: np.ndarray,
    cv_noise_threshold: float,
) -> list[dict[str, float | int]]:
    """Compute per-degree-quintile CV diagnostics on pilot nodes.

    The table is based on the sampled pilot set and is intended for
    reporting/diagnostics (not for gate computation, which keeps `cv_score`).
    """
    df = pd.DataFrame(
        {
            "degree": sampled_degrees.astype(float),
            "per_node_cv": per_node_cv.astype(float),
        }
    )

    # If qcut collapses bins on low-variance slices, keep whatever bins remain.
    df["deg_q"] = pd.qcut(df["degree"], q=5, labels=False, duplicates="drop")

    rows: list[dict[str, float | int]] = []
    for q in sorted(df["deg_q"].dropna().astype(int).unique().tolist()):
        sub = df[df["deg_q"] == q]
        rows.append(
            {
                "quintile": int(q),
                "n_nodes": int(len(sub)),
                "cv_mean": float(sub["per_node_cv"].mean()),
                "cv_median": float(sub["per_node_cv"].median()),
                "cv_noise_count": int((sub["per_node_cv"] > float(cv_noise_threshold)).sum()),
            }
        )
    return rows


def main() -> None:
    args = parse_args()

    csr = load_csr_npz(args.csr)
    degrees = csr["degrees"]
    rows = _sample_labeled_indices(
        degrees=degrees,
        n_sample=int(args.n_pilot_nodes),
        seed=int(args.seed),
    )
    rows = np.asarray(rows, dtype=np.int64)
    pilot_node_ids = csr["node_ids"][rows].astype(str)

    inv_degrees = np.zeros_like(degrees, dtype=float)
    positive = degrees > 0
    inv_degrees[positive] = 1.0 / degrees[positive].astype(float)

    def _worker(seed_offset: int, row: int) -> np.ndarray:
        return _simulate_runs_for_node(
            source_row=int(row),
            indptr=csr["indptr"],
            indices=csr["indices"],
            inv_degrees=inv_degrees,
            n_runs=int(args.n_pilot_runs),
            worker_seed=int(seed_offset) + int(row),
        )

    runs_a = Parallel(n_jobs=int(args.n_jobs), backend="loky")(
        delayed(_worker)(int(args.seed), int(row)) for row in rows
    )
    runs_b = Parallel(n_jobs=int(args.n_jobs), backend="loky")(
        delayed(_worker)(int(args.stability_seed_base), int(row)) for row in rows
    )

    mat_a = np.vstack([np.asarray(x, dtype=float) for x in runs_a])
    mat_b = np.vstack([np.asarray(x, dtype=float) for x in runs_b])

    reach_mean_a = mat_a.mean(axis=1)
    reach_mean_b = mat_b.mean(axis=1)

    mean_reach = float(np.mean(reach_mean_a))
    median_reach = float(np.median(reach_mean_a))
    iqr_reach = float(np.percentile(reach_mean_a, 75) - np.percentile(reach_mean_a, 25))

    top_thresh_a = float(np.quantile(reach_mean_a, 1.0 - float(args.top_pct)))
    top_a = reach_mean_a[reach_mean_a >= top_thresh_a]
    top10_to_median_ratio = float(np.mean(top_a) / (median_reach + 1e-9)) if len(top_a) else float("nan")

    per_node_cv = mat_a.std(axis=1, ddof=0) / (reach_mean_a + 1e-9)
    noise_mask = per_node_cv > float(args.cv_noise_threshold)
    cv_noise_count = int(np.sum(noise_mask))
    if np.any(~noise_mask):
        cv_score = float(np.mean(per_node_cv[~noise_mask]))
    else:
        cv_score = float("nan")

    rho, _ = spearmanr(reach_mean_a, reach_mean_b)
    rank_stability = _safe_float(float(rho))

    top_thresh_b = float(np.quantile(reach_mean_b, 1.0 - float(args.top_pct)))
    top_set_a = set(np.where(reach_mean_a >= top_thresh_a)[0].tolist())
    top_set_b = set(np.where(reach_mean_b >= top_thresh_b)[0].tolist())
    jaccard_stability = _jaccard(top_set_a, top_set_b)

    ks = _ks_results(
        centrality_path=args.centrality,
        pilot_node_ids=pilot_node_ids,
        ks_threshold=float(args.ks_threshold),
    )

    per_quintile_cv = _per_quintile_cv_table(
        sampled_degrees=degrees[rows],
        per_node_cv=per_node_cv,
        cv_noise_threshold=float(args.cv_noise_threshold),
    )

    payload = {
        "timestamp": now_iso(),
        "config": {
            "n_pilot_nodes": int(len(rows)),
            "n_pilot_runs": int(args.n_pilot_runs),
            "seed": int(args.seed),
            "stability_seed_base": int(args.stability_seed_base),
            "worker_seed_rule_A": "seed + node_row",
            "worker_seed_rule_B": "stability_seed_base + node_row",
            "cv_noise_threshold": float(args.cv_noise_threshold),
            "ks_threshold": float(args.ks_threshold),
            "top_pct": float(args.top_pct),
            "p_model": "weighted_cascade",
        },
        "summary": {
            "mean_reach": mean_reach,
            "median_reach": median_reach,
            "iqr_reach": iqr_reach,
            "top10_to_median_ratio": top10_to_median_ratio,
            "rank_stability": rank_stability,
            "cv_score": cv_score,
            "cv_noise_count": cv_noise_count,
            "per_quintile_cv": per_quintile_cv,
            "jaccard_stability": float(jaccard_stability),
            "n_pilot_nodes": int(len(rows)),
            "n_pilot_runs": int(args.n_pilot_runs),
        },
        "ks_results": ks,
    }

    write_json(args.out, payload)
    print(f"[OK] Wrote pilot diagnostics: {Path(args.out)}")


if __name__ == "__main__":
    main()
