"""MAPR2026 v3 — Hybrid IC (degree + views) pilot sweep.

Goal
----
Try a hybrid IC activation model that combines:
- structure: weighted cascade attention constraint 1/deg(v)
- raw signal: sender strength derived from views(u)

Hybrid model (multiplicative)
-----------------------------
  p(u->v) = min(1, (1/deg(v)) * (1 + gamma * views_strength(u)))
  views_strength(u) = clip((log1p(views(u)) - q_low) / (q_high - q_low), 0, 1)

This script runs a small MC pilot over stratified seed nodes and sweeps gamma.
It writes a JSON summary to support selecting a stable, non-degenerate formula.

Outputs
-------
- outputs/day1_benchmark/hybrid_sweep/hybrid_degree_views_pilot.json

Notes
-----
- Designed to be cheap: default 200 seeds x 50 runs.
- Uses the same CSR IC engine as ic_labels_primary.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score

from _shared import PATHS, ensure_parent, load_csr_npz, now_iso, require_columns
from ic_labels_primary import (
    _compute_views_strength_aligned,
    _resolve_io_path,
    _sample_labeled_indices,
    _simulate_ic_node_summary,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hybrid IC (degree+views) pilot gamma sweep")
    p.add_argument("--csr", default=PATHS.csr_npz)
    p.add_argument("--node-attrs", default=PATHS.node_attributes)
    p.add_argument("--views-col", default="views")
    p.add_argument("--views-q-low", type=float, default=0.05)
    p.add_argument("--views-q-high", type=float, default=0.95)
    p.add_argument(
        "--hybrid-variant",
        default="centered",
        choices=["centered", "mult"],
        help=(
            "Hybrid variant. 'mult' uses (1+gamma*strength). 'centered' uses (1+gamma*(strength-mean_strength)) "
            "to keep average activation regime closer to baseline."
        ),
    )
    p.add_argument("--gamma-grid", default="0,0.25,0.5,1,2")
    p.add_argument("--n-pilot-nodes", type=int, default=200)
    p.add_argument("--n-runs", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-pct", type=float, default=0.10)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument(
        "--out-json",
        default="outputs/day1_benchmark/hybrid_sweep/hybrid_degree_views_pilot.json",
    )
    return p.parse_args()


def _parse_float_list(raw: str) -> list[float]:
    out: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            out.append(float(token))
    if not out:
        raise ValueError("gamma-grid cannot be empty")
    return out


def _load_views_raw_aligned(
    node_ids: np.ndarray,
    node_attrs_path: str | Path,
    views_col: str,
) -> np.ndarray:
    attrs_path = _resolve_io_path(node_attrs_path)
    if not attrs_path.exists():
        raise FileNotFoundError(f"Missing node attributes parquet: {attrs_path}")
    attrs = pd.read_parquet(attrs_path, columns=["node_id", views_col])
    require_columns(attrs, ["node_id", views_col], "node_attributes")

    attrs = attrs[["node_id", views_col]].copy()
    attrs["node_id"] = attrs["node_id"].astype(str)
    attrs[views_col] = pd.to_numeric(attrs[views_col], errors="coerce").fillna(0.0)

    series = attrs.set_index("node_id")[views_col]
    return (
        series.reindex(pd.Index(node_ids.astype(str)))
        .fillna(0.0)
        .astype(float)
        .to_numpy()
    )


def _summarize(arr: np.ndarray) -> dict[str, float]:
    arr = np.asarray(arr, dtype=float)
    q25, q50, q75 = np.quantile(arr, [0.25, 0.50, 0.75])
    return {
        "mean": float(arr.mean()),
        "median": float(q50),
        "iqr": float(q75 - q25),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _compute_means(
    rows: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    inv_degrees: np.ndarray,
    n_runs: int,
    seed: int,
    n_jobs: int,
    p_model: str,
    sender_strength: np.ndarray | None,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    def _worker(row: int) -> tuple[float, float]:
        return _simulate_ic_node_summary(
            source=int(row),
            indptr=indptr,
            indices=indices,
            inv_degrees=inv_degrees,
            n_runs=int(n_runs),
            worker_seed=int(seed) + int(row),
            p_model=str(p_model),
            sender_strength=sender_strength,
            hybrid_gamma=float(gamma),
        )

    t0 = time.perf_counter()
    stats = Parallel(n_jobs=int(n_jobs), backend="loky")(
        delayed(_worker)(int(row)) for row in rows
    )
    elapsed = float(time.perf_counter() - t0)

    means = np.array([m for m, _ in stats], dtype=float)
    stds = np.array([s for _, s in stats], dtype=float)
    return means, stds, elapsed


def main() -> None:
    args = parse_args()
    gamma_grid = _parse_float_list(args.gamma_grid)

    print(
        "[INFO] hybrid_pilot_start "
        f"(n_pilot_nodes={int(args.n_pilot_nodes)}, n_runs={int(args.n_runs)}, "
        f"gamma_grid={gamma_grid}, variant={str(args.hybrid_variant)}, n_jobs={int(args.n_jobs)})"
    )

    csr = load_csr_npz(args.csr)
    node_ids = csr["node_ids"].astype(str)
    degrees = csr["degrees"].astype(np.int64, copy=False)

    inv_deg = 1.0 / np.maximum(degrees.astype(np.float64, copy=False), 1.0)

    # Pilot seed set: degree-stratified sampling over full active nodes
    rows = _sample_labeled_indices(degrees=degrees, n_sample=int(args.n_pilot_nodes), seed=int(args.seed))

    print(f"[INFO] selected_pilot_rows n={len(rows)}")

    # Raw views (for correlation checks)
    views_raw = _load_views_raw_aligned(node_ids=node_ids, node_attrs_path=args.node_attrs, views_col=str(args.views_col))

    # Sender strength (for hybrid model)
    raw_strength, strength_meta = _compute_views_strength_aligned(
        node_ids=node_ids,
        node_attrs_path=args.node_attrs,
        views_col=str(args.views_col),
        q_low=float(args.views_q_low),
        q_high=float(args.views_q_high),
    )

    if str(args.hybrid_variant) == "centered":
        mean_strength = float(raw_strength.mean())
        sender_strength = raw_strength - mean_strength
        strength_meta = dict(strength_meta)
        strength_meta["mean_strength"] = float(mean_strength)
        hybrid_p_model = "hybrid_degree_views_centered"
    else:
        sender_strength = raw_strength
        hybrid_p_model = "hybrid_degree_views_mult"

    print(f"[INFO] views_strength_meta {strength_meta}")

    # Baseline: weighted cascade
    print("[INFO] baseline_weighted_cascade_start")
    base_means, base_stds, base_sec = _compute_means(
        rows=rows,
        indptr=csr["indptr"],
        indices=csr["indices"],
        inv_degrees=inv_deg,
        n_runs=int(args.n_runs),
        seed=int(args.seed),
        n_jobs=int(args.n_jobs),
        p_model="weighted_cascade",
        sender_strength=None,
        gamma=0.0,
    )
    print(f"[INFO] baseline_weighted_cascade_done runtime_sec={base_sec:.2f}")

    k = max(1, int(np.ceil(float(args.top_pct) * len(rows))))
    base_top = set(np.argsort(-base_means)[:k].tolist())

    out_rows: list[dict] = []

    for gamma in gamma_grid:
        print(f"[INFO] gamma_run_start gamma={float(gamma)}")
        if float(gamma) == 0.0:
            means, stds, elapsed = base_means, base_stds, 0.0
            p_model = "weighted_cascade"
        else:
            means, stds, elapsed = _compute_means(
                rows=rows,
                indptr=csr["indptr"],
                indices=csr["indices"],
                inv_degrees=inv_deg,
                n_runs=int(args.n_runs),
                seed=int(args.seed),
                n_jobs=int(args.n_jobs),
                p_model=hybrid_p_model,
                sender_strength=sender_strength,
                gamma=float(gamma),
            )
            p_model = hybrid_p_model

        print(f"[INFO] gamma_run_done gamma={float(gamma)} p_model={p_model} runtime_sec={elapsed:.2f}")

        rho_vs_base, _ = spearmanr(base_means, means)
        rho_vs_views, _ = spearmanr(views_raw[rows], means)
        rho_vs_degree, _ = spearmanr(degrees[rows].astype(float), means)

        top = set(np.argsort(-means)[:k].tolist())
        jaccard = float(len(base_top & top) / len(base_top | top)) if (base_top | top) else 1.0
        ndcg = float(ndcg_score(base_means.reshape(1, -1), means.reshape(1, -1), k=k))

        out_rows.append(
            {
                "p_model": p_model,
                "gamma": float(gamma),
                "summary_mean_reach": _summarize(means),
                "summary_std_reach": _summarize(stds),
                "rho_vs_baseline_weighted": float(rho_vs_base),
                "rho_vs_views_raw": float(rho_vs_views),
                "rho_vs_degree": float(rho_vs_degree),
                "topk": {"top_pct": float(args.top_pct), "k": int(k), "jaccard_vs_baseline": float(jaccard), "ndcg_vs_baseline": float(ndcg)},
                "runtime_sec": float(elapsed),
            }
        )

    payload = {
        "timestamp": now_iso(),
        "config": {
            "csr": str(args.csr),
            "node_attrs": str(args.node_attrs),
            "views_col": str(args.views_col),
            "views_q_low": float(args.views_q_low),
            "views_q_high": float(args.views_q_high),
            "strength_meta": strength_meta,
            "gamma_grid": [float(g) for g in gamma_grid],
            "n_pilot_nodes": int(args.n_pilot_nodes),
            "n_runs": int(args.n_runs),
            "seed": int(args.seed),
            "top_pct": float(args.top_pct),
            "n_jobs": int(args.n_jobs),
        },
        "baseline": {
            "p_model": "weighted_cascade",
            "runtime_sec": float(base_sec),
            "summary_mean_reach": _summarize(base_means),
            "summary_std_reach": _summarize(base_stds),
        },
        "results": out_rows,
    }

    out_path = _resolve_io_path(args.out_json)
    ensure_parent(out_path)
    Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] Wrote hybrid pilot sweep JSON: {out_path}")


if __name__ == "__main__":
    main()
