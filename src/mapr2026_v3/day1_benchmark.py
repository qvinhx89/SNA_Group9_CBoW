"""MAPR2026 v3 — Day-1 decisions.

Owner: Person 1 (IC core)

Inputs
------
- data/processed/graph_csr.npz

Outputs
-------
- outputs/day1_benchmark/ic_runtime_benchmark.json
- outputs/day1_benchmark/one_hop_correlation.json

This file is a scaffold. It supports --dry-run to emit placeholder JSON files
with the required keys (so teammates can integrate downstream code early).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

from _shared import PATHS, ensure_dir, load_csr_npz, now_iso, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MAPR2026 v3 Day-1 benchmark + one-hop correlation (scaffold)")
    p.add_argument("--csr", default=PATHS.csr_npz)
    p.add_argument("--out-dir", default=PATHS.day1_dir)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bench-nodes", type=int, default=100)
    p.add_argument("--bench-runs", type=int, default=50)
    p.add_argument("--pilot-nodes", type=int, default=200)
    p.add_argument("--pilot-runs", type=int, default=50)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument(
        "--target-n-sample",
        type=int,
        default=5000,
        help="Target number of labeled seed nodes for primary IC labeling (M0 default: 5000)",
    )
    return p.parse_args()


def _sample_indices_degree_stratified(degrees: np.ndarray, n_sample: int, seed: int) -> np.ndarray:
    n_nodes = len(degrees)
    if n_sample >= n_nodes:
        return np.arange(n_nodes, dtype=np.int64)

    df = pd.DataFrame({"idx": np.arange(n_nodes, dtype=np.int64), "degree": degrees.astype(float)})
    df["deg_q"] = pd.qcut(df["degree"], q=5, labels=False, duplicates="drop")
    stratify_labels = df["deg_q"].to_numpy()
    indices = df["idx"].to_numpy()

    try:
        _, sampled = train_test_split(
            indices,
            test_size=int(n_sample),
            random_state=seed,
            stratify=stratify_labels,
        )
        return sampled.astype(np.int64)
    except ValueError:
        # Fallback when a stratum is too small for strict stratification.
        rng = np.random.default_rng(seed)
        sampled = rng.choice(indices, size=int(n_sample), replace=False)
        return sampled.astype(np.int64)


def _simulate_ic_once(source: int, indptr: np.ndarray, indices: np.ndarray, degrees: np.ndarray, rng: np.random.Generator) -> int:
    activated = {int(source)}
    frontier = [int(source)]

    while frontier:
        next_frontier: list[int] = []
        for node in frontier:
            start_idx = int(indptr[node])
            end_idx = int(indptr[node + 1])
            for ptr in range(start_idx, end_idx):
                nb = int(indices[ptr])
                if nb in activated:
                    continue
                deg_nb = int(degrees[nb])
                if deg_nb <= 0:
                    continue
                if rng.random() < (1.0 / float(deg_nb)):
                    activated.add(nb)
                    next_frontier.append(nb)
        frontier = next_frontier

    return len(activated)


def _simulate_ic_runs_for_node(
    source: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    degrees: np.ndarray,
    n_runs: int,
    worker_seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(worker_seed)
    out = np.empty(n_runs, dtype=np.int32)
    for i in range(n_runs):
        out[i] = _simulate_ic_once(source, indptr, indices, degrees, rng)
    return out


def _one_hop_spread(node: int, indptr: np.ndarray, indices: np.ndarray, degrees: np.ndarray) -> float:
    start_idx = int(indptr[node])
    end_idx = int(indptr[node + 1])
    nbs = indices[start_idx:end_idx]
    if len(nbs) == 0:
        return 0.0
    deg_nbs = degrees[nbs]
    deg_nbs = deg_nbs[deg_nbs > 0]
    if len(deg_nbs) == 0:
        return 0.0
    return float(np.sum(1.0 / deg_nbs.astype(float)))


def _projected_hours(per_sim_ms: float, n_seeds: int, n_runs: int) -> float:
    return float((per_sim_ms * n_seeds * n_runs) / 1000.0 / 3600.0)


def _runtime_decision(projected_hours_default: float) -> tuple[dict[str, int | str], str]:
    if projected_hours_default < 4.0:
        return (
            {"n_seeds": 5000, "n_runs": 200, "rule": "projected_runtime < 4h"},
            "proceed_as_planned",
        )
    if projected_hours_default <= 8.0:
        return (
            {"n_seeds": 3000, "n_runs": 150, "rule": "4h <= projected_runtime <= 8h"},
            "reduce_compute_with_limitation",
        )
    return (
        {"n_seeds": 2000, "n_runs": 100, "rule": "projected_runtime > 8h"},
        "minimum_budget_with_limitation",
    )


def _rho_decision_branch(rho: float) -> str:
    if rho < 0.8:
        return "viable_gnn"
    if rho <= 0.9:
        return "two_hop_primary"
    return "restructure"


def _run_runtime_benchmark(
    indptr: np.ndarray,
    indices: np.ndarray,
    degrees: np.ndarray,
    seed: int,
    bench_nodes: int,
    bench_runs: int,
    n_jobs: int,
) -> tuple[float, np.ndarray]:
    sampled = _sample_indices_degree_stratified(degrees=degrees, n_sample=bench_nodes, seed=seed)

    def _worker(node: int) -> int:
        _ = _simulate_ic_runs_for_node(
            source=int(node),
            indptr=indptr,
            indices=indices,
            degrees=degrees,
            n_runs=bench_runs,
            worker_seed=seed + int(node),
        )
        return int(node)

    t0 = time.time()
    _ = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_worker)(int(node)) for node in sampled)
    elapsed_sec = time.time() - t0

    per_sim_ms = float((elapsed_sec / (len(sampled) * bench_runs)) * 1000.0)
    return per_sim_ms, sampled


def _run_one_hop_correlation(
    indptr: np.ndarray,
    indices: np.ndarray,
    degrees: np.ndarray,
    seed: int,
    pilot_nodes: int,
    pilot_runs: int,
    n_jobs: int,
) -> tuple[float, float | None, int]:
    sampled = _sample_indices_degree_stratified(degrees=degrees, n_sample=pilot_nodes, seed=seed)

    one_hop_scores = np.array(
        [_one_hop_spread(int(node), indptr=indptr, indices=indices, degrees=degrees) for node in sampled],
        dtype=float,
    )

    def _ic_mean(node: int) -> float:
        runs = _simulate_ic_runs_for_node(
            source=int(node),
            indptr=indptr,
            indices=indices,
            degrees=degrees,
            n_runs=pilot_runs,
            worker_seed=seed + int(node),
        )
        return float(runs.mean())

    ic_means = np.array(
        Parallel(n_jobs=n_jobs, backend="loky")(delayed(_ic_mean)(int(node)) for node in sampled),
        dtype=float,
    )

    rho, p_value = spearmanr(one_hop_scores, ic_means)
    if np.isnan(rho):
        raise ValueError("Spearman rho is NaN. Check pilot scores for constant values.")
    if np.isnan(p_value):
        p_out: float | None = None
    else:
        p_out = float(p_value)

    return float(rho), p_out, int(len(sampled))


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)

    if args.dry_run:
        write_json(
            out_dir / "ic_runtime_benchmark.json",
            {
                "timestamp": now_iso(),
                "dry_run": True,
                "per_sim_ms": None,
                "projected_total_hours": None,
                "decision": {"n_seeds": None, "n_runs": None},
                "note": "Scaffold placeholder. Implement benchmark_ic_runtime per MAPR2026 v3.",
            },
        )
        write_json(
            out_dir / "one_hop_correlation.json",
            {
                "timestamp": now_iso(),
                "dry_run": True,
                "spearman_rho": None,
                "p_value": None,
                "decision_branch": None,
                "note": "Scaffold placeholder. Implement IC pilot + one-hop baseline correlation.",
            },
        )
        print(f"[OK] Wrote dry-run placeholders to: {out_dir}")
        return

    # Real mode.
    csr = load_csr_npz(Path(args.csr))
    indptr = csr["indptr"]
    indices = csr["indices"]
    degrees = csr["degrees"]

    per_sim_ms, sampled_bench_nodes = _run_runtime_benchmark(
        indptr=indptr,
        indices=indices,
        degrees=degrees,
        seed=args.seed,
        bench_nodes=args.bench_nodes,
        bench_runs=args.bench_runs,
        n_jobs=args.n_jobs,
    )

    projected_default = _projected_hours(
        per_sim_ms=per_sim_ms,
        n_seeds=args.target_n_sample,
        n_runs=200,
    )
    runtime_decision, runtime_action = _runtime_decision(projected_default)

    runtime_options = [
        {
            "n_seeds": 5000,
            "n_runs": 200,
            "projected_hours": _projected_hours(per_sim_ms, 5000, 200),
        },
        {
            "n_seeds": 3000,
            "n_runs": 150,
            "projected_hours": _projected_hours(per_sim_ms, 3000, 150),
        },
        {
            "n_seeds": 2000,
            "n_runs": 100,
            "projected_hours": _projected_hours(per_sim_ms, 2000, 100),
        },
    ]

    write_json(
        out_dir / "ic_runtime_benchmark.json",
        {
            "timestamp": now_iso(),
            "dry_run": False,
            "per_sim_ms": per_sim_ms,
            "projected_total_hours": projected_default,
            "decision": runtime_decision,
            "decision_action": runtime_action,
            "benchmark_config": {
                "bench_nodes": int(len(sampled_bench_nodes)),
                "bench_runs": int(args.bench_runs),
                "target_n_sample": int(args.target_n_sample),
                "seed": int(args.seed),
                "n_jobs": int(args.n_jobs),
                "p_model": "weighted_cascade",
            },
            "projected_hours_by_option": runtime_options,
        },
    )

    rho, p_value, n_valid = _run_one_hop_correlation(
        indptr=indptr,
        indices=indices,
        degrees=degrees,
        seed=args.seed,
        pilot_nodes=args.pilot_nodes,
        pilot_runs=args.pilot_runs,
        n_jobs=args.n_jobs,
    )
    decision_branch = _rho_decision_branch(rho)

    write_json(
        out_dir / "one_hop_correlation.json",
        {
            "timestamp": now_iso(),
            "dry_run": False,
            "spearman_rho": rho,
            "p_value": p_value,
            "decision_branch": decision_branch,
            "pilot_config": {
                "pilot_nodes": int(args.pilot_nodes),
                "pilot_runs": int(args.pilot_runs),
                "n_valid_nodes": int(n_valid),
                "seed": int(args.seed),
                "sampling": "degree_quintile_stratified",
                "p_model": "weighted_cascade",
            },
            "decision_gate": {
                "lt_0_8": "viable_gnn",
                "between_0_8_0_9": "two_hop_primary",
                "gt_0_9": "restructure",
            },
        },
    )

    print(
        "[OK] Day-1 real mode completed. "
        f"per_sim_ms={per_sim_ms:.4f}, projected_total_hours={projected_default:.3f}, "
        f"rho={rho:.4f}, branch={decision_branch}"
    )


if __name__ == "__main__":
    main()
