"""MAPR2026 v3 — Day-1 decisions.

Owner: Person 1 (IC core)

Inputs
------
- data/processed/graph_csr.npz

Outputs
-------
- outputs/day1_benchmark/ic_runtime_benchmark.json
- outputs/day1_benchmark/one_hop_correlation.json
- docs/day1_decisions.md

Critical Decision Gates:
------------------------
1. Runtime Benchmark → Determines N_seeds, N_runs, projected compute budget
2. One-hop Correlation → Determines GNN narrative branch (primary vs fallback)

Decision Rules (from v3 spec):
- Runtime < 4h: 5,000 seeds × 200 runs
- Runtime 4-8h: 3,000 seeds × 150 runs
- Runtime > 8h: 2,000 seeds × 100 runs + log limitation

- Spearman ρ < 0.8: GNN primary narrative
- ρ 0.8-0.9: Add 2-hop proxy; GNN may still win
- ρ > 0.9: RESTRUCTURE - proxies primary, GNN secondary
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from _shared import PATHS, ensure_dir, load_csr_npz, now_iso, write_json


def run_ic_csr(
    seed_node: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    degrees: np.ndarray,
    n_runs: int = 50,
    worker_seed: int | None = None,
) -> np.ndarray:
    """
    Monte Carlo IC using weighted cascade on CSR format.

    p(u,v) = 1/degree(v) — parameter-free weighted cascade.

    Returns:
        Array of reach sizes (length n_runs)
    """
    rng = np.random.default_rng(seed=worker_seed)
    sizes = []

    for _ in range(n_runs):
        activated = {seed_node}
        frontier = [seed_node]

        while frontier:
            next_frontier = []
            for node in frontier:
                start_idx = indptr[node]
                end_idx = indptr[node + 1]
                for idx in range(start_idx, end_idx):
                    nb = indices[idx]
                    if nb not in activated:
                        p = 1.0 / degrees[nb] if degrees[nb] > 0 else 0.0
                        if rng.random() < p:
                            activated.add(nb)
                            next_frontier.append(nb)
            frontier = next_frontier

        sizes.append(len(activated))

    return np.array(sizes, dtype=np.int32)


def benchmark_ic_runtime(
    indptr: np.ndarray,
    indices: np.ndarray,
    degrees: np.ndarray,
    n_test: int = 100,
    n_runs: int = 50,
) -> dict:
    """
    Benchmark IC runtime to project full compute budget.

    Returns:
        dict with per_sim_ms, projected_hours, and decision
    """
    n_nodes = len(degrees)
    test_seeds = random.sample(range(n_nodes), min(n_test, n_nodes))

    print(f"[Benchmark] Testing {len(test_seeds)} seeds × {n_runs} runs each...")
    t0 = time.time()

    for seed in test_seeds:
        run_ic_csr(seed, indptr, indices, degrees, n_runs=n_runs, worker_seed=42 + seed)

    elapsed = time.time() - t0
    per_sim_ms = elapsed / (len(test_seeds) * n_runs) * 1000

    # Decision table
    configs = [
        (5000, 200, "<4h"),
        (3000, 150, "4-8h"),
        (2000, 100, ">8h"),
    ]

    decisions = []
    for n_seeds, n_runs_config, budget_range in configs:
        projected_hours = per_sim_ms / 1000 * n_seeds * n_runs_config / 3600
        decisions.append({
            "n_seeds": n_seeds,
            "n_runs": n_runs_config,
            "budget_range": budget_range,
            "projected_hours": round(projected_hours, 2),
        })

    # Select configuration based on runtime
    if decisions[0]["projected_hours"] < 4:
        selected = decisions[0]
    elif decisions[1]["projected_hours"] < 8:
        selected = decisions[1]
    else:
        selected = decisions[2]

    return {
        "per_sim_ms": round(per_sim_ms, 3),
        "n_test_seeds": n_test,
        "n_test_runs": n_runs,
        "all_configs": decisions,
        "selected_config": selected,
    }


def compute_one_hop_spread(
    node: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    degrees: np.ndarray,
) -> float:
    """
    One-hop expected spread: Σ 1/degree(v) for v in N(u).

    This is an O(degree) analytical proxy for weighted cascade IC.
    """
    start_idx = indptr[node]
    end_idx = indptr[node + 1]
    total = 0.0

    for idx in range(start_idx, end_idx):
        nb = indices[idx]
        total += 1.0 / max(degrees[nb], 1)

    return total


def one_hop_correlation_check(
    indptr: np.ndarray,
    indices: np.ndarray,
    degrees: np.ndarray,
    n_pilot: int = 200,
    n_runs: int = 50,
) -> dict:
    """
    Critical decision gate: check Spearman ρ between one-hop proxy and IC.

    If ρ > 0.9: GNN narrative must be restructured (proxies become primary).
    """
    n_nodes = len(degrees)
    pilot_nodes = random.sample(range(n_nodes), min(n_pilot, n_nodes))

    print(f"[One-hop check] Running IC pilot on {len(pilot_nodes)} nodes × {n_runs} runs...")

    ic_scores = []
    one_hop_scores = []

    for node in pilot_nodes:
        # IC simulation
        sizes = run_ic_csr(node, indptr, indices, degrees, n_runs=n_runs, worker_seed=42 + node)
        ic_scores.append(float(sizes.mean()))

        # One-hop analytical proxy
        one_hop = compute_one_hop_spread(node, indptr, indices, degrees)
        one_hop_scores.append(one_hop)

    rho, p_value = spearmanr(ic_scores, one_hop_scores)

    # Decision logic
    if rho < 0.8:
        decision_branch = "gnn_primary"
        narrative = "GNN story viable; proceed as planned"
    elif 0.8 <= rho < 0.9:
        decision_branch = "gnn_with_2hop"
        narrative = "Add 2-hop proxy as stronger baseline; GNN may still win"
    else:
        decision_branch = "restructure_proxies_primary"
        narrative = "RESTRUCTURE: paper centers on divergence analysis + analytical proxies; GNN becomes secondary"

    return {
        "spearman_rho": round(rho, 3),
        "p_value": round(p_value, 6),
        "n_pilot_nodes": n_pilot,
        "n_pilot_runs": n_runs,
        "decision_branch": decision_branch,
        "narrative": narrative,
    }


def write_decision_doc(runtime_result: dict, onehop_result: dict, out_dir: Path) -> None:
    """Write docs/day1_decisions.md with locked decisions."""
    doc_path = Path("docs/day1_decisions.md")
    doc_path.parent.mkdir(parents=True, exist_ok=True)

    config = runtime_result["selected_config"]

    content = f"""# Day-1 Decisions (MAPR2026 v3)

**Date**: {now_iso()}

## 1. IC Runtime Benchmark

**Per-simulation time**: {runtime_result['per_sim_ms']:.2f} ms

**Projected total runtime** (selected config):
- N_seeds: {config['n_seeds']:,}
- N_runs: {config['n_runs']}
- Projected: {config['projected_hours']:.1f} hours

**Decision**: Use {config['n_seeds']:,} seeds × {config['n_runs']} runs

**All tested configurations**:
"""

    for cfg in runtime_result["all_configs"]:
        content += f"- {cfg['n_seeds']:,} seeds × {cfg['n_runs']} runs → {cfg['projected_hours']:.1f}h ({cfg['budget_range']})\n"

    content += f"""
## 2. One-Hop Baseline Reality Check

**Spearman ρ** (one-hop vs IC pilot): {onehop_result['spearman_rho']:.3f} (p={onehop_result['p_value']:.6f})

**Decision branch**: `{onehop_result['decision_branch']}`

**Narrative**: {onehop_result['narrative']}

## 3. Locked Parameters for Downstream Stages

```yaml
# IC Labels (Stage 4)
n_seeds: {config['n_seeds']}
n_runs: {config['n_runs']}
p_model: weighted_cascade  # p(u,v) = 1/degree(v)

# GNN Narrative Branch
narrative_branch: {onehop_result['decision_branch']}
primary_baseline: {"one_hop_spread" if onehop_result['decision_branch'] == 'restructure_proxies_primary' else "gnn_raw_attr"}
```

## 4. Action Items

"""

    if onehop_result['decision_branch'] == 'restructure_proxies_primary':
        content += "- [ ] **CRITICAL**: Restructure paper outline - proxies are primary, GNN is secondary\n"
        content += "- [ ] Update RQ3 framing: 'Can GNN provide marginal gains over analytical proxies?'\n"

    if config['projected_hours'] > 8:
        content += "- [ ] Log compute limitation in paper Section 5\n"

    content += f"- [ ] Proceed with IC labels using {config['n_seeds']:,} seeds × {config['n_runs']} runs\n"
    content += "- [ ] Update all downstream scripts with locked decisions\n"

    with doc_path.open("w", encoding="utf-8") as f:
        f.write(content)

    print(f"[OK] Wrote decision doc: {doc_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MAPR2026 v3 Day-1 benchmark + one-hop correlation")
    p.add_argument("--csr", default=PATHS.csr_npz)
    p.add_argument("--out-dir", default=PATHS.day1_dir)
    p.add_argument("--n-test", type=int, default=100, help="Number of test seeds for runtime benchmark")
    p.add_argument("--n-pilot", type=int, default=200, help="Number of pilot seeds for one-hop check")
    p.add_argument("--n-runs", type=int, default=50, help="Number of runs per seed")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


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

    # Real mode: require CSR and run benchmarks
    print("[Day-1 Benchmark] Loading CSR...")
    csr_data = load_csr_npz(Path(args.csr))
    indptr = csr_data["indptr"]
    indices = csr_data["indices"]
    degrees = csr_data["degrees"]

    print(f"[OK] Loaded graph: {len(degrees):,} nodes, {len(indices):,} edges")

    # Task 1: Runtime benchmark
    print("\n" + "="*60)
    print("TASK 1: IC RUNTIME BENCHMARK")
    print("="*60)
    runtime_result = benchmark_ic_runtime(indptr, indices, degrees, n_test=args.n_test, n_runs=args.n_runs)

    print(f"\n[Result] Per-simulation: {runtime_result['per_sim_ms']:.2f} ms")
    print(f"[Result] Selected config: {runtime_result['selected_config']['n_seeds']:,} seeds × {runtime_result['selected_config']['n_runs']} runs")
    print(f"[Result] Projected runtime: {runtime_result['selected_config']['projected_hours']:.1f} hours")

    # Task 2: One-hop correlation
    print("\n" + "="*60)
    print("TASK 2: ONE-HOP BASELINE REALITY CHECK")
    print("="*60)
    onehop_result = one_hop_correlation_check(indptr, indices, degrees, n_pilot=args.n_pilot, n_runs=args.n_runs)

    print(f"\n[Result] Spearman ρ: {onehop_result['spearman_rho']:.3f} (p={onehop_result['p_value']:.6f})")
    print(f"[Result] Decision: {onehop_result['decision_branch']}")
    print(f"[Result] Narrative: {onehop_result['narrative']}")

    # Save artifacts
    runtime_result["timestamp"] = now_iso()
    onehop_result["timestamp"] = now_iso()

    write_json(out_dir / "ic_runtime_benchmark.json", runtime_result)
    write_json(out_dir / "one_hop_correlation.json", onehop_result)

    # Write decision doc
    write_decision_doc(runtime_result, onehop_result, out_dir)

    print("\n" + "="*60)
    print("DAY-1 BENCHMARK COMPLETE")
    print("="*60)
    print(f"Artifacts saved to: {out_dir}")
    print("Review docs/day1_decisions.md before proceeding to Stage 4 (IC labels)")


if __name__ == "__main__":
    main()
