"""MAPR2026 v3 — Louvain resolution sensitivity sweep (B9 soft branch).

Produces:
- outputs/mapr2026_v3_results/louvain_resolution_sensitivity.json

Default sweep:
- resolution in {0.5, 1.0, 2.0}
- n_runs per resolution = 10 (seed sweep)

This artifact supports the B9 decision rule in team plan/checklist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from _shared import PATHS, ensure_parent, now_iso, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Louvain resolution sensitivity sweep")
    p.add_argument("--graph", default=PATHS.graph_edgelist, help="Path to graph_active.edgelist")
    p.add_argument(
        "--resolutions",
        default="0.5,1.0,2.0",
        help="Comma-separated resolution values, e.g. 0.5,1.0,2.0",
    )
    p.add_argument(
        "--include-extra-mid-resolutions",
        action="store_true",
        help="Append extra mid-range resolutions: 1.4, 1.55, 1.6",
    )
    p.add_argument("--n-runs", type=int, default=10, help="Seed sweep runs per resolution")
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--nmi-threshold", type=float, default=0.70)
    p.add_argument(
        "--merge-existing",
        action="store_true",
        help="Merge newly computed resolutions with existing output file results",
    )
    p.add_argument("--accept-mean-nmi", type=float, default=0.67)
    p.add_argument("--target-mean-nmi", type=float, default=0.70)
    p.add_argument("--accept-min-nmi", type=float, default=0.60)
    p.add_argument("--target-min-nmi", type=float, default=0.65)
    p.add_argument("--modularity-retain-ratio", type=float, default=0.95)
    p.add_argument("--target-comm-low", type=int, default=30)
    p.add_argument("--target-comm-high", type=int, default=80)
    p.add_argument("--target-top3-max", type=float, default=50.0)
    p.add_argument("--target-singleton-max", type=float, default=5.0)
    p.add_argument("--weight-stability", type=float, default=0.35)
    p.add_argument("--weight-modularity", type=float, default=0.25)
    p.add_argument("--weight-structure", type=float, default=0.20)
    p.add_argument("--weight-downstream", type=float, default=0.20)
    p.add_argument("--out", default=PATHS.louvain_resolution_sensitivity)
    return p.parse_args()


def _resolve_path(path_like: str | Path) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p

    if p.exists() or p.parent.exists():
        return p

    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / p


def _load_python_louvain() -> Any:
    try:
        import community as community_louvain  # python-louvain
    except Exception as exc:
        raise ImportError(
            "python-louvain is required. Install with: pip install python-louvain"
        ) from exc

    if not hasattr(community_louvain, "best_partition") or not hasattr(community_louvain, "modularity"):
        raise ImportError("Loaded 'community' module is not python-louvain.")
    return community_louvain


def _nmi_pairwise(partitions: list[dict[str, int]], nodes: list[str]) -> tuple[float, float, float]:
    try:
        from sklearn.metrics import normalized_mutual_info_score
    except Exception as exc:
        raise ImportError("scikit-learn is required for NMI. Install with: pip install scikit-learn") from exc

    if len(partitions) < 2:
        return 1.0, 0.0, 1.0

    vals: list[float] = []
    for i in range(len(partitions)):
        for j in range(i + 1, len(partitions)):
            labels_i = [partitions[i].get(n, -1) for n in nodes]
            labels_j = [partitions[j].get(n, -1) for n in nodes]
            vals.append(float(normalized_mutual_info_score(labels_i, labels_j)))

    arr = np.asarray(vals, dtype=float)
    return float(arr.mean()), float(arr.std()), float(arr.min())


def _community_size_stats(partition: dict[str, int], n_nodes: int) -> dict[str, float | int]:
    counts: dict[int, int] = {}
    for cid in partition.values():
        counts[cid] = counts.get(cid, 0) + 1

    sizes = sorted(counts.values(), reverse=True)
    top3 = sum(sizes[:3]) if sizes else 0
    singleton_count = sum(1 for s in sizes if s == 1)

    return {
        "n_communities": int(len(sizes)),
        "pct_nodes_top3_communities": float((top3 / n_nodes) * 100.0 if n_nodes else 0.0),
        "singleton_count": int(singleton_count),
        "singleton_pct": float((singleton_count / n_nodes) * 100.0 if n_nodes else 0.0),
    }


def _parse_resolutions(raw: str) -> list[float]:
    vals = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    if not vals:
        raise ValueError("No valid resolutions provided.")
    return vals


def _dedupe_sorted_resolutions(values: list[float]) -> list[float]:
    uniq = {round(float(v), 10): float(v) for v in values}
    out = list(uniq.values())
    out.sort()
    return out


def _load_existing_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    results = payload.get("results", [])
    if not isinstance(results, list):
        return []
    return [r for r in results if isinstance(r, dict)]


def _merge_results(existing: list[dict[str, Any]], new: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[float, dict[str, Any]] = {}
    for row in existing:
        try:
            key = round(float(row["resolution"]), 10)
            merged[key] = row
        except Exception:
            continue

    for row in new:
        key = round(float(row["resolution"]), 10)
        merged[key] = row

    out = list(merged.values())
    out.sort(key=lambda x: float(x["resolution"]))
    return out


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _lin_score(value: float, low: float, high: float) -> float:
    if high <= low:
        return 1.0 if value >= high else 0.0
    return _clamp01((float(value) - float(low)) / (float(high) - float(low)))


def _comm_count_score(n_communities: int, low: int, high: int) -> float:
    n = int(n_communities)
    if low <= n <= high:
        return 1.0

    if n < low:
        return _clamp01(float(n) / float(max(low, 1)))

    # n > high
    span = float(max(high, 1))
    overflow = float(n - high)
    return _clamp01(1.0 - (overflow / span))


def _build_evaluation_framework(results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    if not results:
        return {
            "criteria": {},
            "ranked_resolutions": [],
            "recommended_resolution": None,
            "notes": ["No results available for evaluation."],
        }

    best_modularity_mean = max(float(r["modularity_mean"]) for r in results)
    modularity_floor = float(best_modularity_mean) * float(args.modularity_retain_ratio)

    w_stability = max(0.0, float(args.weight_stability))
    w_modularity = max(0.0, float(args.weight_modularity))
    w_structure = max(0.0, float(args.weight_structure))
    w_downstream = max(0.0, float(args.weight_downstream))

    rows: list[dict[str, Any]] = []
    for r in results:
        mean_nmi = float(r["mean_nmi_louvain"])
        min_nmi = float(r["min_nmi_louvain"])
        std_nmi = float(r["std_nmi_louvain"])
        mod_mean = float(r["modularity_mean"])
        n_communities = int(r["n_communities"])
        pct_top3 = float(r["pct_nodes_top3_communities"])
        singleton_pct = float(r["singleton_pct"])

        mean_nmi_accept = mean_nmi >= float(args.accept_mean_nmi)
        min_nmi_accept = min_nmi >= float(args.accept_min_nmi)
        modularity_accept = mod_mean >= modularity_floor
        hard_gate_pass = bool(mean_nmi_accept and min_nmi_accept and modularity_accept)

        stability_score = (
            0.55 * _lin_score(mean_nmi, float(args.accept_mean_nmi), float(args.target_mean_nmi))
            + 0.30 * _lin_score(min_nmi, float(args.accept_min_nmi), float(args.target_min_nmi))
            + 0.15 * (1.0 - _clamp01(std_nmi / 0.10))
        )

        modularity_score = _lin_score(mod_mean, modularity_floor, best_modularity_mean)
        structure_score = (
            0.45 * _comm_count_score(n_communities, int(args.target_comm_low), int(args.target_comm_high))
            + 0.40 * (1.0 - _lin_score(pct_top3, float(args.target_top3_max), 100.0))
            + 0.15 * (1.0 - _lin_score(singleton_pct, float(args.target_singleton_max), 100.0))
        )

        # Downstream seed-level robustness is not computed in this script.
        downstream_score = 0.50
        downstream_available = False

        weighted_sum = 0.0
        weight_sum = 0.0
        for score, weight, available in [
            (stability_score, w_stability, True),
            (modularity_score, w_modularity, True),
            (structure_score, w_structure, True),
            (downstream_score, w_downstream, downstream_available),
        ]:
            if available and weight > 0.0:
                weighted_sum += float(score) * float(weight)
                weight_sum += float(weight)

        composite_score = float(weighted_sum / weight_sum) if weight_sum > 0 else 0.0

        rows.append(
            {
                "resolution": float(r["resolution"]),
                "hard_gate_pass": hard_gate_pass,
                "gate_checks": {
                    "mean_nmi_accept": bool(mean_nmi_accept),
                    "min_nmi_accept": bool(min_nmi_accept),
                    "modularity_accept": bool(modularity_accept),
                },
                "component_scores": {
                    "stability": float(stability_score),
                    "modularity": float(modularity_score),
                    "structure": float(structure_score),
                    "downstream": float(downstream_score),
                },
                "composite_score": float(composite_score),
                "summary_metrics": {
                    "mean_nmi_louvain": mean_nmi,
                    "min_nmi_louvain": min_nmi,
                    "modularity_mean": mod_mean,
                    "n_communities": n_communities,
                    "pct_nodes_top3_communities": pct_top3,
                    "singleton_pct": singleton_pct,
                },
            }
        )

    rows.sort(
        key=lambda x: (
            int(x["hard_gate_pass"]),
            float(x["composite_score"]),
            float(x["summary_metrics"]["mean_nmi_louvain"]),
        ),
        reverse=True,
    )

    run_counts = sorted({int(r.get("n_runs", -1)) for r in results if "n_runs" in r})
    mixed_run_counts = len(run_counts) > 1

    recommended = rows[0]["resolution"] if rows else None
    return {
        "criteria": {
            "hard_gate": {
                "mean_nmi_min": float(args.accept_mean_nmi),
                "min_nmi_min": float(args.accept_min_nmi),
                "modularity_mean_min": float(modularity_floor),
                "modularity_retain_ratio": float(args.modularity_retain_ratio),
            },
            "structure_targets": {
                "n_communities_low": int(args.target_comm_low),
                "n_communities_high": int(args.target_comm_high),
                "pct_top3_nodes_max": float(args.target_top3_max),
                "singleton_pct_max": float(args.target_singleton_max),
            },
            "weights_requested": {
                "stability": w_stability,
                "modularity": w_modularity,
                "structure": w_structure,
                "downstream": w_downstream,
            },
            "weights_applied": {
                "stability": w_stability,
                "modularity": w_modularity,
                "structure": w_structure,
                "downstream": 0.0,
            },
        },
        "ranked_resolutions": rows,
        "recommended_resolution": recommended,
        "notes": [
            "Downstream seed-level robustness is not computed in this script; downstream weight is excluded from applied score.",
            "Use this ranking jointly with Task 5 seed-level reruns before changing primary resolution.",
            (
                "Results include mixed n_runs across resolutions; compare with caution."
                if mixed_run_counts
                else "All compared resolutions use the same n_runs."
            ),
        ],
    }


def main() -> None:
    args = parse_args()
    graph_path = _resolve_path(args.graph)
    out_path = _resolve_path(args.out)

    if args.n_runs < 1:
        raise ValueError("--n-runs must be >= 1")

    resolutions = _parse_resolutions(args.resolutions)
    auto_added_resolutions: list[float] = []
    if args.include_extra_mid_resolutions:
        auto_added_resolutions = [1.4, 1.55, 1.6]
        resolutions = _dedupe_sorted_resolutions(resolutions + auto_added_resolutions)
    community_louvain = _load_python_louvain()

    if not graph_path.exists():
        raise FileNotFoundError(f"Graph not found: {graph_path}")

    print(f"[INFO] Loading graph: {graph_path}")
    G = nx.read_edgelist(graph_path)
    nodes = list(G.nodes())
    n_nodes = len(nodes)
    print(f"[INFO] Graph loaded: {n_nodes} nodes, {G.number_of_edges()} edges")

    new_results: list[dict[str, Any]] = []

    for resolution in resolutions:
        partitions: list[dict[str, int]] = []
        modularities: list[float] = []
        run_stats: list[dict[str, Any]] = []

        for i in range(args.n_runs):
            seed = args.seed_start + i
            partition = community_louvain.best_partition(G, random_state=seed, resolution=resolution)
            q = float(community_louvain.modularity(partition, G))
            stats = _community_size_stats(partition, n_nodes)

            partitions.append(partition)
            modularities.append(q)
            run_stats.append({"seed": int(seed), "modularity": q, **stats})

        mod_arr = np.asarray(modularities, dtype=float)
        best_idx = int(np.argmax(mod_arr))
        best = run_stats[best_idx]

        mean_nmi, std_nmi, min_nmi = _nmi_pairwise(partitions, nodes)

        over_merge = (best["n_communities"] < 20) or (best["pct_nodes_top3_communities"] > 50.0)
        over_split = (best["n_communities"] > 200) and (best["singleton_pct"] > 5.0)

        new_results.append(
            {
                "resolution": float(resolution),
                "n_runs": int(args.n_runs),
                "best_seed": int(best["seed"]),
                "best_modularity": float(best["modularity"]),
                "modularity_mean": float(mod_arr.mean()),
                "modularity_std": float(mod_arr.std()),
                "n_communities": int(best["n_communities"]),
                "pct_nodes_top3_communities": float(best["pct_nodes_top3_communities"]),
                "singleton_count": int(best["singleton_count"]),
                "singleton_pct": float(best["singleton_pct"]),
                "mean_nmi_louvain": float(mean_nmi),
                "std_nmi_louvain": float(std_nmi),
                "min_nmi_louvain": float(min_nmi),
                "stability_warning": bool(mean_nmi < args.nmi_threshold),
                "over_merge_warning": bool(over_merge),
                "over_split_warning": bool(over_split),
            }
        )

        print(
            "[INFO] "
            f"resolution={resolution:.2f} "
            f"Q_best={best['modularity']:.4f} "
            f"n_comm={best['n_communities']} "
            f"top3={best['pct_nodes_top3_communities']:.2f}% "
            f"mean_nmi={mean_nmi:.4f}"
        )

    existing_results: list[dict[str, Any]] = []
    if args.merge_existing:
        existing_results = _load_existing_results(out_path)

    results = _merge_results(existing_results, new_results) if args.merge_existing else new_results

    # Harmonize stability warning with current nmi_threshold for all rows.
    for r in results:
        r["stability_warning"] = bool(float(r["mean_nmi_louvain"]) < float(args.nmi_threshold))

    res10 = None
    for r in results:
        if abs(r["resolution"] - 1.0) < 1e-9:
            res10 = r
            break

    if res10 is None:
        res10 = min(results, key=lambda r: abs(float(r["resolution"]) - 1.0))

    evaluation_framework = _build_evaluation_framework(results, args)

    payload = {
        "timestamp": now_iso(),
        "graph_path": str(graph_path),
        "n_nodes": int(n_nodes),
        "n_edges": int(G.number_of_edges()),
        "resolutions_tested": [float(r["resolution"]) for r in results],
        "resolutions_requested_this_run": [float(r) for r in resolutions],
        "resolutions_auto_added_this_run": [float(r) for r in auto_added_resolutions],
        "n_runs_per_resolution": int(args.n_runs),
        "n_runs_by_resolution": {
            str(float(r["resolution"])): int(r.get("n_runs", args.n_runs)) for r in results
        },
        "seed_start": int(args.seed_start),
        "nmi_threshold": float(args.nmi_threshold),
        "merge_existing": bool(args.merge_existing),
        "results": results,
        "evaluation_framework": evaluation_framework,
        "resolution_1_0_decision": {
            "resolution": float(res10["resolution"]),
            "n_communities": int(res10["n_communities"]),
            "pct_nodes_top3_communities": float(res10["pct_nodes_top3_communities"]),
            "singleton_pct": float(res10["singleton_pct"]),
            "over_merge_warning": bool(res10["over_merge_warning"]),
            "over_split_warning": bool(res10["over_split_warning"]),
            "target_practical_range": "30-80 communities",
            "rule_notes": {
                "over_merge": "n_communities < 20 or pct_nodes_top3_communities > 50%",
                "over_split": "n_communities > 200 and singleton_pct > 5%",
            },
        },
    }

    ensure_parent(out_path)
    write_json(out_path, payload)
    print(f"[OK] Wrote B9 sensitivity artifact: {out_path}")


if __name__ == "__main__":
    main()
