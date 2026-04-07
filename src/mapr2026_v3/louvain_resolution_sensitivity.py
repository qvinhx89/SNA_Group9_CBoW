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
    p.add_argument("--n-runs", type=int, default=10, help="Seed sweep runs per resolution")
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--nmi-threshold", type=float, default=0.85)
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


def main() -> None:
    args = parse_args()
    graph_path = _resolve_path(args.graph)
    out_path = _resolve_path(args.out)

    if args.n_runs < 1:
        raise ValueError("--n-runs must be >= 1")

    resolutions = _parse_resolutions(args.resolutions)
    community_louvain = _load_python_louvain()

    if not graph_path.exists():
        raise FileNotFoundError(f"Graph not found: {graph_path}")

    print(f"[INFO] Loading graph: {graph_path}")
    G = nx.read_edgelist(graph_path)
    nodes = list(G.nodes())
    n_nodes = len(nodes)
    print(f"[INFO] Graph loaded: {n_nodes} nodes, {G.number_of_edges()} edges")

    results: list[dict[str, Any]] = []

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

        results.append(
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

    res10 = None
    for r in results:
        if abs(r["resolution"] - 1.0) < 1e-9:
            res10 = r
            break

    if res10 is None:
        res10 = min(results, key=lambda r: abs(float(r["resolution"]) - 1.0))

    payload = {
        "timestamp": now_iso(),
        "graph_path": str(graph_path),
        "n_nodes": int(n_nodes),
        "n_edges": int(G.number_of_edges()),
        "resolutions_tested": [float(r) for r in resolutions],
        "n_runs_per_resolution": int(args.n_runs),
        "seed_start": int(args.seed_start),
        "nmi_threshold": float(args.nmi_threshold),
        "results": results,
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
