from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
MAPR_V3_ROOT = REPO_ROOT / "src" / "mapr2026_v3"
if str(MAPR_V3_ROOT) not in sys.path:
    sys.path.insert(0, str(MAPR_V3_ROOT))

from src.graph.community import compute_cross_community_edge_fraction, run_louvain_single
from src.mapr2026_v3.typology_ic_views import _build_structural_frame, _compute_structural_profiling


def _top3_pct(partition: dict[str, int]) -> float:
    counts: dict[int, int] = {}
    for cid in partition.values():
        counts[cid] = counts.get(cid, 0) + 1
    sizes = sorted(counts.values(), reverse=True)
    n_nodes = sum(sizes)
    if n_nodes == 0:
        return 0.0
    return (sum(sizes[:3]) / n_nodes) * 100.0


def _build_community_features(
    graph_path: Path,
    resolution: float,
    seed: int,
    out_path: Path,
) -> dict:
    import networkx as nx

    g = nx.read_edgelist(str(graph_path))
    nodes = list(g.nodes())

    partition, modularity = run_louvain_single(g, seed=seed, resolution=resolution)
    comm_ids = np.array([int(partition[n]) for n in nodes], dtype=np.int64)
    cross = compute_cross_community_edge_fraction(g, partition, nodes)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "node_id": nodes,
            "community_id": comm_ids,
            "cross_community_edge_fraction": cross,
        }
    ).to_parquet(out_path, index=False)

    return {
        "resolution": float(resolution),
        "seed": int(seed),
        "n_nodes": int(len(nodes)),
        "n_communities": int(len(set(partition.values()))),
        "best_modularity_single_seed": float(modularity),
        "pct_nodes_top3_communities": float(_top3_pct(partition)),
        "community_features_path": str(out_path),
    }


def _extract_cross_comm_row(df: pd.DataFrame) -> dict:
    row = df[df["feature"] == "cross_community_edge_fraction"]
    if row.empty:
        raise ValueError("cross_community_edge_fraction row not found in structural profiling output")
    r = row.iloc[0]
    return {
        "group_hidden_mean": float(r["group_hidden_mean"]),
        "group_overrated_mean": float(r["group_overrated_mean"]),
        "delta_hidden_minus_overrated": float(r["group_hidden_mean"] - r["group_overrated_mean"]),
        "cliffs_delta": float(r["cliffs_delta"]),
        "p_corrected": float(r["p_corrected"]),
        "significant": bool(r["significant"]),
    }


def main() -> None:
    repo = REPO_ROOT

    graph_path = repo / "data/processed/graph_active.edgelist"
    typology_path = repo / "data/processed/typology_labels_ic_views.parquet"
    node_attrs_path = repo / "data/processed/node_attributes.parquet"
    centrality_path = repo / "data/processed/centrality_table.parquet"
    kshell_path = repo / "data/processed/kshell_table.parquet"

    out_results = repo / "outputs/mapr2026_v3_results"
    out_data = repo / "data/processed"

    # Use best seeds observed in the latest sweep artifact.
    settings = [
        ("res1_1", 1.1, 6),
        ("res1_6", 1.6, 2),
    ]

    if not typology_path.exists():
        raise FileNotFoundError(f"Missing typology labels: {typology_path}")

    typology_df = pd.read_parquet(typology_path)

    profiling_by_tag: dict[str, dict] = {}
    partition_stats: dict[str, dict] = {}

    for tag, resolution, seed in settings:
        comm_path = out_data / f"community_features_{tag}.parquet"
        part_info = _build_community_features(graph_path, resolution, seed, comm_path)
        partition_stats[tag] = part_info

        frame = _build_structural_frame(
            typology_df=typology_df,
            node_attrs_path=node_attrs_path,
            centrality_path=centrality_path,
            kshell_path=kshell_path,
            community_path=comm_path,
        )
        profiling = _compute_structural_profiling(frame, delta_threshold=0.20)

        profiling_csv = out_results / f"structural_profiling_{tag}.csv"
        profiling.to_csv(profiling_csv, index=False)

        profiling_by_tag[tag] = {
            "resolution": float(resolution),
            "seed": int(seed),
            "structural_profiling_csv": str(profiling_csv),
            "cross_community_edge_fraction": _extract_cross_comm_row(profiling),
        }

    c11 = profiling_by_tag["res1_1"]["cross_community_edge_fraction"]
    c16 = profiling_by_tag["res1_6"]["cross_community_edge_fraction"]

    same_direction = (c11["delta_hidden_minus_overrated"] > 0 and c16["delta_hidden_minus_overrated"] > 0) or (
        c11["delta_hidden_minus_overrated"] < 0 and c16["delta_hidden_minus_overrated"] < 0
    )
    meaningful_both = abs(c11["cliffs_delta"]) >= 0.20 and abs(c16["cliffs_delta"]) >= 0.20
    significance_not_flipped = bool(c11["significant"] == c16["significant"])

    comparison = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "summary": {
            "primary_candidate": "res1_1",
            "sensitivity_candidate": "res1_6",
            "same_direction_effect": bool(same_direction),
            "effect_size_meaningful_both": bool(meaningful_both),
            "significance_not_flipped": bool(significance_not_flipped),
        },
        "partition_stats": partition_stats,
        "task5_downstream": profiling_by_tag,
        "decision_rule_notes": {
            "same_direction_effect": "Hidden minus Overrated mean of cross_community_edge_fraction keeps same sign",
            "effect_size_meaningful_both": "|Cliff's delta| >= 0.20 in both resolutions",
            "significance_not_flipped": "significant boolean for cross_community_edge_fraction is identical",
        },
    }

    out_json = out_results / "task5_resolution_sensitivity_comparison.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print(f"[OK] Wrote comparison summary: {out_json}")
    print(f"[OK] Wrote profiling CSVs: {out_results / 'structural_profiling_res1_1.csv'} and {out_results / 'structural_profiling_res1_6.csv'}")


if __name__ == "__main__":
    main()
