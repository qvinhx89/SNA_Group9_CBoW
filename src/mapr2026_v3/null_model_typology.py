"""MAPR2026 v3 — Task 7: configuration null model typology comparison.

Owner: Person 2

Inputs
------
- data/processed/typology_labels_ic_views.parquet
- data/processed/graph_csr.npz

Outputs
-------
- outputs/mapr2026_v3_results/null_model_typology_summary.json

Plan-conformant behavior
------------------------
- Sample 500 labeled nodes (seed=42 by default).
- Build real 500-node subgraph from CSR.
- Generate 3 configuration-model realizations (seeds: 0, 100, 200).
- Run IC weighted-cascade on each null graph (100 runs/node).
- Compare rank correlation (rho) and hidden-node betweenness against real subgraph.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

from _shared import PATHS, ensure_dir, load_csr_npz, now_iso, require_columns, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MAPR2026 v3 Task 7 null model typology")
    p.add_argument("--typology", default=PATHS.typology)
    p.add_argument("--csr", default=PATHS.csr_npz)
    p.add_argument("--out-dir", default=PATHS.results_dir)
    p.add_argument("--n-sample", type=int, default=500)
    p.add_argument("--n-realizations", type=int, default=3)
    p.add_argument("--n-runs-per-node", type=int, default=100)
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--top-pct", type=float, default=0.10)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _assign_typology(ic_scores: np.ndarray, views: np.ndarray, top_pct: float) -> np.ndarray:
    if not (0.0 < float(top_pct) < 1.0):
        raise ValueError("top_pct must be in (0,1)")

    ic_thresh = float(np.quantile(ic_scores, 1.0 - float(top_pct)))
    views_thresh = float(np.quantile(views, 1.0 - float(top_pct)))

    ic_high = ic_scores >= ic_thresh
    views_high = views >= views_thresh

    labels = np.full(shape=ic_scores.shape[0], fill_value="Non", dtype=object)
    labels[np.logical_and(ic_high, views_high)] = "True"
    labels[np.logical_and(ic_high, np.logical_not(views_high))] = "Hidden"
    labels[np.logical_and(np.logical_not(ic_high), views_high)] = "Overrated"
    return labels


def _build_real_subgraph_from_csr(sample_node_ids: np.ndarray, csr_path: Path) -> nx.Graph:
    csr = load_csr_npz(csr_path)
    indptr = csr["indptr"]
    indices = csr["indices"]
    csr_node_ids = csr["node_ids"]

    node_to_row = {str(node_id): int(i) for i, node_id in enumerate(csr_node_ids.tolist())}
    missing = [str(n) for n in sample_node_ids.tolist() if str(n) not in node_to_row]
    if missing:
        raise ValueError(f"Sample contains node_ids missing from CSR mapping: n_missing={len(missing)}")

    sampled_rows = [node_to_row[str(n)] for n in sample_node_ids.tolist()]
    row_to_local = {int(row): int(i) for i, row in enumerate(sampled_rows)}
    sampled_row_set = set(sampled_rows)

    g = nx.Graph()
    g.add_nodes_from(range(len(sampled_rows)))

    for row in sampled_rows:
        u_local = row_to_local[int(row)]
        start = int(indptr[int(row)])
        end = int(indptr[int(row) + 1])
        for nb_row_raw in indices[start:end]:
            nb_row = int(nb_row_raw)
            if nb_row not in sampled_row_set:
                continue
            if int(row) < nb_row:
                v_local = row_to_local[nb_row]
                g.add_edge(u_local, v_local)

    return g


def _graph_to_neighbors_and_inv_degree(g: nx.Graph) -> tuple[list[np.ndarray], np.ndarray]:
    n = g.number_of_nodes()
    neighbors: list[np.ndarray] = []
    inv_degree = np.zeros(n, dtype=float)

    for node in range(n):
        neigh = np.array(sorted(g.neighbors(node)), dtype=np.int64)
        neighbors.append(neigh)
        deg = len(neigh)
        if deg > 0:
            inv_degree[node] = 1.0 / float(deg)

    return neighbors, inv_degree


def _simulate_ic_once(source: int, neighbors: list[np.ndarray], inv_degree: np.ndarray, rng: np.random.Generator) -> int:
    activated = np.zeros(shape=len(neighbors), dtype=bool)
    activated[int(source)] = True
    frontier = [int(source)]

    while frontier:
        next_frontier: list[int] = []
        for node in frontier:
            for nb in neighbors[int(node)]:
                nb_i = int(nb)
                if activated[nb_i]:
                    continue
                p = float(inv_degree[nb_i])
                if p <= 0.0:
                    continue
                if rng.random() < p:
                    activated[nb_i] = True
                    next_frontier.append(nb_i)
        frontier = next_frontier

    return int(activated.sum())


def _simulate_ic_means(g: nx.Graph, n_runs_per_node: int, seed_base: int = 42) -> np.ndarray:
    neighbors, inv_degree = _graph_to_neighbors_and_inv_degree(g)
    n = len(neighbors)
    means = np.zeros(n, dtype=float)

    for source in range(n):
        rng = np.random.default_rng(int(seed_base) + int(source))
        runs = np.empty(int(n_runs_per_node), dtype=np.int32)
        for i in range(int(n_runs_per_node)):
            runs[i] = _simulate_ic_once(source=source, neighbors=neighbors, inv_degree=inv_degree, rng=rng)
        means[source] = float(runs.mean())

    return means


def _hidden_betweenness_mean(g: nx.Graph, labels: np.ndarray) -> float:
    if g.number_of_nodes() == 0:
        return 0.0

    bet = nx.betweenness_centrality(g, normalized=True)
    hidden_nodes = [int(i) for i, lab in enumerate(labels.tolist()) if str(lab) == "Hidden"]
    if not hidden_nodes:
        return 0.0
    vals = [float(bet.get(n, 0.0)) for n in hidden_nodes]
    return float(np.mean(vals))


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)

    typology_path = Path(args.typology)
    csr_path = Path(args.csr)
    out_path = out_dir / "null_model_typology_summary.json"

    if int(args.n_sample) < 2:
        raise ValueError("--n-sample must be >= 2")
    if int(args.n_realizations) < 1:
        raise ValueError("--n-realizations must be >= 1")
    if int(args.n_runs_per_node) < 1:
        raise ValueError("--n-runs-per-node must be >= 1")
    if not (0.0 < float(args.top_pct) < 1.0):
        raise ValueError("--top-pct must be in (0,1)")

    if args.dry_run:
        n_nodes = int(args.n_sample)
        write_json(
            out_path,
            {
                "timestamp": now_iso(),
                "dry_run": True,
                "n_nodes": n_nodes,
                "n_realizations": int(args.n_realizations),
                "n_runs_per_node": int(args.n_runs_per_node),
                "rho_mean": 0.0,
                "rho_std": 0.0,
                "hidden_betweenness_real_subgraph_mean": 0.0,
                "hidden_betweenness_null_mean": 0.0,
                "hidden_betweenness_null_std": 0.0,
                "interpretation": "Dry-run placeholder only. Execute real mode for Task 7 results.",
            },
        )
        print(f"[OK] Wrote dry-run placeholder: {out_path}")
        return

    if not typology_path.exists():
        raise FileNotFoundError(f"Missing typology labels: {typology_path}")
    if not csr_path.exists():
        raise FileNotFoundError(f"Missing CSR graph: {csr_path}")

    df = pd.read_parquet(typology_path)
    require_columns(df, ["node_id", "ic_score_mean", "views"], "typology")

    work = df[["node_id", "ic_score_mean", "views"]].copy()
    work["node_id"] = work["node_id"].astype(str)
    work["ic_score_mean"] = pd.to_numeric(work["ic_score_mean"], errors="coerce")
    work["views"] = pd.to_numeric(work["views"], errors="coerce")
    if work[["ic_score_mean", "views"]].isna().any().any():
        na_counts = work[["ic_score_mean", "views"]].isna().sum().to_dict()
        raise ValueError(f"Null-model inputs contain missing numeric values: {na_counts}")

    n_available = int(len(work))
    n_sample = int(min(int(args.n_sample), n_available))
    if n_sample < int(args.n_sample):
        print(f"[WARN] Requested n_sample={args.n_sample} but only {n_available} labeled rows available. Using {n_sample}.")

    rng_sample = np.random.default_rng(int(args.sample_seed))
    sampled_idx = np.sort(rng_sample.choice(np.arange(n_available), size=n_sample, replace=False))
    sampled = work.iloc[sampled_idx].reset_index(drop=True)

    sample_node_ids = sampled["node_id"].to_numpy(dtype=str)
    real_ic_scores = sampled["ic_score_mean"].to_numpy(dtype=float)
    views = sampled["views"].to_numpy(dtype=float)

    g_real_sub = _build_real_subgraph_from_csr(sample_node_ids=sample_node_ids, csr_path=csr_path)

    real_labels = _assign_typology(ic_scores=real_ic_scores, views=views, top_pct=float(args.top_pct))
    hidden_bet_real = _hidden_betweenness_mean(g=g_real_sub, labels=real_labels)

    rho_values: list[float] = []
    hidden_bet_null_values: list[float] = []

    degree_sequence = [int(d) for _, d in g_real_sub.degree()]
    for realization_idx in range(int(args.n_realizations)):
        realization_seed = int(realization_idx) * 100

        g_null = nx.Graph(nx.configuration_model(degree_sequence, seed=realization_seed))
        g_null.remove_edges_from(nx.selfloop_edges(g_null))
        if g_null.number_of_nodes() < n_sample:
            g_null.add_nodes_from(range(g_null.number_of_nodes(), n_sample))

        null_ic_scores = _simulate_ic_means(
            g=g_null,
            n_runs_per_node=int(args.n_runs_per_node),
            seed_base=42,
        )

        rho_result: Any = stats.spearmanr(real_ic_scores, null_ic_scores)
        rho = float(rho_result.correlation) if hasattr(rho_result, "correlation") else float(rho_result[0])
        if np.isnan(rho):
            rho = 0.0
        rho_values.append(rho)

        null_labels = _assign_typology(ic_scores=null_ic_scores, views=views, top_pct=float(args.top_pct))
        hidden_bet_null_values.append(_hidden_betweenness_mean(g=g_null, labels=null_labels))

    rho_mean = float(np.mean(rho_values)) if rho_values else 0.0
    rho_std = float(np.std(rho_values, ddof=1)) if len(rho_values) > 1 else 0.0
    hidden_bet_null_mean = float(np.mean(hidden_bet_null_values)) if hidden_bet_null_values else 0.0
    hidden_bet_null_std = float(np.std(hidden_bet_null_values, ddof=1)) if len(hidden_bet_null_values) > 1 else 0.0

    if hidden_bet_real > (hidden_bet_null_mean + max(0.05, hidden_bet_null_std)):
        interpretation = (
            "Null graph Hidden nodes do NOT show elevated betweenness — typology reflects true structural "
            "position, not degree-distribution artifact."
        )
    else:
        interpretation = (
            "Hidden-node betweenness on real subgraph is comparable to configuration null; "
            "report potential degree-distribution artifact as limitation."
        )

    payload: dict[str, Any] = {
        "timestamp": now_iso(),
        "n_nodes": int(n_sample),
        "n_realizations": int(args.n_realizations),
        "n_runs_per_node": int(args.n_runs_per_node),
        "rho_mean": rho_mean,
        "rho_std": rho_std,
        "hidden_betweenness_real_subgraph_mean": float(hidden_bet_real),
        "hidden_betweenness_null_mean": hidden_bet_null_mean,
        "hidden_betweenness_null_std": hidden_bet_null_std,
        "interpretation": interpretation,
        "top_pct": float(args.top_pct),
        "sample_seed": int(args.sample_seed),
        "realization_seeds": [int(i) * 100 for i in range(int(args.n_realizations))],
    }
    write_json(out_path, payload)
    print(
        "[OK] Wrote null model summary: "
        f"{out_path} (n_nodes={n_sample}, n_realizations={args.n_realizations}, n_runs_per_node={args.n_runs_per_node})"
    )


if __name__ == "__main__":
    main()
