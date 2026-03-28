"""
IC Model Calibration Module
===========================
Calibrate Independent Cascade activation probability (p) to avoid
ceiling/floor effects in influence simulation.

CHANGE-6: Parameters per implementation plan:
- Target: mean_reach / N in [5%, 30%] (not [8%, 25%])
- p values: {0.01, 0.03, 0.05} (not including 0.08)
- Subgraph: 10% (not 20%)
- Seeds: k=10 (not 50)
- Runs: 50 per seed
"""

import networkx as nx
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from typing import List, Tuple
from joblib import Parallel, delayed
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_single_ic(G: nx.Graph, seed_node: int, p: float, max_steps: int = 100) -> int:
    """
    Run single Independent Cascade simulation.

    Parameters
    ----------
    G : nx.Graph
        Network graph
    seed_node : int
        Starting seed node
    p : float
        Activation probability
    max_steps : int
        Maximum propagation steps

    Returns
    -------
    int
        Number of activated nodes (reach)
    """
    activated = {seed_node}
    newly_activated = {seed_node}

    for _ in range(max_steps):
        if not newly_activated:
            break

        next_activated = set()
        for node in newly_activated:
            for neighbor in G.neighbors(node):
                if neighbor not in activated:
                    if np.random.random() < p:
                        next_activated.add(neighbor)
                        activated.add(neighbor)

        newly_activated = next_activated

    return len(activated)


def run_ic_pilot(
    G: nx.Graph,
    p: float,
    n_seeds: int = 50,
    n_runs_per_seed: int = 10,
    seed: int = 42
) -> dict:
    """
    Run pilot IC simulations for a given p value.

    Parameters
    ----------
    G : nx.Graph
        Network graph
    p : float
        Activation probability to test
    n_seeds : int
        Number of seed nodes to test
    n_runs_per_seed : int
        Runs per seed node
    seed : int
        Random seed

    Returns
    -------
    dict
        Statistics for this p value
    """
    np.random.seed(seed)
    nodes = list(G.nodes())
    sample_seeds = np.random.choice(nodes, size=min(n_seeds, len(nodes)), replace=False)

    all_reaches = []

    for seed_node in sample_seeds:
        for _ in range(n_runs_per_seed):
            reach = run_single_ic(G, seed_node, p)
            all_reaches.append(reach)

    reaches = np.array(all_reaches)
    N = G.number_of_nodes()

    return {
        "p": p,
        "mean_reach": float(np.mean(reaches)),
        "std_reach": float(np.std(reaches)),
        "median_reach": float(np.median(reaches)),
        "mean_reach_pct": float(np.mean(reaches) / N * 100),
        "min_reach": int(np.min(reaches)),
        "max_reach": int(np.max(reaches)),
        "n_simulations": len(reaches)
    }


def sample_subgraph(G: nx.Graph, fraction: float = 0.2, seed: int = 42) -> nx.Graph:
    """
    Sample a connected subgraph for pilot testing.

    Parameters
    ----------
    G : nx.Graph
        Original graph
    fraction : float
        Fraction of nodes to sample
    seed : int
        Random seed

    Returns
    -------
    nx.Graph
        Sampled subgraph
    """
    np.random.seed(seed)
    nodes = list(G.nodes())
    n_sample = int(len(nodes) * fraction)

    # Start from random node and do BFS to get connected subgraph
    start_node = np.random.choice(nodes)
    sampled_nodes = set()
    queue = [start_node]

    while queue and len(sampled_nodes) < n_sample:
        node = queue.pop(0)
        if node not in sampled_nodes:
            sampled_nodes.add(node)
            neighbors = list(G.neighbors(node))
            np.random.shuffle(neighbors)
            queue.extend(neighbors)

    return G.subgraph(sampled_nodes).copy()


def calibrate_ic_parameter(
    graph_path: str = "data/processed/graph_active.edgelist",
    output_dir: str = "outputs/stage3_ic_calibration",
    p_values: List[float] = [0.01, 0.03, 0.05],  # CHANGE-6: removed 0.08
    subgraph_fraction: float = 0.10,  # CHANGE-6: 10% instead of 20%
    n_seeds: int = 10,  # CHANGE-6: k=10 instead of 50
    n_runs_per_seed: int = 50,  # 50 runs per seed
    target_range: Tuple[float, float] = (5.0, 30.0),  # CHANGE-6: [5%, 30%] instead of [8%, 25%]
    seed: int = 42,
    n_jobs: int = -1
) -> dict:
    """
    Run IC calibration to find optimal p value.

    Parameters
    ----------
    graph_path : str
        Path to graph edgelist
    output_dir : str
        Output directory
    p_values : List[float]
        P values to test
    subgraph_fraction : float
        Fraction of nodes for pilot subgraph
    n_seeds : int
        Number of seed nodes per p
    n_runs_per_seed : int
        Runs per seed
    target_range : Tuple[float, float]
        Target range for mean_reach_pct [min%, max%]
    seed : int
        Random seed
    n_jobs : int
        Number of parallel jobs (-1 for all cores)

    Returns
    -------
    dict
        Calibration results and selected p
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load graph
    logger.info(f"Loading graph from {graph_path}...")
    G = nx.read_edgelist(graph_path)
    logger.info(f"Full graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Sample subgraph for pilot
    logger.info(f"Sampling {subgraph_fraction*100}% subgraph for pilot...")
    G_pilot = sample_subgraph(G, fraction=subgraph_fraction, seed=seed)
    logger.info(f"Pilot subgraph: {G_pilot.number_of_nodes()} nodes, {G_pilot.number_of_edges()} edges")

    # Test each p value
    logger.info(f"Testing p values: {p_values}")
    results = []

    for p in tqdm(p_values, desc="Calibrating p"):
        result = run_ic_pilot(
            G_pilot, p, n_seeds=n_seeds, n_runs_per_seed=n_runs_per_seed, seed=seed
        )
        results.append(result)
        logger.info(f"p={p}: mean_reach_pct={result['mean_reach_pct']:.2f}%")

    # Find best p in target range
    selected_p = None
    selected_reason = ""

    for result in results:
        pct = result['mean_reach_pct']
        if target_range[0] <= pct <= target_range[1]:
            selected_p = result['p']
            selected_reason = f"mean_reach_pct={pct:.2f}% falls in target range [{target_range[0]}%, {target_range[1]}%]"
            break

    # If none in range, select closest to midpoint
    if selected_p is None:
        target_mid = (target_range[0] + target_range[1]) / 2
        closest = min(results, key=lambda x: abs(x['mean_reach_pct'] - target_mid))
        selected_p = closest['p']
        selected_reason = f"No p in target range. Selected p={selected_p} with mean_reach_pct={closest['mean_reach_pct']:.2f}% (closest to midpoint {target_mid}%)"
        logger.warning(selected_reason)

    # Compile output
    calibration_output = {
        "pilot_subgraph": {
            "n_nodes": G_pilot.number_of_nodes(),
            "n_edges": G_pilot.number_of_edges(),
            "fraction": subgraph_fraction
        },
        "parameters_tested": results,
        "target_range_pct": list(target_range),
        "selected_p": selected_p,
        "selection_reason": selected_reason,
        "seed": seed
    }

    # Save results
    output_file = output_dir / "calibration_results.json"
    with open(output_file, 'w') as f:
        json.dump(calibration_output, f, indent=2)

    # Also save as CSV for easy viewing
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "calibration_results.csv", index=False)

    logger.info(f"Calibration complete. Selected p={selected_p}")
    logger.info(f"Results saved to {output_dir}")

    return calibration_output


if __name__ == "__main__":
    calibrate_ic_parameter()
