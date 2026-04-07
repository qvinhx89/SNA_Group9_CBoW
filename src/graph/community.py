"""
Community Detection Module
==========================
Performs community detection using Louvain algorithm with stability check.

CHANGE-4: Run Louvain 10 times with different seeds, compute NMI stability,
and select the partition with highest modularity Q.

CHANGE-5: Export contract-ready community features:
- community_id
- cross_community_edge_fraction
"""

import networkx as nx
import pandas as pd
import numpy as np
from pathlib import Path
import importlib
import json
import logging
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import yaml

try:
    from sklearn.metrics import normalized_mutual_info_score
except ImportError:
    normalized_mutual_info_score = None
    logging.warning("sklearn not installed for NMI computation")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _import_python_louvain() -> Optional[object]:
    """Import python-louvain robustly even when this file name shadows `community`.

    Running this script directly (src/graph/community.py) can shadow the
    third-party package `community`. To avoid that, temporarily remove this
    script directory from sys.path while importing.
    """
    script_file = Path(__file__).resolve()
    script_dir = script_file.parent

    original_sys_path = list(sys.path)
    existing_community = sys.modules.pop("community", None)
    mod: Optional[object] = None

    try:
        filtered_path: list[str] = []
        for p in original_sys_path:
            resolved = Path(p if p else ".").resolve()
            if resolved == script_dir:
                continue
            filtered_path.append(p)

        sys.path = filtered_path
        mod = importlib.import_module("community")
    except Exception:
        mod = None
    finally:
        sys.path = original_sys_path
        if mod is None and existing_community is not None:
            sys.modules["community"] = existing_community

    if mod is None:
        return None

    mod_file = getattr(mod, "__file__", None)
    if mod_file is not None and Path(mod_file).resolve() == script_file:
        return None

    return mod


def _resolve_louvain_backend() -> Tuple[str, Optional[object]]:
    """
    Resolve a Louvain backend from available libraries.

    Returns
    -------
    Tuple[str, Optional[object]]
        (backend_name, backend_module)
        backend_name is always "python-louvain"
    """
    community_louvain = _import_python_louvain()
    if community_louvain is not None and hasattr(community_louvain, "best_partition") and hasattr(
        community_louvain, "modularity"
    ):
        return "python-louvain", community_louvain

    raise ImportError(
        "python-louvain is required for this pipeline. "
        "Install with: pip install python-louvain. "
        "If already installed, run from project root with the configured interpreter."
    )


def load_config(config_path: str = "src/config/base.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}. Using defaults.")
        return {}


def run_louvain_single(G: nx.Graph, seed: int, resolution: float = 1.0) -> Tuple[Dict, float]:
    """
    Run single Louvain community detection.

    Parameters
    ----------
    G : nx.Graph
        Network graph
    seed : int
        Random seed
    resolution : float
        Resolution parameter

    Returns
    -------
    Tuple[Dict, float]
        (partition dict, modularity Q)
    """
    _, backend_module = _resolve_louvain_backend()
    if backend_module is None:
        raise ImportError("python-louvain backend selected but module is unavailable")

    best_partition_fn = getattr(backend_module, "best_partition")
    modularity_fn = getattr(backend_module, "modularity")
    partition = best_partition_fn(G, random_state=seed, resolution=resolution)
    modularity = modularity_fn(partition, G)
    return partition, float(modularity)


def compute_nmi_between_partitions(partition1: Dict, partition2: Dict, nodes: List) -> float:
    """
    Compute Normalized Mutual Information between two partitions.

    Parameters
    ----------
    partition1 : Dict
        First partition (node -> community)
    partition2 : Dict
        Second partition (node -> community)
    nodes : List
        List of nodes (to ensure same ordering)

    Returns
    -------
    float
        NMI score [0, 1]
    """
    if normalized_mutual_info_score is None:
        raise ImportError("sklearn not installed for NMI computation")

    labels1 = [partition1.get(n, -1) for n in nodes]
    labels2 = [partition2.get(n, -1) for n in nodes]

    return float(normalized_mutual_info_score(labels1, labels2))


def compute_cross_community_edge_fraction(
    G: nx.Graph,
    partition: Dict,
    nodes: List,
) -> np.ndarray:
    """Compute cross-community edge fraction for every node.

    For each node u:
        cross_community_edge_fraction(u) =
            (# neighbors v where community(v) != community(u)) / degree(u)

    Isolated nodes get 0.0 by definition.
    """
    fractions = np.zeros(len(nodes), dtype=np.float64)
    for i, node in enumerate(nodes):
        deg_u = G.degree(node)
        if deg_u == 0:
            fractions[i] = 0.0
            continue

        comm_u = partition[node]
        cross_count = 0
        for nbr in G.neighbors(node):
            if partition[nbr] != comm_u:
                cross_count += 1

        fractions[i] = cross_count / deg_u

    return fractions


def detect_communities(
    graph_path: str = "data/processed/graph_active.edgelist",
    output_dir: str = "outputs/stage2",
    output_data_dir: str = "data/processed",
    config_path: str = "src/config/base.yaml"
) -> pd.DataFrame:
    """
    Detect communities using Louvain with stability check.

    CHANGE-4: Run 10 times, compute NMI stability, select best-Q partition.

    Parameters
    ----------
    graph_path : str
        Path to graph edgelist
    output_dir : str
        Output directory for metrics
    output_data_dir : str
        Output directory for data files
    config_path : str
        Path to configuration file

    Returns
    -------
    pd.DataFrame
        DataFrame with community labels
    """
    output_dir_path = Path(output_dir)
    output_data_dir_path = Path(output_data_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_data_dir_path.mkdir(parents=True, exist_ok=True)

    # Load config
    config = load_config(config_path)

    # CHANGE-4: Louvain parameters from config
    n_runs = config.get('community', {}).get('louvain', {}).get('n_runs', 10)
    resolution = config.get('community', {}).get('louvain', {}).get('resolution', 1.0)
    seed_start = config.get('community', {}).get('louvain', {}).get('seed_start', 0)
    nmi_threshold = config.get('community', {}).get('louvain', {}).get('nmi_threshold', 0.85)

    # Load graph
    logger.info(f"Loading graph from {graph_path}...")
    G = nx.read_edgelist(graph_path)
    nodes = list(G.nodes())
    n_nodes = len(nodes)
    logger.info(f"Graph loaded: {n_nodes} nodes, {G.number_of_edges()} edges")

    backend_name, _ = _resolve_louvain_backend()
    logger.info(f"Using Louvain backend: {backend_name}")

    # CHANGE-4: Run Louvain n_runs times with different seeds
    logger.info(f"Running Louvain {n_runs} times for stability check...")
    partitions = []
    modularities = []

    for i in range(n_runs):
        seed = seed_start + i
        partition, modularity = run_louvain_single(G, seed=seed, resolution=resolution)
        partitions.append(partition)
        modularities.append(modularity)
        logger.info(f"  Run {i+1}/{n_runs}: seed={seed}, Q={modularity:.4f}, "
                    f"n_communities={len(set(partition.values()))}")

    # CHANGE-4: Select best partition (highest modularity Q)
    best_idx = np.argmax(modularities)
    best_partition = partitions[best_idx]
    best_modularity = modularities[best_idx]
    best_seed = seed_start + best_idx
    n_communities = len(set(best_partition.values()))

    logger.info(f"Best partition: run {best_idx+1}, seed={best_seed}, "
                f"Q={best_modularity:.4f}, n_communities={n_communities}")

    # CHANGE-4: Compute NMI between all pairs of partitions
    logger.info("Computing NMI stability across runs...")
    nmi_values = []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            nmi = compute_nmi_between_partitions(partitions[i], partitions[j], nodes)
            nmi_values.append(nmi)

    mean_nmi = np.mean(nmi_values)
    std_nmi = np.std(nmi_values)
    min_nmi = np.min(nmi_values)

    logger.info(f"NMI stability: mean={mean_nmi:.4f}, std={std_nmi:.4f}, min={min_nmi:.4f}")

    # Flag instability warning
    if mean_nmi < nmi_threshold:
        logger.warning(f"Louvain instability detected: mean_NMI={mean_nmi:.4f} < threshold={nmi_threshold}")

    # CHANGE-5: Build contract-ready community features.
    community_ids = np.array([int(best_partition[n]) for n in nodes], dtype=np.int64)
    cross_comm_fraction = compute_cross_community_edge_fraction(G, best_partition, nodes)

    # Keep legacy column name `community` for backward compatibility,
    # and add contract columns used by MAPR2026 v3.
    community_df = pd.DataFrame(
        {
            'node_id': nodes,
            'community': community_ids,
            'community_id': community_ids,
            'cross_community_edge_fraction': cross_comm_fraction,
        }
    )

    # Save community labels (from best-Q run)
    output_path = output_data_dir_path / "community_labels.parquet"
    community_df.to_parquet(output_path, index=False)
    logger.info(f"Saved community labels (best-Q run) to {output_path}")

    # Save explicit contract artifact for Person 2 Track B.
    community_features_path = output_data_dir_path / "community_features.parquet"
    community_df[["node_id", "community_id", "cross_community_edge_fraction"]].to_parquet(
        community_features_path,
        index=False,
    )
    logger.info(
        "Saved community features (community_id + cross_community_edge_fraction) "
        f"to {community_features_path}"
    )

    # CHANGE-4: Save metrics including mean_nmi_louvain
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "backend": backend_name,
        "n_nodes": n_nodes,
        "n_runs": n_runs,
        "best_run_index": int(best_idx),
        "best_seed": int(best_seed),
        "best_modularity": float(best_modularity),
        "n_communities": int(n_communities),
        "all_modularities": [float(m) for m in modularities],
        "mean_nmi_louvain": float(mean_nmi),  # CHANGE-4: key metric for stability
        "std_nmi_louvain": float(std_nmi),
        "min_nmi_louvain": float(min_nmi),
        "cross_community_edge_fraction_mean": float(np.mean(cross_comm_fraction)),
        "cross_community_edge_fraction_std": float(np.std(cross_comm_fraction)),
        "nmi_threshold": nmi_threshold,
        "stability_warning": bool(mean_nmi < nmi_threshold),
    }

    metrics_path = output_dir_path / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics (including mean_nmi_louvain) to {metrics_path}")

    # Save detailed stability report
    stability_report = {
        "timestamp": datetime.now().isoformat(),
        "method": "louvain",
        "backend": backend_name,
        "n_runs": n_runs,
        "resolution": resolution,
        "seeds_used": list(range(seed_start, seed_start + n_runs)),
        "run_results": [
            {
                "run": i + 1,
                "seed": int(seed_start + i),
                "modularity": float(modularities[i]),
                "n_communities": int(len(set(partitions[i].values()))),
            }
            for i in range(n_runs)
        ],
        "nmi_pairwise": {
            f"run{i+1}_vs_run{j+1}": float(compute_nmi_between_partitions(partitions[i], partitions[j], nodes))
            for i in range(n_runs) for j in range(i + 1, n_runs)
        },
        "summary": {
            "mean_nmi": float(mean_nmi),
            "std_nmi": float(std_nmi),
            "min_nmi": float(min_nmi),
            "stability_threshold": nmi_threshold,
            "is_stable": bool(mean_nmi >= nmi_threshold),
        }
    }

    stability_path = output_dir_path / "louvain_stability_report.json"
    with open(stability_path, 'w') as f:
        json.dump(stability_report, f, indent=2)
    logger.info(f"Saved stability report to {stability_path}")

    logger.info("Community detection complete")
    return community_df


if __name__ == "__main__":
    detect_communities()
