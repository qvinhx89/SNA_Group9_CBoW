"""
Centrality Computation Module
=============================
Computes centrality metrics for network nodes:
- Degree centrality
- PageRank
- Betweenness centrality (approximate)

CHANGE-5: Betweenness uses k=1000 pivots with seed=42 per implementation plan.
"""

import networkx as nx
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "src/config/base.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}. Using defaults.")
        return {}


def compute_centrality(
    graph_path: str = "data/processed/graph_active.edgelist",
    node_attrs_path: str = "data/processed/node_attributes.parquet",
    output_dir: str = "outputs/stage1",
    output_data_dir: str = "data/processed",
    config_path: str = "src/config/base.yaml"
) -> pd.DataFrame:
    """
    Compute centrality metrics for all nodes.

    Parameters
    ----------
    graph_path : str
        Path to graph edgelist
    node_attrs_path : str
        Path to node attributes parquet
    output_dir : str
        Output directory for metrics
    output_data_dir : str
        Output directory for data files
    config_path : str
        Path to configuration file

    Returns
    -------
    pd.DataFrame
        DataFrame with centrality metrics
    """
    output_dir = Path(output_dir)
    output_data_dir = Path(output_data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_data_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config = load_config(config_path)

    # CHANGE-5: Betweenness parameters from config or defaults
    betweenness_k = config.get('centrality', {}).get('betweenness', {}).get('k_pivots', 1000)
    betweenness_seed = config.get('centrality', {}).get('betweenness', {}).get('seed', 42)

    pagerank_alpha = config.get('centrality', {}).get('pagerank', {}).get('alpha', 0.85)
    pagerank_max_iter = config.get('centrality', {}).get('pagerank', {}).get('max_iter', 100)
    pagerank_tol = config.get('centrality', {}).get('pagerank', {}).get('tol', 1e-6)

    # Load graph
    logger.info(f"Loading graph from {graph_path}...")
    G = nx.read_edgelist(graph_path)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    logger.info(f"Graph loaded: {n_nodes} nodes, {n_edges} edges")

    # Compute degree centrality
    logger.info("Computing degree centrality...")
    degree_centrality = nx.degree_centrality(G)

    # Also store raw degree
    degree_raw = dict(G.degree())

    # Compute PageRank
    logger.info(f"Computing PageRank (alpha={pagerank_alpha})...")
    pagerank = nx.pagerank(G, alpha=pagerank_alpha, max_iter=pagerank_max_iter, tol=pagerank_tol)

    # Compute Betweenness centrality (approximate)
    # CHANGE-5: k=1000 pivots, seed=42 per implementation plan
    logger.info(f"Computing approximate betweenness (k={betweenness_k}, seed={betweenness_seed})...")
    logger.info("Note: k=1000 on N~163K gives ~3% error bound per Brandes (2001)")
    betweenness = nx.betweenness_centrality(G, k=betweenness_k, seed=betweenness_seed)

    # Create DataFrame
    logger.info("Building centrality table...")
    nodes = list(G.nodes())
    centrality_df = pd.DataFrame({
        'node_id': nodes,
        'degree': [degree_raw[n] for n in nodes],
        'degree_centrality': [degree_centrality[n] for n in nodes],
        'pagerank': [pagerank[n] for n in nodes],
        'betweenness': [betweenness[n] for n in nodes]
    })

    # Load node attributes if available and merge views
    try:
        attrs_df = pd.read_parquet(node_attrs_path)
        if 'views' in attrs_df.columns:
            centrality_df = centrality_df.merge(
                attrs_df[['node_id', 'views']],
                on='node_id',
                how='left'
            )
            logger.info("Merged views from node attributes")
    except FileNotFoundError:
        logger.warning(f"Node attributes file not found: {node_attrs_path}")

    # Save centrality table
    output_path = output_data_dir / "centrality_table.parquet"
    centrality_df.to_parquet(output_path, index=False)
    logger.info(f"Saved centrality table to {output_path}")

    # Compute correlations
    correlations = {
        "pagerank_betweenness": float(centrality_df['pagerank'].corr(
            centrality_df['betweenness'], method='spearman')),
        "pagerank_degree": float(centrality_df['pagerank'].corr(
            centrality_df['degree'], method='spearman')),
        "betweenness_degree": float(centrality_df['betweenness'].corr(
            centrality_df['degree'], method='spearman'))
    }

    if 'views' in centrality_df.columns:
        correlations["pagerank_views"] = float(centrality_df['pagerank'].corr(
            centrality_df['views'], method='spearman'))
        correlations["betweenness_views"] = float(centrality_df['betweenness'].corr(
            centrality_df['views'], method='spearman'))
        correlations["degree_views"] = float(centrality_df['degree'].corr(
            centrality_df['views'], method='spearman'))

    # Save metrics
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "betweenness_k": betweenness_k,  # CHANGE-5: record k
        "betweenness_seed": betweenness_seed,  # CHANGE-5: record seed
        "pagerank_alpha": pagerank_alpha,
        "centrality_stats": {
            "degree": {
                "mean": float(centrality_df['degree'].mean()),
                "std": float(centrality_df['degree'].std()),
                "max": int(centrality_df['degree'].max())
            },
            "pagerank": {
                "mean": float(centrality_df['pagerank'].mean()),
                "std": float(centrality_df['pagerank'].std()),
                "max": float(centrality_df['pagerank'].max())
            },
            "betweenness": {
                "mean": float(centrality_df['betweenness'].mean()),
                "std": float(centrality_df['betweenness'].std()),
                "max": float(centrality_df['betweenness'].max())
            }
        },
        "spearman_correlations": correlations
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")

    # Save params (CHANGE-5: include betweenness k and seed)
    params = {
        "timestamp": datetime.now().isoformat(),
        "graph_path": str(graph_path),
        "betweenness_k": betweenness_k,
        "betweenness_seed": betweenness_seed,
        "betweenness_note": "k=1000 on N~163K gives ~3% error bound per Brandes (2001)",
        "pagerank_alpha": pagerank_alpha,
        "pagerank_max_iter": pagerank_max_iter,
        "pagerank_tol": pagerank_tol
    }

    params_path = output_dir / "params.json"
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2)
    logger.info(f"Saved params to {params_path}")

    logger.info("Centrality computation complete")
    return centrality_df


if __name__ == "__main__":
    compute_centrality()
