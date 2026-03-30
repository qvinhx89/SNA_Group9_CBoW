"""
Null Model Comparison Module
============================
Generate configuration model to validate that typology patterns
are not artifacts of degree distribution.

Compares:
- % Hidden Influencers in real graph vs null model
- If similar → typology may be artifact of degree distribution
"""

import networkx as nx
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict, Tuple
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


def create_configuration_model(G: nx.Graph, seed: int = 42) -> nx.Graph:
    """
    Create configuration model preserving degree sequence.

    Parameters
    ----------
    G : nx.Graph
        Original graph
    seed : int
        Random seed

    Returns
    -------
    nx.Graph
        Configuration model graph
    """
    degree_sequence = [d for n, d in G.degree()]

    # Configuration model may have self-loops and multi-edges
    # Use configuation_model and convert to simple graph
    G_null = nx.configuration_model(degree_sequence, seed=seed)
    G_null = nx.Graph(G_null)  # Remove multi-edges
    G_null.remove_edges_from(nx.selfloop_edges(G_null))  # Remove self-loops

    return G_null


def compute_centrality_on_null(G_null: nx.Graph, k_betweenness: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Compute centrality metrics on null model.

    Parameters
    ----------
    G_null : nx.Graph
        Null model graph
    k_betweenness : int
        Number of pivots for betweenness approximation
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Centrality table for null model
    """
    logger.info("Computing centralities on null model...")

    # Degree
    degree = dict(G_null.degree())

    # PageRank
    pagerank = nx.pagerank(G_null, alpha=0.85)

    # Betweenness (approximate)
    betweenness = nx.betweenness_centrality(G_null, k=min(k_betweenness, G_null.number_of_nodes()), seed=seed)

    # K-shell
    kshell = nx.core_number(G_null)

    nodes = list(G_null.nodes())
    return pd.DataFrame({
        'node_id': nodes,
        'degree': [degree[n] for n in nodes],
        'pagerank': [pagerank[n] for n in nodes],
        'betweenness': [betweenness[n] for n in nodes],
        'kshell': [kshell[n] for n in nodes]
    })


def compute_sis_on_null(centrality_df: pd.DataFrame, weights: list = None) -> pd.DataFrame:
    """
    Compute SIS scores on null model centrality.

    Parameters
    ----------
    centrality_df : pd.DataFrame
        Centrality table from null model
    weights : list, optional
        Weights for SIS formula [w_pr, w_bet, w_ks]
        If None, uses unweighted average

    Returns
    -------
    pd.DataFrame
        DataFrame with SIS scores
    """
    df = centrality_df.copy()

    # Compute ranks (higher value = higher rank)
    df['rank_pagerank'] = df['pagerank'].rank(method='average', ascending=False)
    df['rank_betweenness'] = df['betweenness'].rank(method='average', ascending=False)
    df['rank_kshell'] = df['kshell'].rank(method='average', ascending=False)

    if weights is None:
        # Unweighted average (per proposal)
        df['sis_score'] = (df['rank_pagerank'] + df['rank_betweenness'] + df['rank_kshell']) / 3
    else:
        w_pr, w_bet, w_ks = weights
        df['sis_score'] = (
            w_pr * df['rank_pagerank'] +
            w_bet * df['rank_betweenness'] +
            w_ks * df['rank_kshell']
        )

    return df


def compute_typology_on_null(
    sis_df: pd.DataFrame,
    views_series: pd.Series = None,
    threshold: float = 0.20
) -> Dict:
    """
    Compute typology distribution on null model.

    For null model, we use degree as proxy for views (since no real views data).

    Parameters
    ----------
    sis_df : pd.DataFrame
        SIS scores from null model
    views_series : pd.Series, optional
        Real views data (if available for same nodes)
    threshold : float
        Top % threshold (default 20%)

    Returns
    -------
    Dict
        Typology distribution
    """
    df = sis_df.copy()
    n = len(df)
    top_k = int(n * threshold)

    # Use degree as proxy if no views
    if views_series is None:
        df['views_proxy'] = df['degree']
    else:
        df['views_proxy'] = views_series.values[:n] if len(views_series) >= n else df['degree']

    # SIS threshold
    sis_threshold = df['sis_score'].nlargest(top_k).min()
    df['sis_high'] = df['sis_score'] >= sis_threshold

    # Views/degree threshold
    views_threshold = df['views_proxy'].nlargest(top_k).min()
    df['views_high'] = df['views_proxy'] >= views_threshold

    # Typology
    def assign_typology(row):
        if row['sis_high'] and row['views_high']:
            return 'true'
        elif row['sis_high'] and not row['views_high']:
            return 'hidden'
        elif not row['sis_high'] and row['views_high']:
            return 'overrated'
        else:
            return 'non'

    df['typology'] = df.apply(assign_typology, axis=1)

    # Distribution
    distribution = {
        str(k): float(v)
        for k, v in df['typology'].value_counts(normalize=True).to_dict().items()
    }

    return {
        'distribution': distribution,
        'hidden_pct': float(distribution.get('hidden', 0.0) * 100.0),
        'n_hidden': int((df['typology'] == 'hidden').sum()),
        'n_total': int(n)
    }


def run_null_model_comparison(
    graph_path: str = "data/processed/graph_active.edgelist",
    typology_path: str = "data/processed/typology_labels.parquet",
    output_dir: str = "outputs/stage3",
    config_path: str = "src/config/base.yaml",
    sample_fraction: float = 0.2,
    seed: int = 42
) -> Dict:
    """
    Run full null model comparison.

    Parameters
    ----------
    graph_path : str
        Path to real graph
    typology_path : str
        Path to real typology labels
    output_dir : str
        Output directory
    config_path : str
        Config file path
    sample_fraction : float
        Fraction of nodes to sample (for speed)
    seed : int
        Random seed

    Returns
    -------
    Dict
        Comparison results
    """
    np.random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config = load_config(config_path)
    k_betweenness = config.get('centrality', {}).get('betweenness', {}).get('k_pivots', 1000)
    threshold = config.get('typology', {}).get('threshold_default', 0.20)

    # Load real graph
    logger.info(f"Loading graph from {graph_path}...")
    G = nx.read_edgelist(graph_path)
    n_nodes = G.number_of_nodes()
    logger.info(f"Graph loaded: {n_nodes} nodes, {G.number_of_edges()} edges")

    # Load real typology for comparison
    try:
        typology_df = pd.read_parquet(typology_path)
        real_hidden_pct = float((typology_df['typology_label'] == 'hidden').mean() * 100.0)
        logger.info(f"Real graph: {real_hidden_pct:.2f}% Hidden Influencers")
    except FileNotFoundError:
        logger.warning("Typology file not found. Will compute from scratch.")
        real_hidden_pct = None

    # Sample if needed (for speed)
    if sample_fraction < 1.0:
        sample_size = int(n_nodes * sample_fraction)
        sampled_nodes = np.random.choice(list(G.nodes()), size=sample_size, replace=False)
        G_sample = G.subgraph(sampled_nodes).copy()
        logger.info(f"Sampled {sample_size} nodes ({sample_fraction*100:.0f}%)")
    else:
        G_sample = G

    # Create null model
    logger.info("Creating configuration model...")
    G_null = create_configuration_model(G_sample, seed=seed)
    logger.info(f"Null model: {G_null.number_of_nodes()} nodes, {G_null.number_of_edges()} edges")

    # Compute centrality on null model
    null_centrality = compute_centrality_on_null(G_null, k_betweenness=k_betweenness, seed=seed)

    # Compute SIS on null model
    null_sis = compute_sis_on_null(null_centrality)

    # Compute typology on null model
    null_typology = compute_typology_on_null(null_sis, threshold=threshold)

    logger.info(f"Null model: {null_typology['hidden_pct']:.2f}% Hidden Influencers")

    # Comparison
    results = {
        "timestamp": datetime.now().isoformat(),
        "seed": int(seed),
        "sample_fraction": float(sample_fraction),
        "n_nodes_sampled": int(G_sample.number_of_nodes()),
        "threshold": float(threshold),
        "real_graph": {
            "hidden_pct": float(real_hidden_pct) if real_hidden_pct is not None else None
        },
        "null_model": {
            "hidden_pct": float(null_typology['hidden_pct']),
            "distribution": {k: float(v) for k, v in null_typology['distribution'].items()},
            "n_hidden": int(null_typology['n_hidden'])
        },
        "interpretation": ""
    }

    # Interpretation
    if real_hidden_pct is not None:
        diff = abs(real_hidden_pct - null_typology['hidden_pct'])
        if diff < 2.0:  # Within 2 percentage points
            results["interpretation"] = (
                f"WARNING: Real ({real_hidden_pct:.1f}%) and null ({null_typology['hidden_pct']:.1f}%) "
                f"Hidden % are similar (diff={diff:.1f}pp). Typology may be artifact of degree distribution."
            )
            logger.warning(results["interpretation"])
        else:
            results["interpretation"] = (
                f"Real ({real_hidden_pct:.1f}%) vs null ({null_typology['hidden_pct']:.1f}%) "
                f"show meaningful difference (diff={diff:.1f}pp). Typology reflects structural properties beyond degree."
            )
            logger.info(results["interpretation"])

    # Save results
    output_path = output_dir / "null_model_comparison.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {output_path}")

    # Save CSV summary
    csv_path = output_dir / "null_model_comparison.csv"
    pd.DataFrame([{
        'metric': 'hidden_pct',
        'real_graph': real_hidden_pct,
        'null_model': float(null_typology['hidden_pct']),
        'difference': (
            float(real_hidden_pct) - float(null_typology['hidden_pct'])
            if real_hidden_pct is not None
            else None
        )
    }]).to_csv(csv_path, index=False)

    logger.info("Null model comparison complete")
    return results


if __name__ == "__main__":
    run_null_model_comparison()
