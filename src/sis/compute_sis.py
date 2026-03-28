"""
SIS (Structural Influence Score) Computation Module
====================================================
Computes the Structural Influence Score using UNWEIGHTED rank-average
as defined in the proposal (Section 6).

SIS Formula:
    SIS(v) = [rank(PageRank) + rank(Betweenness) + rank(k-shell)] / 3

This is an UNWEIGHTED average of ranks. Do NOT add weights.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    """
    Compute rank of values in a series.

    Parameters
    ----------
    series : pd.Series
        Values to rank
    ascending : bool
        If True, lowest value gets rank 1 (default for most centralities)

    Returns
    -------
    pd.Series
        Ranks (1 = lowest value when ascending=True)
    """
    return series.rank(method='average', ascending=ascending)


def compute_sis(
    centrality_path: str = "data/processed/centrality_table.parquet",
    output_dir: str = "outputs/stage3",
    output_data_dir: str = "data/processed",
    seed: int = 42
) -> pd.DataFrame:
    """
    Compute Structural Influence Score using UNWEIGHTED rank-average.

    The SIS formula is:
        SIS(v) = [rank(PageRank) + rank(Betweenness) + rank(k-shell)] / 3

    This is the formula defined in the proposal (Section 6).
    NO WEIGHTS are applied - this is an unweighted average.

    Parameters
    ----------
    centrality_path : str
        Path to centrality table parquet
    output_dir : str
        Output directory for metrics
    output_data_dir : str
        Output directory for data files
    seed : int
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        DataFrame with SIS scores and ranks
    """
    np.random.seed(seed)
    output_dir = Path(output_dir)
    output_data_dir = Path(output_data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_data_dir.mkdir(parents=True, exist_ok=True)

    # Load centrality data
    logger.info(f"Loading centrality data from {centrality_path}...")
    centrality_df = pd.read_parquet(centrality_path)

    n_nodes = len(centrality_df)
    logger.info(f"Loaded {n_nodes} nodes")

    # Verify required columns exist
    required_cols = ['node_id', 'pagerank', 'betweenness', 'kshell']
    missing_cols = [c for c in required_cols if c not in centrality_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Compute ranks (higher centrality = higher rank number)
    # ascending=False means highest value gets highest rank
    logger.info("Computing ranks for each centrality metric...")
    centrality_df['rank_pagerank'] = compute_rank(centrality_df['pagerank'], ascending=False)
    centrality_df['rank_betweenness'] = compute_rank(centrality_df['betweenness'], ascending=False)
    centrality_df['rank_kshell'] = compute_rank(centrality_df['kshell'], ascending=False)

    # Compute SIS as UNWEIGHTED average of ranks (per proposal Section 6)
    # SIS(v) = [rank(PageRank) + rank(Betweenness) + rank(k-shell)] / 3
    logger.info("Computing SIS (unweighted rank-average)...")
    centrality_df['sis_score'] = (
        centrality_df['rank_pagerank'] +
        centrality_df['rank_betweenness'] +
        centrality_df['rank_kshell']
    ) / 3

    # Compute SIS rank (higher SIS = higher rank)
    centrality_df['sis_rank'] = compute_rank(centrality_df['sis_score'], ascending=False)

    # Create output DataFrame
    sis_df = centrality_df[['node_id', 'sis_score', 'sis_rank',
                            'rank_pagerank', 'rank_betweenness', 'rank_kshell']].copy()

    # Save SIS table
    output_path = output_data_dir / "sis_table.parquet"
    sis_df.to_parquet(output_path, index=False)
    logger.info(f"Saved SIS table to {output_path}")

    # Compute and save metrics
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "n_nodes": n_nodes,
        "sis_formula": "SIS(v) = [rank(PageRank) + rank(Betweenness) + rank(k-shell)] / 3",
        "sis_formula_note": "Unweighted rank-average as per proposal Section 6",
        "sis_stats": {
            "mean": float(sis_df['sis_score'].mean()),
            "std": float(sis_df['sis_score'].std()),
            "min": float(sis_df['sis_score'].min()),
            "max": float(sis_df['sis_score'].max()),
            "median": float(sis_df['sis_score'].median())
        },
        "rank_correlations": {
            "pagerank_betweenness": float(centrality_df['rank_pagerank'].corr(
                centrality_df['rank_betweenness'], method='spearman')),
            "pagerank_kshell": float(centrality_df['rank_pagerank'].corr(
                centrality_df['rank_kshell'], method='spearman')),
            "betweenness_kshell": float(centrality_df['rank_betweenness'].corr(
                centrality_df['rank_kshell'], method='spearman'))
        }
    }

    metrics_path = output_dir / "sis_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")

    # Save params
    params = {
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "input_path": str(centrality_path),
        "output_path": str(output_path),
        "sis_formula": "SIS(v) = [rank(PageRank) + rank(Betweenness) + rank(k-shell)] / 3",
        "weights": "None (unweighted average)",
        "rank_method": "average"
    }

    params_path = output_dir / "sis_params.json"
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2)
    logger.info(f"Saved params to {params_path}")

    logger.info("SIS computation complete")
    return sis_df


if __name__ == "__main__":
    compute_sis()
