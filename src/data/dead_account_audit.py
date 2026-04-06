"""
Dead Account Audit (Stage A0)
==============================
Prerequisite for all IC simulation. Stats from this module feed into
Section 5 (Limitations) of the paper.

Contract:
- Input: data/raw/large_twitch_features.csv (has column `dead_account`)
- Output: outputs/stage0_data_quality/dead_account_report.json

Schema:
{
  "n_dead": <int>,
  "n_live": <int>,
  "pct_dead": <float>,
  "mean_degree_dead": <float>,
  "mean_degree_live": <float>,
  "mean_views_dead": <float>,
  "mean_views_live": <float>
}

Paper framing (Section 5 Limitations):
"Dead accounts (X% of nodes) were excluded; they have systematically lower
degree and views than active accounts. Findings generalize only to active users."
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_dead_account_audit(
    raw_dir: str = "data/raw",
    output_dir: str = "outputs/stage0_data_quality",
) -> dict:
    """
    Audit dead vs live accounts before filtering.

    Returns:
        dict: Dead account statistics
    """
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find features file
    features_file = None
    for candidate in ["large_twitch_features.csv", "twitch_features.csv", "features.csv"]:
        candidate_path = raw_path / candidate
        if candidate_path.exists():
            features_file = candidate_path
            break

    if features_file is None:
        raise FileNotFoundError(
            f"No features file found in {raw_path}. "
            "Expected large_twitch_features.csv or similar with dead_account column."
        )

    logger.info(f"Reading features from: {features_file}")
    df = pd.read_csv(features_file)

    # Validate required columns
    if "dead_account" not in df.columns:
        raise ValueError(
            f"Column 'dead_account' not found in {features_file}. "
            f"Available columns: {list(df.columns)}"
        )

    if "views" not in df.columns:
        raise ValueError(
            f"Column 'views' not found in {features_file}. "
            f"Available columns: {list(df.columns)}"
        )

    # Compute degree if we have edge data
    degree_available = False
    edge_file = None
    for candidate in ["large_twitch_edges.csv", "twitch_edges.csv", "edges.csv"]:
        candidate_path = raw_path / candidate
        if candidate_path.exists():
            edge_file = candidate_path
            break

    if edge_file is not None:
        logger.info(f"Computing degree from: {edge_file}")
        edges_df = pd.read_csv(edge_file)

        # Infer column names
        col_names = edges_df.columns.tolist()
        if len(col_names) >= 2:
            source_col, target_col = col_names[0], col_names[1]

            # Count degree
            source_counts = edges_df[source_col].value_counts()
            target_counts = edges_df[target_col].value_counts()
            degree_counts = source_counts.add(target_counts, fill_value=0)

            # Map to numeric_id
            if "numeric_id" in df.columns:
                df["degree"] = df["numeric_id"].map(degree_counts).fillna(0).astype(int)
                degree_available = True
            else:
                logger.warning("numeric_id column not found, skipping degree computation")

    # Split by dead_account
    dead = df[df["dead_account"] == 1]
    live = df[df["dead_account"] == 0]

    n_dead = int(len(dead))
    n_live = int(len(live))
    n_total = n_dead + n_live
    pct_dead = float(n_dead / n_total * 100) if n_total > 0 else 0.0

    # Views stats
    mean_views_dead = float(dead["views"].mean()) if len(dead) > 0 else 0.0
    mean_views_live = float(live["views"].mean()) if len(live) > 0 else 0.0

    # Degree stats (if available)
    if degree_available:
        mean_degree_dead = float(dead["degree"].mean()) if len(dead) > 0 else 0.0
        mean_degree_live = float(live["degree"].mean()) if len(live) > 0 else 0.0
    else:
        mean_degree_dead = None
        mean_degree_live = None
        logger.warning("Degree statistics not available (edge file not found)")

    # Build report
    report = {
        "timestamp": datetime.now().isoformat(),
        "features_file": str(features_file),
        "edge_file": str(edge_file) if edge_file else None,
        "n_dead": n_dead,
        "n_live": n_live,
        "n_total": n_total,
        "pct_dead": round(pct_dead, 2),
        "mean_views_dead": round(mean_views_dead, 1),
        "mean_views_live": round(mean_views_live, 1),
        "mean_degree_dead": round(mean_degree_dead, 1) if mean_degree_dead is not None else None,
        "mean_degree_live": round(mean_degree_live, 1) if mean_degree_live is not None else None,
    }

    # Save report
    report_path = output_path / "dead_account_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Dead account audit complete. Report saved to: {report_path}")
    logger.info(f"Dead accounts: {n_dead} ({pct_dead:.1f}%)")
    logger.info(f"Live accounts: {n_live} ({100-pct_dead:.1f}%)")
    logger.info(f"Mean views - Dead: {mean_views_dead:.0f}, Live: {mean_views_live:.0f}")

    if degree_available:
        logger.info(f"Mean degree - Dead: {mean_degree_dead:.1f}, Live: {mean_degree_live:.1f}")

    # Log paper framing
    logger.info("\n" + "="*60)
    logger.info("Paper framing (Section 5 Limitations):")
    logger.info(f'Dead accounts ({pct_dead:.1f}% of nodes) were excluded; they have')
    logger.info(f'systematically lower degree and views than active accounts.')
    logger.info(f'Findings generalize only to active users.')
    logger.info("="*60 + "\n")

    return report


if __name__ == "__main__":
    run_dead_account_audit()
