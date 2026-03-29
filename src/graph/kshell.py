"""
K-shell Computation Module
==========================
Compute k-shell (core number) per node and align with centrality table.

Stage 2 contract:
- `data/processed/kshell_table.parquet`
- Update `data/processed/centrality_table.parquet` with `kshell` column
- `outputs/stage2/kshell_metrics.json`
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import networkx as nx
import pandas as pd


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_kshell(
	graph_path: str = "data/processed/graph_active.edgelist",
	centrality_path: str = "data/processed/centrality_table.parquet",
	output_data_dir: str = "data/processed",
	output_dir: str = "outputs/stage2",
) -> pd.DataFrame:
	"""Compute k-shell values and merge into centrality table."""
	graph_file = Path(graph_path)
	centrality_file = Path(centrality_path)
	data_dir = Path(output_data_dir)
	out_dir = Path(output_dir)

	data_dir.mkdir(parents=True, exist_ok=True)
	out_dir.mkdir(parents=True, exist_ok=True)

	if not graph_file.exists():
		raise FileNotFoundError(f"Missing graph file: {graph_file}")
	if not centrality_file.exists():
		raise FileNotFoundError(f"Missing centrality file: {centrality_file}")

	logger.info("Loading graph from %s", graph_file)
	G = nx.read_edgelist(graph_file)

	logger.info("Computing core numbers (k-shell)")
	core_map = nx.core_number(G)

	kshell_df = pd.DataFrame(
		{
			"node_id": [str(k) for k in core_map.keys()],
			"kshell": [int(v) for v in core_map.values()],
		}
	)

	kshell_out = data_dir / "kshell_table.parquet"
	kshell_df.to_parquet(kshell_out, index=False)

	centrality_df = pd.read_parquet(centrality_file)
	centrality_df["node_id"] = centrality_df["node_id"].astype(str)
	merged = centrality_df.merge(kshell_df, on="node_id", how="left")

	if merged["kshell"].isna().any():
		merged["kshell"] = merged["kshell"].fillna(0)
	merged["kshell"] = merged["kshell"].astype(int)
	merged.to_parquet(centrality_file, index=False)

	metrics = {
		"timestamp": datetime.now().isoformat(),
		"n_nodes_graph": int(G.number_of_nodes()),
		"n_nodes_with_kshell": int(len(kshell_df)),
		"kshell_min": int(kshell_df["kshell"].min()),
		"kshell_max": int(kshell_df["kshell"].max()),
		"kshell_mean": float(kshell_df["kshell"].mean()),
		"outputs": {
			"kshell_table": str(kshell_out),
			"centrality_table": str(centrality_file),
		},
	}

	metrics_path = out_dir / "kshell_metrics.json"
	with open(metrics_path, "w", encoding="utf-8") as f:
		json.dump(metrics, f, indent=2)

	logger.info("K-shell computation complete.")
	return merged


if __name__ == "__main__":
	compute_kshell()
