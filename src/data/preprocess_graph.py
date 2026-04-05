"""
Graph Preprocessing Module
==========================
Clean raw/interim data and export processed graph artifacts.

Stage 0 contract:
- `data/processed/graph_active.edgelist`
- `data/processed/node_attributes.parquet`
- quality artifacts under `outputs/stage0_data_quality/`
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


def _canonical_undirected_edges(edges: pd.DataFrame) -> pd.DataFrame:
	cleaned = edges.copy()
	cleaned = cleaned.dropna(subset=["source", "target"])
	cleaned["source"] = cleaned["source"].astype(str)
	cleaned["target"] = cleaned["target"].astype(str)

	cleaned = cleaned[cleaned["source"] != cleaned["target"]].copy()

	# Canonical ordering avoids duplicate undirected edges (u,v) and (v,u).
	mins = cleaned[["source", "target"]].min(axis=1)
	maxs = cleaned[["source", "target"]].max(axis=1)
	cleaned["source"] = mins
	cleaned["target"] = maxs
	cleaned = cleaned.drop_duplicates(subset=["source", "target"]).reset_index(drop=True)
	return cleaned


def preprocess_graph(
	edges_path: str = "data/interim/edges_raw.parquet",
	nodes_path: str = "data/interim/nodes_raw.parquet",
	processed_dir: str = "data/processed",
	quality_dir: str = "outputs/stage0_data_quality",
) -> dict:
	"""Build a cleaned active graph and aligned node attribute table."""
	edges_file = Path(edges_path)
	nodes_file = Path(nodes_path)
	processed_path = Path(processed_dir)
	quality_path = Path(quality_dir)

	processed_path.mkdir(parents=True, exist_ok=True)
	quality_path.mkdir(parents=True, exist_ok=True)

	if not edges_file.exists():
		raise FileNotFoundError(
			f"Missing interim edge file: {edges_file}. Run src/data/load_raw.py first."
		)
	if not nodes_file.exists():
		raise FileNotFoundError(
			f"Missing interim node file: {nodes_file}. Run src/data/load_raw.py first."
		)

	edges_raw = pd.read_parquet(edges_file)
	nodes_raw = pd.read_parquet(nodes_file)

	if "source" not in edges_raw.columns or "target" not in edges_raw.columns:
		raise ValueError("edges_raw.parquet must contain source and target columns")
	if "node_id" not in nodes_raw.columns:
		raise ValueError("nodes_raw.parquet must contain node_id column")

	n_edges_before = len(edges_raw)
	edges_clean = _canonical_undirected_edges(edges_raw)
	n_edges_after = len(edges_clean)

	G = nx.from_pandas_edgelist(edges_clean, source="source", target="target")
	n_nodes_full = G.number_of_nodes()

	if n_nodes_full == 0:
		raise ValueError("Graph is empty after preprocessing")

	largest_cc_nodes = max(nx.connected_components(G), key=len)
	G_active = G.subgraph(largest_cc_nodes).copy()

	active_nodes = pd.Index([str(n) for n in G_active.nodes()])
	attrs = nodes_raw.copy()
	attrs["node_id"] = attrs["node_id"].astype(str)
	attrs = attrs.drop_duplicates(subset=["node_id"])
	attrs = attrs[attrs["node_id"].isin(active_nodes)].copy()

	if "views" not in attrs.columns:
		attrs["views"] = pd.NA

	attrs["views"] = pd.to_numeric(attrs["views"], errors="coerce")
	if attrs["views"].notna().any():
		fill_value = float(attrs["views"].median())
	else:
		fill_value = 0.0
	attrs["views"] = attrs["views"].fillna(fill_value)

	# Ensure every active node has attributes.
	attrs = pd.DataFrame({"node_id": active_nodes}).merge(attrs, on="node_id", how="left")
	attrs["views"] = pd.to_numeric(attrs["views"], errors="coerce").fillna(fill_value)

	graph_out = processed_path / "graph_active.edgelist"
	attrs_out = processed_path / "node_attributes.parquet"
	nx.write_edgelist(G_active, graph_out, data=False)
	attrs.to_parquet(attrs_out, index=False)

	metrics = {
		"timestamp": datetime.now().isoformat(),
		"n_edges_before_cleaning": int(n_edges_before),
		"n_edges_after_cleaning": int(n_edges_after),
		"n_nodes_full_graph": int(n_nodes_full),
		"n_nodes_active_graph": int(G_active.number_of_nodes()),
		"n_edges_active_graph": int(G_active.number_of_edges()),
		"largest_component_ratio": float(G_active.number_of_nodes() / n_nodes_full),
		"n_missing_views_filled": int((attrs["views"] == fill_value).sum()),
	}

	metrics_path = quality_path / "metrics.json"
	report_path = quality_path / "preprocess_report.json"
	with open(metrics_path, "w", encoding="utf-8") as f:
		json.dump(metrics, f, indent=2)
	with open(report_path, "w", encoding="utf-8") as f:
		json.dump(
			{
				"timestamp": datetime.now().isoformat(),
				"inputs": {"edges": str(edges_file), "nodes": str(nodes_file)},
				"outputs": {"graph_active": str(graph_out), "node_attributes": str(attrs_out)},
			},
			f,
			indent=2,
		)

	logger.info("Preprocessing complete. Active graph and node attributes saved.")
	return metrics


if __name__ == "__main__":
	preprocess_graph()
