"""
Raw Data Loading Module
=======================
Standardize raw graph files into interim parquet artifacts.

Stage 0 contract:
- Produce `data/interim/edges_raw.parquet`
- Produce `data/interim/nodes_raw.parquet`
- Produce `outputs/stage0_data_quality/raw_load_report.json`
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


EDGE_NAME_HINTS = ("edge", "edges", "link")
NODE_NAME_HINTS = ("target", "node", "nodes", "meta", "attribute")


def _detect_separator(file_path: Path) -> str:
	if file_path.suffix.lower() == ".tsv":
		return "\t"
	return ","


def _find_candidate_files(raw_dir: Path) -> Tuple[List[Path], List[Path]]:
	candidates = [
		p for p in raw_dir.iterdir()
		if p.is_file() and p.suffix.lower() in {".csv", ".tsv", ".txt"}
	]

	edge_files = [p for p in candidates if any(h in p.name.lower() for h in EDGE_NAME_HINTS)]
	node_files = [p for p in candidates if any(h in p.name.lower() for h in NODE_NAME_HINTS)]
	return edge_files, node_files


def _standardize_edges(edge_file: Path) -> pd.DataFrame:
	sep = _detect_separator(edge_file)
	df = pd.read_csv(edge_file, sep=sep)
	lowered = {c.lower(): c for c in df.columns}

	source_col = None
	target_col = None

	for cand in ["source", "src", "from", "u"]:
		if cand in lowered:
			source_col = lowered[cand]
			break

	for cand in ["target", "dst", "to", "v"]:
		if cand in lowered:
			target_col = lowered[cand]
			break

	if source_col is None or target_col is None:
		if df.shape[1] >= 2:
			source_col, target_col = df.columns[:2]
		else:
			raise ValueError(f"Cannot infer source/target columns from {edge_file}")

	out = df[[source_col, target_col]].copy()
	out.columns = ["source", "target"]
	out["source"] = out["source"].astype(str)
	out["target"] = out["target"].astype(str)
	return out


def _standardize_nodes(node_file: Optional[Path], edges_df: pd.DataFrame) -> pd.DataFrame:
	if node_file is None:
		nodes = pd.Index(edges_df["source"]).union(pd.Index(edges_df["target"]))
		return pd.DataFrame({"node_id": nodes.astype(str), "views": pd.NA})

	sep = _detect_separator(node_file)
	df = pd.read_csv(node_file, sep=sep)
	lowered = {c.lower(): c for c in df.columns}

	node_col = None
	for cand in ["node_id", "id", "new_id", "user_id", "channel_id"]:
		if cand in lowered:
			node_col = lowered[cand]
			break
	if node_col is None:
		node_col = df.columns[0]

	views_col = None
	for cand in ["views", "view", "n_views", "view_count", "followers"]:
		if cand in lowered:
			views_col = lowered[cand]
			break

	out = pd.DataFrame({"node_id": df[node_col].astype(str)})
	if views_col is not None:
		out["views"] = pd.to_numeric(df[views_col], errors="coerce")
	else:
		out["views"] = pd.NA

	out = out.drop_duplicates(subset=["node_id"])
	return out


def load_raw_data(
	raw_dir: str = "data/raw",
	interim_dir: str = "data/interim",
	quality_dir: str = "outputs/stage0_data_quality",
) -> dict:
	"""Load and standardize raw input files for Stage 0."""
	raw_path = Path(raw_dir)
	interim_path = Path(interim_dir)
	quality_path = Path(quality_dir)

	interim_path.mkdir(parents=True, exist_ok=True)
	quality_path.mkdir(parents=True, exist_ok=True)

	if not raw_path.exists():
		raise FileNotFoundError(f"Raw directory does not exist: {raw_path}")

	edge_files, node_files = _find_candidate_files(raw_path)
	if not edge_files:
		raise FileNotFoundError(
			"No edge file found in data/raw. Expected a csv/tsv/txt with name containing edge/edges/link."
		)

	edge_file = sorted(edge_files)[0]
	node_file = sorted(node_files)[0] if node_files else None

	logger.info("Using edge file: %s", edge_file)
	if node_file is not None:
		logger.info("Using node file: %s", node_file)
	else:
		logger.warning("No node metadata file found. Views will be missing.")

	edges_df = _standardize_edges(edge_file)
	nodes_df = _standardize_nodes(node_file, edges_df)

	edges_out = interim_path / "edges_raw.parquet"
	nodes_out = interim_path / "nodes_raw.parquet"
	edges_df.to_parquet(edges_out, index=False)
	nodes_df.to_parquet(nodes_out, index=False)

	report = {
		"timestamp": datetime.now().isoformat(),
		"raw_dir": str(raw_path),
		"edge_file": str(edge_file),
		"node_file": str(node_file) if node_file else None,
		"n_edges_raw": int(len(edges_df)),
		"n_nodes_raw": int(len(nodes_df)),
		"views_available": int(nodes_df["views"].notna().sum()),
		"outputs": {
			"edges_raw": str(edges_out),
			"nodes_raw": str(nodes_out),
		},
	}

	report_path = quality_path / "raw_load_report.json"
	with open(report_path, "w", encoding="utf-8") as f:
		json.dump(report, f, indent=2)

	logger.info("Saved standardized interim files and report.")
	return report


if __name__ == "__main__":
	load_raw_data()
