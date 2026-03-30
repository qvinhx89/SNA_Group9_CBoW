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
from typing import Dict, List, Sequence, Tuple

import pandas as pd


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


EDGE_NAME_HINTS = ("edge", "edges", "link")
NODE_NAME_HINTS = ("target", "node", "nodes", "meta", "attribute", "feature", "features")

EDGE_COLUMN_CANDIDATE_PAIRS = (
	("source", "target"),
	("src", "dst"),
	("from", "to"),
	("u", "v"),
	("numeric_id_1", "numeric_id_2"),
)

NODE_ID_CANDIDATES = (
	"node_id",
	"numeric_id",
	"id",
	"new_id",
	"user_id",
	"channel_id",
)

VIEWS_CANDIDATES = (
	"views",
	"view",
	"n_views",
	"view_count",
	"followers",
)

MIN_NODE_EDGE_OVERLAP_RATIO = 0.95


def _detect_separator(file_path: Path) -> str:
	if file_path.suffix.lower() == ".tsv":
		return "\t"
	return ","


def _list_candidate_files(raw_dir: Path) -> List[Path]:
	return sorted(
		[
			p for p in raw_dir.iterdir()
			if p.is_file() and p.suffix.lower() in {".csv", ".tsv", ".txt"}
		]
	)


def _read_header_columns(file_path: Path) -> List[str]:
	sep = _detect_separator(file_path)
	try:
		head = pd.read_csv(file_path, sep=sep, nrows=0)
	except Exception as exc:
		logger.warning("Skipping file %s due to header read error: %s", file_path, exc)
		return []
	return [str(c).strip() for c in head.columns]


def _pick_edge_file(candidates: Sequence[Path]) -> Path:
	best: Tuple[int, Path] | None = None

	for path in candidates:
		columns = _read_header_columns(path)
		if len(columns) < 2:
			continue

		lowered = {c.lower() for c in columns}
		score = 0
		if any((src in lowered and dst in lowered) for src, dst in EDGE_COLUMN_CANDIDATE_PAIRS):
			score += 100
		if any(h in path.name.lower() for h in EDGE_NAME_HINTS):
			score += 10
		if len(columns) == 2:
			score += 5

		if best is None or score > best[0]:
			best = (score, path)

	if best is None:
		raise FileNotFoundError(
			"No valid edge file found in data/raw. Expected a csv/tsv/txt with at least 2 columns."
		)

	return best[1]


def _pick_node_file(candidates: Sequence[Path], edge_file: Path) -> Path:
	best: Tuple[int, Path, bool, bool] | None = None

	for path in candidates:
		if path == edge_file:
			continue

		columns = _read_header_columns(path)
		if not columns:
			continue

		lowered = {c.lower() for c in columns}
		has_node_id = any(c in lowered for c in NODE_ID_CANDIDATES)
		has_views = any(c in lowered for c in VIEWS_CANDIDATES)

		score = 0
		if has_node_id:
			score += 80
		if "numeric_id" in lowered:
			score += 20
		if has_views:
			score += 50
		if any(h in path.name.lower() for h in NODE_NAME_HINTS):
			score += 10

		if best is None or score > best[0]:
			best = (score, path, has_node_id, has_views)

	if best is None:
		raise FileNotFoundError(
			"No node metadata file candidate found in data/raw. Expected a features/nodes csv/tsv/txt file."
		)

	_, node_file, has_node_id, has_views = best
	if not has_node_id or not has_views:
		raise ValueError(
			"Node metadata file is invalid. Expected both an ID column "
			f"{list(NODE_ID_CANDIDATES)} and a views column {list(VIEWS_CANDIDATES)} in {node_file}."
		)

	return node_file


def _infer_edge_columns(columns: Sequence[str], file_path: Path) -> Tuple[str, str]:
	lowered = {c.lower(): c for c in columns}
	for source_cand, target_cand in EDGE_COLUMN_CANDIDATE_PAIRS:
		if source_cand in lowered and target_cand in lowered:
			return lowered[source_cand], lowered[target_cand]

	if len(columns) >= 2:
		logger.warning(
			"Edge columns not explicitly detected in %s. Falling back to first two columns.",
			file_path,
		)
		return columns[0], columns[1]

	raise ValueError(f"Cannot infer source/target columns from {file_path}")


def _standardize_edges(edge_file: Path) -> pd.DataFrame:
	sep = _detect_separator(edge_file)
	df = pd.read_csv(edge_file, sep=sep)
	source_col, target_col = _infer_edge_columns(df.columns.tolist(), edge_file)

	out = df[[source_col, target_col]].copy()
	out.columns = ["source", "target"]
	out["source"] = out["source"].astype(str)
	out["target"] = out["target"].astype(str)
	return out


def _infer_node_columns(columns: Sequence[str], node_file: Path) -> Tuple[str, str]:
	lowered = {c.lower(): c for c in columns}

	node_col = None
	for cand in NODE_ID_CANDIDATES:
		if cand in lowered:
			node_col = lowered[cand]
			break

	if node_col is None:
		raise ValueError(
			f"Cannot infer node ID column from {node_file}. Expected one of {list(NODE_ID_CANDIDATES)}."
		)

	views_col = None
	for cand in VIEWS_CANDIDATES:
		if cand in lowered:
			views_col = lowered[cand]
			break

	if views_col is None:
		raise ValueError(
			f"Cannot infer views column from {node_file}. Expected one of {list(VIEWS_CANDIDATES)}."
		)

	return node_col, views_col


def _validate_node_edge_overlap(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> Dict[str, float]:
	edge_nodes = pd.Index(
		pd.concat([edges_df["source"], edges_df["target"]], ignore_index=True)
		.astype(str)
		.unique()
	)
	node_ids = pd.Index(nodes_df["node_id"].astype(str).unique())
	overlap_count = int(len(edge_nodes.intersection(node_ids)))
	total_edge_nodes = int(len(edge_nodes))
	ratio = float(overlap_count / total_edge_nodes) if total_edge_nodes > 0 else 0.0

	if ratio < MIN_NODE_EDGE_OVERLAP_RATIO:
		raise ValueError(
			"Node metadata mapping failed validation: "
			f"overlap ratio with edge nodes is {ratio:.4f} (expected >= {MIN_NODE_EDGE_OVERLAP_RATIO:.2f})."
		)

	return {
		"edge_nodes_total": float(total_edge_nodes),
		"node_ids_total": float(len(node_ids)),
		"node_edge_overlap_count": float(overlap_count),
		"node_edge_overlap_ratio": ratio,
	}


def _standardize_nodes(node_file: Path, edges_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str | float]]:

	sep = _detect_separator(node_file)
	df = pd.read_csv(node_file, sep=sep)
	node_col, views_col = _infer_node_columns(df.columns.tolist(), node_file)

	node_series = df[node_col].astype("string").str.strip()
	out = pd.DataFrame({"node_id": node_series})
	out = out.dropna(subset=["node_id"])
	out = out[out["node_id"] != ""]
	out["node_id"] = out["node_id"].astype(str)
	out["views"] = pd.to_numeric(df.loc[out.index, views_col], errors="coerce")

	out = out.drop_duplicates(subset=["node_id"])
	overlap_stats = _validate_node_edge_overlap(out, edges_df)

	stats: Dict[str, str | float] = {
		"node_id_column": node_col,
		"views_column": views_col,
		"n_node_rows_raw": float(len(df)),
		"n_node_rows_after_clean": float(len(out)),
	}
	stats.update(overlap_stats)
	return out, stats


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

	candidate_files = _list_candidate_files(raw_path)
	if not candidate_files:
		raise FileNotFoundError(
			"No raw input files found in data/raw. Expected csv/tsv/txt files."
		)

	edge_file = _pick_edge_file(candidate_files)
	node_file = _pick_node_file(candidate_files, edge_file)

	logger.info("Using edge file: %s", edge_file)
	logger.info("Using node file: %s", node_file)

	edges_df = _standardize_edges(edge_file)
	nodes_df, node_stats = _standardize_nodes(node_file, edges_df)

	edges_out = interim_path / "edges_raw.parquet"
	nodes_out = interim_path / "nodes_raw.parquet"
	edges_df.to_parquet(edges_out, index=False)
	nodes_df.to_parquet(nodes_out, index=False)

	report = {
		"timestamp": datetime.now().isoformat(),
		"raw_dir": str(raw_path),
		"edge_file": str(edge_file),
		"node_file": str(node_file),
		"n_edges_raw": int(len(edges_df)),
		"n_nodes_raw": int(len(nodes_df)),
		"n_nodes_unique": int(nodes_df["node_id"].nunique()),
		"views_available": int(nodes_df["views"].notna().sum()),
		"node_id_column": str(node_stats["node_id_column"]),
		"views_column": str(node_stats["views_column"]),
		"edge_nodes_total": int(node_stats["edge_nodes_total"]),
		"node_edge_overlap_count": int(node_stats["node_edge_overlap_count"]),
		"node_edge_overlap_ratio": float(node_stats["node_edge_overlap_ratio"]),
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
