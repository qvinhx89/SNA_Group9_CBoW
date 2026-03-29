"""
Typology Construction Module
============================
Build 2x2 typology labels from SIS and views thresholds.

Stage 3 contract:
- `data/processed/typology_labels.parquet`
- `outputs/stage3/typology_summary.json`
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "src/config/base.yaml") -> dict:
	with open(config_path, "r", encoding="utf-8") as f:
		return yaml.safe_load(f)


def _compute_top_flags(series: pd.Series, threshold: float) -> pd.Series:
	n = len(series)
	top_k = max(1, int(n * threshold))
	cutoff = series.nlargest(top_k).min()
	return series >= cutoff


def build_typology(
	sis_path: str = "data/processed/sis_table.parquet",
	centrality_path: str = "data/processed/centrality_table.parquet",
	output_data_dir: str = "data/processed",
	output_dir: str = "outputs/stage3",
	config_path: str = "src/config/base.yaml",
) -> pd.DataFrame:
	"""Build typology labels and save summary artifacts."""
	sis_file = Path(sis_path)
	centrality_file = Path(centrality_path)
	data_dir = Path(output_data_dir)
	out_dir = Path(output_dir)

	data_dir.mkdir(parents=True, exist_ok=True)
	out_dir.mkdir(parents=True, exist_ok=True)

	if not sis_file.exists():
		raise FileNotFoundError(f"Missing SIS file: {sis_file}")
	if not centrality_file.exists():
		raise FileNotFoundError(f"Missing centrality file: {centrality_file}")

	config = load_config(config_path)
	threshold = float(config.get("typology", {}).get("threshold_default", 0.20))

	sis_df = pd.read_parquet(sis_file)
	cent_df = pd.read_parquet(centrality_file)

	merged = sis_df.merge(
		cent_df[["node_id", "views"]], on="node_id", how="left", suffixes=("", "_cent")
	)
	merged["views"] = pd.to_numeric(merged["views"], errors="coerce").fillna(0.0)

	merged["sis_high"] = _compute_top_flags(merged["sis_score"], threshold)
	merged["views_high"] = _compute_top_flags(merged["views"], threshold)

	def assign_label(row: pd.Series) -> str:
		if row["sis_high"] and row["views_high"]:
			return "true"
		if row["sis_high"] and (not row["views_high"]):
			return "hidden"
		if (not row["sis_high"]) and row["views_high"]:
			return "overrated"
		return "non"

	merged["typology_label"] = merged.apply(assign_label, axis=1)

	labels_df = merged[["node_id", "typology_label", "sis_score", "views", "sis_high", "views_high"]].copy()
	labels_out = data_dir / "typology_labels.parquet"
	labels_df.to_parquet(labels_out, index=False)

	distribution = labels_df["typology_label"].value_counts().to_dict()
	distribution_pct = (
		labels_df["typology_label"].value_counts(normalize=True).mul(100).round(3).to_dict()
	)

	summary = {
		"timestamp": datetime.now().isoformat(),
		"threshold": threshold,
		"n_nodes": int(len(labels_df)),
		"distribution": {k: int(v) for k, v in distribution.items()},
		"distribution_pct": {k: float(v) for k, v in distribution_pct.items()},
		"outputs": {
			"typology_labels": str(labels_out),
		},
	}

	summary_path = out_dir / "typology_summary.json"
	with open(summary_path, "w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2)

	labels_df.to_csv(out_dir / "typology_distribution.csv", index=False)
	logger.info("Typology labels generated.")
	return labels_df


if __name__ == "__main__":
	build_typology()
