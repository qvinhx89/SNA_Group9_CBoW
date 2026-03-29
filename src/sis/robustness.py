"""
Robustness Analysis Module
==========================
Evaluate typology stability under threshold sensitivity.

Stage 3 contract:
- `outputs/stage3/robustness_summary.json`
- `outputs/stage3/robustness_thresholds.csv`
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Set

import pandas as pd
import yaml


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "src/config/base.yaml") -> dict:
	with open(config_path, "r", encoding="utf-8") as f:
		return yaml.safe_load(f)


def _top_flags(series: pd.Series, threshold: float) -> pd.Series:
	n = len(series)
	top_k = max(1, int(n * threshold))
	cutoff = series.nlargest(top_k).min()
	return series >= cutoff


def _hidden_nodes(df: pd.DataFrame, threshold: float) -> Set[str]:
	sis_high = _top_flags(df["sis_score"], threshold)
	views_high = _top_flags(df["views"], threshold)
	hidden = df[sis_high & (~views_high)]["node_id"].astype(str)
	return set(hidden.tolist())


def _jaccard(a: Set[str], b: Set[str]) -> float:
	union = a.union(b)
	if not union:
		return 1.0
	return len(a.intersection(b)) / len(union)


def run_robustness(
	sis_path: str = "data/processed/sis_table.parquet",
	centrality_path: str = "data/processed/centrality_table.parquet",
	output_dir: str = "outputs/stage3",
	config_path: str = "src/config/base.yaml",
) -> pd.DataFrame:
	"""Run threshold sensitivity and hidden-set Jaccard stability analysis."""
	sis_file = Path(sis_path)
	centrality_file = Path(centrality_path)
	out_dir = Path(output_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	if not sis_file.exists():
		raise FileNotFoundError(f"Missing SIS file: {sis_file}")
	if not centrality_file.exists():
		raise FileNotFoundError(f"Missing centrality file: {centrality_file}")

	config = load_config(config_path)
	thresholds = config.get("typology", {}).get("thresholds_sensitivity", [0.15, 0.20, 0.25])
	base_threshold = float(config.get("typology", {}).get("threshold_default", 0.20))
	jaccard_target = float(config.get("typology", {}).get("jaccard_target", 0.70))

	sis_df = pd.read_parquet(sis_file)
	cent_df = pd.read_parquet(centrality_file)
	df = sis_df.merge(cent_df[["node_id", "views"]], on="node_id", how="left")
	df["views"] = pd.to_numeric(df["views"], errors="coerce").fillna(0.0)

	hidden_sets: Dict[float, Set[str]] = {
		float(t): _hidden_nodes(df, float(t)) for t in thresholds
	}

	if base_threshold not in hidden_sets:
		hidden_sets[base_threshold] = _hidden_nodes(df, base_threshold)

	base_hidden = hidden_sets[base_threshold]
	rows = []
	for t in sorted(hidden_sets.keys()):
		cur = hidden_sets[t]
		rows.append(
			{
				"threshold": t,
				"hidden_count": len(cur),
				"jaccard_with_base": _jaccard(cur, base_hidden),
			}
		)

	result_df = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
	result_df["jaccard_pass"] = result_df["jaccard_with_base"] >= jaccard_target

	csv_path = out_dir / "robustness_thresholds.csv"
	result_df.to_csv(csv_path, index=False)

	summary = {
		"timestamp": datetime.now().isoformat(),
		"base_threshold": base_threshold,
		"jaccard_target": jaccard_target,
		"all_pass": bool(result_df["jaccard_pass"].all()),
		"rows": result_df.to_dict(orient="records"),
	}

	summary_path = out_dir / "robustness_summary.json"
	with open(summary_path, "w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2)

	logger.info("Robustness analysis complete.")
	return result_df


if __name__ == "__main__":
	run_robustness()
