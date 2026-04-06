"""Enrich node_attributes with v3-required columns.

Adds/updates columns on data/processed/node_attributes.parquet:
- degree (from centrality_table.parquet)
- life_time, language (from raw features file via numeric_id -> node_id)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enrich node_attributes for MAPR2026 v3")
    p.add_argument("--node-attrs", default="data/processed/node_attributes.parquet")
    p.add_argument("--centrality", default="data/processed/centrality_table.parquet")
    p.add_argument("--features", default="data/raw/large_twitch_features.csv")
    p.add_argument("--out", default="data/processed/node_attributes.parquet")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    node_attrs_path = Path(args.node_attrs)
    centrality_path = Path(args.centrality)
    features_path = Path(args.features)
    out_path = Path(args.out)

    if not node_attrs_path.exists():
        raise FileNotFoundError(f"Missing node attributes: {node_attrs_path}")
    if not centrality_path.exists():
        raise FileNotFoundError(f"Missing centrality table: {centrality_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Missing features file: {features_path}")

    node_attrs = pd.read_parquet(node_attrs_path)
    if "node_id" not in node_attrs.columns:
        raise ValueError("node_attributes.parquet must contain node_id")
    node_attrs = node_attrs.copy()
    node_attrs["node_id"] = node_attrs["node_id"].astype(str)

    centrality = pd.read_parquet(centrality_path, columns=["node_id", "degree"])
    centrality = centrality.drop_duplicates(subset=["node_id"]).copy()
    centrality["node_id"] = centrality["node_id"].astype(str)

    features = pd.read_csv(features_path, usecols=["numeric_id", "life_time", "language"])
    features = features.rename(columns={"numeric_id": "node_id"}).copy()
    features["node_id"] = features["node_id"].astype(str)
    features["life_time"] = pd.to_numeric(features["life_time"], errors="coerce")
    features["language"] = features["language"].astype(str)
    features = features.drop_duplicates(subset=["node_id"])

    out = node_attrs.merge(centrality, on="node_id", how="left", suffixes=("", "_new"))
    if "degree_new" in out.columns:
        out["degree"] = out["degree_new"].astype(float)
        out = out.drop(columns=["degree_new"])

    out = out.merge(features, on="node_id", how="left", suffixes=("", "_new"))
    if "life_time_new" in out.columns:
        out["life_time"] = out["life_time_new"]
        out = out.drop(columns=["life_time_new"])
    if "language_new" in out.columns:
        out["language"] = out["language_new"]
        out = out.drop(columns=["language_new"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    print(
        "[OK] Enriched node_attributes: "
        f"{out_path} (rows={len(out):,}, cols={list(out.columns)})"
    )


if __name__ == "__main__":
    main()
