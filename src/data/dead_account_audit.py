"""Dead account audit for MAPR2026 v3 pre-Day1 prerequisites.

Contract output:
- outputs/stage0_data_quality/dead_account_report.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate dead account audit report")
    p.add_argument("--features", default="data/raw/large_twitch_features.csv")
    p.add_argument("--edges", default="data/raw/large_twitch_edges.csv")
    p.add_argument("--out", default="outputs/stage0_data_quality/dead_account_report.json")
    return p.parse_args()


def _infer_edge_columns(df: pd.DataFrame) -> tuple[str, str]:
    candidates = [
        ("source", "target"),
        ("src", "dst"),
        ("numeric_id_1", "numeric_id_2"),
        ("u", "v"),
    ]
    lowered = {c.lower(): c for c in df.columns}
    for a, b in candidates:
        if a in lowered and b in lowered:
            return lowered[a], lowered[b]
    if len(df.columns) >= 2:
        return str(df.columns[0]), str(df.columns[1])
    raise ValueError("Could not infer two edge endpoint columns")


def _build_degree_series(edges_path: Path) -> pd.Series:
    edges = pd.read_csv(edges_path)
    u_col, v_col = _infer_edge_columns(edges)
    u = edges[u_col].astype(str)
    v = edges[v_col].astype(str)
    degree = pd.concat([u, v], ignore_index=True).value_counts(sort=False)
    degree.index = degree.index.astype(str)
    return degree.astype(float)


def main() -> None:
    args = parse_args()
    features_path = Path(args.features)
    edges_path = Path(args.edges)
    out_path = Path(args.out)

    if not features_path.exists():
        raise FileNotFoundError(f"Missing features file: {features_path}")
    if not edges_path.exists():
        raise FileNotFoundError(f"Missing edges file: {edges_path}")

    features = pd.read_csv(features_path)
    required_cols = {"numeric_id", "dead_account", "views"}
    missing = [c for c in required_cols if c not in features.columns]
    if missing:
        raise ValueError(f"Features file missing required columns: {missing}")

    features = features.copy()
    features["node_id"] = features["numeric_id"].astype(str)
    features["dead_account"] = pd.to_numeric(features["dead_account"], errors="coerce").fillna(0).astype(int)
    features["views"] = pd.to_numeric(features["views"], errors="coerce")

    degree = _build_degree_series(edges_path)
    features["degree"] = features["node_id"].map(degree).fillna(0.0)

    dead = features[features["dead_account"] == 1]
    live = features[features["dead_account"] == 0]

    report = {
        "timestamp": datetime.now().isoformat(),
        "n_dead": int(len(dead)),
        "n_live": int(len(live)),
        "pct_dead": float(len(dead) / len(features) * 100.0) if len(features) else 0.0,
        "mean_degree_dead": float(dead["degree"].mean()) if len(dead) else 0.0,
        "mean_degree_live": float(live["degree"].mean()) if len(live) else 0.0,
        "mean_views_dead": float(dead["views"].mean()) if len(dead) else 0.0,
        "mean_views_live": float(live["views"].mean()) if len(live) else 0.0,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[OK] Wrote dead account report: {out_path}")


if __name__ == "__main__":
    main()
