"""MAPR2026 v3 — IC label uncertainty summary (post-Day1).

Owner: Person 1

Purpose
-------
Export uncertainty information so Person 2/3 can identify boundary nodes and
avoid over-interpreting noisy top-10 classification labels.

Outputs
-------
- outputs/day1_benchmark/ic_label_uncertainty.json
- data/processed/ic_scores_primary_with_ci.parquet (default)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from _shared import PATHS, now_iso, require_columns, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IC uncertainty report + CI-enriched parquet")
    p.add_argument("--ic", default=PATHS.ic_scores)
    p.add_argument("--cls", default=PATHS.classification_labels)
    p.add_argument("--out-json", default=f"{PATHS.day1_dir}/ic_label_uncertainty.json")
    p.add_argument("--out-ic-ci", default="data/processed/ic_scores_primary_with_ci.parquet")
    p.add_argument("--z", type=float, default=1.96, help="Z-score for confidence interval")
    p.add_argument("--top-pct", type=float, default=0.10)
    p.add_argument("--stable-pos-prob", type=float, default=0.90)
    p.add_argument("--stable-neg-prob", type=float, default=0.10)
    p.add_argument("--boundary-top", type=int, default=20, help="How many boundary nodes to list")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    df_ic = pd.read_parquet(args.ic)
    df_cls = pd.read_parquet(args.cls)

    require_columns(df_ic, ["node_id", "ic_score_mean", "ic_score_std", "n_runs"], "ic_scores")
    require_columns(df_cls, ["node_id"], "classification_labels")
    if "y_top10" not in df_cls.columns:
        if "y_top10_consensus" in df_cls.columns:
            df_cls = df_cls.rename(columns={"y_top10_consensus": "y_top10"})
        else:
            require_columns(df_cls, ["y_top10"], "classification_labels")

    df = df_ic.merge(df_cls, on="node_id", how="inner")
    if len(df) != len(df_ic):
        raise ValueError(
            "classification_labels node set must match ic_scores node set for uncertainty export"
        )

    means = df["ic_score_mean"].astype(float).to_numpy()
    stds = df["ic_score_std"].astype(float).to_numpy()
    n_runs = df["n_runs"].astype(float).to_numpy()
    n_runs_safe = np.maximum(n_runs, 1.0)

    se = stds / np.sqrt(n_runs_safe)
    ci_low = means - float(args.z) * se
    ci_high = means + float(args.z) * se

    threshold = float(np.quantile(means, 1.0 - float(args.top_pct)))
    zscore = np.empty_like(means, dtype=float)
    np.divide(threshold - means, se, out=zscore, where=se > 0.0)
    zscore = np.where(se > 0.0, zscore, np.where(means >= threshold, -np.inf, np.inf))
    p_above = norm.sf(zscore)

    crossing = (ci_low <= threshold) & (ci_high >= threshold)
    y_top = df["y_top10"].astype(int).to_numpy() == 1

    stable_pos = p_above >= float(args.stable_pos_prob)
    stable_neg = p_above <= float(args.stable_neg_prob)
    ambiguous = ~(stable_pos | stable_neg)

    out_df = df_ic.copy()
    out_df["ic_se"] = se
    out_df["ic_ci_lower"] = ci_low
    out_df["ic_ci_upper"] = ci_high
    out_df["p_above_top10_threshold"] = p_above
    out_df["is_boundary_ci_crossing_threshold"] = crossing.astype(int)

    out_path = Path(args.out_ic_ci)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    boundary_df = df[["node_id", "ic_score_mean", "ic_score_std", "n_runs", "y_top10"]].copy()
    boundary_df["distance_to_threshold"] = np.abs(boundary_df["ic_score_mean"] - threshold)
    boundary_df["ic_se"] = se
    boundary_df["ic_ci_lower"] = ci_low
    boundary_df["ic_ci_upper"] = ci_high
    boundary_df["p_above_top10_threshold"] = p_above

    payload = {
        "timestamp": now_iso(),
        "config": {
            "ic_path": str(args.ic),
            "classification_path": str(args.cls),
            "top_pct": float(args.top_pct),
            "z": float(args.z),
            "stable_pos_prob": float(args.stable_pos_prob),
            "stable_neg_prob": float(args.stable_neg_prob),
        },
        "thresholds": {
            "top10_ic_score_mean_threshold": threshold,
            "n_labeled": int(len(df)),
            "n_runs_unique": sorted(df["n_runs"].astype(int).unique().tolist()),
        },
        "summary": {
            "n_boundary_ci_crossing_threshold": int(crossing.sum()),
            "boundary_ratio": float(crossing.mean()),
            "n_boundary_among_y_top10": int((crossing & y_top).sum()),
            "n_boundary_among_y_non_top10": int((crossing & (~y_top)).sum()),
            "n_stable_positive": int(stable_pos.sum()),
            "n_stable_negative": int(stable_neg.sum()),
            "n_ambiguous": int(ambiguous.sum()),
            "ambiguous_ratio": float(ambiguous.mean()),
        },
        "boundary_nodes_closest_to_threshold": boundary_df.sort_values("distance_to_threshold")
        .head(int(args.boundary_top))
        .to_dict(orient="records"),
        "enriched_ic_output": str(out_path),
    }

    write_json(args.out_json, payload)
    print(f"[OK] Wrote uncertainty report: {Path(args.out_json)}")
    print(f"[OK] Wrote CI-enriched IC parquet: {out_path}")


if __name__ == "__main__":
    main()
