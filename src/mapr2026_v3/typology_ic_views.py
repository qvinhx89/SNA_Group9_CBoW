"""MAPR2026 v3 — Typology: IC high/low × views high/low.

Owner: Person 2 (typology)

Inputs
------
- data/processed/ic_scores_primary.parquet
- data/processed/node_attributes.parquet

Output (contract)
---------------
- data/processed/typology_labels_ic_views.parquet
  columns: node_id, typology_label, ic_high, views_high, ic_score_mean, views

Scaffold behavior
-----------------
- Default mode raises NotImplementedError (for real labels + quadrant sizing).
- Use --dry-run to build typology from whatever IC file exists (including mock outputs).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _shared import PATHS, ensure_parent, now_iso, require_columns


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MAPR2026 v3 typology (IC×views) scaffold")
    p.add_argument("--ic", default=PATHS.ic_scores)
    p.add_argument("--node-attrs", default=PATHS.node_attributes)
    p.add_argument("--out", default=PATHS.typology)
    p.add_argument("--pct", type=float, default=0.10, help="Top-pct threshold (default 10%)")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _load_ic_or_fallback(ic_path: Path, attrs_path: Path, dry_run: bool) -> pd.DataFrame:
    """Load IC scores. In dry-run mode, falls back to SIS/random if IC file is missing."""
    if ic_path.exists():
        df = pd.read_parquet(ic_path)
        require_columns(df, ["node_id", "ic_score_mean"], "ic_scores")
        return df

    if not dry_run:
        raise FileNotFoundError(f"Missing IC scores: {ic_path}")

    # Dry-run fallback: try SIS table, then random mock.
    sis_path = Path("data/processed/sis_table.parquet")
    if sis_path.exists():
        df_sis = pd.read_parquet(sis_path)
        score_col = next((c for c in ["sis_score", "sis", "score", "pagerank"] if c in df_sis.columns), None)
        if "node_id" in df_sis.columns and score_col is not None:
            print(f"[dry-run] IC scores not found; using {sis_path} column '{score_col}' as mock ic_score_mean")
            return df_sis[["node_id", score_col]].rename(columns={score_col: "ic_score_mean"})

    if attrs_path.exists():
        df_attrs = pd.read_parquet(attrs_path)
        if "node_id" in df_attrs.columns:
            import numpy as np
            print("[dry-run] IC scores not found; generating random mock ic_score_mean from node_attributes")
            rng = np.random.default_rng(42)
            return pd.DataFrame({"node_id": df_attrs["node_id"].astype(str),
                                  "ic_score_mean": rng.random(size=len(df_attrs))})

    raise FileNotFoundError(
        f"Dry-run fallback failed: IC scores ({ic_path}), SIS table, and node_attributes all missing. "
        "Run Person 1's dry-run first: python ic_labels_primary.py --dry-run"
    )


def main() -> None:
    args = parse_args()

    ic_path = Path(args.ic)
    attrs_path = Path(args.node_attrs)
    if not attrs_path.exists():
        raise FileNotFoundError(f"Missing node attributes: {attrs_path}")

    df_ic = _load_ic_or_fallback(ic_path, attrs_path, dry_run=args.dry_run)

    df_attrs = pd.read_parquet(attrs_path)
    require_columns(df_attrs, ["node_id", "views"], "node_attributes")

    df = df_ic[["node_id", "ic_score_mean"]].merge(df_attrs[["node_id", "views"]], on="node_id", how="inner")
    if len(df) == 0:
        raise ValueError("No overlap between IC scores and node attributes")

    # Simple typology rules (can be refined by Person 2).
    ic_thresh = df["ic_score_mean"].quantile(1.0 - args.pct)
    views_thresh = df["views"].quantile(1.0 - args.pct)

    df["ic_high"] = df["ic_score_mean"] >= ic_thresh
    df["views_high"] = df["views"] >= views_thresh

    def _label(row) -> str:
        if row["ic_high"] and row["views_high"]:
            return "true_influencer"
        if row["ic_high"] and (not row["views_high"]):
            return "hidden"
        if (not row["ic_high"]) and row["views_high"]:
            return "overrated"
        return "non"

    df["typology_label"] = df.apply(_label, axis=1)

    out = df[["node_id", "typology_label", "ic_high", "views_high", "ic_score_mean", "views"]].copy()
    require_columns(
        out,
        ["node_id", "typology_label", "ic_high", "views_high", "ic_score_mean", "views"],
        "typology_labels",
    )

    ensure_parent(args.out)
    out.to_parquet(args.out, index=False)

    if args.dry_run:
        print(f"[OK] Wrote typology (dry-run OK): {args.out} (timestamp={now_iso()})")
        return

    raise NotImplementedError(
        "Implement quadrant sizing checks, expansion strategy, and reporting per MAPR2026 v3. "
        "This scaffold already writes a baseline typology parquet."
    )


if __name__ == "__main__":
    main()
