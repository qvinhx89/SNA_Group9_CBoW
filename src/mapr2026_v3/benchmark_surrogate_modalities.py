"""Benchmark surrogate feature modalities for IC regression targets.

Compares three input settings on the shared M0 split:
- graph_only: structural features (degree, pagerank, kshell)
- views_only: raw attribute features (views-based)
- graph_plus_views: combined structural + raw attributes

Outputs a per-seed CSV and a summary CSV with paired deltas vs graph_only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from _shared import PATHS, ensure_dir, require_columns
from eval_ranking_harness import apply_test_mask, compute_metrics, load_split_mask


SEEDS = [42, 123, 456, 789, 1024]


def _resolve_repo_path(path_like: str | Path) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    return Path(__file__).resolve().parents[2] / p


def _ensure_node_id_str(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["node_id"] = out["node_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    return out


def _build_views_features(node_attrs: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_node_id_str(node_attrs)
    require_columns(df, ["node_id"], "node_attributes")

    views_raw = pd.to_numeric(df.get("views", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)

    if "life_time" in df.columns:
        life_time = pd.to_numeric(df["life_time"], errors="coerce").fillna(1.0)
    elif "life_time_days" in df.columns:
        life_time = pd.to_numeric(df["life_time_days"], errors="coerce").fillna(1.0)
    else:
        life_time = pd.Series(np.ones(len(df), dtype=float), index=df.index)

    life_time = life_time.clip(lower=1.0)
    return pd.DataFrame(
        {
            "node_id": df["node_id"].astype(str),
            "views_log": np.log1p(views_raw).astype(float),
            "views_per_day": (views_raw / life_time).astype(float),
            "life_time": life_time.astype(float),
        }
    )


def _build_graph_features(centrality: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_node_id_str(centrality)
    require_columns(df, ["node_id"], "centrality_table")

    kshell_col = "kshell" if "kshell" in df.columns else ("k_shell" if "k_shell" in df.columns else None)
    missing = [c for c in ["degree", "pagerank"] if c not in df.columns]
    if kshell_col is None:
        missing.append("kshell|k_shell")
    if missing:
        raise ValueError(f"centrality_table missing required graph features: {missing}")

    out = df[["node_id", "degree", "pagerank", kshell_col]].copy()
    if kshell_col != "kshell":
        out = out.rename(columns={kshell_col: "kshell"})
    out[["degree", "pagerank", "kshell"]] = out[["degree", "pagerank", "kshell"]].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0.0)
    return out


def _topk_jaccard(y_true: np.ndarray, y_pred: np.ndarray, top_pct: float) -> float:
    n = int(y_true.size)
    k = max(1, int(np.ceil(float(top_pct) * n)))
    idx_pred = set(np.argsort(-y_pred)[:k].tolist())
    idx_true = set(np.argsort(-y_true)[:k].tolist())
    union = idx_pred | idx_true
    if not union:
        return 1.0
    return float(len(idx_pred & idx_true) / len(union))


def _load_merged(
    targets_path: str | Path,
    split_mask_path: str | Path,
    node_attrs_path: str | Path,
    centrality_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = _ensure_node_id_str(pd.read_parquet(_resolve_repo_path(targets_path)))
    require_columns(targets, ["node_id", "y"], "regression_targets")
    targets["y"] = pd.to_numeric(targets["y"], errors="coerce").fillna(0.0)

    split_mask = _ensure_node_id_str(load_split_mask(_resolve_repo_path(split_mask_path)))
    views_df = _build_views_features(pd.read_parquet(_resolve_repo_path(node_attrs_path)))
    graph_df = _build_graph_features(pd.read_parquet(_resolve_repo_path(centrality_path)))

    merged = (
        targets[["node_id", "y"]]
        .merge(views_df, on="node_id", how="left")
        .merge(graph_df, on="node_id", how="left")
    )

    feature_cols = [
        "views_log",
        "views_per_day",
        "life_time",
        "degree",
        "pagerank",
        "kshell",
    ]
    merged[feature_cols] = merged[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return merged, split_mask


def _evaluate_mode(
    merged: pd.DataFrame,
    split_mask: pd.DataFrame,
    feature_cols: list[str],
    seed: int,
    top_pct: float,
) -> dict[str, float | int]:
    split_series = split_mask.set_index("node_id")["split"]
    split = split_series.reindex(merged["node_id"]).fillna("train")

    train_mask = split.eq("train").to_numpy(dtype=bool)
    if not train_mask.any():
        raise ValueError("Empty train split after alignment.")

    x = merged[feature_cols].to_numpy(dtype=float)
    y = merged["y"].to_numpy(dtype=float)

    model = RandomForestRegressor(
        n_estimators=400,
        min_samples_leaf=2,
        random_state=int(seed),
        n_jobs=-1,
    )
    model.fit(x[train_mask], y[train_mask])
    y_pred_all = model.predict(x)

    eval_df = pd.DataFrame(
        {
            "node_id": merged["node_id"].astype(str),
            "y_true": y.astype(float),
            "y_pred": y_pred_all.astype(float),
        }
    )
    eval_test = apply_test_mask(eval_df, split_mask, node_id_col="node_id")

    y_true_t = eval_test["y_true"].to_numpy(dtype=float)
    y_pred_t = eval_test["y_pred"].to_numpy(dtype=float)
    metrics = compute_metrics(y_true_t, y_pred_t)

    return {
        "seed": int(seed),
        "spearman_rho": float(metrics.spearman_rho),
        "ndcg_at_10pct": float(metrics.ndcg_at_10pct),
        "precision_at_10pct": float(metrics.precision_at_10pct),
        "jaccard_at_10pct": float(_topk_jaccard(y_true_t, y_pred_t, top_pct=top_pct)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark graph vs views surrogate modalities")
    p.add_argument("--targets-path", default=PATHS.regression_targets)
    p.add_argument("--split-mask-path", default=PATHS.split_masks)
    p.add_argument("--node-attrs-path", default=PATHS.node_attributes)
    p.add_argument("--centrality-path", default="data/processed/centrality_table.parquet")
    p.add_argument("--top-pct", type=float, default=0.10)
    p.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    p.add_argument(
        "--out-dir",
        default="outputs/day1_benchmark/surrogate_ablation",
        help="Directory for per-seed and summary CSV outputs",
    )
    p.add_argument("--tag", default="current_targets")
    return p.parse_args()


def _parse_seed_list(raw: str) -> list[int]:
    out = []
    for t in raw.split(","):
        t = t.strip()
        if t:
            out.append(int(t))
    if not out:
        raise ValueError("seeds cannot be empty")
    return out


def main() -> None:
    args = parse_args()
    seeds = _parse_seed_list(args.seeds)
    out_dir = ensure_dir(_resolve_repo_path(args.out_dir))

    merged, split_mask = _load_merged(
        targets_path=args.targets_path,
        split_mask_path=args.split_mask_path,
        node_attrs_path=args.node_attrs_path,
        centrality_path=args.centrality_path,
    )

    modes = {
        "graph_only": ["degree", "pagerank", "kshell"],
        "views_only": ["views_log", "views_per_day", "life_time"],
        "graph_plus_views": ["degree", "pagerank", "kshell", "views_log", "views_per_day", "life_time"],
    }

    per_seed_rows: list[dict[str, float | int | str]] = []
    for mode_name, cols in modes.items():
        for seed in seeds:
            row = _evaluate_mode(
                merged=merged,
                split_mask=split_mask,
                feature_cols=cols,
                seed=int(seed),
                top_pct=float(args.top_pct),
            )
            row["mode"] = mode_name
            per_seed_rows.append(row)

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_out = out_dir / f"modalities_per_seed_{args.tag}.csv"
    per_seed_df.to_csv(per_seed_out, index=False)

    summary = (
        per_seed_df.groupby("mode", as_index=False)
        .agg(
            spearman_rho_mean=("spearman_rho", "mean"),
            spearman_rho_std=("spearman_rho", "std"),
            ndcg_mean=("ndcg_at_10pct", "mean"),
            ndcg_std=("ndcg_at_10pct", "std"),
            precision_mean=("precision_at_10pct", "mean"),
            precision_std=("precision_at_10pct", "std"),
            jaccard_mean=("jaccard_at_10pct", "mean"),
            jaccard_std=("jaccard_at_10pct", "std"),
        )
    )

    # Paired improvements over graph_only by matching seed.
    base = per_seed_df.loc[per_seed_df["mode"] == "graph_only", ["seed", "spearman_rho", "jaccard_at_10pct"]]
    base = base.rename(
        columns={
            "spearman_rho": "base_spearman_rho",
            "jaccard_at_10pct": "base_jaccard_at_10pct",
        }
    )
    deltas = per_seed_df.merge(base, on="seed", how="left")
    deltas["delta_spearman_vs_graph_only"] = deltas["spearman_rho"] - deltas["base_spearman_rho"]
    deltas["delta_jaccard_vs_graph_only"] = deltas["jaccard_at_10pct"] - deltas["base_jaccard_at_10pct"]

    delta_summary = (
        deltas.groupby("mode", as_index=False)
        .agg(
            delta_spearman_mean=("delta_spearman_vs_graph_only", "mean"),
            delta_spearman_std=("delta_spearman_vs_graph_only", "std"),
            delta_jaccard_mean=("delta_jaccard_vs_graph_only", "mean"),
            delta_jaccard_std=("delta_jaccard_vs_graph_only", "std"),
        )
    )

    summary = summary.merge(delta_summary, on="mode", how="left")
    summary["n_seeds"] = int(len(seeds))

    summary_out = out_dir / f"modalities_summary_{args.tag}.csv"
    summary.to_csv(summary_out, index=False)

    print("[OK] Surrogate modality benchmark completed")
    print(f" - nodes={len(merged):,}")
    print(f" - test_nodes={int((split_mask['split'] == 'test').sum()):,}")
    print(f" - seeds={seeds}")
    print(f" - per_seed={per_seed_out}")
    print(f" - summary={summary_out}")


if __name__ == "__main__":
    main()
