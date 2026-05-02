"""MAPR2026 v3 — Diagnostic graph-augmented flat baseline (HSCC-focused).

Trains LinearRegression on own node attributes plus fixed one-hop *attribute*
means over the full active graph (D^{-1} A X), with MinMaxScaler fit on the
training pool only (same train/val carve-out as surrogate training).

This is intentionally a *sanity / diagnostic* comparator — not the primary
pre-specified flat LR(degree, views, life_time, language) baseline.

Outputs (defaults under PATHS.results_dir):
- graph_augmented_flat_metrics_hscc.csv  (runtime_sec = LR inference only; see feature_precompute_sec)
- graph_augmented_flat_predictions_hscc.parquet  (node_id, y_pred — for frozen paired bootstrap)
- gnn_vs_graph_aug_flat_bootstrap_ci_hscc.json  (unless --skip-gnn-comparison; prefer --gnn-predictions-parquet)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

from _shared import PATHS, ensure_parent, load_csr_npz, require_columns, write_json
from eval_ranking_harness import apply_test_mask, compute_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _ensure_node_id_str(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["node_id"] = out["node_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    return out


def _derive_own_propagation_features(
    node_ids_csr: np.ndarray,
    node_attributes: pd.DataFrame,
    include_language: bool,
) -> tuple[pd.DataFrame, list[str]]:
    """Feature columns used for both 'own' and neighbor-mean blocks (no targets, no IC)."""
    attrs = _ensure_node_id_str(node_attributes)
    require_columns(attrs, ["node_id"], "node_attributes")

    if "views" in attrs.columns:
        views_raw = pd.to_numeric(attrs["views"], errors="coerce").fillna(0.0)
    else:
        views_raw = pd.Series(np.zeros(len(attrs), dtype=float), index=attrs.index)

    if "life_time" in attrs.columns:
        life_time = pd.to_numeric(attrs["life_time"], errors="coerce").fillna(1.0)
    elif "life_time_days" in attrs.columns:
        life_time = pd.to_numeric(attrs["life_time_days"], errors="coerce").fillna(1.0)
    else:
        life_time = pd.Series(np.ones(len(attrs), dtype=float), index=attrs.index)

    life_time = life_time.clip(lower=1.0)
    views_log = np.log1p(views_raw)
    views_per_day = views_raw / life_time

    lang_col = None
    if "language" in attrs.columns:
        lang_col = "language"
    elif "lang" in attrs.columns:
        lang_col = "lang"

    lang_dummies = pd.DataFrame(index=attrs.index)
    if include_language and lang_col is not None:
        lang_series = attrs[lang_col].astype(str).fillna("unknown")
        lang_series = lang_series.replace({"nan": "unknown", "None": "unknown"})
        lang_dummies = pd.get_dummies(lang_series, prefix="lang", dtype=float)
    elif include_language and lang_col is None:
        print("[WARN] --include-language set but no 'language'/'lang' column found.")

    base = pd.DataFrame(
        {
            "node_id": attrs["node_id"].astype(str),
            "views_log": views_log.astype(float),
            "views_per_day": views_per_day.astype(float),
            "life_time": life_time.astype(float),
        }
    )
    if not lang_dummies.empty:
        base = pd.concat([base, lang_dummies], axis=1)

    order = pd.DataFrame({"node_id": node_ids_csr.astype(str)})
    merged = order.merge(base, on="node_id", how="left")
    merged = merged.fillna(0.0)

    prop_cols = [c for c in merged.columns if c != "node_id"]
    return merged, prop_cols


def _neighbor_mean_matrix(
    indptr: np.ndarray,
    indices: np.ndarray,
    degrees: np.ndarray,
    X: np.ndarray,
) -> np.ndarray:
    """Row-wise neighbor-only mean: (D^{-1} A X) with isolated handling for deg=0."""
    n, f = X.shape
    data = np.ones(len(indices), dtype=np.float64)
    a_mat = sparse.csr_matrix((data, indices, indptr), shape=(n, n))
    agg = a_mat.dot(X)
    out = np.zeros_like(agg, dtype=np.float64)
    deg = degrees.astype(np.int64)
    mask = deg > 0
    out[mask] = agg[mask] / deg[mask][:, None]
    return out


def _labeled_masks_for_fit(
    labeled_node_ids: pd.Series,
    split_mask_df: pd.DataFrame,
    fit_mask: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Boolean masks on labeled rows aligned with ``labeled_node_ids`` order.

    Parameters
    ----------
    fit_mask
        - ``surrogate_train``: match ``load_surrogate_data_bundle`` — first 10%% of
          parquet ``split==train`` ids become validation; scaler/LR fit on remainder.
        - ``parquet_train``: all parquet ``split==train`` rows are training (matches
          ``bootstrap_ci._build_linear_predictions`` flat-LR convention; val mask empty).
    """
    mode = str(fit_mask).strip().lower()
    if mode not in {"surrogate_train", "parquet_train"}:
        raise ValueError("fit_mask must be 'surrogate_train' or 'parquet_train'")

    split_df = _ensure_node_id_str(split_mask_df)
    train_ids_parquet = set(split_df.loc[split_df["split"] == "train", "node_id"].astype(str).tolist())
    test_ids = set(split_df.loc[split_df["split"] == "test", "node_id"].astype(str).tolist())

    val_ids: set[str] = set()
    if mode == "parquet_train":
        train_ids = train_ids_parquet.copy()
    else:
        train_ids = train_ids_parquet.copy()
        if train_ids:
            train_sorted = sorted(train_ids)
            n_val = max(1, int(0.1 * len(train_sorted)))
            val_ids = set(train_sorted[:n_val])
            train_ids = set(train_sorted[n_val:])

    ids = labeled_node_ids.astype(str)
    train_m = ids.isin(train_ids).to_numpy(dtype=bool)
    val_m = ids.isin(val_ids).to_numpy(dtype=bool)
    test_m = ids.isin(test_ids).to_numpy(dtype=bool)
    return train_m, val_m, test_m


def _build_design_matrix(
    csr: dict[str, Any],
    node_attributes: pd.DataFrame,
    community_df: pd.DataFrame | None,
    include_language: bool,
    include_comm: bool,
) -> tuple[np.ndarray, list[str]]:
    """Full-graph design [own_block | nbr_mean_block] in CSR node order."""
    node_ids_csr = csr["node_ids"].astype(str)
    degrees = csr["degrees"].astype(np.int64)
    indptr = csr["indptr"]
    indices = csr["indices"]

    deg_log = np.log1p(degrees.astype(np.float64)).reshape(-1, 1)
    prop_df, prop_cols = _derive_own_propagation_features(node_ids_csr, node_attributes, include_language)

    x_prop = prop_df[prop_cols].to_numpy(dtype=np.float64)
    x_own = np.hstack([deg_log, x_prop])

    own_names = ["own_degree_log", *[f"own_{c}" for c in prop_cols]]
    x_nbr = _neighbor_mean_matrix(indptr, indices, degrees, x_own)
    nbr_names = ["nbr_mean_degree_log"] + [f"nbr_mean_{c}" for c in prop_cols]

    parts = [x_own, x_nbr]
    names = own_names + nbr_names

    if include_comm:
        if community_df is None:
            raise ValueError("include_comm requires community_df")
        comm = _ensure_node_id_str(community_df)
        require_columns(comm, ["node_id", "cross_community_edge_fraction"], "community_labels")
        order = pd.DataFrame({"node_id": node_ids_csr.astype(str)})
        cc = order.merge(
            comm[["node_id", "cross_community_edge_fraction"]],
            on="node_id",
            how="left",
        )
        cc_frac = pd.to_numeric(cc["cross_community_edge_fraction"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        cc_col = cc_frac.reshape(-1, 1)
        parts = [x_own, cc_col, x_nbr]
        names = own_names + ["own_cross_community_edge_fraction"] + nbr_names

    x_full = np.hstack(parts)
    return x_full.astype(np.float64), names


def _fit_predict_linear(
    x_labeled: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, MinMaxScaler, LinearRegression, float, float]:
    """Fit scaler+OLS on train_mask; return predictions and (train_sec, inference_sec)."""
    scaler = MinMaxScaler()
    if not train_mask.any():
        raise ValueError("Empty training mask for graph-augmented flat baseline.")
    t_train_0 = time.time()
    scaler.fit(x_labeled[train_mask])
    x_scaled = scaler.transform(x_labeled)

    reg = LinearRegression()
    reg.fit(x_scaled[train_mask], y[train_mask])
    train_sec = float(time.time() - t_train_0)

    t_inf_0 = time.time()
    pred = reg.predict(x_scaled).astype(np.float64)
    inference_sec = float(time.time() - t_inf_0)
    return pred, scaler, reg, train_sec, inference_sec


def main() -> None:
    parser = argparse.ArgumentParser(description="Graph-augmented flat LR baseline (diagnostic, HSCC)")
    parser.add_argument("--targets-path", default="data/processed/regression_targets_hscc_refined.parquet")
    parser.add_argument("--split-mask-path", default=PATHS.split_masks)
    parser.add_argument("--node-attributes-path", default=PATHS.node_attributes)
    parser.add_argument("--csr-npz-path", default=PATHS.csr_npz)
    parser.add_argument("--community-labels-path", default="data/processed/community_labels.parquet")
    parser.add_argument("--include-language", action="store_true", help="Align with HSCC GNN: language dummies.")
    parser.add_argument(
        "--include-comm",
        action="store_true",
        help="Append own-node cross_community_edge_fraction (diagnostic stress-test; uses Louvain used in HSCC labels).",
    )
    parser.add_argument(
        "--fit-mask",
        default="surrogate_train",
        choices=["surrogate_train", "parquet_train"],
        help=(
            "Training rows for scaler/LR: surrogate_train = train minus first 10%% val slice "
            "(matches run_surrogates); parquet_train = all split==train (matches bootstrap_ci flat LR)."
        ),
    )
    parser.add_argument(
        "--out-predictions-parquet",
        default=str(Path(PATHS.results_dir) / "graph_augmented_flat_predictions_hscc.parquet"),
        help="Write node_id,y_pred for all labeled target rows (use with empty string to skip).",
    )
    parser.add_argument(
        "--no-save-predictions",
        action="store_true",
        help="Do not write predictions parquet.",
    )
    parser.add_argument(
        "--surrogate-csv",
        default=str(Path(PATHS.results_dir) / "surrogate_ranking_metrics_hscc_clean.csv"),
        help="CSV with C2 GNN rows to pick best architecture for bootstrap comparison.",
    )
    parser.add_argument(
        "--gnn-predictions-parquet",
        default="",
        help="Parquet with columns node_id,y_pred — REQUIRED for paper-frozen bootstrap vs Table II.",
    )
    parser.add_argument(
        "--require-gnn-predictions-parquet",
        action="store_true",
        help="Fail if --gnn-predictions-parquet is missing (no silent GNN retrain for bootstrap).",
    )
    parser.add_argument("--skip-gnn-comparison", action="store_true", help="Only write flat-baseline metrics CSV.")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--equivalence-bound", type=float, default=0.02)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--seeds", default="", help="Comma-separated GNN training seeds (default: surrogate seeds).")
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--appnp-alpha", type=float, default=0.15)
    parser.add_argument("--appnp-k", type=int, default=10)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--gnn-std-threshold", type=float, default=float("inf"))
    parser.add_argument(
        "--out-metrics-csv",
        default=str(Path(PATHS.results_dir) / "graph_augmented_flat_metrics_hscc.csv"),
    )
    parser.add_argument(
        "--out-bootstrap-json",
        default=str(Path(PATHS.results_dir) / "gnn_vs_graph_aug_flat_bootstrap_ci_hscc.json"),
    )
    args = parser.parse_args()

    from bootstrap_ci import (  # noqa: PLC0415 — heavy deps loaded lazily
        _bootstrap_spearman_ndcg_ci,
        _interpret_ci,
        _predict_gnn_best,
        _select_best_gnn_model_name,
    )

    targets_path = resolve_project_path(args.targets_path)
    split_path = resolve_project_path(args.split_mask_path)
    attrs_path = resolve_project_path(args.node_attributes_path)
    csr_path = resolve_project_path(args.csr_npz_path)

    targets = pd.read_parquet(targets_path)
    targets = _ensure_node_id_str(targets)
    require_columns(targets, ["node_id", "y"], "regression_targets")
    targets["y"] = pd.to_numeric(targets["y"], errors="coerce")

    split_df = pd.read_parquet(split_path)
    split_df = _ensure_node_id_str(split_df)

    node_attributes = pd.read_parquet(attrs_path)
    csr = load_csr_npz(csr_path)
    id_to_idx = {str(nid): i for i, nid in enumerate(csr["node_ids"].astype(str))}

    indptr = csr["indptr"]
    indices = csr["indices"]
    if int(indptr[-1]) != len(indices):
        raise ValueError(f"CSR indptr/indices mismatch: indptr[-1]={indptr[-1]} len(indices)={len(indices)}")

    target_ids = targets["node_id"].astype(str).tolist()
    missing = [n for n in target_ids if n not in id_to_idx]
    if missing:
        raise ValueError(
            f"{len(missing)} regression target node_id not found in CSR node_ids (examples: {missing[:5]})"
        )
    idx = np.array([id_to_idx[n] for n in target_ids], dtype=np.int64)

    comm_df: pd.DataFrame | None = None
    if args.include_comm:
        comm_path = resolve_project_path(args.community_labels_path)
        comm_df = pd.read_parquet(comm_path)

    t0_feat = time.time()
    x_full, feat_names = _build_design_matrix(
        csr,
        node_attributes,
        comm_df,
        include_language=bool(args.include_language),
        include_comm=bool(args.include_comm),
    )
    graph_feature_sec = float(time.time() - t0_feat)

    x_lab = x_full[idx]

    train_m, val_m, test_m = _labeled_masks_for_fit(targets["node_id"], split_df, str(args.fit_mask))
    y = targets["y"].to_numpy(dtype=np.float64)

    pred_lr, _scaler, _reg, train_sec, inference_sec = _fit_predict_linear(x_lab, y, train_m)

    eval_df = pd.DataFrame({"node_id": targets["node_id"].astype(str), "y_true": y, "y_pred": pred_lr})
    eval_test = apply_test_mask(eval_df, split_df, node_id_col="node_id")
    metrics = compute_metrics(
        eval_test["y_true"].to_numpy(dtype=float),
        eval_test["y_pred"].to_numpy(dtype=float),
    )

    if args.include_comm:
        model_name = "lr_own_plus_1hop_attrs_comm_diagnostic"
        diagnostic_level = "community_exposed_stress_test"
    else:
        model_name = "lr_own_plus_1hop_attrs"
        diagnostic_level = "graph_augmented_flat_sanity"

    n_full = int(len(csr["node_ids"]))
    n_train = int(train_m.sum())
    n_val = int(val_m.sum())
    n_test_labeled = int(test_m.sum())

    row: dict[str, Any] = {
        "label_regime": "hscc",
        "model_name": model_name,
        "fit_mask": str(args.fit_mask),
        "diagnostic_level": diagnostic_level,
        "spearman_rho_mean": float(metrics.spearman_rho),
        "spearman_rho_std": 0.0,
        "ndcg_mean": float(metrics.ndcg_at_10pct),
        "ndcg_std": 0.0,
        "precision_mean": float(metrics.precision_at_10pct),
        "precision_std": 0.0,
        "feature_precompute_sec": graph_feature_sec,
        "train_sec": train_sec,
        "inference_sec": inference_sec,
        "runtime_sec": inference_sec,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test_labeled,
        "n_full_nodes": n_full,
        "n_features": int(x_lab.shape[1]),
        "include_language": bool(args.include_language),
        "include_comm": bool(args.include_comm),
        "feature_names": json.dumps(feat_names, ensure_ascii=False),
    }

    out_csv = resolve_project_path(args.out_metrics_csv)
    ensure_parent(out_csv)
    pd.DataFrame([row]).to_csv(out_csv, index=False)
    print(f"[OK] Wrote metrics row to {out_csv}")
    print(
        f"     model={model_name} fit_mask={args.fit_mask} rho={metrics.spearman_rho:.4f} "
        f"ndcg@10%={metrics.ndcg_at_10pct:.4f} p@10%={metrics.precision_at_10pct:.3f}"
    )

    pred_out_arg = str(args.out_predictions_parquet).strip()
    if pred_out_arg and not bool(args.no_save_predictions):
        pred_path = resolve_project_path(pred_out_arg)
        ensure_parent(pred_path)
        pd.DataFrame({"node_id": targets["node_id"].astype(str), "y_pred": pred_lr}).to_parquet(
            pred_path, index=False
        )
        print(f"[OK] Wrote predictions to {pred_path}")

    if args.skip_gnn_comparison:
        print("[INFO] --skip-gnn-comparison: skipping bootstrap JSON.")
        return

    if bool(args.require_gnn_predictions_parquet) and not str(args.gnn_predictions_parquet).strip():
        raise ValueError("--require-gnn-predictions-parquet requires a non-empty --gnn-predictions-parquet path.")

    seed_list = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip() != ""]
    if not seed_list:
        seed_list = [42, 123, 456, 789, 1024]

    surrogate_csv = resolve_project_path(args.surrogate_csv)
    best_gnn = _select_best_gnn_model_name(
        surrogate_csv,
        label_regime="hscc",
        std_threshold=float(args.gnn_std_threshold),
    )

    gnn_pred_path = str(args.gnn_predictions_parquet).strip()
    if gnn_pred_path:
        gnn_pred_df = pd.read_parquet(resolve_project_path(gnn_pred_path))
        gnn_pred_df = _ensure_node_id_str(gnn_pred_df)
        require_columns(gnn_pred_df, ["node_id", "y_pred"], "gnn_predictions")
    else:
        print(
            "[WARN] No --gnn-predictions-parquet: retraining GNN for bootstrap — point estimates may differ "
            "from frozen Table II. Export frozen preds and re-run for submission numbers."
        )
        print(f"[INFO] Retraining best GNN ({best_gnn}) for paired bootstrap (this can take a while)...")
        _split_df, gnn_pred_df = _predict_gnn_best(
            targets_path=str(targets_path),
            split_mask_path=str(split_path),
            model_name=best_gnn,
            max_epochs=int(args.max_epochs),
            seeds=seed_list,
            include_language=bool(args.include_language),
            gat_heads=int(args.gat_heads),
            appnp_alpha=float(args.appnp_alpha),
            appnp_k=int(args.appnp_k),
            hidden_channels=int(args.hidden_channels),
        )

    y_df = apply_test_mask(targets[["node_id", "y"]], split_df, node_id_col="node_id")
    gnn_test = apply_test_mask(
        pd.DataFrame({"node_id": gnn_pred_df["node_id"].astype(str), "y_pred": gnn_pred_df["y_pred"].astype(float)}),
        split_df,
        node_id_col="node_id",
    )
    flat_test = apply_test_mask(
        pd.DataFrame({"node_id": targets["node_id"].astype(str), "y_pred": pred_lr.astype(float)}),
        split_df,
        node_id_col="node_id",
    )

    merged = y_df.merge(gnn_test.rename(columns={"y_pred": "y_pred_gnn"}), on="node_id", how="inner")
    merged = merged.merge(flat_test.rename(columns={"y_pred": "y_pred_flat"}), on="node_id", how="inner")

    ci = _bootstrap_spearman_ndcg_ci(
        y_true=merged["y"].to_numpy(dtype=float),
        y_pred_a=merged["y_pred_gnn"].to_numpy(dtype=float),
        y_pred_b=merged["y_pred_flat"].to_numpy(dtype=float),
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed),
    )

    payload: dict[str, Any] = {
        "label_regime": "hscc",
        "role": "diagnostic_graph_augmented_flat_baseline",
        "diagnostic_level": diagnostic_level,
        "flat_model": model_name,
        "fit_mask": str(args.fit_mask),
        "gnn_model": best_gnn,
        "gnn_predictions_source": (
            str(resolve_project_path(gnn_pred_path)) if gnn_pred_path else "retrained_via_bootstrap_ci._predict_gnn_best"
        ),
        "gnn_predictions_frozen": bool(gnn_pred_path),
        "surrogate_csv_used": str(surrogate_csv),
        "include_language": bool(args.include_language),
        "include_comm": bool(args.include_comm),
        "n_train": n_train,
        "n_val": n_val,
        "n_test": int(merged.shape[0]),
        "n_full_nodes": n_full,
        "n_features": int(x_lab.shape[1]),
        "n_bootstrap": int(args.n_bootstrap),
        "equivalence_bound": float(args.equivalence_bound),
        "flat_spearman_on_test": float(metrics.spearman_rho),
        "spearman": {
            "delta_mean": ci["spearman_delta_mean"],
            "ci_95_lower": ci["spearman_ci_95_lower"],
            "ci_95_upper": ci["spearman_ci_95_upper"],
            "interpretation": _interpret_ci(
                ci["spearman_ci_95_lower"],
                ci["spearman_ci_95_upper"],
                equivalence_bound=float(args.equivalence_bound),
            ),
        },
        "ndcg_at_10pct": {
            "delta_mean": ci["ndcg_delta_mean"],
            "ci_95_lower": ci["ndcg_ci_95_lower"],
            "ci_95_upper": ci["ndcg_ci_95_upper"],
            "interpretation": _interpret_ci(
                ci["ndcg_ci_95_lower"],
                ci["ndcg_ci_95_upper"],
                equivalence_bound=float(args.equivalence_bound),
            ),
        },
    }

    out_json = resolve_project_path(args.out_bootstrap_json)
    write_json(out_json, payload)
    print(f"[OK] Wrote bootstrap CI to {out_json}")


if __name__ == "__main__":
    main()
