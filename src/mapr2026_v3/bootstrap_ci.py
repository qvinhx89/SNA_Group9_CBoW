from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

from _shared import PATHS, ensure_parent
from eval_ranking_harness import apply_test_mask, compute_metrics, load_split_mask

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _ensure_node_id_str(df: pd.DataFrame) -> pd.DataFrame:
    local_df = df.copy()
    local_df["node_id"] = local_df["node_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    return local_df


def _import_run_baselines_symbols() -> dict[str, Any]:
    from run_baselines import (
        MLPRegressor,
        TRAINING_SEEDS as BASELINE_SEEDS,
        _derive_features,
        get_loss_function,
    )

    return {
        "MLPRegressor": MLPRegressor,
        "BASELINE_SEEDS": BASELINE_SEEDS,
        "derive_features": _derive_features,
        "get_loss_function": get_loss_function,
    }


def _import_run_surrogates_symbols() -> dict[str, Any]:
    from run_surrogates import (
        TRAINING_SEEDS as SURROGATE_SEEDS,
        load_surrogate_data_bundle,
        train_surrogate_5seeds,
    )

    return {
        "SURROGATE_SEEDS": SURROGATE_SEEDS,
        "load_surrogate_data_bundle": load_surrogate_data_bundle,
        "train_surrogate_5seeds": train_surrogate_5seeds,
    }


def _compute_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    result = spearmanr(y_true, y_pred)
    statistic = getattr(result, "statistic", result[0])
    return float(np.asarray(statistic).item())


def _bootstrap_spearman_ndcg_ci(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    random_generator = np.random.default_rng(int(seed))
    n = int(len(y_true))
    if n == 0:
        raise ValueError("Bootstrap input is empty.")

    spearman_deltas: list[float] = []
    ndcg_deltas: list[float] = []

    for _ in range(int(n_bootstrap)):
        sample_idx = random_generator.integers(0, n, size=n)
        y_sample = y_true[sample_idx]
        pred_a_sample = y_pred_a[sample_idx]
        pred_b_sample = y_pred_b[sample_idx]

        metric_a = compute_metrics(y_sample, pred_a_sample)
        metric_b = compute_metrics(y_sample, pred_b_sample)

        spearman_deltas.append(float(metric_a.spearman_rho - metric_b.spearman_rho))
        ndcg_deltas.append(float(metric_a.ndcg_at_10pct - metric_b.ndcg_at_10pct))

    spearman_arr = np.asarray(spearman_deltas, dtype=float)
    ndcg_arr = np.asarray(ndcg_deltas, dtype=float)

    return {
        "spearman_delta_mean": float(np.mean(spearman_arr)),
        "spearman_ci_95_lower": float(np.percentile(spearman_arr, 2.5)),
        "spearman_ci_95_upper": float(np.percentile(spearman_arr, 97.5)),
        "ndcg_delta_mean": float(np.mean(ndcg_arr)),
        "ndcg_ci_95_lower": float(np.percentile(ndcg_arr, 2.5)),
        "ndcg_ci_95_upper": float(np.percentile(ndcg_arr, 97.5)),
    }


def _interpret_ci(ci_lower: float, ci_upper: float, equivalence_bound: float) -> str:
    bound = float(abs(equivalence_bound))
    if ci_lower > 0:
        return "gnn_significantly_better"
    if ci_upper < 0:
        return "gnn_significantly_worse"
    if ci_lower >= -bound and ci_upper <= bound:
        return "practically_equivalent"
    return "no_clear_superiority"


def _load_targets_df(targets_path: str | Path) -> pd.DataFrame:
    targets = pd.read_parquet(resolve_project_path(targets_path))
    targets = _ensure_node_id_str(targets)
    if "node_id" not in targets.columns or "y" not in targets.columns:
        raise ValueError(f"Invalid targets schema in {targets_path}: need node_id,y")
    targets = targets[["node_id", "y"]].copy()
    targets["y"] = pd.to_numeric(targets["y"], errors="coerce")
    return targets


def _select_best_gnn_model_name(surrogate_csv: Path, label_regime: str) -> str:
    df = pd.read_csv(surrogate_csv)
    if "label_regime" in df.columns:
        df = df.loc[df["label_regime"].astype(str).str.lower() == str(label_regime).lower()].copy()

    allowed = [
        "gnn_raw_attr",
        "gcn_raw_attr",
        "gin_raw_attr",
        "gat_raw_attr",
        "appnp_raw_attr",
    ]
    df = df.loc[df["model_name"].astype(str).isin(allowed)].copy()
    if df.empty:
        raise RuntimeError(f"No C2 GNN rows found for label_regime={label_regime} in {surrogate_csv}")

    priority = {"appnp_raw_attr": 0, "gat_raw_attr": 1, "gin_raw_attr": 2, "gcn_raw_attr": 3, "gnn_raw_attr": 4}
    best_score = float(df["spearman_rho_mean"].max())
    tie = df.loc[(best_score - df["spearman_rho_mean"]).abs() < 0.001].copy()
    tie["priority"] = tie["model_name"].map(priority)
    tie = tie.sort_values(["priority", "model_name"], ascending=[True, True])
    return str(tie.iloc[0]["model_name"])


def _model_name_to_arch(model_name: str) -> str:
    mapping = {
        "gnn_raw_attr": "sage",
        "gcn_raw_attr": "gcn",
        "gin_raw_attr": "gin",
        "gat_raw_attr": "gat",
        "appnp_raw_attr": "appnp",
    }
    if model_name not in mapping:
        raise ValueError(f"Unsupported GNN model_name for C4 bootstrap: {model_name}")
    return mapping[model_name]


def _predict_gnn_best(
    targets_path: str | Path,
    split_mask_path: str | Path,
    model_name: str,
    max_epochs: int,
    seeds: list[int] | None,
    include_language: bool,
    gat_heads: int,
    appnp_alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    surrogate_symbols = _import_run_surrogates_symbols()
    load_surrogate_data_bundle = surrogate_symbols["load_surrogate_data_bundle"]
    train_surrogate_5seeds = surrogate_symbols["train_surrogate_5seeds"]

    arch = _model_name_to_arch(model_name)
    bundle = load_surrogate_data_bundle(
        feature_mode="raw_attr",
        targets_path=targets_path,
        split_mask_path=split_mask_path,
        node_scope="all",
        include_language=include_language,
    )
    _, y_pred = train_surrogate_5seeds(
        bundle=bundle,
        max_epochs=int(max_epochs),
        model_name=model_name,
        arch=arch,
        randomize_train_target=False,
        early_stop=False,
        patience=20,
        seeds=seeds,
        loss_mode="huber",
        rankloss_alpha=0.5,
        gat_heads=int(gat_heads),
        appnp_alpha=float(appnp_alpha),
    )
    prediction_df = pd.DataFrame({"node_id": bundle.node_ids.astype(str), "y_pred": y_pred.astype(float)})
    return bundle.split_mask_df, prediction_df


def _build_linear_predictions(
    targets_df: pd.DataFrame,
    split_mask_df: pd.DataFrame,
    node_attributes: pd.DataFrame,
    feature_cols: list[str],
    include_language: bool,
) -> np.ndarray:
    baseline_symbols = _import_run_baselines_symbols()
    derive_features = baseline_symbols["derive_features"]

    feature_frame = derive_features(node_attributes, include_language=include_language)
    if "degree" in feature_cols:
        degree_source = _ensure_node_id_str(node_attributes)[["node_id", "degree"]].copy()
        feature_frame = feature_frame.merge(degree_source, on="node_id", how="left")

    merged = targets_df[["node_id", "y"]].merge(feature_frame, on="node_id", how="left")
    for col in feature_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    train_ids = set(split_mask_df.loc[split_mask_df["split"] == "train", "node_id"].astype(str).tolist())
    train_mask = merged["node_id"].astype(str).isin(train_ids).to_numpy()
    if train_mask.sum() == 0:
        raise RuntimeError("Train split is empty for linear baseline bootstrap comparator.")

    regressor = LinearRegression()
    x = merged[feature_cols].to_numpy(dtype=np.float32)
    y = merged["y"].to_numpy(dtype=np.float32)
    regressor.fit(x[train_mask], y[train_mask])
    return regressor.predict(x).astype(float)


def _predict_mlp_raw_attr(
    targets_df: pd.DataFrame,
    split_mask_df: pd.DataFrame,
    node_attributes: pd.DataFrame,
    max_epochs: int,
    seeds: list[int] | None,
    include_language: bool,
) -> np.ndarray:
    import torch

    baseline_symbols = _import_run_baselines_symbols()
    MLPRegressor = baseline_symbols["MLPRegressor"]
    derive_features = baseline_symbols["derive_features"]
    get_loss_function = baseline_symbols["get_loss_function"]
    BASELINE_SEEDS = baseline_symbols["BASELINE_SEEDS"]

    features = derive_features(node_attributes, include_language=include_language)
    merged = targets_df[["node_id", "y"]].merge(features, on="node_id", how="left")
    feature_cols = [c for c in merged.columns if c not in {"node_id", "y"}]
    for col in feature_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    x = torch.tensor(merged[feature_cols].to_numpy(dtype=np.float32), dtype=torch.float32)
    y = torch.tensor(merged["y"].to_numpy(dtype=np.float32), dtype=torch.float32)

    train_ids = set(split_mask_df.loc[split_mask_df["split"] == "train", "node_id"].astype(str).tolist())
    train_mask = torch.tensor(merged["node_id"].astype(str).isin(train_ids).to_numpy(), dtype=torch.bool)

    training_seeds = BASELINE_SEEDS if seeds is None else list(seeds)
    if len(training_seeds) == 0:
        raise ValueError("Empty seeds for MLP comparator.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    y = y.to(device)
    train_mask = train_mask.to(device)

    predictions: list[np.ndarray] = []
    for seed in training_seeds:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        np.random.seed(int(seed))

        model = MLPRegressor(in_features=int(x.shape[1]), hidden_dim=128, dropout=0.3).to(device)
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = get_loss_function()

        for _ in range(int(max_epochs)):
            model.train()
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred[train_mask], y[train_mask])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(x).detach().cpu().numpy().astype(np.float32)
        predictions.append(pred)

    return np.mean(np.stack(predictions, axis=0), axis=0).astype(float)


def _select_strongest_flat_baseline_hscc(
    targets_df: pd.DataFrame,
    split_mask_df: pd.DataFrame,
    node_attributes: pd.DataFrame,
    max_epochs: int,
    seeds: list[int] | None,
    include_language: bool,
) -> tuple[str, np.ndarray, float]:
    baseline_symbols = _import_run_baselines_symbols()
    derive_features = baseline_symbols["derive_features"]

    candidates: list[tuple[str, np.ndarray]] = []

    linear_specs = [
        ("lr_life_time", ["life_time"]),
        ("lr_views_life_time", ["views_log", "life_time"]),
        ("lr_degree_views_life_time", ["degree", "views_log", "life_time"]),
    ]

    derived = derive_features(node_attributes, include_language=include_language)
    lang_cols = [col for col in derived.columns if col.startswith("lang_")]
    if len(lang_cols) > 0:
        linear_specs.extend(
            [
                ("lr_views_life_time_lang", ["views_log", "life_time", *lang_cols]),
                ("lr_degree_views_life_time_lang", ["degree", "views_log", "life_time", *lang_cols]),
            ]
        )

    for model_name, feature_cols in linear_specs:
        pred = _build_linear_predictions(
            targets_df=targets_df,
            split_mask_df=split_mask_df,
            node_attributes=node_attributes,
            feature_cols=feature_cols,
            include_language=include_language,
        )
        candidates.append((model_name, pred))

    mlp_pred = _predict_mlp_raw_attr(
        targets_df=targets_df,
        split_mask_df=split_mask_df,
        node_attributes=node_attributes,
        max_epochs=max_epochs,
        seeds=seeds,
        include_language=include_language,
    )
    candidates.append(("mlp_raw_attr", mlp_pred))

    eval_target = targets_df[["node_id", "y"]].copy()
    eval_target = apply_test_mask(eval_target, split_mask_df, node_id_col="node_id")

    best_model = ""
    best_pred: np.ndarray | None = None
    best_rho = -np.inf

    for model_name, pred in candidates:
        pred_df = pd.DataFrame({"node_id": targets_df["node_id"].astype(str), "y_pred": pred.astype(float)})
        pred_test = apply_test_mask(pred_df, split_mask_df, node_id_col="node_id")
        merged = eval_target.merge(pred_test, on="node_id", how="inner")
        rho = _compute_spearman(
            merged["y"].to_numpy(dtype=float),
            merged["y_pred"].to_numpy(dtype=float),
        )
        if rho > best_rho:
            best_rho = float(rho)
            best_model = model_name
            best_pred = pred

    if best_pred is None:
        raise RuntimeError("Failed to build HSCC strongest flat baseline predictions.")

    return best_model, best_pred, best_rho


def _build_degree_predictions(targets_df: pd.DataFrame, node_attributes: pd.DataFrame) -> np.ndarray:
    degree_df = _ensure_node_id_str(node_attributes)[["node_id", "degree"]].copy()
    merged = targets_df[["node_id"]].merge(degree_df, on="node_id", how="left")
    return pd.to_numeric(merged["degree"], errors="coerce").fillna(0.0).to_numpy(dtype=float)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    out = ensure_parent(path)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MAPR2026 v3.2 bootstrap CI runner for Person 3")
    parser.add_argument("--surrogate-csv", default=str(Path(PATHS.results_dir) / "surrogate_ranking_metrics.csv"))
    parser.add_argument("--surrogate-csv-a0", default="", help="Optional A0-specific surrogate CSV. Falls back to --surrogate-csv.")
    parser.add_argument("--surrogate-csv-hscc", default="", help="Optional HSCC-specific surrogate CSV. Falls back to --surrogate-csv.")
    parser.add_argument("--targets-a0", default="data/processed/regression_targets_a0.parquet")
    parser.add_argument("--targets-hscc", default="data/processed/regression_targets_hscc_refined.parquet")
    parser.add_argument("--split-mask-path", default=PATHS.split_masks)
    parser.add_argument("--node-attributes-path", default=PATHS.node_attributes)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--equivalence-bound", type=float, default=0.02)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--seeds", default="", help="Comma-separated training seeds override.")
    parser.add_argument("--gat-heads", type=int, default=4, help="GAT heads used when bootstrap reruns a GAT best-model.")
    parser.add_argument("--appnp-alpha", type=float, default=0.15, help="APPNP alpha used when bootstrap reruns an APPNP best-model.")
    parser.add_argument(
        "--out-a0",
        default=str(Path(PATHS.results_dir) / "gnn_vs_degree_bootstrap_ci_a0.json"),
    )
    parser.add_argument(
        "--out-hscc",
        default=str(Path(PATHS.results_dir) / "gnn_vs_baseline_bootstrap_ci_hscc.json"),
    )
    parser.add_argument("--include-language-hscc", action="store_true", help="Use language-aware raw_attr/full features for HSCC comparators and GNN reruns.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    surrogate_csv_legacy = resolve_project_path(args.surrogate_csv)
    surrogate_csv_a0 = resolve_project_path(args.surrogate_csv_a0) if str(args.surrogate_csv_a0).strip() else surrogate_csv_legacy
    surrogate_csv_hscc = resolve_project_path(args.surrogate_csv_hscc) if str(args.surrogate_csv_hscc).strip() else surrogate_csv_legacy
    split_mask_path = resolve_project_path(args.split_mask_path)
    targets_a0_path = resolve_project_path(args.targets_a0)
    targets_hscc_path = resolve_project_path(args.targets_hscc)
    node_attributes_path = resolve_project_path(args.node_attributes_path)
    include_language_hscc = bool(args.include_language_hscc)

    if args.dry_run:
        print("[OK] bootstrap_ci dry-run")
        print(f" - surrogate_csv_legacy={surrogate_csv_legacy}")
        print(f" - surrogate_csv_a0={surrogate_csv_a0}")
        print(f" - surrogate_csv_hscc={surrogate_csv_hscc}")
        print(f" - split_mask={split_mask_path}")
        print(f" - targets_a0={targets_a0_path}")
        print(f" - targets_hscc={targets_hscc_path}")
        print(f" - include_language_hscc={include_language_hscc}")
        print(f" - gat_heads={int(args.gat_heads)}")
        print(f" - appnp_alpha={float(args.appnp_alpha)}")
        return

    split_mask_df = load_split_mask(split_mask_path)
    split_mask_df = _ensure_node_id_str(split_mask_df)

    if str(args.seeds).strip():
        seed_list = [int(part.strip()) for part in str(args.seeds).split(",") if part.strip()]
    else:
        seed_list = None

    surrogate_symbols = _import_run_surrogates_symbols()
    baseline_symbols = _import_run_baselines_symbols()
    SURROGATE_SEEDS = surrogate_symbols["SURROGATE_SEEDS"]
    BASELINE_SEEDS = baseline_symbols["BASELINE_SEEDS"]

    node_attributes = pd.read_parquet(node_attributes_path)
    node_attributes = _ensure_node_id_str(node_attributes)

    best_a0_name = _select_best_gnn_model_name(surrogate_csv_a0, label_regime="a0")
    _, gnn_pred_a0_df = _predict_gnn_best(
        targets_path=targets_a0_path,
        split_mask_path=split_mask_path,
        model_name=best_a0_name,
        max_epochs=int(args.max_epochs),
        seeds=seed_list if seed_list is not None else list(SURROGATE_SEEDS),
        include_language=False,
        gat_heads=int(args.gat_heads),
        appnp_alpha=float(args.appnp_alpha),
    )

    targets_a0 = _load_targets_df(targets_a0_path)
    degree_pred = _build_degree_predictions(targets_a0, node_attributes=node_attributes)

    y_a0_df = apply_test_mask(targets_a0[["node_id", "y"]], split_mask_df, node_id_col="node_id")
    pred_gnn_a0_df = apply_test_mask(gnn_pred_a0_df, split_mask_df, node_id_col="node_id")
    pred_degree_df = apply_test_mask(
        pd.DataFrame({"node_id": targets_a0["node_id"].astype(str), "y_pred": degree_pred}),
        split_mask_df,
        node_id_col="node_id",
    )
    merged_a0 = y_a0_df.merge(pred_gnn_a0_df, on="node_id", how="inner", suffixes=("", "_gnn"))
    merged_a0 = merged_a0.merge(pred_degree_df, on="node_id", how="inner", suffixes=("_gnn", "_cmp"))

    ci_a0 = _bootstrap_spearman_ndcg_ci(
        y_true=merged_a0["y"].to_numpy(dtype=float),
        y_pred_a=merged_a0["y_pred_gnn"].to_numpy(dtype=float),
        y_pred_b=merged_a0["y_pred_cmp"].to_numpy(dtype=float),
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed),
    )
    payload_a0 = {
        "label_regime": "a0",
        "gnn_model": best_a0_name,
        "comparator_model": "degree",
        "surrogate_csv_used": str(surrogate_csv_a0),
        "feature_policy": {
            "include_language": False,
            "gnn_model": best_a0_name,
        },
        "n_test": int(merged_a0.shape[0]),
        "n_bootstrap": int(args.n_bootstrap),
        "equivalence_bound": float(args.equivalence_bound),
        "spearman": {
            "delta_mean": ci_a0["spearman_delta_mean"],
            "ci_95_lower": ci_a0["spearman_ci_95_lower"],
            "ci_95_upper": ci_a0["spearman_ci_95_upper"],
            "interpretation": _interpret_ci(
                ci_a0["spearman_ci_95_lower"],
                ci_a0["spearman_ci_95_upper"],
                equivalence_bound=float(args.equivalence_bound),
            ),
        },
        "ndcg_at_10pct": {
            "delta_mean": ci_a0["ndcg_delta_mean"],
            "ci_95_lower": ci_a0["ndcg_ci_95_lower"],
            "ci_95_upper": ci_a0["ndcg_ci_95_upper"],
            "interpretation": _interpret_ci(
                ci_a0["ndcg_ci_95_lower"],
                ci_a0["ndcg_ci_95_upper"],
                equivalence_bound=float(args.equivalence_bound),
            ),
        },
    }
    _write_json(resolve_project_path(args.out_a0), payload_a0)

    best_hscc_name = _select_best_gnn_model_name(surrogate_csv_hscc, label_regime="hscc")
    _, gnn_pred_hscc_df = _predict_gnn_best(
        targets_path=targets_hscc_path,
        split_mask_path=split_mask_path,
        model_name=best_hscc_name,
        max_epochs=int(args.max_epochs),
        seeds=seed_list if seed_list is not None else list(SURROGATE_SEEDS),
        include_language=include_language_hscc,
        gat_heads=int(args.gat_heads),
        appnp_alpha=float(args.appnp_alpha),
    )

    targets_hscc = _load_targets_df(targets_hscc_path)
    strongest_name, strongest_pred, strongest_rho = _select_strongest_flat_baseline_hscc(
        targets_df=targets_hscc,
        split_mask_df=split_mask_df,
        node_attributes=node_attributes,
        max_epochs=int(args.max_epochs),
        seeds=seed_list if seed_list is not None else list(BASELINE_SEEDS),
        include_language=include_language_hscc,
    )

    y_hscc_df = apply_test_mask(targets_hscc[["node_id", "y"]], split_mask_df, node_id_col="node_id")
    pred_gnn_hscc_df = apply_test_mask(gnn_pred_hscc_df, split_mask_df, node_id_col="node_id")
    pred_cmp_hscc_df = apply_test_mask(
        pd.DataFrame({"node_id": targets_hscc["node_id"].astype(str), "y_pred": strongest_pred}),
        split_mask_df,
        node_id_col="node_id",
    )
    merged_hscc = y_hscc_df.merge(pred_gnn_hscc_df, on="node_id", how="inner", suffixes=("", "_gnn"))
    merged_hscc = merged_hscc.merge(pred_cmp_hscc_df, on="node_id", how="inner", suffixes=("_gnn", "_cmp"))

    ci_hscc = _bootstrap_spearman_ndcg_ci(
        y_true=merged_hscc["y"].to_numpy(dtype=float),
        y_pred_a=merged_hscc["y_pred_gnn"].to_numpy(dtype=float),
        y_pred_b=merged_hscc["y_pred_cmp"].to_numpy(dtype=float),
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed),
    )
    payload_hscc = {
        "label_regime": "hscc",
        "gnn_model": best_hscc_name,
        "comparator_model": strongest_name,
        "comparator_spearman_on_test": float(strongest_rho),
        "surrogate_csv_used": str(surrogate_csv_hscc),
        "feature_policy": {
            "include_language": include_language_hscc,
            "gnn_model": best_hscc_name,
        },
        "n_test": int(merged_hscc.shape[0]),
        "n_bootstrap": int(args.n_bootstrap),
        "equivalence_bound": float(args.equivalence_bound),
        "spearman": {
            "delta_mean": ci_hscc["spearman_delta_mean"],
            "ci_95_lower": ci_hscc["spearman_ci_95_lower"],
            "ci_95_upper": ci_hscc["spearman_ci_95_upper"],
            "interpretation": _interpret_ci(
                ci_hscc["spearman_ci_95_lower"],
                ci_hscc["spearman_ci_95_upper"],
                equivalence_bound=float(args.equivalence_bound),
            ),
        },
        "ndcg_at_10pct": {
            "delta_mean": ci_hscc["ndcg_delta_mean"],
            "ci_95_lower": ci_hscc["ndcg_ci_95_lower"],
            "ci_95_upper": ci_hscc["ndcg_ci_95_upper"],
            "interpretation": _interpret_ci(
                ci_hscc["ndcg_ci_95_lower"],
                ci_hscc["ndcg_ci_95_upper"],
                equivalence_bound=float(args.equivalence_bound),
            ),
        },
    }
    _write_json(resolve_project_path(args.out_hscc), payload_hscc)

    print("[OK] Bootstrap CI artifacts written")
    print(f" - a0={resolve_project_path(args.out_a0)}")
    print(f" - hscc={resolve_project_path(args.out_hscc)}")


if __name__ == "__main__":
    main()
