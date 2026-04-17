"""MAPR2026 v3 — Run surrogate models (Node2Vec+LR / MLP / optional GNN) scaffold.

Owner: Person 3

Inputs
------
- data/processed/regression_targets.parquet
- data/processed/node_attributes.parquet
- (optional) graph artifacts for embeddings/GNN

Outputs
-------
- outputs/mapr2026_v3_results/surrogate_ranking_metrics.csv (suggested)
- (optional) model checkpoints under outputs/mapr2026_v3_results/

Scaffold behavior
-----------------
- --dry-run writes a header-only CSV.
- Real mode should implement training (5 seeds) + aggregation per MAPR2026 v3.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import SAGEConv
except Exception:
    Data = None
    SAGEConv = None

from _shared import PATHS, ensure_dir, now_iso, read_edgelist_pairs, require_columns
from eval_ranking_harness import (
    apply_test_mask,
    compute_metrics,
    load_split_mask as load_shared_split_mask,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_EPOCHS = 200
TRAINING_SEEDS = [42, 123, 456, 789, 1024]


def resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


@dataclass
class SurrogateDataBundle:
    node_ids: pd.Series
    graph_data: Any
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    sample_weight: torch.Tensor
    train_eligible_mask: torch.Tensor
    scaler: MinMaxScaler
    split_mask_df: pd.DataFrame
    label_gate: dict[str, Any]


class GraphSAGERegressor(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        if SAGEConv is None:
            raise ImportError("torch_geometric is required for GraphSAGE but is not available.")
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean")
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr="mean")
        self.head = nn.Linear(hidden_channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        out = self.head(x)
        return out.squeeze(-1)

    def reset_parameters(self) -> None:
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.head.reset_parameters()


def get_loss_function() -> nn.Module:
    return nn.HuberLoss(delta=1.0)


def _ensure_node_id_str(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["node_id"] = df["node_id"].astype(str)
    return df


def _derive_features(node_attributes: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_node_id_str(node_attributes)
    require_columns(df, ["node_id"], "node_attributes")

    if "views" in df.columns:
        views_raw = pd.to_numeric(df["views"], errors="coerce").fillna(0.0)
    else:
        views_raw = pd.Series(np.zeros(len(df), dtype=float), index=df.index)

    if "life_time" in df.columns:
        life_time = pd.to_numeric(df["life_time"], errors="coerce").fillna(1.0)
    elif "life_time_days" in df.columns:
        life_time = pd.to_numeric(df["life_time_days"], errors="coerce").fillna(1.0)
    else:
        life_time = pd.Series(np.ones(len(df), dtype=float), index=df.index)

    life_time = life_time.clip(lower=1.0)
    views_log = np.log1p(views_raw)
    views_per_day = views_raw / life_time

    features = pd.DataFrame(
        {
            "node_id": df["node_id"],
            "views_log": views_log.astype(float),
            "views_per_day": views_per_day.astype(float),
            "life_time": life_time.astype(float),
        }
    )
    return features


def _parse_clip_range(raw: str) -> tuple[float, float]:
    parts = [x.strip() for x in str(raw).split(",") if x.strip()]
    if len(parts) != 2:
        raise ValueError("--uncertainty-weight-clip must be in format 'min,max'")
    lo = float(parts[0])
    hi = float(parts[1])
    if lo <= 0 or hi <= 0 or hi < lo:
        raise ValueError("--uncertainty-weight-clip requires 0 < min <= max")
    return lo, hi


def _parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if token:
            values.append(int(token))
    if not values:
        raise ValueError("List cannot be empty")
    return values


def _parse_str_list(raw: str) -> list[str]:
    values = [token.strip() for token in str(raw).split(",") if token.strip()]
    if not values:
        raise ValueError("List cannot be empty")
    return values


def split_sets_from_shared_mask(split_mask_path: str | Path) -> dict[str, set[str]]:
    split_df = load_shared_split_mask(resolve_project_path(split_mask_path))
    split_df = _ensure_node_id_str(split_df)

    out = {
        "train": set(split_df.loc[split_df["split"] == "train", "node_id"].tolist()),
        "test": set(split_df.loc[split_df["split"] == "test", "node_id"].tolist()),
        "val": set(),
    }
    if out["train"]:
        train_sorted = sorted(out["train"])
        n_val = max(1, int(0.1 * len(train_sorted)))
        out["val"] = set(train_sorted[:n_val])
        out["train"] = set(train_sorted[n_val:])
    return out


def load_shared_split_dataframe(split_mask_path: str | Path) -> pd.DataFrame:
    split_df = load_shared_split_mask(resolve_project_path(split_mask_path))
    split_df = _ensure_node_id_str(split_df)
    return split_df


def _build_edge_index(node_ids: pd.Series, edgelist_path: str | Path) -> torch.Tensor:
    node_to_idx: dict[str, int] = {node_id: i for i, node_id in enumerate(node_ids.tolist())}
    src_nodes, dst_nodes = read_edgelist_pairs(resolve_project_path(edgelist_path))

    edges: list[tuple[int, int]] = []
    for src, dst in zip(src_nodes, dst_nodes):
        if src in node_to_idx and dst in node_to_idx:
            s_idx = node_to_idx[src]
            d_idx = node_to_idx[dst]
            edges.append((s_idx, d_idx))
            edges.append((d_idx, s_idx))

    if not edges:
        return torch.empty((2, 0), dtype=torch.long)

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def load_surrogate_data_bundle(
    feature_mode: str = "raw_attr",
    node_attributes_path: str | Path = PATHS.node_attributes,
    centrality_path: str | Path = "data/processed/centrality_table.parquet",
    targets_path: str | Path = PATHS.regression_targets,
    ic_scores_path: str | Path = PATHS.ic_scores,
    split_mask_path: str | Path = PATHS.split_masks,
    edgelist_path: str | Path = PATHS.graph_edgelist,
    use_label_uncertainty_weights: bool = False,
    uncertainty_eps: float = 1e-6,
    uncertainty_weight_clip: tuple[float, float] = (0.25, 4.0),
    drop_noisy_quantile: float = 0.0,
    min_train_kept: int = 200,
    stability_summary_csv: str | Path | None = None,
) -> SurrogateDataBundle:
    if Data is None:
        raise ImportError("torch_geometric is required for surrogate data bundle construction.")

    node_attributes = pd.read_parquet(resolve_project_path(node_attributes_path))
    node_attributes = _ensure_node_id_str(node_attributes)
    require_columns(node_attributes, ["node_id"], "node_attributes")
    base_df = node_attributes[["node_id"]].copy()

    targets = pd.read_parquet(resolve_project_path(targets_path))
    targets = _ensure_node_id_str(targets)
    require_columns(targets, ["node_id", "y"], "regression_targets")
    y_df = targets[["node_id", "y"]].copy()
    y_df["y"] = pd.to_numeric(y_df["y"], errors="coerce")

    raw_features_df = _derive_features(node_attributes)
    merged = base_df.merge(y_df, on="node_id", how="left")

    if feature_mode == "raw_attr":
        merged = merged.merge(raw_features_df, on="node_id", how="left")
        feature_cols = ["views_log", "views_per_day", "life_time"]
    elif feature_mode == "random":
        rng = np.random.default_rng(42)
        random_features = rng.standard_normal((len(merged), 3), dtype=np.float32)
        merged["rand_feat_0"] = random_features[:, 0]
        merged["rand_feat_1"] = random_features[:, 1]
        merged["rand_feat_2"] = random_features[:, 2]
        feature_cols = ["rand_feat_0", "rand_feat_1", "rand_feat_2"]
    elif feature_mode in {"centrality", "graph_only", "full"}:
        centrality_df = pd.read_parquet(resolve_project_path(centrality_path))
        centrality_df = _ensure_node_id_str(centrality_df)

        degree_col = "degree" if "degree" in centrality_df.columns else None
        pagerank_col = "pagerank" if "pagerank" in centrality_df.columns else None
        kshell_col = "kshell" if "kshell" in centrality_df.columns else ("k_shell" if "k_shell" in centrality_df.columns else None)

        selected = [c for c in [degree_col, pagerank_col, kshell_col] if c is not None]
        if not selected:
            raise ValueError("centrality_table.parquet does not contain required columns for GNN ablations.")

        centrality_sel = centrality_df[["node_id", *selected]].copy()
        if kshell_col and kshell_col != "kshell":
            centrality_sel = centrality_sel.rename(columns={kshell_col: "kshell"})
            selected = ["kshell" if c == kshell_col else c for c in selected]

        if feature_mode == "centrality":
            required = ["degree", "pagerank", "kshell"]
            missing = [c for c in required if c not in centrality_sel.columns]
            if missing:
                raise ValueError(f"Missing required centrality features: {missing}")
            merged = merged.merge(centrality_sel[["node_id", *required]], on="node_id", how="left")
            feature_cols = required
        elif feature_mode == "graph_only":
            if "degree" not in centrality_sel.columns:
                raise ValueError("graph_only requires 'degree' in centrality_table.parquet")
            merged = merged.merge(centrality_sel[["node_id", "degree"]], on="node_id", how="left")
            feature_cols = ["degree"]
        else:
            required = ["degree", "pagerank", "kshell"]
            missing = [c for c in required if c not in centrality_sel.columns]
            if missing:
                raise ValueError(f"Missing required full-feature centrality columns: {missing}")
            merged = merged.merge(raw_features_df, on="node_id", how="left")
            merged = merged.merge(centrality_sel[["node_id", *required]], on="node_id", how="left")
            feature_cols = ["views_log", "views_per_day", "life_time", "degree", "pagerank", "kshell"]
    else:
        raise ValueError(f"Unsupported feature_mode={feature_mode}")

    split_mask_df = load_shared_split_dataframe(split_mask_path)
    split = {
        "train": set(split_mask_df.loc[split_mask_df["split"] == "train", "node_id"].tolist()),
        "test": set(split_mask_df.loc[split_mask_df["split"] == "test", "node_id"].tolist()),
        "val": set(),
    }
    if split["train"]:
        train_sorted = sorted(split["train"])
        n_val = max(1, int(0.1 * len(train_sorted)))
        split["val"] = set(train_sorted[:n_val])
        split["train"] = set(train_sorted[n_val:])

    node_ids = merged["node_id"].astype(str)

    train_mask_np = node_ids.isin(split["train"]).to_numpy(dtype=bool)
    val_mask_np = node_ids.isin(split["val"]).to_numpy(dtype=bool)
    test_mask_np = node_ids.isin(split["test"]).to_numpy(dtype=bool)

    merged["y"] = pd.to_numeric(merged["y"], errors="coerce").fillna(0.0)
    merged[feature_cols] = merged[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    x_raw = merged[feature_cols].to_numpy(dtype=np.float32)

    ic_std = np.full(len(merged), np.nan, dtype=float)
    ic_scores_resolved = resolve_project_path(ic_scores_path)
    if ic_scores_resolved.exists():
        ic_scores_df = _ensure_node_id_str(pd.read_parquet(ic_scores_resolved))
        if "ic_score_std" in ic_scores_df.columns:
            std_ser = pd.to_numeric(
                ic_scores_df.set_index("node_id")["ic_score_std"], errors="coerce"
            )
            ic_std = std_ser.reindex(node_ids).to_numpy(dtype=float)

    finite_mask = np.isfinite(ic_std)
    std_fill = float(np.nanmedian(ic_std[finite_mask])) if finite_mask.any() else 1.0
    if not np.isfinite(std_fill) or std_fill < 0:
        std_fill = 1.0
    ic_std = np.where(np.isfinite(ic_std), ic_std, std_fill)
    ic_std = np.maximum(ic_std, 0.0)

    sample_weight_np = np.ones(len(merged), dtype=np.float32)
    if use_label_uncertainty_weights:
        inv_std = 1.0 / (ic_std + float(max(uncertainty_eps, 1e-12)))
        ref = float(np.median(inv_std[train_mask_np])) if train_mask_np.any() else float(np.median(inv_std))
        if not np.isfinite(ref) or ref <= 0:
            ref = 1.0
        lo, hi = uncertainty_weight_clip
        sample_weight_np = np.clip(inv_std / ref, float(lo), float(hi)).astype(np.float32)

    train_eligible_np = np.ones(len(merged), dtype=bool)
    dropped_threshold = None
    dropped_train_nodes = 0
    if float(drop_noisy_quantile) > 0:
        q = float(np.clip(drop_noisy_quantile, 0.0, 0.999))
        if train_mask_np.any():
            dropped_threshold = float(np.quantile(ic_std[train_mask_np], q))
        else:
            dropped_threshold = float(np.quantile(ic_std, q))

        drop_mask = train_mask_np & (ic_std > dropped_threshold)
        train_eligible_np[drop_mask] = False

        n_train_total = int(np.sum(train_mask_np))
        n_train_kept = int(np.sum(train_mask_np & train_eligible_np))
        min_keep = int(min(max(1, int(min_train_kept)), n_train_total)) if n_train_total > 0 else 0
        if n_train_total > 0 and n_train_kept < min_keep:
            train_idx = np.where(train_mask_np)[0]
            order = train_idx[np.argsort(ic_std[train_idx])]
            keep_idx = order[:min_keep]
            train_eligible_np[train_mask_np] = False
            train_eligible_np[keep_idx] = True

        dropped_train_nodes = int(np.sum(train_mask_np & (~train_eligible_np)))

    label_gate: dict[str, Any] = {
        "use_label_uncertainty_weights": bool(use_label_uncertainty_weights),
        "uncertainty_eps": float(uncertainty_eps),
        "uncertainty_weight_clip": [
            float(uncertainty_weight_clip[0]),
            float(uncertainty_weight_clip[1]),
        ],
        "drop_noisy_quantile": float(drop_noisy_quantile),
        "drop_threshold_ic_score_std": (None if dropped_threshold is None else float(dropped_threshold)),
        "n_nodes": int(len(merged)),
        "n_train_total": int(np.sum(train_mask_np)),
        "n_train_kept": int(np.sum(train_mask_np & train_eligible_np)),
        "n_train_dropped": int(dropped_train_nodes),
        "train_dropped_rate": (
            float(dropped_train_nodes / max(1, int(np.sum(train_mask_np)))) if train_mask_np.any() else 0.0
        ),
        "ic_score_std": {
            "min": float(np.min(ic_std)),
            "p50": float(np.median(ic_std)),
            "p90": float(np.quantile(ic_std, 0.90)),
            "max": float(np.max(ic_std)),
        },
        "sample_weight": {
            "min": float(np.min(sample_weight_np)),
            "p50": float(np.median(sample_weight_np)),
            "p90": float(np.quantile(sample_weight_np, 0.90)),
            "max": float(np.max(sample_weight_np)),
        },
    }

    if stability_summary_csv:
        stability_path = resolve_project_path(stability_summary_csv)
        if stability_path.exists():
            try:
                stability_df = pd.read_csv(stability_path)
                if len(stability_df) > 0 and "n_runs" in stability_df.columns:
                    best_row = stability_df.sort_values("n_runs").iloc[-1]
                    label_gate["stability_reference"] = {
                        "path": str(stability_path).replace("\\", "/"),
                        "n_runs": int(best_row.get("n_runs", -1)),
                        "jaccard_mean": float(best_row.get("jaccard_mean", np.nan)),
                        "jaccard_min": float(best_row.get("jaccard_min", np.nan)),
                        "spearman_mean": float(best_row.get("spearman_mean", np.nan)),
                        "spearman_min": float(best_row.get("spearman_min", np.nan)),
                    }
            except Exception as exc:
                label_gate["stability_reference_error"] = str(exc)

    scaler = MinMaxScaler()
    if train_mask_np.any():
        scaler.fit(x_raw[train_mask_np])
    else:
        scaler.fit(x_raw)
    x_scaled = scaler.transform(x_raw)

    y_tensor = torch.tensor(merged["y"].to_numpy(dtype=np.float32), dtype=torch.float32)
    edge_index = _build_edge_index(node_ids=node_ids, edgelist_path=edgelist_path)

    graph_data = Data(
        x=torch.tensor(x_scaled, dtype=torch.float32),
        edge_index=edge_index,
        y=y_tensor,
    )
    graph_data.train_mask = torch.tensor(train_mask_np, dtype=torch.bool)
    graph_data.val_mask = torch.tensor(val_mask_np, dtype=torch.bool)
    graph_data.test_mask = torch.tensor(test_mask_np, dtype=torch.bool)
    graph_data.sample_weight = torch.tensor(sample_weight_np, dtype=torch.float32)
    graph_data.train_eligible_mask = torch.tensor(train_eligible_np, dtype=torch.bool)

    return SurrogateDataBundle(
        node_ids=node_ids,
        graph_data=graph_data,
        train_mask=graph_data.train_mask,
        val_mask=graph_data.val_mask,
        test_mask=graph_data.test_mask,
        sample_weight=graph_data.sample_weight,
        train_eligible_mask=graph_data.train_eligible_mask,
        scaler=scaler,
        split_mask_df=split_mask_df,
        label_gate=label_gate,
    )


def evaluate_on_test_mask(
    node_ids: pd.Series,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    split_mask_df: pd.DataFrame,
) -> dict[str, float]:
    eval_df = pd.DataFrame(
        {
            "node_id": node_ids.astype(str).tolist(),
            "y_true": y_true.detach().cpu().numpy().astype(float),
            "y_pred": y_pred.detach().cpu().numpy().astype(float),
        }
    )
    eval_test = apply_test_mask(eval_df, split_mask_df, node_id_col="node_id")
    if eval_test.empty:
        raise ValueError("Shared split mask produced an empty test set.")

    metrics = compute_metrics(
        eval_test["y_true"].to_numpy(dtype=float),
        eval_test["y_pred"].to_numpy(dtype=float),
    )

    return {
        "spearman_rho": metrics.spearman_rho,
        "ndcg_at_10pct": metrics.ndcg_at_10pct,
        "precision_at_10pct": metrics.precision_at_10pct,
    }


def train_surrogate_5seeds(
    bundle: SurrogateDataBundle,
    max_epochs: int = MAX_EPOCHS,
    model_name: str = "gnn_raw_attr",
    randomize_train_target: bool = False,
    training_seeds: list[int] | None = None,
) -> tuple[dict[str, float], np.ndarray]:
    seed_metrics: list[dict[str, float]] = []
    seed_inference_runtimes: list[float] = []
    seed_train_runtimes: list[float] = []
    seed_predictions: list[np.ndarray] = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = bundle.graph_data.to(device)
    sample_weight = bundle.sample_weight.to(device)
    train_eligible_mask = bundle.train_eligible_mask.to(device)
    seeds = training_seeds if training_seeds is not None else TRAINING_SEEDS

    for seed in seeds:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        model = GraphSAGERegressor(in_channels=data.x.shape[1], hidden_channels=128, dropout=0.3).to(device)
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_mask_eff = data.train_mask & train_eligible_mask
        if int(train_mask_eff.sum().item()) == 0:
            raise ValueError("No train nodes remain after label-gate filtering.")

        y_train_target = data.y.clone()
        if randomize_train_target:
            train_idx = torch.where(train_mask_eff)[0]
            if train_idx.numel() > 1:
                perm = train_idx[torch.randperm(train_idx.numel(), device=device)]
                y_train_target[train_idx] = data.y[perm]

        t_train_0 = time.time()
        for _ in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            pred = model(data.x, data.edge_index)
            loss_vec = F.huber_loss(
                pred[train_mask_eff],
                y_train_target[train_mask_eff],
                delta=1.0,
                reduction="none",
            )
            train_w = sample_weight[train_mask_eff]
            loss = torch.sum(loss_vec * train_w) / torch.clamp(torch.sum(train_w), min=1e-8)
            loss.backward()
            optimizer.step()
        t_train_1 = time.time()

        model.eval()
        with torch.no_grad():
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            y_pred = model(data.x, data.edge_index)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()

        inference_runtime_sec = float(t1 - t0)

        metrics = evaluate_on_test_mask(
            node_ids=bundle.node_ids,
            y_true=data.y,
            y_pred=y_pred,
            split_mask_df=bundle.split_mask_df,
        )
        seed_metrics.append(metrics)
        seed_inference_runtimes.append(inference_runtime_sec)
        seed_train_runtimes.append(float(t_train_1 - t_train_0))
        seed_predictions.append(y_pred.detach().cpu().numpy().astype(np.float32))

    rho_values = np.array([m["spearman_rho"] for m in seed_metrics], dtype=float)
    ndcg_values = np.array([m["ndcg_at_10pct"] for m in seed_metrics], dtype=float)
    precision_values = np.array([m["precision_at_10pct"] for m in seed_metrics], dtype=float)

    row = {
        "model_name": model_name,
        "spearman_rho_mean": float(np.mean(rho_values)),
        "spearman_rho_std": float(np.std(rho_values, ddof=0)),
        "ndcg_mean": float(np.mean(ndcg_values)),
        "ndcg_std": float(np.std(ndcg_values, ddof=0)),
        "precision_mean": float(np.mean(precision_values)),
        "precision_std": float(np.std(precision_values, ddof=0)),
        "runtime_sec": float(np.mean(np.array(seed_inference_runtimes, dtype=float))),
        "train_sec": float(np.mean(np.array(seed_train_runtimes, dtype=float))),
    }
    mean_prediction = np.mean(np.stack(seed_predictions, axis=0), axis=0)
    return row, mean_prediction


def _upsert_rows(csv_path: Path, rows: list[dict[str, float]], cols: list[str], key: str = "model_name") -> None:
    new_df = pd.DataFrame(rows)
    for col in cols:
        if col not in new_df.columns:
            new_df[col] = np.nan
    new_df = new_df[cols].copy()

    if csv_path.exists():
        old_df = pd.read_csv(csv_path)
        for col in cols:
            if col not in old_df.columns:
                old_df[col] = np.nan
        old_df = old_df[cols].copy()
        merged = pd.concat([old_df, new_df], ignore_index=True)
    else:
        merged = new_df

    merged = merged.drop_duplicates(subset=[key], keep="last")
    merged = merged.sort_values(key).reset_index(drop=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(csv_path, index=False)


def _upsert_runtime_rows(rows: list[dict[str, float]]) -> None:
    runtime_path = resolve_project_path(PATHS.runtime_csv)
    cols = ["model_name", "inference_sec_full_graph", "train_sec"]
    runtime_rows = []
    for row in rows:
        runtime_rows.append(
            {
                "model_name": row["model_name"],
                "inference_sec_full_graph": row.get("runtime_sec", np.nan),
                "train_sec": row.get("train_sec", np.nan),
            }
        )
    _upsert_rows(runtime_path, runtime_rows, cols=cols, key="model_name")


def _write_per_group_prediction_error(
    node_ids: pd.Series,
    y_true: torch.Tensor,
    split_mask_df: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    typology_path = resolve_project_path(PATHS.typology)
    base_df = pd.DataFrame(
        {
            "node_id": node_ids.astype(str).tolist(),
            "y_true": y_true.detach().cpu().numpy().astype(float),
        }
    )
    test_df = apply_test_mask(base_df, split_mask_df, node_id_col="node_id")

    group_df = None
    if typology_path.exists():
        raw_group = pd.read_parquet(typology_path)
        raw_group = _ensure_node_id_str(raw_group)
        for candidate in ["typology", "typology_label", "quadrant", "group", "label"]:
            if candidate in raw_group.columns:
                group_df = raw_group[["node_id", candidate]].rename(columns={candidate: "group_label"})
                break

    if group_df is None:
        group_df = pd.DataFrame({"node_id": test_df["node_id"].astype(str), "group_label": "all_test"})

    rows: list[dict[str, float]] = []
    for model_name, pred in predictions.items():
        pred_df = pd.DataFrame({"node_id": node_ids.astype(str).tolist(), "y_pred": pred.astype(float)})
        pred_test = apply_test_mask(pred_df, split_mask_df, node_id_col="node_id")
        merged = test_df.merge(pred_test, on="node_id", how="inner").merge(group_df, on="node_id", how="left")
        merged["group_label"] = merged["group_label"].fillna("unknown")
        merged["abs_err"] = (merged["y_true"] - merged["y_pred"]).abs()
        merged["sq_err"] = (merged["y_true"] - merged["y_pred"]) ** 2

        grouped = merged.groupby("group_label", dropna=False)
        for group_label, g in grouped:
            n_nodes = int(g.shape[0])
            if n_nodes < 20:
                continue
            rho = float(compute_metrics(g["y_true"].to_numpy(dtype=float), g["y_pred"].to_numpy(dtype=float)).spearman_rho)
            rows.append(
                {
                    "model_name": model_name,
                    "typology_group": str(group_label),
                    "n_nodes": n_nodes,
                    "spearman_rho": rho,
                    "mae": float(g["abs_err"].mean()),
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows, columns=["model_name", "typology_group", "n_nodes", "spearman_rho", "mae"])
    out_df.sort_values(["model_name", "typology_group"]).to_csv(out_path, index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MAPR2026 v3 surrogate runner scaffold")
    p.add_argument("--out-dir", default=PATHS.results_dir)
    p.add_argument("--out-csv", default=str(Path(PATHS.results_dir) / "surrogate_ranking_metrics.csv"))
    p.add_argument(
        "--per-group-error-csv",
        default=str(Path(PATHS.results_dir) / "per_group_prediction_error.csv"),
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    p.add_argument("--run-random-sanity", action="store_true")
    p.add_argument("--only-random", action="store_true")
    p.add_argument("--skip-gnn-full", action="store_true")
    p.add_argument("--ic-scores-path", default=PATHS.ic_scores)
    p.add_argument("--use-label-uncertainty-weights", action="store_true")
    p.add_argument("--uncertainty-eps", type=float, default=1e-6)
    p.add_argument("--uncertainty-weight-clip", default="0.25,4.0")
    p.add_argument("--drop-noisy-quantile", type=float, default=0.0)
    p.add_argument("--min-train-kept", type=int, default=200)
    p.add_argument("--stability-summary-csv", default="")
    p.add_argument(
        "--training-seeds",
        default=",".join(str(s) for s in TRAINING_SEEDS),
        help="Comma-separated list of random seeds for training/evaluation.",
    )
    p.add_argument(
        "--models",
        default="",
        help="Optional comma-separated subset of model names to run.",
    )
    p.add_argument(
        "--label-gate-report-json",
        default=str(Path(PATHS.results_dir) / "label_gate_report.json"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = resolve_project_path(args.out_dir)
    ensure_dir(out_dir)

    out_csv = resolve_project_path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    cols = [
        "model_name",
        "spearman_rho_mean",
        "spearman_rho_std",
        "ndcg_mean",
        "ndcg_std",
        "precision_mean",
        "precision_std",
        "runtime_sec",
        "train_sec",
    ]

    if args.dry_run:
        pd.DataFrame(columns=cols).to_csv(out_csv, index=False)
        print(f"[OK] Wrote dry-run surrogate metrics header: {out_csv} (timestamp={now_iso()})")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training device: {device}")
    weight_clip = _parse_clip_range(args.uncertainty_weight_clip)
    training_seeds = _parse_int_list(args.training_seeds)

    model_specs: list[tuple[str, str, bool]] = [
        ("gnn_raw_attr", "raw_attr", False),
        ("gnn_graph_only", "graph_only", False),
        ("gnn_centrality", "centrality", False),
        ("gnn_full", "full", False),
    ]
    if args.skip_gnn_full:
        model_specs = [spec for spec in model_specs if spec[0] != "gnn_full"]
    if args.run_random_sanity:
        model_specs.append(("gnn_random", "random", False))
    if args.only_random:
        model_specs = [("gnn_random", "random", False)]
    if args.models:
        selected_models = set(_parse_str_list(args.models))
        available_models = {name for name, _, _ in model_specs}
        unknown_models = sorted(selected_models - available_models)
        if unknown_models:
            raise ValueError(f"Unknown model names in --models: {unknown_models}")
        model_specs = [spec for spec in model_specs if spec[0] in selected_models]

    results_list: list[dict[str, float]] = []
    predictions_by_model: dict[str, np.ndarray] = {}
    base_bundle: SurrogateDataBundle | None = None

    for model_name, feature_mode, randomize_train_target in model_specs:
        try:
            bundle = load_surrogate_data_bundle(
                feature_mode=feature_mode,
                ic_scores_path=args.ic_scores_path,
                use_label_uncertainty_weights=bool(args.use_label_uncertainty_weights),
                uncertainty_eps=float(args.uncertainty_eps),
                uncertainty_weight_clip=weight_clip,
                drop_noisy_quantile=float(args.drop_noisy_quantile),
                min_train_kept=int(args.min_train_kept),
                stability_summary_csv=(args.stability_summary_csv or None),
            )
            if base_bundle is None:
                base_bundle = bundle
            row, pred = train_surrogate_5seeds(
                bundle=bundle,
                max_epochs=args.max_epochs,
                model_name=model_name,
                randomize_train_target=randomize_train_target,
                training_seeds=training_seeds,
            )
            results_list.append(row)
            predictions_by_model[model_name] = pred
        except Exception as exc:
            print(f"[WARN] Skipping {model_name}: {exc}")

    if not results_list:
        raise RuntimeError("No surrogate models produced results.")

    _upsert_rows(out_csv, rows=results_list, cols=cols, key="model_name")
    _upsert_runtime_rows(results_list)

    if base_bundle is not None:
        gate_path = resolve_project_path(args.label_gate_report_json)
        gate_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": now_iso(),
            "config": {
                "ic_scores_path": str(resolve_project_path(args.ic_scores_path)).replace("\\", "/"),
                "use_label_uncertainty_weights": bool(args.use_label_uncertainty_weights),
                "uncertainty_eps": float(args.uncertainty_eps),
                "uncertainty_weight_clip": [float(weight_clip[0]), float(weight_clip[1])],
                "drop_noisy_quantile": float(args.drop_noisy_quantile),
                "min_train_kept": int(args.min_train_kept),
                "stability_summary_csv": (
                    str(resolve_project_path(args.stability_summary_csv)).replace("\\", "/")
                    if args.stability_summary_csv
                    else ""
                ),
            },
            "label_gate": base_bundle.label_gate,
        }
        with gate_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[OK] Wrote label gate report: {gate_path}")

    if base_bundle is not None and predictions_by_model:
        per_group_path = resolve_project_path(args.per_group_error_csv)
        _write_per_group_prediction_error(
            node_ids=base_bundle.node_ids,
            y_true=base_bundle.graph_data.y,
            split_mask_df=base_bundle.split_mask_df,
            predictions=predictions_by_model,
            out_path=per_group_path,
        )

    print("[OK] Surrogate training/evaluation completed with 5 seeds.")
    print(f" - models_written={len(results_list)}")
    print(f" - n_nodes={base_bundle.graph_data.x.shape[0]}, n_edges={base_bundle.graph_data.edge_index.shape[1]}")
    print(f" - output={out_csv}")


if __name__ == "__main__":
    main()
