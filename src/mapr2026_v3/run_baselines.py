"""MAPR2026 v3 — Run analytical baselines (Group 1–3) and write ranking metrics.

Owner: Person 3

Inputs (expected, depending on baseline)
--------------------------------------
- data/processed/regression_targets.parquet
- data/processed/node_attributes.parquet
- data/processed/centrality_table.parquet
- data/processed/kshell_table.parquet
- data/processed/diffusion_proxies.parquet

Output (contract)
---------------
- outputs/mapr2026_v3_results/baseline_ranking_metrics.csv

Scaffold behavior
-----------------
- --dry-run writes an empty CSV with headers only.
- Real mode should implement: load y targets, compute y_pred per baseline, compute metrics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

try:
    from torch_geometric.nn import Node2Vec
except Exception:
    Node2Vec = None

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
class BaselineDataBundle:
    node_ids: pd.Series
    x_mlp: torch.Tensor
    y: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    edge_index: torch.Tensor
    scaler: MinMaxScaler
    split_mask_df: pd.DataFrame


class MLPRegressor(nn.Module):
    def __init__(self, in_features: int = 3, hidden_dim: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features).squeeze(-1)

    def reset_parameters(self) -> None:
        for module in self.modules():
            if module is self:
                continue
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()


def get_loss_function() -> nn.Module:
    return nn.HuberLoss(delta=1.0)


def _ensure_node_id_str(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Normalize potential float representations like "123.0" to "123"
    df["node_id"] = df["node_id"].astype(str).str.replace(r"\.0$", "", regex=True)
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

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


def load_baseline_data_bundle(
    node_attributes_path: str | Path = PATHS.node_attributes,
    targets_path: str | Path = PATHS.regression_targets,
    split_mask_path: str | Path = PATHS.split_masks,
    edgelist_path: str | Path = PATHS.graph_edgelist,
) -> BaselineDataBundle:
    node_attributes = pd.read_parquet(resolve_project_path(node_attributes_path))
    targets = pd.read_parquet(resolve_project_path(targets_path))
    targets = _ensure_node_id_str(targets)
    require_columns(targets, ["node_id", "y"], "regression_targets")

    features_df = _derive_features(node_attributes)
    merged = features_df.merge(targets[["node_id", "y"]], on="node_id", how="left")
    merged["y"] = pd.to_numeric(merged["y"], errors="coerce").fillna(0.0)

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

    feature_cols = ["views_log", "views_per_day", "life_time"]
    x_raw = merged[feature_cols].to_numpy(dtype=np.float32)
    y = torch.tensor(merged["y"].to_numpy(dtype=np.float32), dtype=torch.float32)

    scaler = MinMaxScaler()
    if train_mask_np.any():
        scaler.fit(x_raw[train_mask_np])
    else:
        scaler.fit(x_raw)
    x_scaled = scaler.transform(x_raw)

    edge_index = _build_edge_index(node_ids=node_ids, edgelist_path=edgelist_path)

    return BaselineDataBundle(
        node_ids=node_ids,
        x_mlp=torch.tensor(x_scaled, dtype=torch.float32),
        y=y,
        train_mask=torch.tensor(train_mask_np, dtype=torch.bool),
        val_mask=torch.tensor(val_mask_np, dtype=torch.bool),
        test_mask=torch.tensor(test_mask_np, dtype=torch.bool),
        edge_index=edge_index,
        scaler=scaler,
        split_mask_df=split_mask_df,
    )


def build_node2vec_model(edge_index: torch.Tensor, embedding_dim: int = 64) -> Any:
    if Node2Vec is None:
        raise ImportError("torch_geometric is required for Node2Vec but is not available.")
    return Node2Vec(
        edge_index=edge_index,
        embedding_dim=embedding_dim,
        walk_length=20,
        context_size=10,
        walks_per_node=20,
        num_negative_samples=1,
        p=1.0,
        q=1.0,
        sparse=True,
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


def eval_heuristic(bundle: BaselineDataBundle, metric_values: pd.Series, name: str) -> dict[str, float]:
    df = pd.DataFrame({"node_id": bundle.node_ids, "val": metric_values.astype(float).fillna(0.0)})
    df_sorted = df.set_index("node_id").loc[bundle.node_ids]
    
    t0 = time.time()
    y_pred = torch.tensor(df_sorted["val"].values, dtype=torch.float32)
    t1 = time.time()
    
    metrics = evaluate_on_test_mask(
        node_ids=bundle.node_ids,
        y_true=bundle.y,
        y_pred=y_pred,
        split_mask_df=bundle.split_mask_df,
    )
    
    return {
        "model_name": name,
        "spearman_rho": metrics["spearman_rho"],
        "spearman_rho_std": 0.0,
        "ndcg_at_10pct": metrics["ndcg_at_10pct"],
        "ndcg_at_10pct_std": 0.0,
        "precision_at_10pct": metrics["precision_at_10pct"],
        "precision_at_10pct_std": 0.0,
        "runtime_sec": float(t1 - t0),
        "train_sec": np.nan,
    }


def _empty_metrics_row(model_name: str) -> dict[str, float]:
    return {
        "model_name": model_name,
        "spearman_rho": np.nan,
        "spearman_rho_std": np.nan,
        "ndcg_at_10pct": np.nan,
        "ndcg_at_10pct_std": np.nan,
        "precision_at_10pct": np.nan,
        "precision_at_10pct_std": np.nan,
        "runtime_sec": np.nan,
        "train_sec": np.nan,
    }


def train_node2vec_5seeds(bundle: BaselineDataBundle, node2vec_epochs: int = 10) -> dict[str, float]:
    if Node2Vec is None:
        print("[WARN] Node2Vec unavailable (torch_geometric missing); writing NaN row.")
        return _empty_metrics_row("node2vec_lr")

    seed_metrics = []
    seed_runtimes = []
    seed_train_times = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_idx = torch.where(bundle.train_mask)[0].cpu().numpy()
    if train_idx.size == 0:
        print("[WARN] Empty training mask for Node2Vec+LR; writing NaN row.")
        return _empty_metrics_row("node2vec_lr")

    y_all = bundle.y.detach().cpu().numpy().astype(np.float32)

    for seed in TRAINING_SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = build_node2vec_model(bundle.edge_index).to(device)
        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
        loader = model.loader(batch_size=128, shuffle=True)

        t_train_0 = time.time()
        for _ in range(max(1, int(node2vec_epochs))):
            model.train()
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            z = model().detach().cpu().numpy().astype(np.float32)
        lr = LinearRegression()
        lr.fit(z[train_idx], y_all[train_idx])
        t_train_1 = time.time()

        model.eval()
        t0 = time.time()
        with torch.no_grad():
            z_eval = model().detach().cpu().numpy().astype(np.float32)
            y_pred_np = lr.predict(z_eval)
            y_pred = torch.tensor(y_pred_np, dtype=torch.float32)
        t1 = time.time()

        metrics = evaluate_on_test_mask(
            node_ids=bundle.node_ids,
            y_true=bundle.y,
            y_pred=y_pred,
            split_mask_df=bundle.split_mask_df,
        )
        seed_metrics.append(metrics)
        seed_runtimes.append(t1 - t0)
        seed_train_times.append(t_train_1 - t_train_0)

    return {
        "model_name": "node2vec_lr",
        "spearman_rho": float(np.mean([m["spearman_rho"] for m in seed_metrics])),
        "spearman_rho_std": float(np.std([m["spearman_rho"] for m in seed_metrics], ddof=0)),
        "ndcg_at_10pct": float(np.mean([m["ndcg_at_10pct"] for m in seed_metrics])),
        "ndcg_at_10pct_std": float(np.std([m["ndcg_at_10pct"] for m in seed_metrics], ddof=0)),
        "precision_at_10pct": float(np.mean([m["precision_at_10pct"] for m in seed_metrics])),
        "precision_at_10pct_std": float(np.std([m["precision_at_10pct"] for m in seed_metrics], ddof=0)),
        "runtime_sec": float(np.mean(seed_runtimes)),
        "train_sec": float(np.mean(seed_train_times)),
    }


def train_mlp_5seeds(bundle: BaselineDataBundle, max_epochs: int = MAX_EPOCHS) -> dict[str, float]:
    seed_metrics: list[dict[str, float]] = []
    seed_inference_runtimes: list[float] = []
    seed_train_runtimes: list[float] = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = bundle.x_mlp.to(device)
    y = bundle.y.to(device)
    train_mask = bundle.train_mask.to(device)
    split_mask_df = bundle.split_mask_df

    for seed in TRAINING_SEEDS:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        model = MLPRegressor(in_features=3, hidden_dim=128, dropout=0.3).to(device)
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = get_loss_function()

        t_train_0 = time.time()
        for _ in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred[train_mask], y[train_mask])
            loss.backward()
            optimizer.step()
        t_train_1 = time.time()

        model.eval()
        with torch.no_grad():
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            y_pred = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()

        inference_runtime_sec = float(t1 - t0)

        metrics = evaluate_on_test_mask(
            node_ids=bundle.node_ids,
            y_true=y,
            y_pred=y_pred,
            split_mask_df=split_mask_df,
        )
        seed_metrics.append(metrics)
        seed_inference_runtimes.append(inference_runtime_sec)
        seed_train_runtimes.append(float(t_train_1 - t_train_0))

    rho_values = np.array([m["spearman_rho"] for m in seed_metrics], dtype=float)
    ndcg_values = np.array([m["ndcg_at_10pct"] for m in seed_metrics], dtype=float)
    prec_values = np.array([m["precision_at_10pct"] for m in seed_metrics], dtype=float)

    return {
        "model_name": "mlp_raw_attr",
        "spearman_rho": float(np.mean(rho_values)),
        "spearman_rho_std": float(np.std(rho_values, ddof=0)),
        "ndcg_at_10pct": float(np.mean(ndcg_values)),
        "ndcg_at_10pct_std": float(np.std(ndcg_values, ddof=0)),
        "precision_at_10pct": float(np.mean(prec_values)),
        "precision_at_10pct_std": float(np.std(prec_values, ddof=0)),
        "runtime_sec": float(np.mean(np.array(seed_inference_runtimes, dtype=float))),
        "train_sec": float(np.mean(np.array(seed_train_runtimes, dtype=float))),
    }


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


def _safe_read_parquet(path_like: str | Path) -> pd.DataFrame | None:
    p = resolve_project_path(path_like)
    if not p.exists():
        return None
    return pd.read_parquet(p)


def _align_series_to_nodes(df: pd.DataFrame, node_ids: pd.Series, value_col: str) -> pd.Series:
    aligned = _ensure_node_id_str(df).set_index("node_id")
    clean_node_ids = node_ids.astype(str).str.replace(r"\.0$", "", regex=True)
    out = aligned.reindex(clean_node_ids)[value_col]
    out_np = pd.to_numeric(out, errors="coerce").fillna(0.0).to_numpy()
    return pd.Series(out_np, index=node_ids.index)


def collect_heuristic_rows(bundle: BaselineDataBundle) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []

    node_attr = _safe_read_parquet(PATHS.node_attributes)
    if node_attr is not None:
        node_attr = _ensure_node_id_str(node_attr)
        if "views" in node_attr.columns:
            rows.append(eval_heuristic(bundle, _align_series_to_nodes(node_attr, bundle.node_ids, "views"), "views"))
            derived = _derive_features(node_attr)
            rows.append(
                eval_heuristic(
                    bundle,
                    _align_series_to_nodes(derived, bundle.node_ids, "views_per_day"),
                    "views_day",
                )
            )
        if "degree" in node_attr.columns:
            rows.append(eval_heuristic(bundle, _align_series_to_nodes(node_attr, bundle.node_ids, "degree"), "degree"))

    centrality = _safe_read_parquet("data/processed/centrality_table.parquet")
    if centrality is not None:
        for col in ["pagerank", "betweenness", "degree"]:
            if col in centrality.columns and col != "degree":
                rows.append(eval_heuristic(bundle, _align_series_to_nodes(centrality, bundle.node_ids, col), col))

    kshell = _safe_read_parquet("data/processed/kshell_table.parquet")
    if kshell is not None:
        kshell_col = "kshell" if "kshell" in kshell.columns else ("k_shell" if "k_shell" in kshell.columns else None)
        if kshell_col is not None:
            rows.append(eval_heuristic(bundle, _align_series_to_nodes(kshell, bundle.node_ids, kshell_col), "kshell"))
    elif centrality is not None and "kshell" in centrality.columns:
        rows.append(eval_heuristic(bundle, _align_series_to_nodes(centrality, bundle.node_ids, "kshell"), "kshell"))

    proxies = _safe_read_parquet(PATHS.proxies)
    if proxies is not None:
        if "one_hop_spread" in proxies.columns:
            rows.append(eval_heuristic(bundle, _align_series_to_nodes(proxies, bundle.node_ids, "one_hop_spread"), "one_hop"))
        if "two_hop_spread" in proxies.columns:
            rows.append(eval_heuristic(bundle, _align_series_to_nodes(proxies, bundle.node_ids, "two_hop_spread"), "two_hop"))

    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MAPR2026 v3 baseline runner scaffold")
    p.add_argument("--out-dir", default=PATHS.results_dir)
    p.add_argument("--out-csv", default=str(Path(PATHS.results_dir) / "baseline_ranking_metrics.csv"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    p.add_argument("--node2vec-epochs", type=int, default=10)
    p.add_argument("--skip-node2vec", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = resolve_project_path(args.out_dir)
    ensure_dir(out_dir)

    out_csv = resolve_project_path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    cols = [
        "model_name",
        "spearman_rho",
        "spearman_rho_std",
        "ndcg_at_10pct",
        "ndcg_at_10pct_std",
        "precision_at_10pct",
        "precision_at_10pct_std",
        "runtime_sec",
        "train_sec",
    ]

    if args.dry_run:
        pd.DataFrame(columns=cols).to_csv(out_csv, index=False)
        print(f"[OK] Wrote dry-run baseline metrics header: {out_csv} (timestamp={now_iso()})")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training device: {device}")

    bundle = load_baseline_data_bundle()
    if Node2Vec is None:
        print("[WARN] torch_geometric not installed; Node2Vec architecture check skipped.")
    else:
        _ = build_node2vec_model(bundle.edge_index)

    results_list = collect_heuristic_rows(bundle)
    results_list.append(train_mlp_5seeds(bundle=bundle, max_epochs=args.max_epochs))
    if args.skip_node2vec:
        print("[INFO] Skipping Node2Vec+LR by flag (--skip-node2vec).")
    else:
        results_list.append(train_node2vec_5seeds(bundle=bundle, node2vec_epochs=args.node2vec_epochs))

    _upsert_rows(out_csv, rows=results_list, cols=cols, key="model_name")
    _upsert_runtime_rows(results_list)

    print("[OK] Baseline training/evaluation completed.")
    print(f" - models_written={len(results_list)}")
    print(f" - n_nodes={bundle.x_mlp.shape[0]}, n_features={bundle.x_mlp.shape[1]}")
    print(f" - train/val/test={int(bundle.train_mask.sum())}/{int(bundle.val_mask.sum())}/{int(bundle.test_mask.sum())}")
    print(f" - output={out_csv}")


if __name__ == "__main__":
    main()
