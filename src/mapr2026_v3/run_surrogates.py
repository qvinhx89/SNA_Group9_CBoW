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
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import MinMaxScaler

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import SAGEConv
except Exception:
    Data = None
    SAGEConv = None

from _shared import PATHS, ensure_dir, now_iso, read_edgelist_pairs, require_columns
from eval_ranking_harness import load_split_mask as load_shared_split_mask


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
    scaler: MinMaxScaler


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
    node_attributes_path: str | Path = PATHS.node_attributes,
    targets_path: str | Path = PATHS.regression_targets,
    split_mask_path: str | Path = PATHS.split_masks,
    edgelist_path: str | Path = PATHS.graph_edgelist,
) -> SurrogateDataBundle:
    if Data is None:
        raise ImportError("torch_geometric is required for surrogate data bundle construction.")

    node_attributes = pd.read_parquet(resolve_project_path(node_attributes_path))
    targets = pd.read_parquet(resolve_project_path(targets_path))
    targets = _ensure_node_id_str(targets)
    require_columns(targets, ["node_id", "y"], "regression_targets")

    features_df = _derive_features(node_attributes)
    merged = features_df.merge(targets[["node_id", "y"]], on="node_id", how="left")
    merged["y"] = pd.to_numeric(merged["y"], errors="coerce").fillna(0.0)

    split = split_sets_from_shared_mask(split_mask_path)
    node_ids = merged["node_id"].astype(str)

    train_mask_np = node_ids.isin(split["train"]).to_numpy(dtype=bool)
    val_mask_np = node_ids.isin(split["val"]).to_numpy(dtype=bool)
    test_mask_np = node_ids.isin(split["test"]).to_numpy(dtype=bool)

    feature_cols = ["views_log", "views_per_day", "life_time"]
    x_raw = merged[feature_cols].to_numpy(dtype=np.float32)

    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x_raw)

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

    return SurrogateDataBundle(
        node_ids=node_ids,
        graph_data=graph_data,
        train_mask=graph_data.train_mask,
        val_mask=graph_data.val_mask,
        test_mask=graph_data.test_mask,
        scaler=scaler,
    )


def evaluate_on_test_mask(y_true: torch.Tensor, y_pred: torch.Tensor, test_mask: torch.Tensor) -> dict[str, float]:
    y_true_test = y_true[test_mask].detach().cpu().numpy().astype(float)
    y_pred_test = y_pred[test_mask].detach().cpu().numpy().astype(float)

    if y_true_test.size == 0:
        raise ValueError("Test mask has zero nodes; cannot evaluate metrics.")

    rho = spearmanr(y_true_test, y_pred_test).statistic
    if rho is None or np.isnan(rho):
        rho = 0.0

    k = max(1, int(np.ceil(0.10 * y_true_test.size)))
    ndcg = float(ndcg_score(y_true_test.reshape(1, -1), y_pred_test.reshape(1, -1), k=k))

    pred_top = set(np.argsort(-y_pred_test)[:k].tolist())
    true_top = set(np.argsort(-y_true_test)[:k].tolist())
    precision = float(len(pred_top.intersection(true_top)) / k)

    return {
        "spearman_rho": float(rho),
        "ndcg_at_10pct": ndcg,
        "precision_at_10pct": precision,
    }


def train_surrogate_5seeds(bundle: SurrogateDataBundle, max_epochs: int = MAX_EPOCHS) -> dict[str, float]:
    seed_metrics: list[dict[str, float]] = []
    seed_inference_runtimes: list[float] = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = bundle.graph_data.to(device)

    for seed in TRAINING_SEEDS:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        model = GraphSAGERegressor(in_channels=data.x.shape[1], hidden_channels=128, dropout=0.3).to(device)
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = get_loss_function()

        for _ in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            pred = model(data.x, data.edge_index)
            loss = loss_fn(pred[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

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

        metrics = evaluate_on_test_mask(y_true=data.y, y_pred=y_pred, test_mask=data.test_mask)
        seed_metrics.append(metrics)
        seed_inference_runtimes.append(inference_runtime_sec)

    rho_values = np.array([m["spearman_rho"] for m in seed_metrics], dtype=float)
    ndcg_values = np.array([m["ndcg_at_10pct"] for m in seed_metrics], dtype=float)
    precision_values = np.array([m["precision_at_10pct"] for m in seed_metrics], dtype=float)

    return {
        "model_name": "gnn_raw_attr",
        "spearman_rho_mean": float(np.mean(rho_values)),
        "spearman_rho_std": float(np.std(rho_values, ddof=0)),
        "ndcg_mean": float(np.mean(ndcg_values)),
        "ndcg_std": float(np.std(ndcg_values, ddof=0)),
        "precision_mean": float(np.mean(precision_values)),
        "precision_std": float(np.std(precision_values, ddof=0)),
        "runtime_sec": float(np.mean(np.array(seed_inference_runtimes, dtype=float))),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MAPR2026 v3 surrogate runner scaffold")
    p.add_argument("--out-dir", default=PATHS.results_dir)
    p.add_argument("--out-csv", default=str(Path(PATHS.results_dir) / "surrogate_ranking_metrics.csv"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = resolve_project_path(args.out_dir)
    ensure_dir(out_dir)

    out_csv = resolve_project_path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    cols = ["model_name", "spearman_rho_mean", "spearman_rho_std", "ndcg_mean", "ndcg_std", "precision_mean", "precision_std", "runtime_sec"]

    if args.dry_run:
        pd.DataFrame(columns=cols).to_csv(out_csv, index=False)
        print(f"[OK] Wrote dry-run surrogate metrics header: {out_csv} (timestamp={now_iso()})")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training device: {device}")

    bundle = load_surrogate_data_bundle()
    results = train_surrogate_5seeds(bundle=bundle, max_epochs=args.max_epochs)
    result_df = pd.DataFrame([results], columns=cols)
    result_df.to_csv(out_csv, mode="a", index=False, header=not out_csv.exists())

    print("[OK] Surrogate training/evaluation completed with 5 seeds.")
    print(f" - n_nodes={bundle.graph_data.x.shape[0]}, n_features={bundle.graph_data.x.shape[1]}, n_edges={bundle.graph_data.edge_index.shape[1]}")
    print(f" - train/val/test={int(bundle.train_mask.sum())}/{int(bundle.val_mask.sum())}/{int(bundle.test_mask.sum())}")
    print(f" - output={out_csv}")


if __name__ == "__main__":
    main()
