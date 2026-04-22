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
from sklearn.preprocessing import MinMaxScaler

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import SAGEConv, GATConv, GCNConv, GINConv, APPNP
except Exception:
    Data = None
    SAGEConv = None
    GATConv = None
    GCNConv = None
    GINConv = None
    APPNP = None

from _shared import PATHS, ensure_dir, now_iso, read_edgelist_pairs, require_columns
from eval_ranking_harness import (
    apply_test_mask,
    compute_metrics,
    load_split_mask as load_shared_split_mask,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_EPOCHS = 200
TRAINING_SEEDS = [42, 123, 456, 789, 1024]


def infer_label_regime_from_targets_path(targets_path: str | Path) -> str:
    name = Path(str(targets_path)).name.lower()
    if "hscc" in name:
        return "hscc"
    if "a2" in name:
        return "a2"
    if "a0" in name:
        return "a0"
    # Backward-compat: legacy default targets were A0 primary.
    if name in {"regression_targets.parquet", "regression_targets.csv"}:
        return "a0"
    return "unknown"


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
    scaler: Any
    split_mask_df: pd.DataFrame


class IdentityScaler:
    def fit(self, x: np.ndarray) -> "IdentityScaler":
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return x


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


class GNNSurrogateRegressor(nn.Module):
    def __init__(
        self,
        arch: str,
        in_channels: int,
        hidden_channels: int = 128,
        dropout: float = 0.3,
        gat_heads: int = 4,
    ) -> None:
        super().__init__()
        if Data is None:
            raise ImportError("torch_geometric is required for GNN surrogates but is not available.")

        self.arch = str(arch).lower()
        self.dropout = nn.Dropout(dropout)

        if self.arch == "sage":
            if SAGEConv is None:
                raise ImportError("torch_geometric is required for GraphSAGE but is not available.")
            self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean")
            self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr="mean")
            self.head = nn.Linear(hidden_channels, 1)
        elif self.arch == "gcn":
            if GCNConv is None:
                raise ImportError("torch_geometric is required for GCN but is not available.")
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.head = nn.Linear(hidden_channels, 1)
        elif self.arch == "gin":
            if GINConv is None:
                raise ImportError("torch_geometric is required for GIN but is not available.")
            mlp1 = nn.Sequential(nn.Linear(in_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
            mlp2 = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
            self.conv1 = GINConv(mlp1)
            self.conv2 = GINConv(mlp2)
            self.head = nn.Linear(hidden_channels, 1)
        elif self.arch == "gat":
            if GATConv is None:
                raise ImportError("torch_geometric is required for GAT but is not available.")
            heads = int(gat_heads)
            self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout)
            self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=True, dropout=dropout)
            self.head = nn.Linear(hidden_channels, 1)
        elif self.arch == "appnp":
            if APPNP is None:
                raise ImportError("torch_geometric is required for APPNP but is not available.")
            # APPNP: MLP feature transform + personalized propagation.
            self.lin1 = nn.Linear(in_channels, hidden_channels)
            self.lin2 = nn.Linear(hidden_channels, hidden_channels)
            self.propagation = APPNP(K=10, alpha=0.1, dropout=dropout)
            self.head = nn.Linear(hidden_channels, 1)
        else:
            raise ValueError(f"Unsupported arch={arch}. Choose from: sage, gcn, gin, gat, appnp")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.arch == "appnp":
            x = self.lin1(x)
            x = torch.relu(x)
            x = self.dropout(x)
            x = self.lin2(x)
            x = torch.relu(x)
            x = self.dropout(x)
            x = self.propagation(x, edge_index)
        else:
            x = self.conv1(x, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)
            x = self.conv2(x, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)
        out = self.head(x)
        return out.squeeze(-1)

    def reset_parameters(self) -> None:
        for mod in [
            getattr(self, "conv1", None),
            getattr(self, "conv2", None),
            getattr(self, "lin1", None),
            getattr(self, "lin2", None),
            getattr(self, "propagation", None),
            getattr(self, "head", None),
        ]:
            if mod is None:
                continue
            reset = getattr(mod, "reset_parameters", None)
            if callable(reset):
                reset()


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

    lang_col = None
    if "language" in df.columns:
        lang_col = "language"
    elif "lang" in df.columns:
        lang_col = "lang"

    lang_dummies = pd.DataFrame(index=df.index)
    if lang_col is not None:
        lang_series = df[lang_col].astype(str).fillna("unknown")
        lang_series = lang_series.replace({"nan": "unknown", "None": "unknown"})
        lang_dummies = pd.get_dummies(lang_series, prefix="lang", dtype=float)

    features = pd.DataFrame(
        {
            "node_id": df["node_id"],
            "views_log": views_log.astype(float),
            "views_per_day": views_per_day.astype(float),
            "life_time": life_time.astype(float),
        }
    )
    if not lang_dummies.empty:
        features = pd.concat([features, lang_dummies], axis=1)
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

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def load_surrogate_data_bundle(
    feature_mode: str = "raw_attr",
    node_attributes_path: str | Path = PATHS.node_attributes,
    centrality_path: str | Path = "data/processed/centrality_table.parquet",
    targets_path: str | Path = PATHS.regression_targets,
    split_mask_path: str | Path = PATHS.split_masks,
    edgelist_path: str | Path = PATHS.graph_edgelist,
    node_scope: str = "all",
) -> SurrogateDataBundle:
    if Data is None:
        raise ImportError("torch_geometric is required for surrogate data bundle construction.")

    node_attributes = pd.read_parquet(resolve_project_path(node_attributes_path))
    node_attributes = _ensure_node_id_str(node_attributes)
    require_columns(node_attributes, ["node_id"], "node_attributes")

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

    node_scope = str(node_scope).lower().strip()
    if node_scope not in {"all", "labeled"}:
        raise ValueError("node_scope must be one of: all, labeled")

    if node_scope == "labeled":
        # Fast local sanity checks: build graph on labeled nodes only.
        labeled_ids = sorted(set(split_mask_df["node_id"].astype(str).tolist()))
        base_df = pd.DataFrame({"node_id": labeled_ids})
    else:
        base_df = node_attributes[["node_id"]].copy()

    targets = pd.read_parquet(resolve_project_path(targets_path))
    targets = _ensure_node_id_str(targets)
    require_columns(targets, ["node_id", "y"], "regression_targets")
    y_df = targets[["node_id", "y"]].copy()
    y_df["y"] = pd.to_numeric(y_df["y"], errors="coerce")

    raw_features_df = _derive_features(node_attributes)
    merged = base_df.merge(y_df, on="node_id", how="left")

    if feature_mode == "constant":
        merged["const_1"] = 1.0
        feature_cols = ["const_1"]
    elif feature_mode == "raw_attr":
        merged = merged.merge(raw_features_df, on="node_id", how="left")
        feature_cols = [c for c in raw_features_df.columns if c != "node_id"]
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
            feature_cols = [c for c in raw_features_df.columns if c != "node_id"] + ["degree", "pagerank", "kshell"]
    else:
        raise ValueError(f"Unsupported feature_mode={feature_mode}")

    node_ids = merged["node_id"].astype(str)

    train_mask_np = node_ids.isin(split["train"]).to_numpy(dtype=bool)
    val_mask_np = node_ids.isin(split["val"]).to_numpy(dtype=bool)
    test_mask_np = node_ids.isin(split["test"]).to_numpy(dtype=bool)

    merged["y"] = pd.to_numeric(merged["y"], errors="coerce").fillna(0.0)
    merged[feature_cols] = merged[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    x_raw = merged[feature_cols].to_numpy(dtype=np.float32)

    if feature_mode == "constant":
        # Do NOT scale constant features to all-zeros; keep x=1 so GIN/sum-style
        # aggregators can still reflect structural degree information.
        scaler: Any = IdentityScaler().fit(x_raw)
        x_scaled = x_raw
    else:
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

    return SurrogateDataBundle(
        node_ids=node_ids,
        graph_data=graph_data,
        train_mask=graph_data.train_mask,
        val_mask=graph_data.val_mask,
        test_mask=graph_data.test_mask,
        scaler=scaler,
        split_mask_df=split_mask_df,
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
    arch: str = "sage",
    early_stop: bool = False,
    patience: int = 20,
    seeds: list[int] | None = None,
) -> tuple[dict[str, float], np.ndarray]:
    seed_metrics: list[dict[str, float]] = []
    seed_inference_runtimes: list[float] = []
    seed_train_runtimes: list[float] = []
    seed_predictions: list[np.ndarray] = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = bundle.graph_data.to(device)

    training_seeds = TRAINING_SEEDS if seeds is None else list(seeds)
    if len(training_seeds) == 0:
        raise ValueError("seeds list is empty")

    for seed in training_seeds:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        model = GNNSurrogateRegressor(arch=arch, in_channels=data.x.shape[1], hidden_channels=128, dropout=0.3).to(device)
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = get_loss_function()

        y_train_target = data.y.clone()
        if randomize_train_target:
            train_idx = torch.where(data.train_mask)[0]
            if train_idx.numel() > 1:
                perm = train_idx[torch.randperm(train_idx.numel(), device=device)]
                y_train_target[train_idx] = data.y[perm]

        best_state = None
        best_val = float("inf")
        no_improve = 0

        t_train_0 = time.time()
        for _ in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            pred = model(data.x, data.edge_index)
            loss = loss_fn(pred[data.train_mask], y_train_target[data.train_mask])
            loss.backward()
            optimizer.step()

            if early_stop and bool(data.val_mask.any()):
                model.eval()
                with torch.no_grad():
                    val_pred = model(data.x, data.edge_index)
                    val_loss = loss_fn(val_pred[data.val_mask], y_train_target[data.val_mask]).item()
                if val_loss + 1e-12 < best_val:
                    best_val = float(val_loss)
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= int(patience):
                        break
        t_train_1 = time.time()

        if early_stop and best_state is not None:
            model.load_state_dict(best_state)

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


def _upsert_rows(
    csv_path: Path,
    rows: list[dict[str, float]],
    cols: list[str],
    key: str | list[str] = "model_name",
) -> None:
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

    key_cols = [key] if isinstance(key, str) else list(key)
    merged = merged.drop_duplicates(subset=key_cols, keep="last")
    merged = merged.sort_values(key_cols).reset_index(drop=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(csv_path, index=False)


def _upsert_runtime_rows(rows: list[dict[str, float]], label_regime: str) -> None:
    runtime_path = resolve_project_path(PATHS.runtime_csv)
    cols = ["label_regime", "model_name", "inference_sec_full_graph", "train_sec"]
    runtime_rows = []
    for row in rows:
        runtime_rows.append(
            {
                "label_regime": str(label_regime),
                "model_name": row["model_name"],
                "inference_sec_full_graph": row.get("runtime_sec", np.nan),
                "train_sec": row.get("train_sec", np.nan),
            }
        )
    _upsert_rows(runtime_path, runtime_rows, cols=cols, key=["label_regime", "model_name"])


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
    p.add_argument(
        "--targets-path",
        default=PATHS.regression_targets,
        help="Regression targets parquet (default: primary A0 targets). Use for A2 reruns.",
    )
    p.add_argument(
        "--label-regime",
        default="",
        help=(
            "Label regime tag to record in output (a0|hscc|a2). "
            "If omitted, inferred from --targets-path filename."
        ),
    )
    p.add_argument(
        "--split-mask-path",
        default=PATHS.split_masks,
        help="Shared split mask parquet (M0-locked).",
    )
    p.add_argument(
        "--node-scope",
        default="all",
        choices=["all", "labeled"],
        help="Graph scope: 'all' uses full graph; 'labeled' builds induced subgraph on labeled nodes (fast local sanity check).",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    p.add_argument("--run-random-sanity", action="store_true")
    p.add_argument("--only-random", action="store_true")
    p.add_argument("--skip-gnn-full", action="store_true")
    p.add_argument(
        "--include-c2-arch",
        action="store_true",
        help="Also run GCN/GIN/GAT/APPNP on raw_attr features (C2 architecture comparison).",
    )
    p.add_argument(
        "--include-edge-only",
        action="store_true",
        help="Also run edge-only (x=1 constant) variants to match 'graph-only' requirement.",
    )
    p.add_argument(
        "--only-edge-only",
        action="store_true",
        help="Run only edge-only variants (x=1) for quick 'graph-only strict' checks.",
    )
    p.add_argument("--early-stop", action="store_true", help="Enable early stopping on val loss (10% of train).")
    p.add_argument("--patience", type=int, default=20)
    p.add_argument(
        "--seeds",
        default="",
        help="Comma-separated training seeds (e.g. '42,123'). Default uses [42,123,456,789,1024].",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = resolve_project_path(args.out_dir)
    ensure_dir(out_dir)

    out_csv = resolve_project_path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    cols = [
        "label_regime",
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

    label_regime = str(args.label_regime).strip().lower() if str(args.label_regime).strip() else infer_label_regime_from_targets_path(args.targets_path)

    seed_list: list[int] | None = None
    if str(args.seeds).strip():
        parts = [p.strip() for p in str(args.seeds).split(",") if p.strip()]
        seed_list = [int(p) for p in parts]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training device: {device}")

    if args.only_edge_only:
        model_specs: list[tuple[str, str, bool, str]] = [
            ("sage_edge_only", "constant", False, "sage"),
            ("gcn_edge_only", "constant", False, "gcn"),
            ("gin_edge_only", "constant", False, "gin"),
            ("gat_edge_only", "constant", False, "gat"),
            ("appnp_edge_only", "constant", False, "appnp"),
        ]
    else:
        model_specs = [
            ("gnn_raw_attr", "raw_attr", False, "sage"),
            ("gnn_graph_only", "graph_only", False, "sage"),
            ("gnn_centrality", "centrality", False, "sage"),
            ("gnn_full", "full", False, "sage"),
        ]
    if args.skip_gnn_full:
        model_specs = [spec for spec in model_specs if spec[0] != "gnn_full"]
    if args.run_random_sanity:
        model_specs.append(("gnn_random", "random", False, "sage"))
    if args.only_random:
        model_specs = [("gnn_random", "random", False, "sage")]

    if args.include_c2_arch:
        model_specs.extend(
            [
                ("gcn_raw_attr", "raw_attr", False, "gcn"),
                ("gin_raw_attr", "raw_attr", False, "gin"),
                ("gat_raw_attr", "raw_attr", False, "gat"),
                ("appnp_raw_attr", "raw_attr", False, "appnp"),
            ]
        )

    if args.include_edge_only:
        model_specs.extend(
            [
                ("sage_edge_only", "constant", False, "sage"),
                ("gcn_edge_only", "constant", False, "gcn"),
                ("gin_edge_only", "constant", False, "gin"),
                ("gat_edge_only", "constant", False, "gat"),
                ("appnp_edge_only", "constant", False, "appnp"),
            ]
        )

    results_list: list[dict[str, float]] = []
    predictions_by_model: dict[str, np.ndarray] = {}
    base_bundle: SurrogateDataBundle | None = None

    for model_name, feature_mode, randomize_train_target, arch in model_specs:
        try:
            bundle = load_surrogate_data_bundle(
                feature_mode=feature_mode,
                targets_path=args.targets_path,
                split_mask_path=args.split_mask_path,
                node_scope=args.node_scope,
            )
            if base_bundle is None:
                base_bundle = bundle
            row, pred = train_surrogate_5seeds(
                bundle=bundle,
                max_epochs=args.max_epochs,
                model_name=model_name,
                randomize_train_target=randomize_train_target,
                arch=arch,
                early_stop=bool(args.early_stop),
                patience=int(args.patience),
                seeds=seed_list,
            )
            row["label_regime"] = label_regime
            results_list.append(row)
            predictions_by_model[model_name] = pred
        except Exception as exc:
            print(f"[WARN] Skipping {model_name}: {exc}")

    if not results_list:
        raise RuntimeError("No surrogate models produced results.")

    _upsert_rows(out_csv, rows=results_list, cols=cols, key=["label_regime", "model_name"])
    _upsert_runtime_rows(results_list, label_regime=label_regime)

    if base_bundle is not None and predictions_by_model:
        per_group_path = resolve_project_path(args.per_group_error_csv)
        _write_per_group_prediction_error(
            node_ids=base_bundle.node_ids,
            y_true=base_bundle.graph_data.y,
            split_mask_df=base_bundle.split_mask_df,
            predictions=predictions_by_model,
            out_path=per_group_path,
        )

    effective_seeds = TRAINING_SEEDS if seed_list is None else list(seed_list)
    print(f"[OK] Surrogate training/evaluation completed with {len(effective_seeds)} seed(s).")
    print(f" - models_written={len(results_list)}")
    print(f" - n_nodes={base_bundle.graph_data.x.shape[0]}, n_edges={base_bundle.graph_data.edge_index.shape[1]}")
    print(f" - output={out_csv}")


if __name__ == "__main__":
    main()
