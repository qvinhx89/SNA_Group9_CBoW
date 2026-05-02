"""MAPR2026 v3 — Ranking metrics harness (Spearman, NDCG@10%, Precision@10%).

Owner: Person 3 (evaluation)

This module is intentionally lightweight and can be imported by other entrypoints
in this folder.

Protocol
--------
- Transductive: compute metrics on held-out labeled nodes only.
- k = ceil(0.10 * n_test)
- Split mask loaded from data/processed/split_masks.parquet (created by Person 1).
  DO NOT create a new split independently — always use the shared artifact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class RankingMetrics:
    spearman_rho: float
    ndcg_at_10pct: float
    precision_at_10pct: float


def _dcg(rels: np.ndarray) -> float:
    # rels assumed ordered by predicted ranking
    denom = np.log2(np.arange(2, rels.size + 2))
    return float(np.sum((2.0 ** rels - 1.0) / denom))


def _argsort_desc_stable(x: np.ndarray) -> np.ndarray:
    """Deterministic descending argsort.

    Uses a stable sort so ties break by original order. Upstream, we enforce
    stable ordering by `node_id` after masking, so this yields deterministic
    top-k membership for NDCG/Precision when there are ties.
    """
    x = np.asarray(x).ravel()
    return np.argsort(-x, kind="mergesort")


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    if k <= 0:
        return float("nan")
    order = _argsort_desc_stable(y_pred)
    topk = order[:k]
    rels = y_true[topk]

    ideal_order = _argsort_desc_stable(y_true)
    ideal_topk = ideal_order[:k]
    ideal_rels = y_true[ideal_topk]

    dcg = _dcg(rels)
    idcg = _dcg(ideal_rels)
    return float(dcg / idcg) if idcg > 0 else float("nan")


def precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    if k <= 0:
        return float("nan")

    pred_top = set(_argsort_desc_stable(y_pred)[:k].tolist())
    true_top = set(_argsort_desc_stable(y_true)[:k].tolist())
    return float(len(pred_top & true_top) / k)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RankingMetrics:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have same shape")

    n = y_true.size
    k = int(math.ceil(0.10 * n))

    res = spearmanr(y_true, y_pred)
    stat = getattr(res, "statistic", res[0])
    stat_arr = np.asarray(stat)
    if stat_arr.size != 1:
        raise ValueError("Spearman statistic is not scalar; ensure y_true/y_pred are 1-D")
    rho = float(stat_arr.item())
    ndcg = ndcg_at_k(y_true=y_true, y_pred=y_pred, k=k)
    prec = precision_at_k(y_true=y_true, y_pred=y_pred, k=k)

    return RankingMetrics(spearman_rho=rho, ndcg_at_10pct=float(ndcg), precision_at_10pct=float(prec))


# ---------------------------------------------------------------------------
# Split-mask helpers (M0-locked protocol)
# ---------------------------------------------------------------------------

def load_split_mask(mask_path: str | Path) -> pd.DataFrame:
    """Load the shared split mask created by Person 1.

    Returns a DataFrame with columns [node_id (str), split ('train'|'test')].

    Usage
    -----
    mask = load_split_mask(PATHS.split_masks)
    test_ids = set(mask.loc[mask["split"] == "test", "node_id"])

    DO NOT create a new split here. Always use the shared artifact so that
    Person 2 (typology) and Person 3 (baselines/surrogates) report metrics
    over the same held-out node set.
    """
    p = Path(mask_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Split mask not found: {p}. "
            "Run ic_labels_primary.py (Person 1) to generate it."
        )
    df = pd.read_parquet(p)
    required = ["node_id", "split"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"split_masks.parquet missing columns: {missing}")
    valid_splits = {"train", "test"}
    unexpected = set(df["split"].unique()) - valid_splits
    if unexpected:
        raise ValueError(f"split_masks.parquet contains unexpected split values: {unexpected}")
    return df


def apply_test_mask(
    df: pd.DataFrame,
    mask: pd.DataFrame,
    node_id_col: str = "node_id",
) -> pd.DataFrame:
    """Filter df to test-split nodes only, using the shared split mask.

    Parameters
    ----------
    df       : DataFrame containing node_id_col + metric columns.
    mask     : Output of load_split_mask().
    node_id_col: Column name for node ids in df.

    Returns
    -------
    Filtered DataFrame (test nodes only), preserving original column order.
    """
    # Guard against accidentally consuming dry-run diffusion proxies.
    proxy_cols = {"one_hop_spread", "two_hop_spread"}
    if proxy_cols.issubset(df.columns):
        if len(df) == 0:
            raise ValueError(
                "apply_test_mask: diffusion proxies input is empty (likely dry-run header-only artifact)."
            )
        if df[["one_hop_spread", "two_hop_spread"]].isna().any().any():
            raise ValueError(
                "apply_test_mask: diffusion proxies contain NaN values (placeholder). "
                "Run diffusion_proxies.py real mode before evaluation/runtime."
            )

    df_local = df.copy()
    df_local[node_id_col] = (
        df_local[node_id_col].astype(str).str.replace(r"\.0$", "", regex=True)
    )

    mask_local = mask.copy()
    mask_local["node_id"] = mask_local["node_id"].astype(str).str.replace(r"\.0$", "", regex=True)

    test_ids = set(mask_local.loc[mask_local["split"] == "test", "node_id"].tolist())
    filtered = df_local[df_local[node_id_col].isin(test_ids)].copy()
    if len(filtered) == 0:
        raise ValueError(
            "apply_test_mask: no rows remain after filtering. "
            "Check that node_id types match between df and split mask."
        )

    # Deterministic row order for downstream metric computations and for
    # tie-breaking in top-k membership.
    filtered = filtered.sort_values(node_id_col, kind="mergesort").reset_index(drop=True)
    return filtered
