import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src" / "mapr2026_v3"))

from eval_ranking_harness import apply_test_mask, compute_metrics, ndcg_at_k, precision_at_k


def test_apply_test_mask_sorts_by_node_id_stably() -> None:
    df = pd.DataFrame(
        {
            "node_id": ["3", "1", "2", "10"],
            "y_true": [0.0, 1.0, 2.0, 3.0],
            "y_pred": [0.0, 1.0, 2.0, 3.0],
        }
    )
    mask = pd.DataFrame(
        {
            "node_id": ["10", "2", "1"],
            "split": ["test", "test", "test"],
        }
    )

    out = apply_test_mask(df, mask, node_id_col="node_id")
    assert out["node_id"].tolist() == ["1", "10", "2"]


def test_tie_breaking_is_deterministic_for_topk_metrics() -> None:
    # n=20 => k=ceil(0.10*n)=2
    n = 20
    y_true = np.zeros(n, dtype=float)
    y_pred = np.zeros(n, dtype=float)

    # True top-2 are indices {0, 2}
    y_true[0] = 3.0
    y_true[2] = 2.0

    # Create a 3-way tie for top predicted values among indices {0, 1, 2}.
    # With stable descending argsort, the top-2 predicted indices are [0, 1]
    # (tie breaks by original index order), which yields precision@2 = 0.5 and
    # NDCG@2 < 1.0.
    y_pred[0] = 1.0
    y_pred[1] = 1.0
    y_pred[2] = 1.0

    k = 2
    ndcg = ndcg_at_k(y_true=y_true, y_pred=y_pred, k=k)
    prec = precision_at_k(y_true=y_true, y_pred=y_pred, k=k)

    # Expected: predicted top2 = {0,1}, true top2 = {0,2}
    assert prec == 0.5

    # Expected DCG = rels [3,0] => 7.0; IDCG = [3,2] => 7 + 3/log2(3)
    expected_ndcg = 7.0 / (7.0 + (3.0 / np.log2(3.0)))
    assert abs(ndcg - expected_ndcg) < 1e-12


def test_compute_metrics_is_consistent_under_ties() -> None:
    n = 20
    y_true = np.zeros(n, dtype=float)
    y_pred = np.zeros(n, dtype=float)

    y_true[0] = 3.0
    y_true[2] = 2.0
    y_pred[0] = 1.0
    y_pred[1] = 1.0
    y_pred[2] = 1.0

    m = compute_metrics(y_true=y_true, y_pred=y_pred)
    assert np.isfinite(m.spearman_rho)
    assert m.ndcg_at_10pct == ndcg_at_k(y_true, y_pred, k=2)
    assert m.precision_at_10pct == precision_at_k(y_true, y_pred, k=2)
