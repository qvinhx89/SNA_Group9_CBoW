from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
MAPR_SRC = REPO_ROOT / "src" / "mapr2026_v3"
if str(MAPR_SRC) not in sys.path:
    sys.path.insert(0, str(MAPR_SRC))

from typology_ic_views import (
    _assign_typology_labels,
    _build_quadrant_report,
    _compute_lifetime_validation,
    _compute_structural_profiling,
)


def test_label_logic_four_quadrants_on_mini_dataframe() -> None:
    # 4 rows -> each quadrant appears exactly once.
    df = pd.DataFrame(
        {
            "node_id": ["n_true", "n_hidden", "n_overrated", "n_non"],
            "ic_high": [True, True, False, False],
            "views_high": [True, False, True, False],
        }
    )

    labels = _assign_typology_labels(df)
    assert labels.tolist() == ["True", "Hidden", "Overrated", "Non"]

    counts = labels.value_counts().to_dict()
    assert counts == {"True": 1, "Hidden": 1, "Overrated": 1, "Non": 1}


def test_quadrant_report_min_quadrant_gate_true_and_false() -> None:
    df = pd.DataFrame(
        {
            "typology_label": ["True", "Hidden", "Overrated", "Non"],
        }
    )

    report_ok = _build_quadrant_report(
        df,
        ic_thresh=0.9,
        views_thresh=0.9,
        min_quadrant_size=1,
    )
    assert report_ok["min_quadrant_ok"] is True

    report_fail = _build_quadrant_report(
        df,
        ic_thresh=0.9,
        views_thresh=0.9,
        min_quadrant_size=2,
    )
    assert report_fail["min_quadrant_ok"] is False


def test_structural_profiling_outputs_required_schema_and_rows() -> None:
    # Synthetic frame with Hidden vs Overrated values separated across all 6 features.
    hidden_rows = [
        {"typology_label": "Hidden", "degree": 20, "pagerank": 0.30, "kshell": 5, "betweenness": 0.050, "cross_community_edge_fraction": 0.60, "life_time": 900},
        {"typology_label": "Hidden", "degree": 22, "pagerank": 0.31, "kshell": 5, "betweenness": 0.055, "cross_community_edge_fraction": 0.62, "life_time": 920},
        {"typology_label": "Hidden", "degree": 24, "pagerank": 0.32, "kshell": 6, "betweenness": 0.060, "cross_community_edge_fraction": 0.64, "life_time": 940},
        {"typology_label": "Hidden", "degree": 26, "pagerank": 0.33, "kshell": 6, "betweenness": 0.065, "cross_community_edge_fraction": 0.66, "life_time": 960},
    ]
    overrated_rows = [
        {"typology_label": "Overrated", "degree": 10, "pagerank": 0.10, "kshell": 2, "betweenness": 0.005, "cross_community_edge_fraction": 0.20, "life_time": 300},
        {"typology_label": "Overrated", "degree": 12, "pagerank": 0.11, "kshell": 2, "betweenness": 0.006, "cross_community_edge_fraction": 0.22, "life_time": 320},
        {"typology_label": "Overrated", "degree": 14, "pagerank": 0.12, "kshell": 3, "betweenness": 0.007, "cross_community_edge_fraction": 0.24, "life_time": 340},
        {"typology_label": "Overrated", "degree": 16, "pagerank": 0.13, "kshell": 3, "betweenness": 0.008, "cross_community_edge_fraction": 0.26, "life_time": 360},
    ]
    df = pd.DataFrame(hidden_rows + overrated_rows)

    out = _compute_structural_profiling(df, delta_threshold=0.20)

    assert len(out) == 6
    assert out["feature"].tolist() == [
        "degree",
        "pagerank",
        "kshell",
        "betweenness",
        "cross_community_edge_fraction",
        "life_time",
    ]
    assert out.columns.tolist() == [
        "feature",
        "group_hidden_mean",
        "group_overrated_mean",
        "mwu_stat",
        "p_raw",
        "p_corrected",
        "cliffs_delta",
        "significant",
    ]
    assert (out["p_raw"] >= 0).all() and (out["p_raw"] <= 1).all()
    assert (out["p_corrected"] >= 0).all() and (out["p_corrected"] <= 1).all()


def test_lifetime_validation_outputs_required_schema_and_success_gate() -> None:
    rows = []
    # Build 5 degree quintiles with strong Hidden vs Non-Hidden life_time separation.
    for q in range(5):
        degree = float(10 + q)
        ic_base = float(100 + q * 10)

        for i in range(12):
            rows.append(
                {
                    "typology_label": "Hidden",
                    "ic_score_mean": ic_base + i,
                    "degree": degree,
                    "life_time": float(900 + q * 20 + i),
                }
            )
        for i in range(12):
            rows.append(
                {
                    "typology_label": "Non",
                    "ic_score_mean": ic_base - i,
                    "degree": degree,
                    "life_time": float(100 + q * 20 + i),
                }
            )

    df = pd.DataFrame(rows)
    out = _compute_lifetime_validation(df)

    assert set(out.keys()) == {
        "partial_spearman_rho",
        "partial_spearman_p",
        "n_quintiles_tested",
        "n_quintiles_significant",
        "success",
        "quintile_results",
    }
    assert isinstance(out["partial_spearman_rho"], float)
    assert isinstance(out["partial_spearman_p"], float)
    assert out["n_quintiles_tested"] == 5
    assert len(out["quintile_results"]) == 5
    assert out["n_quintiles_significant"] >= 3
    assert out["success"] is True

    for row in out["quintile_results"]:
        assert set(row.keys()) == {
            "quintile",
            "n_hidden",
            "n_non_hidden",
            "p_raw",
            "p_corrected",
            "cliffs_delta",
            "significant",
        }


def test_lifetime_validation_applies_min_group_size_guard() -> None:
    rows = []
    for q in range(5):
        degree = float(20 + q)
        for i in range(3):
            rows.append(
                {
                    "typology_label": "Hidden",
                    "ic_score_mean": float(50 + i + q),
                    "degree": degree,
                    "life_time": float(700 + i),
                }
            )
        for i in range(20):
            rows.append(
                {
                    "typology_label": "Non",
                    "ic_score_mean": float(40 + i + q),
                    "degree": degree,
                    "life_time": float(300 + i),
                }
            )

    out = _compute_lifetime_validation(pd.DataFrame(rows), min_group_size=10)
    assert out["n_quintiles_tested"] == 0
    assert out["n_quintiles_significant"] == 0
    assert out["success"] is False
