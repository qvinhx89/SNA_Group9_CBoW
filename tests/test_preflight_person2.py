from __future__ import annotations

import json
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
MAPR_SRC = REPO_ROOT / "src" / "mapr2026_v3"
if str(MAPR_SRC) not in sys.path:
    sys.path.insert(0, str(MAPR_SRC))

preflight_person2 = importlib.import_module("preflight_person2")
_check_metric_correlation_matrix = preflight_person2._check_metric_correlation_matrix
_check_stage5_null_package = preflight_person2._check_stage5_null_package
_check_lifetime_if_problem_handled = preflight_person2._check_lifetime_if_problem_handled


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _expected_metrics() -> list[str]:
    return [
        "ic_score_mean",
        "views",
        "degree",
        "pagerank",
        "kshell",
        "betweenness_approx",
        "one_hop_spread",
        "two_hop_spread",
    ]


def _matrix_payload(n_rows: int, n_rows_expected: int, coverage_ok: bool = True) -> dict:
    n = len(_expected_metrics())
    rho = np.eye(n, dtype=float).tolist()
    p = np.eye(n, dtype=float).tolist()
    return {
        "timestamp": "2026-04-08T00:00:00",
        "n_rows": n_rows,
        "n_rows_expected": n_rows_expected,
        "coverage_ok": coverage_ok,
        "metrics": _expected_metrics(),
        "rho_matrix": rho,
        "p_matrix_corrected": p,
        "column_mapping": {
            "ic_score_mean": "ic_score_mean",
            "views": "views",
            "degree": "degree",
            "pagerank": "pagerank",
            "kshell": "kshell",
            "betweenness_approx": "betweenness",
            "one_hop_spread": "one_hop_spread",
            "two_hop_spread": "two_hop_spread",
        },
    }


def test_preflight_stage5_null_package_passes_with_minimal_valid_schema(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"

    _write_json(
        results_dir / "null_model_typology_summary.json",
        {
            "timestamp": "2026-04-08T00:00:00",
            "n_nodes": 500,
            "n_realizations": 3,
            "n_runs_per_node": 100,
            "rho_mean": 0.4,
            "rho_std": 0.01,
            "hidden_betweenness_real_subgraph_mean": 0.001,
            "hidden_betweenness_null_mean": 0.002,
            "hidden_betweenness_null_std": 0.0001,
            "interpretation": "ok",
        },
    )

    shared_perm = {
        "timestamp": "2026-04-08T00:00:00",
        "n_nodes_labeled": 5000,
        "n_permutations": 200,
        "top_pct": 0.1,
        "real": {"agreement_rate": 0.88},
        "null_distribution": {"agreement_rate_mean": 0.82},
        "empirical_p_values": {"agreement_rate_ge_real": 0.01},
        "interpretation": "ok",
    }
    _write_json(results_dir / "views_permutation_null_summary.json", shared_perm)
    _write_json(results_dir / "ic_permutation_null_summary.json", shared_perm)

    result = _check_stage5_null_package(results_dir)
    assert result.status == "PASS"


def test_preflight_metric_matrix_returns_structured_fail_for_bad_matrix_payload(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    ic_scores_path = tmp_path / "ic_scores_primary.parquet"

    pd.DataFrame({"node_id": [f"n{i}" for i in range(8)]}).to_parquet(ic_scores_path, index=False)

    payload = _matrix_payload(n_rows=8, n_rows_expected=8, coverage_ok=True)
    payload["rho_matrix"] = "not-a-matrix"
    _write_json(results_dir / "metric_correlation_matrix.json", payload)

    result = _check_metric_correlation_matrix(results_dir=results_dir, ic_scores_path=ic_scores_path)
    assert result.status == "FAIL"
    assert "Invalid matrix payload" in result.detail


def test_preflight_metric_matrix_fails_hard_on_n_rows_coverage_mismatch(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    ic_scores_path = tmp_path / "ic_scores_primary.parquet"

    pd.DataFrame({"node_id": [f"n{i}" for i in range(8)]}).to_parquet(ic_scores_path, index=False)

    payload = _matrix_payload(n_rows=7, n_rows_expected=7, coverage_ok=True)
    _write_json(results_dir / "metric_correlation_matrix.json", payload)

    result = _check_metric_correlation_matrix(results_dir=results_dir, ic_scores_path=ic_scores_path)
    assert result.status == "FAIL"
    assert "coverage mismatch vs IC labels" in result.detail


def test_preflight_metric_matrix_fails_when_p_matrix_diagonal_not_one(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    ic_scores_path = tmp_path / "ic_scores_primary.parquet"

    pd.DataFrame({"node_id": [f"n{i}" for i in range(8)]}).to_parquet(ic_scores_path, index=False)

    payload = _matrix_payload(n_rows=8, n_rows_expected=8, coverage_ok=True)
    payload["p_matrix_corrected"][0][0] = 0.0
    _write_json(results_dir / "metric_correlation_matrix.json", payload)

    result = _check_metric_correlation_matrix(results_dir=results_dir, ic_scores_path=ic_scores_path)
    assert result.status == "FAIL"
    assert "diagonal must be 1.0" in result.detail


def test_lifetime_if_problem_status_pass_when_fallback_is_handled(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    assumptions_doc = tmp_path / "docs" / "assumptions_limitations.md"

    _write_json(
        results_dir / "lifetime_validation.json",
        {
            "partial_spearman_rho": -0.02,
            "n_quintiles_significant": 0,
            "success": False,
        },
    )
    _write_json(
        results_dir / "language_validation.json",
        {
            "trigger_condition": {
                "partial_spearman_rho_lt_0_05": True,
                "n_quintiles_significant_lt_3": True,
            },
            "note": "supplementary corroboration",
        },
    )
    _write_text(
        assumptions_doc,
        "Task 6 IF PROBLEM is documented. Language-based corroboration is supplementary evidence.",
    )

    result = _check_lifetime_if_problem_handled(results_dir=results_dir, assumptions_doc=assumptions_doc)
    assert result.status == "PASS"
    assert "IF PROBLEM handled" in result.detail
    assert result.blocking is False


def test_lifetime_if_problem_status_warn_when_triggered_but_language_missing(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    assumptions_doc = tmp_path / "docs" / "assumptions_limitations.md"

    _write_json(
        results_dir / "lifetime_validation.json",
        {
            "partial_spearman_rho": 0.01,
            "n_quintiles_significant": 2,
            "success": False,
        },
    )
    _write_text(
        assumptions_doc,
        "Task 6 IF PROBLEM note is present.",
    )

    result = _check_lifetime_if_problem_handled(results_dir=results_dir, assumptions_doc=assumptions_doc)
    assert result.status == "WARN"
    assert "language fallback artifact is missing" in result.detail
    assert result.blocking is False
