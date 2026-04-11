from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MAPR_SRC = REPO_ROOT / "src" / "mapr2026_v3"
if str(MAPR_SRC) not in sys.path:
    sys.path.insert(0, str(MAPR_SRC))

from null_model_typology import _build_null_interpretation


def test_null_interpretation_detects_structural_signal_when_gap_is_large() -> None:
    interpretation, gap, gap_sigma = _build_null_interpretation(
        hidden_bet_real=1.5e-4,
        hidden_bet_null_mean=5.0e-5,
        hidden_bet_null_std=2.0e-5,
        rho_mean=0.55,
    )

    assert gap > 0.0
    assert gap_sigma >= 1.0
    assert "structural signal" in interpretation
    assert "rho_mean=0.550" in interpretation


def test_null_interpretation_flags_degree_artifact_when_gap_is_small() -> None:
    interpretation, gap, gap_sigma = _build_null_interpretation(
        hidden_bet_real=8.0e-5,
        hidden_bet_null_mean=7.9e-5,
        hidden_bet_null_std=2.0e-5,
        rho_mean=0.44,
    )

    assert abs(gap) < 2.0e-5
    assert gap_sigma < 1.0
    assert "potential degree-distribution artifact" in interpretation
    assert "rho_mean=0.440" in interpretation


def test_null_interpretation_reports_inconclusive_when_gap_small_and_rho_weak() -> None:
    interpretation, gap, gap_sigma = _build_null_interpretation(
        hidden_bet_real=8.0e-5,
        hidden_bet_null_mean=7.9e-5,
        hidden_bet_null_std=2.0e-5,
        rho_mean=0.12,
    )

    assert abs(gap) < 2.0e-5
    assert gap_sigma < 1.0
    assert "inconclusive" in interpretation
    assert "rho_mean=0.120" in interpretation
