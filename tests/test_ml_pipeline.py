import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src" / "mapr2026_v3"))

from day1_benchmark import _projected_hours, _rho_decision_branch, _runtime_decision


def test_projected_hours_is_positive() -> None:
	out = _projected_hours(per_sim_ms=0.5, n_seeds=5000, n_runs=200)
	assert out > 0


def test_runtime_decision_gates() -> None:
	d1, a1 = _runtime_decision(3.9)
	d2, a2 = _runtime_decision(6.0)
	d3, a3 = _runtime_decision(9.0)

	assert d1["n_seeds"] == 5000 and d1["n_runs"] == 200 and a1 == "proceed_as_planned"
	assert d2["n_seeds"] == 3000 and d2["n_runs"] == 150 and a2 == "reduce_compute_with_limitation"
	assert d3["n_seeds"] == 2000 and d3["n_runs"] == 100 and a3 == "minimum_budget_with_limitation"


def test_rho_decision_branch_gates() -> None:
	assert _rho_decision_branch(0.79) == "viable_gnn"
	assert _rho_decision_branch(0.80) == "two_hop_primary"
	assert _rho_decision_branch(0.90) == "two_hop_primary"
	assert _rho_decision_branch(0.91) == "restructure"
