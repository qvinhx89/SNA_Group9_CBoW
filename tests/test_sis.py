import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src" / "mapr2026_v3"))

from ic_pilot_diagnostics import _per_quintile_cv_table


def test_per_quintile_cv_table_basic_shape() -> None:
	sampled_degrees = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
	per_node_cv = np.array([0.1, 0.2, 0.15, 0.25, 0.3, 0.35, 0.4, 0.12, 0.22, 0.32], dtype=float)

	rows = _per_quintile_cv_table(
		sampled_degrees=sampled_degrees,
		per_node_cv=per_node_cv,
		cv_noise_threshold=0.5,
	)

	assert len(rows) > 0
	for row in rows:
		assert "quintile" in row
		assert "n_nodes" in row
		assert "cv_mean" in row
		assert "cv_median" in row
		assert "cv_noise_count" in row
