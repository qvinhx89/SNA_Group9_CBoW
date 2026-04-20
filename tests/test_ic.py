import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src" / "mapr2026_v3"))

from ic_labels_primary import _create_split_mask, _rq2_narrative_tier


def test_create_split_mask_respects_basic_contract() -> None:
	n = 100
	df_ic = pd.DataFrame(
		{
			"node_id": [f"n{i}" for i in range(n)],
			"ic_score_mean": np.linspace(1.0, 2.0, n),
		}
	)
	degrees = np.arange(1, n + 1)

	out = _create_split_mask(df_ic=df_ic, degrees=degrees, test_frac=0.20, seed=42)

	assert set(out.columns) == {"node_id", "split"}
	assert set(out["split"].unique()) == {"train", "test"}
	assert len(out) == n
	assert out["node_id"].nunique() == n

	n_test = int((out["split"] == "test").sum())
	assert n_test == 20


def test_rq2_narrative_tier_thresholds() -> None:
	assert _rq2_narrative_tier(0.69) == "strong_divergence"
	assert _rq2_narrative_tier(0.70) == "moderate"
	assert _rq2_narrative_tier(0.85) == "moderate"
	assert _rq2_narrative_tier(0.851) == "high_agreement"
