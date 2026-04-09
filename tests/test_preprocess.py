import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src" / "mapr2026_v3"))

from export_csr import build_csr_from_edges


def test_build_csr_from_edges_contract() -> None:
	src = ["b", "a"]
	dst = ["c", "b"]

	indptr, indices, degrees, node_ids = build_csr_from_edges(src, dst)

	assert node_ids.tolist() == sorted(node_ids.tolist())
	assert indptr.shape[0] == len(node_ids) + 1
	assert len(indices) == 2 * len(src)  # undirected expansion
	assert np.array_equal(np.diff(indptr), degrees)
	assert int(degrees.sum()) == len(indices)
