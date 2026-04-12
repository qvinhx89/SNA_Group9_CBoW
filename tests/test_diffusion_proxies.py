from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MAPR_SRC = REPO_ROOT / "src" / "mapr2026_v3"
if str(MAPR_SRC) not in sys.path:
    sys.path.insert(0, str(MAPR_SRC))

from diffusion_proxies import _assert_csr_bidirectional, _compute_one_hop, _compute_two_hop


def _toy_path3_csr() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Undirected path 0-1-2 represented with both directions in CSR."""
    indptr = np.array([0, 1, 3, 4], dtype=np.int64)
    indices = np.array([1, 0, 2, 1], dtype=np.int64)
    degrees = np.diff(indptr).astype(np.int64)
    return indptr, indices, degrees


def test_one_hop_two_hop_on_toy_path_graph() -> None:
    indptr, indices, degrees = _toy_path3_csr()
    inv_deg = 1.0 / np.maximum(degrees.astype(np.float64), 1.0)

    one_hop = _compute_one_hop(indptr, indices, inv_deg)
    two_hop = _compute_two_hop(indptr, indices, inv_deg, one_hop)

    np.testing.assert_allclose(one_hop, np.array([0.5, 2.0, 0.5]), atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(two_hop, np.array([1.0, 2.0, 1.0]), atol=1e-12, rtol=0.0)


def test_csr_symmetry_check_passes_for_bidirectional_csr() -> None:
    indptr, indices, _ = _toy_path3_csr()
    _assert_csr_bidirectional(indptr, indices)


def test_two_hop_excludes_source_node_from_v_spread() -> None:
    """Verify that the two-hop formula correctly subtracts inv_deg[u] (excludes u from v's spread).

    Complete graph K3 (0-1, 1-2, 0-2): every node has degree 2, inv_deg = 0.5.
    one_hop(u) = 2 * 0.5 = 1.0 for all nodes.
    two_hop(u): each neighbor v contributes inv_deg[v] * (1 + one_hop[v] - inv_deg[u])
              = 0.5 * (1 + 1.0 - 0.5) = 0.5 * 1.5 = 0.75.
    Two neighbors -> two_hop(u) = 2 * 0.75 = 1.5.
    """
    #   K3: 0-1, 1-2, 0-2 (both directions stored)
    indptr = np.array([0, 2, 4, 6], dtype=np.int64)
    indices = np.array([1, 2, 0, 2, 0, 1], dtype=np.int64)
    degrees = np.diff(indptr).astype(np.int64)
    inv_deg = 1.0 / np.maximum(degrees.astype(np.float64), 1.0)

    one_hop = _compute_one_hop(indptr, indices, inv_deg)
    two_hop = _compute_two_hop(indptr, indices, inv_deg, one_hop)

    np.testing.assert_allclose(one_hop, np.array([1.0, 1.0, 1.0]), atol=1e-12)
    np.testing.assert_allclose(two_hop, np.array([1.5, 1.5, 1.5]), atol=1e-12)


def test_csr_symmetry_check_fails_when_reverse_edge_missing() -> None:
    # Asymmetric toy CSR: edge 0->1 exists, reverse 1->0 is missing.
    indptr = np.array([0, 1, 1], dtype=np.int64)
    indices = np.array([1], dtype=np.int64)

    with pytest.raises(ValueError, match="symmetry check failed"):
        _assert_csr_bidirectional(indptr, indices)
