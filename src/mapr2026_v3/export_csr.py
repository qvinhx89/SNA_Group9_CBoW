"""MAPR2026 v3 — Export CSR graph artifact.

Owner: Person 1 (IC core)

Input
-----
- data/processed/graph_active.edgelist

Output (contract)
---------------
- data/processed/graph_csr.npz containing keys:
  - indptr (int64, shape n+1)
  - indices (int64, shape nnz)
  - degrees (int64, shape n)
  - node_ids (str, shape n)

Notes
-----
- Mapping must be deterministic across runs.
- Default mode is safe: it will NOT run the full export unless you pass --run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _shared import PATHS, read_edgelist_pairs, save_csr_npz


def build_csr_from_edges(src: list[str], dst: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """TODO: Replace with a memory-efficient implementation.

    Current scaffold builds an undirected adjacency list then CSR.
    This is intentionally minimal and should be optimized by Person 1.
    """
    node_ids = np.array(sorted(set(src) | set(dst)), dtype=str)
    node_to_idx = {n: i for i, n in enumerate(node_ids)}

    # Build undirected edge list
    rows = np.fromiter((node_to_idx[u] for u in src), dtype=np.int64)
    cols = np.fromiter((node_to_idx[v] for v in dst), dtype=np.int64)
    rows_ud = np.concatenate([rows, cols])
    cols_ud = np.concatenate([cols, rows])

    # Sort by row then col for deterministic CSR
    order = np.lexsort((cols_ud, rows_ud))
    rows_ud = rows_ud[order]
    cols_ud = cols_ud[order]

    # Build indptr
    n = node_ids.shape[0]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(indptr, rows_ud + 1, 1)
    indptr = np.cumsum(indptr, dtype=np.int64)
    indices = cols_ud.astype(np.int64, copy=False)
    degrees = np.diff(indptr).astype(np.int64, copy=False)

    return indptr, indices, degrees, node_ids


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export CSR artifact for MAPR2026 v3")
    p.add_argument("--edgelist", default=PATHS.graph_edgelist)
    p.add_argument("--out", default=PATHS.csr_npz)
    p.add_argument("--max-edges", type=int, default=None, help="For small-mode only")
    p.add_argument(
        "--run",
        action="store_true",
        help="Actually perform export. Without this flag, script validates inputs and exits.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    edgelist_path = Path(args.edgelist)
    out_path = Path(args.out)

    if not edgelist_path.exists():
        raise FileNotFoundError(f"Missing input: {edgelist_path}")

    if not args.run:
        print(
            "[MAPR2026 v3] export_csr scaffold: input exists. "
            "Re-run with --run to generate CSR (optionally --max-edges for small mode)."
        )
        return

    src, dst = read_edgelist_pairs(edgelist_path, max_edges=args.max_edges)
    if len(src) == 0:
        raise ValueError("Edgelist appears empty after filtering")

    indptr, indices, degrees, node_ids = build_csr_from_edges(src, dst)
    save_csr_npz(out_path, indptr=indptr, indices=indices, degrees=degrees, node_ids=node_ids)

    print(f"[OK] Wrote CSR: {out_path} (n_nodes={len(node_ids):,}, nnz={len(indices):,})")


if __name__ == "__main__":
    main()
