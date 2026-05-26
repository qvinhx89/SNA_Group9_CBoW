"""MAPR2026 v3.1 — IC labels (I-A: attribute-informed, row-normalized views).

This produces an *optional, gated* IC label set:
- I-A pilot gate is MUST (see `ic_pilot_ia.py`).
- This full labeling is conditional-MUST only if the pilot passes.

I-A rule
--------
For each active node u, for each neighbor v:

    p(u,v) = w(v) / sum_{x in N(u)} w(x)

where w(v) = log1p(max(views(v), 0)).

Inputs
------
- data/processed/graph_csr.npz
- data/processed/node_attributes.parquet
- data/processed/ic_scores_primary.parquet  (to reuse the exact labeled node_ids)

Outputs
-------
- outputs/mapr2026_v3_results/ic_scores_ia.parquet
  columns: node_id, ic_score_mean, ic_score_std, n_runs, p_model

- data/processed/regression_targets_ia.parquet
  columns: node_id, y  (y=log1p(ic_score_mean))

Notes
-----
- Uses a stamp-based visited array per worker to avoid O(n) clears per run.
- This is *not* a sensitivity analysis of A0. It is an attribute-informed
  operationalization and must be labeled as such in the paper.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from _shared import PATHS, ensure_parent, load_csr_npz, now_iso, require_columns


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MAPR2026 v3.1 IC labeling (I-A row-normalized views)")
    p.add_argument("--csr", default=PATHS.csr_npz)
    p.add_argument("--node-attributes", default=PATHS.node_attributes)
    p.add_argument(
        "--primary-ic",
        default=PATHS.ic_scores,
        help="Primary IC scores parquet (used only to reuse exact labeled node_ids)",
    )
    p.add_argument(
        "--out-ic",
        default=str(Path(PATHS.results_dir) / "ic_scores_ia.parquet"),
        help="Output parquet for I-A IC scores",
    )
    p.add_argument(
        "--out-reg",
        default="data/processed/regression_targets_ia.parquet",
        help="Output regression targets parquet for I-A (y=log1p(ic_score_mean))",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--n-runs",
        type=int,
        default=0,
        help="MC runs per node. If 0, reuse n_runs from primary IC artifact.",
    )
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument(
        "--chunk-size",
        type=int,
        default=64,
        help="How many labeled nodes each worker processes per task (reduces per-node allocation overhead).",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _simulate_ic_once_ia_stamp(
    source: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    w: np.ndarray,
    neigh_w_sum: np.ndarray,
    rng: np.random.Generator,
    marks: np.ndarray,
    frontier: np.ndarray,
    stamp: int,
) -> int:
    head = 0
    tail = 0

    marks[int(source)] = np.uint32(stamp)
    frontier[tail] = int(source)
    tail += 1

    activated_count = 1

    while head < tail:
        u = int(frontier[head])
        head += 1

        denom = float(neigh_w_sum[u])
        if denom <= 0.0:
            continue

        start = int(indptr[u])
        end = int(indptr[u + 1])
        for ptr in range(start, end):
            v = int(indices[ptr])
            if marks[v] == np.uint32(stamp):
                continue

            p = float(w[v]) / denom
            if p <= 0.0:
                continue

            if rng.random() < p:
                marks[v] = np.uint32(stamp)
                frontier[tail] = v
                tail += 1
                activated_count += 1

    return int(activated_count)


def _simulate_node_summary_ia(
    source: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    w: np.ndarray,
    neigh_w_sum: np.ndarray,
    n_runs: int,
    worker_seed: int,
    marks: np.ndarray,
    frontier: np.ndarray,
    stamp_start: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(int(worker_seed))

    runs = np.empty(int(n_runs), dtype=np.int32)
    stamp = int(stamp_start)
    for i in range(int(n_runs)):
        stamp += 1
        runs[i] = _simulate_ic_once_ia_stamp(
            source=int(source),
            indptr=indptr,
            indices=indices,
            w=w,
            neigh_w_sum=neigh_w_sum,
            rng=rng,
            marks=marks,
            frontier=frontier,
            stamp=stamp,
        )

    return float(runs.mean()), float(runs.std(ddof=0))


def _chunks(arr: np.ndarray, chunk_size: int) -> list[np.ndarray]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    out: list[np.ndarray] = []
    n = int(arr.shape[0])
    for i in range(0, n, int(chunk_size)):
        out.append(arr[i : i + int(chunk_size)])
    return out


def main() -> None:
    args = parse_args()

    out_ic = Path(args.out_ic)
    out_reg = Path(args.out_reg)
    ensure_parent(out_ic)
    ensure_parent(out_reg)

    if args.dry_run:
        df_ic = pd.DataFrame(
            {
                "node_id": pd.Series(dtype=str),
                "ic_score_mean": pd.Series(dtype=float),
                "ic_score_std": pd.Series(dtype=float),
                "n_runs": pd.Series(dtype=int),
                "p_model": pd.Series(dtype=str),
            }
        )
        df_ic.to_parquet(out_ic, index=False)
        pd.DataFrame({"node_id": pd.Series(dtype=str), "y": pd.Series(dtype=float)}).to_parquet(out_reg, index=False)
        print(f"[OK] Dry-run wrote empty I-A artifacts: {out_ic} and {out_reg} (timestamp={now_iso()})")
        return

    csr = load_csr_npz(Path(args.csr))
    indptr = csr["indptr"]
    indices = csr["indices"]
    degrees = csr["degrees"]
    node_ids_csr = csr["node_ids"].astype(str)

    df_primary = pd.read_parquet(Path(args.primary_ic))
    df_primary["node_id"] = df_primary["node_id"].astype(str)
    require_columns(df_primary, ["node_id", "n_runs"], "ic_scores_primary")

    labeled_node_ids = df_primary["node_id"].astype(str).tolist()
    if len(labeled_node_ids) == 0:
        raise ValueError("Primary IC artifact has zero labeled nodes")

    if int(args.n_runs) > 0:
        n_runs = int(args.n_runs)
    else:
        n_runs = int(pd.to_numeric(df_primary["n_runs"], errors="coerce").dropna().iloc[0])
        if n_runs <= 0:
            raise ValueError("Could not infer n_runs from primary IC artifact")

    # Map node_id -> CSR row index (vectorized).
    idx = pd.Index(node_ids_csr)
    labeled_rows = idx.get_indexer(labeled_node_ids)
    missing = int(np.sum(labeled_rows < 0))
    if missing:
        raise ValueError(f"{missing} labeled node_ids are missing from CSR mapping. Aborting.")
    labeled_rows = labeled_rows.astype(np.int64)

    # Load views and align to CSR order.
    df_attr = pd.read_parquet(Path(args.node_attributes))
    df_attr["node_id"] = df_attr["node_id"].astype(str)
    require_columns(df_attr, ["node_id", "views"], "node_attributes")

    views_series = pd.to_numeric(df_attr.set_index("node_id")["views"], errors="coerce").fillna(0.0)
    views_aligned = views_series.reindex(pd.Index(node_ids_csr), fill_value=0.0).to_numpy(dtype=np.float64)

    w = np.log1p(np.maximum(0.0, views_aligned)).astype(np.float64, copy=False)

    # denom per source u: sum_{v in N(u)} w[v]
    neigh_w_sum = np.add.reduceat(w[indices], indptr[:-1]).astype(np.float64, copy=False)
    neigh_w_sum[degrees <= 0] = 0.0

    chunk_size = int(args.chunk_size)
    row_chunks = _chunks(labeled_rows, chunk_size=chunk_size)

    def _worker_chunk(rows_chunk: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Allocate scratch arrays once per task for performance.
        n_nodes = int(neigh_w_sum.shape[0])
        marks = np.zeros(n_nodes, dtype=np.uint32)
        frontier = np.empty(n_nodes, dtype=np.int64)

        means = np.empty(len(rows_chunk), dtype=np.float64)
        stds = np.empty(len(rows_chunk), dtype=np.float64)
        stamp = 1
        for i, row in enumerate(rows_chunk.tolist()):
            m, s = _simulate_node_summary_ia(
                source=int(row),
                indptr=indptr,
                indices=indices,
                w=w,
                neigh_w_sum=neigh_w_sum,
                n_runs=n_runs,
                worker_seed=int(args.seed) + int(row),
                marks=marks,
                frontier=frontier,
                stamp_start=stamp,
            )
            means[i] = m
            stds[i] = s
            stamp += int(n_runs) + 1

        return rows_chunk.astype(np.int64, copy=False), means, stds

    t0 = time.time()
    stats = Parallel(n_jobs=int(args.n_jobs), backend="loky")(
        delayed(_worker_chunk)(chunk) for chunk in row_chunks
    )
    elapsed = float(time.time() - t0)

    means = np.empty(len(labeled_rows), dtype=np.float64)
    stds = np.empty(len(labeled_rows), dtype=np.float64)
    row_to_pos = {int(r): i for i, r in enumerate(labeled_rows.tolist())}
    for rows_chunk, means_chunk, stds_chunk in stats:
        for r, m, s in zip(rows_chunk.tolist(), means_chunk.tolist(), stds_chunk.tolist()):
            pos = row_to_pos[int(r)]
            means[pos] = float(m)
            stds[pos] = float(s)

    df_out = pd.DataFrame(
        {
            "node_id": np.array(labeled_node_ids, dtype=object),
            "ic_score_mean": means,
            "ic_score_std": stds,
            "n_runs": int(n_runs),
            "p_model": "ia_row_norm_views",
        }
    )

    df_reg = df_out[["node_id", "ic_score_mean"]].copy()
    df_reg["y"] = np.log1p(df_reg["ic_score_mean"].astype(np.float64))
    df_reg = df_reg[["node_id", "y"]]

    df_out.to_parquet(out_ic, index=False)
    df_reg.to_parquet(out_reg, index=False)

    print(
        "[OK] I-A IC labeling done "
        f"(n_labeled={len(df_out):,}, n_runs={n_runs}, elapsed_sec={elapsed:.2f}).\n"
        f" - ic_out={out_ic}\n"
        f" - reg_out={out_reg}"
    )


if __name__ == "__main__":
    main()
