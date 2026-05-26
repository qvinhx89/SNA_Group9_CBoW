"""MAPR2026 v3 — IC labels (sensitivity A2: symmetric normalization).

Goal
----
Generate an IC score artifact for the A2 diffusion rule using the *same labeled nodes*
as the primary A0 IC labeling.

A2 rule
-------
    p(u,v) = 1 / sqrt(deg(u) * deg(v))

Inputs
------
- data/processed/graph_csr.npz
- data/processed/ic_scores_primary.parquet  (to reuse the exact 5k labeled node_ids)

Outputs
-------
- outputs/mapr2026_v3_results/ic_scores_sensitivity_a2.parquet
  columns: node_id, ic_score_mean, ic_score_std, n_runs, p_model

- data/processed/regression_targets_a2.parquet
  columns: node_id, y  (y=log1p(ic_score_mean))
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
    p = argparse.ArgumentParser(description="MAPR2026 v3 IC sensitivity A2 (symmetric deg) labeling")
    p.add_argument("--csr", default=PATHS.csr_npz)
    p.add_argument(
        "--primary-ic",
        default=PATHS.ic_scores,
        help="Primary IC scores parquet (used only to reuse exact labeled node_ids)",
    )
    p.add_argument(
        "--out-ic",
        default=str(Path(PATHS.results_dir) / "ic_scores_sensitivity_a2.parquet"),
        help="Output parquet for A2 IC scores",
    )
    p.add_argument(
        "--out-reg",
        default="data/processed/regression_targets_a2.parquet",
        help="Output regression targets parquet for A2 (y=log1p(ic_score_mean))",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--n-runs",
        type=int,
        default=0,
        help="MC runs per node. If 0, reuse n_runs from primary IC artifact.",
    )
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _simulate_ic_once_a2(
    source: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    inv_sqrt_deg: np.ndarray,
    rng: np.random.Generator,
) -> int:
    activated = {int(source)}
    frontier = [int(source)]

    while frontier:
        next_frontier: list[int] = []
        for u in frontier:
            start_idx = int(indptr[u])
            end_idx = int(indptr[u + 1])
            p_u = float(inv_sqrt_deg[u])
            if p_u <= 0.0:
                continue
            for nb_raw in indices[start_idx:end_idx]:
                v = int(nb_raw)
                if v in activated:
                    continue
                p_v = float(inv_sqrt_deg[v])
                if p_v <= 0.0:
                    continue
                p = p_u * p_v
                if p <= 0.0:
                    continue
                if rng.random() < p:
                    activated.add(v)
                    next_frontier.append(v)
        frontier = next_frontier

    return len(activated)


def _simulate_node_summary_a2(
    source: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    inv_sqrt_deg: np.ndarray,
    n_runs: int,
    worker_seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(worker_seed)
    runs = np.empty(int(n_runs), dtype=np.int32)
    for i in range(int(n_runs)):
        runs[i] = _simulate_ic_once_a2(source, indptr, indices, inv_sqrt_deg, rng)
    return float(runs.mean()), float(runs.std(ddof=0))


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
        print(f"[OK] Dry-run wrote empty A2 artifacts: {out_ic} and {out_reg} (timestamp={now_iso()})")
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

    inv_sqrt_deg = np.zeros_like(degrees, dtype=float)
    mask = degrees > 0
    inv_sqrt_deg[mask] = 1.0 / np.sqrt(degrees[mask].astype(float))

    def _worker(row: int) -> tuple[float, float]:
        return _simulate_node_summary_a2(
            source=int(row),
            indptr=indptr,
            indices=indices,
            inv_sqrt_deg=inv_sqrt_deg,
            n_runs=n_runs,
            worker_seed=int(args.seed) + int(row),
        )

    t0 = time.time()
    stats = Parallel(n_jobs=int(args.n_jobs), backend="loky")(delayed(_worker)(int(r)) for r in labeled_rows)
    elapsed = float(time.time() - t0)

    means = np.array([m for m, _ in stats], dtype=float)
    stds = np.array([s for _, s in stats], dtype=float)

    df_out = pd.DataFrame(
        {
            "node_id": np.array(labeled_node_ids, dtype=object),
            "ic_score_mean": means,
            "ic_score_std": stds,
            "n_runs": int(n_runs),
            "p_model": "symmetric_deg",
        }
    )

    # Regression targets for rerunning surrogates.
    df_reg = df_out[["node_id", "ic_score_mean"]].copy()
    df_reg["y"] = np.log1p(df_reg["ic_score_mean"].astype(float))
    df_reg = df_reg[["node_id", "y"]]

    df_out.to_parquet(out_ic, index=False)
    df_reg.to_parquet(out_reg, index=False)

    print(
        "[OK] A2 IC labeling done "
        f"(n_labeled={len(df_out):,}, n_runs={n_runs}, elapsed_sec={elapsed:.2f}).\n"
        f" - ic_out={out_ic}\n"
        f" - reg_out={out_reg}"
    )


if __name__ == "__main__":
    main()
