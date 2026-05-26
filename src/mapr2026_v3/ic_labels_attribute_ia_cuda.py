"""MAPR2026 v3.1 — IC labels (I-A, CUDA-only).

This is a CUDA/GPU implementation of the optional I-A labeling branch.
It preserves the same I-A rule as the CPU version:

    p(u,v) = w(v) / sum_{x in N(u)} w(x)
    w(v) = log1p(max(views(v), 0))

Contract outputs
----------------
- outputs/mapr2026_v3_results/ic_scores_ia.parquet
  columns: node_id, ic_score_mean, ic_score_std, n_runs, p_model

- data/processed/regression_targets_ia.parquet
  columns: node_id, y  (y=log1p(ic_score_mean))

Notes
-----
- Requires CUDA-capable PyTorch; aborts if CUDA is unavailable.
- Control flow remains Pythonic (frontier expansion), but sampling and state
  are stored on GPU to satisfy the GPU-only execution requirement.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch

# IMPORTANT (Windows): torch must be imported before numpy/pandas.
# Importing numpy/pandas first can trigger WinError 1114 during torch DLL init.
import numpy as np
import pandas as pd

from _shared import PATHS, ensure_parent, load_csr_npz, now_iso, require_columns


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MAPR2026 v3.1 IC labeling (I-A CUDA-only)")
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
    p.add_argument(
        "--device",
        default="cuda",
        choices=["cuda"],
        help="CUDA-only: this script will abort if CUDA is unavailable.",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=250,
        help="Print progress every N labeled nodes (0 disables).",
    )
    p.add_argument(
        "--max-nodes",
        type=int,
        default=0,
        help="Optional: limit to first N labeled nodes (0 means all). Useful for smoke tests.",
    )
    return p.parse_args()


@torch.no_grad()
def _simulate_ic_once_ia_cuda(
    source: int,
    indptr_np: np.ndarray,
    indices_cuda: torch.Tensor,
    w_cuda: torch.Tensor,
    neigh_w_sum_np: np.ndarray,
    gen: torch.Generator,
    marks_cuda: torch.Tensor,
    stamp: int,
) -> int:
    marks_cuda[int(source)] = int(stamp)

    frontier: list[int] = [int(source)]
    activated_count = 1

    device = w_cuda.device
    stamp_i = int(stamp)

    while frontier:
        slices: list[torch.Tensor] = []
        degs: list[int] = []
        denoms: list[float] = []

        for u in frontier:
            denom = float(neigh_w_sum_np[int(u)])
            if denom <= 0.0 or not math.isfinite(denom):
                continue

            start = int(indptr_np[int(u)])
            end = int(indptr_np[int(u) + 1])
            deg = int(end - start)
            if deg <= 0:
                continue

            slices.append(indices_cuda[start:end])
            degs.append(deg)
            denoms.append(denom)

        if len(slices) == 0:
            break

        nbrs_all = torch.cat(slices, dim=0)
        degs_t = torch.tensor(degs, device=device, dtype=torch.int64)
        denoms_t = torch.tensor(denoms, device=device, dtype=torch.float32)
        denoms_rep = torch.repeat_interleave(denoms_t, degs_t)

        # Filter already-activated nodes for this run.
        unvisited = marks_cuda[nbrs_all] != stamp_i
        if not bool(unvisited.any()):
            break

        nbrs = nbrs_all[unvisited]
        denoms_rep = denoms_rep[unvisited]
        if nbrs.numel() == 0:
            break

        p = w_cuda[nbrs] / denoms_rep
        p = torch.clamp(p, 0.0, 1.0)

        r = torch.rand((int(nbrs.numel()),), device=device, generator=gen)
        hit = r < p
        if not bool(hit.any()):
            break

        new = nbrs[hit]
        if new.numel() == 0:
            break

        new = torch.unique(new)
        marks_cuda[new] = stamp_i

        new_cpu = new.detach().cpu().to(torch.int64).tolist()
        activated_count += int(len(new_cpu))
        frontier = new_cpu

    return int(activated_count)


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this Python environment. "
            "Install a CUDA-enabled torch build and ensure NVIDIA drivers are working."
        )

    device = torch.device("cuda")

    out_ic = Path(args.out_ic)
    out_reg = Path(args.out_reg)
    ensure_parent(out_ic)
    ensure_parent(out_reg)

    csr = load_csr_npz(Path(args.csr))
    indptr = csr["indptr"].astype(np.int64, copy=False)
    indices_np = csr["indices"].astype(np.int64, copy=False)
    degrees = csr["degrees"].astype(np.int64, copy=False)
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

    if int(args.max_nodes) > 0:
        keep = int(args.max_nodes)
        labeled_rows = labeled_rows[:keep]
        labeled_node_ids = labeled_node_ids[:keep]

    print(
        "[CUDA] start I-A labeling "
        f"(device={torch.cuda.get_device_name(0)}, n_labeled={len(labeled_rows):,}, "
        f"n_runs={n_runs}, ts={now_iso()})"
    )

    # Load views and align to CSR order.
    df_attr = pd.read_parquet(Path(args.node_attributes))
    df_attr["node_id"] = df_attr["node_id"].astype(str)
    require_columns(df_attr, ["node_id", "views"], "node_attributes")

    views_series = pd.to_numeric(df_attr.set_index("node_id")["views"], errors="coerce").fillna(0.0)
    views_aligned = views_series.reindex(pd.Index(node_ids_csr), fill_value=0.0).to_numpy(dtype=np.float64)

    w = np.log1p(np.maximum(0.0, views_aligned)).astype(np.float32, copy=False)

    # denom per source u: sum_{v in N(u)} w[v]
    neigh_w_sum = np.add.reduceat(w[indices_np], indptr[:-1]).astype(np.float64, copy=False)
    neigh_w_sum[degrees <= 0] = 0.0

    # Move large arrays to CUDA once.
    indices_cuda = torch.from_numpy(indices_np).to(device=device, dtype=torch.int64)
    w_cuda = torch.from_numpy(w).to(device=device, dtype=torch.float32)

    n_nodes = int(node_ids_csr.shape[0])
    marks = torch.zeros((n_nodes,), device=device, dtype=torch.int32)

    means = np.empty((len(labeled_rows),), dtype=np.float64)
    stds = np.empty((len(labeled_rows),), dtype=np.float64)

    # One generator per labeled node (matches CPU pattern: seed + row).
    progress_every = int(args.progress_every)
    t0 = time.time()
    global_stamp = 1

    for i, row in enumerate(labeled_rows.tolist()):
        worker_seed = int(args.seed) + int(row)
        gen = torch.Generator(device=device)
        gen.manual_seed(int(worker_seed))

        s1 = 0.0
        s2 = 0.0
        for _ in range(int(n_runs)):
            global_stamp += 1
            reach = _simulate_ic_once_ia_cuda(
                source=int(row),
                indptr_np=indptr,
                indices_cuda=indices_cuda,
                w_cuda=w_cuda,
                neigh_w_sum_np=neigh_w_sum,
                gen=gen,
                marks_cuda=marks,
                stamp=int(global_stamp),
            )
            s1 += float(reach)
            s2 += float(reach) * float(reach)

        mean = s1 / float(n_runs)
        var = max(0.0, (s2 / float(n_runs)) - mean * mean)
        std = math.sqrt(var)

        means[i] = float(mean)
        stds[i] = float(std)

        if progress_every > 0 and ((i + 1) % progress_every == 0):
            elapsed = float(time.time() - t0)
            print(
                f"[CUDA] progress {i+1}/{len(labeled_rows)} "
                f"(elapsed_sec={elapsed:.1f}, stamp={global_stamp}, ts={now_iso()})"
            )

    elapsed = float(time.time() - t0)

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
        "[OK] I-A IC labeling (CUDA) done "
        f"(n_labeled={len(df_out):,}, n_runs={n_runs}, elapsed_sec={elapsed:.2f}).\n"
        f" - ic_out={out_ic}\n"
        f" - reg_out={out_reg}"
    )


if __name__ == "__main__":
    main()
