"""MAPR2026 v3 — IC labels (HSCC-refined).

Formula (from IC_results_feedback.md)
------------------------------------
    p(u,v) = clip(
        lambda * (phi(u) / deg(u)) * (1 + gamma * I[c_u != c_v]),
        0,
        p_max,
    )

where:
    phi(u) = rank(log1p(views_u) / (1 + life_time_u)) / N

Notes
-----
- Reuses the exact labeled node_ids from primary IC artifact.
- Exports both IC scores and regression targets (y=log1p(ic_score_mean)).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

# IMPORTANT (Windows): torch must be imported before numpy/pandas.
# Importing numpy/pandas first can trigger WinError 1114 during torch DLL init.
try:
    import torch
except Exception:  # pragma: no cover - optional dependency for CPU-only runs
    torch = None

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from _shared import PATHS, ensure_parent, load_csr_npz, now_iso, require_columns


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MAPR2026 v3 IC labeling (HSCC-refined)")
    p.add_argument("--csr", default=PATHS.csr_npz)
    p.add_argument("--node-attrs", default=PATHS.node_attributes)
    p.add_argument("--community", default="data/processed/community_labels.parquet")
    p.add_argument(
        "--primary-ic",
        default=PATHS.ic_scores,
        help="Primary IC scores parquet (used only to reuse exact labeled node_ids)",
    )
    p.add_argument(
        "--out-ic",
        default=str(Path(PATHS.results_dir) / "ic_scores_hscc_refined.parquet"),
        help="Output parquet for HSCC-refined IC scores",
    )
    p.add_argument(
        "--out-reg",
        default="data/processed/regression_targets_hscc_refined.parquet",
        help="Output regression targets parquet for HSCC-refined (y=log1p(ic_score_mean))",
    )
    p.add_argument(
        "--out-diag",
        default=str(Path(PATHS.results_dir) / "hscc_refined_label_diagnostics.json"),
        help="Optional diagnostics json",
    )
    p.add_argument("--lambda-coef", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--p-max", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--n-runs",
        type=int,
        default=0,
        help="MC runs per node. If 0, reuse n_runs from primary IC artifact.",
    )
    p.add_argument(
        "--max-labeled",
        type=int,
        default=0,
        help="Optional cap on labeled nodes for smoke testing (0 = all labeled nodes).",
    )
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device. 'auto' uses CUDA if available, else CPU.",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=250,
        help="Print progress every N labeled nodes (0 disables).",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _load_source_strength(node_attrs_path: Path) -> pd.DataFrame:
    attrs = pd.read_parquet(node_attrs_path)
    attrs["node_id"] = attrs["node_id"].astype(str)
    require_columns(attrs, ["node_id", "views"], "node_attributes")

    if "life_time" in attrs.columns:
        life_time = pd.to_numeric(attrs["life_time"], errors="coerce").fillna(1.0)
    elif "life_time_days" in attrs.columns:
        life_time = pd.to_numeric(attrs["life_time_days"], errors="coerce").fillna(1.0)
    else:
        life_time = pd.Series(np.ones(len(attrs), dtype=float), index=attrs.index)

    life_time = life_time.clip(lower=0.0)
    views = pd.to_numeric(attrs["views"], errors="coerce").fillna(0.0).clip(lower=0.0)
    base_score = np.log1p(views) / (1.0 + life_time.astype(float))

    # rank in (0,1], matching phi(u)=rank/N in the feedback formula.
    phi = pd.Series(base_score, index=attrs.index).rank(method="average", pct=True).astype(float)

    out = pd.DataFrame({"node_id": attrs["node_id"], "phi": phi})
    return out


def _load_community_ids(community_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(community_path)
    df["node_id"] = df["node_id"].astype(str)

    if "community_id" in df.columns:
        comm_col = "community_id"
    elif "community" in df.columns:
        comm_col = "community"
    else:
        raise ValueError("Community parquet must include either 'community_id' or 'community'.")

    out = df[["node_id", comm_col]].copy().rename(columns={comm_col: "community_id"})
    out["community_id"] = pd.to_numeric(out["community_id"], errors="coerce").fillna(-1).astype(np.int64)
    return out


def _simulate_ic_once_hscc(
    source: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    base_sender: np.ndarray,
    comm_ids: np.ndarray,
    gamma: float,
    p_max: float,
    rng: np.random.Generator,
) -> int:
    activated = {int(source)}
    frontier = [int(source)]

    while frontier:
        next_frontier: list[int] = []
        for u in frontier:
            start_idx = int(indptr[u])
            end_idx = int(indptr[u + 1])

            base_u = float(base_sender[u])
            if base_u <= 0.0:
                continue

            comm_u = int(comm_ids[u])

            for nb_raw in indices[start_idx:end_idx]:
                v = int(nb_raw)
                if v in activated:
                    continue

                p = base_u
                if comm_u != int(comm_ids[v]):
                    p = p * (1.0 + float(gamma))

                if p > p_max:
                    p = p_max
                if p > 1.0:
                    p = 1.0
                if p <= 0.0:
                    continue

                if rng.random() < p:
                    activated.add(v)
                    next_frontier.append(v)

        frontier = next_frontier

    return len(activated)


def _simulate_node_summary_hscc(
    source: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    base_sender: np.ndarray,
    comm_ids: np.ndarray,
    gamma: float,
    p_max: float,
    n_runs: int,
    worker_seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(worker_seed)
    runs = np.empty(int(n_runs), dtype=np.int32)
    for i in range(int(n_runs)):
        runs[i] = _simulate_ic_once_hscc(
            source=source,
            indptr=indptr,
            indices=indices,
            base_sender=base_sender,
            comm_ids=comm_ids,
            gamma=gamma,
            p_max=p_max,
            rng=rng,
        )
    return float(runs.mean()), float(runs.std(ddof=0))


def _simulate_ic_once_hscc_cuda(
    source: int,
    indptr_np: np.ndarray,
    indices_cuda: Any,
    edge_p_cuda: Any,
    gen: Any,
    marks_cuda: Any,
    stamp: int,
) -> int:
    if torch is None:
        raise RuntimeError("Torch is required for CUDA execution.")

    marks_cuda[int(source)] = int(stamp)

    frontier: list[int] = [int(source)]
    activated_count = 1
    stamp_i = int(stamp)
    device = edge_p_cuda.device

    while frontier:
        nbr_slices: list[Any] = []
        p_slices: list[Any] = []

        for u in frontier:
            start = int(indptr_np[int(u)])
            end = int(indptr_np[int(u) + 1])
            if end <= start:
                continue
            nbr_slices.append(indices_cuda[start:end])
            p_slices.append(edge_p_cuda[start:end])

        if len(nbr_slices) == 0:
            break

        nbrs_all = torch.cat(nbr_slices, dim=0)
        p_all = torch.cat(p_slices, dim=0)

        unvisited = marks_cuda[nbrs_all] != stamp_i
        if not bool(unvisited.any()):
            break

        nbrs = nbrs_all[unvisited]
        p = p_all[unvisited]
        if nbrs.numel() == 0:
            break

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

    out_ic = Path(args.out_ic)
    out_reg = Path(args.out_reg)
    out_diag = Path(args.out_diag)
    ensure_parent(out_ic)
    ensure_parent(out_reg)
    ensure_parent(out_diag)

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

        diag = {
            "timestamp": now_iso(),
            "mode": "dry_run",
            "formula": "hscc_refined",
            "lambda_coef": float(args.lambda_coef),
            "gamma": float(args.gamma),
            "p_max": float(args.p_max),
            "ic_out": str(out_ic),
            "reg_out": str(out_reg),
        }
        out_diag.write_text(pd.Series(diag).to_json(indent=2), encoding="utf-8")
        print(f"[OK] Dry-run wrote empty HSCC-refined artifacts: {out_ic} and {out_reg} (timestamp={now_iso()})")
        return

    csr = load_csr_npz(Path(args.csr))
    indptr = csr["indptr"]
    indices = csr["indices"]
    degrees = csr["degrees"].astype(np.int64, copy=False)
    node_ids_csr = csr["node_ids"].astype(str)

    node_ids_index = pd.Index(node_ids_csr)

    # Labeled nodes are locked to primary IC artifact.
    df_primary = pd.read_parquet(Path(args.primary_ic))
    df_primary["node_id"] = df_primary["node_id"].astype(str)
    require_columns(df_primary, ["node_id", "n_runs"], "ic_scores_primary")
    labeled_node_ids = df_primary["node_id"].astype(str).tolist()
    if len(labeled_node_ids) == 0:
        raise ValueError("Primary IC artifact has zero labeled nodes")

    if int(args.max_labeled) > 0:
        labeled_node_ids = labeled_node_ids[: int(args.max_labeled)]

    if int(args.n_runs) > 0:
        n_runs = int(args.n_runs)
    else:
        n_runs = int(pd.to_numeric(df_primary["n_runs"], errors="coerce").dropna().iloc[0])
        if n_runs <= 0:
            raise ValueError("Could not infer n_runs from primary IC artifact")

    # Build phi(u) from node attributes (all nodes in CSR scope).
    phi_df = _load_source_strength(Path(args.node_attrs))
    phi_rows = node_ids_index.get_indexer(phi_df["node_id"].astype(str).to_numpy())
    valid_phi = phi_rows >= 0
    phi_vec = np.zeros(len(node_ids_csr), dtype=float)
    phi_vec[phi_rows[valid_phi]] = phi_df.loc[valid_phi, "phi"].to_numpy(dtype=float)

    # Build community id vector (all nodes in CSR scope).
    comm_df = _load_community_ids(Path(args.community))
    comm_rows = node_ids_index.get_indexer(comm_df["node_id"].astype(str).to_numpy())
    valid_comm = comm_rows >= 0
    comm_vec = np.full(len(node_ids_csr), -1, dtype=np.int64)
    comm_vec[comm_rows[valid_comm]] = comm_df.loc[valid_comm, "community_id"].to_numpy(dtype=np.int64)

    # base_sender(u) = lambda * phi(u) / deg(u)
    base_sender = np.zeros(len(node_ids_csr), dtype=float)
    deg_mask = degrees > 0
    base_sender[deg_mask] = float(args.lambda_coef) * (phi_vec[deg_mask] / degrees[deg_mask].astype(float))

    # Map labeled ids -> CSR rows
    labeled_rows = node_ids_index.get_indexer(np.array(labeled_node_ids, dtype=object))
    missing = int(np.sum(labeled_rows < 0))
    if missing:
        raise ValueError(f"{missing} labeled node_ids are missing from CSR mapping. Aborting.")

    labeled_rows = labeled_rows.astype(np.int64)

    device_choice = str(args.device).lower().strip()
    if device_choice == "cuda":
        if torch is None or not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA torch is unavailable.")
        use_cuda = True
    elif device_choice == "cpu":
        use_cuda = False
    else:
        use_cuda = bool(torch is not None and torch.cuda.is_available())

    t0 = time.time()
    if use_cuda:
        if torch is None:
            raise RuntimeError("Torch is required for CUDA execution.")

        print(
            "[CUDA] start HSCC-refined labeling "
            f"(device={torch.cuda.get_device_name(0)}, n_labeled={len(labeled_rows):,}, "
            f"n_runs={n_runs}, ts={now_iso()})"
        )

        # Per-edge activation probability for directed edges in CSR.
        edge_p = np.zeros(indices.shape[0], dtype=np.float32)
        for u in range(len(node_ids_csr)):
            start = int(indptr[u])
            end = int(indptr[u + 1])
            if end <= start:
                continue

            base_u = float(base_sender[u])
            if base_u <= 0.0:
                continue

            nbrs = indices[start:end]
            cross = comm_vec[nbrs] != int(comm_vec[u])
            p = base_u * (1.0 + float(args.gamma) * cross.astype(float))
            p = np.clip(p, 0.0, float(args.p_max))
            p = np.minimum(p, 1.0)
            edge_p[start:end] = p.astype(np.float32, copy=False)

        device = torch.device("cuda")
        indices_cuda = torch.from_numpy(indices.astype(np.int64, copy=False)).to(device=device, dtype=torch.int64)
        edge_p_cuda = torch.from_numpy(edge_p).to(device=device, dtype=torch.float32)
        marks = torch.zeros((len(node_ids_csr),), device=device, dtype=torch.int32)

        means = np.empty((len(labeled_rows),), dtype=np.float64)
        stds = np.empty((len(labeled_rows),), dtype=np.float64)

        progress_every = int(args.progress_every)
        global_stamp = 1
        with torch.no_grad():
            for i, row in enumerate(labeled_rows.tolist()):
                gen = torch.Generator(device=device)
                gen.manual_seed(int(args.seed) + int(row))

                runs = np.empty((int(n_runs),), dtype=np.int32)
                for j in range(int(n_runs)):
                    global_stamp += 1
                    runs[j] = _simulate_ic_once_hscc_cuda(
                        source=int(row),
                        indptr_np=indptr,
                        indices_cuda=indices_cuda,
                        edge_p_cuda=edge_p_cuda,
                        gen=gen,
                        marks_cuda=marks,
                        stamp=int(global_stamp),
                    )

                means[i] = float(runs.mean())
                stds[i] = float(runs.std(ddof=0))

                if progress_every > 0 and ((i + 1) % progress_every == 0):
                    elapsed_i = float(time.time() - t0)
                    print(
                        f"[CUDA] progress {i+1}/{len(labeled_rows)} "
                        f"(elapsed_sec={elapsed_i:.1f}, stamp={global_stamp}, ts={now_iso()})"
                    )
    else:
        def _worker(row: int) -> tuple[float, float]:
            return _simulate_node_summary_hscc(
                source=int(row),
                indptr=indptr,
                indices=indices,
                base_sender=base_sender,
                comm_ids=comm_vec,
                gamma=float(args.gamma),
                p_max=float(args.p_max),
                n_runs=n_runs,
                worker_seed=int(args.seed) + int(row),
            )

        stats = Parallel(n_jobs=int(args.n_jobs), backend="loky")(delayed(_worker)(int(r)) for r in labeled_rows)
        means = np.array([m for m, _ in stats], dtype=float)
        stds = np.array([s for _, s in stats], dtype=float)

    elapsed = float(time.time() - t0)

    df_out = pd.DataFrame(
        {
            "node_id": np.array(labeled_node_ids, dtype=object),
            "ic_score_mean": means,
            "ic_score_std": stds,
            "n_runs": int(n_runs),
            "p_model": "hscc_refined",
        }
    )

    df_reg = df_out[["node_id", "ic_score_mean"]].copy()
    df_reg["y"] = np.log1p(df_reg["ic_score_mean"].astype(float))
    df_reg = df_reg[["node_id", "y"]]

    df_out.to_parquet(out_ic, index=False)
    df_reg.to_parquet(out_reg, index=False)

    diag = {
        "timestamp": now_iso(),
        "formula": "hscc_refined",
        "lambda_coef": float(args.lambda_coef),
        "gamma": float(args.gamma),
        "p_max": float(args.p_max),
        "n_labeled": int(len(df_out)),
        "n_runs": int(n_runs),
        "elapsed_sec": float(elapsed),
        "ic_score_mean_summary": {
            "min": float(df_out["ic_score_mean"].min()),
            "max": float(df_out["ic_score_mean"].max()),
            "mean": float(df_out["ic_score_mean"].mean()),
            "std": float(df_out["ic_score_mean"].std(ddof=0)),
            "cv": float(df_out["ic_score_mean"].std(ddof=0) / (df_out["ic_score_mean"].mean() + 1e-12)),
        },
        "ic_out": str(out_ic),
        "reg_out": str(out_reg),
    }
    out_diag.write_text(pd.Series(diag).to_json(indent=2), encoding="utf-8")

    print(
        "[OK] HSCC-refined IC labeling done "
        f"(n_labeled={len(df_out):,}, n_runs={n_runs}, elapsed_sec={elapsed:.2f}).\n"
        f" - ic_out={out_ic}\n"
        f" - reg_out={out_reg}\n"
        f" - diag_out={out_diag}"
    )


if __name__ == "__main__":
    main()
