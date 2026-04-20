"""MAPR2026 v3 — IC label stability check (post-Day1).

Owner: Person 1

Purpose
-------
Compute stability evidence for IC labels on the already-labeled node set.
The check runs independent MC experiments with different seed offsets and
reports pairwise top-decile Jaccard and rank Spearman.

Default protocol (aligned with plan docs):
- mc_seeds: 0,1,2
- n_runs per seed: 150
- worker_seed = mc_seed * seed_multiplier + node_row
- top-pct: 0.10

Output
------
- outputs/day1_benchmark/ic_label_stability.json
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from _shared import PATHS, ensure_parent, load_csr_npz, now_iso, require_columns, write_json
from ic_labels_primary import _compute_views_strength_aligned, _simulate_ic_node_summary


@dataclass(frozen=True)
class StabilityResult:
    seed_i: int
    seed_j: int
    jaccard_top_decile: float
    spearman_rank: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IC label stability check on labeled nodes")
    p.add_argument("--csr", default=PATHS.csr_npz)
    p.add_argument("--ic", default=PATHS.ic_scores)
    p.add_argument("--out", default=f"{PATHS.day1_dir}/ic_label_stability.json")
    p.add_argument("--n-runs", type=int, default=150)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--mc-seeds", default="0,1,2", help="Comma-separated MC seed ids")
    p.add_argument("--seed-multiplier", type=int, default=10000)
    p.add_argument("--top-pct", type=float, default=0.10)
    p.add_argument("--jaccard-threshold", type=float, default=0.85)

    # IC probability model (default: original weighted cascade)
    p.add_argument(
        "--p-model",
        default="weighted_cascade",
        choices=[
            "weighted_cascade",
            "hybrid_degree_views_mult",
            "hybrid_degree_views_centered",
        ],
        help=(
            "Activation probability model. weighted_cascade uses p(u->v)=1/deg(v). "
            "Hybrid variants use sender strength derived from log1p(views)."
        ),
    )
    p.add_argument("--node-attrs", default=PATHS.node_attributes)
    p.add_argument("--views-col", default="views")
    p.add_argument("--hybrid-gamma", type=float, default=0.0)
    p.add_argument("--views-q-low", type=float, default=0.05)
    p.add_argument("--views-q-high", type=float, default=0.95)
    p.add_argument(
        "--max-nodes",
        type=int,
        default=0,
        help="Optional cap for quick smoke tests (0 means all labeled nodes)",
    )
    return p.parse_args()


def _parse_seed_list(raw: str) -> list[int]:
    out: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    if len(out) < 2:
        raise ValueError("Need at least 2 mc seeds for stability check")
    return out


def _simulate_means_for_seed(
    rows: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    inv_degrees: np.ndarray,
    n_runs: int,
    seed_offset: int,
    n_jobs: int,
    p_model: str,
    sender_strength: np.ndarray | None,
    hybrid_gamma: float,
) -> np.ndarray:
    def _worker(row: int) -> float:
        mean_score, _ = _simulate_ic_node_summary(
            source=int(row),
            indptr=indptr,
            indices=indices,
            inv_degrees=inv_degrees,
            n_runs=n_runs,
            worker_seed=seed_offset + int(row),
            p_model=str(p_model),
            sender_strength=sender_strength,
            hybrid_gamma=float(hybrid_gamma),
        )
        return float(mean_score)

    means = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_worker)(int(row)) for row in rows
    )
    return np.asarray(means, dtype=float)


def _top_decile_set(scores: np.ndarray, top_pct: float) -> set[int]:
    thresh = float(np.quantile(scores, 1.0 - top_pct))
    return set(np.where(scores >= thresh)[0].tolist())


def _jaccard(a: set[int], b: set[int]) -> float:
    union = a | b
    if not union:
        return 1.0
    return float(len(a & b) / len(union))


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(pd.Series(x).corr(pd.Series(y), method="spearman"))


def main() -> None:
    args = parse_args()
    mc_seeds = _parse_seed_list(args.mc_seeds)

    csr = load_csr_npz(args.csr)
    df_ic = pd.read_parquet(args.ic)
    require_columns(df_ic, ["node_id", "ic_score_mean"], "ic_scores")

    node_ids = csr["node_ids"].astype(str)
    node_to_row = {node_id: i for i, node_id in enumerate(node_ids.tolist())}

    missing_nodes = sorted(set(df_ic["node_id"].astype(str)) - set(node_to_row.keys()))
    if missing_nodes:
        raise ValueError(
            f"{len(missing_nodes)} labeled nodes not found in CSR mapping. "
            "Cannot run stability check."
        )

    ordered_node_ids = df_ic["node_id"].astype(str).to_numpy()
    rows = np.asarray([node_to_row[n] for n in ordered_node_ids], dtype=np.int64)
    if int(args.max_nodes) > 0:
        rows = rows[: int(args.max_nodes)]

    degrees = csr["degrees"]
    inv_degrees = np.zeros_like(degrees, dtype=float)
    positive = degrees > 0
    inv_degrees[positive] = 1.0 / degrees[positive].astype(float)

    sender_strength: np.ndarray | None = None
    if str(args.p_model) in {"hybrid_degree_views_mult", "hybrid_degree_views_centered"}:
        raw_strength, _meta = _compute_views_strength_aligned(
            node_ids=node_ids,
            node_attrs_path=args.node_attrs,
            views_col=str(args.views_col),
            q_low=float(args.views_q_low),
            q_high=float(args.views_q_high),
        )
        if str(args.p_model) == "hybrid_degree_views_centered":
            sender_strength = (raw_strength - float(raw_strength.mean())).astype(
                np.float64, copy=False
            )
        else:
            sender_strength = raw_strength

    scores_by_seed: dict[int, np.ndarray] = {}
    top_sets: dict[int, set[int]] = {}
    top_counts: dict[int, int] = {}

    for mc_seed in mc_seeds:
        seed_offset = int(mc_seed) * int(args.seed_multiplier)
        means = _simulate_means_for_seed(
            rows=rows,
            indptr=csr["indptr"],
            indices=csr["indices"],
            inv_degrees=inv_degrees,
            n_runs=int(args.n_runs),
            seed_offset=seed_offset,
            n_jobs=int(args.n_jobs),
            p_model=str(args.p_model),
            sender_strength=sender_strength,
            hybrid_gamma=float(args.hybrid_gamma),
        )
        scores_by_seed[int(mc_seed)] = means
        top_set = _top_decile_set(means, top_pct=float(args.top_pct))
        top_sets[int(mc_seed)] = top_set
        top_counts[int(mc_seed)] = int(len(top_set))

    pairwise: list[StabilityResult] = []
    for i in range(len(mc_seeds)):
        for j in range(i + 1, len(mc_seeds)):
            seed_i = int(mc_seeds[i])
            seed_j = int(mc_seeds[j])
            pairwise.append(
                StabilityResult(
                    seed_i=seed_i,
                    seed_j=seed_j,
                    jaccard_top_decile=_jaccard(top_sets[seed_i], top_sets[seed_j]),
                    spearman_rank=_spearman(scores_by_seed[seed_i], scores_by_seed[seed_j]),
                )
            )

    jaccards = [p.jaccard_top_decile for p in pairwise]
    spearmans = [p.spearman_rank for p in pairwise]

    payload = {
        "timestamp": now_iso(),
        "config": {
            "n_labeled_nodes": int(len(rows)),
            "n_runs_per_seed": int(args.n_runs),
            "mc_seeds": [int(x) for x in mc_seeds],
            "seed_multiplier": int(args.seed_multiplier),
            "worker_seed_rule": "mc_seed * seed_multiplier + node_row",
            "top_pct": float(args.top_pct),
            "jaccard_threshold": float(args.jaccard_threshold),
            "p_model": str(args.p_model),
            "hybrid_gamma": float(args.hybrid_gamma),
            "views_col": str(args.views_col),
            "views_q_low": float(args.views_q_low),
            "views_q_high": float(args.views_q_high),
            "max_nodes": int(args.max_nodes),
        },
        "top_decile_counts": {str(k): int(v) for k, v in top_counts.items()},
        "pairwise": [asdict(p) for p in pairwise],
        "summary": {
            "jaccard_mean": float(np.mean(jaccards)),
            "jaccard_min": float(np.min(jaccards)),
            "jaccard_pass_threshold": bool(float(np.min(jaccards)) >= float(args.jaccard_threshold)),
            "spearman_mean": float(np.mean(spearmans)),
            "spearman_min": float(np.min(spearmans)),
        },
    }

    write_json(args.out, payload)
    print(f"[OK] Wrote stability report: {Path(args.out)}")


if __name__ == "__main__":
    main()
