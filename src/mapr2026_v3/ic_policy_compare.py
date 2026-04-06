"""MAPR2026 v3 — P2 policy comparison for noisy top-10 labels.

Owner: Person 1

Purpose
-------
Compare three label policies when P1 stability has not passed:
- Policy A: hard top 10% (current y_top10)
- Policy B: consensus top 10% (selected in >=2/3 MC seeds)
- Policy C: probabilistic top 10% (rank by p_above_top10_threshold)

Selection rule
--------------
Choose the policy with the lowest boundary/ambiguous noise among policies that
keep typology quadrant sizes valid (min quadrant size constraint).

Important export contract
-------------------------
`policy_b` is the official consensus-B label and is defined as:
policy_b == (seed_vote_count >= consensus_k), where `seed_vote_count` is the
number of MC seeds (from `--mc-seeds`) that place a node in top decile.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from _shared import PATHS, ensure_dir, load_csr_npz, now_iso, require_columns
from ic_labels_primary import _simulate_ic_node_summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare A/B/C policies and pick P2 winner")
    p.add_argument("--csr", default=PATHS.csr_npz)
    p.add_argument("--ic", default=PATHS.ic_scores)
    p.add_argument("--ic-ci", default="data/processed/ic_scores_primary_with_ci.parquet")
    p.add_argument("--cls", default=PATHS.classification_labels)
    p.add_argument("--node-attrs", default=PATHS.node_attributes)
    p.add_argument("--out-dir", default="outputs/day1_benchmark/policy_compare")
    p.add_argument("--top-pct", type=float, default=0.10)
    p.add_argument("--mc-seeds", default="0,1,2")
    p.add_argument("--seed-multiplier", type=int, default=10000)
    p.add_argument("--consensus-k", type=int, default=2)
    p.add_argument(
        "--n-runs-consensus",
        type=int,
        default=0,
        help="Runs per node for Policy B consensus. 0 means auto-read from ic_scores n_runs.",
    )
    p.add_argument("--stable-pos-prob", type=float, default=0.90)
    p.add_argument("--stable-neg-prob", type=float, default=0.10)
    p.add_argument("--min-quadrant-size", type=int, default=150)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument(
        "--max-nodes",
        type=int,
        default=0,
        help="Optional cap for quick smoke tests (0 means all labeled nodes)",
    )
    return p.parse_args()


def _parse_int_list(raw: str) -> list[int]:
    out: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            out.append(int(token))
    if not out:
        raise ValueError("List argument cannot be empty")
    return out


def _top_decile_set(scores: np.ndarray, top_pct: float) -> set[int]:
    thresh = float(np.quantile(scores, 1.0 - top_pct))
    return set(np.where(scores >= thresh)[0].tolist())


def _simulate_means_for_seed(
    rows: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    inv_degrees: np.ndarray,
    n_runs: int,
    seed_offset: int,
    n_jobs: int,
) -> np.ndarray:
    def _worker(row: int) -> float:
        mean_score, _ = _simulate_ic_node_summary(
            source=int(row),
            indptr=indptr,
            indices=indices,
            inv_degrees=inv_degrees,
            n_runs=n_runs,
            worker_seed=seed_offset + int(row),
        )
        return float(mean_score)

    means = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_worker)(int(row)) for row in rows
    )
    return np.asarray(means, dtype=float)


def _evaluate_policy(
    policy_name: str,
    y_pos: np.ndarray,
    crossing: np.ndarray,
    ambiguous: np.ndarray,
    views_high: np.ndarray,
    min_quadrant_size: int,
    top_pct: float,
) -> dict[str, Any]:
    n_total = int(y_pos.shape[0])
    n_pos = int(y_pos.sum())
    pos_ratio = float(n_pos / max(n_total, 1))

    n_boundary_pos = int((y_pos & crossing).sum())
    n_ambiguous_pos = int((y_pos & ambiguous).sum())

    boundary_ratio = float(n_boundary_pos / n_pos) if n_pos > 0 else float("nan")
    ambiguous_ratio = float(n_ambiguous_pos / n_pos) if n_pos > 0 else float("nan")

    boundary_ratio_global = float(crossing.mean())
    ambiguous_ratio_global = float(ambiguous.mean())

    true_n = int((y_pos & views_high).sum())
    hidden_n = int((y_pos & (~views_high)).sum())
    overrated_n = int(((~y_pos) & views_high).sum())
    non_n = int(((~y_pos) & (~views_high)).sum())

    quadrants = {
        "True": {"n": true_n, "pct": float(true_n / n_total)},
        "Hidden": {"n": hidden_n, "pct": float(hidden_n / n_total)},
        "Overrated": {"n": overrated_n, "pct": float(overrated_n / n_total)},
        "Non": {"n": non_n, "pct": float(non_n / n_total)},
    }
    min_quadrant_ok = bool(all(v["n"] >= int(min_quadrant_size) for v in quadrants.values()))

    return {
        "policy": policy_name,
        "n_total": n_total,
        "n_positive": n_pos,
        "positive_ratio": pos_ratio,
        "boundary_ratio": boundary_ratio,
        "ambiguous_ratio": ambiguous_ratio,
        "boundary_ratio_global": boundary_ratio_global,
        "ambiguous_ratio_global": ambiguous_ratio_global,
        "n_boundary_positive": n_boundary_pos,
        "n_ambiguous_positive": n_ambiguous_pos,
        "target_positive_ratio": float(top_pct),
        "min_quadrant_size": int(min_quadrant_size),
        "min_quadrant_ok": min_quadrant_ok,
        "quadrants": quadrants,
    }


def _select_winner(rows: list[dict[str, Any]], top_pct: float) -> dict[str, Any]:
    eligible = [r for r in rows if bool(r["min_quadrant_ok"])]
    if not eligible:
        return {
            "winner_policy": None,
            "reason": "No policy satisfies typology size constraints",
            "selection_rule": (
                "Among min_quadrant_ok=True policies, pick lowest boundary_ratio, then "
                "lowest ambiguous_ratio, then closest positive_ratio to target top_pct"
            ),
        }

    winner = min(
        eligible,
        key=lambda r: (
            float(r["boundary_ratio"]),
            float(r["ambiguous_ratio"]),
            abs(float(r["positive_ratio"]) - float(top_pct)),
            str(r["policy"]),
        ),
    )
    return {
        "winner_policy": str(winner["policy"]),
        "reason": "Lowest boundary/ambiguous ratio under typology constraints",
        "selection_rule": (
            "Among min_quadrant_ok=True policies, pick lowest boundary_ratio, then "
            "lowest ambiguous_ratio, then closest positive_ratio to target top_pct"
        ),
    }


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    mc_seeds = _parse_int_list(args.mc_seeds)

    if int(args.consensus_k) < 1 or int(args.consensus_k) > len(mc_seeds):
        raise ValueError("consensus_k must be in [1, len(mc_seeds)]")

    df_ic = pd.read_parquet(args.ic)
    df_ci = pd.read_parquet(args.ic_ci)
    df_cls = pd.read_parquet(args.cls)
    df_attrs = pd.read_parquet(args.node_attrs)

    require_columns(df_ic, ["node_id", "ic_score_mean", "n_runs"], "ic_scores")
    require_columns(
        df_ci,
        ["node_id", "p_above_top10_threshold", "is_boundary_ci_crossing_threshold"],
        "ic_scores_with_ci",
    )
    require_columns(df_cls, ["node_id", "y_top10"], "classification_labels")
    require_columns(df_attrs, ["node_id", "views"], "node_attributes")

    n_runs_unique = sorted(df_ic["n_runs"].astype(int).unique().tolist())
    if int(args.n_runs_consensus) > 0:
        n_runs_consensus = int(args.n_runs_consensus)
    elif len(n_runs_unique) == 1:
        n_runs_consensus = int(n_runs_unique[0])
    else:
        raise ValueError(
            "ic_scores has multiple n_runs values. Please set --n-runs-consensus explicitly."
        )

    df = df_ic[["node_id", "ic_score_mean", "n_runs"]].merge(
        df_ci[["node_id", "p_above_top10_threshold", "is_boundary_ci_crossing_threshold"]],
        on="node_id",
        how="inner",
    )
    if len(df) != len(df_ic):
        raise ValueError("ic_scores_with_ci node set must match ic_scores")

    df = df.merge(df_cls[["node_id", "y_top10"]], on="node_id", how="left")
    if df["y_top10"].isna().any():
        thresh = float(np.quantile(df["ic_score_mean"].astype(float).to_numpy(), 1.0 - float(args.top_pct)))
        df["y_top10"] = (df["ic_score_mean"].astype(float) >= thresh).astype(int)
    else:
        df["y_top10"] = df["y_top10"].astype(int)

    df = df.merge(df_attrs[["node_id", "views"]], on="node_id", how="inner")
    if len(df) == 0:
        raise ValueError("No overlap between policy inputs and node attributes")

    df = df.sort_values("node_id", kind="mergesort").reset_index(drop=True)
    if int(args.max_nodes) > 0:
        df = df.head(int(args.max_nodes)).copy()

    crossing = (df["is_boundary_ci_crossing_threshold"].astype(int).to_numpy() == 1)
    p_above = df["p_above_top10_threshold"].astype(float).to_numpy()
    ambiguous = (p_above > float(args.stable_neg_prob)) & (p_above < float(args.stable_pos_prob))

    views = df["views"].astype(float).to_numpy()
    views_thresh = float(np.quantile(views, 1.0 - float(args.top_pct)))
    views_high = views >= views_thresh

    y_a = df["y_top10"].astype(int).to_numpy() == 1

    csr = load_csr_npz(args.csr)
    node_to_row = {n: i for i, n in enumerate(csr["node_ids"].astype(str).tolist())}
    node_ids = df["node_id"].astype(str).to_numpy()
    missing = sorted(set(node_ids.tolist()) - set(node_to_row.keys()))
    if missing:
        raise ValueError(f"{len(missing)} nodes missing in CSR mapping")

    rows = np.asarray([node_to_row[n] for n in node_ids], dtype=np.int64)
    degrees = csr["degrees"]
    inv_degrees = np.zeros_like(degrees, dtype=float)
    mask = degrees > 0
    inv_degrees[mask] = 1.0 / degrees[mask].astype(float)

    seed_top_sets: dict[int, set[int]] = {}
    seed_top_counts: dict[str, int] = {}
    for mc_seed in mc_seeds:
        seed_offset = int(mc_seed) * int(args.seed_multiplier)
        means = _simulate_means_for_seed(
            rows=rows,
            indptr=csr["indptr"],
            indices=csr["indices"],
            inv_degrees=inv_degrees,
            n_runs=int(n_runs_consensus),
            seed_offset=seed_offset,
            n_jobs=int(args.n_jobs),
        )
        top_set = _top_decile_set(means, top_pct=float(args.top_pct))
        seed_top_sets[int(mc_seed)] = top_set
        seed_top_counts[str(int(mc_seed))] = int(len(top_set))

    votes = np.zeros(len(df), dtype=np.int32)
    for top_set in seed_top_sets.values():
        if not top_set:
            continue
        idx = np.fromiter(sorted(top_set), dtype=np.int64)
        votes[idx] += 1
    y_b = votes >= int(args.consensus_k)

    target_k = int(y_a.sum())
    if target_k <= 0:
        target_k = int(max(1, np.ceil(float(args.top_pct) * len(df))))
    rank_df = df[["node_id", "ic_score_mean", "p_above_top10_threshold"]].copy()
    rank_df["idx"] = np.arange(len(rank_df), dtype=np.int64)
    rank_df = rank_df.sort_values(
        ["p_above_top10_threshold", "ic_score_mean", "node_id"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    chosen_idx = set(rank_df.head(target_k)["idx"].astype(int).tolist())
    y_c = np.asarray([i in chosen_idx for i in range(len(df))], dtype=bool)

    rows_out = [
        _evaluate_policy(
            policy_name="A_hard_top10",
            y_pos=y_a,
            crossing=crossing,
            ambiguous=ambiguous,
            views_high=views_high,
            min_quadrant_size=int(args.min_quadrant_size),
            top_pct=float(args.top_pct),
        ),
        _evaluate_policy(
            policy_name="B_consensus_top10",
            y_pos=y_b,
            crossing=crossing,
            ambiguous=ambiguous,
            views_high=views_high,
            min_quadrant_size=int(args.min_quadrant_size),
            top_pct=float(args.top_pct),
        ),
        _evaluate_policy(
            policy_name="C_probabilistic_top10",
            y_pos=y_c,
            crossing=crossing,
            ambiguous=ambiguous,
            views_high=views_high,
            min_quadrant_size=int(args.min_quadrant_size),
            top_pct=float(args.top_pct),
        ),
    ]

    winner = _select_winner(rows_out, top_pct=float(args.top_pct))

    label_df = pd.DataFrame(
        {
            "node_id": df["node_id"].astype(str),
            "policy_a": y_a.astype(int),
            "policy_b": y_b.astype(int),
            "policy_c": y_c.astype(int),
            "seed_vote_count": votes.astype(int),
            "p_above_top10_threshold": p_above,
            "is_boundary": crossing.astype(int),
            "is_ambiguous": ambiguous.astype(int),
            "views": views,
            "views_high": views_high.astype(int),
        }
    )

    csv_out = Path(out_dir) / "policy_comparison_summary.csv"
    json_out = Path(out_dir) / "policy_comparison_summary.json"
    labels_out = Path(out_dir) / "policy_labels_abc.parquet"

    pd.DataFrame(rows_out).drop(columns=["quadrants"]).to_csv(csv_out, index=False)
    label_df.to_parquet(labels_out, index=False)

    payload = {
        "timestamp": now_iso(),
        "config": {
            "top_pct": float(args.top_pct),
            "mc_seeds": mc_seeds,
            "seed_multiplier": int(args.seed_multiplier),
            "consensus_k": int(args.consensus_k),
            "n_runs_consensus": int(n_runs_consensus),
            "stable_pos_prob": float(args.stable_pos_prob),
            "stable_neg_prob": float(args.stable_neg_prob),
            "min_quadrant_size": int(args.min_quadrant_size),
            "n_nodes": int(len(df)),
            "max_nodes": int(args.max_nodes),
            "policy_b_definition": "policy_b == (seed_vote_count >= consensus_k)",
        },
        "seed_top10_counts": seed_top_counts,
        "seed_vote_count_distribution": {
            str(k): int(v)
            for k, v in pd.Series(votes).value_counts().sort_index().to_dict().items()
        },
        "policy_b_positive_count": int(y_b.sum()),
        "winner": winner,
        "rows": rows_out,
        "artifacts": {
            "summary_csv": str(csv_out).replace("\\", "/"),
            "labels_parquet": str(labels_out).replace("\\", "/"),
        },
    }

    with json_out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[OK] Wrote policy summary CSV: {csv_out}")
    print(f"[OK] Wrote policy summary JSON: {json_out}")
    print(f"[OK] Wrote policy labels parquet: {labels_out}")
    print(f"[OK] Winner policy: {winner.get('winner_policy')}")


if __name__ == "__main__":
    main()
