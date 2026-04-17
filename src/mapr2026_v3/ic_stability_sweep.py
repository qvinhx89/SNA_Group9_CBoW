"""MAPR2026 v3 — P1 stability sweep with evidence logs.

Owner: Person 1

Purpose
-------
Run a controlled sweep over n_runs to evaluate label stability under fixed
labeled node set, fixed split, and fixed MC seed protocol.

Evidence outputs (paper-friendly)
-------------------------------
- outputs/day1_benchmark/stability_sweep/stability_sweep_events.jsonl
- outputs/day1_benchmark/stability_sweep/stability_sweep.log
- outputs/day1_benchmark/stability_sweep/n_runs_<N>.json (per setting)
- outputs/day1_benchmark/stability_sweep/stability_sweep_summary.csv
- outputs/day1_benchmark/stability_sweep/stability_sweep_summary.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from _shared import PATHS, ensure_dir, load_csr_npz, now_iso, require_columns
from ic_labels_primary import _compute_views_strength_aligned, _simulate_ic_node_summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P1 stability sweep with audit logs")
    p.add_argument("--csr", default=PATHS.csr_npz)
    p.add_argument("--ic", default=PATHS.ic_scores)
    p.add_argument("--split", default=PATHS.split_masks)
    p.add_argument("--out-dir", default="outputs/day1_benchmark/stability_sweep")
    p.add_argument("--run-grid", default="150,300,500,800,1200")
    p.add_argument("--mc-seeds", default="0,1,2")
    p.add_argument("--seed-multiplier", type=int, default=10000)
    p.add_argument("--top-pct", type=float, default=0.10)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--min-jaccard-mean", type=float, default=0.85)
    p.add_argument("--min-jaccard-min", type=float, default=0.80)

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


def _parse_int_list(raw: str) -> list[int]:
    out: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            out.append(int(token))
    if not out:
        raise ValueError("List argument cannot be empty")
    return out


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _append_log(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def _print_and_log(text_log_path: Path, line: str) -> None:
    print(line)
    _append_log(text_log_path, line)


def _init_summary_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "n_runs",
                "jaccard_mean",
                "jaccard_min",
                "spearman_mean",
                "spearman_min",
                "runtime_sec",
                "pass_plan_threshold",
            ],
        )
        writer.writeheader()


def _append_summary_csv_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "n_runs",
                "jaccard_mean",
                "jaccard_min",
                "spearman_mean",
                "spearman_min",
                "runtime_sec",
                "pass_plan_threshold",
            ],
        )
        writer.writerow(row)


def _write_summary_json(
    path: Path,
    run_grid: list[int],
    mc_seeds: list[int],
    args: argparse.Namespace,
    n_labeled_nodes: int,
    rows: list[dict[str, Any]],
    events_path: Path,
    text_log_path: Path,
    summary_csv: Path,
    *,
    interrupted: bool,
) -> dict[str, Any]:
    pass_rows = [r for r in rows if bool(r.get("pass_plan_threshold"))]
    best_by_jaccard = max(rows, key=lambda r: (r["jaccard_mean"], r["jaccard_min"])) if rows else None
    selected = pass_rows[0] if pass_rows else None

    payload = {
        "timestamp": now_iso(),
        "interrupted": bool(interrupted),
        "config": {
            "run_grid": run_grid,
            "mc_seeds": mc_seeds,
            "seed_multiplier": int(args.seed_multiplier),
            "top_pct": float(args.top_pct),
            "n_jobs": int(args.n_jobs),
            "n_labeled_nodes": int(n_labeled_nodes),
            "max_nodes": int(args.max_nodes),
            "min_jaccard_mean": float(args.min_jaccard_mean),
            "min_jaccard_min": float(args.min_jaccard_min),
            "p_model": str(args.p_model),
            "hybrid_gamma": float(args.hybrid_gamma),
            "views_col": str(args.views_col),
            "views_q_low": float(args.views_q_low),
            "views_q_high": float(args.views_q_high),
        },
        "rows": rows,
        "selected_first_pass": selected,
        "best_by_jaccard": best_by_jaccard,
        "any_pass": bool(len(pass_rows) > 0),
        "artifacts": {
            "events_jsonl": str(events_path).replace("\\", "/"),
            "text_log": str(text_log_path).replace("\\", "/"),
            "summary_csv": str(summary_csv).replace("\\", "/"),
            "summary_json": str(path).replace("\\", "/"),
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


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


def main() -> None:
    args = parse_args()

    run_grid = _parse_int_list(args.run_grid)
    mc_seeds = _parse_int_list(args.mc_seeds)
    out_dir = ensure_dir(args.out_dir)

    events_path = Path(out_dir) / "stability_sweep_events.jsonl"
    text_log_path = Path(out_dir) / "stability_sweep.log"
    summary_csv = Path(out_dir) / "stability_sweep_summary.csv"
    summary_json = Path(out_dir) / "stability_sweep_summary.json"

    # Reset logs per run for clean evidence package.
    for p in [events_path, text_log_path, summary_csv, summary_json]:
        if p.exists():
            p.unlink()

    _init_summary_csv(summary_csv)

    csr = load_csr_npz(args.csr)
    df_ic = pd.read_parquet(args.ic)
    require_columns(df_ic, ["node_id", "ic_score_mean"], "ic_scores")

    df_split = pd.read_parquet(args.split)
    require_columns(df_split, ["node_id", "split"], "split_masks")

    labeled_ids = df_ic["node_id"].astype(str).to_numpy()
    split_ids = set(df_split["node_id"].astype(str).tolist())
    labeled_set = set(labeled_ids.tolist())
    if labeled_set != split_ids:
        raise ValueError(
            "split_masks node set does not match ic_scores node set. "
            "Sweep requires fixed labeled set and split contract."
        )

    node_ids = csr["node_ids"].astype(str)
    node_to_row = {n: i for i, n in enumerate(node_ids.tolist())}
    missing = sorted(labeled_set - set(node_to_row.keys()))
    if missing:
        raise ValueError(f"{len(missing)} labeled nodes not found in CSR mapping")

    rows = np.asarray([node_to_row[n] for n in labeled_ids], dtype=np.int64)
    if int(args.max_nodes) > 0:
        rows = rows[: int(args.max_nodes)]

    degrees = csr["degrees"]
    inv_degrees = np.zeros_like(degrees, dtype=float)
    mask = degrees > 0
    inv_degrees[mask] = 1.0 / degrees[mask].astype(float)

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

    start_event = {
        "timestamp": now_iso(),
        "event": "sweep_started",
        "config": {
            "run_grid": run_grid,
            "mc_seeds": mc_seeds,
            "seed_multiplier": int(args.seed_multiplier),
            "top_pct": float(args.top_pct),
            "n_jobs": int(args.n_jobs),
            "min_jaccard_mean": float(args.min_jaccard_mean),
            "min_jaccard_min": float(args.min_jaccard_min),
            "n_labeled_nodes": int(len(rows)),
            "max_nodes": int(args.max_nodes),
            "worker_seed_rule": "mc_seed * seed_multiplier + node_row",
            "p_model": str(args.p_model),
            "hybrid_gamma": float(args.hybrid_gamma),
            "views_col": str(args.views_col),
            "views_q_low": float(args.views_q_low),
            "views_q_high": float(args.views_q_high),
            "split_sha256": _sha256_file(Path(args.split)),
            "ic_scores_sha256": _sha256_file(Path(args.ic)),
        },
    }
    _append_jsonl(events_path, start_event)
    _print_and_log(
        text_log_path,
        f"[{now_iso()}] sweep_started run_grid={run_grid} nodes={len(rows)} p_model={args.p_model} gamma={args.hybrid_gamma}",
    )

    rows_out: list[dict[str, Any]] = []
    n_labeled_nodes = int(len(rows))

    interrupted = False
    try:
        for n_runs in run_grid:
            run_start = time.perf_counter()
            _append_jsonl(
                events_path,
                {
                    "timestamp": now_iso(),
                    "event": "run_started",
                    "n_runs": int(n_runs),
                    "mc_seeds": mc_seeds,
                },
            )
            _print_and_log(text_log_path, f"[{now_iso()}] run_started n_runs={n_runs}")

            scores_by_seed: dict[int, np.ndarray] = {}
            seed_runtime_sec: dict[str, float] = {}

            for mc_seed in mc_seeds:
                seed_offset = int(mc_seed) * int(args.seed_multiplier)
                _append_jsonl(
                    events_path,
                    {
                        "timestamp": now_iso(),
                        "event": "seed_started",
                        "n_runs": int(n_runs),
                        "mc_seed": int(mc_seed),
                        "seed_offset": int(seed_offset),
                    },
                )
                _print_and_log(text_log_path, f"[{now_iso()}] seed_started n_runs={n_runs} mc_seed={mc_seed}")

                t0 = time.perf_counter()
                scores_by_seed[int(mc_seed)] = _simulate_means_for_seed(
                    rows=rows,
                    indptr=csr["indptr"],
                    indices=csr["indices"],
                    inv_degrees=inv_degrees,
                    n_runs=int(n_runs),
                    seed_offset=seed_offset,
                    n_jobs=int(args.n_jobs),
                    p_model=str(args.p_model),
                    sender_strength=sender_strength,
                    hybrid_gamma=float(args.hybrid_gamma),
                )
                seed_elapsed = float(time.perf_counter() - t0)
                seed_runtime_sec[str(mc_seed)] = seed_elapsed
                _append_jsonl(
                    events_path,
                    {
                        "timestamp": now_iso(),
                        "event": "seed_completed",
                        "n_runs": int(n_runs),
                        "mc_seed": int(mc_seed),
                        "runtime_sec": seed_elapsed,
                    },
                )
                _print_and_log(
                    text_log_path,
                    f"[{now_iso()}] seed_completed n_runs={n_runs} mc_seed={mc_seed} runtime_sec={seed_elapsed:.2f}",
                )

            pairwise: list[dict[str, float | int]] = []
            jaccards: list[float] = []
            spearmans: list[float] = []

            for i in range(len(mc_seeds)):
                for j in range(i + 1, len(mc_seeds)):
                    s_i = int(mc_seeds[i])
                    s_j = int(mc_seeds[j])
                    top_i = _top_decile_set(scores_by_seed[s_i], top_pct=float(args.top_pct))
                    top_j = _top_decile_set(scores_by_seed[s_j], top_pct=float(args.top_pct))
                    jac = _jaccard(top_i, top_j)
                    rho = _spearman(scores_by_seed[s_i], scores_by_seed[s_j])
                    pairwise.append(
                        {
                            "seed_i": s_i,
                            "seed_j": s_j,
                            "jaccard_top_decile": float(jac),
                            "spearman_rank": float(rho),
                        }
                    )
                    jaccards.append(float(jac))
                    spearmans.append(float(rho))

            runtime_sec = float(time.perf_counter() - run_start)
            j_mean = float(np.mean(jaccards))
            j_min = float(np.min(jaccards))
            s_mean = float(np.mean(spearmans))
            s_min = float(np.min(spearmans))
            pass_plan = bool(
                (j_mean >= float(args.min_jaccard_mean)) and (j_min >= float(args.min_jaccard_min))
            )

            run_payload = {
                "timestamp": now_iso(),
                "n_runs": int(n_runs),
                "n_labeled_nodes": int(len(rows)),
                "mc_seeds": mc_seeds,
                "seed_runtime_sec": seed_runtime_sec,
                "runtime_sec": runtime_sec,
                "pairwise": pairwise,
                "summary": {
                    "jaccard_mean": j_mean,
                    "jaccard_min": j_min,
                    "spearman_mean": s_mean,
                    "spearman_min": s_min,
                    "pass_plan_threshold": pass_plan,
                    "threshold_jaccard_mean": float(args.min_jaccard_mean),
                    "threshold_jaccard_min": float(args.min_jaccard_min),
                },
            }

            per_run_json = Path(out_dir) / f"n_runs_{int(n_runs)}.json"
            with per_run_json.open("w", encoding="utf-8") as f:
                json.dump(run_payload, f, indent=2, ensure_ascii=False)

            row_out = {
                "n_runs": int(n_runs),
                "jaccard_mean": j_mean,
                "jaccard_min": j_min,
                "spearman_mean": s_mean,
                "spearman_min": s_min,
                "runtime_sec": runtime_sec,
                "pass_plan_threshold": pass_plan,
            }
            rows_out.append(row_out)

            # Write incremental summary artifacts so progress is visible and results are durable.
            _append_summary_csv_row(summary_csv, row_out)
            _write_summary_json(
                summary_json,
                run_grid=run_grid,
                mc_seeds=mc_seeds,
                args=args,
                n_labeled_nodes=n_labeled_nodes,
                rows=rows_out,
                events_path=events_path,
                text_log_path=text_log_path,
                summary_csv=summary_csv,
                interrupted=False,
            )

            _append_jsonl(
                events_path,
                {
                    "timestamp": now_iso(),
                    "event": "run_completed",
                    "n_runs": int(n_runs),
                    "jaccard_mean": j_mean,
                    "jaccard_min": j_min,
                    "spearman_mean": s_mean,
                    "runtime_sec": runtime_sec,
                    "pass_plan_threshold": pass_plan,
                },
            )
            _print_and_log(
                text_log_path,
                (
                    f"[{now_iso()}] run_completed n_runs={n_runs} "
                    f"jaccard_mean={j_mean:.6f} jaccard_min={j_min:.6f} "
                    f"spearman_mean={s_mean:.6f} runtime_sec={runtime_sec:.2f} pass={pass_plan}"
                ),
            )

    except KeyboardInterrupt:
        interrupted = True
        _append_jsonl(
            events_path,
            {
                "timestamp": now_iso(),
                "event": "sweep_interrupted",
                "completed_runs": int(len(rows_out)),
                "planned_runs": int(len(run_grid)),
            },
        )
        _print_and_log(
            text_log_path,
            f"[{now_iso()}] sweep_interrupted completed_runs={len(rows_out)}/{len(run_grid)}",
        )
        _write_summary_json(
            summary_json,
            run_grid=run_grid,
            mc_seeds=mc_seeds,
            args=args,
            n_labeled_nodes=n_labeled_nodes,
            rows=rows_out,
            events_path=events_path,
            text_log_path=text_log_path,
            summary_csv=summary_csv,
            interrupted=True,
        )
        return

    # Final summary write (ensures best_by_jaccard / selected are up-to-date and marked not interrupted).
    _write_summary_json(
        summary_json,
        run_grid=run_grid,
        mc_seeds=mc_seeds,
        args=args,
        n_labeled_nodes=n_labeled_nodes,
        rows=rows_out,
        events_path=events_path,
        text_log_path=text_log_path,
        summary_csv=summary_csv,
        interrupted=False,
    )

    pass_rows = [r for r in rows_out if bool(r.get("pass_plan_threshold"))]
    _append_jsonl(
        events_path,
        {
            "timestamp": now_iso(),
            "event": "sweep_completed",
            "any_pass": bool(len(pass_rows) > 0),
            "summary_csv": str(summary_csv).replace("\\", "/"),
            "summary_json": str(summary_json).replace("\\", "/"),
        },
    )
    _print_and_log(
        text_log_path,
        f"[{now_iso()}] sweep_completed any_pass={bool(len(pass_rows) > 0)} summary={summary_csv}",
    )

    print(f"[OK] Wrote sweep summary CSV: {summary_csv}")
    print(f"[OK] Wrote sweep summary JSON: {summary_json}")
    print(f"[OK] Wrote evidence logs: {events_path}, {text_log_path}")


if __name__ == "__main__":
    main()
