"""MAPR2026 v3 — Regression stability report from sweep outputs.

Owner: Person 1

Purpose
-------
Summarize rank-stability evidence for regression target across n_runs settings
using already generated sweep artifacts.

Output
------
- outputs/day1_benchmark/ic_regression_stability.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from _shared import PATHS, now_iso, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build regression stability report from sweep summary")
    p.add_argument(
        "--summary-csv",
        default=f"{PATHS.day1_dir}/stability_sweep/stability_sweep_summary.csv",
    )
    p.add_argument(
        "--summary-json",
        default=f"{PATHS.day1_dir}/stability_sweep/stability_sweep_summary.json",
    )
    p.add_argument(
        "--out",
        default=f"{PATHS.day1_dir}/ic_regression_stability.json",
    )
    p.add_argument(
        "--spearman-threshold",
        type=float,
        default=0.90,
        help="Pass threshold for spearman_mean",
    )
    return p.parse_args()


def _try_read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    csv_path = Path(args.summary_csv)
    json_path = Path(args.summary_json)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing sweep summary CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    required = ["n_runs", "spearman_mean", "spearman_min", "runtime_sec"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Sweep summary CSV missing columns: {missing}")

    df = df.sort_values("n_runs").reset_index(drop=True)
    threshold = float(args.spearman_threshold)
    df["spearman_pass_threshold"] = df["spearman_mean"].astype(float) >= threshold

    any_pass = bool(df["spearman_pass_threshold"].any())
    first_pass = None
    if any_pass:
        row = df[df["spearman_pass_threshold"]].iloc[0]
        first_pass = {
            "n_runs": int(row["n_runs"]),
            "spearman_mean": float(row["spearman_mean"]),
            "spearman_min": float(row["spearman_min"]),
            "runtime_sec": float(row["runtime_sec"]),
        }

    best = df.sort_values(["spearman_mean", "spearman_min"], ascending=False).iloc[0]
    sweep_meta = _try_read_json(json_path)

    payload = {
        "timestamp": now_iso(),
        "source": {
            "summary_csv": str(csv_path).replace("\\", "/"),
            "summary_json": str(json_path).replace("\\", "/"),
            "summary_json_exists": bool(sweep_meta is not None),
        },
        "thresholds": {
            "spearman_mean_min": threshold,
        },
        "rows": [
            {
                "n_runs": int(r["n_runs"]),
                "spearman_mean": float(r["spearman_mean"]),
                "spearman_min": float(r["spearman_min"]),
                "runtime_sec": float(r["runtime_sec"]),
                "spearman_pass_threshold": bool(r["spearman_pass_threshold"]),
            }
            for _, r in df.iterrows()
        ],
        "summary": {
            "any_pass": any_pass,
            "selected_first_pass": first_pass,
            "best_by_spearman": {
                "n_runs": int(best["n_runs"]),
                "spearman_mean": float(best["spearman_mean"]),
                "spearman_min": float(best["spearman_min"]),
                "runtime_sec": float(best["runtime_sec"]),
            },
            "recommendation": (
                "Regression target can be treated as more stable than binary labels; "
                "continue using regression as primary objective and binary labels as supplementary."
            ),
        },
    }

    write_json(args.out, payload)
    print(f"[OK] Wrote regression stability report: {Path(args.out)}")


if __name__ == "__main__":
    main()
