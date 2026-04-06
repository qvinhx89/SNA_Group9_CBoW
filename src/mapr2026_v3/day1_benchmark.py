"""MAPR2026 v3 — Day-1 decisions.

Owner: Person 1 (IC core)

Inputs
------
- data/processed/graph_csr.npz

Outputs
-------
- outputs/day1_benchmark/ic_runtime_benchmark.json
- outputs/day1_benchmark/one_hop_correlation.json

This file is a scaffold. It supports --dry-run to emit placeholder JSON files
with the required keys (so teammates can integrate downstream code early).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _shared import PATHS, ensure_dir, load_csr_npz, now_iso, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MAPR2026 v3 Day-1 benchmark + one-hop correlation (scaffold)")
    p.add_argument("--csr", default=PATHS.csr_npz)
    p.add_argument("--out-dir", default=PATHS.day1_dir)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)

    if args.dry_run:
        write_json(
            out_dir / "ic_runtime_benchmark.json",
            {
                "timestamp": now_iso(),
                "dry_run": True,
                "per_sim_ms": None,
                "projected_total_hours": None,
                "decision": {"n_seeds": None, "n_runs": None},
                "note": "Scaffold placeholder. Implement benchmark_ic_runtime per MAPR2026 v3.",
            },
        )
        write_json(
            out_dir / "one_hop_correlation.json",
            {
                "timestamp": now_iso(),
                "dry_run": True,
                "spearman_rho": None,
                "p_value": None,
                "decision_branch": None,
                "note": "Scaffold placeholder. Implement IC pilot + one-hop baseline correlation.",
            },
        )
        print(f"[OK] Wrote dry-run placeholders to: {out_dir}")
        return

    # Real mode: require CSR and actual implementations.
    _ = load_csr_npz(Path(args.csr))
    raise NotImplementedError(
        "Implement runtime benchmark + one-hop correlation using weighted-cascade IC on CSR. "
        "Run with --dry-run to generate placeholder artifacts."
    )


if __name__ == "__main__":
    main()
