"""MAPR2026 v3 - Export consensus binary labels as supplementary artifact.

Owner: Person 1

Purpose
-------
Create the supplementary consensus label artifact from winner policy B without
replacing canonical classification_labels.parquet.

Definition (locked)
-------------------
y_top10_consensus == policy_b == (seed_vote_count >= consensus_k)

Output
------
- data/processed/classification_labels_consensus.parquet
- outputs/day1_benchmark/policy_compare/classification_labels_consensus_report.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _shared import ensure_parent, now_iso, require_columns, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export supplementary consensus labels")
    p.add_argument(
        "--policy-labels",
        default="outputs/day1_benchmark/policy_compare/policy_labels_abc.parquet",
    )
    p.add_argument(
        "--out",
        default="data/processed/classification_labels_consensus.parquet",
    )
    p.add_argument(
        "--report",
        default="outputs/day1_benchmark/policy_compare/classification_labels_consensus_report.json",
    )
    p.add_argument(
        "--consensus-k",
        type=int,
        default=2,
        help="Positive if vote_count >= consensus-k",
    )
    p.add_argument(
        "--uncertain-vote-count",
        type=int,
        default=1,
        help="Mark uncertain when vote_count equals this value",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    policy_path = Path(args.policy_labels)
    if not policy_path.exists():
        raise FileNotFoundError(f"Missing policy labels artifact: {policy_path}")

    df = pd.read_parquet(policy_path)
    require_columns(
        df,
        ["node_id", "policy_b", "seed_vote_count", "p_above_top10_threshold"],
        "policy_labels_abc",
    )

    votes = df["seed_vote_count"].astype(int)
    y_from_votes = (votes >= int(args.consensus_k)).astype(int)
    y_policy_b = df["policy_b"].astype(int)

    if not bool(np.array_equal(y_from_votes.to_numpy(), y_policy_b.to_numpy())):
        raise ValueError(
            "Inconsistent policy artifact: policy_b != (seed_vote_count >= consensus_k). "
            "Please rerun ic_policy_compare.py to regenerate policy labels."
        )

    y_consensus = y_policy_b
    is_uncertain = (votes == int(args.uncertain_vote_count)).astype(int)

    out_df = pd.DataFrame(
        {
            "node_id": df["node_id"].astype(str),
            "y_top10_consensus": y_consensus,
            "is_uncertain": is_uncertain,
            "vote_count": votes.astype(int),
            "p_above_top10_threshold": df["p_above_top10_threshold"].astype(float),
        }
    )

    out_path = Path(args.out)
    ensure_parent(out_path)
    out_df.to_parquet(out_path, index=False)

    report = {
        "timestamp": now_iso(),
        "policy_source": str(policy_path).replace("\\", "/"),
        "output": str(out_path).replace("\\", "/"),
        "config": {
            "consensus_k": int(args.consensus_k),
            "uncertain_vote_count": int(args.uncertain_vote_count),
            "definition": "y_top10_consensus == policy_b == (seed_vote_count >= consensus_k)",
            "vote_count_source": "seed_vote_count",
        },
        "summary": {
            "n_nodes": int(len(out_df)),
            "n_positive": int(out_df["y_top10_consensus"].sum()),
            "positive_ratio": float(out_df["y_top10_consensus"].mean()),
            "n_uncertain": int(out_df["is_uncertain"].sum()),
            "uncertain_ratio": float(out_df["is_uncertain"].mean()),
            "vote_count_distribution": {
                str(k): int(v)
                for k, v in out_df["vote_count"].value_counts().sort_index().to_dict().items()
            },
        },
    }
    write_json(args.report, report)

    print(f"[OK] Wrote consensus labels: {out_path}")
    print(f"[OK] Wrote consensus report: {Path(args.report)}")


if __name__ == "__main__":
    main()
