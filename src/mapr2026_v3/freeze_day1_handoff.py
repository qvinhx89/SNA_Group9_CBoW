"""MAPR2026 v3 — Freeze split + versioned Day1 handoff package.

Owner: Person 1

Purpose
-------
Create a versioned handoff snapshot with checksums so downstream consumers use
the exact same artifacts and do not accidentally drift split/test sets.

Outputs
-------
- outputs/day1_benchmark/split_freeze_manifest.json
- outputs/handoffs/person1_day1_<version_tag>/manifest.json
- Versioned copies of required artifacts under the handoff folder
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from _shared import PATHS, ensure_dir, ensure_parent, now_iso


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Freeze split + version Day1 handoff")
    p.add_argument("--version-tag", default=now_iso().replace(":", "").replace("-", ""))
    p.add_argument("--handoff-root", default="outputs/handoffs")
    p.add_argument("--split-manifest", default=f"{PATHS.day1_dir}/split_freeze_manifest.json")
    p.add_argument("--quality-report", default=f"{PATHS.day1_dir}/quality_gate_report.json")
    p.add_argument(
        "--quality-mode",
        choices=["final", "provisional"],
        default="final",
        help="final=enforce hard quality gates; provisional=record gate status but continue",
    )
    p.add_argument("--min-jaccard-mean", type=float, default=0.85)
    p.add_argument("--min-jaccard-min", type=float, default=0.80)
    p.add_argument("--min-cv-score", type=float, default=0.30)
    return p.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def collect_required_paths() -> list[Path]:
    required = [
        Path(PATHS.csr_npz),
        Path(PATHS.ic_scores),
        Path("data/processed/ic_scores_primary_with_ci.parquet"),
        Path(PATHS.regression_targets),
        Path(PATHS.classification_labels),
        Path(PATHS.split_masks),
        Path(f"{PATHS.day1_dir}/ic_runtime_benchmark.json"),
        Path(f"{PATHS.day1_dir}/one_hop_correlation.json"),
        Path(f"{PATHS.day1_dir}/ic_pilot_diagnostics.json"),
        Path(f"{PATHS.day1_dir}/ic_label_stability.json"),
        Path(f"{PATHS.day1_dir}/ic_label_uncertainty.json"),
        Path(f"{PATHS.day1_dir}/quality_gate_report.json"),
        Path("docs/day1_decisions.md"),
        Path("docs/day1_handoff_brief_for_person2_3.md"),
        Path("docs/decision_request_p1_stage4_v3c.md"),
        Path("docs/m0_decisions.md"),
    ]

    optional_if_exists = [
        # Optional I-A branch artifacts (v3.1)
        Path("outputs/mapr2026_v3_results/ic_scores_ia.parquet"),
        Path("data/processed/regression_targets_ia.parquet"),
        Path("data/processed/classification_labels_consensus.parquet"),
        Path("data/processed/ic_scores_primary_with_ci_consensus.parquet"),
        Path(f"{PATHS.day1_dir}/ic_label_uncertainty_consensus.json"),
        Path(f"{PATHS.day1_dir}/ic_regression_stability.json"),
        Path(f"{PATHS.day1_dir}/stability_explanation.json"),
        Path("outputs/ic_feasibility/phase1_community_overlap.json"),
        Path("outputs/ic_feasibility/phase2_threshold_analysis.json"),
        Path("outputs/day1_benchmark/policy_compare/policy_comparison_summary.csv"),
        Path("outputs/day1_benchmark/policy_compare/policy_comparison_summary.json"),
        Path("outputs/day1_benchmark/policy_compare/classification_labels_consensus_report.json"),
        Path("outputs/mapr2026_v3_results/runtime_breakdown.csv"),
    ]
    required.extend([p for p in optional_if_exists if p.exists()])

    missing = [p for p in required if not p.exists()]
    if missing:
        missing_str = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(f"Cannot freeze handoff. Missing required files:\n{missing_str}")
    return required


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing JSON file: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_quality_gates(args: argparse.Namespace) -> dict[str, Any]:
    pilot_path = Path(f"{PATHS.day1_dir}/ic_pilot_diagnostics.json")
    stability_path = Path(f"{PATHS.day1_dir}/ic_label_stability.json")
    uncertainty_path = Path(f"{PATHS.day1_dir}/ic_label_uncertainty.json")

    pilot = _read_json(pilot_path)
    st = _read_json(stability_path)
    un = _read_json(uncertainty_path)

    cv_score = float(pilot.get("summary", {}).get("cv_score", float("nan")))
    jaccard_mean = float(st.get("summary", {}).get("jaccard_mean", float("nan")))
    jaccard_min = float(st.get("summary", {}).get("jaccard_min", float("nan")))

    checks = {
        "pilot_cv_score_pass": bool(cv_score > float(args.min_cv_score)),
        "stability_jaccard_mean_pass": bool(jaccard_mean >= float(args.min_jaccard_mean)),
        "stability_jaccard_min_pass": bool(jaccard_min >= float(args.min_jaccard_min)),
    }
    pass_all = bool(all(checks.values()))

    report = {
        "timestamp": now_iso(),
        "quality_mode": str(args.quality_mode),
        "thresholds": {
            "min_cv_score": float(args.min_cv_score),
            "min_jaccard_mean": float(args.min_jaccard_mean),
            "min_jaccard_min": float(args.min_jaccard_min),
        },
        "observed": {
            "cv_score": cv_score,
            "jaccard_mean": jaccard_mean,
            "jaccard_min": jaccard_min,
            "uncertainty_boundary_ratio": float(un.get("summary", {}).get("boundary_ratio", float("nan"))),
            "uncertainty_ambiguous_ratio": float(un.get("summary", {}).get("ambiguous_ratio", float("nan"))),
        },
        "checks": checks,
        "pass_all": pass_all,
        "required_artifacts": {
            "ic_pilot_diagnostics": str(pilot_path).replace("\\", "/"),
            "ic_label_stability": str(stability_path).replace("\\", "/"),
            "ic_label_uncertainty": str(uncertainty_path).replace("\\", "/"),
        },
    }

    _write_json(Path(args.quality_report), report)

    if str(args.quality_mode) == "final" and not pass_all:
        failed = [k for k, v in checks.items() if not v]
        raise ValueError(
            "Quality gate FAILED in final mode. "
            f"Failed checks: {failed}. "
            f"See {args.quality_report}."
        )

    return report


def main() -> None:
    args = parse_args()

    gate_report = evaluate_quality_gates(args)
    print(
        "[OK] Quality gate report: "
        f"{args.quality_report} (pass_all={gate_report['pass_all']}, mode={args.quality_mode})"
    )

    required = collect_required_paths()
    handoff_root = ensure_dir(args.handoff_root)
    handoff_dir = ensure_dir(handoff_root / f"person1_day1_{args.version_tag}")

    files_manifest: list[dict[str, Any]] = []
    for src in required:
        dst = handoff_dir / src
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

        files_manifest.append(
            {
                "path": str(src).replace("\\", "/"),
                "size_bytes": int(src.stat().st_size),
                "sha256": sha256_file(src),
            }
        )

    split_path = Path(PATHS.split_masks)
    split_freeze = {
        "timestamp": now_iso(),
        "split_path": str(split_path).replace("\\", "/"),
        "sha256": sha256_file(split_path),
        "size_bytes": int(split_path.stat().st_size),
        "locked_rule": {
            "test_frac": 0.20,
            "stratify_by": "degree_quintile (q=5, duplicates='drop')",
            "seed": 42,
        },
        "consumer_rule": "Consumers must load this split artifact; no local re-splitting.",
        "versioned_handoff_dir": str(handoff_dir).replace("\\", "/"),
    }
    _write_json(Path(args.split_manifest), split_freeze)

    handoff_manifest = {
        "timestamp": now_iso(),
        "handoff_version": f"person1_day1_{args.version_tag}",
        "handoff_dir": str(handoff_dir).replace("\\", "/"),
        "quality_mode": str(args.quality_mode),
        "quality_gate_pass_all": bool(gate_report.get("pass_all", False)),
        "n_files": len(files_manifest),
        "files": files_manifest,
        "notes": [
            "This package is immutable for downstream reproducibility.",
            "If rerun is needed, generate a new version tag instead of overwriting.",
        ],
    }
    _write_json(handoff_dir / "manifest.json", handoff_manifest)

    print(f"[OK] Wrote split freeze manifest: {args.split_manifest}")
    print(f"[OK] Wrote versioned handoff package: {handoff_dir}")


if __name__ == "__main__":
    main()
