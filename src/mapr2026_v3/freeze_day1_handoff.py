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
        Path(f"{PATHS.day1_dir}/ic_label_stability.json"),
        Path(f"{PATHS.day1_dir}/ic_label_uncertainty.json"),
        Path("docs/day1_decisions.md"),
        Path("docs/m0_decisions.md"),
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        missing_str = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(f"Cannot freeze handoff. Missing required files:\n{missing_str}")
    return required


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()

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
