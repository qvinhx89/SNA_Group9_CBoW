"""Verify a versioned MAPR2026 Day1 handoff package.

Checks:
- Every file listed in manifest exists inside the handoff folder
- SHA256 + size match the manifest
- Quick load sanity for key parquet artifacts (A0 + optional I-A)

Usage (PowerShell):
  .venv/Scripts/python.exe scripts/verify_handoff_package.py \
    --manifest outputs/handoffs/person1_day1_<tag>/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _check_manifest_files(manifest_path: Path) -> tuple[Path, list[str]]:
    manifest = _read_json(manifest_path)
    handoff_dir = Path(manifest["handoff_dir"])
    errors: list[str] = []

    if not handoff_dir.exists():
        errors.append(f"handoff_dir does not exist: {handoff_dir}")
        return handoff_dir, errors

    files = manifest.get("files", [])
    if not isinstance(files, list) or not files:
        errors.append("manifest.files missing or empty")
        return handoff_dir, errors

    for item in files:
        rel_path = Path(item["path"])
        expected_size = int(item["size_bytes"])
        expected_sha256 = str(item["sha256"])

        frozen_path = handoff_dir / rel_path
        if not frozen_path.exists():
            errors.append(f"missing in handoff: {rel_path}")
            continue

        actual_size = int(frozen_path.stat().st_size)
        if actual_size != expected_size:
            errors.append(f"size mismatch: {rel_path} (manifest={expected_size}, actual={actual_size})")

        actual_sha256 = _sha256_file(frozen_path)
        if actual_sha256 != expected_sha256:
            errors.append(f"sha256 mismatch: {rel_path} (manifest={expected_sha256}, actual={actual_sha256})")

    return handoff_dir, errors


def _parquet_quickcheck(handoff_dir: Path, rel_path: str, key_cols: list[str] | None = None) -> str:
    import pandas as pd  # local import

    path = handoff_dir / rel_path
    if not path.exists():
        return f"SKIP {rel_path} (missing)"

    df = pd.read_parquet(path)
    n_rows, n_cols = df.shape

    msg = f"OK {rel_path} rows={n_rows} cols={n_cols}"
    if key_cols:
        missing = [c for c in key_cols if c not in df.columns]
        if missing:
            msg += f" | missing_cols={missing}"
        else:
            msg += " | key_cols=present"

    if "node_id" in df.columns:
        n_dupe = int(df["node_id"].duplicated().sum())
        if n_dupe:
            msg += f" | DUPLICATE node_id={n_dupe}"

    return msg


def _npz_quickcheck(handoff_dir: Path, rel_path: str) -> str:
    import numpy as np  # local import

    path = handoff_dir / rel_path
    if not path.exists():
        return f"SKIP {rel_path} (missing)"

    # We only validate the archive loads; downstream code loads CSR via its own loader.
    with np.load(path, allow_pickle=False) as z:
        keys = sorted(list(z.keys()))
    return f"OK {rel_path} npz_keys={keys[:8]}{'...' if len(keys) > 8 else ''}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify a MAPR2026 handoff package")
    p.add_argument("--manifest", required=True, help="Path to handoff manifest.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)

    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    handoff_dir, errors = _check_manifest_files(manifest_path)

    print(f"[VERIFY] manifest={manifest_path}")
    print(f"[VERIFY] handoff_dir={handoff_dir}")

    if errors:
        print(f"[FAIL] {len(errors)} issues")
        for e in errors[:50]:
            print(f"  - {e}")
        raise SystemExit(2)

    print("[OK] All manifest files present with matching size+sha256")

    # Quick-load sanity for core artifacts (from within handoff_dir)
    print(_npz_quickcheck(handoff_dir, "data/processed/graph_csr.npz"))

    print(_parquet_quickcheck(handoff_dir, "data/processed/ic_scores_primary.parquet", key_cols=["node_id"]))
    print(_parquet_quickcheck(handoff_dir, "data/processed/ic_scores_primary_with_ci.parquet", key_cols=["node_id"]))
    print(_parquet_quickcheck(handoff_dir, "data/processed/regression_targets.parquet", key_cols=["node_id"]))
    print(_parquet_quickcheck(handoff_dir, "data/processed/classification_labels.parquet", key_cols=["node_id"]))
    print(_parquet_quickcheck(handoff_dir, "data/processed/split_masks.parquet"))

    # Optional I-A branch artifacts
    print(_parquet_quickcheck(handoff_dir, "outputs/mapr2026_v3_results/ic_scores_ia.parquet", key_cols=["node_id"]))
    print(_parquet_quickcheck(handoff_dir, "data/processed/regression_targets_ia.parquet", key_cols=["node_id"]))

    print("[OK] Quick-load checks completed")


if __name__ == "__main__":
    main()
