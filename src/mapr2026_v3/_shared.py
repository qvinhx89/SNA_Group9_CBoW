from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MaprPaths:
    graph_edgelist: str = "data/processed/graph_active.edgelist"
    csr_npz: str = "data/processed/graph_csr.npz"

    node_attributes: str = "data/processed/node_attributes.parquet"
    sis_table: str = "data/processed/sis_table.parquet"

    day1_dir: str = "outputs/day1_benchmark"
    results_dir: str = "outputs/mapr2026_v3_results"

    ic_scores: str = "data/processed/ic_scores_primary.parquet"
    regression_targets: str = "data/processed/regression_targets.parquet"
    classification_labels: str = "data/processed/classification_labels.parquet"
    # M0-locked: degree-stratified 80/20 split over labeled nodes, seed=42
    split_masks: str = "data/processed/split_masks.parquet"

    proxies: str = "data/processed/diffusion_proxies.parquet"
    typology: str = "data/processed/typology_labels_ic_views.parquet"
    proxies_status: str = "outputs/mapr2026_v3_results/diffusion_proxies_status.json"
    louvain_resolution_sensitivity: str = "outputs/mapr2026_v3_results/louvain_resolution_sensitivity.json"


PATHS = MaprPaths()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = ensure_parent(path)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def read_edgelist_pairs(edgelist_path: str | Path, max_edges: int | None = None) -> tuple[list[str], list[str]]:
    """Read a whitespace-separated edgelist into two node-id lists.

    Notes
    -----
    - NetworkX `write_edgelist` typically outputs `u v {attr...}` per line.
    - We only use the first two tokens.
    """
    src: list[str] = []
    dst: list[str] = []
    with Path(edgelist_path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            src.append(str(parts[0]))
            dst.append(str(parts[1]))
            if max_edges is not None and len(src) >= max_edges:
                break
    return src, dst


def require_columns(df: pd.DataFrame, required: Iterable[str], df_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} missing required columns: {missing}")


def load_csr_npz(npz_path: str | Path) -> dict[str, Any]:
    """Load CSR arrays saved via np.savez.

    Required keys (minimum contract): indptr, indices, degrees, node_ids.
    """
    p = Path(npz_path)
    if not p.exists():
        raise FileNotFoundError(f"Missing CSR artifact: {p}")

    data = np.load(p, allow_pickle=True)
    required = ["indptr", "indices", "degrees", "node_ids"]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise ValueError(f"CSR NPZ missing keys: {missing}. Found: {data.files}")

    indptr = data["indptr"].astype(np.int64, copy=False)
    indices = data["indices"].astype(np.int64, copy=False)
    degrees = data["degrees"].astype(np.int64, copy=False)
    node_ids = data["node_ids"].astype(str)

    if indptr.ndim != 1 or indices.ndim != 1 or degrees.ndim != 1:
        raise ValueError("CSR arrays must be 1-D")
    if indptr.shape[0] != degrees.shape[0] + 1:
        raise ValueError("CSR shape mismatch: len(indptr) must be len(degrees)+1")
    if node_ids.shape[0] != degrees.shape[0]:
        raise ValueError("CSR mapping mismatch: len(node_ids) must equal len(degrees)")

    # Degree consistency check
    calc_deg = np.diff(indptr)
    if not np.array_equal(calc_deg, degrees):
        raise ValueError("CSR degrees mismatch: degrees[i] != indptr[i+1]-indptr[i]")

    return {
        "indptr": indptr,
        "indices": indices,
        "degrees": degrees,
        "node_ids": node_ids,
    }


def assert_diffusion_proxies_ready_for_eval_runtime(
    proxies_path: str | Path,
    expected_n_nodes: int | None = None,
) -> pd.DataFrame:
    """Guard against accidentally using dry-run diffusion proxies in eval/runtime.

    Conditions enforced:
    - file exists
    - required columns exist
    - non-empty
    - no NaN in one_hop_spread/two_hop_spread
    - unique node_id rows
    - optional full-graph row count check
    """
    p = Path(proxies_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Missing diffusion proxies artifact: {p}. "
            "Run diffusion_proxies.py real mode before eval/runtime."
        )

    df = pd.read_parquet(p)
    require_columns(df, ["node_id", "one_hop_spread", "two_hop_spread"], "diffusion_proxies")

    if len(df) == 0:
        raise ValueError(
            "diffusion_proxies.parquet is header-only (dry-run placeholder). "
            "Do not use this for evaluation/runtime."
        )

    if df[["one_hop_spread", "two_hop_spread"]].isna().any().any():
        raise ValueError(
            "diffusion_proxies.parquet contains NaN proxy values (likely placeholder). "
            "Run diffusion_proxies.py real mode before evaluation/runtime."
        )

    node_ids = df["node_id"].astype(str)
    if node_ids.nunique() != len(df):
        raise ValueError("diffusion_proxies.parquet has duplicate node_id rows.")

    if expected_n_nodes is not None and node_ids.nunique() != expected_n_nodes:
        raise ValueError(
            "diffusion_proxies.parquet does not cover full active graph: "
            f"got {node_ids.nunique()}, expected {expected_n_nodes}."
        )

    return df


def save_csr_npz(
    npz_path: str | Path,
    indptr: np.ndarray,
    indices: np.ndarray,
    degrees: np.ndarray,
    node_ids: np.ndarray,
) -> None:
    out = ensure_parent(npz_path)
    np.savez_compressed(
        out,
        indptr=indptr.astype(np.int64, copy=False),
        indices=indices.astype(np.int64, copy=False),
        degrees=degrees.astype(np.int64, copy=False),
        node_ids=node_ids.astype(str, copy=False),
    )


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def timed(fn):
    def _wrap(*args, **kwargs):
        t0 = time.time()
        out = fn(*args, **kwargs)
        return out, time.time() - t0

    return _wrap
