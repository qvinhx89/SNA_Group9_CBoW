"""MAPR2026 v3 - Person 2 preflight automation.

This script validates prerequisites for Track B (proxies + typology + null model)
and prints PASS/WARN/FAIL checks against the current contract in docs.

Usage
-----
Run from repository root (recommended):

    python src/mapr2026_v3/preflight_person2.py

Exit codes
----------
0: No blocking failures for the requested readiness level.
2: One or more blocking failures.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REQUIRED_P_MODEL = "weighted_cascade"
DEFAULT_PERM_MIN_POLICY = 200

INDEPENDENT_CHECKS = {
    "Python version",
    "Required Python packages",
    "Stage-0 active graph",
    "node_attributes base schema",
    "Results directory writable",
}

REAL_MODE_CHECKS = {
    "Python version",
    "Required Python packages",
    "Stage-0 active graph",
    "node_attributes base schema",
    "Results directory writable",
    "CSR artifact contract",
    "IC scores contract",
    "Community features contract",
}

COMPLETION_READY_CHECKS = {
    "Python version",
    "Required Python packages",
    "Stage-0 active graph",
    "node_attributes base schema",
    "Results directory writable",
    "CSR artifact contract",
    "IC scores contract",
    "Community features contract",
    "Day-1 benchmark artifacts",
    "Day-1 decisions note",
    "Centrality table contract",
    "Input join coherence",
    "Stage-5 null package",
    "Metric correlation matrix",
}


@dataclass
class CheckResult:
    name: str
    status: str  # PASS | WARN | FAIL
    blocking: bool
    detail: str


def _repo_root() -> Path:
    # src/mapr2026_v3/preflight_person2.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


def _load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _require_columns(df: pd.DataFrame, cols: Iterable[str]) -> list[str]:
    return [c for c in cols if c not in df.columns]


def _check_python_version(min_minor: int = 10, max_minor: int = 12) -> CheckResult:
    ver = sys.version_info
    ok = (ver.major == 3) and (min_minor <= ver.minor <= max_minor)
    if ok:
        return CheckResult(
            name="Python version",
            status="PASS",
            blocking=True,
            detail=f"Detected Python {ver.major}.{ver.minor}.{ver.micro} (required: 3.{min_minor}..3.{max_minor})",
        )
    return CheckResult(
        name="Python version",
        status="FAIL",
        blocking=True,
        detail=f"Detected Python {ver.major}.{ver.minor}.{ver.micro}; required range is 3.{min_minor}..3.{max_minor}",
    )


def _check_imports() -> CheckResult:
    required = [
        "numpy",
        "pandas",
        "scipy",
        "networkx",
        "sklearn",
        "pyarrow",
        "community",  # python-louvain
    ]

    missing_required: list[str] = []

    for mod in required:
        try:
            __import__(mod)
        except Exception:
            missing_required.append(mod)

    if missing_required:
        return CheckResult(
            name="Required Python packages",
            status="FAIL",
            blocking=True,
            detail=f"Missing required modules: {missing_required}",
        )

    return CheckResult(
        name="Required Python packages",
        status="PASS",
        blocking=True,
        detail="All required modules import successfully (including python-louvain).",
    )


def _check_file_exists(path: Path, name: str, blocking: bool = True) -> CheckResult:
    if path.exists():
        return CheckResult(name=name, status="PASS", blocking=blocking, detail=f"Found: {path}")
    return CheckResult(name=name, status="FAIL", blocking=blocking, detail=f"Missing: {path}")


def _check_node_attributes_base(path: Path) -> CheckResult:
    if not path.exists():
        return CheckResult(
            name="node_attributes base schema",
            status="FAIL",
            blocking=True,
            detail=f"Missing file: {path}",
        )
    df = _load_parquet(path)
    missing = _require_columns(df, ["node_id", "views", "life_time"])
    if missing:
        return CheckResult(
            name="node_attributes base schema",
            status="FAIL",
            blocking=True,
            detail=f"Missing required columns {missing} in {path}",
        )

    base = df[["node_id", "views", "life_time"]].copy()
    if base["node_id"].astype(str).nunique() != len(base):
        return CheckResult(
            name="node_attributes base schema",
            status="FAIL",
            blocking=True,
            detail=f"node_id must be unique in {path}",
        )

    if base.isna().any().any():
        na_counts = base.isna().sum().to_dict()
        return CheckResult(
            name="node_attributes base schema",
            status="FAIL",
            blocking=True,
            detail=f"Found missing values in base columns {na_counts} in {path}",
        )

    if (pd.to_numeric(base["life_time"], errors="coerce") <= 0).any():
        return CheckResult(
            name="node_attributes base schema",
            status="FAIL",
            blocking=True,
            detail=f"life_time must be > 0 for all rows in {path}",
        )

    return CheckResult(
        name="node_attributes base schema",
        status="PASS",
        blocking=True,
        detail=f"OK columns and non-missing values in {path}: node_id, views, life_time",
    )


def _read_active_nodes(edgelist_path: Path) -> set[str]:
    edge_df = pd.read_csv(
        edgelist_path,
        sep=r"\s+",
        header=None,
        usecols=[0, 1],
        names=["u", "v"],
    )
    return set(edge_df["u"].astype(str)).union(set(edge_df["v"].astype(str)))


def _check_community_features(
    node_attrs_path: Path,
    community_features_path: Path,
    active_nodes: set[str],
) -> CheckResult:
    required = ["node_id", "community_id", "cross_community_edge_fraction"]
    source = ""
    cf: pd.DataFrame

    # Option 1: embedded in node_attributes.parquet
    if node_attrs_path.exists():
        attrs = _load_parquet(node_attrs_path)
        missing = _require_columns(attrs, required)
        if not missing:
            cf = attrs[required].copy()
            source = str(node_attrs_path)
        else:
            cf = pd.DataFrame()
    else:
        cf = pd.DataFrame()

    # Option 2: separate community_features.parquet
    if cf.empty and community_features_path.exists():
        cf = _load_parquet(community_features_path)
        missing = _require_columns(cf, required)
        if not missing:
            cf = cf[required].copy()
            source = str(community_features_path)

    if cf.empty:
        return CheckResult(
            name="Community features contract",
            status="FAIL",
            blocking=True,
            detail=(
                "Missing community features columns (node_id, community_id, cross_community_edge_fraction). "
                f"Checked {node_attrs_path} and {community_features_path}."
            ),
        )

    cf["node_id"] = cf["node_id"].astype(str)
    if cf["node_id"].nunique() != len(cf):
        return CheckResult(
            name="Community features contract",
            status="FAIL",
            blocking=True,
            detail=f"node_id must be unique in community features source: {source}",
        )

    if cf[required].isna().any().any():
        na_counts = cf[required].isna().sum().to_dict()
        return CheckResult(
            name="Community features contract",
            status="FAIL",
            blocking=True,
            detail=f"Found missing values in required columns {na_counts} (source: {source})",
        )

    cross = pd.to_numeric(cf["cross_community_edge_fraction"], errors="coerce")
    if cross.isna().any() or ((cross < 0) | (cross > 1)).any():
        return CheckResult(
            name="Community features contract",
            status="FAIL",
            blocking=True,
            detail="cross_community_edge_fraction must be numeric and within [0, 1]",
        )

    feature_nodes = set(cf["node_id"])
    missing_active = active_nodes - feature_nodes
    if missing_active:
        return CheckResult(
            name="Community features contract",
            status="FAIL",
            blocking=True,
            detail=(
                f"Community features do not cover all active nodes: missing={len(missing_active)} "
                f"(source: {source})"
            ),
        )

    return CheckResult(
        name="Community features contract",
        status="PASS",
        blocking=True,
        detail=f"Valid schema + no missing + full active-node coverage in {source}",
    )


def _check_csr_contract(path: Path) -> CheckResult:
    if not path.exists():
        return CheckResult(name="CSR artifact contract", status="FAIL", blocking=True, detail=f"Missing: {path}")

    try:
        data = np.load(path, allow_pickle=True)
        required = ["indptr", "indices", "degrees", "node_ids"]
        missing = [k for k in required if k not in data.files]
        if missing:
            return CheckResult(
                name="CSR artifact contract",
                status="FAIL",
                blocking=True,
                detail=f"Missing keys {missing} in {path}",
            )

        indptr = data["indptr"].astype(np.int64, copy=False)
        indices = data["indices"].astype(np.int64, copy=False)
        degrees = data["degrees"].astype(np.int64, copy=False)
        node_ids = data["node_ids"].astype(str)

        if indptr.ndim != 1 or degrees.ndim != 1 or indices.ndim != 1:
            return CheckResult(
                name="CSR artifact contract",
                status="FAIL",
                blocking=True,
                detail="indptr/indices/degrees must be 1-D arrays",
            )

        if indptr.shape[0] != degrees.shape[0] + 1:
            return CheckResult(
                name="CSR artifact contract",
                status="FAIL",
                blocking=True,
                detail="len(indptr) must equal len(degrees)+1",
            )

        if node_ids.shape[0] != degrees.shape[0]:
            return CheckResult(
                name="CSR artifact contract",
                status="FAIL",
                blocking=True,
                detail="len(node_ids) must equal len(degrees)",
            )

        calc_deg = np.diff(indptr)
        if not np.array_equal(calc_deg, degrees):
            return CheckResult(
                name="CSR artifact contract",
                status="FAIL",
                blocking=True,
                detail="Degree mismatch: degrees[i] != indptr[i+1]-indptr[i]",
            )

        n_nodes = degrees.shape[0]
        if indptr[0] != 0 or indptr[-1] != indices.shape[0]:
            return CheckResult(
                name="CSR artifact contract",
                status="FAIL",
                blocking=True,
                detail="Invalid CSR pointers: require indptr[0]=0 and indptr[-1]=len(indices)",
            )

        if np.any(np.diff(indptr) < 0):
            return CheckResult(
                name="CSR artifact contract",
                status="FAIL",
                blocking=True,
                detail="indptr must be non-decreasing",
            )

        if np.any(indices < 0) or np.any(indices >= n_nodes):
            return CheckResult(
                name="CSR artifact contract",
                status="FAIL",
                blocking=True,
                detail="indices must be in [0, n_nodes)",
            )

        if np.unique(node_ids).shape[0] != node_ids.shape[0]:
            return CheckResult(
                name="CSR artifact contract",
                status="FAIL",
                blocking=True,
                detail="node_ids must be unique",
            )

        sorted_ids = np.sort(node_ids)
        if not np.array_equal(node_ids, sorted_ids):
            return CheckResult(
                name="CSR artifact contract",
                status="FAIL",
                blocking=True,
                detail="node_ids must be sorted ascending for deterministic mapping",
            )

        return CheckResult(
            name="CSR artifact contract",
            status="PASS",
            blocking=True,
            detail=f"OK schema, bounds, determinism, and degree consistency in {path}",
        )
    except Exception as ex:
        return CheckResult(
            name="CSR artifact contract",
            status="FAIL",
            blocking=True,
            detail=f"Failed to validate CSR NPZ: {ex}",
        )


def _check_ic_scores_contract(path: Path) -> CheckResult:
    if not path.exists():
        return CheckResult(name="IC scores contract", status="FAIL", blocking=True, detail=f"Missing: {path}")

    try:
        df = _load_parquet(path)
        missing = _require_columns(df, ["node_id", "ic_score_mean", "ic_score_std", "n_runs", "p_model"])
        if missing:
            return CheckResult(
                name="IC scores contract",
                status="FAIL",
                blocking=True,
                detail=f"Missing columns {missing} in {path}",
            )

        req = ["node_id", "ic_score_mean", "ic_score_std", "n_runs", "p_model"]
        if df[req].isna().any().any():
            na_counts = df[req].isna().sum().to_dict()
            return CheckResult(
                name="IC scores contract",
                status="FAIL",
                blocking=True,
                detail=f"Found missing values in required columns {na_counts} in {path}",
            )

        node_ids = df["node_id"].astype(str)
        if node_ids.nunique() != len(df):
            return CheckResult(
                name="IC scores contract",
                status="FAIL",
                blocking=True,
                detail=f"node_id must be unique in {path}",
            )

        n_runs = pd.to_numeric(df["n_runs"], errors="coerce")
        if n_runs.isna().any() or (n_runs <= 0).any():
            return CheckResult(
                name="IC scores contract",
                status="FAIL",
                blocking=True,
                detail="n_runs must be numeric and > 0 for all rows",
            )

        p_models = set(df["p_model"].astype(str).str.lower().unique())
        if p_models != {REQUIRED_P_MODEL}:
            return CheckResult(
                name="IC scores contract",
                status="FAIL",
                blocking=True,
                detail=f"p_model must be '{REQUIRED_P_MODEL}' only; found {sorted(p_models)}",
            )

        if not np.isfinite(pd.to_numeric(df["ic_score_mean"], errors="coerce")).all():
            return CheckResult(
                name="IC scores contract",
                status="FAIL",
                blocking=True,
                detail="ic_score_mean must be finite numeric values",
            )

        return CheckResult(
            name="IC scores contract",
            status="PASS",
            blocking=True,
            detail=(
                f"OK schema, uniqueness, non-missing, n_runs>0, and p_model={REQUIRED_P_MODEL} "
                f"in {path} (rows={len(df)})"
            ),
        )
    except Exception as ex:
        return CheckResult(
            name="IC scores contract",
            status="FAIL",
            blocking=True,
            detail=f"Failed to read/validate {path}: {ex}",
        )


def _check_day1_decisions(path: Path) -> CheckResult:
    if not path.exists():
        return CheckResult(
            name="Day-1 decisions note",
            status="FAIL",
            blocking=True,
            detail=f"Missing: {path}",
        )

    text = path.read_text(encoding="utf-8", errors="ignore").lower()

    # Plan uses n_sample/n_runs/narrative_branch (legacy docs may still mention n_seeds/narrative).
    has_sample = ("n_sample" in text) or ("n_seeds" in text)
    has_runs = "n_runs" in text
    has_narrative = ("narrative_branch" in text) or ("narrative" in text)

    missing_tokens: list[str] = []
    if not has_sample:
        missing_tokens.append("n_sample")
    if not has_runs:
        missing_tokens.append("n_runs")
    if not has_narrative:
        missing_tokens.append("narrative_branch")
    if missing_tokens:
        return CheckResult(
            name="Day-1 decisions note",
            status="FAIL",
            blocking=True,
            detail=f"Found {path} but missing decision tokens: {missing_tokens}",
        )

    return CheckResult(
        name="Day-1 decisions note",
        status="PASS",
        blocking=True,
        detail=f"Found locked decisions in {path}",
    )


def _check_day1_benchmark_artifacts(day1_dir: Path) -> CheckResult:
    runtime_path = day1_dir / "ic_runtime_benchmark.json"
    corr_path = day1_dir / "one_hop_correlation.json"

    missing_files = [str(p) for p in [runtime_path, corr_path] if not p.exists()]
    if missing_files:
        return CheckResult(
            name="Day-1 benchmark artifacts",
            status="FAIL",
            blocking=True,
            detail=f"Missing required Day-1 file(s): {missing_files}",
        )

    try:
        runtime_payload = json.loads(runtime_path.read_text(encoding="utf-8"))
        corr_payload = json.loads(corr_path.read_text(encoding="utf-8"))
    except Exception as ex:
        return CheckResult(
            name="Day-1 benchmark artifacts",
            status="FAIL",
            blocking=True,
            detail=f"Failed to parse Day-1 JSON files: {ex}",
        )

    runtime_keys = ["per_sim_ms", "projected_total_hours", "decision"]
    corr_keys = ["spearman_rho", "decision_branch"]

    missing_runtime = [k for k in runtime_keys if k not in runtime_payload]
    missing_corr = [k for k in corr_keys if k not in corr_payload]

    if missing_runtime or missing_corr:
        return CheckResult(
            name="Day-1 benchmark artifacts",
            status="FAIL",
            blocking=True,
            detail=(
                f"Schema mismatch. runtime_missing={missing_runtime}, "
                f"corr_missing={missing_corr}"
            ),
        )

    return CheckResult(
        name="Day-1 benchmark artifacts",
        status="PASS",
        blocking=True,
        detail=f"Found required Day-1 artifacts and keys in {day1_dir}",
    )


def _check_centrality_contract(path: Path, active_nodes: set[str]) -> CheckResult:
    if not path.exists():
        return CheckResult(
            name="Centrality table contract",
            status="FAIL",
            blocking=True,
            detail=f"Missing: {path}",
        )

    try:
        df = _load_parquet(path)
        required = ["node_id", "degree", "pagerank", "betweenness", "kshell"]
        missing = _require_columns(df, required)
        if missing:
            return CheckResult(
                name="Centrality table contract",
                status="FAIL",
                blocking=True,
                detail=f"Missing columns {missing} in {path}",
            )

        dfr = df[required].copy()
        dfr["node_id"] = dfr["node_id"].astype(str)
        if dfr["node_id"].nunique() != len(dfr):
            return CheckResult(
                name="Centrality table contract",
                status="FAIL",
                blocking=True,
                detail=f"node_id must be unique in {path}",
            )

        if dfr[required].isna().any().any():
            na_counts = dfr[required].isna().sum().to_dict()
            return CheckResult(
                name="Centrality table contract",
                status="FAIL",
                blocking=True,
                detail=f"Found missing values in required columns {na_counts}",
            )

        missing_active = active_nodes - set(dfr["node_id"])
        if missing_active:
            return CheckResult(
                name="Centrality table contract",
                status="FAIL",
                blocking=True,
                detail=f"Centrality table missing {len(missing_active)} active nodes",
            )

        return CheckResult(
            name="Centrality table contract",
            status="PASS",
            blocking=True,
            detail=f"Valid schema + non-missing + full active-node coverage in {path}",
        )
    except Exception as ex:
        return CheckResult(
            name="Centrality table contract",
            status="FAIL",
            blocking=True,
            detail=f"Failed to validate centrality table: {ex}",
        )


def _check_input_join_coherence(
    ic_scores_path: Path,
    node_attrs_path: Path,
    community_features_path: Path,
    centrality_path: Path,
) -> CheckResult:
    try:
        ic = _load_parquet(ic_scores_path)
        attrs = _load_parquet(node_attrs_path)
        cent = _load_parquet(centrality_path)
    except Exception as ex:
        return CheckResult(
            name="Input join coherence",
            status="FAIL",
            blocking=True,
            detail=f"Could not load one of required inputs for join checks: {ex}",
        )

    ic_ids = set(ic["node_id"].astype(str))
    attrs2 = attrs.copy()
    attrs_missing = _require_columns(attrs2, ["node_id", "views", "life_time"])
    if attrs_missing:
        return CheckResult(
            name="Input join coherence",
            status="FAIL",
            blocking=True,
            detail=f"node_attributes is missing required columns for join checks: {attrs_missing}",
        )
    attrs2["node_id"] = attrs2["node_id"].astype(str)

    cent2 = cent.copy()
    cent_missing = _require_columns(cent2, ["node_id", "degree", "pagerank", "betweenness", "kshell"])
    if cent_missing:
        return CheckResult(
            name="Input join coherence",
            status="FAIL",
            blocking=True,
            detail=f"centrality_table is missing required columns for join checks: {cent_missing}",
        )
    cent2["node_id"] = cent2["node_id"].astype(str)

    missing_in_attrs = ic_ids - set(attrs2["node_id"])
    if missing_in_attrs:
        return CheckResult(
            name="Input join coherence",
            status="FAIL",
            blocking=True,
            detail=f"{len(missing_in_attrs)} IC-labeled nodes missing in node_attributes",
        )

    # views + life_time must be usable on all IC-labeled nodes
    attrs_ic = attrs2[attrs2["node_id"].isin(ic_ids)]
    if attrs_ic[["views", "life_time"]].isna().any().any():
        na_counts = attrs_ic[["views", "life_time"]].isna().sum().to_dict()
        return CheckResult(
            name="Input join coherence",
            status="FAIL",
            blocking=True,
            detail=f"Missing views/life_time on IC-labeled nodes: {na_counts}",
        )

    missing_in_cent = ic_ids - set(cent2["node_id"])
    if missing_in_cent:
        return CheckResult(
            name="Input join coherence",
            status="FAIL",
            blocking=True,
            detail=f"{len(missing_in_cent)} IC-labeled nodes missing in centrality table",
        )

    # Community features may live in node_attributes or dedicated file.
    if all(c in attrs2.columns for c in ["community_id", "cross_community_edge_fraction"]):
        comm_source = attrs2[["node_id", "community_id", "cross_community_edge_fraction"]].copy()
    elif community_features_path.exists():
        comm_source = _load_parquet(community_features_path)[
            ["node_id", "community_id", "cross_community_edge_fraction"]
        ].copy()
    else:
        return CheckResult(
            name="Input join coherence",
            status="FAIL",
            blocking=True,
            detail="No community source available for join checks",
        )

    comm_source["node_id"] = comm_source["node_id"].astype(str)
    missing_in_comm = ic_ids - set(comm_source["node_id"])
    if missing_in_comm:
        return CheckResult(
            name="Input join coherence",
            status="FAIL",
            blocking=True,
            detail=f"{len(missing_in_comm)} IC-labeled nodes missing community features",
        )

    return CheckResult(
        name="Input join coherence",
        status="PASS",
        blocking=True,
        detail="All IC-labeled nodes can be joined with views/life_time, centrality, and community features",
    )


def _check_outputs_dir(path: Path) -> CheckResult:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".preflight_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return CheckResult(name="Results directory writable", status="PASS", blocking=True, detail=f"Writable: {path}")
    except Exception as ex:
        return CheckResult(
            name="Results directory writable",
            status="FAIL",
            blocking=True,
            detail=f"Cannot write to {path}: {ex}",
        )


def _check_stage5_null_package(results_dir: Path) -> CheckResult:
    """Blocking completion check for Stage-5 null package (Task 7/8/9)."""
    null_model_path = results_dir / "null_model_typology_summary.json"
    views_perm_path = results_dir / "views_permutation_null_summary.json"
    ic_perm_path = results_dir / "ic_permutation_null_summary.json"

    missing_files = [str(p) for p in [null_model_path, views_perm_path, ic_perm_path] if not p.exists()]
    if missing_files:
        return CheckResult(
            name="Stage-5 null package",
            status="FAIL",
            blocking=True,
            detail=f"Missing required null-package artifact(s): {missing_files}",
        )

    def _load_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    try:
        payload_null = _load_json(null_model_path)
        payload_views = _load_json(views_perm_path)
        payload_ic = _load_json(ic_perm_path)
    except Exception as ex:
        return CheckResult(
            name="Stage-5 null package",
            status="FAIL",
            blocking=True,
            detail=f"Failed to parse one or more null-package JSON files: {ex}",
        )

    required_null = [
        "timestamp",
        "n_nodes",
        "n_realizations",
        "n_runs_per_node",
        "rho_mean",
        "rho_std",
        "hidden_betweenness_real_subgraph_mean",
        "hidden_betweenness_null_mean",
        "hidden_betweenness_null_std",
        "interpretation",
    ]
    missing_null = [k for k in required_null if k not in payload_null]
    if missing_null:
        return CheckResult(
            name="Stage-5 null package",
            status="FAIL",
            blocking=True,
            detail=f"null_model_typology_summary.json missing keys: {missing_null}",
        )

    for label, payload in [
        ("views_permutation_null_summary.json", payload_views),
        ("ic_permutation_null_summary.json", payload_ic),
    ]:
        required_top = [
            "timestamp",
            "n_nodes_labeled",
            "n_permutations",
            "top_pct",
            "real",
            "null_distribution",
            "empirical_p_values",
            "interpretation",
        ]
        missing_top = [k for k in required_top if k not in payload]
        if missing_top:
            return CheckResult(
                name="Stage-5 null package",
                status="FAIL",
                blocking=True,
                detail=f"{label} missing top-level keys: {missing_top}",
            )

        if int(payload.get("n_permutations", 0)) < 1:
            return CheckResult(
                name="Stage-5 null package",
                status="FAIL",
                blocking=True,
                detail=f"{label} has invalid n_permutations={payload.get('n_permutations')}",
            )

        if not isinstance(payload.get("real"), dict) or "agreement_rate" not in payload["real"]:
            return CheckResult(
                name="Stage-5 null package",
                status="FAIL",
                blocking=True,
                detail=f"{label} missing real.agreement_rate",
            )

        if not isinstance(payload.get("null_distribution"), dict) or "agreement_rate_mean" not in payload["null_distribution"]:
            return CheckResult(
                name="Stage-5 null package",
                status="FAIL",
                blocking=True,
                detail=f"{label} missing null_distribution.agreement_rate_mean",
            )

        if not isinstance(payload.get("empirical_p_values"), dict):
            return CheckResult(
                name="Stage-5 null package",
                status="FAIL",
                blocking=True,
                detail=f"{label} missing empirical_p_values object",
            )

    return CheckResult(
        name="Stage-5 null package",
        status="PASS",
        blocking=True,
        detail=(
            "Found complete Task 7/8/9 null package: "
            "null_model_typology_summary.json + views_permutation_null_summary.json + ic_permutation_null_summary.json"
        ),
    )


def _soft_check_permutation_quality(results_dir: Path, min_permutations_policy: int) -> CheckResult:
    """Non-blocking quality check for Task 8/9 permutation artifacts.

    This check is intentionally soft (WARN/PASS only) to avoid blocking pipeline
    completion while still surfacing statistical-quality risks.
    """
    if int(min_permutations_policy) < 1:
        return CheckResult(
            name="Permutation quality policy",
            status="WARN",
            blocking=False,
            detail=f"Invalid min_permutations_policy={min_permutations_policy}; expected >=1",
        )

    views_perm_path = results_dir / "views_permutation_null_summary.json"
    ic_perm_path = results_dir / "ic_permutation_null_summary.json"

    missing_files = [str(p) for p in [views_perm_path, ic_perm_path] if not p.exists()]
    if missing_files:
        return CheckResult(
            name="Permutation quality policy",
            status="WARN",
            blocking=False,
            detail=f"Cannot evaluate permutation quality because artifact(s) missing: {missing_files}",
        )

    try:
        payload_views = json.loads(views_perm_path.read_text(encoding="utf-8"))
        payload_ic = json.loads(ic_perm_path.read_text(encoding="utf-8"))
    except Exception as ex:
        return CheckResult(
            name="Permutation quality policy",
            status="WARN",
            blocking=False,
            detail=f"Could not parse permutation artifacts for soft quality check: {ex}",
        )

    warns: list[str] = []

    def _eval_one(label: str, payload: dict[str, Any]) -> None:
        n_perm = int(payload.get("n_permutations", 0))
        if n_perm < int(min_permutations_policy):
            warns.append(
                f"{label}: n_permutations={n_perm} < policy_min={min_permutations_policy}"
            )

        if n_perm > 0:
            min_p_resolution = 1.0 / float(n_perm + 1)
            if min_p_resolution > 0.01:
                warns.append(
                    f"{label}: min empirical p-value resolution={min_p_resolution:.4f} is coarse (>0.01)"
                )

        top_pct = float(payload.get("top_pct", -1.0))
        if not np.isclose(top_pct, 0.10, atol=1e-9):
            warns.append(f"{label}: top_pct={top_pct} differs from M0 lock 0.10")

        null_dist = payload.get("null_distribution")
        if isinstance(null_dist, dict):
            std_agree = float(null_dist.get("agreement_rate_std", 0.0))
            if std_agree <= 0.0:
                warns.append(f"{label}: agreement_rate_std<=0 suggests degenerate/null issue")

    _eval_one("views_permutation_null_summary.json", payload_views)
    _eval_one("ic_permutation_null_summary.json", payload_ic)

    if warns:
        return CheckResult(
            name="Permutation quality policy",
            status="WARN",
            blocking=False,
            detail="; ".join(warns),
        )

    return CheckResult(
        name="Permutation quality policy",
        status="PASS",
        blocking=False,
        detail=(
            "Permutation artifacts satisfy soft policy: "
            f"n_permutations >= {min_permutations_policy}, p-resolution <= 0.01, top_pct=0.10"
        ),
    )


def _check_lifetime_if_problem_handled(results_dir: Path, assumptions_doc: Path) -> CheckResult:
    """Non-blocking visibility check for Task 6 IF PROBLEM fallback handling."""
    lifetime_path = results_dir / "lifetime_validation.json"
    language_path = results_dir / "language_validation.json"

    if not lifetime_path.exists():
        return CheckResult(
            name="Lifetime IF PROBLEM status",
            status="WARN",
            blocking=False,
            detail=f"Cannot evaluate lifetime fallback status because artifact is missing: {lifetime_path}",
        )

    try:
        lifetime_payload = json.loads(lifetime_path.read_text(encoding="utf-8"))
    except Exception as ex:
        return CheckResult(
            name="Lifetime IF PROBLEM status",
            status="WARN",
            blocking=False,
            detail=f"Could not parse lifetime_validation.json for fallback visibility: {ex}",
        )

    if not isinstance(lifetime_payload, dict):
        return CheckResult(
            name="Lifetime IF PROBLEM status",
            status="WARN",
            blocking=False,
            detail="lifetime_validation.json payload is not an object; cannot determine IF PROBLEM status",
        )

    try:
        rho = float(lifetime_payload.get("partial_spearman_rho", np.nan))
    except Exception:
        rho = float("nan")

    try:
        n_sig = int(lifetime_payload.get("n_quintiles_significant"))
    except Exception:
        n_sig = -1

    success_flag = lifetime_payload.get("success")
    trigger_from_metrics = (np.isfinite(rho) and rho < 0.05) or (n_sig >= 0 and n_sig < 3)
    trigger_from_success = isinstance(success_flag, bool) and (not success_flag)
    fallback_triggered = bool(trigger_from_metrics or trigger_from_success)

    if not fallback_triggered:
        return CheckResult(
            name="Lifetime IF PROBLEM status",
            status="PASS",
            blocking=False,
            detail=(
                "IF PROBLEM not triggered: lifetime_validation.json passes trigger conditions "
                f"(partial_spearman_rho={rho:.4f}, n_quintiles_significant={n_sig})"
            ),
        )

    if not language_path.exists():
        return CheckResult(
            name="Lifetime IF PROBLEM status",
            status="WARN",
            blocking=False,
            detail=(
                "IF PROBLEM triggered by lifetime_validation.json but language fallback artifact is missing: "
                f"{language_path}"
            ),
        )

    try:
        language_payload = json.loads(language_path.read_text(encoding="utf-8"))
    except Exception as ex:
        return CheckResult(
            name="Lifetime IF PROBLEM status",
            status="WARN",
            blocking=False,
            detail=f"IF PROBLEM triggered but language_validation.json could not be parsed: {ex}",
        )

    if not isinstance(language_payload, dict):
        return CheckResult(
            name="Lifetime IF PROBLEM status",
            status="WARN",
            blocking=False,
            detail="IF PROBLEM triggered but language_validation.json payload is not an object",
        )

    assumptions_note_present = False
    if assumptions_doc.exists():
        try:
            assumptions_text = assumptions_doc.read_text(encoding="utf-8").lower()
            assumptions_note_present = (
                "task 6 if problem" in assumptions_text
                and "language-based corroboration" in assumptions_text
                and "supplementary" in assumptions_text
            )
        except Exception:
            assumptions_note_present = False

    if assumptions_note_present:
        return CheckResult(
            name="Lifetime IF PROBLEM status",
            status="PASS",
            blocking=False,
            detail=(
                "IF PROBLEM handled: lifetime trigger active and fallback corroboration documented "
                "(language_validation.json present + assumptions_limitations note found)."
            ),
        )

    return CheckResult(
        name="Lifetime IF PROBLEM status",
        status="WARN",
        blocking=False,
        detail=(
            "IF PROBLEM triggered and language_validation.json exists, but assumptions_limitations note "
            "(Task 6 IF PROBLEM + supplementary language corroboration) was not found."
        ),
    )


def _check_metric_correlation_matrix(results_dir: Path, ic_scores_path: Path) -> CheckResult:
    """Blocking completion check for Task 11 metric correlation artifact."""
    path = results_dir / "metric_correlation_matrix.json"
    if not path.exists():
        return CheckResult(
            name="Metric correlation matrix",
            status="FAIL",
            blocking=True,
            detail=f"Missing required artifact: {path}",
        )

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as ex:
        return CheckResult(
            name="Metric correlation matrix",
            status="FAIL",
            blocking=True,
            detail=f"Failed to parse JSON: {ex}",
        )

    if not isinstance(payload, dict):
        return CheckResult(
            name="Metric correlation matrix",
            status="FAIL",
            blocking=True,
            detail="Top-level JSON payload must be an object",
        )

    try:
        df_ic = _load_parquet(ic_scores_path)
        if "node_id" not in df_ic.columns:
            return CheckResult(
                name="Metric correlation matrix",
                status="FAIL",
                blocking=True,
                detail=f"IC scores file missing node_id column: {ic_scores_path}",
            )
        expected_rows_from_ic = int(df_ic["node_id"].astype(str).nunique())
    except Exception as ex:
        return CheckResult(
            name="Metric correlation matrix",
            status="FAIL",
            blocking=True,
            detail=f"Could not read IC scores for coverage check: {ex}",
        )
    try:
        required_top = [
            "timestamp",
            "n_rows",
            "n_rows_expected",
            "coverage_ok",
            "metrics",
            "rho_matrix",
            "p_matrix_corrected",
            "column_mapping",
        ]
        missing_top = [k for k in required_top if k not in payload]
        if missing_top:
            return CheckResult(
                name="Metric correlation matrix",
                status="FAIL",
                blocking=True,
                detail=f"Missing top-level keys: {missing_top}",
            )

        expected_metrics = [
            "ic_score_mean",
            "views",
            "degree",
            "pagerank",
            "kshell",
            "betweenness_approx",
            "one_hop_spread",
            "two_hop_spread",
        ]
        metrics = payload.get("metrics")
        if metrics != expected_metrics:
            return CheckResult(
                name="Metric correlation matrix",
                status="FAIL",
                blocking=True,
                detail=f"metrics list mismatch. expected={expected_metrics}, got={metrics}",
            )

        try:
            n_rows = int(payload.get("n_rows"))
            n_rows_expected = int(payload.get("n_rows_expected"))
        except Exception as ex:
            return CheckResult(
                name="Metric correlation matrix",
                status="FAIL",
                blocking=True,
                detail=f"n_rows/n_rows_expected must be integer-like values: {ex}",
            )

        coverage_ok = payload.get("coverage_ok")
        if not isinstance(coverage_ok, bool):
            return CheckResult(
                name="Metric correlation matrix",
                status="FAIL",
                blocking=True,
                detail="coverage_ok must be a boolean",
            )
        if not coverage_ok:
            return CheckResult(
                name="Metric correlation matrix",
                status="FAIL",
                blocking=True,
                detail="coverage_ok=false in Task 11 artifact",
            )
        if n_rows != n_rows_expected:
            return CheckResult(
                name="Metric correlation matrix",
                status="FAIL",
                blocking=True,
                detail=f"n_rows mismatch inside artifact: n_rows={n_rows}, n_rows_expected={n_rows_expected}",
            )
        if n_rows != expected_rows_from_ic:
            return CheckResult(
                name="Metric correlation matrix",
                status="FAIL",
                blocking=True,
                detail=(
                    "Task 11 coverage mismatch vs IC labels: "
                    f"artifact n_rows={n_rows}, IC unique node_id={expected_rows_from_ic}"
                ),
            )

        n = len(expected_metrics)
        try:
            rho = np.asarray(payload.get("rho_matrix"), dtype=float)
            p_corr = np.asarray(payload.get("p_matrix_corrected"), dtype=float)
        except Exception as ex:
            return CheckResult(
                name="Metric correlation matrix",
                status="FAIL",
                blocking=True,
                detail=f"Invalid matrix payload: {ex}",
            )

        if rho.shape != (n, n) or p_corr.shape != (n, n):
            return CheckResult(
                name="Metric correlation matrix",
                status="FAIL",
                blocking=True,
                detail=f"Matrix shape mismatch: rho={rho.shape}, p_matrix_corrected={p_corr.shape}, expected={(n, n)}",
            )

        if not np.isfinite(rho).all() or not np.isfinite(p_corr).all():
            return CheckResult(
                name="Metric correlation matrix",
                status="FAIL",
                blocking=True,
                detail="rho_matrix or p_matrix_corrected contains non-finite values",
            )

        if np.any(p_corr < 0.0) or np.any(p_corr > 1.0):
            return CheckResult(
                name="Metric correlation matrix",
                status="FAIL",
                blocking=True,
                detail="p_matrix_corrected contains values outside [0,1]",
            )

        # Convention lock: diagonal corresponds to self-correlation (no test), must be 1.0.
        if not np.allclose(np.diag(p_corr), np.ones(n, dtype=float), atol=1e-12):
            return CheckResult(
                name="Metric correlation matrix",
                status="FAIL",
                blocking=True,
                detail="p_matrix_corrected diagonal must be 1.0 for all metrics",
            )

        if not np.allclose(rho, rho.T, atol=1e-12) or not np.allclose(p_corr, p_corr.T, atol=1e-12):
            return CheckResult(
                name="Metric correlation matrix",
                status="FAIL",
                blocking=True,
                detail="rho_matrix or p_matrix_corrected is not symmetric",
            )

        mapping = payload.get("column_mapping")
        if not isinstance(mapping, dict):
            return CheckResult(
                name="Metric correlation matrix",
                status="FAIL",
                blocking=True,
                detail="column_mapping must be an object",
            )

        missing_mapping = [m for m in expected_metrics if m not in mapping]
        if missing_mapping:
            return CheckResult(
                name="Metric correlation matrix",
                status="FAIL",
                blocking=True,
                detail=f"column_mapping missing canonical metrics: {missing_mapping}",
            )

        bet_source = str(mapping.get("betweenness_approx"))
        if bet_source not in {"betweenness_approx", "betweenness"}:
            return CheckResult(
                name="Metric correlation matrix",
                status="FAIL",
                blocking=True,
                detail=(
                    "column_mapping['betweenness_approx'] must map to source 'betweenness_approx' "
                    f"or 'betweenness'; got '{bet_source}'"
                ),
            )

        return CheckResult(
            name="Metric correlation matrix",
            status="PASS",
            blocking=True,
            detail=(
                f"Valid Task 11 artifact in {path} (n_rows={n_rows}, "
                f"n_rows_expected={n_rows_expected}, betweenness_source={bet_source})"
            ),
        )
    except Exception as ex:
        return CheckResult(
            name="Metric correlation matrix",
            status="FAIL",
            blocking=True,
            detail=f"Unexpected validation error (structured fail): {ex}",
        )


def _optional_existing_artifact_checks(repo: Path, active_nodes_count: int) -> list[CheckResult]:
    """Validate already-generated Track B artifacts if they exist.

    These checks are non-blocking for pre-start readiness but useful to catch
    stale/placeholder outputs that do not match final contract.
    """
    results: list[CheckResult] = []

    proxies = repo / "data/processed/diffusion_proxies.parquet"
    if proxies.exists():
        try:
            df_p = _load_parquet(proxies)
            missing = _require_columns(df_p, ["node_id", "one_hop_spread", "two_hop_spread"])
            if missing:
                results.append(
                    CheckResult(
                        name="Existing proxies schema",
                        status="WARN",
                        blocking=False,
                        detail=f"Missing columns {missing} in {proxies}",
                    )
                )
            else:
                na_one = int(df_p["one_hop_spread"].isna().sum())
                na_two = int(df_p["two_hop_spread"].isna().sum())
                rows = int(len(df_p))
                if rows < active_nodes_count or na_one == rows or na_two == rows:
                    results.append(
                        CheckResult(
                            name="Existing proxies schema",
                            status="WARN",
                            blocking=False,
                            detail=(
                                f"Proxies file appears placeholder/non-final: rows={rows} (active={active_nodes_count}), "
                                f"na_one_hop={na_one}, na_two_hop={na_two}"
                            ),
                        )
                    )
                else:
                    results.append(
                        CheckResult(
                            name="Existing proxies schema",
                            status="PASS",
                            blocking=False,
                            detail=f"Proxies appear real-mode complete in {proxies} (rows={rows})",
                        )
                    )
        except Exception as ex:
            results.append(
                CheckResult(
                    name="Existing proxies schema",
                    status="WARN",
                    blocking=False,
                    detail=f"Could not validate {proxies}: {ex}",
                )
            )

    null_summary = repo / "outputs/mapr2026_v3_results/null_model_typology_summary.json"
    if null_summary.exists():
        try:
            payload = json.loads(null_summary.read_text(encoding="utf-8"))
            required_keys = [
                "timestamp",
                "n_nodes",
                "n_realizations",
                "n_runs_per_node",
                "rho_mean",
                "rho_std",
                "hidden_betweenness_real_subgraph_mean",
                "hidden_betweenness_null_mean",
                "hidden_betweenness_null_std",
                "interpretation",
            ]
            missing = [k for k in required_keys if k not in payload]
            if missing:
                results.append(
                    CheckResult(
                        name="Existing null-model summary schema",
                        status="WARN",
                        blocking=False,
                        detail=(
                            f"Found placeholder/non-final schema in {null_summary}; missing keys {missing}. "
                            "This is OK before real implementation."
                        ),
                    )
                )
            else:
                results.append(
                    CheckResult(
                        name="Existing null-model summary schema",
                        status="PASS",
                        blocking=False,
                        detail=f"Schema matches final contract in {null_summary}",
                    )
                )
        except Exception as ex:
            results.append(
                CheckResult(
                    name="Existing null-model summary schema",
                    status="WARN",
                    blocking=False,
                    detail=f"Could not parse {null_summary}: {ex}",
                )
            )

    return results


def _print_results(title: str, results: list[CheckResult]) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    for r in results:
        tag = r.status.ljust(4)
        block = "BLOCK" if r.blocking else "INFO "
        print(f"[{tag}] [{block}] {r.name}: {r.detail}")


def _readiness_summary(results: list[CheckResult]) -> tuple[bool, int, int, int]:
    n_pass = sum(1 for r in results if r.status == "PASS")
    n_warn = sum(1 for r in results if r.status == "WARN")
    n_fail = sum(1 for r in results if r.status == "FAIL")
    blocking_fail = any((r.status == "FAIL") and r.blocking for r in results)
    return (not blocking_fail), n_pass, n_warn, n_fail


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight checks for MAPR2026 v3 Person 2 Track B")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON summary in addition to text output.",
    )
    parser.add_argument(
        "--perm-min-policy",
        type=int,
        default=DEFAULT_PERM_MIN_POLICY,
        help="Soft policy minimum for permutation count in Task 8/9 artifacts (non-blocking WARN if below).",
    )
    args = parser.parse_args()

    repo = _repo_root()

    # Core paths
    graph_active = repo / "data/processed/graph_active.edgelist"
    node_attrs = repo / "data/processed/node_attributes.parquet"
    community_features = repo / "data/processed/community_features.parquet"
    csr_npz = repo / "data/processed/graph_csr.npz"
    ic_scores = repo / "data/processed/ic_scores_primary.parquet"
    centrality_table = repo / "data/processed/centrality_table.parquet"
    day1_dir = repo / "outputs/day1_benchmark"
    day1_note = repo / "docs/day1_decisions.md"
    assumptions_doc = repo / "docs/assumptions_limitations.md"
    results_dir = repo / "outputs/mapr2026_v3_results"

    checks: list[CheckResult] = []

    # 1) Environment checks
    checks.append(_check_python_version())
    checks.append(_check_imports())

    # 2) Independent-start checks (Person 2 can start coding/skeleton work)
    checks.append(_check_file_exists(graph_active, "Stage-0 active graph"))
    checks.append(_check_node_attributes_base(node_attrs))
    checks.append(_check_outputs_dir(results_dir))

    # Active-node universe for coverage checks.
    active_nodes: set[str] = set()
    if graph_active.exists():
        try:
            active_nodes = _read_active_nodes(graph_active)
        except Exception as ex:
            checks.append(
                CheckResult(
                    name="Stage-0 active graph",
                    status="FAIL",
                    blocking=True,
                    detail=f"Failed to parse active edgelist for coverage checks: {ex}",
                )
            )

    # 3) Real-mode/gating checks (needed to run full Track B outputs)
    checks.append(_check_day1_benchmark_artifacts(day1_dir))
    checks.append(_check_csr_contract(csr_npz))
    checks.append(_check_ic_scores_contract(ic_scores))
    checks.append(_check_community_features(node_attrs, community_features, active_nodes))
    checks.append(_check_centrality_contract(centrality_table, active_nodes))
    checks.append(_check_day1_decisions(day1_note))
    checks.append(_check_input_join_coherence(ic_scores, node_attrs, community_features, centrality_table))
    checks.append(_check_stage5_null_package(results_dir))
    checks.append(_check_metric_correlation_matrix(results_dir, ic_scores_path=ic_scores))
    checks.append(_soft_check_permutation_quality(results_dir, min_permutations_policy=int(args.perm_min_policy)))
    checks.append(_check_lifetime_if_problem_handled(results_dir=results_dir, assumptions_doc=assumptions_doc))

    # 4) Non-blocking checks on currently existing artifacts (if present)
    checks.extend(_optional_existing_artifact_checks(repo, active_nodes_count=len(active_nodes)))

    _print_results("MAPR2026 v3 - Person 2 Preflight Report", checks)
    ready, n_pass, n_warn, n_fail = _readiness_summary(checks)

    independent_fail = any((r.status == "FAIL") and r.blocking and r.name in INDEPENDENT_CHECKS for r in checks)
    real_mode_fail = any((r.status == "FAIL") and r.blocking and r.name in REAL_MODE_CHECKS for r in checks)
    completion_ready_fail = any(
        (r.status == "FAIL") and r.blocking and r.name in COMPLETION_READY_CHECKS for r in checks
    )

    print("\n" + "-" * 80)
    print(f"Summary: PASS={n_pass}, WARN={n_warn}, FAIL={n_fail}")
    print(f"Ready for independent Track B coding (dry-run/skeleton): {'PASS' if not independent_fail else 'FAIL'}")
    print(f"Ready for full real-mode Track B execution: {'PASS' if not real_mode_fail else 'FAIL'}")
    print(f"Ready for completion-ready declaration: {'PASS' if not completion_ready_fail else 'FAIL'}")
    print("Overall blocking status:", "PASS" if (ready and not completion_ready_fail) else "FAIL")
    print("-" * 80)

    if args.json:
        payload: dict[str, Any] = {
            "summary": {
                "pass": n_pass,
                "warn": n_warn,
                "fail": n_fail,
                "overall_blocking_status": "PASS" if (ready and not completion_ready_fail) else "FAIL",
                "ready_independent": not independent_fail,
                "ready_real_mode": not real_mode_fail,
                "ready_completion": not completion_ready_fail,
            },
            "checks": [
                {
                    "name": r.name,
                    "status": r.status,
                    "blocking": r.blocking,
                    "detail": r.detail,
                }
                for r in checks
            ],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))

    return 0 if not completion_ready_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
