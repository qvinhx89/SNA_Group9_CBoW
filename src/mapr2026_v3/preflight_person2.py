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
    expected_tokens = ["n_seeds", "n_runs", "narrative"]
    missing_tokens = [tok for tok in expected_tokens if tok not in text]
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
