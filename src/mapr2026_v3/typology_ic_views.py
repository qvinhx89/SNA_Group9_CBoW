"""MAPR2026 v3 — Typology: IC high/low × views high/low.

Owner: Person 2 (typology)

Inputs
------
- data/processed/ic_scores_primary.parquet (labeled subset; Option B compatible)
- data/processed/node_attributes.parquet
- data/processed/community_features.parquet (coverage check for downstream profiling)

Output (contract)
---------------
- data/processed/typology_labels_ic_views.parquet
  columns: node_id, typology_label, ic_high, views_high, ic_score_mean, views
- outputs/mapr2026_v3_results/typology_quadrant_report.json
    fields: timestamp, n_total, ic_threshold, views_threshold, quadrants,
                    min_quadrant_ok, two_sample_applied

Behavior
--------
- Real mode builds typology from labeled IC nodes and writes quadrant report.
- Dry-run can fall back to SIS/random IC mock to unblock parallel coding.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score
from scipy import stats
from statsmodels.stats.multitest import multipletests

from _shared import PATHS, ensure_parent, load_csr_npz, now_iso, require_columns, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MAPR2026 v3 typology (IC×views) scaffold")
    p.add_argument("--ic", default=PATHS.ic_scores)
    p.add_argument("--node-attrs", default=PATHS.node_attributes)
    p.add_argument("--community-features", default="data/processed/community_features.parquet")
    p.add_argument("--centrality-table", default="data/processed/centrality_table.parquet")
    p.add_argument("--kshell-table", default="data/processed/kshell_table.parquet")
    p.add_argument("--out", default=PATHS.typology)
    p.add_argument("--quadrant-json", default="outputs/mapr2026_v3_results/typology_quadrant_report.json")
    p.add_argument("--structural-csv", default="outputs/mapr2026_v3_results/structural_profiling.csv")
    p.add_argument("--lifetime-json", default="outputs/mapr2026_v3_results/lifetime_validation.json")
    p.add_argument("--language-json", default="outputs/mapr2026_v3_results/language_validation.json")
    p.add_argument("--pct", type=float, default=0.10, help="Top-pct threshold (default 10%)")
    p.add_argument("--min-quadrant-size", type=int, default=150)
    p.add_argument("--delta-threshold", type=float, default=0.20)
    p.add_argument("--lifetime-min-group-size", type=int, default=10)
    p.add_argument("--language-min-group-size", type=int, default=10)
    p.add_argument(
        "--require-min-quadrant",
        action="store_true",
        help="Fail with non-zero exit if any quadrant size is below --min-quadrant-size",
    )
    p.add_argument(
        "--force-structural-profiling",
        action="store_true",
        help="Run structural profiling even when min_quadrant_ok is false",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _load_ic_or_fallback(ic_path: Path, attrs_path: Path, dry_run: bool) -> pd.DataFrame:
    """Load IC scores. In dry-run mode, falls back to SIS/random if IC file is missing."""
    if ic_path.exists():
        df = pd.read_parquet(ic_path)
        require_columns(df, ["node_id", "ic_score_mean"], "ic_scores")
        return df

    if not dry_run:
        raise FileNotFoundError(f"Missing IC scores: {ic_path}")

    # Dry-run fallback: try SIS table, then random mock.
    sis_path = Path("data/processed/sis_table.parquet")
    if sis_path.exists():
        df_sis = pd.read_parquet(sis_path)
        score_col = next((c for c in ["sis_score", "sis", "score", "pagerank"] if c in df_sis.columns), None)
        if "node_id" in df_sis.columns and score_col is not None:
            print(f"[dry-run] IC scores not found; using {sis_path} column '{score_col}' as mock ic_score_mean")
            return df_sis[["node_id", score_col]].rename(columns={score_col: "ic_score_mean"})

    if attrs_path.exists():
        df_attrs = pd.read_parquet(attrs_path)
        if "node_id" in df_attrs.columns:
            import numpy as np
            print("[dry-run] IC scores not found; generating random mock ic_score_mean from node_attributes")
            rng = np.random.default_rng(42)
            return pd.DataFrame({"node_id": df_attrs["node_id"].astype(str),
                                  "ic_score_mean": rng.random(size=len(df_attrs))})

    raise FileNotFoundError(
        f"Dry-run fallback failed: IC scores ({ic_path}), SIS table, and node_attributes all missing. "
        "Run Person 1's dry-run first: python ic_labels_primary.py --dry-run"
    )


def _validate_community_coverage(community_path: Path, labeled_node_ids: pd.Series) -> None:
    """Ensure community features exist and cover all labeled nodes.

    Task 4 output does not include community fields directly, but this guard catches
    upstream data drift early and keeps Track B artifacts aligned for Task 5.
    """
    if not community_path.exists():
        raise FileNotFoundError(
            f"Missing community features: {community_path}. "
            "Run community detection pipeline before real typology build."
        )

    df_comm = pd.read_parquet(community_path)
    require_columns(df_comm, ["node_id", "community_id", "cross_community_edge_fraction"], "community_features")

    comm_node_ids = df_comm["node_id"].astype(str)
    missing = set(labeled_node_ids.astype(str).tolist()) - set(comm_node_ids.tolist())
    if missing:
        raise ValueError(
            "community_features.parquet does not cover all labeled IC nodes: "
            f"missing={len(missing)}"
        )


def _build_quadrant_report(
    df: pd.DataFrame,
    ic_thresh: float,
    views_thresh: float,
    min_quadrant_size: int,
) -> dict[str, Any]:
    quadrant_order = ["True", "Hidden", "Overrated", "Non"]
    counts = df["typology_label"].value_counts().to_dict()
    n_total = int(len(df))

    quadrants = {
        q: {
            "n": int(counts.get(q, 0)),
            "pct": float(counts.get(q, 0) / n_total) if n_total > 0 else 0.0,
        }
        for q in quadrant_order
    }

    min_quadrant_ok = all(v["n"] >= int(min_quadrant_size) for v in quadrants.values())

    return {
        "timestamp": now_iso(),
        "n_total": n_total,
        "ic_threshold": float(ic_thresh),
        "views_threshold": float(views_thresh),
        "quadrants": quadrants,
        "min_quadrant_ok": bool(min_quadrant_ok),
        "two_sample_applied": False,
    }


def _assign_typology_label(ic_high: bool, views_high: bool) -> str:
    if ic_high and views_high:
        return "True"
    if ic_high and (not views_high):
        return "Hidden"
    if (not ic_high) and views_high:
        return "Overrated"
    return "Non"


def _assign_typology_labels(df: pd.DataFrame) -> pd.Series:
    require_columns(df, ["ic_high", "views_high"], "typology_frame")
    return df.apply(
        lambda row: _assign_typology_label(bool(row["ic_high"]), bool(row["views_high"])),
        axis=1,
    )


def _cliffs_delta_from_u(u_stat: float, n1: int, n2: int) -> float:
    denom = (n1 * n2) / 2.0
    if denom <= 0.0:
        raise ValueError("Cannot compute Cliff's delta with empty comparison group")
    return float((float(u_stat) - denom) / denom)


def _partial_spearman_rho(ic_score: np.ndarray, life_time: np.ndarray, degree: np.ndarray) -> tuple[float, float]:
    """Compute Spearman(ic_score, life_time | degree) via residualized ranks."""
    ic = np.asarray(ic_score, dtype=float)
    lft = np.asarray(life_time, dtype=float)
    deg = np.asarray(degree, dtype=float)

    if not (len(ic) == len(lft) == len(deg)):
        raise ValueError("partial_spearman inputs must have equal length")
    if len(ic) < 3:
        raise ValueError("partial_spearman requires at least 3 rows")

    rank_ic = stats.rankdata(ic)
    rank_lft = stats.rankdata(lft)
    rank_deg = stats.rankdata(deg)

    def _residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
        x_ = np.column_stack([np.ones(len(x)), x])
        beta = np.linalg.lstsq(x_, y, rcond=None)[0]
        return y - x_ @ beta

    res_ic = _residualize(rank_ic, rank_deg)
    res_lft = _residualize(rank_lft, rank_deg)

    rho_result: Any = stats.spearmanr(res_ic, res_lft)
    rho = float(rho_result.correlation) if hasattr(rho_result, "correlation") else float(rho_result[0])
    p_val = float(rho_result.pvalue) if hasattr(rho_result, "pvalue") else float(rho_result[1])
    if np.isnan(rho) or np.isnan(p_val):
        return 0.0, 1.0
    return rho, p_val


def _compute_lifetime_validation(df: pd.DataFrame, min_group_size: int = 10) -> dict[str, Any]:
    """Compute Task 6 lifetime validation summary from labeled typology rows."""
    require_columns(df, ["typology_label", "ic_score_mean", "degree", "life_time"], "typology_lifetime_frame")
    if int(min_group_size) < 1:
        raise ValueError("lifetime min_group_size must be >= 1")

    work = df[["typology_label", "ic_score_mean", "degree", "life_time"]].copy()
    for col in ["ic_score_mean", "degree", "life_time"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    if work[["ic_score_mean", "degree", "life_time"]].isna().any().any():
        na_counts = work[["ic_score_mean", "degree", "life_time"]].isna().sum().to_dict()
        raise ValueError(f"Lifetime validation requires non-missing numeric columns: {na_counts}")

    rho, p_val = _partial_spearman_rho(
        ic_score=work["ic_score_mean"].to_numpy(dtype=float),
        life_time=work["life_time"].to_numpy(dtype=float),
        degree=work["degree"].to_numpy(dtype=float),
    )

    work["degree_quintile"] = pd.qcut(work["degree"], q=5, labels=False, duplicates="drop")

    quintile_results: list[dict[str, Any]] = []
    tested_indexes: list[int] = []
    tested_p_values: list[float] = []

    quintiles = sorted(work["degree_quintile"].dropna().astype(int).unique().tolist())
    for q in quintiles:
        q_df = work[work["degree_quintile"].astype(float) == float(q)]
        hidden = q_df[q_df["typology_label"] == "Hidden"]["life_time"].to_numpy(dtype=float)
        non_hidden = q_df[q_df["typology_label"] != "Hidden"]["life_time"].to_numpy(dtype=float)

        row: dict[str, Any] = {
            "quintile": int(q),
            "n_hidden": int(len(hidden)),
            "n_non_hidden": int(len(non_hidden)),
            "p_raw": 1.0,
            "p_corrected": 1.0,
            "cliffs_delta": 0.0,
            "significant": False,
        }

        if len(hidden) >= int(min_group_size) and len(non_hidden) >= int(min_group_size):
            mwu = stats.mannwhitneyu(hidden, non_hidden, alternative="two-sided")
            u_stat = float(mwu.statistic)
            p_raw = float(mwu.pvalue)
            row["p_raw"] = p_raw
            row["cliffs_delta"] = _cliffs_delta_from_u(u_stat, len(hidden), len(non_hidden))

            tested_indexes.append(len(quintile_results))
            tested_p_values.append(p_raw)

        quintile_results.append(row)

    if tested_p_values:
        _, p_corrected, _, _ = multipletests(tested_p_values, method="fdr_bh")
        for idx, p_corr in zip(tested_indexes, p_corrected):
            p_corr_f = float(p_corr)
            quintile_results[idx]["p_corrected"] = p_corr_f
            quintile_results[idx]["significant"] = bool(p_corr_f < 0.05)

    n_tested = int(len(tested_indexes))
    n_significant = int(sum(1 for row in quintile_results if bool(row["significant"])))

    return {
        "partial_spearman_rho": float(rho),
        "partial_spearman_p": float(p_val),
        "n_quintiles_tested": n_tested,
        "n_quintiles_significant": n_significant,
        "success": bool(n_significant >= 3),
        "quintile_results": quintile_results,
    }


def _compute_language_validation(
    typology_df: pd.DataFrame,
    node_attrs_path: Path,
    community_path: Path,
    csr_path: Path,
    min_group_size: int = 10,
) -> dict[str, Any]:
    """Compute language fallback corroboration when Task 6 lifetime gate fails."""
    if int(min_group_size) < 1:
        raise ValueError("language min_group_size must be >= 1")

    if not node_attrs_path.exists():
        raise FileNotFoundError(f"Missing node attributes for language fallback: {node_attrs_path}")
    if not community_path.exists():
        raise FileNotFoundError(f"Missing community features for language fallback: {community_path}")

    require_columns(typology_df, ["node_id", "typology_label"], "typology_labels")

    df_attrs = pd.read_parquet(node_attrs_path)
    require_columns(df_attrs, ["node_id", "language"], "node_attributes")
    df_attrs = df_attrs[["node_id", "language"]].copy()
    df_attrs["node_id"] = df_attrs["node_id"].astype(str)
    df_attrs["language"] = df_attrs["language"].fillna("__missing__").astype(str)

    df_comm = pd.read_parquet(community_path)
    require_columns(df_comm, ["node_id", "community_id"], "community_features")
    df_comm = df_comm[["node_id", "community_id"]].copy()
    df_comm["node_id"] = df_comm["node_id"].astype(str)
    df_comm["community_id"] = df_comm["community_id"].fillna("__missing__").astype(str)

    df_typ = typology_df[["node_id", "typology_label"]].copy()
    df_typ["node_id"] = df_typ["node_id"].astype(str)

    nmi_frame = df_typ.merge(df_attrs, on="node_id", how="left").merge(df_comm, on="node_id", how="left")
    nmi_frame["language"] = nmi_frame["language"].fillna("__missing__").astype(str)
    nmi_frame["community_id"] = nmi_frame["community_id"].fillna("__missing__").astype(str)

    if nmi_frame["language"].nunique() <= 1 or nmi_frame["community_id"].nunique() <= 1:
        nmi_value = 0.0
    else:
        nmi_value = float(
            normalized_mutual_info_score(
                nmi_frame["community_id"].to_numpy(dtype=str),
                nmi_frame["language"].to_numpy(dtype=str),
            )
        )

    csr = load_csr_npz(csr_path)
    indptr = csr["indptr"]
    indices = csr["indices"]
    node_ids = csr["node_ids"]

    lang_map = dict(zip(df_attrs["node_id"].tolist(), df_attrs["language"].tolist()))
    lang_aligned = np.array([lang_map.get(str(n), "__missing__") for n in node_ids], dtype=object)
    node_to_row = {str(n): i for i, n in enumerate(node_ids.tolist())}

    def _neighbor_entropy_for_node(node_id: str) -> float:
        row = node_to_row.get(str(node_id))
        if row is None:
            return float("nan")
        start = int(indptr[row])
        end = int(indptr[row + 1])
        if end <= start:
            return float("nan")
        neigh_rows = indices[start:end]
        neigh_langs = lang_aligned[neigh_rows]
        neigh_langs = neigh_langs[neigh_langs != "__missing__"]
        if len(neigh_langs) == 0:
            return float("nan")
        _, counts = np.unique(neigh_langs, return_counts=True)
        probs = counts / counts.sum()
        return float(-np.sum(probs * np.log(probs)))

    hidden_ids = df_typ[df_typ["typology_label"] == "Hidden"]["node_id"].tolist()
    overrated_ids = df_typ[df_typ["typology_label"] == "Overrated"]["node_id"].tolist()

    hidden_entropy = np.array([_neighbor_entropy_for_node(n) for n in hidden_ids], dtype=float)
    hidden_entropy = hidden_entropy[np.isfinite(hidden_entropy)]
    overrated_entropy = np.array([_neighbor_entropy_for_node(n) for n in overrated_ids], dtype=float)
    overrated_entropy = overrated_entropy[np.isfinite(overrated_entropy)]

    diversity_summary: dict[str, Any] = {
        "metric": "neighbor_language_entropy",
        "comparator": "Hidden_vs_Overrated",
        "n_hidden": int(len(hidden_entropy)),
        "n_overrated": int(len(overrated_entropy)),
        "hidden_mean": float(np.mean(hidden_entropy)) if len(hidden_entropy) > 0 else float("nan"),
        "overrated_mean": float(np.mean(overrated_entropy)) if len(overrated_entropy) > 0 else float("nan"),
        "mwu_stat": 0.0,
        "p_raw": 1.0,
        "p_corrected": 1.0,
        "cliffs_delta": 0.0,
        "significant": False,
    }

    if len(hidden_entropy) >= int(min_group_size) and len(overrated_entropy) >= int(min_group_size):
        mwu = stats.mannwhitneyu(hidden_entropy, overrated_entropy, alternative="two-sided")
        u_stat = float(mwu.statistic)
        p_raw = float(mwu.pvalue)
        delta = _cliffs_delta_from_u(u_stat, len(hidden_entropy), len(overrated_entropy))
        _, p_corrected, _, _ = multipletests([p_raw], method="fdr_bh")
        p_corr = float(p_corrected[0])

        diversity_summary["mwu_stat"] = u_stat
        diversity_summary["p_raw"] = p_raw
        diversity_summary["p_corrected"] = p_corr
        diversity_summary["cliffs_delta"] = float(delta)
        diversity_summary["significant"] = bool(p_corr < 0.05 and abs(float(delta)) >= 0.20)

    return {
        "timestamp": now_iso(),
        "n_nodes_labeled": int(len(df_typ)),
        "nmi_community_language_labeled": nmi_value,
        "neighbor_language_diversity": diversity_summary,
    }


def _build_structural_frame(
    typology_df: pd.DataFrame,
    node_attrs_path: Path,
    centrality_path: Path,
    kshell_path: Path,
    community_path: Path,
) -> pd.DataFrame:
    """Join all profile features required by Task 5 onto typology labels."""
    if not node_attrs_path.exists():
        raise FileNotFoundError(f"Missing node attributes: {node_attrs_path}")
    if not centrality_path.exists():
        raise FileNotFoundError(f"Missing centrality table: {centrality_path}")
    if not community_path.exists():
        raise FileNotFoundError(f"Missing community features: {community_path}")

    df_typ = typology_df[["node_id", "typology_label"]].copy()
    df_typ["node_id"] = df_typ["node_id"].astype(str)

    df_attrs = pd.read_parquet(node_attrs_path)
    require_columns(df_attrs, ["node_id", "degree", "life_time"], "node_attributes")
    df_attrs = df_attrs[["node_id", "degree", "life_time"]].copy()
    df_attrs["node_id"] = df_attrs["node_id"].astype(str)

    df_cent = pd.read_parquet(centrality_path)
    require_columns(df_cent, ["node_id", "pagerank", "betweenness"], "centrality_table")
    df_cent = df_cent[["node_id", "pagerank", "betweenness"] + (["kshell"] if "kshell" in df_cent.columns else [])].copy()
    df_cent["node_id"] = df_cent["node_id"].astype(str)

    if "kshell" not in df_cent.columns:
        if not kshell_path.exists():
            raise FileNotFoundError(
                f"Missing kshell source: centrality has no 'kshell' and file not found {kshell_path}"
            )
        df_ks = pd.read_parquet(kshell_path)
        require_columns(df_ks, ["node_id", "kshell"], "kshell_table")
        df_ks = df_ks[["node_id", "kshell"]].copy()
        df_ks["node_id"] = df_ks["node_id"].astype(str)
        df_cent = df_cent.merge(df_ks, on="node_id", how="left")

    df_comm = pd.read_parquet(community_path)
    require_columns(df_comm, ["node_id", "cross_community_edge_fraction"], "community_features")
    df_comm = df_comm[["node_id", "cross_community_edge_fraction"]].copy()
    df_comm["node_id"] = df_comm["node_id"].astype(str)

    merged = (
        df_typ.merge(df_attrs, on="node_id", how="left")
        .merge(df_cent[["node_id", "pagerank", "betweenness", "kshell"]], on="node_id", how="left")
        .merge(df_comm, on="node_id", how="left")
    )

    required = [
        "degree",
        "pagerank",
        "kshell",
        "betweenness",
        "cross_community_edge_fraction",
        "life_time",
    ]
    missing_counts = {c: int(merged[c].isna().sum()) for c in required}
    if any(v > 0 for v in missing_counts.values()):
        raise ValueError(
            "Structural profiling feature join has missing values for typology nodes: "
            f"{missing_counts}"
        )

    return merged


def _compute_structural_profiling(df: pd.DataFrame, delta_threshold: float = 0.20) -> pd.DataFrame:
    """Compute Task 5 structural profiling: MWU + Cliff's delta + BH-FDR."""
    require_columns(
        df,
        [
            "typology_label",
            "degree",
            "pagerank",
            "kshell",
            "betweenness",
            "cross_community_edge_fraction",
            "life_time",
        ],
        "typology_structural_frame",
    )

    hidden_df = df[df["typology_label"] == "Hidden"]
    overrated_df = df[df["typology_label"] == "Overrated"]

    if len(hidden_df) == 0 or len(overrated_df) == 0:
        raise ValueError(
            "Structural profiling requires non-empty Hidden and Overrated groups. "
            f"Got Hidden={len(hidden_df)} Overrated={len(overrated_df)}"
        )

    features = [
        "degree",
        "pagerank",
        "kshell",
        "betweenness",
        "cross_community_edge_fraction",
        "life_time",
    ]

    rows: list[dict[str, Any]] = []
    p_raws: list[float] = []
    for feat in features:
        h_vals = pd.to_numeric(hidden_df[feat], errors="coerce").dropna().to_numpy(dtype=float)
        o_vals = pd.to_numeric(overrated_df[feat], errors="coerce").dropna().to_numpy(dtype=float)

        if len(h_vals) == 0 or len(o_vals) == 0:
            raise ValueError(f"Feature '{feat}' has empty group after numeric coercion")

        mwu = stats.mannwhitneyu(h_vals, o_vals, alternative="two-sided")
        u_stat = float(mwu.statistic)
        p_raw = float(mwu.pvalue)
        cliffs_delta = _cliffs_delta_from_u(u_stat, len(h_vals), len(o_vals))

        rows.append(
            {
                "feature": feat,
                "group_hidden_mean": float(np.mean(h_vals)),
                "group_overrated_mean": float(np.mean(o_vals)),
                "mwu_stat": u_stat,
                "p_raw": p_raw,
                "cliffs_delta": cliffs_delta,
            }
        )
        p_raws.append(p_raw)

    _, p_corrected, _, _ = multipletests(p_raws, method="fdr_bh")

    for i, row in enumerate(rows):
        p_corr = float(p_corrected[i])
        row["p_corrected"] = p_corr
        row["significant"] = bool(p_corr < 0.05 and abs(float(row["cliffs_delta"])) >= float(delta_threshold))

    out = pd.DataFrame(rows)
    out = out[
        [
            "feature",
            "group_hidden_mean",
            "group_overrated_mean",
            "mwu_stat",
            "p_raw",
            "p_corrected",
            "cliffs_delta",
            "significant",
        ]
    ].copy()
    return out


def main() -> None:
    args = parse_args()

    ic_path = Path(args.ic)
    attrs_path = Path(args.node_attrs)
    community_path = Path(args.community_features)
    centrality_path = Path(args.centrality_table)
    kshell_path = Path(args.kshell_table)
    quadrant_json_path = Path(args.quadrant_json)
    structural_csv_path = Path(args.structural_csv)
    lifetime_json_path = Path(args.lifetime_json)
    language_json_path = Path(args.language_json)
    if not attrs_path.exists():
        raise FileNotFoundError(f"Missing node attributes: {attrs_path}")

    if not (0.0 < float(args.pct) < 1.0):
        raise ValueError("--pct must be in (0, 1)")
    if int(args.min_quadrant_size) < 1:
        raise ValueError("--min-quadrant-size must be >= 1")
    if float(args.delta_threshold) < 0.0:
        raise ValueError("--delta-threshold must be >= 0")
    if int(args.lifetime_min_group_size) < 1:
        raise ValueError("--lifetime-min-group-size must be >= 1")
    if int(args.language_min_group_size) < 1:
        raise ValueError("--language-min-group-size must be >= 1")

    df_ic = _load_ic_or_fallback(ic_path, attrs_path, dry_run=args.dry_run)
    df_ic["node_id"] = df_ic["node_id"].astype(str)
    if df_ic["node_id"].nunique() != len(df_ic):
        raise ValueError("IC scores contain duplicate node_id rows")

    df_attrs = pd.read_parquet(attrs_path)
    require_columns(df_attrs, ["node_id", "views", "degree", "life_time"], "node_attributes")
    df_attrs["node_id"] = df_attrs["node_id"].astype(str)

    if not args.dry_run:
        _validate_community_coverage(community_path, df_ic["node_id"])

    df = df_ic[["node_id", "ic_score_mean"]].merge(df_attrs[["node_id", "views"]], on="node_id", how="inner")
    if len(df) == 0:
        raise ValueError("No overlap between IC scores and node attributes")

    if df["node_id"].nunique() != len(df):
        raise ValueError("Merged typology frame contains duplicate node_id rows")

    # Simple typology rules (can be refined by Person 2).
    ic_thresh = df["ic_score_mean"].quantile(1.0 - args.pct)
    views_thresh = df["views"].quantile(1.0 - args.pct)

    df["ic_high"] = df["ic_score_mean"] >= ic_thresh
    df["views_high"] = df["views"] >= views_thresh

    df["typology_label"] = _assign_typology_labels(df)

    out = df[["node_id", "typology_label", "ic_high", "views_high", "ic_score_mean", "views"]].copy()
    require_columns(
        out,
        ["node_id", "typology_label", "ic_high", "views_high", "ic_score_mean", "views"],
        "typology_labels",
    )

    ensure_parent(args.out)
    out.to_parquet(args.out, index=False)

    report = _build_quadrant_report(
        out,
        ic_thresh=float(ic_thresh),
        views_thresh=float(views_thresh),
        min_quadrant_size=int(args.min_quadrant_size),
    )
    ensure_parent(quadrant_json_path)
    write_json(quadrant_json_path, report)

    lifetime_frame = out[["node_id", "typology_label", "ic_score_mean"]].merge(
        df_attrs[["node_id", "degree", "life_time"]],
        on="node_id",
        how="left",
    )
    lifetime_summary = _compute_lifetime_validation(
        lifetime_frame,
        min_group_size=int(args.lifetime_min_group_size),
    )
    ensure_parent(lifetime_json_path)
    write_json(lifetime_json_path, lifetime_summary)

    language_summary: dict[str, Any] | None = None
    lifetime_problem_triggered = bool(
        (float(lifetime_summary["partial_spearman_rho"]) < 0.05)
        or (int(lifetime_summary["n_quintiles_significant"]) < 3)
    )
    if (not args.dry_run) and lifetime_problem_triggered:
        language_summary = _compute_language_validation(
            typology_df=out,
            node_attrs_path=attrs_path,
            community_path=community_path,
            csr_path=Path(PATHS.csr_npz),
            min_group_size=int(args.language_min_group_size),
        )
        language_summary["trigger_condition"] = {
            "partial_spearman_rho_lt_0_05": bool(float(lifetime_summary["partial_spearman_rho"]) < 0.05),
            "n_quintiles_significant_lt_3": bool(int(lifetime_summary["n_quintiles_significant"]) < 3),
        }
        language_summary["note"] = (
            "IF PROBLEM fallback triggered: lifetime validation limited. "
            "Language corroboration is supplementary and does not replace lifetime_validation.json."
        )
        ensure_parent(language_json_path)
        write_json(language_json_path, language_summary)

    if args.require_min_quadrant and (not bool(report["min_quadrant_ok"])):
        raise ValueError(
            "Quadrant minimum-size gate failed: "
            f"min_quadrant_ok=False with min_quadrant_size={args.min_quadrant_size}."
        )

    if args.dry_run:
        print(f"[OK] Wrote typology (dry-run OK): {args.out} (timestamp={now_iso()})")
        print(f"[OK] Wrote quadrant report: {quadrant_json_path}")
        print(f"[OK] Wrote life_time validation: {lifetime_json_path}")
        return

    if bool(report["min_quadrant_ok"]) or args.force_structural_profiling:
        struct_frame = _build_structural_frame(
            typology_df=out,
            node_attrs_path=attrs_path,
            centrality_path=centrality_path,
            kshell_path=kshell_path,
            community_path=community_path,
        )
        structural_df = _compute_structural_profiling(
            struct_frame,
            delta_threshold=float(args.delta_threshold),
        )
        ensure_parent(structural_csv_path)
        structural_df.to_csv(structural_csv_path, index=False)
        print(f"[OK] Wrote structural profiling: {structural_csv_path} rows={len(structural_df)}")
    else:
        print(
            "[WARN] Structural profiling skipped because min_quadrant_ok=False. "
            "Run two-sample strategy first or pass --force-structural-profiling."
        )

    print(f"[OK] Wrote typology labels: {args.out} rows={len(out)}")
    print(f"[OK] Wrote quadrant report: {quadrant_json_path}")
    print(
        "[OK] Wrote life_time validation: "
        f"{lifetime_json_path} "
        f"(n_quintiles_tested={lifetime_summary['n_quintiles_tested']}, "
        f"n_quintiles_significant={lifetime_summary['n_quintiles_significant']}, "
        f"success={lifetime_summary['success']})"
    )
    if lifetime_problem_triggered and language_summary is not None:
        print(
            "[OK] Wrote language fallback validation: "
            f"{language_json_path} "
            f"(nmi_community_language_labeled={language_summary['nmi_community_language_labeled']:.4f}, "
            f"comparator={language_summary['neighbor_language_diversity']['comparator']})"
        )
    if not bool(report["min_quadrant_ok"]):
        print(
            "[WARN] min_quadrant_ok=False. "
            "Per plan, trigger two-sample strategy before structural profiling sign-off."
        )


if __name__ == "__main__":
    main()
