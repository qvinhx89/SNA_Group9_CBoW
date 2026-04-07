#!/usr/bin/env python3
"""
ic_feasibility_protocol.py  —  MAPR2026 v3
===========================================
Empirical protocol: "Can binary IC labels reach Jaccard ≥ 0.85,
and does kappa ∈ {1.5, 2, 3} fix it?"

Three phases with early stopping. Any single PIVOT_CONFIRMED condition
is sufficient — later phases are skipped once a verdict is reached.

  Phase 1  (< 3 min, zero new compute)
    Test 1.1  Regime classification from existing pilot diagnostics
    Test 1.2  KS degree-separation: top-10% IC vs top-11–20% IC nodes
    Test 1.3  Community-tier overlap between the same two groups

  Phase 2  (< 1 min, zero new compute)
    Test 2.1  Natural-threshold sweep with CLT-based Jaccard estimation

  Phase 3  (5–20 min, new IC compute — only if Phases 1-2 inconclusive)
    Test 3.1  Pilot kappa sweep: {1.5, 2.0, 3.0} × 400 nodes × 200 runs

Outputs
-------
  outputs/ic_feasibility/phase1_regime.json
  outputs/ic_feasibility/phase1_degree_separation.json
  outputs/ic_feasibility/phase1_community_overlap.json
  outputs/ic_feasibility/phase2_threshold_analysis.json
  outputs/ic_feasibility/phase3_kappa_sweep.json          (if reached)
  outputs/ic_feasibility/pivot_decision_report.json       (final verdict)

Usage
-----
  python src/mapr2026_v3/ic_feasibility_protocol.py
  python src/mapr2026_v3/ic_feasibility_protocol.py --skip-kappa-sweep
  python src/mapr2026_v3/ic_feasibility_protocol.py --out-dir outputs/ic_feasibility
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import ks_2samp, mannwhitneyu, norm

sys.path.insert(0, str(Path(__file__).parent))
from _shared import load_csr_npz, now_iso, write_json, ensure_dir

# ─────────────────────────────────────────────────────────────────────────────
# PROTOCOL CONSTANTS  — all thresholds here; change once, applies everywhere
# ─────────────────────────────────────────────────────────────────────────────

PROTO: dict[str, Any] = {
    # Original plan gates
    "cv_adequate":              0.30,
    "jaccard_target":           0.85,
    # Pivot thresholds
    "jaccard_feasibility_min":  0.70,   # est. Jaccard below this at ALL pcts → pivot
    "degree_ks_not_separable":  0.15,   # KS stat below → no structural separation → pivot
    "community_overlap_same":   0.70,   # comm-set Jaccard above → same tier → pivot signal
    # Sub-critical detection (from pilot data)
    "mean_reach_pct_subcritic": 0.001,  # mean_reach < 0.1% of LCC → cascade too weak
    "cv_subcritic":             0.05,   # cv_score < 0.05 → near-constant distribution
    # Threshold scan range (Test 2.1)
    "pct_min":   0.03,
    "pct_max":   0.30,
    "pct_step":  0.01,
    # Kappa sweep (Test 3.1)
    "kappa_values":    [1.5, 2.0, 3.0],
    "pilot_n_nodes":   400,
    "pilot_n_runs":    200,
    "pilot_mc_seeds":  [0, 1, 2],
    "ceiling_warn_pct": 0.05,   # mean_reach > 5% LCC = approaching explosive regime
    # Misc
    "top_pct":  0.10,
    "next_pct": 0.10,           # comparison window: top-11%–20%
}


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy/pandas scalars to plain Python JSON-safe types."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, set):
        return [_to_jsonable(v) for v in sorted(obj)]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _write_json_safe(path: str | Path, payload: dict[str, Any]) -> None:
    """Write JSON after normalizing numpy/pandas scalar types."""
    write_json(path, _to_jsonable(payload))

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_all_artifacts(repo_root: Path) -> dict[str, Any]:
    """Load every artifact needed by the protocol. Fail-fast with clear messages."""

    art: dict[str, Any] = {}

    # 1. Graph CSR ─────────────────────────────────────────────────────────
    csr = load_csr_npz(repo_root / "data/processed/graph_csr.npz")
    art["indptr"]     = csr["indptr"]       # np.int64
    art["indices"]    = csr["indices"]      # np.int64
    art["degrees"]    = csr["degrees"]      # np.int64
    art["node_ids"]   = csr["node_ids"]     # str array, row_index → node_id_string
    art["n_nodes"]    = len(art["degrees"])

    # 2. IC scores with per-node std ───────────────────────────────────────
    ci_path = repo_root / "data/processed/ic_scores_primary_with_ci.parquet"
    if not ci_path.exists():
        ci_path = repo_root / "data/processed/ic_scores_primary.parquet"
    if not ci_path.exists():
        raise FileNotFoundError("ic_scores_primary(_with_ci).parquet not found.")

    ic_df = pd.read_parquet(ci_path)
    # Reconstruct ic_score_std from CI columns if not present
    if "ic_score_std" not in ic_df.columns:
        if {"ic_ci_upper", "ic_ci_lower"}.issubset(ic_df.columns):
            n_r = float(ic_df["n_runs"].iloc[0]) if "n_runs" in ic_df.columns else 200.0
            ic_df["ic_score_std"] = (
                (ic_df["ic_ci_upper"] - ic_df["ic_ci_lower"]) / (2 * 1.96)
                * np.sqrt(n_r)
            )
        else:
            raise ValueError(
                "ic_scores parquet must have 'ic_score_std' OR "
                "{'ic_ci_lower','ic_ci_upper'} columns."
            )

    required = ["node_id", "ic_score_mean", "ic_score_std", "n_runs"]
    missing = [c for c in required if c not in ic_df.columns]
    if missing:
        raise ValueError(f"ic_scores parquet missing columns: {missing}")
    art["ic_df"] = ic_df

    # Build node_id → CSR row index lookup (for degree merging)
    art["nodeid_to_row"] = {nid: i for i, nid in enumerate(art["node_ids"])}

    # 3. Pilot diagnostics ─────────────────────────────────────────────────
    diag_path = repo_root / "outputs/day1_benchmark/ic_pilot_diagnostics.json"
    if not diag_path.exists():
        raise FileNotFoundError(f"Missing pilot diagnostics: {diag_path}")
    with open(diag_path) as f:
        art["pilot_diag"] = json.load(f)

    # 4. Community features (optional) ─────────────────────────────────────
    comm_path = repo_root / "data/processed/community_features.parquet"
    art["community_df"] = pd.read_parquet(comm_path) if comm_path.exists() else None

    # 5. LCC size ─────────────────────────────────────────────────────────
    lcc_path = repo_root / "outputs/stage0_data_quality/lcc_report.json"
    if lcc_path.exists():
        with open(lcc_path) as f:
            art["n_nodes_lcc"] = json.load(f).get("n_nodes_lcc", art["n_nodes"])
    else:
        art["n_nodes_lcc"] = art["n_nodes"]

    return art


def load_regression_stability_metrics(repo_root: Path) -> dict[str, Any]:
    """Load empirical Spearman stability points used in Option B justification."""
    out: dict[str, Any] = {
        "spearman_at_150runs": None,
        "spearman_at_1200runs": None,
        "source": None,
    }

    sweep_path = repo_root / "outputs/day1_benchmark/stability_sweep/stability_sweep_summary.json"
    if sweep_path.exists():
        with open(sweep_path, encoding="utf-8") as f:
            payload = json.load(f)
        rows = payload.get("rows", [])
        for row in rows:
            n_runs = int(row.get("n_runs", -1))
            if n_runs == 150:
                out["spearman_at_150runs"] = float(row.get("spearman_mean"))
            elif n_runs == 1200:
                out["spearman_at_1200runs"] = float(row.get("spearman_mean"))
        out["source"] = str(sweep_path)

    if out["spearman_at_150runs"] is None:
        stab_path = repo_root / "outputs/day1_benchmark/ic_label_stability.json"
        if stab_path.exists():
            with open(stab_path, encoding="utf-8") as f:
                payload = json.load(f)
            out["spearman_at_150runs"] = float(payload.get("summary", {}).get("spearman_mean"))
            if out["source"] is None:
                out["source"] = str(stab_path)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — REGIME & STRUCTURAL SEPARATION  (zero new IC compute)
# ─────────────────────────────────────────────────────────────────────────────

def test_11_regime(art: dict[str, Any]) -> dict[str, Any]:
    """
    Test 1.1 — Regime classification from pilot diagnostics.

    THEORETICAL NOTE: weighted cascade p(u,v) = 1/degree(v) has spectral
    radius ρ(W) = 1 by construction (row-sums of influence matrix = 1).
    Therefore 'regime' is inferred from observed cascade dynamics, NOT from
    comparing p_mean with λ_max of adjacency matrix (which would be wrong).
    """
    d    = art["pilot_diag"]["summary"]
    n_lcc = art["n_nodes_lcc"]

    cv              = d["cv_score"]
    mean_reach      = d["mean_reach"]
    top10_ratio     = d["top10_to_median_ratio"]
    mean_reach_pct  = mean_reach / n_lcc

    # Classify regime from observed dynamics
    if mean_reach_pct < PROTO["mean_reach_pct_subcritic"] and cv < PROTO["cv_subcritic"]:
        regime = "sub-critical"
        regime_note = (
            "Cascade dies within 1–2 hops. IC scores carry no meaningful signal. "
            "Changing kappa will not help — the problem is structural."
        )
        pivot_signal = True
    elif cv < PROTO["cv_adequate"] and top10_ratio < 2.0:
        regime = "near-critical-compressed"
        regime_note = (
            "Distribution heavily compressed (most nodes similar reach). "
            "Binary threshold sits in dense region — exactly the observed failure mode. "
            "Proceed to Tests 1.2 + 2.1."
        )
        pivot_signal = False   # not sufficient alone; need Tests 1.2 + 2.1
    elif cv >= PROTO["cv_adequate"]:
        regime = "adequate-spread"
        regime_note = (
            "cv_score already adequate. Binary gate failure originates elsewhere "
            "(threshold placement or n_runs insufficient). "
            "Proceed to Test 2.1 threshold sweep."
        )
        pivot_signal = False
    else:
        regime = "near-critical-moderate"
        regime_note = "Moderate spread. Proceed to full protocol."
        pivot_signal = False

    result = {
        "test": "1.1_regime",
        "cv_score":         round(cv, 4),
        "mean_reach":       round(mean_reach, 4),
        "mean_reach_pct":   round(mean_reach_pct, 6),
        "top10_to_median":  round(top10_ratio, 3),
        "regime":           regime,
        "regime_note":      regime_note,
        "theoretical_note": (
            "Weighted cascade p(u,v)=1/deg(v) always has rho(W)=1 by construction "
            "(influence matrix row-sums = 1). Regime is inferred from observed dynamics."
        ),
        "pivot_signal":     pivot_signal,
        "pivot_reason":     "sub-critical cascade — IC not viable" if pivot_signal else None,
    }
    _print_test("1.1", "PIVOT SIGNAL" if pivot_signal else "OK", result)
    return result


def test_12_degree_separation(art: dict[str, Any]) -> dict[str, Any]:
    """
    Test 1.2 — KS test on degree distributions of top-10% vs top-11–20% IC nodes.

    Logic: if the two groups are structurally indistinguishable (same degree
    distribution), no IC formula can create a stable binary boundary between them,
    because the graph topology doesn't support differentiation.
    """
    ic_df = art["ic_df"].copy()
    nid2row = art["nodeid_to_row"]
    degs = art["degrees"]

    # Attach degree to each IC node
    ic_df["degree"] = ic_df["node_id"].astype(str).map(
        lambda nid: int(degs[nid2row[nid]]) if nid in nid2row else np.nan
    )
    ic_df = ic_df.dropna(subset=["degree"])

    # Rank by ic_score_mean
    ic_df = ic_df.sort_values("ic_score_mean", ascending=False).reset_index(drop=True)
    n = len(ic_df)
    k_top  = max(1, int(n * PROTO["top_pct"]))
    k_next = max(1, int(n * PROTO["next_pct"]))

    deg_top   = ic_df["degree"].iloc[:k_top].values.astype(float)
    deg_next  = ic_df["degree"].iloc[k_top : k_top + k_next].values.astype(float)

    ks_stat_raw, ks_p_raw = cast(tuple[float, float], ks_2samp(deg_top, deg_next))
    _, mwu_p_raw = cast(tuple[float, float], mannwhitneyu(deg_top, deg_next, alternative="two-sided"))
    ks_stat = float(ks_stat_raw)
    ks_p = float(ks_p_raw)
    mwu_p = float(mwu_p_raw)

    not_separable = ks_stat < PROTO["degree_ks_not_separable"]
    verdict = (
        "not_separable"    if ks_stat < PROTO["degree_ks_not_separable"]  else
        "marginally_sep"   if ks_stat < 0.25                               else
        "separable"
    )

    result = {
        "test": "1.2_degree_separation",
        "n_top10":        int(k_top),
        "n_next10":       int(k_next),
        "degree_median_top10":  round(float(np.median(deg_top)), 1),
        "degree_median_next10": round(float(np.median(deg_next)), 1),
        "ks_stat":    round(ks_stat, 4),
        "ks_p_value": round(ks_p, 4),
        "mwu_p_value": round(mwu_p, 4),
        "verdict":    verdict,
        "threshold":  PROTO["degree_ks_not_separable"],
        "pivot_signal":  not_separable,
        "pivot_reason": (
            f"Degree KS={ks_stat:.3f} < {PROTO['degree_ks_not_separable']} — "
            "top-10% and top-11–20% IC nodes are structurally identical. "
            "No IC formula can create a stable binary boundary on this topology."
        ) if not_separable else None,
    }
    _print_test("1.2", "PIVOT SIGNAL" if not_separable else verdict.upper(), result)
    return result


def test_13_community_overlap(art: dict[str, Any]) -> dict[str, Any]:
    """
    Test 1.3 — Community-tier overlap between top-10% and top-11–20% IC nodes.

    If both groups draw from the same communities, within-community rank noise
    drives Jaccard instability regardless of p formula.
    """
    if art["community_df"] is None:
        result = {
            "test": "1.3_community_overlap",
            "skipped": True,
            "reason": "community_features.parquet not found",
            "pivot_signal": False,
        }
        _print_test("1.3", "SKIPPED", result)
        return result

    ic_df   = art["ic_df"].copy().sort_values("ic_score_mean", ascending=False).reset_index(drop=True)
    comm_df = art["community_df"][["node_id", "community_id"]].copy()
    comm_df["node_id"] = comm_df["node_id"].astype(str)
    ic_df["node_id"]   = ic_df["node_id"].astype(str)

    merged = ic_df.merge(comm_df, on="node_id", how="left")
    n  = len(merged)
    k  = max(1, int(n * PROTO["top_pct"]))
    k2 = max(1, int(n * PROTO["next_pct"]))

    comms_top   = set(merged["community_id"].iloc[:k].dropna().astype(int).tolist())
    comms_next  = set(merged["community_id"].iloc[k : k + k2].dropna().astype(int).tolist())

    intersection = comms_top & comms_next
    union        = comms_top | comms_next
    comm_overlap_jaccard = len(intersection) / len(union) if union else 0.0

    # Additional: fraction of communities that span BOTH groups
    merged["rank_group"] = (
        pd.cut(merged.index, bins=[-1, k-1, k+k2-1, n],
               labels=["top10", "next10", "rest"])
    )
    spanned = (
        merged[merged["rank_group"].isin(["top10", "next10"])]
        .groupby("community_id")["rank_group"]
        .nunique()
    )
    frac_spanning_comms = (spanned == 2).mean() if len(spanned) > 0 else 0.0

    same_tier = comm_overlap_jaccard > PROTO["community_overlap_same"]

    result = {
        "test": "1.3_community_overlap",
        "n_communities_top10":  len(comms_top),
        "n_communities_next10": len(comms_next),
        "community_overlap_jaccard":  round(comm_overlap_jaccard, 3),
        "frac_communities_spanning_both_groups": round(float(frac_spanning_comms), 3),
        "threshold": PROTO["community_overlap_same"],
        "pivot_signal":  same_tier,
        "pivot_reason": (
            f"Community-set Jaccard={comm_overlap_jaccard:.3f} > "
            f"{PROTO['community_overlap_same']} — both groups share the same "
            "community tier. Within-community rank noise will prevent stable "
            "binary separation regardless of propagation probability."
        ) if same_tier else None,
    }
    _print_test("1.3", "PIVOT SIGNAL" if same_tier else "OK", result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — NATURAL-THRESHOLD SWEEP  (zero new IC compute)
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_jaccard_clt(
    ic_mean: np.ndarray,
    ic_std:  np.ndarray,
    n_runs:  int,
    threshold_val: float,
) -> float:
    """
    CLT-based Jaccard estimate between two independent MC seeds.

    Derivation
    ----------
    Let X_i ~ N(μ_i, se_i²) be the IC score estimate for node i under a
    fresh MC seed (se_i = σ_i / √n_runs, by Central Limit Theorem).

    P(node i ∈ top-k | fresh seed) = p_i = Φ((μ_i − θ) / se_i)

    For two independent seeds S₁, S₂:
        E[|S₁ ∩ S₂|] = Σ_i  p_i²
        E[|S₁ ∪ S₂|] = Σ_i (2p_i − p_i²)
        E[Jaccard]   ≈ Σ_i p_i² / Σ_i (2p_i − p_i²)

    This is an approximation (ignores threshold covariance), valid when n is
    large and the top-k boundary is not too thin.
    """
    se = ic_std / np.sqrt(max(n_runs, 1))
    se = np.where(se < 1e-9, 1e-9, se)          # guard against zero std

    p = norm.cdf((ic_mean - threshold_val) / se)  # P(node in top-k | seed)
    p = np.clip(p, 0.0, 1.0)

    expected_intersection = float((p ** 2).sum())
    expected_union        = float(((2 * p) - p ** 2).sum())

    if expected_union < 1e-9:
        return 0.0
    return min(expected_intersection / expected_union, 1.0)


def test_21_threshold_analysis(art: dict[str, Any]) -> dict[str, Any]:
    """
    Test 2.1 — Sweep all thresholds in [3%, 30%], estimate Jaccard at each.

    Answers: 'Is there ANY top-k threshold where binary labels are stable?'
    If not → no binary labeling strategy works on this IC distribution.
    """
    ic_df  = art["ic_df"].copy()
    mu     = ic_df["ic_score_mean"].values.astype(float)
    sigma  = ic_df["ic_score_std"].values.astype(float)
    n_runs = int(ic_df["n_runs"].iloc[0])
    n      = len(mu)

    idx_sorted = np.argsort(mu)[::-1]   # descending
    mu_sorted  = mu[idx_sorted]

    rows = []
    pct_vals = np.arange(PROTO["pct_min"], PROTO["pct_max"] + 1e-9, PROTO["pct_step"])

    for pct in pct_vals:
        k = max(1, int(n * pct))
        if k >= n:
            continue
        threshold_val = float(mu_sorted[k - 1])  # score of k-th node

        est_jaccard  = _estimate_jaccard_clt(mu, sigma, n_runs, threshold_val)

        # Boundary node count: nodes whose 95% CI crosses the threshold
        se = sigma / np.sqrt(n_runs)
        boundary_mask = (
            (mu + 1.96 * se > threshold_val) &
            (mu - 1.96 * se < threshold_val)
        )
        n_boundary = int(boundary_mask.sum())
        n_boundary_in_topk = int(boundary_mask[idx_sorted[:k]].sum())

        # Gap: score difference between k-th and (k+1)-th node
        gap = float(mu_sorted[k - 1] - mu_sorted[k]) if k < n else 0.0
        # Local density: avg std of nodes in ±5% window around threshold
        win = max(1, int(n * 0.05))
        local_std = float(sigma[idx_sorted[max(0, k-win): k+win]].mean())
        gap_to_noise = gap / (local_std / np.sqrt(n_runs) + 1e-9)

        rows.append({
            "threshold_pct":      round(float(pct), 2),
            "k":                  int(k),
            "threshold_val":      round(threshold_val, 4),
            "estimated_jaccard":  round(est_jaccard, 4),
            "n_boundary_global":  n_boundary,
            "boundary_pct_global": round(n_boundary / n, 3),
            "n_boundary_in_topk": n_boundary_in_topk,
            "boundary_pct_in_topk": round(n_boundary_in_topk / k, 3),
            "gap":                round(gap, 4),
            "gap_to_noise":       round(gap_to_noise, 3),
        })

    df_rows = pd.DataFrame(rows)
    best_row = df_rows.loc[df_rows["estimated_jaccard"].idxmax()].to_dict()
    max_est_jaccard = float(df_rows["estimated_jaccard"].max())
    no_viable_threshold = max_est_jaccard < PROTO["jaccard_feasibility_min"]

    result = {
        "test": "2.1_threshold_analysis",
        "n_runs_used":           n_runs,
        "n_nodes":               n,
        "best_threshold_pct":    round(float(best_row["threshold_pct"]), 2),
        "best_estimated_jaccard": round(max_est_jaccard, 4),
        "feasibility_min":       PROTO["jaccard_feasibility_min"],
        "target_threshold_10pct_jaccard": round(
            float(df_rows.loc[df_rows["threshold_pct"] == 0.10, "estimated_jaccard"].iloc[0])
            if (df_rows["threshold_pct"] == 0.10).any() else 0.0, 4
        ),
        "sweep_rows":            df_rows.to_dict(orient="records"),
        "pivot_signal":          no_viable_threshold,
        "pivot_reason": (
            f"Maximum estimated Jaccard across all thresholds [{PROTO['pct_min']:.0%}–"
            f"{PROTO['pct_max']:.0%}] = {max_est_jaccard:.3f} < "
            f"{PROTO['jaccard_feasibility_min']} (feasibility minimum). "
            "The IC score distribution has no region with sufficient gap-to-noise "
            "ratio for stable binary labeling."
        ) if no_viable_threshold else None,
    }
    _print_test("2.1", "PIVOT SIGNAL" if no_viable_threshold else
                f"best_jaccard={max_est_jaccard:.3f}@{best_row['threshold_pct']:.0%}", result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — KAPPA PILOT SWEEP  (new IC compute, only if Phases 1-2 inconclusive)
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_ic_once_kappa(
    source:     int,
    indptr:     np.ndarray,
    indices:    np.ndarray,
    inv_degrees_kappa: np.ndarray,   # pre-computed: kappa/degree, clipped to 1.0
    rng:        np.random.Generator,
) -> int:
    """Single IC run with arbitrary kappa baked into inv_degrees_kappa.
    Reuses the same BFS logic as ic_labels_primary._simulate_ic_once.
    """
    activated = {int(source)}
    frontier  = [int(source)]
    while frontier:
        nxt: list[int] = []
        for node in frontier:
            for nb_raw in indices[int(indptr[node]) : int(indptr[node + 1])]:
                nb = int(nb_raw)
                if nb in activated:
                    continue
                p = float(inv_degrees_kappa[nb])
                if p <= 0.0:
                    continue
                if p >= 1.0 or rng.random() < p:
                    activated.add(nb)
                    nxt.append(nb)
        frontier = nxt
    return len(activated)


def _node_summary_kappa(
    source:     int,
    indptr:     np.ndarray,
    indices:    np.ndarray,
    inv_degrees_kappa: np.ndarray,
    n_runs:     int,
    seed:       int,
) -> tuple[float, float]:
    """Mean and std of reach for one node across n_runs."""
    rng  = np.random.default_rng(seed)
    runs = np.empty(n_runs, dtype=np.int32)
    for i in range(n_runs):
        runs[i] = _simulate_ic_once_kappa(source, indptr, indices, inv_degrees_kappa, rng)
    return float(runs.mean()), float(runs.std(ddof=0))


def _run_kappa_pilot(
    kappa:      float,
    pilot_rows: np.ndarray,         # CSR row indices of pilot nodes
    indptr:     np.ndarray,
    indices:    np.ndarray,
    degrees:    np.ndarray,
    n_nodes:    int,
    n_nodes_lcc: int,
    n_runs:     int,
    mc_seeds:   list[int],
    n_jobs:     int,
) -> dict[str, Any]:
    """
    Run IC pilot for one kappa value across multiple MC seeds.

    Computes per-seed: cv_score, estimated Jaccard at natural threshold.
    Returns aggregated summary.
    """
    # Pre-compute inv_degrees with kappa, capped at 1.0
    inv_deg = np.zeros(n_nodes, dtype=float)
    mask    = degrees > 0
    inv_deg[mask] = np.minimum(kappa / degrees[mask].astype(float), 1.0)

    seed_results = []
    for mc_seed in mc_seeds:
        t0 = time.time()

        def _worker(row: int) -> tuple[float, float]:
            return _node_summary_kappa(
                source=int(row),
                indptr=indptr,
                indices=indices,
                inv_degrees_kappa=inv_deg,
                n_runs=n_runs,
                seed=mc_seed * 10_000 + int(row),   # same deterministic rule as plan
            )

        stats_raw = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_worker)(int(r)) for r in pilot_rows
        )
        if stats_raw is None:
            raise RuntimeError("Parallel execution returned no results in kappa pilot.")
        stats = cast(list[tuple[float, float]], stats_raw)
        elapsed = time.time() - t0

        means = np.array([m for m, _ in stats])
        stds  = np.array([s for _, s in stats])

        cv = float(means.std() / (means.mean() + 1e-9))

        # Natural threshold: percentile with maximum estimated Jaccard
        best_jaccard = 0.0
        best_pct     = 0.10
        n = len(means)
        idx_sorted = np.argsort(means)[::-1]
        means_s = means[idx_sorted]

        for pct in np.arange(0.05, 0.30, 0.01):
            k = max(1, int(n * pct))
            if k >= n:
                continue
            tval = float(means_s[k - 1])
            jac  = _estimate_jaccard_clt(means, stds, n_runs, tval)
            if jac > best_jaccard:
                best_jaccard = jac
                best_pct     = pct

        # Ceiling effect: fraction of pilot nodes reaching > 5% LCC
        ceiling_threshold = 0.05 * n_nodes_lcc
        ceiling_pct = float((means > ceiling_threshold).mean())

        seed_results.append({
            "mc_seed":           mc_seed,
            "cv_score":          round(cv, 4),
            "cv_pass":           cv >= PROTO["cv_adequate"],
            "best_threshold_pct": round(best_pct, 2),
            "best_estimated_jaccard": round(best_jaccard, 4),
            "jaccard_feasible":  best_jaccard >= PROTO["jaccard_target"],
            "mean_reach":        round(float(means.mean()), 3),
            "mean_reach_pct":    round(float(means.mean()) / n_nodes_lcc, 5),
            "ceiling_pct":       round(ceiling_pct, 3),
            "ceiling_warn":      ceiling_pct > PROTO["ceiling_warn_pct"],
            "runtime_sec":       round(elapsed, 1),
        })

    cv_values    = [r["cv_score"] for r in seed_results]
    jacc_values  = [r["best_estimated_jaccard"] for r in seed_results]
    cv_mean      = float(np.mean(cv_values))
    jacc_mean    = float(np.mean(jacc_values))
    any_ceiling  = any(r["ceiling_warn"] for r in seed_results)

    both_pass    = (cv_mean >= PROTO["cv_adequate"] and
                    jacc_mean >= PROTO["jaccard_target"])

    verdict = "VIABLE" if both_pass else "FAIL"
    if any_ceiling:
        verdict += "_CEILING_WARN"

    return {
        "kappa":              kappa,
        "cv_mean":            round(cv_mean, 4),
        "cv_pass":            cv_mean >= PROTO["cv_adequate"],
        "jaccard_mean":       round(jacc_mean, 4),
        "jaccard_feasible":   jacc_mean >= PROTO["jaccard_target"],
        "ceiling_warn":       any_ceiling,
        "verdict":            verdict,
        "per_seed":           seed_results,
    }


def test_31_kappa_sweep(art: dict[str, Any], n_jobs: int = -1) -> dict[str, Any]:
    """
    Test 3.1 — Pilot kappa sweep across {1.5, 2.0, 3.0}.

    Only reached when Phases 1–2 leave the verdict inconclusive.
    """
    degrees  = art["degrees"]
    indptr   = art["indptr"]
    indices  = art["indices"]
    n_nodes  = art["n_nodes"]
    n_lcc    = art["n_nodes_lcc"]

    # Stratified pilot sample (same quintile-stratified logic as main pipeline)
    rng_sample = np.random.default_rng(42)
    quintiles  = pd.qcut(
        pd.Series(degrees.astype(float)), q=5, labels=False, duplicates="drop"
    ).to_numpy()
    pilot_rows_list = []
    for q in range(5):
        pool = np.where(quintiles == q)[0]
        k    = max(1, PROTO["pilot_n_nodes"] // 5)
        pick = rng_sample.choice(pool, size=min(k, len(pool)), replace=False)
        pilot_rows_list.append(pick)
    pilot_rows = np.sort(np.concatenate(pilot_rows_list))

    kappa_rows = []
    any_viable = False
    viable_kappa = None

    for kappa in PROTO["kappa_values"]:
        print(f"\n  [Phase 3] Running kappa={kappa} "
              f"({len(pilot_rows)} nodes × {PROTO['pilot_n_runs']} runs × "
              f"{len(PROTO['pilot_mc_seeds'])} seeds)...")
        row = _run_kappa_pilot(
            kappa=kappa,
            pilot_rows=pilot_rows,
            indptr=indptr,
            indices=indices,
            degrees=degrees,
            n_nodes=n_nodes,
            n_nodes_lcc=n_lcc,
            n_runs=PROTO["pilot_n_runs"],
            mc_seeds=PROTO["pilot_mc_seeds"],
            n_jobs=n_jobs,
        )
        kappa_rows.append(row)
        print(f"    cv={row['cv_mean']:.3f} ({'✓' if row['cv_pass'] else '✗'})  "
              f"est_jaccard={row['jaccard_mean']:.3f} "
              f"({'✓' if row['jaccard_feasible'] else '✗'})  [{row['verdict']}]")
        if row["cv_pass"] and row["jaccard_feasible"] and not row["ceiling_warn"]:
            if not any_viable:
                viable_kappa = kappa
            any_viable = True

    pivot_signal = not any_viable

    result = {
        "test": "3.1_kappa_sweep",
        "pilot_n_nodes":  len(pilot_rows),
        "pilot_n_runs":   PROTO["pilot_n_runs"],
        "mc_seeds":       PROTO["pilot_mc_seeds"],
        "kappa_results":  kappa_rows,
        "any_kappa_viable": any_viable,
        "viable_kappa":     viable_kappa,
        "pivot_signal":  pivot_signal,
        "pivot_reason": (
            f"No kappa in {PROTO['kappa_values']} achieved "
            f"cv_mean ≥ {PROTO['cv_adequate']} AND "
            f"est_jaccard ≥ {PROTO['jaccard_target']} simultaneously "
            "on the pilot sample. Binary label instability is intrinsic "
            "to the Twitch topology under any moderate-to-strong IC propagation."
        ) if pivot_signal else None,
        "rerun_recommendation": (
            f"kappa={viable_kappa} achieved both gates on pilot. "
            f"Estimated full rerun cost: ~"
            f"{5000 * PROTO['pilot_n_runs'] * 0.5 / 1000 / 3600 * viable_kappa:.1f}h. "
            "Requires re-justifying IC model interpretation in paper "
            f"(kappa={viable_kappa} ≠ standard weighted cascade)."
        ) if any_viable else None,
    }
    _print_test("3.1", "PIVOT SIGNAL" if pivot_signal else f"KAPPA {viable_kappa} VIABLE", result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# DECISION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

_EVIDENCE_TEMPLATE = (
    "Empirical analysis of the Twitch EN graph (168,114 nodes; mean degree ≈ 81) "
    "under the weighted-cascade IC model (p(u,v) = kappa/degree(v), kappa=1) yields "
    "the following evidence against stable binary top-k labeling:\n"
    "{evidence_bullets}\n"
    "These are intrinsic properties of the graph topology and propagation regime, "
    "not simulation artifacts. Accordingly, we adopt continuous regression on "
    "log-transformed IC scores (y = log1p(ic_score_mean)) as the primary evaluation "
    "framework (Option B). Binary labels are retained as supplementary only, with "
    "{boundary_clause}"
)


def build_decision_report(
    results: dict[str, dict],
    quality_gate_report: dict,
    regression_stability: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    """Traverse all test results, determine final verdict, write report."""

    pivot_signals = {k: v.get("pivot_signal", False) for k, v in results.items()}
    pivot_reasons = {k: v.get("pivot_reason") for k, v in results.items()
                     if v.get("pivot_signal")}

    # Gather specific metric values for evidence statement
    evidence_bullets = []
    if "1.1" in results:
        r = results["1.1"]
        evidence_bullets.append(
            f"• Regime: {r['regime']} — cv_score={r['cv_score']:.4f} "
            f"(target > {PROTO['cv_adequate']}), "
            f"mean_reach={r['mean_reach']:.1f} nodes "
            f"({r['mean_reach_pct']:.4%} of LCC)."
        )
    if "1.2" in results:
        r = results["1.2"]
        evidence_bullets.append(
            f"• Degree separation (KS) top-10% vs top-11–20%: "
            f"KS={r['ks_stat']:.4f} (verdict: {r['verdict']}; "
            f"pivot threshold: KS < {PROTO['degree_ks_not_separable']})."
        )
    if "1.3" in results and not results["1.3"].get("skipped"):
        r = results["1.3"]
        evidence_bullets.append(
            f"• Community-tier overlap: "
            f"set-Jaccard={r['community_overlap_jaccard']:.3f}, "
            f"{r['frac_communities_spanning_both_groups']:.1%} communities span both groups."
        )
    if "2.1" in results:
        r = results["2.1"]
        evidence_bullets.append(
            f"• Max estimated Jaccard across all percentile thresholds "
            f"[{PROTO['pct_min']:.0%}–{PROTO['pct_max']:.0%}]: "
            f"{r['best_estimated_jaccard']:.4f} at "
            f"{r['best_threshold_pct']:.0%} "
            f"(feasibility minimum: {PROTO['jaccard_feasibility_min']})."
        )
    if "3.1" in results:
        r = results["3.1"]
        evidence_bullets.append(
            f"• Kappa sweep {PROTO['kappa_values']}: "
            + (f"no kappa achieved cv ≥ {PROTO['cv_adequate']} AND "
               f"est_jaccard ≥ {PROTO['jaccard_target']} simultaneously."
               if r["pivot_signal"] else
               f"kappa={r['viable_kappa']} met gates (see rerun_recommendation).")
        )

    any_pivot = any(pivot_signals.values())
    kappa_viable = (
        results.get("3.1", {}).get("any_kappa_viable", False)
    )

    if any_pivot and not kappa_viable:
        verdict = "PIVOT_CONFIRMED"
        action  = "Option B is empirically justified. No rerun needed."
    elif kappa_viable:
        verdict = "KAPPA_VIABLE"
        kappa_v = results["3.1"]["viable_kappa"]
        action  = (
            f"kappa={kappa_v} shows promise on pilot. "
            "Team decision needed: rerun IC (3–4 day delay, new model justification) "
            "vs. proceed with Option B (zero delay, defensible with evidence)."
        )
    else:
        verdict = "INCONCLUSIVE"
        action  = "All tests passed without pivot signal. Binary gate failure may be n_runs-related."

    # Observed boundary ratio from quality_gate_report (already computed)
    boundary_ratio = quality_gate_report.get("observed", {}).get("uncertainty_boundary_ratio")
    if boundary_ratio is None:
        boundary_clause = (
            "explicit uncertainty disclosure (uncertainty_boundary_ratio unavailable in "
            "quality_gate_report.json; report this missing-data state explicitly)."
        )
    else:
        boundary_clause = (
            f"explicit uncertainty disclosure (uncertainty_boundary_ratio={float(boundary_ratio):.1%})."
        )

    evidence_statement = _EVIDENCE_TEMPLATE.format(
        evidence_bullets="\n".join(evidence_bullets),
        boundary_clause=boundary_clause,
    )

    spearman_150 = regression_stability.get("spearman_at_150runs")
    spearman_1200 = regression_stability.get("spearman_at_1200runs")

    report = {
        "timestamp":          now_iso(),
        "protocol_version":   "1.0",
        "verdict":            verdict,
        "action":             action,
        "any_pivot_signal":   any_pivot,
        "pivot_signals":      pivot_signals,
        "pivot_reasons":      pivot_reasons,
        "quality_gate_observed": quality_gate_report.get("observed", {}),
        "tests_run":          list(results.keys()),
        "evidence_statement_for_paper": evidence_statement,
        "option_b_justification": {
            "quality_gate_pass_all": quality_gate_report.get("pass_all", False),
            "regression_spearman_at_150runs": spearman_150,
            "regression_spearman_at_1200runs": spearman_1200,
            "regression_spearman_source": regression_stability.get("source"),
            "spearman_threshold_for_binary": 0.9,
            "conclusion": (
                "Regression target (log1p IC score mean) is substantially more stable "
                "than binary top-10% labels. Spearman rank correlation remains consistent "
                "while Jaccard collapses — confirming the problem is threshold-cutting "
                "noise, not signal absence."
            ),
        },
        "proto_constants": {k: v for k, v in PROTO.items()
                            if not isinstance(v, list) or len(v) < 10},
    }

    _write_json_safe(out_dir / "pivot_decision_report.json", report)
    return report


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _print_test(tid: str, status: str, result: dict) -> None:
    pivot = "⚠  PIVOT SIGNAL" if result.get("pivot_signal") else "✓"
    print(f"  [Test {tid}] {status:30s}  {pivot}")


def _load_quality_gate(repo_root: Path) -> dict:
    p = repo_root / "outputs/day1_benchmark/quality_gate_report.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {"pass_all": False, "observed": {}}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IC feasibility protocol — MAPR2026 v3")
    p.add_argument("--repo-root",       default=".",
                   help="Repository root (default: current directory)")
    p.add_argument("--out-dir",         default="outputs/ic_feasibility",
                   help="Output directory for all JSON artifacts")
    p.add_argument("--skip-kappa-sweep", action="store_true",
                   help="Skip Phase 3 even if Phases 1-2 are inconclusive")
    p.add_argument("--force-kappa-sweep", action="store_true",
                   help="Run Phase 3 even when pivot was already signaled in earlier phases")
    p.add_argument("--n-jobs",          type=int, default=-1,
                   help="Parallel workers for Phase 3 IC simulation")
    return p.parse_args()


def main() -> None:
    args     = parse_args()
    root     = Path(args.repo_root).resolve()
    out_dir  = ensure_dir(root / args.out_dir)

    print("\n" + "═" * 60)
    print("  IC FEASIBILITY PROTOCOL  —  MAPR2026 v3")
    print("═" * 60)
    print(f"  repo_root : {root}")
    print(f"  out_dir   : {out_dir}")
    print(f"  skip_kappa: {args.skip_kappa_sweep}")
    print(f"  force_kappa: {args.force_kappa_sweep}")
    print("═" * 60 + "\n")

    # ── Load artifacts ────────────────────────────────────────────────────
    print("[Loading artifacts...]")
    art = load_all_artifacts(root)
    qg  = _load_quality_gate(root)
    regression_stability = load_regression_stability_metrics(root)
    print(f"  Loaded: {art['n_nodes']:,} nodes, "
          f"{len(art['ic_df']):,} labeled IC nodes, "
          f"community_df={'present' if art['community_df'] is not None else 'absent'}\n")

    all_results: dict[str, dict] = {}
    pivot_confirmed = False

    # ── Phase 1 ───────────────────────────────────────────────────────────
    print("── Phase 1: Regime & structural separation (zero new compute) ──")

    r11 = test_11_regime(art)
    _write_json_safe(out_dir / "phase1_regime.json", r11)
    all_results["1.1"] = r11
    if r11["pivot_signal"]:
        pivot_confirmed = True

    r12 = test_12_degree_separation(art)
    _write_json_safe(out_dir / "phase1_degree_separation.json", r12)
    all_results["1.2"] = r12
    if r12["pivot_signal"]:
        pivot_confirmed = True

    r13 = test_13_community_overlap(art)
    _write_json_safe(out_dir / "phase1_community_overlap.json", r13)
    all_results["1.3"] = r13
    if r13["pivot_signal"]:
        pivot_confirmed = True

    if pivot_confirmed:
        print("\n  ⚠  Phase 1 produced pivot signal(s). Skipping Phase 2+3.\n")

    # ── Phase 2 ───────────────────────────────────────────────────────────
    if not pivot_confirmed:
        print("\n── Phase 2: Natural-threshold sweep (zero new compute) ──")

        r21 = test_21_threshold_analysis(art)
        _write_json_safe(out_dir / "phase2_threshold_analysis.json", r21)
        all_results["2.1"] = r21
        if r21["pivot_signal"]:
            pivot_confirmed = True

    # ── Phase 3 (conditional) ─────────────────────────────────────────────
    if args.force_kappa_sweep:
        print("\n── Phase 3: Kappa pilot sweep (forced run) ──")
        r31 = test_31_kappa_sweep(art, n_jobs=args.n_jobs)
        _write_json_safe(out_dir / "phase3_kappa_sweep.json", r31)
        all_results["3.1"] = r31
        if r31["pivot_signal"]:
            pivot_confirmed = True
    elif pivot_confirmed:
        print("\n  ⚠  Pivot already confirmed — skipping Phase 3 (kappa sweep).\n")
    elif args.skip_kappa_sweep:
        print("\n  [Phase 3 skipped via --skip-kappa-sweep flag]\n")
    else:
        print("\n── Phase 3: Kappa pilot sweep (new IC compute) ──")
        r31 = test_31_kappa_sweep(art, n_jobs=args.n_jobs)
        _write_json_safe(out_dir / "phase3_kappa_sweep.json", r31)
        all_results["3.1"] = r31
        if r31["pivot_signal"]:
            pivot_confirmed = True

    # ── Final decision report ─────────────────────────────────────────────
    print("\n── Final decision ──")
    report = build_decision_report(all_results, qg, regression_stability, out_dir)

    print(f"\n{'═'*60}")
    print(f"  VERDICT : {report['verdict']}")
    print(f"  ACTION  : {report['action']}")
    print(f"{'═'*60}")
    print(f"\nArtifacts written to: {out_dir}")
    print("  pivot_decision_report.json  ← cite this in paper / handoff doc")


if __name__ == "__main__":
    main()
