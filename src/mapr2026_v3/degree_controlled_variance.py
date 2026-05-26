"""MAPR2026 v3 — Degree-controlled IC variance (C1).

Owner: Person 1

Purpose
-------
Generate the MAPR-MUST C1 artifact:
- outputs/mapr2026_v3_results/degree_controlled_ic_variance.json

This quantifies whether IC variability persists *within* degree bands, i.e.
"IC is not just degree" evidence without relying on global correlation alone.

Inputs
------
- data/processed/ic_scores_primary.parquet (labeled subset)
- data/processed/graph_csr.npz (degree lookup; deterministic mapping)
Optional:
- data/processed/diffusion_proxies.parquet (one_hop_spread) for tier-2 residual evidence

Output (contract)
-----------------
- outputs/mapr2026_v3_results/degree_controlled_ic_variance.json
  fields include per-band:
    degree_band, n_nodes_in_band, ic_mean_in_band, ic_std_in_band,
    cv_within_band, interpretation

Notes
-----
- Degree bands are quintiles computed on the labeled subset degrees.
- CV is computed as std/mean, with guard for mean==0.
 - If diffusion proxies are available, we also report a tier-2 residual test:
     regress log1p(ic_score_mean) on log1p(one_hop_spread) within each band and
     summarize R^2 and residual scale.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _shared import PATHS, load_csr_npz, now_iso, require_columns, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MAPR2026 v3 C1: degree-controlled IC variance")
    p.add_argument("--ic", default=PATHS.ic_scores)
    p.add_argument("--csr", default=PATHS.csr_npz)
    p.add_argument(
        "--proxies",
        default=PATHS.proxies,
        help="Optional diffusion proxies parquet (uses one_hop_spread if present)",
    )
    p.add_argument(
        "--out",
        default="outputs/mapr2026_v3_results/degree_controlled_ic_variance.json",
    )
    p.add_argument("--n-bands", type=int, default=5, help="Number of degree bands (default quintiles=5)")
    p.add_argument(
        "--disable-tier2",
        action="store_true",
        help="Disable tier-2 one-hop regression residual evidence even if proxies exist",
    )
    return p.parse_args()


def _ensure_node_id_str(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["node_id"] = df["node_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    return df


def _interpret_bandwise(band_rows: list[dict[str, Any]]) -> str:
    cvs = [float(r.get("cv_within_band", float("nan"))) for r in band_rows]
    cvs = [c for c in cvs if np.isfinite(c)]
    if not cvs:
        return "Insufficient data to interpret within-band variance (no finite CV values)."

    med = float(np.median(cvs))
    if med >= 0.30:
        return (
            "Median within-band CV ≥ 0.30: IC variability remains substantial even after conditioning on degree; "
            "supports 'IC is not solely degree'."
        )
    if med >= 0.15:
        return (
            "Median within-band CV in [0.15, 0.30): some within-degree variability persists; evidence is mixed and "
            "should be paired with correlation/ablation results."
        )
    return (
        "Median within-band CV < 0.15: IC scores are relatively tight within degree bands; results may be largely "
        "degree-driven (interpret alongside global rho and proxy correlations)."
    )


def _ols_simple(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Fit y ~ a + b x via least squares and return compact diagnostics."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError("x and y must be 1-D arrays of the same length")

    n = int(x.shape[0])
    if n < 3:
        return {
            "n": float(n),
            "beta0": float("nan"),
            "beta1": float("nan"),
            "r2": float("nan"),
            "rmse": float("nan"),
            "residual_std": float("nan"),
        }

    X = np.column_stack([np.ones(n, dtype=float), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    resid = y - y_hat

    sse = float(np.sum(resid ** 2))
    sst = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = float(1.0 - (sse / sst)) if sst > 0 else float("nan")

    rmse = float(np.sqrt(sse / n))
    residual_std = float(np.std(resid, ddof=1))

    return {
        "n": float(n),
        "beta0": float(beta[0]),
        "beta1": float(beta[1]),
        "r2": r2,
        "rmse": rmse,
        "residual_std": residual_std,
    }


def main() -> None:
    args = parse_args()

    ic_path = Path(args.ic)
    if not ic_path.exists():
        raise FileNotFoundError(f"Missing IC scores: {ic_path}")

    df_ic = _ensure_node_id_str(pd.read_parquet(ic_path))
    require_columns(df_ic, ["node_id", "ic_score_mean"], "ic_scores_primary")

    # Degree lookup via CSR mapping.
    csr = load_csr_npz(args.csr)
    node_ids = csr["node_ids"].astype(str)
    degrees = csr["degrees"].astype(np.int64, copy=False)
    deg_map = dict(zip(node_ids.tolist(), degrees.tolist()))

    df = df_ic[["node_id", "ic_score_mean"]].copy()
    df["ic_score_mean"] = pd.to_numeric(df["ic_score_mean"], errors="coerce")
    if df["ic_score_mean"].isna().any():
        raise ValueError("ic_score_mean contains NaN after numeric coercion")

    df["degree"] = df["node_id"].map(deg_map)
    if df["degree"].isna().any():
        missing = int(df["degree"].isna().sum())
        raise ValueError(f"Degree lookup failed for {missing} labeled nodes (CSR mapping missing node_id)")
    df["degree"] = pd.to_numeric(df["degree"], errors="coerce").astype(int)

    # Optional tier-2: merge one-hop proxy for residual evidence.
    tier2_enabled = False
    tier2_note = "disabled"
    if not bool(args.disable_tier2):
        proxies_path = Path(args.proxies)
        if proxies_path.exists():
            df_p = pd.read_parquet(proxies_path)
            require_columns(df_p, ["node_id", "one_hop_spread"], "diffusion_proxies")
            df_p = df_p[["node_id", "one_hop_spread"]].copy()
            df_p["node_id"] = df_p["node_id"].astype(str)
            df_p["one_hop_spread"] = pd.to_numeric(df_p["one_hop_spread"], errors="coerce")
            df = df.merge(df_p, on="node_id", how="left")

            n_missing_proxy = int(df["one_hop_spread"].isna().sum())
            if n_missing_proxy == 0:
                tier2_enabled = True
                tier2_note = "enabled"
            else:
                tier2_note = f"proxies_missing_for_{n_missing_proxy}_labeled_nodes"
        else:
            tier2_note = "proxies_file_missing"

    n_bands = int(args.n_bands)
    if n_bands < 2:
        raise ValueError("--n-bands must be >= 2")

    # Degree bands on labeled set.
    df["degree_band_idx"] = pd.qcut(df["degree"].astype(float), q=n_bands, labels=False, duplicates="drop")
    if df["degree_band_idx"].isna().any():
        raise ValueError("Failed to create degree bands (qcut produced NaN bands)")

    band_rows: list[dict[str, Any]] = []
    for band_idx in sorted(df["degree_band_idx"].unique().tolist()):
        band_df = df[df["degree_band_idx"] == band_idx]
        n = int(len(band_df))
        mean = float(band_df["ic_score_mean"].mean()) if n else float("nan")
        std = float(band_df["ic_score_mean"].std(ddof=1)) if n >= 2 else 0.0
        cv = float(std / mean) if mean != 0.0 else float("inf")

        deg_min = int(band_df["degree"].min()) if n else None
        deg_max = int(band_df["degree"].max()) if n else None

        row: dict[str, Any] = {
            "degree_band": f"q{int(band_idx) + 1}",
            "degree_range": [deg_min, deg_max],
            "n_nodes_in_band": n,
            "ic_mean_in_band": mean,
            "ic_std_in_band": std,
            "cv_within_band": cv,
            "interpretation": "Within-band variability summary (see top-level interpretation).",
        }

        if tier2_enabled and n >= 3:
            x = band_df["one_hop_spread"].astype(float).to_numpy()
            y = band_df["ic_score_mean"].astype(float).to_numpy()
            # Plan-aligned residual evidence (robust to heavy tails): log1p(y) ~ a + b*log1p(x)
            diag = _ols_simple(np.log1p(x), np.log1p(y))
            row["tier2_one_hop_regression"] = {
                "model": "log1p(ic_score_mean) ~ beta0 + beta1*log1p(one_hop_spread)",
                "n": int(diag["n"]),
                "beta0": float(diag["beta0"]),
                "beta1": float(diag["beta1"]),
                "r2": float(diag["r2"]),
                "rmse": float(diag["rmse"]),
                "residual_std": float(diag["residual_std"]),
            }

        band_rows.append(row)

    payload: dict[str, Any] = {
        "timestamp": now_iso(),
        "n_rows": int(len(df)),
        "n_bands": int(len(band_rows)),
        "bands": band_rows,
        "interpretation": _interpret_bandwise(band_rows),
        "tier2": {
            "enabled": bool(tier2_enabled),
            "mode": "optional_one_hop_residual",
            "proxies_path": str(Path(args.proxies)),
            "note": str(tier2_note),
        },
    }

    out_path = Path(args.out)
    write_json(out_path, payload)
    print(f"[OK] Wrote degree-controlled IC variance: {out_path} (n_rows={len(df)})")


if __name__ == "__main__":
    main()
