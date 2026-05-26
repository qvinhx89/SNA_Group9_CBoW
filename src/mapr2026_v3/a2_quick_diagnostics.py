"""MAPR2026 v3 — Quick diagnostics for A2 IC labels.

This script does NOT train any surrogate model.
It summarizes how A2 labels relate to:
- primary A0 labels
- degree (from CSR)
- diffusion proxies (one-hop / two-hop)

Outputs a small JSON report for tracking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _shared import PATHS, load_csr_npz, now_iso, require_columns


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quick diagnostics for A2 IC labels")
    p.add_argument("--csr", default=PATHS.csr_npz)
    p.add_argument("--ic-a0", default=PATHS.ic_scores)
    p.add_argument("--ic-a2", default=str(Path(PATHS.results_dir) / "ic_scores_sensitivity_a2.parquet"))
    p.add_argument("--proxies", default=PATHS.proxies)
    p.add_argument(
        "--out-json",
        default=str(Path(PATHS.results_dir) / "a2_label_diagnostics.json"),
    )
    p.add_argument("--top-pct", type=float, default=0.10)
    return p.parse_args()


def _ensure_node_id_str(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["node_id"] = out["node_id"].astype(str)
    return out


def _spearman(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    m = x.notna() & y.notna()
    if int(m.sum()) < 3:
        return float("nan")
    return float(x[m].corr(y[m], method="spearman"))


def _stable_rank(scores: np.ndarray, ids: np.ndarray) -> np.ndarray:
    # score desc, tie-break id asc
    return np.lexsort((ids.astype(np.int64), -scores.astype(float)))


def _topk_set(scores: np.ndarray, ids: np.ndarray, top_pct: float) -> set[int]:
    n = int(len(scores))
    k = max(1, int(np.ceil(float(top_pct) * n)))
    rank = _stable_rank(scores, ids)
    return set(rank[:k].tolist())


def _jaccard(a: set[int], b: set[int]) -> float:
    u = a | b
    if not u:
        return 1.0
    return float(len(a & b) / len(u))


def main() -> None:
    args = parse_args()

    df_a0 = _ensure_node_id_str(pd.read_parquet(Path(args.ic_a0)))
    df_a2 = _ensure_node_id_str(pd.read_parquet(Path(args.ic_a2)))
    require_columns(df_a0, ["node_id", "ic_score_mean"], "ic_scores_primary")
    require_columns(df_a2, ["node_id", "ic_score_mean"], "ic_scores_sensitivity_a2")

    df = df_a0[["node_id", "ic_score_mean"]].rename(columns={"ic_score_mean": "ic_a0"}).merge(
        df_a2[["node_id", "ic_score_mean"]].rename(columns={"ic_score_mean": "ic_a2"}),
        on="node_id",
        how="inner",
    )

    # Attach degree via CSR mapping.
    csr = load_csr_npz(Path(args.csr))
    node_ids = pd.Series(csr["node_ids"].astype(str))
    degrees = pd.Series(csr["degrees"].astype(float))
    deg_by_id = pd.Series(degrees.to_numpy(), index=node_ids.to_numpy())
    df["degree"] = df["node_id"].map(deg_by_id)

    # Attach diffusion proxies if available.
    proxies_path = Path(args.proxies)
    if proxies_path.exists():
        df_p = _ensure_node_id_str(pd.read_parquet(proxies_path))
        need = ["node_id", "one_hop_spread", "two_hop_spread"]
        require_columns(df_p, need, "diffusion_proxies")
        df = df.merge(df_p[need], on="node_id", how="left")
    else:
        df["one_hop_spread"] = np.nan
        df["two_hop_spread"] = np.nan

    # Summary stats
    ic_a0 = pd.to_numeric(df["ic_a0"], errors="coerce")
    ic_a2 = pd.to_numeric(df["ic_a2"], errors="coerce")
    cv_a0 = float(ic_a0.std(ddof=0) / ic_a0.mean()) if float(ic_a0.mean()) > 0 else 0.0
    cv_a2 = float(ic_a2.std(ddof=0) / ic_a2.mean()) if float(ic_a2.mean()) > 0 else 0.0

    # Correlations
    rho_a0_a2 = _spearman(df["ic_a0"], df["ic_a2"])
    rho_a0_deg = _spearman(df["ic_a0"], df["degree"])
    rho_a2_deg = _spearman(df["ic_a2"], df["degree"])
    rho_a2_onehop = _spearman(df["ic_a2"], df["one_hop_spread"])
    rho_a2_twohop = _spearman(df["ic_a2"], df["two_hop_spread"])

    # Top-k agreement A0 vs A2
    ids = np.arange(len(df), dtype=np.int64)
    top_a0 = _topk_set(ic_a0.to_numpy(dtype=float), ids, float(args.top_pct))
    top_a2 = _topk_set(ic_a2.to_numpy(dtype=float), ids, float(args.top_pct))
    jaccard_top = _jaccard(top_a0, top_a2)

    out = {
        "timestamp": now_iso(),
        "n_rows_joined": int(df.shape[0]),
        "top_pct": float(args.top_pct),
        "cv": {"a0": cv_a0, "a2": cv_a2},
        "spearman": {
            "a0_vs_a2": rho_a0_a2,
            "a0_vs_degree": rho_a0_deg,
            "a2_vs_degree": rho_a2_deg,
            "a2_vs_one_hop_spread": rho_a2_onehop,
            "a2_vs_two_hop_spread": rho_a2_twohop,
        },
        "topk": {"jaccard_a0_vs_a2": jaccard_top},
        "notes": {
            "scope": "labeled subset only (same node_ids as primary)",
            "no_gnn_trained": True,
        },
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"[OK] Wrote A2 diagnostics: {out_path}")
    print(f" - n_rows={out['n_rows_joined']}, jaccard@{args.top_pct:.2f}={jaccard_top:.3f}")
    print(f" - rho(a0,a2)={rho_a0_a2:.3f}, rho(a2,deg)={rho_a2_deg:.3f}")


if __name__ == "__main__":
    main()
