"""MAPR2026 v3 — IC labels (primary: weighted cascade).

Owner: Person 1 (IC core)

Inputs
------
- data/processed/graph_csr.npz

Optional inputs for dry-run mocking
---------------------------------
- data/processed/sis_table.parquet (to create mock IC scores)

Outputs (contract)
------------------
- data/processed/ic_scores_primary.parquet
  columns: node_id, ic_score_mean, ic_score_std, n_runs, p_model
- data/processed/regression_targets.parquet
  columns: node_id, y (y=log1p(ic_score_mean))
- data/processed/classification_labels.parquet
  columns: node_id, y_top10
- data/processed/split_masks.parquet  [M0-locked]
  columns: node_id, split ('train' | 'test')
  rule: test_frac=0.20, stratify=degree_quintile (q=5), seed=42
  scope: labeled nodes only (same set as ic_scores_primary)

Scaffold behavior
-----------------
- Default mode does not run full IC labeling.
- Use --dry-run to generate mock outputs that satisfy schema contracts.
- Real mode: stratified sampling (--n-sample) → MC IC → split mask.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _shared import PATHS, ensure_parent, load_csr_npz, now_iso, require_columns


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MAPR2026 v3 IC labels (primary) scaffold")
    p.add_argument("--csr", default=PATHS.csr_npz)
    p.add_argument("--out-ic", default=PATHS.ic_scores)
    p.add_argument("--out-reg", default=PATHS.regression_targets)
    p.add_argument("--out-cls", default=PATHS.classification_labels)
    p.add_argument("--out-mask", default=PATHS.split_masks,
                   help="Path for split_masks.parquet (M0-locked: 80/20, degree-stratified, seed=42)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--mock-from-sis", default=PATHS.sis_table)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-runs", type=int, default=50)
    # M0-locked defaults — change only via explicit flag and update docs/m0_decisions.md
    p.add_argument("--n-sample", type=int, default=5000,
                   help="Number of labeled nodes to sample for IC (real mode only)")
    p.add_argument("--test-frac", type=float, default=0.20,
                   help="Held-out fraction for evaluation (M0-locked: 0.20)")
    return p.parse_args()


def _mock_ic_scores(node_ids: np.ndarray, sis_path: Path | None, seed: int, n_runs: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    if sis_path is not None and sis_path.exists():
        df = pd.read_parquet(sis_path)
        # Best-effort: accept either 'node_id' or index-like naming.
        if "node_id" not in df.columns:
            raise ValueError("sis_table.parquet must contain node_id for mock mode")

        # Try common SIS score column names
        score_col = None
        for candidate in ["sis_score", "sis", "score"]:
            if candidate in df.columns:
                score_col = candidate
                break
        if score_col is None:
            # Fall back to pagerank if present
            for candidate in ["pagerank", "betweenness", "kshell"]:
                if candidate in df.columns:
                    score_col = candidate
                    break
        if score_col is None:
            raise ValueError("sis_table.parquet lacks a usable score column for mocking")

        df = df[["node_id", score_col]].rename(columns={score_col: "ic_score_mean"})
        df = df[df["node_id"].isin(set(node_ids))].copy()
        if len(df) == 0:
            raise ValueError("Mock mode: no overlap between CSR node_ids and sis_table node_id")

        df["ic_score_mean"] = df["ic_score_mean"].astype(float)
        df["ic_score_std"] = rng.normal(loc=0.0, scale=1e-6, size=len(df))
        df["n_runs"] = int(n_runs)
        df["p_model"] = "dry_run_mock_from_sis"
        return df

    # If SIS not available, create random mock scores.
    mock_mean = rng.random(size=len(node_ids))
    return pd.DataFrame(
        {
            "node_id": node_ids,
            "ic_score_mean": mock_mean,
            "ic_score_std": np.zeros_like(mock_mean),
            "n_runs": int(n_runs),
            "p_model": "dry_run_random",
        }
    )


def _create_split_mask(
    df_ic: pd.DataFrame,
    degrees: np.ndarray | None = None,
    test_frac: float = 0.20,
    seed: int = 42,
) -> pd.DataFrame:
    """Create degree-stratified train/test split mask over labeled nodes.

    M0-locked rule: test_frac=0.20, stratify=degree_quintile (q=5), seed=42.
    If degrees are not available (dry-run without CSR), falls back to random split.

    Parameters
    ----------
    df_ic    : DataFrame with 'node_id' column — the labeled node set.
    degrees  : Degree array aligned with df_ic rows. None → random split.
    test_frac: Fraction held out for test (M0 default: 0.20).
    seed     : Random seed (M0 default: 42).

    Returns
    -------
    DataFrame with columns [node_id (str), split (str: 'train'|'test')].
    """
    from sklearn.model_selection import train_test_split

    node_ids = df_ic["node_id"].to_numpy()
    n = len(node_ids)
    indices = np.arange(n)

    stratify_labels: np.ndarray | None = None
    if degrees is not None and len(degrees) == n:
        quintiles = pd.qcut(
            pd.Series(degrees.astype(float)), q=5, labels=False, duplicates="drop"
        )
        stratify_labels = quintiles.to_numpy()

    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_frac,
        stratify=stratify_labels,
        random_state=seed,
    )

    split_arr = np.full(n, "train", dtype=object)
    split_arr[test_idx] = "test"

    return pd.DataFrame(
        {"node_id": node_ids.astype(str), "split": split_arr.astype(str)}
    )


def main() -> None:
    args = parse_args()
    csr_path = Path(args.csr)
    node_ids: np.ndarray

    if csr_path.exists():
        csr = load_csr_npz(csr_path)
        node_ids = csr["node_ids"]
    else:
        if not args.dry_run:
            raise FileNotFoundError(
                f"Missing CSR artifact: {csr_path}. Run export_csr.py first (or use --dry-run)."
            )

        # Dry-run fallback: derive node_ids from SIS table or node attributes.
        sis_path = Path(args.mock_from_sis)
        attrs_path = Path(PATHS.node_attributes)

        if sis_path.exists():
            df_sis = pd.read_parquet(sis_path)
            if "node_id" not in df_sis.columns:
                raise ValueError("sis_table.parquet must contain node_id for dry-run fallback")
            node_ids = df_sis["node_id"].astype(str).unique()
        elif attrs_path.exists():
            df_attrs = pd.read_parquet(attrs_path)
            if "node_id" not in df_attrs.columns:
                raise ValueError("node_attributes.parquet must contain node_id for dry-run fallback")
            node_ids = df_attrs["node_id"].astype(str).unique()
        else:
            raise FileNotFoundError(
                "Dry-run fallback requires either sis_table.parquet or node_attributes.parquet to exist "
                "when CSR is missing."
            )

    if not args.dry_run:
        raise NotImplementedError(
            "Implement weighted-cascade IC Monte Carlo labeling on CSR (plus stability checks). "
            "Run with --dry-run to generate mock schema-correct outputs."
        )

    df_ic = _mock_ic_scores(node_ids=node_ids, sis_path=Path(args.mock_from_sis), seed=args.seed, n_runs=args.n_runs)
    require_columns(df_ic, ["node_id", "ic_score_mean", "ic_score_std", "n_runs", "p_model"], "ic_scores")

    ensure_parent(args.out_ic)
    df_ic.to_parquet(args.out_ic, index=False)

    df_reg = df_ic[["node_id", "ic_score_mean"]].copy()
    df_reg["y"] = np.log1p(df_reg["ic_score_mean"].astype(float))
    df_reg = df_reg[["node_id", "y"]]
    df_reg.to_parquet(args.out_reg, index=False)

    # Top-10% classification label
    thresh = df_ic["ic_score_mean"].quantile(0.90)
    df_cls = df_ic[["node_id", "ic_score_mean"]].copy()
    df_cls["y_top10"] = (df_cls["ic_score_mean"] >= thresh).astype(int)
    df_cls = df_cls[["node_id", "y_top10"]]
    df_cls.to_parquet(args.out_cls, index=False)

    # Split mask (M0-locked: 80/20, degree-stratified if CSR available, seed=42)
    degrees_aligned: np.ndarray | None = None
    if csr_path.exists():
        csr = load_csr_npz(csr_path)
        # Align degrees to df_ic rows via node_id lookup
        id_to_deg = dict(zip(csr["node_ids"].tolist(), csr["degrees"].tolist()))
        degrees_aligned = np.array(
            [id_to_deg.get(nid, 0) for nid in df_ic["node_id"].tolist()], dtype=np.int64
        )
    df_mask = _create_split_mask(df_ic, degrees=degrees_aligned,
                                 test_frac=args.test_frac, seed=args.seed)
    require_columns(df_mask, ["node_id", "split"], "split_masks")
    ensure_parent(args.out_mask)
    df_mask.to_parquet(args.out_mask, index=False)

    n_test = int((df_mask["split"] == "test").sum())
    n_train = len(df_mask) - n_test
    print(
        "[OK] Wrote dry-run IC artifacts: "
        f"{args.out_ic}, {args.out_reg}, {args.out_cls}, {args.out_mask} "
        f"(split: {n_train} train / {n_test} test, timestamp={now_iso()})"
    )


if __name__ == "__main__":
    main()
