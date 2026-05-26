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
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

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
    p.add_argument("--n-jobs", type=int, default=-1,
                   help="Parallel jobs for IC simulation in real mode")
    # M0-locked defaults — change only via explicit flag and update docs/m0_decisions.md
    p.add_argument("--n-sample", type=int, default=5000,
                   help="Number of labeled nodes to sample for IC (real mode only)")
    p.add_argument("--test-frac", type=float, default=0.20,
                   help="Held-out fraction for evaluation (M0-locked: 0.20)")
    p.add_argument(
        "--day1-decisions-path",
        default="docs/day1_decisions.md",
        help="Path to day1 decisions markdown for automated M3 alignment update",
    )
    p.add_argument(
        "--skip-day1-decisions-update",
        action="store_true",
        help="Disable automated M3 views/IC alignment section update",
    )
    p.add_argument(
        "--update-m3-only",
        action="store_true",
        help="Only recompute and refresh M3 views/IC alignment section from existing IC artifact",
    )
    return p.parse_args()


def _sample_labeled_indices(degrees: np.ndarray, n_sample: int, seed: int) -> np.ndarray:
    n_nodes = len(degrees)
    if n_sample <= 0:
        raise ValueError("n_sample must be > 0")
    if n_sample >= n_nodes:
        return np.arange(n_nodes, dtype=np.int64)

    all_idx = np.arange(n_nodes, dtype=np.int64)
    quintiles = pd.qcut(
        pd.Series(degrees.astype(float)), q=5, labels=False, duplicates="drop"
    ).to_numpy()

    try:
        _, sampled = train_test_split(
            all_idx,
            test_size=int(n_sample),
            random_state=seed,
            stratify=quintiles,
        )
    except ValueError:
        rng = np.random.default_rng(seed)
        sampled = rng.choice(all_idx, size=int(n_sample), replace=False)

    return np.sort(sampled.astype(np.int64))


def _simulate_ic_once(
    source: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    inv_degrees: np.ndarray,
    rng: np.random.Generator,
) -> int:
    activated = {int(source)}
    frontier = [int(source)]

    while frontier:
        next_frontier: list[int] = []
        for node in frontier:
            start_idx = int(indptr[node])
            end_idx = int(indptr[node + 1])
            for nb_raw in indices[start_idx:end_idx]:
                nb = int(nb_raw)
                if nb in activated:
                    continue
                p = float(inv_degrees[nb])
                if p <= 0.0:
                    continue
                if rng.random() < p:
                    activated.add(nb)
                    next_frontier.append(nb)
        frontier = next_frontier

    return len(activated)


def _simulate_ic_node_summary(
    source: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    inv_degrees: np.ndarray,
    n_runs: int,
    worker_seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(worker_seed)
    runs = np.empty(n_runs, dtype=np.int32)
    for i in range(n_runs):
        runs[i] = _simulate_ic_once(source, indptr, indices, inv_degrees, rng)
    return float(runs.mean()), float(runs.std(ddof=0))


def _real_ic_scores(
    node_ids: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    degrees: np.ndarray,
    n_sample: int,
    n_runs: int,
    seed: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    sampled_rows = _sample_labeled_indices(degrees=degrees, n_sample=n_sample, seed=seed)

    inv_degrees = np.zeros_like(degrees, dtype=float)
    mask = degrees > 0
    inv_degrees[mask] = 1.0 / degrees[mask].astype(float)

    def _worker(row: int) -> tuple[float, float]:
        return _simulate_ic_node_summary(
            source=int(row),
            indptr=indptr,
            indices=indices,
            inv_degrees=inv_degrees,
            n_runs=n_runs,
            worker_seed=seed + int(row),  # M0/plan: primary worker_seed = 42 + node_index
        )

    t0 = time.time()
    stats = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_worker)(int(row)) for row in sampled_rows
    )
    elapsed = time.time() - t0

    means = np.array([m for m, _ in stats], dtype=float)
    stds = np.array([s for _, s in stats], dtype=float)
    sampled_node_ids = node_ids[sampled_rows].astype(str)

    df_ic = pd.DataFrame(
        {
            "node_id": sampled_node_ids,
            "ic_score_mean": means,
            "ic_score_std": stds,
            "n_runs": int(n_runs),
            "p_model": "weighted_cascade",
        }
    )

    print(
        "[OK] Real IC simulation completed "
        f"(n_labeled={len(df_ic):,}, n_runs={n_runs}, elapsed_sec={elapsed:.2f})"
    )
    return df_ic, sampled_rows


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


def _rq2_narrative_tier(rho: float) -> str:
    if rho < 0.70:
        return "strong_divergence"
    if rho <= 0.85:
        return "moderate"
    return "high_agreement"


def _compute_views_ic_alignment(
    df_ic: pd.DataFrame,
    node_attrs_path: str | Path,
) -> tuple[float, float, int]:
    """Compute Spearman correlation between views and IC score over overlap nodes."""
    attrs = pd.read_parquet(node_attrs_path)
    require_columns(attrs, ["node_id", "views"], "node_attributes")

    df = df_ic[["node_id", "ic_score_mean"]].copy()
    df["node_id"] = df["node_id"].astype(str)

    attrs = attrs[["node_id", "views"]].copy()
    attrs["node_id"] = attrs["node_id"].astype(str)
    attrs["views"] = pd.to_numeric(attrs["views"], errors="coerce")

    merged = df.merge(attrs, on="node_id", how="inner").dropna(subset=["views", "ic_score_mean"])
    if len(merged) < 3:
        raise ValueError("Not enough overlap nodes to compute views/IC Spearman.")

    rho, pval = spearmanr(
        merged["views"].astype(float).to_numpy(),
        merged["ic_score_mean"].astype(float).to_numpy(),
    )
    if not np.isfinite(rho) or not np.isfinite(pval):
        raise ValueError("Views/IC Spearman produced non-finite values.")
    return float(rho), float(pval), int(len(merged))


def _render_m3_section(rho: float, pval: float, n_overlap: int) -> str:
    tier = _rq2_narrative_tier(rho)
    return "\n".join(
        [
            "## 13) M3 Views/IC Alignment Check (RQ2 Narrative Lookup)",
            "",
            "Scope:",
            "- Source join: `data/processed/ic_scores_primary.parquet` x `data/processed/node_attributes.parquet`",
            f"- Overlap nodes used: {n_overlap}",
            "",
            "Measured result:",
            f"- `spearmanr(views, ic_score_mean) = {rho}`",
            f"- `p_value = {pval}`",
            "",
            "Narrative tier (from M3 lookup table):",
            f"- `{tier}`",
            "",
            "Locked RQ2 narrative for this cycle:",
            "- Popularity (`views`) does not reliably represent diffusion potential (`ic_score_mean`) on this graph.",
            "- Hidden influencers are expected and should be interpreted through structural signals (betweenness / cross-community connectivity), not raw popularity alone.",
        ]
    )


def _update_day1_decisions_m3(
    day1_path: str | Path,
    rho: float,
    pval: float,
    n_overlap: int,
) -> None:
    p = Path(day1_path)
    if not p.exists():
        return

    text = p.read_text(encoding="utf-8")
    marker = "## 13) M3 Views/IC Alignment Check (RQ2 Narrative Lookup)"
    new_section = _render_m3_section(rho=rho, pval=pval, n_overlap=n_overlap)

    if marker in text:
        prefix = text.split(marker)[0].rstrip()
        updated = prefix + "\n\n" + new_section + "\n"
    else:
        updated = text.rstrip() + "\n\n" + new_section + "\n"

    p.write_text(updated, encoding="utf-8")


def main() -> None:
    args = parse_args()
    csr_path = Path(args.csr)
    node_ids: np.ndarray

    if args.update_m3_only:
        df_existing = pd.read_parquet(args.out_ic)
        require_columns(df_existing, ["node_id", "ic_score_mean"], "ic_scores_existing")
        rho, pval, n_overlap = _compute_views_ic_alignment(
            df_ic=df_existing,
            node_attrs_path=PATHS.node_attributes,
        )
        if not args.skip_day1_decisions_update:
            _update_day1_decisions_m3(
                day1_path=args.day1_decisions_path,
                rho=rho,
                pval=pval,
                n_overlap=n_overlap,
            )
            print(
                f"[OK] Updated M3 section in {args.day1_decisions_path} "
                f"(rho={rho:.6f}, p={pval:.3e}, n={n_overlap})"
            )
        return

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

    if args.dry_run:
        df_ic = _mock_ic_scores(
            node_ids=node_ids, sis_path=Path(args.mock_from_sis), seed=args.seed, n_runs=args.n_runs
        )
        sampled_rows = None
    else:
        csr = load_csr_npz(csr_path)
        df_ic, sampled_rows = _real_ic_scores(
            node_ids=csr["node_ids"],
            indptr=csr["indptr"],
            indices=csr["indices"],
            degrees=csr["degrees"],
            n_sample=int(args.n_sample),
            n_runs=int(args.n_runs),
            seed=int(args.seed),
            n_jobs=int(args.n_jobs),
        )

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
        if sampled_rows is not None:
            degrees_aligned = csr["degrees"][sampled_rows].astype(np.int64)
        else:
            # Dry-run path: align by node_id lookup
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
        "[OK] Wrote IC artifacts: "
        f"{args.out_ic}, {args.out_reg}, {args.out_cls}, {args.out_mask} "
        f"(split: {n_train} train / {n_test} test, timestamp={now_iso()})"
    )

    if not args.skip_day1_decisions_update and not args.dry_run:
        try:
            rho, pval, n_overlap = _compute_views_ic_alignment(
                df_ic=df_ic,
                node_attrs_path=PATHS.node_attributes,
            )
            _update_day1_decisions_m3(
                day1_path=args.day1_decisions_path,
                rho=rho,
                pval=pval,
                n_overlap=n_overlap,
            )
            print(
                f"[OK] Updated M3 section in {args.day1_decisions_path} "
                f"(rho={rho:.6f}, p={pval:.3e}, n={n_overlap})"
            )
        except Exception as exc:
            print(f"[WARN] Could not update M3 section automatically: {exc}")


if __name__ == "__main__":
    main()
