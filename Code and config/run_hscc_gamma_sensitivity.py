"""MAPR2026 v3 — HSCC cross-community amplification (γ) mechanism sensitivity.

This is a *mechanism check*, not a new primary regime: only γ varies; λ, p_max,
Louvain partition, labeled node set (from primary IC artifact), split masks,
and MC budget stay fixed.

Pipeline per γ (except γ=1.0 which can reuse frozen main artifacts):
1) Run ``ic_labels_hscc_refined.py`` with the same labeled IDs as primary IC.
2) Record label/reach diagnostics and optional Spearman structure on the test mask.
3) Optionally evaluate a minimal model set (degree, LR attr, LR 1-hop, GraphSAGE)
   and paired bootstrap Δρ vs frozen test predictions.

γ=0 is *not* A0: it keeps HSCC source term φ(u)/deg(u) and only disables the
(1 + γ·𝟙[c_u≠c_v]) cross-community boost.

LR 1-hop defaults to ``parquet_train`` fit mask so it matches
``bootstrap_ci._build_linear_predictions`` (all ``split==train`` ids). Use
``--lr-1hop-fit-mask surrogate_train`` only when matching the graph-augmented
flat baseline script default (train minus a fixed 10% validation slice).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from _shared import PATHS, ensure_parent, load_csr_npz, require_columns, write_json
from eval_ranking_harness import apply_test_mask, compute_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAP_DIR = Path(__file__).resolve().parent


def resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _ensure_node_id_str(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["node_id"] = out["node_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    return out


def gamma_file_tag(gamma: float) -> str:
    if abs(gamma - 1.0) < 1e-9:
        return "1"
    if abs(gamma) < 1e-9:
        return "0"
    if abs(gamma - 0.5) < 1e-9:
        return "0p5"
    s = f"{float(gamma):g}".replace(".", "p")
    return s.replace("-", "m")


def _spearman_on_test(
    targets_df: pd.DataFrame,
    pred_series: pd.Series | np.ndarray,
    split_mask_df: pd.DataFrame,
    node_id_col: str = "node_id",
) -> float:
    """Spearman between y and a per-node scalar on held-out test only."""
    df = _ensure_node_id_str(targets_df[[node_id_col, "y"]].copy())
    df["pred"] = np.asarray(pred_series, dtype=float)
    test_df = apply_test_mask(df, split_mask_df, node_id_col=node_id_col)
    if len(test_df) < 3:
        return float("nan")
    r = spearmanr(test_df["y"].to_numpy(dtype=float), test_df["pred"].to_numpy(dtype=float))
    stat = getattr(r, "statistic", r[0])
    return float(np.asarray(stat).item())


def _run_ic_labeling_subprocess(
    gamma: float,
    out_ic: Path,
    out_reg: Path,
    out_diag: Path,
    csr: Path,
    node_attrs: Path,
    community: Path,
    primary_ic: Path,
    n_runs: int,
    seed: int,
    device: str,
    n_jobs: int,
) -> float:
    """Invoke ic_labels_hscc_refined.py; return wall-clock seconds."""
    script = MAP_DIR / "ic_labels_hscc_refined.py"
    cmd = [
        sys.executable,
        str(script),
        "--gamma",
        str(float(gamma)),
        "--lambda-coef",
        "1.0",
        "--p-max",
        "1.0",
        "--csr",
        str(csr),
        "--node-attrs",
        str(node_attrs),
        "--community",
        str(community),
        "--primary-ic",
        str(primary_ic),
        "--out-ic",
        str(out_ic),
        "--out-reg",
        str(out_reg),
        "--out-diag",
        str(out_diag),
        "--n-runs",
        str(int(n_runs)),
        "--seed",
        str(int(seed)),
        "--device",
        str(device),
        "--n-jobs",
        str(int(n_jobs)),
    ]
    t0 = time.time()
    subprocess.run(cmd, cwd=str(MAP_DIR), check=True)
    return float(time.time() - t0)


def _label_reach_diagnostics(ic_df: pd.DataFrame) -> dict[str, float]:
    m = pd.to_numeric(ic_df["ic_score_mean"], errors="coerce").astype(float)
    mu = float(m.mean())
    sd = float(m.std(ddof=0))
    return {
        "mean_reach": mu,
        "std_reach": sd,
        "max_reach": float(m.max()),
        "cv_reach": float(sd / (mu + 1e-12)),
    }


def _degree_series_for_targets(
    targets_df: pd.DataFrame,
    node_attributes: pd.DataFrame,
) -> pd.Series:
    attrs = _ensure_node_id_str(node_attributes)
    require_columns(attrs, ["node_id", "degree"], "node_attributes for degree baseline")
    m = targets_df[["node_id"]].merge(attrs[["node_id", "degree"]], on="node_id", how="left")
    return pd.to_numeric(m["degree"], errors="coerce").fillna(0.0)


def _phi_series_for_targets(targets_df: pd.DataFrame, node_attrs_path: Path) -> pd.Series:
    from ic_labels_hscc_refined import _load_source_strength  # noqa: PLC0415

    phi_df = _load_source_strength(node_attrs_path)
    m = targets_df[["node_id"]].merge(phi_df, on="node_id", how="left")
    return pd.to_numeric(m["phi"], errors="coerce").fillna(0.0)


def _lr_1hop_predictions(
    targets_df: pd.DataFrame,
    split_mask_df: pd.DataFrame,
    csr: dict[str, Any],
    node_attributes: pd.DataFrame,
    include_language: bool,
    fit_mask: str,
) -> np.ndarray:
    from run_graph_augmented_flat_baseline import (  # noqa: PLC0415
        _build_design_matrix,
        _fit_predict_linear,
        _labeled_masks_for_fit,
    )

    id_to_idx = {str(nid): i for i, nid in enumerate(csr["node_ids"].astype(str))}
    target_ids = targets_df["node_id"].astype(str).tolist()
    missing = [n for n in target_ids if n not in id_to_idx]
    if missing:
        raise ValueError(f"{len(missing)} targets missing from CSR (examples {missing[:3]})")
    x_full, _ = _build_design_matrix(
        csr,
        node_attributes,
        None,
        include_language=include_language,
        include_comm=False,
    )
    idx = np.array([id_to_idx[n] for n in target_ids], dtype=np.int64)
    x_lab = x_full[idx]
    y = pd.to_numeric(targets_df["y"], errors="coerce").to_numpy(dtype=np.float64)
    train_m, _, _ = _labeled_masks_for_fit(targets_df["node_id"], split_mask_df, fit_mask)
    # Returns (pred, scaler, reg, train_sec, inference_sec) — see run_graph_augmented_flat_baseline.
    pred, _, _, _, _ = _fit_predict_linear(x_lab, y, train_m)
    return pred


def _try_load_sage_parquet(sage_dir: Path, tag: str) -> pd.DataFrame | None:
    """Optional frozen GraphSAGE predictions: ``sage_predictions_hscc_gamma_{tag}.parquet``."""
    path = sage_dir / f"sage_predictions_hscc_gamma_{tag}.parquet"
    if not path.is_file():
        return None
    df = _ensure_node_id_str(pd.read_parquet(path))
    require_columns(df, ["node_id", "y_pred"], f"SAGE predictions {path.name}")
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
    return df[["node_id", "y_pred"]]


def main() -> None:
    p = argparse.ArgumentParser(description="HSCC γ mechanism sensitivity (fixed sample, split, partition)")
    p.add_argument("--gammas", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    p.add_argument("--work-dir", default=str(Path(PATHS.results_dir) / "hscc_gamma_sensitivity"))
    p.add_argument("--csr-npz-path", default=PATHS.csr_npz)
    p.add_argument("--node-attributes-path", default=PATHS.node_attributes)
    p.add_argument("--community-labels-path", default="data/processed/community_labels.parquet")
    p.add_argument("--split-mask-path", default=PATHS.split_masks)
    p.add_argument("--primary-ic-path", default=PATHS.ic_scores, help="Locks labeled node_ids + n_runs.")
    p.add_argument(
        "--gamma1-targets-path",
        default="data/processed/regression_targets_hscc_refined.parquet",
        help="Frozen γ=1.0 regression targets (main HSCC); used as ref y and default paths for γ=1.",
    )
    p.add_argument("--gamma1-ic-path", default=str(Path(PATHS.results_dir) / "ic_scores_hscc_refined.parquet"))
    p.add_argument("--gamma1-diag-path", default=str(Path(PATHS.results_dir) / "hscc_refined_label_diagnostics.json"))
    p.add_argument("--n-runs", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--skip-labeling", action="store_true", help="Skip MC-IC if outputs already exist.")
    p.add_argument(
        "--regenerate-gamma1",
        action="store_true",
        help="If set, rerun labeling for γ=1 into work-dir instead of using frozen main artifacts.",
    )
    p.add_argument("--eval-models", action="store_true", help="Run degree / LR attr / LR 1-hop / GraphSAGE + bootstrap.")
    p.add_argument("--include-language", action="store_true")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--bootstrap-seed", type=int, default=42)
    p.add_argument("--equivalence-bound", type=float, default=0.02)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--seeds", default="", help="Comma-separated GNN seeds (default 42,123,456,789,1024).")
    p.add_argument("--hidden-channels", type=int, default=128)
    p.add_argument("--appnp-alpha", type=float, default=0.15)
    p.add_argument("--appnp-k", type=int, default=10)
    p.add_argument("--gat-heads", type=int, default=4)
    p.add_argument("--skip-sage", action="store_true", help="Skip GraphSAGE train + SAGE-related bootstrap pairs.")
    p.add_argument(
        "--lr-1hop-fit-mask",
        default="parquet_train",
        choices=["parquet_train", "surrogate_train"],
        help="LR 1-hop training mask: parquet_train matches bootstrap_ci._build_linear_predictions; "
        "surrogate_train matches run_graph_augmented_flat_baseline default (train minus first 10%% val slice).",
    )
    p.add_argument(
        "--sage-predictions-dir",
        default="",
        help="Optional directory with sage_predictions_hscc_gamma_{tag}.parquet (node_id, y_pred). "
        "If a file exists for this γ tag, load it instead of retraining GraphSAGE.",
    )
    p.add_argument(
        "--out-summary-csv",
        default=str(Path(PATHS.results_dir) / "hscc_gamma_sensitivity_summary.csv"),
    )
    p.add_argument(
        "--out-bootstrap-json",
        default=str(Path(PATHS.results_dir) / "hscc_gamma_sensitivity_bootstrap.json"),
    )
    args = p.parse_args()

    work_dir = resolve_project_path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    csr_path = resolve_project_path(args.csr_npz_path)
    attrs_path = resolve_project_path(args.node_attributes_path)
    comm_path = resolve_project_path(args.community_labels_path)
    split_path = resolve_project_path(args.split_mask_path)
    primary_ic = resolve_project_path(args.primary_ic_path)

    split_df = pd.read_parquet(split_path)
    split_df = _ensure_node_id_str(split_df)
    node_attrs = pd.read_parquet(attrs_path)
    csr = load_csr_npz(csr_path)

    ref_targets = _ensure_node_id_str(pd.read_parquet(resolve_project_path(args.gamma1_targets_path)))
    require_columns(ref_targets, ["node_id", "y"], "ref gamma=1 targets")

    gammas = [float(g) for g in args.gammas]
    summary_rows: list[dict[str, Any]] = []
    bootstrap_payload: dict[str, Any] = {
        "role": "hscc_gamma_mechanism_sensitivity",
        "gammas": gammas,
        "fixed": {
            "lambda": 1.0,
            "p_max": 1.0,
            "louvain_partition": str(comm_path),
            "split_mask": str(split_path),
            "primary_ic_node_lock": str(primary_ic),
            "n_runs": int(args.n_runs),
        },
        "fit_mask_policy": {
            "lr_attr": "parquet_train",
            "lr_1hop": str(args.lr_1hop_fit_mask),
            "note": "LR attr uses all split==train node_ids (bootstrap_ci._build_linear_predictions). "
            "LR 1-hop uses _labeled_masks_for_fit with the chosen lr_1hop mode.",
        },
        "per_gamma": [],
    }

    from bootstrap_ci import (  # noqa: PLC0415
        _bootstrap_spearman_ndcg_ci,
        _build_linear_predictions,
        _interpret_ci,
        _predict_gnn_best,
    )

    seed_list = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip() != ""]
    if not seed_list:
        seed_list = [42, 123, 456, 789, 1024]

    sage_dir: Path | None = None
    if str(args.sage_predictions_dir).strip():
        sage_dir = resolve_project_path(str(args.sage_predictions_dir).strip())

    for gamma in gammas:
        tag = gamma_file_tag(gamma)
        is_main = abs(gamma - 1.0) < 1e-9 and not bool(args.regenerate_gamma1)

        label_source: str
        label_wall_sec: float | None

        if is_main:
            out_ic = resolve_project_path(args.gamma1_ic_path)
            out_reg = resolve_project_path(args.gamma1_targets_path)
            out_diag = resolve_project_path(args.gamma1_diag_path)
            label_source = "frozen_main"
            label_wall_sec = None
            if not args.skip_labeling:
                print(f"[INFO] γ={gamma}: using frozen main artifacts (skip labeling).")
        else:
            out_ic = work_dir / f"ic_scores_hscc_gamma_{tag}.parquet"
            out_reg = work_dir / f"regression_targets_hscc_gamma_{tag}.parquet"
            out_diag = work_dir / f"hscc_gamma_{tag}_label_diagnostics.json"
            if args.skip_labeling and out_ic.exists() and out_reg.exists():
                print(f"[INFO] γ={gamma}: skip-labeling, found {out_ic.name}")
                label_source = "skipped_existing"
                label_wall_sec = None
            else:
                print(f"[INFO] γ={gamma}: running HSCC MC-IC labeling → {out_reg.name}")
                label_source = "generated"
                label_wall_sec = _run_ic_labeling_subprocess(
                    gamma=gamma,
                    out_ic=out_ic,
                    out_reg=out_reg,
                    out_diag=out_diag,
                    csr=csr_path,
                    node_attrs=attrs_path,
                    community=comm_path,
                    primary_ic=primary_ic,
                    n_runs=int(args.n_runs),
                    seed=int(args.seed),
                    device=str(args.device),
                    n_jobs=int(args.n_jobs),
                )

        ic_df = _ensure_node_id_str(pd.read_parquet(out_ic))
        require_columns(ic_df, ["node_id", "ic_score_mean"], "IC scores (gamma sensitivity)")
        targets_df = _ensure_node_id_str(pd.read_parquet(out_reg))
        require_columns(targets_df, ["node_id", "y"], "regression targets")
        targets_df["y"] = pd.to_numeric(targets_df["y"], errors="coerce")

        reach_d = _label_reach_diagnostics(ic_df)
        deg_series = _degree_series_for_targets(targets_df, node_attrs)
        phi_series = _phi_series_for_targets(targets_df, attrs_path)

        rho_deg = _spearman_on_test(targets_df, deg_series.to_numpy(), split_df)
        rho_phi = _spearman_on_test(targets_df, phi_series.to_numpy(), split_df)
        merged_y = targets_df[["node_id", "y"]].merge(ref_targets.rename(columns={"y": "y_ref"}), on="node_id", how="inner")
        if len(merged_y) >= 3:
            r_all = spearmanr(
                merged_y["y"].to_numpy(dtype=float),
                merged_y["y_ref"].to_numpy(dtype=float),
            )
            stat_all = getattr(r_all, "statistic", r_all[0])
            rho_vs_ref_all = float(np.asarray(stat_all).item())
        else:
            rho_vs_ref_all = float("nan")
        test_merged = apply_test_mask(merged_y, split_df, node_id_col="node_id")
        if len(test_merged) >= 3:
            r_g1 = spearmanr(
                test_merged["y"].to_numpy(dtype=float),
                test_merged["y_ref"].to_numpy(dtype=float),
            )
            stat = getattr(r_g1, "statistic", r_g1[0])
            rho_vs_ref = float(np.asarray(stat).item())
        else:
            rho_vs_ref = float("nan")

        n_test = len(
            apply_test_mask(targets_df[["node_id", "y"]], split_df, node_id_col="node_id"),
        )

        row: dict[str, Any] = {
            "gamma": float(gamma),
            "gamma_tag": tag,
            "n_labeled": int(len(targets_df)),
            "n_test": int(n_test),
            "mean_reach": reach_d["mean_reach"],
            "std_reach": reach_d["std_reach"],
            "max_reach": reach_d["max_reach"],
            "cv_reach": reach_d["cv_reach"],
            "spearman_y_degree_test": rho_deg,
            "spearman_y_phi_test": rho_phi,
            "spearman_y_vs_gamma1_y_test": rho_vs_ref,
            "spearman_y_vs_gamma1_y_all": rho_vs_ref_all,
            "ic_scores_path": str(out_ic),
            "regression_targets_path": str(out_reg),
            "label_source": label_source,
            "label_generation_sec": float(label_wall_sec) if label_wall_sec is not None else float("nan"),
        }

        gamma_boot: dict[str, Any] = {
            "gamma": float(gamma),
            "gamma_tag": tag,
            "label_source": label_source,
            "n_test_eval_nodes": int(n_test),
            "spearman_y_vs_gamma1_y_all": rho_vs_ref_all,
            "spearman_y_vs_gamma1_y_test": rho_vs_ref,
        }

        if args.eval_models:
            deg_pred = deg_series.to_numpy(dtype=float)
            ev_deg = apply_test_mask(
                targets_df.assign(y_pred=deg_pred),
                split_df,
                node_id_col="node_id",
            )
            m_deg = compute_metrics(ev_deg["y"].to_numpy(float), ev_deg["y_pred"].to_numpy(float))

            from bootstrap_ci import _import_run_baselines_symbols  # noqa: PLC0415

            derive_features = _import_run_baselines_symbols()["derive_features"]
            derived = derive_features(node_attrs, include_language=bool(args.include_language))
            lang_cols = [c for c in derived.columns if str(c).startswith("lang_")]
            if bool(args.include_language) and lang_cols:
                feat_cols = ["degree", "views_log", "life_time", *sorted(lang_cols)]
            else:
                feat_cols = ["degree", "views_log", "life_time"]

            lr_attr_pred = _build_linear_predictions(
                targets_df=targets_df,
                split_mask_df=split_df,
                node_attributes=node_attrs,
                feature_cols=feat_cols,
                include_language=bool(args.include_language),
            )
            ev_lr = apply_test_mask(
                pd.DataFrame({"node_id": targets_df["node_id"].astype(str), "y": targets_df["y"], "y_pred": lr_attr_pred}),
                split_df,
                node_id_col="node_id",
            )
            m_lr = compute_metrics(ev_lr["y"].to_numpy(float), ev_lr["y_pred"].to_numpy(float))

            lr_1hop_pred = _lr_1hop_predictions(
                targets_df,
                split_df,
                csr,
                node_attrs,
                bool(args.include_language),
                str(args.lr_1hop_fit_mask),
            )
            ev_1h = apply_test_mask(
                pd.DataFrame({"node_id": targets_df["node_id"].astype(str), "y": targets_df["y"], "y_pred": lr_1hop_pred}),
                split_df,
                node_id_col="node_id",
            )
            m_1h = compute_metrics(ev_1h["y"].to_numpy(float), ev_1h["y_pred"].to_numpy(float))

            row["degree_model_spearman"] = float(m_deg.spearman_rho)
            row["lr_attr_spearman"] = float(m_lr.spearman_rho)
            row["lr_1hop_spearman"] = float(m_1h.spearman_rho)

            y_test = apply_test_mask(targets_df[["node_id", "y"]], split_df, node_id_col="node_id")

            if not args.skip_sage:
                sage_pred_path = (sage_dir / f"sage_predictions_hscc_gamma_{tag}.parquet") if sage_dir is not None else None
                sage_parquet = _try_load_sage_parquet(sage_dir, tag) if sage_dir is not None else None
                if sage_parquet is not None:
                    assert sage_pred_path is not None
                    print(f"[INFO] γ={gamma}: loading GraphSAGE predictions from {sage_pred_path}")
                    aligned = targets_df[["node_id"]].astype(str).merge(sage_parquet, on="node_id", how="left")
                    if bool(aligned["y_pred"].isna().any()):
                        miss = int(aligned["y_pred"].isna().sum())
                        raise ValueError(f"SAGE parquet missing {miss} labeled node predictions")
                    sage_pred = aligned["y_pred"].to_numpy(dtype=float)
                    gamma_boot["sage_source"] = "parquet"
                    gamma_boot["sage_predictions_path"] = str(sage_pred_path)
                else:
                    if sage_dir is not None:
                        print(f"[INFO] γ={gamma}: no SAGE parquet for tag {tag}; retraining GraphSAGE …")
                    print(f"[INFO] γ={gamma}: training GraphSAGE (gnn_raw_attr) — may require torch_geometric …")
                    targets_parquet = str(out_reg if out_reg.is_absolute() else resolve_project_path(out_reg))
                    _split_df, sage_df = _predict_gnn_best(
                        targets_path=targets_parquet,
                        split_mask_path=str(split_path),
                        model_name="gnn_raw_attr",
                        max_epochs=int(args.max_epochs),
                        seeds=seed_list,
                        include_language=bool(args.include_language),
                        gat_heads=int(args.gat_heads),
                        appnp_alpha=float(args.appnp_alpha),
                        appnp_k=int(args.appnp_k),
                        hidden_channels=int(args.hidden_channels),
                    )
                    order = targets_df["node_id"].astype(str).tolist()
                    pred_map = dict(zip(sage_df["node_id"].astype(str), sage_df["y_pred"].astype(float)))
                    sage_pred = np.array([pred_map[str(n)] for n in order], dtype=np.float64)
                    gamma_boot["sage_source"] = "retrained"

                ev_sg = apply_test_mask(
                    pd.DataFrame({"node_id": targets_df["node_id"].astype(str), "y_pred": sage_pred}),
                    split_df,
                    node_id_col="node_id",
                )
                merged_sg = y_test.merge(ev_sg, on="node_id", how="inner")
                m_sg = compute_metrics(
                    merged_sg["y"].to_numpy(float),
                    merged_sg["y_pred"].to_numpy(float),
                )
                row["sage_spearman"] = float(m_sg.spearman_rho)
                row["sage_source"] = str(gamma_boot.get("sage_source", ""))

                pred_lr_df = pd.DataFrame({"node_id": targets_df["node_id"].astype(str), "y_pred": lr_attr_pred})
                pred_lr_t = apply_test_mask(pred_lr_df, split_df, node_id_col="node_id")
                m2 = merged_sg.merge(pred_lr_t.rename(columns={"y_pred": "y_pred_lr"}), on="node_id", how="inner")
                ci_lr = _bootstrap_spearman_ndcg_ci(
                    y_true=m2["y"].to_numpy(float),
                    y_pred_a=m2["y_pred"].to_numpy(float),
                    y_pred_b=m2["y_pred_lr"].to_numpy(float),
                    n_bootstrap=int(args.n_bootstrap),
                    seed=int(args.bootstrap_seed),
                )
                pred_1h_df = pd.DataFrame({"node_id": targets_df["node_id"].astype(str), "y_pred": lr_1hop_pred})
                pred_1h_t = apply_test_mask(pred_1h_df, split_df, node_id_col="node_id")
                m3 = merged_sg.merge(pred_1h_t.rename(columns={"y_pred": "y_pred_1h"}), on="node_id", how="inner")
                ci_1h = _bootstrap_spearman_ndcg_ci(
                    y_true=m3["y"].to_numpy(float),
                    y_pred_a=m3["y_pred"].to_numpy(float),
                    y_pred_b=m3["y_pred_1h"].to_numpy(float),
                    n_bootstrap=int(args.n_bootstrap),
                    seed=int(args.bootstrap_seed),
                )

                row["delta_sage_vs_lr_attr"] = float(m_sg.spearman_rho - m_lr.spearman_rho)
                row["ci_sage_vs_lr_attr_lower"] = ci_lr["spearman_ci_95_lower"]
                row["ci_sage_vs_lr_attr_upper"] = ci_lr["spearman_ci_95_upper"]
                row["delta_sage_vs_lr_1hop"] = float(m_sg.spearman_rho - m_1h.spearman_rho)
                row["ci_sage_vs_lr_1hop_lower"] = ci_1h["spearman_ci_95_lower"]
                row["ci_sage_vs_lr_1hop_upper"] = ci_1h["spearman_ci_95_upper"]

                gamma_boot["sage_vs_lr_attr"] = {
                    "delta_mean": ci_lr["spearman_delta_mean"],
                    "ci_95_lower": ci_lr["spearman_ci_95_lower"],
                    "ci_95_upper": ci_lr["spearman_ci_95_upper"],
                    "interpretation": _interpret_ci(
                        ci_lr["spearman_ci_95_lower"],
                        ci_lr["spearman_ci_95_upper"],
                        float(args.equivalence_bound),
                    ),
                }
                gamma_boot["sage_vs_lr_1hop"] = {
                    "delta_mean": ci_1h["spearman_delta_mean"],
                    "ci_95_lower": ci_1h["spearman_ci_95_lower"],
                    "ci_95_upper": ci_1h["spearman_ci_95_upper"],
                    "interpretation": _interpret_ci(
                        ci_1h["spearman_ci_95_lower"],
                        ci_1h["spearman_ci_95_upper"],
                        float(args.equivalence_bound),
                    ),
                }
            else:
                row["sage_spearman"] = float("nan")
                row["sage_source"] = "skipped"
                row["delta_sage_vs_lr_attr"] = float("nan")
                row["ci_sage_vs_lr_attr_lower"] = float("nan")
                row["ci_sage_vs_lr_attr_upper"] = float("nan")
                row["delta_sage_vs_lr_1hop"] = float("nan")
                row["ci_sage_vs_lr_1hop_lower"] = float("nan")
                row["ci_sage_vs_lr_1hop_upper"] = float("nan")
                gamma_boot["note"] = "GraphSAGE skipped (--skip-sage)"

        summary_rows.append(row)
        bootstrap_payload["per_gamma"].append(gamma_boot)

    out_csv = resolve_project_path(args.out_summary_csv)
    ensure_parent(out_csv)
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
    print(f"[OK] Wrote summary CSV: {out_csv}")

    out_json = resolve_project_path(args.out_bootstrap_json)
    write_json(out_json, bootstrap_payload)
    print(f"[OK] Wrote bootstrap / run metadata JSON: {out_json}")


if __name__ == "__main__":
    main()
