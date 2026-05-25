"""Publication-style stacked bar charts: Spearman rho by IC operationalization.

Two rows, one column (IEEE single-column friendly): avoids cramped side-by-side
panels and keeps value labels for near-zero / negative correlations from colliding
with y-axis tick labels.

Representative values align with frozen clean metrics:
  - Structural (A0): baseline_ranking_metrics_a0_clean + surrogate_ranking_metrics_a0_clean
  - Source-community (HSCC): baseline_ranking_metrics_hscc_clean + surrogate_ranking_metrics_hscc_clean

GNN rows (GCN, GraphSAGE) use a light fill plus ``///`` hatch; baselines use solid grays.

Suggested LaTeX caption (put Delta rho in the caption, not on the plot; values match
paired-bootstrap means in ``outputs/mapr2026_v3_results/gnn_vs_degree_bootstrap_ci_a0.json``
and ``gnn_vs_baseline_bootstrap_ci_hscc.json``):

  Representative Spearman correlations across the two IC operationalizations.
  Bars show selected main-comparator rows from Tables~II--III, not complete
  leaderboards; hatched bars denote GNN rows. The headline paired-bootstrap
  comparisons are GCN versus degree ($\\Delta\\rho\\approx-0.018$) and GraphSAGE
  versus flat LR ($\\Delta\\rho\\approx+0.033$). Rank-loss GraphSAGE and the
  1-hop diagnostic are omitted because they are secondary or diagnostic analyses.

Outputs vector PDF (and optional high-DPI PNG) under figures/ for LaTeX.

Usage (from repo root):
  python scripts/make_regime_comparison_figure.py
  python scripts/make_regime_comparison_figure.py --figheight 2.05 --basename regime_comparison
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Representative main-comparator rows (Tables II--III in paper; see *_clean.csv).
STRUCTURAL_LABELS = ["Degree", "Node2Vec+LR", "GCN", "GraphSAGE"]
STRUCTURAL_VALS = [0.826343, 0.810306, 0.807657, 0.534131]

SOURCE_LABELS = ["Degree", "Node2Vec+LR", "Flat LR", "GraphSAGE"]
# Flat LR = lr_degree_views_life_time_lang (HSCC primary flat comparator).
SOURCE_VALS = [-0.006361, 0.570143, 0.884302, 0.915493]

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 6.5,
        "axes.titlesize": 6.8,
        "axes.labelsize": 6.5,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "hatch.linewidth": 0.35,
    }
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def bar_style_for(label: str) -> dict[str, object]:
    """Visual encoding: hatched = GNN; solid grays = baselines."""
    if label in {"GCN", "GraphSAGE"}:
        return {
            "color": "0.97",
            "edgecolor": "black",
            "hatch": "///",
            "linewidth": 0.65,
        }
    if label == "Degree":
        return {"color": "0.45", "edgecolor": "black", "linewidth": 0.5}
    if label == "Node2Vec+LR":
        return {"color": "0.70", "edgecolor": "black", "linewidth": 0.5}
    if label == "Flat LR":
        return {"color": "0.85", "edgecolor": "black", "linewidth": 0.5}
    return {"color": "0.78", "edgecolor": "black", "linewidth": 0.5}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        default="figures",
        help="Output directory (relative to repo root unless absolute).",
    )
    p.add_argument(
        "--basename",
        default="regime_comparison",
        help="Stem for regime_comparison.pdf / .png",
    )
    p.add_argument(
        "--figwidth",
        type=float,
        default=3.45,
        help="Figure width in inches (IEEE single-column friendly).",
    )
    p.add_argument(
        "--figheight",
        type=float,
        default=1.90,
        help="Figure height in inches (stacked 2-row default; use ~2.05 if labels feel cramped).",
    )
    p.add_argument(
        "--xticks",
        default="0,0.5,1",
        help="Comma-separated x-axis tick positions (default: sparse summary ticks).",
    )
    p.add_argument(
        "--xmin",
        type=float,
        default=-0.10,
        help="Left x-axis limit (slightly negative so tiny negative bars stay visible).",
    )
    p.add_argument(
        "--value-threshold",
        type=float,
        default=0.03,
        help="Place value labels at anchor x when rho < this (avoids overlap near 0 for small/negative rho).",
    )
    p.add_argument(
        "--value-anchor-x",
        type=float,
        default=0.035,
        help="x position for value labels when rho < threshold (right of zero).",
    )
    p.add_argument("--dpi", type=int, default=600, help="PNG resolution.")
    p.add_argument("--no-png", action="store_true", help="Skip writing PNG.")
    return p.parse_args()


def _place_value_label(
    ax: plt.Axes,
    yi: float,
    val: float,
    *,
    threshold: float,
    anchor_x: float,
    offset: float = 0.018,
) -> None:
    """Avoid overlapping bar-end labels with y tick text for rho near zero."""
    lab = f"{val:.3f}"
    if val < threshold:
        ax.text(anchor_x, yi, lab, va="center", ha="left", fontsize=5.8)
    else:
        ax.text(val + offset, yi, lab, va="center", ha="left", fontsize=5.8)


def main() -> None:
    args = parse_args()
    root = _repo_root()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    panels = [
        ("Structural WC", STRUCTURAL_LABELS, STRUCTURAL_VALS),
        ("Source-community", SOURCE_LABELS, SOURCE_VALS),
    ]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(float(args.figwidth), float(args.figheight)),
        sharex=True,
    )

    thr = float(args.value_threshold)
    anchor = float(args.value_anchor_x)
    x_min = float(args.xmin)
    xticks = [float(x.strip()) for x in str(args.xticks).split(",") if x.strip() != ""]

    for ax, (title, labels, vals) in zip(axes, panels):
        y = np.arange(len(labels))
        v = np.asarray(vals, dtype=float)
        ax.axvline(0.0, color="black", linewidth=0.5, zorder=0)
        for yi, label, val in zip(y, labels, v):
            ax.barh(yi, val, height=0.55, zorder=1, **bar_style_for(str(label)))
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_title(title, loc="left", pad=2)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", length=2)
        ax.set_xlim(x_min, 1.02)

        for yi, val in zip(y, v):
            _place_value_label(ax, yi, val, threshold=thr, anchor_x=anchor)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes:
        if xticks:
            ax.set_xticks(xticks)

    axes[-1].set_xlabel(r"Spearman $\rho$")
    fig.tight_layout(pad=0.25, h_pad=0.45)

    stem = str(args.basename).strip() or "regime_comparison"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight", format="pdf")
    if not bool(args.no_png):
        png_path = out_dir / f"{stem}.png"
        fig.savefig(png_path, bbox_inches="tight", format="png", dpi=int(args.dpi))
        print(f"[OK] wrote {png_path}")
    plt.close(fig)
    print(f"[OK] wrote {pdf_path}")


if __name__ == "__main__":
    main()
