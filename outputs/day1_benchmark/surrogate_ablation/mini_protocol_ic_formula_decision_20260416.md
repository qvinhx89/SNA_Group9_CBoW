# Mini Protocol Decision (2026-04-16)

## Scope
Compare IC formula candidates using two lenses:
1) Label stability (from `outputs/day1_benchmark/hybrid_sweep/*/stability_sweep_summary.csv`, nodes=200, seeds=0..4)
2) Surrogate learnability gain from adding views (`graph_plus_views` vs `graph_only`) using `benchmark_surrogate_modalities.py`

## Candidate Summary

### A) weighted_cascade (baseline)
- Stability @ n_runs=150:
  - jaccard_mean = 0.3041
  - spearman_mean = 0.7024
  - runtime_sec = 179.27
- Learnability (current targets, n=5000):
  - delta_spearman(graph_plus_views - graph_only) = +0.00184
  - delta_jaccard_top10 = -0.00211
- Interpretation:
  - Fastest and most stable rank metric among tested runs.
  - Adding views gives almost no regression gain.

### B) hybrid_degree_views_centered, gamma=0.1
- Stability @ n_runs=150:
  - jaccard_mean = 0.3916 (best top-decile stability)
  - spearman_mean = 0.6645
  - runtime_sec = 359.54
- Learnability:
  - n=1000, n_runs=30: delta_spearman = +0.01182, delta_jaccard_top10 = +0.00000
  - n=2000, n_runs=30: delta_spearman = +0.02108, delta_jaccard_top10 = +0.01522
- Interpretation:
  - Strong and consistent Spearman gain when views are added.
  - Top-k gain is modest but non-negative in larger sample run.

### C) hybrid_degree_views_centered, gamma=0.2
- Stability @ n_runs=150:
  - jaccard_mean = 0.3774
  - spearman_mean = 0.6983
  - runtime_sec = 496.15 (slowest)
- Learnability (n=2000, n_runs=30):
  - delta_spearman = +0.00510
  - delta_jaccard_top10 = -0.01311
- Interpretation:
  - Weaker learnability gain than gamma=0.1 and negative top-k delta.

## Decision
Recommended IC formula for current phase: **hybrid_degree_views_centered with gamma=0.1**.

Rationale:
- Satisfies instructor direction to combine raw signals + structure.
- Best top-decile stability among tested formulas.
- Strongest and most consistent surrogate regression gain from adding views.
- Runtime is higher than baseline but still substantially lower than gamma=0.2.

## Deployment Defaults (proposed)
- p_model: `hybrid_degree_views_centered`
- hybrid_gamma: `0.1`
- n_runs: `100` for routine generation, `150` for final/report artifacts
- report both graph_only and graph_plus_views surrogate results, with graph_plus_views as primary under hybrid labels

## Notes / Limitations
- Full 5000-node hybrid run at n_runs=200 did not complete reliably in-session due to long runtime/progress visibility.
- Mini protocol relies on completed n=1000 and n=2000 hybrid learnability runs plus existing stability sweeps.
