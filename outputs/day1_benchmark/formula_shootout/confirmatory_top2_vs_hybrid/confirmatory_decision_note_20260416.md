# Confirmatory Run: Top-2 vs Hybrid (2026-04-16)

## Setup
Direct comparison under larger budget:
- formulas: `hybrid_centered(gamma=0.1)`, `sender_boost(alpha=0.1)`, `sender_receiver(alpha=0.15,beta=0.1)`
- n_sample=1000
- n_runs=20
- mc_seeds={0,1,2}
- same data/features/splits and same evaluation protocol

Raw outputs:
- `outputs/day1_benchmark/formula_shootout/confirmatory_top2_vs_hybrid/hybrid/ic_formula_shootout_summary.csv`
- `outputs/day1_benchmark/formula_shootout/confirmatory_top2_vs_hybrid/sender_receiver/ic_formula_shootout_summary.csv`
- `outputs/day1_benchmark/formula_shootout/confirmatory_top2_vs_hybrid/sender_boost/ic_formula_shootout_summary.csv`

Unified ranking table:
- `outputs/day1_benchmark/formula_shootout/confirmatory_top2_vs_hybrid/confirmatory_ranking.csv`

## Key Results
1) **hybrid_centered(gamma=0.1)**
- runtime_sec: 168.95 (fastest)
- stability: jaccard_mean=0.2073, spearman_mean=0.6325
- learnability: graph_plus_views_spearman=0.7398, delta_spearman=+0.0225
- score_total: **0.7946** (rank #1)

2) sender_receiver
- runtime_sec: 265.44
- stability: jaccard_mean=0.2031, spearman_mean=0.6512
- learnability: graph_plus_views_spearman=0.7304, delta_spearman=+0.0087
- score_total: 0.5185 (rank #2)

3) sender_boost
- runtime_sec: 514.75 (slowest)
- stability: jaccard_mean=0.1790, spearman_mean=0.6376
- learnability: graph_plus_views_spearman=0.7364, delta_spearman=+0.0262
- score_total: 0.3466 (rank #3)

## Decision
Under confirmatory conditions, **hybrid_centered(gamma=0.1) remains the best overall formula**.

Why:
- best runtime by a wide margin,
- strongest combined stability + learnability balance,
- no evidence that top-2 alternatives beat hybrid on total trade-off.

## Practical Recommendation
- Keep production formula as:
  - `p_model=hybrid_degree_views_centered`
  - `hybrid_gamma=0.1`
  - `n_runs=150` for final labels/report artifacts
- Optionally keep sender_receiver as backup candidate for future ablation sections.
