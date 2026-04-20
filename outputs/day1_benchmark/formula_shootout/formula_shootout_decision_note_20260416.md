# IC Formula Shootout Note (2026-04-16)

## Goal
Test alternative IC probability formulas to identify the strongest candidate under:
- label stability (inter-seed Spearman/Jaccard)
- surrogate learnability (graph_only vs graph_plus_views Spearman)
- runtime

## Benchmark setup (exploratory)
- n_sample=300 nodes
- n_runs=5 MC runs per seed
- mc_seeds={0,1}
- features used:
  - sender strength: robust-scaled mean of {log1p(views), pagerank, kshell}
  - receiver resistance: robust-scaled mean of {degree, cross_community_edge_fraction}

## Completed formulas
- weighted_cascade
- sender_boost
- sender_receiver
- community_boost

`convex_mixture` did not complete within practical runtime budgets in this session (repeated timeouts), therefore treated as computationally infeasible at this stage.

## Unified ranking (raw metrics + normalized composite score)
1) sender_boost
- runtime_sec: 47.58
- stability: jaccard_mean=0.1765, spearman_mean=0.5627
- learnability: graph_plus_views_spearman=0.6082, delta_spearman=-0.0326
- composite score (exploratory): 0.7433

2) sender_receiver
- runtime_sec: 16.89
- stability: jaccard_mean=0.1538, spearman_mean=0.5468
- learnability: graph_plus_views_spearman=0.6362, delta_spearman=-0.0202
- composite score (exploratory): 0.6523

3) weighted_cascade
- runtime_sec: 13.61
- stability: jaccard_mean=0.1731, spearman_mean=0.5588
- learnability: graph_plus_views_spearman=0.5204, delta_spearman=-0.0618
- composite score (exploratory): 0.5619

4) community_boost
- runtime_sec: 181.46 (slowest)
- stability: jaccard_mean=0.0714, spearman_mean=0.5389
- learnability: graph_plus_views_spearman=0.7097, delta_spearman=+0.0181
- composite score (exploratory): 0.4000

## Practical takeaway
- In this exploratory shootout, sender_boost is the best overall trade-off.
- sender_receiver is a strong runner-up with much better runtime.
- community_boost shows learnability upside but currently unstable and too slow.
- convex_mixture is currently not operationally feasible.

## Recommendation before replacing current production formula
Run a confirmatory pass on top-2 formulas (sender_boost, sender_receiver) with larger budget:
- n_sample >= 1000
- n_runs >= 20
- mc_seeds >= 3
and compare again versus current `hybrid_degree_views_centered(gamma=0.1)`.
