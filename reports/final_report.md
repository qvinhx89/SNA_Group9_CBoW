# Final Report - SNA Twitch Influencer Analysis

> **Project**: Identifying Hidden Influencers in Twitch Streamer Network
> **Group**: 9
> **Date**: [YYYY-MM-DD]

---

## Executive Summary

[Brief summary of key findings for all 4 research questions]

---

## 1. Introduction

### 1.1 Background
[Context about social network analysis and influencer marketing]

### 1.2 Research Questions
1. **RQ1**: How do structural centrality metrics diverge from visibility metrics (views) in the Twitch network?
2. **RQ2**: Do "Hidden Influencers" (high structural, low visibility) demonstrate real influence in IC simulations?
3. **RQ3**: Which seeding strategy performs best for information diffusion?
4. **RQ4**: Can Hidden Influencers be detected using only surface metrics?

### 1.3 Dataset
- Dataset: Twitch Gamers Social Network
- Nodes: 168,114 streamers
- Edges: 6,797,557 mutual follows
- Source: `benedekrozemberczki/datasets` (Twitch Gamers)

---

## 2. Methodology

### 2.1 Structural Influence Score (SIS)

Formula (unweighted rank-average):
```
SIS(v) = [rank(PageRank) + rank(Betweenness) + rank(k-shell)] / 3
```

### 2.2 Typology Construction
- Top 20% SIS × Top 20% Views → 4 groups
- True Influencer, Hidden Influencer, Overrated, Non-Influencer

### 2.3 Independent Cascade (IC) Model
- Calibrated activation probability p = [X.XX]
- Single-seed validation: 50 runs per node
- Multi-seed benchmark: 300 runs per strategy

### 2.4 ML Detectability
- Models: Majority class, LR (views), LR (degree), LR (views+degree), RandomForest
- Split: 70/10/20 stratified

---

## 3. Results

### 3.1 RQ1: Structural vs Visibility Divergence

**Key Findings:**
- [Finding 1]
- [Finding 2]

**Evidence:**
- See: `outputs/stage1/rq1_ranking_metrics.csv`
- Figure: `reports/figures/fig_rank_divergence.png`

### 3.2 RQ2: Hidden Influencer Validation

**Key Findings:**
- [Finding 1]
- [Finding 2]

**Statistical Tests:**
| Comparison | Effect Size (r) | p_raw | p_corrected |
|------------|-----------------|-------|-------------|
| Hidden vs Overrated | X.XX | X.XXX | X.XXX |
| Hidden vs True | X.XX | X.XXX | X.XXX |

**Evidence:**
- See: `outputs/stage4_single_seed/rq2_hidden_validation.csv`
- Figure: `reports/figures/fig_hidden_vs_overrated_ic.png`

### 3.3 RQ3: Seeding Strategy Benchmark

**Strategy Rankings:**
| Rank | Strategy | Mean Reach | 95% CI |
|------|----------|------------|--------|
| 1 | [Strategy] | XXX | [CI] |
| 2 | [Strategy] | XXX | [CI] |
| ... | ... | ... | ... |

**Evidence:**
- See: `outputs/stage5_multi_seed/rq3_strategy_benchmark.csv`
- Figure: `reports/figures/fig_ic_strategy_comparison.png`

### 3.4 RQ4: ML Detectability

**Model Performance:**
| Model | Test F1 (macro) | CV F1 ± std |
|-------|-----------------|-------------|
| Majority | X.XXX | - |
| LR (views) | X.XXX | X.XX ± X.XX |
| LR (degree) | X.XXX | X.XX ± X.XX |
| LR (views+degree) | X.XXX | X.XX ± X.XX |
| RandomForest | X.XXX | X.XX ± X.XX |

**SHAP Interpretation:**
- [Key insight about feature importance]

**Evidence:**
- See: `outputs/stage6_ml/rq4_detectability.csv`
- Figures: `reports/figures/fig_confusion_matrix.png`, `fig_shap_beeswarm.png`

---

## 4. Discussion

### 4.1 Interpretation
[Discuss findings in context of research questions]

### 4.2 Null Model Analysis
- Real graph: X.X% Hidden Influencers
- Null model: X.X% Hidden Influencers
- Interpretation: [Is typology artifact of degree distribution?]

### 4.3 Limitations
- See: `docs/assumptions_limitations.md`

---

## 5. Conclusion

[Summary of key contributions and implications]

---

## References

1. Kitsak, M., et al. (2010). Identification of influential spreaders in complex networks. Nature Physics.
2. [Other references]

---

## Appendix

### A. Artifact Index
| Artifact | Location | Description |
|----------|----------|-------------|
| Centrality table | `data/processed/centrality_table.parquet` | All centrality metrics |
| ... | ... | ... |

### B. Reproducibility
- All seeds and parameters logged in `docs/experiment_registry.md`
- Run pipeline: `python run_all.py` or `make run_all`
