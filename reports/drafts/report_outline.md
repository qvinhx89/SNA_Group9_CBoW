# Report Outline - SNA Twitch Influencer Project

> **Purpose**: Planning document for final report structure
> **Status**: Draft

---

## Report Structure

### 1. Title Page
- Project title
- Group number and members
- Date
- Course information

### 2. Abstract (200-300 words)
- Problem statement
- Methods overview
- Key findings summary
- Main conclusions

### 3. Introduction (1-1.5 pages)
- Background and motivation
- Problem definition
- Research questions (RQ1-RQ4)
- Contributions summary

### 4. Related Work (0.5-1 page)
- Prior work on influencer identification
- Network centrality measures
- Information diffusion models

### 5. Data Description (0.5 page)
- Dataset source and characteristics
- Preprocessing steps
- Final graph statistics

### 6. Methodology (2-3 pages)
- SIS formulation and rationale
- Typology construction
- IC simulation setup
- ML pipeline design
- Statistical testing approach

### 7. Results (3-4 pages)
- RQ1 findings + figure/table
- RQ2 findings + statistical tests
- RQ3 findings + benchmark table
- RQ4 findings + model comparison

### 8. Discussion (1-1.5 pages)
- Interpretation of results
- Null model analysis
- Practical implications
- Limitations

### 9. Conclusion (0.5 page)
- Summary of contributions
- Future work

### 10. References
- All citations in proper format

### 11. Appendix
- Detailed tables
- Additional figures
- Reproducibility instructions

---

## Figure List

| Figure ID | Description | Script Source |
|-----------|-------------|---------------|
| fig_rank_divergence | Rank comparison: views vs centralities | `src/evaluation/ranking_overlap.py` |
| fig_typology_distribution | 2x2 typology distribution | `src/sis/build_typology.py` |
| fig_hidden_vs_overrated_ic | IC reach comparison box plot | `src/simulation/run_single_seed_ic.py` |
| fig_ic_strategy_comparison | Multi-seed strategy benchmark | `src/simulation/run_multi_seed_ic.py` |
| fig_confusion_matrix | ML confusion matrix | `src/ml/evaluate_metrics.py` |
| fig_shap_beeswarm | SHAP feature importance | `src/ml/shap_analysis.py` |
| fig_sensitivity_heatmap | Robustness analysis | `src/sis/robustness.py` |

---

## Table List

| Table ID | Description | Source File |
|----------|-------------|-------------|
| table_rq1_metrics | Centrality-views correlations | `outputs/stage1/` |
| table_rq2_hidden_validation | IC validation with effect sizes | `outputs/stage4_single_seed/` |
| table_rq3_ic_benchmark | Strategy ranking | `outputs/stage5_multi_seed/` |
| table_rq4_detectability_report | ML model comparison | `outputs/stage6_ml/` |

---

## Writing Assignments

| Section | Primary Author | Reviewer | Deadline |
|---------|----------------|----------|----------|
| Abstract | | | Week 10 |
| Introduction | | | Week 10 |
| Methodology | | | Week 9 |
| Results | | | Week 9 |
| Discussion | | | Week 10 |

---

## Checklist Before Submission

- [ ] All figures generated and placed in `reports/figures/`
- [ ] All tables generated and placed in `reports/tables/`
- [ ] No claims without evidence reference
- [ ] All parameters documented in `experiment_registry.md`
- [ ] Reproducibility verified with `run_all.py` or `make run_all`
- [ ] Spell check completed
- [ ] Format consistent throughout
