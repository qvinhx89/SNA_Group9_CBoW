# Experiment Registry

> **Purpose**: Track all experimental configurations, parameter changes, and decisions.
> **Rule**: Every config change must be logged with timestamp and rationale.

---

## How to Use This Registry

1. **Before making changes**: Check existing entries for context
2. **When changing parameters**: Add new entry with timestamp
3. **Format**: Use the template below for consistency
4. **Config hash**: Run `python -c "import hashlib; print(hashlib.md5(open('src/config/base.yaml').read().encode()).hexdigest()[:8])"` to generate

---

## Registry Entries

### [2026-04-20 21:56] - HSCC-refined formula lock + regression target artifact backfill

- **Config hash**: n/a (formula + artifact registration)
- **Parameter changed**: `ic_formula_hscc_refined`, `hscc_regression_target_artifact`
- **Old value**: HSCC-refined run artifacts existed, but registry entry and `regression_targets_hscc_refined.parquet` were missing in working tree.
- **New value**:
   - Formula lock (implemented in `src/mapr2026_v3/ic_labels_hscc_refined.py`):
      `p(u,v)=clip(lambda * phi(u)/deg(u) * (1 + gamma * I[c_u!=c_v]), p_max)`
      with `phi(u)=rank(log1p(views_u)/(1+life_time_u))/N`.
   - Added artifact: `data/processed/regression_targets_hscc_refined.parquet` (`node_id`, `y=log1p(ic_score_mean)`, n=5000).
- **Rationale**: Close reproducibility gap before HSCC surrogate training and keep experiment traceability aligned with anti-HARKing protocol.
- **Impact**:
   - Person 3 can now train surrogates directly on HSCC targets via `--targets-path data/processed/regression_targets_hscc_refined.parquet`.
   - HSCC run now has explicit registry provenance.
- **Author**: Person 1

### [2026-04-18 12:46] - Enable CUDA torch + I-A labeling CUDA-only implementation

- **Config hash**: n/a (environment + implementation change)
- **Parameter changed**: `runtime_dependency`, `ia_labeling_backend`
- **Old value**: torch (cpu) or absent; I-A labeling CPU-only (`ic_labels_attribute_ia.py`)
- **New value**: CUDA-enabled torch (cu126); I-A labeling can run CUDA-only via `src/mapr2026_v3/ic_labels_attribute_ia_cuda.py`
- **Rationale**: User requirement “chỉ chạy bằng GPU CUDA” for full I-A labeling.
- **Impact**: Generates the same contract artifacts but requires `torch.cuda.is_available()==True`.
- **Install note (Windows)**: torch CUDA wheel installed from PyTorch CUDA index (e.g. cu126). Also requires import order `import torch` before `numpy/pandas` to avoid WinError 1114.
- **Author**: Person 1

### [2026-04-18 09:00] - MAPR2026 v3.1 Pre-registration: H-IA (Attribute-Informed IC branch)

- **Config hash**: n/a (hypothesis pre-registration)
- **Parameter changed**: `hypothesis_registry`
- **Old value**: not recorded
- **New value**: `H-IA: Under I-A labels, GATv2 sẽ outperform degree (degree blind to row-norm IC); FAIL → A0-only narrative.`
- **Rationale**: Prevent HARKing; lock the I-A branch interpretation before producing full I-A labels.
- **Impact**:
   - If I-A pilot PASS → run full I-A labeling and allow I-A surrogate experiments.
   - If I-A pilot FAIL → skip full I-A; commit A0-only narrative.
- **Evidence**: `outputs/mapr2026_v3_results/ia_pilot_diagnostics.json` (current run shows `pass=true`).
- **Author**: Person 1

### [2026-04-18 09:05] - MAPR2026 v3.1 Construct-validity config lock (no calibration target)

- **Config hash**: n/a (paper-facing lock)
- **Parameter changed**: `src/config/experiment.yaml::mapr2026_v3_1`
- **Old value**: missing/unspecified
- **New value**: `graph_directed=false`, `calibration_mode=variance_check`, `p_primary=weighted_cascade`, `ic_backend=csr_numpy`, `ic_parallel=joblib_loky`
- **Rationale**: Align repo configuration record with MAPR2026 v3.1 plan flow and master plan constraints.
- **Impact**: Stabilizes construct-validity statements across reruns and handoffs.
- **Author**: Person 1

### [2026-04-07 19:30] - Task 6 Comparator Lock + IF PROBLEM Fallback Activated

- **Config hash**: n/a (analysis/protocol lock for MAPR2026 v3 Track B)
- **Parameter changed**: `task6_lifetime_comparator`
- **Old value**: Hidden vs Overrated (pseudocode drift in implementation plan section 10.2)
- **New value**: Hidden vs Non-Hidden
- **Rationale**: Lock implementation to team parallel coding plan contract and avoid protocol drift across docs.
- **Impact**: Lifetime validation uses one comparator consistently across reruns and sign-off checks.
- **Author**: Person 2 / Group 9

### [2026-04-07 19:35] - Task 6 Power Guard + Language Fallback Artifact

- **Config hash**: n/a (code-level guard and fallback branch)
- **Parameter changed**: `lifetime_min_group_size`, `language_fallback_enabled`
- **Old value**: Stratified MWU accepted non-empty groups (`n > 0`)
- **New value**: Stratified MWU requires `n_hidden >= 10` and `n_non_hidden >= 10`; IF PROBLEM branch writes `language_validation.json`
- **Rationale**: Address Bottleneck #3 power issue and execute mandatory IF PROBLEM branch when lifetime gate fails.
- **Impact**: `lifetime_validation.json` remains primary artifact; fallback corroboration is now produced at `outputs/mapr2026_v3_results/language_validation.json`.
- **Author**: Person 2 / Group 9

### Entry Template

```
### [YYYY-MM-DD HH:MM] - Brief Description
- **Config hash**: xxxxxxxx
- **Parameter changed**: `parameter_name`
- **Old value**: X
- **New value**: Y
- **Rationale**: Why this change was made
- **Impact**: What this affects
- **Author**: Name
```

---

## Implementation Plan Revision (2026-03)

### [2026-03-27] - Implementation Plan Aligned with Proposal

- **Changes applied per proposal alignment review**:

1. **SIS Formula (FORBIDDEN-1)**:
   - **Formula**: `SIS(v) = [rank(PageRank) + rank(Betweenness) + rank(k-shell)] / 3`
   - **Type**: UNWEIGHTED rank-average (no weights w1, w2, w3)
   - **Rationale**: Fixed per proposal Section 6

2. **ML Baselines (FORBIDDEN-2)**:
   - **Models allowed**: Majority class, LR(views), LR(degree), LR(views+degree)
   - **Removed**: RandomForest, SHAP analysis, Node2Vec ablation
   - **Rationale**: Proposal Section 5 defines only LR baselines

3. **Removed Components (FORBIDDEN-3)**:
   - **Removed**: Null model (configuration model) comparison
   - **Removed**: Eigenvector centrality check
   - **Rationale**: Not in proposal scope

4. **Betweenness Parameters (CHANGE-5)**:
   - **k_pivots**: 1000
   - **seed**: 42
   - **Note**: k=1000 on N~163K gives ~3% error bound per Brandes (2001)

5. **IC Calibration (CHANGE-6)**:
   - **Subgraph**: 10% (not 20%)
   - **p values**: {0.01, 0.03, 0.05} (removed 0.08)
   - **Seeds**: k=10 (not 50)
   - **Runs**: 50 per seed
   - **Target range**: [5%, 30%] (not [8%, 25%])

6. **Statistical Tests (CHANGE-7)**:
   - **Correction**: Benjamini-Hochberg (fdr_bh)
   - **Effect size**: rank-biserial r
   - **Output columns**: effect_size_r, p_raw, p_corrected_bh

7. **Louvain Stability (CHANGE-4)**:
   - **Runs**: 10 with different seeds
   - **Metric**: mean_nmi_louvain in metrics.json
   - **Selection**: Best-Q partition

- **Author**: Group 9

---

## Week 1-2: Setup Phase

### [XXXX-XX-XX] - Initial Configuration

- **Config hash**: (generate after creating config)
- **Parameters set**:
  - `sis.formula`: unweighted rank-average (NO WEIGHTS per proposal)
  - `typology.threshold`: 0.20 (top 20%)
  - `random_seed`: 42
  - `betweenness.k_pivots`: 1000
  - `betweenness.seed`: 42
- **Rationale**: Values defined in proposal and implementation plan
- **Author**: Group 9

---

## Week 3-4: Centrality & Community Phase

### [XXXX-XX-XX] - Betweenness Computation

- **Config hash**:
- **k_pivots used**: 1000
- **seed used**: 42
- **Estimated error**: ~3% per Brandes (2001)
- **Author**:

### [XXXX-XX-XX] - Louvain Stability Check (CHANGE-4)

- **Config hash**:
- **Number of runs**: 10
- **Seeds used**: 0-9
- **NMI values between runs**: [X.XX, X.XX, ...]
- **Mean NMI (mean_nmi_louvain)**: X.XX
- **Stability threshold**: 0.85
- **Decision**: [STABLE/UNSTABLE]
- **Selected partition**: Run #X (seed=X, modularity=X.XX)
- **Author**:

---

## Week 5-6: SIS & Typology Phase

### [XXXX-XX-XX] - SIS Computation

- **Config hash**:
- **Formula used**: `SIS(v) = [rank(PageRank) + rank(Betweenness) + rank(k-shell)] / 3`
- **Weights**: None (unweighted per FORBIDDEN-1)
- **Verification**: Formula matches proposal Section 6
- **Author**:

### [XXXX-XX-XX] - Robustness Analysis Results

- **Config hash**:
- **Threshold sensitivity**:
  | Threshold | Hidden count | Jaccard with 20% |
  |-----------|--------------|------------------|
  | 15% | XXX | X.XX |
  | 20% | XXX | 1.00 |
  | 25% | XXX | X.XX |
- **Target**: Jaccard >= 0.70 between variants
- **Result**: [PASS/FAIL]
- **Author**:

---

## Week 7-8: IC Calibration & Validation Phase

### [XXXX-XX-XX] - IC Parameter Calibration (CHANGE-6)

- **Config hash**:
- **Pilot subgraph**: 10% of nodes (N = XXX)
- **p values tested**: {0.01, 0.03, 0.05}
- **Seeds per p**: k=10
- **Runs per seed**: 50
- **Results**:
  | p value | Mean reach | Mean reach / N |
  |---------|------------|----------------|
  | 0.01 | XXX | X.X% |
  | 0.03 | XXX | X.X% |
  | 0.05 | XXX | X.X% |
- **Target range**: [5%, 30%]
- **Selected p (ic_p_chosen)**: X.XX
- **Justification (ic_p_justification)**: Falls in target range [5%, 30%]
- **Author**:

### [XXXX-XX-XX] - IC Robustness Check (NUANCED-1)

- **Config hash**:
- **p_chosen**: X.XX
- **p_high (p_chosen × 2)**: X.XX
- **Runs for robustness check**: 50
- **Strategy ranking preserved**: [YES/NO]
- **Note**: Single robustness check, NOT full sensitivity sweep
- **Author**:

---

## Week 9-10: ML Phase

### [XXXX-XX-XX] - ML Pipeline Configuration (FORBIDDEN-2)

- **Config hash**:
- **Train/Val/Test split**: 70/10/20
- **Stratification**: By typology
- **Models trained**:
  - Majority class baseline
  - LR (views only)
  - LR (degree only)
  - LR (views + degree)
- **NOT included** (per proposal): RandomForest, SHAP
- **Random seed**: 42
- **Author**:

### [XXXX-XX-XX] - CV Variance Estimation (NUANCED-2)

- **Config hash**:
- **Folds**: 5
- **Data used**: train+val combined
- **Purpose**: Supplementary variance estimation
- **Note**: Primary result is 70/10/20 split
- **Results**:
  | Model | Mean F1 | Std F1 |
  |-------|---------|--------|
  | lr_views_only | X.XX | X.XX |
  | lr_views_degree | X.XX | X.XX |
- **Author**:

### [XXXX-XX-XX] - Data Leakage Check

- **Verification method**:
- **Result**: [PASS/FAIL]
- **Author**:

---

## Summary Statistics

| Phase     | Entries | Critical Decisions                    |
| --------- | ------- | ------------------------------------- |
| Week 1-2  | X       | Initial config, SIS formula           |
| Week 3-4  | X       | Betweenness params, Louvain stability |
| Week 5-6  | X       | SIS computation, Robustness           |
| Week 7-8  | X       | IC calibration                        |
| Week 9-10 | X       | ML config                             |

---

## Config File Hashes (Checkpoints)

| Date | Stage     | Hash | Notes                  |
| ---- | --------- | ---- | ---------------------- |
|      | Stage 0   |      | Initial                |
|      | Stage 1-2 |      | Centrality + Community |
|      | Stage 3   |      | Post-SIS               |
|      | Stage 4   |      | Post-IC calibration    |
|      | Stage 6   |      | Final                  |

---

_This registry is the source of truth for all experimental decisions._
