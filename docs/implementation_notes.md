# Implementation Notes - SNA Twitch Influencer Project

> **Purpose**: Document key methodological decisions, formulas, and literature grounding.
> **Last updated**: Week 1 (Revised per proposal alignment)

---

## 1. Structural Influence Score (SIS) Formula

### 1.1 Mathematical Definition

<!-- FORBIDDEN-1: SIS formula is UNWEIGHTED rank-average per proposal Section 6 -->

The Structural Influence Score (SIS) uses an **UNWEIGHTED rank-average**:

$$
\text{SIS}(v) = \frac{\text{rank}(\text{PageRank}) + \text{rank}(\text{Betweenness}) + \text{rank}(k\text{-shell})}{3}
$$

**IMPORTANT**: This formula is **fixed per proposal Section 6**. Do NOT add weights.

where:
- $\text{rank}(x_i)$ = rank of node $i$ when sorted by metric $x$ (higher centrality = higher rank)
- The result is an average of three ranks

### 1.2 Rationale

| Metric | Purpose |
|--------|---------|
| PageRank | Captures recursive importance from network structure (Brin & Page, 1998) |
| Betweenness | Identifies information brokers and bridge nodes (Freeman, 1977) |
| k-shell | Captures core position in network (Kitsak et al., 2010) |

**Why unweighted average**:
- Treats all three structural dimensions equally
- Simple and interpretable
- Defined in proposal - not subject to modification

### 1.3 Literature Grounding

**Primary citations**:

1. **Kitsak, M., et al. (2010)**. "Identification of influential spreaders in complex networks." *Nature Physics*, 6(11), 888-893.
   - Established k-shell decomposition as predictor of spreading influence
   - Showed k-shell outperforms degree in predicting cascade size

2. **Freeman, L. C. (1977)**. "A set of measures of centrality based on betweenness." *Sociometry*, 40(1), 35-41.
   - Original betweenness centrality definition

3. **Brin, S., & Page, L. (1998)**. "The anatomy of a large-scale hypertextual web search engine." *Computer Networks*, 30(1-7), 107-117.
   - PageRank algorithm foundation

---

## 2. Graph Data Format <!-- CHANGE-2 -->

### 2.1 Storage Format

**Graph storage**: Use portable formats, NOT pickle files.

- **Edge list**: `data/processed/graph_active.edgelist`
  - Saved via `nx.write_edgelist(G, 'graph_active.edgelist')`
  - Portable across Python and NetworkX versions

- **Node attributes**: `data/processed/node_attributes.parquet`
  - Contains: `node_id`, `views`, and placeholder columns for centrality metrics
  - Parquet format for efficient storage and cross-platform compatibility

### 2.2 Load Procedure

```python
import networkx as nx
import pandas as pd

# Step 1: Load graph structure
G = nx.read_edgelist('data/processed/graph_active.edgelist')

# Step 2: Attach node attributes
attrs_df = pd.read_parquet('data/processed/node_attributes.parquet')
for _, row in attrs_df.iterrows():
    node_id = row['node_id']
    if G.has_node(node_id):
        for col in attrs_df.columns:
            if col != 'node_id':
                G.nodes[node_id][col] = row[col]
```

### 2.3 Python Version <!-- CHANGE-3 -->

**Record the Python version used** for reproducibility:

```
Python version: 3.11.9
```

---

## 3. Typology Construction (2x2 Matrix)

### 3.1 Definition

Split nodes into 4 groups based on **top-20% thresholds**:

| | High SIS (top 20%) | Low SIS (bottom 80%) |
|---|---|---|
| **High Views (top 20%)** | True Influencer | Overrated |
| **Low Views (bottom 80%)** | Hidden Influencer | Non-Influencer |

### 3.2 Threshold Sensitivity

Test robustness with:
- 15% threshold
- 20% threshold (default)
- 25% threshold

**Report Jaccard stability**:
$$
J(\text{Hidden}_{20\%}, \text{Hidden}_{15\%}) \geq 0.7 \text{ (target)}
$$

---

## 4. Centrality Computation <!-- CHANGE-5 -->

### 4.1 Betweenness Centrality Approximation

For large graphs, use approximate betweenness with specified parameters:

```python
import networkx as nx

# Parameters per implementation plan (CHANGE-5)
k_pivots = 1000  # Number of pivot nodes
seed = 42        # Random seed for reproducibility

betweenness = nx.betweenness_centrality(G, k=k_pivots, seed=seed)
```

**Note**: k=1000 on N≈163K active nodes gives ~3% error bound per Brandes (2001).

Record in `params.json`:
- `betweenness_k`: 1000
- `betweenness_seed`: 42

---

## 5. Independent Cascade (IC) Model

### 5.1 Model Definition

- **Activation probability** $p$: probability that active node $u$ activates neighbor $v$
- **Propagation**: each newly activated node has one chance to activate each inactive neighbor
- **Termination**: when no new activations occur

### 5.2 Calibration Protocol <!-- CHANGE-6 -->

**Goal**: Choose $p$ such that mean cascade reach is neither trivially small nor saturating.

**Target range**: $\text{mean\_reach} / N \in [5\%, 30\%]$

**Procedure** (see `src/simulation/ic_calibration.py`):
1. Sample **10%** of nodes as pilot subgraph
2. Test $p \in \{0.01, 0.03, 0.05\}$
3. Use **k=10 seeds**, **50 runs per seed**
4. Select $p$ where mean_reach/N falls in target range [5%, 30%]
5. Document selection in `experiment_registry.md` with justification

**Output fields** in `params.json`:
- `ic_p_chosen`: selected p value
- `ic_p_justification`: reason for selection

---

## 6. Statistical Methods

### 6.1 Multiple Testing Correction <!-- CHANGE-7 -->

Use **Benjamini-Hochberg** procedure (controls FDR at 0.05):

```python
from statsmodels.stats.multitest import multipletests

reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
```

### 6.2 Effect Size Measures <!-- CHANGE-7 -->

**Rank-biserial correlation** (for Mann-Whitney U):
$$
r_{rb} = 1 - \frac{2U}{n_1 \cdot n_2}
$$

**Output columns** for `table_rq2_validation.csv`:
- `effect_size_r`: rank-biserial correlation
- `p_raw`: raw p-value
- `p_corrected_bh`: Benjamini-Hochberg corrected p-value

### 6.3 Primary Comparisons for RQ2

1. **Primary**: Hidden vs Overrated (most important)
2. **Secondary**: Hidden vs True, Overrated vs Non

All comparisons receive BH correction.

---

## 7. ML Pipeline Notes <!-- FORBIDDEN-2 -->

### 7.1 Models (per proposal Section 5 - Baselines)

| Model | Purpose |
|-------|---------|
| Majority class | Baseline |
| LR (views only) | Surface metric baseline |
| LR (degree only) | Basic structural baseline |
| LR (views + degree) | Combined surface baseline |

**Note**: NO RandomForest, NO SHAP analysis. The proposal defines only LR baselines. The point of RQ4 is to show surface metrics are insufficient — LR is intentionally simple.

### 7.2 Data Split

- Train: 70%
- Validation: 10%
- Test: 20%
- Stratified by typology

### 7.3 Variance Estimation <!-- NUANCED-2 -->

After training on the 70/10/20 split, additionally run **5-fold stratified CV** on train+val combined to estimate metric variance.

Report: **mean ± std F1** alongside primary test-set results.

The 70/10/20 split remains the **primary reported result**. CV is supplementary only.

---

## 8. Version History

| Date | Change | Author |
|------|--------|--------|
| Week 1 | Initial document | Group 9 |
| Week 1 (Rev) | Aligned with proposal: unweighted SIS, removed null model, removed RF/SHAP | Group 9 |
| Week 1 (Rev) | Added CHANGE-2 through CHANGE-7 per implementation plan | Group 9 |

---

*This document should be updated as methodological decisions are made during implementation.*
