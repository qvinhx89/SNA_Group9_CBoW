# Assumptions and Limitations

> **Purpose**: Document all assumptions underlying the analysis and known limitations.
> **Importance**: Transparency for reviewers and correct interpretation of results.

---

## 1. Data Assumptions

### 1.1 Graph Structure

| Assumption           | Description                                   | Implication                                   |
| -------------------- | --------------------------------------------- | --------------------------------------------- |
| **Undirected edges** | Friendship/mutual follow treated as symmetric | Influence flows both directions equally       |
| **Static snapshot**  | Single point-in-time network                  | Cannot capture temporal dynamics of influence |
| **No edge weights**  | All connections treated equally               | Strong/weak ties not distinguished            |
| **No self-loops**    | Nodes cannot connect to themselves            | Verified in preprocessing                     |

### 1.2 Node Attributes

| Assumption                    | Description                                 | Implication                                       |
| ----------------------------- | ------------------------------------------- | ------------------------------------------------- |
| **Views as visibility proxy** | Aggregate views represent public popularity | May not reflect recent activity                   |
| **Active accounts only**      | Dead accounts (dead_account=1) excluded     | Reduces noise but may miss historical influencers |

### 1.3 Data Scope

| Assumption          | Description                      | Implication                                 |
| ------------------- | -------------------------------- | ------------------------------------------- |
| **Twitch DE only**  | German-speaking Twitch community | Results may not generalize to other regions |
| **Single platform** | Twitch network only              | Cross-platform influence not captured       |

---

## 2. Methodological Assumptions

### 2.1 SIS Formula

| Assumption                               | Description                    | Justification                                    |
| ---------------------------------------- | ------------------------------ | ------------------------------------------------ |
| **Equal PageRank + Betweenness weights** | w1 = w2 = 0.4                  | Both represent distinct aspects of influence     |
| **Lower k-shell weight**                 | w3 = 0.2                       | Core position less directly related to influence |
| **Rank normalization valid**             | Percentile ranking appropriate | Handles heavy-tailed distributions               |
| **Metrics independent enough**           | Not perfectly correlated       | Verified via Spearman correlation                |

### 2.2 Typology (2x2 Matrix)

| Assumption                            | Description                    | Implication                        |
| ------------------------------------- | ------------------------------ | ---------------------------------- |
| **20% threshold meaningful**          | Top 20% represents "high"      | Sensitivity analysis tests 15%/25% |
| **Binary split valid**                | High/low dichotomy appropriate | Loses gradation information        |
| **SIS captures structural influence** | Composite score meaningful     | Validated via IC simulation        |

### 2.3 Independent Cascade Model

| Assumption                             | Description             | Implication                           |
| -------------------------------------- | ----------------------- | ------------------------------------- |
| **Homogeneous activation probability** | Same p for all edges    | Real influence varies by relationship |
| **Independent cascades**               | Activations independent | No synergy effects modeled            |
| **Single activation attempt**          | One chance per edge     | May underestimate stubborn influence  |
| **Synchronous propagation**            | Discrete time steps     | Real influence continuous             |

---

## 3. Statistical Assumptions

### 3.1 Hypothesis Testing

| Assumption                     | Description                    | Verification                 |
| ------------------------------ | ------------------------------ | ---------------------------- |
| **Non-parametric tests valid** | No normality assumption needed | Mann-Whitney U used          |
| **Independent samples**        | Typology groups independent    | Stratified sampling          |
| **FDR control appropriate**    | Benjamini-Hochberg valid       | Multiple comparisons context |

### 3.2 Power Analysis

| Assumption                      | Description         | Note                      |
| ------------------------------- | ------------------- | ------------------------- |
| **Medium effect size expected** | d ≈ 0.5             | Based on pilot/literature |
| **80% power target**            | Standard convention | May need adjustment       |

---

## 4. Known Limitations

### 4.1 Data Limitations

1. **Temporal dynamics**: Network is a single snapshot; influence may change over time
2. **Missing data**: Some streamers may have incomplete view counts
3. **Selection bias**: Only includes accounts meeting certain activity thresholds
4. **Regional scope**: Results specific to German Twitch community

### 4.2 Methodological Limitations

1. **SIS weights arbitrary**: 0.4/0.4/0.2 chosen based on literature, not optimized
2. **Threshold sensitivity**: 20% cutoff affects typology composition
3. **IC model simplicity**: Real information spreading more complex than IC
4. **No ground truth**: "True influence" not directly observable
5. **External validation limited (Task 6 IF PROBLEM)**: life_time validation was inconclusive after degree control (`partial_spearman_rho = -0.02`, `n_quintiles_significant = 0/3 quintiles tested` after BH-FDR; 2 quintiles skipped because `n_hidden < 10`), so language-based corroboration is reported as supplementary evidence and does not replace lifetime validation.

### 4.3 Scope Limitations

1. **No causal claims**: Correlation between structure and influence, not causation
2. **Platform-specific**: Results may not transfer to other social networks
3. **Content-agnostic**: Does not consider content type or quality

---

## 5. Threats to Validity

### 5.1 Internal Validity

| Threat                | Mitigation                             |
| --------------------- | -------------------------------------- |
| Confounding variables | Control for degree in analyses         |
| Selection bias        | Random sampling within typology        |
| Measurement error     | Multiple metrics, sensitivity analysis |

### 5.2 External Validity

| Threat               | Mitigation             |
| -------------------- | ---------------------- |
| Generalizability     | Document scope clearly |
| Temporal validity    | Note snapshot nature   |
| Platform specificity | Discuss in limitations |

### 5.3 Construct Validity

| Threat                | Mitigation                     |
| --------------------- | ------------------------------ |
| SIS validity          | IC simulation validation       |
| Influence measurement | Multiple validation approaches |
| Typology validity     | Null model comparison          |

---

## 6. Interpretation Guidelines

### What Results CAN Support:

- Structural position correlates with simulated spreading reach
- Hidden Influencers exist (high-SIS, low-views nodes)
- Simple surface metrics insufficient for detecting hidden influencers

### What Results CANNOT Support:

- Causal claims about influence
- Predictions of actual content virality
- Generalizations to other platforms/regions
- Claims about streamer quality or value

---

## 7. Recommendations for Report

When presenting results, always:

1. State scope (Twitch DE, static snapshot)
2. Acknowledge IC model limitations
3. Note sensitivity to threshold choices
4. Discuss null model comparison results
5. Avoid causal language ("affects" → "associated with")

---

_This document should be referenced in the final report's limitations section._
