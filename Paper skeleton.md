# When Does Graph Learning Add Value Beyond Strong Baselines?
## A Comparative Study of IC Operationalizations for Influence Approximation on Social Networks

---

> **PAPER SKELETON — DRAFT v0.1 — 2026-04-26**
>
> **Provenance of numbers:**
> - `[✅ FROZEN]` — from clean, authoritative output files; can be cited in final paper
> - `[⚠ PRELIMINARY]` — from combined/older run files; verify after Person 3 A100 rerun
> - `[🔲 PLACEHOLDER]` — data not yet available; fill after fresh run
>
> **Data sources:**
> - A0 baselines: `baseline_ranking_metrics_a0_clean.csv` ✅
> - A0 GNN surrogates: `surrogate_ranking_metrics_a0_clean.csv` ✅
> - HSCC baselines: `baseline_ranking_metrics_hscc_clean.csv` ✅
> - HSCC GNN surrogates: `surrogate_ranking_metrics.csv` (hscc rows) ⚠
> - A0 bootstrap: `gnn_vs_degree_bootstrap_ci_a0.json` ⚠ (stale — no feature_policy field)
> - HSCC bootstrap: `gnn_vs_baseline_bootstrap_ci_hscc.json` ⚠ (stale — no feature_policy field)
> - Runtime: `runtime_breakdown.csv` ✅
>
> **Action required before submission:** After Person 3 A100 rerun completes, replace all ⚠ PRELIMINARY numbers with frozen values from `surrogate_ranking_metrics_hscc_clean.csv` and fresh bootstrap JSONs.

---

## Abstract

Identifying influential users in static social networks without behavioral cascade logs requires simulation-based operationalizations of influence. Whether graph representation learning adds value beyond the strongest available baseline — analytical or flat, depending on the regime — remains poorly understood. We study two Monte Carlo Independent Cascade (MC-IC) operationalizations on the Twitch social network (168K nodes, 6.8M edges): a structural weighted-cascade regime (A0) and a domain-informed source-community regime (HSCC). Within-degree-band IC variability motivates continuous regression over binary classification. Under A0, analytical structural baselines nearly saturate the approximation ceiling (degree ρ=0.826 [✅]); under HSCC, the dominant signal shifts to source-attribute models (best flat baseline LR ρ=0.884 [✅]), with structural centralities near zero. Evaluated across five architectures (GraphSAGE, GCN, GIN, GAT, APPNP), GNN surrogates exhibit regime-dependent value. Under A0, the best GNN (GCN, ρ=0.808 [✅]) falls below degree under bootstrap testing (Δ=−0.016 [⚠ PRELIMINARY]); under HSCC, preliminary results suggest the best GNN (SAGE, ρ=0.916 [⚠]) improves over the strongest flat baseline (Δ=+0.033 [⚠ PRELIMINARY]). In both regimes, surrogates reduce inference cost by several orders of magnitude relative to MC-IC. These findings suggest that operationalization choice — not architecture — is the primary driver of when graph learning adds value for influence approximation on dense social networks.

---

## 1. Introduction

### 1.1 Problem

Identifying influential users in online social networks is important for applications such as viral marketing, community management, and platform recommendation. Monte Carlo Independent Cascade (MC-IC) simulation provides a principled operationalization of influence potential grounded in the diffusion model of Kempe et al. [1]. However, MC-IC is computationally expensive, requiring hundreds of stochastic simulations per node and thereby rendering repeated evaluation impractical on large-scale graphs. In our setting on the Twitch social network, labeling 5,000 nodes via 200 MC-IC runs requires approximately 480 seconds under the structural A0 operationalization, and ~1,956 seconds under the more compute-intensive HSCC regime.

### 1.2 Motivation for Surrogates

Graph Neural Networks (GNNs) offer a natural surrogate approach: they can learn to approximate IC scores from graph structure and node attributes and then be deployed for fast inference. Prior work on GNN-based influence estimation has largely focused on a single diffusion model, leaving open the question of when learned graph representations genuinely add value beyond the strongest available baseline — which may be an analytical centrality (e.g., degree) under degree-coupled operationalizations, or a strong flat attribute model under engagement-driven ones. In dense social networks where cascades attenuate quickly, degree itself may already capture most of the diffusion signal, leaving limited room for additional gains from graph learning.

This paper studies not whether GNNs are universally superior, but under which diffusion operationalizations graph message passing adds value beyond analytical and flat baselines.

### 1.3 Core Idea and Contributions

In this paper, we investigate how the choice of IC operationalization determines whether GNN surrogate learning provides value beyond the strongest available baseline — a regime-dependent comparator that is analytical (degree centrality) under A0 and flat (LR with full node-attribute access) under HSCC. We compare two defensible operationalizations on the Twitch MUSAE social network (168K nodes, 6.8M edges) [2]: (1) weighted cascade (A0), in which transmission probability depends only on target degree, and (2) HSCC, a domain-informed variant that incorporates source engagement velocity and cross-community amplification. Our contributions are threefold:

1. **We analyze Monte Carlo IC as a simulation-defined operational metric for influence potential on Twitch, showing that continuous regression targets are more appropriate than binary top-k labels in this dense-graph setting.** Within each degree band, intra-band coefficient of variation is at least 1.12 (ranging to 2.29 across degree quintiles), indicating substantial IC variability that cannot be reduced to degree alone and that makes binary threshold assignment unreliable.
2. **We compare two diffusion operationalizations, A0 and HSCC, and show that they induce qualitatively different approximation regimes.** A0 yields a degree-dominated regime in which analytical baselines are near-optimal. HSCC yields a graph-aware attribute-community regime in which source-side engagement drives labels and preliminary evidence suggests neighborhood structure may add additional value.
3. **We benchmark analytical, flat, and GNN surrogates under both regimes, showing that GNN value is regime-dependent rather than universal while inference remains orders of magnitude faster than repeated MC simulation.** Under A0, the best GNN using raw node attributes (GCN, ρ=0.808 [✅]) falls below degree (ρ=0.826 [✅]) under bootstrap testing. Under HSCC, preliminary results suggest the best GNN (SAGE, ρ=0.916 [⚠]) improves over the strongest matched flat baseline (ρ=0.884 [✅]) by Δ=+0.033 [⚠ PRELIMINARY — pending A100 rerun].

---

## 2. Background and Related Work

### 2.1 Independent Cascade and Influence Estimation

Kempe, Kleinberg, and Tardos [1] formalized the Independent Cascade (IC) model and the influence maximization (IM) problem. In IC, each directed edge (u,v) is independently activated with probability p(u,v), and activated nodes in turn attempt to activate their neighbors in subsequent rounds. For influence estimation without cascade logs, a common operational choice is the *weighted cascade* model, where p(u,v) = 1/deg(v), proposed by Kempe et al. and used subsequently in DeepIM [3] and related learning-based IM methods. Our task differs from seed-set optimization: we perform node-level IC score regression to learn a surrogate that rapidly approximates simulation-derived influence scores.

### 2.2 Node Importance and Structural Baselines

Prior work has established several strong structural baselines for ranking influential nodes. Degree centrality is often the most predictive structural measure under degree-coupled IC [3]. PageRank [4] captures global influence through random-walk stationary distribution. K-shell decomposition [5] identifies nodes in dense network cores. Two-hop spread counts second-order neighborhood size, approximating local cascade reach. Guille et al. [6] discuss evaluation challenges when behavioral data is unavailable; we follow their approach of using simulation-derived scores as evaluation targets.

### 2.3 GNN Architectures for Graph-Level Tasks

We evaluate five GNN architectures under a unified training protocol: GraphSAGE [7] (neighborhood sampling and aggregation), GCN [8] (spectral graph convolution), GIN [9] (graph isomorphism network with maximally expressive aggregation), GAT [10] (attention-weighted aggregation over neighbors), and APPNP [11] (personalized propagation of neural predictions). All architectures are applied as node-level regressors trained to predict continuous IC scores, not as graph classifiers. Community structure is identified using the Louvain algorithm [12] for HSCC parameter computation.

---

## 3. MC-IC as an Operational Metric

### 3.1 Construct Validity

The Twitch dataset [2] provides a follower graph where edges represent subscription relationships, not observed diffusion pathways. We treat the follower graph as a structural substrate for simulation: it specifies the topology through which influence could propagate, rather than recording actual propagation events. All findings in this paper are therefore properties of *simulation-defined influence approximation*, not measurements of real influence. We make no claim that MC-IC scores correspond to actual empirical influence in real-world propagation. Our contribution is a comparative study of how operationalization choice affects the learnability of the resulting influence surrogate.

### 3.2 Operationalizations

**A0 — Weighted Cascade (structural baseline regime)**

$$p(u, v) = \frac{1}{\deg(v)}$$

A0 is the standard structural operationalization proposed by Kempe et al. [1]. Transmission probability depends only on the in-degree of the target node, making it degree-coupled by design. A0 serves as our structural reference regime: it generates labels that should be highly predictable by degree-based analytical baselines, allowing us to characterize the learnability ceiling imposed by a purely structural IC specification.

**HSCC — Source-Community Regime (domain-informed)**

$$p(u, v) = \text{clip}\!\left(\lambda \cdot \frac{\phi(u)}{\deg(u)} \cdot \left(1 + \gamma \cdot \mathbf{1}[c_u \neq c_v]\right),\ 0,\ p_{\max}\right)$$

where the source engagement velocity term is:

$$\phi(u) = \frac{\text{rank}\!\left(\frac{\log(1 + \text{views}_u)}{1 + \text{life\_time}_u}\right)}{N}$$

with fixed parameters λ=1.0, γ=1.0, p_max=1.0. Three design choices merit brief justification:

- **Rank normalization on φ(u):** The views distribution is heavy-tailed; rank normalization bounds the source term to [0, 1] and prevents a small number of extreme accounts from distorting the transmission scale.
- **log1p(views)/(1+life\_time):** This inner term approximates engagement velocity rather than cumulative popularity — log1p compresses outliers, and dividing by (1+life\_time) avoids rewarding longevity alone.
- **Community amplification (1+γ·𝟏[c\_u≠c\_v]):** Encodes the structural-hole interpretation of cross-community bridging [13]; nodes at community boundaries have heightened transmission probability toward cross-community neighbors.

All parameters (λ, γ, p\_max) are fixed prior to any model training and not tuned to maximize downstream surrogate performance.

HSCC is introduced as a domain-informed comparative operationalization rather than as a validated generative law of Twitch diffusion. The fixed community-amplification configuration is kept as a transparent, frozen comparative setting instead of being tuned to maximize downstream surrogate gains. Estimating edge-level transmission probabilities from behavioral data would require supervised cascade logs unavailable in this dataset; weighted cascade and HSCC therefore provide principled zero-shot alternatives.

> **[AUTHOR NOTE — delete before submission]** Anticipating reviewer pushback on HSCC validity: we claim only that HSCC is a *comparative operationalization*, not a validated generative model of Twitch behavior. Its role is to introduce source-side and community-side signal into label generation in a transparent, frozen configuration, allowing a controlled test of whether graph message passing recovers structure that source-attribute flat models cannot. The §3.1 construct validity statement handles this.

### 3.3 Dataset and Simulation Protocol

We use the Twitch MUSAE EN dataset [2]: **168,114 nodes**, **6,797,557 edges** (directed follower graph). We select 5,000 nodes as labeled seeds (uniform random sample), running 200 MC-IC simulations per seed for each operationalization independently. Simulation is implemented with sparse CSR propagation; A0 labeling completes in ~480 seconds [✅] for 5,000 nodes; HSCC labeling requires ~1,956 seconds [✅] due to the additional community-lookup and source-velocity computation per edge traversal.

HSCC IC score summary [✅]:

| Statistic | Value |
|-----------|-------|
| Mean reach | 4.83 nodes |
| Max reach | 16.31 nodes |
| Std | 2.82 |
| CV (coefficient of variation) | 0.583 |
| n_runs | 200 |

A0 IC scores exhibit heavy-tailed, degree-coupled behavior: nodes in the top degree band (degree 93–7,613) achieve mean IC reach of 109.3 nodes (std=176.6), while low-degree nodes (degree 1–10) achieve mean IC reach of only 2.5 nodes. HSCC reach is substantially lower (mean=4.83, max=16.31), reflecting selective local-community diffusion rather than broad viral spread.

> [🔲 PLACEHOLDER — Add A0 full IC distribution summary (mean/median/CV) from a0 IC diagnostics once available. Temporary: use degree-controlled variance data above as evidence of A0 range.]

### 3.4 Discriminativeness and Regime Contrast

| Regime | Mean Reach | Max | CV | Degree Signal |
|--------|-----------|-----|----|---------------|
| A0 | [🔲 full dist] | [🔲] | [🔲] | Very strong (p∝1/deg(v)) |
| HSCC | 4.83 [✅] | 16.31 [✅] | 0.583 [✅] | Absent (degree ρ=−0.006 [✅]) |

Both operationalizations generate non-degenerate continuous targets (CV > 0 in both regimes). The key contrast is that A0 is structurally dominated by degree, while HSCC completely decouples from degree: the Spearman correlation between degree centrality and HSCC IC scores is −0.006 [✅].

### 3.5 Regression as the Principled Formulation

We treat regression not as a fallback, but as the natural formulation for a simulation-derived continuous influence target. IC scores are inherently continuous: they represent expected cascade reach under a stochastic process and vary smoothly across nodes. Imposing a binary top-k cutoff discards this gradient without gain in predictability.

The continuous formulation receives additional empirical support from two observations. First, IC scores are heavy-tailed: under A0, nodes in the top degree band achieve mean reach of 109.3 nodes versus 2.5 nodes for low-degree nodes, creating a smooth ranking gradient rather than a clean two-class partition. Second, within each degree band the IC score distribution remains highly variable: intra-band CV ranges from 1.12 to 2.29 across degree quintiles [✅] (degree-controlled IC variance analysis), meaning many nodes near any top-k threshold share nearly identical expected IC scores. Binary label assignment at such thresholds would be sensitive to the specific simulation runs chosen.

> **[AUTHOR NOTE]** Formal label stability (Jaccard overlap across repeated MC campaigns) would further quantify boundary sensitivity. Artifact `stability_explanation.json` is pending — add once available with quantified label flip rate. The within-band CV argument is defensible on its own but Jaccard evidence would strengthen Claim 1.

All downstream evaluation uses Spearman rank correlation (ρ), NDCG@10%, and Precision@10%, which directly evaluate ranking quality over the continuous target.

---

## 4. Surrogate Learning Across Operationalizations

### 4.1 Experimental Setup

**Dataset split:** 5,000 labeled nodes; transductive setting; 1,000 test nodes held out; remaining 4,000 used for training. Split is regime-independent and consistent across all models.

**Feature access (locked per regime):**
- A0: `raw_attr` = [degree, life_time, views] (3 features; `include_language=False`)
- HSCC: `raw_attr` = [degree, life_time, views, language_dummies] (24 features; `include_language=True`)
- All flat fairness baselines receive the same feature access as the GNN comparator for that regime.

**GNN training protocol:** Adam optimizer, 5 independent random seeds, hidden_channels=128, dropout=0.3, 200 epochs. APPNP uses K=10 propagation steps with α=0.15 teleport probability and gradient clipping (max_norm=1.0).

**Evaluation metrics:** Spearman ρ (primary), NDCG@10%, Precision@10%.

**Bootstrap CI protocol:** 1,000-sample bootstrap resampling over test predictions, paired comparison, 95% percentile CI. Pre-registered equivalence bound δ₀=0.02. Results labeled as: gnn_significantly_better, practically_equivalent, or gnn_significantly_worse.

### 4.2 Results: A0 — Structural Ceiling

Under A0, the degree-coupled label generation creates a structural ceiling that analytical baselines nearly saturate. **This is not a failure of GNNs but rather a property of the operationalization**: degree already encodes nearly all available diffusion signal.

#### Table 2: A0 Surrogate Results — Spearman ρ and NDCG@10% on 1,000 held-out test nodes; primary comparator: degree centrality. GNN rows averaged over 5 seeds; analytical and flat baselines are deterministic. [✅ FROZEN]

| Model | Type | ρ (mean) | ρ (std) | NDCG@10% | P@10% |
|-------|------|---------|--------|---------|-------|
| **degree** | Analytical | **0.826** | — | 0.881 | 0.60 |
| pagerank | Analytical | 0.824 | — | 0.857 | 0.56 |
| kshell | Analytical | 0.816 | — | 0.687 | 0.50 |
| two_hop | Analytical | 0.804 | — | 0.848 | 0.55 |
| one_hop | Analytical | 0.688 | — | 0.833 | 0.52 |
| betweenness | Analytical | 0.716 | — | 0.735 | 0.42 |
| LR(deg+views+life_time) | Flat | 0.522 | — | 0.803 | 0.52 |
| MLP(raw_attr) | Flat | 0.435 | 0.004 | 0.601 | 0.42 |
| **GCN (raw_attr)** | **GNN** | **0.808** | **0.001** | **0.825** | **0.53** |
| GIN (raw_attr) | GNN | 0.622 | 0.010 | 0.730 | 0.40 |
| SAGE (raw_attr) | GNN | 0.534 | 0.009 | 0.674 | 0.45 |
| APPNP (raw_attr) | GNN | 0.141 | 0.697 | 0.516 | 0.34 |
| GAT (raw_attr) | GNN | [⚠ OOM] | — | — | — |
| *SAGE (centrality feats)* | *Diag.* | *0.828* | *0.0002* | *0.881* | *0.62* |

*Note: SAGE(centrality feats) uses degree/pagerank/kshell as input features — included as a diagnostic ceiling only, not a fair comparator.*
*GAT OOM encountered (VRAM threshold at hidden_channels=128 on test GPU); excluded from A0 main results.*
*APPNP shows catastrophically high variance across seeds (ρ std=0.697) — training completed but predictions are unreliable, likely due to over-smoothing under K=10 propagation steps on this dense 6.8M-edge graph with only 3 input features; excluded from claims.*

**Key finding (A0) [⚠ PRELIMINARY — bootstrap pending frozen A100 rerun]:** The best GNN using raw attributes (GCN, ρ=0.808 [✅]) falls below degree centrality under preliminary bootstrap testing:

> **Bootstrap A0:** GCN(raw_attr) vs degree — Δ=−0.016 [95% CI: −0.025, −0.007] — *gnn_significantly_worse* [⚠ PRELIMINARY — pending A100 rerun with fresh model]
> NDCG: Δ=−0.034 [95% CI: −0.057, −0.013] — *gnn_significantly_worse* [⚠ PRELIMINARY]

**Interpretation:** Under A0, analytical structural baselines are already near-optimal. GCN with raw features (degree, life_time, views) achieves ρ=0.808, approaching but not reaching the degree ceiling of ρ=0.826. The binding constraint is the diffusion operationalization, not the model family.

The gap between GCN (ρ=0.808) and MLP (ρ=0.435) confirms that graph message passing does learn structure — SAGE and GCN consistently exceed MLP by 0.37–0.38 ρ points — but the operationalization-imposed ceiling prevents this graph signal from exceeding degree centrality. When centrality features are fed directly as inputs (SAGE-centrality diagnostic), ρ=0.828 ≈ degree=0.826, confirming the ceiling is reachable but only by explicitly encoding degree-type information in the feature space.

> **Paste-ready paper sentence [⚠ — pending frozen A0 bootstrap rerun; use this template]:** Under A0, the best-performing GNN (GCN, Spearman ρ=0.808 [✅]) falls below degree centrality (ρ=0.826 [✅]) under the bootstrap comparison (Δ=−0.016, 95% CI=[−0.025, −0.007] [⚠ PRELIMINARY]), indicating that the limiting factor is the degree-coupled operationalization rather than the neural architecture. [Drop the [⚠] tags after rerunning bootstrap on fresh A100 outputs; keep this sentence only if the rerun confirms the same inferential direction.]

### 4.3 Results: HSCC — Graph-Aware Regime

Under HSCC, the source-engagement and community-bridging signal in label generation creates a qualitatively different approximation landscape. Degree collapses as a predictor (ρ=−0.006 [✅]); it is retained in Table 3 only as contextual evidence of this regime shift, not as the relevant comparator. Under HSCC, degree is no longer the meaningful reference — the appropriate comparator is the strongest flat non-graph baseline under matched feature access, which is LR(deg+views+life\_time+lang) at ρ=0.884 [✅]. The key question is whether graph message passing can recover community-mediated signal that this strong flat baseline already misses.

#### Table 3: HSCC Surrogate Results — Spearman ρ and NDCG@10% on 1,000 held-out test nodes; primary comparator: LR(deg+views+life_time+lang), the strongest flat baseline under matched feature access. GNN rows averaged over 5 seeds. [✅ baselines frozen; ⚠ GNN rows preliminary]

| Model | Type | ρ (mean) | ρ (std) | NDCG@10% | P@10% |
|-------|------|---------|--------|---------|-------|
| degree | Analytical | −0.006 [✅] | — | 0.465 | 0.04 |
| kshell | Analytical | −0.027 [✅] | — | 0.425 | 0.05 |
| betweenness | Analytical | 0.040 [✅] | — | 0.498 | 0.09 |
| one_hop | Analytical | 0.081 [✅] | — | 0.498 | 0.05 |
| LR(life_time) | Flat | 0.790 [✅] | — | 0.827 | 0.46 |
| LR(views+life_time) | Flat | 0.868 [✅] | — | 0.800 | 0.41 |
| LR(deg+views+life_time) | Flat | 0.868 [✅] | — | 0.801 | 0.41 |
| MLP(raw_attr) | Flat | 0.837 [✅] | 0.026 | 0.751 | 0.29 |
| **LR(deg+views+life_time+lang)** | **Flat** | **0.884 [✅]** | — | **0.829** | **0.45** |
| LR(views+life_time+lang) | Flat | 0.884 [✅] | — | 0.830 | 0.45 |
| **SAGE (raw_attr)** | **GNN** | **0.916 [⚠]** | **0.004** | **0.902** | **0.58** |
| SAGE (all feats) | GNN | 0.918 [⚠] | 0.003 | 0.904 | 0.60 |
| GAT (raw_attr) | GNN | 0.608 [⚠] | 0.034 | 0.708 | 0.25 |
| GCN (raw_attr) | GNN | 0.602 [⚠] | 0.014 | 0.694 | 0.21 |
| GIN (raw_attr) | GNN | 0.051 [⚠] | 0.062 | 0.482 | 0.05 |
| APPNP (raw_attr) | GNN | −0.031 [⚠] | 0.098 | 0.456 | 0.03 |
| SAGE (rankloss) | GNN | 0.671 [⚠] | 0.017 | 0.754 | 0.35 |

*Comparator: LR(deg+views+life_time+lang), the strongest flat baseline with matched language feature access.*
*[⚠ PRELIMINARY] HSCC GNN rows from pre-rerun combined file; to be replaced with `surrogate_ranking_metrics_hscc_clean.csv` after A100 rerun.*

**Key finding (HSCC) [⚠ PRELIMINARY — pending frozen rerun]:** The best GNN (SAGE with raw_attr features including language) achieves ρ=0.916 [⚠], observationally exceeding the strongest flat baseline; preliminary bootstrap analysis gives:

> **Bootstrap HSCC:** SAGE(raw_attr) vs LR(deg+views+life_time+lang) — Δ=+0.033 [95% CI: +0.021, +0.044] — *gnn_significantly_better* [⚠ PRELIMINARY]
> NDCG: Δ=+0.074 [95% CI: +0.050, +0.099] — *gnn_significantly_better* [⚠ PRELIMINARY]

**Interpretation [⚠ PRELIMINARY — language below conditional on frozen rerun confirming gnn_significantly_better]:** Preliminary results suggest GraphSAGE neighborhood aggregation recovers structure beyond what flat source-attribute models can capture from node-level attributes alone. The comparator LR(+lang) aggregates multiple node-level source attributes (degree, views, life_time, language dummies) without graph access, achieving ρ=0.884; the additional +0.033 ρ points gained by SAGE are consistent with residual neighborhood-structured signal beyond those node-level features. Whether this reflects SAGE's mean-aggregation partially reconstructing the community-bridging context encoded in the HSCC transmission formula cannot be determined from the current experiments alone.

Notably, GCN (ρ=0.602) and GIN (ρ=0.051) perform dramatically worse than SAGE (ρ=0.916) under HSCC. This architecture sensitivity suggests that SAGE's mean-aggregation may be better suited to capturing the neighborhood engagement patterns relevant to the HSCC mechanism, whereas GCN's symmetric degree normalization may partially re-couple representations to degree — a feature with near-zero signal under HSCC. This observation is discussed briefly in the Appendix; a full architecture study is beyond the scope of this paper.

Ranking-aware training (SAGE rankloss, ρ=0.671 [⚠]) performs substantially below MSE-trained SAGE (ρ=0.916 [⚠]) under HSCC, suggesting that the listwise ranking objective may not align well with the engagement-velocity structure of the HSCC labels; this does not alter the regime-level conclusion and is not pursued further here.

> **Paste-ready paper sentence [⚠ — use this template; fill outcome label after frozen rerun]:** Under HSCC, preliminary results suggest that GNN message passing improves upon the strongest flat baseline (SAGE ρ=0.916 vs LR ρ=0.884, observed Δ=+0.033, 95% CI=[+0.021, +0.044] [⚠ PRELIMINARY]), consistent with neighborhood structure contributing information that cannot be recovered from node-level attributes alone. [Replace "preliminary results suggest" with "we find" and drop [⚠] tags once bootstrap is re-confirmed on frozen HSCC GNN outputs.]

### 4.4 Regime Contrast

The contrast between A0 and HSCC reveals that surrogate learnability is not a property of the model alone; rather, it is jointly determined by the diffusion operationalization and the information already recoverable by simple baselines.

| Property | A0 | HSCC |
|----------|-----|------|
| Label signal source | Target degree (structural) | Source velocity + community bridging |
| Best analytical baseline | degree ρ=0.826 [✅] | degree ρ=−0.006 [✅] — collapses |
| Best trained (flat) baseline | LR(deg+views+life_time) ρ=0.522 [✅] | LR(+lang) ρ=0.884 [✅] |
| Best GNN (raw attrs) | GCN ρ=0.808 [✅] | SAGE ρ=0.916 [⚠] |
| Bootstrap comparator | degree (analytical) | LR(deg+views+life_time+lang) |
| Bootstrap result | prelim. Δ=−0.016 (bootstrap label: gnn worse) [⚠] | prelim. Δ=+0.033 (bootstrap label: gnn better) [⚠] |
| Best GNN architecture | GCN | GraphSAGE |
| APPNP seed variance (std) | 0.697 — excluded | 0.098 — excluded |
| Primary insight | Degree-coupled IC = degree ceiling | Engagement+community IC = graph gains |

Under A0, the degree-coupling creates a ceiling that neural architectures cannot overcome with only raw node attributes. Under HSCC, the source-side velocity signal allows flat models to reach ρ=0.884, but the remaining gap is filled by the SAGE graph aggregation (ρ=0.916 [⚠]), consistent with residual neighborhood-structured signal that node-level attributes alone cannot recover.

### 4.5 Runtime

All surrogates provide inference dramatically faster than MC-IC simulation.

#### Table 4: Runtime Summary [✅ FROZEN]

| Model | Inference (full graph) | Training | Speedup vs MC-IC |
|-------|----------------------|----------|-----------------|
| MC-IC (labeling) | 480.3 s [✅] | — | 1× |
| degree (analytical) | 0.004 s [✅] | — | ~120,000× |
| LR (flat) | <0.003 s [✅] | <0.004 s | ~160,000× |
| SAGE (raw_attr) | 0.086 s [✅] | ~27 s | ~5,600× |
| GCN (raw_attr) | 0.165 s [✅] | ~59 s | ~2,900× |
| GIN (raw_attr) | 0.067 s [✅] | ~24 s | ~7,200× |
| GAT (raw_attr) | 0.139 s [✅] | ~61 s | ~3,500× |
| APPNP (raw_attr) | 0.790 s [✅] | ~286 s | ~610× |

*GNN inference times: full 168K-node graph in a single forward pass. Training: averaged over 5 seeds. MC-IC baseline: 480s to label 5,000 nodes under A0 (the cheaper operationalization). Speedup ratios compare GNN full-graph inference to this MC-IC campaign cost; the asymmetry (5K vs 168K nodes) means speedups are conservative — extrapolating MC-IC to all 168K nodes (~16,140s) would increase GNN speedup by ~34×. Analytical baselines are faster still, with per-inference costs under 5ms.*

*Runtime justification for GNNs is regime-dependent: under A0, degree inference is ~120K× faster than MC-IC and already near-optimal (ρ=0.826), making GNNs unnecessary. Under HSCC, structural analytical baselines collapse (degree ρ=−0.006), while strong flat baselines (LR, ρ=0.884) retain both high accuracy and fast inference (<3ms). GNNs provide an additional accuracy gain over these flat baselines (preliminary: SAGE Δ=+0.033) with comparable inference speed, at the cost of training overhead (~27s for SAGE).*

---

## 5. Discussion and Limitations

### 5.1 When Does GNN Surrogate Learning Help?

Our results suggest that the usefulness of graph learning for IC approximation is governed by the information structure of the diffusion operationalization rather than by architecture choice alone. The A0 finding is strongly supported and likely generalizes: whenever the IC transmission formula is degree-coupled, structural analytical baselines provide a competitive approximation ceiling that message passing cannot easily surpass with raw node features. The HSCC finding rests on preliminary evidence and should be read with corresponding caution: when transmission probability encodes source-side behavioral attributes and community-bridging structure, flat models appear to capture the node-level component while neighborhood aggregation may add residual community-mediated signal — pending confirmation from frozen HSCC outputs.

Under A0, degree-coupling ensures that degree centrality is sufficient for most diffusion signal, leaving little room for graph aggregation to contribute. Under HSCC, source engagement velocity is largely capturable by flat models (LR ρ=0.884 [✅]), but preliminary results suggest a residual neighborhood-structured signal — consistent with cross-community amplification — that SAGE may exploit. The operative condition is therefore: GNN surrogates appear to add value when the target operationalization encodes graph-mediated signal that node-level attributes alone cannot recover, though this conclusion is pending confirmation on frozen HSCC outputs.

### 5.2 Limitations

1. **Follower graph ≠ observed diffusion:** The Twitch dataset provides a subscription graph, not behavioral cascade logs. All findings are properties of simulation-defined influence approximation.
2. **Operationalization validity:** Neither A0 nor HSCC is an empirically validated model of actual Twitch diffusion; both are principled zero-shot operationalizations among many plausible alternatives. For HSCC specifically, the source-velocity and community-amplification terms are motivated by structural-hole theory [13] and platform engagement patterns, but have not been calibrated against observed cascade data.
3. **Transductive evaluation:** Our evaluation is transductive; test nodes are drawn from the same graph as training nodes. Inductive generalization to new graphs or unseen nodes is not tested.
4. **Small mean reach under HSCC:** Mean IC reach of 4.83 nodes should be interpreted as selective local-community diffusion, not a limitation of discriminativeness (CV=0.583 confirms meaningful variation).
5. **Feature access parity required:** Our HSCC claims require matched feature access between GNN and flat baselines. Any change to the GNN feature set (e.g., removing `language`) would invalidate the current comparator selection.
6. **Single dataset:** All findings are from the Twitch MUSAE EN graph. Evaluating the regime-dependent contrast on a dataset with observed behavioral cascades (e.g., the Higgs Twitter dataset [14], which records real retweet/reply/mention cascades) would test whether the finding generalizes under empirical diffusion data rather than simulation-defined labels.

### 5.3 Scope of the Operationalization Choice

Our experiments fix two operationalizations (A0 and HSCC) with frozen hyperparameters (λ=1.0, γ=1.0). The HSCC parameters were not optimized to maximize downstream surrogate gains — they were fixed prior to any GNN training to preserve experimental integrity. This means the observed HSCC GNN gap (+0.033 [⚠ PRELIMINARY]) does not arise from operationalization overfitting — this conclusion holds as a matter of experimental design regardless of the final bootstrap outcome. Learning edge-level transmission probabilities directly from behavioral data would require supervised cascade logs unavailable in this dataset; A0 and HSCC therefore represent principled zero-shot operationalization choices appropriate for the no-cascade-log setting.

---

## 6. Conclusion

We have shown that the value of GNN surrogates for Monte Carlo influence estimation is regime-dependent, not universal. Under a degree-coupled operationalization (A0), analytical structural baselines represent a near-optimal ceiling that GNNs cannot surpass when using only raw node attributes. Under a source-community operationalization (HSCC), preliminary results suggest that graph message passing yields measurable improvements over the strongest flat baselines, consistent with neighborhood aggregation recovering community-mediated signal that node-level attributes alone cannot capture; these results are pending confirmation after A100 rerun. Our findings suggest that diffusion operationalization — rather than model architecture — is the primary driver of surrogate learnability on dense social networks.

---

## References

[1] D. Kempe, J. Kleinberg, and É. Tardos, "Maximizing the spread of influence through a social network," in *Proc. KDD*, 2003.

[2] B. Rozemberczki, C. Allen, and R. Sarkar, "Multi-scale attributed node embedding," *J. Complex Networks*, vol. 9, no. 2, 2021.

[3] [🔲 ACTION REQUIRED — Resolve before submission. In §2.1, [3] is cited for two things: (a) the weighted cascade model (p(u,v)=1/deg(v)) — this is actually Kempe et al. [1]; if [3] is redundant for that claim, remove the citation and cite [1] only. (b) Learning-based influence estimation that uses weighted cascade — candidate papers: S. Manchanda, A. Mittal, A. Dhawan, S. Medya, S. Ranu, A. Singh, "GCOMB: Learning Budget-constrained Combinatorial Optimization over Graphs," *Proc. NeurIPS*, 2020; or Z. Ling, Z. Chen, H. Tong, "PIANO: Influence Maximization Meets Deep Reinforcement Learning," *IEEE TKDE*, 2023. Confirm which paper the text intends to cite, find the correct venue/year, and replace this block.]

[4] L. Page, S. Brin, R. Motwani, and T. Winograd, "The PageRank citation ranking: Bringing order to the web," Stanford Technical Report, 1999.

[5] M. Kitsak, L. K. Gallos, S. Havlin, et al., "Identification of influential spreaders in complex networks," *Nature Physics*, vol. 6, pp. 888–893, 2010.

[6] A. Guille, H. Hacid, C. Favre, and D. A. Zighed, "Information diffusion in online social networks: A survey," *SIGMOD Record*, vol. 42, no. 2, 2013.

[7] W. L. Hamilton, Z. Ying, and J. Leskovec, "Inductive representation learning on large graphs," in *Proc. NeurIPS*, 2017.

[8] T. N. Kipf and M. Welling, "Semi-supervised classification with graph convolutional networks," in *Proc. ICLR*, 2017.

[9] K. Xu, W. Hu, J. Leskovec, and S. Jegelka, "How powerful are graph neural networks?" in *Proc. ICLR*, 2019.

[10] P. Veličković, G. Cucurull, A. Casanova, et al., "Graph attention networks," in *Proc. ICLR*, 2018.

[11] J. Klicpera, A. Bojchevski, and S. Günnemann, "Predict then propagate: Graph neural networks meet personalized PageRank," in *Proc. ICLR*, 2019.

[12] V. D. Blondel, J.-L. Guillaume, R. Lambiotte, and E. Lefebvre, "Fast unfolding of communities in large networks," *J. Statistical Mechanics*, 2008.

[13] R. S. Burt, *Structural Holes: The Social Structure of Competition*. Harvard University Press, 1992.

[14] M. De Domenico, A. Lima, P. Mougel, and M. Musolesi, "The anatomy of a scientific rumor," *Scientific Reports*, vol. 3, no. 2980, 2013. [Higgs Twitter dataset: real retweet/reply/mention cascades following the Higgs boson discovery announcement.]

---

## Appendix: Architecture Sensitivity Analysis

> [ℹ NOTE: Include as appendix if page budget allows; cut first if tight]

The dramatic architecture sensitivity under HSCC (SAGE ρ=0.916 [⚠] vs GCN ρ=0.602 [⚠] vs GIN ρ=0.051 [⚠]) warrants a brief diagnostic note. We hypothesize:

- **SAGE leads under HSCC [⚠ PRELIMINARY]:** Mean-aggregation over neighbors captures the *average* community engagement velocity of a node's neighborhood, which is consistent with the HSCC community-bridging term.
- **GCN trails under HSCC [⚠ PRELIMINARY]:** Symmetric normalization 1/√deg(u)·1/√deg(v) introduces a degree-weighting effect that partially re-couples representations to degree — counterproductive under HSCC where degree has zero predictive value.
- **GIN remains weak under HSCC [⚠ PRELIMINARY]:** GIN's sum-aggregation without degree normalization may create degenerate representations on the heavily skewed Twitch degree distribution (max degree=7,613).
- **APPNP appears unstable in both regimes [⚠ PRELIMINARY]:** K=10 multi-hop propagation on the dense 6.8M-edge graph with only 3-24 input features leads to over-smoothing instability (A0 std=0.697, HSCC std=0.098).

These hypotheses suggest that aggregation strategy and degree normalization choices interact strongly with IC operationalization in ways not captured by standard GNN benchmarking. This observation motivates future work on architecture selection criteria under different diffusion regimes.

---

## Figure Descriptions (for actual figure generation)

> [ℹ For paper typesetting — team should generate these figures from frozen output files]

### Figure 1: Pipeline Diagram
Two-branch pipeline showing:
- Left: Twitch graph + node attributes → A0 IC simulation → A0 labels (degree-coupled) → Analytical/Flat/GNN evaluation → "Structural ceiling: degree dominates" [✅ confirmed — annotate with final frozen label]
- Right: Twitch graph + node attributes + community context → HSCC IC simulation → HSCC labels (engagement+community) → Flat/GNN evaluation → "[regime outcome label — fill after A100 rerun: e.g. 'Graph-aware: SAGE improves over LR' if bootstrap confirms]"
- Both branches: speedup arrow from MC-IC (480s) → surrogate inference (<1s)

### Figure 2: Two-Panel Results Figure
- Left panel (A0): Dot plot with 95% error bars, x-axis = Spearman ρ, y-axis = models (degree baseline highlighted with dashed line, GCN highlighted as best GNN). Show regime where GNNs cluster below degree. [✅ layout confirmed from frozen A0 data]
- Right panel (HSCC): Dot plot with 95% error bars, x-axis = Spearman ρ, y-axis = models (LR(+lang) baseline highlighted with dashed line, SAGE highlighted as best GNN). [⚠ PRELIMINARY — generate after A100 rerun; annotate bootstrap outcome label (gnn_significantly_better / practically_equivalent / gnn_significantly_worse) from frozen bootstrap JSON, not from pre-rerun numbers]
- Grayscale-compatible; both panels share same x-axis scale [−0.1, 1.0].

---

## Section Status Tracker

| Section | Status | Action |
|---------|--------|--------|
| Abstract | Draft with ⚠ PRELIMINARY numbers | Update after A100 rerun |
| §1 Introduction | Draft — numbers from clean files | Verify [X] and [Y] fill-ins |
| §2 Background | Draft — literature only | Resolve [3]: determine if citation needed beyond [1] for weighted cascade; see [3] action note in References |
| §3 MC-IC Metric | Draft — A0 table placeholder; §3.5 retitled | Add A0 IC distribution stats; add stability_explanation.json once ready |
| §4.2 A0 Results | Complete [✅] | Bootstrap pending fresh rerun |
| §4.3 HSCC Results | ⚠ GNN rows preliminary | Rerun: replace all ⚠ with fresh numbers |
| §4.4 Contrast | Complete | Update Table 4 values after rerun |
| §4.5 Runtime | Complete [✅] | None |
| §5 Discussion | Complete | None |
| §6 Conclusion | Complete | Update claims after rerun |
| References | [14] De Domenico et al. added; [3] is a pending action item | Resolve [3] (see inline action note); confirm [1]–[2], [4]–[13] are correct; [14] ✓ |
| Appendix | Optional — check page budget | Cut if >6 pages |
