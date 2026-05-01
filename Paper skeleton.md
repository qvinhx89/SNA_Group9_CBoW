# When Does Graph Learning Add Value Beyond Strong Baselines?
## A Comparative Study of IC Operationalizations for Influence Approximation on Social Networks

---

> **PAPER SKELETON — POST-FREEZE DRAFT — 2026-04-28**
>
> **Authority for drafting:** Factual numbers (ρ, Δρ, CIs, comparator names) follow [Paper guide.md](Paper%20guide.md) and the frozen outputs below. Framing, tone, and claim discipline follow [Paper rules.md](Paper%20rules.md). This skeleton is a drafting scaffold, not a second source of truth for numbers.
>
> **Provenance of numbers:**
> - `[✅ FROZEN]` — from clean output files listed below; cite in final paper only if those files are the submission freeze
> - `[🔲 PLACEHOLDER]` — reserved for items explicitly left open in the writing plan (e.g. full A0 IC distribution table); not for reruns
>
> **Data sources (frozen Person 3 run):**
> - A0 baselines: `baseline_ranking_metrics_a0_clean.csv` ✅
> - A0 GNN surrogates: `surrogate_ranking_metrics_a0_clean.csv` ✅
> - HSCC baselines: `baseline_ranking_metrics_hscc_clean.csv` ✅
> - HSCC GNN surrogates: `surrogate_ranking_metrics_hscc_clean.csv` ✅
> - A0 bootstrap: `gnn_vs_degree_bootstrap_ci_a0.json` ✅
> - HSCC bootstrap: `gnn_vs_baseline_bootstrap_ci_hscc.json` ✅
> - HSCC rankloss (C3) bootstrap: `gnn_vs_rankloss_bootstrap_ci_hscc.json` ✅ (`loss_mode=rankloss_combined`, CI vs flat comparator)
> - Runtime: `runtime_breakdown.csv` ✅
>
> **Before submission:** Proofread prose against [Paper guide.md](Paper%20guide.md) §3.3 (abstract) and §4 (claims). C3 rankloss is frozen; include or cut it for **page budget only**, not because the artifact is preliminary. If C3 is omitted from the main paper, remove matching abstract/conclusion/table rows at the same time.

---

> **[CLEAN DRAFT PRODUCTION — HOW TO STRIP THIS SCAFFOLD FOR SUBMISSION]**
>
> **Step 1 — Remove all AUTHOR NOTEs:** Delete every block starting with `> **[AUTHOR NOTE` through its closing `>` line.
>
> **Step 2 — Strip inline markers:** Remove all ` [✅]` and ` [🔲]` tags from prose and table cells. Keep `†` and `‡` daggers — these belong in the submission.
>
> **Step 3 — Use clean paragraphs from AUTHOR NOTEs (not scaffold text):**
> - §5.1: Use the paste-ready paragraph from the §5.1 AUTHOR NOTE (not the scaffold prose above it)
> - §6 Conclusion: Use the 5-sentence template from the §6 AUTHOR NOTE
> - §4.5 prose: Follow the 6-step order from the §4.5 AUTHOR NOTE
>
> **Step 4 — Resolve remaining [🔲] placeholders:**
> - HSCC labeling time (§3.3): verify from HSCC simulation logs before citing
> - Any remaining `[🔲]` in prose: do not submit with unresolved placeholders
>
> **Word budget targets (IEEE 6-page ~3,000 words body + ~250 words refs):**
> §1 Introduction: ~300 words | §2 Background: ~250 words | §3 Dataset+Method: ~400 words
> §4 Results: ~700 words | §5 Discussion: ~300 words | §6 Conclusion: ~150 words
>
> **Table-first discipline:** Fill Table 2 (A0) and Table 3 (HSCC) with frozen numbers first — prose should describe what the reader sees in the table, not repeat all numbers in full sentences.

---

## Abstract

Identifying influential users in static social networks without behavioral cascade logs requires simulation-based operationalizations of influence, yet it remains unclear when learned graph representations add value beyond strong non-graph baselines. We study two Monte Carlo Independent Cascade (MC-IC) operationalizations on the Twitch Gamers social network (168K nodes, 6.8M edges): a structural weighted-cascade regime (A0) and a domain-informed source-community regime (HSCC). We find that binary top-k influence labels are structurally unstable under the degree-coupled A0 operationalization, motivating continuous regression on simulation-derived influence scores. Under A0, the best GNN using raw node attributes (GCN, ρ=0.808 [✅]) remains statistically below degree centrality (ρ=0.826 [✅]; Δρ=−0.018, 95% CI [−0.029, −0.008] [✅]). Under HSCC, degree centrality collapses (ρ=−0.006 [✅]) and GraphSAGE (`gnn_raw_attr`, ρ=0.915 [✅]) significantly outperforms the official matched flat comparator locked in the frozen bootstrap artifact — LR(degree, views, life_time, language), ρ=0.884 [✅] — with Δρ=+0.033, 95% CI [+0.021, +0.044] [✅]; a ranking-aware variant further achieves Δρ=+0.041 vs the same comparator (95% CI [+0.030, +0.053] [✅]). Once trained, the GNN surrogate provides full-graph inference in approximately 0.086 seconds compared with 480 seconds for a single MC-IC labeling pass, a speedup of approximately 5,500× [✅].

> **[AUTHOR NOTE — detailed reference version for writers; trim sentences S3–S6 to guide §3.3 template before submission. Key points NOT in 6-sentence abstract but must appear in body text: (a) APPNP excluded from best-arch (std>0.1 in both regimes); (b) near-tie HSCC comparator (0.00012 difference) — justified by pre-specification; (c) +0.009 vs standard SAGE is descriptive only, no paired bootstrap vs standard training.]**

---

## 1. Introduction

### 1.1 Problem

Identifying influential users in online social networks is important for applications such as viral marketing, community management, and platform recommendation. Monte Carlo Independent Cascade (MC-IC) simulation provides a principled operationalization of influence potential grounded in the diffusion model of Kempe et al. [1]. However, MC-IC is computationally expensive, requiring hundreds of stochastic simulations per node and thereby rendering repeated evaluation impractical on large-scale graphs. In our setting on the Twitch Gamers graph, labeling 5,000 nodes via 200 MC-IC runs requires approximately 480 seconds under the structural A0 operationalization [✅]; HSCC labeling is more compute-intensive and should be reported only after verifying the corresponding simulation logs (see §3.3).

### 1.2 Motivation for Surrogates

Graph Neural Networks (GNNs) offer a natural surrogate approach: they can learn to approximate IC scores from graph structure and node attributes and then be deployed for fast inference. Prior work on GNN-based influence estimation has largely focused on a single diffusion model, leaving open the question of when learned graph representations genuinely add value beyond the strongest non-graph baselines — whether structural metrics such as degree centrality, or flat attribute models such as logistic regression. The answer depends critically on the IC operationalization: regimes that couple transmission probability to degree already make degree a near-optimal surrogate, whereas regimes driven by source-side attributes shift competitiveness toward flat attribute models and potentially toward graph message passing.

This paper studies not whether GNNs are universally superior, but under which diffusion operationalizations graph message passing adds value beyond the strongest valid non-graph comparator in each regime.

### 1.3 Core Idea and Contributions

In this paper, we investigate how the choice of IC operationalization determines whether GNN surrogate learning provides value beyond the strongest non-graph baselines — regime-dependent comparators that are analytical (degree centrality) under A0 and flat (LR with full node-attribute access) under HSCC. We compare two defensible operationalizations on the Twitch Gamers social network (168K nodes, 6.8M edges) [2]: (1) weighted cascade (A0), in which transmission probability depends only on target degree, and (2) HSCC, a domain-informed variant that incorporates source engagement velocity and cross-community amplification. Our contributions are threefold:

1. **We analyze Monte Carlo IC as a simulation-defined operational metric for influence potential on Twitch, showing that continuous regression targets are more appropriate than binary top-k labels in this dense-graph setting.** Within each degree band, intra-band coefficient of variation is at least 1.12 (ranging to 2.29 across degree quintiles), indicating substantial IC variability that cannot be reduced to degree alone and that makes binary threshold assignment unreliable.
2. **We compare two diffusion operationalizations, A0 and HSCC, and show that they induce qualitatively different approximation regimes.** A0 yields a degree-dominated regime in which analytical baselines are near-optimal. HSCC yields a source-community regime in which strong flat baselines become the relevant comparator and graph value must be assessed against matched feature access rather than assumed.
3. **We benchmark analytical, flat, and GNN surrogates under both regimes, showing that GNN value is regime-dependent rather than universal while inference remains orders of magnitude faster than repeated MC simulation.** Under A0, the best GNN using raw node attributes (GCN, ρ=0.808 [✅]) remains statistically below degree (ρ=0.826 [✅]) under bootstrap testing (Δρ=−0.018, 95% CI [−0.029, −0.008]). Under HSCC, the best standard GNN (`gnn_raw_attr`, SAGE, ρ=0.915 [✅]) significantly outperforms the official flat comparator locked in the frozen HSCC bootstrap artifact (ρ≈0.884; Δρ=+0.033, 95% CI [+0.021, +0.044]). An optional ranking-aware training variant reaches ρ=0.924 with inferential gain vs the same comparator (Δρ=+0.041, 95% CI [+0.030, +0.053]); +0.009 vs standard training is descriptive only.

---

## 2. Background and Related Work

### 2.1 Independent Cascade and Influence Estimation

Kempe, Kleinberg, and Tardos [1] formalized the Independent Cascade (IC) model and the influence maximization (IM) problem. In IC, each directed edge (u,v) is independently activated with probability p(u,v), and activated nodes in turn attempt to activate their neighbors in subsequent rounds. For influence estimation without cascade logs, a common operational choice is the *weighted cascade* model, where p(u,v) = 1/deg(v), proposed by Kempe et al. and used subsequently in related learning-based influence maximization methods. GNN-based approaches to influence *estimation* (as opposed to seed-set maximization) are discussed in §2.2. Our task differs from seed-set optimization: we perform node-level IC score regression to learn a surrogate that rapidly approximates simulation-derived influence scores.

### 2.2 Node Importance and Structural Baselines

Prior work has established several strong structural baselines for ranking influential nodes. Degree centrality is often the most predictive structural measure under degree-coupled IC [1]. PageRank [4] captures global influence through random-walk stationary distribution. K-shell decomposition [5] identifies nodes in dense network cores. Two-hop spread counts second-order neighborhood size, approximating local cascade reach. Guille et al. [6] discuss evaluation challenges when behavioral data is unavailable; we follow their approach of using simulation-derived scores as evaluation targets.

Node2Vec [14] generates structure-aware node embeddings via biased random walk sampling on the graph, providing a shallow structural baseline that is separate from both analytical centrality measures and flat attribute models. We include it to distinguish graph-structure signal (captured through walk-based co-occurrence statistics) from learned message-passing signal (captured through GNN neighborhood aggregation).

Prior GNN-based approaches to social influence estimation — most notably DeepInf [3] (Qiu et al., KDD 2018), which predicts local social influence using graph attention networks trained on observed interaction logs — have demonstrated that neighborhood structure improves influence prediction when behavioral cascade data is available for supervision. Our setting differs in a key respect: we lack observed cascade logs and instead use Monte Carlo IC simulation to generate influence score labels from the static follower graph. This motivates our regime-dependent analysis: we ask when GNN message passing adds value *beyond* degree centrality and flat source-attribute models, given that the supervision signal itself is simulation-derived rather than observed.

### 2.3 GNN Architectures for Node-Level Surrogate Regression

We evaluate four GNN architectures under a unified training protocol: GraphSAGE [7] (neighborhood sampling and aggregation), GCN [8] (spectral graph convolution), GIN [9] (graph isomorphism network with maximally expressive aggregation), and APPNP [10] (personalized propagation of neural predictions). All architectures are applied as node-level regressors trained to predict continuous IC scores, not as graph classifiers. Community structure is identified using the Louvain algorithm [11] for HSCC parameter computation.

---

## 3. MC-IC as an Operational Metric

### 3.1 Construct Validity

The Twitch dataset [2] provides a follower graph where edges represent mutual follower relationships, not observed diffusion pathways. We treat the follower graph as a structural substrate for simulation: it specifies the topology through which influence could propagate, rather than recording actual propagation events. All findings in this paper are therefore properties of *simulation-defined influence approximation*, not measurements of real influence. We make no claim that MC-IC scores correspond to actual empirical influence in real-world propagation. Our contribution is a comparative study of how operationalization choice affects the learnability of the resulting influence surrogate.

### 3.2 Operationalizations

**A0 — Weighted Cascade (structural baseline regime)**

$$p(u, v) = \frac{1}{\deg(v)}$$

A0 is the standard structural operationalization proposed by Kempe et al. [1]. Transmission probability depends only on the in-degree of the target node, making it degree-coupled by design. A0 serves as our structural reference regime: it generates labels that should be highly predictable by degree-based analytical baselines, allowing us to characterize the learnability ceiling imposed by a purely structural IC specification.

**HSCC — Source-Community Regime (domain-informed)**

$$p(u, v) = \text{clip}\!\left(\lambda \cdot \frac{\phi(u)}{\deg(u)} \cdot \left(1 + \gamma \cdot \mathbf{1}[c_u \neq c_v]\right),\ 0,\ p_{\max}\right)$$

where the source engagement velocity term is:

$$\phi(u) = \frac{\text{rank}\!\left(\frac{\log(1 + \text{views}_u)}{1 + \text{life\_time}_u}\right)}{N}$$

This operationalization combines source-side engagement intensity (the φ(u) velocity term) with a structural incentive for cross-community spread (the 1+γ·𝟏[c_u≠c_v] amplification), making transmission probability depend on the *source* node's relative activity rather than the target's connectivity.

Fixed parameters: λ=1.0, γ=1.0, p_max=1.0. Three design choices merit brief justification:

- **Rank normalization on φ(u):** The views distribution is heavy-tailed; rank normalization bounds the source term to [0, 1] and prevents a small number of extreme accounts from distorting the transmission scale.
- **log1p(views)/(1+life\_time):** This inner term approximates engagement velocity rather than cumulative popularity — log1p compresses outliers, and dividing by (1+life\_time) avoids rewarding longevity alone.
- **Community amplification (1+γ·𝟏[c\_u≠c\_v]):** Encodes the structural-hole interpretation of cross-community bridging [12]; nodes at community boundaries have heightened transmission probability toward cross-community neighbors.

All parameters (λ, γ, p\_max) are fixed prior to any model training and not tuned to maximize downstream surrogate performance.

HSCC is introduced as a domain-informed comparative operationalization rather than as a validated generative law of Twitch diffusion. The fixed community-amplification configuration is kept as a transparent, frozen comparative setting instead of being tuned to maximize downstream surrogate gains. Estimating edge-level transmission probabilities from behavioral data would require supervised cascade logs unavailable in this dataset; weighted cascade and HSCC therefore provide principled zero-shot alternatives.

> **[AUTHOR NOTE — delete before submission]** Anticipating reviewer pushback on HSCC validity: we claim only that HSCC is a *comparative operationalization*, not a validated generative model of Twitch behavior. Its role is to introduce source-side and community-side signal into label generation in a transparent, frozen configuration, allowing a controlled test of whether graph message passing recovers structure that source-attribute flat models cannot. The §3.1 construct validity statement handles this.

### 3.3 Dataset and Simulation Protocol

We use the Twitch Gamers dataset [2]: **168,114 nodes**, **6,797,557 edges** (undirected mutual-follow graph). We label 5,000 nodes selected via degree-stratified sampling (degree quintiles, q=5; seed=42), running 200 MC-IC simulations per labeled node for each operationalization independently. Simulation is implemented with sparse CSR propagation; A0 labeling completes in ~480 seconds [✅] for 5,000 nodes (from `runtime_breakdown.csv`). HSCC labeling is more compute-intensive due to additional community lookup and source-velocity computation per edge traversal; report its wall-clock time only after verifying the corresponding HSCC simulation logs (do not infer it from `runtime_breakdown.csv`).

HSCC IC score summary (from `outputs/mapr2026_v3_results/hscc_refined_label_diagnostics.json`) [✅]:

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

The continuous formulation receives additional empirical support from two sources. First, under A0, formal label stability analysis across three independent MC campaigns (150 runs/seed in the stability experiment — separate from the main labeling protocol's 200 runs/seed; weighted-cascade model; `ic_label_stability.json`) yields a mean top-decile Jaccard of 0.31 [✅] — far below the 0.85 stability target — with maximum estimated Jaccard of 0.66 across all percentile thresholds from 3% to 30% (`phase2_threshold_analysis.json`). The cause is structural rather than sampling: 84.2% of communities span both the top-k and the boundary band (`phase1_community_overlap.json`), meaning within-community rank noise prevents stable binary separation regardless of propagation probability or threshold choice (`stability_explanation.json`, `interpretation: "structural"` [✅]). Second, within each degree band the IC score distribution remains highly variable under A0: intra-band CV ranges from 1.12 to 2.29 across degree quintiles [✅] (`degree_controlled_ic_variance.json`), confirming a smooth ranking gradient rather than a clean two-class partition.

Under HSCC, the structural instability argument is at least as strong: degree centrality collapses completely as a predictor (ρ = −0.006 [✅]), reach is substantially lower (mean = 4.83, CV = 0.583 [✅]), and label signal originates from source-velocity and community-bridging terms — creating an even more arbitrary top-k boundary. The community-overlap structural argument (84.2% spanning boundary) is a graph-topology property invariant to IC model parameterization, and therefore extends directly to HSCC. Formal Jaccard stability was measured under A0; the structural and topology-based arguments apply to both regimes.

> **[AUTHOR NOTE — scope of evidence]** All formal stability artifacts (`ic_label_stability.json`, `stability_explanation.json`, `phase1_community_overlap.json`, `phase2_threshold_analysis.json`) are from A0 analysis and exist under `outputs/day1_benchmark/` and `outputs/ic_feasibility/`. Path confirmed — NOT pending. For HSCC, cite the topology argument (community overlap is regime-invariant) and the degree-collapse evidence (ρ=−0.006 makes any degree-anchored threshold meaningless). Do not cite "Jaccard=0.31" for HSCC — that is an A0 measurement only.

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

**Bootstrap CI protocol:** 1,000-resample bootstrap over test predictions (note: "1,000 resamples" is the draw count — separate from the "1,000 test nodes" above), paired comparison, 95% percentile CI. Pre-registered equivalence bound δ₀=0.02. Results labeled as: gnn_significantly_better, practically_equivalent, or gnn_significantly_worse.

**Architectures and additional baselines:** The official active GNN comparison covers GraphSAGE, GCN, GIN, and APPNP. Node2Vec+LR is retained as a shallow-embedding baseline; per `runtime_breakdown.csv`, **`train_sec` ≈153s** bundles **Node2Vec embedding training + downstream LR fit**, while **`inference_sec` ≈0.04s** is **LR prediction on frozen embeddings only** — report precomputation separately from real-time analytical baselines (see Table 4 footnote).

### 4.2 Results: A0 — Structural Ceiling

Under A0, the degree-coupled label generation creates a structural ceiling that analytical baselines nearly saturate. **This is not a failure of GNNs but rather a property of the operationalization**: degree already encodes nearly all available diffusion signal.

#### Table 2: A0 Surrogate Results — Spearman ρ, NDCG@10%, and P@10% on 1,000 held-out test nodes; primary comparator: degree centrality (ρ=0.826). GNN rows averaged over 5 random seeds (± std). Analytical and flat baselines are deterministic. [✅ FROZEN]

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
| Node2Vec + LR | Shallow Embedding | 0.810 | 0.005 | 0.859 | 0.58 |
| **GCN (raw_attr)** | **GNN** | **0.808** | **0.001** | **0.825** | **0.53** |
| GIN (raw_attr) | GNN | 0.615 | 0.022 | 0.731 | 0.39 |
| SAGE (raw_attr) | GNN | 0.534 | 0.009 | 0.674 | 0.45 |
| APPNP (raw_attr)† | GNN | 0.585 | 0.417 | 0.724 | 0.49 |
| SAGE (centrality feats)‡ | Diag. | 0.828 | 0.0002 | 0.881 | 0.62 |

‡ SAGE (centrality feats): uses degree/pagerank/kshell as input features — diagnostic ceiling only; oracle feature access not available at inference time; excluded from main comparisons.
*Node2Vec: `train_sec` ≈153s bundles embedding + LR fit; reported runtime column in baselines is inference/predict only (~0.04s). If the team de-scopes Node2Vec before submission, synchronize deletion across all 4 locations: (a) §4.1 setup list (this mention), (b) Table 2 and Table 3 rows, (c) §4.5 runtime table, (d) §2.3 background mention. Search keyword "Node2Vec" in both Paper guide.md and Paper rules.md to confirm full sync.*
*† APPNP excluded from best-architecture selection: seed σ=0.417 ≥ `--gnn-std-threshold 0.1`; kept in table for completeness with dagger footnote (Rule B3).*

**Key finding (A0) [✅]:** The best raw-attribute GNN remains statistically below degree centrality under bootstrap testing:

> **Bootstrap A0:** GCN(raw_attr) vs degree — Δρ=−0.018, 95% CI [−0.029, −0.008] — *gnn_significantly_worse* [✅]
> NDCG@10%: Δ=−0.045, 95% CI [−0.076, −0.013] — *gnn_significantly_worse* [✅]

**Interpretation:** Under A0, the degree-coupled operationalization yields a structural ceiling that degree centrality nearly saturates; the binding constraint is the operationalization, not the model family. The centrality-feature diagnostic indicates the ceiling is reachable when degree-type information is explicitly provided as input features.

> **Paste-ready paper sentence [✅]:** Under degree-coupled IC (A0), the best GNN (GCN, raw node attributes) remains statistically below degree centrality (Δρ=−0.018, 95% CI [−0.029, −0.008]), consistent with a structural ceiling imposed by the degree-coupled operationalization — not a generic “GNN failure.”

### 4.3 Results: HSCC — Graph-Aware Regime

Under HSCC, the source-engagement and community-bridging signal in label generation creates a qualitatively different approximation landscape. Degree centrality collapses to ρ = −0.006 [✅] — a shift of 0.832 Spearman points from its A0 value (ρ = 0.826). This dramatic collapse signals a fundamental regime change: degree is no longer informative under source-community propagation, and the relevant comparator shifts to LR with source-side attributes. Degree is retained in Table 3 only as **contextual evidence of regime shift**, not as the primary comparator (footnote per Rule E5). The **official flat comparator** for bootstrap-tested claims is the model locked in `gnn_vs_baseline_bootstrap_ci_hscc.json`: **`lr_degree_views_life_time_lang`** (LR with degree, views, life_time, language), ρ≈0.884 [✅]. The key question is whether graph message passing recovers signal beyond what those flat baselines already encode.
The near-tie disclosure and comparator-lock justification are given once in the Table 3 footnote.

#### Table 3: HSCC Surrogate Results — Spearman ρ, NDCG@10%, and P@10% on 1,000 held-out test nodes; **primary comparator (bootstrap):** LR(degree, views, life_time, language) — `lr_degree_views_life_time_lang` (ρ=0.884) [✅]. GNN rows averaged over 5 random seeds (± std). [✅ FROZEN]

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
| Node2Vec + LR | Shallow Embedding | 0.570 | 0.015 | 0.723 | 0.29 |
| **LR(deg+views+life_time+lang)** | **Flat** | **0.884 [✅]** | — | **0.829** | **0.45** |
| LR(views+life_time+lang) | Flat | 0.884 [✅] | — | 0.830 | 0.45 |
| **SAGE (`gnn_raw_attr`)** | **GNN** | **0.915** | **0.004** | **0.902** | **0.58** |
| SAGE + rankloss (C3) | GNN | 0.924 | 0.002 | 0.911 | 0.60 |
| SAGE (all feats)‡ | GNN | 0.916 | 0.005 | 0.908 | 0.62 |
| GCN (raw_attr) | GNN | 0.602 | 0.014 | 0.694 | 0.21 |
| GIN (raw_attr) | GNN | 0.028 | 0.046 | 0.469 | 0.04 |
| APPNP (raw_attr)† | GNN | −0.037 | 0.146 | 0.491 | 0.03 |

*Comparator (bootstrap): `lr_degree_views_life_time_lang` — official per frozen `gnn_vs_baseline_bootstrap_ci_hscc.json`. Near-tie: `lr_views_life_time_lang` ρ=0.88442 vs `lr_degree_views_life_time_lang` ρ=0.88430 (Δ=0.00012); selection justified by pre-specification, not margin.*
*† APPNP excluded from best-architecture pool: σ=0.146 ≥ 0.1 (HSCC); σ=0.417 (A0). Listed for completeness.*
*Node2Vec: `train_sec` ≈153s bundles embedding + LR fit; inference ≈0.04s is predict-only (see §4.1).*
*Degree row: included for regime-contrast reference only (ρ=−0.006); not the HSCC decision comparator.*
*‡ SAGE (all feats): includes language dummies + degree/pagerank/kshell as input features — diagnostic ceiling only, not a fair main comparator (oracle feature access).*

**Key finding (HSCC) [✅]:** Paired bootstrap vs the official flat comparator gives:

> **Bootstrap HSCC:** `gnn_raw_attr` vs `lr_degree_views_life_time_lang` — Δρ=+0.033, 95% CI [+0.021, +0.044] — *gnn_significantly_better* [✅]
> NDCG@10%: Δ=+0.074, 95% CI [+0.050, +0.099] — *gnn_significantly_better* [✅]

**C3 rankloss [✅ frozen — optional for page budget only]:** `gnn_vs_rankloss_bootstrap_ci_hscc.json` — `best_arch_raw_attr_rankloss(sage)` vs same comparator: Δρ=+0.041, 95% CI [+0.030, +0.053] (*gnn_significantly_better*). The +0.009 gap vs standard SAGE (0.924 − 0.915) is **descriptive only** — no paired bootstrap vs standard training in artifacts; do not call it “significant.”

**Interpretation [✅]:** Under HSCC, GraphSAGE message passing significantly outperforms the official matched flat comparator under bootstrap testing, consistent with residual neighborhood-structured signal beyond node-level attributes. The architecture–regime interaction (e.g., GCN strongest under A0 vs SAGE strongest under HSCC; GIN collapsing under HSCC) is summarized in §4.4 and the Appendix.

> **Paste-ready paper sentence [✅ HSCC — significant vs flat comparator]:** Under source-community IC (HSCC), the best GNN (`gnn_raw_attr`, SAGE, raw attributes including language) significantly outperforms the official matched flat baseline — LR(degree, views, life_time, language), ρ≈0.884 — achieving ρ=0.915 (Δρ=+0.033, 95% CI [+0.021, +0.044]), consistent with residual neighborhood-structured signal beyond node-level attributes.
>
> **Optional C3 sentence [✅ inferential vs comparator + descriptive vs SAGE]:** A ranking-aware variant reaches ρ=0.924 (+0.009 descriptive over standard training) and Δρ=+0.041 vs the same flat comparator (95% CI [+0.030, +0.053]).

### 4.4 Regime Contrast

The contrast between A0 and HSCC reveals that surrogate learnability is not a property of the model alone; rather, it is jointly determined by the diffusion operationalization and the information already recoverable by simple baselines.

| Property | A0 | HSCC |
|----------|-----|------|
| Label signal source | Target degree (structural) | Source velocity + community bridging |
| Best analytical baseline | degree ρ=0.826 [✅] | degree ρ=−0.006 [✅] — collapses |
| Best trained (flat) baseline | LR(deg+views+life_time) ρ=0.522 [✅] | Official comparator: `lr_degree_views_life_time_lang` ρ≈0.884 [✅]; near-tie `lr_views_life_time_lang` |
| Best GNN (raw attrs) | GCN ρ=0.808 [✅] | SAGE (`gnn_raw_attr`) ρ=0.915 [✅] |
| Optional C3 (`best_arch_raw_attr_rankloss`) | — | SAGE ρ=0.924 [✅]; inferential Δρ vs same flat comparator |
| Bootstrap comparator | degree (analytical) | `lr_degree_views_life_time_lang` (locked in HSCC JSON) |
| Bootstrap result | Δρ=−0.018, CI [−0.029, −0.008] — *gnn_significantly_worse* [✅] | Δρ=+0.033, CI [+0.021, +0.044] — *gnn_significantly_better* [✅] |
| Best GNN architecture | GCN | GraphSAGE |
| APPNP seed variance (std) | 0.417 — excluded from best-arch pool [✅] | 0.146 — excluded [✅] |
| Primary insight | Degree-coupled IC = degree ceiling | Comparator shifts to language-aware flat models; SAGE adds inferential margin vs official comparator [✅] |

Under A0, degree-coupling creates a ceiling that raw-attribute GNNs do not exceed vs degree (bootstrap [✅]). Under HSCC, flat models reach ρ≈0.884 under matched features; SAGE (`gnn_raw_attr`) adds a significant increment vs the **official** flat comparator per frozen bootstrap (Δρ=+0.033 [✅]), with an optional rankloss variant at ρ=0.924 and Δρ=+0.041 vs the same comparator (CI [✅]; +0.009 vs standard SAGE is descriptive only).
The best-performing raw-attribute architecture differs by regime (GCN under A0; SAGE under HSCC), while APPNP is excluded from the best-architecture pool by the variance threshold; full architecture sensitivity is summarized in the Appendix.

### 4.5 Runtime

All surrogates provide inference dramatically faster than MC-IC simulation.

#### Table 4: Runtime Summary [✅ FROZEN]

| Model | Inference (full graph) | Training | Speedup vs MC-IC |
|-------|----------------------|----------|-----------------|
| MC-IC labeling (training labels; 5,000 nodes × 200 runs) | 480.3 s [✅] | — | 1× |
| degree (analytical) | 0.004 s [✅] | — | ~120,000× |
| LR (flat) | <0.003 s [✅] | <0.004 s | ~160,000× |
| Node2Vec + LR (shallow embedding) | ~0.040 s (predict) [✅] | ~153 s (embedding + LR fit bundled) [✅] | — |
| SAGE (raw_attr) | 0.086 s [✅] | ~27 s | ~5,590× |
| GCN (raw_attr) | 0.165 s [✅] | ~59 s | ~2,900× |
| GIN (raw_attr) | 0.067 s [✅] | ~24 s | ~7,200× |
| APPNP (raw_attr) | 0.790 s [✅] | ~286 s | ~610× |

*GNN inference times: full 168K-node graph in a single forward pass. Training: averaged over 5 seeds. MC-IC baseline: 480.3s for generating training labels (5,000 nodes × 200 runs; `mc_ic_labeling` in `runtime_breakdown.csv`). **Headline runtime anchor:** use SAGE `gnn_raw_attr` in the HSCC row. Speedup vs MC-IC for SAGE: ≈480.3 / 0.086 ≈ **5,590×** — round narrative to **~5,500×** (guide §4.5) to avoid false precision. Node2Vec: **`train_sec` ≈153s** = offline embedding training + LR fit bundled; **`inference_sec` ≈0.04s** = LR prediction only — not comparable to degree/LR(raw) without the precomputation disclaimer.*

*Runtime framing is regime-dependent: under A0, degree is ~120K× faster than MC-IC and near ceiling; under HSCC, flat baselines and GNNs achieve fast inference, while the **inferential** HSCC story comes from bootstrap JSONs, not wall-clock.*

> **[AUTHOR NOTE — guide §4.5 prose order for writing the runtime paragraph (6 steps):]**
> 1. MC-IC cost first: "A single MC-IC labeling pass requires 480 seconds" (the expensive step being replaced)
> 2. GNN inference cost: "Once trained, the GNN provides full-graph inference in approximately 0.086 seconds"
> 3. Speedup ratio: "a speedup of approximately 5,500×"
> 4. Clarify speedup scope: "speedup is inference vs one labeling pass — not training time"
> 5. Node2Vec separately: "Node2Vec requires ~153s offline precomputation per regime; downstream LR prediction is ~0.04s"
> 6. Analytical baselines last: "degree and k-shell are near-instantaneous (analytical formula)"
>
> Paste-ready English (guide §4.5):
> "Once trained, the GNN surrogate provides full-graph influence score inference in approximately 0.086 seconds, compared with 480 seconds for a single MC-IC labeling pass — a speedup of approximately 5,500×."
>
> Do NOT compare SAGE speedup with degree as if they have the same labeling cost. Do NOT use "0.09s" — inconsistent with frozen headline value 0.086s.

---

## 5. Discussion and Limitations

### 5.1 When Does GNN Surrogate Learning Help?

Our results suggest that the usefulness of graph learning for IC approximation is governed by the information structure of the diffusion operationalization rather than by architecture choice alone. The A0 finding is strongly supported: whenever the IC transmission formula is degree-coupled, structural analytical baselines provide a competitive ceiling that raw-feature GNNs do not beat vs degree under bootstrap testing (Δρ=−0.018 [✅]). The HSCC finding is supported at the inferential level for the **official** flat comparator in the frozen bootstrap JSON: SAGE (`gnn_raw_attr`) significantly outperforms that comparator (Δρ=+0.033, CI [+0.021, +0.044] [✅]), with a practical near-tie between two language-aware LR baselines in the point-estimate table (disclosed in Table 3 footnote).

Under A0, degree-coupling ensures that degree centrality is sufficient for most diffusion signal, leaving little room for graph aggregation to contribute beyond the structural ceiling. Under HSCC, strong flat models already reach ρ≈0.884 under matched features; yet neighborhood aggregation still adds a significant increment **vs the pre-specified comparator** per bootstrap (not vs an alternate near-tied LR row). The operative condition is therefore: GNN surrogates add inferential value when the operationalization leaves graph-mediated signal that matched flat models do not fully recover — as measured by the declared comparator and CI protocol.

> **[AUTHOR NOTE — for paper draft, use guide §5.1 paste-ready paragraph instead of the above:]**
>
> Our results reveal a regime-dependent answer: graph message passing adds value when the label generation process encodes structure that node-level attributes cannot fully capture alone. Under A0 — where transmission probability is degree-coupled — degree centrality already provides a near-optimal surrogate, leaving little residual for GNN message passing. Under HSCC — where transmission depends on source engagement velocity and cross-community bridging — node-level flat models remain competitive but graph message passing recovers additional signal (Δρ = +0.033, CI [+0.021, +0.044]). The implication is that the question "do GNNs help for influence prediction?" has no universal answer; it depends on what the diffusion operationalization encodes.
>
> The scaffold text above is retained for defensive detail (bootstrapped claims, comparator lock citation). The paste-ready version above is the preferred prose for the paper submission.

### 5.2 Limitations

1. **Follower graph ≠ observed diffusion:** Our graph is a follower network, whereas actual information cascades propagate along a different, unobserved subgraph of active interactions. Results should therefore be interpreted as properties of MC-IC surrogates on the follower topology, not as measurements of real Twitch information spread.

2. **Operationalization validity:** Both A0 and HSCC are simulation-based operationalizations of influence rather than empirically validated diffusion laws. Findings describe the learnability of each operationalization's output, not the learnability of real influence.

3. **HSCC not empirically calibrated:** The HSCC formula encodes domain-informed hypotheses (source velocity, community bridging) without direct calibration to observed cascade data. We claim only that it introduces community-side signal into label generation in a transparent, frozen configuration — not that it accurately reflects Twitch diffusion mechanics.

4. **Transductive evaluation:** Our evaluation is transductive: models are trained and evaluated on the same graph with held-out test nodes. Inductive generalization — applying the trained surrogate to new graphs or temporally shifted snapshots — is not assessed and remains an open evaluation challenge.

5. **Small mean reach under HSCC:** Under HSCC, mean cascade reach is 4.83 nodes — substantially below A0. This reflects the selective local-community propagation structure of the HSCC formula, not a deficiency of the operationalization. Rankings over these small cascades are still meaningful for identifying the most-connected community hubs, but the absolute influence magnitudes are not comparable to broad viral spread scenarios.

6. **Single dataset:** All experiments use a single snapshot of the Twitch Gamers follower graph. Generalizability to other social platforms, temporal dynamics, or different community structures has not been tested.

> **[AUTHOR NOTE — feature access parity, removed from numbered list]** HSCC claims require matched feature access between GNN and flat baselines (guide §4.2 fairness rule): if the GNN uses `language`, the flat comparator must also use `language`. The current comparator lock (`lr_degree_views_life_time_lang`, ρ=0.884) satisfies this. This is an experimental design constraint, not a limitation of findings — removed from limitations list to avoid reviewer confusion.

### 5.3 Scope of the Operationalization Choice

Our experiments fix two operationalizations (A0 and HSCC) with frozen hyperparameters (λ=1.0, γ=1.0). The HSCC parameters were not optimized to maximize downstream surrogate gains — they were fixed prior to any GNN training to preserve experimental integrity. This means the observed HSCC GNN increment (+0.033 ρ vs the official flat comparator [✅]) does not arise from tuning HSCC constants to favor GNNs — a design fact independent of the sign of the final bootstrap gap. Learning edge-level transmission probabilities directly from behavioral data would require supervised cascade logs unavailable in this dataset; A0 and HSCC therefore represent principled zero-shot operationalization choices appropriate for the no-cascade-log setting.

---

## 6. Conclusion

> **[AUTHOR NOTE — guide §5.4 provides a cleaner 5-sentence template. Use this for submission prose:]**
>
> We studied the learnability of two IC operationalizations on the Twitch Gamers social graph. Under a degree-coupled regime (A0), the best GNN remains statistically below degree centrality (Δρ = −0.018, CI [−0.029, −0.008]), confirming that the diffusion operationalization imposes a structural ceiling that degree already saturates. Under a source-community regime (HSCC), graph message passing significantly outperforms the strongest matched flat baseline (Δρ = +0.033, CI [+0.021, +0.044]), with an optional ranking-aware variant reaching Δρ = +0.041. In all cases, trained surrogates provide influence score inference orders of magnitude faster than repeated MC simulation. Our central finding is that operationalization choice, rather than architectural capacity, governs when graph learning adds value for simulation-based influence prediction.
>
> The scaffold paragraph below retains defensive framing (Δρ=+0.041 qualifications, +0.009 descriptive-only note) useful for reviewer response prep; trim to the template above for submission.

We have shown that the value of GNN surrogates for Monte Carlo influence estimation is regime-dependent, not universal. Under a degree-coupled operationalization (A0), analytical structural baselines represent a near-optimal ceiling that the best raw-attribute GNN does not exceed vs degree under bootstrap testing (Δρ=−0.018, CI [−0.029, −0.008] [✅]). Under a source-community operationalization (HSCC), graph message passing adds a significant increment over the **official** matched flat comparator locked in the frozen HSCC bootstrap artifact (Δρ=+0.033, CI [+0.021, +0.044] [✅]), with an optional ranking-aware variant reaching Δρ=+0.041 vs the same comparator (CI [+0.030, +0.053] [✅]; +0.009 vs standard training is descriptive only). Learned surrogates remain orders of magnitude faster than repeated MC-IC. Ranking-aware training provides a further marginal gain under HSCC, suggesting that explicit rank supervision is a viable extension when the operationalization is source-side-driven. Our findings suggest that diffusion operationalization — rather than architecture alone — is the primary driver of when graph learning adds value for influence approximation on dense social networks.

---

## References

[1] D. Kempe, J. Kleinberg, and É. Tardos, "Maximizing the spread of influence through a social network," in *Proc. KDD*, 2003.

[2] B. Rozemberczki and R. Sarkar, "Twitch Gamers: a Dataset for Evaluating Proximity Preserving and Structural Role-based Node Embeddings," arXiv:2101.03091, 2021.

[3] J. Qiu, J. Tang, H. Ma, Y. Dong, K. Wang, and J. Tang, "DeepInf: Social influence prediction with deep learning," in *Proc. ACM KDD*, 2018, pp. 2110–2119.

[4] L. Page, S. Brin, R. Motwani, and T. Winograd, "The PageRank citation ranking: Bringing order to the web," Stanford Technical Report, 1999.

[5] M. Kitsak, L. K. Gallos, S. Havlin, et al., "Identification of influential spreaders in complex networks," *Nature Physics*, vol. 6, pp. 888–893, 2010.

[6] A. Guille, H. Hacid, C. Favre, and D. A. Zighed, "Information diffusion in online social networks: A survey," *SIGMOD Record*, vol. 42, no. 2, 2013.

[7] W. L. Hamilton, Z. Ying, and J. Leskovec, "Inductive representation learning on large graphs," in *Proc. NeurIPS*, 2017.

[8] T. N. Kipf and M. Welling, "Semi-supervised classification with graph convolutional networks," in *Proc. ICLR*, 2017.

[9] K. Xu, W. Hu, J. Leskovec, and S. Jegelka, "How powerful are graph neural networks?" in *Proc. ICLR*, 2019.

[10] J. Klicpera, A. Bojchevski, and S. Günnemann, "Predict then propagate: Graph neural networks meet personalized PageRank," in *Proc. ICLR*, 2019.

[11] V. D. Blondel, J.-L. Guillaume, R. Lambiotte, and E. Lefebvre, "Fast unfolding of communities in large networks," *J. Statistical Mechanics*, 2008.

[12] R. S. Burt, *Structural Holes: The Social Structure of Competition*. Harvard University Press, 1992.

[13] M. De Domenico, A. Lima, P. Mougel, and M. Musolesi, "The anatomy of a scientific rumor," *Scientific Reports*, vol. 3, no. 2980, 2013. [Higgs Twitter dataset: real retweet/reply/mention cascades following the Higgs boson discovery announcement.]

[14] A. Grover and J. Leskovec, "node2vec: Scalable feature learning for networks," in *Proc. ACM KDD*, 2016.

> **[AUTHOR NOTE — References budget guidance for 6-page IEEE format]**
>
> Current count: 14 references. Target for final submission: ~12–14 (depends on venue template column width).
>
> **Lowest-priority references (candidates for removal if space is tight):**
> - **[13] Higgs dataset (De Domenico et al.):** Cited only in §5.2 Limitation 6 as a future-work dataset. If the limitation is reworded to "graphs with observed behavioral cascade logs" without naming a specific dataset, [13] can be removed.
>
> **Must-keep references (do not remove):** [1] Kempe (IC model), [2] Rozemberczki & Sarkar (Twitch Gamers dataset), [3] Qiu/DeepInf (closest prior GNN influence work), [7] Hamilton/SAGE, [8] Kipf/GCN, [9] Xu/GIN, [10] Klicpera/APPNP, [11] Blondel/Louvain (community detection used in HSCC), [14] Grover/Node2Vec.
>
> **Lakens (2017) equivalence testing:** Add as reference ONLY if the main paper text explicitly uses "equivalence bound" or TOST language. Otherwise keep in `Paper guide.md` §6.4 (reviewer defense only) — not in main paper references.

---

## Appendix: Architecture Sensitivity Analysis

> [ℹ NOTE: Include as appendix if page budget allows; cut first if tight]

The architecture–regime interaction is substantial: best raw_attr architecture is **GCN under A0** (ρ=0.808) vs **SAGE under HSCC** (`gnn_raw_attr`, ρ=0.915), while **GIN is near-random under HSCC** (ρ=0.028) despite moderate A0 performance — report as a comparative finding (Rule E12), not as “GIN is generally weak.” We hypothesize:

- **SAGE leads under HSCC [✅]:** Mean aggregation may better aggregate source-side engagement signals under the HSCC label structure than symmetric normalization or sum aggregation.
- **GCN trails under HSCC [✅]:** Symmetric normalization may partially re-couple representations to degree — unhelpful when degree is uninformative (ρ=−0.006).
- **GIN collapses under HSCC [✅]:** ρ=0.028 (σ=0.046) is near-random ranking performance; frame as operationalization mismatch for sum aggregation under source-velocity labels.
- **APPNP unstable in both regimes [✅]:** excluded from best-architecture pool by policy (A0 σ=0.417; HSCC σ=0.146 > 0.1); reported for completeness only.

These hypotheses suggest that aggregation strategy and degree normalization choices interact strongly with IC operationalization in ways not captured by standard GNN benchmarking. This observation motivates future work on architecture selection criteria under different diffusion regimes.

---

## Figure Descriptions (for actual figure generation)

> [ℹ For paper typesetting — team should generate these figures from frozen output files]

### Figure 1: Pipeline Diagram
Five-box left-to-right pipeline (no feedback loops; Operationalization box uses fork shape):
- **Box 1 — Input:** "Twitch Gamers Graph (168K nodes, 6.8M edges) + node attributes (views, life_time, language)"
- **Box 2 — Operationalization (fork):**
  - Branch A0: "p(u,v) = 1/deg(v)" → degree-coupled labels
  - Branch HSCC: "p(u,v) = clip(λ·φ(u)/deg(u)·(1+γ·𝟏[c_u≠c_v]), 0, p_max)" → source-velocity + community labels
- **Box 3 — MC-IC Simulation:** "5,000 labeled nodes, 200 runs/node → continuous influence scores"
- **Box 4 — Surrogate Models:** "Analytical (degree, k-shell) | Flat (LR, MLP) | GNN (SAGE, GCN, GIN, APPNP†)"
- **Box 5 — Outputs:** "Regime-specific claims + runtime (MC-IC 480s [✅] vs GNN ~0.086s → ≈5,500× speedup [✅])"

†APPNP: shown in Box 4 with dagger; footnote "excluded from best-arch (std > 0.1)".
Each box ≤ 2 lines of text. Arrows: strictly left → right.

### Figure 2: Two-Panel Results Figure
- Left panel (A0): Dot plot with 95% error bars, x-axis = Spearman ρ, y-axis = models (degree baseline highlighted with dashed line, GCN highlighted as best GNN). Show regime where GNNs cluster below degree. [✅ layout confirmed from frozen A0 data]
- Right panel (HSCC): Dot plot with 95% error bars, x-axis = Spearman ρ, y-axis = models (vertical dashed line at official comparator LR(degree, views, life_time, language) ρ≈0.884; highlight SAGE `gnn_raw_attr` ρ=0.915; optional SAGE+rankloss ρ=0.924; dagger † on APPNP). Bootstrap label from `gnn_vs_baseline_bootstrap_ci_hscc.json`: *gnn_significantly_better* [✅].
- Grayscale-compatible; both panels share same x-axis scale [−0.1, 1.0].

---

## Section Status Tracker

| Section | Status | Action |
|---------|--------|--------|
| Abstract | Frozen Δρ/CIs + correct direction [✅] | Trim to ≤150 words for venue limit; remove [✅] tags; keep C3 only if Table 3 includes rankloss row |
| §1 Introduction | Draft — core numbers frozen; HSCC timing unverified | Polishing pass; do NOT cite an HSCC labeling-time number without verifying |
| §2 Background | Draft — [3] resolved as DeepInf (Qiu et al. KDD 2018) [✅]; GNN prior work paragraph added to §2.2 [✅]; Node2Vec [14] added [✅] | Done — proofread §2.2 prose flow before submission |
| §3 MC-IC Metric | Draft — HSCC stats [✅]; A0 regime row 🔲 placeholder | Optional: fill A0 full distribution row from `ic_scores_a0.parquet` |
| §4 Results | Frozen surrogate + bootstrap values [✅]; APPNP †, SAGE (all feats) ‡ marked | Decide C3 row under page limit; write prose using §4.3 paste-ready sentences |
| §4.5 Runtime | Frozen [✅]; AUTHOR NOTE with 6-step prose order added | Use 0.086s / ~5,590× (round to ~5,500× in prose); write per AUTHOR NOTE order |
| §5.1 | Scaffold + AUTHOR NOTE with clean paste-ready paragraph [✅] | Use AUTHOR NOTE paragraph for submission prose |
| §5.2 Limitations | Rewritten to guide paste-ready English [✅]; 6 limitations | Trim to 5 if space-constrained (cut L5 HSCC reach first) |
| §6 Conclusion | Scaffold + AUTHOR NOTE with 5-sentence template [✅] | Use AUTHOR NOTE template for submission; remove [✅] tags |
| References | [3] resolved as DeepInf (Qiu et al., KDD 2018) [✅]; [14] Grover & Leskovec 2016 [✅]; 14 total refs | See AUTHOR NOTE after refs for budget guidance if page-constrained |
| Appendix | Optional — architecture note [✅] | Cut first if tight |
