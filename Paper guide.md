# Supervisor Guidance: Writing the MAPR 2026 Paper

## 9-Day Execution Plan for a Defensible Dual-Operationalization Paper

---

## Part 1: Paper Identity — What This Paper Actually Is

Before writing a single word, the team must internalize what this paper contributes and what it does not claim.

**This paper is:** A comparative empirical study showing that the effectiveness of GNN surrogate learning for IC influence approximation depends critically on the IC operationalization. Under degree-coupled IC (A0), analytical baselines are near-optimal. Under attribute-community IC (HSCC), GNN message passing captures cross-community engagement patterns that flat baselines cannot access. The stability analysis revealing structural binary instability is a secondary but genuinely novel methodological finding.

**This paper is not:** A claim that GNN is universally superior for influence prediction. It is not a claim that HSCC is "the correct" diffusion model for Twitch. It is not a claim that MC-IC scores represent real influence.

**The title should reflect the contrast, not a blanket GNN claim.** Something like: "When Does Graph Learning Outperform Analytical Baselines? A Comparative Study of IC Operationalizations for Influence Approximation" or more concisely: "Regime-Dependent GNN Surrogate Learning for Monte Carlo Influence Estimation on Social Networks."

---

## Part 2: Section-by-Section Writing Guide

### Section 1 — Introduction (0.5 pages, ~400 words)

**Paragraph 1 (3-4 sentences).** Identifying influential users in online social networks is critical for viral marketing, community management, and platform recommendation. Monte Carlo Independent Cascade (MC-IC) simulation provides a principled operationalization of influence potential grounded in the diffusion model of Kempe et al. (2003). However, MC-IC is computationally expensive — requiring hundreds of stochastic simulations per node — making it impractical for repeated evaluation on large-scale graphs.

**Paragraph 2 (3-4 sentences).** Graph Neural Networks (GNNs) offer a natural surrogate: learn to approximate IC scores from graph structure and node attributes, then deploy the trained model for fast inference. Prior work on GNN-based influence estimation (Kumar et al., 2022; Ling et al., 2023) has focused on single diffusion models, leaving open the question of when learned representations actually outperform simple analytical baselines such as degree centrality. On dense social networks where cascades die quickly, degree itself may capture most of the diffusion signal, leaving little room for GNN improvement.

**Paragraph 3 (4-5 sentences — contributions).** In this paper, we investigate how the choice of IC operationalization determines whether GNN surrogate learning adds value over analytical baselines. We compare two defensible operationalizations on the Twitch social network (168K nodes, 6.8M edges): (1) weighted cascade (A0), where transmission probability depends only on target degree, and (2) HSCC, a domain-informed variant incorporating source engagement velocity and cross-community amplification. Our contributions are threefold. First, we show that binary influence classification is structurally unstable on dense networks, motivating continuous regression as the principled prediction formulation. Second, we demonstrate that under degree-coupled IC, all GNN architectures converge to the degree centrality ceiling, confirming that the operationalization — not the model architecture — is the binding constraint. Third, under HSCC, GNN message passing captures cross-community engagement patterns that flat baselines cannot access, achieving Spearman ρ = [X] compared to [Y] for the strongest non-graph baseline.

### Section 2 — Background (0.75 pages)

**2.1 Independent Cascade and Weighted Cascade.** Define the IC model following Kempe et al. (2003). State the weighted cascade parameterization p(u,v) = 1/deg(v) and cite its use in DeepIM (Ling et al., 2023). Explain that MC estimation requires R simulation runs per seed node, producing a mean reach score. Define the surrogate learning problem: given IC scores for a subset of nodes, learn a function f(G, X) → IC scores for all nodes.

**2.2 GNN Architectures.** One paragraph covering the five architectures tested, with one sentence each: GraphSAGE (mean aggregation, Hamilton et al. 2017), GCN (symmetric normalization, Kipf and Welling 2017), GIN (sum aggregation with WL-equivalent expressiveness, Xu et al. 2019), GAT (learned attention weights, Veličković et al. 2018), and APPNP (decoupled embedding and K-step Personalized PageRank propagation, Klicpera et al. 2019). State that all are evaluated with identical hyperparameters for fair comparison.

**2.3 Evaluation Protocol.** State the transductive setting. Define Spearman ρ (primary), NDCG@10% (secondary). State that metrics are computed on held-out labeled nodes only, with full-graph inference reported solely for runtime assessment. Pre-register the practical equivalence bound of |Δ Spearman| ≤ 0.02.

### Section 3 — MC-IC as Operational Metric (1.0 pages)

This section must accomplish three things: justify IC as a metric, present the stability finding, and introduce the two operationalizations.

**3.1 Construct Validity and Operationalization Design.** Use the paragraph from Implementation Plan Section 1.1 almost verbatim — it is well-written. Then introduce A0 and HSCC as two defensible but qualitatively different operationalizations. A0 models attention dilution (each neighbor receives equal probability inversely proportional to target degree). HSCC models engagement-velocity-driven diffusion amplified by cross-community bridging. State explicitly that HSCC is a domain-informed design, not a claim about true Twitch diffusion mechanics.

**3.2 Discriminativeness.** Present the IC reach distribution for A0 (mean 31.1, median 6.25, top-10/median ratio ~8×). State that IC scores follow a heavy-tailed distribution consistent with real influence dynamics where most nodes have limited reach but a small fraction can trigger large cascades. For HSCC, report mean reach 4.83 with CV 0.583.

**3.3 Label Stability — A Structural Finding.** This is the paper's most novel methodological contribution. Present the Jaccard instability (0.307 at 150 runs, never exceeding 0.68 at 1200 runs). Present the structural cause: 84.2% of Louvain communities span the top-10% boundary, and gap-to-noise ratios are near zero at all tested thresholds. State the conclusion: binary influence classification is structurally unstable on dense social networks with heavy-tailed IC distributions. This instability is irreducible by increasing simulation runs — it reflects a property of the graph topology, not simulation variance.

**3.4 Regression Formulation.** State that the stability analysis empirically motivates continuous regression on log-transformed IC scores as the principled prediction formulation. Cite the Spearman stability (0.827 at 1200 runs) as evidence that rank ordering is stable even when binary top-k membership is not. Present this as a positive design choice, not a fallback.

### Section 4 — GNN Surrogate Learning (2.0 pages)

**4.1 Experimental Setup (0.3 pages).** Dataset: Twitch MUSAE, 168,114 nodes, 6,797,557 edges, undirected mutual-follow. IC: 5,000 stratified nodes × 200 runs. Split: 80/20 degree-stratified, shared across operationalizations. GNN: 5 architectures, hidden_dim=128, 2 layers, dropout=0.3, HuberLoss, 200 epochs, 5 seeds. Baselines: degree, PageRank, k-shell, one-hop spread, two-hop spread, LR variants, MLP.

**4.2 A0 Results — Structural Ceiling (0.5 pages).** Present the main results table for A0. Degree achieves 0.826, two-hop 0.804, best GNN approximately 0.82. State the bootstrap CI result. If CI includes zero within ±0.02: "Under A0, all GNN architectures achieve Spearman ρ statistically equivalent to degree centrality (bootstrap 95% CI: [X, Y]). This confirms that when IC transmission probability is a direct function of target degree, analytical baselines capture the dominant signal." Present the +0.099 message passing finding (GNN-raw-attr 0.534 vs MLP 0.435) as evidence that graph structure provides signal, but the signal is already fully captured by precomputed centrality.

**4.3 HSCC Results — Graph-Aware Regime (0.5 pages).** Present the HSCC baseline table. Degree: ~0.04. LR(life_time): ~0.80. LR(views+life_time): ~0.81. MLP(raw attrs): ~0.85-0.87. Best GNN: ~0.88-0.93. Bootstrap CI comparing GNN vs strongest flat baseline. The key narrative: under HSCC, degree is no longer informative (rho=0.037), but life_time is a strong single predictor (rho=-0.801). GNN's advantage comes from learning cross-community structure through language-based message passing — a signal that flat baselines cannot access because it requires aggregating neighborhood community composition.

**If GNN dùng language feature:** You must include LR(views+life_time+language) and MLP(raw attrs+language) as fairness baselines. If these fairness baselines also reach ~0.87-0.89, the GNN advantage narrows to the message passing component only. Report this honestly.

**4.4 Contrast Analysis (0.4 pages).** This is the core intellectual contribution. Why does GNN win under HSCC but not A0? Under A0, R²(IC, degree) = 0.887 — degree explains 89% of variance. Adding any attribute explains less than 0.1% additional variance. There is no signal for GNN to learn beyond what degree already provides. Under HSCC, degree explains only 0.1% of variance. The IC score decomposes into phi(u) (engagement velocity, accessible to MLP from raw attributes) and cross_community_fraction (structural bridging, accessible only to GNN through message passing). The oracle analysis (phi × cross_frac = 0.931) confirms this decomposition.

**4.5 Runtime (0.3 pages).** MC-IC labeling: 480s for 5,000 nodes. GNN inference: 0.067s for 168,114 nodes. Speedup: 7,169×. Frame this as the practical motivation for surrogate learning regardless of whether GNN beats degree: even if GNN only matches degree, it provides the same ranking 7,000× faster.

### Section 5 — Discussion and Limitations (0.5 pages)

**5.1 When Does GNN Add Value?** Only when the diffusion model encodes information that requires neighborhood aggregation to recover — specifically, when IC scores depend on the composition of a node's neighborhood (which neighbors have high engagement, which are in different communities) rather than just the node's local degree. This is a structural condition on the IC formula, not a property of GNN architecture.

**5.2 Limitations.** State all four clearly. First, the follower graph is not observed diffusion — all findings are properties of the simulation, not measurements of real influence. Second, HSCC is a novel, domain-informed formula designed to test whether neighborhood composition matters; it is not validated as the true Twitch diffusion model. Third, HSCC has small mean reach (4.83 nodes) — while realistic for selective social diffusion, this limits the dynamic range of the regression target. Fourth, life_time is a strong baseline predictor under HSCC (rho=-0.801); the GNN advantage depends on fairness of baseline feature access.

**5.3 Why Not Learn p From Data?** One sentence: learning edge-level transmission probabilities requires supervised cascade logs unavailable in this dataset; weighted cascade and HSCC provide principled zero-shot alternatives.

---

## Part 3: Papers to Read and Cite

### Must-Cite (Core Framework)

**Kempe, Kleinberg, and Tardos (2003).** "Maximizing the Spread of Influence through a Social Network." KDD. This is the foundational reference for the IC model and weighted cascade parameterization. Cite for the IC model definition, the NP-hardness of influence maximization, and the weighted cascade p(u,v) = 1/in-degree(v) formulation.

**Ling, Jiang, Wang, Thai, Xue, Song, Qiu, and Zhao (2023).** "Deep Graph Representation Learning and Optimization for Influence Maximization." ICML. The DeepIM paper. Cite for two things: (1) the weighted cascade experimental setup that you follow, and (2) as a representative of learning-based influence maximization methods. This paper uses GAT with monotonicity constraints and knowledge distillation for seed set optimization — a different task from yours (they optimize seed sets, you approximate individual IC scores), but the diffusion model setup is the same.

**Hamilton, Ying, and Leskovec (2017).** "Inductive Representation Learning on Large Graphs." NeurIPS. GraphSAGE paper. Cite for the SAGE architecture and the inductive learning paradigm.

**Kipf and Welling (2017).** "Semi-Supervised Classification with Graph Convolutional Networks." ICLR. GCN paper. Cite for the GCN architecture and specifically for the symmetric normalization D^{-1/2}AD^{-1/2} that is structurally analogous to the A2 sensitivity variant.

**Xu, Hu, Leskovec, and Jegelka (2019).** "How Powerful are Graph Neural Networks?" ICLR. GIN paper. Cite for the GIN architecture and the WL-equivalence expressiveness result that motivates including GIN as the maximally expressive baseline.

**Veličković, Cucurull, Casanova, Romero, Liò, and Bengio (2018).** "Graph Attention Networks." ICLR. GAT paper. Cite for the attention-based aggregation mechanism and the hypothesis that attention could learn degree-inversely-proportional weighting.

**Klicpera, Bojchevski, and Günnemann (2019).** "Predict Then Propagate: Graph Neural Networks Meet Personalized PageRank." ICLR. APPNP paper. Cite for the decoupled embed-then-propagate architecture and the K-step Personalized PageRank propagation that motivates testing deeper receptive fields for IC approximation.

**Rozemberczki, Allen, and Sarkar (2021).** "Multi-Scale Attributed Node Embedding." Journal of Complex Networks. The Twitch MUSAE dataset paper. Cite for the dataset description, the mutual-follow edge semantics, and the node attributes (views, life_time, language, dead_account).

### Should-Cite (Strengthens Specific Claims)

**Kitsak, Gallos, Havlin, Liljeros, Muchnik, Stanley, and Makse (2010).** "Identification of Influential Spreaders in Complex Networks." Nature Physics. Cite for the finding that k-shell coreness predicts spreading ability, which provides context for why structural centrality baselines are strong competitors under degree-coupled IC.

**Guille, Hacid, Favre, and Zighed (2013).** "Information Diffusion in Online Social Networks: A Survey." ACM SIGMOD Record. Cite specifically Section 4 on evaluation challenges when behavioral ground truth is unavailable. This supports the construct validity discussion.

**Burt (1992).** "Structural Holes: The Social Structure of Competition." Harvard University Press. Cite for the structural holes theory that provides domain justification for the cross-community amplification component of HSCC. Nodes bridging structural holes have disproportionate influence because information through them reaches otherwise disconnected groups.

**Benjamini and Hochberg (1995).** "Controlling the False Discovery Rate." Journal of the Royal Statistical Society Series B. Cite if you use BH-FDR correction for any multiple testing.

**Blondel, Guillaume, Lambiotte, and Lefebvre (2008).** "Fast Unfolding of Communities in Large Networks." Journal of Statistical Mechanics. Cite for the Louvain community detection algorithm used to compute community assignments and cross-community edge fractions.

### Consider-Citing (Adds Depth If Space Permits)

**Chen, Wang, and Wang (2010).** "Scalable Influence Maximization for Prevalent Viral Marketing in Large-Scale Social Networks." KDD. Cite for the PMIA model showing influence decays exponentially with hop count, supporting your finding that cascades die within 1-3 hops on dense graphs.

**Kumar, Mallik, Khetarpal, and Panda (2022).** "Influence Maximization in Social Networks Using Graph Embedding and Graph Neural Network." Information Sciences. Cite as a representative of GNN-based influence approaches that use IC-simulated labels for training — the same paradigm you follow, but on smaller networks and without the dual-operationalization comparison.

**Aral and Walker (2012).** "Identifying Influential and Susceptible Members of Social Networks." Science. Cite for the empirical finding that social ties correlate with influence pathways, supporting the construct validity of using follower graphs for diffusion simulation.

**Grover and Leskovec (2016).** "node2vec: Scalable Feature Learning for Networks." KDD. Cite if Node2Vec is included as a baseline.

**Lü, Chen, Ren, Zhang, Zhang, and Zhou (2016).** "Vital Nodes Identification in Complex Networks." Physics Reports. Cite for the comprehensive survey on node importance metrics, providing context for why degree, PageRank, k-shell, and betweenness are standard baselines.

---

## Part 4: Critical Execution Items for the Next 9 Days

### Day 21 (Today) — Lock and Verify

Person 1 must verify that `regression_targets_hscc_refined.parquet` exists and is readable. If it does not exist, regenerate it immediately from the existing HSCC IC scores. This is a one-line operation and blocks everything else. Person 1 must also add the HSCC formula entry to `experiment_registry.md` with the exact parameters (lambda=1.0, gamma=1.0, p_max=1.0).

Person 2 must confirm that `community_features.parquet` covers 100% of active nodes with both `community_id` and `cross_community_edge_fraction`. This is an upstream dependency for HSCC interpretation.

Person 3 must update the evaluation harness to accept a `label_regime` parameter so that baseline and GNN results are clearly tagged as A0 or HSCC. This is a code change, not a new experiment.

### Days 22-23 — Baseline Fairness (HSCC) + GNN Training (Both Regimes)

Person 3 runs HSCC flat baselines first, before any GNN: LR(life_time), LR(views+life_time), LR(degree+views+life_time), MLP(views, life_time). If GNN will use language as a feature, also run LR(views+life_time+language) and MLP(views+life_time+language). Record the strongest flat baseline Spearman — this becomes the HSCC bootstrap comparator.

Person 3 simultaneously trains GNN architectures on both A0 and HSCC labels. For A0, raw_attr features (views_log, views_per_day, life_time). For HSCC, same raw_attr plus language if the team decides to include it. Five architectures × five seeds × two regimes = 50 training runs. At ~23 seconds per run, this is approximately 20 minutes total on GPU.

### Day 24 — Bootstrap CI + Result Locking

Person 3 runs bootstrap CI for both regimes. A0: GNN best vs degree. HSCC: GNN best vs strongest flat baseline (identified from Day 22 results). Record both Spearman and NDCG CIs.

All experimental results are frozen after Day 24. No new experiments, no parameter changes, no additional IC formulations.

### Days 25-27 — Paper Writing

Day 25: Draft Sections 1-2 (Introduction + Background). These do not depend on exact numbers.

Day 26: Draft Sections 3-4 (MC-IC Metric + GNN Results) with actual numbers from the frozen result tables. Create Figure 1 (pipeline diagram) and Figure 2 (two-panel results showing A0 and HSCC contrast).

Day 27: Draft Section 5 (Discussion + Limitations). Complete all tables. Internal review pass.

### Days 28-29 — Polish and Format

Day 28: IEEE formatting, double-blind compliance check (remove all author names, affiliations, acknowledgments from PDF), figure readability in grayscale, reference formatting.

Day 29: Final read-through by all team members. Fix any claim that is not supported by the frozen results. Ensure the abstract accurately reflects actual findings, not hoped-for findings.

### Day 30 — Submit

Submit. Do not make last-minute changes to results or claims.

---

## Part 5: The Three Claims the Paper Must Support With Evidence

Every claim in the paper must be traceable to a specific artifact. Here is the mapping:

**Claim 1: Binary influence classification is structurally unstable on dense social networks.** Evidence: `stability_explanation.json` (pct_communities_spanning_boundary = 0.842, mean_gap_to_noise near zero), Jaccard stability sweep (0.307 → 0.682, never reaching 0.85), Spearman stability (0.685 → 0.827 — rank ordering stabilizes but binary membership does not).

**Claim 2: Under degree-coupled IC (A0), GNN is practically equivalent to degree centrality.** Evidence: `gnn_vs_degree_bootstrap_ci_a0.json` (bootstrap CI including zero within ±0.02), A0 results table showing degree 0.826 vs best GNN ~0.82. Message passing contribution: GNN-raw-attr 0.534 vs MLP-raw-attr 0.435 = +0.099.

**Claim 3: Under attribute-community IC (HSCC), GNN outperforms flat baselines by capturing cross-community engagement structure.** Evidence: `gnn_vs_baseline_bootstrap_ci_hscc.json` (bootstrap CI for GNN vs strongest flat baseline), HSCC results table showing degree ~0.04 and GNN ~0.88-0.93. Oracle decomposition: phi × cross_frac = 0.931 confirming that the IC signal decomposes into an attribute component and a structural component.

If any of these three claims cannot be supported by the actual experimental results, the paper must be rewritten to reflect what the data actually shows. The contrast story (Claims 2+3 together) is publishable even if Claim 3 shows only a modest GNN margin, because the contrast itself is the finding.

---

## Part 6: What Reviewers Will Ask and How to Answer

**"Why not use real cascade data?"** Answer: The Twitch dataset does not contain behavioral cascade logs. MC-IC provides a principled simulation-based proxy following established methodology (Kempe et al., 2003; Ling et al., 2023). All findings should be interpreted as properties of the simulation, not measurements of real influence. This is stated explicitly in Section 3.1 and Limitations.

**"Why is HSCC a good diffusion model?"** Answer: We do not claim HSCC is the true Twitch diffusion model. HSCC is a domain-informed operationalization designed to test whether neighborhood composition adds learnable value for GNN. The engagement velocity component (views per unit active time) is a reasonable proxy for content production rate. The cross-community amplification is motivated by structural holes theory (Burt, 1992). The paper's contribution is the comparative finding, not a claim about HSCC's realism.

**"Why not compare against published influence maximization methods like DeepIM?"** Answer: DeepIM and similar methods solve a different problem: selecting optimal seed sets to maximize total cascade reach. Our task is node-level IC score regression — predicting individual influence potential, not optimizing a collective seed set. The comparison is not directly applicable, though the diffusion model parameterization follows DeepIM's weighted cascade setup.

**"The Twitch dataset is from 2021. Is it still relevant?"** Answer: The MUSAE Twitch dataset is a standard benchmark for graph-level analysis, cited in numerous recent works. The contribution is methodological (when does GNN add value for IC approximation), not findings specific to the Twitch platform circa 2021. Dataset age does not affect the methodological contribution.

**"Life_time dominates HSCC labels. Isn't GNN just learning life_time?"** Answer: This is addressed through baseline fairness. LR(life_time) achieves rho ~0.80 and MLP(views+life_time) achieves ~0.85-0.87. GNN's additional margin comes from cross-community structure learned through message passing, not from better life_time processing. The oracle analysis (phi × cross_frac = 0.931) decomposes the signal into attribute and structural components, confirming that the structural component is what GNN uniquely captures.

---

Dưới đây là hướng dẫn của mình với vai trò **supervisor SNA + AI/ML**, dựa trên:

- `MAPR2026_Implementation_Plan_updated`
- `MAPR2026_v3_team_parallel_coding_plan_updated`
- và `Paper guide` của supervisor khác

Mình sẽ đi theo tinh thần: **practical, defendable, nộp được MAPR**, không lan man.

---

# 1. Đánh giá tổng quan về 2 plan mới nhất

## Verdict
**Hai plan mới nhất đã đủ tốt để viết một paper defensible cho MAPR 2026**, với điều kiện:

1. **đóng băng scope ngay bây giờ**,
2. **không mở thêm operationalization mới ngoài A0 + HSCC**,
3. **baseline fairness dưới HSCC phải đầy đủ**,
4. **paper phải được viết như một comparative operationalization study**, không phải một paper “GNN luôn tốt hơn”.

## Paper mạnh nhất lúc này là gì?
Không phải:
- “GNN beats all baselines”
- cũng không phải “HSCC là true Twitch diffusion model”

Mà là:

> **“Giá trị của GNN surrogate phụ thuộc mạnh vào IC operationalization. Dưới A0, analytical baselines gần như trần. Dưới HSCC, graph message passing có thể thêm giá trị vượt quá các flat baselines.”**

Đây là một câu chuyện:
- **honest**
- **đủ mới cho MAPR**
- và **rất khó bị reviewer đánh gục** nếu viết đúng.

---

# 2. Paper này thực chất là paper gì?

Đây là điểm phải chốt trước khi viết.

## Identity của paper
**Paper này là một bài empirical comparative study về operationalization của influence và surrogate learning.**

### Nó KHÔNG phải:
- paper về “real influence”
- paper về “novel GNN architecture”
- paper về “new IC theory”
- paper về “influence maximization”

### Nó LÀ:
- paper về **simulation-defined influence approximation**
- paper về **regime-dependent value of GNNs**
- paper về **khi nào analytical baselines đủ, khi nào graph learning bắt đầu hữu ích**

---

# 3. Cấu trúc narrative nên giữ chặt

Mình đồng ý mạnh với hướng đi của supervisor guide, nhưng sẽ làm nó sắc hơn và practical hơn cho MAPR.

## Narrative trục chính
### Bước 1
**MC-IC là một operational metric hợp lý nhưng chỉ là proxy**

### Bước 2
**Binary top-k labels không ổn định → regression/ranking là formulation đúng**

### Bước 3
**Không có một IC operationalization nào “tự nhiên đúng” — operationalization choice quyết định learnability**

### Bước 4
**GNN không có giá trị cố hữu; giá trị của nó phụ thuộc regime của label**
- A0: structural ceiling, degree gần tối ưu
- HSCC: GNN có thể học thêm cross-community signal beyond flat baselines

---

# 4. Hướng dẫn viết paper theo từng section

Mình sẽ cho luôn hướng viết cụ thể, gần như có thể dùng để drafting.

---

## Section 1 — Introduction (0.5 trang)

## Mục tiêu
Trả lời 4 câu:
1. Bài toán là gì?
2. Vì sao khó?
3. Vì sao MC-IC cần surrogate?
4. Đóng góp cụ thể là gì?

## Cấu trúc đề xuất

### Paragraph 1 — Problem
- identify influential users / power users trong static social graph
- thiếu behavioral cascade logs
- nên phải operationalize influence gián tiếp

### Paragraph 2 — Tension
- MC-IC là proxy hợp lý nhưng đắt
- GNN là surrogate candidate
- nhưng chưa rõ khi nào GNN thực sự hơn các heuristic/baseline đơn giản

### Paragraph 3 — Core idea
- compare two IC operationalizations:
  - A0: structural weighted cascade
  - HSCC: source-velocity + cross-community amplification
- show that operationalization choice changes the learning problem

### Contributions — 3 bullet là đẹp nhất
Mình khuyên viết như sau:

1. **We analyze Monte Carlo IC as a simulation-defined operational metric for influence potential on Twitch, and show that binary top-k labels are structurally unstable while continuous regression targets remain usable.**
2. **We compare two diffusion operationalizations, A0 and HSCC, and show that they induce qualitatively different approximation regimes: degree-dominated versus graph-aware attribute-community structure.**
3. **We benchmark analytical, flat, and GNN surrogates under both regimes, showing that GNN value is regime-dependent rather than universal, while retaining orders-of-magnitude speedups over repeated MC simulation.**

## Cần tránh
- “we identify real power users”
- “we propose a superior GNN”
- “we discover the correct Twitch diffusion model”

---

## Section 2 — Background / Related Work (0.5–0.75 trang)

## Nên chia 3 cụm rất ngắn

### 2.1 Influence / IC / diffusion
- Kempe et al. (2003)
- weighted cascade setup
- DeepIM chỉ cite như example của learning-based IM dùng weighted cascade

### 2.2 Node importance / structural baselines
- degree / PageRank / k-shell
- Kitsak et al. (2010)
- optionally Lü et al. survey nếu còn reference budget

### 2.3 Graph surrogate / GNN
- GraphSAGE
- GCN
- GIN
- GAT
- APPNP
- surrogate / graph regression framing

## Important note
Không đi sâu influence maximization literature quá nhiều.  
Bài của bạn không optimize seed set; bài của bạn **approximate node-level IC scores**.

---

## Section 3 — MC-IC as Operational Metric (đây là section quan trọng nhất)  
**~1.0–1.25 trang**

Mình đồng ý với supervisor guide: **Section 3 phải được mở rộng hơn Section 4 so với bản plan cũ**, vì đây mới là contribution methodological mạnh nhất.

## 3.1 Construct validity
Dùng gần nguyên văn đoạn bạn đã chuẩn bị:
- follower graph ≠ observed diffusion
- nhưng là structural substrate hợp lý trong absence of logs
- all findings are about the operationalization, not real influence

## 3.2 Operationalizations
Giới thiệu rất rõ:

### A0
\[
p(u,v)=1/\deg(v)
\]
- standard structural operationalization
- degree-coupled by design

### HSCC
\[
p(u,v)=\mathrm{clip}\left(\lambda \frac{\phi(u)}{\deg(u)}(1+\gamma \mathbf{1}[c_u\neq c_v]),0,p_{\max}\right)
\]
- domain-informed alternative
- source engagement velocity + community bridging
- not “true Twitch diffusion”, only a comparative operationalization

## 3.3 Discriminativeness
Ở đây nên có **1 bảng ngắn**:

| Regime | Mean reach | Median | CV | Comment |
|---|---:|---:|---:|---|
| A0 | ... | ... | ... | heavy-tailed, structural |
| HSCC | ... | ... | ... | selective, non-degenerate |

### Key writing point
- A0: broad but degree-dominated
- HSCC: smaller mean reach nhưng vẫn discriminative
- I-A: chỉ mention 1 câu trong Discussion/Appendix như negative archive, đừng đưa vào main text dài

## 3.4 Stability and regression
Đây là phần bạn phải viết thật crisp:

- binary top-k unstable
- community boundary spanning
- gap-to-noise gần 0
- instability is structural, not just Monte Carlo noise
- therefore:
  - **continuous regression is the principled formulation**
  - binary labels are only secondary/provisional

### Câu rất nên có
> We treat regression not as a fallback but as the natural formulation for a simulation-derived continuous influence target.

## 3.5 Why not degree?
Đây là gap phải fill.

### Mình khuyên viết như sau:
- Under A0, degree is indeed a very strong approximation.
- But the degree-controlled (and ideally one-hop-controlled) variance analysis shows whether IC retains variance beyond local connectivity.
- Under HSCC, degree collapses as a predictor almost entirely.

=> như vậy bạn không cần overclaim rằng “IC always goes beyond degree”.  
Bạn viết đúng hơn:
> The extent to which IC goes beyond degree depends on the operationalization.

Đây là câu rất mạnh.

---

## Section 4 — Surrogate Learning Across Operationalizations  
**~2.0 trang**

Đây là section kết quả chính, nhưng phải viết gọn và có logic.

---

## 4.1 Setup
Ngắn gọn:
- Twitch MUSAE
- 168k nodes
- 5k labeled nodes
- transductive split
- 5 seeds
- metrics: Spearman, NDCG@10, P@10
- runtime reported separately

---

## 4.2 A0 results — “structural ceiling”
Đây phải là subsection riêng và **không được xem là thất bại**.

### Main message:
- degree / two-hop are already near-optimal
- GNNs converge toward that ceiling
- bootstrap comparator = degree

### Nếu CI cho practical equivalence:
Câu chuẩn:
> Under A0, the best GNN architecture is practically equivalent to degree centrality under the pre-registered equivalence bound, indicating that the limiting factor is the diffusion operationalization rather than the GNN architecture.

### Cần đưa:
- degree
- one-hop
- two-hop
- best flat baseline
- 4–5 GNN architectures (raw_attr)
- maybe best ablation result

---

## 4.3 HSCC results — “graph-aware regime”
Đây là subsection main-claim.

### Main message:
- degree không còn là comparator đúng
- strongest flat baseline mới là comparator thật
- GNN có thể thêm giá trị nếu học được phần graph/community structure beyond raw attrs

### Bắt buộc:
Bảng HSCC phải có:
- LR(life_time)
- LR(views + life_time)
- LR(degree + views + life_time)
- MLP(raw attrs)
- nếu GNN dùng language:
  - LR(... + language)
  - MLP(... + language)

### Nếu GNN thắng strongest flat baseline:
Câu chuẩn:
> Under HSCC, GNN message passing significantly improves over the strongest flat baseline, suggesting that neighborhood structure contributes information not recoverable from node-level attributes alone.

### Nếu GNN chỉ xấp xỉ / hơn rất ít:
Câu chuẩn:
> Under HSCC, flat attribute models explain most of the source-driven component, while GNN contributes a smaller but measurable improvement attributable to graph-mediated community structure.

### Nếu GNN thua strongest flat baseline:
Vẫn viết được:
> Under HSCC, most of the predictive signal is already captured by source-side engagement attributes, with only limited incremental benefit from graph message passing.

=> paper vẫn ổn nếu contrast A0 vs HSCC còn mạnh.

---

## 4.4 Contrast analysis
Đây là trái tim của paper.

### Cần giải thích rõ:
#### A0
- label is degree-coupled
- analytical baselines strong
- GNN near ceiling

#### HSCC
- label decomposes into:
  - source engagement term
  - graph/community amplification term
- flat baselines capture mostly source term
- GNN can only add value on the structural amplification part

### Đây là nơi dùng:
- `phi`
- `phi × (1+cross_frac)`
- nhưng chỉ như **ceiling / interpretation**, không phải main baseline

### Câu rất hay để dùng:
> The contrast between A0 and HSCC shows that surrogate learnability is not a property of the model alone; it is jointly determined by the diffusion operationalization and the information already recoverable by simple baselines.

---

## 4.5 Runtime
Giữ ngắn và sạch:

- MC-IC labeling cost
- GNN training cost
- GNN inference cost
- analytical baseline cost (near-zero inference)

### Important wording
- speedup **vs MC-IC**
- not vs degree

---

## Section 5 — Discussion & Limitations (0.5 trang)

Đây là nơi bạn “khóa” reviewer.

## 5.1 When does GNN help?
Câu trả lời:
- not universally
- only when target depends on graph-mediated information not already captured by strong flat baselines

## 5.2 Limitations
Tối thiểu nên có 4 ý:

1. follower graph ≠ observed diffusion path
2. A0 and HSCC are operationalizations, not ground truth
3. HSCC is novel and domain-informed, not empirically validated diffusion law
4. transductive evaluation only
5. small mean reach under HSCC should be interpreted as selective diffusion, not broad viral spread

## 5.3 Why not learn p from data?
Giữ câu này:
- no supervised diffusion logs
- hence zero-shot operationalizations

---

# 5. Figures và tables nên có gì

Do MAPR chỉ có 6 trang, đừng tham.

## Must-have figures
### Figure 1
Pipeline diagram:
- graph
- A0 / HSCC
- MC-IC labels
- regression target
- baselines + GNN surrogates

### Figure 2
**Two-panel results figure**
- trái: A0
- phải: HSCC
- bar chart / dot plot with CI
- line reference:
  - A0: degree
  - HSCC: strongest flat baseline

Đây là figure quan trọng nhất của paper.

## Must-have tables
### Table 1
Dataset + operationalizations
- nodes, edges
- A0 formula
- HSCC formula
- mean/median/CV

### Table 2
A0 results (main subset)
- degree, one-hop, two-hop, MLP, GNNs

### Table 3
HSCC results
- LR(life_time), LR(views+life_time), LR(deg+views+lt), MLP, GNNs

### Table 4 (nếu còn chỗ)
Runtime mini-table

Nếu thiếu chỗ:
- merge runtime vào main results table dưới dạng cột cuối

---

# 6. Những claim nào được phép và không được phép

## Được phép
- “MC-IC is a principled operational metric”
- “A0 and HSCC induce different approximation regimes”
- “binary labels are structurally unstable”
- “under A0, analytical baselines are near-optimal”
- “under HSCC, GNNs may add value beyond flat baselines”
- “surrogate value depends on operationalization”

## Không được phép
- “we identify real power users”
- “HSCC is the true Twitch diffusion model”
- “GNN always outperforms baselines”
- “MC-IC is ground truth”
- “practical equivalence” nếu chưa pre-register bound và chưa có CI phù hợp
- “GNN is feature-agnostic” theo nghĩa tuyệt đối

### Thay “feature-agnostic” bằng:
> **without precomputed structural summaries**

---

# 7. Kịch bản viết paper theo kết quả cuối

Đây là phần cực quan trọng.

---

## Case 1 — Tốt nhất
### A0: GNN ≈ degree  
### HSCC: GNN > strongest flat baseline

=> paper rất mạnh cho MAPR

### Abstract nên viết:
- A0 = structural ceiling
- HSCC = graph-aware regime
- GNN advantage is regime-dependent
- 7000x speedup over MC-IC

---

## Case 2 — A0: GNN ≈ degree  
### HSCC: GNN ≈ strongest flat baseline
=> paper vẫn publishable

### Framing:
- operationalization contrast is main contribution
- HSCC shows that source-side attributes dominate much of the signal
- graph message passing adds limited but interpretable value
- still useful as fast surrogate

---

## Case 3 — A0: GNN < degree  
### HSCC: GNN < strong flat baseline
=> vẫn cứu được paper nếu viết đúng

### Main contribution chuyển thành:
1. binary instability finding
2. operationalization contrast
3. analytical/flat baselines often suffice depending on the regime
4. GNN is not universally superior

Đây vẫn là một paper empirical negative-result tốt cho MAPR.

---

# 8. Những paper cần đọc và cite

Mình chia thành 3 nhóm:
- **must cite in main paper**
- **should read / cite if space allows**
- **read only, not necessarily cite**

---

## 8.1 Must cite (main paper)
### Influence / IC / operationalization
1. **Kempe, Kleinberg, Tardos (2003)**  
   *Maximizing the Spread of Influence through a Social Network*  
   → foundational IC model

2. **Ling et al. (2023), DeepIM**  
   *Deep Graph Representation Learning and Optimization for Influence Maximization*  
   → weighted cascade setup, learning-based IM context

3. **Rozemberczki et al. (2021)**  
   MUSAE / Twitch dataset paper  
   → dataset semantics

### GNN architectures
4. **Hamilton et al. (2017)** — GraphSAGE  
5. **Kipf & Welling (2017)** — GCN  
6. **Xu et al. (2019)** — GIN  
7. **Veličković et al. (2018)** — GAT  
8. **Klicpera et al. (2019)** — APPNP  

### Evaluation / construct validity / node importance
9. **Kitsak et al. (2010)**  
   → k-shell spreaders

10. **Guille et al. (2013)**  
   → evaluation without behavioral logs

11. **Aral & Walker (2012)**  
   → social ties and influence pathways

### Stats / community
12. **Benjamini & Hochberg (1995)**  
13. **Blondel et al. (2008)** — Louvain

### Nếu cần HSCC justification
14. **Burt (1992)** — Structural Holes

> Nếu reference budget thực sự phải ≤12, mình khuyên giữ:
- Kempe
- DeepIM
- Rozemberczki
- GraphSAGE
- GCN
- GIN
- GAT
- APPNP
- Kitsak
- Guille
- Burt
- Blondel  
và có thể bỏ Aral & Walker hoặc BH nếu space quá chặt (BH có thể chỉ mention methodologically).

---

## 8.2 Should read (có thể cite nếu cần mở rộng)
1. **Lü et al. (2016), Physics Reports**  
   survey về vital nodes  
2. **Chen, Wang & Wang (2010), KDD**  
   PMIA / local influence approximation  
3. **Grover & Leskovec (2016)**  
   Node2Vec  
4. **Brody et al. (2022)**  
   GATv2 — chỉ nếu GATv2 còn xuất hiện ở appendix/future work  
5. **Hu et al. (2019)**  
   GINE — chỉ nếu C5 được mention  
6. **Fosdick et al. (2018)**  
   configuration model/nulls, nếu bạn còn nói về nulls ở extended version

---

## 8.3 Read but probably not cite in MAPR version
- GCNII
- HGT
- GraphGPS
- fairness/per-group analysis papers
- strong journal-only methodological papers

Chúng tốt cho extended version, không cần nhồi vào MAPR 6 trang.

---

# 9. Hướng dẫn viết Abstract rất cụ thể

Mình khuyên abstract theo template này:

### Sentence 1
Problem + difficulty:
> Identifying influential users on static social networks without behavioral cascade logs requires simulation-based operationalizations of influence, but the learnability of such operationalizations remains poorly understood.

### Sentence 2
Method:
> We study two Monte Carlo Independent Cascade (MC-IC) operationalizations on the Twitch social network: a structural weighted-cascade regime (A0) and a domain-informed source-community regime (HSCC).

### Sentence 3
Stability/regression:
> We show that binary top-k influence labels are structurally unstable, motivating continuous regression on simulation-derived influence scores.

### Sentence 4
Main contrast:
> Under A0, analytical structural baselines are already near-optimal, whereas under HSCC the strongest baselines shift to flat source-attribute models.

### Sentence 5
GNN result:
> Across GraphSAGE, GCN, GIN, GAT, and APPNP, GNN surrogates provide regime-dependent value, ranging from practical equivalence to structural baselines under A0 to measurable gains over flat baselines under HSCC.

### Sentence 6
Runtime:
> In all cases, learned surrogates provide orders-of-magnitude faster inference than repeated MC simulation.

---

# 10. Những việc mình yêu cầu team làm ngay trước khi viết

## Must-do ngay hôm nay
### Person 1
- fix HSCC regression target file
- registry entry for HSCC
- freeze config

### Person 2
- verify community feature coverage
- ensure diffusion proxies file clean
- provide quick note on language-community alignment if already available

### Person 3
- implement **all HSCC fairness baselines**
- run regime-tagged output tables
- ensure bootstrap comparators are regime-specific

---

# 11. Kết luận cuối của supervisor

## Supervisor verdict
**Hai plan update mới nhất đã đủ tốt để viết một paper defensible cho MAPR, nếu bạn giữ đúng đường `A0 + HSCC` và không mở thêm scope.**

### Điểm mạnh nhất của paper:
- stability finding
- regime contrast
- honest demonstration rằng GNN value is conditional, not universal
- runtime story sạch

### Điểm dễ bị reviewer tấn công nhất:
- HSCC fairness baselines chưa đủ
- HSCC có nguy cơ bị xem là engineered nếu không frame đúng
- claim về GNN phải map đúng với bootstrap của từng regime

### Nếu sửa 3 điểm này tốt:
1. baseline fairness for HSCC
2. regime-specific comparator logic
3. tight writing with no extra branches

=> **paper hoàn toàn có thể defend được ở MAPR**.
