# When Does Graph Learning Add Value Beyond Strong Baselines?
## A Comparative Study of IC Operationalizations for Influence Approximation

*Anonymous Submission*

---

## Abstract

Approximating node influence in static social networks without observed cascade logs requires defining an operational target rather than inferring influence directly from behavior. We study two Monte Carlo Independent Cascade (MC-IC) operationalizations on the Twitch Gamers network [12]: a structural weighted-cascade regime and a community-conditioned source regime. Under the structural operationalization, binary top-k labels are unstable, and the best graph surrogate remains below degree centrality (GCN ρ = 0.808 vs. degree ρ = 0.826; Δρ = −0.018, 95% CI [−0.029, −0.008]). Under the source-community operationalization, degree collapses (ρ = −0.006), and GraphSAGE with raw attributes significantly outperforms the pre-specified matched flat comparator, LR(degree, views, life_time, language), with ρ = 0.915 versus 0.884 (Δρ = +0.033, 95% CI [+0.021, +0.044]). Once trained, the source-community GraphSAGE surrogate performs a single post-training forward pass over the full 168K-node graph in about 0.086 seconds. For scale, a recorded structural MC-IC labeling pass for 5,000 labeled nodes with 200 runs takes 480.3 seconds, while source-community label generation takes approximately 1956 seconds. Taken together, these results show that the value of graph learning for influence approximation is operationalization-dependent: it emerges when the target preserves neighborhood-structured signal beyond what strong structural and tabular baselines already encode.

**Index Terms** — influence approximation, social networks, graph neural networks, baseline comparison, diffusion modeling, surrogate learning

---

## I. Introduction

Many applications on large social graphs require ranking nodes by potential influence even when no reliable cascade logs are available. In such settings, influence cannot be observed directly and must instead be approximated through an operational proxy. Monte Carlo diffusion simulation is a natural option, but repeated simulation is expensive and its outputs depend strongly on how the influence process is defined [1], [2]. This creates a dual problem: one must learn an efficient surrogate while also being explicit about what notion of influence that surrogate is meant to approximate.

Graph neural networks are a plausible solution because influence is inherently relational, yet their practical value is often stated too broadly [5], [6]. A strong graph model may look impressive when compared with weak baselines, but that comparison does not answer the more relevant methodological question: does graph message passing add information beyond what strong structural or tabular baselines already capture? For influence approximation, the answer depends not only on model architecture but also on the target operationalization used to generate the labels.

This paper studies that dependency on a single social graph under two Monte Carlo Independent Cascade operationalizations. The first is a structural weighted-cascade regime, which serves as a degree-coupled structural baseline. The second is a community-conditioned source regime, which injects source- and community-dependent structure into the propagation process. Holding the dataset, prediction task, and evaluation protocol fixed while changing the target operationalization allows us to ask a sharper question: under which operationalizations does graph-aware learning provide measurable value beyond strong non-graph comparators?

Our results show that the answer is operationalization-dependent rather than universal. Under the structural weighted-cascade regime, binary top-k labels are too unstable to serve as the primary target, and the best GNN remains statistically below degree centrality. Under the community-conditioned source regime, degree collapses as a useful proxy, and GraphSAGE significantly outperforms the pre-specified matched flat comparator. The same study therefore yields both a negative and a positive result for graph learning, depending on what kind of influence signal the operationalization makes available.

The paper makes three contributions. First, it compares two influence operationalizations that induce qualitatively different predictive settings on the same social network. Second, it shows that graph learning does not uniformly dominate strong baselines: under a degree-coupled target it does not win, whereas under a community-conditioned target it provides a reliable gain. Third, it extracts a practical methodological lesson for influence approximation: before investing in graph-based surrogates, one should first ask whether the chosen operationalization contains graph-specific signal beyond what strong baseline features already encode.

---

## II. Background and Scope

### A. Influence approximation without cascade logs

Influence estimation is often framed as a diffusion problem, but many real social-network datasets do not provide trustworthy cascade logs at the granularity needed to supervise a ranking model directly. In those settings, a common alternative is to simulate a plausible propagation process and treat the resulting scores as a proxy target for learning [1]. This move is practical, but it changes the epistemic status of the problem: the model is no longer learning "true" influence from observed behavior, but rather approximating a chosen operationalization of influence.

That distinction matters for both writing and evaluation. Throughout this paper, we therefore refer to the targets as operationalizations of influence rather than as direct measurements of real-world causal impact. This wording is deliberately narrower and helps keep the claims aligned with the evidence actually provided by the experiments. It also distinguishes our setting from social influence prediction work that is trained on observed behavioral adoption or exposure data rather than on simulation-derived labels [7].

### B. Strong baselines versus graph-aware surrogates

The methodological challenge is not simply to design a strong predictor, but to determine whether graph-aware learning contributes anything beyond strong non-graph baselines. Structural heuristics such as degree centrality can already be highly competitive when the target is closely aligned with local connectivity [2], [3]. Likewise, flat models built from node attributes and handcrafted features can absorb substantial signal without explicit message passing. As a result, a meaningful graph-learning comparison must be made against competent analytical and tabular baselines rather than against weak strawman models.

This framing also clarifies the contribution of the paper. We are not claiming that graph neural networks dominate classical methods in general. Instead, we ask when neighborhood aggregation adds non-redundant information once strong structural and flat alternatives have already been given a fair chance. This question is adjacent to, but distinct from, the broader influence-maximization literature and recent learning-based approaches that use graph representations to predict spreading ability or select seeds [13], [14].

### C. Scope guard

The scope of the paper is intentionally narrow. We do not claim access to true diffusion ground truth, and we do not treat the simulated targets as observational evidence of real-world causal influence. We also do not claim that graph learning is universally superior across datasets, tasks, or diffusion assumptions. The study is confined to one social graph and two carefully defined IC operationalizations, chosen to test whether the apparent value of graph-aware learning changes when the target construction changes. This narrower scope is a feature rather than a weakness: it lets the paper make a conditional claim that is easier to defend than a broad technology-level claim.

---

## III. Influence Operationalizations and Evaluation Protocol

### A. Operationalization 1: Structural weighted-cascade regime

Our first operationalization is a structural weighted-cascade regime, in which transmission follows a weighted-cascade rule whose signal is strongly coupled to local degree structure. In this structural specification, the edge-level transmission probability for an attempted activation from node u to node v is

$$p(u, v) = \frac{1}{\deg(v)}, \tag{1}$$

following the weighted-cascade parameterization of Kempe et al. [1]. This operationalization is useful as a controlled structural baseline because it defines influence without any domain-specific source mechanism, but that same simplicity also makes it prone to producing labels that track degree too closely for graph message passing to add much new information.

Under the structural weighted-cascade operationalization, formal label-stability diagnostics show that binary top-k labels are not stable enough to serve as the primary learning target. Across independent MC-IC campaigns, the mean top-decile Jaccard is approximately 0.31, far below the pre-specified stability target of 0.85. Additional diagnostics show heavy-tailed but non-degenerate score variation within degree bands: mean reach ranges from 2.5 nodes in the lowest degree band to 109.3 nodes in the highest, with intra-band coefficients of variation between 1.12 and 2.29. We therefore treat this structural operationalization primarily as a regression problem and predict a continuous simulation-derived score rather than a brittle binary label.

This scope boundary matters for the rest of the paper. All evidence based on threshold sweeps, Jaccard behavior, uncertainty boundaries, or binary-label fragility is used only to justify the target choice in the structural weighted-cascade regime. We do not transfer those diagnostics mechanically to the second regime.

### B. Operationalization 2: Source-community regime

Our source-community operationalization is a community-conditioned source regime that modifies the influence process so that source-side engagement and community-conditioned propagation matter directly. In this operationalization, the transmission rule is

$$p(u, v) = \text{clip}_\lambda\!\left(\frac{\phi(u)}{\deg(u)}(1 + \gamma\,\mathbf{1}[c_u \neq c_v]),\ 0,\ p_{\max}\right), \tag{2}$$

with

$$\phi(u) = \frac{\text{rank}(\log(1 + \text{views}_u)/(1 + \text{life\_time}_u))}{N}. \tag{3}$$

Here $c_u$ and $c_v$ denote community assignments, and the fixed configuration uses λ = 1.0, γ = 1.0, and p_max = 1.0.

The rank-normalized source term is intended to approximate engagement velocity rather than cumulative popularity: log(1 + views) compresses heavy-tailed outliers, dividing by 1 + life_time avoids rewarding longevity alone, and rank normalization improves robustness. Community structure is derived using Louvain through a 10-run seed sweep at resolution 1.1, after which the best-modularity partition is fixed for label generation rather than tuned jointly with the predictive models [10].

This distinction is empirically visible. Under the structural operationalization, degree centrality is a strong approximation target, whereas under the community-conditioned source regime its Spearman correlation with the target drops to ρ = −0.006. This target also remains a non-degenerate continuous one despite its smaller absolute cascades: summary statistics give mean reach 4.83, maximum reach 16.31, standard deviation 2.82, and coefficient of variation 0.583. We use that collapse as evidence that the source-community construction is not just a mild parameter variation of the structural one, but a qualitatively different predictive setting in which strong flat models remain necessary while degree alone is no longer an adequate benchmark.

The claim boundary is therefore different here. We motivate this regime by its construction and by the observed failure of simple structural ranking, not by reusing the binary-stability evidence from the structural weighted-cascade regime. It is introduced as a domain-informed comparative operationalization rather than as a validated generative law of Twitch diffusion. This separation keeps the interpretation of both operationalizations defensible.

### C. Prediction targets and train/test protocol

For both operationalizations, the main prediction target is a continuous transformation of the simulation output, specifically $y_u = \log(1 + \bar{s}_u)$ where $\bar{s}_u$ is the mean Monte Carlo influence score for node u. This is the natural formulation for a simulation-derived continuous target and preserves rank information while avoiding the threshold sensitivity of binary top-k labels. For inferential comparisons, we use 1,000 paired bootstrap resamples of the 1,000 held-out test nodes, sampling node indices with replacement. Within each resample, the same sampled indices are applied to the target vector and to both model-prediction vectors; Spearman correlation is recomputed for each model and differenced as Δρ = ρ_GNN − ρ_comparator. We report the 2.5th–97.5th percentile interval of this paired bootstrap distribution and use a pre-specified practical-equivalence bound of δ₀ = 0.02 for Spearman ρ [11].

The experiments use the Twitch Gamers graph with 168,114 nodes and 6,797,557 edges. The graph is undirected because edges represent mutual follower relationships, not observed diffusion pathways. We label 5,000 nodes via degree-quintile-stratified sampling, freeze an 80/20 train/test split with seed 42, run 200 MC-IC simulations per labeled node for each operationalization, and evaluate on the fixed held-out set of 1,000 labeled test nodes; the remaining 4,000 labeled nodes are used for model development in the transductive setting, with a validation split carved from the training pool. This design balances simulation cost against statistical power while avoiding a labeled subset concentrated only in the highest- or lowest-degree part of the graph.

IC activation attempts are therefore simulated over an undirected structural substrate rather than over observed Twitch cascades, and the resulting targets should not be interpreted as direct measurements of real information diffusion. Community assignments are used only inside the source-community label-generation process and are not provided as input features to either flat baselines or GNNs in the main comparison; comparator inputs are restricted to the pre-specified structural and attribute features available under each model family, with language included where applicable.

All reported ranking metrics are computed on held-out labeled nodes only. The graph is treated transductively, but evaluation is restricted to the test mask within the labeled subset rather than to the full graph. Our primary metric is Spearman rank correlation, which best matches the paper's emphasis on ranking quality. We report NDCG@10% and Precision@10% as secondary metrics to capture behavior near the top of the ranking, with k defined as ⌈0.10 × n_test⌉.

### D. Model families and comparator policy

We compare four model families: analytical structural baselines such as degree centrality; flat baselines built from node attributes and selected handcrafted features; shallow embedding baselines such as Node2Vec [4]; and graph-aware surrogate models based on message passing. This comparator structure is important because the paper's claim is not that GNNs outperform weak baselines, but that they may add value beyond already-competitive non-graph alternatives. Where the relevant comparison results are available, we report matched flat and shallow-embedding baselines directly in the regime-specific results tables; otherwise, we restrict the prose claim to the pre-specified comparator.

Comparator policy is fixed before interpretation. In the structural weighted-cascade regime, degree centrality is the primary reference baseline because the regime is explicitly degree-coupled. In the community-conditioned source regime, the designated comparator is the pre-specified flat model LR(degree, views, life_time, language). A second flat model using views, life_time, and language alone is effectively tied to the fourth decimal place (0.88442 versus 0.88430), so we retain the former only because it was pre-specified in the bootstrap comparison.

The active graph-model comparison includes GraphSAGE, GCN, GIN, and APPNP [5], [6], [8], [9]. Models with cross-seed Spearman standard deviation above 0.1 are retained descriptively but excluded from best-architecture claims.

---

## IV. Results

### A. Results under the structural weighted-cascade regime

Under the structural weighted-cascade regime, the best graph-based surrogate does not surpass the strongest structural baseline. Among the active GNN architectures, the best result comes from GCN with raw attributes (ρ = 0.808), but this remains below degree centrality (ρ = 0.826). Bootstrap comparison against the locked structural comparator yields Δρ = −0.018 with a 95% confidence interval of [−0.029, −0.008], placing the interval fully below zero and outside the pre-registered equivalence window δ₀ = 0.02. Under that pre-specified decision rule, the result is correctly interpreted as statistically worse rather than practically equivalent.

This pattern is most naturally read as a structural ceiling: when the operationalization is degree-coupled, a strong analytical baseline already captures most of the available ranking signal.

**Table II: Structural Weighted-Cascade Surrogate Results on 1,000 Held-Out Test Nodes. Degree Centrality is the Primary Comparator. Stochastic Learned Rows Report Mean±Std over 5 Seeds.**

| Model | Type | ρ | NDCG@10% | P@10% |
|---|---|---|---|---|
| Degree | Analytical | 0.826 | 0.881 | 0.60 |
| PageRank | Analytical | 0.824 | 0.857 | 0.56 |
| k-shell | Analytical | 0.816 | 0.687 | 0.50 |
| Two-hop | Analytical | 0.804 | 0.848 | 0.55 |
| Node2Vec + LR | Shallow | 0.810 ± 0.005 | 0.859 | 0.58 |
| GCN (raw attr.) | GNN | 0.808 ± 0.001 | 0.825 | 0.53 |
| GIN (raw attr.) | GNN | 0.615 ± 0.022 | 0.731 | 0.39 |
| GraphSAGE (raw attr.) | GNN | 0.534 ± 0.009 | 0.674 | 0.45 |
| APPNP (raw attr.)† | GNN | 0.585 ± 0.417 | 0.724 | 0.49 |

*†APPNP is retained for completeness but excluded from best-architecture claims because its cross-seed variance exceeds the stability threshold.*

### B. Results under the community-conditioned source regime

The conclusion changes under the community-conditioned source regime. Here the best standard graph-aware surrogate is GraphSAGE with raw attributes, which reaches ρ = 0.915 on the held-out test nodes. The designated matched flat comparator, LR(degree, views, life_time, language), reaches ρ = 0.884, and the pre-specified bootstrap comparison gives Δρ = +0.033 with a 95% confidence interval of [+0.021, +0.044]. This interval lies entirely above zero and above the pre-specified equivalence bound, supporting the claim that graph-aware learning adds measurable value under this source-community operationalization.

Two disclosures matter for interpretability. First, degree centrality drops to ρ = −0.006 under this operationalization, so it is retained as contextual evidence of target change rather than as the main comparator. Second, the flat model using views, life_time, and language alone is effectively tied with the designated comparator to the fourth decimal place; we report LR(degree, views, life_time, language) because it was the pre-specified comparator in the bootstrap comparison, not because the point estimate gap is practically meaningful.

**Table III: Source-Community Surrogate Results on 1,000 Held-Out Test Nodes. Degree is Contextual Only; the Pre-Specified Comparator for Inferential Claims is LR(degree, views, life_time, language). Stochastic Learned Rows Report Mean±Std over 5 Seeds.**

| Model | Type | ρ | NDCG@10% | P@10% |
|---|---|---|---|---|
| Degree | Analytical | −0.006 | 0.465 | 0.04 |
| LR(life_time) | Flat | 0.790 | 0.827 | 0.46 |
| LR(views+life_time) | Flat | 0.868 | 0.800 | 0.41 |
| LR(deg+views+life_time+lang) | Flat | 0.884 | 0.829 | 0.45 |
| LR(views+life_time+lang) | Flat | 0.884 | 0.830 | 0.45 |
| GraphSAGE (raw attr.) | GNN | 0.915 ± 0.004 | 0.902 | 0.58 |
| GraphSAGE + rankloss | GNN | 0.924 ± 0.002 | 0.911 | 0.60 |
| GCN (raw attr.) | GNN | 0.602 ± 0.014 | 0.694 | 0.21 |
| GIN (raw attr.) | GNN | 0.028 ± 0.046 | 0.469 | 0.04 |
| APPNP (raw attr.)† | GNN | −0.037 ± 0.146 | 0.491 | 0.03 |

*Comparator note: LR(views+life_time+lang) is effectively tied at the fourth decimal place (0.88442 vs. 0.88430); LR(deg+views+life_time+lang) remains the designated comparator because it was pre-specified in the bootstrap comparison. †APPNP is retained for completeness but excluded from best-architecture selection due to unstable cross-seed variance.*

As a secondary result, the ranking-aware GraphSAGE variant reaches ρ = 0.924 and improves over the same pre-specified comparator by Δρ = +0.041 with a 95% confidence interval of [+0.030, +0.053]. However, the +0.009 gap between the ranking-aware and standard GraphSAGE variants is descriptive only, since the available comparison results do not provide a paired significance test for that within-GNN comparison.

### C. Cross-operationalization contrast

Taken together, the two operationalizations show that the value of graph message passing is strictly target-dependent. The dataset is the same, the evaluation protocol is the same, and the broad prediction task is the same, yet the conclusion reverses once the target operationalization changes. Under the structural weighted-cascade regime, the target is sufficiently aligned with degree that a strong analytical baseline sets the ceiling. Under the community-conditioned source regime, the target encodes source- and neighborhood-dependent structure that is not fully recoverable from node-level attributes alone, and graph-aware learning becomes useful.

This contrast is the paper's main methodological point. It suggests that debates about whether GNNs "work" for influence approximation are under-specified unless they also state which target operationalization is being learned. The more defensible conclusion is therefore conditional rather than universal: graph learning helps when the target contains graph-specific signal beyond what strong structural and flat baselines already encode.

### D. Runtime and practical value

The runtime results support the practical motivation for surrogate learning without replacing the need for predictive validity. Runtime comparisons therefore separate label generation, surrogate training, and post-training inference. A recorded structural MC-IC labeling pass over 5,000 labeled nodes with 200 runs takes approximately 480.3 seconds and serves as an empirical anchor for simulation-based labeling cost. Source-community label generation is more expensive, taking approximately 1956.1 seconds in the corresponding diagnostics. By contrast, the trained GraphSAGE surrogate in the community-conditioned source regime requires about 27.3 seconds of one-time training and then performs a single post-training forward pass over the full 168K-node graph in about 0.08596 seconds; all reported ranking metrics remain computed only on the held-out labeled test nodes.

**Table IV: Runtime Summary from Recorded Measurements. Ratios are Inference-Only Comparisons Against the Recorded 480.3 s Structural MC-IC Labeling Anchor, Not the Source-Community Label-Generation Wall-Clock.**

| Component | Measured time | Training | Anchor ratio |
|---|---|---|---|
| Structural MC-IC labeling anchor | 480.3 s | — | 1× |
| Degree baseline | 0.004 s | — | ~120,000× |
| LR flat baseline | <0.003 s | <0.004 s | ~160,000× |
| Node2Vec + LR baseline | 0.040 s | ~153 s | — |
| Source-community GraphSAGE | 0.086 s | ~27 s | ~5,590× |
| Source-community GCN | 0.165 s | ~59 s | ~2,900× |
| Source-community GIN | 0.086 s | ~28 s | ~5,600× |
| Source-community APPNP | 0.790 s | ~286 s | ~610× |

*Runtime notes: Measured time denotes the recorded deployment-style pass: labeling time for the structural MC-IC anchor and full-graph forward-pass time for trained surrogates and baselines. Anchor ratio is defined only against the recorded 480.3 s structural MC-IC labeling anchor. Main-text runtime values are rounded for readability, while exact frozen-artifact values are retained here for reproducibility. The source-community GraphSAGE row gives a single post-training forward pass over the full 168K-node graph in about 0.08596 s, with one-time training of about 27.3 s; the corresponding source-community label-generation wall-clock is approximately 1956.1 s, implying an inference-only speedup of roughly 22,700×. Ranking metrics remain computed only on held-out labeled test nodes. Node2Vec training bundles embedding precomputation and LR fit.*

This runtime gap matters because it turns an expensive simulation-derived target into a fast reusable approximation once training is complete. The reported ~5,500× speedup is an inference-only comparison against the recorded structural MC-IC labeling anchor; using the same-regime source-community label-generation cost yields an inference-only speedup of roughly 22,700×. At the same time, the paper should avoid presenting speed alone as the contribution. The runtime story is meaningful only because the surrogate is also accurate under the operationalization in which graph-aware learning is substantively justified.

---

## V. Discussion

### A. When does graph learning help?

The two operationalizations suggest a simple methodological principle: graph learning helps when the target contains relational signal not already captured by strong degree-based or attribute-based baselines. In the structural weighted-cascade regime, the target is tightly coupled to degree structure; in the source-community operationalization, neighborhood aggregation becomes useful because the target depends on source-side and community-conditioned interactions. The lesson is therefore conditional, not universal: whether GNNs help depends on the influence target being operationalized.

### B. Limitations

Several limitations are worth stating plainly. First, the study uses a single dataset, so the conclusions should be read as evidence of operationalization dependence rather than as a universal law of social-network influence modeling. Second, the targets are simulation-derived rather than based on observed cascades, which means the paper evaluates surrogate fidelity to a target operationalization, not fidelity to ground-truth behavioral spread. Third, evaluation is performed on a labeled subset rather than on exhaustively simulated scores for all nodes, reflecting a realistic compute budget but still limiting the scope of inference. In addition, the paired bootstrap intervals quantify node-level evaluation uncertainty only on the fixed held-out test set; they do not resample MC-IC simulation runs, graph structure, train/test splits, or potential dependencies among neighboring nodes. Fourth, the source-community operationalization depends on a single fixed Louvain partition selected from a 10-run sweep at resolution 1.1; different resolution parameters could change community boundaries and therefore the community-conditioned transmission term, and we do not claim that this partition is globally optimal. Fifth, we do not claim that no engineered graph-augmented flat model could close part of the HSCC gap; our inferential claim is limited to the pre-specified flat comparator and the evaluated shallow/GNN surrogates. Finally, GAT-style models were omitted because of memory limits on this dense graph. We therefore do not claim real-world causal influence, automatic transfer of the structural instability diagnosis to the community-conditioned source regime, or universal GNN superiority.

---

## VI. Conclusion

This paper examined when graph-aware surrogate learning adds value beyond strong baselines for influence approximation on a static social network. By comparing two target operationalizations on the same graph, we found that the answer is not universal. Under a degree-coupled structural operationalization, the best GNN remained below degree centrality; under a source-community operationalization, GraphSAGE delivered a reliable gain over the pre-specified matched flat comparator while preserving a substantial runtime advantage over repeated simulation.

The broader implication is methodological. Claims about the usefulness of graph learning for influence approximation should be conditioned on the target construction being learned, rather than stated as architecture-level generalities. Before investing in a graph-based surrogate, one should first ask whether the chosen operationalization contains graph-specific signal beyond what strong structural and tabular baselines already encode.

---

## References

[1] D. Kempe, J. Kleinberg, and E. Tardos, "Maximizing the spread of influence through a social network," in *Proc. 9th ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining (KDD)*, 2003, pp. 137–146.

[2] S. P. Borgatti, "Centrality and network flow," *Social Networks*, vol. 27, no. 1, pp. 55–71, 2005.

[3] L. C. Freeman, "Centrality in social networks: Conceptual clarification," *Social Networks*, vol. 1, no. 3, pp. 215–239, 1978.

[4] A. Grover and J. Leskovec, "node2vec: Scalable feature learning for networks," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining (KDD)*, 2016, pp. 855–864.

[5] T. N. Kipf and M. Welling, "Semi-supervised classification with graph convolutional networks," in *Proc. Int. Conf. Learn. Representations (ICLR)*, 2017.

[6] W. L. Hamilton, R. Ying, and J. Leskovec, "Inductive representation learning on large graphs," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2017, pp. 1024–1034.

[7] J. Qiu, J. Tang, H. Ma, Y. Dong, K. Wang, and J. Tang, "DeepInf: Social influence prediction with deep learning," in *Proc. 24th ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining (KDD)*, 2018, pp. 2110–2119.

[8] K. Xu, W. Hu, J. Leskovec, and S. Jegelka, "How powerful are graph neural networks?," in *Proc. Int. Conf. Learn. Representations (ICLR)*, 2019.

[9] J. Klicpera, A. Bojchevski, and S. Günnemann, "Predict then propagate: Graph neural networks meet personalized PageRank," in *Proc. Int. Conf. Learn. Representations (ICLR)*, 2019.

[10] V. D. Blondel, J.-L. Guillaume, R. Lambiotte, and E. Lefebvre, "Fast unfolding of communities in large networks," *J. Stat. Mech. Theory Exp.*, vol. 2008, no. 10, P10008, 2008.

[11] D. Lakens, "Equivalence tests: A practical primer for t tests, correlations, and meta-analyses," *Social Psychological and Personality Science*, vol. 8, no. 4, pp. 355–362, 2017.

[12] R. Sarkar and B. Rózemberczki, "Twitch Gamers: A dataset for evaluating proximity preserving and structural role-based node embeddings," in *Proc. Workshop Graph Learn. Benchmarks @ The Web Conf. 2021*, 2021.

[13] A. Zareie and R. Sakellariou, "Influence maximization in social networks: A survey of behaviour-aware methods," *Social Network Analysis and Mining*, vol. 13, no. 1, Art. no. 78, 2023.

[14] B. S. Panda and D. K. Tosh, "Influence maximization in social networks using graph embedding and graph neural network," *Information Sciences*, vol. 607, pp. 1617–1636, 2022.
