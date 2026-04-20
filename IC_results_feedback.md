

# Expert Analysis: GNN vs Baselines — Current Situation and Path Forward

---

## Addendum (2026-04-20): HSCC-refined Fix Status + Verified Outputs

This addendum records implementation updates and verified metrics after the latest audit.

### A. Fix status for previously flagged issues

1. **Missing HSCC regression target artifact**: **FIXED**
  - File created: `data/processed/regression_targets_hscc_refined.parquet`
  - Contract verified: columns `node_id, y`, with `y = log1p(ic_score_mean)`.

2. **Feedback/process traceability (registry gap)**: **FIXED**
  - Added registry entry in `docs/experiment_registry.md`:
    - `[2026-04-20 21:56] - HSCC-refined formula lock + regression target artifact backfill`

3. **HSCC baseline run was sanity-only**: **FIXED**
  - Official HSCC baseline run completed to:
    - `outputs/mapr2026_v3_results/baseline_ranking_metrics_hscc.csv`

4. **Surrogate `raw_attr` missing language feature**: **FIXED**
  - `run_surrogates.py` updated so `raw_attr` includes language-derived features (one-hot dummies).
  - Quick check on labeled scope shows `raw_attr` now has 24 features (`x_shape=(5000, 24)`).

### B. Official HSCC baseline results (from `baseline_ranking_metrics_hscc.csv`)

Key rows (test-split metrics):

- `degree`: Spearman = **-0.0064**
- `views`: Spearman = **0.1467**
- `life_time`: Spearman = **-0.7900**
- `lr_life_time`: Spearman = **0.7900**
- `lr_views_life_time`: Spearman = **0.8678**
- `lr_degree_views_life_time`: Spearman = **0.8679**
- `phi`: Spearman = **0.8779**
- `lr_phi`: Spearman = **0.8779**
- `mlp_raw_attr`: Spearman = **0.8640 ± 0.0018** (5 seeds)

Interpretation update:
- HSCC still solves the **degree-dominance** problem strongly.
- However, **life_time-aware linear baselines are very strong** and must remain in the main baseline table.

### C. Environment limitation noted during official run

- Node2Vec baseline was skipped because `torch-cluster`/`pyg-lib` is unavailable in the current environment.
- The run completed successfully with all other baselines.

### D. Minor numeric correction from previous draft

- `rho(HSCC, I-A)` verified at approximately **0.0323** (not ~0.017).

---

---

## 1. Honest Assessment of Where Things Stand

Let me be direct about the situation as revealed by the data.

Under A0 (the current primary IC formulation), GNN cannot beat degree. This is not a bug or an implementation problem. It is a mathematical consequence: `p(u,v) = 1/deg(v)` makes IC scores a function of local degree structure, and degree itself is the most efficient summary of that structure. The regression analysis confirms this definitively — `R²(IC_A0 ~ degree) = 0.887`, meaning degree explains 89% of IC variance. Adding views, life_time, or any node attribute to the regression adds less than 0.1% explained variance. There is simply no signal in the labels that GNN can access but degree cannot.

The I-A experiment confirmed the opposite extreme: row-normalization kills all variance, making labels essentially unpredictable by any method. CV = 0.239 on the full run, with all degree quintiles showing nearly identical IC scores (mean range 27.8–28.4).

These are not failures of the research plan. They are genuine empirical findings about the relationship between IC formulation, graph topology, and learnability. The question now is how to move forward productively.

---

## 2. The Real Risk: A Baseline Stronger Than Degree

The question about "what if a baseline beats degree and also beats GNN" is important and the answer is: **this has already happened in your data**.

Look at the existing results carefully:

```
degree:          0.826
two_hop_spread:  0.804
one_hop_spread:  0.688 (test split)
pagerank:        0.824
kshell:          0.816
```

Two-hop spread (0.804) is already close to degree (0.826). These are analytical proxies computable in O(E) without any learning. Under A0, any formula that better approximates the multi-hop cascade dynamics — including a three-hop analytical proxy — could potentially match or exceed degree. If someone computes `three_hop_expected_spread` and it achieves 0.83+, then every GNN variant is beaten by a formula that takes seconds to compute.

Under the proposed source-driven formulas (SRC/HSCC), the risk shifts. The one-hop analytical proxy for SRC is `rank(views_u)/C` — literally just a rank transformation of views. If someone computes `rank(views)` as a baseline (which is already in Group 1 as `views_rank`), and SRC IC scores correlate perfectly with views rank (which they will, since `E[one_hop_SRC(u)] = rank(views_u)/C`), then `views_rank` baseline achieves rho ≈ 1.0 with SRC-IC labels. GNN would need to beat views_rank, not degree. And views_rank is a zero-cost baseline.

This is the fundamental trap of source-driven formulas: **you engineer labels that GNN can predict well, but in doing so you also engineer labels that a trivial baseline can predict perfectly**.

---

## 3. Why the Proposed Formulas Are Problematic

### SRC (Source Rank Cascade)

The one-hop expectation is:

```
E[one_hop(u)] = rank(views_u) / (N × C_budget)
```

This is a deterministic function of views_u alone. Multi-hop dynamics add noise around this expectation but on a dense graph with mean degree 82, the law of large numbers ensures that the actual IC score converges quickly to the one-hop expectation. The result: `rho(IC_SRC, views_rank)` will be approximately 0.95–1.0.

GNN with views as input feature will learn `IC ≈ f(views)` and achieve rho ~0.90. But `views_rank` baseline achieves rho ~0.98 without any learning. GNN loses to a trivial baseline.

### HSCC (Hybrid Source-Community Cascade)

The community boost `(1 + γ × I[comm(u) ≠ comm(v)])` introduces genuine structural signal that views_rank alone cannot capture. However, two problems emerge.

First, the analytical proxy `views_rank × (1 + γ × cross_community_fraction)` is computable in O(E) — just multiply two precomputed vectors. If this proxy achieves rho > 0.85 with HSCC-IC labels, GNN again loses to an analytical formula.

Second, γ = 3.0 means cross-community edges have 4× the propagation probability of within-community edges. This is a very strong assumption with no empirical grounding in the Twitch dataset. A reviewer will ask why γ = 3 rather than γ = 1 or γ = 10, and the honest answer is "because it makes GNN win." This is exactly the kind of parameter tuning that undermines scientific credibility.

### The General Problem

Any IC formula where the one-hop expectation can be written as a simple function of node-level features will produce labels that a feature-based baseline can predict. The formula needs to create labels that depend on **multi-hop neighborhood composition in a way that cannot be summarized by node-level statistics**. This is extremely difficult to achieve on a dense graph where local averaging rapidly converges to global means.

---

## 4. What Would Actually Work — And Why It Might Not Be Worth Doing

The theoretical framework identifying four conditions (label information, mixing property, source dominance, GNN computability) is correct. But the practical implication is uncomfortable: to satisfy all four conditions simultaneously on Twitch, you need a formula where IC scores depend on the specific **pattern** of attributes in the 2-hop neighborhood, not just the sum or average.

An example that would genuinely require GNN:

```
p(u,v) = clip(similarity(feature_vector_u, feature_vector_v) / deg(v), p_max)
```

where `similarity` is cosine similarity between multi-dimensional feature vectors. This creates IC labels that depend on the **composition** of the neighborhood (which neighbors are similar to the source), which is exactly what GNN message passing computes and degree/views cannot.

The problem: this formula has no principled justification for influence modeling. It is purely engineered to make GNN necessary. A reviewer will see through this immediately.

---

## 5. The Honest Path Forward

Given 13 days to deadline and the structural constraints revealed by the data, I see three viable paths. They are ordered by intellectual honesty, not by "GNN wins" likelihood.

### Path A: Accept Equivalence Under A0, Lead With Stability Finding

This is the most defensible paper. The contribution is:

**Section 3 (primary contribution):** MC-IC as operational metric reveals that binary influence classification is structurally unstable on dense social networks (84.2% community boundary spanning, Jaccard ceiling 0.68). Continuous regression is the principled formulation. This is a genuinely novel finding.

**Section 4 (secondary contribution):** GNN achieves statistically equivalent ranking performance to degree centrality (bootstrap CI) while requiring no precomputed graph statistics. Message passing adds +0.099 Spearman over feature-only MLP. Runtime speedup 7,169×.

**Section 4 (architecture analysis):** Five architectures (SAGE/GCN/GIN/GAT/APPNP) all converge to the same performance ceiling. This is evidence that under degree-coupled IC, the ceiling is structural, not architectural. APPNP's multi-hop propagation does not overcome the degree dominance.

The paper title becomes something like: "Stability Analysis of Monte Carlo Influence Estimation in Dense Social Networks with GNN Surrogate Approximation."

The strength of this path: every claim is supported by data, no formula engineering required, the stability finding is genuinely interesting, and the equivalence + runtime story is practical. The weakness: "GNN matches degree" is a modest ML contribution.

### Path B: Run SRC With Honest Framing

If the instructor requires GNN to beat degree, SRC is the fastest path. But it must be framed honestly.

Run SRC (source rank cascade) with `φ = rank(views)/N`. Verify that GNN with views features achieves rho ~0.90 while degree achieves ~0.51 on SRC labels. This is a genuine win for GNN.

**But also report:** `views_rank` baseline achieves rho ~0.98 on SRC labels. GNN does not beat the trivial views-rank baseline. The finding is: "Under popularity-driven diffusion, views rank is the dominant predictor. GNN learns this relationship from graph structure but cannot improve on direct views ranking."

The paper framing: "We compare two IC operationalizations — structural (A0, degree-coupled) and popularity-driven (SRC). Under A0, GNN matches degree. Under SRC, GNN substantially outperforms degree but not views-rank. This reveals that GNN's advantage depends critically on the alignment between the diffusion model and the features available to the learner."

This is a more interesting paper than Path A because it has a comparative finding. The risk: a reviewer may ask "why is SRC a good influence model?" and the answer ("because it makes GNN outperform degree") is circular.

### Path C: A2 Sensitivity + Architecture Analysis as Moderate Improvement

A2 (`p = 1/sqrt(deg_u × deg_v)`) already exists and has R² = 0.546 with degree — meaning 45% of variance is unexplained by degree alone. This is the most promising existing variant for GNN improvement without engineering new formulas.

Run all five architectures on A2 labels. GCN is theoretically aligned with A2 (the GCN normalization `D^{-1/2}AD^{-1/2}` is structurally analogous). If GCN on A2 labels achieves Spearman significantly above degree's 0.762 (the degree-A2 correlation), this is a clean finding about architectural inductive bias.

The paper framing: "We evaluate GNN surrogate learning under two IC diffusion rules: weighted cascade (A0) and symmetric normalization (A2). Under A0, degree-based baselines are near-optimal. Under A2, GCN's symmetric normalization provides a natural inductive bias, achieving [X] Spearman compared to degree's [Y]. This demonstrates that GNN architecture selection should be informed by the diffusion dynamics being approximated."

This is scientifically sound because A2 has independent justification (symmetric normalization is natural for undirected graphs) and the GCN-A2 alignment is a pre-registered hypothesis with theoretical grounding in spectral graph theory.

---

## 6. The Baseline That Could Beat Everything

The concern about "what if a baseline beats degree AND GNN" deserves a specific answer. The most dangerous baselines are:

**Three-hop analytical proxy under A0:** Extends two-hop (0.804) by one more level. Could reach 0.82+. Implementation: O(deg³) per node, feasible with batching. If this beats both degree and GNN on A0, the finding is "analytical multi-hop proxies are sufficient for weighted cascade IC."

**Views_rank under SRC:** As analyzed above, this trivially achieves near-perfect correlation with SRC labels.

**One-hop analytical proxy × views under HSCC:** The product `views_rank × (1 + γ × cross_frac)` is computable in O(E) and may match GNN on HSCC labels.

The way to handle this risk is to **include these analytical proxies as baselines in your table**. If they beat GNN, report that honestly. The finding "analytical proxies sufficient, GNN not needed" is publishable — it tells practitioners they can skip expensive GNN training.

---

## 7. My Recommended Formula If GNN Must Win

If the constraint is that GNN must demonstrably beat all baselines including degree, views_rank, and analytical proxies, the only formula that achieves this without being transparently engineered is one that requires **multi-hop neighborhood composition** — something no single-node metric or one-hop proxy can compute.

```python
def compute_interaction_ic(u, v, indptr, indices, views_log, community, degrees):
    """
    p(u,v) depends on the INTERACTION between source u's neighborhood
    and target v's neighborhood — requiring 2-hop information.
    """
    # Component 1: source popularity (views of u)
    source_pop = views_log[u] / max(degrees[u], 1)
    
    # Component 2: neighborhood diversity mismatch
    # How different is u's neighborhood from v's neighborhood?
    u_neighbors = set(indices[indptr[u]:indptr[u+1]])
    v_neighbors = set(indices[indptr[v]:indptr[v+1]])
    jaccard_uv = len(u_neighbors & v_neighbors) / max(len(u_neighbors | v_neighbors), 1)
    
    # Low Jaccard = diverse connection → information reaches new parts of graph
    diversity_bonus = 1.0 - jaccard_uv  # higher when neighborhoods don't overlap
    
    # Combined probability
    p = clip(source_pop * (1 + 2.0 * diversity_bonus), p_max=0.3)
    return p
```

**Why this requires GNN:** The Jaccard similarity between neighborhoods of u and v cannot be computed from any single-node feature (degree, views, kshell, etc.). It requires knowing the actual neighbor sets of both endpoints. A 2-layer GNN naturally computes this: Layer 1 aggregates neighbor information for each node, Layer 2 compares aggregated representations of connected nodes.

**Why this is defensible:** Neighborhood overlap affects information redundancy in cascades — if u and v share many neighbors, activating v from u adds fewer new potential targets. This is the "structural holes" argument from Burt (1992), well-established in SNA literature.

**The analytical proxy risk:** Computing Jaccard for all edges is O(E × mean_degree) ≈ O(E²/N). For Twitch, this is roughly 6.8M × 82 ≈ 560M operations — feasible but slow (minutes, not seconds). If someone precomputes edge Jaccard and uses it as a feature, they could match GNN. But this requires edge-level precomputation that GNN does implicitly through message passing — the runtime comparison favors GNN for repeated inference.

**The honest caveat:** This formula is more complex than A0 and harder to justify as "the right model for Twitch influence." It should be framed as "testing whether GNN can learn neighborhood-composition-dependent diffusion patterns" rather than "this is how influence works on Twitch."

---

## 8. Practical Recommendation for the Next 13 Days

Given the timeline, here is what I would do.

**Days 17–18:** Run APPNP and all other architectures on A0 labels (already planned as C2). Simultaneously, run the A2 sensitivity. Generate `regression_targets_a2.parquet` and run C2-A2 (all architectures on A2 labels). This is the highest-value experiment for the remaining time: if GCN on A2 beats degree-on-A2 correlation (0.762) by a meaningful margin, you have a clean paper.

**Day 18 evening:** Look at C2 results. Decision point:
- If any architecture on A0 beats degree (0.826): lead with that. Run bootstrap CI.
- If GCN on A2 beats degree-on-A2 (0.762) significantly: lead with the dual-operationalization story (A0 equivalence + A2 GCN advantage).
- If neither: commit to Path A (stability finding as primary contribution).

**Days 19–21:** Run ranking loss (C3) and bootstrap CI (C4) based on the decision above. Lock all experimental results by day 21.

**Days 22–27:** Write the paper based on actual results, not hoped-for results.

Do not attempt to implement SRC, HSCC, or the Jaccard-based formula unless A2 results are clearly insufficient and the instructor explicitly requests a new IC formulation. Each new formula requires pilot validation, full simulation, GNN retraining, and baseline recomputation — at minimum 3 days of work. With 13 days to deadline including paper writing, there is no margin for another formula iteration.

---

## 9. Bottom Line

The data tells a clear story: on dense social networks under standard IC formulations, analytical baselines are extremely competitive with learned representations. This is itself a publishable finding. The plan's existing framework — stability analysis, regression justification, architecture comparison, runtime speedup — provides sufficient material for a MAPR paper regardless of whether GNN beats degree.

The strongest remaining path to GNN advantage within the current data is A2 labels + GCN architecture alignment. If that produces even a 0.02–0.03 Spearman improvement over degree-on-A2, combined with the A0 equivalence story, the paper has a clean dual-operationalization narrative that is both honest and interesting.

Engineering a formula specifically to make GNN win is technically possible but scientifically risky. It shifts the contribution from "understanding influence approximation" to "designing labels that favor GNN" — and reviewers at SNA-adjacent venues will notice.

---

Dưới góc nhìn **reviewer SNA**, mình sẽ trả lời rất thẳng:

# Kết luận ngắn

**Không — các công thức hiện có trong file chưa đủ để đảm bảo GNN thắng mọi baseline.**  
Thậm chí, với dữ liệu hiện tại, ta đã thấy rõ:

- **A0**: degree gần như là “oracle baseline” theo cấu trúc nhãn
- **A2**: degree yếu đi nhưng vẫn rất mạnh
- **I-A**: label gần như mất variance nên không learnable
- **SRC/SPC** kiểu source-views thuần: rất dễ làm **`views_rank`** trở thành baseline mới mạnh hơn cả degree và vẫn thắng GNN
- **HSCC** tốt hơn SRC, nhưng vẫn có nguy cơ bị một **composite analytical baseline** đánh bại

Nói ngắn gọn hơn:

> **Nếu mục tiêu là “GNN phải thắng mọi baseline”, thì đó không phải mục tiêu khoa học lành mạnh.**  
> Với bất kỳ công thức IC nào đủ tường minh, gần như luôn có thể viết ra một baseline analytical tương ứng để cạnh tranh hoặc thắng GNN.

Vì vậy, câu hỏi đúng không phải là:

> “Công thức nào đảm bảo GNN thắng tất cả baseline?”

mà là:

> **“Operationalization nào tạo ra label vừa defensible, vừa không trivially reducible về một scalar baseline đơn giản, để GNN có cơ hội thắng một bộ baseline hợp lý?”**

---

# 1. Tình hình hiện tại: chẩn đoán đúng vấn đề

Dựa trên các kết quả bạn gửi, tình hình bây giờ rất rõ:

---

## 1.1 A0: GNN thua degree không phải do model yếu, mà do label bị degree-coupled
Bạn đã có:

- `R²(IC_A0 ~ degree) = 0.8868`
- `rho(IC_A0, degree) ≈ 0.828`
- `rho(IC_A0, one_hop) ≈ 0.717`
- `rho(IC_A0, two_hop) ≈ 0.804`

Điều này nói rằng:

- **IC_A0 gần như là hàm của cấu trúc degree/local neighborhood**
- degree đã giải thích gần 89% variance
- two-hop analytical proxy đã rất sát

=> Trong regime này, GNN gần như **không còn “room”** để thắng degree một cách đáng kể.

**Kết luận reviewer:**  
A0 vẫn hoàn toàn publishable, nhưng **không phải nơi tốt để ép GNN thắng baseline**.

---

## 1.2 A2: tốt hơn A0 cho GNN, nhưng chưa chắc đủ
Bạn có:

- `R²(IC_A2 ~ degree) ≈ 0.546`
- `rho(IC_A2, degree) ≈ 0.762`

Điều này tốt hơn A0 rất nhiều:
- degree yếu đi
- còn ~45% variance chưa giải thích

=> **A2 là candidate tốt nhất trong các variant structural hiện có** để thử GCN/APPNP/GIN/GAT.

**Kết luận reviewer:**  
Nếu muốn một hướng **defensible** mà vẫn có cơ hội cho GNN, thì **A2 là best existing option**.

---

## 1.3 I-A: pilot pass nhưng full fail — và đây là failure mode có tính toán học
Bạn có:

- pilot CV ổn
- nhưng full run:
  - `CV ≈ 0.239`
  - `rho(IC, degree) ≈ 0.042`
  - `rho(IC, views) ≈ ~0`
  - degree quintile means gần như bằng nhau

=> I-A row-normalization làm:
\[
\sum_{v\in N(u)} p(u,v)=1
\]
khiến one-hop expectation gần như giống nhau cho mọi node → label collapse.

**Kết luận reviewer:**  
I-A **không nên dùng làm main target**.  
Nó chỉ có giá trị như **negative evidence**:
- “attribute-informed row-normalized diffusion can destroy learnability on dense power-law social graphs.”

---

# 2. Câu hỏi quan trọng nhất: nếu một baseline khác còn mạnh hơn degree thì sao?

Đây là câu hỏi rất đúng.

## Câu trả lời thẳng:
**Nếu có baseline khác mạnh hơn degree và vẫn thắng GNN, thì đừng cố “né” nó.**
Phải:

1. **đưa baseline đó vào bảng chính**
2. **report trung thực**
3. **đổi contribution từ “GNN wins” sang “analytical proxy suffices under this operationalization”**
4. hoặc **bỏ công thức hiện tại và chuyển sang một operationalization mới**, nhưng chỉ khi có lý do phương pháp luận rõ ràng

Nếu không làm vậy, paper sẽ bị xem là:
- p-hacking
- label engineering to favor deep learning
- thiếu honesty

---

## Cụ thể với các công thức source-driven:
### SRC / SPC
Nếu:
\[
p(u,v)\propto \frac{\phi(\text{views}_u)}{\deg(u)}
\]
thì one-hop expectation sẽ thành hàm của **source views**.

=> baseline `views_rank` hoặc `source_score_rank` sẽ gần như thành oracle.

**Kết luận:**  
SRC/SPC có thể giúp GNN thắng **degree**, nhưng rất dễ thua một baseline khác mạnh hơn degree: **`views_rank`**.

Nên nếu dùng SRC/SPC, đừng tự hỏi “GNN có thắng degree không”, mà phải hỏi:
> **GNN có thắng `views_rank` không?**

Câu trả lời thường là: **rất khó**.

---

## Với HSCC / community-boosted source formula
Nếu:
\[
p(u,v)\propto \frac{\phi(\text{source})}{\deg(u)}(1+\gamma \mathbf{1}[c_u\neq c_v])
\]

thì `views_rank` một mình không còn đủ, nhưng một baseline kiểu:
\[
\phi(u)\cdot (1+\gamma\cdot \text{cross\_community\_fraction}(u))
\]
có thể lại rất mạnh.

=> HSCC tốt hơn SRC ở chỗ **khó bị một scalar baseline đánh bại hơn**, nhưng vẫn **không đảm bảo GNN thắng tất cả**.

---

# 3. Các công thức trong file có đủ mạnh để GNN thắng mọi baseline không?

## 3.1 SRC — **không đủ**
### Ưu điểm:
- phá degree dominance
- dễ implement
- CV ổn
- GNN raw-attr có thể predict tốt

### Nhưng vấn đề chí mạng:
- `views_rank` baseline sẽ cực mạnh
- có thể gần như thắng GNN

### Reviewer verdict:
**Không đủ mạnh nếu bộ baseline của bạn có `views_rank`.**

---

## 3.2 SPC — **cũng không đủ**
SPC chỉ là SRC mềm hơn (log views thay rank views), nhưng bản chất vẫn là source-popularity driven.

=> baseline popularity vẫn mạnh.

**Reviewer verdict:**  
**Không giải quyết được baseline problem, chỉ đổi degree problem thành views problem.**

---

## 3.3 HSCC — **tốt nhất trong các công thức bạn nêu**
### Vì sao tốt hơn:
- không phụ thuộc source popularity một chiều
- inject edge-level / community-level structure
- degree baseline yếu đi
- views baseline một mình cũng không còn đủ
- GNN có thể dùng:
  - source attrs
  - community structure
  - message passing

### Nhưng:
- composite baseline analytical vẫn có thể mạnh
- nếu γ quá lớn, reviewer sẽ hỏi “tại sao 3 mà không phải 1 hay 10?”

### Reviewer verdict:
**HSCC là ứng viên tốt nhất trong các công thức bạn đã nêu, nhưng vẫn không đảm bảo GNN thắng mọi baseline.**

---

## 3.4 II-B (target-side views/deg) — **không nên kỳ vọng nhiều**
Target-side attribute formulas trên graph dense thường dính mixing trap:

\[
\sum_{v\in N(u)} f(\text{attr}_v)\approx \deg(u)\cdot \mathbb{E}[f(\text{attr})]
\]

Bạn đã có smoking gun:
- `rho(neigh_logviews_sum, degree)=0.9915`

=> mọi công thức kiểu “sum target-side views over neighbors” gần như sẽ quay về degree.

**Reviewer verdict:**  
**Không nên đầu tư thêm vào family này.**

---

# 4. Nguyên lý quan trọng nhất: điều kiện để GNN có cơ hội thắng

Mình đồng ý với framework 4 điều kiện mà bạn nêu, và tóm gọn lại:

## Để GNN có genuine advantage, label phải:
1. **có variance đủ**
2. **không quá tương quan với degree**
3. **không quá tương quan với views rank hoặc một scalar source score**
4. **phụ thuộc vào multi-hop / edge-interaction / neighborhood composition**
5. **nhưng vẫn có semantic SNA hợp lý**

### Tức là:
- A0 fail ở (2)
- I-A fail ở (1)
- SRC/SPC fail ở (3)
- HSCC partial pass
- A2 partial pass

---

# 5. Giải pháp mình đề xuất: đừng hỏi “formula nào đảm bảo GNN thắng”, hãy dùng **screening protocol**

Nếu bạn còn muốn thử công thức mới, thì **mọi candidate formula phải qua screening trước khi full simulation**.

## Screening criteria mình khuyên:
Trên pilot 200–500 nodes, reject ngay nếu:

### Label quality
- `CV <= 0.30`
- hoặc `top10/median <= 2`

### Baseline resistance
- `rho(IC, degree) >= 0.80`
- hoặc `rho(IC, views_rank) >= 0.85`
- hoặc `rho(IC, one-hop/two-hop/custom analytical proxy) >= 0.85`

### Semantic defensibility
- công thức phải có diễn giải SNA hợp lý
- không được là “deep-learning-friendly by design” một cách lộ liễu

Nếu fail một trong các cái trên:
- **không full run**
- không train GNN
- coi như rejected candidate

Đây là cách làm khoa học hơn rất nhiều so với việc cứ chạy full rồi mới thấy baseline lại thắng.

---

# 6. Công thức mình thực sự khuyên dùng nếu bạn muốn thử một hướng mới

Nếu bạn **bắt buộc** cần một operationalization mới có cơ hội cho GNN thắng hơn, mình không khuyên SRC thuần.  
Mình khuyên một công thức **mixed, source-driven + structurally grounded + anti-trivial-baseline**.

---

## Đề xuất chính thức của mình:
# **Residual Source-Bridge Cascade (RSBC)**

### Bước 1 — tạo source strength không trivially bằng views hoặc degree
Định nghĩa residual popularity:

\[
z_u=\log(1+\text{views}_u)-\hat f\big(\log(1+\deg_u),\log(1+\text{life\_time}_u)\big)
\]

trong đó \(\hat f\) là hồi quy đơn giản (linear regression hoặc GAM nhẹ).

Sau đó:
\[
s_u=\frac{\operatorname{rank}(\max(z_u,0))}{N}
\]

Ý nghĩa:
- `s_u` đo phần popularity **không giải thích được** bởi degree và tenure
- như vậy:
  - degree baseline không còn là oracle
  - views_rank baseline cũng không còn oracle
- đây là cách triệt luôn cả hai baseline quá mạnh

### Bước 2 — đưa edge-level bridge signal
\[
b(u,v)=1+\gamma\cdot \mathbf{1}[c_u\neq c_v]
\]

hoặc nếu còn sức:
\[
b(u,v)=1+\gamma_1\mathbf{1}[c_u\neq c_v]+\gamma_2(1-\text{Jaccard}(N(u),N(v)))
\]

### Bước 3 — pha thêm thành phần structural vừa đủ để vẫn plausible
\[
p(u,v)=\operatorname{clip}\left(
\lambda\left[
\beta\frac{s_u}{\deg(u)}+(1-\beta)\frac{1}{\sqrt{\deg(u)\deg(v)}}
\right] b(u,v),
\ p_{\max}
\right)
\]

## Khuyến nghị tham số ban đầu:
- \(\beta = 0.7\)
- \(\gamma = 1.5\)
- \(p_{\max}=0.3\)
- \(\lambda\) chọn theo pilot sao cho:
  - mean reach không degenerate
  - CV > 0.30

---

## Tại sao công thức này là hợp lý hơn SRC thuần?

### So với SRC:
- không bị `views_rank` trivially dominate
- vì dùng **residualized source popularity**, không phải raw views rank

### So với A0/A2:
- degree yếu đi
- nhưng vẫn giữ một thành phần structural plausible

### So với HSCC:
- chặt hơn vì source popularity đã được “deconfound” khỏi degree/lifetime
- edge bridge term vẫn giữ được SNA meaning

### GNN advantage:
GNN có thể dùng:
- node attrs: `views`, `life_time`, maybe residual score
- graph structure
- community information
- message passing để hấp thụ interaction giữa source strength và bridge topology

### Analytical baseline risk:
vẫn còn, nhưng thấp hơn rõ so với SRC/SPC.

---

# 7. Nếu không muốn residualization vì sợ bị reviewer nói engineered thì sao?

Khi đó mình khuyên dùng bản đơn giản hơn:

# **HSCC-refined**
\[
p(u,v)=\operatorname{clip}\left(
\lambda \cdot \frac{\operatorname{rank}(\log(1+\text{views}_u)/(1+\text{life\_time}_u))}{N\cdot \deg(u)}
\cdot \bigl(1+\gamma\mathbf{1}[c_u\neq c_v]\bigr),
\ p_{\max}
\right)
\]

### Đây là bản:
- dễ giải thích hơn
- ít “engineered” hơn
- vẫn có source-driven signal
- still community-aware

### Nhưng:
- baseline `source_rank × cross_comm_fraction` vẫn có thể mạnh

=> nếu dùng công thức này, **bắt buộc phải benchmark thêm composite baseline đó**.

---

# 8. Nếu baseline mới vẫn thắng GNN thì làm sao?

## Câu trả lời reviewer-honest:
Nếu baseline mới vẫn thắng GNN, thì:
1. **đừng tiếp tục đổi công thức chỉ để GNN thắng**
2. thêm baseline đó vào bảng
3. reframe contribution thành:
   - “we characterize when analytical proxies suffice”
   - “GNN helps only when diffusion operationalization encodes signals unavailable to simple node-level summaries”
   - “surrogate value lies in runtime and flexibility, not always superiority”

### Điều này vẫn publishable ở MAPR
Đặc biệt nếu bạn có:
- strong stability/regression finding
- runtime story
- architecture comparison
- one or two alternative operationalizations

---

# 9. Khuyến nghị thực chiến cuối cùng cho bạn

Với tình trạng hiện tại, mình khuyên **không nên mở quá nhiều hướng mới**.

## Phương án an toàn nhất
### Track chính:
- **A0 primary**
- **A2 sensitivity**
- architecture comparison + bootstrap CI
- paper claim = equivalence / ceiling / runtime + message passing helps over MLP

### Nếu giảng viên rất muốn “GNN thắng”:
Chỉ thử **1 công thức mới duy nhất**.

## Mình khuyên:
### **HSCC-refined** nếu muốn đơn giản/defensible hơn
hoặc
### **RSBC** nếu muốn maximize chance GNN thắng và chấp nhận formula engineered hơn một chút

Nhưng chỉ chọn **một**.

---

# 10. Câu trả lời trực diện cho từng câu hỏi của bạn

## “Nếu lỡ có 1 baseline nào đó đạt kết quả cao hơn degree và vẫn thắng được GNN thì sao?”
**Thì phải report baseline đó và đổi claim.**  
Không nên tiếp tục “đào label” cho tới khi GNN thắng — reviewer sẽ coi đó là p-hacking.

---

## “Liệu các công thức đưa trong file có đủ mạnh để GNN thắng được mọi baseline và đạt kết quả tốt?”
**Không.**
- SRC/SPC: không đủ, vì `views_rank` sẽ rất mạnh
- HSCC: tốt hơn, nhưng vẫn không đảm bảo thắng composite analytical baseline
- A0/A2: không đủ để GNN chắc thắng
- I-A: không learnable

---

## “Giải pháp và công thức bạn đề xuất là gì để GNN đạt được kết quả tốt nhất?”
Nếu buộc phải thử công thức mới, thứ tự mình khuyên là:

### Ưu tiên 1 — **HSCC-refined** (defensible hơn)
\[
p(u,v)=\operatorname{clip}\left(
\lambda \cdot \frac{\phi(u)}{\deg(u)}
\cdot (1+\gamma\mathbf{1}[c_u\neq c_v]),
\ p_{\max}
\right)
\]
với \(\phi(u)\) = source popularity score chuẩn hóa (ưu tiên rank của log-views-per-day)

### Ưu tiên 2 — **RSBC** (khả năng cho GNN lợi thế cao hơn)
\[
p(u,v)=\operatorname{clip}\left(
\lambda\left[
\beta\frac{s_u}{\deg(u)}+(1-\beta)\frac{1}{\sqrt{\deg(u)\deg(v)}}
\right](1+\gamma\mathbf{1}[c_u\neq c_v]),
\ p_{\max}
\right)
\]
với \(s_u\) là residualized source popularity score

---

# 11. Final reviewer recommendation

## Nếu mục tiêu là paper MAPR defensible:
**Đừng tối ưu theo “GNN phải thắng mọi baseline”.**  
Hãy tối ưu theo:

1. **label operationalization có nghĩa**
2. **stability story mạnh**
3. **baseline comparison honest**
4. **runtime contribution rõ**
5. **một supplemental operationalization nơi GNN có cơ hội thắng hợp lý**

## Nếu mục tiêu phụ là “thử tạo một setting mà GNN có khả năng thắng”:
- thử **1 công thức mới duy nhất**
- pre-register rõ
- benchmark cả:
  - degree
  - views_rank / source_rank
  - one-hop / two-hop
  - composite analytical proxy phù hợp với công thức đó
- nếu GNN vẫn không thắng → dừng và report

---
