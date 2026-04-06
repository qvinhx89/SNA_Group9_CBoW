

# Phân tích Bottleneck và Tài liệu Tham khảo Cần thiết

---

## 1. Bottleneck #1: One-Hop Spread Có Thể Dominate Toàn Bộ IC Ranking

### Tại sao đây là rủi ro lớn nhất

Dưới weighted cascade với `p(u,v) = 1/degree(v)` trên undirected graph, kỳ vọng một hop spread cho node u là `Σ_{v ∈ N(u)} 1/degree(v)`. Với Twitch mean degree khoảng 81, giá trị p trung bình xấp xỉ 1/81 ≈ 0.012. Ở mức p này, đại đa số cascade chết trong 1–2 hops. Điều đó có nghĩa là IC reach gần như hoàn toàn được xác định bởi cấu trúc neighborhood cục bộ — chính xác là thứ mà one-hop analytical formula đã capture.

Quyết định narrative phụ thuộc vào **3-metric gate** (không phải chỉ Spearman): nếu cả ba cùng pass — `ρ > 0.9` **VÀ** `Jaccard@10% > 0.8` **VÀ** `NDCG@10% > 0.9` — thì GNN narrative phải restructure sang "analytical proxy + divergence analysis." Nếu chỉ `ρ > 0.9` nhưng top-k alignment chưa cao (Jaccard hoặc NDCG chưa pass), giữ nguyên GNN + 2-hop head-to-head comparison — đây không phải rescue, đây là planned narrative trong M2. Team cần chuẩn bị cả hai hướng trước khi biết kết quả Day 1.

### Những thứ cần thử nghiệm

Đầu tiên, cần đo ρ(one-hop, IC) trên pilot 200 nodes trước khi commit bất kỳ compute nào. Nếu ρ nằm trong khoảng 0.8–0.9, two-hop proxy trở thành baseline then chốt — cần verify xem two-hop có capture thêm variance đáng kể không, hay chỉ thêm noise. > ⚠ **Three-hop KHÔNG khả thi tại Twitch scale:** O(Σ d(v)³) ≈ 81³ per node × 168k nodes → infeasible. Giới hạn tối đa là two-hop (O(Σ d(v)²)). Mọi thực nghiệm dừng ở one-hop vs two-hop.

Nếu cả 3 metrics đều pass: không thay đổi IC semantics — tránh "directed reinterpretation" hoặc "scaling p" vì cả hai thay đổi model definition và làm paper mất reproducibility với literature. Narrative restructure là đủ và defensible.

Ngoài ra, cần xem xét liệu GNN có thể win trên top-k metrics ngay cả khi Spearman toàn cục cao — vì one-hop có thể rank đúng globally nhưng sai ở top nodes nơi multi-hop structure matters hơn. Đây là lý do Jaccard@10% và NDCG@10% là bắt buộc trong gate, không phải optional.

### Tài liệu cần đọc

Kempe, Kleinberg & Tardos (2003), "Maximizing the Spread of Influence through a Social Network" (KDD 2003) — paper gốc define IC model và weighted cascade. Cần hiểu chính xác điều kiện nào làm cascade die nhanh (low p regime) để justify tại sao one-hop dominate.

Chen, Wang & Wang (2010), "Scalable Influence Maximization for Prevalent Viral Marketing in Large-Scale Social Networks" (KDD 2010) — introduce PMIA (Path-based Maximum Influence Arborescence) model, chứng minh rằng influence giảm exponentially theo hop count, support lập luận "local neighborhood dominates."

Lü, Chen, Ren, Zhang, Zhang & Zhou (2016), "Vital nodes identification in complex networks" (Physics Reports) — survey toàn diện về node importance metrics, bao gồm phân tích khi nào local metrics (degree, one-hop) sufficient và khi nào cần global metrics. Đặc biệt Section 4 về spreading dynamics trên real networks cho thấy regime nào local/global matters.

Chen, Lu & Zhang (2012), "Identifying Influential Nodes in Complex Networks" (Physica A) — propose LocalRank algorithm, đo influence bằng local structure (4-hop neighborhood). So sánh LocalRank với degree/betweenness/k-shell cho thấy khi nào local đủ tốt.

Morone & Makse (2015), "Influence maximization in complex networks through optimal percolation" (Nature 524:65–68) — introduce Collective Influence (CI) algorithm, chứng minh rằng influential spreaders KHÔNG phải luôn là high-degree hubs mà là nodes ở "optimal percolation threshold." Relevant để frame "one-hop/degree proxy vs actual IC rank divergence" là hiện tượng đã được document trong spreading literature — không chỉ là artifact của paper này.

---

## 2. Bottleneck #2: IC Cascade Distribution Có Thể Degenerate

### Tại sao đây là rủi ro cao

Với p ≈ 0.012 (weighted cascade trên graph degree ~81) và graph clustering coefficient ~0.13, rất có khả năng phần lớn single-seed cascades chết ngay sau seed node (reach = 1) hoặc chỉ spread được 2–5 nodes. Distribution sẽ cực kỳ right-skewed với median gần 1 và mean bị kéo bởi vài outlier nodes ở core. Coefficient of variation (CV) có thể thấp hơn ngưỡng 0.3 mà plan yêu cầu, không phải vì simulation sai mà vì dynamics thực sự degenerate ở mức p này.

Nếu CV < 0.3, ranking IC scores gần như vô nghĩa — đa số nodes có score giống nhau (≈ 1–2) và sự khác biệt chủ yếu là noise từ MC simulation. GNN sẽ không thể học gì useful từ labels như vậy.

> **Correction quan trọng — Jensen's inequality và bimodal IC distribution:**
>
> Review này dùng p ≈ 1/81 như activation probability "trung bình" — điều này không chính xác. Với weighted cascade `p(u,v) = 1/degree(v)`, xác suất activation phụ thuộc vào degree của **node đích** v, không phải nguồn. Vì 1/x là hàm lồi, **Jensen's inequality** cho: `E[1/degree(v)] > 1/E[degree(v)] = 1/81`. Trên scale-free network (heavy-tailed degree), các low-degree nodes đóng góp mạnh vào kỳ vọng này — p thực sự trung bình qua tất cả các cạnh cao hơn 1/81 đáng kể.
>
> Hệ quả: IC distribution rất có khả năng **bimodal theo degree quintile**, không phải uniformly degenerate:
> - **Hub nodes (degree quintile Q5)**: nhiều neighbors có degree trung bình thấp → p cao → cascade spread meaningful → reach phân bố rộng
> - **Low-degree nodes (Q1–Q4)**: neighbors cũng thường là large-degree (preferential attachment) → p thấp → reach ≈ 1–3
>
> Điều này có nghĩa: **global CV (toàn bộ n_sample nodes) sẽ rất có thể PASS ngưỡng 0.3** nhờ between-quintile variance lớn. Nhưng **within-quintile CV cho Q1–Q4 có thể fail** — ranking IC trong cùng degree band dominated by MC noise, không phải structural signal. GNN có thể học tốt signal ở cấp quintile nhưng fail ở cấp fine-grained ranking bên trong Q1–Q4. Đây là limitation cần report, không phải failure cần tránh.

### Những thứ cần thử nghiệm

Pilot diagnostics 200 nodes × 50 runs phải report đầy đủ: mean, median, IQR, top-10%/median ratio, và histogram của reach distribution.

**Bước 1 — Primary Diagnostics (PHẢI chạy trước, trước khi xem xét bất kỳ giải pháp nào):**

**Per-quintile CV diagnostic (bắt buộc trên pilot 200 nodes):** Chia 200 pilot nodes thành 5 degree quintiles (Q1–Q5, ~40 nodes mỗi nhóm). Tính CV riêng cho từng quintile. Nếu CV_Q1 < 0.15 và CV_Q2 < 0.20, confirm rằng ranking trong low-degree group là noise-dominated — ghi vào `docs/day1_decisions.md` là known limitation. GNN evaluation nên focus vào overall ranking (toàn bộ n_sample) chứ không claim "accurately ranks nodes within a degree band."

**Diagnostics robust hơn CV** — đặc biệt cần thiết ở low-mean regimes khi CV dễ bị nhiễu:
- **P(reach > 1)**: tỷ lệ cascades có ít nhất 1 node bị lây nhiễm ngoài seed. Nếu P(reach > 1) < 0.2 → regime quá degenerate, cần thêm runs hoặc check LCC.
- **P(reach > 5)**: tỷ lệ cascades spread được ≥ 5 nodes. Nếu P(reach > 5) < 0.05 → hầu hết influence chỉ trong immediate neighborhood.
- **Tail separation ratio**: `mean_reach(top-10%) / median_reach`. Nếu ratio > 5 → tail phân hóa đủ mạnh, labels vẫn usable dù median ≈ 1–2.
- **Run-count stability curve**: plot Kendall τ giữa N_runs và N_runs/2 cho mỗi quintile. Nếu τ > 0.95 ở N_runs=100, giảm xuống 100 để tiết kiệm compute (50% savings).

→ **Nếu kết quả ổn** (CV_global > 0.3, bimodal signal rõ qua quintile table, Kendall τ > 0.85, tail separation ratio > 3×): **không cần ba hướng bên dưới**. Ghi kết quả vào `ic_pilot_diagnostics.json` và tiến hành full IC labeling.

**Bước 2 — Last Resort (chỉ nếu primary diagnostics fail toàn bộ — median reach < 2 VÀ P(reach > 1) < 0.2 VÀ tail separation < 2×):**

Nếu median reach < 2, cần xem xét ba hướng. Hướng thứ nhất là restrict analysis về LCC subgraph nếu graph có nhiều component nhỏ — đảm bảo cascade có "room to spread." Hướng thứ hai là thử uniform p (κ-target với κ=2 hoặc κ=3) như sensitivity variant — nếu reach distribution meaningful hơn dưới uniform p nhưng degenerate dưới weighted cascade, đây là finding về regime sensitivity. Hướng thứ ba là thay đổi metric: thay vì dùng raw reach count, dùng normalized reach (reach/degree) hoặc reach relative to component size — nhưng cần justify tại sao normalization phù hợp.

Cũng cần test xem tăng MC runs (từ 200 lên 500) có stabilize ranking ở regime degenerate không — vì khi reach = 1 với probability 0.95 và reach > 1 với probability 0.05, cần rất nhiều runs để phân biệt node nào có 5% chance vs 3% chance.

### Tài liệu cần đọc

Kitsak, Gallos, Havlin et al. (2010), "Identification of influential spreaders in complex networks" (Nature Physics) — phân tích k-shell vs degree vs betweenness cho SIR spreading. Quan trọng nhất là Figure 3 và SI, cho thấy spreading outcome phụ thuộc critically vào effective spreading rate β/μ. Giúp hiểu regime nào cascade meaningful.

Pastor-Satorras & Vespignani (2001), "Epidemic Spreading in Scale-Free Networks" (Physical Review Letters) — epidemic threshold trên scale-free networks. Với Twitch (heavy-tailed degree distribution), effective threshold có thể rất thấp, nghĩa là ngay cả small p cũng có thể produce non-trivial cascades. Cần kiểm tra.

Bao, Hao, Li & Cao (2022), "Identifying influential spreaders in complex networks based on k-shell hybrid method" (Physica A) — phân tích sensitivity của influence ranking với p trên nhiều real-world networks. Table 3 và 4 cho thấy ranking stability across p regimes.

Ling, Jiang, Wang et al. (2023), "Deep Graph Representation Learning and Optimization for Influence Maximization" (ICML 2023 — DeepIM paper đã có) — Section 5 Experiment Setup ghi rõ: weighted cascade `p_{u,v} = 1/d_in(v)`, LT threshold uniform [0.3, 0.6], SIS infection probability 0.001. Table 2 kết quả cho thấy reach 8–14% với seed set = 1% nodes trên Cora-ML (2810 nodes) nhưng chỉ 7–11% trên Digg (280k nodes). Twitch (168k, dense) có thể có behavior khác hoàn toàn.

---

## 3. Bottleneck #3: Typology Quadrant Sizes và Statistical Power

### Tại sao đây là vấn đề thực tế

2×2 typology cắt tại top-10% trên cả IC và views tạo ra 4 quadrants. "Hidden" (high IC, low views) và "Overrated" (low IC, high views) là hai quadrants quan trọng nhất, nhưng kích thước của chúng phụ thuộc vào correlation giữa views và IC. Nếu views và IC highly correlated (ρ > 0.8 — rất có thể trên Twitch vì high-degree nodes có cả views cao và IC reach cao), thì "True" và "Non" quadrants sẽ lớn, còn "Hidden" và "Overrated" sẽ rất nhỏ.

Với 5000 labeled nodes, top-10% IC = 500 nodes. Nếu 80% của 500 nodes IC-high cũng views-high (do correlation), Hidden chỉ có 100 nodes. Stratified MWU across 5 degree quintiles cần ~20 nodes per quintile per group — với 100 Hidden nodes, mỗi quintile chỉ có ~20 nodes, borderline cho Mann-Whitney U. Cliff's delta ≥ 0.20 yêu cầu effect size đáng kể — với sample nhỏ, power sẽ thấp.

### Những thứ cần thử nghiệm

Cần estimate correlation ρ(views, IC) sớm nhất có thể (từ pilot 200 nodes) để project quadrant sizes trước khi commit compute cho full 5000 nodes. Nếu ρ > 0.8, cần quyết định ngay: tăng n_sample lên 10000 (double compute cost) hoặc adjust threshold từ top-10% sang top-20% (thay đổi interpretation nhưng tăng sample per quadrant).

Two-sample strategy (augment Sample B với high-betweenness/low-views nodes) có risk: Sample B là biased sample, và statistical tests trên biased sample cần careful interpretation. Phải tách rõ: Sample A cho unbiased statistics, Sample B chỉ cho structural profiling và illustration.

Ngoài ra cần thử threshold alternatives: thay vì top-10% cứng, dùng top-20% và report cả hai trong sensitivity analysis. Lưu ý: plan đã lock `threshold=0.90` (top-10%) trong `m0_decisions.md` — nếu cần thay đổi phải lock lại ngay, không được thay đổi midway.

**Backup plan khi quadrants quá nhỏ — residual-based divergence:** Thay vì hard cut 2×2 quadrant, định nghĩa `divergence_score = standardized_rank(IC) - standardized_rank(views)`. Nodes với divergence_score ở top decile = "Hidden-like"; bottom decile = "Overrated-like." Advantages: (a) không phụ thuộc hard threshold, (b) sample size tự động tăng lên ~10% × n_sample = 500 nodes ở mỗi đầu, (c) continuous measure phù hợp hơn cho regression analysis. Frame trong paper là "rank-divergence analysis" thay vì "quadrant typology" — vẫn publishable và thực ra arguable strong hơn về mặt statistical rigor.

### Tài liệu cần đọc

Cha, Haddadi, Benevenuto & Gummadi (2010), "Measuring User Influence in Twitter: The Million Follower Fallacy" (ICWSM 2010) — phân tích divergence giữa follower count và retweet/mention influence. Figure 2 cho thấy Spearman ρ giữa followers và retweets chỉ ~0.5 trên Twitter. Nếu Twitch views-IC correlation cao hơn đáng kể, đó là finding riêng, cần discuss.

Bakshy, Hofman, Mason & Watts (2011), "Everyone's an Influencer: Quantifying Influence on Twitter" (WSDM 2011) — empirical analysis showing influence is highly variable even among similar users. Giúp frame "Hidden influencer" concept với data evidence.

Cohen (1988), "Statistical Power Analysis for the Behavioral Sciences" — reference chuẩn cho effect size interpretation. Cliff's delta ≥ 0.20 tương đương "small-to-medium" effect. Với n < 50 per group, power để detect small effect < 50%. Cần biết để report limitations đúng.

Benjamini & Hochberg (1995), "Controlling the False Discovery Rate" (JRSS Series B) — FDR correction method. Cần hiểu rằng BH correction trên 5 quintile tests (hay 6 feature tests) sẽ inflate p-values, đặc biệt khi sample nhỏ. Có thể cần report cả raw và corrected p.

---

## 4. Bottleneck #4: GNN Feature Fairness/Confounding và Ablation Interpretation

### Tại sao phức tạp hơn plan viết

Plan v3 thiết kế 4 GNN variants: raw-attr, graph-only, centrality, full. Mục tiêu: tách "giá trị của message passing" (raw-attr vs MLP) và "giá trị của attributes" (raw-attr vs graph-only). Tuy nhiên interpretation sạch chỉ khi assumptions đúng.

Vấn đề thứ nhất: GNN-raw-attr dùng views_log, views/day, life_time làm features. IC labels dưới weighted cascade KHÔNG dùng views — nhưng views correlate với degree (popularity ↔ connectivity), và degree trực tiếp determine IC reach. GNN sẽ learn rằng "nodes có views cao → thường degree cao → IC reach cao" — đây là indirect leakage qua correlation, không phải direct leakage. Interpretation "GNN captures higher-order structure" bị muddied.

Vấn đề thứ hai: GNN-graph-only chỉ có degree_norm làm feature. Với weighted cascade, IC reach gần như hoàn toàn determined bởi local topology (xem Bottleneck 1). Degree + 2-hop aggregation trong GraphSAGE sẽ approximate one-hop/two-hop spread rất tốt — nhưng đó không phải "learned higher-order structure," đó là "GNN re-discovers analytical formula." Cần so sánh GNN-graph-only output vs two-hop proxy trực tiếp để check.

Vấn đề thứ ba: GNN-full (all 6 features) sẽ hầu như chắc chắn outperform GNN-raw-attr vì centrality features encode IC-relevant information trực tiếp. Nhưng finding "more features → better" là trivial và không publishable. Cần careful framing: GNN-full là "upper bound with oracle features."

### Những thứ cần thử nghiệm

**So sánh quan trọng nhất (primary ablation):** Thêm **MLP-raw-attr** (same features như GNN-raw-attr nhưng không có message passing / graph structure). Nếu MLP-raw-attr ≈ GNN-raw-attr → message passing adds marginal value, main contribution là feature engineering. Nếu GNN-raw-attr >> MLP-raw-attr → message passing genuinely helps. Đây là comparison sạch nhất để justify "why GNN, not just feature regression." Nên đặt MLP-raw-attr là **primary baseline** trong ablation table, không phải secondary.

Cần thêm một variant quan trọng mà plan hiện thiếu: GNN với random node features (hoặc constant features = ones). Variant này test giá trị thuần túy của graph topology through message passing, không confound bởi bất kỳ node attribute nào. Nếu GNN-random ≈ GNN-graph-only (degree only), degree là sufficient summary; nếu GNN-random < GNN-graph-only, degree feature itself adds value beyond topology. **Add if time permits** (Day 18+).

Cần plot learning curves: Spearman ρ vs training epochs cho mỗi variant. Nếu GNN-raw-attr converge nhanh hơn GNN-graph-only, attributes đang giúp training — finding nhỏ nhưng publishable.

Cần test xem removing views_log khỏi GNN-raw-attr (chỉ giữ views/day + life_time) có thay đổi performance đáng kể không. Nếu có → views là key feature, indirect leakage concern lớn hơn. Nếu không → GNN thực sự learn từ temporal attributes (và views_log chỉ là correlated noise).

### Tài liệu cần đọc

Hamilton, Ying & Leskovec (2017), "Inductive Representation Learning on Large Graphs" (NeurIPS 2017) — GraphSAGE paper gốc. Section 5 analysis cho thấy aggregation function (mean vs LSTM vs pool) ảnh hưởng performance khác nhau trên từng dataset. Mean aggregation (plan v3 dùng) ổn nhất nhưng least expressive.

Xu, Hu, Leskovec & Jegelka (2019), "How Powerful are Graph Neural Networks?" (ICLR 2019) — lý thuyết expressiveness của GNN, liên hệ với Weisfeiler-Lehman test. Giúp hiểu tại sao 2-layer mean-aggregation GraphSAGE có thể approximate one-hop/two-hop analytical formula.

You, Ying & Leskovec (2019), "Position-aware Graph Neural Networks" (ICML 2019) — thảo luận limitation của message-passing GNN trong capture global position. Relevant vì IC reach phụ thuộc vào global graph position (core vs periphery) nhưng standard GNN chỉ capture local neighborhood.

Errica, Podda, Bacciu & Micheli (2020), "A Fair Comparison of Graph Neural Networks for Graph Classification" (ICLR 2020) — empirical study cho thấy đơn giản feature engineering + MLP thường competitive với GNN trên nhiều benchmark. Giúp justify narrative "centrality features sufficient" nếu GNN không win.

---

## 5. Bottleneck #5: Null Model — Configuration Model Có Thể Không Informative

### Vấn đề cụ thể

Configuration model bảo toàn degree sequence nhưng phá hủy community structure, clustering, và core-periphery organization. Trên Twitch (clustering coefficient ~0.13, nontrivial community structure), null graph sẽ có clustering gần 0 và topology fundamentally khác real graph.

Điều này tạo ra hai vấn đề. Thứ nhất, IC dynamics trên null graph sẽ rất khác real graph (vì weighted cascade spread pattern phụ thuộc vào local clustering), nên ρ(real IC, null IC) sẽ thấp — nhưng đó không chứng minh "higher-order structure matters," nó chỉ chứng minh "clustering affects cascade," điều ai cũng biết. Thứ hai, typology quadrants trên null graph sẽ có kích thước rất khác real graph vì IC distribution thay đổi hoàn toàn. So sánh "Hidden betweenness on real vs null" trên hai typology distributions khác nhau là apples-to-oranges.

### Những thứ cần thử nghiệm

Cần thử dk-series graph hoặc stochastic block model thay vì configuration model — những null models này bảo toàn thêm clustering hoặc community structure, tạo so sánh fair hơn. Tuy nhiên implementation phức tạp hơn và có thể ngoài scope 25 ngày.

Phương án pragmatic: chạy configuration model nhưng report rõ "null model differs from real graph in clustering and community structure" và interpret kết quả accordingly. Nếu null Hidden nodes CŨNG có high betweenness, finding mạnh hơn (betweenness elevated even when topology randomized). Nếu không, inconclusive (expected, because clustering destroyed).

Cần phân biệt rõ **hai null model bổ sung** với mục tiêu khác nhau, cả hai đều rẻ compute:

**Null model 2 — IC-score permutation:** Random shuffle `ic_score_mean` values across nodes (giữ nguyên graph). Nếu shuffled labels vẫn produce similar quadrant sizes và structural profiles, typology là **threshold artifact** (chỉ phụ thuộc vào distribution shape), không phải structural phenomenon. Chi phí: zero simulations, chỉ cần permute một array.

**Null model 3 — Views-permutation (mạnh hơn và trực tiếp hơn):** Giữ nguyên graph và IC scores, nhưng **permute `views` ngẫu nhiên** across nodes, rồi rebuild typology. Nếu Hidden/Overrated profile vẫn tồn tại sau permutation (Hidden vẫn elevated betweenness, Overrated vẫn reduced), thì divergence finding không phụ thuộc vào correlation views–IC — đây là structural phenomenon thật. Nếu profile biến mất sau permutation → views alignment là key mechanism, không phải graph structure. Đây là null model rẻ nhất, defensible nhất, và trả lời trực tiếp câu hỏi "is Hidden influencer a real structural category?"

### Tài liệu cần đọc

Orsini, Dankulov, Jamakovic et al. (2015), "Quantifying randomness in real networks" (Nature Communications) — dk-series null models bảo toàn degree correlations ở nhiều orders. Section về 2.5K-series null model relevant nhất cho social networks.

Fosdick, Larremore, Nishimura & Ugander (2018), "Configuring Random Graph Models with Fixed Degree Sequences" (SIAM Review) — comprehensive review configuration model variants, biases, và implementation issues. Section 4 về edge-swapping MCMC algorithm là alternative implementation.

Blondel, Guillaume, Lambiotte & Lefebvre (2008), "Fast unfolding of communities in large networks" (Journal of Statistical Mechanics) — Louvain algorithm paper gốc. Cần hiểu resolution parameter effect: resolution=1.0 (default) có thể over-partition hoặc under-partition tùy graph density. Twitch có mean degree ~81 (dense) → Louvain có thể merge communities quá aggressive.

Newman (2006), "Modularity and community structure in networks" (PNAS) — giải thích tại sao modularity optimization (Louvain dùng) có resolution limit. Trên dense graphs, communities nhỏ bị merge. Cần check modularity Q và xem community size distribution có reasonable không.

---

## 6. Bottleneck #6: Runtime và Compute Budget

### Vấn đề thực tế

5000 nodes × 200 runs trên graph 168k nodes/6.8M edges. Nếu per-simulation = 0.5ms (best case, C backend) → 500 seconds total = 8 phút. Nếu per-simulation = 5ms (Python with CSR) → 5000 seconds = 83 phút. Nếu per-simulation = 50ms (poorly optimized) → 14 giờ.

Variance lớn phụ thuộc vào implementation quality. Plan v3 dùng pure Python loop trên CSR arrays — đây không phải C backend thực sự, chỉ là Python accessing numpy arrays. Với Twitch mean degree 81, mỗi BFS step duyệt ~81 neighbors, và Python loop overhead significant.

Label stability (3 seeds × 5000 nodes × 150 runs) adds 2.25M simulations. Null model (3 realizations × 500 nodes × 100 runs) adds 150k simulations. **Total: ~3.4M simulations** (1M + 2.25M + 0.15M). Ở 5ms/sim → ~4.7 giờ; ở 50ms/sim → ~47 giờ (2 ngày liên tục).

> **Lưu ý:** Con số trên dùng n_sample=5000 và N_runs=200 (best-case budget). Với budget tier trung bình (n_sample=3000, N_runs=150), tổng giảm xuống còn ~1.8M simulations (0.45M + 1.35M + 0.15M). Budget tier tối thiểu (n_sample=2000, N_runs=100) → ~0.85M simulations.

### Những thứ cần thử nghiệm

Cần benchmark Day 1 thực sự nghiêm túc — không chỉ 100 nodes × 50 runs (5000 simulations) mà phải profile rõ: per-simulation ms decomposed thành graph access time vs random number generation vs Python loop overhead.

Nếu per-sim > 10ms, cần optimize. Ba hướng theo thứ tự effort: Numba JIT compilation (decorate IC function với @numba.jit, expect 10–50x speedup), Cython compilation (rewrite IC loop, expect 50–100x), hoặc chuyển sang NetworKit/igraph native diffusion (nếu available).

Ngoài ra cần xem xét: giảm n_runs từ 200 xuống 100 có ảnh hưởng rank stability đáng kể không? Run convergence check ở 50, 100, 150, 200 trên 200 pilot nodes để plot stability curve. Nếu Spearman ρ giữa 100 và 200 runs > 0.99, dùng 100 runs tiết kiệm 50% compute.

### Tài liệu cần đọc

Borgs, Brautbar, Chayes & Lucier (2014), "Maximizing Social Influence in Nearly Optimal Time" (SODA 2014) — introduce RIS (Reverse Influence Sampling) algorithm. Phần quan trọng nhất: **convergence analysis** cho thấy số MC simulations cần thiết để đạt (ε,δ)-approximation của IC reach. Relevant để justify lựa chọn N_runs (200 vs 100) với error bound thay vì arbitrary heuristic.

ndlib documentation (https://ndlib.readthedocs.io/) — Network Diffusion Library, có IC model implementation optimized cho large graphs. Nếu API compatible, dùng trực tiếp thay vì tự implement. **Cảnh báo:** NDlib implementation có thể vẫn dùng Python/NetworkX backend — benchmark trước khi commit.

Numba documentation — @numba.njit decorator có thể accelerate pure Python loops 10–100x. Đặc biệt effective cho BFS-style IC simulation với numpy arrays. Ưu tiên trước NDlib nếu per-sim > 10ms.

igraph documentation (https://python-igraph.org/) — igraph.Graph.neighborhood() và igraph's built-in epidemic simulation functions. Nếu convert graph sang igraph format, có thể dùng C backend cho BFS.

---

## 7. Bottleneck #7: life_time Validation Có Thể Fail

### Tại sao rất possible

life_time (account age) correlate với degree (older accounts accumulate more connections) và views (older accounts accumulate more views). Sau khi control for degree, residual correlation giữa life_time và IC score có thể rất nhỏ hoặc non-significant — đặc biệt nếu IC score đã được largely explained bởi degree (which controls for age effect indirectly).

Stratified MWU across degree quintiles cần Hidden vs Overrated to differ on life_time WITHIN each degree band. Nhưng within a degree quintile, life_time variance có thể nhỏ (similar-degree nodes có similar tenure), reducing statistical power.

Nếu life_time validation fails, paper mất external validation layer duy nhất. Typology claims resting entirely on structural metrics (betweenness, cross-community fraction) mà tất cả đều derived from same graph — no exogenous evidence.

### Những thứ cần thử nghiệm

Cần check trước (pilot): partial Spearman ρ(IC, life_time | degree) trên 200 pilot nodes. Nếu ρ < 0.05 (near zero), life_time validation sẽ gần chắc chắn fail ở full sample — cần prepare narrative ngay.

Alternative external validator: language attribute. Twitch dataset có language per node. Nếu communities align with language groups (expected) và Hidden influencers are cross-language bridges, đây là external corroboration không phụ thuộc life_time. NMI(community, language) + check xem Hidden nodes có higher language diversity trong neighborhood không.

Nếu cả life_time và language đều inconclusive, paper cần reframe: typology là descriptive framework, structural profiling results là main finding, external validation là acknowledged limitation.

### Tài liệu cần đọc

Rozemberczki, Allen & Sarkar (2021), "Multi-Scale Attributed Node Embedding" (Journal of Complex Networks) — paper gốc của Twitch dataset. Section 3 mô tả attributes bao gồm views, mature, life_time, dead_account, language. Cần hiểu chính xác definition: life_time = number of days since account creation? Active days? Total streaming hours? Interpretation phụ thuộc vào definition chính xác.

Aral & Walker (2012), "Identifying Influential and Susceptible Members of Social Networks" (Science) — randomized experiment showing influence and susceptibility are distinct. Relevant vì life_time có thể correlate với susceptibility (older users more/less susceptible) rather than influence.

---

## 8. Bottleneck #8: Paper Fit 6 Trang IEEE

### Vấn đề thực tế

Plan v3 có 4 RQs, 5 baseline groups, 4 GNN variants, null model, life_time validation, runtime table, sensitivity analysis. Đây là content cho 10+ trang, không phải 6. IEEE double-column format cho khoảng 5000 words + figures/tables. Mỗi table chiếm 1/4 trang, mỗi figure chiếm 1/3–1/2 trang.

Must-have content: pipeline figure (1/3 trang), main results table (1/4 trang), typology scatter (1/3 trang), runtime table (1/6 trang) = đã chiếm ~1 trang chỉ figures/tables. Methodology cần 1.5 trang minimum (IC setup + GNN + baselines + evaluation). Introduction + Related Work = 1 trang. Experiments cần 2 trang minimum. Discussion + Limitations = 0.5 trang. References = 0.5 trang. Total = 6.5+ trang.

### Những thứ cần quyết định sớm

**Budget từng section (target word count):**
| Section | Target | Notes |
|---------|--------|-------|
| Abstract | 150w | 1 câu/RQ |
| Introduction | 300w | Hook + gap + contributions |
| Related Work | 250w | 1 dòng/paper, 7–8 papers |
| Methodology | 700w | IC setup + typology + GNN + baselines |
| Experiments | 900w | Results + ablation + runtime |
| Discussion + Limitations | 300w | 2 paragraphs |
| References | ~30 refs | Fit ~0.5 trang 2-col |
| **Total** | **~2600w** | Leaves room for figures/tables |

Cắt bao nhiêu RQ? RQ1 (IC quality) có thể compress thành 2 câu trong Methodology. RQ4 (structural profiles) có thể merge vào RQ2 (divergence analysis). Giảm từ 4 RQ xuống **3 chính** (IC operationalization + divergence typology + GNN surrogate) là pragmatic nhất cho 6 trang — đủ specific để be rigorous nhưng không over-commit.

GNN ablation table: 4 variants × 3 metrics × mean±std = table rất lớn. Nên report trong main paper chỉ **2 key comparisons**: (1) MLP-raw-attr vs GNN-raw-attr (value of message passing), (2) GNN-graph-only vs GNN-full (value of features). Full ablation vào supplementary material nếu MAPR cho phép.

Null model results: compress thành **1 câu + 1 số**: "Configuration model comparison confirms typology reflects higher-order structure beyond degree sequence: null Hidden betweenness = X vs real = Y (p < 0.05)." Không cần full section.

### Tài liệu cần đọc

IEEE conference paper formatting guidelines (2-column template). Xem cụ thể MAPR 2025 proceedings để biết typical paper length và section structure của accepted papers. Download 2–3 MAPR 2025 papers từ IEEE Xplore để calibrate content density.

---

## 9. Bottleneck #9: Louvain Resolution Limit trên Dense Graph

### Tại sao đây là rủi ro methodological

Louvain algorithm tối ưu modularity Q — nhưng modularity có **resolution limit** (Fortunato & Barthélemy, 2007): các community nhỏ hơn một scale threshold nhất định bị merge vào community lớn hơn dù structure rõ ràng. Threshold này xấp xỉ `O(√E)` với E là số cạnh — cụ thể hơn là `√(E/2)` cho hai community dày đặc ngang nhau, hoặc `√(2E)` cho clique-like communities; exact value phụ thuộc vào mật độ nội bộ của community (xem Figure 1 của F&B 2007). Trên Twitch (`E ≈ 6.8M`), threshold này nằm trong khoảng **~1800–3700 nodes** tùy theo community density. Điểm mấu chốt: mọi community nhỏ hơn ~2000 nodes đều có nguy cơ bị absorbed — và với Twitch là dense graph (mean degree 81), nguy cơ này là **thực tế, không phải lý thuyết**.

Với `mean_degree ≈ 81` (dense graph) và `resolution=1.0`, Louvain có thể:
- **Over-merge**: nhiều tight-knit gaming sub-community (e.g., "Dota 2 streamers," "CS:GO streamers") bị gộp vào 1–2 mega-communities
- **Produce trivial partition**: vài community >10k nodes + nhiều singletons → `cross_community_edge_fraction` mất discriminative power
- **Under-detect language sub-communities**: Twitch graph có natural language-based structure (EN, DE, FR, PT, RU). Nếu EN community quá lớn, Louvain split arbitrarily thay vì theo meaningful topological seams

Nếu `community_features.parquet` có community partition không meaningful, mọi analysis dùng `cross_community_edge_fraction` (structural profiles của Hidden/Overrated, RQ4) đều có interpretation nhạy cảm với lựa chọn resolution.

### Những thứ cần thử nghiệm

**Bước 1 — Sensitivity sweep:** Chạy Louvain với 3 giá trị `resolution ∈ {0.5, 1.0, 2.0}` và report với mỗi giá trị:
- Tổng số communities
- Size distribution: (min, Q1, median, Q3, max) và % nodes trong top-3 communities
- Modularity Q
- NMI(community_label, language) — xem community có align với language groups không

**Bước 2 — Decision rule:**
- Nếu `resolution=1.0` cho < 20 communities hoặc > 50% nodes trong 3 communities lớn nhất → over-merging, tăng resolution
- Nếu `resolution=1.0` cho > 200 communities với nhiều singletons → over-splitting, giảm resolution
- Target: 30–80 communities với size distribution reasonably non-degenerate

**Bước 3 — Lock và document:** Ghi quyết định resolution vào `docs/m0_decisions.md` trước mọi downstream analysis. `community_features.parquet` phải được regenerate nếu resolution thay đổi.

**Alternative — Leiden Algorithm:** Leiden (Traag et al., 2019) sửa các artifact của Louvain (disconnected communities, non-convergence ở late iterations). Nếu `leidenalg` package available (`pip install leidenalg`), đáng chạy thử để compare partition quality.

**Bước 4 — Robustness check cho paper:** Report chính 1 resolution nhưng footnote "NMI(community, language) = X; sensitivity analysis với resolution={0.5, 2.0} cho qualitatively similar community structure profiles."

### Tài liệu cần đọc

Fortunato & Barthélemy (2007), "Resolution limit in community detection" (PNAS 104(1):36–41) — formal proof rằng modularity optimization merge communities nhỏ hơn threshold. Figure 1 cho thấy heuristic để estimate threshold. **Must-read** trước khi chọn resolution.

Traag, Waltman & van Eck (2019), "From Louvain to Leiden: guaranteeing well-connected communities" (Scientific Reports 9:5233) — Leiden algorithm, fixes Louvain's disconnected community artifact. Relevant nếu Louvain cho disconnected communities (sign of poor convergence).

Lancichinetti & Fortunato (2009), "Community detection algorithms: A comparative analysis" (Physical Review E 80:056117) — benchmark comparison trên LFR graphs. Table 1 và 2 cho realistic expectations: Louvain performs well on large networks but susceptible to resolution limit on dense ones.

Blondel, Guillaume, Lambiotte & Lefebvre (2008), "Fast unfolding of communities in large networks" (Journal of Statistical Mechanics P10008) — Louvain original paper. Appendix giải thích resolution parameter semantics và effect on partition granularity.

---

## 10. Bottleneck #10: GNN Transductive Evaluation và Deployment Claim

### Tại sao đây là vấn đề paper-level

Plan v3 dùng **transductive evaluation**: GNN trained và evaluated trên cùng graph, IC labels available chỉ cho `n_sample` nodes (train/val/test split trong tập labeled này). Tuy nhiên paper muốn claim "GNN có thể deployed để predict IC rank cho toàn bộ 168k nodes" — claim này **không được support bởi evaluation protocol hiện tại**.

Cụ thể:
- **Evaluation protocol**: GNN learns từ `n_sample ≈ 2k–5k` labeled nodes, accuracy được đo trên held-out test set trong `n_sample` này
- **Full-graph inference** (168k nodes) được report chỉ cho **runtime** assessment (speedup vs MC IC)
- **Accuracy của predictions trên 163k+ unlabeled nodes**: completely unknown và **không được measured**
- Reviewer sẽ hỏi: _"How do you know GNN predictions are accurate for unlabeled nodes? You only evaluated on the labeled subset."_

Đây là fundamental limitation của transductive learning + partial labeling. Nó không làm paper invalid nhưng nếu paper text overclaims deployment capability, reviewer sẽ reject hoặc require major revision.

### Những thứ cần quyết định trước khi viết paper

**Option A — Safe framing (đề xuất cho 25-ngày timeline):** Không claim deployment accuracy. Paper chỉ claim: _"GNN approximates IC ranking for evaluated nodes at X× inference speedup vs MC IC simulation."_ Full-graph inference chỉ dùng để compute wall-clock runtime speedup. Add explicit limitation statement: _"Accuracy of predictions for nodes outside the evaluated subset is not assessed; this remains an open question for future work."_

**Option B — Quasi-inductive evaluation (thêm effort, mạnh hơn):** Chạy IC simulation thêm 500–1000 nodes **không nằm trong train/val/test** (có thể từ degree-stratified sample ngoài `n_sample`). Evaluate GNN predictions trên batch này. Cost: 500 × 150 runs = 75k simulations (feasible). Claim: _"GNN generalizes to out-of-sample nodes with ρ ≈ X."_

**Option C — Framing chuẩn (cite precedent):** Nhiều node-level GNN papers dùng transductive setting khi label acquisition là expensive (đây là justification cho partial IC labeling). Cite Yang & Leskovec (2016) và Hamilton et al. (2017) để frame transductive evaluation là standard — reviewer khó reject nếu claim có precedent.

**Recommendation:** Option A + Option C là đường an toàn nhất. Nếu có time buffer trước Day 21, thêm Option B validation trên 500 out-of-sample nodes — cost thấp, claim strength tăng đáng kể.

### Tài liệu cần đọc

Yang & Leskovec (2016), "Revisiting Semi-Supervised Learning with Graph Embeddings" (ICML 2016) — phân tích transductive vs inductive settings cho node classification. Sections 3–4 cung cấp standard framing cho partial-label transductive evaluation. **Cite để justify evaluation protocol.**

Hamilton, Ying & Leskovec (2017), "Inductive Representation Learning on Large Graphs" (NeurIPS 2017) — GraphSAGE paper. Mặc dù GraphSAGE designed cho inductive, paper này cũng có transductive baseline cho comparison. Cite để contextualize transductive vs inductive trade-off.

Kipf & Welling (2017), "Semi-Supervised Classification with Graph Convolutional Networks" (ICLR 2017) — GCN paper, standard transductive node classification. Provide precedent rằng transductive evaluation là widely accepted khi full-graph structure is known but labels are scarce.

Zhu, Ghahramani & Lafferty (2003), "Semi-supervised learning using Gaussian fields and harmonic functions" (ICML 2003) — classical framing of semi-supervised learning: "labels are expensive; use graph structure to propagate." Helps justify why IC labels for only `n_sample` nodes is scientifically sound.

---

## Tổng hợp: Priority Matrix

| # | Bottleneck | Khả năng xảy ra | Impact nếu xảy ra | Khi nào biết | Hành động | Đã xử lý trong plan? |
|---|---|---|---|---|---|---|
| B1 | One-hop dominate IC | Trung bình-Cao | **Restructure paper** | Day 1 chiều | Pilot 200 nodes trước mọi thứ | ✅ 3-metric gate trong M2 |
| B2 | IC cascade degenerate (global) | Thấp-Trung bình | Labels vô nghĩa nếu global CV fail | Day 1–2 | CV check + histogram + **per-quintile CV** | ✅ pilot diagnostics JSON |
| B2′ | Within-quintile noise (Q1–Q4) | **Cao** | Limitation trong paper, GNN fine-ranking fail | Day 1–2 | Per-quintile CV diagnostic | ⚠ Chưa có trong plan — thêm vào Day-1 report |
| B3 | Typology quadrant quá nhỏ | Trung bình-Cao | Statistical power thấp (MWU) | Day 3–4 | Estimate ρ(views,IC) từ pilot sớm | ⚠ Plan có two-sample strategy nhưng chưa có power pre-check |
| B4 | GNN không beat proxy | Trung bình | Thấp (publishable với fallback) | Day 15–18 | Fallback narrative sẵn sàng | ✅ Fallback narratives trong plan |
| B5 | Null model inconclusive | Trung bình | Thấp — compress thành 1 câu | Day 16–18 | Permutation null thêm vào B5 | ⚠ Permutation null không có trong plan |
| B6 | Runtime vượt budget | Trung bình | **Delay pipeline** | Day 1 sáng | Benchmark 100×50 trước tiên | ✅ Budget tiers trong plan |
| B7 | life_time validation fail | Trung bình-Cao | Mất external validation | Day 17–18 | Test partial ρ trên pilot | ⚠ Language fallback chưa có trong plan |
| B8 | Paper > 6 trang | Cao | **Blocker submission** | Day 21+ | Quyết định cắt RQ sớm (Day 5) | ⚠ Không có explicit cut plan |
| B9 | Louvain resolution sai | Trung bình | Community features vô nghĩa | Day 5–6 | Resolution sweep {0.5,1.0,2.0} | ⚠ Plan lock resolution=1.0 nhưng không có sensitivity check |
| B10 | GNN deployment claim quá mạnh | Trung bình | Reviewer reject/major revision | Paper writing | Framing cẩn thận + cite precedent | ⚠ Chưa có explicit framing note trong plan |

### Chú giải cột "Đã xử lý trong plan?"
- ✅ = mitigation đã có trong `MAPR2026_v3_team_parallel_coding_plan.md` hoặc `MAPR2026_Implementation_Plan_v3.md`
- ⚠ = bottleneck được identify nhưng plan chưa có explicit action item — cần team chủ động handle

### Bottlenecks cần action ngay (Day 1–2):
1. **B1**: Measure `one_hop_rho`, `jaccard_at_10pct`, `ndcg_at_10pct` → write to `one_hop_correlation.json`
2. **B2′**: Per-quintile CV → ghi vào `ic_pilot_diagnostics.json` hoặc `docs/day1_decisions.md`
3. **B6**: `per_sim_ms` benchmark → decision tier → lock `n_sample`, `N_runs`

### Bottlenecks cần quyết định trước Day 8 (paper writing start):
4. **B8**: Quyết định structure 2 RQ hay 4 RQ — commit trước khi viết Section 3
5. **B9**: Lock Louvain resolution vào `docs/m0_decisions.md` trước khi Person 2 runs community detection
6. **B10**: Agree on framing: transductive evaluation + cite Yang & Leskovec (2016)


---

> ⚠ **[REVIEW PHASE ADDENDUM — đọc trước khi dùng phần này]**
>
> Phần dưới đây là đánh giá chi tiết từng bottleneck và các "missing bottleneck" được identify trong quá trình cross-analysis với 3 file MAPR2026. **Lưu ý numbering:**
>
> - "Bottleneck 9–13" trong phần này là **analysis-phase identifiers** (missing gaps tìm thấy khi review), **KHÔNG** cùng số thứ tự với Bottleneck #9 và #10 đã được bổ sung vào danh sách chính ở trên.
> - **Bottleneck 9 (Inconsistency giữa các files)** — ✅ RESOLVED: tất cả inconsistencies (GNN-raw-attr independence, n_sample vs N_seeds, views vs views_log, community_features separate file) đã được fix trong docs.
> - **Bottleneck 12 (Louvain resolution sensitivity)** — ✅ Đã bổ sung thành **Bottleneck #9** đầy đủ trong danh sách chính ở trên.
> - **Bottleneck 10 (Sampling bias)** — vẫn còn relevant, xem B3 và B10 trong danh sách chính.
> - **Bottleneck 11 (Label uncertainty weighting)** — nice-to-have, không trong scope 25 ngày; ghi limitation thay thế.
> - **Bottleneck 13 (Pilot gate thin)** — vẫn còn relevant: nếu pilot ρ ∈ [0.78, 0.92], rerun trên 500 nodes.

---

Dưới góc nhìn **reviewer SNA/graph learning**, mình đánh giá:

## Kết luận ngắn

**File Bottleneck docs là tốt, đúng hướng và hữu ích thực chiến. Khoảng 75–85% các bottleneck được nêu là hợp lý và bám rất sát rủi ro thật của bộ MAPR2026 v3.**  
Đặc biệt, các bottleneck mạnh nhất là:

1. **one-hop proxy có thể gần như thay IC**
2. **IC label có thể bị degenerate**
3. **quadrant Hidden/Overrated có thể quá nhỏ**
4. **runtime của MC IC là bottleneck compute lớn nhất**
5. **6 trang IEEE không đủ nếu không cắt story sớm**

Tuy nhiên, **vẫn có một số chỗ cần chỉnh/giảm overclaim**, và cũng có **một vài bottleneck quan trọng chưa được nêu đủ rõ**.  
Ngoài ra, giữa các file MAPR2026 vẫn còn vài **inconsistency nội bộ** mà nếu không sửa thì sẽ làm bottleneck trở nên “self-inflicted”.

---

# 1. Đánh giá tổng quan Bottleneck docs so với 3 file MAPR2026

## 1.1 Nhận định chung
Bottleneck docs **không đi chệch plan v3**, ngược lại nó giúp plan v3 thực tế hơn.  
Nó đúng ở chỗ:

- không giả định GNN sẽ thắng
- coi one-hop check là **gating decision**
- xem runtime là **critical-path**
- nhìn thấy rủi ro statistical power của typology
- cảnh báo null model/configuration model có thể chưa đủ
- cảnh báo external validation bằng `life_time` có thể fail
- nhìn đúng bài toán page-budget 6 trang

## 1.2 Nhưng có 3 điểm cần hiệu chỉnh lớn
### (A) Một số claim đang **hơi quá mạnh**
Ví dụ:
- “one-hop dominate toàn bộ IC ranking”
- “GNN story sụp đổ”
- “configuration model không informative”

Những câu này **đúng như cảnh báo**, nhưng nên viết là **empirical risk / gating condition**, không phải định đề chắc chắn.

### (B) Một số remedy trong Bottleneck docs còn **chưa tối ưu cho 25 ngày**
Ví dụ:
- đề xuất three-hop / nhiều null model phức tạp / dk-series
- thêm directed reinterpretation của Twitch
- nhiều sensitivity layer quá sâu

Về mặt khoa học thì hay, nhưng với MAPR 25 ngày thì cần phân tầng:
- **must-have**
- **should-have**
- **nice-to-have**

### (C) Có vài bottleneck quan trọng **chưa được chỉ ra hoặc chưa nói đủ**
Mình sẽ bổ sung ở phần 4.

---

# 2. Đánh giá từng bottleneck: đúng tới đâu, cần sửa gì

---

## Bottleneck 1 — One-hop spread có thể dominate IC ranking

## Đánh giá
**Đúng và rất quan trọng.** Đây là bottleneck đúng nhất trong cả tài liệu.

Với weighted cascade:
\[
p(u,v)=1/\deg(v)
\]
trên một graph undirected và khá dense như Twitch, hoàn toàn có khả năng:
- cascade chủ yếu sống trong 1–2 hops
- one-hop analytical proxy có Spearman rất cao với MC IC

Điều này phù hợp với trực giác từ IM literature:
- local structure thường rất mạnh trong low-probability diffusion regime  
(Kempe et al., 2003; Chen, Wang & Wang, 2010; Lü et al., 2016).

## Điểm cần chỉnh
### 1) Không nên coi đây là “gần như chắc chắn”
Nó là **gating question**, không phải assumption.

### 2) Spearman một mình là chưa đủ để “kill GNN story”
Cần thêm ít nhất:
- **Jaccard@top-10%**
- hoặc **NDCG@10%**
- hoặc **Precision@10%**

Vì có thể:
- Spearman toàn cục rất cao
- nhưng top influencer set vẫn khác đáng kể

### 3) Two-hop complexity đang bị understated ở vài chỗ
Trong docs có đoạn ngầm coi two-hop vẫn “rẻ”.  
Thực ra full-graph naive two-hop có complexity gần:

\[
\sum_{v} d(v)^2
\]

chứ **không phải O(E)** theo nghĩa đẹp.  
Trên heavy-tailed graph, đây có thể khá đắt.

## Đề xuất sửa
### Gating decision cho Day 1 nên là:
- **Spearman ρ**
- **Jaccard@10%**
- **NDCG@10%**

Ví dụ:
- nếu ρ > 0.9 **và** Jaccard@10% > 0.8 → GNN không còn là headline
- nếu ρ > 0.9 nhưng Jaccard@10% thấp → vẫn còn lý do giữ GNN

### Baseline nên bổ sung
Ngoài one-hop / two-hop, có thể cân nhắc 1 baseline “local-global bridge”:
- **LocalRank** (Chen et al., 2012)
- hoặc **Collective Influence** (Morone & Makse, 2015)

Nếu kịp, mình nghiêng về **LocalRank** hơn vì dễ frame hơn cho spreading proxy.

## Tài liệu phù hợp
- Kempe, Kleinberg & Tardos (2003), KDD — IC model foundation
- Chen, Wang & Wang (2010), KDD — PMIA / local influence approximation
- Lü et al. (2016), *Physics Reports* — survey vital nodes
- Chen et al. (2012), *Physica A* — LocalRank
- Morone & Makse (2015), *Nature* — Collective Influence

---

## Bottleneck 2 — IC cascade distribution có thể degenerate

## Đánh giá
**Đúng**, nhưng phần hiện tại cần tinh chỉnh.

Weighted cascade có thể cho:
- median reach rất thấp
- distribution lệch mạnh
- nhiều node gần như reach = 1
- ranking noisy ở boundary

Đây là risk thật.

## Điểm cần chỉnh
### 1) `CV > 0.3` không nên là điều kiện duy nhất
CV hữu ích, nhưng với mean thấp thì CV rất dễ méo.  
Nếu median reach gần 1, CV có thể đánh lừa.

### 2) Không nên dùng “mean p ≈ 1/81” để suy ra regime quá mạnh
Twitch có heavy-tailed degree. Với weighted cascade, important thing không phải mean degree đơn giản, mà là:
- degree distribution
- harmonic structure
- hub-to-leaf interaction
- component structure / clustering

Nói cách khác: **mean degree alone không đủ** để kết luận cascade chắc chắn chết nhanh.

## Đề xuất sửa
Thay vì chỉ dùng:
- mean
- median
- IQR
- CV

hãy thêm 3 diagnostic mạnh hơn:

### (a) Proportion of non-trivial cascades
\[
P(\text{reach} > 1), \quad P(\text{reach} > 5)
\]

### (b) Tail separation
\[
\text{top-10\% mean reach} / \text{median reach}
\]

### (c) Pairwise ordering stability
Spearman hoặc Kendall giữa:
- 50 vs 100 runs
- 100 vs 150 runs
- 150 vs 200 runs

Nếu ordering ổn định thì label vẫn usable dù median thấp.

### (d) Gini coefficient hoặc entropy của reach distribution
Nếu có thời gian, Gini rất hợp để đo “phân hóa influence proxy”.

## Điểm framing quan trọng
**Median reach = 1 không tự động làm paper chết.**  
Nếu tail tách đủ mạnh và top-k ordering ổn định, ranking task vẫn có ý nghĩa.

## Tài liệu phù hợp
- Kitsak et al. (2010), *Nature Physics*
- Pastor-Satorras & Vespignani (2001), *PRL* — epidemic threshold
- Lü et al. (2016), *Physics Reports*
- Ling et al. (2023), DeepIM — weighted cascade usage

---

## Bottleneck 3 — Typology quadrant sizes và statistical power

## Đánh giá
**Rất đúng**. Đây là bottleneck thật cho Task B.

Nếu \(\rho(\text{views}, \text{IC})\) cao:
- Hidden và Overrated sẽ rất nhỏ
- MWU theo degree quintile sẽ thiếu power
- Cliff’s delta khó significant sau BH correction

## Điểm cần chỉnh
### 1) Chỉ tăng sample size chưa chắc đủ
Nếu correlation quá cao, tăng từ 5k lên 10k có thể vẫn không cứu được nhiều.

### 2) Two-sample strategy hợp lý nhưng phải được frame cẩn thận
Sample B là **targeted sample**, nên:
- dùng cho **profiling / illustration**
- không dùng cho unbiased prevalence claims
- không dùng để report population rates như thể representative

## Đề xuất sửa
### Backup analysis nên có thêm một nhánh threshold-free
Nếu quadrants nhỏ, dùng thêm:
- **residual analysis**
  - residual = standardized(IC rank) − standardized(views rank)
- hoặc rank-difference score

Rồi define:
- Hidden-like = residual top decile
- Overrated-like = residual bottom decile

Cách này giúp:
- giảm phụ thuộc vào top-10 × top-10 hard cut
- tăng sample size của nhóm phân kỳ

### Nên report thêm
- quadrant counts + CI đơn giản
- sensitivity at top-10 và top-20 (chỉ nếu cần, không cần 5/10/15/20 đầy đủ)

## Tài liệu phù hợp
- Cha et al. (2010), ICWSM — *The Million Follower Fallacy*
- Bakshy et al. (2011), WSDM — influence variability
- Cohen (1988) — power/effect size background
- Benjamini & Hochberg (1995)

---

## Bottleneck 4 — GNN feature leakage / ablation interpretation

## Đánh giá
**Đúng một nửa, cần sửa ngôn ngữ.**

Ở đây nên phân biệt:

- **Leakage thật**: feature chứa thông tin được derive trực tiếp từ target construction
- **Confounding / fairness issue**: feature là strong proxy của label vì cùng phụ thuộc graph structure

Với plan v3:
- dùng centrality features để predict IC labels **không hẳn là leakage**
- nhưng nếu làm vậy thì claim “message passing học higher-order structure beyond hand-crafted features” sẽ yếu

## Điểm rất quan trọng cần sửa trong v3
### Có lỗi nội bộ trong v3:
Ở phần paper structure / independence matrix có dòng kiểu:

> GNN-raw-attr features | Views-independent? ✅ Yes (no views)

Điều này **sai**.  
Vì `gnn_raw_attr` trong config đang dùng:
- `views_log`
- `views_per_day`
- `life_time`

=> **không thể ghi là views-independent**.

## Đề xuất sửa
### 1) Đổi wording bottleneck
Từ “feature leakage” thành:
> **feature fairness / confounding with hand-crafted structural summaries**

### 2) Thêm một baseline rất đáng giá:
- **GNN-constant-features** hoặc **GNN-random-features**

Lợi ích:
- test pure topology + message passing
- tách rõ vai trò của node attributes

### 3) Sắp xếp narrative cho GNN nên là:
- **Primary comparison**: MLP-raw-attr vs GNN-raw-attr
- **Ablation**: GNN-graph-only / GNN-centrality / GNN-full

Cách này sạch hơn so với lấy GNN-full làm trọng tâm.

## Tài liệu phù hợp
- Hamilton et al. (2017), GraphSAGE
- Xu et al. (2019), GIN / expressiveness
- Errica et al. (2020), fair comparison of GNNs
- You et al. (2019), P-GNN

---

## Bottleneck 5 — Null model: configuration model có thể không đủ informative

## Đánh giá
**Đúng.** Đây là critique hợp lý.

Configuration model chỉ bảo toàn degree sequence, nhưng phá:
- clustering
- modularity/community
- core-periphery
- assortativity

Nên nếu null khác xa real graph thì kết luận “higher-order structure matters” cần viết khiêm tốn hơn.

## Điểm cần chỉnh
### 1) Không nên nói configuration model “vô dụng”
Nó vẫn là **first-order null** rất hợp lý nếu claim của bạn là:
> “degree sequence alone cannot explain the observed typology.”

### 2) Nhưng không nên over-interpret
Nếu real vs null khác nhau, kết luận đúng nhất là:
> **degree sequence alone is insufficient**

chứ không phải đã chứng minh đầy đủ mọi higher-order structure.

## Đề xuất sửa
### Null model nên có 2 tầng nếu kịp
#### Tier 1 — bắt buộc
- **degree-preserving null**
  - configuration model **hoặc**
  - degree-preserving edge swaps

#### Tier 2 — rất nên có nếu còn sức
- **views-permutation null**
  - giữ graph + IC score cố định
  - permute `views` across nodes
  - rebuild typology
  - kiểm tra Hidden/Overrated structure còn không

Đây là null rất mạnh và rẻ compute.  
Nó trực tiếp trả lời:
> “phân kỳ views–IC có phải chỉ là artifact của marginal views distribution không?”

### Community sensitivity
Nếu dùng Louvain, nên report tối thiểu:
- số communities
- modularity Q
- resolution = 1.0
- maybe một quick sensitivity 0.5 / 1.0 / 2.0 nếu kịp

## Tài liệu phù hợp
- Fosdick et al. (2018), *SIAM Review* — configuration model review
- Orsini et al. (2015), *Nature Communications* — dk-series/randomness
- Newman (2006), PNAS — modularity
- Blondel et al. (2008), Louvain

---

## Bottleneck 6 — Runtime và compute budget

## Đánh giá
**Đúng hoàn toàn**, và thậm chí có thể vẫn đang hơi lạc quan.

Đây là bottleneck implementation lớn nhất.

## Điểm cần chỉnh
### 1) `CSR + Python loop` chưa phải “C backend”
Điều này phải nói thật rõ.

Trong v3 có chỗ phrasing làm người đọc có thể hiểu như:
- CSR + numpy = rất nhanh như C

Không đúng.  
Nếu inner loop vẫn là Python thì vẫn rất có thể chậm.

### 2) `joblib loky` không tự động đảm bảo shared memory tối ưu
Nếu không memmap rõ ràng, large arrays vẫn có thể tốn RAM hoặc serialize nhiều.

### 3) Two-hop full-graph cũng có thể là bottleneck không nhỏ
Bottleneck docs có nhắc nhưng nên nhấn mạnh hơn.

## Đề xuất sửa
### Must-have engineering changes
1. **Benchmark thật ngày 1**: đúng như docs
2. Nếu per-sim quá chậm:
   - ưu tiên **Numba JIT (`@njit`)**
   - hoặc Cython / C++ backend
3. Với `joblib loky`:
   - explicit memmap / shared read-only arrays
4. Tránh dùng Python `set` trong inner loop nếu có thể
   - dùng boolean mask / preallocated arrays

### Cảnh báo nhỏ về NDlib
Bottleneck docs gợi ý NDlib là hướng đáng xem.  
**Mình khuyên cẩn thận**: NDlib hữu ích về API, nhưng historical implementations của nó không hẳn là lựa chọn nhanh nhất cho workload kiểu này nếu backend vẫn xoay quanh Python/NetworkX.

Nếu mục tiêu là speed:
- **Numba**
- **igraph / graph-tool**
- hoặc custom compiled loop  
thường đáng tin hơn.

## Tài liệu phù hợp
- DeepIM (2023) cho weighted cascade setup
- NDlib docs: nên coi là exploratory, không phải mặc định giải pháp nhanh nhất
- Numba docs / igraph docs: engineering guidance, không nhất thiết cần cite trong paper

---

## Bottleneck 7 — `life_time` external validation có thể fail

## Đánh giá
**Đúng**, và đây là cảnh báo nên giữ nguyên.

`life_time` nhiều khả năng correlate với:
- degree
- views
- tenure
- account maturity

nên sau khi control, signal còn lại có thể rất yếu.

## Điểm cần chỉnh
### 1) Không nên đặt quá nhiều kỳ vọng vào `life_time`
Nó là:
- **corroboration**
- không phải validation mạnh

### 2) Nếu GNN primary dùng `life_time`, không được dùng `life_time` để validate GNN predictions
Điểm này v3 đã hiểu, nhưng nên nhấn mạnh hơn.

## Đề xuất sửa
### Nếu raw data có các field này, có thể cân nhắc thêm:
- `language`
- `mature`
- `created_at` / `updated_at` derived recency
- neighborhood language diversity

Nhưng chỉ dùng nếu thật sự có và sạch.

### Nếu không có external variable tốt hơn:
hãy chuẩn bị từ đầu câu limitation:
> “External corroboration is limited because all available covariates in the Twitch snapshot are themselves entangled with tenure and popularity.”

## Tài liệu phù hợp
- Rozemberczki et al. (2021) — dataset semantics
- Aral & Walker (2012), *Science*
- Bakshy et al. (2011), WSDM

---

## Bottleneck 8 — 6 trang IEEE không đủ

## Đánh giá
**Đúng 100%**.  
Bottleneck docs phân tích rất đúng chỗ này.

Hiện v3 vẫn có xu hướng muốn kể quá nhiều:
- IC operationalization
- stability
- typology
- null model
- structural profiling
- external validation
- 5 baseline groups
- 4 GNN variants
- runtime
- sensitivity

=> chắc chắn chật.

## Đề xuất sửa
### Core story của paper nên ép về 3 câu hỏi thôi:
#### RQ1
Weighted-cascade IC có đủ stable/discriminative để dùng làm influence proxy không?

#### RQ2
Views và IC rank khác nhau như thế nào? Hidden/Overrated có profile cấu trúc ra sao?

#### RQ3
Analytical proxies và GNN xấp xỉ IC tốt đến đâu, với trade-off accuracy/runtime ra sao?

### Cắt/merge
- merge RQ4 vào RQ2
- đưa nhiều sensitivity vào appendix / omit
- main paper chỉ giữ:
  - 1 main results table
  - 1 typology figure
  - 1 runtime mini-table

---

# 3. Những điểm trong Bottleneck docs cần sửa trực tiếp

Dưới đây là các chỉnh sửa mình đề xuất cho chính file Bottleneck docs.

---

## 3.1 Sửa Bottleneck 1
### Từ:
> one-hop spread có thể dominate toàn bộ IC ranking

### Thành:
> one-hop spread is a critical **empirical gating baseline** under weighted cascade and may match MC IC surprisingly well, especially in low-hop regimes. This must be tested using both global rank metrics (Spearman/Kendall) and top-k metrics (NDCG/Jaccard@k).

---

## 3.2 Sửa Bottleneck 2
### Từ:
> nếu CV < 0.3 thì labels gần như vô nghĩa

### Thành:
> low CV is a warning sign, but not sufficient to conclude the ranking is unusable. Label usefulness should be assessed jointly by distributional dispersion, tail separation, and rank stability across MC run counts.

---

## 3.3 Sửa Bottleneck 4
### Từ:
> GNN feature leakage

### Thành:
> fairness/confounding in GNN feature design

Và thêm note:
> centrality-derived inputs are legitimate for ablation, but they weaken any claim that message passing alone contributes beyond hand-crafted structural summaries.

---

## 3.4 Sửa Bottleneck 5
### Từ:
> configuration model có thể không informative

### Thành:
> configuration model is a valid degree-sequence null, but it supports only a **first-order** interpretation (“degree alone is insufficient”). It should not be over-interpreted as a full higher-order structural null.

---

## 3.5 Sửa Bottleneck 6
Thêm câu:
> two-hop spread on the full graph is not O(E) in the naive implementation; its total complexity is closer to \(O(\sum_v d(v)^2)\), which can be substantial on heavy-tailed networks.

---

# 4. Những bottleneck còn thiếu hoặc chưa nói đủ

Đây là phần mình thấy quan trọng nhất để bổ sung.

---

## Bottleneck 9 — Inconsistency giữa các file MAPR2026
Hiện giữa v3, migration checklist và team plan vẫn có một số mâu thuẫn nhỏ nhưng nguy hiểm:

### Ví dụ:
1. **GNN-raw-attr có views nhưng independence matrix lại ghi views-independent**
2. typology chỗ dùng `views_log`, chỗ dùng `views`
3. `N_seeds` đôi lúc bị dùng để chỉ:
   - số labeled nodes
   - lúc khác lại nghe như số random seeds
4. community features lúc ghi merge vào `node_attributes`, lúc ghi file riêng
5. full-graph inference vs held-out labeled evaluation: có nơi rõ, có nơi mơ hồ

## Khuyến nghị
Tạo một file **`docs/schema_lock.md`** hoặc sửa `_shared.py` + M0 decisions để khóa cứng:
- typology dùng **raw views**
- GNN-raw-attr **not views-independent**
- `n_labeled_nodes` thay cho `N_seeds`
- community features = file riêng
- metrics = test labeled only; runtime = full graph only

---

## Bottleneck 10 — Sampling bias của labeled subset
Bottleneck docs có nói quadrant size, nhưng chưa nhấn mạnh đủ **selection bias**.

Nếu label chỉ có trên 2k–5k nodes sampled theo degree quintile, thì:
- surrogate evaluation phản ánh sampled population
- không tự động phản ánh full graph population

## Khuyến nghị
- giữ representative Sample A
- nếu augment Sample B thì **không dùng nó cho surrogate metrics**
- trong paper ghi rõ:
  > “All predictive metrics are estimated on the representative labeled subset; targeted augmentation is used only for descriptive typology analysis.”

Nếu kịp, thêm:
- community-aware sanity check cho sample

---

## Bottleneck 11 — Label uncertainty chưa được tận dụng trong training
V3 có bootstrap CI cho IC scores, nhưng chưa tận dụng chúng cho model training.

## Khuyến nghị
Nếu còn thời gian:
- dùng **sample weights**
\[
w_u = \frac{1}{\text{CI width}_u + \epsilon}
\]
hoặc inverse variance weighting trong regression loss.

Nếu không làm, ít nhất:
- report uncertainty bands
- flag nodes near top-10 threshold with high uncertainty

Đây là cách tăng defensibility mà không cần thêm model mới.

---

## Bottleneck 12 — Louvain resolution sensitivity
Bottleneck docs có nói null model và community structure, nhưng chưa nói rõ:

- resolution = 1.0 là arbitrary
- trên dense graph, Louvain có thể merge communities quá mạnh

## Khuyến nghị nhẹ
Không cần full sensitivity study. Chỉ cần:
- report number of communities
- modularity Q
- maybe một quick check resolution 0.5 / 1.0 / 2.0 trên subgraph hoặc whole graph nếu rẻ

Nếu cross-community fraction của Hidden vẫn cao ổn định thì claim mạnh hơn.

---

## Bottleneck 13 — One-hop gating trên 200 pilot nodes có thể hơi mỏng
200 nodes là ổn cho smoke test, nhưng nếu ρ rơi vào vùng sát ngưỡng 0.8–0.9 thì quyết định narrative có thể bị nhạy.

## Khuyến nghị
- nếu pilot rho ở vùng **0.78–0.92**, rerun thêm với **500 nodes** trước khi chốt narrative
- đặc biệt nếu paper direction phụ thuộc hoàn toàn vào ngưỡng đó

---

# 5. Đề xuất tài liệu/paper bổ sung theo từng bottleneck

Dưới đây là bộ tài liệu mình khuyên giữ/prioritize.

---

## Nhóm A — diffusion / local-vs-global influence
1. **Kempe, Kleinberg & Tardos (2003)** — KDD  
   Foundation của IC / influence maximization.

2. **Chen, Wang & Wang (2010)** — KDD  
   PMIA; rất hữu ích để justify local approximation.

3. **Lü et al. (2016)** — *Physics Reports*  
   Survey mạnh nhất cho vital nodes / spreading.

4. **Chen et al. (2012)** — *Physica A*  
   LocalRank; hợp để giải quyết bottleneck one-hop vs broader local proxy.

5. **Morone & Makse (2015)** — *Nature*  
   Collective Influence; tốt nếu muốn thêm 1 baseline local-global bridge.

---

## Nhóm B — spreading regimes / sensitivity
6. **Kitsak et al. (2010)** — *Nature Physics*  
   k-shell vs degree in spreading.

7. **Pastor-Satorras & Vespignani (2001)** — *PRL*  
   Epidemic threshold intuition.

8. **Ling et al. (2023), DeepIM** — ICML  
   Dùng để justify weighted cascade setup, **không dùng cho 8% single-seed calibration**.

---

## Nhóm C — popularity vs influence divergence
9. **Cha et al. (2010)** — ICWSM  
   *The Million Follower Fallacy*

10. **Bakshy et al. (2011)** — WSDM  
   empirical variability of influence

11. **Aral & Walker (2012)** — *Science*  
   social ties and influence pathways, useful for construct-validity discussion

---

## Nhóm D — GNN fairness / expressiveness
12. **Hamilton et al. (2017)** — NeurIPS  
   GraphSAGE

13. **Xu et al. (2019)** — ICLR  
   How Powerful Are GNNs?

14. **Errica et al. (2020)** — ICLR  
   fair comparison / benchmark mindset

15. **You et al. (2019)** — ICML  
   Position-aware GNNs; useful for limits of local message passing

---

## Nhóm E — null models / community
16. **Fosdick et al. (2018)** — *SIAM Review*  
   fixed-degree random graph models

17. **Orsini et al. (2015)** — *Nature Communications*  
   dk-series/randomness in real networks

18. **Blondel et al. (2008)** — Louvain

19. **Newman (2006)** — modularity and community structure

---

## Nhóm F — stats
20. **Benjamini & Hochberg (1995)**
21. **Cohen (1988)** — power/effect size background

---

# 6. Khuyến nghị sửa Bottleneck docs theo mức ưu tiên

> **Status update:** Danh sách dưới đây phản ánh trạng thái sau khi đã update file. ✅ = đã xử lý, ⚠ = còn open.

## Must-fix
1. ✅ **Sửa lại Bottleneck 1**: dùng thêm top-k metrics (Jaccard@10%, NDCG@10%), không chỉ Spearman — đã có trong 3-metric gate
2. ✅ **Sửa lại Bottleneck 2**: CV không phải criterion duy nhất — đã thêm Jensen's inequality correction + per-quintile CV diagnostic
3. ✅ **Sửa lại Bottleneck 4**: independence matrix đã sửa GNN-raw-attr = ❌ views-independent trong v3 plan
4. ✅ **Sửa lại Bottleneck 5**: configuration model = first-order null — note đã có trong B5 text
5. ✅ **Sửa lại complexity của two-hop**: `O(Σ d(v)²)` đã cập nhật trong cả 3 files
6. ✅ **Inconsistency giữa các MAPR2026 files** — tất cả đã RESOLVED trong session trước
7. ⚠ **Sampling bias của labeled subset** — vẫn cần explicit mention trong paper limitations (xem B3 + B10)

## Strongly recommended
8. ⚠ **views-permutation null** — mentioned in B5 nhưng KHÔNG có trong plan; cần team xem xét thêm
9. ⚠ **residual-based divergence analysis** làm backup nếu quadrants nhỏ — không có trong plan; good backup nếu B3 materializes
10. ⚠ **GNN-random/constant-features** baseline — mentioned in B4 nhưng không có trong plan; thêm nếu có time
11. ⚠ **uncertainty-aware training** note — out of 25-day scope; chỉ ghi limitation

## Nice-to-have
12. ✅ **Louvain resolution sensitivity** — đã bổ sung thành Bottleneck #9 đầy đủ trong danh sách chính
13. ⚠ community-aware sample sanity check — nhẹ, thêm nếu B9 trigger cần rerun
14. ⚠ one-hop gate rerun trên 500 pilot nodes nếu near threshold (pilot ρ ∈ [0.78, 0.92])

---

# 7. Kết luận cuối cùng

## Đánh giá của mình
**Bottleneck docs nhìn chung là đúng, hợp lý, và rất có giá trị thực chiến cho project này.**  
Nó đặc biệt tốt ở việc:
- phá ảo tưởng “GNN chắc sẽ thắng”
- nhìn đúng bottleneck compute
- nhìn đúng bottleneck page-limit
- nhìn đúng statistical fragility của typology

Tuy nhiên, để align hoàn toàn với **MAPR2026 plan v3**, mình khuyên:

### Cần chỉnh ngay:
- giảm các câu over-strong thành empirical gate
- sửa một số hiểu nhầm về complexity / null model interpretation
- bổ sung bottleneck về **sample bias**, **file inconsistency**, **uncertainty use**
- thêm 1–2 null/backup analysis rẻ mà mạnh:
  - **views permutation null**
  - **residual-based divergence**

