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
>
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
- **P(reach >= 5)**: tỷ lệ cascades spread được ≥ 5 nodes. Nếu P(reach >= 5) < 0.05 → hầu hết influence chỉ trong immediate neighborhood.
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

**Fallback Paper Narrative nếu cả GNN-raw-attr VÀ GNN-graph-only đều không beat MLP-raw-attr (tức là message passing không giúp ích):**

_(a) Adjusted claim (thay thế claim "GNN superior"):_

> "GNN-based models did not outperform feature-only baselines (MLP-raw-attr) on influence prediction, suggesting that local neighborhood structure — as captured by message-passing at this graph scale — does not provide additional signal beyond node-level degree features under the weighted cascade model."

_(b) Mandatory limitation sentence (bắt buộc đưa vào Discussion hoặc Conclusions):_

> "We note that the GNN was trained on a transductive split with n_sample ≪ |V|; performance may improve with full-graph inductive training or with larger labeled sets, which we leave for future work."

_(c) Những gì vẫn giữ trong paper — đây là finding defensible, không phải failure:_

- **Table 3 runtime comparison** (GNN inference speedup vs MC IC) → **vẫn giữ nguyên** — speedup claim không phụ thuộc vào accuracy của GNN
- **Ablation table** (MLP-raw-attr vs GNN-raw-attr vs GNN-graph-only) → **vẫn báo cáo đầy đủ** — cho thấy topology không giúp ích ở scale Twitch, đây là finding về regime của graph learning
- **RQ3 answer**: "Topology-aware GNN does not improve over topology-free baseline at Twitch scale under transductive training" — câu này IS a defensible paper contribution vì nó resolves an open empirical question
- **Framing chuyển hướng**: Focus narrative sang "IC-based GNN speedup" (runtime benefit) thay vì "IC-based GNN accuracy" (predictive benefit)

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

> ⚠ **[NICE-TO-HAVE — Future work. OUT OF SCOPE cho 25-ngày execution.]** Cần thử dk-series graph hoặc stochastic block model thay vì configuration model — những null models này bảo toàn thêm clustering hoặc community structure, tạo so sánh fair hơn. Tuy nhiên implementation phức tạp hơn và hoàn toàn ngoài scope 25 ngày. Bỏ qua trong execution hiện tại; chỉ ghi là future work trong paper.

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

## Bảng Phân Tầng Ưu Tiên (Must / Should / Nice)

| Bottleneck | Tên                  | Tier       | Lý do                                                         |
| ---------- | -------------------- | ---------- | ------------------------------------------------------------- |
| B1         | One-Hop ρ Check      | **MUST**   | Day-1 gate — quyết định toàn bộ narrative GNN                 |
| B2         | Weighted Cascade CV  | **MUST**   | Validates IC metric trước khi labeling                        |
| B2′        | Per-Quintile CV      | **MUST**   | Sub-experiment của B2, primary diagnostic bắt buộc            |
| B3         | Views/IC Divergence  | **MUST**   | RQ2 core claim — typology validity                            |
| B4         | Feature Fairness     | **MUST**   | Ablation cho GNN paper claim; MLP-raw-attr baseline           |
| B5         | Typology Null Models | **MUST**   | Null 2 (IC-perm) + Null 3 (views-perm) đều cần                |
| B6         | MC Simulation Budget | **MUST**   | Day-1 runtime benchmark quyết định n_sample + N_runs          |
| B7         | life_time Validation | **MUST**   | External validation; prepare fallback narrative sớm           |
| B8         | Paper Writing Budget | **SHOULD** | Planning tool — không ảnh hưởng results nhưng prevent overrun |
| B9         | Louvain Resolution   | **SHOULD** | Sensitivity sweep cần thiết; nếu skip → footnote kết quả      |
| B10        | GNN Eval Gap         | **SHOULD** | Framing claim — bắt buộc address nếu submit conference        |
| dk-series  | dk-series typing     | **NICE**   | Future work — OUT OF SCOPE 25 ngày                            |

> **Quy tắc tối giản khi bị áp lực thời gian:** Drop theo thứ tự B8 → B9 → B10. **B1–B7 là core, không được bỏ.** Nếu B9 bị skip, ghi vào limitations: "Louvain resolution=1.0 without sensitivity analysis." Nếu B10 bị skip, dùng Option A+C framing trong paper (cite Yang & Leskovec 2016, Kipf & Welling 2017).

---

## Tổng hợp: Priority Matrix

| #   | Bottleneck                     | Khả năng xảy ra | Impact nếu xảy ra                             | Khi nào biết  | Hành động                                  | Đã xử lý trong plan?                                        |
| --- | ------------------------------ | --------------- | --------------------------------------------- | ------------- | ------------------------------------------ | ----------------------------------------------------------- |
| B1  | One-hop dominate IC            | Trung bình-Cao  | **Restructure paper**                         | Day 1 chiều   | Pilot 200 nodes trước mọi thứ              | ✅ 3-metric gate trong M2                                   |
| B2  | IC cascade degenerate (global) | Thấp-Trung bình | Labels vô nghĩa nếu global CV fail            | Day 1–2       | CV check + histogram + **per-quintile CV** | ✅ pilot diagnostics JSON                                   |
| B2′ | Within-quintile noise (Q1–Q4)  | **Cao**         | Limitation trong paper, GNN fine-ranking fail | Day 1–2       | Per-quintile CV diagnostic                 | ⚠ Chưa có trong plan — thêm vào Day-1 report                |
| B3  | Typology quadrant quá nhỏ      | Trung bình-Cao  | Statistical power thấp (MWU)                  | Day 3–4       | Estimate ρ(views,IC) từ pilot sớm          | ⚠ Plan có two-sample strategy nhưng chưa có power pre-check |
| B4  | GNN không beat proxy           | Trung bình      | Thấp (publishable với fallback)               | Day 15–18     | Fallback narrative sẵn sàng                | ✅ Fallback narratives trong plan                           |
| B5  | Null model inconclusive        | Trung bình      | Thấp — compress thành 1 câu                   | Day 16–18     | Permutation null thêm vào B5               | ⚠ Permutation null không có trong plan                      |
| B6  | Runtime vượt budget            | Trung bình      | **Delay pipeline**                            | Day 1 sáng    | Benchmark 100×50 trước tiên                | ✅ Budget tiers trong plan                                  |
| B7  | life_time validation fail      | Trung bình-Cao  | Mất external validation                       | Day 17–18     | Test partial ρ trên pilot                  | ⚠ Language fallback chưa có trong plan                      |
| B8  | Paper > 6 trang                | Cao             | **Blocker submission**                        | Day 21+       | Quyết định cắt RQ sớm (Day 5)              | ⚠ Không có explicit cut plan                                |
| B9  | Louvain resolution sai         | Trung bình      | Community features vô nghĩa                   | Day 5–6       | Resolution sweep {0.5,1.0,2.0}             | ⚠ Plan lock resolution=1.0 nhưng không có sensitivity check |
| B10 | GNN deployment claim quá mạnh  | Trung bình      | Reviewer reject/major revision                | Paper writing | Framing cẩn thận + cite precedent          | ⚠ Chưa có explicit framing note trong plan                  |

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

## Kế Hoạch Hành Động Theo Ngày (Day 1–25)

| Ngày        | Bottleneck | Owner    | Deliverable chính                                                                       | Gate / Điều kiện tiếp tục                          |
| ----------- | ---------- | -------- | --------------------------------------------------------------------------------------- | -------------------------------------------------- |
| Day 1 sáng  | B6         | Person 1 | `ic_runtime_benchmark.json` — per_sim_ms, projected_hours                               | n_sample + N_runs locked                           |
| Day 1 chiều | B1         | Person 1 | `one_hop_correlation.json` — spearman ρ, Jaccard@10%, NDCG@10%                          | GNN narrative branch locked                        |
| Day 1 cuối  | B2′        | Person 1 | Per-quintile CV table (Q1–Q5), P(reach>1), P(reach>5), tail sep ratio                   | CV_global > 0.3? Bimodal signal rõ?                |
| Day 2–4     | B2         | Person 1 | IC pilot run (200 nodes × N_runs) → `ic_pilot_diagnostics.json` + Jaccard stability     | jaccard_stability ≥ 0.85                           |
| Day 5–10    | B6 (main)  | Person 1 | IC labels full n_sample nodes → `ic_labels_primary.parquet`                             | IC labels done; CV + stability confirmed           |
| Day 5–7     | B9         | Person 2 | Louvain sensitivity sweep {0.5, 1.0, 2.0} → số communities, modularity Q, NMI(language) | resolution locked → `m0_decisions.md`              |
| Day 7–12    | B3         | Person 2 | Views/IC ρ, 2×2 typology quadrants, residual divergence backup                          | Quadrant sizes checked; narrative tier decided     |
| Day 10–15   | B5         | Person 2 | Null 2 (IC-score perm) + Null 3 (views-perm) → p-values                                 | p < 0.05 OR fallback narrative ready               |
| Day 8–18    | B7         | Person 2 | life_time + language external validation → partial ρ, NMI                               | Pass OR limitation sentence drafted                |
| Day 8–18    | B4         | Person 3 | GNN train (all 4 variants) + MLP-raw-attr baseline → ablation table                     | RMSE vs baselines; "both fail" narrative if needed |
| Day 18–20   | B10        | Person 3 | Eval gap framing → choose Option A/B/C                                                  | Paper claim adjusted; cite precedent               |
| Day 20–25   | B8         | All      | Paper draft, section word budget enforced, figures/tables finalized                     | ≤ 6 pages, ≤ 2600 words                            |

> **Note về parallel track:** Person 2 và Person 3 có thể start track của mình ngay từ Day 5–8 khi Person 1's IC labels từ pilot (Day 2–4) sẵn sàng — không cần chờ full IC labeling hoàn thành. Artifact contracts trong coding plan quy định exactly những file nào mỗi người cần để unblock.

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

> ✅ **Tất cả inconsistency nội bộ giữa các file MAPR2026 đã được giải quyết qua các lần review.** Phần Phụ lục này lưu lại lịch sử phân tích và danh sách tài liệu tham khảo để đối chiếu khi viết paper. Không còn “self-inflicted bottleneck” do file mâu thuẫn nhau.

---

# Phụ lục A: Tài liệu Tham Khảo

Danh sách đầy đủ các tài liệu cần đọc cho project, phân nhóm theo chủ đề.

---

## Nhóm A — Diffusion / Local-vs-Global Influence

1. **Kempe, Kleinberg & Tardos (2003)** — KDD
   Foundation của IC model / influence maximization. Must-read.

2. **Chen, Wang & Wang (2010)** — KDD
   PMIA; rất hữu ích để justify local approximation trong low-p regime.

3. **Lü et al. (2016)** — _Physics Reports_
   Survey mạnh nhất cho vital nodes / spreading dynamics.

4. **Chen et al. (2012)** — _Physica A_
   LocalRank; hợp để giải quyết bottleneck one-hop vs broader local proxy.

5. **Morone & Makse (2015)** — _Nature, 524_(7563), 65–68
   Collective Influence algorithm; chứng minh influential spreaders ≠ luôn là high-degree hubs. Dùng để frame "one-hop/degree proxy vs actual IC rank divergence" là hiện tượng có precedent trong literature.

---

## Nhóm B — Spreading Regimes / Sensitivity

6. **Kitsak et al. (2010)** — _Nature Physics_
   k-shell vs degree in spreading; regime analysis rất relevant.

7. **Pastor-Satorras & Vespignani (2001)** — _Physical Review Letters_
   Epidemic threshold trên scale-free networks; intuition cho low-p regime.

8. **Ling et al. (2023), DeepIM** — ICML
   Dùng để justify weighted cascade setup; **không dùng cho 8% single-seed calibration trên Twitch.**

9. **Borgs, Brautbar, Chayes & Lucier (2014)** — SODA
   RIS algorithm; convergence analysis cho MC IC simulations. Dùng để justify lựa chọn N_runs với (ε,δ)-approximation bound thay vì arbitrary heuristic.

---

## Nhóm C — Popularity vs Influence Divergence

10. **Cha et al. (2010)** — ICWSM
    _The Million Follower Fallacy_ — ρ(followers, retweets) ≈ 0.5 trên Twitter. Frame "views–IC divergence" với precedent.

11. **Bakshy et al. (2011)** — WSDM
    Empirical variability of influence; "Hidden influencer" concept.

12. **Aral & Walker (2012)** — _Science_
    Social ties and influence vs susceptibility; useful for construct-validity discussion.

---

## Nhóm D — GNN Fairness / Expressiveness

13. **Hamilton, Ying & Leskovec (2017)** — NeurIPS
    GraphSAGE — inductive representation learning. Transductive baseline provided for comparison.

14. **Kipf & Welling (2017)** — ICLR
    GCN — standard transductive node classification. Cite để justify transductive evaluation protocol khi labels scarce.

15. **Xu et al. (2019)** — ICLR
    How Powerful Are GNNs? (GIN / expressiveness theory). Mean-aggregation GNN có thể approximate local analytical formula.

16. **Errica et al. (2020)** — ICLR
    Fair comparison của GNN vs MLP; hữu ích khi "GNN không thắng MLP-raw-attr."

17. **You et al. (2019)** — ICML
    Position-aware GNNs; limits of local message passing cho global rank prediction.

18. **Yang & Leskovec (2016)** — ICML
    "Revisiting Semi-Supervised Learning with Graph Embeddings" — standard framing cho transductive evaluation với partial labels. **Cite để justify B10 Option A+C.**

---

## Nhóm E — Null Models / Community Detection

19. **Fosdick, Larremore, Nishimura & Ugander (2018)** — _SIAM Review_
    Configuration model variants, biases, và edge-swapping MCMC. First-order null.

20. **Orsini et al. (2015)** — _Nature Communications_
    dk-series null models; bảo toàn degree correlations nhiều orders hơn configuration model. NICE-TO-HAVE — future work.

21. **Blondel, Guillaume, Lambiotte & Lefebvre (2008)** — _Journal of Statistical Mechanics_
    Louvain algorithm; resolution parameter semantics.

22. **Newman (2006)** — PNAS
    Modularity và resolution limit của community detection.

23. **Fortunato & Barthélemy (2007)** — _PNAS, 104_(1), 36–41
    Formal proof của resolution limit trong modularity optimization. **Must-read trước khi chọn Louvain resolution.**

24. **Traag, Waltman & van Eck (2019)** — _Scientific Reports, 9_, 5233
    Leiden algorithm — fixes Louvain's disconnected community artifact. Dùng nếu `leidenalg` available.

25. **Lancichinetti & Fortunato (2009)** — _Physical Review E, 80_, 056117
    Community detection benchmark (LFR graphs); realistic expectations cho Louvain trên dense graphs.

---

## Nhóm F — Statistics / Power Analysis

26. **Benjamini & Hochberg (1995)** — _JRSS Series B_
    FDR correction; cần hiểu khi BH correction trên 5 quintile tests inflate p-values.

27. **Cohen (1988)** — "Statistical Power Analysis for the Behavioral Sciences"
    Effect size interpretation; Cliff's delta ≥ 0.20 là small-to-medium effect.

---

## Nhóm G — Dataset Reference

28. **Rozemberczki, Allen & Sarkar (2021)** — _Journal of Complex Networks_
    Twitch dataset paper gốc. Semantic definitions của `views`, `life_time`, `language`, `mature`. Must-read.

---

# Phụ lục B: Status Checklist (cập nhật sau mỗi lần review)

> **Status update (chốt sổ — final):** ✅ = đã xử lý và verified, ⚠ = open item cần team handle khi execute, 🚫 = OUT OF SCOPE 25 ngày.

## Must-fix — Tất cả đã RESOLVED ✅

1. ✅ **B1 three-hop removed**: đã thay bằng infeasibility statement O(Σ d(v)³) — không làm được ở Twitch scale
2. ✅ **B1 3-metric gate**: Spearman ρ + Jaccard@10% + NDCG@10% — không chỉ Spearman đơn lẻ
3. ✅ **B2 reordered**: Primary diagnostics (per-quintile CV, P(reach>k), tail sep, Kendall τ) TRƯỚC; ba hướng là LAST RESORT
4. ✅ **B2 Jensen's inequality**: IC distribution bimodal by degree quintile, không uniformly degenerate
5. ✅ **B4 MLP-raw-attr**: primary ablation baseline thêm vào; "both fail" fallback narrative có template
6. ✅ **B5 Null 2 + Null 3**: IC-score permutation và views-permutation đều có trong B5 với mục tiêu phân biệt rõ
7. ✅ **B5 dk-series labeled NICE-TO-HAVE / OUT OF SCOPE**
8. ✅ **B6 simulation count**: 3.4M (không phải 5.4M); budget tier table có
9. ✅ **Complexity two-hop**: O(Σ d(v)²) đã cập nhật trong cả 3 MAPR2026 files
10. ✅ **GNN-raw-attr independence matrix**: ❌ views-independent đã sửa trong v3 plan
11. ✅ **B9 (Louvain resolution)**: section đầy đủ với 4-step protocol + resolution sweep {0.5,1.0,2.0}
12. ✅ **B10 (GNN eval gap)**: section đầy đủ với Options A/B/C
13. ✅ **Inconsistency giữa MAPR2026 files**: tất cả đã RESOLVED (views vs views_log, N_seeds terminology, community_features path, independence matrix)
14. ✅ **Priority Tiers table**: Must/Should/Nice cho tất cả 10 bottleneck
15. ✅ **Day 1-25 Action Plan**: timeline với owner + deliverable + gate cho từng bottleneck
16. ✅ **Phụ lục A**: 28 references đầy đủ phân nhóm theo chủ đề

## Strongly Recommended — Cần team handle khi execute

17. ⚠ **Sampling bias**: trong paper ghi rõ "All predictive metrics estimated on representative labeled subset; targeted augmentation used only for descriptive typology analysis" — xem B3 + B10
18. ✅ **views-permutation null (Null 3)**: đã được lock trong coding plan + checklist (execution lock + artifact requirement)
19. ✅ **residual-based divergence**: đã được lock trong coding plan + checklist theo trigger rõ ràng + artifact requirement
20. ✅ **GNN-random-features baseline**: đã thêm dưới dạng add-if-time optional trong plan + checklist (`gnn_random`)

## Nice-to-Have / Future Work

21. 🚫 **dk-series null model**: OUT OF SCOPE 25 ngày — ghi footnote future work
22. ⚠ **Community-aware sample sanity check**: nhẹ, thêm nếu B9 sweep trigger cần rerun partition
23. ⚠ **One-hop gate rerun trên 500 nodes**: nếu pilot ρ ∈ [0.78, 0.92] → rerun trước khi chốt narrative
24. 🚫 **Uncertainty-aware training** (inverse-variance weighting): OUT OF SCOPE 25 ngày — ghi limitation sentence
