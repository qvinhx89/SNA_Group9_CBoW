# Hướng dẫn Supervisor: Viết paper MAPR 2026

## Cách dùng file này

Đây là **bản final guide đã gộp** từ hai lớp hướng dẫn trước đó. Mục tiêu của file này là cho team một tài liệu duy nhất, đi từ trên xuống là đủ để:

- chốt paper identity và scope,
- viết từng section của paper,
- biết rõ claim nào được phép nói và cần artifact nào để chống đỡ,
- hoàn tất drafting, freeze, và final submission theo một workflow gọn và defensible.

**Source of truth khi có xung đột:**

- Framing, claims, comparator policy, allowed/forbidden wording: `MAPR2026_Implementation_Plan_v3.md`
- Artifact names, ownership, execution order, freeze discipline: `docs/MAPR2026_v3_team_parallel_coding_plan.md`

**Quy ước đọc file này:**

- Phần hướng dẫn, giải thích, và decision logic viết bằng tiếng Việt.
- Các đoạn được đánh dấu rõ là **paste-ready English** có thể dùng trực tiếp hoặc chỉnh rất nhẹ để đưa vào paper.
- Mọi placeholder số liệu như `[X]`, `[Y]`, `[X, Y]` chỉ được điền sau khi frozen outputs đã được khóa.

---

## Phần 1: Paper identity, scope, và narrative backbone

### 1.1 Paper này là gì / không phải là gì

**Paper này là:** một bài **comparative empirical study** về việc cách operationalize IC ảnh hưởng như thế nào đến learnability của influence surrogate. Dưới `A0`, analytical baselines gần như chạm trần. Dưới `HSCC`, graph message passing có thể thêm giá trị nếu neighborhood/community structure thực sự mang signal vượt quá flat baselines.

**Paper này không phải:**

- claim rằng GNN luôn vượt trội cho influence prediction,
- claim rằng `HSCC` là diffusion model đúng cho Twitch,
- claim rằng MC-IC scores là real influence,
- paper về influence maximization theo nghĩa tối ưu seed set,
- paper về kiến trúc GNN mới.

### 1.2 Core thesis và title direction

**Core thesis của paper:**

> Giá trị của GNN surrogate phụ thuộc mạnh vào IC operationalization. Dưới `A0`, analytical baselines gần như là structural ceiling. Dưới `HSCC`, graph message passing chỉ có giá trị khi nó học được phần signal graph/community mà flat baselines không recover được.

**Gợi ý title bằng tiếng Anh:**

- `"When Does Graph Learning Outperform Analytical Baselines? A Comparative Study of IC Operationalizations for Influence Approximation"`
- `"Regime-Dependent GNN Surrogate Learning for Monte Carlo Influence Estimation on Social Networks"`

### 1.3 Scope guard

Trong MAPR window, final guide này chỉ công nhận một active path:

- `A0`: weighted cascade structural regime, `p(u,v)=1/deg(v)`
- `HSCC`: source-velocity + community-aware regime

Các nhánh như `I-A`, `II-B`, `A1` chỉ được nhắc nếu thật cần như archive note rất ngắn hoặc appendix/future-work note. Không để chúng xuất hiện như active execution branch trong main paper.

### 1.4 Narrative backbone

Giữ chặt 4 bước này làm xương sống cho mọi section:

1. **MC-IC là một operational metric hợp lý nhưng vẫn chỉ là proxy.**
2. **Binary top-k labels không ổn định, nên regression/ranking là formulation đúng.**
3. **Không có một IC operationalization nào tự nhiên đúng; operationalization choice quyết định learnability.**
4. **GNN không có giá trị cố hữu; giá trị của nó phụ thuộc vào regime của label.**

Nếu giữ đúng backbone này, paper vẫn defensible kể cả khi biên lợi thế của GNN dưới `HSCC` chỉ nhỏ.

---

## Phần 2: Hướng dẫn viết theo từng mục / section

### Mục 1 / Section 1 - Introduction (0.5 trang, khoảng 400 từ)

#### Mục tiêu

Section mở đầu phải trả lời được 4 câu:

1. Bài toán là gì?
2. Vì sao khó?
3. Vì sao MC-IC cần surrogate?
4. Đóng góp cụ thể là gì?

#### Đoạn 1 / Paragraph 1 - Problem

**Paste-ready English:**

Identifying influential users in online social networks is important for applications such as viral marketing, community management, and platform recommendation. Monte Carlo Independent Cascade (MC-IC) simulation provides a principled operationalization of influence potential grounded in the diffusion model of Kempe et al. (2003). However, MC-IC is computationally expensive, requiring hundreds of stochastic simulations per node and thereby rendering repeated evaluation impractical on large-scale graphs.

#### Đoạn 2 / Paragraph 2 - Tension

**Paste-ready English:**

Graph Neural Networks (GNNs) offer a natural surrogate approach: they can learn to approximate IC scores from graph structure and node attributes and then be deployed for fast inference. Prior work on GNN-based influence estimation has largely focused on a single diffusion model, leaving open the question of when learned graph representations genuinely outperform simple analytical baselines such as degree centrality. In dense social networks where cascades attenuate quickly, degree itself may already capture most of the diffusion signal, leaving limited room for additional gains from graph learning.

#### Đoạn 3 / Paragraph 3 - Core idea

**Paste-ready English:**

In this paper, we investigate how the choice of IC operationalization determines whether GNN surrogate learning provides value beyond analytical baselines. We compare two defensible operationalizations on the Twitch social network (168K nodes, 6.8M edges): (1) weighted cascade (A0), in which transmission probability depends only on target degree, and (2) HSCC, a domain-informed variant that incorporates source engagement velocity and cross-community amplification. Our contributions are threefold. First, we show that binary influence classification is structurally unstable on dense networks, motivating continuous regression as the principled prediction formulation. Second, we demonstrate that under degree-coupled IC, GNN architectures converge to the degree-centrality ceiling, confirming that the operationalization, rather than the model family, is the binding constraint. Third, under HSCC, graph message passing can exploit cross-community engagement structure beyond flat baselines, achieving Spearman rho = [X] compared with [Y] for the strongest non-graph baseline.

#### Contributions - nên giữ 3 bullet

**Paste-ready English:**

1. **We analyze Monte Carlo IC as a simulation-defined operational metric for influence potential on Twitch, showing that binary top-k labels are structurally unstable whereas continuous regression targets remain informative.**
2. **We compare two diffusion operationalizations, A0 and HSCC, and show that they induce qualitatively different approximation regimes: a degree-dominated regime and a graph-aware attribute-community regime.**
3. **We benchmark analytical, flat, and GNN surrogates under both regimes, showing that the value of GNNs is regime-dependent rather than universal while retaining orders-of-magnitude speedups over repeated MC simulation.**

#### Cần tránh

- `"we identify real power users"`
- `"we propose a superior GNN"`
- `"we discover the correct Twitch diffusion model"`

### Mục 2 / Section 2 - Background / Related Work (0.5-0.75 trang)

#### 2.1 Influence / IC / diffusion

Phần này nên rất ngắn:

- định nghĩa IC theo Kempe et al. (2003),
- nêu weighted cascade setup,
- nói rõ MC estimation cần nhiều stochastic runs cho mỗi seed node,
- chốt bài toán của bạn là **node-level IC score approximation**, không phải seed-set optimization.

#### 2.2 Node importance / structural baselines

Chỉ cần đủ để reviewer thấy vì sao các baseline này hợp lệ:

- degree,
- PageRank,
- k-shell,
- one-hop / two-hop spread.

#### 2.3 Graph surrogate / GNN

Nhắc ngắn 5 kiến trúc đã test:

- GraphSAGE
- GCN
- GIN
- GAT
- APPNP

Mục tiêu là cho thấy shortlist hợp lý và được chạy dưới cùng một training protocol để fair comparison.

#### Lưu ý quan trọng

Không đi quá sâu vào literature của influence maximization. Paper này không tối ưu seed set; paper này học surrogate cho **continuous node-level IC scores**.

### Mục 3 / Section 3 - MC-IC as Operational Metric (1.0-1.25 trang)

Đây là section methodological mạnh nhất của paper. Nó phải làm được 4 việc:

1. biện minh vì sao IC là một operational metric hợp lý,
2. giới thiệu `A0` và `HSCC`,
3. chứng minh binary instability là structural,
4. biện minh vì sao regression là primary formulation.

#### 3.1 Construct validity

Phải giữ logic này:

- follower graph khác observed diffusion,
- nhưng vẫn là structural substrate hợp lý trong absence of cascade logs,
- mọi finding trong paper là về **simulation-defined influence approximation**,
- không được viết như thể paper đang đo real influence ngoài đời.

#### 3.2 Hai operationalization đang active

**A0**

\[
p(u,v)=1/\deg(v)
\]

- standard structural operationalization,
- degree-coupled by design,
- dùng như structural contrast / reference regime.

**HSCC**

\[
p(u,v)=\mathrm{clip}\left(\lambda \frac{\phi(u)}{\deg(u)}(1+\gamma \mathbf{1}[c_u\neq c_v]),0,p_{\max}\right)
\]

với:

\[
\phi(u)=\mathrm{rank}\left(\frac{\log(1+\mathrm{views}_u)}{1+\mathrm{life\_time}_u}\right)/N
\]

- domain-informed alternative,
- source engagement velocity + community bridging,
- không phải "true Twitch diffusion model",
- chỉ là một comparative operationalization hợp lý để kiểm tra khi nào graph structure tạo thêm learnable value.

#### 3.3 Discriminativeness

Nên có một bảng ngắn kiểu này:

| Regime | Mean reach | Median | CV | Comment |
|---|---:|---:|---:|---|
| A0 | ... | ... | ... | heavy-tailed, structural |
| HSCC | ... | ... | ... | selective, non-degenerate |

**Điểm cần viết rõ:**

- `A0`: broad nhưng degree-dominated,
- `HSCC`: mean nhỏ hơn nhưng vẫn discriminative,
- không kéo `I-A` vào main text; nếu cần chỉ nhắc như archive note một câu.

#### 3.4 Stability and regression

Phần này phải rất crisp:

- binary top-k unstable,
- many communities span the decision boundary,
- gap-to-noise gần 0,
- instability là structural chứ không chỉ do Monte Carlo noise,
- vì vậy continuous regression là principled formulation.

**Paste-ready English:**

> Binary influence classification is structurally unstable in dense social networks with heavy-tailed IC distributions. This instability is not eliminated by increasing the number of simulation runs and instead reflects a property of the underlying graph topology rather than simulation variance.

**Câu nên giữ thêm:**

> We treat regression not as a fallback, but as the natural formulation for a simulation-derived continuous influence target.

#### 3.5 Why not degree?

Phần này dùng để tránh overclaim:

- Under `A0`, degree đúng là approximation rất mạnh.
- Nhưng variance analysis vẫn cho biết IC có giữ thêm variance beyond local connectivity hay không.
- Under `HSCC`, degree collapse gần như hoàn toàn.

**Câu chốt nên dùng:**

> The extent to which IC provides signal beyond degree depends on the operationalization.

### Mục 4 / Section 4 - Surrogate Learning Across Operationalizations (~2.0 trang)

#### 4.1 Setup

Giữ ngắn gọn:

- Twitch MUSAE
- 168k nodes
- 6.8M edges
- 5k labeled nodes
- transductive split
- 5 seeds
- metrics: Spearman, NDCG@10, P@10
- runtime reported separately

#### 4.2 A0 results - structural ceiling

Phần này **không được viết như một failure**.

**Thông điệp chính:**

- degree / two-hop đã rất mạnh,
- GNN tiến gần ceiling đó,
- comparator bootstrap chính thức của `A0` là **degree**.

**Trong bảng nên có:**

- degree
- one-hop
- two-hop
- best flat baseline
- 4-5 GNN architectures

**Paste-ready English nếu CI cho practical equivalence:**

> Under A0, the best-performing GNN is practically equivalent to degree centrality under the pre-registered equivalence bound, indicating that the limiting factor is the diffusion operationalization rather than the neural architecture.

Ngoài ra, vẫn nên nêu diagnostic kiểu `GNN-raw-attr vs MLP` để cho thấy message passing học được graph signal, nhưng signal đó không vượt được structural ceiling do `A0` áp đặt.

#### 4.3 HSCC results - graph-aware regime

Đây là subsection main-claim.

**Thông điệp chính:**

- degree không còn là comparator đúng,
- strongest flat baseline mới là comparator thật,
- GNN chỉ có giá trị nếu học được graph/community structure beyond raw attributes.

**Bảng HSCC bắt buộc nên có:**

- `LR(life_time)`
- `LR(views + life_time)`
- `LR(degree + views + life_time)`
- `MLP(raw attrs)`
- nếu GNN dùng pipeline `raw_attr` hiện tại:
  - `LR(... + language)`
  - `MLP(... + language)`

**Lưu ý bắt buộc cho codebase hiện tại:** nếu `raw_attr` dùng cột `language`, language dummies sẽ được đưa vào features tự động. Vì vậy, trừ khi team chủ động disable và ghi lại quyết định đó trước frozen run, fairness baselines có `language` là bắt buộc.

**Nếu GNN thắng strongest flat baseline:**

> Under HSCC, GNN message passing significantly improves upon the strongest flat baseline, suggesting that neighborhood structure contributes information that cannot be recovered from node-level attributes alone.

**Nếu GNN chỉ xấp xỉ / hơn rất ít:**

> Under HSCC, flat attribute models explain most of the source-driven component, while GNNs contribute a smaller but still measurable improvement attributable to graph-mediated community structure.

**Nếu GNN thua strongest flat baseline:**

> Under HSCC, most of the predictive signal is already captured by source-side engagement attributes, with only limited incremental benefit from graph-based message passing.

Paper vẫn ổn nếu contrast `A0 vs HSCC` còn mạnh và viết trung thực.

**Comparator lock cho bản MAPR:**

- comparator chính của `HSCC` trong main paper là **strongest flat baseline từ frozen fairness table dưới matched feature access**,
- `phi`, `lr_phi`, và các oracle-style decomposition liên quan chỉ dùng để giải thích cơ chế,
- không dùng chúng làm comparator chính nếu comparator policy chưa được re-lock trong plans.

#### 4.4 Contrast analysis

Đây là trái tim intellectual của paper.

**A0**

- label degree-coupled,
- analytical baselines mạnh,
- GNN near ceiling.

**HSCC**

- label có source term + graph/community amplification term,
- flat baselines chủ yếu capture source term,
- GNN chỉ có thể thêm giá trị ở structural amplification part.

`phi`, `lr_phi`, và related oracle decompositions nên được dùng như interpretation rows, không phải main baselines.

**Paste-ready English:**

> The contrast between A0 and HSCC shows that surrogate learnability is not a property of the model alone; rather, it is jointly determined by the diffusion operationalization and the information already recoverable by simple baselines.

#### 4.5 Runtime

Giữ phần này ngắn và thực dụng:

- MC-IC labeling cost
- GNN training cost
- GNN inference cost
- analytical baseline inference gần như bằng 0

**Cách diễn đạt đúng:**

- speedup **vs MC-IC**
- không so speedup của GNN với degree như thể chúng cùng chi phí labeling

### Mục 5 / Section 5 - Discussion & Limitations (0.5 trang)

#### 5.1 When does GNN help?

Câu trả lời phải nhất quán:

- not universally,
- only when the target depends on graph-mediated information not already captured by strong flat baselines.

#### 5.2 Limitations

Nên giữ 5 điểm này:

1. follower graph khác observed diffusion path,
2. `A0` và `HSCC` là operationalizations chứ không phải ground truth,
3. `HSCC` là domain-informed formula mới nhưng chưa được empirically validated như diffusion law thật,
4. evaluation hiện tại là transductive,
5. small mean reach dưới `HSCC` nên được hiểu như selective diffusion, không phải broad viral spread.

Nếu paper dùng `language` trong GNN, phải nói rõ rằng fairness của baseline feature access là điều kiện để mọi claim về GNN có giá trị.

#### 5.3 Why not learn p from data?

**Paste-ready English:**

> Estimating edge-level transmission probabilities would require supervised cascade logs that are unavailable in this dataset; weighted cascade and HSCC therefore provide principled zero-shot alternatives.

---

## Phần 3: Hình, bảng, và abstract

### 3.1 Hình bắt buộc nên có

#### Hình 1 / Figure 1

Pipeline diagram:

- graph
- `A0 / HSCC`
- MC-IC labels
- regression targets
- baselines + GNN surrogates

#### Hình 2 / Figure 2

Two-panel results figure:

- trái: `A0`
- phải: `HSCC`
- bar chart hoặc dot plot với CI
- comparator line:
  - `A0`: degree
  - `HSCC`: strongest flat baseline

### 3.2 Bảng bắt buộc nên có

#### Bảng 1 / Table 1

Dataset + operationalizations:

- nodes, edges
- A0 formula
- HSCC formula
- mean / median / CV

#### Bảng 2 / Table 2

A0 results:

- degree
- one-hop
- two-hop
- MLP
- GNNs

#### Bảng 3 / Table 3

HSCC results:

- `LR(life_time)`
- `LR(views+life_time)`
- `LR(degree+views+life_time)`
- `MLP`
- GNNs

#### Bảng 4 / Table 4 (nếu còn chỗ)

Runtime mini-table.

Nếu thiếu chỗ, merge runtime vào main results table dưới dạng cột cuối.

### 3.3 Abstract template

**Câu 1 / Sentence 1 - Problem + difficulty**

> Identifying influential users in static social networks without behavioral cascade logs requires simulation-based operationalizations of influence, yet the learnability of such operationalizations remains poorly understood.

**Câu 2 / Sentence 2 - Method**

> We study two Monte Carlo Independent Cascade (MC-IC) operationalizations on the Twitch social network: a structural weighted-cascade regime (A0) and a domain-informed source-community regime (HSCC).

**Câu 3 / Sentence 3 - Stability / regression**

> We show that binary top-k influence labels are structurally unstable, motivating continuous regression on simulation-derived influence scores.

**Câu 4 / Sentence 4 - Main contrast**

> Under A0, analytical structural baselines are already near-optimal, whereas under HSCC the strongest competitors shift to flat source-attribute models.

**Câu 5 / Sentence 5 - GNN result**

> Across GraphSAGE, GCN, GIN, GAT, and APPNP, GNN surrogates provide regime-dependent value, ranging from practical equivalence to structural baselines under A0 to measurable gains over flat baselines under HSCC.

**Câu 6 / Sentence 6 - Runtime**

> In all cases, learned surrogates provide inference that is orders of magnitude faster than repeated MC simulation.

---

## Phần 4: Claims, evidence, và wording guardrails

### 4.1 Claim 1

**Paste-ready English claim:**

> Binary influence classification is structurally unstable in dense social networks.

**Required evidence:**

- `stability_explanation.json`
- Jaccard stability sweep
- Spearman stability sweep

Claim này chỉ được nói mạnh khi frozen copies đã được xác nhận là đúng artifacts dùng cho bản nộp.

### 4.2 Claim 2

**Paste-ready English claim:**

> Under degree-coupled IC (A0), GNNs are practically equivalent to degree centrality.

**Required evidence:**

- `gnn_vs_degree_bootstrap_ci_a0.json`
- frozen A0 results table
- diagnostic `GNN-raw-attr vs MLP-raw-attr`

Comparator bootstrap chính thức của `A0` là **degree**. one-hop và two-hop là narrative/table baselines mạnh, nhưng không thay thế bootstrap-vs-degree nếu chưa re-lock plan.

### 4.3 Claim 3

**Paste-ready English claim:**

> Under attribute-community IC (HSCC), GNNs outperform flat baselines by capturing cross-community engagement structure.

**Required evidence:**

- `gnn_vs_baseline_bootstrap_ci_hscc.json`
- frozen HSCC results table
- fairness stack đầy đủ dưới matched feature access
- oracle-style decomposition rows chỉ để interpretation

`phi`, `lr_phi`, và related oracle rows không phải comparator chính của main paper.

### 4.4 Những claim được phép

- `"MC-IC is a principled operational metric"`
- `"A0 and HSCC induce different approximation regimes"`
- `"binary labels are structurally unstable"`
- `"under A0, analytical baselines are near-optimal"`
- `"under HSCC, GNNs may add value beyond flat baselines"`
- `"surrogate value depends on operationalization"`

### 4.5 Những claim không được phép

- `"we identify real power users"`
- `"HSCC is the true Twitch diffusion model"`
- `"GNN always outperforms baselines"`
- `"MC-IC is ground truth"`
- `"practical equivalence"` khi chưa có SESOI + CI phù hợp
- `"GNN is feature-agnostic"` theo nghĩa tuyệt đối

Thay `"feature-agnostic"` bằng:

> without precomputed structural summaries

### 4.6 Kịch bản viết paper theo kết quả cuối

#### Kịch bản 1 / Case 1 - Tốt nhất

- `A0`: GNN ≈ degree
- `HSCC`: GNN > strongest flat baseline

Framing:

- A0 = structural ceiling
- HSCC = graph-aware regime
- GNN advantage is regime-dependent
- runtime story sạch

#### Kịch bản 2 / Case 2 - Trung tính nhưng vẫn tốt

- `A0`: GNN ≈ degree
- `HSCC`: GNN ≈ strongest flat baseline

Framing:

- operationalization contrast vẫn là main contribution,
- HSCC cho thấy source-side attributes dominate much of the signal,
- graph message passing adds limited but interpretable value.

#### Kịch bản 3 / Case 3 - Kết quả yếu

- `A0`: GNN < degree
- `HSCC`: GNN < strong flat baseline

Framing:

1. binary instability finding,
2. operationalization contrast,
3. analytical / flat baselines often suffice depending on regime,
4. GNN is not universally superior.

Đây vẫn có thể là một empirical negative-result paper ổn cho MAPR nếu viết trung thực và gọn.

---

## Phần 5: Các paper nên đọc và cite

### 5.1 Must cite trong main paper

1. **Kempe, Kleinberg, Tardos (2003)** - foundational IC model
2. **Ling et al. (2023), DeepIM** - weighted cascade setup, learning-based IM context
3. **Rozemberczki et al. (2021)** - MUSAE / Twitch dataset
4. **Hamilton et al. (2017)** - GraphSAGE
5. **Kipf & Welling (2017)** - GCN
6. **Xu et al. (2019)** - GIN
7. **Veličković et al. (2018)** - GAT
8. **Klicpera et al. (2019)** - APPNP
9. **Kitsak et al. (2010)** - k-shell spreaders
10. **Guille et al. (2013)** - evaluation without behavioral logs
11. **Burt (1992)** - structural holes for HSCC justification
12. **Blondel et al. (2008)** - Louvain

### 5.2 Có thể thêm nếu còn budget

- **Benjamini & Hochberg (1995)** nếu dùng BH-FDR
- **Aral & Walker (2012)** nếu cần nhấn mạnh social ties and influence pathways
- **Lü et al. (2016)** cho survey về vital nodes
- **Chen, Wang & Wang (2010)** cho hop-decay / local influence approximation
- **Grover & Leskovec (2016)** nếu Node2Vec xuất hiện như baseline

### 5.3 Read but probably not cite cho MAPR version

- GCNII
- HGT
- GraphGPS
- fairness/per-group analysis papers
- journal-only methodological extensions

---

## Phần 6: Execution và freeze checklist

### 6.1 Giai đoạn 1 - Before writing

- **Người 1 / Person 1**
  - verify `regression_targets_hscc_refined.parquet`
  - verify HSCC formula lock trong `experiment_registry.md`
  - freeze config

- **Người 2 / Person 2**
  - verify `community_features.parquet` coverage
  - ensure `community_id` và `cross_community_edge_fraction` đều đầy đủ
  - xác nhận các downstream profiling inputs sạch

- **Người 3 / Person 3**
  - hoàn tất HSCC fairness baselines
  - regenerate hoặc refresh regime-tagged outputs nếu file hiện có stale / thiếu rows
  - đảm bảo bootstrap comparators là regime-specific

### 6.2 Giai đoạn 2 - During drafting

- viết theo thứ tự: Introduction / Background -> MC-IC metric -> GNN results -> Discussion
- chỉ dùng frozen outputs khi điền số liệu
- không mở thêm experiment mới khi đã bước sang drafting
- không đổi operationalization, comparator policy, hoặc feature policy sau freeze

### 6.3 Giai đoạn 3 - Before submission

- IEEE formatting
- double-blind compliance
- figure readability ở grayscale
- reference formatting
- claim-to-artifact cross-check lần cuối
- abstract phải phản ánh findings thật, không phản ánh điều team kỳ vọng

### 6.4 Freeze rules bắt buộc

- `A0`: bootstrap vs `degree`
- `HSCC`: bootstrap vs strongest flat baseline từ frozen matched-access fairness table
- nếu GNN dùng `language`, HSCC fairness baselines cũng phải dùng `language`
- outputs cần regime-tagged rõ ràng
- `community_features.parquet` là blocking dependency để diễn giải HSCC

---

## Phần 7: Reviewer defense appendix

### Reviewer hỏi: "Why not use real cascade data?"

Twitch dataset không có behavioral cascade logs. MC-IC chỉ là simulation-based proxy có cơ sở phương pháp luận; mọi findings phải được hiểu là properties của simulation, không phải real influence measurements.

### Reviewer hỏi: "Why is HSCC a good diffusion model?"

Paper không claim `HSCC` là diffusion model thật của Twitch. `HSCC` chỉ là một domain-informed operationalization để kiểm tra khi nào neighborhood composition tạo learnable value cho GNN. Đóng góp của paper là comparative finding, không phải realism claim.

### Reviewer hỏi: "Why not compare against DeepIM or other IM methods?"

DeepIM giải bài toán khác: chọn seed set tối ưu để maximize total cascade reach. Task ở đây là **node-level IC score regression**, nên comparison trực tiếp không cùng problem setting.

### Reviewer hỏi: "The Twitch dataset is from 2021. Is it still relevant?"

Đóng góp của paper là methodological, không phải phát hiện đặc thù cho Twitch năm 2021. MUSAE Twitch vẫn là benchmark hợp lý cho graph-level analysis và surrogate-learning evaluation.

### Reviewer hỏi: "Life_time dominates HSCC labels. Isn't GNN just learning life_time?"

Đây là lý do fairness baselines là điều kiện bắt buộc. Nếu `language` đi vào GNN, nó cũng phải đi vào LR/MLP fairness baselines. Chỉ residual margin sau matched baselines mới được phép diễn giải là graph/community message-passing gain.

---

## Phần 8: Kết luận của supervisor

Nếu team giữ đúng các điều sau, paper hoàn toàn có thể defensible ở MAPR:

1. chỉ đi theo `A0 + HSCC`,
2. giữ comparator policy theo regime,
3. khóa fairness trước khi nói về GNN win,
4. viết paper như một comparative operationalization study,
5. không để claims vượt quá frozen evidence.

**Điểm mạnh nhất của paper:**

- stability finding,
- regime contrast,
- honest demonstration rằng GNN value is conditional, not universal,
- runtime story sạch.

**Điểm dễ bị reviewer tấn công nhất:**

- HSCC fairness baselines chưa đủ,
- HSCC bị frame như engineered-but-undeclared,
- claim về GNN không map chặt với bootstrap artifacts theo từng regime.

Nếu khóa chặt ba điểm đó, final guide này đủ để team follow trực tiếp và viết ra một paper mạch lạc, đúng đắn, hợp lý, và defensible.
