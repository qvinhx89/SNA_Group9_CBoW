# Hướng dẫn Supervisor: Viết paper MAPR 2026

## Cách dùng file này

Đây là **bản final guide đã gộp** từ hai lớp hướng dẫn trước đó. Mục tiêu của file này là cho team một tài liệu duy nhất, đi từ trên xuống là đủ để:

- chốt paper identity và scope,
- viết từng section của paper,
- biết rõ claim nào được phép nói và cần artifact nào để chống đỡ,
- hoàn tất drafting, freeze, và final submission theo một workflow gọn và defensible.

**Two-file authority model — đọc cả hai file, mỗi file trả lời một loại câu hỏi khác nhau:**

| Câu hỏi                                                           | File trả lời                    |
| ----------------------------------------------------------------- | ------------------------------- |
| Số liệu frozen, paste-ready sentences, claim templates là gì?     | **File này (`Paper guide.md`)** |
| Viết như thế nào — style, tone, language patterns, section rules? | **`Paper rules.md`**            |
| Claim nào được phép và cần artifact nào?                          | Cả hai file cùng nhau           |

Khi hai file có vẻ mâu thuẫn về **số liệu cụ thể** (ρ, CI, comparator name): **file này thắng**. Khi mâu thuẫn về **nguyên tắc viết/framing**: **`Paper rules.md` thắng**.

- `MAPR2026_Implementation_Plan_v3.md` — archive reference cho execution decisions cũ; không dùng để override bất cứ điều gì trong hai file trên
- `docs/MAPR2026_v3_team_parallel_coding_plan.md` — ownership/execution order; chỉ đọc khi cần trace artifact provenance

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

- `"When Does Graph Learning Add Value Beyond Strong Baselines? A Comparative Study of IC Operationalizations for Influence Approximation"`
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

### 1.5 Main-paper freeze rule

Để paper không bị quá tải trong giới hạn 6 trang IEEE, main paper chỉ nên xoay quanh 4 khối evidence đã được plans chống đỡ rõ:

1. **construct validity + stability/regression justification**,
2. **A0 results + bootstrap vs degree**,
3. **HSCC results + fairness-complete strongest-flat comparison**,
4. **runtime + một đoạn contrast interpretation ngắn**.

Những thứ sau chỉ nên để ở appendix/supplementary hoặc cắt trước nếu thiếu chỗ:

- full correlation matrix,
- mọi architecture-by-feature ablation dài,
- `phi`, `lr_phi`, và các oracle-style decomposition chi tiết,
- `A2` hoặc archive regimes,
- extended reviewer-defense prose,
- sensitivity notes không đi thẳng vào 3 claims chính.

Rule vận hành rất quan trọng: review skeleton có thể gợi ý cấu trúc mạnh hơn, nhưng main paper không được mở thêm narrative branch mới ngoài 4 khối trên nếu artifacts freeze chưa chống đỡ được.

> **⚠️ MAIN-PAPER SCOPE LOCK — IEEE 6-page limit**
>
> Main paper claims are bounded to exactly **4 evidence blocks** (+ optional C3):
>
> 1. **Binary instability → continuous regression** (§3 + `ic_label_stability.json`, `phase2_threshold_analysis.json`)
> 2. **A0 result: GCN statistically below degree — structural ceiling** (§4.2 + `gnn_vs_degree_bootstrap_ci_a0.json`)
> 3. **HSCC result: SAGE significantly above `lr_degree_views_life_time_lang` — message passing value** (§4.3 + `gnn_vs_baseline_bootstrap_ci_hscc.json`)
> 4. **Runtime: ~5,500× speedup** (§4.5 + `runtime_breakdown.csv`)
>
> **[🟡 C3 BOOST — ✅ FROZEN]** SAGE rankloss (Δρ=+0.041 vs comparator) may be added as a 5th result only if page budget allows. It does NOT replace any of the 4 blocks above.
>
> **DO NOT ADD** to main paper: full arch ablation tables, oracle-style decompositions (φ, lr_phi), archive regimes (A2), extended reviewer defense prose, sensitivity notes not directly supporting claims 1–3.

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

Graph Neural Networks (GNNs) offer a natural surrogate approach: they can learn to approximate IC scores from graph structure and node attributes and then be deployed for fast inference. Prior work on GNN-based influence estimation has largely focused on a single diffusion model, leaving open the question of when learned graph representations genuinely add value beyond the strongest non-graph baselines — whether structural metrics such as degree centrality, or flat attribute models such as logistic regression. The answer depends critically on the IC operationalization: regimes that couple transmission probability to degree already make degree a near-optimal surrogate, whereas regimes driven by source-side attributes shift competitiveness toward flat attribute models and potentially toward graph message passing.

#### Đoạn 3 / Paragraph 3 - Core idea

**Paste-ready English:**

In this paper, we investigate how the choice of IC operationalization determines whether GNN surrogate learning provides value beyond strong baselines. We compare two defensible operationalizations on the Twitch Gamers social network (168K nodes, 6.8M edges): (1) weighted cascade (A0), in which transmission probability depends only on target degree, and (2) HSCC, a domain-informed variant that incorporates source engagement velocity and cross-community amplification. Our contributions are threefold. First, we show that binary influence classification is structurally unstable on dense networks, motivating continuous regression as the principled prediction formulation. Second, we demonstrate that under degree-coupled IC, GNN architectures converge to the degree-centrality ceiling, confirming that the operationalization, rather than the model family, is the binding constraint. Third, under HSCC, we evaluate whether graph message passing adds value beyond the strongest matched flat baseline rather than assuming that neighborhood structure is automatically beneficial.

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

**Paragraph structure guidance cho §2.2 (1–2 câu per baseline group):**

_Paragraph 1 — Analytical baselines:_ "Degree centrality and k-core decomposition are widely used analytical proxies for influence potential in dense networks [Kitsak et al., 2010]. While computationally trivial, their relevance to simulation-based IC scores depends critically on the propagation model: degree-coupled transmission models structurally reward high-degree nodes, while engagement-driven models may decorrelate from degree."

_Paragraph 2 — Flat attribute models:_ "Logistic regression on node-level source attributes provides a strong flat baseline that captures engagement signals (views, account lifetime) without graph structure. A fair comparison against GNNs requires that flat baselines have matched feature access — both receiving the same attribute set — to isolate the contribution of message passing."

_Paragraph 3 — Node2Vec as shallow embedding baseline:_ "Node2Vec [Grover & Leskovec, 2016] generates structure-aware node embeddings via random walk sampling, providing a shallow structural baseline separate from both analytical and flat attribute approaches. We include it to distinguish graph-structure signal (Node2Vec) from message-passing signal (GNN)."

**Length target:** §2.2 nên là ~3 câu / 50–70 từ. Không cần phải cite mỗi baseline survey — chỉ cite những references trực tiếp support lý do chọn baseline đó.

#### 2.3 Graph surrogate / GNN

Nhắc ngắn **4 architectures đã test (active trong official rerun)**:

- GraphSAGE (SAGE — mean aggregation)
- GCN (symmetric normalization)
- GIN (sum + MLP, expressiveness-focused)
- APPNP (K=10 PPR propagation, decoupled transformation)

Mục tiêu là cho thấy shortlist hợp lý và được chạy dưới cùng một training protocol để fair comparison. Khi nhắc số kiến trúc trong paper body, dùng **"four architectures"** (không phải "five").

**Node2Vec (Grover & Leskovec, 2016):** active baseline trong shallow embedding group. Xem ghi chú precomputation bên dưới.

#### Lưu ý quan trọng

Không đi quá sâu vào literature của influence maximization. Paper này không tối ưu seed set; paper này học surrogate cho **continuous node-level IC scores**.

**Node2Vec (đã quyết định giữ trong table):** Khi viết paper, phân biệt rõ Node2Vec với degree/LR/MLP:

- Embedding generation cần ~153 giây (~2.5 phút) precomputation per regime theo frozen `runtime_breakdown.csv` — không phải real-time inference
- Trong setup section hoặc runtime table, ghi rõ: _"Node2Vec embedding generation + LR fit (training): ~153s; LR prediction (inference only): ~0.04s"_
- **Lưu ý semantics:** `train_sec` (153s) = embedding generation + LR fit bundled together; `inference_sec` (0.04s) = LR predict only. Không tách nhỏ hơn vì code không ghi riêng từng phần.
- Không frame Node2Vec như "fast inference" baseline — nó có offline preprocessing cost cao hơn degree/LR
- Trong Table 2/3, tách Node2Vec vào nhóm **"shallow embedding"** riêng hoặc thêm footnote về precomputation cost để không so sánh ngang hàng tốc độ với analytical baselines
- Nếu team **đã quyết định de-scope Node2Vec** trước freeze, phải xóa đồng bộ khỏi: (a) §4.1 setup list, (b) Table 2/3 plan, (c) §4.5 runtime table, (d) §2.3 background mention. Không để guide ở trạng thái vừa "giữ" vừa "không list". Tìm bằng keyword "Node2Vec" trong cả 2 file để đồng bộ.

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
p(u,v)=\mathrm{clip}\left(\lambda \frac{\phi(u)}{\deg(u)}(1+\gamma \mathbf{1}[c_u\neq c_v]),0,p\_{\max}\right)
\]

với:

\[
\phi(u)=\mathrm{rank}\left(\frac{\log(1+\mathrm{views}\_u)}{1+\mathrm{life_time}\_u}\right)/N
\]

- domain-informed alternative,
- source engagement velocity + community bridging,
- không phải "true Twitch diffusion model",
- chỉ là một comparative operationalization hợp lý để kiểm tra khi nào graph structure tạo thêm learnable value.

**Ba câu reviewer gần như chắc chắn sẽ hỏi và phải được trả lời ngắn ngay trong Section 3.2:**

- **Vì sao dùng rank thay vì raw value?** Vì `views` có phân phối heavy-tailed; rank normalization giúp hạn chế việc một số account cực lớn kéo toàn bộ scale và làm cho source term ổn định hơn giữa các rerun.
- **Vì sao dùng `log1p(views)/(1+life_time)`?** Vì paper cần một proxy gần với engagement velocity hơn cumulative popularity thuần túy; `log1p` nén outlier, còn chia cho `1+life_time` giúp tránh reward tuyệt đối cho account chỉ vì tồn tại lâu.
- **Vì sao khóa `lambda`, `gamma`, `p_max` thay vì tune?** Vì HSCC trong paper này là comparative operationalization đã freeze để kiểm tra regime learnability, không phải tham số được tối ưu để tạo ra GNN win lớn nhất.

**Paste-ready English để giải thích HSCC ngắn gọn nhưng defensible:**

> HSCC is introduced as a domain-informed comparative operationalization rather than as a validated generative law of Twitch diffusion. The rank-based source term improves robustness to the heavy-tailed views distribution, while the `log1p(views)/(1+life_time)` form is intended to approximate engagement velocity rather than cumulative popularity. The fixed community-amplification configuration is kept as a transparent, frozen comparative setting instead of being tuned to maximize downstream surrogate gains.

#### 3.3 Discriminativeness

Nên có một bảng ngắn kiểu này:

| Regime |                                                                                         Mean reach |             CV (per-band range) |                                                                                                               Max reach | Comment                                                                              |
| ------ | -------------------------------------------------------------------------------------------------: | ------------------------------: | ----------------------------------------------------------------------------------------------------------------------: | ------------------------------------------------------------------------------------ |
| A0     | 2.5–109 per degree band [✅ from `outputs/mapr2026_v3_results/degree_controlled_ic_variance.json`] | 1.12–2.29 across quintiles [✅] | [🔲 compute from the frozen Person 1 A0 score parquet (`ic_scores_a0.parquet` per team handoff) before filling Table 1] | heavy-tailed, degree-dominated; top band (deg 93–7613) mean=109 vs low band mean=2.5 |
| HSCC   |                                                                                          4.83 [✅] |            0.583 (overall) [✅] |                                                                                                              16.31 [✅] | selective, community-driven; low max reach reflects local propagation                |

**Frozen sources cho A0:**

- `outputs/mapr2026_v3_results/degree_controlled_ic_variance.json`: per-quintile CV [2.29, 1.89, 1.44, 1.12, 1.62] và mean reach [2.51, 7.46, 13.10, 24.30, 109.31]
- Overall A0 mean/median: [🔲 compute from the frozen Person 1 A0 score parquet (`ic_scores_a0.parquet` per team handoff) nếu thật sự cần exact number cho Table 1 trong paper]

**Frozen sources cho HSCC:**

- `outputs/mapr2026_v3_results/hscc_refined_label_diagnostics.json` — label stats: mean=4.83, max=16.31, std=2.82, CV=0.583

**Điểm cần viết rõ:**

- `A0`: broad nhưng degree-dominated,
- `HSCC`: mean nhỏ hơn nhưng vẫn discriminative,
- mean reach nhỏ dưới `HSCC` phải được diễn giải như **selective local-community diffusion**, không phải bằng chứng rằng operationalization vô nghĩa,
- không kéo `I-A` vào main text; nếu cần chỉ nhắc như archive note một câu.

#### 3.4 Stability and regression

Phần này phải rất crisp:

- binary top-k unstable,
- many communities span the decision boundary,
- gap-to-noise gần 0,
- instability là structural chứ không chỉ do Monte Carlo noise,
- vì vậy continuous regression là principled formulation.

**Paste-ready English:**

> Binary influence classification is structurally unstable: under A0 (weighted-cascade, formal diagnostic), mean top-decile Jaccard across three independent MC campaigns is 0.31 (exact: 0.307 — `ic_label_stability.json` [✅]) — far below the 0.85 stability target. This instability is not eliminated by increasing simulation runs, and instead reflects graph topology: 84.2% of communities span the top-k boundary band [✅], confirming structural rather than sampling origin. The argument extends to HSCC via invariant community topology and degree collapse (ρ = −0.006 [✅]).

**Rounding policy cho Jaccard:** Dùng **0.31** trong paper prose (hợp lý 2 decimal places). Dùng **0.307** khi cite artifact trực tiếp trong supplementary. Không dùng cả hai không nhất quán trong cùng một đoạn.

**Scope note:** Jaccard=0.31 là kết quả A0 formal diagnostic saja. Đừng attribute cho HSCC. Structural explanation (community overlap) và degree-collapse argument có thể nói "applies to both regimes."

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

- Twitch Gamers
- 168k nodes
- 6.8M edges
- 5k labeled nodes
- transductive split
- 5 seeds
- metrics: Spearman, NDCG@10, P@10
- **4 GNN architectures: SAGE, GCN, GIN, APPNP**
- Node2Vec + LR as shallow embedding baseline (offline precomputation ~153s/regime ≈ 2.5 min — from frozen `runtime_breakdown.csv`; separate from real-time inference)
- runtime reported separately

**APPNP evaluation status (bắt buộc phải ghi rõ trong setup và table):**

APPNP được chạy trong cả hai regime nhưng có seed variance vượt ngưỡng policy (`--gnn-std-threshold 0.1`):

- A0: APPNP std=0.417 ≥ 0.1 → **excluded from best-arch selection**
- HSCC: APPNP std=0.146 ≥ 0.1 → **excluded from best-arch selection**

APPNP **vẫn được list trong results table** để so sánh, nhưng phải có footnote: _"APPNP excluded from best-architecture comparison due to seed variance exceeding threshold (std > 0.1 in both regimes)."_ Không được chọn APPNP làm representative GNN cho bất kỳ claim nào.

**Node2Vec status (✅ IN table, với group separation bắt buộc):**

Node2Vec **được giữ trong main results table** (không bị loại). Tuy nhiên phải đặt ở nhóm riêng **"Shallow Embedding"**, tách khỏi analytical baselines (degree, etc.) và flat attribute models (LR, MLP). Lý do: Node2Vec cần ~153s precomputation offline — không so sánh ngang với analytical inference. Trong table caption và/hoặc experimental setup, phải ghi rõ:

> Node2Vec: embedding precomputation ≈153s offline per regime; reported inference time is downstream LR only (~0.04s).

> **[Rule HSCC-F1 — Feature Access Parity Lock (NON-NEGOTIABLE)]**
>
> The HSCC GNN (`gnn_raw_attr`) uses `include_language=true`. Therefore:
>
> - The official HSCC flat comparator **MUST** also include language features
> - **Locked comparator: `lr_degree_views_life_time_lang`** (ρ = 0.884) — cannot be changed post-freeze
> - Any flat baseline WITHOUT language gives the GNN an unfair feature advantage → invalid comparison
>
> **Verification:** `gnn_vs_baseline_bootstrap_ci_hscc.json` was computed vs `lr_degree_views_life_time_lang` (ρ=0.884). The paper's HSCC main claim (Δρ=+0.033, CI=[+0.021, +0.044]) is **defined relative to this comparator**. Changing the comparator changes the inferential claim — requires team consensus and re-documentation.
>
> **Paper-text signal:** Any sentence claiming "SAGE significantly outperforms the strongest flat baseline" must name this baseline explicitly at least once (in main text, table, or caption).

**Feature-access lock phải được chốt trước frozen run và giữ nhất quán trong toàn paper:**

- **Mode A (active for this paper):** GNN main comparison dùng `language` (`include_language=true`) → flat baselines tương ứng trong fairness table **cũng phải dùng `language`** → `lr_degree_views_life_time_lang` (ρ=0.884) là comparator chốt.
- **Mode B:** nếu team không muốn `language` đi vào fairness stack, phải bỏ `language` khỏi tất cả model của main comparison, không chỉ khỏi LR/MLP.

Không được viết paper trong trạng thái lửng lơ kiểu GNN có `language` nhưng baseline chính thì không.

#### 4.2 A0 results - structural ceiling

Phần này **không được viết như một failure**.

**Thông điệp chính:**

- degree / two-hop đã rất mạnh,
- GNN tiến gần ceiling đó,
- comparator bootstrap chính thức của `A0` là **degree**.

**Trong bảng nên có:**

- degree (ρ=0.826 — primary A0 comparator)
- one-hop / two-hop (contextual baselines)
- best flat LR variant (ví dụ: `lr_life_time` hoặc `lr_degree_views_life_time` — whichever is strongest in frozen baseline CSV)
- MLP (raw_attr)
- 4 GNN architectures (SAGE, GCN, GIN, APPNP) — APPNP có footnote về high variance
- Node2Vec + LR (shallow embedding group — riêng, không merge với GNN rows)

**FROZEN A0 RESULT — Paste-ready English (`gnn_significantly_worse`):**

> Under A0, the best GNN (GCN, raw node attributes) remains statistically below degree centrality (Δρ = −0.018, 95% CI [−0.029, −0.008]), indicating that the degree-coupled operationalization imposes a structural learnability ceiling: the GNN cannot overcome the ceiling that degree itself defines as an analytical baseline.

**Lưu ý quan trọng:** Kết quả KHÔNG phải "practically equivalent" — CI = [−0.029, −0.008] không nằm trong [−0.02, +0.02]. Cụ thể: |CI lower bound| = 0.029 > δ₀ = 0.02, nên không thỏa equivalence. Phải dùng "statistically below", không được dùng "practically equivalent" cho A0. Đây vẫn là kết quả **tốt cho paper** — nó confirm hypothesis structural ceiling.

**Cách viết A0 result KHÔNG như failure — hướng dẫn cụ thể:**

Khi viết §4.2 prose, phải theo logic sau:

1. **Mở đầu bằng narrative của regime**, không phải bằng failure: _"Under A0, the label-generation mechanism is degree-coupled by construction..."_
2. **Nêu strength của baselines trước**: degree ρ=0.826, k-shell=0.816, two-hop strong — "analytical baselines already capture most of the surrogate signal"
3. **Kết quả GNN trong context**: GCN ρ=0.808 — "approaches the ceiling but remains statistically below it" — không phải "GNN fails"
4. **Đóng bằng mechanism**: "This result confirms the structural ceiling hypothesis: when labels are degree-coupled, graph learning cannot recover signal that degree already encodes analytically"
5. **Architecture-regime insight**: Ghi rõ GCN wins A0 (ρ=0.808), SAGE poor (ρ=0.534) — đây là finding, không phải sidenote

**A0 architecture results (frozen):**

- GCN: ρ=0.808 ± 0.001 → best under A0
- GIN: ρ=0.615 ± 0.029 → moderate
- SAGE: ρ=0.534 ± 0.009 → poor under A0
- APPNP: std=0.417 → excluded from best-arch comparison

Ngoài ra, vẫn nên nêu diagnostic kiểu `GNN-raw-attr vs MLP` để cho thấy message passing học được graph signal (GCN raw_attr ρ=0.808 > MLP ρ=0.435 [✅ from `baseline_ranking_metrics_a0_clean.csv`]), nhưng signal đó không vượt được structural ceiling do `A0` áp đặt.

#### 4.3 HSCC results - graph-aware regime

Đây là subsection main-claim.

**Thông điệp chính:**

- degree collapse từ ρ=0.826 (A0) xuống ρ=−0.006 (HSCC) — phải được nêu rõ trong body text
- official comparator: `lr_degree_views_life_time_lang` (ρ=0.884) — locked by frozen bootstrap CI artifact
- GNN chỉ có giá trị nếu học được graph/community structure beyond raw attributes

**Degree collapse — bắt buộc phải nêu trong §4.3 body (không chỉ trong abstract):**

Khi viết §4.3, phải có ít nhất một câu/đoạn explicit về degree collapse:

> Under HSCC, degree centrality collapses to ρ = −0.006 — a shift of 0.832 Spearman points from its A0 value (ρ = 0.826). This dramatic collapse signals a fundamental regime change: degree is no longer informative under source-community propagation, and the relevant comparator shifts to LR with source-side attributes.

Câu này **phải xuất hiện trong body text của §4.3**, không chỉ trong abstract. Reviewer sẽ hỏi nếu paper chỉ claim degree collapse ở abstract mà không explain mechanism trong body.

**Cấu trúc §4.3 prose được đề xuất:**

1. **Mở đầu bằng regime context**: "Under HSCC, the source-velocity operationalization introduces source-side engagement and community structure into the label..."
2. **Nêu degree collapse với số cụ thể**: ρ=0.826 → ρ=−0.006, shift=0.832
3. **Chuyển sang comparator**: "The relevant comparator therefore shifts to LR with source-side attributes..."
4. **Nêu flat baselines**: LR(life_time), LR(views+life_time), LR(degree+views+life_time+language) ρ=0.884
5. **GNN result**: SAGE ρ=0.915, Δρ=+0.033, CI=[+0.021, +0.044] — "significant improvement suggesting residual graph-structured signal"
6. **Architecture-regime insight**: SAGE wins HSCC, GCN drops to 0.602, GIN collapses to 0.028
7. **GIN collapse reporting** (bắt buộc — xem dưới)

**GIN collapse — bắt buộc report trong paper, không hide:**

GIN ρ=0.028 dưới HSCC là finding quan trọng. Cần viết một câu trong §4.3 về điều này:

> GIN exhibits near-random performance under HSCC (ρ = 0.028), consistent with the hypothesis that sum aggregation without normalization fails to stabilize message passing under the heavy-tailed source-velocity signal — in contrast to SAGE's mean aggregation (ρ = 0.915) and GCN's symmetric normalization (ρ = 0.602).

GIN collapse không được loại bỏ khỏi table hoặc từ bỏ không nhắc trong prose — đây là kiến trúc được chạy chính thức và kết quả của nó là một comparative finding.

**HSCC architecture results (frozen):**

- SAGE: ρ=0.915 ± 0.004 → best under HSCC
- GCN: ρ=0.602 ± 0.014 → moderate (drops from A0)
- GIN: ρ=0.028 ± 0.046 → near-random collapse
- APPNP: std=0.146 → excluded from best-arch comparison
- SAGE rankloss: ρ=0.924 ± 0.002 → BOOST variant

**Bảng HSCC bắt buộc nên có:**

- degree (footnote: "included for regime-contrast reference only; ρ = −0.006")
- `LR(life_time)` ở vị trí nổi bật đầu tiên trong nhóm flat baselines
- `LR(views + life_time)`
- `LR(degree + views + life_time)`
- `MLP(raw attrs)`
- nếu GNN dùng pipeline `raw_attr` hiện tại (Mode A — active):
  - **`lr_degree_views_life_time_lang`** (ρ=0.884) ← **primary HSCC comparator** — phải in đậm, đánh dấu ★ hoặc tô nền nhạt trong table để reviewer thấy ngay đây là reference row
  - `MLP(... + language)`

**Cross-reference:** `lr_degree_views_life_time_lang` = `LR(degree, views, life_time, language)` — hai cách viết tương đương. Trong table: dùng tên ngắn gọn nhất mà vẫn unique; trong caption: viết đầy đủ. Nhất quán trong toàn paper (xem Rule F4 trong Paper rules.md).

**Lưu ý bắt buộc cho codebase hiện tại:** nếu `raw_attr` dùng cột `language`, language dummies sẽ được đưa vào features tự động. Vì vậy, trừ khi team chủ động disable và ghi lại quyết định đó trước frozen run, fairness baselines có `language` là bắt buộc.

**Comparator rule cho HSCC phải viết thật rõ trong guide và trong paper:**

- comparator chính trong main paper là **official flat comparator locked in the frozen bootstrap CI artifact under matched node-level feature access**,
- `LR(life_time)` vẫn nên được hiện diện nổi bật vì reviewer gần như chắc chắn sẽ nhìn vào nó như một baseline nguồn-thuộc-tính đơn giản và mạnh, **không phải** vì nó là strongest HSCC baseline,
- degree chỉ còn là contextual evidence cho regime shift, không phải comparator quyết định claim,
- `phi`, `lr_phi`, `phi × (1 + cross_community_fraction)`, và mọi formula-derived oracle row chỉ được dùng như interpretation / ceiling / appendix diagnostics.

**Rule thao tác khi draft câu kết quả HSCC:** trong bản final, thay cụm generic như **"strongest flat baseline"** bằng đúng tên comparator frozen (ví dụ `LR(life_time)`, `LR(views+life_time)`, hoặc `LR(... + language)`). Reviewer gần như chắc chắn sẽ hỏi baseline nào là comparator thật.

**Rule cho main baseline table:** không đặt các oracle-style rows cạnh LR/MLP/GNN như thể đó là comparator công bằng của main paper. Nếu cần dùng, tách thành diagnostic note hoặc appendix mini-table riêng.

**FROZEN HSCC RESULT — Paste-ready English (`gnn_significantly_better`) ← ✅ ACTIVE:**

> Under HSCC, the best GNN (SAGE, raw node attributes including language features) significantly outperforms the strongest matched flat baseline — LR with degree, views, life_time, and language (ρ = 0.884) — achieving ρ = 0.915 (Δρ = +0.033, 95% CI [+0.021, +0.044]), consistent with residual neighborhood-structured signal beyond node-level attributes.

**Comparator được chốt: `lr_degree_views_life_time_lang` (ρ=0.884).** Không được thay đổi comparator này trong các draft sau khi đã viết claim này (Rule F3 comparator lock).

**Near-tie disclosure (bắt buộc ghi vào paper hoặc footnote):** Theo `baseline_ranking_metrics_hscc_clean.csv`, `lr_views_life_time_lang` đạt ρ=0.88442 và `lr_degree_views_life_time_lang` đạt ρ=0.88430 — chênh lệch chỉ 0.00012 (4th decimal place). Hai baseline này thực tế là **tied at ρ ≈ 0.884**. Official comparator được chọn là `lr_degree_views_life_time_lang` vì đó là model được lock trong frozen bootstrap CI artifact (`gnn_vs_baseline_bootstrap_ci_hscc.json`), không phải vì nó mạnh hơn theo point estimate.

**Cách viết defensible trong paper (dùng một trong hai cách):**

- Option A (ngắn): "the strongest flat baseline under matched feature access (ρ = 0.884)" — chấp nhận được vì ρ=0.88430 ≈ ρ=0.88442
- Option B (chính xác hơn): "the official comparator from the pre-specified bootstrap CI artifact — LR(degree, views, life_time, language), ρ = 0.884 — which is effectively tied with LR(views, life_time, language) at the fourth decimal place"

Nếu reviewer hỏi tại sao không dùng `lr_views_life_time_lang`: answer = "Both baselines are within 0.001 ρ points. We use `lr_degree_views_life_time_lang` as the official comparator because it was pre-specified in the bootstrap CI artifact before results were analyzed."

**[Archive — không dùng cho paper này — frozen result đã xác nhận `gnn_significantly_better`]**

_Nếu GNN chỉ xấp xỉ:_ `lr_degree_views_life_time_lang (ρ = 0.884) already explains most of the source-driven component...`

_Nếu GNN thua:_ `The best GNN remains below lr_degree_views_life_time_lang (ρ = 0.884)...`

Paper vẫn ổn nếu contrast `A0 vs HSCC` còn mạnh và viết trung thực.

**Comparator lock cho bản MAPR (đã resolved — không cần thay đổi):**

- comparator chính của `HSCC`: **`lr_degree_views_life_time_lang`** (ρ=0.884) — frozen, không thay đổi
- `phi`, `lr_phi`, và các oracle-style decomposition liên quan chỉ dùng để giải thích cơ chế, không dùng làm comparator chính

#### 4.4 Contrast analysis

Đây là trái tim intellectual của paper.

**A0**

- label degree-coupled,
- analytical baselines mạnh (degree ρ=0.826),
- GNN near ceiling nhưng dưới degree (GCN ρ=0.808, Δρ=−0.018).

**HSCC**

- label có source term + graph/community amplification term,
- degree collapse (ρ=0.826 → ρ=−0.006, shift=0.832),
- flat baselines chủ yếu capture source term (lr_dvtl_lang ρ=0.884),
- GNN thêm được Δρ=+0.033 residual (SAGE ρ=0.915).

`phi`, `lr_phi`, và related oracle decompositions nên được dùng như interpretation rows, không phải main baselines.

**Paste-ready English (regime contrast — frozen):**

> The contrast between A0 and HSCC shows that surrogate learnability is not a property of the model alone; rather, it is jointly determined by the diffusion operationalization and the information already recoverable by simple baselines. Under A0, degree centrality captures most of the label signal, leaving no recoverable margin for graph learning. Under HSCC, degree becomes uninformative (ρ = −0.006), shifting the competitive landscape toward source-side attribute models — and graph message passing adds a further increment beyond even the strongest of these.

#### 4.4.1 Architecture-regime interaction (finding bắt buộc phải viết)

Đây là finding cần xuất hiện trong §4 body, không chỉ trong checklist hoặc guide.

**Frozen architecture-regime results:**

| Architecture         | A0 (ρ)               | HSCC (ρ)             | Pattern                    |
| -------------------- | -------------------- | -------------------- | -------------------------- |
| GCN (symmetric norm) | **0.808**            | 0.602                | Best A0, moderate HSCC     |
| SAGE (mean agg)      | 0.534                | **0.915**            | Poor A0, best HSCC         |
| GIN (sum agg)        | 0.615                | 0.028                | Moderate A0, collapse HSCC |
| APPNP (K-PPR)        | excluded (std=0.417) | excluded (std=0.146) | Unstable both              |

**Framing cho paper body (paste-ready English):**

> The best-performing architecture differs by regime: GCN under A0 (ρ = 0.808) and SAGE under HSCC (ρ = 0.915). This architecture-regime interaction is consistent with the hypothesis that operationalization choice governs inductive bias alignment — GCN's symmetric normalization aligns naturally with degree-coupled propagation, while SAGE's mean aggregation is better suited to aggregate source-side engagement signals across community boundaries.

**GIN collapse (paste-ready sentence for §4.3 or §4.4):**

> GIN exhibits near-random ranking performance under HSCC (ρ = 0.028), in contrast to its moderate A0 performance (ρ = 0.615). This collapse under source-velocity operationalization — despite GIN's theoretical expressiveness advantages — suggests that sum aggregation without normalization fails to stabilize the message-passing process when the label signal originates from source-side attributes with heavy-tailed distributions.

**Rule viết architecture comparison trong paper:** Một đoạn ngắn (~3-4 câu) trong §4.3 hoặc §4.4 là đủ. Không viết §4.4.1 như một riêng lẻ section nếu không đủ chỗ — tích hợp vào regime results paragraphs là tốt nhất cho 6-trang MAPR format.

#### 4.4.2 C3 Rankloss — placement guidance (✅ FROZEN, đưa vào main paper)

**Status:** FROZEN — `gnn_vs_rankloss_bootstrap_ci_hscc.json` đã verified, `loss_mode=rankloss_combined`, `rankloss_alpha=0.5`, best arch=SAGE (ρ=0.924).

**Vị trí tốt nhất trong paper:**

1. **Table 3 (HSCC results):** Thêm một row riêng cho SAGE + rankloss, đặt ngay dưới row `gnn_raw_attr` (SAGE standard). Label: `SAGE + rankloss` hoặc `GNN (rankloss, C3)`. Ghi ρ=0.924 ± 0.002.
2. **§4.3 prose:** Thêm một câu ngắn: _"A ranking-aware training variant achieves ρ = 0.924 (+0.009 descriptive gain over standard Huber training) and significantly outperforms the flat comparator by Δρ = +0.041 (95% CI [+0.030, +0.053])."_
   - **Lưu ý:** "+0.009 vs standard SAGE" là **descriptive** (không có paired bootstrap → không dùng "significantly"). "+0.041 vs comparator" là **inferential** (có CI từ frozen JSON → dùng "significantly").
3. **Conclusion:** Thêm optional sentence (Rule C2): _"Ranking-aware training provides a further marginal gain under HSCC, suggesting that explicit rank supervision is a viable extension when the operationalization is source-side-driven."_

**Không nên đẩy xuống appendix** vì:

- Artifact đã frozen và kết quả significant vs comparator (CI [+0.030, +0.053] không span zero)
- Là một đóng góp nhỏ nhưng concrete — giúp làm phong phú thêm §4.3 mà không cần nhiều chỗ
- Nếu thiếu chỗ, cắt oracle decomposition rows hoặc discussion prose trước khi cắt C3 row

#### 4.5 Runtime

Giữ phần này ngắn và thực dụng:

- MC-IC labeling cost
- GNN training cost
- GNN inference cost
- analytical baseline inference gần như bằng 0

**Cách diễn đạt đúng:**

- speedup **vs MC-IC**
- không so speedup của GNN với degree như thể chúng cùng chi phí labeling

**Frozen runtime values (từ `runtime_breakdown.csv`):**

| Step                                                      | Time                                                                                                |
| --------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| MC-IC labeling (1 pass, full graph)                       | **480.3s**                                                                                          |
| GNN inference (full graph, SAGE `gnn_raw_attr`, HSCC row) | **~0.086s**                                                                                         |
| Speedup (MC-IC ÷ GNN inference)                           | **~5,590×** (using `hscc,gnn_raw_attr` in `runtime_breakdown.csv`; round to ~5,500× in paper prose) |
| Node2Vec precomputation                                   | ~153s (~2.5 min)                                                                                    |
| Node2Vec downstream LR inference                          | ~0.040s                                                                                             |
| degree / analytical baseline                              | ~0.001s                                                                                             |

**Paste-ready English:**

> Once trained, the GNN surrogate provides full-graph influence score inference in approximately 0.086 seconds, compared with 480 seconds for a single MC-IC labeling pass — a speedup of approximately 5,500×.

**Runtime anchor:** Số headline ở đây dùng đúng row `hscc,gnn_raw_attr` trong `runtime_breakdown.csv` (`inference_sec_full_graph` ≈ 0.08596s). Nếu đổi sang row khác như `gnn_full` (~0.091s), phải đổi cả tốc độ và speedup cùng lúc.

**Rounding policy:** Bảng frozen dùng `~0.086s`; paper prose dùng `0.086s` hoặc `under 0.1 seconds`.

Không dùng "5,580×" trực tiếp trong paper — round về "approximately 5,500×" hoặc "over 5,000×" để tránh false precision.

**Prose order cho §4.5 runtime paragraph:**

1. MC-IC cost first (480s — the expensive step being replaced)
2. GNN inference cost (0.086s from `hscc,gnn_raw_attr` — the fast surrogate)
3. Speedup ratio (~5,500×)
4. Clarify: speedup is inference vs one labeling pass — not training time
5. Node2Vec separately: "Node2Vec requires ~153s offline precomputation per regime; downstream LR prediction is ~0.04s"
6. Analytical baselines last: "degree and k-shell are near-instantaneous (analytical formula)"

**Report GNN inference with best-performing arch:** Speedup dùng SAGE `gnn_raw_attr` (~0.086s in the HSCC row used for the headline runtime claim) — không cần report per-architecture times trong §4.5. Một câu về SAGE là sufficient; "GNN" = best-performing arch in that regime.

**Test set size clarification (for reviewer defense):** `1,000 test nodes` is the held-out test set size. `1,000 bootstrap resamples` is the resampling count for CI computation. These are independent numbers — do not conflate in paper text or reviewer response.

### Mục 5 / Section 5 - Discussion & Limitations (0.5 trang)

#### 5.1 When does GNN help?

Câu trả lời phải nhất quán:

- not universally,
- only when the target depends on graph-mediated information not already captured by strong flat baselines.

#### 5.1 When does GNN help? — Paste-ready paragraph

> Our results reveal a regime-dependent answer: graph message passing adds value when the label generation process encodes structure that node-level attributes cannot fully capture alone. Under A0 — where transmission probability is degree-coupled — degree centrality already provides a near-optimal surrogate, leaving little residual for GNN message passing. Under HSCC — where transmission depends on source engagement velocity and cross-community bridging — node-level flat models remain competitive but graph message passing recovers additional signal (Δρ = +0.033, CI [+0.021, +0.044]). The implication is that the question "do GNNs help for influence prediction?" has no universal answer; it depends on what the diffusion operationalization encodes.

#### 5.2 Limitations — với paste-ready English cho từng điểm

**Limitation 1 — follower graph ≠ observed diffusion:**

> Our graph is a follower network, whereas actual information cascades propagate along a different, unobserved subgraph of active interactions. Results should therefore be interpreted as properties of MC-IC surrogates on the follower topology, not as measurements of real Twitch information spread.

**Limitation 2 — operationalizations, not ground truth:**

> Both A0 and HSCC are simulation-based operationalizations of influence rather than empirically validated diffusion laws. Findings describe the learnability of each operationalization's output, not the learnability of real influence.

**Limitation 3 — HSCC not empirically validated:**

> The HSCC formula encodes domain-informed hypotheses (source velocity, community bridging) without direct calibration to observed cascade data. We claim only that it introduces community-side signal into label generation in a transparent, frozen configuration — not that it accurately reflects Twitch diffusion mechanics.

**Limitation 4 — transductive evaluation:**

> Our evaluation is transductive: models are trained and evaluated on the same graph with held-out test nodes. Inductive generalization — applying the trained surrogate to new graphs or temporally shifted snapshots — is not assessed and remains an open evaluation challenge.

**Limitation 5 — small HSCC reach:**

> Under HSCC, mean cascade reach is 4.83 nodes — substantially below A0. This reflects the selective local-community propagation structure of the HSCC formula, not a deficiency of the operationalization. Rankings over these small cascades are still meaningful for identifying the most-connected community hubs, but the absolute influence magnitudes are not comparable to broad viral spread scenarios.

**Limitation 6 (optional, if needed for space-filling) — single dataset:**

> All experiments use a single snapshot of the Twitch Gamers follower graph. Generalizability to other social platforms, temporal dynamics, or different community structures has not been tested.

Nếu paper dùng `language` trong GNN, phải nói rõ rằng fairness của baseline feature access là điều kiện để mọi claim về GNN có giá trị.

#### 5.3 Why not learn p from data?

**Paste-ready English:**

> Estimating edge-level transmission probabilities would require supervised cascade logs that are unavailable in this dataset; weighted cascade and HSCC therefore provide principled zero-shot alternatives.

#### 5.4 Conclusion section — Paste-ready paragraph (≤5 sentences)

> We studied the learnability of two IC operationalizations on the Twitch Gamers social graph. Under a degree-coupled regime (A0), the best GNN remains statistically below degree centrality (Δρ = −0.018, CI [−0.029, −0.008]), confirming that the diffusion operationalization imposes a structural ceiling that degree already saturates. Under a source-community regime (HSCC), graph message passing significantly outperforms the strongest matched flat baseline (Δρ = +0.033, CI [+0.021, +0.044]), with an optional ranking-aware variant reaching Δρ = +0.041. In all cases, trained surrogates provide influence score inference orders of magnitude faster than repeated MC simulation. Our central finding is that operationalization choice, rather than architectural capacity, governs when graph learning adds value for simulation-based influence prediction.

**Sentence assignment:**

- Sentence 1: Context / scope statement (one graph, two operationalizations)
- Sentence 2: A0 finding with frozen CI (structural ceiling confirmed)
- Sentence 3: HSCC finding with frozen CI (graph learning adds value here)
- Sentence 4: Runtime / speedup (optional if space-constrained)
- Sentence 5: Central take-away (operationalization governs GNN value)

---

## Phần 3: Hình, bảng, và abstract

### 3.1 Hình bắt buộc nên có

#### Hình 1 / Figure 1

Pipeline diagram — **5 boxes, trái sang phải:**

1. **Input** — "Twitch Gamers Graph (168K nodes, 6.8M edges) + node attributes (views, life_time, language)"
2. **Operationalization** — hai nhánh từ cùng một graph:
   - Nhánh trên: "A0: p(u,v) = 1/deg(v)"
   - Nhánh dưới: "HSCC: p(u,v) = clip(λ·φ(u)/deg(u)·(1+γ·1[c_u≠c_v]), 0, p_max)"
3. **MC-IC Simulation** — "Labeling: 5,000 nodes, 200 runs/node → continuous influence scores"
4. **Surrogate Models** — "Analytical (degree, k-shell) | Flat (LR, MLP) | GNN (SAGE, GCN, GIN, APPNP)"
5. **Outputs** — "Regime-specific claims + runtime comparison vs MC-IC"

Mỗi box ngắn gọn, không quá 2 dòng text. Arrows chỉ chảy trái → phải; không có feedback loop. Operationalization branch nên được đánh dấu rõ là "fork" (ví dụ: fork shape hoặc 2 parallel arrows labelled A0 / HSCC).

**Figure 1 nên trả lời được đúng một câu hỏi:** từ cùng một graph, khi đổi operationalization thì downstream surrogate-learning story đổi như thế nào. Không nhét `I-A`, `A2`, hoặc archive branches vào figure chính.

#### Hình 2 / Figure 2

Two-panel results figure — **layout cụ thể:**

- **Trái panel (A0):** dot plot, trục x = Spearman ρ [0.0, 1.0], trục y = model rows (sorted by ρ descending)
- **Phải panel (HSCC):** dot plot, cùng model rows, trục x = Spearman ρ [−0.1, 1.0] để degree collapse thấy được
- **Model rows (cùng order ở cả 2 panel, top-to-bottom):**
  - Analytical group: degree, k-shell, pagerank, one-hop, two-hop
  - Flat attr group: LR(life_time), LR(views+LT), LR(degree+views+LT+lang) ← `lr_degree_views_life_time_lang`
  - Shallow embedding: Node2Vec+LR _(italic or lighter weight)_
  - GNN group: SAGE, GCN, GIN, APPNP† — với SAGE rankloss ở cuối GNN group
- **Comparator line (vertical dashed):**
  - A0: degree ρ=0.826
  - HSCC: `lr_degree_views_life_time_lang` ρ=0.884
- **Error bars:** ± 1 std dev từ 5 seeds (not CI — CI được báo cáo trong bảng)
- **Color/grayscale policy:** tất cả phải readable ở grayscale — dùng marker shape (●▲■◆) để phân biệt group, không chỉ dùng màu

†APPNP: plot điểm nhưng mark bằng dagger (†) và ghi footnote: "Excluded from best-arch comparison (seed std > 0.1)."

Nếu còn chỗ, NDCG@10 có thể để trong table thay vì figure. Figure chính nên ưu tiên làm rõ regime contrast thay vì nhồi nhiều metric.

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
- strongest flat baseline
- MLP
- Node2Vec + LR (shallow embedding; note offline precomputation separately)
- GNNs

Nếu thiếu chỗ, giữ đủ các rows để reviewer vẫn thấy được 3 lớp comparator: analytical, flat, GNN.

#### Bảng 3 / Table 3

HSCC results:

- `LR(life_time)` ở vị trí nổi bật đầu tiên
- `LR(views+life_time)`
- `LR(degree+views+life_time)`
- nếu có `language` trong main GNN features thì thêm matched `LR(... + language)` và `MLP(... + language)`
- `MLP`
- `Node2Vec + LR` (shallow embedding group; footnote precomputation cost)
- GNNs

Không để `phi`, `lr_phi`, hoặc oracle decomposition rows trong main Table 3. Nếu cần dùng để giải thích cơ chế, chuyển chúng sang note diễn giải hoặc appendix.

#### Bảng 4 / Table 4 (nếu còn chỗ)

Runtime mini-table.

Nếu thiếu chỗ, merge runtime vào main results table dưới dạng cột cuối.

### 3.3 Abstract template

Abstract nên giữ đúng **6 câu** (template dưới đây), lý tưởng **không quá 150 từ**. Tất cả số đã frozen — có thể dùng trực tiếp. Câu 6 (runtime) là optional nếu thiếu chỗ.

**Câu 1 / Sentence 1 - Problem + difficulty**

> Identifying influential users in static social networks without behavioral cascade logs requires simulation-based operationalizations of influence, yet the learnability of such operationalizations remains poorly understood.

**Câu 2 / Sentence 2 - Method**

> We study two Monte Carlo Independent Cascade (MC-IC) operationalizations on the Twitch Gamers social network: a structural weighted-cascade regime (A0) and a domain-informed source-community regime (HSCC).

**Câu 3 / Sentence 3 - Stability / regression**

> We show that binary top-k influence labels are structurally unstable, motivating continuous regression on simulation-derived influence scores.

**Câu 4 / Sentence 4 - Main contrast (FROZEN)**

> Under the degree-coupled regime (A0), the best GNN (GCN) remains statistically below degree centrality (Δρ = −0.018), confirming an analytical structural ceiling; under the source-community regime (HSCC), degree centrality collapses (from ρ = 0.826 to ρ = −0.006) and the strongest flat baseline shifts to LR with source-side attributes (ρ = 0.884).

**Câu 5 / Sentence 5 - GNN result (FROZEN)**

> Across four GNN architectures, graph message passing fails to improve upon degree centrality under A0 — in fact remaining statistically below it (Δρ = −0.018, 95% CI [−0.029, −0.008]) — but achieves Δρ = +0.033 (95% CI [+0.021, +0.044]) over the strongest flat baseline under HSCC; a ranking-aware variant further reaches Δρ = +0.041 over the same comparator — demonstrating that operationalization, not architecture, governs when graph learning adds value.

**⚠️ Lưu ý:** "No significant lift" (cũ) ngụ ý CI spans zero (inconclusive). Sai. CI A0 = [−0.029, −0.008] là fully negative → GNN **statistically WORSE** than degree, không phải neutral. Câu trên đã sửa để phản ánh đúng direction.

**Câu 6 / Sentence 6 - Runtime**

> In all cases, learned surrogates provide inference that is orders of magnitude faster than repeated MC simulation.

**Rule bắt buộc cho abstract:**

- không viết `"outperforms"` hoặc `"significantly improves"` nếu chưa có bootstrap artifact đúng regime chống đỡ,
- câu về `HSCC` phải so với **strongest frozen flat baseline**, không so với degree,
- nếu GNN dưới `HSCC` chỉ xấp xỉ hoặc thua strongest flat baseline, phải rewrite Sentence 5 theo framing comparative-regime thay vì superiority framing.

---

## Phần 4: Claims, evidence, và wording guardrails

### Claim-to-artifact map bắt buộc

| Claim / paper sentence type                                                                                                         | Required artifact(s) tối thiểu                                                                                                                                                                                                                                                                                        |
| ----------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Binary instability là structural (A0 formal; HSCC by topology extension)                                                            | `outputs/day1_benchmark/stability_explanation.json` [✅ EXISTS], `ic_label_stability.json` [✅ Jaccard=0.307], `phase1_community_overlap.json` [✅ 84.2% community overlap], `phase2_threshold_analysis.json` [✅ max Jaccard=0.657 across all k] — tất cả tại `outputs/day1_benchmark/` và `outputs/ic_feasibility/` |
| `A0` là structural ceiling / GNN statistically below degree (**NOT equivalence**)                                                   | `gnn_vs_degree_bootstrap_ci_a0.json` ✅ — GCN ρ=0.808 vs degree ρ=0.826, delta=−0.018, CI=[−0.029, −0.008]; `surrogate_ranking_metrics_a0_clean.csv` ✅                                                                                                                                                               |
| `HSCC` là graph-aware regime / GNN gain beyond flat baselines                                                                       | `gnn_vs_baseline_bootstrap_ci_hscc.json` ✅ — SAGE ρ=0.915 vs `lr_degree_views_life_time_lang` ρ=0.884, delta=+0.033, CI=[+0.021, +0.044]; `surrogate_ranking_metrics_hscc_clean.csv` ✅                                                                                                                              |
| Runtime / speedup claim                                                                                                             | `runtime_breakdown.csv` ✅ — MC-IC=480.3s; headline runtime uses `hscc,gnn_raw_attr` inference≈0.086s; speedup≈5,590× (round to ~5,500× in paper prose)                                                                                                                                                               |
| **C3 rankloss significantly improves over the flat comparator; +0.009 vs standard SAGE is descriptive only [🟡 BOOST — ✅ FROZEN]** | `gnn_vs_rankloss_bootstrap_ci_hscc.json` ✅ — SAGE rankloss ρ=0.924 vs `lr_degree_views_life_time_lang` ρ=0.884; delta=+0.041, CI=[+0.030, +0.053]; `loss_mode=rankloss_combined`, `rankloss_alpha=0.5` confirmed                                                                                                     |

**Universal wording rule:** mọi câu có từ như `"outperforms"`, `"significantly improves"`, `"practically equivalent"`, `"ceiling"`, hoặc `"dominates"` đều phải map được về đúng artifact freeze của regime tương ứng. Nếu chưa map được, hạ câu đó xuống mức mô tả mềm hơn hoặc bỏ.

### 4.1 Claim 1

**Paste-ready English claim (✅ FULLY SUPPORTED — all artifacts confirmed):**

> Binary top-k influence labels are structurally unstable under IC simulation: across three independent MC campaigns (A0 weighted-cascade regime), mean top-decile Jaccard is 0.31 — far below the 0.85 stability target. The instability arises from graph topology rather than simulation variance: 84.2% of communities span both the top-k and boundary band, preventing stable binary separation regardless of threshold or run count. This motivates continuous regression on simulation-derived influence score means as the primary formulation, applied consistently across both A0 and HSCC regimes.

**Scope và evidence — quan trọng:**

| Artifact                                                         | Nội dung                                                             | Phạm vi                                                |
| ---------------------------------------------------------------- | -------------------------------------------------------------------- | ------------------------------------------------------ |
| `outputs/day1_benchmark/ic_label_stability.json`                 | Jaccard_mean=0.307, Spearman_mean=0.685 across 3 seeds               | A0 only (weighted_cascade)                             |
| `outputs/day1_benchmark/stability_explanation.json`              | interpretation="structural"; pct_communities_spanning_boundary=0.842 | A0 scores; community topology = regime-invariant       |
| `outputs/ic_feasibility/phase1_community_overlap.json`           | Community-set Jaccard=0.842 > 0.7 threshold                          | A0 scores; community structure applies to both regimes |
| `outputs/ic_feasibility/phase2_threshold_analysis.json`          | Max estimated Jaccard=0.657 across k=3%–30%                          | A0 only                                                |
| `outputs/ic_feasibility/pivot_decision_report.json`              | evidence_statement_for_paper ready; PIVOT_CONFIRMED                  | A0 only                                                |
| `outputs/mapr2026_v3_results/degree_controlled_ic_variance.json` | CV 1.12–2.29 across quintiles                                        | A0 only                                                |

**Quy tắc khi viết về Claim 1:**

- Jaccard=0.31 → chỉ cite cho A0 ("under the weighted-cascade regime" hoặc "formal diagnostic under A0")
- Structural argument (community overlap 84.2%) → có thể nói áp dụng cho "cả hai regimes" vì graph topology không đổi theo IC model
- HSCC extension argument (không cần artifact riêng): degree collapse (ρ=−0.006) làm cho bất kỳ degree-anchored binary threshold nào vô nghĩa; reach thấp (mean=4.83, CV=0.583) và source-velocity signal → binary boundary còn arbitrary hơn A0
- KHÔNG nói "Jaccard=0.31 under HSCC" — đó là A0 number

**Paste-ready English cho §3.5 — A0 formal diagnostic:**

> Under the weighted-cascade operationalization (A0), formal label stability analysis yields a mean top-decile Jaccard of 0.31 across three independent MC campaigns — far below the 0.85 stability target. Maximum estimated Jaccard remains below 0.66 across all percentile thresholds (3%–30%), indicating that no choice of k rescues binary labeling. The root cause is structural: 84.2% of communities span both the top-k and boundary band, confirming that rank noise is topology-driven rather than a sampling artifact.

**Paste-ready English cho HSCC extension (same §3.5):**

> Under HSCC, the structural instability argument extends by analogy: community structure is invariant to IC parameterization, and the degree-collapse result (ρ = −0.006) renders any degree-anchored binary threshold meaningless. The lower mean reach (4.83 vs. >100 nodes under A0) and source-velocity-driven signal further increase boundary arbitrariness. Accordingly, continuous regression is adopted as the primary formulation for both operationalizations.

### 4.2 Claim 2

**Paste-ready English claim (FROZEN — `gnn_significantly_worse`):**

> Under degree-coupled IC (A0), the best GNN (GCN, raw node attributes) remains statistically below degree centrality (Δρ = −0.018, 95% CI [−0.029, −0.008]), consistent with a structural ceiling imposed by the degree-coupled operationalization.

**Lưu ý quan trọng:** Claim này KHÔNG phải "practical equivalence". CI = [−0.029, −0.008] không nằm trong [−0.02, +0.02] → không đạt equivalence bound. Interpretation đúng: `gnn_significantly_worse`. Đây vẫn là kết quả tốt — nó confirm hypothesis structural ceiling, không phải counterevidence.

**Required evidence (✅ all frozen):**

- `gnn_vs_degree_bootstrap_ci_a0.json` — GCN, delta=−0.018, CI=[−0.029, −0.008] ✅
- `surrogate_ranking_metrics_a0_clean.csv` — GCN ρ=0.808 ✅
- `baseline_ranking_metrics_a0_clean.csv` — degree ρ=0.826 ✅
- diagnostic `GNN-raw-attr vs MLP-raw-attr` (GCN raw_attr ρ=0.808 > MLP ρ=0.435)

Comparator bootstrap chính thức của `A0` là **degree**. one-hop và two-hop là narrative/table baselines mạnh, nhưng không thay thế bootstrap-vs-degree.

### 4.3 Claim 3

**Paste-ready English claim (FROZEN — `gnn_significantly_better`):**

> Under attribute-community IC (HSCC), the best GNN (SAGE, raw node attributes including language) significantly outperforms the strongest flat baseline with matched feature access — LR(degree, views, life_time, language), ρ = 0.884 — by Δρ = +0.033 (95% CI [+0.021, +0.044]), suggesting that neighborhood message passing captures community-level propagation structure beyond what node-level source attributes alone can encode.

**Comparator lock:** `lr_degree_views_life_time_lang` (ρ=0.884). Nếu paper dùng tên ngắn hơn, phải consistent: "LR(dvtl+lang)" hoặc "LR(full attr)". Không thay đổi comparator giữa các draft.

**Required evidence (✅ all frozen):**

- `gnn_vs_baseline_bootstrap_ci_hscc.json` — SAGE, delta=+0.033, CI=[+0.021, +0.044] ✅
- `surrogate_ranking_metrics_hscc_clean.csv` — SAGE (gnn_raw_attr) ρ=0.915 ✅
- `baseline_ranking_metrics_hscc_clean.csv` — lr_degree_views_life_time_lang ρ=0.884 ✅
- fairness: GNN dùng `include_language=true` → flat baseline cũng dùng language ✅

`phi`, `lr_phi`, và related oracle rows không phải comparator chính của main paper.

Nếu strongest frozen flat baseline thắng hoặc hòa GNN, phải đổi claim này sang bản mềm hơn, ví dụ:

> Under attribute-community IC (HSCC), flat baselines already capture much of the source-driven signal, while graph-based models provide at most limited additional value from community-mediated structure.

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
- `"practical equivalence"` khi chưa có SESOI + CI phù hợp — xem giải thích dưới
- `"GNN is feature-agnostic"` theo nghĩa tuyệt đối

Thay `"feature-agnostic"` bằng:

> without precomputed structural summaries

**Khi nào "practical equivalence" hợp lệ và khi nào KHÔNG hợp lệ:**

"Practically equivalent" (hay "statistically equivalent") là một **inferential claim** — chỉ được dùng khi:

1. Có pre-registered equivalence bound δ₀ (ở đây δ₀=0.02),
2. Bootstrap CI được tính đúng cách,
3. **Toàn bộ CI nằm trong [−δ₀, +δ₀]** = [−0.02, +0.02].

**Với frozen A0 result:** CI = [−0.029, −0.008]:

- CI không nằm trong [−0.02, +0.02]: lower bound −0.029 có |−0.029| = 0.029 > 0.02
- CI hoàn toàn âm → GNN dưới degree → `gnn_significantly_worse`
- **KHÔNG được dùng "practically equivalent"** cho A0
- Correct framing: "remains statistically below degree centrality"

**Khi nào "practically equivalent" sẽ hợp lệ (hypothetical):** Nếu CI = [−0.015, +0.010] → toàn bộ nằm trong [−0.02, +0.02] → có thể dùng "practically equivalent under the pre-registered equivalence bound (δ₀=0.02)". Đây là kịch bản không xảy ra với frozen results của paper này.

**Lưu ý quan trọng:** Không nhầm lẫn giữa "practically equivalent" và "not significantly different". CI span zero (ví dụ [−0.010, +0.008]) chỉ có nghĩa là "không significant theo tiêu chí thông thường" — không có nghĩa là "practically equivalent" trừ khi toàn bộ CI nằm trong equivalence window.

### 4.6 Kịch bản viết paper theo kết quả cuối

**Rewrite rule bắt buộc trước khi draft title và abstract:**

Nếu best GNN dưới `HSCC` **không beat strongest frozen flat baseline** với comparator và bootstrap đúng regime, paper phải được viết như một **comparative operationalization paper** chứ không phải một **GNN-superiority paper**. Khi đó title, abstract, và introduction phải chuyển trọng tâm sang regime contrast, fairness-complete baselines, và điều kiện mà graph learning có hoặc không có giá trị.

> **[ACTIVE SCENARIO — FROZEN RESULTS]** Paper này rơi vào **Kịch bản 1b** (xem bên dưới), không phải Kịch bản 1, 2, hay 3. Kịch bản 1b là frozen framing đúng cho tất cả drafts.

#### Kịch bản 1b / ACTIVE for this paper — ✅ FROZEN

- `A0`: GNN **significantly worse** than degree (CI=[−0.029, −0.008], fully negative) — không phải "≈ degree"
- `HSCC`: GNN > strongest flat baseline (CI=[+0.021, +0.044]) — đúng như Kịch bản 1

Framing:

- A0 = structural learnability ceiling — GNN cannot overcome the label-degree coupling analytically baked into A0
- HSCC = graph-aware regime — neighborhood message passing adds signal beyond source attributes
- GNN advantage is **strictly regime-dependent** — A0 explicitly shows no advantage, HSCC explicitly shows advantage
- runtime story sạch, speedup ~5,500× unchanged

**Paste-ready framing sentence:**

> The value of graph message passing is strictly regime-dependent: under A0, the best GNN (GCN) remains statistically below degree centrality (Δρ = −0.018, CI [−0.029, −0.008]), while under HSCC, the best GNN (SAGE) significantly outperforms the strongest matched flat baseline (Δρ = +0.033, CI [+0.021, +0.044]).

---

#### Kịch bản 1 / Case 1 - [NOT ACTIVE — frozen result A0 is significantly_worse, not equivalent]

- `A0`: GNN ≈ degree ← **KHÔNG ĐÚNG với frozen results; A0 frozen = significantly_worse**
- `HSCC`: GNN > strongest flat baseline

Framing (hypothetical — không dùng cho paper này):

- A0 = structural ceiling
- HSCC = graph-aware regime
- GNN advantage is regime-dependent
- runtime story sạch

#### Kịch bản 2 / Case 2 - [NOT ACTIVE]

- `A0`: GNN ≈ degree ← **KHÔNG ĐÚNG với frozen results**
- `HSCC`: GNN ≈ strongest flat baseline ← **KHÔNG ĐÚNG với frozen results (HSCC = significantly_better)**

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
2. **Rozemberczki & Sarkar (2021)** - Twitch Gamers dataset
3. **Hamilton et al. (2017)** - GraphSAGE
4. **Kipf & Welling (2017)** - GCN
5. **Xu et al. (2019)** - GIN
6. **Klicpera et al. (2019)** - APPNP
7. **Kitsak et al. (2010)** - k-shell spreaders
8. **Guille et al. (2013)** - evaluation without behavioral logs
9. **Burt (1992)** - structural holes for HSCC justification
10. **Blondel et al. (2008)** - Louvain
11. **Grover & Leskovec (2016)** - Node2Vec **(must cite — Node2Vec đã được giữ ở main table như shallow-embedding baseline; nếu Node2Vec de-scoped khỏi paper, move reference này xuống §5.2)**

### 5.2 Có thể thêm nếu còn budget

- **Ling et al. (2023), DeepIM** — learning-based influence maximization context (cite only if you explicitly contrast IM vs node-level surrogate regression)
- **Benjamini & Hochberg (1995)** nếu dùng BH-FDR
- **Aral & Walker (2012)** nếu cần nhấn mạnh social ties and influence pathways
- **Lü et al. (2016)** cho survey về vital nodes
- **Chen, Wang & Wang (2010)** cho hop-decay / local influence approximation

### 5.3 Reference budget rule

MAPR version nên cố giữ khoảng **12 references cốt lõi** trong main paper. Nếu thiếu chỗ:

1. bỏ các reference chỉ để tăng background nhưng không đỡ claim chính,
2. chỉ giữ reference cho baseline nào thực sự xuất hiện ở main table,
3. để `Aral & Walker (2012)` hoặc các survey phụ ở trạng thái cut-first nếu narrative hiện tại đã đủ được chống đỡ bởi các nguồn chính.

### 5.4 Read but probably not cite cho MAPR version

- GCNII
- HGT
- GraphGPS
- fairness/per-group analysis papers
- journal-only methodological extensions

---

## Phần 6: Execution và freeze checklist

### 6.1 Giai đoạn 1 - Before writing

- **Người 1 / Person 1**
  - confirm shared upstream artifacts `data/processed/graph_csr.npz` + `data/processed/split_masks.parquet` are the frozen ones used by downstream evaluation
  - verify `ic_scores_a0.parquet` / `regression_targets_a0.parquet`
  - verify `ic_scores_hscc_refined.parquet` / `regression_targets_hscc_refined.parquet`
  - confirm `data/processed/node_attributes.parquet` is the same frozen handoff used by downstream joins/evaluation
  - verify HSCC formula lock trong `experiment_registry.md`
  - freeze config

- **Người 2 / Person 2**
  - verify `community_features.parquet` coverage
  - ensure `community_id` và `cross_community_edge_fraction` đều đầy đủ
  - verify `diffusion_proxies.parquet` full-graph coverage
  - xác nhận các downstream profiling inputs sạch

- **Người 3 / Person 3 — ✅ RERUN COMPLETED (hidden_channels=128, 2026-04-28)**
  - ✅ `surrogate_ranking_metrics_a0_clean.csv` — frozen
  - ✅ `surrogate_ranking_metrics_hscc_clean.csv` — frozen
  - ✅ `gnn_vs_degree_bootstrap_ci_a0.json` — frozen (GCN, delta=−0.018, CI=[−0.029, −0.008])
  - ✅ `gnn_vs_baseline_bootstrap_ci_hscc.json` — frozen (SAGE, delta=+0.033, CI=[+0.021, +0.044])
  - ✅ `gnn_vs_rankloss_bootstrap_ci_hscc.json` — frozen (SAGE rankloss, delta=+0.041, CI=[+0.030, +0.053])
  - ✅ `runtime_breakdown.csv` — frozen (MC-IC=480.3s; headline runtime uses `hscc,gnn_raw_attr` inference≈0.086s; speedup≈5,590×, rounded to ~5,500× in paper prose)
  - **Best arch A0:** GCN raw_attr (ρ=0.808, std=0.001) — APPNP excluded (std=0.417≥0.1), SAGE poor (ρ=0.534), GIN moderate (ρ=0.615)
  - **Best arch HSCC:** SAGE / gnn_raw_attr (ρ=0.915, std=0.004) — APPNP excluded (std=0.146≥0.1), GIN collapses (ρ=0.028), GCN moderate (ρ=0.602)
  - **Architecture-regime insight (ghi vào paper):** GCN (symmetric normalization) suits A0; SAGE (mean aggregation) suits HSCC. GCN: 0.808@A0 → 0.602@HSCC; SAGE: 0.534@A0 → 0.915@HSCC
  - **GIN collapse under HSCC (ρ=0.028) là finding — cần report trong paper, không hide**
  - **Node2Vec precomp:** ~153s per regime (từ `runtime_breakdown.csv`) — ghi đúng vào runtime table, không merge với GNN inference time

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
- mọi con số trong bảng/abstract phải trace được về frozen CSV/JSON đúng regime
- không dùng `"outperforms"` nếu chưa có bootstrap support tương ứng
- nếu quá 6 trang, cắt appendix-style diagnostics trước khi cắt 4 evidence blocks chính
- **[✅ C3 FROZEN — đã confirmed]** `gnn_vs_rankloss_bootstrap_ci_hscc.json` frozen: `loss_mode=rankloss_combined`, `rankloss_alpha=0.5`, SAGE ρ=0.924, delta=+0.041 vs lr_dvtl_lang, CI=[+0.030, +0.053]. Có thể include trong paper. Khi submit, verify lại một lần cuối rằng Table 3 C3 row dùng đúng ρ=0.924 và CI từ file này.

### 6.4 Freeze rules bắt buộc — ✅ TẤT CẢ ĐÃ FROZEN (Person 3 rerun 2026-04-28)

- ✅ `A0`: `gnn_vs_degree_bootstrap_ci_a0.json` — GCN, delta=−0.018, CI=[−0.029, −0.008], `gnn_significantly_worse`
- ✅ `HSCC`: `gnn_vs_baseline_bootstrap_ci_hscc.json` — SAGE vs `lr_degree_views_life_time_lang` (ρ=0.884), delta=+0.033, CI=[+0.021, +0.044], `gnn_significantly_better`
- ✅ Feature access matched: GNN dùng `include_language=true` → flat baseline cũng dùng language ✅
- ✅ Outputs regime-tagged: `baseline_ranking_metrics_{a0|hscc}_clean.csv`, `surrogate_ranking_metrics_{a0|hscc}_clean.csv`
- ✅ `community_features.parquet` — HSCC formula inputs verified
- ✅ **[🟡 C3 BOOST — FROZEN]** `gnn_vs_rankloss_bootstrap_ci_hscc.json` — SAGE rankloss, delta=+0.041, CI=[+0.030, +0.053]; `feature_policy.loss_mode="rankloss_combined"`, `rankloss_alpha=0.5` confirmed ✅

**Tất cả freeze dependencies đã resolved — sẵn sàng finalize title, abstract, contributions, contrast paragraph.**

---

## Phần 7: Reviewer defense appendix

### Reviewer hỏi: "Why not use real cascade data?"

Twitch dataset không có behavioral cascade logs. MC-IC chỉ là simulation-based proxy có cơ sở phương pháp luận; mọi findings phải được hiểu là properties của simulation, không phải real influence measurements.

### Reviewer hỏi: "Why is HSCC a good diffusion model?"

Paper không claim `HSCC` là diffusion model thật của Twitch. `HSCC` chỉ là một domain-informed operationalization để kiểm tra khi nào neighborhood composition tạo learnable value cho GNN. Đóng góp của paper là comparative finding, không phải realism claim.

### Reviewer hỏi: "Why not compare against DeepIM or other IM methods?"

DeepIM giải bài toán khác: chọn seed set tối ưu để maximize total cascade reach. Task ở đây là **node-level IC score regression**, nên comparison trực tiếp không cùng problem setting.

### Reviewer hỏi: "The Twitch dataset is from 2021. Is it still relevant?"

Đóng góp của paper là methodological, không phải phát hiện đặc thù cho Twitch năm 2021. Twitch Gamers vẫn là benchmark hợp lý cho graph-level analysis và surrogate-learning evaluation.

### Reviewer hỏi: "Life_time dominates HSCC labels. Isn't GNN just learning life_time?"

Đây là lý do fairness baselines là điều kiện bắt buộc. Nếu `language` đi vào GNN, nó cũng phải đi vào LR/MLP fairness baselines. Chỉ residual margin sau matched baselines mới được phép diễn giải là graph/community message-passing gain.

### Reviewer hỏi: "A0 and HSCC look like the same IC model with different parameters, not qualitatively different regimes."

Câu trả lời có hai phần:

1. **Structurally:** A0 decouples transmission probability from any source-side content — all variance comes from target degree. HSCC explicitly couples transmission to source engagement velocity and cross-community exposure. These are not just different parameter values; they are different information sources encoded in the label generation.

2. **Empirically:** The results demonstrate qualitative differentiation — degree ρ = 0.826 under A0 vs ρ = −0.006 under HSCC (shift = 0.832). This is not a minor parameter difference; it is a structural reversal of the relationship between degree and the label. If reviewers insist the regimes are not qualitatively different, point to this collapse as the empirical evidence of regime qualitative change.

**Paste-ready rebuttal sentence:**

> The two regimes differ not only parametrically but structurally: A0 couples transmission probability entirely to target degree, while HSCC replaces this dependence with source-side engagement velocity and cross-community amplification. The empirical consequence — degree centrality shifting from ρ = 0.826 under A0 to ρ = −0.006 under HSCC — demonstrates that this is a qualitative, not merely quantitative, difference in the information structure of the labels.

### Reviewer hỏi: "Why should we trust your GNN results if you only test on one dataset?"

Câu trả lời: đóng góp chính của paper là methodological (operationalization matters), không phải chứng minh GNN luôn tốt. Twitch Gamers là một chuẩn benchmark hợp lý cho surrogate learning. Một dataset duy nhất đủ để demonstrate the pattern nếu paper frame nó đúng cách — không claim "always works on all datasets"; claim "this is what happens under these two operationalizations on this graph."

**Paste-ready rebuttal sentence:**

> Our claim is methodological rather than universal: we demonstrate that operationalization choice governs the value of graph learning on this benchmark. Replicating this experiment on datasets with observed cascade logs (e.g., Higgs Twitter) would be a natural extension to validate generalizability — a point we acknowledge in the limitations section.

### Reviewer hỏi: "Δρ = +0.033 is marginal. Is this practically significant?"

Đây là **reviewer attack nguy hiểm nhất** cho HSCC claim. Hai cách counter:

**Cách 1 — CI framing (không cần thêm evidence):**

> The improvement of Δρ = +0.033 is statistically significant at the 95% level (CI [+0.021, +0.044]), with the confidence interval fully above zero and both bounds exceeding the pre-registered equivalence threshold (δ₀ = 0.02). Under our pre-registration, this constitutes evidence of genuine improvement, not marginal noise.

**Cách 2 — Rank position framing (nếu có space):**

> In a test set of 1,000 nodes, Δρ = +0.033 corresponds to systematic rank differences that compound across the top-k subset critical for influence maximization — a 3–4 percentile rank gain for the most influential nodes is practically meaningful in seed selection contexts.

**Cách 3 — Baseline comparison framing:**

> The Δρ = +0.033 gain is over the _strongest matched flat baseline_ (LR with degree, views, life_time, and language — ρ = 0.884), not over a trivial baseline. Achieving any gain over an already-strong attribute model confirms that neighborhood message passing adds non-redundant information.

**Lưu ý:** Đừng dùng cách 2 nếu không có exact computation. Cách 1 là safest và fully supported by frozen artifact.

### Reviewer hỏi: "Why is your equivalence bound δ₀ = 0.02? This seems arbitrary."

> The equivalence bound δ₀ = 0.02 was pre-specified prior to data analysis in the experiment design document (implementation plan) and is consistent with a conservative small-effect threshold for Spearman correlation in ranking benchmarks. Under this pre-registration, A0 CI = [−0.029, −0.008] falls outside the equivalence window (the lower bound |−0.029| exceeds δ₀), correctly classified as significantly worse. HSCC CI = [+0.021, +0.044] has both bounds above δ₀, correctly classified as significantly better. The bound is symmetric and was fixed before unblinding results.

**Paste-ready rebuttal:**

> The equivalence bound δ₀ = 0.02 was pre-registered prior to data analysis, following Lakens (2017) two one-sided tests (TOST) methodology for statistical equivalence testing. The threshold was chosen as a conservative small-effect bound for Spearman rank correlations in ranking benchmarks — consistent with Cohen's small-effect convention in correlation analysis. Changing the threshold post-hoc would constitute p-hacking; we report results under the pre-specified bound and provide full bootstrap CIs for readers who prefer alternative thresholds.

**Methodological citation (for paper if space permits; for reviewer response always):**

> Lakens, D. (2017). Equivalence tests: A practical primer for t tests, correlations, and meta-analyses. _Social Psychological and Personality Science_, 8(4), 355–362.

**Usage note:** If the main paper text uses "equivalence bound" or "TOST" language, add Lakens (2017) to the paper's reference list. If equivalence testing is implicit only (CI-based interpretation without naming TOST), this citation is reviewer-response-only and need not appear in the main reference list.

### Reviewer hỏi: "APPNP has strong theoretical properties. Why did it fail in both regimes?"

> APPNP was excluded from the best-arch comparison under both regimes due to high seed variance (std = 0.417 under A0; std = 0.146 under HSCC), exceeding the pre-registered threshold of 0.1. The instability is consistent with PPR propagation sensitivity: with K=10 propagation steps and teleport α=0.15, APPNP's iterative approximation may amplify variance in IC scores that are themselves heavy-tailed (CV > 1.1 across all degree bands). We report APPNP results in the Appendix for completeness. Its exclusion from the best-arch selection follows the pre-registered protocol and does not represent a post-hoc decision.

### Reviewer hỏi: "Why 5,000 labeled nodes out of 168K?"

> The 5,000-node labeled set was chosen to balance simulation cost (200 runs × 5,000 nodes × two regimes = substantial compute) against statistical power. To avoid concentrating labels only among high- or low-degree nodes, labeled nodes are sampled with degree-stratified coverage (degree quintiles, q=5; seed=42), consistent with the shared split-mask protocol. The 1,000-node held-out test set provides sufficient power for the bootstrap CI procedure (1,000 resamples). We acknowledge that scaling to the full 168K graph would require approximate MC-IC or importance sampling — a natural extension noted in the limitations section.

### Reviewer hỏi: "Louvain community detection is resolution-sensitive. Did you test different resolution parameters?"

> Community labels for HSCC are derived from a single Louvain run with default resolution (as specified in the frozen implementation plan). We acknowledge that different resolutions would yield different community structures and potentially different HSCC results. However, the contribution of HSCC is to introduce community-membership signal into label generation — not to claim that this specific partition is optimal. The fixed configuration is transparent and reproducible; sensitivity to Louvain resolution is acknowledged as a limitation in §5.2.

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
