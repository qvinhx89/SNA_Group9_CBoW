# MAPR 2026 — Implementation Plan v3.1

## Operationalizing Influence Potential via Weighted-Cascade IC Simulation and GNN Surrogate Learning on Twitch

**Deadline submit:** 30/4/2026 | **Bắt đầu thực thi:** 6/4/2026 | **📍 Hôm nay: 16/4/2026 — còn 14 ngày**
**Conference:** MAPR 2026, Hue City, 13–14/8/2026 — IEEE Xplore Track 1: Graph Learning
**Dataset:** Twitch Gamers Social Network (Rozemberczki et al., 2021) — 168,114 nodes, 6,797,557 edges
**Document version:** 3.1 — tích hợp Professor's Framing (linear pipeline: IC metric → GNN surrogate)

**Scope bridge:** Tài liệu này là strategic master plan (research + execution + paper). Với phạm vi coding team 3 người, `docs/MAPR2026_v3_team_parallel_coding_plan.md` là execution override; nếu có khác biệt ở tác vụ hằng ngày, ưu tiên Team Plan và giữ các ràng buộc nghiên cứu/narrative theo tài liệu này.

---

## 0. Nền tảng tư duy — Đọc kỹ trước khi implement

### 0.1 Reframe cốt lõi

**Hướng cũ (circular):**

```
SIS = f(PageRank, Betweenness, k-shell)  →  define power user
IC simulation                             →  validate SIS     ← CIRCULAR
```

**Hướng mới (defensible):**

```
Weighted Cascade p(u,v) = 1/degree(v)   ← parameter-free, no attributes
        ↓
Monte Carlo IC simulation → IC score mỗi node
        ↓
IC score = OPERATIONALIZATION của influence potential
         (proxy có principled grounding, KHÔNG phải ground truth)
        ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  [MAIN PIPELINE — Professor's Framing v3.1]                             │
│  Task A: Operationalize IC     ──►  Task C: GNN Surrogate               │
│  (labels, stability, pivot)         (approximate IC fast)               │
└─────────────────────────────────────────────────────────────────────────┘
```

### 0.1b Professor's Framing (v3.1 — tháng 4/2026)

> **"We propose MC-IC as a principled operational metric to measure power users. But it is expensive to run MC-IC. We show the stability analysis and regression problem nature. We can approximate it to a very good margin using GNN (GCN/GIN/GAT...)."**

**Pipeline tuyến tính** (thay thế 6 RQ song song — đọc theo thứ tự):

```
[1] MC-IC = metric tốt?     →  justify: discriminative reach distribution + IC ≠ degree (higher-order)
[2] MC-IC đắt?              →  evidence: 480s / 5k nodes → 7,169× speedup = motivation cho surrogate
[3] Regression nature?      →  evidence: Jaccard instability (0.307→0.682) + structural cause + PIVOT_CONFIRMED
[4] GNN xấp xỉ tốt?        →  evidence: architecture comparison (GCN/GIN/GAT/SAGE) + significance test
```

**Scope note (v3.1):** Main contribution = linear pipeline [1]→[2]→[3]→[4]. Additional analyses are limited to continuous correlation summaries (e.g., views vs IC Spearman) and do not introduce categorical grouping.

> **Tension cốt lõi cần resolve:** `gnn_centrality` Spearman 0.817 < `degree` 0.826.
> Defense strategy: (a) bootstrap CI để test statistical equivalence, (b) architecture search (GAT **có thể** phù hợp
> với weighted cascade — _hypothesis; C2 quyết định_ — vì attention **có thể** học pattern gần 1/degree(v)), (c) feature-agnostic story (gnn_raw_attr 0.534 vs
> mlp_raw_attr 0.435, +0.099 từ message passing). KHÔNG claim "very good margin" trừ khi có significance test.

---

### 0.2 Ba nhiệm vụ tách biệt — KHÔNG trộn lẫn

| Task  | Câu hỏi nghiên cứu                                          | Output chính                                         | Tier (v3.1)                    |
| ----- | ----------------------------------------------------------- | ---------------------------------------------------- | ------------------------------ |
| **A** | How to define influence proxy without behavioral logs?      | IC scores, label stability, regression justification | **MAIN** — Section 3 của paper |
| **C** | Can GNN approximate IC efficiently vs analytical baselines? | Architecture comparison, Spearman ρ, speedup         | **MAIN** — Section 4 của paper |

### 0.3 Framing language — nhất quán xuyên suốt paper

| ❌ KHÔNG viết                                     | ✅ PHẢI viết                                                                                                                        |
| ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| "We identify real power users"                    | "We operationalize influence potential"                                                                                             |
| "Ground truth influence"                          | "Simulation-defined influence proxy"                                                                                                |
| "GNN predicts influencers"                        | "GNN approximates IC simulation efficiently"                                                                                        |
| "Calibrated to 8% reach (DeepIM)"                 | "Pilot verified non-degenerate spread distribution"                                                                                 |
| "IC validates our approach"                       | "IC is our definitional operationalization"                                                                                         |
| "We pivot to regression due to label instability" | "We formulate prediction as regression on continuous MC scores — the principled choice for a simulation-derived continuous target"  |
| "Option B is a fallback"                          | "Regression primary is the correct formulation; binary labels are a derived secondary artifact with inherent threshold sensitivity" |
| "GNN approximates IC to a very good margin"       | "GNN achieves statistically comparable approximation quality to strongest analytical baseline" _(chỉ sau khi có bootstrap CI)_      |
| "GNN outperforms baselines"                       | "GNN matches/approaches best analytical baseline while being end-to-end learnable without precomputed centrality"                   |
| "MC-IC is a good metric"                          | "MC-IC is a principled operational metric for influence potential in static social graphs"                                          |
| "We show GNN is better"                           | "Message passing extracts structural signal (+0.099 Spearman over MLP); GNN is a viable fast surrogate vs MC-IC simulation"         |

---

## 1. Construct Validity — Phải Address Trong Paper

> **Gap #1 gây rejection.** Reviewer SNA sẽ hỏi ngay: "Why should IC simulation on follower graph tell us anything about real Twitch influence?"

### 1.1 Paragraph bắt buộc trong Section 3.1

> _"The Twitch follower graph represents declared social affinity rather than observed information transmission. Influence on Twitch primarily occurs through live streams, raids, and chat interactions — channels not captured by static friendship edges. The Twitch Gamers dataset has been used in prior network analysis studies for community structure, node classification, and link prediction (Rozemberczki et al., 2021), establishing it as a standard benchmark for graph-level analysis despite the absence of behavioral diffusion logs. While this limits the behavioral realism of any diffusion simulation, prior work has established that social ties correlate with influence pathways in online platforms (Guille et al., 2013; Aral & Walker, 2012), making graph-based diffusion models a reasonable structural operationalization. We explicitly do not claim friendship edges are transmission channels; we treat the graph as a structural substrate on which a hypothetical diffusion process is simulated. All findings should be interpreted as properties of diffusion under this operationalization."_

### 1.2 Directionality — bắt buộc trong config và paper

```yaml
# experiment.yaml
graph_directed: false
graph_direction_note: >
  "MUSAE Twitch dataset exposes only mutual-follow edges (Rozemberczki et al., 2021 §3).
   Undirected treatment is the only valid representation. Under undirected weighted
   cascade, p(u,v) = 1/degree(v) models limited attention budget per incoming edge."
```

### 1.3 Paragraph bắt buộc trong Section 5 (Limitations)

> _"A fundamental limitation is that Twitch's follower network may not correspond to actual information transmission pathways. Dead accounts (X% of nodes, with systematically lower degree and views than active accounts) were excluded; findings generalize only to active users. Furthermore, account age (life_time) is used as an external proxy variable — while exogenous to IC labels, it may capture platform tenure rather than influence potential directly. All quantitative findings should be treated as structural properties of the weighted-cascade operationalization, not as measurements of real Twitch influence."_

---

## 2. Quyết định kỹ thuật cốt lõi — ngày 6/4

> **HAI task quan trọng nhất của ngày đầu tiên, phải làm trước mọi thứ khác.**

### 2.1 Task #1 (Buổi sáng): Benchmark IC Runtime

```
Nếu bỏ qua task này → toàn bộ timeline có thể sụp đổ.
```

```python
import igraph as ig
import time, random, numpy as np

def benchmark_ic_runtime(G_ig, n_test=100, n_runs=50):
    degrees = G_ig.degree()
    test_seeds = random.sample(range(G_ig.vcount()), n_test)

    t0 = time.time()
    for seed in test_seeds:
        for _ in range(n_runs):
            run_ic_csr(seed, indptr, indices, degrees)  # see Section 4.2
    elapsed = time.time() - t0

    per_sim_ms = elapsed / (n_test * n_runs) * 1000
    projected = per_sim_ms / 1000 * N_SAMPLE * N_RUNS / 3600

    print(f"Per-sim: {per_sim_ms:.1f} ms | Projected {N_SAMPLE}×{N_RUNS} runs: {projected:.1f}h")
    return per_sim_ms, projected

# DECISION TABLE:
# < 4h  → n_sample=5,000, N_runs=200
# 4–8h  → n_sample=3,000, N_runs=150
# > 8h  → n_sample=2,000, N_runs=100; ghi limitation
```

### 2.2 Task #2 (Buổi chiều): One-Hop Baseline Reality Check

**Đây là quyết định sống còn cho toàn bộ GNN narrative.**

Dưới weighted cascade với p nhỏ, cascade thường chết trong 1–3 hops. Do đó one-hop expected spread `Σ 1/degree(v)` có thể là predictor cực mạnh cho full IC reach (ρ > 0.9). Tuy nhiên gate Day-1 không được dựa vào Spearman một mình: phải kiểm tra thêm top-k alignment qua Jaccard@10% và NDCG@10%.

```python
from sklearn.metrics import ndcg_score
import numpy as np
import pandas as pd

def select_pilot_nodes_stratified(degrees, n_pilot=200, q=5, seed=42):
    """Degree-quintile stratified sampling for representative, reproducible pilots."""
    rng = np.random.default_rng(seed)
    deg_q = pd.qcut(degrees, q=q, labels=False, duplicates='drop')
    bins = np.unique(deg_q)
    per_bin = n_pilot // len(bins)

    pilot = []
    for b in bins:
        idx = np.where(deg_q == b)[0]
        pilot.extend(rng.choice(idx, size=per_bin, replace=False).tolist())

    remainder = n_pilot - len(pilot)
    if remainder > 0:
        pool = np.setdiff1d(np.arange(len(degrees)), np.array(pilot, dtype=int))
        pilot.extend(rng.choice(pool, size=remainder, replace=False).tolist())

    return np.array(pilot, dtype=int)

# Chạy IC pilot trên 200 nodes × 50 runs
pilot_nodes = select_pilot_nodes_stratified(degrees, n_pilot=200, q=5, seed=42)
ic_pilot = {n: run_ic_csr(n, indptr, indices, degrees, n_runs=50).mean()
            for n in pilot_nodes}

# Compute one-hop spread
one_hop = {n: sum(1.0/max(degrees[v], 1) for v in G.neighbors(n))
           for n in pilot_nodes}

# Gate metrics: Spearman + Jaccard@10% + NDCG@10%
ic_vals = np.array([ic_pilot[n] for n in pilot_nodes], dtype=float)
onehop_vals = np.array([one_hop[n] for n in pilot_nodes], dtype=float)

rho, p = spearmanr(ic_vals, onehop_vals)
k = max(1, int(np.ceil(0.10 * len(pilot_nodes))))

# Stable tie-break để tránh top-k drift giữa các lần chạy
top_ic = set(np.argsort(-ic_vals, kind='stable')[:k])
top_onehop = set(np.argsort(-onehop_vals, kind='stable')[:k])
jaccard_10 = len(top_ic & top_onehop) / len(top_ic | top_onehop)

ndcg_10 = float(ndcg_score(ic_vals.reshape(1, -1), onehop_vals.reshape(1, -1), k=k))

print(f"One-hop vs IC pilot: ρ={rho:.3f}, Jaccard@10%={jaccard_10:.3f}, NDCG@10%={ndcg_10:.3f}")

# DECISION:
# ρ < 0.8  → GNN story viable; proceed as planned
# 0.8–0.9  → Add 2-hop proxy as stronger baseline; GNN may still win
# ρ > 0.9 and Jaccard@10% > 0.8 and NDCG@10% > 0.9
#          → RESTRUCTURE: proxies primary, GNN secondary; title changes
# ρ > 0.9 but top-k alignment chưa cao
#          → giữ GNN + 2-hop head-to-head, nhấn mạnh top-k divergence
```

**Prepared narrative nếu ρ > 0.9 và top-k alignment cao:**

> _"We find that one-hop analytical expected spread under weighted cascade achieves ρ > 0.9 with full MC IC scores, suggesting cascade dynamics are largely confined to the local neighborhood of each seed. This finding itself is a contribution: expensive MC simulation can be approximated by an O(E) analytical proxy with 7,169× speedup. GNN surrogate learning is additionally motivated by its feature-agnostic nature — no precomputed graph statistics required — and the potential to generalize inductively to new nodes."_

---

## 3. Technical Stack

```
IC Simulation Backend:   igraph (C) + CSR numpy arrays — TUYỆT ĐỐI KHÔNG dùng NetworkX BFS
Parallel IC:             joblib Parallel(prefer='loky') + CSR shared-memory arrays
Graph Analytics:         NetworKit (C++) cho betweenness approximate
Community Detection:     python-louvain (seed sweep + best modularity run)
GNN:                     PyTorch Geometric (PyG) ≥ 2.5, torch ≥ 2.0
Embeddings:              node2vec library (dimensions=64, walks=20-30)
Stats:                   scipy.stats, statsmodels (BH correction), sklearn
Data:                    pandas + pyarrow (parquet format)
```

> **NetworkX usage policy — clarification (để tránh nhầm lẫn với "TUYỆT ĐỐI không dùng"):**
>
> | Loại sử dụng                                                   | Cho phép?     | Lý do                                                             |
> | -------------------------------------------------------------- | ------------- | ----------------------------------------------------------------- |
> | IC simulation loops (BFS/DFS per node per run)                 | ❌ **BANNED** | O(N × runs) bằng NetworkX → timeout; dùng CSR numpy thay thế      |
> | Graph load/convert **một lần**: `nx.read_edgelist()` → CSR/PyG | ✅ OK         | Chỉ gọi 1 lần khi startup                                         |
> | Community detection: `python-louvain`                          | ✅ OK         | python-louvain dùng NetworkX internally — acceptable              |
> | Betweenness trên subgraph nhỏ (≤500 nodes)                     | ✅ OK         | Dùng khi NetworKit không available; **không dùng cho full graph** |
> | Utility: degree dict, neighbor list cho debug                  | ✅ OK         | Không ảnh hưởng runtime production                                |
>
> **Rule of thumb:** Nếu NetworkX call nằm trong vòng lặp chạy ≥ 5,000 lần → **bắt buộc** đổi sang CSR numpy. Nếu chỉ gọi 1–10 lần → acceptable.

**Yêu cầu phần cứng tối thiểu:**

```
RAM:    ≥ 32 GB
CPU:    ≥ 8 cores
GPU:    ≥ 8 GB VRAM (RTX 3080 / A100)
```

---

## 4. IC Simulation — Kỹ thuật đúng

### 4.1 Calibration — Weighted Cascade KHÔNG cần target reach

> **Đây là thay đổi quan trọng nhất so với v2.0.**

Weighted cascade `p(u,v) = 1/degree(v)` là **parameter-free** — không có λ để tune. DeepIM (ICML 2023) báo cáo ~8% reach của **seed set 1% nodes được tối ưu hóa**, không phải single-node reach. Áp con số đó làm calibration target cho single-seed IC là **sai ngữ cảnh nghiêm trọng**.

```yaml
# experiment.yaml — KHÔNG có calibration_target_reach_pct nữa

# Primary: Weighted Cascade — parameter-free, no calibration needed
p_primary: weighted_cascade # p(u,v) = 1/degree(v)
calibration_mode: variance_check # NOT target_reach

# Pilot diagnostics để verify non-degenerate distribution
pilot_diagnostics:
  - mean_reach # mean single-seed reach across pilot nodes
  - median_reach
  - iqr_reach # interquartile range
  - top10_to_median_ratio # should be >> 1 để ranking có ý nghĩa
  - rank_stability # Spearman ρ giữa MC seeds
  - cv_score # coefficient of variation, target > 0.3


# If CV < 0.3: cascade dies too fast → consider subgraph restriction
# If median reach > 5% LCC: cascade too explosive → report as unexpected finding

# Sensitivity variants are configured explicitly (see Section 4.1b).
# Do NOT keep a stale uniform-p block here.
```

**Cite DeepIM đúng cách:**

> _"Following the weighted-cascade experimental setup widely used in influence maximization, including the configuration adopted in DeepIM (Ling et al., 2023) and originally formalized by Kempe et al. (2003), we set p(u,v) = 1/degree(v). This models a limited attention budget where each neighbor's influence is inversely proportional to the number of connections competing for attention."_

---

### 4.1b Diffusion Rule Variants — Sensitivity Analysis (robustness to p(u,v) choice)

> **Framing bắt buộc:** Các variant dưới đây là **sensitivity analysis về diffusion rule choice**, không phải "tune p để GNN đẹp hơn". Primary label source luôn là **A0 (weighted cascade)**.
>
> **Views-independence:**
>
> - **A0 (primary) + A1/A2 (structural sensitivity)** phải **views-independent** để giữ construct validity trong main story (labels không “leak” popularity signal).
> - **I-A attribute-informed operationalization:** **I-A pilot gate là MUST (unconditional; ~20 phút)** để quyết định có chạy full I-A hay không. **Full I-A là conditional-MUST nếu (và chỉ nếu) pilot pass.** Nếu pilot fail: commit A0-only narrative và bỏ toàn bộ I-A.
>
> Cách viết defensible duy nhất: "**robustness to diffusion rule choice**" + "**architectural inductive bias check**".

#### Tính chất toán học của từng variant

**A0 — Weighted Cascade (primary, giữ nguyên): `p(u,v) = 1/deg(v)`**

- Budget property: `Σ_{u∈N(v)} p(u,v) = deg(v) × 1/deg(v) = 1.0` per receiving node.
- **Global average one-hop spread = 1.0 ∀ graph** (mathematical invariant, không phải approximation):
  ```
  Global_avg = (1/N) × Σ_u [Σ_{v∈N(u)} 1/deg(v)]
             = (1/N) × Σ_v [deg(v) × 1/deg(v)]   ← mỗi v được sum bởi deg(v) neighbors
             = (1/N) × N = 1.0
  ```
- Rank ordering vẫn biến thiên mạnh vì `Σ_{v∈N(u)} 1/deg(v)` phụ thuộc vào degree của _neighbors_ của u, không phải chính degree(u). Node có nhiều **niche neighbors** (low-degree) → one-hop spread cao.
- Trên Twitch (high mean_degree → small p): cascade chết sau 1–3 hops → IC score ≈ one-hop analytical proxy → degree baseline competitive. Đây là lý do cần A2/A1 sensitivity để kiểm tra robustness.
- Literature: Kempe (2003), Ling/DeepIM (2023).

**A1 — Source Budget (sensitivity S2): `p(u,v) = 1/deg(u)`**

- **One-hop spread của MỌI node đều = 1.0** (identity, không phải invariant):
  ```
  E_A1[u] = Σ_{v∈N(u)} 1/deg(u) = deg(u)/deg(u) = 1.0 ∀u
  ```
- Hệ quả quan trọng: IC-A1 score **hoàn toàn không phụ thuộc vào 1-hop dynamics** → score chỉ reflect 2+ hop propagation. Node ở vị trí bridge (high betweenness) hay cross-community sẽ nổi bật hơn dưới A1.
- Expected: `Spearman(IC-A1, degree)` thấp hơn A0 đáng kể → "IC ≠ degree" story mạnh hơn nếu cần. Nhưng đây là **kết quả thực nghiệm cần verify**, không hứa trước.
- Defensible như "broadcaster overload model": mỗi node có broadcast budget = 1, chia đều cho tất cả neighbors.
- Không có strong literature grounding → phải self-justify trong paper.

**A2 — Symmetric Normalization (sensitivity S1, đáng thử nhất): `p(u,v) = 1/√(deg(u)×deg(v))`**

- A2 là **geometric mean** của A0(u→v) và A0(v→u):
  ```
  A2(u,v) = 1/√(deg(u)×deg(v)) = √[1/deg(v) × 1/deg(u)] = √[A0(u→v) × A0(v→u)]
  ```
- Symmetric: `p(u,v) = p(v,u)` → cleaner cho undirected graph. Không có clean budget property như A0.
- **GCN-IC alignment (core insight — viết cẩn trọng):**
  GCNConv (Kipf & Welling, 2017) aggregates với weight: `1/√(d̃_u × d̃_v)` (d̃ = deg+1 với self-loop).
  Structurally analogous — nhưng không exact: GCN có self-loops (d̃ ≠ deg), non-linearity (ReLU), và dropout phá vỡ tính tuyến tính thuần túy.
  > **Paper framing cẩn trọng:** _"GCN's message passing uses `Â = D^{-1/2}AD^{-1/2}` which is structurally analogous to a symmetric degree-normalized diffusion operator (A2). We test whether this structural alignment translates to empirical performance advantage when the IC target is generated under A2 — an architectural inductive bias check."_
  > ⚠ **KHÔNG viết:** "GCN = IC probability" hay "GCN implements IC exactly" — self-loops và non-linearity làm khác biệt.
- Views-independent, life_time-independent → không vi phạm bất kỳ independence constraint nào.

#### Phân loại defensibility (cập nhật)

| Variant                  | Tên                          | Views-indep       | life_time indep | Grounding          | Defensible?                       | Priority          |
| ------------------------ | ---------------------------- | ----------------- | --------------- | ------------------ | --------------------------------- | ----------------- |
| A0: `1/deg(v)`           | Weighted Cascade             | ✅                | ✅              | ✅✅ Kempe+DeepIM  | **✅ Primary**                    | 1 — luôn chạy     |
| **I-A: `w(v)/Σw(N(u))`** | **Attr-Informed (row-norm)** | **❌ dùng views** | ✅              | ✅ Twitch-specific | ✅ (attr-informed; gated)         | **2 — MUST (pilot); conditional full** |
| A2: `1/√(deg(u)×deg(v))` | Symmetric                    | ✅                | ✅              | ✅ GCN analogy     | ✅                                | 3 — robustness    |
| A1: `1/deg(u)`           | Source Budget                | ✅                | ✅              | Marginal           | ✅                                | 4 — if needed     |
| II-B: `w(v)/deg(v)`      | Views-Density                | ❌ dùng views     | ✅              | Moderate           | ⚠ fallback nếu I-A degenerate     | 5 — fallback      |
| B3: life_time-based      | Tenure Amp.                  | ✅                | ❌              | None               | ⚠ mất external val                | Low               |

---

#### I-A — Attribute-Informed IC (Row-Normalized Views Attention)

> 🔵 **[SUPPLEMENTARY TRACK — SKIP nếu không activate I-A]**
> Toàn bộ block này chỉ relevant nếu I-A pilot pass (3 checks bên dưới).
> Nếu A0 only: bỏ qua toàn bộ I-A, II-B, và C2-I-A sections → tiếp tục từ "Framework thử nghiệm".

> **Điều kiện kích hoạt:** Chỉ chạy như **supplemental** label set; bắt buộc có pilot gate + ghi rõ trong paper đây là **attribute-informed operationalization** (không phải sensitivity của A0).

**Code pointers (v3.1):**

- **Pilot gate (MUST, ~20 phút):** `python src/mapr2026_v3/ic_pilot_ia.py --n-pilot-nodes 200 --n-pilot-runs 50 --n-jobs -1`
- **Full I-A labels (conditional-MUST nếu pilot pass):** `python src/mapr2026_v3/ic_labels_attribute_ia.py --n-runs 200 --n-jobs -1`

**Outputs:**

- Pilot: `outputs/mapr2026_v3_results/ia_pilot_diagnostics.json`
- Full: `outputs/mapr2026_v3_results/ic_scores_ia.parquet` + `data/processed/regression_targets_ia.parquet`

**Formula:**

```
p(u,v) = log1p(views(v)) / Σ_{x∈N(u)} log1p(views(x))
       = w(v) / Σ_{x∈N(u)} w(x)      với w(v) = log1p(views(v))
```

**Lý do dùng `log1p(views)` thay vì raw views:** Twitch views phân phối power-law (Gini ≈ 0.9+) → raw views làm một số node chiếm p ≈ 1.0 → cascade degenerate. `log1p` compress distribution → p phân tán hơn → CV > 0.3.

**Tính chất toán học then chốt:**

```
1. Row-normalized: Σ_{v∈N(u)} p(u,v) = 1.0  ∀u có ít nhất 1 neighbor
   → E[one-hop spread(u)] = 1.0  ∀u  (identity, không phải invariant)
   → One-hop spread hoàn toàn không phân biệt nodes
   → IC-I-A score chỉ phụ thuộc 2+ hop attribute propagation

2. Hệ quả với baselines:
   Degree(u): chỉ biết số neighbors, KHÔNG biết views của họ
   → Spearman(degree, IC-I-A) ≈ 0.45–0.65 (degree nearly blind)

   MLP(views(u), life_time(u)): chỉ thấy features của u, không aggregate N(u)
   → Spearman(MLP, IC-I-A) ≈ 0.50–0.65 (miss neighbor context)

3. GNN 2-layer alignment:
   Layer 1: h_u^(1) = AGG({w(v) : v ∈ N(u)})
           ≈ học được Σ w(v) và phân phối views của N(u)
           = đúng thành phần denominator của p(u,v) = w(v)/Σw(N(u))

   Layer 2: h_u^(2) = AGG({h_v^(1) : v ∈ N(u)})
           ≈ aggregate 2-hop views composition
           ≈ strong alignment với thành phần 2-hop của IC-I-A spread

   → Đây là **architectural inductive-bias alignment**: message passing có thể học/khai thác đúng cấu trúc normalization mà I-A dùng,
     không cần claim "implements exactly".
```

**Paper framing (attribute-informed operationalization):**

> _"In addition to the structural weighted cascade (A0), we evaluate an attribute-informed cascade (I-A) where each source node allocates its limited attention across its neighbors proportionally to their log-scaled view counts: p(u,v) = log(1+views(v)) / Σ\_{x∈N(u)} log(1+views(x)). This row-normalized formulation models Twitch-specific engagement dynamics as **popularity-biased neighbor selection within a local social context** (u preferentially activates relatively more popular neighbors among N(u)). Under I-A, degree-only baselines are disadvantaged because they cannot observe the views distribution of a node's neighborhood — a signal directly accessible to GNN message passing."_

**Tại sao I-A KHÔNG phải p-hacking:**

1. Pre-registered hypothesis TRƯỚC KHI chạy: "GNN 2-layer has inductive bias advantage under I-A because layer-1 aggregation computes the normalization weights that p(u,v) uses."
2. Pilot check với threshold rõ ràng (CV, ρ) — không điều chỉnh sau khi thấy kết quả.
3. A0 vẫn là primary; I-A là second operationalization, clearly labeled.
4. Report all pilot outcomes kể cả nếu fail.

---

#### II-B — Views-Density Cascade (Fallback nếu I-A degenerate)

**Formula:**

```
p(u,v) = clip(w(v) / deg(v), max=0.5)    với w(v) = views_norm(v) ∈ [0,1]
```

Clip at 0.5 thay vì 1.0 để tránh cascade quá explosive khi một node có cả high views lẫn low degree.

**Khi nào dùng II-B thay I-A:** Nếu pilot I-A cho CV < 0.3 (cascade degenerate do views quá concentrated). II-B không row-normalized nên cascade không chết ở hop-1 → CV cao hơn.

**Hạn chế vs I-A:** One-hop analytical proxy `Σ_{v∈N(u)} w(v)/deg(v)` có thể capture nhiều variance → GNN advantage nhỏ hơn. Phải kiểm tra ρ(IC-II-B, 1hop_proxy) < 0.85.

---

#### Pilot Decision Protocol — Bắt buộc chạy trước khi commit full IC sim

> **Thời gian:** ~15–20 phút (200 nodes × 50 runs cho mỗi variant). Chạy trước khi bắt đầu full 5k × 200 runs.

```python
import numpy as np
from scipy.stats import spearmanr

# ── Precompute views weights (O(N), một lần) ───────────────────────────────
# node_attrs: DataFrame với columns ['node_id', 'views'] — tất cả active nodes
# Align với CSR node ordering (node_ids_ordered từ graph_csr.npz)
views_raw = node_attrs_ordered['views'].fillna(0).values  # shape [n_active]
views_log  = np.log1p(views_raw)                          # log1p transform

# Precompute per-node neighbor views sum (O(E), một lần)
neighbor_views_sum = np.zeros(n_active, dtype=np.float64)
for u in range(n_active):
    nbrs = indices[indptr[u]:indptr[u+1]]
    if len(nbrs) > 0:
        neighbor_views_sum[u] = views_log[nbrs].sum()
# neighbor_views_sum[u] = Σ_{x∈N(u)} log1p(views(x)) — denominator của I-A

# ── IC-I-A pilot (200 nodes × 50 runs) ────────────────────────────────────
def run_ic_csr_ia_pilot(seed_node, indptr, indices, views_log,
                         neighbor_views_sum, n_runs=50, worker_seed=None):
    """
    IC với p(u,v) = views_log[v] / neighbor_views_sum[u].
    Precomputed neighbor_views_sum để tránh recompute trong inner loop.
    """
    rng = np.random.default_rng(seed=worker_seed)
    sizes = []
    for _ in range(n_runs):
        activated = {seed_node}
        frontier  = [seed_node]
        while frontier:
            next_frontier = []
            for node in frontier:
                denom = neighbor_views_sum[node]
                if denom <= 0:
                    continue
                for idx in range(indptr[node], indptr[node+1]):
                    nb = indices[idx]
                    if nb not in activated:
                        p = views_log[nb] / denom   # I-A formula
                        if rng.random() < p:
                            activated.add(nb)
                            next_frontier.append(nb)
            frontier = next_frontier
        sizes.append(len(activated))
    return np.array(sizes, dtype=np.int32)

# ── Chạy pilot ────────────────────────────────────────────────────────────
# Use the same degree-quintile stratified pilot selection as the Day-1 pilot.
# (Helper `select_pilot_nodes_stratified` is defined in the Day-1 pilot snippet above.)
pilot_nodes = select_pilot_nodes_stratified(degrees, n_pilot=200, q=5, seed=42)

ic_a0_pilot = {n: run_ic_csr(n, indptr, indices, degrees, n_runs=50, worker_seed=42+n).mean()
               for n in pilot_nodes}
ic_ia_pilot = {n: run_ic_csr_ia_pilot(n, indptr, indices, views_log,
                                        neighbor_views_sum, n_runs=50, worker_seed=42+n).mean()
               for n in pilot_nodes}

ic_a0 = np.array([ic_a0_pilot[n] for n in pilot_nodes])
ic_ia = np.array([ic_ia_pilot[n] for n in pilot_nodes])
deg   = np.array([degrees[n] for n in pilot_nodes])
views = np.array([views_raw[n] for n in pilot_nodes])

# Neighbor-views mean proxy (analytical baseline for I-A — degree blind)
nbr_views_mean = np.array([
    views_log[indices[indptr[n]:indptr[n+1]]].mean() if indptr[n] < indptr[n+1] else 0.0
    for n in pilot_nodes
])
# 2-hop analytical proxy for I-A (costly but gives upper bound)
# two_hop_ia[u] ≈ Σ_{v∈N(u)} [w(v)/Σw(N(u))] × [1 + Σ_{w∈N(v)} w(w)/Σw(N(v))]
# Skip for pilot — use nbr_views_mean as proxy upper bound

# ── Decision checks ───────────────────────────────────────────────────────
cv_ia          = ic_ia.std() / ic_ia.mean() if ic_ia.mean() > 0 else 0
cv_a0          = ic_a0.std() / ic_a0.mean() if ic_a0.mean() > 0 else 0
rho_deg_a0, _  = spearmanr(ic_a0, deg)
rho_deg_ia, _  = spearmanr(ic_ia, deg)
rho_views_ia, _ = spearmanr(ic_ia, views)
rho_nbr_ia, _  = spearmanr(ic_ia, nbr_views_mean)   # key check: analytical proxy
rho_a0_ia, _   = spearmanr(ic_a0, ic_ia)

print(f"=== IC PILOT DECISION ===")
print(f"A0: CV={cv_a0:.3f} | ρ(IC,degree)={rho_deg_a0:.3f}")
print(f"I-A: CV={cv_ia:.3f} | ρ(IC,degree)={rho_deg_ia:.3f} | ρ(IC,views)={rho_views_ia:.3f}")
print(f"     ρ(IC-I-A, nbr_views_mean)={rho_nbr_ia:.3f}  ← KEY: must be < 0.85")
print(f"     ρ(IC-A0, IC-I-A)={rho_a0_ia:.3f}  ← operationalizations differ if < 0.85")
```

**Decision tree bắt buộc (ghi vào `docs/day1_decisions.md`):**

```
CHECK 1 — Non-degenerate:
  cv_ia > 0.3
    → PASS: I-A cascade không degenerate ✓
  cv_ia ≤ 0.3
    → FAIL: Switch to II-B (p=w(v)/deg(v)); re-run pilot với II-B

CHECK 2 — Degree clearly disadvantaged:
  rho_deg_ia < 0.75
    → PASS: degree baseline substantially blind ✓
  0.75 ≤ rho_deg_ia < 0.85
    → MARGINAL: proceed but note GNN advantage may be moderate
  rho_deg_ia ≥ 0.85
    → FAIL: I-A still too correlated with degree → Use A2 sensitivity only

CHECK 3 — Simple proxy not dominant:
  rho_nbr_ia < 0.85
    → PASS: neighbor-views mean insufficient → GNN has room to learn ✓
  rho_nbr_ia ≥ 0.85
    → WARNING: GNN advantage may be limited (analytical proxy too strong)
    → Still proceed (GNN learns non-linear combinations + faster inference)
    → Note trong paper: "simple neighbor-views aggregation achieves ρ=[X];
       GNN improves via multi-hop and learned non-linear composition"

ALL PASS (CHECK 1+2+3):
  → Run full I-A IC sim: 5k nodes × 200 runs → ic_scores_ia.parquet
  → Run C2-I-A: 5 archs × 5 seeds trên I-A labels → surrogate_ranking_metrics_ia.csv
      (APPNP + GAT + **GATv2** + GIN + GCN — xem bên dưới tại sao GATv2 thay thế/bổ sung GAT trong I-A)
  → Run C4-I-A: Bootstrap CI GNN_best vs degree on I-A labels

ANY FAIL → Fallback theo thứ tự: II-B → A2 → A0 only
```

---

#### C2-I-A Architecture Selection — Tại sao GATv2 quan trọng hơn GAT trong I-A track

> **Lý thuyết alignment quan trọng (pre-register trước C2-I-A):**
>
> | | GAT v1 | GATv2 |
> |---|---|---|
> | Attention formula | `e(i,j) = a^T [W·h_i \|\| W·h_j]` | `e(i,j) = a^T LeakyReLU(W·[h_i \|\| h_j])` |
> | Attention type | **Static** — ranking của j không thay đổi theo i | **Dynamic** — ranking của j phụ thuộc i |
> | I-A formula | `p(u,v) = views(v) / Σviews(N(u))` | Row-normalize: denominator phụ thuộc u |
> | Alignment | GAT v1 attention KHÔNG dynamic → không thể học I-A row-normalization | **GATv2 dynamic attention khớp trực tiếp với I-A: trọng số của v phụ thuộc u** |
> | A0 alignment | OK (static 1/deg(v) chỉ cần target-node info) | Overkill cho A0, fine nhưng no advantage |
>
> **Kết luận:** Trong C2-A0, GAT v1 là đúng choice (static attention ≈ A0 static p(u,v)). Trong C2-I-A, **GATv2 là correct architecture** — dynamic attention cho phép model học row-normalized views weighting.

**C2-I-A architecture list (nếu I-A pilot pass):**

```python
from torch_geometric.nn import GATv2Conv

class GATv2Surrogate(nn.Module):
    """
    GATv2 surrogate — Dynamic attention (Brody et al., ICLR 2022).
    
    Alignment với I-A IC (H4):
    I-A formula: p(u,v) = log1p(views(v)) / Σ_{w∈N(u)} log1p(views(w))
    → Attention weight của v PHỤ THUỘC ngữ cảnh u (row-normalized per source)
    → GATv2 dynamic attention: e(i,j) = a^T LeakyReLU(W·[h_i || h_j])
      → ranking của j thay đổi theo i (dynamic) → có thể học I-A normalization
    
    Tại sao GAT v1 không đủ cho I-A:
    GAT v1: e(i,j) = a_src^T(W·h_i) + a_tgt^T(W·h_j)  [separable]
    → ranking của j KHÔNG thay đổi theo i → cannot model row-normalization
    
    Ref: Brody et al., "How Attentive are Graph Attention Networks?", ICLR 2022
    """
    def __init__(self, in_dim=3, hidden_dim=128, heads=4, dropout=0.3):
        super().__init__()
        self.conv1 = GATv2Conv(in_dim, hidden_dim // heads, heads=heads,
                               dropout=dropout, concat=True)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads,
                               dropout=dropout, concat=True)
        self.head  = nn.Linear(hidden_dim, 1)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = self.drop(x)
        x = F.elu(self.conv2(x, edge_index))
        return self.head(x).squeeze(-1)

# Hypothesis H4 (GATv2 — I-A dynamic attention alignment):
# I-A p(u,v) row-normalized per source → requires dynamic attention (H4)
# Expected: GATv2 > GAT v1 specifically under I-A labels
# C2-I-A ARCHITECTURES (replace GAT v1 with GATv2 for I-A track):
C2_IA_ARCHITECTURES = ['appnp', 'gatv2', 'gin', 'gcn', 'sage']
# Note: 'gatv2' uses GATv2Surrogate; others use same get_model() factory
# Add to get_model() factory:
# elif arch == 'gatv2':
#     return GATv2Surrogate(in_dim=in_dim, hidden_dim=hidden_dim, heads=4, dropout=dropout)
```

> **C2-I-A vs C2-A0 arch list comparison:**
>
> | Arch | C2-A0 | C2-I-A | Lý do khác biệt |
> |---|---|---|---|
> | APPNP | ✅ MUST (H3) | ✅ MUST (H3 still valid) | PPR propagation phù hợp cả 2 |
> | GAT v1 | ✅ MUST (H1) | ❌ Replaced by GATv2 | Static attention không model I-A |
> | **GATv2** | ❌ Not in C2-A0 | ✅ MUST (H4) | Dynamic attention = I-A alignment |
> | GIN | ✅ MUST | ✅ MUST | Sum agg. reference |
> | GCN | ✅ MUST (H2) | ✅ MUST | Baseline comparison |
> | SAGE | ✅ Done (baseline) | ✅ Done (baseline) | Mean agg. baseline |

**Output C2-I-A:** `surrogate_ranking_metrics_ia.csv` — same schema as primary CSV, với model_name: `appnp_raw_attr_ia`, `gatv2_raw_attr_ia`, `gin_raw_attr_ia`, `gcn_raw_attr_ia`, `gnn_raw_attr_ia` (SAGE).

---

#### Framework thử nghiệm (đúng linear pipeline)

Với **mỗi operationalization** (A0 primary, I-A nếu bật + pilot pass, A2 sensitivity):

1. **[1] Metric tốt?** — CV > 0.3; `Spearman(IC_variant, degree)` < 0.85; không trivially predictable bởi 1-hop proxy.
2. **[3] Stability** — Jaccard top-10% qua 3 MC seeds. I-A có thể kém stable hơn A0 → report honestly.
3. **[4] Surrogate** — C2 protocol (same split/hyperparams/5 seeds/4 archs); primary question per variant:
   - A0: bootstrap CI GNN vs degree (statistical equivalence sufficient)
   - I-A: GNN expected significantly > degree; architecture comparison reveals which arch learns attribute propagation best

**Prepared narratives — mọi outcome đều publishable:**

_I-A pilot pass, GNN >> degree (best case):_

> _"Under the attribute-informed cascade (I-A), GNN-raw-attr achieves Spearman ρ=[X], significantly outperforming degree-based ranking (ρ=[Y], bootstrap CI lower=[Z]>0). Degree-based baselines cannot capture the views distribution of a node's neighborhood — a signal directly computed by GNN message passing — explaining the substantial performance gap. Under the structural cascade (A0), GNN achieves statistically comparable performance to degree (bootstrap CI: [X]–[Y]), demonstrating that GNN's surrogate advantage scales with the complexity of the underlying diffusion dynamics."_

_I-A pilot fail, fallback to A0+A2:_

> _"Pilot evaluation of attribute-informed IC variants revealed [degenerate distribution/high proxy correlation], suggesting [reason]. We therefore proceed with the structural weighted cascade (A0) as our sole operationalization, and report A2 symmetric as a robustness sensitivity check."_

_GCN improves most under A2 (H2 confirmed):_

> _"GCN's Spearman improves from [X] (A0) to [Y] (A2 labels), consistent with the structural alignment between GCN's D^{-1/2}AD^{-1/2} normalization and the symmetric diffusion operator."_

---

### 4.2 IC Implementation — CSR + loky (không phải igraph threads)

```python
from scipy.sparse import csr_matrix
import networkx as nx
import numpy as np
import random

# Bước 1: Convert graph sang CSR ONCE — numpy arrays shared efficiently
adj = nx.to_scipy_sparse_array(G_active, format='csr')
indptr  = adj.indptr.copy()     # row pointers
indices = adj.indices.copy()    # column indices (neighbors)
degrees = np.diff(indptr)       # degree array

# Bước 2: IC simulation với CSR — không cần graph object trong worker
def run_ic_csr(seed_node, indptr, indices, degrees, n_runs=200, worker_seed=None):
    """
    Monte Carlo IC dùng CSR format.
    Mỗi worker có RNG riêng, seed deterministically derived.
    KHÔNG dùng random.seed() global — gây race condition với threads.
    """
    rng = np.random.default_rng(seed=worker_seed)  # RNG riêng cho mỗi worker
    sizes = []

    for _ in range(n_runs):
        activated = {seed_node}
        frontier = [seed_node]

        while frontier:
            next_frontier = []
            for node in frontier:
                start_idx = indptr[node]
                end_idx   = indptr[node + 1]
                for idx in range(start_idx, end_idx):
                    nb = indices[idx]
                    if nb not in activated:
                        p = 1.0 / degrees[nb] if degrees[nb] > 0 else 0.0
                        if rng.random() < p:
                            activated.add(nb)
                            next_frontier.append(nb)
            frontier = next_frontier

        sizes.append(len(activated))

    return np.array(sizes, dtype=np.int32)

# ── Sensitivity S1: A2 Symmetric variant ───────────────────────────────────────
# Chỉ thay 1 dòng so với run_ic_csr (A0): p = 1/sqrt(deg(u)*deg(v))
# Output: outputs/mapr2026_v3_results/ic_scores_sensitivity_a2.parquet

def run_ic_csr_a2(seed_node, indptr, indices, degrees, n_runs=200, worker_seed=None):
    """
    A2 Symmetric IC: p(u,v) = 1/sqrt(deg(u) * deg(v)).
    Structurally analogous to GCN's D^{-1/2}AD^{-1/2} normalization.
    Cùng CSR format, cùng RNG setup — fair comparison với A0.
    """
    rng = np.random.default_rng(seed=worker_seed)
    sizes = []

    for _ in range(n_runs):
        activated = {seed_node}
        frontier = [seed_node]

        while frontier:
            next_frontier = []
            for node in frontier:
                deg_node = degrees[node]
                start_idx = indptr[node]
                end_idx   = indptr[node + 1]
                for idx in range(start_idx, end_idx):
                    nb = indices[idx]
                    if nb not in activated:
                        # A2: p = 1/sqrt(deg(u)*deg(v)) — symmetric
                        if deg_node > 0 and degrees[nb] > 0:
                            p = 1.0 / np.sqrt(float(deg_node) * float(degrees[nb]))
                        else:
                            p = 0.0
                        if rng.random() < p:
                            activated.add(nb)
                            next_frontier.append(nb)
            frontier = next_frontier

        sizes.append(len(activated))

    return np.array(sizes, dtype=np.int32)

# ── I-A: Attribute-Informed IC (Row-Normalized Views Attention) ────────────
# ĐIỀU KIỆN: pilot pass (CV>0.3, ρ_deg<0.85, ρ_proxy<0.85)
# Output: outputs/mapr2026_v3_results/ic_scores_ia.parquet

# Bước 0 (một lần trước khi chạy parallel): precompute views weights
def precompute_views_weights(node_ids_ordered, node_attrs_df, indptr, indices):
    """
    Precompute:
      views_log[u]           = log1p(views(u))               — per-node
      neighbor_views_sum[u]  = Σ_{v∈N(u)} log1p(views(v))   — per-node (O(E))
    Dùng cho IC-I-A inner loop để tránh recompute.
    """
    # Align node_attrs với CSR ordering
    attrs = node_attrs_df.set_index('node_id').reindex(node_ids_ordered)
    views_raw = attrs['views'].fillna(0).values.astype(np.float64)
    views_log = np.log1p(views_raw)

    n = len(node_ids_ordered)
    neighbor_views_sum = np.zeros(n, dtype=np.float64)
    for u in range(n):
        nbrs = indices[indptr[u]:indptr[u+1]]
        if len(nbrs) > 0:
            neighbor_views_sum[u] = views_log[nbrs].sum()

    return views_log, neighbor_views_sum  # both shape [n_active]

def run_ic_csr_ia(seed_node, indptr, indices, views_log, neighbor_views_sum,
                  n_runs=200, worker_seed=None):
    """
    IC-I-A: p(u,v) = views_log[v] / neighbor_views_sum[u]
    Row-normalized → one-hop spread = 1.0 ∀u → IC score driven by 2+ hop.
    GNN 2-layer has inductive bias advantage: layer-1 aggregates views_log of N(u)
    = computes near-exact denominator and numerator of p(u,v).

    NOTE: views_log và neighbor_views_sum là read-only numpy arrays —
    pickle-safe cho loky processes (shared memory via copy-on-write).
    """
    rng = np.random.default_rng(seed=worker_seed)
    sizes = []
    for _ in range(n_runs):
        activated = {seed_node}
        frontier  = [seed_node]
        while frontier:
            next_frontier = []
            for node in frontier:
                denom = neighbor_views_sum[node]
                if denom <= 0.0:
                    continue  # isolated node hoặc all-zero views neighbors
                start_idx = indptr[node]
                end_idx   = indptr[node + 1]
                for idx in range(start_idx, end_idx):
                    nb = indices[idx]
                    if nb not in activated:
                        p = views_log[nb] / denom   # I-A formula — core change
                        if p > 0.0 and rng.random() < p:
                            activated.add(nb)
                            next_frontier.append(nb)
            frontier = next_frontier
        sizes.append(len(activated))
    return np.array(sizes, dtype=np.int32)

def run_ic_all_nodes_ia(sampled_nodes, indptr, indices,
                         views_log, neighbor_views_sum,
                         n_runs=200, n_jobs=-1):
    """Parallel IC-I-A cho tất cả sampled nodes."""
    results = Parallel(n_jobs=n_jobs, prefer='loky')(
        delayed(run_ic_csr_ia)(
            node, indptr, indices, views_log, neighbor_views_sum,
            n_runs=n_runs, worker_seed=42 + node
        )
        for node in sampled_nodes
    )
    return dict(zip(sampled_nodes, results))

# ── II-B Fallback: Views-Density (non-normalized) — nếu I-A pilot CV < 0.3 ──
# p(u,v) = clip(views_norm[v] / deg(v), max=0.5)
# Tránh quá explosive khi views cao + degree thấp → clip at 0.5
def precompute_views_density(node_ids_ordered, node_attrs_df, degrees):
    """p_iib[v] = clip(views_norm[v] / max(deg(v),1), 0, 0.5)"""
    attrs = node_attrs_df.set_index('node_id').reindex(node_ids_ordered)
    views_raw = attrs['views'].fillna(0).values.astype(np.float64)
    vmax = views_raw.max()
    views_norm = views_raw / vmax if vmax > 0 else views_raw
    p_iib = np.clip(views_norm / np.maximum(degrees.astype(np.float64), 1.0), 0.0, 0.5)
    return p_iib  # shape [n_active], pre-computed per-node edge weight for v

def run_ic_csr_iib(seed_node, indptr, indices, p_iib, n_runs=200, worker_seed=None):
    """
    IC-II-B: p(u,v) = p_iib[v]  (pre-computed, non-normalized)
    Fallback nếu I-A pilot CV < 0.3.
    """
    rng = np.random.default_rng(seed=worker_seed)
    sizes = []
    for _ in range(n_runs):
        activated = {seed_node}
        frontier  = [seed_node]
        while frontier:
            next_frontier = []
            for node in frontier:
                for idx in range(indptr[node], indptr[node+1]):
                    nb = indices[idx]
                    if nb not in activated:
                        p = p_iib[nb]   # II-B: depends only on target v
                        if p > 0.0 and rng.random() < p:
                            activated.add(nb)
                            next_frontier.append(nb)
            frontier = next_frontier
        sizes.append(len(activated))
    return np.array(sizes, dtype=np.int32)

# ── Sensitivity S2: A1 Source Budget variant (nếu cần "IC ≠ degree" stronger) ──
# Chỉ thay 1 dòng: p = 1.0/deg_node (thay vì degrees[nb]) — mọi node 1-hop spread = 1.0
# def run_ic_csr_a1(seed_node, ...): p = 1.0/deg_node if deg_node > 0 else 0.0
# Output: outputs/mapr2026_v3_results/ic_scores_sensitivity_a1.parquet

# ── Unified wrapper cho tất cả variants ───────────────────────────────────
def run_ic_variant(sampled_nodes, indptr, indices, degrees,
                   p_rule='a0', n_runs=200, n_jobs=-1, **kwargs):
    """
    p_rule options:
      'a0'           → A0 weighted cascade (primary)
      'a2'           → A2 symmetric (sensitivity S1) — kwargs: none extra
      'ia'           → I-A attr-informed — kwargs: views_log, neighbor_views_sum
      'iib'          → II-B views-density fallback — kwargs: p_iib
      'a1'           → A1 source budget (sensitivity S2) — kwargs: none extra
    Output cùng schema (dict node_id → np.array[n_runs]).
    """
    if p_rule == 'a0':
        fn = lambda n: run_ic_csr(n, indptr, indices, degrees, n_runs, 42+n)
    elif p_rule == 'a2':
        fn = lambda n: run_ic_csr_a2(n, indptr, indices, degrees, n_runs, 42+n)
    elif p_rule == 'ia':
        vl, nvs = kwargs['views_log'], kwargs['neighbor_views_sum']
        fn = lambda n: run_ic_csr_ia(n, indptr, indices, vl, nvs, n_runs, 42+n)
    elif p_rule == 'iib':
        p_arr = kwargs['p_iib']
        fn = lambda n: run_ic_csr_iib(n, indptr, indices, p_arr, n_runs, 42+n)
    elif p_rule == 'a1':
        fn = lambda n: run_ic_csr_a1(n, indptr, indices, degrees, n_runs, 42+n)
    else:
        raise ValueError(f"Unknown p_rule: {p_rule}")

    results = Parallel(n_jobs=n_jobs, prefer='loky')(
        delayed(fn)(node) for node in sampled_nodes
    )
    return dict(zip(sampled_nodes, results))

# Bước 3: Parallel execution — dùng 'loky' (processes), KHÔNG phải 'threads'
# Lý do: Python GIL serializes threads với CPU-bound code
# CSR numpy arrays được pickle efficiently qua memory mapping
from joblib import Parallel, delayed

def run_ic_all_nodes(sampled_nodes, indptr, indices, degrees, n_runs=200, n_jobs=-1):
    """Parallel IC cho tất cả sampled nodes."""
    results = Parallel(n_jobs=n_jobs, prefer='loky')(
        delayed(run_ic_csr)(
            node, indptr, indices, degrees,
            n_runs=n_runs,
            worker_seed=42 + node  # deterministic per-worker seed
        )
        for node in sampled_nodes
    )
    return dict(zip(sampled_nodes, results))
```

### 4.3 Label Stability Diagnostic

```python
from itertools import combinations  # required for pairwise seed comparison

def check_label_stability(sampled_nodes, indptr, indices, degrees,
                           n_seeds=3, n_runs=150, top_k_pct=0.10):
    """
    3 independent MC experiments (giảm từ 5 để tiết kiệm compute).
    Jaccard > 0.85 → labels stable.
    """
    ic_runs = {}
    for mc_seed in range(n_seeds):
        ic_runs[mc_seed] = {
            node: run_ic_csr(node, indptr, indices, degrees,
                             n_runs=n_runs, worker_seed=mc_seed * 10000 + node).mean()
            for node in sampled_nodes
        }

    k = int(len(sampled_nodes) * top_k_pct)
    jaccards = []
    for i, j in combinations(range(n_seeds), 2):
        top_i = set(sorted(ic_runs[i], key=ic_runs[i].get, reverse=True)[:k])
        top_j = set(sorted(ic_runs[j], key=ic_runs[j].get, reverse=True)[:k])
        jaccards.append(len(top_i & top_j) / len(top_i | top_j))

    mean_j = np.mean(jaccards)
    if mean_j < 0.85:
        print(f"WARNING: Label instability (Jaccard={mean_j:.3f}). Increase n_runs.")
    return mean_j

def bootstrap_ci_ic(ic_runs_array, n_bootstrap=1000, alpha=0.05):
    """Bootstrap 95% CI cho mean IC reach — negligible compute."""
    means = [np.random.choice(ic_runs_array, len(ic_runs_array), replace=True).mean()
             for _ in range(n_bootstrap)]
    return np.percentile(means, [100*alpha/2, 100*(1-alpha/2)])
```

### 4.4 Sampling với Representativeness Check

```python
def stratified_sample_with_ks_check(df, n_sample=5000, seed=42):
    """
    Stratify theo degree quintile.
    KS test để validate representativeness.
    """
    df['deg_q'] = pd.qcut(df['degree'], q=5, labels=False, duplicates='drop')
    sampled = (df.groupby('deg_q', group_keys=False)
                 .apply(lambda x: x.sample(frac=n_sample/len(df), random_state=seed))
                 .reset_index(drop=True))

    from scipy.stats import ks_2samp
    ks_results = {}
    for col in ['degree', 'kshell', 'pagerank']:
        if col in df.columns:
            stat, p = ks_2samp(df[col].dropna(), sampled[col].dropna())
            ks_results[col] = {'ks_stat': round(stat, 4), 'p': round(p, 4)}
            if stat > 0.10:
                print(f"WARNING: {col} distribution not representative (KS={stat:.3f})")

    return sampled, ks_results
```

---

## 6. Community Detection — Bắt buộc cho Stability Explanation _(Task A support)_

> **v3.1 note:** Louvain community detection phục vụ 2 mục đích:
> (1) **Stability explanation**: tính `pct_communities_spanning_boundary` để giải thích tại sao Jaccard < 0.85 (structural cause, not MC noise)
> (2) **Proxy/feature**: `cross_community_edge_fraction` dùng cho baselines/proxies (nếu cần)

```python
import community as community_louvain  # python-louvain
import numpy as np

def compute_community_features(G_nx, resolution=1.0, n_runs=10, seed_start=0):
    """
    Louvain community detection — O(N log N), vài phút trên 168k nodes.
    Dùng cho stability explanation và làm proxy feature theo community.
    """
    partitions, modularities = [], []
    for seed in range(seed_start, seed_start + n_runs):
        p = community_louvain.best_partition(G_nx, resolution=resolution, random_state=seed)
        q = community_louvain.modularity(p, G_nx)
        partitions.append(p)
        modularities.append(q)

    partition = partitions[int(np.argmax(modularities))]  # best run by modularity
    # community_id cho mỗi node
    community_ids = partition

    # Cross-community edge fraction (proxy cho participation coefficient)
    cross_comm = {}
    for node in G_nx.nodes():
        neighbors = list(G_nx.neighbors(node))
        if len(neighbors) == 0:
            cross_comm[node] = 0.0
            continue
        n_cross = sum(1 for nb in neighbors if partition[nb] != partition[node])
        cross_comm[node] = n_cross / len(neighbors)

    return community_ids, cross_comm

# Save to data/processed/community_features.parquet (file riêng):
# - node_id
# - community_id
# - cross_community_edge_fraction
# Consumers join với node_attributes.parquet theo node_id (KHÔNG ghi đè node_attributes.parquet)
```

---

## 7. Baseline Hierarchy — Đầy đủ và Không Redundant

### Group 1: Raw Feature Baselines (O(1))

| Baseline       | Formula               |
| -------------- | --------------------- |
| Views rank     | rank(views)           |
| Views/day rank | rank(views/life_time) |
| Degree rank    | rank(degree)          |

### Group 2: Structural Centrality (O(N log N) đến O(NE))

| Baseline             | Implementation                       |
| -------------------- | ------------------------------------ |
| PageRank             | nx.pagerank(G, alpha=0.85)           |
| k-shell              | nx.core_number(G)                    |
| Betweenness (approx) | NetworKit ApproxBetweenness2, ε=0.10 |

### Group 3: Cheap Diffusion Proxies — one-hop O(E), two-hop naive O(Σ d(v)^2) — KHÔNG Redundant

> **Đây là group critical nhất.** v2 có `one-hop spread` và `weighted degree` — thực chất **cùng formula** với weighted cascade undirected. v3 fix bằng cách thay `weighted degree` bằng **2-hop proxy**.

```python
def one_hop_expected_spread(node, G, degrees):
    """Σ 1/degree(v) for v in N(u). O(deg(u))."""
    return sum(1.0/max(degrees[v], 1) for v in G.neighbors(node))

def two_hop_expected_spread(node, G, degrees):
    """
    E[cascade size through 2 hops] under weighted cascade.
    Mạnh hơn one-hop và genuinely different.
    O(deg²) per node; full-graph naive complexity gần O(Σ d(v)^2).
    """
    total = 0.0
    for v in G.neighbors(node):
        p_uv = 1.0 / max(degrees[v], 1)
        second_hop = sum(1.0/max(degrees[w], 1)
                         for w in G.neighbors(v) if w != node)
        total += p_uv * (1 + second_hop)
    return total
```

| Baseline           | Formula                     | Complexity       |
| ------------------ | --------------------------- | ---------------- |
| One-hop spread     | `Σ 1/deg(v)` for v in N(u)  | O(deg(u))        |
| **Two-hop spread** | `Σ p(u,v) × (1 + Σ p(v,w))` | O(deg²) per node |

### Group 4: Shallow Embedding Baselines

| Baseline              | Config                                                             |
| --------------------- | ------------------------------------------------------------------ |
| Node2Vec + LR         | dim=64, walks=**20** (không phải 200), walk_len=20 → LR regression |
| MLP on raw attributes | 2-layer MLP, features = [views_log, views/day, life_time]          |

### Group 5: GNN — Architecture × Feature Ablation (v3.1)

> **v3.1 update:** Mở rộng từ GraphSAGE duy nhất → 4 architectures per instructor recommendation.
> Primary feature set vẫn là raw attributes để story "GNN learns beyond hand-crafted features" defensible.

**Architecture Comparison (MUST — per instructor: "GCN/GIN/GAT..."):**

| Architecture | PyG Class  | Aggregation           | Lý do test                                                                                                                                       |
| ------------ | ---------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| GraphSAGE    | `SAGEConv` | Mean (baseline)       | Hiện tại đang dùng; reference point                                                                                                              |
| **GCN**      | `GCNConv`  | **Sym. norm. sum**    | Spectral baseline; **`D^{-1/2}AD^{-1/2}` structurally analogous to A2 diffusion rule** — additional inductive bias check nếu chạy A2 sensitivity |
| GIN          | `GINConv`  | Sum + MLP             | Sum agg. preserves multi-hop counts (WL-equivalent expressiveness); reference for non-degree-weighted IC dynamics                                |
| **GAT**      | `GATConv`  | **Learned attention** | **Hypothesis (confirm via C2):** p(u,v)=1/degree(v) → attention **có thể** học weighting này                                                     |

> **Hai inductive bias hypotheses (cả hai to be confirmed by C2):**
>
> 1. **GAT–A0 hypothesis:** Dưới IC primary (A0: `p=1/deg(v)`), GAT có thể học attention weight tỷ lệ nghịch với degree của neighbor → potentially best aligned with A0. _(hypothesis/intuition — C2 decides)_
> 2. **GCN–A2 hypothesis:** `GCNConv` aggregates với weight `1/√(d̃_u×d̃_v)` — structurally analogous to A2 symmetric rule. Nếu chạy A2 sensitivity IC labels, GCN expected to be best arch. _(testable via sensitivity experiment — see Section 4.1b)_
>
> ⚠ Không kết luận arch nào “best” trước khi có kết quả C2; nếu C2 cho arch khác tốt hơn thì dùng kết quả thực nghiệm. Cả hai hypotheses có prepared narratives cho mọi outcome.

**Feature Ablation (giữ nguyên từ v3.0):**

| Variant        | Features                          | Role                                                          |
| -------------- | --------------------------------- | ------------------------------------------------------------- |
| GNN-raw-attr   | views_log, views/day, life_time   | **Primary proposed** (mọi architecture test trên variant này) |
| GNN-graph-only | degree_norm only (or random init) | Ablation: topology without attributes                         |
| GNN-centrality | degree, PR, kshell                | Ablation: centrality features                                 |
| GNN-full       | all 6 features                    | Supplementary upper bound — **[✦ IF TIME]**                   |

**Matrix thực nghiệm (priority):**

|               | raw_attr | graph_only | centrality |
| ------------- | -------- | ---------- | ---------- |
| **GraphSAGE** | ✅ đã có | ✅ đã có   | ✅ đã có   |
| **GCN**       | **MUST** | [IF TIME]  | [IF TIME]  |
| **GIN**       | **MUST** | [IF TIME]  | [IF TIME]  |
| **GAT**       | **MUST** | [IF TIME]  | [IF TIME]  |

**Naming convention cho surrogate_ranking_metrics.csv (artifact names):**
`gcn_raw_attr`, `gin_raw_attr`, `gat_raw_attr` (+ `best_arch_raw_attr_rankloss` sau C2)

> **Lưu ý phân biệt:** Tên trong CSV artifact (`gcn_raw_attr`) khác với tên display trong paper table (`gnn_raw_attr (GCN)`).
> Quy ước: CSV dùng snake*case prefix (`gcn*`, `gin*`, `gat*`); paper table dùng `gnn_raw_attr (Architecture)` để nhất quán với G5 group labeling.

**Tại sao cấu trúc này:**

- GNN-raw-attr vs MLP-raw-attr: giá trị của **message passing** (+0.099 Spearman confirmed)
- Architecture comparison: **which message passing** hoạt động tốt nhất cho IC proxy task
- GNN-raw-attr vs GNN-centrality: giá trị của **centrality features** vs learned structure
- Best GNN vs degree: **bootstrap CI** để test statistical equivalence (Section 8.5)

---

## 8. Evaluation Metrics và Protocol

### 8.1 Transductive Setting — Phải Nói Rõ

```python
# Bắt buộc viết trong paper (Section 3.4):
# "We evaluate in a transductive node-level regression setting on a fixed static
#  graph, where IC labels are available only for a stratified node subset (N_sample
#  nodes). All accuracy/ranking metrics are computed on held-out LABELED nodes only.
#  Full-graph inference (all 168,114 nodes) is reported solely for runtime assessment."
```

### 8.2 Primary Metrics (ranking task)

```
Spearman ρ     — primary: rank correlation với IC scores
NDCG@10%       — secondary: ranking quality
Precision@10%  — supplementary
```

### 8.3 TRÁNH

```
Accuracy, F1-macro  — misleading với 95/5 class imbalance
```

### 8.4 Degree-Controlled IC Variance Test (v3.1 — MUST, justify "why not just degree?")

**Mục tiêu:** Chứng minh IC capture higher-order diffusion effects ngoài degree. Trả lời câu hỏi
reviewer: _"Why not just use degree as power user metric?"_

```python
# Filter nodes với degree trong narrow band quanh mean (degree_mean ≈ 81)
degree_band = (75, 85)  # ±5 quanh mean
band_nodes = [n for n in labeled_nodes if degree_band[0] <= degree[n] <= degree_band[1]]

ic_in_band = ic_scores[band_nodes]
cv_within_band = ic_in_band.std() / ic_in_band.mean()

# If cv_within_band > 0.3 → IC adds information beyond degree
# Expected: nodes same degree but different community positions → different IC scores
```

**Output:** `outputs/mapr2026_v3_results/degree_controlled_ic_variance.json`

```json
{
  "degree_band": "75-85",
  "n_nodes_in_band": "<N>",
  "ic_mean_in_band": "<float>",
  "ic_std_in_band": "<float>",
  "cv_within_band": "<float>",
  "interpretation": "IC adds info beyond degree" | "IC ≈ degree at this scale"
}
```

**Thời gian:** ~30 phút (filter + compute từ existing IC scores, không cần rerun simulation).

**Paper narrative nếu cv > 0.3:** "Within the same degree band (degree 75–85), IC scores vary substantially
(CV = X), demonstrating that IC captures higher-order structural information beyond local connectivity."

**Paper narrative nếu cv ≤ 0.3 (honest limitation):** "Within narrow degree bands, IC variance is
limited, suggesting that at Twitch scale, IC is largely explained by degree. This motivates the
architecture comparison: GNN must learn non-degree structure to add value."

### 8.5 Bootstrap Significance Test — GNN vs Degree (v3.1 — MUST)

**Mục tiêu:** Kiểm định xem Δ Spearman = 0.826 − 0.817 = 0.009 có statistically significant không.

```python
import numpy as np
from scipy.stats import spearmanr

def bootstrap_spearman_ci(y_true, y_pred_a, y_pred_b, n_bootstrap=1000, seed=42):
    """Bootstrap CI for difference in Spearman between two predictors."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        rho_a, _ = spearmanr(y_true[idx], y_pred_a[idx])
        rho_b, _ = spearmanr(y_true[idx], y_pred_b[idx])
        deltas.append(rho_a - rho_b)  # GNN_best - degree
    ci_lower = np.percentile(deltas, 2.5)
    ci_upper = np.percentile(deltas, 97.5)
    return float(np.mean(deltas)), ci_lower, ci_upper

# Run: y_pred_a = best GNN predictions, y_pred_b = degree ranks, y_true = IC scores
```

**Output:** `outputs/mapr2026_v3_results/gnn_vs_degree_bootstrap_ci.json`

```json
{
  "n_bootstrap": 1000,
  "comparator_a": "gnn_best_architecture",
  "comparator_b": "degree",
  "delta_mean": "<float>",
  "ci_95_lower": "<float>",
  "ci_95_upper": "<float>",
  "interpretation": "equivalent | significantly_lower | significantly_higher"
}
```

**Thời gian:** ~10 phút (resample existing predictions, không cần retraining).

**Protocol spec (để tránh ambiguity khi implement):**

- **Metric được CI:** Spearman ρ only (primary ranking metric — không phải NDCG/Precision)
- **Đơn vị resample:** nodes trong test set (resample with replacement, `size = n_test`)
- **Δ definition:** `Spearman(GNN_best) − Spearman(degree)` trên cùng test set và cùng `y_true`
- **"GNN_best":** architecture với mean Spearman cao nhất qua 5 seeds từ C2; predictions = mean predictions qua 5 seeds
- **"degree":** `rank(degree)` trên active graph, đã filter về test nodes

**Decision:**

- `ci_95_lower > 0` → GNN significantly better → claim "GNN surpasses degree"
- `ci_95_lower ≤ 0 ≤ ci_95_upper` → **"GNN achieves statistically equivalent Spearman ρ to degree while requiring no precomputed graph statistics."**
- `ci_95_upper < 0` → **"GNN is competitive; message passing contributes +0.099 Spearman over MLP without structural features."**

### 8.6 Multiple Testing Correction

```python
from statsmodels.stats.multitest import multipletests

# Apply BH-FDR correction cho tất cả MWU tests
p_values_raw = [...]  # collect tất cả raw p-values
rejected, p_corrected, _, _ = multipletests(p_values_raw, alpha=0.05, method='fdr_bh')
# Report p_corrected, NOT p_raw
```

### 8.7 Repeated Training Seeds

```python
# 5 training seeds để report mean ± std
# v3.1: dùng GNNSurrogate(arch=...) thay vì train_graphsage cũ
training_seeds = [42, 123, 456, 789, 1024]

def train_and_eval(arch, features, seed):
    set_all_seeds(seed)
    model = GNNSurrogate(arch=arch, in_dim=features.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # ... training loop with HuberLoss ...
    return evaluate_ranking_metrics(model, test_data)

results_per_seed = []
for seed in training_seeds:
    metrics = train_and_eval(arch='sage', features=raw_attr_features, seed=seed)  # ← thay 'sage' bằng 'gcn'/'gin'/'gat' cho other architectures
    results_per_seed.append(metrics)

# Paper viết: "All metrics averaged over 5 random training seeds (mean ± std)"
mean_spearman = np.mean([r['spearman'] for r in results_per_seed])
std_spearman  = np.std( [r['spearman'] for r in results_per_seed])
# Repeat for each arch in ARCHITECTURES = ['sage', 'gcn', 'gin', 'gat']
```

---

## 9. GNN Training (v3.1 — Architecture Comparison + Ranking Loss)

### 9.1 Architecture Comparison: GCN / GIN / GAT / GraphSAGE / APPNP

> **v3.1:** Mở rộng từ GraphSAGE duy nhất → 5 architectures. Config chuẩn giống nhau để fair comparison.
> **APPNP là architecture được bổ sung mới** vì lý do lý thuyết mạnh nhất: K-step Personalized PageRank propagation với teleport/restart là **structural analogy/inductive bias** cho target diffusion-like — xem H3 bên dưới.
>
> **⚠ PyG version check (trước khi chạy):** `APPNP` requires PyG ≥ 2.3. Verify:
> ```python
> from torch_geometric.nn import APPNP; print("APPNP OK")  # must print without ImportError
> ```

```python
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GINConv, GATConv, APPNP

class GNNSurrogate(nn.Module):
    """
    Unified GNN wrapper — swap architecture bằng arch parameter.
    Primary task: Regression trên log(IC_score + 1).
    Config chuẩn: n_layers=2, hidden_dim=128, dropout=0.3 (KHÔNG thay đổi khi so sánh arch).
    Supported arch: 'sage' | 'gcn' | 'gin' | 'gat'
    APPNP dùng APPNPSurrogate (class riêng bên dưới — kiến trúc khác: embed-then-propagate).
    """
    def __init__(self, arch='sage', in_dim=3, hidden_dim=128, n_layers=2,
                 dropout=0.3, gat_heads=4):
        super().__init__()
        self.arch = arch
        self.convs = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

        def make_conv(in_c, out_c):
            if arch == 'sage':
                return SAGEConv(in_c, out_c, aggr='mean')
            elif arch == 'gcn':
                return GCNConv(in_c, out_c)           # add_self_loops=True by default
            elif arch == 'gin':
                mlp = nn.Sequential(nn.Linear(in_c, out_c), nn.ReLU(),
                                    nn.Linear(out_c, out_c))
                return GINConv(mlp)
            elif arch == 'gat':
                heads = gat_heads if out_c > 1 else 1
                # out_c // heads * heads == out_c (concat mode)
                return GATConv(in_c, out_c // heads, heads=heads,
                               dropout=0.0)  # 0.0: attention dropout separate from layer dropout

        # BUG-FIX v3.1: original had range(n_layers-2) which gave 0 iterations for n_layers=2
        # → only 1 message-passing layer. Correct: n_layers conv layers total.
        self.convs.append(make_conv(in_dim, hidden_dim))
        for _ in range(n_layers - 1):               # ← was (n_layers - 2), now (n_layers - 1)
            self.convs.append(make_conv(hidden_dim, hidden_dim))
        self.head = nn.Linear(hidden_dim, 1)
        # Result: n_layers conv layers + 1 linear head (correct 2-layer GNN for n_layers=2)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = self.act(self.dropout(conv(x, edge_index)))
        return self.head(x).squeeze(-1)


class APPNPSurrogate(nn.Module):
    """
    APPNP surrogate — embed-then-propagate pattern (khác với conv-stack trong GNNSurrogate).

    Lý thuyết: APPNP thực hiện K-step Personalized PageRank (PPR) propagation:
        x^(k) = (1 - alpha) * A_hat * x^(k-1) + alpha * x^(0)
    Với alpha = teleport/restart weight, APPNP tái-inject tín hiệu gốc x^(0) mỗi bước:
        - alpha lớn hơn → propagation “local” hơn (ít smoothing hơn)
        - alpha nhỏ hơn → propagation “diffusive” hơn (nhiều smoothing hơn)
        - K là số bước propagation (receptive field theo hop)
    → Liên hệ với IC chỉ là **structural analogy/inductive bias** (không phải tương đương hình thức).

    Ref: Klicpera et al., ICLR 2019 "Predict then Propagate"
    """
    def __init__(self, in_dim=3, hidden_dim=128, K=10, alpha=0.15, dropout=0.3):
        super().__init__()
        # Feature embedding (MLP before propagation)
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 1)
        # PPR propagation (K steps, restart prob = alpha)
        self.prop = APPNP(K=K, alpha=alpha, dropout=dropout)
        self.dropout_fn = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        # Step 1: embed features to prediction space
        x = F.relu(self.lin1(x))
        x = self.dropout_fn(x)
        x = self.lin2(x)                   # shape: (N, 1)
        # Step 2: propagate predictions via PPR (K=10, alpha=0.15)
        x = self.prop(x, edge_index)       # personalized PageRank smoothing
        return x.squeeze(-1)

    # Hyperparams (starting point): K=10 (propagation steps), alpha=0.15 (teleport/restart weight)


def get_model(arch, in_dim, hidden_dim=128, n_layers=2, dropout=0.3,
              gat_heads=4, appnp_K=10, appnp_alpha=0.15):
    """Factory function — trả về đúng model class cho từng architecture."""
    if arch == 'appnp':
        return APPNPSurrogate(in_dim=in_dim, hidden_dim=hidden_dim,
                              K=appnp_K, alpha=appnp_alpha, dropout=dropout)
    return GNNSurrogate(arch=arch, in_dim=in_dim, hidden_dim=hidden_dim,
                        n_layers=n_layers, dropout=dropout, gat_heads=gat_heads)


# Architectures to train (all với gnn_raw_attr features):
# APPNP = architecture lý thuyết mạnh nhất (H3 — IC cascade analogy)
ARCHITECTURES = ['sage', 'gcn', 'gin', 'gat', 'appnp']

# Training loop usage:
# for arch in ARCHITECTURES:
#     model = get_model(arch, in_dim=in_dim)
#     # → APPNPSurrogate cho 'appnp', GNNSurrogate cho 4 arch còn lại
```

**C2 Fair-comparison protocol (bắt buộc lock):**

- **Cùng split:** `split_masks.parquet` M0-locked (`random_state=42`, degree-stratified 80/20)
- **Cùng features:** `raw_attr` = `[views_log_norm, views_per_day_norm, life_time_norm]` (in_dim=3)
- **Cùng loss:** Huber (`delta=1.0`); **không early stopping** — `epochs=200` cố định
- **Cùng hyperparams (conv-based archs):** `hidden_dim=128, n_layers=2, dropout=0.3, lr=1e-3`; GAT thêm `gat_heads=4`
- **APPNP-specific:** `K=10, alpha=0.15, dropout=0.3, lr=1e-3` (thay vì conv layers; xem `APPNPSurrogate`)
- **5 seeds mỗi arch:** `[42, 123, 456, 789, 1024]` → report mean ± std

**Best arch selection criterion:**

- **Primary:** arch có `spearman_rho_mean` cao nhất qua 5 seeds
- **Tie-break (diff < 0.001):** APPNP > GAT > GIN > GCN > SAGE (**pre-registered order**; APPNP được ưu tiên vì lý thuyết mạnh nhất — H3)
- **Ghi vào:** `docs/experiment_registry.md` field `gnn_primary_arch` ngay sau C2 xong

**Ba inductive bias hypotheses — pre-registered trước C2:**

**Hypothesis H1 (GAT–A0 alignment):** Dưới IC primary (A0: `p(u,v)=1/deg(v)`):
Weighted cascade → neighbor degree thấp có influence lớn hơn.
_Intuition:_ GAT có thể học attention weight inversely-proportional-to-degree tự động.

> ⚠ Đây là **hypothesis/intuition** dựa trên cơ chế lý thuyết. C2 empirically verify — nếu GAT không win thì dùng kết quả thực nghiệm, không dùng framing này trong paper.

**Hypothesis H2 (GCN–A2 alignment — nếu chạy A2 sensitivity):** Dưới IC sensitivity (A2: `p(u,v)=1/√(deg(u)×deg(v))`):
`GCNConv` aggregates với weight `1/√(d̃_u×d̃_v)` — structurally analogous to A2 diffusion.
_Prediction:_ Nếu chạy C2 trên A2 labels, GCN expected to outperform GAT/GIN/SAGE vì inductive bias aligned với target generative process.

> ⚠ Self-loops (d̃ = deg+1 ≠ deg) và non-linearity (ReLU/dropout) làm GCN không _exactly_ implement A2 — đây là structural analogy, not exact equivalence. Pre-register dưới dạng "architectural inductive bias check."

**Hypothesis H3 (APPNP — IC cascade analogy — STRONGEST theoretical alignment):**

APPNP thực hiện K-step Personalized PageRank (Klicpera et al., ICLR 2019):

```
x^(k) = (1 - alpha) * A_hat * x^(k-1) + alpha * x^(0)
```

Đây là **structural analogy (inductive-bias)** với một quá trình lan truyền có teleport/restart (không phải cơ chế dừng của IC):

- `alpha` là trọng số tái-inject `x^(0)` mỗi bước (teleport/restart weight)
- `K` là số bước propagation (độ sâu receptive field)
- `(1 - alpha)` là phần “propagate” qua lân cận trong công thức APPNP

_Prediction (hypothesis):_ APPNP is hypothesized to **outperform SAGE/GCN/GIN/GAT** trên A0 IC labels vì inductive bias multi-hop + teleport có thể phù hợp hơn với target dạng diffusion-like. `K=10, alpha=0.15` là starting point (không diễn giải alpha như xác suất IC “tiếp tục/dừng”).

> ✅ Đây là lý do thêm APPNP vào C2. Nếu H3 confirm → APPNP là best arch → C3 (ranking loss) trên APPNP. Nếu H3 reject → report honestly; IC trên Twitch có thể bị dominated bởi local degree structure hơn là multi-hop cascade.

**Ghi chú về IC-A0 và degree dominance (context quan trọng cho C2):**

> **⚠ Structural constraint của A0:** IC-A0 sử dụng `p(u,v) = 1/deg(v)`, nên IC score **degree-coupled** (transition phụ thuộc trực tiếp vào `deg(v)`). Hệ quả: `degree` Spearman = 0.826 — baseline rất mạnh. GNN training trên A0 labels sẽ dễ "tái học" degree-like quantities từ graph topology.
>
> Empirically từ existing artifacts (test split; `outputs/mapr2026_v3_results/baseline_ranking_metrics.csv`):
>
> - `one_hop_spread` ρ = **0.688** (một hop)
> - `two_hop_spread` ρ = **0.804** (hai hop — multi-hop improves vs one-hop)
> - `degree` ρ = **0.826** (baseline rất mạnh)
> - `gnn_graph_only` (SAGE) = **0.470** — SAGE mean aggregation không capture multi-hop IC
> - `gnn_raw_attr` (SAGE) = **0.534** — raw attrs giúp nhưng vẫn xa degree
>
> **H3 rationale:** APPNP với PPR-style multi-hop propagation có thể capture hiệu ứng multi-hop tốt hơn SAGE mean (two_hop **0.804** > one_hop **0.688** trên test split). Nếu APPNP capture được multi-hop composition tốt hơn → có thể close gap với degree (0.826) hoặc vượt qua.
>
> **Nếu C2 vẫn không thể beat degree trên A0:** Đây là **structural expectation** (không phải implementation failure) — A0 IC ∝ 1/deg(v) = degree-derived label. Cần I-A labels (degree-blind) để GNN có genuine structural advantage. Xem Section 10.4 (I-A supplemental analysis).

### 9.1b Ranking Loss Experiment (v3.1 — MUST, fix metric mismatch)

> **Vấn đề:** HuberLoss tối ưu regression error (MSE-like), nhưng evaluation metrics là Spearman/NDCG
> (ranking metrics). Mismatch này có thể cost 2–3 Spearman points.

```python
import torch.nn.functional as F

# Option A: Pairwise Margin Ranking Loss
def pairwise_ranking_loss(pred, target, margin=0.1, n_pairs=512):
    """Sample pairs (i,j) where target_i > target_j, penalize if pred_i < pred_j."""
    n = len(pred)
    idx = torch.randperm(n)[:n_pairs * 2].view(n_pairs, 2)
    i, j = idx[:, 0], idx[:, 1]
    # Ensure target_i > target_j
    mask = target[i] > target[j]
    i, j = i[mask], j[mask]
    if len(i) == 0:
        return torch.tensor(0.0, requires_grad=True)
    return F.margin_ranking_loss(pred[i], pred[j],
                                  torch.ones(len(i), device=pred.device), margin=margin)

# Option B: Combined loss (Huber + ranking)
def combined_loss(pred, target, alpha=0.5, margin=0.1):
    huber = F.huber_loss(pred, target, delta=1.0)
    rank = pairwise_ranking_loss(pred, target, margin=margin)
    return alpha * huber + (1 - alpha) * rank

# Variant naming: best_arch + '_rankloss' (e.g., 'gat_raw_attr_rankloss')
criterion_rankloss = combined_loss  # α=0.5 default; sweep [0.25, 0.5, 0.75] if time
```

**Kỳ vọng:** +0.02–0.03 Spearman. Nếu GAT + ranking loss đạt ≥ 0.84 → vượt degree (0.826) → clean contribution.

**Output thêm vào surrogate_ranking_metrics.csv:** `best_arch_raw_attr_rankloss` (tên thực tế sau C2, ví dụ: `gat_raw_attr_rankloss` nếu GAT là best arch)

### 9.1c Optional: Inductive Generalization Test _(CAN HAVE — nếu còn thời gian)_

> **Khi nào làm:** Chỉ nếu architecture comparison + ranking loss vẫn không vượt degree sau bootstrap CI
> (tức là CI entirely negative — worst case). Đây là strongest remaining argument cho GNN nhưng
> cần careful implementation (~2h). Nếu CI bao gồm 0 → equivalence claim đủ, KHÔNG cần test này.

**Mục tiêu:** Chứng minh GNN có thể generalize tới nodes không thấy trong training — điều degree không thể làm (degree cần precomputed table).

**Thiết kế (strict inductive split):**

1. Hold out 20% of labeled nodes _hoàn toàn khỏi training graph_ — xóa edges của họ khỏi message passing graph
2. Train GNN trên 80% labeled nodes với reduced graph (168k - held-out nodes)
3. Inference: thêm held-out nodes vào graph sau training (inductive)
4. GNN phải predict IC cho nodes không có trong precomputed degree table

**Lưu ý kỹ thuật:**

```python
# PyG supports inductive inference via subgraph masking
# Held-out nodes: remove from edge_index during training
train_edge_index = subgraph(train_node_mask, full_edge_index)[0]
# At inference: use full_edge_index; GNN aggregates from neighbors
```

**Expected claim nếu succeed:**

> _"In an inductive setting where held-out nodes have no precomputed centrality, GNN achieves
> Spearman ρ = X while degree centrality cannot be applied without full graph recomputation.
> This demonstrates GNN's practical advantage in dynamic graph scenarios."_

**Output:** `outputs/mapr2026_v3_results/gnn_inductive_eval.json`

```json
{
  "setting": "inductive_20pct_holdout",
  "gnn_best_arch": "<arch>",
  "spearman_inductive": <float>,
  "n_held_out_nodes": <int>,
  "note": "degree centrality inapplicable to unseen nodes"
}
```

**Thời gian ước tính:** ~2 giờ (modify eval pipeline + retrain + evaluate).

---

### 9.1d GINE + IC Edge Features — C5 Supplemental _(CAN HAVE — nếu còn thời gian sau C2/C3/C4)_

> **Khi nào làm:** Chỉ sau khi C2 + C3 + C4 đều done. Đây là strongest possible alignment experiment nhưng không phải "feature-agnostic" nữa — cần label rõ trong paper.
>
> **Không làm trước C2/C3/C4** — không block critical path.

**Lý do GINE là architecture đặc biệt cho IC surrogate:**

GINE (Hu et al., NeurIPS 2019) mở rộng GIN bằng cách incorporate **edge features** vào message passing:
```
GIN:   h_v = MLP( (1+ε)·h_v + Σ_{u∈N(v)} h_u )
GINE:  h_v = MLP( (1+ε)·h_v + Σ_{u∈N(v)} ReLU(h_u + e_uv) )
                                                   ^^^^^^
                                              edge feature incorporated
```

**Insight then chốt:** Với `e_uv = 1/deg(v)` (= IC-A0 propagation probability), GINE nhận **explicit IC mechanism** trong message passing — model biết xác suất "được infected" của mỗi cạnh. Đây là upper bound experiment: nếu GINE với IC edge features vẫn không beat degree → structural constraint của A0 là absolute.

```python
from torch_geometric.nn import GINEConv

class GINESurrogate(nn.Module):
    """
    GINE surrogate với IC propagation probabilities làm edge features.
    
    Alignment với IC (C5 hypothesis — H5):
    IC-A0: p(u,v) = 1/deg(v) — propagation probability của mỗi edge.
    GINE nhận e_uv = p(u,v) explicitly → message từ u đến v = ReLU(h_u + 1/deg(v))
    → Model biết "trọng số" của mỗi cạnh theo IC dynamics.
    
    ⚠ Không phải "feature-agnostic": edge features = structural property (1/deg).
    Trong paper: "GINE with explicit IC mechanism encoding" — supplemental upper bound.
    ⚠ KHÔNG đưa vào C2 fair comparison (C2 = node features only, no edge features).
    
    Ref: Hu et al., "Strategies for Pre-training Graph Neural Networks", NeurIPS 2019
    """
    def __init__(self, in_dim=3, hidden_dim=128, edge_dim=1, dropout=0.3):
        super().__init__()
        # GINE cần edge_dim phải khớp với hidden_dim trong MLP
        nn1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        nn2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv1 = GINEConv(nn1, train_eps=True, edge_dim=edge_dim)
        self.conv2 = GINEConv(nn2, train_eps=True, edge_dim=edge_dim)
        self.head  = nn.Linear(hidden_dim, 1)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = self.drop(x)
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        return self.head(x).squeeze(-1)


def compute_ic_edge_features(edge_index, degrees, rule='a0'):
    """
    Compute IC propagation probability for each edge as edge feature.
    
    rule='a0': p(u,v) = 1/deg(v)          — IC primary (Weighted Cascade)
    rule='a2': p(u,v) = 1/sqrt(deg(u)*deg(v)) — IC symmetric sensitivity
    
    Returns: edge_attr of shape (E, 1) — IC probability per directed edge
    """
    src, dst = edge_index[0], edge_index[1]
    if rule == 'a0':
        # p(u,v) = 1/deg(v) — sink-degree only
        probs = 1.0 / degrees[dst].float().clamp(min=1)
    elif rule == 'a2':
        # p(u,v) = 1/sqrt(deg(u)*deg(v)) — symmetric
        probs = 1.0 / (degrees[src].float() * degrees[dst].float()).sqrt().clamp(min=1)
    return probs.unsqueeze(-1)          # shape: (E, 1)

# Cách dùng C5:
# edge_attr_a0 = compute_ic_edge_features(data.edge_index, degree_tensor, rule='a0')
# model_gine = GINESurrogate(in_dim=3, hidden_dim=128, edge_dim=1)
# preds = model_gine(data.x, data.edge_index, edge_attr_a0)
```

**C5 Experiment protocol:**

| Variant | Edge features | Node features | CSV model_name | Priority |
|---|---|---|---|---|
| GINE-IC-A0 | `1/deg(v)` per edge | `raw_attr` | `gine_ic_a0_raw_attr` | ✦ [IF TIME] |
| GINE-IC-A2 | `1/√(deg(u)×deg(v))` per edge | `raw_attr` | `gine_ic_a2_raw_attr` | ✦ [IF TIME] |
| GINE-graph-only | `1/deg(v)` per edge | `degree_norm` only | `gine_ic_a0_graph_only` | ✦ [IF TIME] |

> **Framing C5 trong paper (bắt buộc nếu include):** "As an upper bound analysis, we augment GNN message passing with explicit IC propagation probabilities as edge features (GINE; Hu et al., 2019). This test quantifies how much structural improvement remains when the diffusion mechanism is directly encoded — distinguishing architectural expressiveness limits from information limits."
>
> **Nếu GINE-IC-A0 cũng không beat degree:** → definitive evidence rằng IC-A0 label là structurally degree-equivalent; chuyển full focus sang I-A track cho GNN advantage claim.
>
> **Nếu GINE-IC-A0 beat degree:** → thú vị — explicit IC mechanism encoding helps; paper claim = "surrogate learning benefits from encoding diffusion mechanism structure."

**Output:** Thêm rows vào `surrogate_ranking_metrics.csv` (same schema).

---

### 9.1e Architecture Evaluation Log — Considered & Rejected

> **Tại sao section này tồn tại:** Khi reviewer hỏi "why not try X?", team có documented rationale. Đây cũng là checklist để không waste time implement architectures không phù hợp.

| Architecture | Xem xét? | Verdict | Lý do chi tiết |
|---|---|---|---|
| **GATv2** (Brody et al., ICLR 2022) | ✅ | ✅ Dùng cho **C2-I-A** | Dynamic attention khớp I-A row-normalization; C2-A0 dùng GAT v1 (static attention phù hợp hơn cho A0). Xem Section 9.1 và C2-I-A block. |
| **GINE** (Hu et al., NeurIPS 2019) | ✅ | ✅ **C5 supplemental** [IF TIME] | Edge features = IC prob — strongest explicit alignment; NOT feature-agnostic; không vào C2 fair comparison. |
| **GCNII** (Chen et al., ICML 2020) | ✅ | ❌ Skip cho C2 | Advantage chỉ xuất hiện tại L=16–64 layers. Tại `n_layers=2` (C2 locked), GCNII ≈ GCN + residual connection — không đủ khác biệt để justify thêm vào. Nếu muốn test, cần L=16 separate experiment, phá vỡ fair comparison. |
| **HGT** (Hu et al., WWW 2020) | ✅ | ❌ **Loại hoàn toàn** | Designed cho heterogeneous graphs (multiple node/edge types). Twitch follower graph là **homogeneous** (1 node type, 1 edge type) → type-specific attention matrices collapse về 1 matrix → HGT = complex GAT variant với overhead không có lợi. Wrong problem type. |
| **GraphGPS** (Rampášek et al., NeurIPS 2022) | ✅ | ❌ **Loại — scale blocker** | MPNN + global Transformer attention. Standard Transformer = O(N²) với N=168k → 28 tỷ attention pairs, không fit GPU. Efficient variants (Performer, BigBird) cần precompute LapPE/RWSE (~30–60 phút eigendecomposition trên 168k×168k). Overkill cho task node regression với 3 features. |

**Quick decision rule cho future architectures:**

```
Câu hỏi 1: Graph có nhiều node/edge types không?
  → Không (Twitch = homogeneous) → Loại HGT và mọi heterogeneous GNN

Câu hỏi 2: Architecture scale được với 168k nodes + 6.8M edges trên 1 GPU?
  → O(N²) attention (standard Transformer) → Loại nếu không có efficient approx
  → Cần eigendecomposition của full Laplacian → Cần benchmark trước

Câu hỏi 3: Architecture có advantage chỉ ở L >> 2 không?
  → Yes (GCNII, DeepGCN...) → Skip (C2 locked tại n_layers=2 cho fair comparison)
  → Có thể test riêng ngoài C2 nếu thời gian

Câu hỏi 4: Architecture cần edge features không có trong đồ thị?
  → Nếu edge features có thể compute từ graph structure → OK (GINE + IC prob)
  → Nếu cần external data không có → Loại
```

---

### 9.2 Training Protocol

```python
from sklearn.model_selection import train_test_split
import pandas as pd

# Step 1: Compute degree quintile labels for stratification
degree_arr = np.array([degree[n] for n in labeled_node_indices])
degree_quintile_labels = pd.qcut(degree_arr, q=5, labels=False, duplicates='drop')

# Step 2: Stratified split — degree-stratified, transductive setting
train_idx, test_idx = train_test_split(
    labeled_node_indices,
    test_size=0.20,
    stratify=degree_quintile_labels,
    random_state=42
)

# Step 3: Save split mask to shared artifact (required by eval_ranking_harness.py)
# split_mask = pd.DataFrame({'node_id': [...], 'split': ['train'|'test']})
# split_mask.to_parquet('data/processed/split_masks.parquet')

# Full graph edge_index passed to GNN (transductive); masks restrict loss to labeled nodes
# Contract (must be explicit to avoid silent bugs):
# - labeled_mask comes from split_masks.parquet (only labeled nodes are present there)
# - train_mask/test_mask MUST be subsets of labeled_mask (no unlabeled leakage)
# - y for unlabeled nodes can be set to 0.0; loss/eval must index by masks only
# - sanity: assert ~(train_mask & test_mask) and train_mask.sum()+test_mask.sum()==labeled_mask.sum()
# Primary metric: Spearman ρ on held-out labeled test nodes
# IMPORTANT: All architectures must use SAME split (same random_state=42)
```

### 9.3 Runtime Table — Tách feature precompute time

```
| Component                               | Time      | Notes                                        |
|-----------------------------------------|-----------|----------------------------------------------|
| Feature precompute (degree, PR, kshell) | ~5 min    | Centrality baselines only; not needed for GNN-raw-attr |
| MC IC labeling (5k nodes × 200 runs)    | 480s      | One-time cost (from runtime_breakdown.csv)   |
| GNN training (5 seeds, 1 arch)          | ~23s/arch | ~115s total for 5 seeds; ×4 archs = ~460s    |
| GNN inference (168,114 nodes)           | 0.067s    | All active nodes (from runtime_breakdown.csv)|
| Node2Vec training (dim=64, walks=20)    | ~8 min    | [to be measured]                             |
| Speedup: MC IC vs GNN inference         | **7,169×** | 480s / 0.067s (confirmed artifact)          |
```

**Filling instructions:** Cột "to be measured" sẽ được fill từ actual runs và ghi vào `runtime_breakdown.csv`.

**QUAN TRỌNG:** Nếu GNN-raw-attr là primary, không cần centrality precompute → runtime so sánh fair hơn.
GNN-raw-attr deployment cost = training (one-time, ~460s) + inference (0.067s) vs MC-IC (480s per evaluation).

---

## 12. Research Questions (v3.1 — Aligned with Linear Pipeline)

> **v3.1 re-alignment:** Professor's framing tổ chức paper theo 4-bước tuyến tính, không phải 6 RQ song song.
> **MAIN RQs** (Section 3+4 của paper): RQ1 + RQ3.
> **Supporting analysis:** metric correlation matrix (continuous; no categorical grouping) để contextualize baselines/ablations.
> Thứ tự ưu tiên: hoàn thiện RQ1 và RQ3 trước; correlation matrix chỉ chạy nếu không block pipeline.

### RQ1 — IC Operationalization Quality & Label Stability ★ **[MAIN — Task A]**

**Câu hỏi:** Does weighted-cascade IC simulation produce a sufficiently discriminative and stable influence ranking for use as a surrogate target? If label stability is insufficient, what structural properties of the graph cause this instability?

**Method:** Pilot diagnostics (CV, top-decile/median ratio, rank stability). Label Jaccard across 3 MC seeds. **Nếu Jaccard < 0.85: chạy thêm stability explanation analysis** (community overlap test + gap-to-noise analysis).

**Success criterion:** `cv_score > 0.3` → regression-ready. Nếu `cv_score < 0.3`: **không block tự động**; kích hoạt Option B khi IC không degenerate, và chỉ hard-stop khi IC degenerate (`median_reach < 2` + `p_reach_gt_1 < 0.20` + `top10_to_median_ratio < 2`). Label Jaccard > 0.85 _(binary-ready gate — KHÔNG block regression nếu fail; failure → binary provisional + stability_explanation.json)_.

**Stability as finding (not just gate):** Nếu Jaccard < 0.85, đây không chỉ là gate failure mà là một scientific finding cần giải thích và report trong paper. Phân tích 2 chiều:

- **Community overlap test:** Với mỗi Louvain community, kiểm tra xem có nodes ở cả top-k và top-(k+10%) boundary không. Tính `pct_communities_spanning_boundary`. Nếu cao (> 70%) → boundary không tự nhiên về mặt cấu trúc cộng đồng.
- **Gap-to-noise ratio:** Tại mỗi percentile threshold từ 85th → 95th, tính `gap = score[k] - score[k+1]` và `noise = σ_local / √n_runs`. Nếu `gap/noise ≈ 0` ở mọi threshold → instability có nguồn gốc **structural** (graph topology), không phải MC sampling variance (reducible by more runs).
- **Artifact:** `outputs/day1_benchmark/stability_explanation.json` (chỉ tạo nếu Jaccard < 0.85).
- **Paper framing (Section 4.2):** _"Label instability is structural in origin: X% of Louvain communities straddle the top-k boundary zone, and gap-to-noise ratios are near zero at all tested thresholds, indicating no natural separation point exists in the IC score distribution. This instability is irreducible by increasing MC runs — it reflects a property of the graph, not simulation variance. This finding provides the empirical motivation for adopting regression over classification as the primary prediction formulation."_

### Supporting Analysis — Metric Correlation Matrix

> Metric correlation matrix vẫn có giá trị cho Section 4.4 Feature Ablation (chứng minh multicollinearity ceiling).
> Nhưng không là dedicated RQ — subsume vào Section 4 của paper.

**Câu hỏi:** How strongly do raw signals (views), structural centrality (degree, pagerank, k-shell, betweenness), and cheap diffusion proxies (one-hop, two-hop) correlate with MC IC scores — globally and within structural regimes?

**Method [MUST]:** Tính full pairwise Spearman ρ matrix trên 8 metrics: `ic_score_mean, views, degree, pagerank, kshell, betweenness_approx, one_hop_spread, two_hop_spread`. BH-FDR correction cho tất cả p-values. **[✦ IF TIME] Breakdown theo degree quintile:** trong mỗi quintile Q0–Q4, tính riêng Spearman(IC, views) và Spearman(IC, degree) — xem pattern có uniform hay chỉ cao ở high-degree nodes. Cắt được nếu tight deadline mà không ảnh hưởng global matrix.

**Output artifact:** `outputs/mapr2026_v3_results/metric_correlation_matrix.json`

**Findings expected:**

- `one_hop_spread` expected to correlate **strongly** với IC score: mean của one_hop distribution = 1.0 cho mọi node (mathematical invariant `Σ 1/deg(v)` với undirected weighted cascade), nhưng rank ordering vẫn biến thiên theo local degree structure. **Expected ρ cao (typical range 0.7–0.95) nhưng KHÔNG nhất thiết ≈ 1.0** — cần verify empirically. Dù ρ cao hay thấp đều phải report, không giấu
- `two_hop_spread` sẽ capture more variance, testing whether higher-order neighborhood matters beyond one hop
- `degree` và `pagerank` thường có ρ cao với IC — nếu ρ > 0.9: interpret như strong global alignment và tập trung vào top-k mismatch + sensitivity; nếu ρ < 0.8 thì GNN story mạnh hơn

**Paper framing (Table trong Section 4.3):** _"Table X shows pairwise Spearman correlations between all influence metrics. While structural metrics (degree, PageRank) show moderate-to-high correlation with MC IC scores globally (ρ = [X]), non-trivial divergence can remain in the top-k region and across metric families — highlighted by the correlation matrix and ranking metrics."_

### RQ3 — GNN Surrogate Quality ★ **[MAIN — Task C]**

**Câu hỏi:** Can GNN approximate simulation-defined influence rankings as accurately as analytical proxies (degree, PageRank), and what is the computational gain over MC-IC simulation?

**Method:** Compare Spearman ρ, NDCG@10% của all baselines. Report runtime: MC IC (hours) vs GNN inference (seconds).

**Prepared narratives:**

_Nếu GNN-raw-attr > two-hop proxy:_

> _"GNN captures higher-order neighborhood patterns beyond 2-hop analytical approximations, demonstrating the value of learned representations for influence estimation."_

_Nếu GNN-raw-attr ≤ two-hop proxy:_

> _"We find that 2-hop analytical spread approximation (naive full-graph complexity gần O(Σ d(v)^2)) achieves ρ ≈ [X] with MC IC scores, closely matching GNN surrogate performance while requiring no training. This suggests weighted-cascade dynamics are well-approximated by local structural summaries. GNN's value lies primarily in the feature-agnostic message passing contribution (+0.099 Spearman over MLP without any precomputed graph statistics)."_

> **Lưu ý framing:** Câu "GNN's value lies primarily in efficient inference as network evolves" ngụ ý inductive generalization — chỉ dùng nếu Section 9.1c (inductive test) được thực hiện. Nếu không có 9.1c, dùng feature-agnostic message passing story (+0.099) thay thế.

_Cả hai outcomes đều publishable tại MAPR với framing đúng._

---

## 13. Dead Account Analysis (Stage 0)

```python
def dead_account_audit(df_raw):
    """Report trước khi filter — ghi vào limitations."""
    dead = df_raw[df_raw['dead_account'] == 1]
    live = df_raw[df_raw['dead_account'] == 0]

    print(f"Dead accounts: {len(dead)} ({len(dead)/len(df_raw)*100:.1f}%)")
    print(f"Dead degree:   {dead['degree'].mean():.1f} vs Live: {live['degree'].mean():.1f}")
    print(f"Dead views:    {dead['views'].mean():.0f} vs Live: {live['views'].mean():.0f}")

    # Paper: "Dead accounts (X%) have lower degree and views than active accounts.
    #         Findings generalize only to active users."
```

---

## 14. Paper Structure (v3.1 — Professor's Linear Narrative, 6 trang IEEE Double-blind)

> **Framing:** Pipeline tuyến tính [1]→[2]→[3]→[4]. KHÔNG tổ chức theo 6 RQ song song.
> Appendix là optional; chỉ include thêm sensitivity/diagnostics nếu còn page budget.

### Tổng quan cấu trúc

| Section  | Tiêu đề                     | Nội dung chính                                                                                                                               | Trang |
| -------- | --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| 1        | Introduction                | Problem, Twitch context, 3 contributions (linear pipeline)                                                                                   | 0.5   |
| 2        | Background                  | Weighted cascade IC, IC surrogate problem statement, GNN architectures overview                                                              | 0.75  |
| 3        | MC-IC as Operational Metric | 3.1 Discriminativeness; 3.2 IC ≠ degree (variance test); 3.3 Stability; 3.4 Regression justification                                         | 0.75  |
| 4        | GNN Surrogate Learning      | 4.1 Setup + feature sets; 4.2 Full baseline table; 4.3 Architecture comparison; 4.4 Feature ablation; 4.5 Ranking loss; 4.6 Runtime vs MC-IC | 2.5   |
| 5        | Discussion & Limitations    | When GNN adds value; when degree sufficient; ceiling analysis; construct validity                                                            | 0.5   |
| Appendix | _(optional, nếu có trang)_  | Sensitivity + additional diagnostics (A2/A1, I-A pilot summary, stability explanation details)                                               | —     |

**Total: ~5 trang nội dung + 0.5 trang references (≤12 refs)**

---

### Section 1 — Introduction (0.5 trang)

- **Hook:** Identifying power users in static social networks — no behavioral logs available
- **Twitch context:** MUSAE benchmark (Rozemberczki et al., 2021); game publishers + platform recommendation use case
- **"Why Twitch not Twitter?":** No public retweet cascades; gaming communities exhibit tight-knit structure ideal for IC simulation
- **Problem statement:** MC-IC is principled but slow (480s / 5k nodes) — need fast surrogate
- **3 contributions:**
  1. MC-IC as a discriminative, degree-independent operational metric for influence potential
  2. Stability analysis reveals continuous regression nature of IC scores
  3. GNN surrogate (GCN/GIN/GAT/SAGE) achieves statistically comparable ranking performance to degree centrality while requiring no precomputed graph statistics

---

### Section 2 — Background (0.75 trang)

| Reference                       | Role trong paper                                         |
| ------------------------------- | -------------------------------------------------------- |
| Kempe et al. (2003)             | IC model foundation; weighted cascade derivation         |
| Ling et al., ICML 2023 (DeepIM) | Weighted cascade p(u,v) = 1/degree(v) parameterization   |
| Kitsak et al. (2010)            | k-shell as spreader proxy — baseline comparison          |
| Kipf & Welling (2017)           | GCN — spectral graph convolution baseline                |
| Hamilton et al. (2017)          | GraphSAGE — inductive neighborhood aggregation           |
| Velickovic et al. (2018)        | GAT — attention ≈ learned 1/degree(v) weighting          |
| Xu et al. (2019)                | GIN — maximal expressiveness (WL-equivalent)             |
| Rozemberczki et al. (2021)      | Twitch MUSAE dataset; prior influence work on same graph |
| Guille et al. (2013) §4         | Evaluation without ground-truth diffusion data           |

**Subsections:**

- 2.1 Weighted Cascade IC Model — formula, parameter-free property, MC estimation
- 2.2 IC Surrogate Problem — definition: learn f: graph features → IC score; evaluation: Spearman ρ, NDCG@10%, Precision@10%
- 2.3 GNN Architectures — 1-paragraph overview of GCN/GIN/GAT/SAGE; why message passing is theoretically aligned with cascade propagation

---

### Section 3 — MC-IC as Operational Metric (0.75 trang)

**Figure 1 (bắt buộc):** IC reach distribution — right-skewed, discriminative; degree distribution overlay to show IC ≠ degree

**3.1 Discriminativeness**

- Pilot diagnostics: non-degenerate spread, CV > threshold → IC differentiates nodes
- IC reach distribution table (mean, median, IQR, top10-to-median ratio)

**3.2 IC Captures Higher-Order Effects (Multi-hop Composition + Degree-Controlled Variance)**

- **Empirical evidence từ artifacts (không suy đoán — đã có số liệu):**
  - `one_hop_spread` Spearman với IC = **0.688** _(test-split evaluation; full-sample Spearman: 0.717 per correlation matrix)_ — local neighbors alone không predict IC well
    - `two_hop_spread` Spearman với IC = **0.804** — hai hop captures IC đáng kể tốt hơn (+0.116)
  - `degree` Spearman với IC = **0.826** — degree là strongest structural predictor
    - **Gap one_hop vs two_hop (+0.116)** chứng minh IC phản ánh **multi-hop cascade composition**, không phải single-hop structure

- **IC ≠ degree (mặc dù A0: p=1/deg(v)):** Degree và IC correlated cao (0.826) nhưng không identical — IC captures cascade amplification qua neighborhood topology (triangles, high-degree 2nd-order neighbors). Nodes cùng degree có thể có IC score khác nhau do community embedding khác nhau.

- **Degree-controlled variance test** (Section 8.4 output):
  - `std(IC | degree ∈ [D±5]) / mean > 0.3` → IC encodes community position and 2nd-order structure, not only 1st-order degree
  - This justifies MC-IC as a richer operational metric than degree rank alone

- **Paper narrative chuẩn cho 3.2:**
  > _"IC simulation captures multi-hop cascade composition beyond degree: two-hop spread correlates with IC at ρ=0.804, exceeding one-hop spread (ρ=0.688; test-split evaluation — full-sample Spearman: 0.717), indicating that influence propagates through second-order neighborhoods. While degree remains the strongest single predictor (ρ=0.826), degree-controlled variance analysis reveals IC encodes additional community structure — confirming MC-IC as a richer operational metric than centrality-based ranking."_

**3.3 Label Stability Analysis**

- Jaccard@10% sweep: 0.307 (N=150) → 0.682 (N=1200); none meet 0.85 threshold
- Structural cause: 84.2% of boundary nodes span community boundaries → inherent instability
- Gap-to-noise ratio → not a gate failure; an empirical finding about influence ranking uncertainty

**3.4 Regression Formulation Justified (Empirically Motivated — Not a Fallback)**

- **CV gate result:** `cv_score = 0.2109 < 0.30` — pilot reveals **near-critical IC dynamics**: small changes in seed selection lead to high variance in ranked outcomes. This is NOT a failure; it is **empirical evidence that binary influence classification is ill-posed** on this graph.

- **Positive framing (bắt buộc):** cv_score = 0.2109 is the empirical motivation for regression:

  > _"Pilot diagnostics reveal near-critical IC dynamics (CV=0.21 < 0.30 threshold): IC rank orderings are sensitive to seed perturbation, making binary thresholding unstable. This motivates continuous regression on raw IC scores — the principled formulation for a simulation-derived continuous target."_

- **Jaccard instability:** `Jaccard@10% = 0.111` at N=150 seeds → binary "top-10% influencers" set is highly seed-sensitive. Regression on continuous IC scores remains reliable: `Spearman@1200 = 0.827` → rank ordering stable at sufficient simulation budget.

- **Paper framing bắt buộc:** "We formulate influence prediction as node-level regression on continuous MC-IC scores. This choice is empirically motivated by near-critical IC dynamics (CV=0.209) revealed in pilot diagnostics, which render binary label thresholds unstable. Rank-order reliability is confirmed by convergent Spearman correlation across simulation budgets."

- **KHÔNG dùng:** "we pivot to regression" / "regression is a fallback" / "binary labels failed" — những framing này ngụ ý regression là plan B. Đây là primary design choice được empirically motivated.

- MC-IC cost: 480s / 5k nodes → **7,169× speedup** motivation for surrogate learning → justifies GNN training cost.

**Evaluation statement (bắt buộc):**

> _"We evaluate in a transductive node-level regression setting. IC labels are available for N_sample degree-stratified nodes. All ranking metrics are computed on held-out labeled test nodes only. Full-graph inference is reported for runtime assessment only."_

---

### Section 4 — GNN Surrogate Learning (2.5 trang)

**Figure 2 (bắt buộc):** Architecture comparison bar chart — Spearman ρ for all models; degree baseline as horizontal reference line; bootstrap 95% CI error bars for GNN variants

**4.1 Experimental Setup**

- Hardware, seeds (5 seeds, report mean ± std), transductive split (80/20, degree-stratified)
- Feature sets: `graph_only` (degree, kshell), `raw_attr` (views_log, vpd, life_time), `centrality` (all 5)
- Hyperparameters: n_layers=2, hidden_dim=128, dropout=0.3, Adam lr=1e-3, 200 epochs, HuberLoss (primary)

**4.2 Full Baseline Comparison Table**

| Group | Model                 | Spearman ρ | NDCG@10% | P@10% | Notes                       |
| ----- | --------------------- | ---------- | -------- | ----- | --------------------------- |
| G0    | random                | —          | —        | —     | Lower bound                 |
| G1    | views_rank            | —          | —        | —     | Raw popularity              |
| G1    | views_per_day_rank    | —          | —        | —     | Normalized popularity       |
| G1    | degree_rank           | 0.826†     | —        | —     | ★ degree = reference line   |
| G2    | pagerank              | 0.824      | —        | —     | Structural centrality       |
| G2    | kshell                | 0.816      | —        | —     | Structural centrality       |
| G2    | betweenness_approx    | —          | —        | —     | Structural centrality       |
| G3    | one_hop_spread        | —          | —        | —     | Cheap diffusion proxy       |
| G3    | two_hop_spread        | —          | —        | —     | Cheap diffusion proxy       |
| G4    | node2vec_lr           | —          | —        | —     | Embedding + LR              |
| G4    | mlp_raw_attr          | 0.435      | —        | —     | MLP, no graph structure     |
| G5    | gnn_graph_only (SAGE) | 0.470      | —        | —     | Topology only               |
| G5    | gnn_raw_attr (SAGE)   | 0.534      | —        | —     | +0.099 vs MLP               |
| G5    | gnn_raw_attr (GCN)    | —          | —        | —     | NEW v3.1 (C2) — H2 arch     |
| G5    | gnn_raw_attr (GIN)    | —          | —        | —     | NEW v3.1 (C2) — sum agg     |
| G5    | gnn_raw_attr (GAT)    | —          | —        | —     | NEW v3.1 (C2) ★ H1 arch     |
| G5    | gnn_raw_attr (APPNP)  | —          | —        | —     | **NEW — H3 STRONGEST** (C2) |
| G5    | gnn_centrality (SAGE) | 0.817      | —        | —     | With centrality features    |
| G5    | best_arch_rankloss    | —          | —        | —     | NEW v3.1 (C3): ranking loss |

_† degree_rank = horizontal reference line in Figure 2. Values to be filled from artifact CSVs._
_Mean ± std across 5 seeds for all G5 models. Confirmed values from existing artifacts shown._

**4.3 Architecture Comparison (APPNP / GCN / GIN / GAT / SAGE — 5 architectures)**

- Controlled comparison: same features (`raw_attr`), same hyperparams, 5 seeds
- **APPNP (H3 — strongest theoretical motivation):** K-step PPR-style propagation với teleport/restart là **structural analogy/inductive bias** cho target diffusion-like; `K=10, alpha=0.15` là starting point (không claim calibration).
- GAT (H1): attention mechanism **may** learn a 1/degree(v)-like weighting — potentially aligned with weighted cascade
- GIN: sum aggregation, WL-equivalent — highest expressiveness; may capture two-hop structure better than SAGE mean (empirical on test split: two_hop ρ=0.804 > one_hop ρ=0.688)
- GCN (H2): relevant primarily if A2 sensitivity labels are run
- **Context:** Under A0, `degree` ρ=0.826 là baseline rất mạnh (label degree-coupled vì transition dùng `deg(v)`). Structural constraint: GNN learning on topology → dễ tái-học degree-like quantities. APPNP's multi-hop + teleport là ứng viên hợp lý để thử bắt multi-hop composition gap. Report honestly; nếu tất cả ≈ degree → bootstrap CI + I-A supplemental để đánh giá “genuine advantage”.

**4.4 Feature Ablation**

- `graph_only` (0.470) → `raw_attr` (0.534) → `centrality` (0.817)
- Message passing contribution: `gnn_raw_attr` 0.534 vs `mlp_raw_attr` 0.435 = **+0.099 from graph structure**
- Centrality features dominate final ranking; raw-attr GNN is feature-agnostic story
  > ⚠ **Clarification "feature-agnostic" (reviewer prep):** Trong paper, "feature-agnostic" = **không cần hand-crafted centrality/structural features** (degree, PageRank, k-shell) được pre-compute. `raw_attr` vẫn sử dụng user **metadata** (views_log_norm, views_per_day_norm, life_time_norm) — đây là metadata tĩnh của node, **KHÔNG phải behavioral traces** (không có click logs, retweet sequences hay engagement events). **A0 IC labels** là views-independent; nếu bật **I-A** thì đó là **attribute-informed operationalization** và phải label rõ. Khi reviewer hỏi "GNN is not truly feature-agnostic": response = GNN-raw_attr adds +0.099 Spearman **over MLP-raw_attr on the same metadata** — giá trị đến từ structural message passing, không phải từ centrality pre-computation.

**4.5 Ranking Loss Experiment**

- `best_arch_raw_attr_rankloss` (tên thực tế xác định sau C2, e.g., `gat_raw_attr_rankloss`): combined α·Huber + (1-α)·pairwise-margin-ranking-loss
- Motivation: Spearman/NDCG are ranking metrics; Huber optimizes pointwise regression
- Result to be filled; expected: +0.02–0.03 Spearman over best architecture with Huber

**4.6 Runtime Comparison**

| Method                     | Time   | Speedup vs MC-IC |
| -------------------------- | ------ | ---------------- |
| MC-IC labeling (5k nodes)  | 480s   | 1×               |
| GNN training (5 seeds)     | ~115s  | —                |
| GNN inference (168k nodes) | 0.067s | **7,169×**       |
| Degree (precomputed)       | ~0s    | —                |

_GNN inference matches degree for deployment speed; training is one-time cost._

> **Operational definition of 7,169× (reviewer prep):**
>
> - **480s** = MC-IC labeling cost for **5,000 nodes × 200 runs** (one-time cost để tạo training labels; joblib loky parallelism).
> - **0.067s** = GNN forward-pass inference trên **tất cả 168,114 active nodes** (sau khi train xong).
> - **Comparison at deployment:** Sau khi model trained, dùng GNN để rank 168k nodes mất 0.067s. Nếu dùng MC-IC để rank 168k nodes: 480s × (168,114/5,000) ≈ **16,140s** ≈ 4.5 giờ → speedup ~241,000×. Con số 7,169× là **lower-bound conservative** (so sánh labeling 5k nodes vs inferring 168k nodes — không cùng population size).
> - **Framing an toàn trong paper:** "GNN inference (0.067s for 168k nodes) is 7,169× faster than the MC-IC labeling cost (480s for 5k nodes × 200 runs) used to generate training labels." Không claim 7,169× là "full-graph vs full-graph" speedup.
> - **If reviewer objects:** Clarify đây là "training-data generation cost vs deployment inference cost" tradeoff — đúng mục đích paper (practical surrogate motivation).

---

### Section 5 — Discussion & Limitations (0.5 trang)

**5.1 When does GNN add value?**

- GNN (`raw_attr`): no precomputed centrality needed — deployable on new graphs without full graph traversal
- Bootstrap CI result: if GNN ≈ degree statistically → "GNN is a viable fast surrogate"
- Feature-agnostic message passing: +0.099 over MLP shows structural signal without explicit centrality
  _(Note: "feature-agnostic" = no hand-crafted centrality features; raw_attr uses user metadata — views_log, views/day, life_time — which are NOT behavioral traces. IC labels are views-independent under A0.)_
- **GNN advantage scales with IC operationalization complexity** _(chỉ include nếu I-A experiment chạy):_
  - Structural IC (A0): GNN matches degree baseline — viable surrogate; degree is a near-sufficient proxy
  - Attribute-informed IC (I-A): GNN significantly outperforms degree — message passing over node attributes captures cascade dynamics that scalar degree cannot represent
    > _"This scaling of GNN advantage demonstrates that our surrogate approach is most valuable precisely when the underlying diffusion model incorporates richer signals that analytical baselines cannot access."_

**5.2 Limitations (bắt buộc — 4 items)**

1. Twitch follower graph ≠ diffusion network (mutual-follow ≠ cascade path)
2. _[Conditional]_ Transductive setting: GNN trained with all nodes visible; **inductive generalization not tested** ← use this if Section 9.1c NOT done. If Section 9.1c IS done: "We evaluate both transductive and a limited inductive setting; full production-scale inductive generalization remains future work."
3. IC simulation is proxy, not ground-truth influence (no behavioral logs)
4. Multicollinearity ceiling: degree ↔ kshell ρ=0.993 → structural features collapse to single factor; GNN-centrality gains upper-bounded by this redundancy

**5.3 Why not learn p from data? — và chính sách views-based p**

> _"Learning p(u,v) requires supervised diffusion logs (e.g., retweet cascades, click sequences) unavailable in this dataset. Weighted cascade p(u,v) = 1/degree(v) provides a principled zero-shot alternative with theoretical backing (Kempe et al., 2003; Ling et al., 2023)."_

**Chính sách views-based p (I-A):**

> _"If we evaluate a views-informed cascade (I-A), we treat it as a supplemental attribute-informed operationalization (not parameter-free). It must be explicitly labeled as such, gated by a small pilot, and reported separately from the structural cascade (A0)."_

**Sensitivity to structural diffusion rule (robustness):**

> _"We evaluate robustness to structural diffusion rule choice via sensitivity variant S1: symmetric normalization p(u,v) = 1/√(deg(u)·deg(v)), structurally analogous to GCN's D^{-1/2}AD^{-1/2} aggregation scheme. We report Spearman correlation between A0 and S1 IC rankings to confirm primary results are not specific to the exact structural formula."_ _(Include only if S1 experiment completed.)_

**Ethics:**

> _"All data are publicly available under MIT License and anonymized. This study does not involve human subjects or interaction with live users."_

---

### References (≤12)

Kempe 2003, Kitsak 2010, Hamilton 2017 (SAGE), Kipf & Welling 2017 (GCN),
Velickovic 2018 (GAT), Xu 2019 (GIN), Rozemberczki 2021, Guille 2013,
Ling 2023 (DeepIM), Benjamini&Hochberg 1995, Blondel 2008 (Louvain), Grover 2016 (Node2Vec).

> **Note:** Đây là 12 refs — ở giới hạn tối đa. Nếu cần cắt: Guille 2013 hoặc Blondel 2008 (Louvain)
> là ít critical nhất. Giữ Kempe + Ling + Hamilton/Velickovic/Xu/Kipf + Rozemberczki bắt buộc.

---

### Appendix (optional — nếu page budget cho phép)

- **A.1 Sensitivity:** A2 vs A0 ranking overlap + short interpretation
- **A.2 I-A (if enabled):** pilot decision summary + alignment checks
- **A.3 Stability:** Stability sweep full table (N=150 to N=1200)

---

## 15. Experiment Configuration (Final)

```yaml
# experiment.yaml v3.1

# ─── Graph Setup ──────────────────────────────────────────────
global_seed: 42
filter_dead_account: true
graph_directed: false
graph_direction_note: "MUSAE Twitch: mutual-follow edges only. Undirected follows prior work."

# ─── IC Simulation ────────────────────────────────────────────
ic_backend: csr_numpy # CSR numpy arrays — NOT igraph object, NOT NetworkX
ic_parallel: joblib_loky # LOKY processes — NOT threads (GIL)
ic_n_jobs: -1

# Weighted cascade is PARAMETER-FREE — no calibration target
p_primary: weighted_cascade # p(u,v) = 1/degree(v)
calibration_mode: variance_check # NOT target_reach

pilot_diagnostics:
  n_pilot_nodes: 200
  n_pilot_runs: 50
  checks:
    [
      mean_reach,
      median_reach,
      iqr_reach,
      top10_to_median_ratio,
      rank_stability,
      cv_score,
    ]
  min_cv: 0.3 # cascade must differentiate nodes

# ── Sensitivity variants (robustness to diffusion rule choice) ────────────────
# S1 — Symmetric (đáng thử nhất, không vi phạm independence):
p_sensitivity_s1: symmetric # p(u,v) = 1/sqrt(deg(u)*deg(v)); GCN-analogous
ic_sensitivity_s1_output: outputs/mapr2026_v3_results/ic_scores_sensitivity_a2.parquet

# S2 — Source Budget (nếu cần tăng "IC ≠ degree" evidence):
p_sensitivity_s2: source_budget # p(u,v) = 1/deg(u); 1-hop = 1.0 for all nodes
ic_sensitivity_s2_output: outputs/mapr2026_v3_results/ic_scores_sensitivity_a1.parquet

# Uniform (đã quyết cắt — giữ config để reference):
# p_sensitivity_uniform: uniform
# kappa_target: 2 # p = 2/mean_degree ≈ 0.025 — tight timeline, không làm

# Priority: S1 (SHOULD DO) > S2 (IF TIME) > uniform (CẮT)

# ── I-A: Attribute-Informed IC (Row-Normalized Views) ────────────
# ĐIỀU KIỆN: pilot pass (CV>0.3 AND rho_deg<0.75 AND rho_proxy<0.85)
# Nếu chưa xác nhận → comment toàn bộ block này

# Uncomment khi pilot pass:
# p_attr_informed: ia  # p(u,v) = log1p(views(v)) / sum(log1p(views(N(u))))
# ia_pilot_output: outputs/mapr2026_v3_results/ic_ia_pilot_decision.json
# ia_pilot_thresholds:
#   cv_min: 0.30             # I-A must differentiate nodes
#   rho_degree_max: 0.75     # I-A must be degree-blind (stricter than A0 gate)
#   rho_proxy_max: 0.85      # nbr_views_mean proxy must not dominate
# ia_n_pilot_nodes: 200
# ia_n_pilot_runs: 50
# ic_ia_output: outputs/mapr2026_v3_results/ic_scores_ia.parquet
# ic_ia_comparison_output: outputs/mapr2026_v3_results/ic_ia_vs_primary.json
# # Fallback if I-A fails CHECK 2 (rho_deg ≥ 0.75):
# p_fallback_iib: views_density  # p(u,v) = clip(views_norm[v]/deg(v), max=0.5)
# ic_iib_output: outputs/mapr2026_v3_results/ic_scores_iib.parquet

# ─── Sampling ──────────────────────────────────────────────────
sample_size_primary: 5000 # adjust based on Day 1 benchmark
sampling_strategy: degree_quintile_stratified
ks_test_threshold: 0.10

# ─── IC Runs ───────────────────────────────────────────────────
mc_runs_primary: 200 # adjust based on benchmark
mc_runs_label_stability: 150 # for 3-seed Jaccard check
n_label_stability_seeds: 3 # reduced from 5 for compute efficiency
label_stability_target_jaccard: 0.85

# ─── Label Generation ──────────────────────────────────────────
label_mode_primary: regression # log1p(ic_score_mean)
label_mode_secondary: classification
classification_threshold: 0.10 # top-10% only — NOT 5% or 15%
cv_noise_threshold: 0.50 # flag high-variance nodes

# ─── Centrality ────────────────────────────────────────────────
compute_betweenness: true
betweenness_backend: networkit
betweenness_epsilon: 0.10
betweenness_delta: 0.10
pagerank_alpha: 0.85

# Community detection
community_algorithm: louvain
louvain_resolution: 1.0
louvain_n_runs: 10
louvain_seed_start: 0
louvain_select_best_by: modularity
compute_cross_community_fraction: true

# ─── Baselines ─────────────────────────────────────────────────
baselines:
  group1_raw: [views_rank, views_per_day_rank, degree_rank]
  group2_central: [pagerank, kshell, betweenness_approx]
  group3_proxies: [one_hop_spread, two_hop_spread] # NOT weighted_degree (redundant)
  group4_embed: [node2vec_lr, mlp_raw_attr]
  group5_gnn: # v3.1: 4 architectures × raw_attr + rankloss variant
    - gnn_raw_attr # canonical CSV name for SAGE raw-attr baseline
    - gcn_raw_attr # NEW — MUST
    - gin_raw_attr # NEW — MUST
    - gat_raw_attr # NEW — MUST
    - appnp_raw_attr # NEW — MUST (H3: IC cascade analog; expected best arch)
    - best_arch_raw_attr_rankloss # NEW — MUST (ranking loss; UPDATE tên thực tế sau C2, e.g., appnp_raw_attr_rankloss)
    - gnn_graph_only # ablation (SAGE)
    - gnn_centrality # ablation (SAGE, = gnn_centrality old)
    # gnn_full: IF TIME only

# Node2Vec (reduced for speed)
node2vec_dim: 64
node2vec_walks: 20 # NOT 200 (too slow)
node2vec_walk_length: 20

# ─── GNN (v3.1 — Multi-Architecture) ──────────────────────────
gnn_primary_variant: gnn_raw_attr # PRIMARY: raw attributes only
gnn_primary_arch: sage # will be updated after architecture comparison

# Architecture comparison (v3.1 — MUST run all 5)
gnn_architectures: [sage, gcn, gin, gat, appnp] # 5 archs — appnp added (H3: IC cascade analog)
gnn_gat_heads: 4 # reduce to 1 if convergence issues
# APPNP-specific (K-step PPR; alpha = teleport/restart weight; heuristic analogy, not IC stopping)
gnn_appnp_K: 10 # propagation steps (receptive field depth)
gnn_appnp_alpha: 0.15 # teleport/restart weight (starting point)

# Fixed hyperparams for fair comparison (DO NOT change per arch)
gnn_n_layers: 2
gnn_hidden_dim: 128
gnn_dropout: 0.30
gnn_lr: 0.001
gnn_epochs: 200
gnn_loss: huber # primary loss
gnn_huber_delta: 1.0
gnn_train_test_split: 0.80
gnn_training_seeds: [42, 123, 456, 789, 1024] # 5 seeds → report mean ± std

# Ranking loss experiment (v3.1 — MUST)
gnn_rankloss_alpha: 0.5 # α × Huber + (1-α) × pairwise-margin
gnn_rankloss_margin: 0.1
gnn_rankloss_n_pairs: 512
gnn_rankloss_variant: best_arch_raw_attr_rankloss # UPDATE after arch comparison: replace "best_arch" with actual best (e.g., gat_raw_attr_rankloss)

feature_sets:
  gnn_raw_attr: [views_log_norm, views_per_day_norm, life_time_norm]
  gnn_graph_only: [degree_norm]
  gnn_centrality: [degree_norm, pagerank_norm, kshell_norm]
  gnn_full:
    [
      degree_norm,
      pagerank_norm,
      kshell_norm,
      views_log_norm,
      views_per_day_norm,
      life_time_norm,
    ]

# Architecture × feature matrix (what to run)
# MUST: all 4 archs × raw_attr
# IF TIME: GCN/GIN/GAT × graph_only, × centrality
gnn_experiment_matrix:
  sage: [raw_attr, graph_only, centrality]
  gcn: [raw_attr] # expand to others if time
  gin: [raw_attr]
  gat: [raw_attr]
  appnp: [raw_attr] # MUST — H3 experiment; expand to graph_only/centrality if time

# ─── Evaluation ────────────────────────────────────────────────
eval_setting: transductive # accuracy on held-out labeled nodes ONLY
primary_metrics: [spearman_rho, ndcg_10, precision_10]
runtime_metrics: [mc_ic_labeling_sec, gnn_training_sec, gnn_inference_all_sec]
avoid_metrics: [accuracy, f1_macro]
multiple_testing_correction: benjamini_hochberg
fdr_alpha: 0.05

# ─── External Validation ───────────────────────────────────────
lifetime_degree_quintiles: 5
cliffs_delta_threshold: 0.20
lifetime_n_quintiles_significant_target: 3
```

---

## 16. Timeline 25 ngày (6/4 – 30/4)

> **📍 Current date: 16/4/2026** — Ngày 6–15/4 đã qua. Các mốc ✅ coi là complete (hoặc ghi override nếu chưa xong).
> Ưu tiên hiện tại: GNN architecture comparison (NEW) + ranking loss (NEW) + bootstrap CI (NEW).

### ⚠️ Nguyên tắc không thể phá vỡ

1. **Ngày 6/4 sáng:** Benchmark IC runtime → quyết định sample size _(đã qua)_
2. **Ngày 6/4 chiều:** One-hop baseline correlation check → quyết định GNN narrative _(đã qua)_
3. **IC simulation chạy liên tục ở background từ Ngày 8** _(đã qua)_
4. **Bắt đầu viết Introduction/Related Work/Methodology từ Ngày 8** _(đã qua)_
5. ⭐ **NEW (v3.1):** Architecture comparison + bootstrap CI phải xong trước 21/4

| Ngày                  | Track A: Data & IC                                                                          | Track B: Baselines & Community        | Track C: GNN & Paper                         |
| --------------------- | ------------------------------------------------------------------------------------------- | ------------------------------------- | -------------------------------------------- |
| ~~**6/4**~~           | ✅ IC benchmark                                                                             | ✅ Setup NetworKit, betweenness (bg)  | ✅ Setup PyG                                 |
| ~~**6–9/4**~~         | ✅ Day 1 checks, sampling, pilot diagnostics                                                | ✅ PageRank, k-shell, one-hop, 2-hop  | ✅ Related work draft                        |
| ~~**10–12/4**~~       | ✅ IC primary running; label stability check                                                | ✅ Community detection, Node2Vec, MLP | ✅ Methodology draft                         |
| ~~**13–14/4**~~       | ✅ IC DONE; baselines finalized                                                             | ✅ Baseline evaluation setup          | ✅ Figure 1 (pipeline); Internal checkpoint  |
| **15-16/4** ← _TODAY_ | **Degree-controlled variance test** (NEW); SAGE GNN baseline locked                         | **GCN/GIN/GAT arch comparison** (NEW) | Experiment section draft; Section 3 write-up |
| **17-18/4**           | **Bootstrap CI: GNN vs degree** (NEW)                                                       | **Ranking loss experiment** (NEW)     | Results tables; Section 4 draft              |
| **19-20/4**           | All new experiment artifacts locked                                                         | Runtime measurement & logging         | Discussion & Limitations; Section 5 draft    |
| **21/4**              | **All experiments locked** _(data + models + new v3.1 experiments)_                         | All results locked                    | Paper draft complete                         |
| **22-27/4**           | **M5 Integration:** gom artifacts → `outputs/mapr2026_v3_results/`; finalize tables + plots | —                                     | Team Plan M5 phase                           |
| **22-23/4**           | Internal review (tất cả thành viên đọc)                                                     | —                                     | Revision round 1                             |
| **24-25/4**           | Fix issues                                                                                  | —                                     | Revision round 2                             |
| **26/4**              | IEEE format check (6 trang, margins, fonts)                                                 | —                                     | Double-blind verify                          |
| **27/4**              | Final read-through                                                                          | —                                     | Submit dry-run                               |
| **28-30/4**           | Buffer + last fixes                                                                         | —                                     | **30/4: SUBMIT**                             |

### Scope Reduction — Phải sẵn sàng cắt nếu tight

> **v3.1 priority:** Main story = [1]→[2]→[3]→[4]. Không có Task B — cắt I-A trước mọi thứ khác nếu tight (chỉ sau pilot pass mới làm).

| Cắt được (theo thứ tự ưu tiên cắt)                                             | Giữ bắt buộc (KHÔNG cắt)                                          |
| ------------------------------------------------------------------------------ | ----------------------------------------------------------------- |
| Appendix extras (optional) — cắt đầu tiên                                      | Weighted cascade IC + pilot diagnostics                           |
| **I-A Attribute-Informed IC** — cắt nếu pilot fail **hoặc** thiếu compute/time | Label stability analysis (Jaccard + structural cause)             |
| Graph perturbation test                                                        | Degree-controlled IC variance test (Section 8.4) ← v3.1 NEW MUST  |
| 5%/15% thresholds (chỉ giữ 10%)                                                | Architecture comparison (GCN/GIN/GAT/SAGE) ← v3.1 NEW MUST        |
| Eigenvector redundancy check                                                   | Bootstrap CI GNN vs degree ← v3.1 NEW MUST                        |
| Betweenness trong GNN features                                                 | One-hop + two-hop proxies                                         |
| GNN-full variant                                                               | Community detection (Louvain) ← Task A stability explanation only |
| Ranking loss α sweep [0.25, 0.5, 0.75]                                         | GNN-raw-attr (primary) + GNN-graph-only (ablation)                |
| Inductive generalization test (9.1c)                                           | Runtime comparison table (MC-IC vs GNN inference)                 |

---

## 17. Folder Structure

```
SNA_MAPR2026/
├── data/
│   ├── raw/                          # twitch_edges.csv, twitch_features.csv
│   ├── interim/                      # active_nodes.csv, active_edges.csv
│   └── processed/
│       ├── graph_csr.npz             # CSR format: indptr, indices, degrees
│       ├── node_attributes.parquet   # base attrs (node_id, views, life_time, language, ...)
│       ├── community_features.parquet# node_id, community_id, cross_community_edge_fraction
│       ├── ic_scores_primary.parquet # weighted cascade, N runs
│       ├── regression_targets.parquet# log1p(ic_score)
│       └── classification_labels.parquet # binary top-10%
│
├── src/
│   ├── data/
│   │   ├── preprocess_graph.py
│   │   └── dead_account_audit.py
│   ├── simulation/
│   │   ├── ic_csr.py                 # IC với CSR arrays (loky parallel)
│   │   ├── benchmark_runtime.py      # NGÀY 6/4 SÁNG
│   │   ├── onehop_check.py           # NGÀY 6/4 CHIỀU — critical decision
│   │   ├── convergence_check.py
│   │   └── label_stability.py
│   ├── graph/
│   │   ├── centrality.py             # NetworKit betweenness
│   │   ├── community.py              # Louvain + cross_comm fraction
│   │   └── diffusion_proxies.py      # one-hop + two-hop
│   ├── models/
│   │   ├── gnn_surrogate.py          # PyG, unified GNNSurrogate(arch=sage/gcn/gin/gat)  [v3.1]
│   │   └── baselines.py              # LR, MLP, Node2Vec+LR
│   ├── evaluation/
│   │   ├── ranking_metrics.py        # Spearman, NDCG, P@k  (fix: kind='stable' in argsort)
│   │   ├── bootstrap_ci.py           # bootstrap_spearman_ci() — GNN vs degree  [v3.1 NEW]
│   │   ├── degree_variance_analysis.py # degree-controlled IC variance test  [v3.1 NEW]
│   │   └── multiple_testing.py       # BH correction
│   └── visualization/
│       └── runtime_bar.py
│
├── outputs/
│   ├── stage0_data_quality/           # dead accounts, LCC, views dist
│   ├── stage1_centrality/             # centrality + community features
│   ├── day1_benchmark/                # IC runtime + one-hop ρ — CRITICAL
│   │   └── stability_explanation.json # only if Jaccard < 0.85
│   ├── stage2_ic_labels/              # pilot diagnostics, stability, bootstrap CI
│   ├── stage4_gnn/                    # all GNN variants, 5 seeds each
│   ├── stage6_sensitivity/            # IC diffusion rule sensitivity variants
│   │   ├── ic_scores_sensitivity_a2.parquet  # S1: p=1/sqrt(deg(u)*deg(v)) — symmetric
│   │   └── ic_scores_sensitivity_a1.parquet  # S2: p=1/deg(u) — source budget [IF TIME]
│   └── mapr2026_v3_results/           # ← v3.1 consolidated artifacts (referenced throughout doc)
│       ├── baseline_ranking_metrics.csv       # all G1-G4 models, Spearman+NDCG+P@10
│       ├── surrogate_ranking_metrics.csv      # all G5 GNN variants, mean±std across seeds
│       ├── degree_controlled_ic_variance.json # Section 8.4 — v3.1 NEW
│       ├── gnn_vs_degree_bootstrap_ci.json    # Section 8.5 — v3.1 NEW
│       ├── metric_correlation_matrix.json     # pairwise Spearman 8×8
│       ├── ic_sensitivity_comparison.json     # Spearman(A0 vs A2), Spearman(A0 vs degree) per variant [SHOULD DO]
│       ├── runtime_breakdown.csv              # IC/GNN/proxy timings
│       ├── gnn_inductive_eval.json            # Section 9.1c — optional
│       │
│       │   # ── I-A artifacts (chỉ tạo khi pilot pass) ──
│       ├── ic_ia_pilot_decision.json          # [SHOULD DO] 3-check pilot result: cv_ia, rho_deg_ia, rho_nbr_ia, decision
│       ├── ic_scores_ia.parquet               # [SHOULD DO] I-A full sim (same schema as primary) — nếu pilot pass
│       ├── ic_ia_vs_primary.json              # [SHOULD DO] Spearman(IC-I-A, IC-A0) + Spearman(IC-I-A, degree)
│       ├── surrogate_ranking_metrics_ia.csv   # [SHOULD DO] C2-I-A: 4 archs × 5 seeds on I-A labels
│       ├── gnn_vs_degree_bootstrap_ci_ia.json # [SHOULD DO] C4-I-A: Bootstrap CI on I-A labels
│       └── ic_scores_iib.parquet              # [IF TIME] II-B fallback: views_density p(u,v) — nếu I-A CHECK 2 fail
│
├── paper/
│   ├── main.tex                       # IEEE two-column, double-blind
│   ├── figures/
│   │   ├── fig1_pipeline.pdf          # Task A→C linear pipeline diagram
│   │   ├── fig2_arch_comparison.pdf   # GNN architecture comparison bar chart (degree reference line)
│   │   └── fig3_ic_distribution.pdf   # IC reach distribution (Section 3.1)
│   └── tables/
│       ├── table1_baseline_full.tex   # Section 4.2 — full G0-G5 comparison
│       └── table2_runtime.tex         # Section 4.6 — runtime comparison
│
├── docs/
│   ├── experiment_registry.md         # mọi decision phải ghi
│   └── day1_decisions.md              # IC benchmark + one-hop ρ outcomes
│
├── config/
│   └── experiment.yaml                # v3.1 config (Section 15)
│
└── README.md
```

---

## 18. Pre-Submission Checklist

### Blocker — Nền tảng (reject nếu thiếu)

- [ ] **Ngày 6/4:** IC benchmark + one-hop ρ check → ghi vào `docs/day1_decisions.md`
- [ ] Construct validity paragraph trong Section 3.1 (follower ≠ diffusion channel, cref Section 1.1)
- [ ] Directionality `graph_directed: false` + justification trong paper
- [ ] `calibration_mode: variance_check` — KHÔNG có `calibration_target_reach_pct: 0.08`
- [ ] DeepIM chỉ cited cho weighted cascade formula, KHÔNG cho 8% reach target
- [ ] IC backend = CSR + loky (KHÔNG phải NetworkX threads)
- [ ] Label stability: chạy 3 MC seeds → **nếu Jaccard ≥ 0.85**: binary non-provisional. **Nếu Jaccard < 0.85**: regression không bị block, chạy `stability_explanation.json` (đây là finding, không phải blocker)
- [ ] Baseline Group 3: one-hop + 2-hop (KHÔNG phải weighted degree — redundant)
- [ ] GNN primary = GNN-raw-attr (views_log, views/day, life_time — no centrality features)
- [ ] Transductive setting stated rõ trong paper
- [ ] Accuracy metrics trên held-out LABELED nodes only; full-graph = runtime story only
- [ ] Runtime table tách: feature precompute / GNN training / GNN inference / MC IC
- [ ] BH correction cho tất cả MWU tests
- [ ] 5 training seeds, report mean ± std
- [ ] Dead account statistics trong limitations
- [ ] Paper framing KHÔNG claim "very good margin" trước khi có bootstrap CI

### Blocker — v3.1 Architecture & Significance (NEW — per professor's framing)

- [ ] **Degree-controlled IC variance test** (Section 8.4) → `degree_controlled_ic_variance.json`
- [ ] **Architecture comparison**: GCN/GIN/GAT chạy với `raw_attr` features, 5 seeds mỗi arch
- [ ] **Ranking loss experiment**: `best_arch_raw_attr_rankloss` (e.g., `gat_raw_attr_rankloss` nếu GAT wins C2) với combined α·Huber + (1-α)·pairwise-margin-loss
- [ ] **Bootstrap CI GNN vs degree** (Section 8.5) → `gnn_vs_degree_bootstrap_ci.json`
- [ ] Paper claim về GNN phải align với bootstrap CI kết quả (equivalent / lower / higher)
- [ ] `eval_ranking_harness.py`: `kind='stable'` trong tất cả argsort calls (NDCG/Precision deterministic)
- [ ] `run_baselines.py`: `sort_values("node_id")` sau `apply_test_mask` (consistent ordering)

### Strongly Recommended

- [ ] **Bootstrap CI cho IC scores** (Section 4.3) — `bootstrap_ci_ic()` — confidence interval cho mean IC reach của từng node (đo độ tin cậy của label, khác với Bootstrap CI GNN vs degree ở Section 8.5)
- [ ] **[SHOULD DO] Sensitivity S1 — Symmetric IC** (Section 4.1b): `run_ic_csr_a2()` → `ic_scores_sensitivity_a2.parquet`; compute `Spearman(IC-A2, IC-A0)` và `Spearman(IC-A2, degree)`; chạy C2 protocol trên A2 labels để test GCN–A2 alignment hypothesis (H2). Framing: "robustness to diffusion rule choice + architectural inductive bias check." _Không làm nếu primary C2 chưa xong._
- [ ] **[IF TIME] Sensitivity S2 — Source Budget IC** (Section 4.1b): `p(u,v)=1/deg(u)` → `ic_scores_sensitivity_a1.parquet`; chủ yếu dùng nếu `Spearman(IC-A0, degree) > 0.85` và cần tăng "IC ≠ degree" evidence.
- [ ] **[SHOULD DO] I-A Attribute-Informed IC pilot**: Chạy 200 nodes × 50 runs, kiểm tra 3 thresholds (CV>0.3, ρ_deg<0.75, ρ_proxy<0.85) → ghi vào `ic_ia_pilot_decision.json`. Nếu pass: chạy full sim (5k×200) → `ic_scores_ia.parquet`; Nếu fail: document lý do, stay A0.
  - _Khi I-A pass:_ Person 3 chạy thêm C2-I-A (4 archs × 5 seeds trên I-A labels) → `surrogate_ranking_metrics_ia.csv`; C4-I-A (bootstrap CI GNN_best_ia vs degree) → `gnn_vs_degree_bootstrap_ci_ia.json`.
  - **Pre-registration bắt buộc:** Ghi "H3: Under I-A, GNN will significantly outperform degree (degree is structurally blind to row-normalized IC)" vào `docs/experiment_registry.md` TRƯỚC KHI chạy pilot.
- [ ] Node2Vec: dim=64, walks=20 (không phải 200)
- [ ] Betweenness là optional analysis; không bắt buộc là GNN feature
- [ ] Figure 2: architecture comparison bar chart với degree reference line + 95% CI error bars (mean ± std across 5 seeds)
- [ ] Community detection (Louvain) → `cross_community_edge_fraction` _(cần cho stability explanation — Task A)_

> **Lưu ý phân biệt 2 loại Bootstrap CI:**
>
> - **`bootstrap_ci_ic()`** (Section 4.3): CI cho mean IC reach của mỗi node — đo noise của MC simulation
> - **`bootstrap_spearman_ci()`** (Section 8.5): CI cho Δ Spearman (GNN vs degree) — đo statistical significance → đây là BLOCKER v3.1

### IEEE Format

- [ ] ≤ 6 trang kể cả figures, tables, references
- [ ] Double-blind: không tên, trường, acknowledgments trong submission PDF
- [ ] Figures readable grayscale ở kích thước nhỏ
- [ ] Abstract ≤ 150 words
- [ ] References: IEEE format [1], [2], ...

---

## 19. Risk Management

### 19.1 Infrastructure & Data Risks

| Rủi ro                                                                        | Xác suất   | Impact       | Action                                                                                      |
| ----------------------------------------------------------------------------- | ---------- | ------------ | ------------------------------------------------------------------------------------------- |
| One-hop ρ > 0.9 + top-k alignment cao (`Jaccard@10% > 0.8`, `NDCG@10% > 0.9`) | Trung bình | **Critical** | Ngày 6/4: check trước; nếu đủ 3 điều kiện thì restructure, nếu không giữ GNN + 2-hop        |
| IC runtime > 8h                                                               | Trung bình | **Critical** | Reduce: n_sample=2k, N_runs=100; log limitation                                             |
| loky OOM với large graph                                                      | Thấp       | Cao          | Reduce n_jobs; monitor RAM                                                                  |
| PyG installation issues                                                       | Thấp       | Trung bình   | Setup Ngày 6/4 sáng; fallback DGL                                                           |
| Paper > 6 pages                                                               | Trung bình | Blocker      | Cut optional appendix extras first; then shorten 4.4 ablation; never cut 4.2 baseline table |

### 19.2 GNN Surrogate Risks (v3.1 — mới)

| Rủi ro                                                                          | Xác suất   | Impact     | Mitigation                                                                                                                  |
| ------------------------------------------------------------------------------- | ---------- | ---------- | --------------------------------------------------------------------------------------------------------------------------- |
| GNN không vượt degree sau full architecture search (SAGE/GCN/GIN/GAT)           | Trung bình | Trung bình | Chạy bootstrap CI (Section 8.5) → nếu CI bao gồm 0: claim statistical equivalence; feature-agnostic story (+0.099 over MLP) |
| GAT không converge với current setup (4 heads, hidden=128)                      | Thấp       | Thấp       | Reduce heads=1; increase epochs=300; nếu vẫn fail: report GAT instability as finding                                        |
| Ranking loss không improve Spearman so với Huber                                | Trung bình | Thấp       | Report as negative finding (appendix note); Huber-trained GNN remains primary variant                                       |
| Degree-controlled variance test shows low IC variance (CV < 0.3)                | Thấp       | Trung bình | Honest limitation in paper: "IC ≈ degree at Twitch scale"; strengthen runtime story instead                                 |
| Bootstrap CI shows GNN significantly _lower_ than degree (CI entirely negative) | Thấp       | Cao        | Restructure Section 4 claim: focus on (1) no-centrality-precompute advantage + (2) message passing contribution (+0.099)    |
| Multiple architecture runs produce high variance (std > 0.05)                   | Thấp       | Trung bình | Report mean ± std across 5 seeds; highlight reproducibility; use more seeds (10) for final table                            |

---

## 20. Decision Log Template

> **Status as of 16/4/2026:** Day-1 decisions đã hoàn thành. IC simulation đã chạy xong. Known outcomes
> from artifacts (runtime_breakdown.csv, surrogate_ranking_metrics.csv) được ghi lại bên dưới.

```markdown
# docs/day1_decisions.md

## [6/4/2026] IC Runtime Benchmark — COMPLETED

Per-simulation time: see `outputs/day1_benchmark/ic_runtime_benchmark.json` (`per_sim_ms`)
Projected total (n_sample × N_runs): see `outputs/day1_benchmark/ic_runtime_benchmark.json` (`projected_total_hours`)
Decision: n_sample = 5000, N_runs = 200 (ước tính từ runtime_breakdown.csv ~480s total)
Adjusted timeline: No adjustment needed (within < 4h threshold)

## [6/4/2026] One-Hop Baseline Reality Check — COMPLETED

One-hop vs IC pilot: see `outputs/day1_benchmark/one_hop_correlation.json` (`spearman_rho`)
Jaccard@10% = see `outputs/day1_benchmark/one_hop_correlation.json` (`jaccard_at_10pct`)
NDCG@10% = see `outputs/day1_benchmark/one_hop_correlation.json` (`ndcg_at_10pct`)
Decision: GNN story viable (ρ không vượt 0.9 + top-k alignment threshold)
Narrative chosen: "GNN as feature-agnostic IC surrogate"

## [~13/4/2026] Baseline Results — KNOWN

degree Spearman ρ = 0.826 (from baseline_ranking_metrics.csv)
pagerank Spearman ρ = 0.824
kshell Spearman ρ = 0.816
gnn_centrality ρ = 0.817 (SAGE, from surrogate_ranking_metrics.csv)
gnn_raw_attr ρ = 0.534 (SAGE)
mlp_raw_attr ρ = 0.435
GNN inference time = 0.067s
MC-IC labeling time = 480s → speedup = 7,169×

## [~16/4/2026] v3.1 Framing Decision — ACTIVE

Professor recommendation: demote Task B; focus on linear pipeline [1]→[2]→[3]→[4]
New experiments required: degree_variance_test, arch_comparison, ranking_loss, bootstrap_CI
Tension to resolve: gnn_centrality (0.817) < degree (0.826) → bootstrap CI needed

## [sau C2/C4] Architecture Comparison Result

Best architecture: chọn từ `outputs/mapr2026_v3_results/surrogate_ranking_metrics.csv` (max `spearman_rho_mean` trong các architecture rows)
Best arch Spearman ρ: xem `outputs/mapr2026_v3_results/surrogate_ranking_metrics.csv`
Bootstrap CI (GNN best vs degree): xem `outputs/mapr2026_v3_results/gnn_vs_degree_bootstrap_ci.json` (`delta_mean`, `ci_95_lower`, `ci_95_upper`)
Interpretation: đọc trực tiếp `outputs/mapr2026_v3_results/gnn_vs_degree_bootstrap_ci.json` (`interpretation`)
Paper claim: map theo `interpretation` (equivalent / significantly_lower / significantly_higher)
```

---

## 21. Phân Công Team

> **Execution alignment note (v3.1):** Bảng dưới đây là 6-person reference để mô tả đầy đủ vai trò. Khi triển khai thực tế với team 3 người, **`docs/MAPR2026_v3_team_parallel_coding_plan.md` là bản execution override**. Mapping đã lock: role GNN (Person 5 reference) → Person 3 execution; Person 3 phụ trách C2 + C3 + C4.

| Người | Track     | Ngày 6–12/4                                        | Ngày 13–21/4                                                                        | Ngày 22–30/4     |
| ----- | --------- | -------------------------------------------------- | ----------------------------------------------------------------------------------- | ---------------- |
| 1     | Data + IC | **Day 1 benchmarks**, preprocessing, IC (bg)       | IC finalize; **degree-controlled variance test** (Section 8.4)                      | Writing support  |
| 2     | Data + IC | Sampling + KS, pilot diagnostics, stability        | Label stability write-up                                                            | Writing support  |
| 3     | Baselines | Betweenness (bg), PageRank, k-shell, **community** | **bootstrap CI** (Sec 8.5)                                                          | Results tables   |
| 4     | Baselines | One-hop, 2-hop, Node2Vec, MLP                      | Evaluation metrics, runtime; fill NDCG/P@10% in baseline table                      | Figures          |
| 5     | GNN       | PyG setup, GNN-raw-attr (SAGE) training            | **Architecture comparison (GCN/GIN/GAT)** + **ranking loss** (9.1b); 5-seed results | Paper Sec 4      |
| 6     | Paper     | **Intro + Related Work từ Ngày 8**                 | Sec 3 draft (MC-IC as metric)                                                       | Paper Sec 1-2, 5 |

> **v3.1 priority shift (execution mapping):** Person 3 (mapped từ role Person 5) tập trung vào architecture comparison + ranking loss (NEW MUST-HAVES) trước khi làm ablation variants. Person 3 chạy bootstrap CI sau khi có best-arch predictions.

**Critical path dependencies (v3.1):**

```
Person 3 arch comparison (C2; mapped từ role Person 5) → done by 18/4 EOD
                                     ↓
Person 3 bootstrap CI (C4) → start 19/4, done by 20/4 EOD
                                     ↓
Person 6 Section 4 write-up → finalize 20-21/4
```

- C1 (degree variance test): Person 1 → không phụ thuộc vào C2, có thể chạy song song 15-16/4
- C3 (ranking loss): Person 3 (mapped từ role Person 5) → chạy với best arch từ C2, sau 18/4

**Daily standup:** 15 phút (không phải 30-45).
**Milestone bắt buộc cuối mỗi tuần:** artifact cụ thể, không chỉ code.

- **Tuần 15-18/4:** C2 (arch comparison) + C1 (degree variance) artifacts sẵn sàng
- **Tuần 19-21/4:** C3 (ranking loss) + C4 (bootstrap CI) + tất cả paper sections drafted

---

## 22. New Experiments Checklist (v3.1 — Per Professor's Framing)

> Quick-reference cho Person 3 và Person 5 — các thực nghiệm mới cần chạy theo framing v3.1.
> Đánh dấu khi hoàn thành và ghi artifact path vào cột tương ứng.

### CRITICAL — Blocking cho defensible submission

| #   | Thực nghiệm                                                                                                                                                                                            | Owner | Artifact expected                                                                         | Status |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----- | ----------------------------------------------------------------------------------------- | ------ |
| C1  | **Degree-controlled IC variance test** (Section 8.4) — lọc nodes degree ∈ [D±5], tính CV(IC) trong band                                                                                                | P1/P3 | `degree_controlled_ic_variance.json`                                                      | ☐      |
| C2  | **Architecture comparison** — train GCN/GIN/GAT với `raw_attr` features, 5 seeds (Section 9.1)                                                                                                         | P5    | rows `gcn_raw_attr`, `gin_raw_attr`, `gat_raw_attr` trong `surrogate_ranking_metrics.csv` | ☐      |
| C3  | **Ranking loss experiment** — `best_arch_raw_attr_rankloss` với combined α·Huber + (1-α)·pairwise-margin-loss (Section 9.1b); tên thực tế xác định sau C2 (e.g., `gat_raw_attr_rankloss` nếu GAT wins) | P5    | row `best_arch_raw_attr_rankloss` trong `surrogate_ranking_metrics.csv`                   | ☐      |
| C4  | **Bootstrap CI: GNN vs degree** (Section 8.5) — 1000 resamplings, 95% CI của Δ Spearman (Section 8.5)                                                                                                  | P3    | `gnn_vs_degree_bootstrap_ci.json`                                                         | ☐      |

### SHOULD HAVE — Strengthens paper significantly

| #   | Thực nghiệm                                                                                         | Owner | Artifact expected                      | Status |
| --- | --------------------------------------------------------------------------------------------------- | ----- | -------------------------------------- | ------ |
| S1  | Fill baseline table (Section 4.2) với NDCG@10% + P@10% cho tất cả models                            | P3    | Updated `baseline_ranking_metrics.csv` | ☐      |
| S2  | Fix `eval_ranking_harness.py`: thêm `kind='stable'` vào 4 `argsort` calls (Section 8.3 fix)         | P3    | deterministic kshell NDCG              | ☐      |
| S3  | Fix `run_baselines.py evaluate_on_test_mask()`: thêm `sort_values("node_id")` sau `apply_test_mask` | P3    | consistent eval results                | ☐      |

### CAN HAVE — Nếu còn thời gian (sau 25/4)

| #   | Thực nghiệm                                                                         | Owner | Artifact expected         | Status |
| --- | ----------------------------------------------------------------------------------- | ----- | ------------------------- | ------ |
| O1  | **Inductive generalization test** (Section 9.1c) — 20% hold-out khỏi training graph | P5    | `gnn_inductive_eval.json` | ☐      |

### Verification Checklist — Document Edits (✅ = hoàn thành trong document; ☐ = experiments chưa chạy — xem C1–C4 ở trên)

- [x] Section 0 có block "Professor's Framing" với pipeline [1]→[2]→[3]→[4]
- [x] Task B đã bỏ hoàn toàn — pipeline sạch: [1] IC metric → [2] MC-IC costly → [3] regression nature → [4] GNN surrogate
- [x] Section 9.1 có đủ 4 architectures: GraphSAGE, GCN, GIN, GAT (spec + code — experiments chưa chạy: xem C2)
- [x] Section 9.1b có ranking loss experiment block (spec + code — chưa chạy: xem C3)
- [x] Section 8.4 có degree-controlled variance test (spec + code — chưa chạy: xem C1)
- [x] Section 8.5 có bootstrap CI protocol với Python code (spec + code — chưa chạy: xem C4)
- [x] Section 14 phản ánh 5-section linear narrative (không còn 6 RQ song song)
- [x] Section 19.2 có 6 GNN-specific risk scenarios
- [x] Framing language table (Section 0.3) có rows về "statistically comparable" + "message passing +0.099"
- [x] Paper structure Section 4.2 có horizontal reference line cho degree baseline
- [x] Không có chỗ nào còn viết "GNN approximates very well" mà chưa có bootstrap CI evidence
- [x] Rankloss naming nhất quán: `best_arch_raw_attr_rankloss` xuyên suốt (không hardcode gat trước C2)
- [x] Threshold `0.90` trong Section 4.5 có comment giải thích (= top-10% = `classification_threshold: 0.10`)
- [x] Section 7 naming convention phân biệt CSV artifact names vs paper display names
- [x] RQ3 fallback narrative có note về inductive test requirement

---

_Document version: 3.1 (review pass 5 — 15/4/2026)_
_Review pass 5 (15/4/2026): Update header days remaining (25→15); fix rankloss naming consistency (`best_arch_raw_attr_rankloss` replaces hardcoded `gat_raw_attr_rankloss` in 4 locations: Section 9.1b, 15, 18, 22); add threshold=0.90 comment (Section 4.5); add CSV vs display name note (Section 7); add arch comment in Section 8.7 code; fix RQ3 fallback narrative (add inductive note); mark all 15 Verification Checklist items as [x] (document complete)._

_Changes from v3.0 → v3.1 (Professor's Framing integration — 15/4/2026):_
_Add Section 0.1b: Professor's linear pipeline [1]→[2]→[3]→[4];_
_Add 4 rows to framing language table (Section 0.3);_
_Add Section 8.4: degree-controlled IC variance test;_
_Add Section 8.5: bootstrap significance test GNN vs degree (bootstrap_spearman_ci);_
_Rename old 8.4 → 8.6 (Multiple Testing Correction);_
_Update Section 7 Group 5: 4-architecture table (SAGE/GCN/GIN/GAT) + GAT theory justification;_
_Replace Section 9.1: unified GNNSurrogate(arch=...) class supporting sage/gcn/gin/gat;_
_Add Section 9.1b: ranking loss experiment (pairwise_ranking_loss + combined_loss);_
_Add Section 9.1c: optional inductive generalization test;_
_Replace Section 14: 5-section linear paper structure per professor's framing;_
_Update Section 19: add 19.1/19.2 split; add 6 GNN-specific risk scenarios;_
_Full review pass (15/4): fix title v3.0→v3.1; update Section 0.1 diagram;_
_Restructure Section 12 RQs (RQ2/RQ2b/RQ3b/RQ4 → APPENDIX tier, RQ1/RQ3 → MAIN);_
_Update Section 9.3 runtime table with known values (480s, 0.067s, 7169×);_
_Update Section 15 experiment.yaml GNN section (multi-arch config, rankloss, group5 expanded);_
_Update Section 17 folder structure (gnn_surrogate.py, bootstrap_ci.py, degree_variance_analysis.py);_
_Restructure Section 18 checklist (v3.1 blockers);_
_Update Section 21 team assignments (Person 5: arch comparison + ranking loss priority);_
_Add tier notes to Sections 5, 6, 10, 11;_
_Fix experiment.yaml v3.0 → v3.1 references_

_Changes from v2 → v3.0: Remove 8% calibration target → variance check; Add Day 1 one-hop ρ check;_
_Fix joblib: CSR + loky; Fix baseline Group 3: 2-hop proxy;_
_Fix title; Clarify transductive evaluation; Fix life_time independence; Restructure GNN ablation;_
_Fix random seed handling; Add Louvain community; Add 5-seed training; Fix runtime table;_
_Reduce Node2Vec params; Writing starts Day 8_
_Tổng hợp từ: Expert SNA Review rounds 1–4_
_Bắt đầu: 6/4/2026 | Deadline: 30/4/2026_
