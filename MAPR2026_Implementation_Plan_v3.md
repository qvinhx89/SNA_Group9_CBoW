# MAPR 2026 — Implementation Plan v3.2

## Dual-Operationalization IC Study and GNN Surrogate Learning on Twitch

**Deadline submit:** 30/4/2026 | **Bắt đầu thực thi:** 6/4/2026 | **📍 Hôm nay: 21/4/2026 — còn 9 ngày**
**Conference:** MAPR 2026, Hue City, 13–14/8/2026 — IEEE Xplore Track 1: Graph Learning
**Dataset:** Twitch Gamers Social Network (Rozemberczki et al., 2021) — 168,114 nodes, 6,797,557 edges
**Document version:** 3.2 — reframed theo dual-operationalization contrast: `A0 + HSCC`

**Scope bridge:** Tài liệu này là strategic master plan (research + execution + paper). Với phạm vi coding team 3 người, `docs/MAPR2026_v3_team_parallel_coding_plan.md` là execution override; nếu có khác biệt ở tác vụ hằng ngày, ưu tiên Team Plan và giữ các ràng buộc nghiên cứu/narrative theo tài liệu này. **v3.2 override:** MAPR path chính thức không còn là `A0 primary + I-A rescue`, mà là **dual-operationalization contrast** giữa `A0` và `HSCC`.

> **Consistency note (v3.2 freeze):** Historical v3.1 terms chỉ nên còn xuất hiện trong các **archive notes được gắn nhãn rõ ràng**. Active execution trong MAPR v3.2 dùng naming hiện tại theo regime: `ic_scores_a0.parquet`, `regression_targets_a0.parquet`, `ic_scores_hscc_refined.parquet`, `regression_targets_hscc_refined.parquet`, `gnn_vs_degree_bootstrap_ci_a0.json`, `gnn_vs_baseline_bootstrap_ci_hscc.json`, và `gnn_vs_rankloss_bootstrap_ci_hscc.json` (khi C3 được chạy). Nếu có xung đột nội dung, **Team Plan + các section freeze trong file này** là nguồn sự thật cao nhất.
>
> _Lưu ý kỹ thuật:_ một số scripts trong codebase vẫn có default legacy path như `ic_scores_primary.parquet` / `regression_targets.parquet`. Nếu gặp, **treat đó là alias của A0** (không “đổi tên” artifacts tự phát giữa chừng) và ghi rõ trong handoff để tránh mismatch giữa người chạy.

---

> **TIER LEGEND — đọc phần này trước:**
> 🔴 **MAPR-MUST** — Bắt buộc cho defensible submission. Thiếu = paper bị reject.
> 🟡 **BOOST** — Tăng chất lượng; làm sau khi xong hết 🔴 trước deadline 30/4.
> 🔵 **FUTURE[Venue]** — Không kịp MAPR nhưng valuable; giữ lại; submit venue khác sau 30/4.
> ⚪ **REF** — Chỉ đọc để align context; không phát sinh task mới.
> **First-pass protocol:** Đọc chỉ 🔴 + Section 0. Skip tất cả 🟡 và 🔵 cho đến khi xong MUST.

## 0. Nền tảng tư duy — Đọc kỹ trước khi implement [⚪ REF]

### 0.1 Reframe cốt lõi

**Hướng cũ (circular):**

```
SIS = f(PageRank, Betweenness, k-shell)  →  define power user
IC simulation                             →  validate SIS     ← CIRCULAR
```

**Hướng mới (defensible, v3.2):**

```
A0:   p(u,v) = 1/degree(v)                                ← structural operationalization
HSCC: p(u,v) = clip( λ × φ(u)/deg(u) × (1 + γ·I[c_u≠c_v]), 0, p_max )
      with φ(u) = rank(log1p(views_u)/(1+life_time_u))/N ← attribute-community operationalization
        ↓
Monte Carlo IC simulation → IC score mỗi node dưới 2 operationalizations
        ↓
IC score = OPERATIONALIZATION của influence potential
         (simulation-defined proxy, KHÔNG phải ground truth)
        ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ [MAIN PIPELINE — v3.2]                                                      │
│ Task A: Compare IC operationalizations ──► Task C: Learn surrogate where   │
│ (A0 contrast + HSCC main target)        graph message passing adds value    │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 0.1b Professor's Framing (v3.1 — tháng 4/2026)

> **"We use MC-IC as a principled operational metric for influence potential, but we explicitly compare two operationalizations. Under degree-coupled IC, analytical baselines may already be near-optimal; under attribute-community IC, graph message passing may add measurable value. The paper contribution is the contrast, not a blanket claim that one GNN always wins."**

**Pipeline tuyến tính, nhưng có 2 operationalization branches** (thay thế 6 RQ song song — đọc theo thứ tự):

```
[1] MC-IC operationalization nào cho insight gì?
    → A0 = structural contrast; HSCC = domain-informed graph-aware regime
[2] MC-IC đắt?
    → evidence: labeling cost vẫn là motivation cho surrogate
[3] Regression nature?
    → stability/continuous-target justification cho từng regime
[4] GNN thêm giá trị khi nào?
    → compare against đúng comparator của từng regime, không mặc định chỉ vs degree
```

**Scope note (v3.2):** Main contribution vẫn là pipeline [1]→[2]→[3]→[4], nhưng Section 4 phải được tổ chức theo **2 operationalizations**: `A0` và `HSCC`. `A2` là sensitivity. `I-A` được archive khỏi critical MAPR path.

> **Tension cốt lõi cần resolve trong v3.2:** `A0` và `HSCC` yêu cầu **hai comparator khác nhau**.
>
> - `A0`: câu hỏi đúng là GNN có đạt mức **practically equivalent** với degree / two-hop hay không.
> - `HSCC`: câu hỏi đúng là GNN có vượt **strongest standard non-graph baseline** (`LR/MLP` với `life_time`, `views`, và nếu dùng thì `language`) hay không.
>   Defense strategy: (a) giữ `A0` như negative control/structural regime, (b) dùng `HSCC` làm graph-aware regime chính, (c) bootstrap CI theo **đúng comparator của từng regime**, (d) không claim GNN thắng nếu baseline fairness chưa đủ.

---

### 0.2 Ba nhiệm vụ tách biệt — KHÔNG trộn lẫn

| Task  | Câu hỏi nghiên cứu                                                              | Output chính                                                    | Tier     |
| ----- | ------------------------------------------------------------------------------- | --------------------------------------------------------------- | -------- |
| **A** | How do different IC operationalizations define influence without behavior logs? | `A0` labels, `HSCC` labels, stability, regression justification | **MAIN** |
| **C** | When does GNN add value over analytical or flat baselines?                      | Regime-specific comparison tables, bootstrap CI, speedup        | **MAIN** |

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
| "GNN approximates IC to a very good margin"       | "GNN is evaluated against the strongest valid comparator for each operationalization"                                               |
| "GNN outperforms baselines"                       | "Under A0, analytical baselines are near-optimal; under HSCC, message passing may outperform flat baselines"                        |
| "MC-IC is a good metric"                          | "MC-IC is a principled operational metric for influence potential in static social graphs"                                          |
| "We show GNN is better"                           | "The added value of GNN depends on the operationalization and the information available to non-graph baselines"                     |

---

## 1. Construct Validity — Phải Address Trong Paper [🔴 MAPR-MUST]

> **Gap #1 gây rejection.** Reviewer SNA sẽ hỏi ngay: "Why should IC simulation on follower graph tell us anything about real Twitch influence?"

### 1.1 Paragraph bắt buộc trong Section 3.1

> _"The Twitch follower graph represents declared social affinity rather than observed information transmission. Influence on Twitch primarily occurs through live streams, raids, and chat interactions — channels not captured by static friendship edges. The Twitch Gamers dataset has been used in prior network analysis studies for community structure, node classification, and link prediction (Rozemberczki et al., 2021), establishing it as a standard benchmark for graph-level analysis despite the absence of behavioral diffusion logs. While this limits the behavioral realism of any diffusion simulation, prior work has established that social ties correlate with influence pathways in online platforms (Guille et al., 2013; Aral & Walker, 2012), making graph-based diffusion models a reasonable structural operationalization. We explicitly do not claim friendship edges are transmission channels; we treat the graph as a structural substrate on which a hypothetical diffusion process is simulated. All findings should be interpreted as properties of diffusion under this operationalization."_

### 1.2 Directionality — bắt buộc trong config và paper

```yaml
# experiment.yaml
graph_directed: false
graph_direction_note: >
  "Twitch Gamers dataset exposes only mutual-follow edges (Rozemberczki & Sarkar, 2021).
   Undirected treatment is the only valid representation. Under undirected weighted
   cascade, p(u,v) = 1/degree(v) models limited attention budget per incoming edge."
```

### 1.3 Paragraph bắt buộc trong Section 5 (Limitations)

> _"A fundamental limitation is that Twitch's follower network may not correspond to actual information transmission pathways. Dead accounts (X% of nodes, with systematically lower degree and views than active accounts) were excluded; findings generalize only to active users. Furthermore, account age (life_time) is used as an external proxy variable — while exogenous to IC labels, it may capture platform tenure rather than influence potential directly. All quantitative findings should be treated as structural properties of the weighted-cascade operationalization, not as measurements of real Twitch influence."_

---

## 2. Quyết định kỹ thuật cốt lõi — ngày 6/4 [⚪ REF — decisions locked & completed]

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

Dưới weighted cascade với p nhỏ, cascade thường chết trong 1–3 hops, nên one-hop expected spread `Σ 1/degree(v)` có thể tương quan với IC reach. Tuy nhiên **không được mặc định “near-perfect alignment”**; proxies (one-hop/two-hop) được xem là **baselines/diagnostics**, còn surrogate learning vẫn cần thiết nếu top-k ranking không trùng khớp. Gate Day-1 không được dựa vào Spearman một mình: phải kiểm tra thêm top-k alignment qua Jaccard@10% và NDCG@10%.

> **Post-freeze note (v3.2):** Frozen rerun không cho thấy one-hop đạt mức “saturating” (ρ gần 1). Vì vậy **không** restructure paper theo hướng “proxies primary”; giữ proxies như baseline và nhấn mạnh multi-hop/feature interactions khi nói về GNN.

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

**Template narrative (CHỈ dùng nếu điều kiện ρ > 0.9 + top-k alignment thật sự đạt):**

> _"We find that a simple one-hop diffusion proxy under weighted cascade achieves near-saturation with MC-IC influence rankings, suggesting cascade dynamics are largely confined to the local neighborhood of each seed. We therefore report diffusion proxies as strong analytical baselines, while still evaluating learned surrogates for cases where top‑k alignment and multi-hop effects matter."_

---

## 3. Technical Stack [⚪ REF]

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

## 4. IC Simulation — Kỹ thuật đúng [🔴 MAPR-MUST]

### 4.1 A0 Calibration — Weighted Cascade KHÔNG cần target reach

> **v3.2 clarification:** `A0` vẫn là structural reference operationalization, nhưng **không còn là label regime duy nhất** của MAPR path.

Weighted cascade `p(u,v) = 1/degree(v)` là **parameter-free** — không có λ để tune. DeepIM (ICML 2023) báo cáo ~8% reach của **seed set 1% nodes được tối ưu hóa**, không phải single-node reach. Áp con số đó làm calibration target cho single-seed IC là **sai ngữ cảnh nghiêm trọng**.

```yaml
# experiment.yaml — KHÔNG có calibration_target_reach_pct nữa

# A0 contrast track: parameter-free structural operationalization
p_a0: weighted_cascade # p(u,v) = 1/degree(v)
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

### 4.1b Operationalization Hierarchy and Sensitivity Rules [🔴/🟡/🔵 MIXED]

> **Framing bắt buộc (v3.2):**
>
> - MAPR main paper dùng **2 operationalizations đồng cấp**: `A0` và `HSCC`.
> - `A0` = structural contrast / negative control.
> - `HSCC` = main graph-aware target where message passing has a plausible advantage.
> - `A2` = structural sensitivity only.
> - `I-A` và các views-based alternatives khác = archive / post-MAPR note, **không còn nằm trên critical path**.
>
> **Tuyệt đối không viết** như thể `HSCC` là "true Twitch diffusion model". Cách viết defensible duy nhất là:
> "**comparative operationalization study**" + "**regime-dependent surrogate learnability**".

#### Tính chất toán học của từng variant

**A0 — Weighted Cascade (structural contrast / reference regime): `p(u,v) = 1/deg(v)`**

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

| Variant                    | Tên                               | Views-indep             | life_time indep | Grounding                  | Defensible?                        | Priority v3.2 |
| -------------------------- | --------------------------------- | ----------------------- | --------------- | -------------------------- | ---------------------------------- | ------------- |
| `A0: 1/deg(v)`             | Weighted Cascade                  | ✅                      | ✅              | ✅✅ Kempe + DeepIM        | **✅ Main paper (contrast track)** | 1             |
| `HSCC`                     | Source-velocity + community boost | ❌ dùng views/life_time | ❌              | ✅ Twitch-motivated        | **✅ Main paper (main target)**    | 2             |
| `A2: 1/sqrt(deg(u)deg(v))` | Symmetric                         | ✅                      | ✅              | ✅ GCN analogy             | ✅ Sensitivity                     | 3             |
| `A1: 1/deg(u)`             | Source Budget                     | ✅                      | ✅              | Marginal                   | 🟡 Optional sensitivity            | 4             |
| `I-A: w(v)/sum w(N(u))`    | Attr-Informed (row-norm)          | ❌                      | ✅              | Negative-result value only | 🔵 Archive / appendix note         | 5             |
| `II-B: w(v)/deg(v)`        | Views-Density                     | ❌                      | ✅              | Moderate                   | 🔵 Archive fallback only           | 6             |

---

#### I-A / II-B Archive Note (v3.2)

> 🔵 **[ARCHIVE / reference only]** `I-A` và `II-B` được giữ lại chỉ như historical record của các attribute-informed operationalizations đã từng cân nhắc. Chúng **không còn là execution branch, không còn pilot gate, không còn fallback path, và không còn là basis cho architecture selection trong MAPR 2026 window**.
>
> Nếu cần nhắc trong paper, chỉ dùng 1-2 câu ở Discussion/Appendix:
>
> - `I-A`: row-normalized views có thể làm sụp discriminative structure trên graph dense.
> - `II-B`: là một fallback historical variant, không thuộc main paper path.
>
> **Docs-first rule:** bất kỳ script, checklist, hay note nào còn mô tả `I-A pilot`, `C2-I-A`, `C4-I-A`, `GATv2-for-I-A`, hoặc decision tree `I-A -> II-B -> A2` đều phải được hiểu là **legacy only**, không dùng để ra quyết định execution hiện tại.

**Tóm tắt archival value:**

- `I-A`: useful như negative-result reference về một attribute-informed operationalization không phù hợp cho current MAPR path.
- `II-B`: useful như note lịch sử cho một fallback design, không phải active scope.
- `A2`: sensitivity duy nhất còn relevant trong v3.2 nếu main `A0 + HSCC` đã ổn.

**Explicit anti-confusion rule:**

1. Không gọi `A0` là "primary vì I-A là second operationalization" nữa.
2. Không pre-register hay chạy pilot cho `I-A` trong current cycle.
3. Không thêm `I-A` rows vào baseline/GNN/bootstrap outputs của MAPR v3.2.
4. Nếu cần giữ chi tiết kỹ thuật cũ, chuyển chúng sang appendix note hoặc archive doc riêng sau deadline.

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

# ── Sensitivity S1: A2 Symmetric variant [🟡 BOOST — sau khi primary IC xong] ──
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

# ── Legacy archive only: II-B Views-Density (không dùng cho MAPR v3.2) ──
# Giữ lại như reference historical snippet; current docs-first execution path không gọi branch này.
# p(u,v) = clip(views_norm[v] / deg(v), max=0.5)
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
    Legacy II-B reference only; not part of current MAPR execution path.
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

# ── Sensitivity S2: A1 Source Budget variant [🔵 FUTURE:SNA-Journal — không làm cho MAPR] ──
# Chỉ thay 1 dòng: p = 1.0/deg_node (thay vì degrees[nb]) — mọi node 1-hop spread = 1.0
# def run_ic_csr_a1(seed_node, ...): p = 1.0/deg_node if deg_node > 0 else 0.0
# Output: outputs/mapr2026_v3_results/ic_scores_sensitivity_a1.parquet

# ── Unified wrapper sketch (docs-first note: MAPR v3.2 chỉ cần a0/a2/hscc) ──
def run_ic_variant(sampled_nodes, indptr, indices, degrees,
                   p_rule='a0', n_runs=200, n_jobs=-1, **kwargs):
    """
    p_rule options:
      'a0'           → A0 weighted cascade (structural contrast)
      'a2'           → A2 symmetric (sensitivity S1) — kwargs: none extra
      'hscc'         → HSCC graph-aware operationalization
      'ia'/'iib'/'a1'→ legacy archive branches; do not activate in current MAPR path
    Output cùng schema (dict node_id → np.array[n_runs]); execution docs should only
    treat a0/hscc as active and a2 as optional sensitivity.
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

## 6. Community Detection — Stability Explanation Support [🟡 BOOST]

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

## 7. Baseline Hierarchy — Đầy đủ và Không Redundant [🔴 MAPR-MUST — Groups 1–3 + 5; Group 4 = 🟡]

### 7.0 Comparator policy theo operationalization (v3.2 override)

> **Single-source-of-truth cho Section 7-9:** Không còn một baseline ladder áp cho mọi label regime.
>
> - **A0 track:** comparator chính = `degree_rank`, `one_hop_spread`, `two_hop_spread`.
> - **HSCC track:** comparator chính = strongest **standard non-graph baseline** gồm tối thiểu `LR(life_time)`, `LR(views + life_time)`, `LR(degree + views + life_time)`, `MLP(raw attrs)`.
> - **Nếu GNN dùng `language` trên HSCC:** phải có fairness versions cho flat baselines với `language`, ví dụ `LR(views + life_time + language)` và `MLP(raw attrs + language)`.
> - **Không** cho `community_id` hoặc `cross_community_edge_fraction` vào raw input của baseline/main GNN comparison. Đây là signal graph-level hoặc oracle-level, chỉ được dùng cho analysis/ceiling.

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

### Group 4: Flat and Shallow Baselines [🔴/🟡 MIXED]

| Baseline                         | Config                                                      | Regime             |
| -------------------------------- | ----------------------------------------------------------- | ------------------ |
| `LR(life_time)`                  | linear regression on `life_time`                            | **HSCC MUST**      |
| `LR(views + life_time)`          | linear regression on engagement attrs                       | **HSCC MUST**      |
| `LR(degree + views + life_time)` | linear regression on full raw attrs                         | **HSCC MUST**      |
| `MLP(raw attrs)`                 | 2-layer MLP, features = `[views_log, views/day, life_time]` | **HSCC MUST**      |
| `LR/MLP + language`              | fairness versions nếu GNN dùng `language`                   | **HSCC SHOULD DO** |
| Node2Vec + LR                    | dim=64, walks=**20** (không phải 200), walk_len=20          | 🟡 secondary       |

### Group 5: GNN — Architecture × Feature Ablation (v3.1)

> **v3.2 update:** Architecture comparison phải đọc theo **2 regimes**.
>
> - `A0`: raw-attr 3 chiều là đủ cho contrast track.
> - `HSCC`: feature policy phải explicit về `life_time` và `language`; nếu dùng `language` cho GNN thì phải bật fairness versions cho flat baselines.

**Architecture Comparison (MUST — per instructor: "GCN/GIN/GAT..."):**

| Architecture | PyG Class  | Aggregation           | Lý do test                                                                                                                                        |
| ------------ | ---------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| GraphSAGE    | `SAGEConv` | Mean (baseline)       | Hiện tại đang dùng; reference point                                                                                                               |
| **GCN**      | `GCNConv`  | **Sym. norm. sum**    | Spectral baseline; **`D^{-1/2}AD^{-1/2}` structurally analogous to A2 diffusion rule** — additional inductive bias check nếu chạy A2 sensitivity  |
| GIN          | `GINConv`  | Sum + MLP             | Sum agg. preserves multi-hop counts (WL-equivalent expressiveness); reference for non-degree-weighted IC dynamics                                 |
| **GAT**      | `GATConv`  | **Learned attention** | ~~Hypothesis (confirm via C2)~~ **DROPPED — OOM tại A100-40GB, h=128. Dùng `--skip-gat`. H1 archived (xem Section 9.1).**                         |
| **APPNP**    | `APPNP`    | **K-step PPR**        | **H3:** embed-then-propagate với teleport/restart là structural analogy/inductive bias cho IC multi-hop; test như ứng viên “deep receptive field” |

> **Ba inductive bias hypotheses (to be confirmed by C2):**
>
> 1. **GAT–A0 hypothesis — [⚪ ARCHIVED: GAT dropped OOM]:** _(Không testable — GAT bị drop do OOM tại A100-40GB. H1 archived trong Section 9.1.)_
> 2. **GCN–A2 hypothesis:** `GCNConv` aggregates với weight `1/√(d̃_u×d̃_v)` — structurally analogous to A2 symmetric rule. Nếu chạy A2 sensitivity IC labels, GCN expected to be best arch. _(testable via sensitivity experiment — see Section 4.1b)_
> 3. **APPNP–multi-hop hypothesis (H3):** APPNP's K-step PPR propagation với teleport/restart weight là structural analogy/inductive bias cho target diffusion-like; hypothesized to outperform conv-stack baselines trên A0 labels. _(hypothesis — C2 decides; see Section 9.1)_
>
> ⚠ Không kết luận arch nào “best” trước khi có kết quả C2; nếu C2 cho arch khác tốt hơn thì dùng kết quả thực nghiệm. Cả ba hypotheses có prepared narratives cho mọi outcome.

**Feature Ablation (giữ nguyên từ v3.0):**

| Variant        | Features                          | Role                                                          |
| -------------- | --------------------------------- | ------------------------------------------------------------- |
| GNN-raw-attr   | views_log, views/day, life_time   | **Primary proposed** (mọi architecture test trên variant này) |
| GNN-graph-only | degree_norm only (or random init) | Ablation: topology without attributes                         |
| GNN-centrality | degree, PR, kshell                | Ablation: centrality features                                 |
| GNN-full       | all 6 features                    | Supplementary upper bound — **[✦ IF TIME]**                   |

**Matrix thực nghiệm (priority):**

|               | raw_attr                                                       | graph_only | centrality |
| ------------- | -------------------------------------------------------------- | ---------- | ---------- |
| **GraphSAGE** | ✅ đã có                                                       | ✅ đã có   | ✅ đã có   |
| **GCN**       | **MUST**                                                       | [IF TIME]  | [IF TIME]  |
| **GIN**       | **MUST**                                                       | [IF TIME]  | [IF TIME]  |
| **GAT**       | ~~MUST~~ **DROPPED** (OOM A100-40GB, h=128; dùng `--skip-gat`) | —          | —          |
| **APPNP**     | **MUST**                                                       | [IF TIME]  | [IF TIME]  |

**Naming convention cho `surrogate_ranking_metrics_{regime}_clean.csv` (artifact names):**
`gcn_raw_attr`, `gin_raw_attr`, `appnp_raw_attr` (+ `best_arch_raw_attr_rankloss` sau C2). `gat_raw_attr` = **dropped** (OOM; không xuất hiện trong official rerun — dùng `--skip-gat`). Mỗi regime ra 1 file riêng: `*_a0_clean.csv` và `*_hscc_clean.csv`.

> **Lưu ý phân biệt:** Tên trong CSV artifact (`gcn_raw_attr`) khác với tên display trong paper table (`gnn_raw_attr (GCN)`).
> Quy ước: CSV dùng snake*case prefix cho active architectures (`gcn*`, `gin*`, `appnp*`); paper table dùng `gnn_raw_attr (Architecture)`để nhất quán với G5 group labeling.`gat\*` chỉ còn là legacy/archive naming, không thuộc official rerun.

**Tại sao cấu trúc này:**

- GNN-raw-attr vs MLP-raw-attr: giá trị của **message passing** (+0.099 Spearman confirmed)
- Architecture comparison: **which message passing** hoạt động tốt nhất cho IC proxy task
- GNN-raw-attr vs GNN-centrality: giá trị của **centrality features** vs learned structure
- Bootstrap CI phải dùng **đúng comparator theo regime** (Section 8.5)

---

## 8. Evaluation Metrics và Protocol [🔴 MAPR-MUST]

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

**Mục tiêu:** Chứng minh IC capture higher-order diffusion effects ngoài degree **và** one-hop structure. Trả lời câu hỏi reviewer: _"Why not just use degree as power user metric?"_

> **⚠ Enhancement (per reviewer feedback):** Degree-band-only test có weakness: variance trong band [75–85] có thể chủ yếu phản ánh one-hop spread variance (vì p=1/deg(v) → IC ≈ analytical one-hop). Test cần **2 tầng kiểm soát** để claim genuine higher-order effect:
>
> 1. **Tầng 1:** Control degree → check IC variance còn không (test hiện tại)
> 2. **Tầng 2:** Control degree + one-hop spread → check IC variance còn không (new)
>    Nếu variance collapse sau tầng 2: IC ≈ degree + local structure (still honest finding). Nếu variance vẫn còn: genuine 2nd-order IC signal.

```python
import numpy as np
from scipy.stats import spearmanr
from scipy import stats

# ── TIER 1: Degree-band variance (original test) ──────────────────────────────
degree_band = (75, 85)  # ±5 quanh mean (≈81 trên Twitch)
band_mask = (degree >= degree_band[0]) & (degree <= degree_band[1])
band_ic   = ic_scores[band_mask]

cv_tier1 = band_ic.std() / band_ic.mean() if band_ic.mean() > 0 else 0.0

# ── TIER 2: Degree + one-hop-spread controlled residual variance ───────────────
# 2a. Lấy nodes trong degree band
band_one_hop = one_hop_spread[band_mask]   # one_hop_spread từ diffusion_proxies.parquet

# 2b. Regress IC on one-hop spread (linear, controls first-order info)
slope, intercept, *_ = stats.linregress(band_one_hop, band_ic)
ic_residual = band_ic - (slope * band_one_hop + intercept)   # residuals after one-hop control

cv_tier2 = ic_residual.std() / abs(band_ic.mean()) if abs(band_ic.mean()) > 0 else 0.0

# ── INTERPRETATION ─────────────────────────────────────────────────────────────
# cv_tier1 > 0.3:  IC adds info beyond degree  (basic test passes)
# cv_tier2 > 0.15: IC adds info beyond degree + one-hop  (strong higher-order evidence)
# cv_tier2 ≤ 0.15: IC ≈ degree + local neighborhood → honest "IC is degree + one-hop" finding
```

**Output:** `outputs/mapr2026_v3_results/degree_controlled_ic_variance.json`

```json
{
  "degree_band": "75-85",
  "n_nodes_in_band": "<N>",
  "ic_mean_in_band": "<float>",
  "cv_tier1_degree_only": "<float>",
  "cv_tier2_degree_plus_onehop": "<float>",
  "interpretation_tier1": "IC adds info beyond degree | IC ≈ degree",
  "interpretation_tier2": "IC adds genuine higher-order signal | IC ≈ degree + local structure"
}
```

**Paper narrative tương ứng:**

| Tier 2 result     | Paper narrative                                                                                                                                                                                                                                                                                                    |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `cv_tier2 > 0.15` | "Even controlling for both degree and one-hop spread, IC scores retain substantial variance (CV=X within the degree band), demonstrating that IC captures multi-hop cascade composition — second-order neighborhood structure — beyond local connectivity."                                                        |
| `cv_tier2 ≤ 0.15` | "Within narrow degree bands, IC scores are well-explained by degree and one-hop spread (residual CV=X after one-hop control), suggesting that at Twitch scale, IC is primarily determined by local structure. This motivates framing IC as a learnable structural summary rather than a fundamentally new metric." |

**Thời gian:** ~45 phút (filter + regression + compute từ existing IC scores và proxies — không cần rerun simulation).

**Paper narrative nếu cv > 0.3:** "Within the same degree band (degree 75–85), IC scores vary substantially
(CV = X), demonstrating that IC captures higher-order structural information beyond local connectivity."

**Paper narrative nếu cv ≤ 0.3 (honest limitation):** "Within narrow degree bands, IC variance is
limited, suggesting that at Twitch scale, IC is largely explained by degree. This motivates the
architecture comparison: GNN must learn non-degree structure to add value."

### 8.5 Bootstrap Significance Tests — Regime-Specific Comparators (v3.2 — MUST)

**Mục tiêu:** Không dùng một comparator duy nhất cho mọi IC labels.

- **A0:** kiểm định `GNN_best` vs `degree` (và đọc cùng với `two_hop_spread`).
- **HSCC:** kiểm định `GNN_best` vs **strongest standard non-graph baseline** (ưu tiên `LR/MLP` có `life_time`, và nếu áp dụng thì `language`).

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

def ndcg_at_k(y_true, y_pred, k=None):
    """NDCG@k for a single sample. k=None → full ranking."""
    from sklearn.metrics import ndcg_score
    return float(ndcg_score(y_true.reshape(1, -1), y_pred.reshape(1, -1), k=k))

def bootstrap_spearman_ndcg_ci(y_true, y_pred_a, y_pred_b, n_bootstrap=1000, seed=42, ndcg_k=None):
    """Bootstrap CI for Spearman AND NDCG simultaneously — same resample loop."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas_sp, deltas_ndcg = [], []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, ya, yb = y_true[idx], y_pred_a[idx], y_pred_b[idx]
        deltas_sp.append(spearmanr(yt, ya).correlation - spearmanr(yt, yb).correlation)
        deltas_ndcg.append(ndcg_at_k(yt, ya, ndcg_k) - ndcg_at_k(yt, yb, ndcg_k))
    return {
        "spearman": (np.percentile(deltas_sp, 2.5), np.percentile(deltas_sp, 97.5)),
        "ndcg":     (np.percentile(deltas_ndcg, 2.5), np.percentile(deltas_ndcg, 97.5)),
    }

# Run:
# - A0:   y_pred_b = degree ranks
# - HSCC: y_pred_b = strongest standard non-graph baseline
# y_true = IC scores of the corresponding regime
```

**Outputs:**

- `outputs/mapr2026_v3_results/gnn_vs_degree_bootstrap_ci_a0.json` — C4: A0 comparator = degree
- `outputs/mapr2026_v3_results/gnn_vs_baseline_bootstrap_ci_hscc.json` — C4: HSCC comparator = strongest flat baseline
- `outputs/mapr2026_v3_results/gnn_vs_rankloss_bootstrap_ci_hscc.json` — C3 [🟡 BOOST]: rankloss variant vs strongest flat baseline HSCC; chỉ tạo khi dùng `--include-rankloss-comparison` trong `bootstrap_ci.py`

```json
{
  "n_bootstrap": 1000,
  "comparator_a": "gnn_best_architecture",
  "comparator_b": "degree_or_strongest_non_graph_baseline",
  "equivalence_bound": 0.02,
  "spearman": {
    "delta_mean": "<float>",
    "ci_95_lower": "<float>",
    "ci_95_upper": "<float>",
    "interpretation": "practically_equivalent | no_clear_superiority | gnn_better | degree_better"
  },
  "ndcg_at_10pct": {
    "delta_mean": "<float>",
    "ci_95_lower": "<float>",
    "ci_95_upper": "<float>",
    "interpretation": "practically_equivalent | no_clear_superiority | gnn_better | degree_better"
  }
}
```

**Thời gian:** ~10 phút (resample existing predictions, không cần retraining).

**Protocol spec (để tránh ambiguity khi implement):**

- **Metric được CI:** **Spearman ρ (primary) + NDCG@10% (secondary)** — compute cả hai trong cùng một resampling loop, zero extra cost. Lý do: Spearman và NDCG có thể diverge (observed: gnn_graph_only Spearman 0.470 vs NDCG 0.835); reviewer sẽ hỏi.
- **Đơn vị resample:** nodes trong test set (resample with replacement, `size = n_test`)
- **Δ definition:** `Δ = metric(GNN_best) − metric(comparator_b)` với `comparator_b` phụ thuộc regime.
- **"GNN_best":** architecture với mean Spearman cao nhất qua 5 seeds từ C2; predictions = mean predictions qua 5 seeds
- **A0 comparator:** `rank(degree)` trên active graph, đã filter về test nodes
- **HSCC comparator:** strongest valid flat baseline sau khi baseline fairness complete
- **Practical equivalence bound (pre-registered):** `|Δ_spearman| ≤ 0.02` = practically equivalent (SESOI — Smallest Effect of Interest). Pre-register threshold này trước khi xem kết quả.

**Decision (3-tier, dùng cho cả Spearman và NDCG):**

| CI outcome                                         | Interpretation                      | Paper claim                                                                                                  |
| -------------------------------------------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `ci_95_lower > 0`                                  | GNN significantly better            | "GNN surpasses degree under IC-A0"                                                                           |
| CI chứa 0 **và** toàn bộ CI trong `[-0.02, +0.02]` | **Practically equivalent**          | "GNN achieves statistically equivalent Spearman ρ to degree while requiring no precomputed graph statistics" |
| CI chứa 0 **nhưng** rộng hơn `[-0.02, +0.02]`      | No clear superiority (underpowered) | "No significant difference detected; GNN provides learnable alternative with +0.099 over MLP"                |
| `ci_95_upper < 0`                                  | GNN significantly worse             | "GNN competitive within bound; focus on +0.099 message passing story vs MLP"                                 |

### 8.6 Multiple Testing Correction [🟡 BOOST]

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

## 9. GNN Training (v3.1 — Architecture Comparison + Ranking Loss) [🔴 MAPR-MUST — xem subsection tiers]

### 9.1 Architecture Comparison: GCN / GIN / GraphSAGE / APPNP [GAT: dropped — OOM tại A100-40GB, h=128]

> **v3.1:** Mở rộng từ GraphSAGE duy nhất → 5 architectures (ban đầu). **v3.2 official:** 4 active architectures (SAGE, GCN, GIN, APPNP) sau khi GAT bị drop OOM — dùng `--skip-gat`. Config chuẩn giống nhau để fair comparison.
> **APPNP là architecture được bổ sung mới** vì lý do lý thuyết mạnh nhất: APPNP **decouples feature transformation from propagation** — embed node features first (MLP), then propagate embeddings via K=10 PPR steps with teleport weight α. Điều này cho phép **receptive field sâu (K=10) mà không bị oversmoothing** (vì transformation và propagation tách biệt). Đây là inductive bias plausible cho IC regression vì IC scores reflect multi-hop cascade reach. Framing: "plausible candidate for deeper multi-hop influence propagation" — không claim APPNP mimics IC mechanics (stochastic vs deterministic là khác nhau). Xem H3 bên dưới.
>
> **⚠ PyG version check (trước khi chạy):** `APPNP` requires PyG ≥ 2.3. Verify:
>
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
    Supported arch: 'sage' | 'gcn' | 'gin' | 'gat' (archived — OOM tại A100-40GB h=128; dùng --skip-gat trong official run)
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
ARCHITECTURES = ['sage', 'gcn', 'gin', 'appnp']
# NOTE: 'gat' dropped — OOM tại A100-40GB (hidden_channels=128). Official rerun dùng --skip-gat.
# GATConv implementation trong GNNSurrogate giữ lại như reference code nhưng không được invoke.

# Training loop usage:
# for arch in ARCHITECTURES:
#     model = get_model(arch, in_dim=in_dim)
#     # → APPNPSurrogate cho 'appnp', GNNSurrogate cho 3 arch còn lại (sage/gcn/gin)
```

**C2 Fair-comparison protocol (bắt buộc lock):**

- **Cùng split:** `split_masks.parquet` M0-locked (`random_state=42`, degree-stratified 80/20)
- **Feature policy by regime:**
  - `A0 raw_attr` = `[views_log_norm, views_per_day_norm, life_time_norm]` (in_dim=3)
  - `HSCC raw_attr` = `[views_log_norm, views_per_day_norm, life_time_norm, language_encoded]` nếu mục tiêu là learn cross-community signal qua language structure
  - Nếu bật `language` cho GNN ở HSCC, phải chạy fairness baselines với `language`
- **Cùng loss:** Huber (`delta=1.0`); **không early stopping** — `epochs=200` cố định
- **Cùng hyperparams (conv-based archs):** `hidden_dim=128, n_layers=2, dropout=0.3, lr=1e-3`; GAT thêm `gat_heads=4` _(archived — GAT dropped OOM; param giữ lại cho reference code)_
- **APPNP-specific:** `K=10, alpha=0.15, dropout=0.3, lr=1e-3` (thay vì conv layers; xem `APPNPSurrogate`)
- **5 seeds mỗi arch:** `[42, 123, 456, 789, 1024]` → report mean ± std

**Best arch selection criterion:**

- **Primary:** arch có `spearman_rho_mean` cao nhất qua 5 seeds
- **Tie-break (diff < 0.001):** APPNP > GIN > GCN > SAGE (**pre-registered order** — GAT dropped OOM; không được chọn làm best arch; APPNP được ưu tiên vì lý thuyết mạnh nhất — H3)
- **Ghi vào:** `docs/experiment_registry.md` field `gnn_primary_arch` ngay sau C2 xong

**Ba inductive bias hypotheses — pre-registered trước C2:**

**Hypothesis H1 (GAT–A0 alignment) — [⚪ REF: ARCHIVED, GAT dropped]:** _(Không testable trong official MAPR rerun — GAT bị drop do OOM tại A100-40GB với hidden_channels=128.)_

> Ghi chú historical: Intuition rằng GAT có thể học attention weight inversely-proportional-to-degree (phù hợp với A0: `p(u,v)=1/deg(v)`) là plausible về lý thuyết. Nếu chạy trên hardware khác (h=64 hoặc multi-GPU), C2 có thể verify. Kết quả empirical từ h=64 pilot (nếu có) có thể dùng như qualitative note trong Appendix. **Không dùng H1 framing trong main paper vì GAT không run trong current MAPR window.**

**Hypothesis H2 (GCN–A2 alignment — nếu chạy A2 sensitivity):** Dưới `A2` sensitivity (`p(u,v)=1/√(deg(u)×deg(v))`):
`GCNConv` aggregates với weight `1/√(d̃_u×d̃_v)` — structurally analogous to A2 diffusion.
_Prediction:_ Nếu chạy C2 trên A2 labels, GCN expected to outperform GIN/SAGE vì inductive bias aligned với target generative process. _(GAT excluded — dropped OOM)_

> ⚠ Self-loops (d̃ = deg+1 ≠ deg) và non-linearity (ReLU/dropout) làm GCN không _exactly_ implement A2 — đây là structural analogy, not exact equivalence. Pre-register dưới dạng "architectural inductive bias check."

**Hypothesis H3 (APPNP — HSCC/A0 transferable multi-hop inductive bias):**

APPNP thực hiện K-step Personalized PageRank (Klicpera et al., ICLR 2019):

```
x^(k) = (1 - alpha) * A_hat * x^(k-1) + alpha * x^(0)
```

Đây là **structural analogy (inductive-bias)** với một quá trình lan truyền có teleport/restart (không phải cơ chế dừng của IC):

- `alpha` là trọng số tái-inject `x^(0)` mỗi bước (teleport/restart weight)
- `K` là số bước propagation (độ sâu receptive field)
- `(1 - alpha)` là phần “propagate” qua lân cận trong công thức APPNP

_Prediction (hypothesis):_ APPNP is hypothesized to be a strong candidate across both regimes because multi-hop propagation may help under `A0`, while wider receptive fields may help integrate language/community context under `HSCC`. `K=10, alpha=0.15` là starting point.

> ✅ Đây là lý do thêm APPNP vào C2. Nếu H3 confirm → APPNP là best arch → C3 (ranking loss) trên APPNP. Nếu H3 reject → report honestly; IC trên Twitch có thể bị dominated bởi local degree structure hơn là multi-hop cascade.

**Ghi chú về IC-A0 và degree dominance (context quan trọng cho C2):**

> **⚠ Structural constraint của A0:** IC-A0 sử dụng `p(u,v) = 1/deg(v)`, nên IC score **degree-coupled** (transition phụ thuộc trực tiếp vào `deg(v)`). Hệ quả: `degree` Spearman = 0.826 — baseline rất mạnh. GNN training trên A0 labels sẽ dễ "tái học" degree-like quantities từ graph topology.
>
> Empirically từ existing artifacts (test split; `outputs/mapr2026_v3_results/baseline_ranking_metrics_a0_clean.csv`):
>
> - `one_hop_spread` ρ = **0.688** (một hop)
> - `two_hop_spread` ρ = **0.804** (hai hop — multi-hop improves vs one-hop)
> - `degree` ρ = **0.826** (baseline rất mạnh)
> - `gnn_graph_only` (SAGE) = **0.470** — SAGE mean aggregation không capture multi-hop IC
> - `gnn_raw_attr` (SAGE) = **0.534** — raw attrs giúp nhưng vẫn xa degree
>
> **H3 rationale:** APPNP với PPR-style multi-hop propagation có thể capture hiệu ứng multi-hop tốt hơn SAGE mean (two_hop **0.804** > one_hop **0.688** trên test split). Nếu APPNP capture được multi-hop composition tốt hơn → có thể close gap với degree (0.826) hoặc vượt qua.
>
> **Nếu C2 không beat degree trên A0:** đây là **structural expectation**, không phải implementation failure.
> **Nếu C2 không beat strongest flat baseline trên HSCC:** vẫn giữ paper theo contrast story; không mở thêm formula mới trong MAPR window.

### 9.1b Ranking Loss Experiment — C3 [🟡 BOOST]

> **Vấn đề:** HuberLoss tối ưu regression error (MSE-like), nhưng evaluation metrics là Spearman/NDCG
> (ranking metrics). Mismatch này có thể cost 2–3 Spearman points.

```python
import torch.nn.functional as F

# Option A: Pairwise Margin Ranking Loss (random pairs — baseline)
def pairwise_ranking_loss(pred, target, margin=0.1, n_pairs=512):
    """Sample pairs (i,j) where target_i > target_j, penalize if pred_i < pred_j."""
    n = len(pred)
    idx = torch.randperm(n)[:n_pairs * 2].view(n_pairs, 2)
    i, j = idx[:, 0], idx[:, 1]
    mask = target[i] > target[j]
    i, j = i[mask], j[mask]
    if len(i) == 0:
        return torch.tensor(0.0, requires_grad=True)
    return F.margin_ranking_loss(pred[i], pred[j],
                                  torch.ones(len(i), device=pred.device), margin=margin)

# ⭐ Option A2: Top-k Focused Ranking Loss (RECOMMENDED over random pairs)
# Lý do: IC scores heavy-tailed → random pairs → hầu hết (low, low) pairs → weak gradient signal
# cho top-k metrics (NDCG@10%). Top-k focused: sample pairs where ≥1 member is in top 20%.
def pairwise_ranking_loss_topk_focused(pred, target, margin=0.1, n_pairs=512, top_frac=0.2):
    """Top-k focused pair sampling: one from top 20%, one from rest. Better NDCG gradient."""
    n = len(pred)
    top_k = max(1, int(n * top_frac))
    top_idx = torch.argsort(target, descending=True)[:top_k]
    top_sample  = top_idx[torch.randint(top_k, (n_pairs,))]
    rest_sample = torch.randint(n, (n_pairs,), device=pred.device)
    mask = target[top_sample] > target[rest_sample]
    i, j = top_sample[mask], rest_sample[mask]
    if len(i) == 0:
        return torch.tensor(0.0, requires_grad=True)
    return F.margin_ranking_loss(pred[i], pred[j],
                                  torch.ones(len(i), device=pred.device), margin=margin)

# Option B: Combined loss (Huber + top-k focused ranking)
def combined_loss(pred, target, alpha=0.5, margin=0.1):
    huber = F.huber_loss(pred, target, delta=1.0)
    rank = pairwise_ranking_loss_topk_focused(pred, target, margin=margin)  # ← use top-k version
    return alpha * huber + (1 - alpha) * rank

# Variant naming: best_arch + '_rankloss' (e.g., 'appnp_raw_attr_rankloss')
criterion_rankloss = combined_loss  # α=0.5 default; sweep [0.25, 0.5, 0.75] if time
```

**Kỳ vọng:** +0.02–0.03 Spearman, và cải thiện NDCG@10% đáng kể hơn random pairs (vì gradient tập trung vào top nodes). Nếu best_arch + ranking loss đạt ≥ 0.84 → vượt degree (0.826) → clean contribution.

**Output thêm vào `surrogate_ranking_metrics_{regime}_clean.csv`:** `best_arch_raw_attr_rankloss` (tên thực tế sau C2, ví dụ: `appnp_raw_attr_rankloss` nếu APPNP là best arch, `gin_raw_attr_rankloss` nếu GIN là best arch — GAT không thể là best arch vì dropped OOM)

### 9.1c Inductive Generalization Test [🔵 FUTURE:ICLR2027]

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

### 9.1d GINE + IC Edge Features — C5 [🔵 FUTURE:TKDE/WWW2027]

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

| Variant         | Edge features                 | Node features      | CSV model_name          | Priority    |
| --------------- | ----------------------------- | ------------------ | ----------------------- | ----------- |
| GINE-IC-A0      | `1/deg(v)` per edge           | `raw_attr`         | `gine_ic_a0_raw_attr`   | ✦ [IF TIME] |
| GINE-IC-A2      | `1/√(deg(u)×deg(v))` per edge | `raw_attr`         | `gine_ic_a2_raw_attr`   | ✦ [IF TIME] |
| GINE-graph-only | `1/deg(v)` per edge           | `degree_norm` only | `gine_ic_a0_graph_only` | ✦ [IF TIME] |

> **Framing C5 trong paper (bắt buộc nếu include):** "As an upper bound analysis, we augment GNN message passing with explicit IC propagation probabilities as edge features (GINE; Hu et al., 2019). This test quantifies how much structural improvement remains when the diffusion mechanism is directly encoded — distinguishing architectural expressiveness limits from information limits."
>
> **Nếu GINE-IC-A0 cũng không beat degree:** → definitive evidence rằng IC-A0 label là structurally degree-equivalent; chuyển full focus sang I-A track cho GNN advantage claim.
>
> **Nếu GINE-IC-A0 beat degree:** → thú vị — explicit IC mechanism encoding helps; paper claim = "surrogate learning benefits from encoding diffusion mechanism structure."

**Output:** Thêm rows vào `surrogate_ranking_metrics_{regime}_clean.csv` của regime tương ứng (same schema).

---

### 9.1e Architecture Evaluation Log — Considered & Rejected [⚪ REF]

> **Tại sao section này tồn tại:** Khi reviewer hỏi "why not try X?", team có documented rationale. Đây cũng là checklist để không waste time implement architectures không phù hợp.

| Architecture                                 | Xem xét? | Verdict                          | Lý do chi tiết                                                                                                                                                                                                                                                                            |
| -------------------------------------------- | -------- | -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **GATv2** (Brody et al., ICLR 2022)          | ✅       | 🔵 Archive only                  | Historical note từ I-A branch cũ. Không thuộc active `A0 + HSCC` MAPR path; nếu muốn test thì để journal extension/post-MAPR.                                                                                                                                                             |
| **GINE** (Hu et al., NeurIPS 2019)           | ✅       | ✅ **C5 supplemental** [IF TIME] | Edge features = IC prob — strongest explicit alignment; NOT feature-agnostic; không vào C2 fair comparison.                                                                                                                                                                               |
| **GCNII** (Chen et al., ICML 2020)           | ✅       | ❌ Skip cho C2                   | Advantage chỉ xuất hiện tại L=16–64 layers. Tại `n_layers=2` (C2 locked), GCNII ≈ GCN + residual connection — không đủ khác biệt để justify thêm vào. Nếu muốn test, cần L=16 separate experiment, phá vỡ fair comparison.                                                                |
| **HGT** (Hu et al., WWW 2020)                | ✅       | ❌ **Loại hoàn toàn**            | Designed cho heterogeneous graphs (multiple node/edge types). Twitch follower graph là **homogeneous** (1 node type, 1 edge type) → type-specific attention matrices collapse về 1 matrix → HGT = complex GAT variant với overhead không có lợi. Wrong problem type.                      |
| **GraphGPS** (Rampášek et al., NeurIPS 2022) | ✅       | ❌ **Loại — scale blocker**      | MPNN + global Transformer attention. Standard Transformer = O(N²) với N=168k → 28 tỷ attention pairs, không fit GPU. Efficient variants (Performer, BigBird) cần precompute LapPE/RWSE (~30–60 phút eigendecomposition trên 168k×168k). Overkill cho task node regression với 3 features. |

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

### 9.1f Future Work — Other Venues [⚪ REF — NOT in active MAPR plan]

> **Mục đích:** Những items dưới đây được reviewer khuyên cắt khỏi active plan cho MAPR. Trong v3.2, chúng đều là **post-MAPR / archive only** trừ khi main `A0 + HSCC` path hoàn tất sớm.

| Item                                     | MAPR verdict            | Venue value                                                                                     | Venue phù hợp                                                            | Ghi chú                                                                           |
| ---------------------------------------- | ----------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ | --------------------------------------------------------------------------------- |
| **C5 GINE + IC edge features**           | Cắt khỏi active         | **Cao** — First paper encode IC probability as GNN edge feature; novel contribution riêng       | TKDE / IEEE TNNLS / WWW 2027 workshop, WSDM 2027                         | Cần: main `A0/HSCC` path done. Code đã có trong Section 9.1d                      |
| **Inductive Generalization Test (9.1c)** | Cắt nếu bootstrap CI OK | **Rất cao** — "Train on one network, predict on another" = independent paper-level contribution | ICLR 2027 / NeurIPS 2027 graph learning track                            | Cần: 2nd dataset (Facebook MUSAE / Reddit / Twitch-ES). Thiết kế đã có trong 9.1c |
| **GATv2 I-A branch**                     | Archive                 | **Trung bình** — GAT v1 vs GATv2 distinction for row-normalized IC is theoretically sound       | Journal extension sau MAPR (full architecture comparison per IC variant) | Không thuộc active v3.2 path                                                      |
| **A1 source budget `p=1/deg(u)`**        | Cắt — marginal insight  | **Thấp** — Small variation of A0; no reviewer asks for it in isolation                          | Comprehensive sensitivity study (SNA journal)                            | Implement: 1-line change từ A0                                                    |
| **II-B views-density `w(v)/deg(v)`**     | Archive                 | **Minimal** — historical attribute-informed fallback note                                       | Không thấy venue riêng biệt                                              | Giữ như archive/reference; không thuộc active MAPR path                           |
| **Per-group prediction error**           | Cắt                     | **Trung bình** — Fairness angle: does GNN favor high-degree nodes?                              | FAccT 2027 / AIES 2027 / CIKM fairness track                             | Cần: `per_group_prediction_error.csv` (script exists in plan)                     |
| **Louvain resolution sensitivity**       | Cắt                     | **Thấp** — Robustness check for community detection                                             | Methodological note trong journal appendix                               | ~1h additional; only if Louvain used in stability explanation                     |

> **Tóm tắt venue strategy post-MAPR:**
>
> - **MAPR 2026 (30/4):** `A0 + HSCC` contrast story + regime-specific baselines + bootstrap per regime
> - **Journal extension (2026-2027):** Full architecture comparison per IC variant (A0/A2/I-A) + GATv2 + GINE edge features + inductive test. Target: IEEE TNNLS / TKDE / Pattern Recognition
> - **Full paper (2027):** Inductive GNN surrogate for IC with cross-dataset evaluation. Target: ICLR / NeurIPS / WWW

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
| MC IC labeling (5k nodes × 200 runs)    | 480.3s    | One-time cost (from runtime_breakdown.csv)   |
| GNN training (5 seeds, 1 arch)          | ~23s/arch | ~115s total for 5 seeds; ×4 archs = ~460s    |
| GNN inference (168,114 nodes)           | 0.086s    | All active nodes (from runtime_breakdown.csv; headline row = `hscc,gnn_raw_attr`)|
| Node2Vec training (dim=64, walks=20)    | ~8 min    | [to be measured]                             |
| Speedup: MC IC vs GNN inference         | **~5,590×** | 480.3s / 0.086s (headline; round to ~5,500× in paper prose) |
```

**Filling instructions:** Cột "to be measured" sẽ được fill từ actual runs và ghi vào `runtime_breakdown.csv`.

**QUAN TRỌNG:** Nếu GNN-raw-attr là primary, không cần centrality precompute → runtime so sánh fair hơn.
GNN-raw-attr deployment cost = training (one-time, ~460s) + inference (0.086s) vs MC-IC labeling (480.3s per labeling pass used for training labels).

---

## 12. Research Questions (v3.1 — Aligned with Linear Pipeline) [⚪ REF]

> **v3.1 re-alignment:** Professor's framing tổ chức paper theo 4-bước tuyến tính, không phải 6 RQ song song.
> **MAIN RQs** (Section 3+4 của paper): RQ1 + RQ3.
> **Supporting analysis:** metric correlation matrix (continuous; no categorical grouping) để contextualize baselines/ablations.
> **Priority rule:** hoàn thiện RQ1 và RQ3 trước; correlation matrix là **MUST** nhưng phải được lên lịch sao cho **không làm chậm** `A0/HSCC` labels / C2 / bootstrap comparison (chạy sau khi `ic_scores_a0.parquet` + `diffusion_proxies.parquet` đã có).

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

> _"We find that 2-hop analytical spread approximation achieves ρ ≈ [X] with MC IC scores, closely matching GNN surrogate performance while requiring no training. This suggests weighted-cascade dynamics are well-approximated by local structural summaries. GNN's value lies primarily in the message passing contribution — **+0.099 Spearman over MLP without precomputed structural summaries** (learns from raw user metadata: views, activity, account age — no hand-crafted centrality)."_

> **Lưu ý framing:** "Without precomputed structural summaries" thay thế "feature-agnostic" (per reviewer: raw_attr vẫn là features → "feature-agnostic" misleading). Câu "GNN's value lies primarily in efficient inference as network evolves" ngụ ý inductive generalization — chỉ dùng nếu Section 9.1c (inductive test) được thực hiện. Nếu không có 9.1c, dùng "+0.099 without precomputed structural summaries" story thay thế.

_Cả hai outcomes đều publishable tại MAPR với framing đúng._

---

## 13. Dead Account Analysis (Stage 0) [⚪ REF — completed]

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

## 14. Paper Structure (v3.2 — Dual-Operationalization Contrast, 6 trang IEEE Double-blind) [🔴 MAPR-MUST]

> **Framing:** Giữ pipeline tuyến tính, nhưng Section 3-4 phải được tổ chức quanh **hai operationalizations đồng cấp**: `A0` và `HSCC`.
> Appendix là optional; `A2` sensitivity có thể giữ 1 đoạn ngắn nếu đã chạy. `I-A` không còn là execution path chính.

### Tổng quan cấu trúc

| Section  | Tiêu đề                                       | Nội dung chính                                                                               | Trang    |
| -------- | --------------------------------------------- | -------------------------------------------------------------------------------------------- | -------- |
| 1        | Introduction                                  | Problem, Twitch context, vì sao cần so sánh operationalizations chứ không chỉ một IC formula | 0.5      |
| 2        | Background                                    | IC model, surrogate learning, GNN architectures, evaluation principles                       | 0.75     |
| 3        | MC-IC as Comparative Operational Metric       | `A0` và `HSCC`, discriminativeness, stability, regression justification, construct validity  | **1.25** |
| 4        | Surrogate Learning Across Operationalizations | setup, baseline tables, `A0` results, `HSCC` results, contrast analysis, runtime             | **2.0**  |
| 5        | Discussion & Limitations                      | when graph learning helps, why `A0` and `HSCC` differ, HARKing and small-reach limitations   | 0.5      |
| Appendix | optional                                      | `A2` sensitivity, extra stability diagnostics, oracle/ceiling notes                          | —        |

**Total target:** ~5.0 trang nội dung + 0.75 trang references.

### Section 1 — Introduction (0.5 trang)

- Hook: identifying influential actors without behavioral cascade logs.
- Problem statement: MC-IC provides a principled operationalization but is expensive and formulation-dependent.
- Main paper direction: **when does GNN surrogate learning outperform analytical or flat baselines, and how does the answer depend on the IC operationalization?**
- Core contribution statement:
  1. Compare `A0` and `HSCC` as two defensible but qualitatively different operationalizations.
  2. Show that binary classification remains unstable, motivating node-level regression on continuous IC scores.
  3. Demonstrate that GNN value is **regime-dependent**: near-ceiling under `A0`, potentially stronger under `HSCC`.

### Section 2 — Background (0.75 trang)

- 2.1 IC model and MC estimation.
- 2.2 Surrogate learning problem and ranking metrics.
- 2.3 GNN architectures with neutral wording: `SAGE`, `GCN`, `GIN`, `APPNP` (4 architectures; note GAT was excluded due to GPU memory constraints at our scale — state this in Section 2.3 or Appendix).
- 2.4 Short note on evaluation fairness: comparator must match the information available in each operationalization.

### Section 3 — MC-IC as Comparative Operational Metric (1.25 trang)

**Figure 1 (bắt buộc):** IC reach distributions for `A0` and `HSCC`, plus a concise table of mean/median/CV.

**3.1 Construct validity and operationalization framing**

- State clearly that follower graph is a structural substrate, not observed diffusion.
- Present `A0` as standard structural operationalization.
- Present `HSCC` as a Twitch-motivated attribute-community operationalization designed to test whether neighborhood composition adds learnable value.

**3.2 Discriminativeness and stability**

- `A0`: degree-coupled, analytically compressible, still useful as structural regime.
- `HSCC`: non-degenerate labels, degree-decoupled, but smaller mean reach and stronger dependence on source-side attributes.
- Regression remains the primary formulation for both regimes.

**3.3 Why the two regimes are scientifically complementary**

- `A0`: answers what happens when IC is mostly determined by structural dilution/degree.
- `HSCC`: answers what happens when diffusion depends on engagement velocity plus community crossing.
- This contrast is the core scientific story, stronger than selling either regime alone.

### Section 4 — Surrogate Learning Across Operationalizations (2.0 trang)

**Figure 2 (bắt buộc):** 2-panel result figure:

- left = `A0` panel with degree/two-hop reference lines,
- right = `HSCC` panel with strongest non-graph baseline reference line.

**4.1 Shared setup**

- fixed split, transductive evaluation, mean±std across 5 seeds.
- same architecture family, but feature policy must be explicit by regime.
- no `community_id` or `cross_community_edge_fraction` in raw model inputs.

**4.2 A0 results**

- degree and cheap structural proxies are expected to be strong.
- paper claim: under degree-coupled IC, analytical baselines are near-optimal; GNN is mainly valuable as a learnable fast surrogate.
- bootstrap comparator = `degree`.

**4.3 HSCC results**

- baseline table must include `LR(life_time)`, `LR(views + life_time)`, `LR(degree + views + life_time)`, `MLP(raw attrs)`, and fairness variants with `language` if GNN uses `language`.
- paper claim: under `HSCC`, degree is no longer the right benchmark; the right comparator is the strongest standard flat baseline.
- bootstrap comparator = strongest non-graph baseline.

**4.4 Contrast analysis**

- explain why `A0` is degree-coupled while `HSCC` decomposes into source-side attribute signal + structural community-crossing signal.
- reviewer-friendly summary:
  - `A0`: negative/contrast regime.
  - `HSCC`: main graph-aware regime.

**4.5 Runtime**

- retain the surrogate value story: labeling is expensive, inference is cheap.
- make runtime claims independent of whether GNN wins outright on `HSCC`.

### Section 5 — Discussion & Limitations (0.5 trang)

**5.1 When does GNN add value?**

- only when the target requires information not already recoverable by the strongest valid non-graph baseline.
- under `A0`, that added value may be small.
- under `HSCC`, added value is plausible only if GNN captures community-crossing structure beyond raw attributes.

**5.2 Limitations (bắt buộc)**

1. Follower graph is not observed diffusion.
2. `HSCC` is novel and only defensible as a **domain-informed operationalization**, not as the true Twitch diffusion law.
3. `HSCC` has small mean reach and HARKing risk; both must be acknowledged explicitly.
4. `life_time` is a very strong baseline signal under `HSCC`, so fairness of flat baselines is non-negotiable.

**5.3 Optional sensitivity**

- `A2` may be reported as structural robustness if already available.
- `I-A` may be mentioned only as an archived negative result, not a central branch.

### Appendix (optional)

- `A2` sensitivity summary.
- oracle/ceiling notes for `HSCC`.
- extra stability diagnostics.

---

## 15. Experiment Configuration (Final, v3.2) [🔴 MAPR-MUST]

```yaml
# experiment.yaml v3.2

# ─── Graph Setup ──────────────────────────────────────────────
global_seed: 42
filter_dead_account: true
graph_directed: false
graph_direction_note: "Twitch Gamers: mutual-follow edges only. Undirected treatment is required."

# ─── IC Simulation Backend ────────────────────────────────────
ic_backend: csr_numpy
ic_parallel: joblib_loky
ic_n_jobs: -1

# ─── Active MAPR Operationalizations ──────────────────────────
operationalizations_main: [a0, hscc]
operationalizations_sensitivity: [a2]
operationalizations_archive: [ia, iib]

# A0 — structural contrast / negative control
p_a0: weighted_cascade # p(u,v) = 1/degree(v)
a0_calibration_mode: variance_check
a0_outputs:
  ic_scores: data/processed/ic_scores_a0.parquet
  regression_targets: data/processed/regression_targets_a0.parquet
  classification_labels: data/processed/classification_labels_a0.parquet

# HSCC — main graph-aware target
p_hscc: hscc_refined
hscc_formula_note: >
  p(u,v) = clip(lambda * phi(u)/deg(u) * (1 + gamma * I[c_u != c_v]), 0, p_max),
  phi(u) = rank(log1p(views_u)/(1 + life_time_u)) / N
hscc_freeze_policy: no_more_tuning
hscc_lambda: 1.0
hscc_gamma: 1.0
hscc_p_max: 1.0
hscc_outputs:
  ic_scores: data/processed/ic_scores_hscc_refined.parquet
  regression_targets: data/processed/regression_targets_hscc_refined.parquet

# Structural sensitivity only
p_sensitivity_a2: symmetric
ic_sensitivity_a2_output: outputs/mapr2026_v3_results/ic_scores_sensitivity_a2.parquet

# Archive only — not on MAPR critical path
# p_attr_informed: ia
# p_fallback_iib: views_density

# ─── Sampling and IC Runs ─────────────────────────────────────
sample_size_main: 5000
sampling_strategy: degree_quintile_stratified
mc_runs_main: 200
mc_runs_label_stability: 150
n_label_stability_seeds: 3
classification_threshold: 0.10

# ─── Shared Graph Artifacts ───────────────────────────────────
community_algorithm: louvain
louvain_resolution: 1.0
louvain_n_runs: 10
compute_cross_community_fraction: true
compute_betweenness: true
betweenness_backend: networkit
pagerank_alpha: 0.85

# ─── Baselines by Regime ──────────────────────────────────────
baselines_a0:
  structural: [degree_rank, one_hop_spread, two_hop_spread, pagerank, kshell]
  flat: [views, views_day, mlp_raw_attr]

baselines_hscc:
  flat_must:
    [lr_life_time, lr_views_life_time, lr_degree_views_life_time, mlp_raw_attr]
  flat_fairness_if_language:
    [lr_views_life_time_lang, lr_degree_views_life_time_lang]
  structural_context:
    [degree_rank, one_hop_spread, two_hop_spread, pagerank, kshell]
  forbidden_raw_features: [community_id, cross_community_edge_fraction]

# ─── GNN ───────────────────────────────────────────────────────
gnn_architectures: [sage, gcn, gin, appnp] # gat dropped — OOM A100-40GB h=128; --skip-gat
gnn_primary_arch_a0: auto_after_c2
gnn_primary_arch_hscc: auto_after_c2
gnn_gat_heads: 4 # archived — not invoked when --skip-gat
gnn_appnp_K: 10
gnn_appnp_alpha: 0.15
gnn_hidden_dim: 128
gnn_n_layers: 2
gnn_dropout: 0.30
gnn_lr: 0.001
gnn_epochs: 200
gnn_loss: huber
gnn_huber_delta: 1.0
gnn_training_seeds: [42, 123, 456, 789, 1024]

feature_sets:
  a0_raw_attr: [views_log_norm, views_per_day_norm, life_time_norm]
  hscc_raw_attr:
    [views_log_norm, views_per_day_norm, life_time_norm, language_encoded]
  graph_only: [degree_norm]
  centrality: [degree_norm, pagerank_norm, kshell_norm]

# ─── Bootstrap Comparators ────────────────────────────────────
bootstrap_outputs:
  a0: outputs/mapr2026_v3_results/gnn_vs_degree_bootstrap_ci_a0.json # C4
  hscc: outputs/mapr2026_v3_results/gnn_vs_baseline_bootstrap_ci_hscc.json # C4
  hscc_rankloss: outputs/mapr2026_v3_results/gnn_vs_rankloss_bootstrap_ci_hscc.json # C3 [BOOST] — chỉ khi --include-rankloss-comparison
bootstrap_equivalence_bound: 0.02

# ─── Evaluation ───────────────────────────────────────────────
eval_setting: transductive
primary_metrics: [spearman_rho, ndcg_10, precision_10]
avoid_metrics: [accuracy, f1_macro]
multiple_testing_correction: benjamini_hochberg
fdr_alpha: 0.05
```

---

## 16. Timeline cuối cho MAPR (21/4 – 30/4) [🔴 MAPR-MUST]

> **📍 Current date: 21/4/2026**. Từ thời điểm này, mọi planning phải phục vụ submission path ngắn nhất và defensible nhất theo `A0 + HSCC`.

### Ưu tiên không thể phá vỡ

1. Khóa `HSCC` config và regenerate artifact còn thiếu.
2. Hoàn tất baseline fairness cho `HSCC` trước khi đọc bất kỳ kết quả GNN nào như một win.
3. Giữ `A0` như contrast track, không cố biến nó thành headline "GNN thắng degree".
4. Dùng bootstrap với **đúng comparator theo regime**.
5. Không mở thêm operationalization mới trong 9 ngày còn lại.

| Ngày        | Track A: IC & Artifacts                                                                   | Track B: Baselines & Community                                                   | Track C: GNN & Paper                                            |
| ----------- | ----------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| **21/4**    | Regenerate `regression_targets_hscc_refined.parquet`; freeze HSCC config; update registry | Confirm `community_features.parquet`; lock HSCC comparator set                   | Patch docs + align harness naming                               |
| **22/4**    | Lock `A0` + `HSCC` label artifacts                                                        | Run HSCC flat baselines incl. `life_time`; fairness versions if using `language` | Start regime-specific GNN training                              |
| **23/4**    | Optional `A2` only if main path stable                                                    | Assemble regime-specific baseline tables                                         | Continue GNN on `A0` + `HSCC`; collect mean±std                 |
| **24/4**    | No new label regimes                                                                      | Runtime + metrics consolidation                                                  | Bootstrap CI: `A0 vs degree`, `HSCC vs strongest flat baseline` |
| **25/4**    | Artifact freeze                                                                           | Final result tables and contrast analysis                                        | Draft Section 4 and Discussion                                  |
| **26/4**    | Internal validation                                                                       | Consistency sweep on terminology/artifacts                                       | IEEE format check                                               |
| **27/4**    | Buffer for missing artifacts only                                                         | Buffer for table fixes only                                                      | Full paper dry-run                                              |
| **28–29/4** | No scope expansion                                                                        | Final edits only                                                                 | Submission package finalize                                     |
| **30/4**    | —                                                                                         | —                                                                                | **SUBMIT**                                                      |

### Scope Reduction — thứ tự cắt mới (v3.2)

| Cắt trước                                                          | Giữ bắt buộc                                                  |
| ------------------------------------------------------------------ | ------------------------------------------------------------- |
| `I-A`, `II-B`, archived views-based variants                       | `A0` contrast run                                             |
| `A1` source-budget                                                 | `HSCC` main run                                               |
| full exhaustive multi-arch trên cả hai regimes nếu thiếu thời gian | HSCC baseline fairness (`life_time`, `views+life_time`, etc.) |
| `GNN-full`, `GNN-random`, inductive test                           | bootstrap đúng comparator cho từng regime                     |
| ranking-loss sweep nhiều alpha                                     | runtime table + contrast narrative                            |
| `A2` nếu main path còn chưa khóa                                   | community artifact đủ để support HSCC interpretation          |

---

## 17. Folder Structure [⚪ REF]

```
SNA_MAPR2026/
├── data/
│   ├── raw/                          # twitch_edges.csv, twitch_features.csv
│   ├── interim/                      # active_nodes.csv, active_edges.csv
│   └── processed/
│       ├── graph_csr.npz             # CSR format: indptr, indices, degrees
│       ├── node_attributes.parquet   # base attrs (node_id, views, life_time, language, ...)
│       ├── community_features.parquet# node_id, community_id, cross_community_edge_fraction
│       ├── ic_scores_a0.parquet      # A0 weighted-cascade contrast track
│       ├── regression_targets_a0.parquet
│       ├── classification_labels_a0.parquet
│       ├── ic_scores_hscc_refined.parquet
│       └── regression_targets_hscc_refined.parquet
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
│       ├── baseline_ranking_metrics_a0_clean.csv       # A0 G1-G4 models, Spearman+NDCG+P@10
│       ├── baseline_ranking_metrics_hscc_clean.csv     # HSCC G1-G4 models, fairness-aware rows
│       ├── surrogate_ranking_metrics_a0_clean.csv      # A0 G5 GNN variants, mean±std across seeds
│       ├── surrogate_ranking_metrics_hscc_clean.csv    # HSCC G5 GNN variants, mean±std across seeds
│       ├── degree_controlled_ic_variance.json # Section 8.4 — v3.1 NEW
│       ├── gnn_vs_degree_bootstrap_ci_a0.json # A0 comparator = degree
│       ├── gnn_vs_baseline_bootstrap_ci_hscc.json # HSCC comparator = strongest flat baseline
│       ├── metric_correlation_matrix.json     # pairwise Spearman 8×8
│       ├── ic_sensitivity_comparison.json     # Spearman(A0 vs A2), Spearman(A0 vs degree) per variant [SHOULD DO]
│       ├── runtime_breakdown.csv              # IC/GNN/proxy timings
│       ├── gnn_inductive_eval.json            # Section 9.1c — optional
│       │
│       │   # ── archived / post-MAPR artifacts ──
│       ├── ic_scores_ia.parquet               # archive only
│       ├── surrogate_ranking_metrics_ia.csv   # archive only
│       └── ic_scores_iib.parquet              # archive only
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

## 18. Pre-Submission Checklist [🔴 MAPR-MUST]

### Blocker — framing và protocol

- [ ] Paper framing dùng đúng `A0 + HSCC`, không còn câu nào imply `A0 primary + I-A rescue`
- [ ] Construct validity paragraph vẫn giữ rõ follower graph != diffusion channel
- [ ] `graph_directed: false` và IC backend `CSR + loky`
- [ ] `HSCC` được mô tả là domain-informed operationalization, không phải true diffusion law
- [ ] `HSCC` đã được ghi vào `docs/experiment_registry.md`
- [ ] `regression_targets_hscc_refined.parquet` tồn tại thật, không chỉ xuất hiện trong diagnostics

### Blocker — fairness của baselines

- [ ] `HSCC` baseline set có ít nhất `LR(life_time)`, `LR(views + life_time)`, `LR(degree + views + life_time)`, `MLP(raw attrs)`
- [ ] Nếu GNN dùng `language` ở HSCC, có fairness baselines với `language`
- [ ] Không model nào trong main comparison dùng `community_id` hoặc `cross_community_edge_fraction` làm raw input
- [ ] `A0` vẫn có `degree`, `one_hop`, `two_hop` trong main table

### Blocker — significance / outputs

- [ ] Bootstrap `A0`: `gnn_vs_degree_bootstrap_ci_a0.json`
- [ ] Bootstrap `HSCC`: `gnn_vs_baseline_bootstrap_ci_hscc.json`
- [ ] Bootstrap rankloss `HSCC` [🟡 BOOST]: `gnn_vs_rankloss_bootstrap_ci_hscc.json` (chỉ bắt buộc nếu C3 được chạy)
- [ ] Claim trong paper map đúng với bootstrap của từng regime
- [ ] Runtime table tách labeling cost và inference cost
- [ ] 5 training seeds, report mean ± std

### Strongly Recommended

- [ ] `A2` sensitivity nếu main path đã khóa
- [ ] Oracle/ceiling analysis cho HSCC ở appendix, không đưa vào main baseline table
- [ ] Figure 2 phải có 2 panels hoặc 2 comparator lines tương ứng `A0` và `HSCC`

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

## 19. Risk Management [⚪ REF]

### 19.1 Infrastructure & Data Risks

| Rủi ro                                                                        | Xác suất   | Impact       | Action                                                                                      |
| ----------------------------------------------------------------------------- | ---------- | ------------ | ------------------------------------------------------------------------------------------- |
| One-hop ρ > 0.9 + top-k alignment cao (`Jaccard@10% > 0.8`, `NDCG@10% > 0.9`) | Trung bình | **Critical** | Ngày 6/4: check trước; nếu đủ 3 điều kiện thì restructure, nếu không giữ GNN + 2-hop        |
| IC runtime > 8h                                                               | Trung bình | **Critical** | Reduce: n_sample=2k, N_runs=100; log limitation                                             |
| loky OOM với large graph                                                      | Thấp       | Cao          | Reduce n_jobs; monitor RAM                                                                  |
| PyG installation issues                                                       | Thấp       | Trung bình   | Setup Ngày 6/4 sáng; fallback DGL                                                           |
| Paper > 6 pages                                                               | Trung bình | Blocker      | Cut optional appendix extras first; then shorten 4.4 ablation; never cut 4.2 baseline table |

### 19.2 GNN Surrogate Risks (v3.1 — mới)

| Rủi ro                                                                          | Xác suất              | Impact     | Mitigation                                                                                                                                     |
| ------------------------------------------------------------------------------- | --------------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| GNN không vượt degree sau full architecture search (SAGE/GCN/GIN/APPNP)         | Trung bình            | Trung bình | Bootstrap CI (Section 8.5) → claim practical equivalence (nếu CI within ±0.02); "+0.099 without precomputed structural summaries" story vs MLP |
| GAT không converge với current setup (4 heads, hidden=128)                      | ~~Thấp~~ **Resolved** | **N/A**    | **Resolved:** GAT đã bị drop chính thức do OOM tại A100-40GB. Official rerun dùng `--skip-gat`. Không cần mitigation.                          |
| Ranking loss không improve Spearman so với Huber                                | Trung bình            | Thấp       | Report as negative finding (appendix note); Huber-trained GNN remains primary variant                                                          |
| Degree-controlled variance test shows low IC variance (CV < 0.3)                | Thấp                  | Trung bình | Honest limitation in paper: "IC ≈ degree at Twitch scale"; strengthen runtime story instead                                                    |
| Bootstrap CI shows GNN significantly _lower_ than degree (CI entirely negative) | Thấp                  | Cao        | Restructure Section 4 claim: focus on (1) no-centrality-precompute advantage + (2) message passing contribution (+0.099)                       |
| Multiple architecture runs produce high variance (std > 0.05)                   | Thấp                  | Trung bình | Report mean ± std across 5 seeds; highlight reproducibility; use more seeds (10) for final table                                               |

---

## 20. Decision Log Template [⚪ REF]

> **Status as of 16/4/2026:** Day-1 decisions đã hoàn thành. IC simulation đã chạy xong. Known outcomes
> from artifacts (runtime_breakdown.csv, `surrogate_ranking_metrics_a0_clean.csv`, `surrogate_ranking_metrics_hscc_clean.csv`) được ghi lại bên dưới.

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

degree Spearman ρ = 0.826 (from `baseline_ranking_metrics_a0_clean.csv`)
pagerank Spearman ρ = 0.824
kshell Spearman ρ = 0.816
gnn_centrality ρ = 0.817 (SAGE, from `surrogate_ranking_metrics_a0_clean.csv`)
gnn_raw_attr ρ = 0.534 (SAGE)
mlp_raw_attr ρ = 0.435
GNN inference time = 0.086s
MC-IC labeling time = 480.3s → speedup = ~5,590× (round to ~5,500× in paper prose)

## [~16/4/2026] v3.1 Framing Decision — ACTIVE

Professor recommendation: demote Task B; focus on linear pipeline [1]→[2]→[3]→[4]
New experiments required: degree_variance_test, arch_comparison, ranking_loss, bootstrap_CI
Tension to resolve: gnn_centrality (0.817) < degree (0.826) → bootstrap CI needed

## [sau C2/C4] Architecture Comparison Result

Best architecture: chọn **riêng theo regime** từ `outputs/mapr2026_v3_results/surrogate_ranking_metrics_a0_clean.csv` hoặc `outputs/mapr2026_v3_results/surrogate_ranking_metrics_hscc_clean.csv` (max `spearman_rho_mean` trong các architecture rows)
Best arch Spearman ρ: xem file `_a0_clean` hoặc `_hscc_clean` tương ứng với regime đang report
Bootstrap `A0`: xem `outputs/mapr2026_v3_results/gnn_vs_degree_bootstrap_ci_a0.json`
Bootstrap `HSCC`: xem `outputs/mapr2026_v3_results/gnn_vs_baseline_bootstrap_ci_hscc.json`
Paper claim: map theo interpretation của từng regime, không dùng một comparator duy nhất cho cả paper
```

---

## 21. Phân Công Team [⚪ REF]

> **Execution alignment note (v3.2):** Bảng dưới đây là 6-person reference để mô tả đầy đủ vai trò. Khi triển khai thực tế với team 3 người, **`docs/MAPR2026_v3_team_parallel_coding_plan.md` là bản execution override**. Từ v3.2, execution path được hiểu là `A0 contrast + HSCC main target`.

| Người | Track     | Ngày 6–12/4                                        | Ngày 13–21/4                                                                                                | Ngày 22–30/4     |
| ----- | --------- | -------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | ---------------- |
| 1     | Data + IC | **Day 1 benchmarks**, preprocessing, IC (bg)       | IC finalize; **degree-controlled variance test** (Section 8.4)                                              | Writing support  |
| 2     | Data + IC | Sampling + KS, pilot diagnostics, stability        | Label stability write-up                                                                                    | Writing support  |
| 3     | Baselines | Betweenness (bg), PageRank, k-shell, **community** | **bootstrap CI** (Sec 8.5)                                                                                  | Results tables   |
| 4     | Baselines | One-hop, 2-hop, Node2Vec, MLP                      | Evaluation metrics, runtime; fill NDCG/P@10% in baseline table                                              | Figures          |
| 5     | GNN       | PyG setup, GNN-raw-attr (SAGE) training            | **Architecture comparison (GCN/GIN/APPNP)** + **ranking loss** (9.1b); 5-seed results (**GAT dropped OOM**) | Paper Sec 4      |
| 6     | Paper     | **Intro + Related Work từ Ngày 8**                 | Sec 3 draft (MC-IC as metric)                                                                               | Paper Sec 1-2, 5 |

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

## 22. Execution Checklist (v3.2 — Dual-Operationalization MAPR Path) [🔴 MAPR-MUST]

> Quick-reference cuối file. Nếu có xung đột với nội dung cũ ở các section tham chiếu lịch sử, section này là **v3.2 override**.

### CRITICAL — Blocking

> ⚠ **Note:** Labels B1–B5 trong bảng này là "blocking task" numbers, KHÔNG phải "experiment C1–C5". Experiment numbering (C1/C2/C3/C4/C5) nằm ở Section 8.4–8.7 và Section 9.1.

| #   | Thực nghiệm / việc                                                         | Owner | Artifact expected                                                                     | Status |
| --- | -------------------------------------------------------------------------- | ----- | ------------------------------------------------------------------------------------- | ------ |
| B1  | Regenerate HSCC regression targets + freeze config                         | P1    | `regression_targets_hscc_refined.parquet` + registry entry                            | ☐      |
| B2  | HSCC baseline fairness set hoàn chỉnh (Exp C1)                             | P3    | HSCC rows trong `baseline_ranking_metrics_hscc_clean.csv`                             | ☐      |
| B3  | Architecture comparison trên `A0` + `HSCC` (Exp C2; `--skip-gat`, 4 archs) | P3    | `surrogate_ranking_metrics_a0_clean.csv` + `surrogate_ranking_metrics_hscc_clean.csv` | ☐      |
| B4  | Bootstrap `A0`: GNN vs degree (Exp C4-A0)                                  | P3    | `gnn_vs_degree_bootstrap_ci_a0.json`                                                  | ☐      |
| B5  | Bootstrap `HSCC`: GNN vs strongest flat baseline (Exp C4-HSCC)             | P3    | `gnn_vs_baseline_bootstrap_ci_hscc.json`                                              | ☐      |

### SHOULD HAVE

| #   | Thực nghiệm / việc                                   | Owner | Artifact expected                        | Status |
| --- | ---------------------------------------------------- | ----- | ---------------------------------------- | ------ |
| S1  | Rankloss variant trên HSCC (Exp C3 — [🟡 BOOST])     | P3    | `gnn_vs_rankloss_bootstrap_ci_hscc.json` | ☐      |
| S2  | `A2` sensitivity nếu main path đã ổn                 | P1/P3 | `ic_scores_sensitivity_a2.parquet`       | ☐      |
| S3  | NDCG@10% + P@10% đầy đủ cho cả `A0` và `HSCC` tables | P3    | updated metrics CSVs                     | ☐      |
| S4  | Ceiling/oracle appendix note cho HSCC                | P1/P3 | appendix text / note                     | ☐      |

### CUT FIRST

| #   | Item                                          | Why cut first                               |
| --- | --------------------------------------------- | ------------------------------------------- |
| X1  | `I-A`, `II-B`                                 | Không còn thuộc MAPR critical path          |
| X2  | multi-alpha rankloss sweep                    | Không làm đổi paper claim cốt lõi           |
| X3  | exhaustive architecture grid beyond main path | Tốn thời gian hơn giá trị trong 9 ngày cuối |
| X4  | inductive / GINE / extra archive branches     | Thuộc post-MAPR                             |

### Verification Checklist — Document Consistency

- [x] Title và Section 0 phản ánh `A0 + HSCC`
- [x] `A0` không còn bị mô tả như primary target duy nhất
- [x] `HSCC` được mô tả như domain-informed operationalization, không phải true diffusion model
- [x] Comparator policy đã tách `A0` vs `HSCC`
- [x] Folder/artifact names không còn neo vào `primary` cho main MAPR path
- [x] Checklist pre-submission đã thêm HSCC fairness requirements
- [x] GAT drop đã propagate đến: Section 9.1, experiment.yaml, risk table, architecture table, paper structure §2.3
- [x] 4-arch list (`sage, gcn, gin, appnp`) locked trong experiment.yaml + code snippet Section 3
- [x] Bootstrap outputs yaml có đủ 3 keys (a0, hscc, hscc_rankloss)
- [x] Section 22 blocking items (B1–B5) dùng canonical `_clean` CSV names

---

_Document version: 3.2_
_Last strategic rewrite: 21/4/2026_
_Last audit: 28/4/2026 — GAT drop fully propagated; Section 22 labels disambiguated_
