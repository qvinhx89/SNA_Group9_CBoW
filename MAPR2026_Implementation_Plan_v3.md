# MAPR 2026 — Implementation Plan v3.0

## Operationalizing Influence Potential via Weighted-Cascade IC Simulation and GNN Surrogate Learning on Twitch

**Deadline submit:** 30/4/2026 | **Bắt đầu thực thi:** 6/4/2026 (còn 25 ngày)
**Conference:** MAPR 2026, Hue City, 13–14/8/2026 — IEEE Xplore Track 1: Graph Learning
**Dataset:** Twitch Gamers Social Network (Rozemberczki et al., 2021) — 168,114 nodes, 6,797,557 edges
**Document version:** 3.0 — tích hợp feedback từ Expert SNA Review rounds 1–4

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
┌─────────────────────┬─────────────────────────┬──────────────────────┐
│  Task A             │  Task B                 │  Task C              │
│  Operationalize     │  Divergence Analysis    │  GNN Surrogate       │
│  (IC labels)        │  (views vs IC typology) │  (approximate IC)    │
└─────────────────────┴─────────────────────────┴──────────────────────┘
```

### 0.2 Ba nhiệm vụ tách biệt — KHÔNG trộn lẫn

| Task  | Câu hỏi nghiên cứu                                           | Output chính                      |
| ----- | ------------------------------------------------------------ | --------------------------------- |
| **A** | How to define influence proxy without behavioral logs?       | IC scores, label stability        |
| **B** | When do popularity metrics disagree with diffusion rankings? | 2×2 typology, structural profiles |
| **C** | Can GNN approximate IC efficiently vs analytical baselines?  | Spearman ρ, NDCG, speedup         |

### 0.3 Framing language — nhất quán xuyên suốt paper

| ❌ KHÔNG viết                     | ✅ PHẢI viết                                        |
| --------------------------------- | --------------------------------------------------- |
| "We identify real power users"    | "We operationalize influence potential"             |
| "Ground truth influence"          | "Simulation-defined influence proxy"                |
| "GNN predicts influencers"        | "GNN approximates IC simulation efficiently"        |
| "Calibrated to 8% reach (DeepIM)" | "Pilot verified non-degenerate spread distribution" |
| "IC validates our approach"       | "IC is our definitional operationalization"         |

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

# Chạy IC pilot trên 200 nodes × 50 runs
pilot_nodes = random.sample(all_active_nodes, 200)
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

top_ic = set(np.argsort(-ic_vals)[:k])
top_onehop = set(np.argsort(-onehop_vals)[:k])
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

> _"We find that one-hop analytical expected spread under weighted cascade achieves ρ > 0.9 with full MC IC scores, suggesting cascade dynamics are largely confined to the local neighborhood of each seed. This finding itself is a contribution: expensive MC simulation can be approximated by an O(E) analytical proxy. GNN surrogate learning provides further gains in the divergence analysis and handles the views-IC typology classification task."_

---

## 3. Technical Stack

```
IC Simulation Backend:   igraph (C) + CSR numpy arrays — TUYỆT ĐỐI KHÔNG dùng NetworkX BFS
Parallel IC:             joblib Parallel(prefer='loky') + CSR shared-memory arrays
Graph Analytics:         NetworKit (C++) cho betweenness approximate
Community Detection:     python-louvain hoặc cdlib (Louvain algorithm)
GNN:                     PyTorch Geometric (PyG) ≥ 2.5, torch ≥ 2.0
Embeddings:              node2vec library (dimensions=64, walks=20-30)
Stats:                   scipy.stats, statsmodels (BH correction), sklearn
Data:                    pandas + pyarrow (parquet format)
```

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

# Uniform baseline only (sensitivity)
p_sensitivity: uniform
kappa_target: 2 # p = kappa/mean_degree ≈ 0.025
```

**Cite DeepIM đúng cách:**

> _"Following the weighted-cascade experimental setup widely used in influence maximization, including the configuration adopted in DeepIM (Ling et al., 2023) and originally formalized by Kempe et al. (2003), we set p(u,v) = 1/degree(v). This models a limited attention budget where each neighbor's influence is inversely proportional to the number of connections competing for attention."_

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

### 4.3 Label Stability — Prerequisite trước GNN

```python
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

### 4.5 Quadrant Size Check + Two-Sample Strategy

```python
def check_and_expand_typology_sample(df_sampled, df_full, G,
                                      ic_col='ic_score_mean',
                                      views_col='views',
                                      threshold=0.90, min_size=150):
    """
    Two-sample strategy:
    - Sample A (representative): dùng cho surrogate GNN training/eval
    - Sample B (targeted): augment Hidden quadrant nếu quá nhỏ
    """
    ic_cut    = df_sampled[ic_col].quantile(threshold)
    views_cut = df_sampled[views_col].quantile(threshold)

    df_sampled = df_sampled.copy()
    df_sampled['typology'] = 'non'
    df_sampled.loc[(df_sampled[ic_col] >= ic_cut) & (df_sampled[views_col] >= views_cut), 'typology'] = 'true'
    df_sampled.loc[(df_sampled[ic_col] >= ic_cut) & (df_sampled[views_col] < views_cut),  'typology'] = 'hidden'
    df_sampled.loc[(df_sampled[ic_col] < ic_cut)  & (df_sampled[views_col] >= views_cut), 'typology'] = 'overrated'

    counts = df_sampled['typology'].value_counts()
    print(f"Quadrant sizes: {dict(counts)}")

    hidden_count = counts.get('hidden', 0)
    if hidden_count < min_size:
        print(f"Hidden quadrant: {hidden_count} < {min_size}. Augmenting Sample B...")
        # Thêm high-betweenness / low-views / mid-degree nodes từ full graph
        candidates = df_full[
            (df_full['betweenness'] > df_full['betweenness'].quantile(0.7)) &
            (df_full['views'] < df_full['views'].quantile(0.3)) &
            (~df_full['node_id'].isin(df_sampled['node_id']))
        ].sample(min(500, min_size * 2), random_state=42)
        # NOTE: Sample B chỉ dùng cho typology analysis, KHÔNG dùng để train GNN
        return df_sampled, candidates, counts
    return df_sampled, None, counts
```

---

## 5. Null Model — Typology Comparison (không chỉ rank correlation)

```python
import networkx as nx
from networkx import configuration_model

def null_model_typology_comparison(G_nx, sampled_nodes_500,
                                    ic_scores_real, views_col,
                                    n_realizations=3, n_runs=100):
    """
    Null model mạnh hơn: so sánh TYPOLOGY QUADRANT giữa real và null,
    không chỉ so Spearman ρ của IC rankings.

    Câu hỏi: Nếu null graph cũng có Hidden quadrant với betweenness cao
             → typology là degree-distribution artifact.
    """
    rho_results = []
    hidden_betweenness_nulls = []

    for realization in range(n_realizations):
        # Generate configuration model
        degree_seq = [d for _, d in G_nx.degree()]
        G_null = configuration_model(degree_seq, seed=realization * 100)
        G_null = nx.Graph(G_null)              # remove multi-edges
        G_null.remove_edges_from(nx.selfloop_edges(G_null))

        # Build CSR for null
        adj_null = nx.to_scipy_sparse_array(G_null, format='csr')
        ip_null  = adj_null.indptr
        ix_null  = adj_null.indices
        dg_null  = np.diff(ip_null)

        # IC on null (500 nodes)
        ic_null = {}
        for node in sampled_nodes_500:
            if node < G_null.number_of_nodes():
                ic_null[node] = run_ic_csr(node, ip_null, ix_null, dg_null, n_runs=n_runs).mean()

        # Rank correlation
        common = [n for n in sampled_nodes_500 if n in ic_null]
        rho, _ = spearmanr(
            [ic_scores_real[n] for n in common],
            [ic_null[n] for n in common]
        )
        rho_results.append(rho)

        # Typology on null
        ic_null_arr = np.array([ic_null.get(n, 0) for n in common])
        ic_null_cut    = np.quantile(ic_null_arr, 0.80)
        views_arr      = np.array([views_col.get(n, 0) for n in common])
        views_null_cut = np.quantile(views_arr, 0.80)
        hidden_mask_null = (ic_null_arr >= ic_null_cut) & (views_arr < views_null_cut)
        # Report betweenness of Hidden nodes in null graph
        hidden_bet_null = [G_null.degree(common[i]) for i, m in enumerate(hidden_mask_null) if m]
        hidden_betweenness_nulls.append(np.mean(hidden_bet_null) if hidden_bet_null else 0)

    print(f"IC rank corr real vs null: ρ = {np.mean(rho_results):.3f} ± {np.std(rho_results):.3f}")
    print(f"Mean degree of Hidden nodes in null: {np.mean(hidden_betweenness_nulls):.2f}")
    print("Interpretation:", "Degree dominates" if np.mean(rho_results) > 0.8 else "Higher-order structure matters")

    return rho_results, hidden_betweenness_nulls
```

---

## 6. Community Detection — Bắt buộc cho Bridge Claims

```python
import community as community_louvain  # python-louvain

def compute_community_features(G_nx, resolution=1.0, seed=42):
    """
    Louvain community detection — O(N log N), vài phút trên 168k nodes.
    BẮT BUỘC để support claim "Hidden nodes are cross-community bridges."
    """
    partition = community_louvain.best_partition(G_nx, resolution=resolution,
                                                  random_state=seed)
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

### Group 5: GNN — Restructured Ablation

> **Thay đổi quan trọng từ v2:** Primary GNN dùng raw attributes (không phải centrality features) để story "GNN learns beyond hand-crafted features" defensible hơn.

| Variant        | Features                          | Role                                  |
| -------------- | --------------------------------- | ------------------------------------- |
| GNN-raw-attr   | views_log, views/day, life_time   | **Primary proposed**                  |
| GNN-graph-only | degree_norm only (or random init) | Ablation: topology without attributes |
| GNN-centrality | degree, PR, kshell                | Ablation: centrality features         |
| GNN-full       | all 6 features                    | Supplementary upper bound             |

**Tại sao cấu trúc này:**

- GNN-raw-attr vs MLP-raw-attr: giá trị của **message passing**
- GNN-raw-attr vs GNN-graph-only: giá trị của **attributes**
- GNN-raw-attr vs centrality baselines: giá trị của **learned representations**

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

### 8.4 Multiple Testing Correction

```python
from statsmodels.stats.multitest import multipletests

# Apply BH-FDR correction cho tất cả MWU tests
p_values_raw = [...]  # collect tất cả raw p-values
rejected, p_corrected, _, _ = multipletests(p_values_raw, alpha=0.05, method='fdr_bh')
# Report p_corrected, NOT p_raw
```

### 8.5 Repeated Training Seeds

```python
# 5 training seeds để report mean ± std
training_seeds = [42, 123, 456, 789, 1024]

results_per_seed = []
for seed in training_seeds:
    set_all_seeds(seed)
    model = train_graphsage(train_data, seed=seed)
    metrics = evaluate(model, test_data)
    results_per_seed.append(metrics)

# Paper viết: "All metrics averaged over 5 random training seeds (mean ± std)"
mean_spearman = np.mean([r['spearman'] for r in results_per_seed])
std_spearman  = np.std( [r['spearman'] for r in results_per_seed])
```

---

## 9. GNN Training

### 9.1 Architecture: GraphSAGE Regression

```python
import torch, torch.nn as nn
from torch_geometric.nn import SAGEConv

class GraphSAGESurrogate(nn.Module):
    """
    Primary task: Regression trên log(IC_score + 1).
    Loss: Huber Loss (robust với IC simulation noise).
    Features: raw attributes ONLY (views_log, views/day, life_time).
    """
    def __init__(self, in_dim=3, hidden_dim=128, n_layers=2, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim, aggr='mean'))
        for _ in range(n_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr='mean'))
        self.convs.append(SAGEConv(hidden_dim, 1, aggr='mean'))
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = self.act(self.dropout(conv(x, edge_index)))
        return self.convs[-1](x, edge_index).squeeze(-1)

criterion = nn.HuberLoss(delta=1.0)  # Robust với IC noise
```

### 9.2 Training Protocol

```python
# Split: degree-stratified, transductive setting
train_idx, test_idx = train_test_split(
    labeled_node_indices,
    test_size=0.20,
    stratify=degree_quintile_labels,
    random_state=42
)

# Full graph passed to GNN (transductive); masks restrict loss to labeled nodes
# Primary metric: Spearman ρ on held-out labeled test nodes
```

### 9.3 Runtime Table — Tách feature precompute time

```
| Component                          | Time   | Notes                    |
|------------------------------------|--------|--------------------------|
| Feature precompute (degree, PR, kshell) | X min | Centrality baselines only |
| MC IC labeling (n_sample × N_runs)      | X h   | One-time cost            |
| GNN training (5 seeds)                  | X min | With GPU                 |
| GNN inference (168,114 nodes)           | X sec | All active nodes         |
| Node2Vec training                        | X min |                          |
| Speedup: MC IC vs GNN inference         | **Zx** | For repeated evaluation  |
```

**QUAN TRỌNG:** Nếu GNN-raw-attr là primary, không cần centrality precompute → runtime so sánh fair hơn.

---

## 10. External Validation — Life_time

### 10.1 Quy tắc quan trọng

> **life_time xuất hiện ở 2 vai trò — phải tách rõ:**
>
> - **Validate IC-based typology** (IC labels KHÔNG dùng life_time → genuinely independent) ✅
> - **KHÔNG validate GNN-full predictions** nếu GNN-full đã thấy life_time trong features ❌

**Cho GNN-raw-attr (primary proposed):** life_time là feature → KHÔNG dùng để validate GNN predictions.
**Cho IC typology:** life_time genuinely independent → dùng làm external corroboration.

### 10.2 Implementation

```python
def validate_typology_with_lifetime(df_typology):
    """
    Validate IC-based typology bằng life_time.
    IC labels KHÔNG dùng life_time → genuinely independent.
    """
    # Method 1: Partial Spearman — IC rank vs life_time | degree
    ic_rank  = rankdata(df_typology['ic_score_mean'])
    deg_rank = rankdata(df_typology['degree'])
    lt_rank  = rankdata(df_typology['life_time'])

    coef = np.polyfit(deg_rank, ic_rank, 1)
    ic_residual = ic_rank - np.polyval(coef, deg_rank)
    rho_partial, p_partial = spearmanr(ic_residual, lt_rank)

    # Method 2: Stratified MWU by degree quintile
    df = df_typology.copy()
    df['deg_q'] = pd.qcut(df['degree'], q=5, labels=False, duplicates='drop')
    strat_results = []

    for q in range(5):
        sub = df[df['deg_q'] == q]
        h = sub[sub['typology'] == 'hidden']['life_time']
        o = sub[sub['typology'] == 'overrated']['life_time']
        if len(h) < 10 or len(o) < 10:
            continue
        _, p = mannwhitneyu(h, o, alternative='greater')
        nh, no = len(h), len(o)
        delta = (sum(hi > oi for hi in h for oi in o)
                 - sum(hi < oi for hi in h for oi in o)) / (nh * no)
        strat_results.append({'quintile': q, 'n_h': nh, 'n_o': no,
                               'p_raw': p, 'delta': delta,
                               'sig': p < 0.05 and abs(delta) >= 0.20})

    # BH correction
    p_vals = [r['p_raw'] for r in strat_results]
    _, p_corr, _, _ = multipletests(p_vals, method='fdr_bh')
    for i, r in enumerate(strat_results):
        r['p_corrected'] = p_corr[i]

    n_sig = sum(r['sig'] for r in strat_results)
    return rho_partial, strat_results, n_sig
```

**Framing nếu validation không significant:**

> _"The life_time correlation, while not statistically significant at the chosen effect size threshold, provides suggestive evidence. This may reflect that account tenure captures platform experience rather than influence potential."_

---

## 11. Structural Profiling — Hidden vs Overrated

```python
def structural_profile_comparison(df_typed, G_nx):
    """Profile Hidden vs Overrated nodes trên structural features."""
    hidden   = df_typed[df_typed['typology'] == 'hidden']
    overrated = df_typed[df_typed['typology'] == 'overrated']

    cols = ['degree', 'pagerank', 'kshell', 'betweenness',
            'cross_community_edge_fraction',  # từ community detection
            'life_time']

    for col in [c for c in cols if c in df_typed.columns]:
        h_vals = hidden[col].dropna()
        o_vals = overrated[col].dropna()
        if len(h_vals) < 10 or len(o_vals) < 10:
            continue
        stat, p = mannwhitneyu(h_vals, o_vals)
        nh, no = len(h_vals), len(o_vals)
        delta = (sum(hi > oi for hi in h_vals for oi in o_vals)
                 - sum(hi < oi for hi in h_vals for oi in o_vals)) / (nh * no)
        print(f"{col:35s}: δ={delta:+.3f}, p={p:.4f}")

    # Expected findings:
    # Hidden → higher betweenness, higher cross_community_fraction
    # Overrated → higher degree, higher views, but peripheral topology
```

---

## 12. Research Questions (Final)

### RQ1 — IC Operationalization Quality

**Câu hỏi:** Does weighted-cascade IC simulation produce a sufficiently discriminative and stable influence ranking for use as a surrogate target?

**Method:** Pilot diagnostics (CV, top-decile/median ratio, rank stability). Label Jaccard across 3 MC seeds.

**Success criterion:** CV > 0.3, label Jaccard > 0.85.

### RQ2 — Divergence Analysis

**Câu hỏi:** To what extent do popularity metrics (views) agree with diffusion-based influence rankings?

**Method:** 2×2 typology (IC high/low × views high/low). Structural profiling (Hidden vs Overrated). life_time external corroboration (validates typology only). Null model: compare typology quadrant profiles real vs configuration model.

**RQ2 Fallback nếu ρ(views, IC) cao (> 0.8):**

> _"We find high agreement between popularity and diffusion rankings (ρ > 0.8), suggesting that on Twitch's dense social graph, structural influence and popularity are largely aligned. The small divergent subset (Hidden influencers) shows systematically higher betweenness and cross-community connectivity."_

### RQ3 — GNN Surrogate Quality

**Câu hỏi:** Can GNN approximate simulation-defined influence rankings more accurately than analytical proxies, and what is the computational gain?

**Method:** Compare Spearman ρ, NDCG@10% của all baselines. Report runtime: MC IC (hours) vs GNN inference (seconds).

**Prepared narratives:**

_Nếu GNN-raw-attr > two-hop proxy:_

> _"GNN captures higher-order neighborhood patterns beyond 2-hop analytical approximations, demonstrating the value of learned representations for influence estimation."_

_Nếu GNN-raw-attr ≤ two-hop proxy:_

> _"We find that 2-hop analytical spread approximation (naive full-graph complexity gần O(Σ d(v)^2)) achieves ρ ≈ [X] with MC IC scores, closely matching GNN surrogate performance while requiring no training. This suggests weighted-cascade dynamics are well-approximated by local structural summaries. GNN's value lies primarily in efficient inference as network evolves."_

_Cả hai outcomes đều publishable tại MAPR với framing đúng._

### RQ4 — User Profile Analysis

**Câu hỏi:** What structural characteristics distinguish users whose popularity rank disagrees with diffusion rank?

**Method:** Structural profiles of Hidden vs Overrated. Community bridging positions (cross_community_edge_fraction). life_time comparison với degree control.

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

## 14. Paper Structure (6 trang IEEE Double-blind)

### Core Story — Phải fit trong 6 trang

```
Must-have (không thể cắt):
1. Weighted cascade operationalization + pilot diagnostics
2. Label stability + bootstrap CI
3. Divergence analysis: views vs IC (typology + null model)
4. Baseline comparison table (all groups)
5. Runtime story
6. life_time external corroboration (typology only)
7. Honest limitations

Optional (cắt nếu thiếu trang):
- Uniform p sensitivity variant
- 2nd/3rd threshold analysis
- Detailed betweenness profiling
- Secondary metrics
```

### Section 1 — Introduction (0.5 trang)

- Hook: Defining power users in static networks without behavioral data
- Twitch context + use case (game publishers, platform recommendation)
- "Why Twitch not Twitter?": No public retweet data; gaming communities have tight-knit structure; MUSAE is standard benchmark
- 3 specific contributions

### Section 2 — Related Work (0.5 trang)

| Reference                       | Role trong paper                |
| ------------------------------- | ------------------------------- |
| Kempe et al. (2003)             | IC model foundation             |
| Kitsak et al. (2010)            | k-shell spreaders               |
| Hamilton et al. (2017)          | GraphSAGE                       |
| Ling et al., ICML 2023 (DeepIM) | Weighted cascade setup          |
| Guille et al. (2013) §4         | Evaluation without ground truth |
| Aral & Walker (2012)            | Social ties and influence       |
| Rozemberczki et al. (2021)      | Twitch MUSAE dataset            |

### Section 3 — Methodology (1.5 trang)

**Figure 1 (bắt buộc):** Pipeline — raw data → IC operationalization → divergence typology → GNN surrogate

**Table 1 — Independence Matrix:**
| Component | Views-independent? | IC-independent? |
|---|---|---|
| Weighted cascade p | ✅ Yes (structure-only) | N/A |
| IC labels | ✅ Yes | N/A |
| GNN-raw-attr features | ❌ No (uses `views_log_norm`, `views_per_day_norm`) | ✅ Yes |
| life_time validation (of typology) | ✅ Yes (with degree control) | ✅ Yes |
| Null model | ✅ Yes | ✅ Yes |

**Evaluation statement (bắt buộc):**

> _"We evaluate in a transductive node-level regression setting. IC labels are available for N_sample stratified nodes. All accuracy metrics are computed on held-out labeled nodes only. Full-graph inference is reported for runtime assessment."_

### Section 4 — Experiments (2.5 trang)

- 4.1 Setup: hardware, runtime benchmark, pilot diagnostics
- 4.2 RQ1: IC operationalization quality (pilot diagnostics table)
- 4.3 RQ2: Divergence analysis + null model (Figure 2: scatter plot IC vs views)
- 4.4 RQ3: Main results table + GNN ablation + runtime comparison
- 4.5 RQ4: Structural profiles (Hidden vs Overrated)
- 4.6 life_time external corroboration
- 4.7 Sensitivity: one-hop vs two-hop correlation

### Section 5 — Discussion & Limitations (0.5 trang)

Bốn limitations bắt buộc (từ Section 1.3 trên). Thêm:

> _"Why not learn p from data? Learning p requires supervised diffusion logs unavailable in this dataset. Weighted cascade provides a principled zero-shot alternative with theoretical backing (Kempe et al., 2003; Ling et al., 2023)."_

Ethics:

> _"All data are publicly available under MIT License and anonymized. This study does not involve human subjects or interaction with live users."_

### References (≤12)

Kempe 2003, Kitsak 2010, Hamilton 2017, Rozemberczki 2021, Guille 2013, Aral&Walker 2012, DeepIM 2023 (full citation), Benjamini&Hochberg 1995, Cliff 1993, Blondel 2008, Grover 2016.

---

## 15. Experiment Configuration (Final)

```yaml
# experiment.yaml v3.0

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

# Sensitivity only
p_sensitivity: uniform
kappa_target: 2 # p_uniform = 2/mean_degree ≈ 0.025

# ─── Sampling ──────────────────────────────────────────────────
sample_size_primary: 5000 # adjust based on Day 1 benchmark
sample_size_typology_min: 8000 # expand if Hidden quadrant < 150 nodes
sampling_strategy: degree_quintile_stratified
ks_test_threshold: 0.10

# ─── IC Runs ───────────────────────────────────────────────────
mc_runs_primary: 200 # adjust based on benchmark
mc_runs_label_stability: 150 # for 3-seed Jaccard check
mc_runs_null_model: 100
n_label_stability_seeds: 3 # reduced from 5 for compute efficiency
n_null_realizations: 3 # mean ± std for null model
label_stability_target_jaccard: 0.85
min_quadrant_size: 150

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
louvain_seed: 42
compute_cross_community_fraction: true

# ─── Baselines ─────────────────────────────────────────────────
baselines:
  group1_raw: [views_rank, views_per_day_rank, degree_rank]
  group2_central: [pagerank, kshell, betweenness_approx]
  group3_proxies: [one_hop_spread, two_hop_spread] # NOT weighted_degree (redundant)
  group4_embed: [node2vec_lr, mlp_raw_attr]
  group5_gnn: [gnn_raw_attr, gnn_graph_only, gnn_centrality, gnn_full]

# Node2Vec (reduced for speed)
node2vec_dim: 64
node2vec_walks: 20 # NOT 200 (too slow)
node2vec_walk_length: 20

# ─── GNN ───────────────────────────────────────────────────────
gnn_primary_variant: gnn_raw_attr # PRIMARY: raw attributes only
gnn_model: graphsage
gnn_aggregation: mean
gnn_n_layers: 2
gnn_hidden_dim: 128
gnn_dropout: 0.30
gnn_lr: 0.001
gnn_epochs: 200
gnn_loss: huber
gnn_huber_delta: 1.0
gnn_train_test_split: 0.80
gnn_cv_folds: 5
gnn_training_seeds: [42, 123, 456, 789, 1024] # 5 seeds → report mean ± std

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

# ─── Evaluation ────────────────────────────────────────────────
eval_setting: transductive # accuracy on held-out labeled nodes ONLY
primary_metrics: [spearman_rho, ndcg_10, precision_10]
runtime_metrics: [mc_ic_labeling_sec, gnn_training_sec, gnn_inference_all_sec]
avoid_metrics: [accuracy, f1_macro]
multiple_testing_correction: benjamini_hochberg
fdr_alpha: 0.05

# ─── External Validation ───────────────────────────────────────
lifetime_validate_target: typology # NOT gnn_predictions (dependency issue)
lifetime_degree_quintiles: 5
cliffs_delta_threshold: 0.20
lifetime_n_quintiles_significant_target: 3
```

---

## 16. Timeline 25 ngày (6/4 – 30/4)

### ⚠️ Nguyên tắc không thể phá vỡ

1. **Ngày 6/4 sáng:** Benchmark IC runtime → quyết định sample size
2. **Ngày 6/4 chiều:** One-hop baseline correlation check → quyết định GNN narrative
3. **IC simulation chạy liên tục ở background từ Ngày 8**
4. **Bắt đầu viết Introduction/Related Work/Methodology từ Ngày 8** (không đợi đến Ngày 15)

| Ngày          | Track A: Data & IC                             | Track B: Baselines & Community           | Track C: GNN & Paper                 |
| ------------- | ---------------------------------------------- | ---------------------------------------- | ------------------------------------ |
| **6/4**       | **#1: IC benchmark**                           | Setup NetworKit, betweenness (bg)        | Setup PyG                            |
| **6/4 chiều** | **#2: One-hop ρ check** → decide GNN narrative | PageRank, k-shell                        | PyG smoke test                       |
| **7/4**       | Dead account audit, LCC, sampling              | One-hop & 2-hop spread                   | Related work draft                   |
| **8/4**       | Stratified 5k sample + KS test                 | Community detection (Louvain)            | **Bắt đầu viết Intro + Methodology** |
| **9/4**       | P model pilot diagnostics (CV check)           | Community features (cross-comm fraction) | Methodology section draft            |
| **10/4**      | **IC primary bắt đầu chạy background**         | MLP baselines                            | Methodology finalize                 |
| **11-12/4**   | IC đang chạy; label stability check            | Node2Vec (dim=64, walks=20)              | Figure 1 (pipeline)                  |
| **13/4**      | IC DONE; bootstrap CI; quadrant sizing         | Null model (3 realizations)              | Internal checkpoint                  |
| **14/4**      | Expand to 8-10k nếu Hidden < 150               | Finalize all baselines                   | Related work finalize                |
| **15-16/4**   | GNN training (5 seeds, all ablations)          | Structural profiles (Hidden/Overrated)   | Experiment section draft             |
| **17-18/4**   | life_time validation (typology only)           | Null model typology comparison           | Results tables & figures             |
| **19-20/4**   | Uniform p sensitivity (parallel với GNN)       | Runtime measurement & logging            | Discussion & Limitations             |
| **21/4**      | All experiments locked                         | All results locked                       | Paper draft complete                 |
| **22-23/4**   | Internal review (tất cả thành viên đọc)        | —                                        | Revision round 1                     |
| **24-25/4**   | Fix issues                                     | —                                        | Revision round 2                     |
| **26/4**      | IEEE format check (6 trang, margins, fonts)    | —                                        | Double-blind verify                  |
| **27/4**      | Final read-through                             | —                                        | Submit dry-run                       |
| **28-30/4**   | Buffer + last fixes                            | —                                        | **30/4: SUBMIT**                     |

### Scope Reduction — Phải sẵn sàng cắt nếu tight

| Cắt được                        | Giữ bắt buộc                                       |
| ------------------------------- | -------------------------------------------------- |
| Attribute-informed p variant    | Weighted cascade IC (primary)                      |
| Graph perturbation test         | Label stability (Jaccard)                          |
| 5%/15% thresholds (chỉ giữ 10%) | Null model (3 realizations)                        |
| Eigenvector redundancy check    | One-hop + two-hop proxies                          |
| Betweenness trong GNN features  | Community detection (Louvain)                      |
| GNN-full variant                | GNN-raw-attr (primary) + GNN-graph-only (ablation) |

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
│       ├── classification_labels.parquet # binary top-10%
│       └── typology_labels_ic_views.parquet   # 2×2 IC×views
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
│   │   ├── label_stability.py
│   │   └── null_model.py             # 3 realizations
│   ├── graph/
│   │   ├── centrality.py             # NetworKit betweenness
│   │   ├── community.py              # Louvain + cross_comm fraction
│   │   └── diffusion_proxies.py      # one-hop + two-hop
│   ├── models/
│   │   ├── graphsage.py              # PyG, raw-attr primary
│   │   └── baselines.py              # LR, MLP, Node2Vec+LR
│   ├── evaluation/
│   │   ├── ranking_metrics.py        # Spearman, NDCG, P@k
│   │   ├── external_validation.py    # life_time (typology only)
│   │   ├── structural_profiles.py
│   │   └── multiple_testing.py       # BH correction
│   └── visualization/
│       ├── typology_scatter.py
│       └── runtime_bar.py
│
├── outputs/
│   ├── stage0_data_quality/           # dead accounts, LCC, views dist
│   ├── stage1_centrality/             # centrality + community features
│   ├── day1_benchmark/                # IC runtime + one-hop ρ — CRITICAL
│   ├── stage2_ic_labels/              # pilot diagnostics, stability, bootstrap CI
│   ├── stage3_typology/               # quadrant sizes, null model comparison
│   ├── stage4_gnn/                    # all GNN variants, 5 seeds each
│   ├── stage5_validation/             # life_time, structural profiles
│   └── stage6_sensitivity/            # uniform p variant
│
├── paper/
│   ├── main.tex                       # IEEE two-column, double-blind
│   ├── figures/                       # Fig1: pipeline, Fig2: typology scatter
│   └── tables/                        # Table1: independence, Table2: results
│
├── docs/
│   ├── experiment_registry.md         # mọi decision phải ghi
│   └── day1_decisions.md              # IC benchmark + one-hop ρ outcomes
│
├── config/
│   └── experiment.yaml                # v3.0 config (Section 15)
│
└── README.md
```

---

## 18. Pre-Submission Checklist

### Blocker (reject nếu thiếu)

- [ ] **Ngày 6/4:** IC benchmark + one-hop ρ check → ghi vào `docs/day1_decisions.md`
- [ ] Construct validity paragraph trong Section 3.1 (follower ≠ diffusion channel)
- [ ] Directionality `graph_directed: false` + justification trong paper
- [ ] `calibration_mode: variance_check` — KHÔNG có `calibration_target_reach_pct: 0.08`
- [ ] DeepIM chỉ cited cho weighted cascade formula, KHÔNG cho 8% reach target
- [ ] IC backend = CSR + loky (KHÔNG phải NetworkX threads)
- [ ] Label stability: Jaccard > 0.85 across 3 MC seeds
- [ ] Quadrant size: Hidden ≥ 150 nodes; expand sample nếu cần
- [ ] Null model: 3 realizations, compare TYPOLOGY QUADRANTS real vs null
- [ ] Baseline Group 3: one-hop + 2-hop (KHÔNG phải weighted degree — redundant)
- [ ] GNN primary = GNN-raw-attr (views_log, views/day, life_time — no centrality features)
- [ ] Transductive setting stated rõ trong paper
- [ ] Accuracy metrics trên held-out LABELED nodes only; full-graph = runtime story only
- [ ] life_time validates IC TYPOLOGY only — KHÔNG validate GNN-full predictions
- [ ] Runtime table tách: feature precompute / GNN training / GNN inference / MC IC
- [ ] BH correction cho tất cả MWU tests
- [ ] 5 training seeds, report mean ± std
- [ ] Community detection (Louvain) → cross_community_edge_fraction
- [ ] Dead account statistics trong limitations
- [ ] Fallback narrative cho RQ3 nếu GNN ≤ cheap proxies
- [ ] Fallback narrative cho RQ2 nếu views/IC highly correlated

### Strongly Recommended

- [ ] Bootstrap CI cho IC scores (30 phút implement)
- [ ] Two-sample strategy nếu Hidden quadrant nhỏ
- [ ] Node2Vec: dim=64, walks=20 (không phải 200)
- [ ] Betweenness chỉ dùng cho structural profiling, không bắt buộc là GNN feature

### IEEE Format

- [ ] ≤ 6 trang kể cả figures, tables, references
- [ ] Double-blind: không tên, trường, acknowledgments trong submission PDF
- [ ] Figures readable grayscale ở kích thước nhỏ
- [ ] Abstract ≤ 150 words
- [ ] References: IEEE format [1], [2], ...

---

## 19. Risk Management

| Rủi ro                                                                        | Xác suất   | Impact       | Action                                                                               |
| ----------------------------------------------------------------------------- | ---------- | ------------ | ------------------------------------------------------------------------------------ |
| One-hop ρ > 0.9 + top-k alignment cao (`Jaccard@10% > 0.8`, `NDCG@10% > 0.9`) | Trung bình | **Critical** | Ngày 6/4: check trước; nếu đủ 3 điều kiện thì restructure, nếu không giữ GNN + 2-hop |
| IC runtime > 8h                                                               | Trung bình | **Critical** | Reduce: n_sample=2k, N_runs=100; log limitation                                      |
| GNN không beat cheap proxies                                                  | Trung bình | Thấp         | Prepare negative-result narrative (RQ3)                                              |
| Hidden quadrant < 150 nodes                                                   | Trung bình | Cao          | Expand to 8-10k; two-sample strategy                                                 |
| views/IC highly correlated                                                    | Trung bình | Thấp         | Prepare fallback narrative (RQ2)                                                     |
| loky OOM với large graph                                                      | Thấp       | Cao          | Reduce n_jobs; monitor RAM                                                           |
| PyG installation issues                                                       | Thấp       | Trung bình   | Setup Ngày 6/4 sáng; fallback DGL                                                    |
| Paper > 6 pages                                                               | Trung bình | Blocker      | Cut sensitivity variants, shorten ablation                                           |

---

## 20. Decision Log Template

```markdown
# docs/day1_decisions.md

## [6/4/2026] IC Runtime Benchmark

Per-simulation time: ** ms
Projected (n_sample × N_runs): ** hours
Decision: n_sample = **, N_runs = **
Adjusted timeline: \_\_

## [6/4/2026] One-Hop Baseline Reality Check

One-hop vs IC pilot: ρ = **
Jaccard@10% = **
NDCG@10% = **
Decision: [GNN story viable / GNN+2hop head-to-head / Restructure narrative]
If restructure: new paper angle = **
```

---

## 21. Phân Công Team

| Người | Track     | Ngày 6–12/4                                        | Ngày 13–21/4                     | Ngày 22–30/4     |
| ----- | --------- | -------------------------------------------------- | -------------------------------- | ---------------- |
| 1     | Data + IC | **Day 1 benchmarks**, preprocessing, IC (bg)       | IC finalize, quadrant sizing     | Writing support  |
| 2     | Data + IC | Sampling + KS, pilot diagnostics, stability        | Null model (3 realizations)      | Writing support  |
| 3     | Baselines | Betweenness (bg), PageRank, k-shell, **community** | Structural profiles              | Results tables   |
| 4     | Baselines | One-hop, 2-hop, Node2Vec, MLP                      | Evaluation metrics, runtime      | Figures          |
| 5     | GNN       | PyG setup, GNN-raw-attr training                   | 5-seed ablation, GNN results     | Paper Sec 3-4    |
| 6     | Paper     | **Intro + Related Work từ Ngày 8**                 | life_time validation, Discussion | Paper Sec 1-2, 5 |

**Daily standup:** 15 phút (không phải 30-45).
**Milestone bắt buộc cuối mỗi tuần:** artifact cụ thể, không chỉ code.

---

_Document version: 3.0_
_Changes from v2: Remove 8% calibration target → variance check; Add Day 1 one-hop ρ check;_
_Fix joblib: CSR + loky; Fix null model: typology comparison; Fix baseline Group 3: 2-hop proxy;_
_Fix title; Clarify transductive evaluation; Fix life_time independence; Restructure GNN ablation;_
_Fix random seed handling; Add Louvain community; Add 5-seed training; Fix runtime table;_
_Reduce Node2Vec params; Add two-sample typology strategy; Writing starts Day 8_
_Tổng hợp từ: Expert SNA Review rounds 1–4_
_Bắt đầu: 6/4/2026 | Deadline: 30/4/2026_
