# MAPR2026 v3 — Migration Checklist (SIS-based → IC-based)

Mục tiêu của checklist này là “đóng băng” **artifact contract tối thiểu** và **thứ tự stage cần viết/đổi** để chuyển pipeline cũ (SIS-based) sang pipeline mới (IC-based) mà không bị đứt mạch.

> Quy ước: “artifact” = file đầu ra được downstream dùng lại. Nếu artifact chưa tồn tại / không đúng schema → stage sau coi như không chạy được.

---

## M0. Quyết định lock trước khi code (chốt tại buổi kick-off)

> Ghi kết quả vào `docs/m0_decisions.md` và commit trước khi ai bắt đầu code.

| Quyết định | Giá trị đã lock | Ghi chú |
|---|---|---|
| `test_frac` | **0.20** | arg `--test-frac` trong `ic_labels_primary.py` |
| `stratify_by` | **degree_quintile** (pd.qcut q=5, duplicates='drop') | áp dụng khi tạo split mask |
| `split_seed` | **42** | arg `--seed` trong `ic_labels_primary.py` |
| `classification_threshold` | **top 10%** (quantile 0.90) | dùng cho `y_top10` và typology axis |
| `min_quadrant_size` | **150** | stop condition của Person 2 |
| `n_sample` (default) | **5.000** labeled nodes | arg `--n-sample` trong `ic_labels_primary.py` |
| **Proxies scope** | **FULL active graph** | `diffusion_proxies.parquet` cover tất cả active nodes |
| `runtime_sec` definition | **full-graph inference only** | không tính precompute/training |

| Quyết định | Giá trị | Khi nào quyết |
|---|---|---|
| `N_seeds × N_runs` | TBD | **M2** (sau Day-1 runtime benchmark) |
| GNN narrative branch | TBD | **M2** (sau one-hop ρ check) |
| Uniform-p sensitivity | TBD | M2 |

---

## A0. Dead Account Audit (Stage 0 prerequisite — bắt buộc trước sampling)

> **Cần làm trước bất kỳ IC simulation nào.** Stats từ đây đi vào Section 5 (Limitations) của paper.

- Script: `src/data/dead_account_audit.py` (hoặc inline trong Stage 0 preprocessing)
- Input: `data/raw/large_twitch_features.csv` (có cột `dead_account`)
- Output: `outputs/stage0_data_quality/dead_account_report.json`

**Contract tối thiểu:**
```json
{
  "n_dead": <int>,
  "n_live": <int>,
  "pct_dead": <float>,
  "mean_degree_dead": <float>,
  "mean_degree_live": <float>,
  "mean_views_dead": <float>,
  "mean_views_live": <float>
}
```

**Paper framing bắt buộc (Section 5 Limitations):**
> *"Dead accounts (X% of nodes) were excluded; they have systematically lower degree and views than active accounts. Findings generalize only to active users."*

---

## A. Bạn được reuse gì từ pipeline cũ (đã có sẵn trong repo)

Các artifact này đã tồn tại và **được xem là input nền tảng** cho MAPR2026 v3 (trừ khi bạn quyết định đổi rule preprocess):

### A1) Stage 0 — Processed graph + node attributes

- `data/processed/graph_active.edgelist`
- `data/processed/node_attributes.parquet`
- `outputs/stage0_data_quality/raw_load_report.json`
- `outputs/stage0_data_quality/metrics.json`
- `outputs/stage0_data_quality/preprocess_report.json`

**Contract tối thiểu**

- `graph_active.edgelist` đọc được bằng NetworkX/igraph.
- `node_attributes.parquet` có ít nhất: `node_id`, `views` (và ideally có thêm `life_time`, `language` nếu có trong raw features).

### A2) Stage 1–2 — Centrality + community + k-shell (Baselines Group 2)

- `data/processed/centrality_table.parquet`
- `data/processed/community_labels.parquet`
- `data/processed/kshell_table.parquet`
- `outputs/stage1/metrics.json`, `outputs/stage1/params.json`
- `outputs/stage2/metrics.json`, `outputs/stage2/louvain_stability_report.json`, `outputs/stage2/kshell_metrics.json`

**Contract tối thiểu**

- `centrality_table.parquet` có `node_id`, `degree`, `pagerank`, `betweenness`, `kshell`, `views`.
- `community_labels.parquet` map phủ 100% active nodes.

### A3) Stage 3 — SIS artifacts (không còn là “label” trong plan mới)

Các file Stage 3 hiện có chỉ còn hữu ích như **baseline cũ / sanity-check**.

- `data/processed/sis_table.parquet`
- `data/processed/typology_labels.parquet` (đây là typology SIS×views — không phải IC×views)
- `outputs/stage3/*` (robustness, null model comparison theo SIS-based)

---

## B. Artifact tối thiểu cần có để pipeline mới (IC-based) chạy được

MAPR2026 v3 coi **IC score (weighted cascade)** là operationalization chính. Tối thiểu cần thêm các artifact sau:

### B1) CSR graph artifact (để IC chạy nhanh và share được)

- `data/processed/graph_csr.npz` (hoặc format tương đương): chứa `indptr`, `indices`, `degrees`, và mapping `node_id ↔ row_index`.

**Contract tối thiểu**

- Có mapping deterministic giữa `node_id` (string) và CSR index (int).
- `degrees[i] == indptr[i+1]-indptr[i]`.

**Gợi ý keys trong `.npz` (để consumer dùng thống nhất):** `indptr`, `indices`, `degrees`, `node_ids` (với `node_ids[i]` là node_id của row i).

### B2) Day-1 decision artifacts (benchmark + one-hop check)

- `outputs/day1_benchmark/ic_runtime_benchmark.json` (per-sim ms + projected runtime + quyết định N_seeds×N_runs)
- `outputs/day1_benchmark/one_hop_correlation.json` (Spearman ρ + kết luận narrative branch)
- `docs/day1_decisions.md` (ghi lại decision)

### B3) IC labels artifacts (Task A)

Tất cả artifacts dưới đây được tạo bởi **Person 1** qua `ic_labels_primary.py`.

- `data/processed/ic_scores_primary.parquet`
  - columns tối thiểu: `node_id`, `ic_score_mean`, `ic_score_std`, `n_runs`, `p_model` (phải ghi rõ `”weighted_cascade”`)
  - columns strongly recommended: `ic_ci_lower`, `ic_ci_upper` (bootstrap 95% CI, `n_bootstrap=1000`)
  - scope: **n_sample labeled nodes** (mặc định 5.000)
- `data/processed/regression_targets.parquet`
  - columns tối thiểu: `node_id`, `y` với `y = log1p(ic_score_mean)`
- `data/processed/classification_labels.parquet`
  - columns tối thiểu: `node_id`, `y_top10` (1 nếu nằm trong top-10% `ic_score_mean`)

> **Xem thêm B7** — `split_masks.parquet` được tạo **cùng lúc** với B3, bởi cùng script `ic_labels_primary.py`. Không thể có B3 mà không có B7.

### B3b) Split mask artifact (M0-locked — tạo cùng với B3)

- `data/processed/split_masks.parquet`
  - columns tối thiểu: `node_id` (str), `split` (str: `'train'` hoặc `'test'`)
  - Rule cứng: `test_frac=0.20`, `stratify=degree_quintile` (pd.qcut q=5), `seed=42`
  - Scope: **labeled nodes only** (cùng set với `ic_scores_primary.parquet`)
  - Owner: **Person 1** (tạo cùng lúc với B3, trong `ic_labels_primary.py --out-mask`)

**Stop condition:** Nếu artifact này chưa tồn tại hoặc sai schema → Person 3 KHÔNG được tự tạo split để thay thế. Báo Person 1 tạo lại.

### B4a) Community features (BẮT BUỘC cho structural profiling — v3 Section 6)

Tạo trước typology vì structural profiling cần `cross_community_edge_fraction`.

- **Thêm vào `data/processed/node_attributes.parquet`** (hoặc file riêng `community_features.parquet`):
  - `community_id`: int — Louvain partition, `resolution=1.0`, `seed=42`
  - `cross_community_edge_fraction`: float — fraction of neighbors in a different community
- Script: `src/graph/community.py` (đã có sẵn trong repo)
- **Library: `python-louvain`** (`community.best_partition(G_nx, resolution=1.0, random_state=42)`) hoặc `cdlib`. KHÔNG dùng NetworkX Louvain (không reproducible với fixed seed).
- Scope: ALL active nodes (phủ 100%)
- **Lý do bắt buộc:** claim "Hidden nodes are cross-community bridges" không support được nếu thiếu cột này.

### B4b) IC×views typology (Task B)

- `data/processed/typology_labels_ic_views.parquet`
  - columns tối thiểu: `node_id`, `typology_label` (true/hidden/overrated/non), `ic_high`, `views_high`, `ic_score_mean`, `views`
  - scope: **tất cả labeled nodes** (không filter theo train/test)
  - threshold: top-10% cho cả IC và views (M0-locked, đồng bộ `classification_threshold=0.10`)

**Output phụ đi kèm:**
- `outputs/mapr2026_v3_results/structural_profiling.csv` — MWU + Cliff's Δ + p_corrected (BH-FDR) cho 6 columns: `degree, pagerank, kshell, betweenness, cross_community_edge_fraction, life_time`
- `outputs/mapr2026_v3_results/lifetime_validation.json` — partial Spearman + stratified MWU results

**CẢNH BÁO `life_time` (v3 Section 10):**
- DÙNG để validate IC-based typology (IC labels không thấy life_time → independent) ✅
- KHÔNG DÙNG để validate GNN-full predictions (GNN-full đã thấy life_time trong features) ❌

### B5) Cheap diffusion proxies (Baselines Group 3)

- `data/processed/diffusion_proxies.parquet`
  - columns tối thiểu: `node_id`, `one_hop_spread`, `two_hop_spread`

**Scope rule (M0-locked):** artifact này phải cover **TOÀN BỘ active nodes** (~168k), không chỉ labeled subset.

Lý do: `runtime_sec` trong `baseline_ranking_metrics.csv` đo full-graph inference để so sánh fair với GNN inference. Person 3 sẽ filter theo `split_masks.parquet` khi tính ranking metrics — Person 2 không cần filter.

Cần thêm output phụ: `outputs/mapr2026_v3_results/runtime_breakdown.csv` (ghi ít nhất `model_name, inference_sec_full_graph`).

### B6) Evaluation-ready tables (Task C metrics)

- `outputs/mapr2026_v3_results/baseline_ranking_metrics.csv`
  - columns tối thiểu: `model_name`, `spearman_rho`, `ndcg_at_10pct`, `precision_at_10pct`, `runtime_sec`
  - `runtime_sec` = full-graph inference only (M0-locked)

**Consumer rule (M0-locked):** Person 3 load metrics target từ `regression_targets.parquet`, filter theo `split_masks.parquet` dùng `load_split_mask()` + `apply_test_mask()` từ `eval_ranking_harness.py`. Không tạo split mới.

> Lưu ý: phần GNN/Node2Vec có thể “scope cut” nếu tight; nhưng **Group 3 proxies + ranking metrics** là bắt buộc (vì quyết định narrative phụ thuộc one-hop).


---

## C. Thứ tự stage cần viết/đổi để không đứt mạch (ưu tiên theo dependency)

Đây là thứ tự tối thiểu để bạn triển khai, sao cho mỗi stage tạo ra artifact mà stage sau cần.

### Stage 0 (đã có) — Graph + node attributes

Không đụng nếu không cần. Chỉ đảm bảo chạy ổn và artifact A1 tồn tại.

### Stage 1 (đã có) — Centrality + community + k-shell

Có thể reuse y nguyên nếu chưa đổi preprocessing.

### Stage 2 (MỚI, phải viết trước) — Export CSR + mapping

**Lý do:** mọi IC/proxy đều cần CSR + mapping.

- Input: `graph_active.edgelist`
- Output: `graph_csr.npz`

### Stage 3 (MỚI, “Day-1 critical”) — Benchmark runtime + one-hop ρ check

**Lý do:** quyết định toàn bộ compute budget và GNN narrative.

- Input: CSR + degrees (+ list nodes)
- Output: artifacts B2

### Stage 4 (MỚI) — IC pilot diagnostics + label stability + split mask

**Lý do:** validate Task A (IC labels đủ discriminative và ổn định) trước khi làm typology/GNN.

- Input: CSR
- Output:
  - `ic_scores_primary.parquet` (ít nhất trên pilot sample; có `ic_ci_lower/upper` nếu bootstrap)
  - `regression_targets.parquet`, `classification_labels.parquet`
  - **`split_masks.parquet`** — tạo ngay sau IC labels (M0-locked: 80/20, degree_quintile, seed=42)
  - stability report (Jaccard across MC seeds)

**Seed rules (critical correctness):**
- **Primary IC**: `worker_seed = 42 + node_index` — mọi lần chạy production dùng cùng seed per-node
- **Stability check**: `worker_seed = mc_seed * 10000 + node` với `mc_seed ∈ {0,1,2}` — 3 genuinely independent MC experiments, `n_runs=150` each
- **Pilot diagnostics**: `n_pilot_nodes=200`, `n_pilot_runs=50` — subset để verify non-degenerate
- `cv_noise_threshold=0.50`: node nào có per-node CV > 0.50 → exclude khỏi stability metrics

**Stop condition:** Nếu `split_masks.parquet` chưa tồn tại sau Stage 4 → Stage 7 (surrogate) không được chạy.

### Stage 4b (MỚI, song song với Stage 4) — Community detection features

**Lý do:** `cross_community_edge_fraction` bắt buộc cho structural profiling — không cần IC labels.

- Input: `data/processed/graph_active.edgelist`
- Output: thêm `community_id` + `cross_community_edge_fraction` vào `node_attributes.parquet`
- Script: `src/graph/community.py` (đã có sẵn)
- Library: `python-louvain` (`community.best_partition(G_nx, resolution=1.0, random_state=42)`)
- Params: Louvain `resolution=1.0`, `seed=42`
- **Có thể chạy song song với Stage 4 (không phụ thuộc IC labels)**

### Stage 5 (ĐỔI/VIẾT MỚI) — IC×views typology + structural profiling + life_time validation + null model

**Lý do:** Task B phụ thuộc IC labels + community features.

- Input: `ic_scores_primary.parquet` + `node_attributes.parquet` (phải có `community_id`, `cross_community_edge_fraction`)
- Output:
  - `typology_labels_ic_views.parquet` (all labeled nodes, threshold top-10%, M0-locked)
  - `outputs/mapr2026_v3_results/structural_profiling.csv` (MWU + Cliff's Δ + BH-FDR p_corrected)
  - `outputs/mapr2026_v3_results/lifetime_validation.json` (partial Spearman + stratified MWU)
  - `null_model_typology_summary.json` (**spec: 500 nodes × 3 realizations × 100 runs/node**)
- Scripts: `src/mapr2026_v3/typology_ic_views.py`, `src/mapr2026_v3/null_model_typology.py`

**Null model spec chi tiết (v3 Section 5):**
- 500 nodes: subsample từ labeled nodes
- 3 realizations: configuration model `seed=realization*100`
- 100 runs/node: đủ để ổn định IC estimate trên null graph
- So sánh: TYPOLOGY QUADRANT (không chỉ rank correlation) — câu hỏi: "null graph có Hidden với betweenness cao không?"

**Two-sample strategy (v3 Section 4.5):**
- Nếu Hidden quadrant < 150 sau lần sample đầu → tăng `n_sample` lên **8.000–10.000 nodes**, đồng thời augment Sample B (high-betweenness + low-views nodes từ full graph)
- Sample B candidates: `betweenness > quantile(0.70)` AND `views < quantile(0.30)` AND chưa có trong Sample A; lấy tối đa `min(500, min_size × 2)` nodes
- Sample B chỉ dùng cho typology analysis, KHÔNG dùng để train GNN

### Stage 6 (MỚI) — Diffusion proxies (one-hop + two-hop)

**Lý do:** baseline critical; cần để so sánh với GNN và để “fallback narrative”.

- Input: `graph_csr.npz`
- Output:
  - `data/processed/diffusion_proxies.parquet` — **FULL active graph** (M0-locked; không filter)
  - `outputs/mapr2026_v3_results/runtime_breakdown.csv` — ghi `model_name`, `inference_sec_full_graph`
- Script: `src/mapr2026_v3/diffusion_proxies.py` (real mode, không `--dry-run`)

**Stop condition:** Nếu output chỉ có labeled subset → rebuild. Person 3 cần full graph để đo runtime fair.

### Stage 7 (MỚI, optional depending on Day-1 decision) — Surrogate learning (Node2Vec/GNN)

Chỉ làm sau khi đã có **tất cả**:

- `data/processed/ic_scores_primary.parquet` (B3)
- `data/processed/split_masks.parquet` (B3b) — **bắt buộc**
- `data/processed/diffusion_proxies.parquet` (B5)
- `outputs/mapr2026_v3_results/baseline_ranking_metrics.csv` — ít nhất Group 1–2 (B6)
- Day-1 narrative branch đã lock (M2)

**Phân chia Group 4 vs Group 5 (quan trọng cho cấu trúc results table):**
- **Group 4 (Shallow Embedding Baselines):** Node2Vec+LR (`dim=64, walks=20, walk_len=20`) + MLP raw attr (`[views_log, views/day, life_time]`) → kết quả vào `baseline_ranking_metrics.csv` (cùng với Group 1–3)
- **Group 5 (GNN ablation, 4 variants):** GNN-raw-attr / GNN-graph-only / GNN-centrality / GNN-full → kết quả vào `surrogate_ranking_metrics.csv` (với `mean±std` 5 seeds)

**GNN hyperparams bắt buộc (v3 Section 9):**
```
Model:       GraphSAGE (PyTorch Geometric ≥ 2.5, torch ≥ 2.0)
hidden_dim:  128,  n_layers: 2,  dropout: 0.30
lr:          0.001,  epochs: 200
Loss:        HuberLoss(delta=1.0)
Hardware:    GPU ≥ 8GB VRAM; fallback: DGL nếu PyG fail
```

---

## D. Mapping “pipeline cũ” → “pipeline mới” (để reuse code tối đa)

**Reuse trực tiếp (không cần sửa):**
- Data loading/preprocess: Stage 0 (`src/data/`)
- Centrality/k-shell: Stage 1–2 (`src/graph/centrality.py`, `src/graph/kshell.py`)
- **Community detection: `src/graph/community.py`** — reuse Louvain, chỉ thêm `cross_community_edge_fraction` output
- Stats utilities: `src/evaluation/stats_tests.py` (BH correction, Cliff's delta)

**Cần fork / viết mới** (không sửa đè SIS-based):
- IC labels pipeline (weighted cascade + CSR): `src/mapr2026_v3/ic_labels_primary.py`
- Typology builder (trục IC thay SIS): `src/mapr2026_v3/typology_ic_views.py`
- Evaluation harness (ranking metrics thay F1): `src/mapr2026_v3/eval_ranking_harness.py`
- Diffusion proxies (one-hop + two-hop): `src/mapr2026_v3/diffusion_proxies.py`
- Surrogate learning (GNN ablation): `src/mapr2026_v3/run_surrogates.py`

---

## E. “Stop conditions” (để tránh làm tiếp khi contract chưa đạt)

Dừng và fix trước khi đi tiếp nếu:

| Condition | Triệu chứng | Action |
|---|---|---|
| CSR không deterministic | Rerun ra mapping khác | Fix `export_csr.py`: sort node_id trước khi build |
| `split_masks.parquet` thiếu | Person 3 raise `FileNotFoundError` | Person 1 chạy `ic_labels_primary.py --dry-run` ngay |
| `split_masks.parquet` sai schema | `load_split_mask()` raise `ValueError` | Person 1 regenerate |
| One-hop ρ > 0.9 + cố build GNN như primary | Narrative không defensible | Restructure: xem Section 2.2 plan v3 |
| IC pilot degenerate | CV < 0.3 hoặc top10/median ≈ 1 | Tăng n_runs hoặc restrict subgraph |
| Hidden quadrant < 150 | `check_and_expand_typology_sample` cảnh báo | Tăng n_sample hoặc đổi threshold tạm → ghi limitation |
| `diffusion_proxies.parquet` chỉ có labeled subset | `n_nodes` trong file << 168k | Person 2 rebuild ở real mode |

---

## F. “Minimal handoff package” giữa teammates (đỡ phải chạy lại)

Nếu Person 1 đã chạy IC labels, Person 2/3 cần tất cả các file dưới đây để không phải rerun:

**Từ Person 1 → Person 2 và 3:**
- `data/processed/graph_csr.npz`
- `data/processed/ic_scores_primary.parquet`
- `data/processed/regression_targets.parquet`
- `data/processed/classification_labels.parquet`
- **`data/processed/split_masks.parquet`** ← critical cho Person 3
- `data/processed/node_attributes.parquet`
- `outputs/day1_benchmark/ic_runtime_benchmark.json`
- `outputs/day1_benchmark/one_hop_correlation.json`
- `docs/day1_decisions.md` (chứa N_seeds, N_runs, narrative_branch)

**Từ Person 2 → Person 3:**
- `data/processed/diffusion_proxies.parquet` (full graph)
- `outputs/mapr2026_v3_results/runtime_breakdown.csv`

---

## F2. Risk Management (v3 Section 19)

| Rủi ro | Xác suất | Impact | Action |
|---|---|---|---|
| `one_hop_rho > 0.9` → GNN story invalid | Trung bình | **Critical** | M2: check trước; dùng prepared fallback narrative |
| IC runtime > 8h | Trung bình | **Critical** | M2: reduce 2k seeds, 100 runs; ghi limitation |
| GNN không beat cheap proxies | Trung bình | Thấp | Prepared "negative result" narrative — vẫn publishable |
| Hidden quadrant < 150 nodes | Trung bình | Cao | Expand 8–10k sample + Sample B (Stage 5) |
| `views/IC ρ > 0.8` | Trung bình | Thấp | Prepared "high agreement" fallback narrative |
| loky OOM với full graph | Thấp | Cao | Reduce `n_jobs`; RAM ≥ 32 GB khi chạy parallel IC |
| PyG installation issues | Thấp | Trung bình | Setup M0; fallback: DGL |
| Paper > 6 trang IEEE | Trung bình | **Blocker** | Cắt theo bảng Scope Reduction dưới |

## F3. Scope Reduction — Cắt theo thứ tự ưu tiên (v3 Section 16)

| Có thể cắt | Phải giữ bắt buộc |
|---|---|
| Uniform-p sensitivity variant | Weighted cascade IC (primary) |
| Graph perturbation test | Label stability (Jaccard ≥ 0.85, 3 seeds) |
| 5% / 15% thresholds | Null model (3 realizations × 500 × 100 runs) |
| Eigenvector centrality baseline | One-hop + two-hop proxies (Group 3) |
| GNN-full variant | Community detection (Louvain + cross_comm_fraction) |
| Bootstrap CI (strongly recommended, optional) | GNN-raw-attr + GNN-graph-only ablation |
| Detailed betweenness profiling | BH-FDR correction cho tất cả MWU p-values |
| Secondary metrics (Precision@10%) | life_time validation của IC typology |
| Dead account detailed breakdown | Dead account % stat trong limitations |

---

## G. Gợi ý triển khai (sát MAPR2026 v3 folder structure)

Repo hiện tại đang dùng `src/config/base.yaml` và `run_all.py` stages 0–7.
Để migrate “không đập vỡ” pipeline cũ:

- Giữ nguyên Stage 0–3 SIS-based (không sửa).
- Tất cả script mới cho MAPR2026 v3 nằm trong `src/mapr2026_v3/`.

**Mapping script ↔ Stage:**

| Stage v3 | Script | Owner | Ghi chú |
|---|---|---|---|
| **Stage 0b (dead account audit)** | `src/data/dead_account_audit.py` | **Person 1** | Phải có trước sampling; stats → limitations |
| Stage 2 (CSR) | `src/mapr2026_v3/export_csr.py` | Person 1 | |
| Stage 3 (Day-1) | `src/mapr2026_v3/day1_benchmark.py` | Person 1 | Gating cho M2 |
| Stage 4 (IC labels + split mask) | `src/mapr2026_v3/ic_labels_primary.py` | Person 1 | Gating cho M3 |
| **Stage 4b (community features)** | `src/graph/community.py` | **Person 2** | Độc lập với IC, chạy sớm |
| Stage 5 (typology + profiling + life_time) | `src/mapr2026_v3/typology_ic_views.py` | Person 2 | Cần IC labels + community |
| Stage 5 (null model) | `src/mapr2026_v3/null_model_typology.py` | Person 2 | 500 nodes × 3 × 100 runs |
| Stage 6 (proxies full graph) | `src/mapr2026_v3/diffusion_proxies.py` | Person 2 | Full active graph |
| Stage 7 (Group 1–4 baselines) | `src/mapr2026_v3/run_baselines.py` | Person 3 | Group 4 = Node2Vec+LR, MLP → baseline CSV |
| Stage 7 (Group 5 GNN ablation) | `src/mapr2026_v3/run_surrogates.py` | Person 3 | 4 GNN variants; mean±std → surrogate CSV |
| (shared) | `src/mapr2026_v3/eval_ranking_harness.py` | Person 3 | `load_split_mask()` + metrics |
| (shared) | `src/mapr2026_v3/_shared.py` | All | Đọc, không sửa riêng |
