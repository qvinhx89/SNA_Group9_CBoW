# MAPR2026 v3 — Team 3 người: Kế hoạch coding song song (không bao gồm viết paper)

Mục tiêu của tài liệu này là thiết kế các đầu việc **có thể triển khai song song tối đa** cho team 3 người để migrate từ pipeline hiện tại (SIS-based, stage 0–3 đã ổn) sang MAPR2026 v3 (IC-based operationalization + divergence analysis + surrogate learning).

Phạm vi:

- **Chỉ phần thực thi code + tạo artifacts + chạy pipeline**.
- Không bao gồm viết paper/narrative (đã có người khác phụ trách).

**Scope bridge:** Tài liệu này là execution plan cho team 3 người coding. `MAPR2026_Implementation_Plan_v3.md` là strategic master plan (research + narrative + publication framing). Nếu khác biệt ở thao tác thực thi hằng ngày, ưu tiên file này; nếu khác biệt về framing nghiên cứu/paper, ưu tiên master plan.

---

## Cách đọc file này — 4 mức độ task

> **Đọc phần này trước.** File plan có 4 loại nội dung với mức độ ưu tiên khác nhau:

| Loại                  | Ký hiệu                                   | Ý nghĩa                                                   | Khi nào thực thi                                    |
| --------------------- | ----------------------------------------- | --------------------------------------------------------- | --------------------------------------------------- |
| **MUST**              | _(không có marker — body text thường)_    | Bắt buộc cho paper defensible                             | Luôn làm theo đúng thứ tự                           |
| **Dự phòng**          | `> ⚠ [IF PROBLEM: <điều kiện>]`           | Phương án thay thế khi gặp vấn đề cụ thể                  | CHỈ khi điều kiện trigger xảy ra                    |
| **Nếu còn thời gian** | `> ✦ [IF TIME]`                           | Tăng thêm chất lượng/robustness                           | Sau khi hoàn thành tất cả MUST trước deadline       |
| **Tham khảo**         | `> 📋 [REFERENCE — không phải task thêm]` | Chỉ để align narrative/decision, không phát sinh task mới | Đọc để thống nhất ngữ cảnh, KHÔNG đưa vào task list |

> **Quy tắc vàng:** Lần đọc đầu — đọc body text bình thường, **bỏ qua toàn bộ các block `⚠ [IF PROBLEM]` và `✦ [IF TIME]`**. Quay lại các block đó khi và chỉ khi điều kiện trigger xảy ra hoặc còn thời gian thừa sau khi done MUST. Không để chúng làm phình scope hoặc trễ timeline.

---

## 0) Quickstart (để ai cũng chạy được trong 10 phút)

Mục tiêu: trước khi chia việc, cả team xác nhận **environment đúng** và **Stage 0 artifacts đã có**.

### 0.1 Environment

- Dùng Python **3.10–3.12** (khuyến nghị: conda env `sna_group9_cbow_py312` theo `environment.yml`).
- Quick check (PowerShell):

```powershell
conda activate sna_group9_cbow_py312
python --version
pip install -r requirements.txt
```

### 0.2 Base artifacts phải tồn tại (đầu vào chung)

Tối thiểu cần các file sau (Stage 0 của pipeline cũ):

- `data/processed/graph_active.edgelist`
- `data/processed/node_attributes.parquet` (tối thiểu có `node_id`, `views`)

Nếu thiếu, chạy lại Stage 0:

```powershell
python run_all.py --stage 0
```

Nếu Person 3 cần baselines Group 2 (PageRank/k-shell/betweenness), đảm bảo Stage 1–2 đã có:

```powershell
python run_all.py --stage 1
python run_all.py --stage 2
```

## 1) Nguyên tắc để “song song thật”

1. **Chia theo artifact contract**, không chia theo “ý tưởng”. Ai sở hữu artifact nào thì chịu trách nhiệm schema + reproducibility của artifact đó.
2. **Không sửa đè Stage 0–3 đang chạy ổn** trừ khi bắt buộc. Nếu cần đổi logic theo MAPR2026 v3, ưu tiên tạo script mới để tránh phá pipeline cũ.
3. Mọi người đều có thể code/test trước bằng **mock artifacts** (từ SIS/centrality) để không bị chặn bởi IC labels.
4. Có **3 artifact là “gating”** bắt buộc sớm — Person 1 phải tạo trước:
   - `data/processed/graph_csr.npz` → unblock Person 2 và 3
   - `outputs/day1_benchmark/*` → lock compute budget và GNN narrative (M2)
   - `data/processed/split_masks.parquet` → Person 3 mới chạy metrics thật được (M3)

### Scope guard (để không lệch MAPR2026 v3)

- IC **primary** phải là **weighted cascade** `p(u,v)=1/degree(v)` (parameter-free). Không dùng “calibration target reach %” cho primary. `calibration_mode: variance_check` (không phải `target_reach`).
- IC backend: **CSR numpy + joblib (loky)**. Tránh NetworkX BFS trong vòng lặp IC.
- Labels IC phải **views-independent** (views chỉ dùng ở typology/eval, không dùng trong IC simulation).
- Graph dùng **undirected** (`graph_directed: false`) — MUSAE Twitch chỉ có mutual-follow edges.
- **Uniform p** — **KHÔNG** report như primary; weighted cascade là bắt buộc.

> ✦ **[IF TIME] Uniform-p sensitivity variant** — nếu còn budget sau khi primary IC xong: `p_uniform = kappa/mean_degree = 2/mean_degree ≈ 0.025` (`kappa_target: 2`). Chỉ report là supplementary sensitivity check, không thay thế primary.

- **LCC check** (Stage 0): xác nhận graph active là mostly connected (1 LCC lớn >> các component còn lại). Output: `lcc_report.json`.

> ⚠ **[IF PROBLEM: pct_lcc < 90%]** Báo cả team ngay; restrict IC sampling to LCC (loại bỏ non-LCC nodes khỏi `n_sample`). Ghi quyết định vào `docs/m0_decisions.md`.

### M0 Decision Record — đọc và xác nhận trước khi code

> Các quyết định dưới đây đã được **lock**. Không thay đổi mà không commit vào `docs/m0_decisions.md` và báo cả team.

| Quyết định                  | Giá trị đã lock                                         | Ai chịu trách nhiệm                |
| --------------------------- | ------------------------------------------------------- | ---------------------------------- |
| `test_frac`                 | **0.20**                                                | Person 1 (trong `--test-frac` arg) |
| `stratify_by`               | **degree_quintile** (`pd.qcut`, q=5, duplicates='drop') | Person 1                           |
| `split_seed`                | **42**                                                  | Person 1 (trong `--seed` arg)      |
| `classification_threshold`  | **top 10%** (`quantile(0.90)`)                          | Person 1                           |
| `min_quadrant_size`         | **150**                                                 | Person 2                           |
| `n_sample` (default)        | **5.000** labeled nodes                                 | Person 1                           |
| **Proxies scope**           | **FULL active graph** (~168k nodes)                     | Person 2                           |
| `runtime_sec` trong metrics | **full-graph inference only**                           | Person 3                           |

| Quyết định            | Giá trị | Khi nào quyết                |
| --------------------- | ------- | ---------------------------- |
| `n_sample × N_runs`   | TBD     | **M2** (sau Day-1 benchmark) |
| GNN narrative branch  | TBD     | **M2** (sau one-hop ρ check) |
| Uniform-p sensitivity | TBD     | M2                           |

**⚠ Rule quan trọng:** Person 3 **KHÔNG được tự tạo split**. Phải load từ `data/processed/split_masks.parquet` do Person 1 tạo, dùng `load_split_mask()` trong `eval_ranking_harness.py`.

---

## 1b) Bảng hằng số chung (Shared Constants)

> ⚠ **Đây là single source of truth cho toàn team.** Mọi thay đổi phải cập nhật bảng này trước, sau đó propagate sang code/config/artifacts. Không được hard-code khác đi ở bất kỳ chỗ nào.

| Hằng số                         | Giá trị chuẩn                                 | Ý nghĩa                                                                                              |
| ------------------------------- | --------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `cv_gate`                       | **0.3**                                       | Regression-ready gate: `cv_score > 0.3` → pipeline tiếp tục bình thường                              |
| `jaccard_gate`                  | **0.85**                                      | Binary-ready gate: `jaccard_stability >= 0.85` → binary non-provisional                              |
| `top_k_pct`                     | **0.10**                                      | Top-10% threshold cho classification labels + NDCG@10% + Precision@10%                               |
| `n_sample`                      | **5,000** (locked sau M2)                     | Số labeled nodes (stratified sample)                                                                 |
| `N_runs`                        | **200** (default; locked sau Day-1 benchmark) | MC IC runs per node                                                                                  |
| `n_mc_stability`                | **3**                                         | Số MC seeds cho label stability check                                                                |
| `gnn_seeds`                     | **5**                                         | Số seeds cho GNN training (mean±std)                                                                 |
| `louvain_seed`                  | **42**                                        | Seed cho Louvain community detection                                                                 |
| `split_seed`                    | **42**                                        | Seed cho train/test split (80/20 stratified)                                                         |
| `test_frac`                     | **0.20**                                      | Test fraction cho split_masks                                                                        |
| **Milestone dates**             |                                               |                                                                                                      |
| M0-M2                           | **6/4 – 10/4**                                | Setup, benchmark, GNN narrative locked                                                               |
| M3                              | **13/4**                                      | IC labels done; split_masks sẵn sàng                                                                 |
| M4                              | **18/4**                                      | All intermediate results done                                                                        |
| M5                              | **22–27/4**                                   | Integration + paper hand-off                                                                         |
| Experiments locked              | **21/4**                                      | Data generation + model training xong                                                                |
| Per-group upgrade trigger       | **25/4**                                      | Nếu predictions sẵn sàng trước 25/4 → per_group_error MUST                                           |
| Submit deadline                 | **30/4**                                      | Hard deadline                                                                                        |
| **CSV scope mapping**           |                                               |                                                                                                      |
| `baseline_ranking_metrics.csv`  | Groups **1–4**                                | Baselines (raw/centrality/proxies/embeddings)                                                        |
| `surrogate_ranking_metrics.csv` | Group **5** (GNN variants)                    | GNN models only                                                                                      |
| `runtime_breakdown.csv`         | Groups **1–5 + proxies**                      | ALL models cho speedup calculation                                                                   |
| **Metric list (RQ2b matrix)**   | **8 metrics**                                 | `ic_score_mean, views, degree, pagerank, kshell, betweenness_approx, one_hop_spread, two_hop_spread` |

---

## 2) Artifact contracts (đóng băng giao diện giữa 3 người)

> Các schema dưới đây bám theo `docs/MAPR2026_v3_migration_checklist.md`. Nếu cần đổi tên/format, phải đổi đồng bộ và ghi vào `docs/experiment_registry.md`.

| Artifact (path)                                                   | Owner                                  | Consumers                                 | Contract tối thiểu                                                                                                                                                                                                                                                                                                |
| ----------------------------------------------------------------- | -------------------------------------- | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `outputs/stage0_data_quality/lcc_report.json`                     | Person 1                               | Person 1,2                                | fields: `n_nodes_total, n_nodes_lcc, pct_lcc, n_components`. **Dùng cho pilot threshold:** `median_reach < 5% of n_nodes_lcc`. Phải có trước khi chạy IC pilot.                                                                                                                                                   |
| `data/processed/graph_csr.npz`                                    | Person 1                               | All                                       | `indptr`, `indices`, `degrees`, mapping `node_id↔row_index` deterministic                                                                                                                                                                                                                                         |
| `outputs/day1_benchmark/ic_runtime_benchmark.json`                | Person 1                               | All                                       | per-sim ms + projected runtime + decision table                                                                                                                                                                                                                                                                   |
| `outputs/day1_benchmark/one_hop_correlation.json`                 | Person 1                               | All                                       | Day-1 gate metrics: `spearman_rho`, `jaccard_at_10pct`, `ndcg_at_10pct`, `decision_branch` (không dùng Spearman đơn lẻ)                                                                                                                                                                                           |
| `data/processed/ic_scores_primary.parquet`                        | Person 1                               | Person 2,3                                | columns: `node_id, ic_score_mean, ic_score_std, n_runs, p_model` (**sample-only: n_sample nodes**)                                                                                                                                                                                                                |
| `data/processed/regression_targets.parquet`                       | Person 1                               | Person 3                                  | columns: `node_id, y` với `y=log1p(ic_score_mean)`                                                                                                                                                                                                                                                                |
| `data/processed/classification_labels.parquet`                    | Person 1                               | Person 3                                  | columns: `node_id, y_top10` (top 10%)                                                                                                                                                                                                                                                                             |
| `data/processed/split_masks.parquet` **[M0-locked]**              | Person 1                               | Person 3                                  | columns: `node_id (str), split ('train'\|'test')`. 80/20, degree-stratified q=5, seed=42. Scope = labeled nodes only. **Không ai tự tạo split khác.** _(Person 2 không cần — typology là descriptive toàn bộ labeled set; filter theo test mask là việc của Person 3 khi eval surrogate)_                         |
| `data/processed/community_features.parquet`                       | Person 2                               | Person 2,3                                | columns: `node_id, community_id, cross_community_edge_fraction`. Scope: ALL active nodes. **File riêng** — KHÔNG ghi đè `node_attributes.parquet` (Person 1 owns).                                                                                                                                                |
| `data/processed/diffusion_proxies.parquet`                        | Person 2                               | Person 3                                  | columns: `node_id, one_hop_spread, two_hop_spread`. **Scope: FULL active graph** (không phải chỉ labeled subset)                                                                                                                                                                                                  |
| `data/processed/typology_labels_ic_views.parquet`                 | Person 2                               | Person 2 _(+Person 3 cho deliverable 1b)_ | columns: `node_id, typology_label, ic_high, views_high, ic_score_mean, views`. Person 3 **không cần** cho **core eval** (dùng `regression_targets.parquet` + `split_masks.parquet`). **Ngoại lệ:** deliverable 1b [✦ IF TIME] cần file này — dependency Person 2 → Person 3 chỉ active khi 1b được thực hiện.     |
| `outputs/mapr2026_v3_results/null_model_typology_summary.json`    | Person 2                               | All                                       | 10 fields bắt buộc: `timestamp, n_nodes(500), n_realizations(3), n_runs_per_node(100), rho_mean, rho_std, hidden_betweenness_real_subgraph_mean, hidden_betweenness_null_mean, hidden_betweenness_null_std, interpretation` — xem Format spec dưới                                                                |
| `outputs/mapr2026_v3_results/views_permutation_null_summary.json` | Person 2                               | All                                       | Permutation null #1 (bắt buộc B5 core): views-permutation typology summary                                                                                                                                                                                                                                        |
| `outputs/mapr2026_v3_results/ic_permutation_null_summary.json`    | Person 2                               | All                                       | Permutation null #2 (bắt buộc B5 core): IC-score permutation typology summary                                                                                                                                                                                                                                     |
| `outputs/mapr2026_v3_results/baseline_ranking_metrics.csv`        | Person 3                               | All                                       | columns: `model_name, spearman_rho, ndcg_at_10pct, precision_at_10pct, runtime_sec`                                                                                                                                                                                                                               |
| `outputs/mapr2026_v3_results/surrogate_ranking_metrics.csv`       | Person 3                               | All                                       | **[MUST nếu M2: gnn_branch_viable=true]** — columns: `model_name, spearman_rho_mean, spearman_rho_std, ndcg_mean, ndcg_std, precision_mean, precision_std, runtime_sec` (mean±std trên 5 seeds)                                                                                                                   |
| `outputs/mapr2026_v3_results/runtime_breakdown.csv`               | Person 2 (proxies) + Person 3 (models) | All                                       | columns: `model_name, inference_sec_full_graph, train_sec(optional/null)` — ghi runtime toàn active graph cho từng model (Group 1–5 + diffusion_proxies); dùng cho Speedup calculation                                                                                                                            |
| `outputs/day1_benchmark/stability_explanation.json`               | Person 1                               | All                                       | **[MUST — triggered khi Jaccard < 0.85 (current observed: 0.307 — recheck sau mỗi re-run); ~30 phút extract từ phase1_community_overlap.json + phase2_threshold_analysis.json, không cần chạy lại IC]** fields: `pct_communities_spanning_boundary`, `mean_gap_to_noise`, `n_thresholds_tested`, `interpretation` |
| `outputs/mapr2026_v3_results/metric_correlation_matrix.json`      | Person 2                               | All                                       | **[MUST — ~2–3h; tất cả data có sẵn ngay bây giờ]** `rho_matrix` 8×8 (8 metrics: ic_score_mean, views, degree, pagerank, kshell, **betweenness_approx**, one_hop_spread, two_hop_spread) + `p_matrix_corrected` (bắt buộc). `rho_by_degree_quintile`: **[✦ IF TIME]** — optional, không ảnh hưởng global matrix   |
| `outputs/mapr2026_v3_results/per_group_prediction_error.csv`      | Person 3                               | All                                       | **[✦ IF TIME — chỉ làm sau M5 khi có đủ predictions; ~1h; Hidden_test ≈ 57 nodes ✓ đủ power]** columns: `model_name, typology_group, n_nodes, spearman_rho, mae`                                                                                                                                                  |

### Format spec chi tiết (để khỏi hiểu khác nhau)

#### `data/processed/graph_csr.npz`

Yêu cầu tối thiểu trong file `.npz` (keys):

- `indptr`: int64, shape `(n_nodes+1,)`
- `indices`: int32/int64, shape **`(2*m_edges,)`** — **bắt buộc lưu cả 2 chiều** cho mỗi edge undirected (u→v VÀ v→u). IC simulation cần truy cập neighbors của mỗi node, nên phải có đủ hai chiều.
- `degrees`: int32/int64, shape `(n_nodes,)`, phải thỏa `degrees[i] == indptr[i+1]-indptr[i]`
- `node_ids`: array of strings, shape `(n_nodes,)`, với `node_ids[i]` là node_id tương ứng row `i`

**Determinism rule:** mapping phải deterministic giữa các lần chạy. Khuyến nghị: sort `node_id` tăng dần trước khi build CSR.

**Build pattern chuẩn cho undirected CSR:**

```python
import scipy.sparse as sp
import numpy as np

sorted_nodes = sorted(G.nodes())
node2idx = {n: i for i, n in enumerate(sorted_nodes)}
n = len(sorted_nodes)

rows, cols = [], []
for u, v in G.edges():
    i, j = node2idx[u], node2idx[v]
    rows += [i, j]; cols += [j, i]   # cả 2 chiều

data = np.ones(len(rows), dtype=np.int8)
A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))

np.savez(
    "graph_csr.npz",
    indptr=A.indptr.astype(np.int64),
    indices=A.indices.astype(np.int32),
    degrees=np.diff(A.indptr).astype(np.int32),
    node_ids=np.array(sorted_nodes)
)
```

#### `outputs/day1_benchmark/*.json`

- `ic_runtime_benchmark.json`: phải có tối thiểu `per_sim_ms`, `projected_total_hours`, `decision` (`n_sample`, `n_runs`)
- `one_hop_correlation.json`: phải có tối thiểu `spearman_rho`, `jaccard_at_10pct`, `ndcg_at_10pct`, `p_value` (nếu có), `decision_branch`

#### `data/processed/split_masks.parquet` (M0-locked)

- `node_id`: string — phủ toàn bộ **labeled nodes** (cùng set với `ic_scores_primary.parquet`)
- `split`: string — giá trị chỉ được là `'train'` hoặc `'test'`
- Rule tạo: `test_frac=0.20`, `stratify=degree_quintile` (pd.qcut q=5), `random_state=42`
- **Consumer rule:** Person 3 load bằng `load_split_mask(PATHS.split_masks)` và filter qua `apply_test_mask()` từ `eval_ranking_harness.py`. Không tạo split mới.

#### `outputs/mapr2026_v3_results/baseline_ranking_metrics.csv`

- `model_name`: string
- `spearman_rho`: float
- `ndcg_at_10pct`: float
- `precision_at_10pct`: float
- `runtime_sec`: float — **full-graph inference time** (M0-locked; không tính precompute/training)

#### `outputs/mapr2026_v3_results/surrogate_ranking_metrics.csv` (**[MUST nếu M2: gnn_branch_viable=true]**)

Schema bắt buộc (mean±std trên 5 training seeds `[42, 123, 456, 789, 1024]`):

- `model_name`: string — một trong 4 tên chuẩn: `gnn_raw_attr`, `gnn_graph_only`, `gnn_centrality`, `gnn_full`
- `spearman_rho_mean`, `spearman_rho_std`: float
- `ndcg_mean`, `ndcg_std`: float
- `precision_mean`, `precision_std`: float
- `runtime_sec`: float — GNN inference time trên full active graph (không tính training)

#### `outputs/mapr2026_v3_results/runtime_breakdown.csv`

Schema bắt buộc — dùng để tính "Speedup: MC IC vs GNN inference" trong Table runtime của paper:

- `model_name`: string — dùng tên chuẩn (vd. `gnn_raw_attr`, `diffusion_proxies`, `node2vec_ridge`, `mc_ic_labeling`, ...)
- `inference_sec_full_graph`: float — thời gian inference trên toàn bộ active graph (~168k nodes); với `mc_ic_labeling` = tổng IC labeling time tính bằng giây
- `train_sec`: float or null — thời gian training (embed + Ridge.fit cho Node2Vec; training 1 seed cho GNN). null cho Group 1–3 và diffusion proxies (không có training phase riêng)

**Owner rule:**

- **Person 1** ghi 1 row: `mc_ic_labeling` với `inference_sec_full_graph = projected_total_hours * 3600` (đọc từ `ic_runtime_benchmark.json`), `train_sec = null`. **PHẢI ghi trước M5** — Person 3 cần để tính Speedup.
- Person 2 ghi `diffusion_proxies` (one_hop + two_hop, inference trên full graph), `train_sec = null`
- Person 3 ghi tất cả Group 1–5 models sau khi chạy inference
- Append vào cùng file — không overwrite

#### `outputs/mapr2026_v3_results/null_model_typology_summary.json`

Schema bắt buộc (để các thành viên khác đọc được không cần hỏi):

```json
{
  "timestamp": "<ISO 8601 string, e.g. 2026-04-18T14:32:00>",
  "n_nodes": 500,
  "n_realizations": 3,
  "n_runs_per_node": 100,
  "rho_mean": 0.42,
  "rho_std": 0.03,
  "hidden_betweenness_real_subgraph_mean": 0.00089,
  "hidden_betweenness_null_mean": 0.00031,
  "hidden_betweenness_null_std": 0.00005,
  "interpretation": "Null graph Hidden nodes do NOT show elevated betweenness — typology reflects true structural position, not degree-distribution artifact."
}
```

- `rho_mean/rho_std`: Spearman ρ (null IC rank vs real IC rank) trung bình ± std qua 3 realizations
- `hidden_betweenness_real_subgraph_mean`: mean betweenness của Hidden nodes trong **real** graph 500-node subgraph (cùng scope với null để comparable)
- `hidden_betweenness_null_mean/std`: mean betweenness của Hidden nodes trên **null** graph 500-node subgraph, averaged over 3 realizations
- Cả hai `*_betweenness_*` đều dùng `nx.betweenness_centrality(G.subgraph(500_nodes), normalized=True)` — cùng scope, cùng hàm → comparable
- `interpretation`: câu kết luận tự động (viết code tự sinh dựa trên so sánh `hidden_betweenness_real_subgraph_mean` vs `hidden_betweenness_null_mean`)

#### Join rule (để khỏi dính lỗi dtype)

- Tất cả parquet/CSV dùng `node_id` nên thống nhất kiểu **string**.

### Protocol lock (để Person 2/3 không implement lệch nhau)

#### Evaluation setting (MAPR2026 v3: transductive)

- Graph là cố định; label IC chỉ có cho một tập node được sample.
- **Tất cả metric accuracy/ranking phải tính trên held-out labeled nodes (test mask)**. Không tính trên toàn bộ nodes.

**Lưu ý quan trọng về “scope” dữ liệu:**

- `ic_scores_primary.parquet` và `regression_targets.parquet` thường là **labeled subset** (do compute budget). Đây là đúng với plan v3.
- Proxies/baselines/surrogates có thể dự đoán cho full-graph để đo runtime, nhưng **khi tính metric thì chỉ dùng test mask trên labeled nodes**.
- **Transductive claim lock (B10):** không claim deployment accuracy trên unlabeled nodes từ các metrics này; muốn claim rộng hơn phải có out-of-sample IC audit riêng (gợi ý 500-1000 nodes).

#### Metric definitions (đã lock tại M0 — không thay đổi)

- `spearman_rho`: Spearman correlation giữa `y_true` và `y_pred` trên **test labeled nodes** (sau khi apply test mask).
- `ndcg_at_10pct`: NDCG@k với $k=\lceil 0.10 \times n_{test}\rceil$; relevance lấy theo `y_true` (regression target = `log1p(ic_score_mean)`).
- `precision_at_10pct`: Precision@k với cùng k; “true top-k” định nghĩa theo top-k của `y_true` trên test set.

> **[M0-locked]** k được tính theo `n_test` (số test nodes), không phải tổng active nodes. Không thay đổi định nghĩa này mà không update `docs/m0_decisions.md`.

### Mock artifacts để không phải chờ nhau

- **Trước khi có `ic_scores_primary.parquet`** (giai đoạn M1–M2): dùng `data/processed/sis_table.parquet` (hoặc `pagerank`) làm nhãn tạm (`ic_score_mean ≈ sis_score`), để Person 2/3 viết pipeline và unit-test I/O.
- **Trước khi có `graph_csr.npz`** (trước M1): Person 2/3 tạm chạy trên subgraph nhỏ được export từ `graph_active.edgelist` để validate logic.

### Smoke checks bắt buộc cho mọi artifact mới

1. **Schema check**: đủ cột/keys bắt buộc.
2. **Coverage check**:
   - Với artifact **full-graph**: tỷ lệ missing gần 0.
   - Với artifact **sample-only** (vd. IC labels): đảm bảo _không missing trong tập labeled nodes_ và report rõ `n_labeled`.
3. **Determinism check**:
   - CSR mapping deterministic.
   - Với cùng seed + cùng sample: danh sách nodes được sample phải trùng; IC MC chấp nhận noise nhỏ nhưng rank/top-decile không được drift mạnh.

---

## 2.1 Lệnh chạy chuẩn (để mọi người test I/O ngay)

Các entrypoints hiện đã scaffold sẵn ở `src/mapr2026_v3/`. Mục tiêu của các lệnh dưới đây là tạo **placeholder artifacts đúng schema** để downstream code không bị chặn.

### (A) Person 1 — CSR + Day-1 artifacts + IC labels (mock/dry-run)

```powershell
# Chạy từ thư mục src/mapr2026_v3/
cd src/mapr2026_v3

# (1) CSR small-mode (test nhanh) — tạo data/processed/graph_csr.npz
python export_csr.py --run --max-edges 200000

# (2) Day-1 artifacts placeholder
python day1_benchmark.py --dry-run

# (3) IC labels + split_masks placeholder (mock từ SIS nếu có)
#     --out-mask tự động ghi data/processed/split_masks.parquet
python ic_labels_primary.py --dry-run --seed 42 --n-runs 50 --test-frac 0.20
#     ↳ tạo: ic_scores_primary.parquet, regression_targets.parquet,
#             classification_labels.parquet, split_masks.parquet
```

> **Sau lệnh (3)**: Person 1 share ngay `split_masks.parquet` lên repo (hoặc gửi file) để Person 3 có thể test harness với mock labels. Đây là "unblock" quan trọng nhất cho Person 3.

### (B) Person 2 — proxies + typology + null model (dry-run)

```powershell
cd src/mapr2026_v3

# (1) Proxies placeholder (dry-run header-only, 0 rows; schema only; KHÔNG dùng cho evaluation/runtime)
python diffusion_proxies.py --dry-run --seed 42

# (2) Typology builder (chạy được cả với mock IC labels từ SIS)
python typology_ic_views.py --dry-run --pct 0.10

# (3) Null model placeholder
python null_model_typology.py --dry-run
```

### (C) Person 3 — baselines/surrogates headers (dry-run)

```powershell
cd src/mapr2026_v3

# Cần có split_masks.parquet trước (lấy từ dry-run của Person 1, xem (A) lệnh 3)
python run_baselines.py --dry-run
python run_surrogates.py --dry-run
```

> ⚠ **[IF PROBLEM: `split_masks.parquet` chưa có]** Person 3 nhờ Person 1 chạy lệnh (A.3) ở chế độ `--dry-run` và commit file `data/processed/split_masks.parquet`. **Không tự tạo split thay thế.**

---

## 3) Workstreams song song (3 người)

### Map theo team

- **Phạm Quốc Vĩnh (Lead)** → **Person 1** (IC core + Day-1 decisions + cung cấp IC labels)
- **Trần Hùng Vĩ** → **Person 2** (structure/proxies/typology + null model)
- **Trần Quốc Hải** → **Person 3** (evaluation harness + surrogate learning/ML)

### Sơ đồ dependency giữa 3 track

```
Person 1 (Track A)              Person 2 (Track B)         Person 3 (Track C)
─────────────────               ──────────────────         ──────────────────
[Stage 0 graph] ──────────────► proxies skeleton ────────► harness skeleton
      │                         (mock graph nhỏ)           (mock labels từ SIS)
      ▼
graph_csr.npz ────────────────► proxy thật (full graph)
      │
      ▼
Day-1 benchmark ──────────────► (quyết định narrative)──► (quyết định GNN scope)
      │
      ▼
IC scores + split_masks ──────► typology thật ──────────► baselines thật
                                                           surrogates thật
```

**Ghi chú dependency chính:**

- `graph_csr.npz` (Person 1) → unblock cả Person 2 và Person 3
- `split_masks.parquet` (Person 1) → Person 3 mới chạy metrics thật
- `ic_scores_primary.parquet` (Person 1) → Person 2 mới build typology thật
- `diffusion_proxies.parquet` (Person 2) → Person 3 mới có Group 3 baseline
- Day-1 ρ result (Person 1) → quyết định toàn bộ GNN narrative cho cả team

### Person 1 — Track A: IC core (CSR + weighted-cascade IC + Day-1 decisions)

**Mục tiêu:** tạo “grounding” cho toàn bộ plan v3: CSR backend + IC labels + stability.

**Không phụ thuộc vào ai** (chỉ cần Stage 0 graph).

**Deliverables theo thứ tự ưu tiên (có thể merge dần):**

1. CSR export + mapping deterministic
   - Input: `data/processed/graph_active.edgelist`
   - Output: `data/processed/graph_csr.npz`
2. **Dead account audit + LCC check** (bắt buộc trước sampling — Day 7/4, theo timeline)
   - **Dead account audit:**
     - Script: `src/data/dead_account_audit.py` (hoặc inline trong preprocessing)
     - Input: `data/raw/large_twitch_features.csv` (cột `dead_account`)
     - Output: `outputs/stage0_data_quality/dead_account_report.json`
     - Report tối thiểu: `n_dead`, `n_live`, `pct_dead`, `mean_degree_dead`, `mean_degree_live`, `mean_views_dead`, `mean_views_live`
     - **Lý do:** stats phải có trong Section 5 (Limitations): "Dead accounts (X%) have lower degree and views."
   - **LCC check** (chạy cùng lúc, hoặc trước IC sampling):
     - Compute connected components của `graph_active`; verify 1 dominant LCC
     - Output: ghi vào `outputs/stage0_data_quality/lcc_report.json` → fields: `n_nodes_total`, `n_nodes_lcc`, `pct_lcc`, `n_components`
     - ⚠ **[IF PROBLEM: pct_lcc < 90%]** Báo cả team ngay; IC sampling restrict to LCC (loại bỏ non-LCC nodes khỏi `n_sample`). Ghi vào `docs/day1_decisions.md`.
     - **Dùng trong pilot**: `median_reach < 5% of LCC size` là threshold "cascade too explosive"
3. Day-1 benchmark scripts
   - `ic_runtime_benchmark.json` → ghi `per_sim_ms`, `projected_total_hours`, `decision` (`n_sample`, `n_runs`)
   - `one_hop_correlation.json` → ghi `spearman_rho`, `jaccard_at_10pct`, `ndcg_at_10pct`, `p_value`, `decision_branch`
   - `docs/day1_decisions.md` → điền đầy đủ metrics gate + quyết định branch theo bảng dưới

   **IC runtime benchmark — cách đo và project (để implement đúng):**

   ```python
   import time
   # Bench trên 100 nodes × 50 runs để ổn định estimate
   BENCH_N_NODES = 100
   BENCH_N_RUNS  = 50

   t0 = time.time()
   for node_idx in bench_nodes[:BENCH_N_NODES]:
       run_ic_simulation(node_idx, n_runs=BENCH_N_RUNS, ...)
   elapsed_sec = time.time() - t0

   per_sim_ms = (elapsed_sec / (BENCH_N_NODES * BENCH_N_RUNS)) * 1000
    # Sau Day-1 decision: n_sample và N_runs được lock
    # Project: total_hours = per_sim_ms * n_sample * N_runs / 1000 / 3600
    projected_total_hours = per_sim_ms * n_sample_candidate * N_runs_candidate / 1e3 / 3600
   ```

   **One-hop gate metrics — cách đo (dùng cùng 200 pilot nodes):**

   ```python
   import numpy as np
   from sklearn.metrics import ndcg_score

   # Bước 1: tính one_hop_spread cho 200 pilot nodes (từ CSR)
   one_hop_scores = [one_hop(node, G_neighbors, degrees) for node in pilot_node_ids]
   # Bước 2: chạy IC pilot (50 runs/node) → ic_score_mean của 200 pilot nodes
   ic_pilot_scores = [ic_mean_per_node[node] for node in pilot_node_ids]

   # Bước 3: Spearman + top-k alignment (k=10% pilot)
   rho, p = scipy.stats.spearmanr(one_hop_scores, ic_pilot_scores)
   k = max(1, int(np.ceil(0.10 * len(pilot_node_ids))))

   order_ic = np.argsort(-np.asarray(ic_pilot_scores, dtype=float))
   order_oh = np.argsort(-np.asarray(one_hop_scores, dtype=float))
   top_ic = set(order_ic[:k]); top_oh = set(order_oh[:k])
   jaccard_at_10pct = len(top_ic & top_oh) / len(top_ic | top_oh)

   ndcg_at_10pct = float(
       ndcg_score(
           np.asarray(ic_pilot_scores, dtype=float).reshape(1, -1),
           np.asarray(one_hop_scores, dtype=float).reshape(1, -1),
           k=k,
       )
   )
   ```

   **IC runtime decision gate (phải ghi vào `day1_decisions.md`):**
   | Projected runtime | n_sample | N_runs |
   |---|---|---|
   | < 4h | 5.000 | 200 |
   | 4–8h | 3.000 | 150 |
   | > 8h | 2.000 | 100 (ghi limitation) |

   **One-hop decision gate (multi-metric):**
   | Condition | GNN narrative branch |
   |---|---|
   | `ρ < 0.8` | GNN là primary contribution — proceed as planned |
   | `0.8 ≤ ρ ≤ 0.9` | Add 2-hop proxy as stronger baseline; GNN may still win |
   | `ρ > 0.9` **and** `Jaccard@10% > 0.8` **and** `NDCG@10% > 0.9` | **RESTRUCTURE**: proxies là primary, GNN là secondary; báo cả team ngay |
   | `ρ > 0.9` nhưng top-k alignment chưa cao | Giữ GNN + 2-hop head-to-head; nhấn mạnh top-k divergence |

4. IC pilot + diagnostics (CV / non-degenerate checks)
   - Output: `outputs/day1_benchmark/ic_pilot_diagnostics.json`
   - **Fields bắt buộc (10 fields + ks_results):** `n_pilot_nodes`, `n_pilot_runs`, `mean_reach`, `median_reach`, `iqr_reach`, `top10_to_median_ratio`, `rank_stability`, `cv_score`, `cv_noise_count` (số nodes có CV > 0.50), `jaccard_stability` (ghi SAU khi chạy 3 MC stability experiments), `ks_results` (dict per feature — xem schema dưới)
   - **Regression-ready gate (primary):** `cv_score > 0.3` → được phép chạy full IC và tiếp tục pipeline với `regression_targets.parquet`.
   - **Binary-ready gate (secondary):** `jaccard_stability >= 0.85` → `classification_labels.parquet` non-provisional. Nếu thấp hơn: binary = provisional (không block nhánh regression).
     > ⚠ **[IF PROBLEM: cv_score < 0.3]** Hai trường hợp — đọc kỹ trước khi dừng pipeline:
     >
     > - **Nếu IC không degenerate** (spearman_mean > 0.65 và reach metrics OK — diagnosis cho thấy noise do binary threshold, không phải IC broken): **KHÔNG dừng pipeline**. Kích hoạt **Option B** (xem block Option B bên dưới) — regression **tiếp tục** với `quality_mode=provisional`, `quality_gate_pass_all=false` ghi trung thực vào manifest. Báo team biết nhưng không block.
     > - **Nếu IC degenerate** (cả 3 điều kiện đồng thời: `median_reach < 2` + `p_reach_gt_1 < 0.20` + `top10_to_median_ratio < 2`): **Dừng pipeline** — báo team ngay; không chạy full IC cho đến khi team có quyết định; xem ⚠ [IF PROBLEM: median_reach...] block bên dưới để biết last-resort fallbacks.
     >
     > **[MUST — narrative only, zero code] Framing note (đọc trước khi implement Option B):** Regression primary KHÔNG phải là fallback do gate fail — đây là formulation đúng về mặt nguyên tắc cho một simulation-derived continuous target. MC simulation tạo ra `ic_score_mean` là continuous quantity; `y = log1p(ic_score_mean)` là regression target tự nhiên. Binary labels là một derived artifact thứ cấp với threshold sensitivity cố hữu. Jaccard instability là _bằng chứng bổ sung_ ủng hộ formulation này, không phải lý do duy nhất để chuyển sang regression. Paper phải trình bày regression primary như là lựa chọn đúng, không phải như là "chúng tôi buộc phải pivot".
     >
     > **Option B — resolution khi gate fire nhưng IC không degenerate** (spearman_mean > 0.65 và reach metrics OK, diagnosis cho thấy noise do binary threshold chứ không phải IC broken): team kích hoạt Option B để không block toàn team:
     >
     > 1. Regression target (`regression_targets.parquet`) = **PRIMARY** — tiếp tục pipeline bình thường.
     > 2. Binary labels (`classification_labels.parquet`) = **provisional/secondary** — phải khai báo uncertainty; dùng consensus branch (`classification_labels_consensus.parquet`) cho binary metrics và loại `is_uncertain=1` khi claim strict binary performance.
     > 3. Freeze handoff package với `quality_mode=provisional` (ghi `quality_gate_pass_all=false` trung thực vào manifest — không fake pass).
     > 4. Áp dụng lockstep rules toàn team (xem Section 8b): cùng 1 version tag, không re-split local.
     > 5. Ghi rõ quyết định Option B vào `docs/day1_decisions.md`.

   > **Note về thứ tự ghi file:** `ic_pilot_diagnostics.json` ghi 2 lần — lần 1 sau pilot run (chưa có `jaccard_stability`), lần 2 sau 3 MC stability experiments (thêm `jaccard_stability`). Script `ic_labels_primary.py` tự update bằng `json.load` → `json.dump`.

   > ✦ **[IF TIME] Robust diagnostics** — thêm vào `ic_pilot_diagnostics.json` nếu còn thời gian sau khi xong MUST:
   >
   > - `p_reach_gt_1 = P(reach > 1)` — nếu < 0.20: flag "regime degenerate"
   > - `p_reach_ge_5 = P(reach >= 5)` — nếu < 0.05: flag "influence mainly local"
   > - `per_quintile_cv` (Q1..Q5): CV riêng từng degree quintile
   > - `run_count_stability_tau_by_quintile`: Kendall τ giữa N_runs và N_runs/2 cho từng quintile; nếu τ > 0.95 ở N_runs=100 có thể downshift runs để tiết kiệm compute
   >
   > Nếu `per_quintile_cv` sẵn có và Q1.cv < 0.15 & Q2.cv < 0.20: ghi `known limitation` vào `docs/day1_decisions.md` — claim ở overall ranking, tránh claim fine-grained within-degree ranking.

   > ⚠ **[IF PROBLEM: median_reach < 2 VÀ p_reach_gt_1 < 0.20 VÀ top10_to_median_ratio < 2]** Last-resort fallbacks — chỉ kích hoạt khi cả 3 điều kiện đồng thời xảy ra:
   >
   > 1. Restrict analysis to LCC (nếu graph có nhiều component nhỏ)
   > 2. Thử uniform-p sensitivity (`p_uniform = kappa/mean_degree`, kappa=2) — báo cáo như sensitivity variant, KHÔNG thay thế primary weighted cascade
   > 3. Dùng normalized reach (reach/degree) thay raw reach — cần justify rõ trong paper
   >
   > Nếu chưa thỏa đồng thời 3 điều kiện: giữ weighted-cascade primary, tiếp tục pipeline bình thường.

   **Schema của `ks_results` (KS representativeness check — 3 features):**

   ```python
   from scipy.stats import ks_2samp

   # Kiểm tra pilot_nodes có representative của toàn labeled pool không
   # So sánh distribution của pilot vs toàn bộ labeled nodes trên 3 features
   ks_results = {}
   for feat in ["degree", "kshell", "pagerank"]:
       pilot_vals = node_attrs.loc[node_attrs["node_id"].isin(pilot_node_ids), feat].values
       full_vals  = node_attrs[feat].values
       stat, p = ks_2samp(pilot_vals, full_vals)
       ks_results[feat] = {
           "ks_stat": float(stat),
           "p_value": float(p),
           "warn": bool(stat > 0.10)   # threshold ks_test_threshold=0.10
       }
   # Ghi vào ic_pilot_diagnostics.json:
   # "ks_results": {"degree": {"ks_stat": 0.07, "p_value": 0.18, "warn": false}, ...}
   ```

   **Pilot node selection (200 nodes — PHẢI reproducible):**

   ```python
   import pandas as pd
   import numpy as np
   rng = np.random.default_rng(seed=42)
   node_attrs = pd.read_parquet("data/processed/node_attributes.parquet")
   # Stratified by degree quintile (same rule as split mask)
   node_attrs["degree_q"] = pd.qcut(node_attrs["degree"], q=5, labels=False, duplicates="drop")
   pilot_nodes = (node_attrs.groupby("degree_q", group_keys=False)
                             .apply(lambda g: g.sample(frac=200/len(node_attrs), random_state=42)))
   pilot_node_ids = pilot_nodes["node_id"].tolist()[:200]
   ```

   **Định nghĩa chính xác 6 metrics (để implement đúng):**

   ```python
   # reach_matrix: shape [n_pilot, n_pilot_runs] — số nodes bị infected mỗi run
   reach_per_node = reach_matrix.mean(axis=1)    # shape [n_pilot]
   mean_reach     = reach_per_node.mean()
   median_reach   = np.median(reach_per_node)
   iqr_reach      = np.percentile(reach_per_node, 75) - np.percentile(reach_per_node, 25)

   # top10_to_median_ratio: top-10% nodes có reach cao nhất vs median
   top10_thresh = np.percentile(reach_per_node, 90)
   top10_mean   = reach_per_node[reach_per_node >= top10_thresh].mean()
   top10_to_median_ratio = top10_mean / (median_reach + 1e-9)

   # cv_score: mean của per-node CV (std/mean trên n_runs của mỗi node)
   per_node_cv = reach_matrix.std(axis=1) / (reach_matrix.mean(axis=1) + 1e-9)
   cv_score    = per_node_cv[per_node_cv <= 0.50].mean()   # exclude high-variance nodes
   cv_noise_count = (per_node_cv > 0.50).sum()

   # rank_stability: Spearman giữa reach trung bình của 2 independent MC seeds
   # Chạy pilot 2 lần với seed khác nhau: seed_A = 42+node, seed_B = 10000+node
   rho_stab, _ = scipy.stats.spearmanr(reach_mean_seedA, reach_mean_seedB)
   rank_stability = rho_stab
   ```

5. IC primary labels + label stability (Jaccard top-decile across 3 MC seeds)
   - `ic_scores_primary.parquet`
   - `regression_targets.parquet`, `classification_labels.parquet`
   - ✦ **[IF TIME] Bootstrap 95% CI** cho mỗi node: `n_bootstrap=1000`, lưu `ic_ci_lower`, `ic_ci_upper` (~30 phút implement — chỉ làm sau khi Jaccard stability pass)

   **Jaccard stability — cách tính (để implement đúng):**

   ```python
   # 3 MC experiments với mc_seed ∈ {0, 1, 2}, mỗi seed chạy n_runs=150
   # Với mỗi seed: lấy top-10% nodes theo ic_score_mean → set của node_ids
   def top10_set(scores_dict):
       df = pd.Series(scores_dict)
       thresh = df.quantile(0.90)
       return set(df[df >= thresh].index)

   sets = [top10_set(scores_seed_0), top10_set(scores_seed_1), top10_set(scores_seed_2)]

   # Mean pairwise Jaccard trên 3 cặp (0,1), (0,2), (1,2)
   def jaccard(A, B): return len(A & B) / len(A | B) if A | B else 1.0
   pairs = [(0,1), (0,2), (1,2)]
   jaccard_stability = np.mean([jaccard(sets[i], sets[j]) for i, j in pairs])
    # Threshold: jaccard_stability >= 0.85 → binary labels non-provisional (regression vẫn là primary gate)
   ```

   > ✦ **[IF TIME] Bootstrap 95% CI per node — implementation reference** (chỉ implement khi làm task [IF TIME] bên trên, sau khi Jaccard stability pass):
   >
   > ```python
   > rng = np.random.default_rng(seed=42)
   > for node_idx in range(n_labeled):
   >     runs = reach_values[node_idx]   # array of n_runs reach values
   >     boot_means = [rng.choice(runs, size=len(runs), replace=True).mean()
   >                   for _ in range(1000)]
   >     ic_ci_lower[node_idx] = np.percentile(boot_means, 2.5)
   >     ic_ci_upper[node_idx] = np.percentile(boot_means, 97.5)
   > ```

5b. **[MUST khi Jaccard < 0.85 — triggered trong run hiện tại (observed: 0.307); recheck sau mỗi re-run; ~30 phút; KHÔNG cần chạy lại IC]** Stability explanation analysis — chạy ngay sau khi confirm `jaccard_stability < 0.85`

- **Tại sao MUST (khi triggered):** Jaccard < 0.85 đã xảy ra. Tất cả số liệu cần thiết đã có trong `outputs/ic_feasibility/phase1_community_overlap.json` và `phase2_threshold_analysis.json`. Chỉ cần đọc 2 file này và reformat — không cần re-run. **Nếu Jaccard ≥ 0.85 ở run tương lai: skip toàn bộ deliverable 5b.**
- **Mục tiêu:** Phân biệt instability do MC sampling (reducible by more runs) vs instability do graph structure (irreducible) — đây là scientific finding, không chỉ là diagnosis.
- **Community overlap test:**
  ```python
  # Với mỗi Louvain community (từ community_features.parquet):
  # Check xem community có nodes ở cả top-k và top-(k+10%) boundary không
  # top_k_set = top 10% nodes theo ic_score_mean
  # boundary_set = nodes ở rank 10%–20%
  n_spanning = sum(
      1 for comm_id in unique_communities
      if (any(partition[n] == comm_id for n in top_k_set) and
          any(partition[n] == comm_id for n in boundary_set))
  )
  pct_communities_spanning_boundary = n_spanning / len(unique_communities)
  ```
- **Gap-to-noise ratio:**
  ```python
  # Tại mỗi percentile threshold pct ∈ {85, 87, 89, 91, 93, 95}:
  sorted_scores = np.sort(ic_scores)[::-1]
  k = int(len(sorted_scores) * pct / 100)
  gap = sorted_scores[k-1] - sorted_scores[k]      # gap tại threshold
  local_std = sorted_scores[max(0,k-5):k+5].std()  # local noise
  noise = local_std / np.sqrt(n_runs)
  gap_to_noise = gap / (noise + 1e-12)
  ```
- **Output:** `outputs/day1_benchmark/stability_explanation.json`
  ```json
  {
    "pct_communities_spanning_boundary": 0.842,
    "mean_gap_to_noise": 0.031,
    "n_thresholds_tested": 6,
    "interpretation": "structural"
  }
  ```
- **Cycle lock note (current run):** giá trị đã lock trong `docs/day1_decisions.md` dùng `n_thresholds_tested = 28` (legacy sweep summary). Tập ngưỡng `{85,87,89,91,93,95}` ở block trên là reference implementation tối giản; không tự ý ghi đè lock values của cycle đang chạy nếu chưa re-freeze version mới.
- **Interpretation rule:** `"structural"` nếu `pct_communities_spanning_boundary > 0.70` VÀ `mean_gap_to_noise < 0.10`; `"sampling"` nếu không. `"structural"` → paper claim instability là irreducible.
- **Lưu ý scope:** Chỉ tạo artifact này nếu `jaccard_stability < 0.85`. Nếu Jaccard pass, không cần chạy.

6. **[M0-locked] Split mask** — tạo ngay sau khi có `ic_scores_primary.parquet`
   - `data/processed/split_masks.parquet`
   - Rule cứng: `test_frac=0.20`, `stratify=degree_quintile` (q=5), `seed=42`
   - Dùng flag `--test-frac 0.20 --seed 42` trong `ic_labels_primary.py`
   - Ghi số `n_train / n_test` vào `docs/day1_decisions.md` để team biết

7. **[M3] Views/IC alignment check** — chạy ngay sau khi có `ic_scores_primary.parquet` (final run)
   ```python
   from scipy.stats import spearmanr
   df = pd.read_parquet(PATHS.ic_scores)
   rho_views_ic, pval = spearmanr(df["views"], df["ic_score_mean"])
   # Ghi vào docs/day1_decisions.md Phần 4:
   # views/IC Spearman ρ = {rho_views_ic:.3f} (p = {pval:.4f})
   ```
   Kết quả quyết định narrative RQ2 — xem bảng fallback tại Milestone M3.

**Gợi ý phân tách file để giảm conflict:**

- Chỉ Person 1 đụng `src/simulation/*` và/hoặc tạo thêm `src/ic/*`.
- Tránh sửa `src/sis/*`.

**Thông số kỹ thuật bắt buộc (từ plan v3 Section 4):**

```
IC backend        : CSR numpy + joblib loky — TUYỆT ĐỐI không dùng NetworkX BFS
worker_seed       : 42 + node_index          (primary IC production run)
stability_seed    : mc_seed * 10000 + node   (label stability check — 3 MC experiments)
p_model           : weighted_cascade  p(u,v) = 1/degree(v)
n_mc_seeds_stab   : 3  (independent MC experiments cho Jaccard check)
n_runs_primary    : 200  (điều chỉnh sau Day-1 benchmark)
n_runs_stability  : 150  (stability check dùng ít runs hơn để tiết kiệm compute)
Jaccard threshold : 0.85 (binary-ready gate; nếu thấp hơn thì binary = provisional, chỉ tăng n_runs khi budget cho phép)
```

**IC simulation core — pseudocode bắt buộc (implement từ đây, không dùng NetworkX BFS):**

```python
import numpy as np
from joblib import Parallel, delayed

def simulate_ic_from_source(source_row, indptr, indices, degrees, n_runs, worker_seed):
    """
    Weighted cascade IC từ source_row (CSR row index).
    Returns: array of shape [n_runs] — reach count (số nodes infected) mỗi run.
    """
    rng = np.random.default_rng(worker_seed)
    reach_counts = np.zeros(n_runs, dtype=np.int32)
    n_nodes = len(indptr) - 1

    for run in range(n_runs):
        infected = np.zeros(n_nodes, dtype=bool)
        infected[source_row] = True
        frontier = [source_row]

        while frontier:
            next_frontier = []
            for u in frontier:
                # Neighbors của u theo CSR
                for v in indices[indptr[u]:indptr[u+1]]:
                    if not infected[v]:
                        # p(u→v) = 1 / degree(v) — weighted cascade
                        p_uv = 1.0 / max(degrees[v], 1)
                        if rng.random() < p_uv:
                            infected[v] = True
                            next_frontier.append(v)
            frontier = next_frontier

        reach_counts[run] = infected.sum()
    return reach_counts

def run_ic_parallel(node_rows, indptr, indices, degrees, n_runs, seed_offset=42, n_jobs=-1):
    """
    Chạy IC song song trên list node_rows.
    Returns: dict {node_row: reach_array[n_runs]}
    """
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(simulate_ic_from_source)(
            row, indptr, indices, degrees, n_runs,
            worker_seed=seed_offset + row   # primary: seed_offset=42; stability: mc_seed*10000
        )
        for row in node_rows
    )
    return {row: arr for row, arr in zip(node_rows, results)}

# Sau khi có results dict:
# ic_score_mean[row] = results[row].mean()
# ic_score_std[row]  = results[row].std()
# n_runs[row]        = n_runs
# p_model            = "weighted_cascade"
```

> **Lưu ý performance:** với `n_nodes=168k` và `n_runs=200`, vòng lặp Python thuần sẽ rất chậm. Nếu cần tăng tốc → vectorize frontier bằng numpy boolean array (thay list frontier bằng np.where). Nhưng pseudocode trên là functionally correct — verify correctness trước, optimize sau.

> **⚠ CRITICAL — worker_seed cho stability check phải KHÁC với primary run:**
>
> - Primary: `worker_seed = 42 + node` → mỗi node có RNG riêng, tất cả runs dùng cùng seed
> - Stability: `worker_seed = mc_seed * 10000 + node` → mc*seed=0,1,2 → 3 \_genuinely independent* MC experiments
> - Nếu dùng cùng seed cho cả 3 MC experiments → Jaccard sẽ = 1.0 (artificial, không phải ổn định thật sự)

**Pilot diagnostics — metrics reference (v3 Section 4.1), pilot = 200 nodes × 50 runs:**

| Metric                                    | Threshold                                | Ý nghĩa                                                       | Loại                       |
| ----------------------------------------- | ---------------------------------------- | ------------------------------------------------------------- | -------------------------- |
| `mean_reach`                              | report                                   | mean single-seed reach                                        | MUST                       |
| `median_reach`                            | < 5% LCC                                 | nếu cao hơn → cascade explosive                               | MUST                       |
| `iqr_reach`                               | report                                   | spread của distribution                                       | MUST                       |
| `top10_to_median_ratio`                   | >> 1                                     | tail separation ratio; nếu ≈ 1 → ranking vô nghĩa             | MUST                       |
| `rank_stability` (Spearman giữa MC seeds) | report                                   |                                                               | MUST                       |
| `cv_score`                                | **> 0.3**                                | nếu thấp → cascade chết quá nhanh                             | **MUST — regression gate** |
| `p_reach_gt_1` (`P(reach > 1)`)           | > 0.20                                   | nếu thấp hơn → regime quá degenerate                          | ✦ [IF TIME]                |
| `p_reach_ge_5` (`P(reach >= 5)`)          | > 0.05                                   | nếu thấp hơn → influence chủ yếu quanh immediate neighborhood | ✦ [IF TIME]                |
| `run_count_stability_tau_by_quintile`     | > 0.95 @ N_runs=100 (optional downshift) | nếu đạt ngưỡng có thể giảm runs để tiết kiệm compute          | ✦ [IF TIME]                |

> ✦ **[IF TIME] metrics** (3 hàng cuối) — chỉ tính sau khi xong 6 MUST metrics; xem block `✦ [IF TIME] Robust diagnostics` ở Deliverable 4 phía trên để biết cách implement.

> **`cv_noise_threshold = 0.50`** (v3 Section 15): node nào có per-node CV > 0.50 là "high-variance node" → exclude khỏi stability metrics, ghi lại count. Cascade quá noisy trên node đó → cần tăng n_runs.

> ✦ **[IF TIME] Run-count stability curve:** với mỗi degree quintile, tính Kendall τ giữa ranking ở `N_runs` và `N_runs/2` (100 vs 50). Nếu τ > 0.95 ở N_runs=100 → có thể downshift N_runs để tiết kiệm compute trong full IC run.

**Stratified sampling với KS check (v3 Section 4.4):**

- Dùng `pd.qcut(degree, q=5)` để stratify (same rule với split mask)
- Sau sampling: chạy KS test trên `degree`, `kshell`, `pagerank` — warn nếu KS stat > 0.10
- Ghi `ks_results` vào stability report

**Definition of Done (DoD) cho Track A — MUST (sign-off bắt buộc):**

- [ ] CSR mapping deterministic: `degrees[i] == indptr[i+1]-indptr[i]`; node_ids sorted tăng dần.
- [ ] Dead account report tồn tại: `outputs/stage0_data_quality/dead_account_report.json` có `n_dead`, `pct_dead`, `mean_degree_dead`, `mean_degree_live`.
- [ ] LCC report tồn tại: `outputs/stage0_data_quality/lcc_report.json` có `n_nodes_total`, `n_nodes_lcc`, `pct_lcc`, `n_components`.
- [ ] Day-1 artifacts sinh ra được: `ic_runtime_benchmark.json` và `one_hop_correlation.json`; `docs/day1_decisions.md` đã điền `n_sample`, `N_runs`, `narrative_branch`.
- [ ] IC pilot diagnostics JSON tồn tại: `outputs/day1_benchmark/ic_pilot_diagnostics.json` có đủ **10 fields bắt buộc** + `ks_results`; `cv_score > 0.3` (regression-ready gate). ⚠ Nếu fail và IC không degenerate: kích hoạt Option B (xem Deliverable 4) — regression tiếp tục với `quality_mode=provisional`, `quality_gate_pass_all=false` ghi trung thực vào manifest.
- [ ] `jaccard_stability` trong pilot JSON: nếu ≥ 0.85 → binary non-provisional; nếu < 0.85 → ghi provisional, không block regression.
- [ ] **[MUST khi Jaccard < 0.85 — hiện triggered; recheck sau re-run; ~30 phút]** `outputs/day1_benchmark/stability_explanation.json` tồn tại. Nếu Jaccard ≥ 0.85 ở run tương lai: checkbox này không áp dụng.
- [ ] `ic_scores_primary.parquet` tồn tại với `n_runs` locked, coverage = n_sample nodes.
- [ ] `split_masks.parquet` tồn tại, schema đúng, coverage = 100% labeled nodes, test_frac ≈ 0.20.

> ✦ **[IF TIME] Soft DoD** — thêm vào khi xong MUST trước deadline:
>
> - [ ] Bootstrap 95% CI: `ic_ci_lower`, `ic_ci_upper` trong `ic_scores_primary.parquet`
> - [ ] Robust diagnostics trong pilot JSON: `p_reach_gt_1`, `p_reach_ge_5`, `per_quintile_cv`, `run_count_stability_tau_by_quintile`
> - [ ] Low-degree limitation note trong `docs/day1_decisions.md` (nếu Q1.cv < 0.15 & Q2.cv < 0.20)

**Gợi ý entrypoint:**

- `src/mapr2026_v3/export_csr.py`
- `src/mapr2026_v3/day1_benchmark.py`
- `src/mapr2026_v3/ic_labels_primary.py`

**Runbook tối thiểu:**

- Unblock team ngay: chạy 3 lệnh ở Mục 2.1 (A).
- Real mode: implement IC (primary: worker_seed=42+node; stability: mc_seed\*10000+node), pilot diagnostics **6 MUST core metrics** + KS check, stability 3 MC seeds (n_runs=150 each).

> ✦ **[IF TIME]** Robust diagnostics (`p_reach_gt_1`, `p_reach_ge_5`, `per_quintile_cv`, `tau_by_quintile`) — chỉ thêm sau khi xong toàn bộ MUST trên.

---

### Person 2 — Track B: Divergence analysis (typology IC×views + proxies + null model)

**Mục tiêu:** Task B (views vs IC typology) + baseline Group 3 (one-hop/two-hop) + null-model typology comparison.

**Có thể làm trước khi IC labels xong** bằng mock nhãn (SIS/pagerank) để hoàn thiện pipeline.

**Deliverables (theo thứ tự dependency):**

1. **[MUST] Community detection + cross-community features** (v3 Section 6)
   - Input: `data/processed/graph_active.edgelist` (hoặc graph_csr.npz)
   - Script: `src/graph/community.py` (đã có sẵn) — Louvain seed sweep (`n_runs=10`, `seed_start=0`, `resolution=1.0`) + chọn **best run** theo modularity
   - **Library:** `python-louvain` only — KHÔNG fallback sang NetworkX Louvain
   - Output: **`data/processed/community_features.parquet`** (file riêng — KHÔNG ghi vào `node_attributes.parquet`)
     - `community_id`: int (Louvain partition)
     - `cross_community_edge_fraction`: float (fraction neighbors in different community)
     - **Lý do file riêng:** Person 1 owns `node_attributes.parquet`. Person 2 ghi đè sẽ gây merge conflict. Consumers (Person 2, 3) join on `node_id` khi cần.
     - Scope: ALL active nodes (phủ 100%)
       > ✦ **[IF TIME] B9 — Louvain resolution sensitivity sweep** — chạy sau khi xong community detection MUST:
       >
       > - Resolution sweep `{0.5, 1.0, 2.0}` — sanity-check resolution limit trên dense graph
       > - Output: `outputs/mapr2026_v3_results/louvain_resolution_sensitivity.json` (fields: `resolution, modularity, n_communities, pct_top3_nodes`)
       > - Rule: giữ `resolution=1.0` làm primary; sweep chỉ là supporting evidence
       > - **Signal for escalation:** nếu `n_communities < 20` hoặc `pct_top3_nodes > 50%` → report "possible over-merge" và team re-lock resolution trong `docs/m0_decisions.md`. Chỉ đổi primary resolution sau khi team đồng ý — không tự đổi unilaterally.
   - **Công thức `cross_community_edge_fraction` (per node):**

     ```python
     import numpy as np

     seeds = range(0, 10)
     partitions, modularities = [], []
     for s in seeds:
         p = community.best_partition(G_nx, resolution=1.0, random_state=s)
         q = community.modularity(p, G_nx)
         partitions.append(p)
         modularities.append(q)

     partition = partitions[int(np.argmax(modularities))]
     # partition: dict {node_id: community_id} from best run

     def cross_community_fraction(node, G_neighbors, partition):
         neighbors = list(G_neighbors[node])
         if len(neighbors) == 0:
             return 0.0
         n_cross = sum(1 for v in neighbors if partition[v] != partition[node])
         return n_cross / len(neighbors)
     ```

   - **Lý do bắt buộc:** structural profiling claim "Hidden nodes are cross-community bridges" cần `cross_community_edge_fraction` — không có thì không support được finding này.

2. Diffusion proxies (Group 3) — **scope: FULL active graph** (M0-locked)
   - Input: `graph_csr.npz`
   - Output: `data/processed/diffusion_proxies.parquet` (ALL active nodes)

- Output phụ: `outputs/mapr2026_v3_results/runtime_breakdown.csv` (`model_name`, `inference_sec_full_graph`, `train_sec` optional/null)
- **Lưu ý:** Không filter — Person 3 apply test mask khi tính metrics
- Two-hop complexity: O(deg²) per node — estimate runtime trước khi chạy full graph

> ⚠ **[IF PROBLEM: two-hop wall-clock > 2h]** Tăng tốc bằng cách: (1) chạy với `n_jobs=-1` (joblib loky parallelism), hoặc (2) batch theo node degree (xử lý low-degree nodes trước để phát hiện sớm nếu still too slow). Two-hop vẫn là **MUST** — đây là tối ưu implementation, **không phải** fallback sang one-hop.

**Công thức bắt buộc (không được dùng `weighted_degree` thay thế — redundant với one-hop):**

```python
# One-hop expected spread — O(deg(u)) per node
def one_hop(node, G_neighbors, degrees):
    return sum(1.0 / max(degrees[v], 1) for v in G_neighbors[node])

# Two-hop expected spread — O(deg²) per node, genuinely different from one-hop
def two_hop(node, G_neighbors, degrees):
    total = 0.0
    for v in G_neighbors[node]:
        p_uv = 1.0 / max(degrees[v], 1)
        second = sum(1.0 / max(degrees[w], 1)
                     for w in G_neighbors[v] if w != node)
        total += p_uv * (1 + second)
    return total
```

3. Typology IC×views (2×2 quadrant) + quadrant sizing
   - Input: `ic_scores_primary.parquet` + `node_attributes.parquet` + `community_features.parquet` (join theo `node_id`; KHÔNG giả định 2 cột community nằm trong `node_attributes.parquet`)
   - Output: `data/processed/typology_labels_ic_views.parquet`
   - Threshold: top 10% cho cả IC và **raw `views`** (M0-locked; KHÔNG dùng `views_log` làm threshold axis — cần consistent với paper text)

   **Label assignment logic bắt buộc (4 quadrants):**

   ```python
   ic_thresh  = df["ic_score_mean"].quantile(0.90)   # top 10% IC
   views_thresh = df["views"].quantile(0.90)          # top 10% raw views (KHÔNG dùng views_log)

   df["ic_high"]    = df["ic_score_mean"] >= ic_thresh
   df["views_high"] = df["views"] >= views_thresh

   def assign_label(row):
       if row["ic_high"] and row["views_high"]:
           return "True"        # high IC + high views
       elif row["ic_high"] and not row["views_high"]:
           return "Hidden"      # high IC + low views  ← target quadrant
       elif not row["ic_high"] and row["views_high"]:
           return "Overrated"   # low IC + high views
       else:
           return "Non"         # low IC + low views

   df["typology_label"] = df.apply(assign_label, axis=1)
   ```

   Output columns bắt buộc: `node_id, typology_label, ic_high, views_high, ic_score_mean, views`

   **Schema bắt buộc cho quadrant JSON report** (ghi ra `outputs/mapr2026_v3_results/typology_quadrant_report.json`):

   ```json
   {
     "timestamp": "<ISO 8601>",
     "n_total": <int>,
     "ic_threshold": <float>,
     "views_threshold": <float>,
     "quadrants": {
       "True":      {"n": <int>, "pct": <float>},
       "Hidden":    {"n": <int>, "pct": <float>},
       "Overrated": {"n": <int>, "pct": <float>},
       "Non":       {"n": <int>, "pct": <float>}
     },
     "min_quadrant_ok": <bool>,
     "two_sample_applied": <bool>
   }
   ```

   - `min_quadrant_ok = all(q["n"] >= 150 for q in quadrants.values())`
   - Ghi ra file này ngay sau khi gán label — trước khi chạy structural profiling

   > ⚠ **[IF PROBLEM: min_quadrant_ok=false (lần 1 — trước two-sample)]** Two-sample strategy: tăng `n_sample` lên **8.000–10.000 nodes** (từ mặc định 5.000), augment với Sample B (high-betweenness + low-views nodes từ full graph). Sample B chỉ dùng cho typology analysis, **KHÔNG** dùng để train GNN. Candidates: `betweenness > quantile(0.70)` AND `views < quantile(0.30)` AND chưa có trong Sample A.

   > ⚠ **[IF PROBLEM: min_quadrant_ok=false sau khi đã áp dụng two-sample strategy]** Residual-based divergence backup:
   >
   > - `divergence_score = z(rank(IC)) - z(rank(views))`; Hidden-like = top decile; Overrated-like = bottom decile
   > - Output: `outputs/mapr2026_v3_results/residual_divergence_report.json`
   > - Đây là backup analysis, **KHÔNG thay thế** typology top-10 M0-locked
   > - Nếu trigger xảy ra: Person 2 chạy và commit artifact trước sign-off Stage 5, ghi rõ trong `docs/experiment_registry.md`

4. Structural profiling — Hidden vs Overrated (v3 Section 11)
   - Columns cần: `degree`, `pagerank`, `kshell`, `betweenness`, **`cross_community_edge_fraction`**, `life_time`
   - Method: MWU + Cliff's delta (Δ ≥ 0.20 là effect size meaningful)
   - BH-FDR correction trên tất cả p-values (Section 8.4) với `statsmodels.multipletests(method='fdr_bh')`
   - Expected: Hidden → higher betweenness + cross_community_fraction; Overrated → higher degree + views

   **MWU exact call + Cliff's delta — code chuẩn (để tránh sai alternative):**

   ```python
   from scipy import stats
   from statsmodels.stats.multitest import multipletests
   import numpy as np

   features = ["degree", "pagerank", "kshell", "betweenness",
               "cross_community_edge_fraction", "life_time"]
   hidden_df   = df[df["typology_label"] == "Hidden"]
   overrated_df = df[df["typology_label"] == "Overrated"]

   rows = []
   p_raws = []
   for feat in features:
       h_vals = hidden_df[feat].dropna().values
       o_vals = overrated_df[feat].dropna().values
       stat, p_raw = stats.mannwhitneyu(h_vals, o_vals, alternative="two-sided")
       n1, n2 = len(h_vals), len(o_vals)
       # Cliff's delta: (U - n1*n2/2) / (n1*n2/2)
       # Positive = Hidden > Overrated (consistent với mannwhitneyu definition)
       cliffs_delta = (stat - n1 * n2 / 2) / (n1 * n2 / 2)
       rows.append({
           "feature": feat,
           "group_hidden_mean": h_vals.mean(),
           "group_overrated_mean": o_vals.mean(),
           "mwu_stat": stat,
           "p_raw": p_raw,
           "cliffs_delta": cliffs_delta
       })
       p_raws.append(p_raw)

   # BH-FDR correction trên tất cả 6 p-values cùng lúc
   reject, p_corrected, _, _ = multipletests(p_raws, method="fdr_bh")
   for i, row in enumerate(rows):
       row["p_corrected"] = p_corrected[i]
       row["significant"] = bool(p_corrected[i] < 0.05 and abs(rows[i]["cliffs_delta"]) >= 0.20)
   ```

   **Schema bắt buộc cho `structural_profiling.csv`:**
   | Cột | Kiểu | Mô tả |
   |---|---|---|
   | `feature` | str | tên cột profile (`degree`, `pagerank`, …) |
   | `group_hidden_mean` | float | mean của Hidden group |
   | `group_overrated_mean` | float | mean của Overrated group |
   | `mwu_stat` | float | Mann-Whitney U statistic |
   | `p_raw` | float | raw p-value |
   | `p_corrected` | float | BH-FDR corrected p-value |
   | `cliffs_delta` | float | Cliff's Δ (+ = hidden > overrated) |
   | `significant` | bool | `p_corrected < 0.05` AND `abs(delta) >= 0.20` |

4b. **[MUST — ~2–3h; tất cả parquet artifacts đã tồn tại; Person 2 có thể chạy ngay hôm nay]** Metric correlation matrix (full pairwise Spearman)

- **Tại sao MUST:** Reviewer sẽ hỏi "how do IC scores relate to simpler metrics?" — không có Table này thì phải trả lời verbal trong rebuttal. Tất cả data đã có sẵn: `ic_scores_primary.parquet`, `diffusion_proxies.parquet`, `node_attributes.parquet`, `centrality_table.parquet`. Zero new computation needed.
- **Mục tiêu:** Trả lời RQ2b — khi nào degree/pagerank/views fail làm proxy cho IC? Provide số liệu định lượng cho Table trong Section 4.3 paper.
- **Input:** join `ic_scores_primary.parquet` + `node_attributes.parquet` + `diffusion_proxies.parquet` + `centrality_table.parquet`; tất cả filter về labeled nodes (5,000 nodes)
- **8 metrics:** `ic_score_mean`, `views`, `degree`, `pagerank`, `kshell`, `betweenness_approx`, `one_hop_spread`, `two_hop_spread`
- **Global matrix:**

  ```python
  from scipy.stats import spearmanr
  from statsmodels.stats.multitest import multipletests
  import numpy as np

  metrics = ["ic_score_mean", "views", "degree", "pagerank",
             "kshell", "betweenness_approx", "one_hop_spread", "two_hop_spread"]
  n = len(metrics)
  rho_matrix = np.zeros((n, n))
  p_matrix   = np.zeros((n, n))

  for i, m1 in enumerate(metrics):
      for j, m2 in enumerate(metrics):
          rho, p = spearmanr(df[m1].values, df[m2].values)
          rho_matrix[i, j] = rho
          p_matrix[i, j]   = p

  # BH-FDR trên tất cả off-diagonal p-values (upper triangle)
  upper_idx = np.triu_indices(n, k=1)
  p_upper = p_matrix[upper_idx]
  _, p_corrected, _, _ = multipletests(p_upper, method='fdr_bh')
  # Ghi p_corrected vào p_matrix upper triangle
  ```

- **[✦ IF TIME] Breakdown by degree quintile** — làm sau khi xong global matrix; nếu tight deadline thì bỏ qua, chỉ giữ global 8×8:
  ```python
  df["deg_q"] = pd.qcut(df["degree"], q=5, labels=False, duplicates="drop")
  rho_by_quintile = {}
  for q in range(5):
      sub = df[df["deg_q"] == q]
      if len(sub) < 30:
          continue
      rho_ic_views, _ = spearmanr(sub["ic_score_mean"], sub["views"])
      rho_ic_degree, _ = spearmanr(sub["ic_score_mean"], sub["degree"])
      rho_by_quintile[f"Q{q}"] = {
          "n": len(sub),
          "rho_ic_views": float(rho_ic_views),
          "rho_ic_degree": float(rho_ic_degree)
      }
  ```
- **Output:** `outputs/mapr2026_v3_results/metric_correlation_matrix.json`
  ```json
  {
    "metrics": ["ic_score_mean", "views", "degree", ...],
    "rho_matrix": [[1.0, 0.42, 0.71, ...], ...],
    "p_matrix_corrected": [[0.0, 0.001, 0.0, ...], ...],
    "rho_by_degree_quintile": {
      "Q0": {"n": 1000, "rho_ic_views": 0.23, "rho_ic_degree": 0.55},
      ...
    }
  }
  ```
- **Timing:** Chạy sau khi có `diffusion_proxies.parquet` và `ic_scores_primary.parquet` — có thể chạy cùng lúc với structural profiling.

5. **life_time external validation của typology** (v3 Section 10 — quy tắc quan trọng)
   - IC labels KHÔNG dùng `life_time` → genuinely independent → valid external corroboration
   - Method 1: Partial Spearman (IC rank vs life_time | degree controlled)
   - Method 2: Stratified MWU by degree quintile, BH-FDR corrected
   - **CẢNH BÁO:** KHÔNG dùng `life_time` để validate GNN-full predictions (GNN-full đã thấy life_time trong features)
     > ⚠ **[IF PROBLEM: partial_spearman_rho < 0.05 HOẶC n_quintiles_significant < 3]** Language fallback corroboration:
     >
     > - `NMI(community_id, language)` — kiểm tra community-language alignment
     > - So sánh language diversity trong neighborhood: Hidden vs Overrated
     > - Output: `outputs/mapr2026_v3_results/language_validation.json`
     > - Đây là corroboration bổ sung, KHÔNG thay thế `lifetime_validation.json` trong main report

   **Partial Spearman implementation (không có hàm sẵn trong scipy):**

   ```python
   from scipy import stats
   import numpy as np

   def partial_spearman_rho(ic_score, life_time, degree):
       """Spearman(ic_score, life_time | degree) — residualize both on degree first."""
       # Rank all 3 variables (Spearman = Pearson trên ranks)
       rank_ic       = stats.rankdata(ic_score)
       rank_lifetime = stats.rankdata(life_time)
       rank_degree   = stats.rankdata(degree)

       # Residualize rank_ic on rank_degree (OLS)
       from numpy.polynomial import polynomial as P
       def resid(y, x):
           x_ = np.column_stack([np.ones(len(x)), x])
           beta = np.linalg.lstsq(x_, y, rcond=None)[0]
           return y - x_ @ beta

       res_ic  = resid(rank_ic, rank_degree)
       res_lft = resid(rank_lifetime, rank_degree)

       rho, p = stats.spearmanr(res_ic, res_lft)
       return rho, p
   ```

   **Stratified MWU — cụ thể:**
   - Tạo 5 degree quintiles từ TOÀN BỘ labeled nodes (`pd.qcut(degree, q=5, duplicates='drop')`)
   - Trong mỗi quintile: so sánh `life_time` của Hidden vs Non-Hidden nodes (MWU)
   - Apply BH-FDR trên 5 p-values: `statsmodels.multipletests(p_values, method='fdr_bh')`
   - Ghi mỗi quintile vào `quintile_results` array trong `lifetime_validation.json`

   **Schema bắt buộc cho `lifetime_validation.json`:**

   ```json
   {
     "partial_spearman_rho": <float>,
     "partial_spearman_p": <float>,
     "n_quintiles_tested": <int>,
     "n_quintiles_significant": <int>,
     "success": <bool>,
     "quintile_results": [
             {"quintile": 0, "n_hidden": <int>, "n_non_hidden": <int>,
        "p_raw": <float>, "p_corrected": <float>, "cliffs_delta": <float>, "significant": <bool>},
       ...
     ]
   }
   ```

   - `success = (n_quintiles_significant >= 3)` — Success target: ≥ 3/5 quintiles significant

6. Null model comparison (configuration model) trên typology (v3 Section 5)
   - **Spec cụ thể:** 500 nodes × **3 realizations** × **100 runs/node**
   - So sánh TYPOLOGY QUADRANT (không chỉ rank correlation) giữa real graph và null
   - Câu hỏi: "Nếu null cũng có Hidden quadrant với betweenness cao → typology là degree-distribution artifact"
   - Output: `null_model_typology_summary.json` (rho_mean±std, hidden_betweenness_null_mean)
     - **Permutation null (B5 core - execution-locked):**
       - Mục tiêu: giảm nguy cơ kết luận "configuration null inconclusive" và kiểm tra trực tiếp cơ chế divergence views-IC.
       - Nhánh 1 (**bắt buộc**): **views-permutation null**
         - Giữ nguyên graph + IC scores, permute `views` across labeled nodes, rebuild typology nhiều lần.
         - Output gợi ý: `outputs/mapr2026_v3_results/views_permutation_null_summary.json`.
       - Nhánh 2 (**bắt buộc**): **IC-score permutation null**
         - Giữ nguyên graph + views, permute `ic_score_mean`, rebuild typology.
         - Output gợi ý: `outputs/mapr2026_v3_results/ic_permutation_null_summary.json`.
       - Rule: configuration model vẫn là primary null trong contract; permutation null package (views-perm + IC-perm) là core của B5.
       - **Execution lock (bắt buộc):** Person 2 phải chạy và commit đủ cả 2 artifact permutation trước sign-off Stage 5.

   **Configuration model generation (Python API bắt buộc):**

   ```python
   import networkx as nx

   def generate_null_graph(G_real, realization_seed):
       degree_sequence = [d for _, d in G_real.degree()]
       # nx.configuration_model trả về MultiGraph → convert sang simple Graph
       G_null = nx.Graph(nx.configuration_model(
           degree_sequence, seed=realization_seed
       ))
       G_null.remove_edges_from(nx.selfloop_edges(G_null))  # loại bỏ self-loops
       return G_null
   # realization_seed = realization_index * 100  (0, 100, 200)
   ```

   **IC trên null graph — dùng cùng engine, cùng params:**
   - `p(u,v) = 1 / degree_in_null_graph(v)` (weighted cascade — cùng formula với real graph)
   - `n_runs_per_node = 100`, `worker_seed = 42 + node_index`
   - Node set: 500 nodes được sample từ labeled nodes (seed=42 để reproducible)
   - Sau IC trên null: apply cùng typology threshold (IC top-10%, views top-10%) → đếm Hidden nodes → lấy betweenness trung bình của Hidden group trên null graph

   **Betweenness trên null graph — cách tính (scope: 500-node subgraph cho CẢ real và null):**

   > ⚠ **Scope consistency rule:** KHÔNG so sánh subgraph betweenness (null) với full-graph betweenness (từ centrality_table.parquet via NetworKit). Hai giá trị này có đơn vị khác nhau — không comparable. Phải tính subgraph betweenness cho CẢ HAI phía với cùng 500-node scope.

   ```python
   import networkx as nx

   # ─── Real graph: subgraph betweenness trên 500-node subset ─────────────────
   # Dùng cùng G_real (real NetworkX graph) và sample_500_node_ids
   G_real_sub = G_real.subgraph(sample_500_node_ids)
   betweenness_real = nx.betweenness_centrality(G_real_sub, normalized=True)
   hidden_on_real = [n for n in sample_500_node_ids if real_typology.get(n) == "Hidden"]
   hidden_bet_real = np.mean([betweenness_real.get(n, 0.0) for n in hidden_on_real]) if hidden_on_real else 0.0

   # ─── Null graph: subgraph betweenness trên cùng 500-node subset ────────────
   G_null_sub = G_null.subgraph(sample_500_node_ids)
   betweenness_null = nx.betweenness_centrality(G_null_sub, normalized=True)
   hidden_on_null = [n for n in sample_500_node_ids if null_typology.get(n) == "Hidden"]
   hidden_bet_null = np.mean([betweenness_null.get(n, 0.0) for n in hidden_on_null]) if hidden_on_null else 0.0

   # ─── Interpretation ─────────────────────────────────────────────────────────
   # Nếu hidden_bet_real >> hidden_bet_null:
   #     Hidden-node betweenness là structural (không phải degree artifact) ✅
   # Nếu hidden_bet_real ≈ hidden_bet_null:
   #     Betweenness là consequence của degree distribution → report as limitation
   ```

   - KHÔNG dùng NetworKit betweenness từ `centrality_table.parquet` phía để compare với null (scope khác)
   - 500 nodes → NetworkX exact betweenness fast enough (~vài giây)

**Gợi ý entrypoint (để review dễ):**

- `src/mapr2026_v3/diffusion_proxies.py`
- `src/mapr2026_v3/typology_ic_views.py`
- `src/mapr2026_v3/null_model_typology.py`

**Runbook tối thiểu (để review nhanh):**

- Unblock team ngay (placeholder artifacts): chạy 3 lệnh ở Mục 2.1 (B).
- Khi implement real mode:
  - `diffusion_proxies.py`: implement one-hop và two-hop proxies theo MAPR2026 v3 (không redundant).
  - `typology_ic_views.py`: thêm quadrant sizing checks + expansion strategy + report (JSON/CSV dưới `outputs/mapr2026_v3_results/`).
  - `null_model_typology.py`: configuration model realizations + compare quadrant profiles, write `null_model_typology_summary.json`.

**Gợi ý phân tách file để giảm conflict:**

- Person 2 đụng `src/graph/null_model.py` (nếu mở rộng) hoặc tạo script mới riêng cho MAPR2026 v3 (khuyến nghị).
- Các report/log phụ của Person 2 (quadrant counts, null-model summary, plots) nên ghi vào `outputs/mapr2026_v3_results/` để không lẫn với outputs SIS (stage3).

**Ghi chú quan trọng về scope typology:**

- `typology_labels_ic_views.parquet` covers **tất cả labeled nodes** (không filter theo train/test mask).
- Lý do: typology là phân tích descriptive toàn bộ labeled set; Person 3 sẽ filter theo test mask riêng khi cần đánh giá surrogate performance.
- Person 2 **không cần đọc** `split_masks.parquet` — đó là việc của Person 3.

**DoD cho Track B — MUST (sign-off bắt buộc):**

- [ ] `data/processed/community_features.parquet` (file riêng, KHÔNG ghi vào `node_attributes.parquet`), phủ 100% active nodes; có `node_id`, `community_id`, `cross_community_edge_fraction`.
- [ ] Proxies (one-hop + two-hop) trên FULL active graph, missing = 0; `runtime_breakdown.csv` có `inference_sec_full_graph`.
- [ ] `typology_quadrant_report.json` tồn tại, mỗi quadrant ≥ 150 nodes (`min_quadrant_ok: true`). ⚠ **[IF PROBLEM: min_quadrant_ok=false]** → xem block ⚠ ở Deliverable 3 để apply two-sample strategy (`two_sample_applied: true`) rồi tiếp tục.
- [ ] Structural profiling: MWU + Cliff's delta (Δ ≥ 0.20) + BH-FDR cho 6 columns → `structural_profiling.csv` đúng 6 hàng.
- [ ] `life_time` validation: chạy partial Spearman + stratified MWU, ghi `p_corrected` vào `lifetime_validation.json`. Gate: `n_quintiles_significant ≥ 3` → sign-off. ⚠ **[IF PROBLEM: < 3/5 significant]** → ghi fallback "external validation limited" vào Limitations — không block pipeline.
- [ ] Configuration null model: 3 realizations × 500 nodes × 100 runs → `null_model_typology_summary.json` đúng 10 fields.
- [ ] Permutation null package: `views_permutation_null_summary.json` + `ic_permutation_null_summary.json` — bắt buộc trước sign-off Stage 5.
- [ ] **[MUST — ~2–3h; chạy được ngay hôm nay]** `metric_correlation_matrix.json` tồn tại với 8×8 `rho_matrix` và `p_matrix_corrected` (global matrix — 8 metrics: ic_score_mean, views, degree, pagerank, kshell, betweenness_approx, one_hop_spread, two_hop_spread — bắt buộc). `rho_by_degree_quintile` là **[✦ IF TIME]** — sign-off không phụ thuộc vào phần này.

> ✦ **[IF TIME] Soft DoD:**
>
> - [ ] Louvain resolution sensitivity: `louvain_resolution_sensitivity.json` cho {0.5, 1.0, 2.0}

> _(Residual divergence — xem ⚠ [IF PROBLEM: min_quadrant_ok=false sau two-sample] block ở Deliverable 3 phía trên. Đây là [IF PROBLEM], không phải [IF TIME] — không nằm trong Soft DoD.)_

**Threshold rule (đã lock tại M0):** top-10% cho cả IC và views (`classification_threshold: 0.10`).

---

### Person 3 — Track C: Surrogate learning + evaluation harness (metrics/runtimes)

**Mục tiêu:** Task C (surrogate learning) và quan trọng nhất là **evaluation harness** (Spearman/NDCG@10%/P@10% + runtime table).

**Có thể làm trước khi IC labels xong** bằng mock nhãn, vì cần build sớm:

- loader + **shared split mask** (load từ `split_masks.parquet`, KHÔNG tự tạo split)
- baseline runner
- logging runtime (full-graph inference only)

**Deliverables:**

1. Evaluation harness (model-agnostic)
   - Input: `regression_targets.parquet` + **`split_masks.parquet`** (M0-locked)
   - Cách dùng: `load_split_mask()` → `apply_test_mask()` → `compute_metrics()`
   - Output: `outputs/mapr2026_v3_results/baseline_ranking_metrics.csv`
   - **⚠ Không tạo split mới** — luôn dùng artifact của Person 1
   - Metrics: Spearman ρ (primary), NDCG@10% (secondary), Precision@10% (supplementary)
   - **TRÁNH:** Accuracy, F1-macro — misleading với 95/5 class imbalance

   **Function signatures bắt buộc trong `eval_ranking_harness.py`:**

   ```python
   def load_split_mask(path: str) -> pd.DataFrame:
       """Load split_masks.parquet. Returns DataFrame với columns [node_id (str), split ('train'|'test')]."""

   def apply_test_mask(df: pd.DataFrame, mask_df: pd.DataFrame) -> pd.DataFrame:
       """Inner join df với mask_df trên node_id, filter split=='test'. Returns test-only DataFrame."""

   def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
       """Returns dict: {spearman_rho, ndcg_at_10pct, precision_at_10pct}. y_true/y_pred: 1D float arrays."""
   ```

   **NDCG@10% + Precision@10% implementation:**

   ```python
   import math
   from sklearn.metrics import ndcg_score
   from scipy.stats import spearmanr

   def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
       k = math.ceil(0.10 * len(y_true))   # 10% của test set — không hardcode

       # Spearman rho
       rho, _ = spearmanr(y_true, y_pred)

       # NDCG@k — relevance = y_true (continuous IC scores), higher = more relevant
       ndcg = ndcg_score(y_true.reshape(1, -1), y_pred.reshape(1, -1), k=k)

       # Precision@k — true top-k defined by y_true, predicted top-k by y_pred
       true_top_k = set(np.argsort(y_true)[-k:])    # top-k indices theo y_true
       pred_top_k = set(np.argsort(y_pred)[-k:])    # top-k indices theo y_pred
       precision_at_k = len(true_top_k & pred_top_k) / k

       return {
           "spearman_rho":      float(rho),
           "ndcg_at_10pct":     float(ndcg),
           "precision_at_10pct": float(precision_at_k),
       }
   ```

   **Chuẩn hóa `model_name` — dùng đúng tên này trong CSV (để join sau không bị lỗi):**
   | model_name | Group | Mô tả |
   |---|---|---|
   | `views_rank` | 1 | rank(views) |
   | `views_per_day_rank` | 1 | rank(views/life_time) |
   | `degree_rank` | 1 | rank(degree) |
   | `pagerank` | 2 | PageRank α=0.85 |
   | `kshell` | 2 | k-core shell |
   | `betweenness` | 2 | ApproxBetweenness2 |
   | `one_hop_spread` | 3 | one-hop proxy |
   | `two_hop_spread` | 3 | two-hop proxy |
   | `node2vec_ridge` | 4 | Node2Vec + Ridge LR |
   | `mlp_raw_attr` | 4 | MLP raw attributes |
   | `gnn_raw_attr` | 5 | GraphSAGE raw-attr (→ surrogate CSV) |
   | `gnn_graph_only` | 5 | GraphSAGE graph-only (→ surrogate CSV) |
   | `gnn_centrality` | 5 | GraphSAGE centrality (→ surrogate CSV) |
   | `gnn_full` | 5 | GraphSAGE full features (→ surrogate CSV) |

1b. **[✦ IF TIME — chỉ làm sau M5 khi có đủ predictions từ tất cả models; ~1h; Hidden_test ≈ 57 nodes ✓ đủ power n≥20]** Per-typology-group prediction error analysis

- **Tại sao IF TIME:** Không thể làm trước khi Person 3 hoàn thành toàn bộ predictions (baseline + GNN). Khi M5 xong, đây là ~1h marginal work để tạo một table có giá trị bổ sung. **Nếu predictions từ tất cả models sẵn sàng trước 25/4 → nâng lên MUST (cần 2 ngày buffer để chạy và review trước deadline 27/4).**
- **Mục tiêu:** Trả lời RQ3b — node types nào khó predict nhất bằng cheap models? Hidden nodes expected worst.
- **Input:** `regression_targets.parquet` + `typology_labels_ic_views.parquet` + `split_masks.parquet` + predictions từ tất cả models
- **Với mỗi (model × typology_group), tính trên test nodes thuộc group đó:**

  ```python
  from scipy.stats import spearmanr
  import numpy as np, pandas as pd

  results = []
  for model_name, y_pred_all in model_predictions.items():
      # y_pred_all: dict {node_id: predicted_score}
      for group in ["True", "Hidden", "Overrated", "Non"]:
          group_nodes = test_df[test_df["typology_label"] == group]["node_id"].tolist()
          if len(group_nodes) < 20:   # power guard
              continue
          y_true = test_df.loc[test_df["node_id"].isin(group_nodes), "y"].values
          y_pred = np.array([y_pred_all[n] for n in group_nodes
                             if n in y_pred_all])
          if len(y_true) != len(y_pred) or len(y_true) < 20:
              continue
          rho, _ = spearmanr(y_true, y_pred)
          mae = np.mean(np.abs(y_true - y_pred))
          results.append({
              "model_name": model_name,
              "typology_group": group,
              "n_nodes": len(y_true),
              "spearman_rho": float(rho),
              "mae": float(mae)
          })
  pd.DataFrame(results).to_csv(
      "outputs/mapr2026_v3_results/per_group_prediction_error.csv", index=False
  )
  ```

- **Output:** `outputs/mapr2026_v3_results/per_group_prediction_error.csv`
- **Timing:** Chạy SAU khi tất cả models đã có predictions trên test set (cuối pipeline Person 3).
- **DoD:** File tồn tại, có ít nhất 1 row per model, Hidden group có row nếu `n_nodes >= 20`.

> ⚠ **[IF PROBLEM: Hidden group n_nodes < 20 trên test set]** Ghi `"Hidden": "skipped — n_nodes < 20"` vào một separate JSON field `skipped_groups` trong summary; không tạo row trong CSV. Đây là power limitation, ghi vào `docs/assumptions_limitations.md`.

2. Baselines (tất cả filter qua test mask trước khi tính metrics):
   - **Group 1 — Raw features O(1):** `rank(views)`, `rank(views/life_time)`, `rank(degree)`
     - Column `views`: lấy từ `node_attributes.parquet`, cột `views` (raw count — KHÔNG normalize)
     - Column `views/life_time`: nếu `node_attributes.parquet` đã có `views_per_day` → dùng trực tiếp; nếu không → tính on-the-fly: `df["views"] / df["life_time"].clip(lower=1)`
     - `runtime_sec` Group 1 = **inference-only** trên full active graph (rank/score generation), không tính file load
   - **Group 2 — Centrality O(N log N → NE):** PageRank (α=0.85), k-shell, Betweenness (NetworKit `ApproxBetweenness2`, `epsilon=0.10`, `delta=0.10`) — reuse artifacts Stage 1–2
     - **Centrality scores đã được precompute từ Stage 1–2** (Person 1) → Person 3 load từ `centrality_table.parquet`, KHÔNG recompute
     - `runtime_sec` Group 2 = **inference-only** trên full active graph (generate score vector từ artifact precompute), không tính file load/metric computation
   - **Group 3 — Diffusion proxies:** one-hop O(E) + two-hop naive O(Σ d(v)^2) từ `diffusion_proxies.parquet` (full graph, filter test mask)

3. **Group 4 — Shallow Embedding Baselines** (v3 Section 7 Group 4 — ghi vào `baseline_ranking_metrics.csv`, KHÔNG phải surrogate CSV):
   - **Node2Vec + LR:**
     - Library: **`node2vec`** (`pip install node2vec`) hoặc **`pecanpy`** (nhanh hơn cho large graph — `pip install pecanpy`)
     - Params: `dim=64, walks=20` (⚠ KHÔNG phải 200 — 10x chậm hơn), `walk_len=20`, `p=1, q=1` (unbiased random walk)
     - Downstream: LR regression (`sklearn.linear_model.Ridge` hoặc `LinearRegression`) trên embedding → predict `y = log1p(ic_score_mean)`
     - Measure runtime tách riêng: `train_sec` (embed + fit) và `runtime_sec` (inference-only `predict` trên full active graph)

     **Flow chuẩn (để không nhầm train/test split):**

     ```python
     # Bước 1: Embed TẤT CẢ labeled nodes (train + test) — chỉ tính 1 lần
     t0_train = time.time()
     embeddings = model.fit_transform(G_labeled)   # shape [n_labeled, 64]

     # Bước 2: Fit Ridge trên TRAIN nodes
     X_train = embeddings[train_mask_local]   # local index trong labeled subset
     y_train = y[train_mask_local]
     ridge = Ridge(alpha=1.0).fit(X_train, y_train)
     train_sec = time.time() - t0_train

     # Bước 3: Inference-only trên full active graph (runtime contract)
     t0_inf = time.time()
     y_pred_full = ridge.predict(embeddings)
     runtime_sec = time.time() - t0_inf

     # Bước 4: Lấy test slice để compute_metrics
     y_pred = y_pred_full[test_mask_local]
     ```

     - `runtime_sec` trong `baseline_ranking_metrics.csv` = **inference-only** (`predict`) trên full active graph.
     - `inference_sec_full_graph` trong `runtime_breakdown.csv` = thời gian `predict` trên full active graph (không phải test-only).
     - Embed + fit = training cost → ghi `train_sec` trong `runtime_breakdown.csv`.
     - ⚠ Node2Vec là **inductive cần re-embed** nếu graph thay đổi — không giống GNN inference (one-pass full graph). Đây là điểm yếu cần note trong paper.
     - Embed trên labeled nodes (không nhất thiết cần 168k) — nhưng dùng FULL graph để random walks có context đủ rộng

   - **MLP raw attributes:**
     - Features: `[log1p(views), views/life_time, life_time]` (normalize min-max trước khi vào MLP)
     - Architecture: `Linear(3→128) → ReLU → Dropout(0.3) → Linear(128→1)`
     - Optimizer: Adam, `lr=0.001`, **epochs=100 cố định** (KHÔNG dùng early stopping — tránh cần val split thêm; epochs đủ nhỏ để không overfit)
     - Loss: HuberLoss(`delta=1.0`) — nhất quán với GNN

   **Min-max scaler — fit trên train_mask ONLY (tránh data leakage):**

   ```python
   from sklearn.preprocessing import MinMaxScaler

   df_labeled = pd.read_parquet("data/processed/node_attributes.parquet")
   df_labeled = df_labeled.merge(split_mask_df, on="node_id")

   df_labeled["feat_views_log"]     = np.log1p(df_labeled["views"])
   df_labeled["feat_views_per_day"] = df_labeled["views"] / df_labeled["life_time"].clip(lower=1)
   feat_cols = ["feat_views_log", "feat_views_per_day", "life_time"]

   scaler = MinMaxScaler()
   train_rows = df_labeled[df_labeled["split"] == "train"]
   test_rows  = df_labeled[df_labeled["split"] == "test"]

   X_train = scaler.fit_transform(train_rows[feat_cols].values)   # fit on train
   X_test  = scaler.transform(test_rows[feat_cols].values)        # transform test
   # Không fit lại scaler trên test — đây là tiêu chuẩn để tránh leakage
   ```

   - **Lưu ý naming:** Master plan v3 gọi đây là "Group 4 Baselines" (không phải "surrogates"). Kết quả phải vào `baseline_ranking_metrics.csv` cùng với Group 1–3 để so sánh đầy đủ trong Table 2 của paper.

4. **[MUST nếu M2: gnn_branch_viable=true] Group 5 — GNN — 4 ablation variants (+1 ✦ [IF TIME])** (v3 Section 7 Group 5):

   | Variant          | Features (in_dim)                                        | Role                                                                    |
   | ---------------- | -------------------------------------------------------- | ----------------------------------------------------------------------- |
   | **GNN-raw-attr** | `views_log_norm, views_per_day_norm, life_time_norm` (3) | **MUST — Primary proposed**                                             |
   | GNN-graph-only   | `degree_norm` only (1)                                   | **MUST** — Ablation: topology without attributes                        |
   | GNN-centrality   | `degree_norm, pagerank_norm, kshell_norm` (3)            | **MUST** — Ablation: hand-crafted features                              |
   | GNN-full         | all 6 features (normalized)                              | ✦ [IF TIME] — supplementary upper bound (có thể cắt nếu tight timeline) |
   | GNN-random       | random/constant node features (1)                        | ✦ [IF TIME] — sanity-check message passing value                        |

> ✦ **[IF TIME]** `GNN-random` — không block deadline; chỉ chạy sau khi xong toàn bộ MUST GNN variants. Nếu chạy: ghi vào `surrogate_ranking_metrics.csv` với `model_name=gnn_random`.

> **Feature normalization bắt buộc**: tất cả features phải normalize trước khi vào GNN (min-max hoặc z-score). Column names trong experiment.yaml là `*_norm`. Không dùng raw values trực tiếp.

Architecture: GraphSAGE, `hidden_dim=128`, `n_layers=2`, `dropout=0.3`, Huber Loss (`delta=1.0`), `lr=0.001`, `epochs=200`.
Framework: **PyTorch Geometric (PyG) ≥ 2.5**, `torch ≥ 2.0`. Hardware yêu cầu: GPU ≥ 8GB VRAM (RTX 3080 / A100).

> ⚠ **[IF PROBLEM: PyG install fail HOẶC không có GPU ≥ 8GB]** Fallback: DGL + CPU (chậm hơn ~5× — thêm khoảng 2–3 ngày training time). Thay `from torch_geometric.nn import SAGEConv` bằng `from dgl.nn import SAGEConv`.

**Transductive GNN Data object setup (PyG):**

```python
from torch_geometric.data import Data
import torch

# x: features cho TẤT CẢ active nodes (168k × in_dim) — kể cả unlabeled
# y: regression target cho TẤT CẢ nodes; unlabeled nodes gán NaN hoặc 0 (bị mask)
# edge_index: toàn bộ edges của active graph (2 × 2m)
# train_mask, test_mask: boolean tensor size 168k; True chỉ tại labeled train/test nodes

# y_all: set unlabeled targets = 0.0 (KHÔNG phải NaN — HuberLoss sẽ fail với NaN)
# Train/test mask sẽ loại unlabeled ra khỏi loss/eval nên giá trị 0.0 không ảnh hưởng
y_all = torch.zeros(n_active_nodes, dtype=torch.float32)
y_all[labeled_mask] = torch.tensor(y_labeled, dtype=torch.float32)

data = Data(
    x=x_all,               # shape [168114, in_dim], float32
    edge_index=edge_index,  # shape [2, 2*m_edges]
    y=y_all,               # shape [168114], float32; 0.0 cho unlabeled (bị mask)
)
data.train_mask = train_mask   # bool tensor [168114], True tại labeled train nodes
data.test_mask  = test_mask    # bool tensor [168114], True tại labeled test nodes

# Loss chỉ tính trên train nodes:
# loss = criterion(out[data.train_mask], data.y[data.train_mask])
# Eval chỉ trên test nodes:
# spearman(out[data.test_mask], data.y[data.test_mask])
```

- `x_all` cần normalize: min-max trên TOÀN active graph (không chỉ labeled subset)
- Labeled nodes = những nodes trong `split_masks.parquet`; unlabeled = còn lại

**Cách build `x_all` cho toàn bộ 168k active nodes (không nhầm nguồn data):**

```python
# Source: node_attributes.parquet — có ALL active nodes (Person 1 không filter)
node_attrs = pd.read_parquet("data/processed/node_attributes.parquet")
# node_attrs index theo CSR row order (dùng node_ids từ graph_csr.npz để align)
csr_data   = np.load("data/processed/graph_csr.npz", allow_pickle=True)
node_ids_ordered = csr_data["node_ids"]   # shape [n_active], sorted ascending

df_all = pd.DataFrame({"node_id": node_ids_ordered})
df_all = df_all.merge(node_attrs[["node_id","views","life_time"]], on="node_id", how="left")

df_all["views_log"]      = np.log1p(df_all["views"].fillna(0))
df_all["views_per_day"]  = df_all["views"].fillna(0) / df_all["life_time"].clip(lower=1)

# Normalize min-max trên TOÀN active graph
from sklearn.preprocessing import MinMaxScaler
feat_cols = ["views_log", "views_per_day", "life_time"]
scaler_gnn = MinMaxScaler()
x_all = scaler_gnn.fit_transform(df_all[feat_cols].values)   # shape [168114, 3]
x_all = torch.tensor(x_all, dtype=torch.float32)
```

- Scaler fit trên TOÀN active graph (cả labeled + unlabeled) — đây là đúng với transductive setting vì GNN thấy toàn bộ node features
- Khác với MLP: MLP fit scaler chỉ trên train_mask (inductive baseline); GNN là transductive nên được dùng full graph statistics

**Build `edge_index` từ `graph_csr.npz` (COO format cho PyG):**

```python
import numpy as np, torch
csr = np.load("data/processed/graph_csr.npz", allow_pickle=True)
indptr  = csr["indptr"]    # shape [n+1]
indices = csr["indices"]   # shape [2*m]
n_nodes = len(indptr) - 1

# Tạo source array: lặp lại mỗi node i theo số neighbors của nó
row_idx = np.repeat(np.arange(n_nodes), np.diff(indptr))  # shape [2*m]
col_idx = indices                                           # shape [2*m]

edge_index = torch.tensor(
    np.stack([row_idx, col_idx], axis=0),
    dtype=torch.long
)  # shape [2, 2*m_edges]
```

**Build `x_all` cho GNN-centrality và GNN-full (cần thêm degree/pagerank/kshell):**

```python
# centrality_table.parquet từ Stage 1–2 (Person 1) — có tất cả active nodes
centrality = pd.read_parquet("data/processed/centrality_table.parquet")
# centrality columns: node_id, degree, pagerank, betweenness, kshell, views

df_all = df_all.merge(centrality[["node_id","degree","pagerank","kshell"]], on="node_id", how="left")
df_all[["degree","pagerank","kshell"]] = df_all[["degree","pagerank","kshell"]].fillna(0)

# Normalize min-max trên full graph (cùng scaler_gnn instance)
# Cho mỗi variant, chọn feat_cols phù hợp rồi fit_transform
feat_map = {
    "gnn_raw_attr":   ["views_log",  "views_per_day", "life_time"],
    "gnn_graph_only": ["degree"],
    "gnn_centrality": ["degree",     "pagerank",      "kshell"],
    "gnn_full":       ["degree",     "pagerank",      "kshell",
                       "views_log",  "views_per_day", "life_time"],
}
# Với mỗi variant: x_all = MinMaxScaler().fit_transform(df_all[feat_map[variant]])
```

**GraphSAGE architecture cụ thể (PyG):**

```python
from torch_geometric.nn import SAGEConv
import torch.nn as nn, torch.nn.functional as F

class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim, aggr='mean')   # aggr='mean' bắt buộc
        self.conv2 = SAGEConv(hidden_dim, 1,      aggr='mean')
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x.squeeze(-1)   # shape [n_nodes] — regression output
```

Ablation story:

- GNN-raw-attr vs MLP-raw-attr → giá trị của message passing
- GNN-raw-attr vs GNN-graph-only → giá trị của attributes
- GNN-raw-attr vs Group 2 baselines → giá trị của learned representations

5. Repeated training seeds + reporting (v3 Sections 8.4–8.5):
   - **5 seeds:** `[42, 123, 456, 789, 1024]` → report `mean ± std` cho mỗi metric trong `surrogate_ranking_metrics.csv`
   - **Lưu ý về BH-FDR:** Person 3 KHÔNG chạy MWU test (đó là việc của Person 2 trong structural profiling). BH-FDR correction ở đây chỉ áp dụng nếu Person 3 muốn so sánh multiple GNN variants bằng test thống kê — trong scope bình thường thì report mean±std là đủ, không cần BH-FDR. Xem Person 2 Deliverable 4 nếu cần làm thêm.

6. Runtime table (v3 Section 9.3):

   | Component                               | Metric          | Notes                       |
   | --------------------------------------- | --------------- | --------------------------- |
   | Feature precompute (degree, PR, kshell) | time            | Centrality baselines only   |
   | MC IC labeling (n_sample × N_runs)      | time            | One-time cost — từ Person 1 |
   | GNN training (5 seeds)                  | time            | With GPU                    |
   | **GNN inference (168,114 nodes)**       | **runtime_sec** | Full active graph           |
   | Node2Vec training                       | time            |                             |
   | Speedup: MC IC vs GNN inference         | **Zx**          | Key claim cho paper         |

   `runtime_sec` trong CSV = **inference only** (không tính load + precompute).

**Runtime rule (để so sánh fair):** log riêng 3 phần (precompute / train / inference). Trong `baseline_ranking_metrics.csv` để `runtime_sec` là inference time trên full active nodes, và ghi chi tiết breakdown ở file phụ `outputs/mapr2026_v3_results/runtime_breakdown.csv` (contract bắt buộc trong M0).

> **QUAN TRỌNG (v3 Section 9.3):** Nếu GNN-raw-attr là primary, **không cần tính centrality precompute time** (degree/PR/kshell) vào runtime GNN — centrality chỉ cần cho GNN-centrality và GNN-full. Việc loại bỏ centrality precompute khỏi primary GNN pipeline làm runtime so sánh **fair hơn** (và là một điểm mạnh của GNN-raw-attr: không cần expensive precompute).

**Gợi ý entrypoint (để review dễ):**

- `src/mapr2026_v3/eval_ranking_harness.py`
- `src/mapr2026_v3/run_baselines.py`
- `src/mapr2026_v3/run_surrogates.py`

**Contract của `src/mapr2026_v3/_shared.py` (để không cần hỏi nhau path):**

```python
# _shared.py — KHÔNG sửa riêng; mọi người đều đọc file này
from pathlib import Path

class PATHS:
    # ─── Processed data ──────────────────────────────────────────────────────
    # NORMALIZATION RULE: tất cả parquet chứa RAW columns (views, life_time, degree...).
    # Normalize (min-max/z-score) on-the-fly TRONG MỖI model script. KHÔNG pre-normalize.
    # Lý do: MLP dùng train-only scaler (inductive); GNN dùng full-graph scaler (transductive).
    # Pre-normalize trong parquet sẽ gây conflict giữa 2 scaler policy này.
    graph_csr         = "data/processed/graph_csr.npz"
    node_attributes   = "data/processed/node_attributes.parquet"
    centrality_table  = "data/processed/centrality_table.parquet"
    community_features= "data/processed/community_features.parquet"   # Person 2 owns; separate from node_attributes
    ic_scores         = "data/processed/ic_scores_primary.parquet"
    regression_tgts   = "data/processed/regression_targets.parquet"
    split_masks       = "data/processed/split_masks.parquet"
    diffusion_proxies = "data/processed/diffusion_proxies.parquet"
    typology_labels   = "data/processed/typology_labels_ic_views.parquet"
    # ─── Outputs ─────────────────────────────────────────────────────────────
    lcc_report        = "outputs/stage0_data_quality/lcc_report.json"
    day1_dir              = "outputs/day1_benchmark"
    ic_runtime_benchmark  = "outputs/day1_benchmark/ic_runtime_benchmark.json"
    one_hop_correlation   = "outputs/day1_benchmark/one_hop_correlation.json"
    ic_pilot_diagnostics  = "outputs/day1_benchmark/ic_pilot_diagnostics.json"
    results_dir           = "outputs/mapr2026_v3_results"
    baseline_csv      = "outputs/mapr2026_v3_results/baseline_ranking_metrics.csv"
    surrogate_csv     = "outputs/mapr2026_v3_results/surrogate_ranking_metrics.csv"
    runtime_csv       = "outputs/mapr2026_v3_results/runtime_breakdown.csv"
```

> Nếu cần thêm path mới → thêm vào `PATHS` trong `_shared.py` rồi commit riêng — không hardcode path string trong business logic.

**Cách test harness trước khi IC labels xong (mock mode):**

```python
# Dùng sis_table.parquet làm mock regression target
import pandas as pd
from _shared import PATHS
from eval_ranking_harness import load_split_mask, apply_test_mask, compute_metrics

# Nếu split_masks.parquet chưa có: nhờ Person 1 chạy ic_labels_primary.py --dry-run
mask = load_split_mask(PATHS.split_masks)

# Dùng pagerank hoặc sis_score làm y_true tạm
df_mock = pd.read_parquet("data/processed/sis_table.parquet")
df_mock = df_mock.rename(columns={"sis_score": "y"})  # hoặc cột tương đương

df_test = apply_test_mask(df_mock, mask)
# y_pred = views rank (Group 1 baseline)
df_test["y_pred"] = df_test["y"].rank()
metrics = compute_metrics(df_test["y"].values, df_test["y_pred"].values)
print(metrics)
```

**Runbook tối thiểu:**

- M1: Chạy 2 lệnh dry-run (Mục 2.1C). Verify `load_split_mask()` không lỗi với mock split_masks.
- M3: Sau khi Person 1 cung cấp IC labels thật: thay mock y bằng `regression_targets.parquet`, chạy lại toàn bộ.
- `run_baselines.py` real mode: load `regression_targets.parquet` → filter test mask → `y_pred` cho Group 1–3 → `compute_metrics()`.
- `run_surrogates.py`: train 5 seeds (`training_seeds = [42, 123, 456, 789, 1024]`), report mean±std.

**Gợi ý phân tách file để giảm conflict:**

- Person 3 tập trung `src/mapr2026_v3/eval_ranking_harness.py`, `run_baselines.py`, `run_surrogates.py` và `src/ml/*`.
- Không chạm vào `src/simulation/*` hay `src/mapr2026_v3/ic_labels_primary.py`.

**DoD cho Track C:**

- `load_split_mask()` + `apply_test_mask()` chạy không lỗi với mock artifacts (M1).
- `baseline_ranking_metrics.csv` có đủ **Group 1–4** rows với real IC labels (M4): Group 1 (views/views_day/degree), Group 2 (PR/kshell/betweenness), Group 3 (one-hop/two-hop), Group 4 (Node2Vec+LR, MLP raw attr).
- GNN-raw-attr chạy được 5 seeds, `surrogate_ranking_metrics.csv` có `spearman_rho_mean`, `spearman_rho_std`, `ndcg_mean`, `ndcg_std`, `runtime_sec` (M5).
- Runtime table có `Speedup: MC IC vs GNN inference` được tính (M5).
- `runtime_sec` = full-graph inference time (đo `time.time()` bao toàn bộ forward pass, không tính file load).
- ⚠ **Person 3 KHÔNG chạy MWU** — BH-FDR là trách nhiệm của Person 2 (structural profiling). Person 3 chỉ report `mean±std` trên 5 seeds.
- **[✦ IF TIME — sau M5; ~1h; nâng thành MUST nếu predictions sẵn sàng trước 25/4]** `per_group_prediction_error.csv` tồn tại; Hidden group có row (n_test≈57 ✓); Hidden là hardest-to-predict group → highlight trong paper Section 4.4.

---

## 4) Nhịp tích hợp (deadline 30/4/2026 — còn 25 ngày kể từ 6/4)

### Milestone M0 — Kick-off (6/4, buổi sáng, ~1 giờ)

**Mục tiêu:** đồng thuận trước khi ai code. Không skip.

Agenda bắt buộc:

1. Xác nhận Stage 0–2 artifacts tồn tại trên máy mọi người (chạy quickstart Mục 0)
2. Điền và commit `docs/m0_decisions.md` (xác nhận 8 quyết định đã lock)
3. Phân công branch: `feature/mapr-ic-core`, `feature/mapr-typology-proxies`, `feature/mapr-surrogate-eval`
4. Mỗi người chạy dry-run script của mình (Mục 2.1) và confirm không lỗi

**Done khi:** `docs/m0_decisions.md` được commit, cả 3 người chạy được dry-run.

> ⚠ **Theo v3 Implementation Plan:** paper writing (Intro + Related Work + Methodology) phải bắt đầu từ **Ngày 8/4**. Nếu chưa assign người viết paper tại M0, phải quyết định ngay tại buổi M0 — không để unassigned.

---

### Milestone M1 — Unblock song song (6/4 buổi chiều → 7/4)

**Mục tiêu:** tạo placeholder artifacts đúng schema để 3 track không bị block nhau.

| Person   | Việc phải merge                    | Artifact tạo ra                                                           |
| -------- | ---------------------------------- | ------------------------------------------------------------------------- |
| Person 1 | `export_csr.py` (small mode)       | `graph_csr.npz`                                                           |
| Person 1 | Dead account audit + LCC check     | `outputs/stage0_data_quality/lcc_report.json`, `dead_account_report.json` |
| Person 1 | `ic_labels_primary.py --dry-run`   | `split_masks.parquet` (mock)                                              |
| Person 2 | `diffusion_proxies.py --dry-run`   | `diffusion_proxies.parquet` (header)                                      |
| Person 3 | `eval_ranking_harness.py` skeleton | import `load_split_mask()` chạy được                                      |

**Done khi:** Person 3 có thể import `load_split_mask(PATHS.split_masks)` mà không lỗi (mock file tồn tại).

---

### Milestone M2 — Day-1 decisions (7/4, cuối ngày)

**Mục tiêu:** lock compute budget và GNN narrative — không implement surrogate trước milestone này.

| Person   | Việc                                                      | Output                                                                                  |
| -------- | --------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Person 1 | Chạy `day1_benchmark.py` (real mode, 100 nodes × 50 runs) | `ic_runtime_benchmark.json`                                                             |
| Person 1 | Chạy one-hop ρ check (200 pilot nodes × 50 runs)          | `one_hop_correlation.json`                                                              |
| Cả team  | Họp online 30 phút, đọc kết quả                           | Điền **Phần 3** của `docs/day1_decisions.md` (`n_sample`, `N_runs`, `narrative_branch`) |

**Decision gate:**

> ⚠ **Naming note:** dùng thống nhất `n_sample` (số labeled nodes được chọn để chạy IC simulation), tránh dùng thuật ngữ khác để không nhầm với IC random seeds; mỗi node chỉ dùng 1 deterministic seed = `42 + node_index`.

- `projected_hours < 4h` → **n_sample=5000**, N_runs=200
- `4h–8h` → **n_sample=3000**, N_runs=150
- `> 8h` → **n_sample=2000**, N_runs=100 (ghi limitation)
- `one_hop_rho < 0.8` → GNN là primary contribution
- `one_hop_rho 0.8–0.9` → GNN + 2-hop baseline head-to-head
- `one_hop_rho > 0.9` **and** `jaccard_at_10pct > 0.8` **and** `ndcg_at_10pct > 0.9` → **restructure:** proxies là primary, GNN là secondary; title changes
- `one_hop_rho > 0.9` nhưng `jaccard_at_10pct <= 0.8` hoặc `ndcg_at_10pct <= 0.9` → giữ GNN + 2-hop, không restructure ngay

> 📋 **[REFERENCE — không phải task thêm]** Sau khi có Day-1 results, chọn narrative phù hợp và ghi vào `docs/day1_decisions.md`. Bảng dưới cung cấp câu văn sẵn có để không mất thời gian viết từ đầu.

**Prepared fallback narratives:**

| Tình huống                                                                         | Narrative                                                                                                                                                                                                                                                                           |
| ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `one_hop_rho > 0.9` **and** `jaccard_at_10pct > 0.8` **and** `ndcg_at_10pct > 0.9` | "We find that one-hop analytical spread (O(E)) strongly aligns with IC both globally and at top-k. Analytical proxies become the primary modeling narrative; GNN is retained as a secondary lens for divergence/evolution analysis."                                                |
| `one_hop_rho > 0.9` nhưng top-k alignment chưa cao                                 | "Although global ranking correlation is high, top-k mismatch remains non-trivial. We keep a head-to-head GNN vs 2-hop comparison and focus on top-k divergence patterns."                                                                                                           |
| `GNN-raw-attr ≤ two-hop`                                                           | "2-hop analytical approximation (naive full-graph complexity gần O(Σ d(v)^2)) achieves ρ ≈ X with MC IC, closely matching GNN surrogate — weighted-cascade dynamics are well-approximated by local structural summaries. GNN's value lies in efficient inference as graph evolves." |
| `views/IC ρ > 0.8`                                                                 | "We find high popularity-diffusion agreement (ρ > 0.8) on Twitch's dense graph. The small divergent subset (Hidden influencers) shows systematically higher betweenness and cross-community connectivity."                                                                          |

**Template của `docs/day1_decisions.md`** (Person 1 tạo file, team cùng điền):

```markdown
## Phần 1 — IC Runtime Benchmark (6/4 sáng)

- per_sim_ms: \_\_
- projected_total_hours: \_\_
- Decision: n_sample = **, N_runs = **

## Phần 2 — One-Hop ρ Check (6/4 chiều)

- spearman_rho: \_\_
- jaccard_at_10pct: \_\_
- ndcg_at_10pct: \_\_
- narrative_branch: [gnn_primary / gnn_and_2hop / restructure]

## Phần 3 — Compute Budget Lock (điền tại M2 team meeting — 7/4)

- n_sample (final confirmed): \_\_
- N_runs (final confirmed): \_\_
- narrative_branch (confirmed): \_\_
- Signed off: Person 1 **, Person 2 **, Person 3 \_\_

## Phần 4 — Views/IC Alignment (điền sau khi có IC labels — M3)

- spearmanr(views, ic_score_mean): rho = **, p = **
- RQ2 narrative tier: [strong_divergence / moderate / high_agreement]
```

> **Phân biệt 2 files:** `docs/m0_decisions.md` = quyết định kiến trúc cố định (threshold, split, scope — lock tại M0). `docs/day1_decisions.md` = quyết định runtime (compute budget, GNN narrative, views/IC ρ — điền dần theo tiến độ).

**Done khi:** `docs/day1_decisions.md` Phần 3 được commit với `n_sample`, `N_runs`, `narrative_branch`.

---

### Milestone M3 — IC labels primary (8–12/4)

**Mục tiêu:** IC labels thật → unblock toàn bộ pipeline.

| Person   | Việc                                                                  | Deadline gợi ý |
| -------- | --------------------------------------------------------------------- | -------------- |
| Person 1 | MC IC simulation (full n_sample × N_runs từ M2)                       | 10/4           |
| Person 1 | Label stability report (Jaccard + per-quintile CV) + split_masks thật | 10/4           |
| Person 2 | Build typology IC×views (real IC labels)                              | 12/4           |
| Person 3 | Chạy baseline ranking thật (Group 1–2)                                | 12/4           |

**Done khi:** `baseline_ranking_metrics.csv` có ít nhất Group 1–2 rows với real IC labels.

**[M3] Views/IC alignment check — narrative lookup cho RQ2:**

> 📋 **[REFERENCE — không phải task thêm]** Sau khi Person 1 ghi `spearmanr(views, ic_score_mean)` vào `docs/day1_decisions.md` Phần 4, cả team tra bảng và chọn narrative tương ứng. Không cần chạy thêm experiment.

| views/IC Spearman ρ | Narrative RQ2                                                                                                                                                                                        |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ρ < 0.70            | "Strong divergence: popularity không phản ánh diffusion potential — Hidden influencers tồn tại và chiếm vai trò cấu trúc (betweenness cao, cross-community cao)."                                    |
| 0.70–0.85           | "Moderate divergence: Hidden quadrant tồn tại và structurally distinct — standard narrative cho paper."                                                                                              |
| ρ > 0.85            | "High popularity-diffusion agreement. Hidden influencers (small subset) vẫn show systematically higher betweenness và cross-community connectivity — emphasize structural distinction thay vì size." |

---

### Milestone M4 — Full pipeline (12–22/4)

| Person   | Việc                                                                           | Deadline gợi ý |
| -------- | ------------------------------------------------------------------------------ | -------------- |
| Person 2 | **Community detection** (Louvain + cross_community_edge_fraction)              | 10/4           |
| Person 2 | Structural profiling (MWU + Cliff's Δ + BH-FDR)                                | 18/4           |
| Person 2 | life_time validation (partial Spearman + stratified MWU)                       | 18/4           |
| Person 2 | Null model comparison (500 nodes × 3 × 100) + summary JSON                     | 18/4           |
| Person 3 | Group 3 baselines (one-hop/two-hop từ proxies full graph)                      | 15/4           |
| Person 3 | Node2Vec (`dim=64, walks=20`) + LR + MLP raw attr                              | 18/4           |
| Person 3 | GNN-raw-attr + 3 ablation variants (**[MUST nếu M2: gnn_branch_viable=true]**) | 22/4           |
| Person 3 | Runtime table + Speedup calculation                                            | 22/4           |

---

### Milestone M5 — Integration + paper hand-off (22–27/4)

- Tất cả artifacts gom vào `outputs/mapr2026_v3_results/`
- Final `baseline_ranking_metrics.csv` (Groups 1–4) + `surrogate_ranking_metrics.csv` (Group 5 — GNN) hoàn chỉnh
- `runtime_breakdown.csv` hoàn chỉnh (precompute / train / inference riêng biệt)
- Bàn giao cho người viết paper: bảng kết quả + plots chính

---

## 4b) Risk Management (v3 Section 19)

| Rủi ro                                                                                      | Xác suất   | Impact       | Action                                                                                                                                                                                      |
| ------------------------------------------------------------------------------------------- | ---------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `one_hop_rho > 0.9` + top-k alignment cao (`jaccard_at_10pct > 0.8`, `ndcg_at_10pct > 0.9`) | Trung bình | **Critical** | M2: check trước; nếu đủ 3 điều kiện thì restructure, nếu không giữ GNN + 2-hop                                                                                                              |
| IC runtime > 8h                                                                             | Trung bình | **Critical** | M2: reduce n_sample=2k, N_runs=100; ghi limitation                                                                                                                                          |
| GNN không beat cheap proxies                                                                | Trung bình | Thấp         | Prepared narrative "negative result" vẫn publishable                                                                                                                                        |
| Hidden quadrant < 150 nodes                                                                 | Trung bình | Cao          | Expand 8–10k sample + Sample B strategy (Person 2 Mục 3)                                                                                                                                    |
| `views/IC ρ > 0.8`                                                                          | Trung bình | Thấp         | Prepared narrative "high agreement" (Mục 4 M2)                                                                                                                                              |
| Thiếu permutation null package (B5)                                                         | Trung bình | Cao          | Không sign-off Stage 5 nếu thiếu `views_permutation_null_summary.json` hoặc `ic_permutation_null_summary.json`                                                                              |
| Louvain partition quá nhạy với resolution (B9)                                              | Trung bình | Trung bình   | Chạy `louvain_resolution_sensitivity.json`; nếu `<20 communities` hoặc `top3>50%` thì nghi over-merge, nếu `>200` + nhiều singleton thì nghi over-split; chỉ đổi resolution sau khi re-lock |
| Overclaim accuracy trên unlabeled nodes (B10)                                               | Trung bình | Trung bình   | Khóa wording transductive: metrics chỉ trên held-out labeled; full-graph chỉ runtime; nếu cần claim rộng hơn thì chạy out-of-sample IC audit                                                |
| loky OOM với full graph                                                                     | Thấp       | Cao          | Reduce `n_jobs`; monitor RAM ≥ 32 GB khi chạy                                                                                                                                               |
| PyG installation issues                                                                     | Thấp       | Trung bình   | Setup M0; fallback: DGL nếu PyG fail                                                                                                                                                        |
| Paper > 6 trang                                                                             | Trung bình | **Blocker**  | Cắt theo bảng dưới                                                                                                                                                                          |

## 4c) Scope Reduction — Cắt khi cần (v3 Section 16)

Nếu timeline tight, cắt theo thứ tự này (an toàn nhất trước):

| Có thể cắt                                 | Phải giữ bắt buộc                                                            |
| ------------------------------------------ | ---------------------------------------------------------------------------- |
| Uniform-p sensitivity variant              | Weighted cascade IC (primary)                                                |
| Graph perturbation test                    | Label stability report (3 seeds; binary-ready nếu Jaccard ≥ 0.85)            |
| 5% / 15% thresholds (chỉ giữ 10%)          | Null package: configuration model + views-permutation + IC-score permutation |
| Eigenvector/betweenness trong GNN features | One-hop + two-hop proxies (Group 3)                                          |
| GNN-full variant                           | Community detection (Louvain + cross_comm_fraction)                          |
| Bootstrap CI                               | GNN-raw-attr (primary) + GNN-graph-only (ablation)                           |
| Detailed betweenness profiling             | BH-FDR correction cho tất cả MWU                                             |
| Secondary metrics (P@10%)                  | life_time validation của typology                                            |

**Quy tắc:** Không cắt bất kỳ mục nào ở cột phải mà không thảo luận cả team + ghi vào `docs/experiment_registry.md`.

---

## 5) Cách tích hợp vào runner mà vẫn giữ song song

Để tránh 3 người cùng sửa `run_all.py` và gây conflict, khuyến nghị làm theo 2 bước:

1. **Mỗi track có entrypoint script riêng** (chạy trực tiếp `python path/to/script.py`) để dev/test độc lập.
2. Chỉ sau khi đạt Milestone M2 hoặc M3 mới gom vào runner (1 PR nhỏ) để chạy end-to-end.

Gợi ý (không bắt buộc) cách “đặt chỗ” trong runner:

- Giữ Stage 0–3 như cũ (SIS-based).
- Tái sử dụng Stage 4–7 hiện có như “MAPR2026 v3 stages” và thay các stubs bằng script thật, **hoặc** thêm stage mới (8+) nếu muốn giữ backward-compat.

### Chuẩn hóa output folders (đỡ lo mỗi người ghi một kiểu)

Tối thiểu cần có (tạo folder nếu chưa có):

- `outputs/day1_benchmark/`
- `outputs/mapr2026_v3_results/`

### Code reuse vs. fork — quy tắc không sửa đè code cũ

Repo đang có pipeline SIS-based (Stage 0–3). Tất cả script MAPR2026 v3 mới phải nằm trong `src/mapr2026_v3/` — **không sửa đè code cũ**.

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

### Quick Reference: Stage ↔ Script ↔ Owner

| Stage v3                                   | Script                                    | Owner        | Ghi chú                                     |
| ------------------------------------------ | ----------------------------------------- | ------------ | ------------------------------------------- |
| **Stage 0b (dead account audit)**          | `src/data/dead_account_audit.py`          | **Person 1** | Phải có trước sampling; stats → limitations |
| Stage 2 (CSR)                              | `src/mapr2026_v3/export_csr.py`           | Person 1     |                                             |
| Stage 3 (Day-1)                            | `src/mapr2026_v3/day1_benchmark.py`       | Person 1     | Gating cho M2                               |
| Stage 4 (IC labels + split mask)           | `src/mapr2026_v3/ic_labels_primary.py`    | Person 1     | Gating cho M3                               |
| **Stage 4b (community features)**          | `src/graph/community.py`                  | **Person 2** | Độc lập với IC, chạy sớm                    |
| Stage 5 (typology + profiling + life_time) | `src/mapr2026_v3/typology_ic_views.py`    | Person 2     | Cần IC labels + community                   |
| Stage 5 (null model)                       | `src/mapr2026_v3/null_model_typology.py`  | Person 2     | 500 nodes × 3 × 100 runs                    |
| Stage 6 (proxies full graph)               | `src/mapr2026_v3/diffusion_proxies.py`    | Person 2     | Full active graph                           |
| Stage 7 (Group 1–4 baselines)              | `src/mapr2026_v3/run_baselines.py`        | Person 3     | Group 4 = Node2Vec+LR, MLP → baseline CSV   |
| Stage 7 (Group 5 GNN ablation)             | `src/mapr2026_v3/run_surrogates.py`       | Person 3     | 4 GNN variants; mean±std → surrogate CSV    |
| (shared)                                   | `src/mapr2026_v3/eval_ranking_harness.py` | Person 3     | `load_split_mask()` + metrics               |
| (shared)                                   | `src/mapr2026_v3/_shared.py`              | All          | Đọc, không sửa riêng                        |

---

## 6) Giảm phụ thuộc bằng test nhỏ + smoke runs

- Mỗi track có 1 chế độ `--dry-run` hoặc “small graph mode” (subgraph vài nghìn nodes) để:
  - test correctness
  - đo runtime sơ bộ
  - giảm thời gian review/CI

- Luôn có 3 check tối thiểu cho artifact mới:
  1. schema check (cột bắt buộc)
  2. coverage check (tỷ lệ node missing)
  3. determinism check (rerun cùng seed cho ra kết quả giống/close)

---

## 7) PR strategy (để merge nhanh mà ít conflict)

- Mỗi người làm 1 branch riêng theo pattern:
  - `feature/mapr-ic-core-*` (Person 1)
  - `feature/mapr-typology-proxies-*` (Person 2)
  - `feature/mapr-surrogate-eval-*` (Person 3)

- Ưu tiên PR nhỏ theo milestone (M1 → M2 → M3) thay vì chờ “xong hết mới merge”.
- Khi PR chạm vào artifact contract: phải update cùng lúc doc contract (Mục 2) + registry.

---

## 7b) Hard-stop hoặc Branch-switch conditions

Nếu gặp condition dưới đây, thực hiện action tương ứng; **chỉ dừng pipeline khi action yêu cầu hard-stop**:

| Condition                                                                                                  | Triệu chứng                                              | Action                                                                                                                                                                                                                          |
| ---------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CSR không deterministic                                                                                    | Rerun ra mapping khác                                    | Fix `export_csr.py`: sort `node_id` trước khi build                                                                                                                                                                             |
| `split_masks.parquet` thiếu                                                                                | Person 3 raise `FileNotFoundError`                       | Person 1 chạy `ic_labels_primary.py --dry-run` ngay                                                                                                                                                                             |
| `split_masks.parquet` sai schema                                                                           | `load_split_mask()` raise `ValueError`                   | Person 1 regenerate                                                                                                                                                                                                             |
| One-hop ρ > 0.9 + top-k alignment cao (`Jaccard@10% > 0.8`, `NDCG@10% > 0.9`) nhưng vẫn cố giữ GNN primary | Narrative không defensible                               | Restructure: xem Section 2.2 plan v3                                                                                                                                                                                            |
| `cv_score < 0.3` (đọc nhánh Option B vs hard-stop)                                                         | Gate fail ở pilot diagnostics                            | ⚠ [IF PROBLEM: cv_score < 0.3] — nếu IC không degenerate: kích hoạt Option B và regression tiếp tục; chỉ hard-stop khi IC degenerate (đủ 3 điều kiện: `median_reach < 2` + `p_reach_gt_1 < 0.20` + `top10_to_median_ratio < 2`) |
| Hidden quadrant < 150                                                                                      | `check_and_expand_typology_sample` cảnh báo              | ⚠ [IF PROBLEM: min_quadrant_ok=false] — xem Person 2 Deliverable 3                                                                                                                                                              |
| `diffusion_proxies.parquet` chỉ có labeled subset                                                          | `n_nodes` trong file << 168k                             | Person 2 rebuild ở real mode (full active graph)                                                                                                                                                                                |
| Louvain partition quá nhạy với resolution                                                                  | `n_communities`/modularity drift mạnh giữa `0.5/1.0/2.0` | ⚠ [IF PROBLEM: louvain_partition_instability] Chạy `louvain_resolution_sensitivity.json`; chỉ đổi resolution sau khi re-lock                                                                                                    |

---

## 8) Checklist nhanh theo người (tóm tắt 1 trang)

### Person 1 — Phạm Quốc Vĩnh

| #   | Việc                                                | Script                           | Artifact output                                                                                             | Deadline |
| --- | --------------------------------------------------- | -------------------------------- | ----------------------------------------------------------------------------------------------------------- | -------- |
| 1   | CSR export                                          | `export_csr.py`                  | `graph_csr.npz`                                                                                             | M1 (6/4) |
| 2   | **Dead account audit**                              | `src/data/dead_account_audit.py` | `outputs/stage0_data_quality/dead_account_report.json`                                                      | M0 (6/4) |
| 3   | **LCC check**                                       | `src/data/dead_account_audit.py` | `outputs/stage0_data_quality/lcc_report.json`                                                               | M0 (6/4) |
| 4   | Day-1 benchmark                                     | `day1_benchmark.py`              | `ic_runtime_benchmark.json`                                                                                 | M2 (7/4) |
| 5   | One-hop ρ check                                     | `day1_benchmark.py`              | `one_hop_correlation.json`                                                                                  | M2 (7/4) |
| 6   | IC pilot + stability                                | `ic_labels_primary.py`           | `outputs/day1_benchmark/ic_pilot_diagnostics.json` (`jaccard_stability`, `cv_score`, per-quintile CV table) | 9/4      |
| 7   | IC labels (full N×R)                                | `ic_labels_primary.py`           | `ic_scores_primary.parquet`, `regression_targets.parquet`, `classification_labels.parquet`                  | 10/4     |
| 8   | **[MUST khi Jaccard < 0.85] Stability explanation** | manual/script (extract phase1/2) | `outputs/day1_benchmark/stability_explanation.json`                                                         | 10/4     |
| 9   | **Split mask** [M0-locked]                          | `ic_labels_primary.py`           | `split_masks.parquet` (cùng lúc #7)                                                                         | 10/4     |
| 10  | Ghi `day1_decisions.md`                             | manual                           | `docs/day1_decisions.md`                                                                                    | M2 (7/4) |
| 11  | **[M3] Views/IC alignment check**                   | `ic_labels_primary.py`           | cập nhật `docs/day1_decisions.md` Phần 4 (`spearmanr(views, ic_score_mean)`)                                | M3       |

### Person 2 — Trần Hùng Vĩ

| #   | Việc                                                                     | Script                            | Artifact output                                                                                                                                                                 | Deadline |
| --- | ------------------------------------------------------------------------ | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| 1   | Proxies skeleton (dry-run)                                               | `diffusion_proxies.py --dry-run`  | dry-run header cho `data/processed/diffusion_proxies.parquet` (schema only; KHÔNG dùng cho evaluation/runtime)                                                                  | M1 (7/4) |
| 2   | **[MUST] Community detection**                                           | `src/graph/community.py`          | `data/processed/community_features.parquet` (columns: `node_id`, `community_id`, `cross_community_edge_fraction`; scope=ALL active nodes, coverage=100%, `node_id` kiểu string) | 10/4     |
| 3   | **Proxies thật (full graph)**                                            | `diffusion_proxies.py`            | `data/processed/diffusion_proxies.parquet` + `outputs/mapr2026_v3_results/runtime_breakdown.csv`                                                                                | 15/4     |
| 4   | Typology IC×views                                                        | `typology_ic_views.py --pct 0.10` | `data/processed/typology_labels_ic_views.parquet` + `outputs/mapr2026_v3_results/typology_quadrant_report.json`                                                                 | 12/4     |
| 5   | Structural profiling (MWU + Cliff's Δ + BH-FDR)                          | `typology_ic_views.py`            | `outputs/mapr2026_v3_results/structural_profiling.csv`                                                                                                                          | 18/4     |
| 6   | **life_time validation typology**                                        | `typology_ic_views.py`            | `outputs/mapr2026_v3_results/lifetime_validation.json`                                                                                                                          | 18/4     |
| 7   | Null model (500 nodes × 3 × 100 runs)                                    | `null_model_typology.py`          | `outputs/mapr2026_v3_results/null_model_typology_summary.json`                                                                                                                  | 18/4     |
| 8   | Views-permutation null (**[MUST — B5 core]**)                            | `typology_ic_views.py`            | `views_permutation_null_summary.json`                                                                                                                                           | 18/4     |
| 9   | IC-score permutation null (**[MUST — B5 core]**)                         | `typology_ic_views.py`            | `ic_permutation_null_summary.json`                                                                                                                                              | 18/4     |
| 10  | ⚠ [IF PROBLEM: min_quadrant_ok=false sau two-sample] Residual divergence | `typology_ic_views.py`            | `residual_divergence_report.json` — chỉ chạy khi `min_quadrant_ok=false` sau two-sample strategy                                                                                | 18/4     |
| 11  | **[MUST] Metric correlation matrix (global 8×8)**                        | `typology_ic_views.py`            | `outputs/mapr2026_v3_results/metric_correlation_matrix.json` (`rho_matrix`, `p_matrix_corrected`; `rho_by_degree_quintile` là ✦ IF TIME)                                        | 18/4     |

> Chú ý: **Bước 4 cần `ic_scores_primary.parquet`** (~10/4). Trong khi chờ: dùng `sis_table.parquet` làm mock.
> **Bước 2 không phụ thuộc IC labels** — có thể làm ngay từ đầu song song với bước 1.

### Person 3 — Trần Quốc Hải

| #   | Việc                                                                                                              | Script                                 | Artifact output                                                            | Deadline |
| --- | ----------------------------------------------------------------------------------------------------------------- | -------------------------------------- | -------------------------------------------------------------------------- | -------- |
| 1   | Harness skeleton                                                                                                  | `eval_ranking_harness.py`              | `load_split_mask()` + `compute_metrics()` OK                               | M1 (7/4) |
| 2   | Baselines Group 1–2 (mock labels)                                                                                 | `run_baselines.py`                     | `baseline_ranking_metrics.csv` (mock)                                      | 9/4      |
| 3   | **Baselines Group 1–2 (real IC)**                                                                                 | `run_baselines.py`                     | CSV real (Group 1: views/views_day/degree, Group 2: PR/kshell/betweenness) | 12/4     |
| 4   | Baselines Group 3 (proxies)                                                                                       | `run_baselines.py`                     | CSV + one-hop/two-hop rows                                                 | 15/4     |
| 5   | **Group 4 — Node2Vec + LR** (`dim=64, walks=20`)                                                                  | `run_baselines.py`                     | `baseline_ranking_metrics.csv` (thêm rows Group 4)                         | 18/4     |
| 6   | **Group 4 — MLP raw attr** (`views_log, views/day, life_time`)                                                    | `run_baselines.py`                     | cập nhật `baseline_ranking_metrics.csv`                                    | 18/4     |
| 7   | **[MUST nếu M2: gnn_branch_viable=true] Group 5 — GNN-raw-attr** (5 seeds)                                        | `run_surrogates.py`                    | `surrogate_ranking_metrics.csv` (mean±std)                                 | 22/4     |
| 8   | **[MUST nếu M2: gnn_branch_viable=true] Group 5 — GNN ablation**: graph-only + centrality; ✦ [IF TIME] + GNN-full | `run_surrogates.py`                    | cập nhật `surrogate_ranking_metrics.csv`                                   | 22/4     |
| 9   | Runtime table + Speedup MC vs GNN                                                                                 | manual/script                          | `outputs/mapr2026_v3_results/runtime_breakdown.csv` hoàn chỉnh             | 22/4     |
| 10  | ✦ [IF TIME nếu M2: gnn_branch_viable=true] GNN-random sanity-check                                                | `run_surrogates.py`                    | thêm `gnn_random` row trong `surrogate_ranking_metrics.csv`                | 22/4     |
| 11  | ✦ [IF TIME; nâng thành MUST nếu predictions đủ trước 25/4] Per-group prediction error                             | `run_baselines.py`/`run_surrogates.py` | `outputs/mapr2026_v3_results/per_group_prediction_error.csv`               | 25/4     |

> **Các bước có tính metrics (2–8, 10, 11):** load `split_masks.parquet` → `apply_test_mask()` → `compute_metrics()`. Không tự tạo split.
> **Group 4 vs Group 5:** Node2Vec+LR và MLP vào `baseline_ranking_metrics.csv` (comparable với Group 1–3). GNN variants vào `surrogate_ranking_metrics.csv` (với mean±std vì 5 seeds).
> **BH-FDR:** áp dụng cho MWU của Person 2 (structural profiling + `life_time` validation). Person 3 không bắt buộc chạy MWU.

---

## 8b) Minimal handoff package giữa teammates

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
- `docs/day1_decisions.md` (chứa `n_sample`, `N_runs`, `narrative_branch`)

**Từ Person 2 → Person 3:**

- `data/processed/diffusion_proxies.parquet` (full graph)
- `outputs/mapr2026_v3_results/runtime_breakdown.csv`

**Option B lockstep rules — áp dụng khi `quality_gate_pass_all=false`:**

Active handoff version: `person1_day1_20260408_p1_day1_v3h_optionB_lockstep`

1. Dùng đúng 1 version tag handoff cho toàn bộ experiment cycle — không mix artifacts từ các version khác nhau.
2. Không tự re-split data local — chỉ load `data/processed/split_masks.parquet` từ handoff (SHA256: `005de40762f6c75e4df66a53efeaa883d126d52abd5c4af0224d736992362104`).
3. Giữ canonical branch (`classification_labels.parquet`) và consensus branch (`classification_labels_consensus.parquet`) tách biệt — không ghi đè canonical.
4. Binary metrics phải khai báo uncertainty: loại `is_uncertain=1` hoặc `vote_count=1` khi claim strict binary performance; ghi rõ evaluation scope (all nodes vs non-uncertain subset).
5. Regression là PRIMARY objective — dùng `regression_targets.parquet` (`y = log1p(ic_score_mean)`) cho toàn bộ surrogate ranking pipeline.
6. Nếu cần thay đổi artifacts: tạo version tag mới (`freeze_day1_handoff.py --version-tag <new_tag>`) — không overwrite handoff directory đã có.
