# MAPR2026 v3 — Team 3 người: Kế hoạch coding song song (không bao gồm viết paper)

Mục tiêu của tài liệu này là thiết kế các đầu việc **có thể triển khai song song tối đa** cho team 3 người để migrate từ pipeline hiện tại (SIS-based, stage 0–3 đã ổn) sang MAPR2026 v3.1 — **Linear pipeline**: [1] MC-IC as metric → [2] MC-IC đắt → [3] Regression nature → [4] GNN surrogate (GCN/GIN/GAT/SAGE).

Phạm vi:

- **Chỉ phần thực thi code + tạo artifacts + chạy pipeline**.
- Không bao gồm viết paper/narrative (đã có người khác phụ trách).

**Scope bridge:** Tài liệu này là execution plan cho team 3 người coding. `MAPR2026_Implementation_Plan_v3.md` là strategic master plan (research + narrative + publication framing). Nếu khác biệt ở thao tác thực thi hằng ngày, ưu tiên file này; nếu khác biệt về framing nghiên cứu/paper, ưu tiên master plan. **v3.1 override:** Framing research/paper theo linear pipeline [1]→[4].

---

## Cách đọc file này — 5 mức độ task

> **Đọc phần này trước.** File plan có 5 loại nội dung với mức độ ưu tiên khác nhau:

| Loại                  | Ký hiệu                                                        | Ý nghĩa                                                                                 | Khi nào thực thi                                                  |
| --------------------- | -------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| **MUST**              | _(không có marker — body text thường; hoặc `[MUST]` explicit)_ | Bắt buộc cho paper defensible                                                           | Luôn làm theo đúng thứ tự                                         |
| **APPENDIX support**  | `**[APPENDIX support — v3.1 demoted]**`                        | Phần hỗ trợ/secondary; KHÔNG block main submission; cắt **đầu tiên** nếu tight deadline | Chỉ sau khi xong hoàn toàn tất cả MUST; làm nếu có thời gian thừa |
| **Dự phòng**          | `> ⚠ [IF PROBLEM: <điều kiện>]`                                | Phương án thay thế khi gặp vấn đề cụ thể                                                | CHỈ khi điều kiện trigger xảy ra                                  |
| **Nếu còn thời gian** | `> ✦ [IF TIME]`                                                | Tăng thêm chất lượng/robustness                                                         | Sau khi hoàn thành tất cả MUST trước deadline                     |
| **Tham khảo**         | `> 📋 [REFERENCE — không phải task thêm]`                      | Chỉ để align narrative/decision, không phát sinh task mới                               | Đọc để thống nhất ngữ cảnh, KHÔNG đưa vào task list               |

> **Quy tắc vàng:** Lần đọc đầu — đọc body text bình thường, **bỏ qua toàn bộ các block `⚠ [IF PROBLEM]`, `✦ [IF TIME]`, và `[APPENDIX support]`**. Quay lại các block đó khi và chỉ khi: điều kiện trigger xảy ra (IF PROBLEM), hoặc còn thời gian thừa sau khi done MUST (IF TIME), hoặc main story hoàn chỉnh và còn budget (APPENDIX support). Không để chúng làm phình scope hoặc trễ timeline.
>
> **⚠ Chú ý đặc biệt cho `[APPENDIX support]`:** Đây KHÔNG phải MUST dù không có marker `[IF TIME]`. Tất cả deliverables mang nhãn `[APPENDIX support — v3.1 demoted]` là **optional** và phải được cắt trước mọi thứ khác khi tight deadline. Chỉ làm sau khi Task A (IC/labels) và Task C (GNN surrogate) đã xong hoàn toàn.

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
- **Views-independence:**
  - **A0 (primary) + A1/A2 (structural sensitivity)** phải **views-independent** (views chỉ dùng ở evaluation/runtime breakdown, không đi vào p(u,v)).
  - **I-A Attribute-Informed IC (supplemental, nếu bật):** được phép dùng `views` trong $p(u,v)$ như **label set bổ sung**; bắt buộc label rõ "attribute-informed operationalization" và phải qua pilot decision protocol trước khi chạy full. A0 vẫn là primary.
- Graph dùng **undirected** (`graph_directed: false`) — MUSAE Twitch chỉ có mutual-follow edges.
- **Uniform p** — **KHÔNG** report như primary; weighted cascade là bắt buộc.

> ✦ **[SHOULD DO] Sensitivity S1 — Symmetric p**: `p(u,v) = 1/√(deg(u)×deg(v))` — parameter-free, views-independent, không vi phạm independence nào. Lý do ưu tiên: structurally analogous to GCN's `D^{-1/2}AD^{-1/2}` normalization → tạo testable "GCN–A2 alignment hypothesis". Output: `ic_scores_sensitivity_a2.parquet`. Chạy sau khi primary IC và C2 xong. Framing: "robustness to diffusion rule choice." Xem Section 4.1b của Implementation Plan.
>
> ✦ **[IF TIME] Sensitivity S2 — Source Budget p**: `p(u,v) = 1/deg(u)` — mọi node có one-hop spread = 1.0, IC score thuần 2+ hop. Chỉ làm nếu `Spearman(IC-A0, degree) > 0.85` và cần tăng "IC ≠ degree" evidence. Output: `ic_scores_sensitivity_a1.parquet`.
>
> ~~Uniform-p~~ (đã cắt — tight timeline, không add giá trị bằng A2).

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
| `n_sample` (default)        | **5.000** labeled nodes                                 | Person 1                           |
| **Proxies scope**           | **FULL active graph** (~168k nodes)                     | Person 2                           |
| `runtime_sec` trong metrics | **full-graph inference only**                           | Person 3                           |

| Quyết định            | Giá trị                                                                                          | Khi nào quyết    |
| --------------------- | ------------------------------------------------------------------------------------------------ | ---------------- |
| `n_sample × N_runs`   | **5,000 × 200** _(default; xem `docs/day1_decisions.md` Phần 3 để confirm — M2 đã qua ngày 7/4)_ | **M2** (đã pass) |
| GNN narrative branch  | Xem `docs/day1_decisions.md` Phần 2 (`narrative_branch`)                                         | **M2** (đã pass) |
| Uniform-p sensitivity | Cắt (không làm — tight timeline per scope reduction table)                                       | M2 (đã quyết)    |

> **📍 M2 đã pass (7/4):** `n_sample` và `N_runs` phải đã được ghi vào `docs/day1_decisions.md` Phần 3. Implementation Plan default = **5,000 × 200** (`mc_runs_primary: 200`, `sample_size_primary: 5000` trong `experiment.yaml`). Nếu Day-1 benchmark cho `projected_hours > 4h`, xem bảng quyết định tại Milestone M2 để biết giá trị đã chọn. **Không tự điều chỉnh n_sample/N_runs** mà không đọc `day1_decisions.md` trước.

**⚠ Rule quan trọng:** Person 3 **KHÔNG được tự tạo split**. Phải load từ `data/processed/split_masks.parquet` do Person 1 tạo, dùng `load_split_mask()` trong `eval_ranking_harness.py`.

---

## 1b) Bảng hằng số chung (Shared Constants)

> ⚠ **Đây là single source of truth cho toàn team.** Mọi thay đổi phải cập nhật bảng này trước, sau đó propagate sang code/config/artifacts. Không được hard-code khác đi ở bất kỳ chỗ nào.

| Hằng số                              | Giá trị chuẩn                                 | Ý nghĩa                                                                                              |
| ------------------------------------ | --------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `cv_gate`                            | **0.3**                                       | Regression-ready gate: `cv_score > 0.3` → pipeline tiếp tục bình thường                              |
| `jaccard_gate`                       | **0.85**                                      | Binary-ready gate: `jaccard_stability >= 0.85` → binary non-provisional                              |
| `top_k_pct`                          | **0.10**                                      | Top-10% threshold cho classification labels + NDCG@10% + Precision@10%                               |
| `n_sample`                           | **5,000** (locked sau M2)                     | Số labeled nodes (stratified sample)                                                                 |
| `N_runs`                             | **200** (default; locked sau Day-1 benchmark) | MC IC runs per node                                                                                  |
| `n_mc_stability`                     | **3**                                         | Số MC seeds cho label stability check                                                                |
| `gnn_seeds`                          | **5**                                         | Số seeds cho GNN training (mean±std)                                                                 |
| `louvain_seed`                       | **42**                                        | Seed cho Louvain community detection                                                                 |
| `split_seed`                         | **42**                                        | Seed cho train/test split (80/20 stratified)                                                         |
| `test_frac`                          | **0.20**                                      | Test fraction cho split_masks                                                                        |
| **Milestone dates**                  |                                               |                                                                                                      |
| M0-M2                                | **6/4 – 10/4**                                | Setup, benchmark, GNN narrative locked                                                               |
| M3                                   | **13/4**                                      | IC labels done; split_masks sẵn sàng                                                                 |
| M4                                   | **18/4**                                      | All intermediate results done                                                                        |
| M5                                   | **22–27/4**                                   | Integration + paper hand-off                                                                         |
| Experiments locked                   | **21/4**                                      | Data generation + model training xong                                                                |
| Submit deadline                      | **30/4**                                      | Hard deadline                                                                                        |
| C1 — degree variance test            | **16/4**                                      | Person 1/3                                                                                           |
| C2 — arch comparison done            | **18/4**                                      | Person 3                                                                                             |
| C3 — ranking loss done               | **20/4**                                      | Person 3 (sau C2)                                                                                    |
| C4 — bootstrap CI done               | **20/4**                                      | Person 3 (sau C2)                                                                                    |
| All v3.1 experiments locked          | **21/4**                                      | Toàn team                                                                                            |
| **CSV scope mapping**                |                                               |                                                                                                      |
| `baseline_ranking_metrics.csv`       | Groups **1–4**                                | Baselines (raw/centrality/proxies/embeddings)                                                        |
| `surrogate_ranking_metrics.csv`      | Group **5** (GNN variants)                    | GNN models only                                                                                      |
| `runtime_breakdown.csv`              | Groups **1–5 + proxies**                      | ALL models cho speedup calculation                                                                   |
| **Metric list (correlation matrix)** | **8 metrics**                                 | `ic_score_mean, views, degree, pagerank, kshell, betweenness_approx, one_hop_spread, two_hop_spread` |

---

## 2) Artifact contracts (đóng băng giao diện giữa 3 người)

> Các schema dưới đây bám theo `docs/MAPR2026_v3_migration_checklist.md`. Nếu cần đổi tên/format, phải đổi đồng bộ và ghi vào `docs/experiment_registry.md`.

| Artifact (path)                                                  | Owner                                             | Consumers  | Contract tối thiểu                                                                                                                                                                                                                                                                                                                                        |
| ---------------------------------------------------------------- | ------------------------------------------------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `outputs/stage0_data_quality/lcc_report.json`                    | Person 1                                          | Person 1,2 | fields: `n_nodes_total, n_nodes_lcc, pct_lcc, n_components`. **Dùng cho pilot threshold:** `median_reach < 5% of n_nodes_lcc`. Phải có trước khi chạy IC pilot.                                                                                                                                                                                           |
| `data/processed/graph_csr.npz`                                   | Person 1                                          | All        | `indptr`, `indices`, `degrees`, mapping `node_id↔row_index` deterministic                                                                                                                                                                                                                                                                                 |
| `outputs/day1_benchmark/ic_runtime_benchmark.json`               | Person 1                                          | All        | per-sim ms + projected runtime + decision table                                                                                                                                                                                                                                                                                                           |
| `outputs/day1_benchmark/one_hop_correlation.json`                | Person 1                                          | All        | Day-1 gate metrics: `spearman_rho`, `jaccard_at_10pct`, `ndcg_at_10pct`, `decision_branch` (không dùng Spearman đơn lẻ)                                                                                                                                                                                                                                   |
| `data/processed/ic_scores_primary.parquet`                       | Person 1                                          | Person 2,3 | columns: `node_id, ic_score_mean, ic_score_std, n_runs, p_model` (**sample-only: n_sample nodes**)                                                                                                                                                                                                                                                        |
| `data/processed/regression_targets.parquet`                      | Person 1                                          | Person 3   | columns: `node_id, y` với `y=log1p(ic_score_mean)`                                                                                                                                                                                                                                                                                                        |
| `data/processed/classification_labels.parquet`                   | Person 1                                          | Person 3   | columns: `node_id, y_top10` (top 10%)                                                                                                                                                                                                                                                                                                                     |
| `data/processed/split_masks.parquet` **[M0-locked]**             | Person 1                                          | Person 3   | columns: `node_id (str), split ('train'\|'test')`. 80/20, degree-stratified q=5, seed=42. Scope = labeled nodes only. **Không ai tự tạo split khác.**                                                                                                                                                                                                     |
| `data/processed/community_features.parquet`                      | Person 2                                          | Person 2,3 | columns: `node_id, community_id, cross_community_edge_fraction`. Scope: ALL active nodes. **File riêng** — KHÔNG ghi đè `node_attributes.parquet` (Person 1 owns).                                                                                                                                                                                        |
| `data/processed/diffusion_proxies.parquet`                       | Person 2                                          | Person 3   | columns: `node_id, one_hop_spread, two_hop_spread`. **Scope: FULL active graph** (không phải chỉ labeled subset)                                                                                                                                                                                                                                          |
| `outputs/mapr2026_v3_results/baseline_ranking_metrics.csv`       | Person 3                                          | All        | columns: `model_name, spearman_rho, ndcg_at_10pct, precision_at_10pct, runtime_sec`                                                                                                                                                                                                                                                                       |
| `outputs/mapr2026_v3_results/surrogate_ranking_metrics.csv`      | Person 3                                          | All        | **[MUST — v3.1 unconditional]** — columns: `model_name, spearman_rho_mean, spearman_rho_std, ndcg_mean, ndcg_std, precision_mean, precision_std, runtime_sec` (mean±std trên 5 seeds). model_names: `gnn_raw_attr`, `gnn_graph_only`, `gnn_centrality`, `gcn_raw_attr` (C2), `gin_raw_attr` (C2), `gat_raw_attr` (C2), `best_arch_raw_attr_rankloss` (C3) |
| `outputs/mapr2026_v3_results/runtime_breakdown.csv`              | Person 2 (proxies) + Person 3 (models)            | All        | columns: `model_name, inference_sec_full_graph, train_sec(optional/null)` — ghi runtime toàn active graph cho từng model (Group 1–5 + diffusion_proxies); dùng cho Speedup calculation                                                                                                                                                                    |
| `outputs/day1_benchmark/stability_explanation.json`              | Person 1                                          | All        | **[MUST — triggered khi Jaccard < 0.85 (current observed: 0.307 — recheck sau mỗi re-run); ~30 phút extract từ phase1_community_overlap.json + phase2_threshold_analysis.json, không cần chạy lại IC]** fields: `pct_communities_spanning_boundary`, `mean_gap_to_noise`, `n_thresholds_tested`, `interpretation`                                         |
| `outputs/mapr2026_v3_results/metric_correlation_matrix.json`     | Person 2                                          | All        | **[MUST — ~2–3h; tất cả data có sẵn ngay bây giờ]** `rho_matrix` 8×8 (8 metrics: ic_score_mean, views, degree, pagerank, kshell, **betweenness_approx**, one_hop_spread, two_hop_spread) + `p_matrix_corrected` (bắt buộc). `rho_by_degree_quintile`: **[✦ IF TIME]** — optional, không ảnh hưởng global matrix                                           |
| `outputs/mapr2026_v3_results/degree_controlled_ic_variance.json` | Person 1 (chạy sau IC labels xong)                | All        | **[MUST — C1; ~30 phút từ existing IC scores]** fields: `degree_band, n_nodes_in_band, ic_mean_in_band, ic_std_in_band, cv_within_band, interpretation`. Deadline: **16/4**                                                                                                                                                                               |
| `outputs/mapr2026_v3_results/gnn_vs_degree_bootstrap_ci.json`    | Person 3 (sau khi có best-arch predictions từ C2) | All        | **[MUST — C4; ~10 phút resample]** fields: `n_bootstrap(1000), comparator_a(gnn_best_architecture), comparator_b(degree), delta_mean, ci_95_lower, ci_95_upper, interpretation`. Deadline: **20/4**                                                                                                                                                       |

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

#### `outputs/mapr2026_v3_results/surrogate_ranking_metrics.csv` (**[MUST — v3.1 unconditional]** — `gnn_branch_viable` gate không còn áp dụng per v3.1 professor's framing)

Schema bắt buộc (mean±std trên 5 training seeds `[42, 123, 456, 789, 1024]`):

- `model_name`: string — tên chuẩn: `gnn_raw_attr`, `gnn_graph_only`, `gnn_centrality`, `gnn_full`, `gcn_raw_attr` (C2), `gin_raw_attr` (C2), `gat_raw_attr` (C2), `appnp_raw_attr` (C2 — **H3 expected best**), `best_arch_raw_attr_rankloss` (C3 — UPDATE tên thực tế sau C2, e.g., `appnp_raw_attr_rankloss`)
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

### (B) Person 2 — community + diffusion proxies (dry-run)

```powershell
cd src/mapr2026_v3

# Proxies placeholder (dry-run header-only, 0 rows; schema only; KHÔNG dùng cho evaluation/runtime)
python diffusion_proxies.py --dry-run --seed 42
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
- **Trần Hùng Vĩ** → **Person 2** (community features + diffusion proxies + correlation matrix)
- **Trần Quốc Hải** → **Person 3** (evaluation harness + surrogate learning/ML)

> **⚠ Team-size mapping note:** `MAPR2026_Implementation_Plan_v3.md` (Section 21) sử dụng **6-person** assignment (Person 1–6) cho critical path. File này dùng **3 người** — mapping như sau:
>
> | Implementation Plan (6 người)    | Team Plan (3 người)        | Scope                                             |
> | -------------------------------- | -------------------------- | ------------------------------------------------- |
> | Person 1 (Data + IC)             | **Person 1**               | IC benchmark, labels, degree variance C1          |
> | Person 2 (Sampling + stability)  | **Person 1** (absorbed)    | Pilot diagnostics, KS check                       |
> | Person 3 (Baselines + community) | **Person 2**               | Community detection, proxies, baselines Group 1–3 |
> | Person 4 (Baselines Group 4)     | **Person 3**               | Node2Vec, MLP, evaluation harness                 |
> | Person 5 (GNN)                   | **Person 3**               | GNN-raw-attr, arch comparison C2, ranking loss C3 |
> | Person 6 (Paper)                 | _(người viết paper riêng)_ | Không trong scope file này                        |
>
> **Critical path mapping:** Implementation Plan nói "Person 5 → C2, Person 3 → C4" — trong Team Plan, **Person 3 làm cả C2, C3, C4**. Mọi tham chiếu "Person 5" trong Implementation Plan = "Person 3" trong file này.

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
IC scores + split_masks ───────────────────────────────► baselines thật
                                                           surrogates thật
```

**Ghi chú dependency chính:**

- `graph_csr.npz` (Person 1) → unblock cả Person 2 và Person 3
- `split_masks.parquet` (Person 1) → Person 3 mới chạy metrics thật
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
     - Script: `src/data/lcc_audit.py`
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

    # Stable tie-break để tránh drift top-k khi có nhiều điểm bằng nhau
    order_ic = np.argsort(-np.asarray(ic_pilot_scores, dtype=float), kind='stable')
    order_oh = np.argsort(-np.asarray(one_hop_scores, dtype=float), kind='stable')
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

   > ✦ **[SHOULD DO — sau khi primary IC + C2 xong] Sensitivity S1: Symmetric IC (A2)**
   >
   > **Mục đích:** Robustness check về diffusion rule choice; tạo architectural inductive bias test.
   > **Code:** Thay 1 dòng trong `run_ic_csr`: `p = 1.0 / np.sqrt(float(degrees[node]) * float(degrees[nb]))` thay vì `p = 1.0 / degrees[nb]`. Giữ nguyên toàn bộ pipeline CSR/loky.
   >
   > ```python
   > # A2 symmetric — chỉ khác 1 dòng:
   > deg_node = degrees[node]
   > p = 1.0 / np.sqrt(float(deg_node) * float(degrees[nb])) \
   >     if deg_node > 0 and degrees[nb] > 0 else 0.0
   > ```
   >
   > **Output:** `outputs/mapr2026_v3_results/ic_scores_sensitivity_a2.parquet` (same schema as primary).
   > **Sau đó compute:**
   >
   > - `Spearman(IC-A2, IC-A0)` — nếu ρ > 0.95: variants gần như identical → sensitivity confirms robustness
   > - `Spearman(IC-A2, degree)` — so sánh với `Spearman(IC-A0, degree)` từ metric correlation matrix
   > - **Ghi vào `ic_sensitivity_comparison.json`**: `{"a0_vs_degree": X, "a2_vs_degree": Y, "a0_vs_a2": Z}`
   >
   > **Handoff sang Person 3:** Nếu A2 labels sẵn sàng trước deadline C2, Person 3 có thể chạy thêm C2-A2 (4 archs × 5 seeds trên A2 labels) để test GCN–A2 hypothesis. Ưu tiên: C2-A0 (primary) xong trước, C2-A2 chỉ nếu còn time budget.
   >
   > **Framing bắt buộc:** "Sensitivity to diffusion rule choice" — không phải "chọn rule giúp GNN thắng degree".

   > 🔵 **[SUPPLEMENTARY TRACK — SKIP nếu không activate I-A]**
   > Toàn bộ block I-A bên dưới chỉ relevant nếu pilot pass (3 checks bên dưới).
   > Nếu A0 only: bỏ qua toàn bộ I-A block, C2-I-A, và C4-I-A → tiếp tục sang Person 2.

   > ✦ **[SHOULD DO — nếu bật I-A] I-A: Attribute-Informed IC (Row-Normalized Views)**
   >
   > **Điều kiện kích hoạt (PHẢI thỏa TẤT CẢ):**
   >
   > 1. Pilot pass: CV > 0.3, ρ(IC-I-A, degree) < 0.75 (kỳ vọng thấp hơn A0 vì one-hop expectation không phụ thuộc degree), ρ(IC-I-A, nbr_views_mean_proxy) < 0.85
   > 2. C2-A0 (architecture comparison trên primary labels) đã chạy xong để không block critical path
   >
   > **Lý do cơ học tại sao I-A giúp GNN thắng degree:**
   >
   > - A0 (primary): p = 1/deg(v) → IC ≈ one-hop analytical proxy → Spearman(IC, degree) ≈ 0.826 → degree đã capture hầu hết variance → GNN khó improve
   > - I-A: p(u,v) = log1p(views(v)) / Σ\_{x∈N(u)} log1p(views(x)) — **row-normalized**
   >   - Mọi node u đều có E[one-hop(u)] = 1.0 (bất kể degree của u)
   >   - One-hop expectation độc lập với degree(u); degree-only baseline bị bất lợi vì không quan sát được **phân phối views của neighborhood** (N(u))
   >   - GNN Layer 1: AGG({log1p(views(v)) : v∈N(u)}) ≈ tính denominator của p(u,v) → structural alignment
   >   - GNN Layer 2: 2-hop attribute propagation ≈ strong inductive-bias alignment với IC-I-A (neighbor-attribute propagation)
   >   - **Hypothesis (pre-registered):** Spearman(GNN, IC-I-A) sẽ tăng đáng kể so với degree-only baselines; magnitude là kết quả thực nghiệm
   >
   > **Pre-registration (bắt buộc — anti-p-hacking):** Phải ghi hypothesis vào `docs/experiment_registry.md` TRƯỚC KHI chạy pilot: "H3: Under I-A, GNN will significantly outperform degree because I-A makes one-hop expectation degree-independent and requires neighborhood attribute aggregation." Decision tree với thresholds cố định trước khi thấy kết quả.
   >
   > **Bước 1 — Pilot (200 nodes × 50 runs):**
   >
   > ```python
   > # Precompute I-A weights (một lần)
   > import numpy as np, scipy.sparse as sp
   > from scipy.stats import spearmanr
   >
   > data  = np.load("data/processed/graph_csr.npz", allow_pickle=True)
   > indptr, indices, degrees, node_ids = data["indptr"], data["indices"], data["degrees"], data["node_ids"].astype(str)
   > n = len(node_ids)
   >
   > import pandas as pd
   > node_attrs = pd.read_parquet("data/processed/node_attributes.parquet")
   >
   > # Map views → CSR row order (IMPORTANT: align by node_id, not by row position in parquet)
   > views_by_id = node_attrs.set_index("node_id")["views"]
   > views_raw = views_by_id.reindex(node_ids).fillna(0).to_numpy(dtype=np.float64)
   > views_log = np.log1p(views_raw)
   >
   > # Precompute denominator: Σ log1p(views(v)) for v ∈ N(u)
   > neighbor_views_sum = np.zeros(n, dtype=np.float64)
   > for u in range(n):
   >     nbrs = indices[indptr[u]:indptr[u+1]]
   >     if len(nbrs) > 0:
   >         neighbor_views_sum[u] = views_log[nbrs].sum()
   >
   > # Run I-A IC pilot (200 nodes × 50 runs):
   > from src.mapr2026_v3.ic_labels_primary import run_ic_csr_ia
   > pilot_ids = ...  # same 200 pilot_node_ids từ primary pilot (stratified)
   > ic_ia = np.array([
   >     run_ic_csr_ia(u, indptr, indices, views_log, neighbor_views_sum,
   >                   n_runs=50, worker_seed=42+u).mean()
   >     for u in pilot_ids
   > ])
   > ```
   >
   > **Bước 2 — Pilot Decision Protocol (3 checks — phải pass TẤT CẢ):**
   >
   > ```python
   > # CHECK 1: Non-degenerate variance (degree-blind → spread phải có variance)
   > cv_ia = ic_ia.std() / (ic_ia.mean() + 1e-9)
   > print(f"CV = {cv_ia:.3f}  (need > 0.3)")
   >
   > # CHECK 2: Degree correlation (phải thấp — đây là lý do chính chạy I-A)
   > deg_pilot = degrees[pilot_ids]
   > rho_deg_ia, _ = spearmanr(ic_ia, deg_pilot)
   > print(f"ρ(IC-I-A, degree) = {rho_deg_ia:.3f}  (need < 0.75)")
   >
   > # CHECK 3: Neighbor-views-mean proxy (nếu quá cao → GNN chỉ cần 1 hop, không thú vị)
   > neighbor_views_mean = np.array([
   >     views_log[indices[indptr[u]:indptr[u+1]]].mean() if indptr[u+1] > indptr[u] else 0.0
   >     for u in pilot_ids
   > ])
   > rho_nbr_ia, _ = spearmanr(ic_ia, neighbor_views_mean)
   > print(f"ρ(IC-I-A, nbr_views_mean) = {rho_nbr_ia:.3f}  (need < 0.85)")
   >
   > # Decision tree:
   > if cv_ia > 0.3 and rho_deg_ia < 0.75 and rho_nbr_ia < 0.85:
   >     print("✅ ALL PASS → Run full I-A IC sim (5k nodes × 200 runs)")
   >     print("   → Run C2-I-A: 4 archs × 5 seeds on I-A labels")
   >     print("   → Run C4-I-A: Bootstrap CI GNN_best vs degree on I-A labels")
   > elif cv_ia <= 0.3:
   >     print("❌ FAIL CHECK 1: IC-I-A degenerate (low variance) → Stay A0 primary, I-A abandoned")
   > elif rho_deg_ia >= 0.75:
   >     print("⚠ FAIL CHECK 2: degree still correlates (rho_deg ≥ 0.75)")
   >     print("  → Fallback: try II-B (views_density) with same checks; if also fail → stay A0")
   > elif rho_nbr_ia >= 0.85:
   >     print("⚠ FAIL CHECK 3: 1-hop proxy too strong → GNN not needed for I-A")
   >     print("  → Report as limitation; stay A0 primary; abandon I-A")
   >
   > # Save pilot result:
   > import json
   > with open("outputs/mapr2026_v3_results/ic_ia_pilot_decision.json", "w") as f:
   >     json.dump({
   >         "cv_ia": float(cv_ia), "rho_deg_ia": float(rho_deg_ia), "rho_nbr_ia": float(rho_nbr_ia),
   >         "pass_cv": bool(cv_ia > 0.3), "pass_deg": bool(rho_deg_ia < 0.75),
   >         "pass_proxy": bool(rho_nbr_ia < 0.85),
   >         "decision": "run_full_ia" if (cv_ia > 0.3 and rho_deg_ia < 0.75 and rho_nbr_ia < 0.85)
   >                     else "fallback_a0"
   >     }, f, indent=2)
   > ```
   >
   > **Nếu ALL PASS → Full I-A sim:**
   >
   > - Output: `outputs/mapr2026_v3_results/ic_scores_ia.parquet` (same schema as primary)
   > - Compute: `Spearman(IC-I-A, degree)` và `Spearman(IC-I-A, IC-A0)` → `ic_ia_vs_primary.json`
   > - **Handoff sang Person 3:** Person 3 chạy **C2-I-A** (**5 archs**: APPNP + **GATv2** + GIN + GCN + SAGE × 5 seeds trên I-A labels) → `surrogate_ranking_metrics_ia.csv`; sau đó C4-I-A (bootstrap CI GNN_best_ia vs degree on I-A labels) → `gnn_vs_degree_bootstrap_ci_ia.json`
   > - **Lưu ý:** C2-I-A dùng **GATv2** thay cho GAT v1 — xem H4 hypothesis bên dưới.
   >
   > **Nếu ANY FAIL → Fallback:**
   >
   > - Ghi rõ vào `docs/experiment_registry.md`: lý do abandon I-A + checkpoint values
   > - Nếu rho_deg fail → thử II-B fallback: `p(u,v) = clip(views_norm[v]/deg(v), max=0.5)` với cùng bộ checks
   > - Nếu cả I-A lẫn II-B fail → giữ A0 primary + S1 sensitivity là đủ
   >
   > **Paper narrative — Nếu I-A pass:**
   >
   > - Section 3.1: "We additionally test an attribute-informed variant (I-A) where propagation probability is proportional to the target node's log-scaled view count, row-normalized per source. Under I-A, degree provides no information about IC rank (Spearman = X), enabling us to test whether GNN message passing captures attribute-driven diffusion."
   > - Table caption: "I-A labels are used for supplementary GNN advantage analysis; A0 (weighted cascade) remains the primary IC operationalization."
   >
   > **Framing bắt buộc:** "GNN advantage under attribute-informed diffusion" — không phải "I-A labels tốt hơn A0 labels". A0 LUÔN là primary framing; I-A là supplementary analysis chứng minh GNN advantage mechanism.

   > **H4 — GATv2 cho C2-I-A (quan trọng — đọc trước khi implement):**
   >
   > | | GAT v1 | GATv2 |
   > |---|---|---|
   > | Attention | Static: `e(i,j) = a_src·(W·h_i) + a_tgt·(W·h_j)` | Dynamic: `e(i,j) = a^T LeakyReLU(W·[h_i \|\| h_j])` |
   > | Tính chất | Ranking của j KHÔNG đổi theo i | Ranking của j PHỤ THUỘC ngữ cảnh i |
   > | I-A formula | `p(u,v) = views(v)/Σviews(N(u))` — denominator phụ thuộc u | ← Cần dynamic attention để model |
   > | Verdict A0 | ✅ OK (static `1/deg(v)` chỉ cần target node) | Overkill cho A0 |
   > | Verdict I-A | ❌ Cannot model row-normalization | **✅ H4: GATv2 là correct arch** |
   >
   > **GATv2 implementation (thêm vào `run_surrogates.py`):**
   >
   > ```python
   > from torch_geometric.nn import GATv2Conv
   > import torch.nn.functional as F
   >
   > class GATv2Surrogate(nn.Module):
   >     """
   >     GATv2 — Dynamic attention (Brody et al., ICLR 2022).
   >     Dùng cho C2-I-A (H4): I-A p(u,v) row-normalized per source → cần dynamic attention.
   >     KHÔNG dùng trong C2-A0 (GAT v1 phù hợp hơn cho static A0).
   >     """
   >     def __init__(self, in_dim=3, hidden_dim=128, heads=4, dropout=0.3):
   >         super().__init__()
   >         self.conv1 = GATv2Conv(in_dim, hidden_dim // heads, heads=heads,
   >                                dropout=dropout, concat=True)
   >         self.conv2 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads,
   >                                dropout=dropout, concat=True)
   >         self.head  = nn.Linear(hidden_dim, 1)
   >         self.drop  = nn.Dropout(dropout)
   >
   >     def forward(self, x, edge_index):
   >         x = F.elu(self.conv1(x, edge_index))
   >         x = self.drop(x)
   >         x = F.elu(self.conv2(x, edge_index))
   >         return self.head(x).squeeze(-1)
   >
   > # Thêm vào get_model() factory:
   > # elif arch == 'gatv2':
   > #     return GATv2Surrogate(in_dim=in_dim, hidden_dim=hidden_dim,
   > #                           heads=4, dropout=dropout)
   >
   > # C2-I-A architectures (GATv2 thay GAT v1):
   > C2_IA_ARCHITECTURES = ['appnp', 'gatv2', 'gin', 'gcn', 'sage']
   > # model_name trong surrogate_ranking_metrics_ia.csv:
   > # appnp_raw_attr_ia, gatv2_raw_attr_ia, gin_raw_attr_ia, gcn_raw_attr_ia, gnn_raw_attr_ia
   > ```

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
   > 2. Dùng normalized reach (reach/degree) thay raw reach — cần justify rõ trong paper
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
   bins = sorted(node_attrs["degree_q"].dropna().unique().tolist())
   n_per_bin = 200 // len(bins)

   parts = []
   for b in bins:
       g = node_attrs[node_attrs["degree_q"] == b]
       parts.append(g.sample(n=n_per_bin, random_state=42))

   pilot_nodes = pd.concat(parts, ignore_index=True)

   # Fill remainder (if any) from the remaining pool, deterministically.
   remainder = 200 - len(pilot_nodes)
   if remainder > 0:
       remaining = node_attrs[~node_attrs["node_id"].isin(pilot_nodes["node_id"])].copy()
       pilot_nodes = pd.concat(
           [pilot_nodes, remaining.sample(n=remainder, random_state=42)],
           ignore_index=True,
       )

   pilot_node_ids = pilot_nodes["node_id"].tolist()
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

> **NetworkX usage policy — clarification (tránh nhầm "TUYỆT ĐỐI không dùng" = cấm hoàn toàn):**
>
> | Loại sử dụng                                                     | Cho phép?     | Lý do                                                        |
> | ---------------------------------------------------------------- | ------------- | ------------------------------------------------------------ |
> | IC simulation loops (BFS/DFS per node per run)                   | ❌ **BANNED** | O(N × runs) bằng NetworkX → timeout; phải dùng CSR numpy     |
> | Graph load/convert **một lần**: `nx.read_edgelist()` → CSR/PyG   | ✅ OK         | Chỉ gọi 1 lần khi startup                                    |
> | Community detection: `python-louvain` (dùng NetworkX internally) | ✅ OK         | Acceptable; không phải IC loop                               |
> | Betweenness centrality trên subgraph nhỏ (≤500 nodes)            | ✅ OK         | Không phải full-graph IC; NetworKit preferred cho full graph |
> | Debug/utility: degree dict, neighbor list                        | ✅ OK         | Không production-path                                        |
>
> **Rule of thumb:** NetworkX call trong vòng lặp ≥ 5,000 lần → **bắt buộc đổi sang CSR numpy**. Gọi 1–10 lần → okay.

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
- [ ] **[v3.1 MUST — C1, deadline 16/4]** `degree_controlled_ic_variance.json` tồn tại với đủ 6 fields. cv_within_band có giá trị, không null.

> ✦ **[IF TIME] Soft DoD** — thêm vào khi xong MUST trước deadline:
>
> - [ ] Bootstrap 95% CI: `ic_ci_lower`, `ic_ci_upper` trong `ic_scores_primary.parquet`
> - [ ] Robust diagnostics trong pilot JSON: `p_reach_gt_1`, `p_reach_ge_5`, `per_quintile_cv`, `run_count_stability_tau_by_quintile`
> - [ ] Low-degree limitation note trong `docs/day1_decisions.md` (nếu Q1.cv < 0.15 & Q2.cv < 0.20)

**[NEW v3.1 — C1] Degree-Controlled IC Variance Test (~30 phút, deadline 16/4):**

> Chứng minh IC captures higher-order diffusion effects beyond local degree. Trả lời reviewer: _"Why not just use degree?"_

```python
import numpy as np, json, pandas as pd

def run_c1_degree_variance_test(ic_scores_path, centrality_path,
                                degree_band=(75, 85),
                                output_path="outputs/mapr2026_v3_results/degree_controlled_ic_variance.json"):
    ic_df  = pd.read_parquet(ic_scores_path)    # columns: node_id, ic_score_mean
    cen_df = pd.read_parquet(centrality_path)   # columns: node_id, degree
    df = ic_df.merge(cen_df[["node_id","degree"]], on="node_id")

    band_df = df[(df["degree"] >= degree_band[0]) & (df["degree"] <= degree_band[1])]
    ic_in_band = band_df["ic_score_mean"].values
    cv = ic_in_band.std() / ic_in_band.mean() if ic_in_band.mean() > 0 else 0.0

    interpretation = ("IC adds info beyond degree" if cv > 0.3
                      else "IC ≈ degree at this scale")
    result = {
        "degree_band": f"{degree_band[0]}-{degree_band[1]}",
        "n_nodes_in_band": int(len(band_df)),
        "ic_mean_in_band": float(ic_in_band.mean()),
        "ic_std_in_band":  float(ic_in_band.std()),
        "cv_within_band":  float(cv),
        "interpretation":  interpretation
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"C1 done: cv={cv:.3f} → {interpretation}")
    return result

# Run:
# run_c1_degree_variance_test("data/processed/ic_scores_primary.parquet",
#                              "data/processed/centrality_table.parquet")
```

**Gợi ý entrypoint:**

- `src/mapr2026_v3/export_csr.py`
- `src/mapr2026_v3/day1_benchmark.py`
- `src/mapr2026_v3/ic_labels_primary.py`

**Runbook tối thiểu:**

- Unblock team ngay: chạy 3 lệnh ở Mục 2.1 (A).
- Real mode: implement IC (primary: worker_seed=42+node; stability: mc_seed\*10000+node), pilot diagnostics **6 MUST core metrics** + KS check, stability 3 MC seeds (n_runs=150 each).

> ✦ **[IF TIME]** Robust diagnostics (`p_reach_gt_1`, `p_reach_ge_5`, `per_quintile_cv`, `tau_by_quintile`) — chỉ thêm sau khi xong toàn bộ MUST trên.

---

### Person 2 — Track B: Structure + diffusion proxies (community + baselines + correlation matrix)

**Mục tiêu:** Community detection (support structural interpretation) + Diffusion proxies (Group 3 baseline) + Metric correlation matrix (multicollinearity evidence).

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

   - **Lý do bắt buộc:** dùng để mô tả cấu trúc network (bridge-ness/cross-community mixing) và làm input cho các bảng tương quan/proxy; không có thì thiếu bằng chứng định lượng.

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

3. **[MUST — ~2–3h; tất cả parquet artifacts đã tồn tại; Person 2 có thể chạy ngay hôm nay]** Metric correlation matrix (full pairwise Spearman) _(MUST — phục vụ Section 4.3 multicollinearity claim)_

- **Tại sao MUST:** Reviewer sẽ hỏi "how do IC scores relate to simpler metrics?" — không có Table này thì phải trả lời verbal trong rebuttal. Tất cả data đã có sẵn: `ic_scores_primary.parquet`, `diffusion_proxies.parquet`, `node_attributes.parquet`, `centrality_table.parquet`. Zero new computation needed.
- **Mục tiêu:** Trả lời RQ2b — khi nào degree/pagerank/views fail làm proxy cho IC? Provide số liệu định lượng cho Table trong Section 4.3 paper.
- **Input:** join `ic_scores_primary.parquet` + `node_attributes.parquet` + `diffusion_proxies.parquet` + `centrality_table.parquet`; tất cả filter về labeled nodes (5,000 nodes)
- **8 metrics:** `ic_score_mean`, `views`, `degree`, `pagerank`, `kshell`, `betweenness_approx`, `one_hop_spread`, `two_hop_spread`

  > 📌 **Toán học của `one_hop_spread` (đọc trước khi interpret kết quả):**
  >
  > - Dưới A0 (`p=1/deg(v)`): Global average one-hop spread = 1.0 ∀ node **là mathematical invariant** (không phải approximation). Proof: `Global_avg = (1/N)×Σ_u[Σ_{v∈N(u)} 1/deg(v)] = (1/N)×Σ_v[deg(v)×1/deg(v)] = 1.0`.
  > - Rank ordering vẫn biến thiên vì `E[one_hop(u)] = Σ_{v∈N(u)} 1/deg(v)` phụ thuộc degree của **neighbors** của u, không phải degree(u). Node có nhiều niche neighbors → one_hop cao hơn hub có cùng degree.
  > - Expected `Spearman(one_hop, IC-A0) ∈ [0.7, 0.95]` — **không nhất thiết = 1.0** vì IC có 2+ hop dynamics. Nếu ρ > 0.9: cascade rất local → proxies competitive → GNN story dùng +0.099 message passing narrative.
  > - Dưới A1 (`p=1/deg(u)`): **Mọi node đều có one_hop = 1.0** (identity, không phải invariant) → `one_hop` hoàn toàn uninformative → IC-A1 chỉ phản ánh 2+ hop → `Spearman(IC-A1, degree)` expected thấp hơn A0.

- **⚠ Column rename (betweenness):** `centrality_table.parquet` lưu cột tên `betweenness`; phải rename thành `betweenness_approx` trước khi chạy matrix để consistent với metric naming convention:
  ```python
  centrality_df = pd.read_parquet(PATHS.centrality_table)
  if "betweenness" in centrality_df.columns and "betweenness_approx" not in centrality_df.columns:
      centrality_df = centrality_df.rename(columns={"betweenness": "betweenness_approx"})
  ```
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
- **Timing:** Chạy sau khi có `diffusion_proxies.parquet` và `ic_scores_primary.parquet`.

**DoD cho Track B — MUST (sign-off bắt buộc):**

**DoD Track B — MUST (không cắt):**

- [ ] `data/processed/community_features.parquet` (file riêng, KHÔNG ghi vào `node_attributes.parquet`), phủ 100% active nodes; có `node_id`, `community_id`, `cross_community_edge_fraction`.
- [ ] Proxies (one-hop + two-hop) trên FULL active graph, missing = 0; `runtime_breakdown.csv` có `inference_sec_full_graph`.
- [ ] **[MUST — ~2–3h; chạy được ngay hôm nay]** `metric_correlation_matrix.json` tồn tại với 8×8 `rho_matrix` và `p_matrix_corrected` (global matrix — 8 metrics: ic_score_mean, views, degree, pagerank, kshell, betweenness_approx, one_hop_spread, two_hop_spread — bắt buộc). `rho_by_degree_quintile` là **[✦ IF TIME]** — sign-off không phụ thuộc vào phần này.

> ✦ **[IF TIME] Soft DoD:**
>
> - [ ] Louvain resolution sensitivity: `louvain_resolution_sensitivity.json` cho {0.5, 1.0, 2.0}

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
       # ⚠ S2 fix: kind='stable' prevents non-deterministic tie-breaking
       true_top_k = set(np.argsort(y_true, kind='stable')[-k:])    # top-k indices theo y_true
       pred_top_k = set(np.argsort(y_pred, kind='stable')[-k:])    # top-k indices theo y_pred
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
   | `gnn_raw_attr` | 5 | GraphSAGE raw-attr / `sage_raw_attr` (backward compat — → surrogate CSV) |
   | `gnn_graph_only` | 5 | GraphSAGE graph-only (→ surrogate CSV) |
   | `gnn_centrality` | 5 | GraphSAGE centrality (→ surrogate CSV) |
   | `gnn_full` | 5 | GraphSAGE full features (→ surrogate CSV) |
   | `gcn_raw_attr` | 5 | GCN raw-attr ← **NEW v3.1 (C2)** |
   | `gin_raw_attr` | 5 | GIN raw-attr ← **NEW v3.1 (C2)** |
   | `gat_raw_attr` | 5 | GAT raw-attr ← **NEW v3.1 (C2)** |
   | `best_arch_raw_attr_rankloss` | 5 | Best arch + ranking loss ← **NEW v3.1 (C3)** |

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

4. **[MUST — v3.1 unconditional] Group 5 — GNN — Architecture comparison + ablation variants** (v3.1 Section 9.1):

> _v3.1: GNN architecture comparison là unconditionally MUST theo professor's framing. `gnn_branch_viable` gate từ M2 không còn áp dụng cho architecture comparison (C2) và bootstrap CI (C4)._

**[NEW v3.1 — MUST] Architecture Comparison (C2):**
Chạy với `raw_attr` features, 5 seeds mỗi arch — **5 architectures total** (SAGE + GCN + GIN + GAT + **APPNP**):

> **⚠ Naming canonical rule:** SAGE raw-attr baseline **phải được ghi vào surrogate CSV với tên `gnn_raw_attr`** (không phải `sage_raw_attr`) để backward compatibility với existing artifacts và consumer scripts. `sage_raw_attr` chỉ là alias giải thích trong table này; **KHÔNG ghi tên `sage_raw_attr` vào file CSV**. Các arch mới dùng prefix arch: `gcn_raw_attr`, `gin_raw_attr`, `gat_raw_attr`, `appnp_raw_attr`.

| Architecture      | **CSV model_name (canonical)**                                                  | Priority             | Inductive bias hypothesis                                                                                                |
| ----------------- | ------------------------------------------------------------------------------- | -------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| GraphSAGE (đã có) | **`gnn_raw_attr`** ← tên CSV chuẩn (alias: sage_raw_attr — chỉ dùng trong docs) | ✅ Done              | Mean agg. — baseline                                                                                                     |
| **GCN**           | `gcn_raw_attr`                                                                  | **MUST (C2)**        | **H2: `D^{-1/2}AD^{-1/2}` ≈ A2 symmetric diffusion — expected better under A2 labels (nếu chạy sensitivity S1)**         |
| GIN               | `gin_raw_attr`                                                                  | **MUST (C2)**        | Sum agg. preserves multi-hop counts (WL-equivalent expressiveness); reference for non-degree-weighted IC dynamics        |
| **GAT**           | `gat_raw_attr`                                                                  | **MUST (C2)**        | **H1: learned attention có thể học 1/deg(v) weighting** _(hypothesis — C2 decides)_                                      |
| **🆕 APPNP**      | `appnp_raw_attr`                                                                | **MUST (C2) — H3 ★** | **H3: K-step PPR propagation + teleport/restart (structural analogy/inductive bias) — STRONGEST theoretical motivation** |

> **Ba inductive bias hypotheses — pre-registered trước C2 (để report theo framing đúng):**
>
> - **H1 (GAT–A0):** Dưới IC primary (A0), GAT có thể học attention weight tỷ lệ nghịch với neighbor degree — IC `p=1/deg(v)` phù hợp với attention mechanism. _(hypothesis, C2 decides)_
> - **H2 (GCN–A2):** Nếu chạy Sensitivity S1 (A2 labels), GCN expected to improve vì `D^{-1/2}AD^{-1/2}` ≈ A2. _(testable, phụ thuộc S1 có chạy không)_
> - **H3 (APPNP — IC cascade analogy):** APPNP thực hiện K-step Personalized PageRank: `x^(k) = (1-α)·Â·x^(k-1) + α·x^(0)`. Với `K=10, alpha=0.15`: α là teleport/restart weight (tái-inject `x^(0)` mỗi bước; không diễn giải như xác suất IC “dừng”). Đây là **structural analogy/inductive bias** cho target diffusion-like — hypothesized best arch. _(Klicpera et al., ICLR 2019)_
>
> Cả ba hypotheses đều có **prepared narratives cho mọi outcome**. Không claim kết quả trước khi chạy C2.
>
> **Tie-break (nếu diff < 0.001):** APPNP > GAT > GIN > GCN > SAGE (**pre-registered**; APPNP ưu tiên vì H3 theory).

> **Context:** Xem bảng real numbers trong **ablation story** bên dưới để hiểu structural constraint của A0 (tại sao GNN khó beat degree, H3 rationale, và outcome interpretations).

**[NEW v3.1 — MUST] Ranking Loss (C3):**
Sau khi C2 xong → train best arch với combined α·Huber + (1-α)·pairwise-margin-loss.
CSV name: `best_arch_raw_attr_rankloss`

**[NEW v3.1 — MUST] Bootstrap CI (C4):**
`bootstrap_spearman_ci(y_true, gnn_best_preds, degree_preds)` → `gnn_vs_degree_bootstrap_ci.json`

> **⚠ C4 Protocol spec (bắt buộc — để tránh lệch triển khai):**
>
> | Tham số             | Giá trị locked                                                                                                                         |
> | ------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
> | **Metric được CI**  | Spearman ρ **only** (không phải NDCG hay P@10% — Spearman là primary ranking metric)                                                   |
> | **Đơn vị resample** | Nodes trong **test set** (resample with replacement, size = n_test; không resample training set)                                       |
> | **Δ definition**    | `Δ = Spearman(GNN_best, y_true) − Spearman(degree, y_true)` trên **cùng test set**                                                     |
> | **"GNN_best"**      | Architecture có **mean Spearman cao nhất** qua 5 seeds từ C2 (nếu C2 chưa xong: dùng SAGE); predictions = mean predictions qua 5 seeds |
> | **"degree"**        | `rank(degree)` trên toàn active graph, đã filter về test nodes (cùng y_true vector)                                                    |
> | **n_bootstrap**     | 1,000                                                                                                                                  |
> | **seed**            | 42                                                                                                                                     |
> | **CI**              | 95% → percentile(2.5) và percentile(97.5)                                                                                              |
>
> **Diễn giải output (quyết định paper claim):**
>
> - `ci_95_lower > 0` → GNN significantly better → claim "GNN surpasses degree"
> - `ci_95_lower ≤ 0 ≤ ci_95_upper` → statistically equivalent → claim "GNN achieves statistically equivalent Spearman ρ to degree while requiring no precomputed graph statistics"
> - `ci_95_upper < 0` → GNN significantly worse → focus on "+0.099 message passing story" (GNN vs MLP), không claim GNN vs degree superiority

---

**Ablation variants (per best architecture, hoặc SAGE nếu C2 chưa xong):**

| Variant          | Features (in_dim)                                        | Role                                                                    |
| ---------------- | -------------------------------------------------------- | ----------------------------------------------------------------------- |
| **GNN-raw-attr** | `views_log_norm, views_per_day_norm, life_time_norm` (3) | **MUST — Primary proposed**                                             |
| GNN-graph-only   | `degree_norm` only (1)                                   | **MUST** — Ablation: topology without attributes                        |
| GNN-centrality   | `degree_norm, pagerank_norm, kshell_norm` (3)            | **MUST** — Ablation: hand-crafted features                              |
| GNN-full         | all 6 features (normalized)                              | ✦ [IF TIME] — supplementary upper bound (có thể cắt nếu tight timeline) |
| GNN-random       | random/constant node features (1)                        | ✦ [IF TIME] — sanity-check message passing value                        |

> ✦ **[IF TIME]** `GNN-random` — không block deadline; chỉ chạy sau khi xong toàn bộ MUST GNN variants. Nếu chạy: ghi vào `surrogate_ranking_metrics.csv` với `model_name=gnn_random`.

> **Feature normalization bắt buộc**: tất cả features phải normalize trước khi vào GNN (min-max hoặc z-score). Column names trong experiment.yaml là `*_norm`. Không dùng raw values trực tiếp.

**Config chuẩn cho TẤT CẢ 5 architectures — locked để fair comparison:**

| Hyperparameter | Giá trị (conv-based archs)                   | Ghi chú APPNP                                                                |
| -------------- | -------------------------------------------- | ---------------------------------------------------------------------------- |
| `hidden_dim`   | 128                                          | Không thay đổi per arch (APPNP dùng hidden_dim cho MLP embedding)            |
| `n_layers`     | 2                                            | Conv-based only; APPNP không dùng n_layers                                   |
| `dropout`      | 0.3                                          | Không thay đổi per arch                                                      |
| `gat_heads`    | 4                                            | Chỉ cho GAT (`out_dim=128//4=32 per head → concat → 128`)                    |
| `appnp_K`      | **10**                                       | **APPNP only** — cascade depth (propagation steps)                           |
| `appnp_alpha`  | **0.15**                                     | **APPNP only** — teleport/restart weight (starting point; controls locality) |
| Loss           | Huber (`delta=1.0`)                          | Không dùng early stopping — **giống nhau cho tất cả 5 archs**                |
| `lr`           | 0.001 (Adam)                                 | Không thay đổi per arch                                                      |
| `epochs`       | 200 (cố định)                                | **Không early stopping** — cố định để fair comparison                        |
| Training seeds | `[42, 123, 456, 789, 1024]`                  | 5 seeds mỗi arch                                                             |
| Split          | `split_masks.parquet` (M0-locked)            | **Cùng split cho mọi arch**                                                  |
| Features (C2)  | `raw_attr` (views_log, views/day, life_time) | C2 chỉ so sánh trên raw_attr — 5 archs × 1 feature set                       |

**Best arch selection criterion (cho C3, C4, ablation):**

> **Best arch = architecture có `spearman_rho_mean` cao nhất** qua 5 seeds trong `surrogate_ranking_metrics.csv`. Nếu tie (diff < 0.001): ưu tiên theo thứ tự **APPNP > GAT > GIN > GCN > SAGE** (pre-registered; APPNP ưu tiên vì H3 theory). Ghi `gnn_primary_arch` vào `docs/experiment_registry.md` ngay sau khi C2 xong — C3 và C4 depend on this value.

Architectures: `sage` (SAGEConv, mean) | `gcn` (GCNConv) | `gin` (GINConv+MLP) | `gat` (GATConv, heads=4) | **`appnp`** (K=10, alpha=0.15, **H3 expected best**).
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

# Safety checks to prevent silent unlabeled leakage
assert torch.all(~(data.train_mask & data.test_mask))
assert torch.all(data.train_mask <= labeled_mask)
assert torch.all(data.test_mask <= labeled_mask)
assert int(data.train_mask.sum() + data.test_mask.sum()) == int(labeled_mask.sum())

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

**GraphSAGE architecture cụ thể (PyG) — SAGE reference:**

```python
# ⚠ v3.1 UPDATE: class GraphSAGE dưới đây chỉ là SAGE reference.
# Dùng GNNSurrogate(arch='sage'|'gcn'|'gin'|'gat') từ Implementation Plan v3.1 Section 9.1
# để support tất cả 4 architectures. Class dưới chỉ giữ cho reference / backward compat.
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

# v3.1 — Unified class (thay GraphSAGE cho architecture comparison):
# ARCHITECTURES = ['sage', 'gcn', 'gin', 'gat']
# model = GNNSurrogate(arch=arch, in_dim=in_dim, hidden_dim=128, n_layers=2, dropout=0.3)
```

**GNNSurrogate + APPNPSurrogate — unified implementation cho C2 (copy từ Implementation Plan v3.1 Section 9.1):**

```python
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GINConv, GATConv, APPNP

class GNNSurrogate(nn.Module):
    """
    Unified GNN wrapper — swap architecture bằng arch parameter.
    Config chuẩn: hidden_dim=128, n_layers=2, dropout=0.3 (KHÔNG thay đổi khi so sánh arch).
    Supported arch: 'sage' | 'gcn' | 'gin' | 'gat' (KHÔNG phải 'appnp' — xem APPNPSurrogate bên dưới).
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
                return GCNConv(in_c, out_c)
            elif arch == 'gin':
                mlp = nn.Sequential(nn.Linear(in_c, out_c), nn.ReLU(),
                                    nn.Linear(out_c, out_c))
                return GINConv(mlp)
            elif arch == 'gat':
                heads = gat_heads if out_c > 1 else 1
                return GATConv(in_c, out_c // heads, heads=heads, dropout=0.0)

        self.convs.append(make_conv(in_dim, hidden_dim))
        for _ in range(n_layers - 1):   # n_layers=2 → 2 conv layers total
            self.convs.append(make_conv(hidden_dim, hidden_dim))
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = self.act(self.dropout(conv(x, edge_index)))
        return self.head(x).squeeze(-1)


class APPNPSurrogate(nn.Module):
    """
    🆕 APPNP surrogate — embed-then-propagate (khác với conv-stack).

    Theory: APPNP = K-step Personalized PageRank (Klicpera et al., ICLR 2019):
        x^(k) = (1 - alpha) * A_hat * x^(k-1) + alpha * x^(0)

    IC cascade analogy (H3):
        - alpha (0.15) là teleport/restart weight: tái-inject x^(0) mỗi bước (không diễn giải như cơ chế dừng của IC)
        - K (=10 steps) là số bước propagation (độ sâu receptive field)
        - (1 - alpha) là phần “propagate” qua lân cận trong công thức APPNP

    Tại sao APPNP là H3 (strongest theoretical alignment):
    - IC multi-hop (test split): two_hop ρ=0.804 > one_hop ρ=0.688 → cascade là multi-hop process
    - SAGE mean aggregation: không phân biệt depth → bị smoothed → 0.470 (graph_only)
    - APPNP PPR: weighted multi-hop + teleport → structural analogy/inductive bias (không phải tương đương hình thức)
    → Expected best arch cho IC regression.
    """
    def __init__(self, in_dim=3, hidden_dim=128, K=10, alpha=0.15, dropout=0.3):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 1)
        self.prop = APPNP(K=K, alpha=alpha, dropout=dropout)
        self.dropout_fn = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        # Step 1: embed features → prediction space
        x = F.relu(self.lin1(x))
        x = self.dropout_fn(x)
        x = self.lin2(x)                   # (N, 1)
        # Step 2: propagate via K-step PPR
        x = self.prop(x, edge_index)
        return x.squeeze(-1)


def get_model(arch, in_dim, hidden_dim=128, n_layers=2, dropout=0.3,
              gat_heads=4, appnp_K=10, appnp_alpha=0.15):
    """Factory — trả về đúng model class. Dùng thay vì khởi tạo trực tiếp."""
    if arch == 'appnp':
        return APPNPSurrogate(in_dim=in_dim, hidden_dim=hidden_dim,
                              K=appnp_K, alpha=appnp_alpha, dropout=dropout)
    return GNNSurrogate(arch=arch, in_dim=in_dim, hidden_dim=hidden_dim,
                        n_layers=n_layers, dropout=dropout, gat_heads=gat_heads)


# C2: 5 architectures (APPNP added — H3: IC cascade analog)
ARCHITECTURES = ['sage', 'gcn', 'gin', 'gat', 'appnp']
training_seeds = [42, 123, 456, 789, 1024]

def train_and_eval(arch, features, seed, data, epochs=200, lr=1e-3):
    import torch.nn.functional as F
    set_all_seeds(seed)
    model = get_model(arch, in_dim=features.shape[1])   # ← dùng get_model (không phải GNNSurrogate trực tiếp)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.huber_loss(out[data.train_mask], data.y[data.train_mask], delta=1.0)
        loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad():
        preds = model(data.x, data.edge_index)[data.test_mask].cpu().numpy()
    return preds
```

**✦ [IF TIME — C5] GINESurrogate với IC Edge Features (supplemental upper bound):**

> **Khi nào làm:** Chỉ sau khi C2 + C3 + C4 xong. Không block critical path.
>
> **Lý do GINE đặc biệt:** GINEConv incorporates edge features vào message passing. Nếu `edge_attr = 1/deg(v)` (= IC-A0 probability của mỗi cạnh), model nhận **explicit IC mechanism** trong từng message — message từ u đến v = `ReLU(h_u + p(u,v))`.
>
> Đây là **upper bound experiment**: quantify gain khi model được "cho biết" xác suất IC của mỗi cạnh. Nếu GINE + IC edge feat vẫn không beat degree → IC-A0 structural constraint là absolute.
>
> **⚠ KHÔNG phải "feature-agnostic":** edge features = structural property (1/deg). **KHÔNG** đưa vào C2 fair comparison. Label rõ trong paper là "C5 supplemental: GNN with explicit IC mechanism encoding."

```python
from torch_geometric.nn import GINEConv

class GINESurrogate(nn.Module):
    """
    GINE + IC edge features — explicit IC mechanism encoding (C5 supplemental).
    edge_attr = IC propagation probability per edge (shape: E × 1).
    
    ⚠ Không phải feature-agnostic. Không đưa vào C2 fair comparison.
    Dùng: C5 supplemental — upper bound analysis (sau khi C2/C3/C4 xong).
    
    Ref: Hu et al., NeurIPS 2019.
    """
    def __init__(self, in_dim=3, hidden_dim=128, edge_dim=1, dropout=0.3):
        super().__init__()
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
    Tạo IC probability làm edge feature cho mỗi directed edge (u, v).
    
    rule='a0': p(u,v) = 1/deg(v)               — IC primary (Weighted Cascade)
    rule='a2': p(u,v) = 1/sqrt(deg(u)*deg(v))  — IC symmetric (A2 sensitivity)
    
    Returns: edge_attr tensor shape (E, 1)
    """
    src, dst = edge_index[0], edge_index[1]
    if rule == 'a0':
        probs = 1.0 / degrees[dst].float().clamp(min=1)
    elif rule == 'a2':
        probs = 1.0 / (degrees[src].float() * degrees[dst].float()).sqrt().clamp(min=1)
    return probs.unsqueeze(-1)   # (E, 1)

# Cách dùng C5:
# edge_attr = compute_ic_edge_features(data.edge_index, degree_tensor, rule='a0')
# model_gine = GINESurrogate(in_dim=3, hidden_dim=128, edge_dim=1)
# preds = model_gine(data.x, data.edge_index, edge_attr)[data.test_mask]
# CSV model_name: 'gine_ic_a0_raw_attr'
```

> **C5 experiment variants (nếu có thời gian):**
>
> | Variant | Edge features | Node features | CSV model_name |
> |---|---|---|---|
> | GINE-IC-A0 (primary) | `1/deg(v)` | `raw_attr` | `gine_ic_a0_raw_attr` |
> | GINE-IC-A2 | `1/√(deg(u)×deg(v))` | `raw_attr` | `gine_ic_a2_raw_attr` |
>
> **Paper framing C5:** "As supplemental upper-bound analysis, we encode IC propagation probabilities directly as edge features (GINE; Hu et al., 2019). Comparing GINE-IC-A0 against GNN-raw-attr (no edge features) quantifies the information gain from explicit IC mechanism encoding."

---

**📋 Architecture Evaluation Log — Tổng kết các model GNN đã đánh giá**

> **Mục đích:** Khi reviewer hỏi "why not try X?", team có documented rationale sẵn. Cũng là checklist để không waste time implement architectures không phù hợp với project này.

| Architecture | Verdict | Dùng ở đâu | Lý do chi tiết |
|---|---|---|---|
| **SAGE** (mean agg.) | ✅ **Trong C2-A0** (baseline) | C2-A0 row `gnn_raw_attr` | Baseline reference. Mean agg. bị smoothing → 0.470 (graph_only), 0.534 (raw_attr). |
| **GCN** | ✅ **MUST C2-A0** (H2) | C2-A0 `gcn_raw_attr` | H2: D^{-1/2}AD^{-1/2} ≈ A2 symmetric IC. Test cả C2-A0 và C2-A2. |
| **GIN** | ✅ **MUST C2-A0** | C2-A0 `gin_raw_attr` | Sum agg. — highest WL expressiveness; preserves hop counts. |
| **GAT v1** | ✅ **MUST C2-A0** (H1) | C2-A0 `gat_raw_attr` | H1: static attention có thể học 1/deg(v). Static = đúng cho A0. |
| **APPNP** | ✅ **MUST C2-A0** (H3) | C2-A0 `appnp_raw_attr` | H3: K-step PPR ≈ IC cascade. Expected best arch. |
| **GATv2** | ✅ **MUST C2-I-A** (H4) | C2-I-A `gatv2_raw_attr_ia` | H4: dynamic attention khớp I-A row-normalization. **KHÔNG dùng trong C2-A0** (static GAT phù hợp hơn cho A0). |
| **GINE + IC edge feat** | ✅ **C5 [IF TIME]** | `gine_ic_a0_raw_attr` | Strongest alignment: explicit IC prob làm edge feature. NOT feature-agnostic. Upper bound experiment. |
| **GCNII** | ❌ **Skip C2** | — | Advantage chỉ tại L=16–64. Tại `n_layers=2` (C2 locked) ≈ GCN + residual. Cần separate L=16 experiment → phá fair comparison. |
| **HGT** | ❌ **Loại hoàn toàn** | — | Designed cho **heterogeneous graphs** (many node/edge types). Twitch = **homogeneous** (1 type). Type matrices collapse → complex GAT variant, không có lợi. |
| **GraphGPS** | ❌ **Loại — scale** | — | MPNN + Transformer O(N²) với N=168k = 28 tỷ pairs. LapPE eigendecomposition 168k×168k tốn 30–60 phút. Overkill cho 3-feature node regression. |

> **Quick rule cho future architectures:**
> - Graph homogeneous? → Loại HGT, DGI heterogeneous variants
> - Scale O(N²)? → Loại nếu không có efficient approx + benchmark trước
> - Advantage chỉ tại L >> 2? → Không đưa vào C2, test riêng
> - Cần edge features ngoài structural? → Verify có data trước khi implement

---

**C3 — Ranking Loss Experiment (sau khi C2 xác định best arch):**

```python
import torch.nn.functional as F

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

def combined_loss(pred, target, alpha=0.5, margin=0.1):
    """Combined Huber + pairwise ranking loss. alpha=0.5 default."""
    huber = F.huber_loss(pred, target, delta=1.0)
    rank  = pairwise_ranking_loss(pred, target, margin=margin)
    return alpha * huber + (1 - alpha) * rank

# CSV name: best_arch_raw_attr_rankloss (e.g., gat_raw_attr_rankloss nếu GAT wins C2)
# Variant: alpha=0.5; sweep [0.25, 0.5, 0.75] nếu còn time
```

**C4 — Bootstrap CI: GNN best vs Degree (sau khi C2 có best-arch predictions):**

```python
import numpy as np
from scipy.stats import spearmanr

def bootstrap_spearman_ci(y_true, y_pred_a, y_pred_b, n_bootstrap=1000, seed=42):
    """Bootstrap CI for difference in Spearman: pred_a (GNN best) vs pred_b (degree)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        rho_a, _ = spearmanr(y_true[idx], y_pred_a[idx])
        rho_b, _ = spearmanr(y_true[idx], y_pred_b[idx])
        deltas.append(rho_a - rho_b)   # GNN_best - degree
    ci_lower = np.percentile(deltas, 2.5)
    ci_upper = np.percentile(deltas, 97.5)
    return float(np.mean(deltas)), ci_lower, ci_upper

# Output: outputs/mapr2026_v3_results/gnn_vs_degree_bootstrap_ci.json
# {
#   "n_bootstrap": 1000,
#   "comparator_a": "gnn_best_architecture",   # e.g., "gat_raw_attr"
#   "comparator_b": "degree",
#   "delta_mean": <float>,
#   "ci_95_lower": <float>,
#   "ci_95_upper": <float>,
#   "interpretation": "equivalent | significantly_lower | significantly_higher"
# }
# Decision: CI bao gồm 0 → "statistically equivalent to degree"
#           CI entirely negative → focus on +0.099 message passing story
# Thời gian: ~10 phút (resample existing predictions, không cần retrain)
```

Ablation story:

> **📊 Real numbers từ existing artifacts (bắt buộc hiểu trước khi chạy C2):**
>
> | Model                   | Spearman ρ | Ghi chú                                          |
> | ----------------------- | ---------- | ------------------------------------------------ |
> | `degree`                | **0.826**  | Baseline rất mạnh dưới A0 (label degree-coupled) |
> | `two_hop_spread`        | **0.804**  | IC is multi-hop (+0.116 vs one_hop)              |
> | `gnn_centrality` (SAGE) | **0.817**  | Circular: degree feature given as input          |
> | `one_hop_spread`        | **0.688**  | One hop alone không đủ                           |
> | `gnn_raw_attr` (SAGE)   | **0.534**  | SAGE + raw attrs — still far from degree         |
> | `gnn_graph_only` (SAGE) | **0.470**  | SAGE mean agg bị smoothing — fails               |
> | `mlp_raw_attr`          | **0.435**  | No message passing                               |
> | `cv_score`              | **0.2109** | Near-critical IC dynamics → regression motivated |
>
> **Key insight:** IC-A0 dùng `p=1/deg(v)` → IC score degree-coupled (tương quan rất cao với degree) → `degree` là baseline rất mạnh. Đây là **structural constraint của A0**, không phải implementation bug.

- **Architecture comparison (C2 primary — A0 labels):** 5 archs (SAGE / GCN / GIN / GAT / **APPNP**) trên `raw_attr` → _which message passing best captures IC's multi-hop dynamics?_
  - **H3 (APPNP — STRONGEST, expected best):** APPNP K-step PPR propagation `x^(k) = (1-α)·Â·x^(k-1) + α·x^(0)` là structural analogy/inductive bias cho target diffusion-like (`K=10, alpha=0.15` là starting point). IC là multi-hop process (two_hop 0.804 > one_hop 0.688) → APPNP's depth-aware propagation expected to capture this better than SAGE mean.
  - **H1 (GAT–A0):** _(hypothesis — to be confirmed by C2)_ weighted cascade `p=1/deg(v)` → GAT attention **có thể** học inversely-proportional-to-degree weighting tự động
  - **H2 (GCN–A2):** _(hypothesis — chỉ testable nếu Sensitivity S1 chạy được)_ GCN's `D^{-1/2}AD^{-1/2}` ≈ A2 symmetric IC rule
  - **GIN:** sum aggregation → có thể capture two-hop count tốt hơn SAGE mean (vì sum preserves hop counts, không smooth out)
  - Cả 5 arch hypothesis đều có prepared narratives (xem Section 4.1b của Implementation Plan)
  - **Nếu C2 vẫn không beat degree = EXPECTED (không phải failure):** A0 label ∝ 1/deg(v) → structural ceiling. → I-A supplemental analysis sẽ unlock genuine GNN advantage với degree-blind labels.
- **Feature ablation (dùng best arch từ C2):**
  - GNN-raw-attr vs MLP-raw-attr → giá trị của **message passing** (**+0.099 Spearman confirmed**: 0.534 vs 0.435)
  - GNN-raw-attr vs GNN-graph-only → giá trị của **content attributes** (0.534 vs 0.470)
  - GNN-raw-attr vs GNN-centrality → so sánh learned vs hand-crafted features (0.534 vs 0.817)
  - GNN-raw-attr vs Group 2 (degree=0.826) → **bootstrap CI** (C4) để test statistical equivalence
    > 📌 **"Feature-agnostic" clarification (reviewer prep):** Trong paper = GNN không cần pre-compute centrality/structural features (degree, PageRank, k-shell). GNN-raw_attr vẫn dùng user **metadata** (views_log_norm, views_per_day_norm, life_time_norm) — **metadata tĩnh**, KHÔNG phải behavioral traces. **A0 IC labels** là views-independent; nếu bật **I-A supplemental**, đó là **attribute-informed operationalization** và phải label rõ. +0.099 story = message passing adds structural signal **beyond same metadata** (GNN vs MLP với raw_attr giống nhau → chỉ khác message passing).
- **Ranking loss (C3):** best_arch + combined α·Huber + (1-α)·pairwise-margin → improve ranking metrics directly
- **CV=0.2109 paper framing:** "Near-critical IC dynamics (CV=0.209) empirically motivated continuous regression formulation — NOT a fallback; the principled choice for simulation-derived continuous targets."

5. Repeated training seeds + reporting (v3.1 Sections 8.7 + 9.1):
   - **5 seeds:** `[42, 123, 456, 789, 1024]` → report `mean ± std` cho mỗi metric trong `surrogate_ranking_metrics.csv`
   - **Lưu ý về BH-FDR:** Chỉ áp dụng nếu chạy nhiều MWU tests (multiple comparisons). Trong scope bình thường thì report mean±std là đủ.

6. Runtime table (v3.1 Section 9.3):

   | Component                               | Metric          | Notes                                                                                               |
   | --------------------------------------- | --------------- | --------------------------------------------------------------------------------------------------- |
   | Feature precompute (degree, PR, kshell) | time            | Centrality baselines only                                                                           |
   | MC IC labeling (n_sample × N_runs)      | time            | One-time cost — từ Person 1                                                                         |
   | GNN training (5 seeds)                  | time            | With GPU                                                                                            |
   | **GNN inference (168,114 nodes)**       | **runtime_sec** | Full active graph                                                                                   |
   | Node2Vec training                       | time            |                                                                                                     |
   | Speedup: MC IC vs GNN inference         | **7,169×**      | Key claim: 480s / 0.067s (confirmed từ runtime_breakdown.csv) — xem operational definition bên dưới |

   `runtime_sec` trong CSV = **inference only** (không tính load + precompute).

   > **Operational definition 7,169× (reviewer prep):**
   >
   > - **480s** = MC-IC labeling 5,000 nodes × 200 runs (one-time training label generation, joblib parallelism).
   > - **0.067s** = GNN inference forward-pass trên toàn bộ 168,114 active nodes (sau training xong).
   > - **Conservative lower-bound:** 480s/0.067s = 7,169× so sánh "labeling cost 5k nodes" vs "inferring all 168k nodes" — không cùng population. Full-graph-vs-full-graph speedup ~241,000× (16,140s IC vs 0.067s GNN).
   > - **Framing an toàn trong paper:** "GNN inference (0.067s, 168k nodes) is 7,169× faster than MC-IC label generation (480s, 5k×200 runs) used for training." Không claim 7,169× là same-population comparison.

**Runtime rule (để so sánh fair):** log riêng 3 phần (precompute / train / inference). Trong `baseline_ranking_metrics.csv` để `runtime_sec` là inference time trên full active nodes, và ghi chi tiết breakdown ở file phụ `outputs/mapr2026_v3_results/runtime_breakdown.csv` (contract bắt buộc trong M0).

> **QUAN TRỌNG (v3.1 Section 9.3):** Nếu GNN-raw-attr là primary, **không cần tính centrality precompute time** (degree/PR/kshell) vào runtime GNN — centrality chỉ cần cho GNN-centrality và GNN-full. Việc loại bỏ centrality precompute khỏi primary GNN pipeline làm runtime so sánh **fair hơn** (và là một điểm mạnh của GNN-raw-attr: không cần expensive precompute).

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
- **[v3.1 MUST — C2, deadline 18/4]** `surrogate_ranking_metrics.csv` có rows `gcn_raw_attr`, `gin_raw_attr`, `gat_raw_attr` (5 seeds each, mean±std).
- **[v3.1 MUST — C3, deadline 20/4]** `surrogate_ranking_metrics.csv` có row `best_arch_raw_attr_rankloss` (best arch từ C2 + combined_loss α=0.5).
- **[v3.1 MUST — C4, deadline 20/4]** `gnn_vs_degree_bootstrap_ci.json` tồn tại với `n_bootstrap=1000`, `ci_95_lower`, `ci_95_upper`, `interpretation`.
- Runtime table có `Speedup: MC IC vs GNN inference` được tính (M5).
- `runtime_sec` = full-graph inference time (đo `time.time()` bao toàn bộ forward pass, không tính file load).
- **[v3.1 code fix — S2]** Kiểm tra `eval_ranking_harness.py`: tất cả 4 `argsort` calls phải dùng `kind='stable'` để tránh tie-breaking non-determinism trong NDCG@10% / P@10%.
- **[v3.1 code fix — S3]** Kiểm tra `run_baselines.py evaluate_on_test_mask()`: thêm `sort_values("node_id")` sau `apply_test_mask()` để đảm bảo node order nhất quán khi merge predictions.

---

## 4) Nhịp tích hợp (deadline 30/4/2026 — **📍 Hôm nay: 17/4/2026, còn 13 ngày**)

> 📍 **Execution status (17/4/2026):**
> - M0–M3: ✅ Hoàn thành (IC labels, split mask, baselines Group 1–3 done)
> - M4: 🔄 Đang chạy — C1 (16/4 done), C2 (19/4), C3/C4 (21/4)
> - M5: ⏳ Pending — 22–27/4
> _(Update dòng này mỗi ngày khi milestone mới complete)_

### Milestone M0 — Kick-off (6/4, buổi sáng, ~1 giờ)

**Mục tiêu:** đồng thuận trước khi ai code. Không skip.

Agenda bắt buộc:

1. Xác nhận Stage 0–2 artifacts tồn tại trên máy mọi người (chạy quickstart Mục 0)
2. Điền và commit `docs/m0_decisions.md` (xác nhận 8 quyết định đã lock)
3. Phân công branch: `feature/mapr-ic-core`, `feature/mapr-community-proxies`, `feature/mapr-surrogate-eval`
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
| `views/IC ρ > 0.8`                                                                 | "We find high popularity-diffusion agreement (ρ > 0.8) on Twitch's dense graph. The small divergent subset shows systematically higher betweenness and cross-community connectivity."                                                                                               |

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
| Person 2 | Community features + diffusion proxies + correlation matrix           | 12/4           |
| Person 3 | Chạy baseline ranking thật (Group 1–2)                                | 12/4           |

**Done khi:** `baseline_ranking_metrics.csv` có ít nhất Group 1–2 rows với real IC labels.

**[M3] Views/IC alignment check — narrative lookup cho RQ2:**

> 📋 **[REFERENCE — không phải task thêm]** Sau khi Person 1 ghi `spearmanr(views, ic_score_mean)` vào `docs/day1_decisions.md` Phần 4, cả team tra bảng và chọn narrative tương ứng. Không cần chạy thêm experiment.

| views/IC Spearman ρ | Narrative RQ2 (no categorical labels)                                                                                                                          |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ρ < 0.70            | "Strong divergence: popularity (views) không phản ánh diffusion potential (IC). Nhấn mạnh định lượng tương quan thấp + so sánh với degree/centrality/proxies." |
| 0.70–0.85           | "Moderate divergence: views có tương quan đáng kể với IC nhưng còn residual variance. Dùng correlation matrix để chỉ ra metric nào proxy tốt/kém."             |
| ρ > 0.85            | "High agreement: views là proxy mạnh cho IC trong dataset này. Nhấn mạnh robustness/sensitivity + limitations; tránh over-claim value-add của complex models." |

---

### Milestone M4 — Full pipeline (12–22/4)

| Person   | Việc                                                                                        | Deadline gợi ý |
| -------- | ------------------------------------------------------------------------------------------- | -------------- |
| Person 2 | **Community detection** (Louvain + cross_community_edge_fraction)                           | 10/4           |
| Person 3 | Group 3 baselines (one-hop/two-hop từ proxies full graph)                                   | 15/4           |
| Person 3 | Node2Vec (`dim=64, walks=20`) + LR + MLP raw attr                                           | 18/4           |
| Person 3 | GNN-raw-attr + 3 ablation variants — **[MUST — v3.1 unconditional]** (dùng best arch từ C2) | 22/4           |
| Person 3 | Runtime table + Speedup calculation                                                         | 22/4           |
| Person 1 | **[C1] Degree-controlled IC variance test**                                                 | 16/4           |
| Person 3 | **[C2] GCN/GIN/GAT/APPNP arch comparison** (5 archs × 5 seeds = 25 runs)                    | **19/4**       |
| Person 3 | **[C3] Ranking loss experiment** (best arch từ C2)                                          | 21/4           |
| Person 3 | **[C4] Bootstrap CI GNN vs degree**                                                         | 21/4           |

> **Critical path v3.1 experiments (updated cho 5 archs):**
> C2 (5 arch comparison: SAGE/GCN/GIN/GAT/APPNP) → done by **19/4** EOD
> ↓
> C3 (ranking loss: best arch) + C4 (bootstrap CI) → start 20/4, done by **21/4**
> ↓
> [IF I-A pass] C2-I-A → start ≈22/4, done ≈23/4

---

### Milestone M5 — Integration + paper hand-off (22–27/4)

- Tất cả artifacts gom vào `outputs/mapr2026_v3_results/`
- Final `baseline_ranking_metrics.csv` (Groups 1–4) + `surrogate_ranking_metrics.csv` (Group 5 — GNN) hoàn chỉnh
- `runtime_breakdown.csv` hoàn chỉnh (precompute / train / inference riêng biệt)
- Bàn giao cho người viết paper: bảng kết quả + plots chính

---

## 4b) Risk Management (v3 Section 19)

| Rủi ro                                                                                      | Xác suất             | Impact       | Action                                                                                                                                                                                                                                                                                                                      |
| ------------------------------------------------------------------------------------------- | -------------------- | ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `one_hop_rho > 0.9` + top-k alignment cao (`jaccard_at_10pct > 0.8`, `ndcg_at_10pct > 0.9`) | Trung bình           | **Critical** | M2: check trước; nếu đủ 3 điều kiện thì restructure, nếu không giữ GNN + 2-hop                                                                                                                                                                                                                                              |
| IC runtime > 8h                                                                             | Trung bình           | **Critical** | M2: reduce n_sample=2k, N_runs=100; ghi limitation                                                                                                                                                                                                                                                                          |
| GNN không beat cheap proxies                                                                | Trung bình           | Thấp         | Prepared narrative "negative result" vẫn publishable                                                                                                                                                                                                                                                                        |
| `views/IC ρ > 0.8`                                                                          | Trung bình           | Thấp         | Prepared narrative "high agreement" (Mục 4 M2)                                                                                                                                                                                                                                                                              |
| Louvain partition quá nhạy với resolution (B9)                                              | Trung bình           | Trung bình   | Chạy `louvain_resolution_sensitivity.json`; nếu `<20 communities` hoặc `top3>50%` thì nghi over-merge, nếu `>200` + nhiều singleton thì nghi over-split; chỉ đổi resolution sau khi re-lock                                                                                                                                 |
| Overclaim accuracy trên unlabeled nodes (B10)                                               | Trung bình           | Trung bình   | Khóa wording transductive: metrics chỉ trên held-out labeled; full-graph chỉ runtime; nếu cần claim rộng hơn thì chạy out-of-sample IC audit                                                                                                                                                                                |
| loky OOM với full graph                                                                     | Thấp                 | Cao          | Reduce `n_jobs`; monitor RAM ≥ 32 GB khi chạy                                                                                                                                                                                                                                                                               |
| PyG installation issues                                                                     | Thấp                 | Trung bình   | Setup M0; fallback: DGL nếu PyG fail                                                                                                                                                                                                                                                                                        |
| Paper > 6 trang                                                                             | Trung bình           | **Blocker**  | Cắt theo bảng dưới                                                                                                                                                                                                                                                                                                          |
| GNN-A0 không vượt degree sau full arch search (SAGE/GCN/GIN/GAT/**APPNP**)                  | **Cao** (structural) | Trung bình   | **Đây là structural expectation, KHÔNG phải bug:** IC-A0 label degree-coupled (`p=1/deg(v)`) → degree là baseline rất mạnh (ρ=0.826). → APPNP (H3) là best bet; nếu vẫn thua: Bootstrap CI equivalence claim + +0.099 message passing story. **Primary mitigation: I-A labels** (degree-blind → GNN has genuine advantage). |
| APPNP không improve (H3 rejected)                                                           | Thấp                 | Trung bình   | Report honestly: PPR propagation không capture IC dynamics under Twitch topology; stay với GAT/GIN best result; focus on +0.099 message passing story và equivalence claim                                                                                                                                                  |
| GAT không converge (4 heads, hidden=128)                                                    | Thấp                 | Thấp         | Reduce heads=1; increase epochs=300; report instability trong experiment notes                                                                                                                                                                                                                                              |
| Ranking loss không improve Spearman so với Huber                                            | Trung bình           | Thấp         | Negative finding vào appendix note; Huber GNN remains primary                                                                                                                                                                                                                                                               |
| Degree-controlled variance test: CV < 0.3 (IC không add beyond degree)                      | Thấp                 | Trung bình   | Honest limitation; strengthen runtime story; không claim IC captures structural info beyond degree                                                                                                                                                                                                                          |
| Bootstrap CI entirely negative (GNN < degree on full arch search)                           | Thấp                 | Cao          | Restructure: focus on no-centrality advantage + message passing +0.099 story; không claim GNN superiority                                                                                                                                                                                                                   |
| High seed variance (std > 0.05 across 5 seeds)                                              | Thấp                 | Trung bình   | Report mean±std; increase to 10 seeds for final table if budget allows                                                                                                                                                                                                                                                      |
| **Sensitivity S1 (A2) không thay đổi IC ranking (Spearman(A0,A2) > 0.95)**                  | Thấp                 | Thấp         | Positive: "sensitivity confirms robustness of primary IC operationalization"; mention in 1 sentence, không cần full section                                                                                                                                                                                                 |
| **GCN không improve dưới A2 labels (H2 rejected)**                                          | Trung bình           | Thấp         | Prepared narrative: "GNN performance robust to IC rule choice; +0.099 message passing advantage holds across operationalizations"; dùng kết quả thực nghiệm thay framing GCN–A2                                                                                                                                             |
| **Views-based p(u,v) request từ reviewer**                                                  | Thấp                 | Trung bình   | Pre-documented exclusion: không có edge-level behavioral logs để justify views-based transmission; primary IC (A0/A1/A2) là views-independent → cite Section 5.3 paper                                                                                                                                                      |
| **I-A pilot fail CHECK 1 (CV ≤ 0.3 — IC-I-A degenerate)**                                   | Thấp                 | Thấp         | Abandon I-A; stay A0 primary + S1 sensitivity; ghi vào experiment_registry.md; không tốn compute thêm                                                                                                                                                                                                                       |
| **I-A pilot fail CHECK 2 (ρ_deg ≥ 0.75 — degree vẫn correlate mạnh)**                       | Trung bình           | Thấp         | Thử II-B fallback (views_density): `p(u,v)=clip(views_norm[v]/deg(v), max=0.5)` với cùng 3 checks; nếu II-B cũng fail → stay A0                                                                                                                                                                                             |
| **I-A pilot pass nhưng C2-I-A: GNN không vượt degree trên I-A labels**                      | Thấp                 | Trung bình   | Vẫn là positive finding: "GNN advantage requires attribute-informed diffusion; under structural A0, degree captures most variance." Báo cáo trung thực cả hai outcomes.                                                                                                                                                     |

## 4c) Scope Reduction — Cắt khi cần (v3 Section 16)

Nếu timeline tight, cắt theo thứ tự này (an toàn nhất trước):

| Cắt được (theo thứ tự ưu tiên CẮT ĐẦU TIÊN)           | Giữ bắt buộc (KHÔNG cắt)                                                       |
| ----------------------------------------------------- | ------------------------------------------------------------------------------ |
| ~~Uniform-p sensitivity~~ (đã cắt — không làm)        | Label stability (Jaccard + structural cause)                                   |
| **Sensitivity S2 (A1 source budget)** — cắt nếu tight | **Sensitivity S1 (A2 symmetric)** ← SHOULD DO sau primary C2                   |
| **I-A Attribute-Informed IC** — cắt nếu pilot fail    | **I-A** ← SHOULD DO nếu pilot pass + không block C2-A0                         |
| Graph perturbation test                               | **Degree-controlled IC variance test** ← v3.1 NEW MUST                         |
| 5% / 15% thresholds (chỉ giữ 10%)                     | **Architecture comparison (GCN/GIN/GAT/APPNP/SAGE)** ← v3.1 NEW MUST — 5 archs |
| Eigenvector/betweenness trong GNN features            | **Bootstrap CI GNN vs degree (A0 labels)** ← v3.1 NEW MUST                     |
| Detailed betweenness profiling                        | One-hop + two-hop proxies (Group 3)                                            |
| GNN-full variant                                      | Community detection (stability explanation only)                               |
| Ranking loss α sweep (chỉ dùng 1 α tốt nhất)          | GNN-raw-attr (primary) + GNN-graph-only (ablation)                             |
| Inductive generalization test (9.1c)                  | Runtime comparison table                                                       |
| **C5 GINE + IC edge features** ← cắt đầu tiên nếu tight sau C4 | C2-I-A + GATv2 ← SHOULD DO nếu I-A pilot pass |
| **C2-I-A GATv2** ← cắt nếu I-A pilot fail (không có I-A labels) | Bootstrap CI (A0 labels) ← MUST regardless |
| Secondary metrics (P@10%)                             | BH-FDR correction cho correlation matrix p-values                              |

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
- Evaluation harness (ranking metrics thay F1): `src/mapr2026_v3/eval_ranking_harness.py`
- Diffusion proxies (one-hop + two-hop): `src/mapr2026_v3/diffusion_proxies.py`
- Surrogate learning (GNN ablation): `src/mapr2026_v3/run_surrogates.py`

### Quick Reference: Stage ↔ Script ↔ Owner

| Stage v3                          | Script                                    | Owner        | Ghi chú                                     |
| --------------------------------- | ----------------------------------------- | ------------ | ------------------------------------------- |
| **Stage 0b (dead account audit)** | `src/data/dead_account_audit.py`          | **Person 1** | Phải có trước sampling; stats → limitations |
| Stage 2 (CSR)                     | `src/mapr2026_v3/export_csr.py`           | Person 1     |                                             |
| Stage 3 (Day-1)                   | `src/mapr2026_v3/day1_benchmark.py`       | Person 1     | Gating cho M2                               |
| Stage 4 (IC labels + split mask)  | `src/mapr2026_v3/ic_labels_primary.py`    | Person 1     | Gating cho M3                               |
| **Stage 4b (community features)** | `src/graph/community.py`                  | **Person 2** | Độc lập với IC, chạy sớm                    |
| Stage 6 (proxies full graph)      | `src/mapr2026_v3/diffusion_proxies.py`    | Person 2     | Full active graph                           |
| Stage 7 (Group 1–4 baselines)     | `src/mapr2026_v3/run_baselines.py`        | Person 3     | Group 4 = Node2Vec+LR, MLP → baseline CSV   |
| Stage 7 (Group 5 GNN ablation)    | `src/mapr2026_v3/run_surrogates.py`       | Person 3     | 4 GNN variants; mean±std → surrogate CSV    |
| (shared)                          | `src/mapr2026_v3/eval_ranking_harness.py` | Person 3     | `load_split_mask()` + metrics               |
| (shared)                          | `src/mapr2026_v3/_shared.py`              | All          | Đọc, không sửa riêng                        |

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
  - `feature/mapr-community-proxies-*` (Person 2)
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
| `diffusion_proxies.parquet` chỉ có labeled subset                                                          | `n_nodes` trong file << 168k                             | Person 2 rebuild ở real mode (full active graph)                                                                                                                                                                                |
| Louvain partition quá nhạy với resolution                                                                  | `n_communities`/modularity drift mạnh giữa `0.5/1.0/2.0` | ⚠ [IF PROBLEM: louvain_partition_instability] Chạy `louvain_resolution_sensitivity.json`; chỉ đổi resolution sau khi re-lock                                                                                                    |

---

## 8) Checklist nhanh theo người (tóm tắt 1 trang)

### Person 1 — Phạm Quốc Vĩnh

| #   | Việc                                                                                                            | Script                              | Artifact output                                                                                             | Deadline               |
| --- | --------------------------------------------------------------------------------------------------------------- | ----------------------------------- | ----------------------------------------------------------------------------------------------------------- | ---------------------- |
| 1   | CSR export                                                                                                      | `export_csr.py`                     | `graph_csr.npz`                                                                                             | M1 (6/4)               |
| 2   | **Dead account audit**                                                                                          | `src/data/dead_account_audit.py`    | `outputs/stage0_data_quality/dead_account_report.json`                                                      | M0 (6/4)               |
| 3   | **LCC check**                                                                                                   | `src/data/lcc_audit.py`             | `outputs/stage0_data_quality/lcc_report.json`                                                               | M0 (6/4)               |
| 4   | Day-1 benchmark                                                                                                 | `day1_benchmark.py`                 | `ic_runtime_benchmark.json`                                                                                 | M2 (7/4)               |
| 5   | One-hop ρ check                                                                                                 | `day1_benchmark.py`                 | `one_hop_correlation.json`                                                                                  | M2 (7/4)               |
| 6   | **IC pilot + stability (gate fail)**                                                                            | `ic_labels_primary.py`              | `outputs/day1_benchmark/ic_pilot_diagnostics.json` (`jaccard_stability`, `cv_score`, per-quintile CV table) | 9/4                    |
| 7   | IC labels (full N×R)                                                                                            | `ic_labels_primary.py`              | `ic_scores_primary.parquet`, `regression_targets.parquet`, `classification_labels.parquet`                  | 10/4                   |
| 8   | **[MUST khi Jaccard < 0.85] Stability explanation**                                                             | manual/script (extract phase1/2)    | `outputs/day1_benchmark/stability_explanation.json`                                                         | 10/4                   |
| 9   | **Split mask** [M0-locked]                                                                                      | `ic_labels_primary.py`              | `split_masks.parquet` (cùng lúc #7)                                                                         | 10/4                   |
| 10  | Ghi `day1_decisions.md`                                                                                         | manual                              | `docs/day1_decisions.md`                                                                                    | M2 (7/4)               |
| 11  | **[M3] Views/IC alignment check**                                                                               | `ic_labels_primary.py`              | cập nhật `docs/day1_decisions.md` Phần 4 (`spearmanr(views, ic_score_mean)`)                                | M3                     |
| 12  | **[NEW v3.1 — C1] Degree-controlled IC variance test**                                                          | manual/`ic_labels_primary.py`       | `outputs/mapr2026_v3_results/degree_controlled_ic_variance.json`                                            | 16/4                   |
| 13  | **[SHOULD DO] Sensitivity S1 — Symmetric IC (A2)** `p=1/√(deg(u)×deg(v))`                                       | `ic_labels_primary.py` (A2 variant) | `ic_scores_sensitivity_a2.parquet` + `ic_sensitivity_comparison.json` (Spearman A0 vs A2 vs degree)         | Sau C2 xong (≈21/4)    |
| 14  | **[SHOULD DO — nếu bật I-A] I-A Pilot** (200 nodes × 50 runs) — 3 checks: CV>0.3, ρ_deg<0.75, ρ_proxy<0.85      | `ic_labels_primary.py` (ia variant) | `outputs/mapr2026_v3_results/ic_ia_pilot_decision.json`                                                     | ≈18/4 (sau C2-A0 xong) |
| 15  | **[SHOULD DO — chỉ khi row 14 pass ALL 3 checks] I-A Full Sim** (5k nodes × 200 runs) + `ic_ia_vs_primary.json` | `ic_labels_primary.py` (ia variant) | `outputs/mapr2026_v3_results/ic_scores_ia.parquet` + `ic_ia_vs_primary.json`                                | ≈19/4                  |

> **Dependency cho row 13:** Chờ C2 (Person 3) xong trước để không block critical path. Nếu A2 labels kịp trước 21/4, Person 3 có thể chạy thêm C2-A2 (4 archs × 5 seeds) để test GCN–A2 alignment hypothesis.
> **Dependency cho rows 14–15:** Row 14 phải pass trước row 15. Nếu row 14 fail: dừng lại, không tốn compute cho row 15. Ghi lý do vào `docs/experiment_registry.md`.

### Person 2 — Trần Hùng Vĩ

| #   | Việc                                              | Script                           | Artifact output                                                                                                                                                                 | Deadline |
| --- | ------------------------------------------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| 1   | Proxies skeleton (dry-run)                        | `diffusion_proxies.py --dry-run` | dry-run header cho `data/processed/diffusion_proxies.parquet` (schema only; KHÔNG dùng cho evaluation/runtime)                                                                  | M1 (7/4) |
| 2   | **[MUST] Community detection**                    | `src/graph/community.py`         | `data/processed/community_features.parquet` (columns: `node_id`, `community_id`, `cross_community_edge_fraction`; scope=ALL active nodes, coverage=100%, `node_id` kiểu string) | 10/4     |
| 3   | **[MUST] Proxies thật (full graph)**              | `diffusion_proxies.py`           | `data/processed/diffusion_proxies.parquet` + `outputs/mapr2026_v3_results/runtime_breakdown.csv`                                                                                | 15/4     |
| 4   | **[MUST] Metric correlation matrix (global 8×8)** | manual/script                    | `outputs/mapr2026_v3_results/metric_correlation_matrix.json` (`rho_matrix`, `p_matrix_corrected`; `rho_by_degree_quintile` là ✦ IF TIME)                                        | 18/4     |

> Chú ý: **Bước 4 cần `ic_scores_primary.parquet` + `diffusion_proxies.parquet`**. Trong khi chờ: dùng `sis_table.parquet` làm mock `ic_score_mean` để test I/O.
> **Bước 2 không phụ thuộc IC labels** — có thể làm ngay từ đầu song song với bước 1.

### Person 3 — Trần Quốc Hải

| #   | Việc                                                                                                                                                                                   | Script                                   | Artifact output                                                                                                     | Deadline                           |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| 1   | Harness skeleton                                                                                                                                                                       | `eval_ranking_harness.py`                | `load_split_mask()` + `compute_metrics()` OK                                                                        | M1 (7/4)                           |
| 2   | Baselines Group 1–2 (mock labels)                                                                                                                                                      | `run_baselines.py`                       | `baseline_ranking_metrics.csv` (mock)                                                                               | 9/4                                |
| 3   | **Baselines Group 1–2 (real IC)**                                                                                                                                                      | `run_baselines.py`                       | CSV real (Group 1: views/views_day/degree, Group 2: PR/kshell/betweenness)                                          | 12/4                               |
| 4   | Baselines Group 3 (proxies)                                                                                                                                                            | `run_baselines.py`                       | CSV + one-hop/two-hop rows                                                                                          | 15/4                               |
| 5   | **Group 4 — Node2Vec + LR** (`dim=64, walks=20`)                                                                                                                                       | `run_baselines.py`                       | `baseline_ranking_metrics.csv` (thêm rows Group 4)                                                                  | 18/4                               |
| 6   | **Group 4 — MLP raw attr** (`views_log, views/day, life_time`)                                                                                                                         | `run_baselines.py`                       | cập nhật `baseline_ranking_metrics.csv`                                                                             | 18/4                               |
| 7   | **[MUST — v3.1 unconditional] Group 5 — GNN-raw-attr** (SAGE, 5 seeds) — base variant trước khi C2 xác định best arch                                                                  | `run_surrogates.py`                      | `surrogate_ranking_metrics.csv` (mean±std)                                                                          | 22/4                               |
| 8   | **[MUST — v3.1 unconditional] Group 5 — GNN ablation**: graph-only + centrality (dùng best arch từ C2); ✦ [IF TIME] + GNN-full                                                         | `run_surrogates.py`                      | cập nhật `surrogate_ranking_metrics.csv`                                                                            | 22/4                               |
| 9   | Runtime table + Speedup MC vs GNN                                                                                                                                                      | manual/script                            | `outputs/mapr2026_v3_results/runtime_breakdown.csv` hoàn chỉnh                                                      | 22/4                               |
| 10  | ✦ [IF TIME] GNN-random sanity-check (message passing value baseline)                                                                                                                   | `run_surrogates.py`                      | thêm `gnn_random` row trong `surrogate_ranking_metrics.csv`                                                         | 22/4                               |
| 11  | ✦ [IF TIME; nâng thành MUST nếu predictions đủ trước 25/4] Per-group prediction error                                                                                                  | `run_baselines.py`/`run_surrogates.py`   | `outputs/mapr2026_v3_results/per_group_prediction_error.csv`                                                        | 25/4                               |
| 12  | **[NEW v3.1 — C2] Architecture comparison** — 5 archs (GCN/GIN/GAT/APPNP + SAGE baseline) × raw_attr, 5 seeds each; dùng `get_model(arch, ...)` factory; **APPNP là H3 expected best** | `run_surrogates.py` (dùng `get_model()`) | rows `gcn_raw_attr`, `gin_raw_attr`, `gat_raw_attr`, **`appnp_raw_attr`** trong `surrogate_ranking_metrics.csv`     | **19/4** (1 ngày thêm cho APPNP)   |
| 13  | **[NEW v3.1 — C3] Ranking loss** best arch (kết quả C2) + combined α·Huber + (1-α)·pairwise-margin; UPDATE model_name với tên arch thực tế (e.g., `appnp_raw_attr_rankloss`)           | `run_surrogates.py`                      | row `best_arch_raw_attr_rankloss` (tên thực: e.g., `appnp_raw_attr_rankloss`) trong `surrogate_ranking_metrics.csv` | 21/4 (sau C2)                      |
| 14  | **[NEW v3.1 — C4] Bootstrap CI** GNN best (mean preds qua 5 seeds) vs degree (1000 resamplings) → xem 3 outcome interpretations                                                        | `run_surrogates.py` / standalone         | `outputs/mapr2026_v3_results/gnn_vs_degree_bootstrap_ci.json`                                                       | 21/4 (song song C3)                |
| 15  | **[SHOULD DO, phụ thuộc Person 1 row 13] C2-A2** — GCN/GIN/GAT/APPNP/SAGE trên A2 labels (test H2: GCN–A2 alignment hypothesis)                                                        | `run_surrogates.py` (p_rule='symmetric') | rows `gcn_a2_raw_attr`, `gin_a2_raw_attr`, `appnp_a2_raw_attr`, etc. trong `surrogate_ranking_metrics.csv`          | Sau A2 labels + C2-A0 xong (≈22/4) |
| 16  | **[SHOULD DO — chỉ khi Person 1 row 15 pass] C2-I-A + C4-I-A** — **5 archs**: APPNP + **GATv2** (H4!) + GIN + GCN + SAGE × 5 seeds trên I-A labels + Bootstrap CI; dùng `GATv2Surrogate` | `run_surrogates.py` (p_rule='ia')        | `surrogate_ranking_metrics_ia.csv` (model_name: `appnp_raw_attr_ia`, `gatv2_raw_attr_ia`, ...) + `gnn_vs_degree_bootstrap_ci_ia.json` | ≈23/4 (sau Person 1 row 15 xong)   |
| 17  | **✦ [IF TIME — C5] GINE + IC edge features** — upper bound experiment sau khi C2+C3+C4 done; `edge_attr = 1/deg(v)` per edge; dùng `GINESurrogate`; **KHÔNG** đưa vào C2 fair comparison | `run_surrogates.py` + `GINESurrogate`    | rows `gine_ic_a0_raw_attr`, `gine_ic_a2_raw_attr` trong `surrogate_ranking_metrics.csv`                             | ≈25/4 (sau C2/C3/C4 xong)          |

> **Các bước có tính metrics (2–8, 10, 11):** load `split_masks.parquet` → `apply_test_mask()` → `compute_metrics()`. Không tự tạo split.
> **Group 4 vs Group 5:** Node2Vec+LR và MLP vào `baseline_ranking_metrics.csv` (comparable với Group 1–3). GNN variants vào `surrogate_ranking_metrics.csv` (với mean±std vì 5 seeds).
> **BH-FDR:** Chỉ cần nếu chạy nhiều MWU tests (multiple comparisons); nếu không thì report mean±std là đủ.

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

Active handoff version: `person1_day1_20260409_p1_day1_v3i_optionB_lockstep`

1. Dùng đúng 1 version tag handoff cho toàn bộ experiment cycle — không mix artifacts từ các version khác nhau.
2. Không tự re-split data local — chỉ load `data/processed/split_masks.parquet` từ handoff (SHA256: `005de40762f6c75e4df66a53efeaa883d126d52abd5c4af0224d736992362104`).
3. Giữ canonical branch (`classification_labels.parquet`) và consensus branch (`classification_labels_consensus.parquet`) tách biệt — không ghi đè canonical.
4. Binary metrics phải khai báo uncertainty: loại `is_uncertain=1` hoặc `vote_count=1` khi claim strict binary performance; ghi rõ evaluation scope (all nodes vs non-uncertain subset).
5. Regression là PRIMARY objective — dùng `regression_targets.parquet` (`y = log1p(ic_score_mean)`) cho toàn bộ surrogate ranking pipeline.
6. Nếu cần thay đổi artifacts: tạo version tag mới (`freeze_day1_handoff.py --version-tag <new_tag>`) — không overwrite handoff directory đã có.
22