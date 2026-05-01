# MAPR2026 v3.2 — Team 3 người: Kế hoạch coding song song (không bao gồm viết paper)

Mục tiêu của tài liệu này là thiết kế các đầu việc **có thể triển khai song song tối đa** cho team 3 người theo MAPR path mới: **dual-operationalization contrast** giữa `A0` và `HSCC`. Execution focus không còn là `A0 primary + I-A conditional`, mà là:

- `A0` = structural contrast / negative-control track.
- `HSCC` = main graph-aware track.
- `A2` = sensitivity nếu còn thời gian.
- `I-A` = archive / post-MAPR only.

Phạm vi:

- **Chỉ phần thực thi code + tạo artifacts + chạy pipeline**.
- Không bao gồm viết paper/narrative (đã có người khác phụ trách).
- **Có một số “paper-facing handoff checks”** (protocol statements, figures/tables checklist, pre-submission sweep) để đảm bảo defensibility/construct-validity; mục tiêu là **đảm bảo các điều kiện/đoạn bắt buộc có mặt**, không phải viết narrative mới.

**Scope bridge:** Tài liệu này là execution plan cho team 3 người coding. `MAPR2026_Implementation_Plan_v3.md` là strategic master plan (research + narrative + publication framing). Nếu khác biệt ở thao tác thực thi hằng ngày, ưu tiên file này; nếu khác biệt về framing nghiên cứu/paper, ưu tiên master plan. **v3.2 override:** mọi task và artifact trong file này phải được hiểu theo `A0 + HSCC`.

> **Consistency note (v3.2 freeze):** Historical v3.1 terms chỉ nên còn xuất hiện trong các archive notes được gắn nhãn rõ ràng (ví dụ GAT/I-A history). Canonical naming trong handoff/contract của v3.2 dùng regime-suffixed artifacts: `ic_scores_a0.parquet`, `regression_targets_a0.parquet`, `ic_scores_hscc_refined.parquet`, `regression_targets_hscc_refined.parquet`, `gnn_vs_degree_bootstrap_ci_a0.json`, `gnn_vs_baseline_bootstrap_ci_hscc.json`, và `gnn_vs_rankloss_bootstrap_ci_hscc.json` (khi C3 được chạy).
>
> _Lưu ý:_ codebase có thể vẫn nhắc tới legacy filenames như `ic_scores_primary.parquet` / `regression_targets.parquet`. Khi gặp, **treat đó là alias của A0** và **không** tạo “hai bộ artifacts song song” với naming khác nhau. Quan trọng nhất là **một mapping rõ ràng, nhất quán** trong handoff để Person 3 không load nhầm.

---

## Cách đọc file này — Tier System [⚪ REF]

> **Đọc phần này trước.** File plan dùng 4-tier labeling system + một số section **MIXED**. Nhìn vào section header để biết tier ngay:

| Tier                 | Tag                                         | Nghĩa                                               | Khi thực thi                                |
| -------------------- | ------------------------------------------- | --------------------------------------------------- | ------------------------------------------- |
| 🔴 **MAPR-MUST**     | `[🔴 MAPR-MUST]` hoặc body text bình thường | Bắt buộc cho submission defensible — thiếu = reject | Luôn làm; ưu tiên tuyệt đối                 |
| 🔴/🟡 **MIXED**      | `[🔴/🟡 MIXED]`                             | Một section có cả MUST và BOOST tasks               | Làm phần 🔴 trước; phần 🟡 sau nếu còn time |
| 🟡 **BOOST**         | `[🟡 BOOST]`                                | Cải thiện paper; không block submission             | Sau khi xong hết 🔴 trước 30/4              |
| 🔵 **FUTURE[Venue]** | `[🔵 FUTURE:Venue]`                         | Valuable cho venue khác; không kịp MAPR             | Skip MAPR; giữ code; revisit sau 30/4       |
| ⚪ **REF**           | `[⚪ REF]`                                  | Chỉ đọc để align; không phát sinh task              | Đọc 1 lần khi onboard; skip khi đang chạy   |
| **APPENDIX support** | `[APPENDIX support — v3.1 demoted]`         | Secondary/optional; cắt đầu tiên khi tight          | Sau khi Task A + C xong hoàn toàn           |
| **IF PROBLEM**       | `> ⚠ [IF PROBLEM: điều kiện]`               | Phương án thay thế khi vấn đề cụ thể xảy ra         | CHỈ khi trigger condition xảy ra            |

> **Quy tắc vàng (v3.2):** Lần đọc đầu → chỉ đọc 🔴 sections. Xong 🔴 → đọc 🟡. Không kịp deadline → dừng ở `A0 + HSCC` MUST path; `A2` và mọi archive branch giữ cho tương lai.
> Trong Section 3/4/8: Person 1 = artifacts `A0 + HSCC`; Person 2 = proxies + **community as blocking dependency for HSCC**; Person 3 = baseline fairness + regime-specific GNN/bootstraps.
> **⚠ APPENDIX support = cắt TRƯỚC MỌI THỨ KHÁC khi tight deadline.**

---

## 0) Quickstart (để ai cũng chạy được trong 10 phút) [🔴 MAPR-MUST]

Mục tiêu: trước khi chia việc, cả team xác nhận **environment đúng** và **Stage 0 artifacts đã có**.

### 0.1 Environment

- Dùng Python **3.10–3.12** (khuyến nghị: conda env `sna_group9_cbow_py312` theo `environment.yml`).
- Quick check (PowerShell):

```powershell
conda activate sna_group9_cbow_py312
python --version
pip install -r requirements.txt
```

- **PyG ≥ 2.3 bắt buộc** cho APPNP (C2 — H3 architecture). Verify:

```python
import torch_geometric as pyg
from torch_geometric.nn import APPNP
print(f"PyG version: {pyg.__version__}")  # Must be ≥ 2.3
```

> ⚠ Nếu `ImportError: cannot import name 'APPNP'` → `pip install torch_geometric --upgrade`

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

## 1) Nguyên tắc để “song song thật” [⚪ REF]

1. **Chia theo artifact contract**, không chia theo “ý tưởng”. Ai sở hữu artifact nào thì chịu trách nhiệm schema + reproducibility của artifact đó.
2. **Không sửa đè Stage 0–3 đang chạy ổn** trừ khi bắt buộc. Nếu cần đổi logic theo MAPR2026 v3, ưu tiên tạo script mới để tránh phá pipeline cũ.
3. Mọi người đều có thể code/test trước bằng **mock artifacts** (từ SIS/centrality) để không bị chặn bởi IC labels.
4. Có **3 artifact là “gating”** bắt buộc sớm — Person 1 phải tạo trước:
   - `data/processed/graph_csr.npz` → unblock Person 2 và 3
   - `outputs/day1_benchmark/*` → lock compute budget và GNN narrative (M2)
   - `data/processed/split_masks.parquet` → Person 3 mới chạy metrics thật được (M3)

### Scope guard (để không lệch MAPR2026 v3)

- Execution path chính thức gồm **2 label regimes**:
  - `A0`: `p(u,v)=1/degree(v)` — contrast track.
  - `HSCC`: source-velocity + community-boost — main track.
- IC backend: **CSR numpy + joblib (loky)**. Tránh NetworkX BFS trong vòng lặp IC.
- **Views-independence policy (revised):**
  - `A0` + `A2` phải views-independent.
  - `HSCC` được phép dùng `views/life_time/community boost` vì đây là domain-informed alternative operationalization.
  - `I-A` không còn là active MAPR branch.
- Graph dùng **undirected** (`graph_directed: false`) — Twitch Gamers chỉ có mutual-follow edges.
- **Uniform p** — không report.
- **Comparator policy:**
  - `A0` → comparator chính: `degree`, `one_hop`, `two_hop`.
  - `HSCC` → comparator chính: strongest standard non-graph baseline (`LR/MLP` với `life_time`, `views`, và nếu dùng thì `language`).

> 🟡 **[BOOST — A2 sensitivity]** `p(u,v) = 1/√(deg(u)×deg(v))` — structural robustness only. Chạy sau khi `A0 + HSCC` path đã khóa.
>
> 🔵 `I-A`, `II-B`, `A1`, `GATv2-I-A`, `GINE` = archive / future work. Không lên critical path của team trong 9 ngày cuối.

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

## 1b) Bảng hằng số chung (Shared Constants) [🔴 MAPR-MUST]

> ⚠ **Đây là single source of truth cho toàn team.** Mọi thay đổi phải cập nhật bảng này trước, sau đó propagate sang code/config/artifacts. Không được hard-code khác đi ở bất kỳ chỗ nào.

| Hằng số | Giá trị chuẩn | Ý nghĩa |
| ------ | ------------- | ------- |
| `cv_gate` | **0.3** | regression-ready gate |
| `jaccard_gate` | **0.85** | binary-ready gate |
| `top_k_pct` | **0.10** | top-10% threshold |
| `n_sample` | **5,000** | labeled nodes per regime |
| `N_runs` | **200** | MC runs per node |
| `gnn_seeds` | **5** | report mean±std |
| `split_seed` | **42** | shared split |
| `test_frac` | **0.20** | held-out test fraction |
| `active_regimes` | **A0 + HSCC** | current MAPR execution path |
| `archive_regimes` | **I-A, II-B** | not on critical path |
| `community_blocking_for_hscc` | **true** | Person 2 artifact is upstream dependency |
| `bootstrap_a0` | `gnn_vs_degree_bootstrap_ci_a0.json` | A0 comparator = degree |
| `bootstrap_hscc` | `gnn_vs_baseline_bootstrap_ci_hscc.json` | HSCC comparator = strongest flat baseline at rerun time; frozen official rerun comparator = `lr_degree_views_life_time_lang` |
| `submit_deadline` | **30/4** | hard deadline |

---

## 2) Artifact contracts (đóng băng giao diện giữa 3 người) [🔴 MAPR-MUST]

> Các schema dưới đây bám theo `docs/MAPR2026_v3_migration_checklist.md`. Nếu cần đổi tên/format, phải đổi đồng bộ và ghi vào `docs/experiment_registry.md`.

| Artifact (path) | Owner | Consumers | Contract tối thiểu |
| --------------- | ----- | --------- | ------------------ |
| `data/processed/graph_csr.npz` | Person 1 | All | deterministic CSR mapping |
| `data/processed/ic_scores_a0.parquet` | Person 1 | Person 2,3 | sample-only A0 labels |
| `data/processed/regression_targets_a0.parquet` | Person 1 | Person 3 | `node_id, y` for A0 |
| `data/processed/classification_labels_a0.parquet` | Person 1 | Person 3 | optional binary derived from A0 |
| `data/processed/ic_scores_hscc_refined.parquet` | Person 1 | Person 2,3 | sample-only HSCC labels |
| `data/processed/regression_targets_hscc_refined.parquet` | Person 1 | Person 3 | `node_id, y` for HSCC |
| `data/processed/split_masks.parquet` | Person 1 | Person 3 | shared split across regimes |
| `data/processed/community_features.parquet` | Person 2 | Person 2,3 | **blocking for HSCC**; `node_id, community_id, cross_community_edge_fraction` |
| `data/processed/diffusion_proxies.parquet` | Person 2 | Person 3 | full-graph `one_hop_spread, two_hop_spread` |
| `outputs/mapr2026_v3_results/baseline_ranking_metrics_a0_clean.csv` | Person 3 | All | regime=a0; columns: label_regime, model_name, spearman_rho, ndcg, precision, runtime |
| `outputs/mapr2026_v3_results/baseline_ranking_metrics_hscc_clean.csv` | Person 3 | All | regime=hscc; same schema + fairness rows nếu dùng language |
| `outputs/mapr2026_v3_results/surrogate_ranking_metrics_a0_clean.csv` | Person 3 | All | regime=a0; GNN rows incl. gnn_raw_attr, gcn, gin, appnp, best_arch_rankloss |
| `outputs/mapr2026_v3_results/surrogate_ranking_metrics_hscc_clean.csv` | Person 3 | All | regime=hscc; same schema; không có gat_raw_attr (dropped OOM) |
| `outputs/mapr2026_v3_results/runtime_breakdown.csv` | Person 2 + Person 3 | All | runtime rows for both regimes |
| `outputs/mapr2026_v3_results/metric_correlation_matrix.json` | Person 2 | All | at minimum for A0; HSCC addendum if time |
| `outputs/mapr2026_v3_results/gnn_vs_degree_bootstrap_ci_a0.json` | Person 3 | All | C4: A0 comparator = degree |
| `outputs/mapr2026_v3_results/gnn_vs_baseline_bootstrap_ci_hscc.json` | Person 3 | All | C4: HSCC comparator = strongest flat baseline at rerun time; frozen official rerun comparator = `lr_degree_views_life_time_lang` |
| `outputs/mapr2026_v3_results/gnn_vs_rankloss_bootstrap_ci_hscc.json` | Person 3 | All | C3 [🟡 BOOST]: rankloss variant CI; chỉ tạo khi `--include-rankloss-comparison` |

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

- `node_id`: string — phủ toàn bộ **labeled nodes** dùng chung cho `A0` và `HSCC` (canonical set được khóa từ `ic_scores_a0.parquet`)
- `split`: string — giá trị chỉ được là `'train'` hoặc `'test'`
- Rule tạo: `test_frac=0.20`, `stratify=degree_quintile` (pd.qcut q=5), `random_state=42`
- **Consumer rule:** Person 3 load bằng `load_split_mask(PATHS.split_masks)` và filter qua `apply_test_mask()` từ `eval_ranking_harness.py`. Không tạo split mới.

#### `outputs/mapr2026_v3_results/baseline_ranking_metrics_{a0|hscc}_clean.csv`

> File được tạo riêng theo regime: `*_a0_clean.csv` và `*_hscc_clean.csv`. Schema giống nhau cho cả hai. Tên cũ `baseline_ranking_metrics.csv` là legacy — không dùng.

- `label_regime`: string — `a0` hoặc `hscc`
- `model_name`: string
- `spearman_rho`: float
- `ndcg_at_10pct`: float
- `precision_at_10pct`: float
- `runtime_sec`: float — **full-graph inference time** (M0-locked; không tính precompute/training)

#### `outputs/mapr2026_v3_results/surrogate_ranking_metrics_{a0|hscc}_clean.csv` (v3.2 regime-separated)

> File được tạo riêng theo regime: `*_a0_clean.csv` và `*_hscc_clean.csv`. Schema giống nhau cho cả hai. Tên cũ `surrogate_ranking_metrics.csv` là legacy — không dùng.

Schema bắt buộc (mean±std trên 5 training seeds `[42, 123, 456, 789, 1024]`):

- `label_regime`: string — `a0`, `hscc`, hoặc `a2` nếu chạy sensitivity
- `model_name`: string — tên chuẩn cho current MAPR rerun: `gnn_raw_attr`, `gnn_graph_only`, `gnn_centrality`, `gnn_full`, `gcn_raw_attr` (C2), `gin_raw_attr` (C2), `appnp_raw_attr` (C2 — **H3 expected best**), `best_arch_raw_attr_rankloss` (C3). `gat_raw_attr` chỉ còn là legacy/archived candidate; **không mong đợi xuất hiện trong official rerun** vì GAT đã bị drop do OOM và current execution dùng `--skip-gat`.
- `spearman_rho_mean`, `spearman_rho_std`: float
- `ndcg_mean`, `ndcg_std`: float
- `precision_mean`, `precision_std`: float
- `runtime_sec`: float — GNN inference time trên full active graph (không tính training)

#### `outputs/mapr2026_v3_results/runtime_breakdown.csv`

Schema bắt buộc — dùng để tính "Speedup: MC IC vs GNN inference" trong Table runtime của paper:

- `model_name`: string — dùng tên chuẩn (vd. `gnn_raw_attr`, `diffusion_proxies`, `node2vec_lr`, `mc_ic_labeling`, ...)
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

- `ic_scores_a0.parquet`, `regression_targets_a0.parquet`, `ic_scores_hscc_refined.parquet`, và `regression_targets_hscc_refined.parquet` thường là **labeled subset** (do compute budget). Đây là đúng với plan v3.2.
- Proxies/baselines/surrogates có thể dự đoán cho full-graph để đo runtime, nhưng **khi tính metric thì chỉ dùng test mask trên labeled nodes**.
- **Transductive claim lock (B10):** không claim deployment accuracy trên unlabeled nodes từ các metrics này; muốn claim rộng hơn phải có out-of-sample IC audit riêng (gợi ý 500-1000 nodes).

#### Metric definitions (đã lock tại M0 — không thay đổi)

- `spearman_rho`: Spearman correlation giữa `y_true` và `y_pred` trên **test labeled nodes** (sau khi apply test mask).
- `ndcg_at_10pct`: NDCG@k với $k=\lceil 0.10 \times n_{test}\rceil$; relevance lấy theo `y_true` (regression target = `log1p(ic_score_mean)`).
- `precision_at_10pct`: Precision@k với cùng k; “true top-k” định nghĩa theo top-k của `y_true` trên test set.

> **[M0-locked]** k được tính theo `n_test` (số test nodes), không phải tổng active nodes. Không thay đổi định nghĩa này mà không update `docs/m0_decisions.md`.

### Mock artifacts để không phải chờ nhau

- **Trước khi có `ic_scores_a0.parquet`** (giai đoạn M1–M2): dùng `data/processed/sis_table.parquet` (hoặc `pagerank`) làm nhãn tạm (`ic_score_mean ≈ sis_score`), để Person 2/3 viết pipeline và unit-test I/O.
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
#     ↳ canonical v3.2 handoff: ic_scores_a0.parquet, regression_targets_a0.parquet,
#                               classification_labels_a0.parquet, split_masks.parquet
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

## 3) Workstreams song song (3 người) [🔴 MAPR-MUST — Track A + C; Track B = 🔴/🟡 MIXED]

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

### Person 1 — Track A: IC core cho `A0 + HSCC` (CSR + labels + shared split) [🔴 MAPR-MUST]

> 🔴 **[MAPR-MUST]** Track A chỉ còn phục vụ `A0 + HSCC`. `I-A`, `II-B`, `C2-I-A`, `C4-I-A`, và mọi pilot/fallback liên quan **không thuộc execution path MAPR v3.2**.
> Nếu scripts còn nhắc `ic_scores_primary.parquet` / `regression_targets.parquet`, **coi đó là legacy alias của A0**. Không tự phát đổi tên artifacts giữa chừng; thay vào đó, ghi rõ mapping trong handoff (để tránh tình trạng “A0 có 2 filename khác nhau” làm Person 3 load nhầm).

**Mục tiêu:** cung cấp bộ artifacts canonical để cả team có thể chạy baselines, surrogates, và bootstrap theo đúng hai regime của v3.2.

**Dependency map:**

- `A0` branch: sau khi có Stage 0 graph thì Person 1 có thể chạy độc lập.
- `HSCC` branch: cần artifact community từ Person 2 trước khi freeze labels cuối.
- `A2`: chỉ chạy nếu `A0 + HSCC` đã ổn định; không được làm chậm critical path.

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
   | Condition | Cách đọc trong v3.2 |
   |---|---|
   | `ρ < 0.8` | Giữ nguyên `A0 + HSCC` path; dưới `A0`, GNN comparison vẫn có giá trị vì degree/proxy chưa near-ceiling |
   | `0.8 ≤ ρ ≤ 0.9` | Giữ nguyên `A0 + HSCC` path; bổ sung `two_hop` như comparator A0 mạnh hơn khi đọc kết quả |
   | `ρ > 0.9` **and** `Jaccard@10% > 0.8` **and** `NDCG@10% > 0.9` | Không đổi paper sang proxy-only story; chỉ ghi rõ rằng dưới `A0`, analytical baselines là near-optimal và `A0` đóng vai trò contrast regime |
   | `ρ > 0.9` nhưng top-k alignment chưa cao | Giữ contrast framing; nhấn mạnh divergence ở top-k thay vì claim blanket win/loss cho GNN |

4. IC pilot + diagnostics (CV / non-degenerate checks)
   - Output: `outputs/day1_benchmark/ic_pilot_diagnostics.json`
   - **Fields bắt buộc (10 fields + ks_results):** `n_pilot_nodes`, `n_pilot_runs`, `mean_reach`, `median_reach`, `iqr_reach`, `top10_to_median_ratio`, `rank_stability`, `cv_score`, `cv_noise_count` (số nodes có CV > 0.50), `jaccard_stability` (ghi SAU khi chạy 3 MC stability experiments), `ks_results` (dict per feature — xem schema dưới)
   - **Regression-ready gate (primary):** `cv_score > 0.3` → được phép chạy full IC và tiếp tục pipeline với `regression_targets_a0.parquet`.
   - **Binary-ready gate (secondary):** `jaccard_stability >= 0.85` → `classification_labels_a0.parquet` non-provisional. Nếu thấp hơn: binary = provisional (không block nhánh regression).
     > ⚠ **[IF PROBLEM: cv_score < 0.3]** Hai trường hợp — đọc kỹ trước khi dừng pipeline:
     >
     > - **Nếu IC không degenerate** (spearman_mean > 0.65 và reach metrics OK — diagnosis cho thấy noise do binary threshold, không phải IC broken): **KHÔNG dừng pipeline**. Kích hoạt **Option B** (xem block Option B bên dưới) — regression **tiếp tục** với `quality_mode=provisional`, `quality_gate_pass_all=false` ghi trung thực vào manifest. Báo team biết nhưng không block.
     > - **Nếu IC degenerate** (cả 3 điều kiện đồng thời: `median_reach < 2` + `p_reach_gt_1 < 0.20` + `top10_to_median_ratio < 2`): **Dừng pipeline** — báo team ngay; không chạy full IC cho đến khi team có quyết định; xem ⚠ [IF PROBLEM: median_reach...] block bên dưới để biết last-resort fallbacks.
     >
     > **[MUST — narrative only, zero code] Framing note (đọc trước khi implement Option B):** Regression primary KHÔNG phải là fallback do gate fail — đây là formulation đúng về mặt nguyên tắc cho một simulation-derived continuous target. MC simulation tạo ra `ic_score_mean` là continuous quantity; `y = log1p(ic_score_mean)` là regression target tự nhiên. Binary labels là một derived artifact thứ cấp với threshold sensitivity cố hữu. Jaccard instability là _bằng chứng bổ sung_ ủng hộ formulation này, không phải lý do duy nhất để chuyển sang regression. Paper phải trình bày regression primary như là lựa chọn đúng, không phải như là "chúng tôi buộc phải pivot".
     >
     > **Option B — resolution khi gate fire nhưng IC không degenerate** (spearman_mean > 0.65 và reach metrics OK, diagnosis cho thấy noise do binary threshold chứ không phải IC broken): team kích hoạt Option B để không block toàn team:
     >
     > 1. Regression target (`regression_targets_a0.parquet`) = **PRIMARY** — tiếp tục pipeline bình thường.
     > 2. Binary labels (`classification_labels_a0.parquet`) = **provisional/secondary** — phải khai báo uncertainty; nếu cần consensus branch thì đặt tên lại theo v3.2 contract, không quay về generic naming.
     > 3. Freeze handoff package với `quality_mode=provisional` (ghi `quality_gate_pass_all=false` trung thực vào manifest — không fake pass).
     > 4. Áp dụng lockstep rules toàn team (xem Section 8b): cùng 1 version tag, không re-split local.
     > 5. Ghi rõ quyết định Option B vào `docs/day1_decisions.md`.

   > **Note về thứ tự ghi file:** `ic_pilot_diagnostics.json` ghi 2 lần — lần 1 sau pilot run (chưa có `jaccard_stability`), lần 2 sau 3 MC stability experiments (thêm `jaccard_stability`). Script `ic_labels_primary.py` tự update bằng `json.load` → `json.dump`.

   > ✦ **[SHOULD DO — sau khi A0 + HSCC + C2 chính đã xong] Sensitivity S1: Symmetric IC (A2)**
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

   > **Archive note (v3.2 docs-first):** `I-A`, `II-B`, và mọi nhánh `C2-I-A`/`C4-I-A` chỉ còn là historical reference. Không chạy pilot, không tạo label set, không dùng để quyết định scope hiện tại. Nếu team muốn giữ record khoa học thì chỉ ghi ngắn trong `docs/experiment_registry.md` hoặc appendix note, nhưng **không để xuất hiện như active deliverable trong Track A**.

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

5. IC A0 labels + label stability (Jaccard top-decile across 3 MC seeds)
   - `ic_scores_a0.parquet`
   - `regression_targets_a0.parquet`, `classification_labels_a0.parquet`
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

6. **[M0-locked] Split mask** — tạo ngay sau khi có `ic_scores_a0.parquet`
   - `data/processed/split_masks.parquet`
   - Rule cứng: `test_frac=0.20`, `stratify=degree_quintile` (q=5), `seed=42`
   - Dùng flag `--test-frac 0.20 --seed 42` trong `ic_labels_primary.py`
   - Ghi số `n_train / n_test` vào `docs/day1_decisions.md` để team biết

7. **[M3] Views/IC alignment check** — chạy ngay sau khi có `ic_scores_a0.parquet` (final run)
   ```python
   from scipy.stats import spearmanr
   df = pd.read_parquet(PATHS.ic_scores_a0)
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
worker_seed       : 42 + node_index          (A0 production run)
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
- [ ] `ic_scores_a0.parquet` tồn tại với `n_runs` locked, coverage = n_sample nodes.
- [ ] `regression_targets_hscc_refined.parquet` tồn tại sau khi community artifact đã khóa và dùng đúng shared labeled-node set của `A0`.
- [ ] `split_masks.parquet` tồn tại, schema đúng, coverage = 100% labeled nodes, test_frac ≈ 0.20.
- [ ] **[v3.1 MUST — C1, deadline 16/4]** `degree_controlled_ic_variance.json` tồn tại với đủ 6 fields. cv_within_band có giá trị, không null.

> ✦ **[IF TIME] Soft DoD** — thêm vào khi xong MUST trước deadline:
>
> - [ ] Bootstrap 95% CI: `ic_ci_lower`, `ic_ci_upper` trong `ic_scores_a0.parquet`
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
# run_c1_degree_variance_test("data/processed/ic_scores_a0.parquet",
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

### Person 2 — Track B: Structure + diffusion proxies (community + baselines + correlation matrix) [🔴/🟡 MIXED]

> 🔴 **[MAPR-MUST]** Diffusion proxies + **community artifact cho HSCC** + correlation matrix là baseline-completeness items.
> `community_features.parquet` không còn là nice-to-have thuần túy; nó là upstream dependency cho HSCC interpretation và fairness logic.
> Deliverables marked **[APPENDIX support]** = cắt đầu tiên nếu tight deadline.

**Mục tiêu:** Community detection cho HSCC + diffusion proxies cho A0 + correlation summaries để contextualize baselines.

**Có thể làm trước khi IC labels xong** bằng mock nhãn (SIS/pagerank) để hoàn thiện pipeline.

**Deliverables (theo thứ tự dependency):**

1. 🔴 **[MAPR-MUST trong v3.2] Community detection + cross-community features**
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

2. 🔴 **[MAPR-MUST] Diffusion proxies (Group 3)** — **scope: FULL active graph** (M0-locked)
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

3. 🔴 **[MAPR-MUST] Metric correlation matrix / summary**

- **Tại sao MUST:** Reviewer sẽ hỏi "how do the IC scores relate to simpler metrics?" Tối thiểu phải có summary cho `A0`; nếu đủ thời gian thì thêm HSCC summary riêng.
- **Mục tiêu:** Trả lời RQ2b — khi nào degree/pagerank/views fail làm proxy cho IC? Provide số liệu định lượng cho Table trong Section 4.3 paper.
- **Input tối thiểu:** join `ic_scores_a0.parquet` + `node_attributes.parquet` + `diffusion_proxies.parquet` + `centrality_table.parquet`
- **HSCC add-on nếu kịp:** join thêm `ic_scores_hscc_refined.parquet` để tạo summary phụ cho flat baselines
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
- **Timing:** Chạy sau khi có `diffusion_proxies.parquet` và `ic_scores_a0.parquet`; HSCC add-on chỉ làm sau khi HSCC artifacts đã khóa.

**DoD cho Track B — MUST (sign-off bắt buộc):**

**DoD Track B — MUST (không cắt):**

- [ ] Proxies (one-hop + two-hop) trên FULL active graph, missing = 0; `runtime_breakdown.csv` có `inference_sec_full_graph`.
- [ ] `metric_correlation_matrix.json` tồn tại với 8×8 `rho_matrix` và `p_matrix_corrected` (global matrix — 8 metrics: ic_score_mean, views, degree, pagerank, kshell, betweenness_approx, one_hop_spread, two_hop_spread — bắt buộc). `rho_by_degree_quintile` là **[✦ IF TIME]** — sign-off không phụ thuộc vào phần này.

> ✦ **[IF TIME] Soft DoD:**
>
> - [ ] `data/processed/community_features.parquet` (file riêng, KHÔNG ghi vào `node_attributes.parquet`), phủ 100% active nodes; có `node_id`, `community_id`, `cross_community_edge_fraction`.
> - [ ] Louvain resolution sensitivity: `louvain_resolution_sensitivity.json` cho {0.5, 1.0, 2.0}

---

### Person 3 — Track C: Surrogate learning + evaluation harness (metrics/runtimes) [🔴 MAPR-MUST — C1+C2+C4]

> 🔴 **[MAPR-MUST]** baseline fairness, architecture comparison, và bootstrap theo đúng comparator của từng regime là blocking.
> C3 Ranking Loss = 🟡 BOOST (làm sau khi C2 xong nếu còn thời gian).
> C5 GINE = 🔵 FUTURE:TKDE/WWW2027 (không làm cho MAPR).

**Mục tiêu:** Task C (surrogate learning) và quan trọng nhất là **evaluation harness** cho `A0` và `HSCC`.

**Có thể làm trước khi IC labels xong** bằng mock nhãn, vì cần build sớm:

- loader + **shared split mask** (load từ `split_masks.parquet`, KHÔNG tự tạo split)
- baseline runner
- logging runtime (full-graph inference only)

**Deliverables:**

1. Evaluation harness (model-agnostic, regime-aware)
   - Input: `regression_targets_a0.parquet` hoặc `regression_targets_hscc_refined.parquet` + **`split_masks.parquet`** (M0-locked)
   - Cách dùng: `load_split_mask()` → `apply_test_mask()` → `compute_metrics()`
   - Output: `outputs/mapr2026_v3_results/baseline_ranking_metrics_a0_clean.csv` hoặc `outputs/mapr2026_v3_results/baseline_ranking_metrics_hscc_clean.csv` (tùy regime)
   - Bắt buộc phân biệt `label_regime` = `a0` / `hscc`
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
   | `node2vec_lr` | 4 | Node2Vec + LR |
   | `mlp_raw_attr` | 4 | MLP raw attributes |
   | `gnn_raw_attr` | 5 | GraphSAGE raw-attr / `sage_raw_attr` (backward compat — → surrogate CSV) |
   | `gnn_graph_only` | 5 | GraphSAGE graph-only (→ surrogate CSV) |
   | `gnn_centrality` | 5 | GraphSAGE centrality (→ surrogate CSV) |
   | `gnn_full` | 5 | GraphSAGE full features (→ surrogate CSV) |
   | `gcn_raw_attr` | 5 | GCN raw-attr ← **NEW v3.1 (C2)** |
   | `gin_raw_attr` | 5 | GIN raw-attr ← **NEW v3.1 (C2)** |
   | `gat_raw_attr` | 5 | GAT raw-attr — **legacy C2 candidate; dropped in current official rerun due to OOM (`--skip-gat`)** |
   | `appnp_raw_attr` | 5 | APPNP raw-attr ← **NEW v3.1 (C2)** |
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

3. **Group 4 — Shallow Embedding Baselines** (v3 Section 7 Group 4 — ghi vào `baseline_ranking_metrics_{a0|hscc}_clean.csv`, KHÔNG phải surrogate CSV):
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

    - `runtime_sec` trong `baseline_ranking_metrics_{a0|hscc}_clean.csv` = **inference-only** (`predict`) trên full active graph.
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

   - **Lưu ý naming:** Master plan v3 gọi đây là "Group 4 Baselines" (không phải "surrogates"). Kết quả phải vào `baseline_ranking_metrics_a0_clean.csv` hoặc `baseline_ranking_metrics_hscc_clean.csv` cùng với Group 1–3 của đúng regime để so sánh đầy đủ trong Table 2 của paper.

4. **[MUST — v3.1 unconditional] Group 5 — GNN — Architecture comparison + ablation variants** (v3.1 Section 9.1):

> _v3.1: GNN architecture comparison là unconditionally MUST theo professor's framing. `gnn_branch_viable` gate từ M2 không còn áp dụng cho architecture comparison (C2) và bootstrap CI (C4)._

**[NEW v3.1 — MUST] Architecture Comparison (C2):**
Chạy với `raw_attr` features, 5 seeds mỗi arch — **4 active architectures total** trong current official rerun (SAGE + GCN + GIN + **APPNP**). `GAT` được giữ lại như historical/archived candidate trong docs nhưng **không còn thuộc active execution path** vì official rerun dùng `--skip-gat` sau quyết định drop do OOM.

> **⚠ Naming canonical rule:** SAGE raw-attr baseline **phải được ghi vào surrogate CSV với tên `gnn_raw_attr`** (không phải `sage_raw_attr`) để backward compatibility với existing artifacts và consumer scripts. `sage_raw_attr` chỉ là alias giải thích trong table này; **KHÔNG ghi tên `sage_raw_attr` vào file CSV**. Current active C2 arch names: `gcn_raw_attr`, `gin_raw_attr`, `appnp_raw_attr`; `gat_raw_attr` chỉ là legacy name trong archive notes, không phải expectation của official rerun.

| Architecture      | **CSV model_name (canonical)**                                                  | Priority             | Inductive bias hypothesis                                                                                                |
| ----------------- | ------------------------------------------------------------------------------- | -------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| GraphSAGE (đã có) | **`gnn_raw_attr`** ← tên CSV chuẩn (alias: sage_raw_attr — chỉ dùng trong docs) | ✅ Done              | Mean agg. — baseline                                                                                                     |
| **GCN**           | `gcn_raw_attr`                                                                  | **MUST (C2)**        | **H2: `D^{-1/2}AD^{-1/2}` ≈ A2 symmetric diffusion — expected better under A2 labels (nếu chạy sensitivity S1)**         |
| GIN               | `gin_raw_attr`                                                                  | **MUST (C2)**        | Sum agg. preserves multi-hop counts (WL-equivalent expressiveness); reference for non-degree-weighted IC dynamics        |
| **GAT**           | `gat_raw_attr`                                                                  | Archive / dropped current rerun | Historical H1 candidate; official MAPR rerun dùng `--skip-gat` vì GAT OOM ở `hidden_channels=128`.                         |
| **🆕 APPNP**      | `appnp_raw_attr`                                                                | **MUST (C2) — H3 ★** | **H3: K-step PPR propagation + teleport/restart (structural analogy/inductive bias) — STRONGEST theoretical motivation** |

> **Ba inductive bias hypotheses — pre-registered trước C2 (để report theo framing đúng):**
>
> **v3.2 execution note:** architecture matrix dưới đây là shortlist cho **cả hai active regimes**. Cách gọi `C2-A0` được giữ lại như historical shorthand vì A0 là nơi so sánh architecture bắt đầu trước, nhưng cùng shortlist này phải được reuse cho `HSCC` sau khi baseline fairness hoàn tất. Không đọc section này như thể C2 chỉ tồn tại cho `A0`.
>
> - **H1 (GAT–A0) — [⚪ ARCHIVED: dropped OOM]:** Historical hypothesis from v3.1. Current official rerun KHÔNG test GAT; dùng `--skip-gat`, nên H1 chỉ còn là archive note.
> - **H2 (GCN–A2):** Nếu chạy Sensitivity S1 (A2 labels), GCN expected to improve vì `D^{-1/2}AD^{-1/2}` ≈ A2. _(testable, phụ thuộc S1 có chạy không)_
> - **H3 (APPNP — IC cascade analogy):** APPNP thực hiện K-step Personalized PageRank: `x^(k) = (1-α)·Â·x^(k-1) + α·x^(0)`. Với `K=10, alpha=0.15`: α là teleport/restart weight (tái-inject `x^(0)` mỗi bước; không diễn giải như xác suất IC “dừng”). Đây là **structural analogy/inductive bias** cho target diffusion-like — hypothesized best arch. _(Klicpera et al., ICLR 2019)_
>
> Cả ba hypotheses đều có **prepared narratives cho mọi outcome**. Không claim kết quả trước khi chạy C2.
>
> **Tie-break (nếu diff < 0.001):** APPNP > GIN > GCN > SAGE (**pre-registered**; APPNP ưu tiên vì H3 theory; GAT dropped OOM nên không tham gia tie-break).

> **Context:** Xem bảng real numbers trong **ablation story** bên dưới để hiểu structural constraint của A0 (tại sao GNN khó beat degree, H3 rationale, và outcome interpretations).

**[NEW v3.1 — 🟡 BOOST] Ranking Loss (C3):**
Sau khi C2 xong → train best arch với combined α·Huber + (1-α)·pairwise-margin-loss.
CSV name: `best_arch_raw_attr_rankloss`

**[v3.2 MUST] Bootstrap CI theo regime:**
- `A0`: `bootstrap_spearman_ci(y_true_a0, gnn_best_preds_a0, degree_preds)` → `gnn_vs_degree_bootstrap_ci_a0.json`
- `HSCC`: `bootstrap_spearman_ci(y_true_hscc, gnn_best_preds_hscc, strongest_flat_baseline_preds)` → `gnn_vs_baseline_bootstrap_ci_hscc.json`

> **⚠ C4 Protocol spec (bắt buộc — để tránh lệch triển khai):**
>
> | Tham số                                          | Giá trị locked                                                                                                                                                                            |
> | ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | --------------------------------------------------------------------------------- |
> | **Metric được CI**                               | **Spearman ρ (primary) + NDCG@10% (secondary)** — compute cả hai trong cùng 1 resampling loop, zero extra cost; Spearman và NDCG có thể diverge (observed: gnn_graph_only 0.470 vs 0.835) |
> | **Đơn vị resample**                              | Nodes trong **test set** (resample with replacement, size = n_test; không resample training set)                                                                                          |
> | **Δ definition**                                 | `Δ_spearman = Spearman(GNN_best) − Spearman(degree)` và `Δ_ndcg = NDCG@10%(GNN_best) − NDCG@10%(degree)` trên **cùng test set**                                                           |
> | **"GNN_best"**                                   | Architecture có **mean Spearman cao nhất** qua 5 seeds từ C2 (nếu C2 chưa xong: dùng SAGE); predictions = mean predictions qua 5 seeds                                                    |
> | **"degree"**                                     | `rank(degree)` trên toàn active graph, đã filter về test nodes (cùng y_true vector)                                                                                                       |
> | **n_bootstrap**                                  | 1,000                                                                                                                                                                                     |
> | **seed**                                         | 42                                                                                                                                                                                        |
> | **CI**                                           | 95% → percentile(2.5) và percentile(97.5)                                                                                                                                                 |
> | **Practical equivalence bound (pre-registered)** | `                                                                                                                                                                                         | Δ_spearman | ≤ 0.02` = practically equivalent (SESOI). Pre-register **trước khi xem kết quả**. |
>
> **Diễn giải output (quyết định paper claim) — 4-tier:**
>
> | CI outcome                                         | Interpretation             | Paper claim                                                                                                |
> | -------------------------------------------------- | -------------------------- | ---------------------------------------------------------------------------------------------------------- |
> | `ci_95_lower > 0`                                  | GNN significantly better   | "GNN surpasses degree under IC-A0"                                                                         |
> | CI chứa 0 **và** toàn bộ CI trong `[-0.02, +0.02]` | **Practically equivalent** | "GNN achieves statistically equivalent Spearman to degree while requiring no precomputed graph statistics" |
> | CI chứa 0 **nhưng** rộng hơn `[-0.02, +0.02]`      | No clear superiority       | "No significant difference; GNN provides learnable alternative with +0.099 over MLP"                       |
> | `ci_95_upper < 0`                                  | GNN significantly worse    | "GNN competitive; focus on +0.099 without-structural-summaries story vs MLP"                               |

---

**Ablation variants (per best architecture, hoặc SAGE nếu C2 chưa xong):**

| Variant          | Features (in_dim)                                        | Role                                                                    |
| ---------------- | -------------------------------------------------------- | ----------------------------------------------------------------------- |
| **GNN-raw-attr** | `views_log_norm, views_per_day_norm, life_time_norm` (3) | **MUST — Primary proposed**                                             |
| GNN-graph-only   | `degree_norm` only (1)                                   | **MUST** — Ablation: topology without attributes                        |
| GNN-centrality   | `degree_norm, pagerank_norm, kshell_norm` (3)            | **MUST** — Ablation: hand-crafted features                              |
| GNN-full         | all 6 features (normalized)                              | ✦ [IF TIME] — supplementary upper bound (có thể cắt nếu tight timeline) |
| GNN-random       | random/constant node features (1)                        | ✦ [IF TIME] — sanity-check message passing value                        |

> ✦ **[IF TIME]** `GNN-random` — không block deadline; chỉ chạy sau khi xong toàn bộ MUST GNN variants. Nếu chạy: ghi vào `surrogate_ranking_metrics_{a0|hscc}_clean.csv` với `model_name=gnn_random` cho đúng regime.

> **Feature normalization bắt buộc**: tất cả features phải normalize trước khi vào GNN (min-max hoặc z-score). Column names trong experiment.yaml là `*_norm`. Không dùng raw values trực tiếp.

**Config chuẩn cho 4 active architectures (SAGE, GCN, GIN, APPNP) — locked để fair comparison:** _(GAT dropped OOM; dùng `--skip-gat`)_

| Hyperparameter | Giá trị (conv-based archs)                   | Ghi chú APPNP                                                                |
| -------------- | -------------------------------------------- | ---------------------------------------------------------------------------- |
| `hidden_dim`   | 128                                          | Không thay đổi per arch (APPNP dùng hidden_dim cho MLP embedding)            |
| `n_layers`     | 2                                            | Conv-based only; APPNP không dùng n_layers                                   |
| `dropout`      | 0.3                                          | Không thay đổi per arch                                                      |
| `gat_heads`    | 4                                            | *(archived — GAT dropped OOM; param giữ lại trong run_surrogates.py nhưng không invoke khi `--skip-gat`)* |
| `appnp_K`      | **10**                                       | **APPNP only** — cascade depth (propagation steps)                           |
| `appnp_alpha`  | **0.15**                                     | **APPNP only** — teleport/restart weight (starting point; controls locality) |
| Loss           | Huber (`delta=1.0`)                          | Không dùng early stopping — **giống nhau cho tất cả 4 active archs**         |
| `lr`           | 0.001 (Adam)                                 | Không thay đổi per arch                                                      |
| `epochs`       | 200 (cố định)                                | **Không early stopping** — cố định để fair comparison                        |
| Training seeds | `[42, 123, 456, 789, 1024]`                  | 5 seeds mỗi arch                                                             |
| Split          | `split_masks.parquet` (M0-locked)            | **Cùng split cho mọi arch**                                                  |
| Features (C2)  | `raw_attr` (views_log, views/day, life_time) | C2 chỉ so sánh trên raw_attr — **4 active archs** × 1 feature set            |

**Best arch selection criterion (cho C3, C4, ablation):**

> **Best arch = architecture có `spearman_rho_mean` cao nhất** qua 5 seeds trong `surrogate_ranking_metrics_{regime}_clean.csv`. Nếu tie (diff < 0.001): ưu tiên theo thứ tự **APPNP > GIN > GCN > SAGE** (pre-registered — GAT dropped OOM; không được chọn làm best arch). Ghi `gnn_primary_arch` vào `docs/experiment_registry.md` ngay sau khi C2 xong — C3 và C4 depend on this value.

Architectures: `sage` (SAGEConv, mean) | `gcn` (GCNConv) | `gin` (GINConv+MLP) | ~~`gat` (GATConv, heads=4)~~ **DROPPED OOM** | **`appnp`** (K=10, alpha=0.15, **H3 expected best**). **Official run: 4 active archs** (`--skip-gat`).
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


# C2: 4 active architectures — GAT dropped OOM (A100-40GB, h=128); use --skip-gat
ARCHITECTURES = ['sage', 'gcn', 'gin', 'appnp']
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
> | Variant              | Edge features        | Node features | CSV model_name        |
> | -------------------- | -------------------- | ------------- | --------------------- |
> | GINE-IC-A0 (primary) | `1/deg(v)`           | `raw_attr`    | `gine_ic_a0_raw_attr` |
> | GINE-IC-A2           | `1/√(deg(u)×deg(v))` | `raw_attr`    | `gine_ic_a2_raw_attr` |
>
> **Paper framing C5:** "As supplemental upper-bound analysis, we encode IC propagation probabilities directly as edge features (GINE; Hu et al., 2019). Comparing GINE-IC-A0 against GNN-raw-attr (no edge features) quantifies the information gain from explicit IC mechanism encoding."

---

**📋 Architecture Evaluation Log — Tổng kết các model GNN đã đánh giá**

> **Mục đích:** Khi reviewer hỏi "why not try X?", team có documented rationale sẵn. Cũng là checklist để không waste time implement architectures không phù hợp với project này.
> **v3.2 clarification:** bảng này không định nghĩa scope chỉ cho `A0`. Hãy hiểu đây là shortlist architecture cho active MAPR path; `A0` là regime chạy trước để lock C2, còn `HSCC` reuse shortlist sau khi flat-baseline fairness đã xong.

| Architecture            | Verdict                       | Dùng ở đâu                 | Lý do chi tiết                                                                                                                                               |
| ----------------------- | ----------------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **SAGE** (mean agg.)    | ✅ **Trong C2-A0** (baseline) | C2-A0 row `gnn_raw_attr`   | Baseline reference. Mean agg. bị smoothing → 0.470 (graph_only), 0.534 (raw_attr).                                                                           |
| **GCN**                 | ✅ **MUST C2-A0** (H2)        | C2-A0 `gcn_raw_attr`       | H2: D^{-1/2}AD^{-1/2} ≈ A2 symmetric IC. Test cả C2-A0 và C2-A2.                                                                                             |
| **GIN**                 | ✅ **MUST C2-A0**             | C2-A0 `gin_raw_attr`       | Sum agg. — highest WL expressiveness; preserves hop counts.                                                                                                  |
| **GAT v1**              | Archive / dropped current rerun | Legacy `gat_raw_attr` note | Historical H1 candidate. Official MAPR rerun không yêu cầu row `gat_raw_attr`; current execution dùng `--skip-gat` do OOM.                                   |
| **APPNP**               | ✅ **MUST C2-A0** (H3)        | C2-A0 `appnp_raw_attr`     | H3: K-step PPR ≈ IC cascade. Expected best arch.                                                                                                             |
| **GATv2**               | 🔵 **Archive only**           | Không thuộc MAPR path      | Giữ như historical note cho I-A branch cũ. Không implement trong current `A0 + HSCC` execution cycle.                                                          |
| **GINE + IC edge feat** | ✅ **C5 [IF TIME]**           | `gine_ic_a0_raw_attr`      | Strongest alignment: explicit IC prob làm edge feature. NOT feature-agnostic. Upper bound experiment.                                                        |
| **GCNII**               | ❌ **Skip C2**                | —                          | Advantage chỉ tại L=16–64. Tại `n_layers=2` (C2 locked) ≈ GCN + residual. Cần separate L=16 experiment → phá fair comparison.                                |
| **HGT**                 | ❌ **Loại hoàn toàn**         | —                          | Designed cho **heterogeneous graphs** (many node/edge types). Twitch = **homogeneous** (1 type). Type matrices collapse → complex GAT variant, không có lợi. |
| **GraphGPS**            | ❌ **Loại — scale**           | —                          | MPNN + Transformer O(N²) với N=168k = 28 tỷ pairs. LapPE eigendecomposition 168k×168k tốn 30–60 phút. Overkill cho 3-feature node regression.                |

> **Quick rule cho future architectures:**
>
> - Graph homogeneous? → Loại HGT, DGI heterogeneous variants
> - Scale O(N²)? → Loại nếu không có efficient approx + benchmark trước
> - Advantage chỉ tại L >> 2? → Không đưa vào C2, test riêng
> - Cần edge features ngoài structural? → Verify có data trước khi implement

---

**C3 — Ranking Loss Experiment (sau khi C2 xác định best arch):**

```python
import torch.nn.functional as F

# Standard pairwise loss (baseline — random pairs)
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

# ⭐ Top-k focused loss (RECOMMENDED — better NDCG gradient signal)
# Lý do: IC heavy-tailed → random pairs mostly (low,low) → weak gradient for NDCG@10%.
# Top-k sampling ensures gradient focuses on nodes that matter for evaluation.
def pairwise_ranking_loss_topk_focused(pred, target, margin=0.1, n_pairs=512, top_frac=0.2):
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

def combined_loss(pred, target, alpha=0.5, margin=0.1):
    """Combined Huber + top-k focused ranking loss. alpha=0.5 default."""
    huber = F.huber_loss(pred, target, delta=1.0)
    rank  = pairwise_ranking_loss_topk_focused(pred, target, margin=margin)  # ← top-k version
    return alpha * huber + (1 - alpha) * rank

# CSV name: best_arch_raw_attr_rankloss (e.g., appnp_raw_attr_rankloss nếu APPNP wins C2)
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

# Output (A0): outputs/mapr2026_v3_results/gnn_vs_degree_bootstrap_ci_a0.json
# Output (HSCC): outputs/mapr2026_v3_results/gnn_vs_baseline_bootstrap_ci_hscc.json
# {
#   "n_bootstrap": 1000,
#   "comparator_a": "gnn_best_architecture",
#   "comparator_b": "degree",
#   "equivalence_bound": 0.02,           # pre-registered SESOI
#   "spearman": {
#     "delta_mean": <float>,
#     "ci_95_lower": <float>,
#     "ci_95_upper": <float>,
#     "interpretation": "practically_equivalent | no_clear_superiority | gnn_better | degree_better"
#   },
#   "ndcg_at_10pct": {                   # zero extra cost — same resample loop
#     "delta_mean": <float>,
#     "ci_95_lower": <float>,
#     "ci_95_upper": <float>,
#     "interpretation": "practically_equivalent | no_clear_superiority | gnn_better | degree_better"
#   }
# }
#           CI entirely negative → focus on +0.099 message passing story
# Thời gian: ~10 phút (resample existing predictions, không cần retrain)
```

Ablation story:

> **📐 Paper section budget (revised per reviewer feedback — xem Implementation Plan Section 14):**
> | Section | Content | Budget |
> |---|---|---|
> | Section 3 (MC-IC as Metric) | Discriminativeness + C1 enhanced 2-tier variance test + Stability + Regression justification | **1.0 trang** (tăng từ 0.75) |
> | Section 4 (GNN Surrogate) | Architecture comparison (ceiling finding) + Ablation + Ranking loss + Runtime | **2.0 trang** (giảm từ 2.5) |
> | Section 5 (Discussion) | Ceiling analysis + When GNN adds value + Construct validity | 0.5 trang |
> Lý do: Section 3 (stability + structural analysis) là strongest original finding → deserves more space. Section 4 framed như "empirical ceiling characterization" không phải architecture search.

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

- **Architecture comparison (C2 primary — A0 labels):** **4 active archs** (SAGE / GCN / GIN / **APPNP**; GAT dropped OOM — `--skip-gat`) trên `raw_attr` → _which message passing best captures IC's multi-hop dynamics?_
  - **H3 (APPNP — STRONGEST, expected best):** APPNP **decouples feature transformation from propagation** → K=10 PPR steps `x^(k) = (1-α)·Â·x^(k-1) + α·x^(0)` without oversmoothing (`K=10, alpha=0.15` là starting point). IC là multi-hop process (two_hop 0.804 > one_hop 0.688) → APPNP's deeper receptive field expected to capture multi-hop composition better than SAGE mean. **Framing trong paper:** "plausible deeper-propagation inductive bias for IC reach" — không claim APPNP mimics IC mechanics (IC stochastic; APPNP deterministic).
  - **H1 (GAT–A0) — [⚪ ARCHIVED: GAT dropped OOM]:** _(Không testable trong current MAPR rerun — xem Section 9.1 IP file cho historical note)_
  - **H2 (GCN–A2):** _(hypothesis — chỉ testable nếu Sensitivity S1 chạy được)_ GCN's `D^{-1/2}AD^{-1/2}` ≈ A2 symmetric IC rule
  - **GIN:** sum aggregation → có thể capture two-hop count tốt hơn SAGE mean (vì sum preserves hop counts, không smooth out)
  - Cả 4 active arch hypotheses đều có prepared narratives; GAT chỉ còn là archived note (xem Section 4.1b của Implementation Plan)
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
   - **5 seeds:** `[42, 123, 456, 789, 1024]` → report `mean ± std` cho mỗi metric trong `surrogate_ranking_metrics_{a0|hscc}_clean.csv`
   - **Lưu ý về BH-FDR:** Chỉ áp dụng nếu chạy nhiều MWU tests (multiple comparisons). Trong scope bình thường thì report mean±std là đủ.

6. Runtime table (v3.1 Section 9.3):

   | Component                               | Metric          | Notes                                                                                               |
   | --------------------------------------- | --------------- | --------------------------------------------------------------------------------------------------- |
   | Feature precompute (degree, PR, kshell) | time            | Centrality baselines only                                                                           |
   | MC IC labeling (n_sample × N_runs)      | time            | One-time cost — từ Person 1                                                                         |
   | GNN training (5 seeds)                  | time            | With GPU                                                                                            |
   | **GNN inference (168,114 nodes)**       | **runtime_sec** | Full active graph                                                                                   |
   | Node2Vec training                       | time            |                                                                                                     |
   | Speedup: MC IC vs GNN inference         | **~5,590×**     | Headline uses 480.3s / 0.086s from `hscc,gnn_raw_attr` in `runtime_breakdown.csv`; round to ~5,500× in paper prose |

   `runtime_sec` trong CSV = **inference only** (không tính load + precompute).

   > **Operational definition (~5,590×, frozen rerun):**
   >
   > - **480.3s** = MC-IC labeling 5,000 nodes × 200 runs (one-time training label generation, joblib parallelism).
   > - **0.086s** = GNN inference forward-pass trên toàn bộ 168,114 active nodes (headline row = `hscc,gnn_raw_attr` trong `runtime_breakdown.csv`, sau training xong).
   > - **Headline ratio:** 480.3 / 0.086 ≈ 5,590×. Trong paper prose, round về **~5,500×** hoặc "over 5,000×" để tránh false precision.
   > - **Framing an toàn trong paper:** "Once trained, the GNN surrogate provides full-graph inference in approximately 0.086 seconds, compared with 480.3 seconds for a single MC-IC labeling pass used for training." Không dùng lại claim cũ `0.067s / 7,169×`.

**Runtime rule (để so sánh fair):** log riêng 3 phần (precompute / train / inference). Trong `baseline_ranking_metrics_{a0|hscc}_clean.csv` để `runtime_sec` là inference time trên full active nodes, và ghi chi tiết breakdown ở file phụ `outputs/mapr2026_v3_results/runtime_breakdown.csv` (contract bắt buộc trong M0).

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
    ic_scores_a0      = "data/processed/ic_scores_a0.parquet"
    regression_tgts_a0= "data/processed/regression_targets_a0.parquet"
    ic_scores_hscc    = "data/processed/ic_scores_hscc_refined.parquet"
    regression_tgts_hscc = "data/processed/regression_targets_hscc_refined.parquet"
    split_masks       = "data/processed/split_masks.parquet"
    diffusion_proxies = "data/processed/diffusion_proxies.parquet"
    # ─── Outputs ─────────────────────────────────────────────────────────────
    lcc_report        = "outputs/stage0_data_quality/lcc_report.json"
    day1_dir              = "outputs/day1_benchmark"
    ic_runtime_benchmark  = "outputs/day1_benchmark/ic_runtime_benchmark.json"
    one_hop_correlation   = "outputs/day1_benchmark/one_hop_correlation.json"
    ic_pilot_diagnostics  = "outputs/day1_benchmark/ic_pilot_diagnostics.json"
    results_dir           = "outputs/mapr2026_v3_results"
    # Regime-split canonical names (dùng trong production):
    baseline_csv_a0   = "outputs/mapr2026_v3_results/baseline_ranking_metrics_a0_clean.csv"
    baseline_csv_hscc = "outputs/mapr2026_v3_results/baseline_ranking_metrics_hscc_clean.csv"
    surrogate_csv_a0  = "outputs/mapr2026_v3_results/surrogate_ranking_metrics_a0_clean.csv"
    surrogate_csv_hscc= "outputs/mapr2026_v3_results/surrogate_ranking_metrics_hscc_clean.csv"
    # Legacy note: trước v3.2 file này từng dùng 2 CSV generic; official rerun hiện tại dùng 4 file `_clean` tách theo regime như trên.
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
- M3: Sau khi Person 1 cung cấp IC labels thật: thay mock y bằng `regression_targets_a0.parquet` hoặc `regression_targets_hscc_refined.parquet`, chạy lại toàn bộ theo đúng regime.
- `run_baselines.py` real mode: load targets đúng regime (`regression_targets_a0.parquet` hoặc `regression_targets_hscc_refined.parquet`) → filter test mask → `y_pred` cho baseline rows tương ứng → `compute_metrics()`.
- `run_surrogates.py`: train 5 seeds (`training_seeds = [42, 123, 456, 789, 1024]`), report mean±std.

**Gợi ý phân tách file để giảm conflict:**

- Person 3 tập trung `src/mapr2026_v3/eval_ranking_harness.py`, `run_baselines.py`, `run_surrogates.py` và `src/ml/*`.
- Không chạm vào `src/simulation/*` hay `src/mapr2026_v3/ic_labels_primary.py`.

**DoD cho Track C:**

- `load_split_mask()` + `apply_test_mask()` chạy không lỗi với mock artifacts (M1).
- `baseline_ranking_metrics_a0_clean.csv` + `baseline_ranking_metrics_hscc_clean.csv` có đủ rows theo **regime** (M4): `A0` có Group 1-3 + shallow baselines; `HSCC` có flat fairness baselines tối thiểu `LR(life_time)`, `LR(views+life_time)`, `LR(degree+views+life_time)`, `MLP(raw attrs)`, và nếu dùng `language` thì có fairness rows tương ứng.
- `surrogate_ranking_metrics_a0_clean.csv` + `surrogate_ranking_metrics_hscc_clean.csv` có `label_regime`, `spearman_rho_mean`, `spearman_rho_std`, `ndcg_mean`, `ndcg_std`, `runtime_sec` cho từng regime đã chạy (M5).
- `C2` architecture comparison trên regime active có đủ rows `gcn_raw_attr`, `gin_raw_attr`, `appnp_raw_attr` (5 seeds each, mean±std) theo đúng regime đang đánh giá; **không mong đợi `gat_raw_attr` trong official rerun** vì GAT đã bị drop và run dùng `--skip-gat`.
- `C3` rank-loss row chỉ áp dụng sau khi đã chọn best arch trong đúng regime; không block nếu team cắt rank-loss để giữ `A0 + HSCC` core path.
- `A0`: `gnn_vs_degree_bootstrap_ci_a0.json` tồn tại với `n_bootstrap=1000`, `ci_95_lower`, `ci_95_upper`, `interpretation`.
- `HSCC`: `gnn_vs_baseline_bootstrap_ci_hscc.json` tồn tại với comparator = strongest valid flat baseline sau khi fairness complete.
- Runtime table có `Speedup: MC IC vs GNN inference` được tính (M5).
- `runtime_sec` = full-graph inference time (đo `time.time()` bao toàn bộ forward pass, không tính file load).
- **[v3.1 code fix — S2]** Kiểm tra `eval_ranking_harness.py`: tất cả 4 `argsort` calls phải dùng `kind='stable'` để tránh tie-breaking non-determinism trong NDCG@10% / P@10%.
- **[v3.1 code fix — S3]** Kiểm tra `run_baselines.py evaluate_on_test_mask()`: thêm `sort_values("node_id")` sau `apply_test_mask()` để đảm bảo node order nhất quán khi merge predictions.

---

## 4) Nhịp tích hợp (deadline 30/4/2026 — **📍 Hôm nay: 21/4/2026, còn 9 ngày**) [🔴 MAPR-MUST]

> **Execution status v3.2:** từ bây giờ chỉ ưu tiên path `A0 + HSCC`.

### Milestone M4.1 — Re-lock execution path (21/4)

| Person | Việc | Output |
| ------ | ---- | ------ |
| Person 1 | regenerate HSCC targets + freeze config | `regression_targets_hscc_refined.parquet`, registry update |
| Person 2 | confirm community artifact coverage | `community_features.parquet` ready for HSCC |
| Person 3 | patch regime-aware evaluation naming | runners/harness aligned |

### Milestone M4.2 — Baseline fairness before GNN claims (22/4)

| Person | Việc | Output |
| ------ | ---- | ------ |
| Person 3 | chạy HSCC flat baselines: `LR(life_time)`, `LR(views+life_time)`, `LR(degree+views+life_time)`, `MLP(raw attrs)` | HSCC rows in `baseline_ranking_metrics_hscc_clean.csv` |
| Person 3 | nếu GNN dùng `language`, thêm fairness baselines với `language` | fairness rows |
| Person 2 | provide any missing joins for community/language checks | support files |

### Milestone M4.3 — Regime-specific model runs (23–24/4)

| Person | Việc | Output |
| ------ | ---- | ------ |
| Person 3 | run GNN on `A0` | A0 surrogate rows |
| Person 3 | run GNN on `HSCC` | HSCC surrogate rows |
| Person 1 | optional `A2` only if main path stable | sensitivity artifact |

### Milestone M4.4 — Bootstrap and locking (24–25/4)

| Person | Việc | Output |
| ------ | ---- | ------ |
| Person 3 | bootstrap `A0`: GNN vs degree | `gnn_vs_degree_bootstrap_ci_a0.json` |
| Person 3 | bootstrap `HSCC`: GNN vs strongest flat baseline (frozen official rerun comparator = `lr_degree_views_life_time_lang`) | `gnn_vs_baseline_bootstrap_ci_hscc.json` |
| Cả team | lock results and table wording | result freeze |

### Milestone M5 — Integration + paper hand-off (26–30/4)

- Final `baseline_ranking_metrics_a0_clean.csv` + `baseline_ranking_metrics_hscc_clean.csv` và `surrogate_ranking_metrics_a0_clean.csv` + `surrogate_ranking_metrics_hscc_clean.csv` phải đọc được theo regime.
- `runtime_breakdown.csv` hoàn chỉnh.
- Bàn giao 2 kết quả chính cho paper:
  - `A0` contrast finding.
  - `HSCC` main comparison.

## 4b) Risk Management (v3.2) [⚪ REF]

| Rủi ro | Impact | Action |
| ------ | ------ | ------ |
| `A0` GNN không vượt degree | Expected | Không coi là bug; viết như structural finding |
| HSCC baseline fairness thiếu | Critical | Không đọc bất kỳ GNN win nào trước khi thêm đủ `life_time` baselines |
| GNN dùng `language` nhưng flat baselines không có `language` | Critical | Bổ sung fairness versions ngay |
| HSCC GNN chỉ ngang baseline mạnh nhất | Medium | Giữ contrast paper; không mở formula mới |
| Community artifact trễ | Critical | Ưu tiên Person 2 trước mọi boost item khác |
| Paper > 6 trang | Blocker | Cắt `I-A`, `A1`, exhaustive extras trước |

## 4c) Scope Reduction — Cắt khi cần (v3.2) [⚪ REF]

| Cắt trước | Giữ bắt buộc |
| --------- | ------------ |
| `I-A`, `II-B`, `GATv2-I-A` | `A0` contrast run |
| `A1`, inductive, GINE | `HSCC` main run |
| multi-alpha rankloss sweep | HSCC baseline fairness |
| exhaustive all-arch all-regime grid | bootstrap theo đúng comparator |
| per-group diagnostics | community + proxy artifacts |

---

## 5) Cách tích hợp vào runner mà vẫn giữ song song [🔴 MAPR-MUST]

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

## 6) Giảm phụ thuộc bằng test nhỏ + smoke runs [🔴 MAPR-MUST]

- Mỗi track có 1 chế độ `--dry-run` hoặc “small graph mode” (subgraph vài nghìn nodes) để:
  - test correctness
  - đo runtime sơ bộ
  - giảm thời gian review/CI

- Luôn có 3 check tối thiểu cho artifact mới:
  1. schema check (cột bắt buộc)
  2. coverage check (tỷ lệ node missing)
  3. determinism check (rerun cùng seed cho ra kết quả giống/close)

---

## 7) PR strategy (để merge nhanh mà ít conflict) [⚪ REF]

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

## 8) Checklist nhanh theo người (tóm tắt 1 trang) [🔴 MAPR-MUST]

### Person 1 — IC artifacts

| # | Việc | Artifact output | Deadline |
| - | ---- | --------------- | -------- |
| 1 | Confirm `graph_csr.npz` + `split_masks.parquet` | shared upstream artifacts | ngay |
| 2 | Lock `A0` artifacts | `ic_scores_a0.parquet`, `regression_targets_a0.parquet` | ngay |
| 3 | Regenerate và verify HSCC targets | `ic_scores_hscc_refined.parquet`, `regression_targets_hscc_refined.parquet` | **21–22/4** |
| 4 | Freeze HSCC config, add registry entry | registry updated | **21–22/4** |
| 5 | Optional `A2` only if main path stable | `ic_scores_sensitivity_a2.parquet` | sau 24/4 |

### Person 2 — Community + proxies

| # | Việc | Artifact output | Deadline |
| - | ---- | --------------- | -------- |
| 1 | Confirm `community_features.parquet` coverage 100% | community artifact ready | **21/4** |
| 2 | Confirm `diffusion_proxies.parquet` full graph | proxy artifact ready | **21/4** |
| 3 | Support joins/checks for HSCC interpretation | merged support tables if needed | 22–23/4 |
| 4 | Optional correlation add-on for HSCC | summary note / json | nếu kịp |

### Person 3 — Baselines + GNN + CI

| # | Việc | Artifact output | Deadline | Tier |
| - | ---- | --------------- | -------- | ---- |
| 1 | Deterministic evaluation harness, regime-aware | harness fixed | ngay | [🔴] |
| 2 | **C1** — Chạy A0 + HSCC flat baselines (degree-controlled variance check) | `baseline_ranking_metrics_a0_clean.csv`, `baseline_ranking_metrics_hscc_clean.csv` | **22/4** | [🔴 C1] |
| 3 | **C1** — Nếu dùng `language`, thêm HSCC fairness baselines | extra HSCC rows in baseline CSV | **22/4** | [🔴 C1] |
| 4 | **C2** — Chạy GNN architecture comparison trên `A0` (`--skip-gat`) | A0 rows in `surrogate_ranking_metrics_a0_clean.csv` | 23/4 | [🔴 C2] |
| 5 | **C2** — Chạy GNN architecture comparison trên `HSCC` (`--skip-gat`) | HSCC rows in `surrogate_ranking_metrics_hscc_clean.csv` | 23–24/4 | [🔴 C2] |
| 6 | **C3** — Rankloss variant của best C2 arch trên HSCC (sau khi C2 xong) | via `bootstrap_ci.py --include-rankloss-comparison` | 24/4 | [🟡 C3] |
| 7 | **C4** — Bootstrap `A0`: GNN vs degree baseline | `gnn_vs_degree_bootstrap_ci_a0.json` | **24/4** | [🔴 C4] |
| 8 | **C4** — Bootstrap `HSCC`: GNN vs strongest flat baseline (frozen official rerun comparator = `lr_degree_views_life_time_lang`) | `gnn_vs_baseline_bootstrap_ci_hscc.json` | **24/4** | [🔴 C4] |
| 9 | Runtime assembly + table handoff (không có row `gat_raw_attr`) | `runtime_breakdown.csv` | 25/4 | [🔴] |
| 10 | **C5** — GINE + IC edge features supplemental | post-MAPR artifact | POST-MAPR | [🔵 FUTURE:TKDE/WWW2027] |

### Handoff tối thiểu

**Person 1 → Person 2/3**

- `data/processed/graph_csr.npz`
- `data/processed/ic_scores_a0.parquet`
- `data/processed/regression_targets_a0.parquet`
- `data/processed/ic_scores_hscc_refined.parquet`
- `data/processed/regression_targets_hscc_refined.parquet`
- `data/processed/split_masks.parquet`
- `data/processed/node_attributes.parquet`

**Person 2 → Person 3**

- `data/processed/community_features.parquet`
- `data/processed/diffusion_proxies.parquet`

**Person 3 → paper owner**

- `baseline_ranking_metrics_a0_clean.csv`
- `baseline_ranking_metrics_hscc_clean.csv`
- `surrogate_ranking_metrics_a0_clean.csv`
- `surrogate_ranking_metrics_hscc_clean.csv`
- `gnn_vs_degree_bootstrap_ci_a0.json` (C4)
- `gnn_vs_baseline_bootstrap_ci_hscc.json` (C4)
- `gnn_vs_rankloss_bootstrap_ci_hscc.json` (C3 — [🟡 BOOST]; chỉ nếu C3 đã chạy)
- `runtime_breakdown.csv`
