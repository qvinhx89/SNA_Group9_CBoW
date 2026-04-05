# MAPR2026 v3 — Team 3 người: Kế hoạch coding song song (không bao gồm viết paper)

Mục tiêu của tài liệu này là thiết kế các đầu việc **có thể triển khai song song tối đa** cho team 3 người để migrate từ pipeline hiện tại (SIS-based, stage 0–3 đã ổn) sang MAPR2026 v3 (IC-based operationalization + divergence analysis + surrogate learning).

Phạm vi:

- **Chỉ phần thực thi code + tạo artifacts + chạy pipeline**.
- Không bao gồm viết paper/narrative (đã có người khác phụ trách).

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
- **Uniform p** chỉ là sensitivity variant (optional): `p_uniform = kappa/mean_degree = 2/mean_degree ≈ 0.025` (`kappa_target: 2`). Không report uniform p như primary.
- **LCC check** (Stage 0): xác nhận graph active là mostly connected (1 LCC lớn >> các component còn lại). Nếu LCC < 90% active nodes → report, xem xét restrict analysis to LCC.

### M0 Decision Record — đọc và xác nhận trước khi code

> Các quyết định dưới đây đã được **lock**. Không thay đổi mà không commit vào `docs/m0_decisions.md` và báo cả team.

| Quyết định | Giá trị đã lock | Ai chịu trách nhiệm |
|---|---|---|
| `test_frac` | **0.20** | Person 1 (trong `--test-frac` arg) |
| `stratify_by` | **degree_quintile** (`pd.qcut`, q=5, duplicates='drop') | Person 1 |
| `split_seed` | **42** | Person 1 (trong `--seed` arg) |
| `classification_threshold` | **top 10%** (`quantile(0.90)`) | Person 1 |
| `min_quadrant_size` | **150** | Person 2 |
| `n_sample` (default) | **5.000** labeled nodes | Person 1 |
| **Proxies scope** | **FULL active graph** (~168k nodes) | Person 2 |
| `runtime_sec` trong metrics | **full-graph inference only** | Person 3 |

| Quyết định | Giá trị | Khi nào quyết |
|---|---|---|
| `N_seeds × N_runs` | TBD | **M2** (sau Day-1 benchmark) |
| GNN narrative branch | TBD | **M2** (sau one-hop ρ check) |
| Uniform-p sensitivity | TBD | M2 |

**⚠ Rule quan trọng:** Person 3 **KHÔNG được tự tạo split**. Phải load từ `data/processed/split_masks.parquet` do Person 1 tạo, dùng `load_split_mask()` trong `eval_ranking_harness.py`.

---

## 2) Artifact contracts (đóng băng giao diện giữa 3 người)

> Các schema dưới đây bám theo `docs/MAPR2026_v3_migration_checklist.md`. Nếu cần đổi tên/format, phải đổi đồng bộ và ghi vào `docs/experiment_registry.md`.

| Artifact (path)                                                | Owner    | Consumers  | Contract tối thiểu                                                                        |
| -------------------------------------------------------------- | -------- | ---------- | ----------------------------------------------------------------------------------------- |
| `data/processed/graph_csr.npz`                                 | Person 1 | All        | `indptr`, `indices`, `degrees`, mapping `node_id↔row_index` deterministic                 |
| `outputs/day1_benchmark/ic_runtime_benchmark.json`             | Person 1 | All        | per-sim ms + projected runtime + decision table                                           |
| `outputs/day1_benchmark/one_hop_correlation.json`              | Person 1 | All        | Spearman ρ + decision branch (ρ<0.8 / 0.8–0.9 / >0.9)                                     |
| `data/processed/ic_scores_primary.parquet`                     | Person 1 | Person 2,3 | columns: `node_id, ic_score_mean, ic_score_std, n_runs, p_model` (**sample-only: n_sample nodes**) |
| `data/processed/regression_targets.parquet`                    | Person 1 | Person 3   | columns: `node_id, y` với `y=log1p(ic_score_mean)`                                        |
| `data/processed/classification_labels.parquet`                 | Person 1 | Person 3   | columns: `node_id, y_top10` (top 10%)                                                     |
| `data/processed/split_masks.parquet` **[M0-locked]**           | Person 1 | Person 2,3 | columns: `node_id (str), split ('train'\|'test')`. 80/20, degree-stratified q=5, seed=42. Scope = labeled nodes only. **Không ai tự tạo split khác.** |
| `data/processed/diffusion_proxies.parquet`                     | Person 2 | Person 3   | columns: `node_id, one_hop_spread, two_hop_spread`. **Scope: FULL active graph** (không phải chỉ labeled subset) |
| `data/processed/typology_labels_ic_views.parquet`              | Person 2 | Person 3   | columns: `node_id, typology_label, ic_high, views_high, ic_score_mean, views`             |
| `outputs/mapr2026_v3_results/null_model_typology_summary.json` | Person 2 | All        | JSON summary + metadata tối thiểu (vd. `n_nodes`)                                         |
| `outputs/mapr2026_v3_results/baseline_ranking_metrics.csv`     | Person 3 | All        | columns: `model_name, spearman_rho, ndcg_at_10pct, precision_at_10pct, runtime_sec`       |
| `outputs/mapr2026_v3_results/surrogate_ranking_metrics.csv`    | Person 3 | All        | (khuyến nghị) metrics mean±std (5 seeds) + `runtime_sec`                                  |

### Format spec chi tiết (để khỏi hiểu khác nhau)

#### `data/processed/graph_csr.npz`

Yêu cầu tối thiểu trong file `.npz` (keys):

- `indptr`: int64, shape `(n_nodes+1,)`
- `indices`: int32/int64, shape `(2*m_edges,)` (hoặc `(m_edges,)` tuỳ bạn encode undirected)
- `degrees`: int32/int64, shape `(n_nodes,)`, phải thỏa `degrees[i] == indptr[i+1]-indptr[i]`
- `node_ids`: array of strings, shape `(n_nodes,)`, với `node_ids[i]` là node_id tương ứng row `i`

**Determinism rule:** mapping phải deterministic giữa các lần chạy. Khuyến nghị: sort `node_id` tăng dần trước khi build CSR.

#### `outputs/day1_benchmark/*.json`

- `ic_runtime_benchmark.json`: phải có tối thiểu `per_sim_ms`, `projected_total_hours`, `decision` (n_seeds, n_runs)
- `one_hop_correlation.json`: phải có tối thiểu `spearman_rho`, `p_value` (nếu có), `decision_branch`

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

#### `outputs/mapr2026_v3_results/surrogate_ranking_metrics.csv` (khuyến nghị)

Tối thiểu nên thống nhất schema (đang bám theo scaffold):

- `model_name`: string
- `spearman_rho_mean`, `spearman_rho_std`: float
- `ndcg_mean`, `ndcg_std`: float
- `precision_mean`, `precision_std`: float
- `runtime_sec`: float

#### `outputs/mapr2026_v3_results/null_model_typology_summary.json`

Tối thiểu cần có:

- `timestamp`
- `n_nodes`
- metadata cần thiết để reproduce (ít nhất: số lần realization, seed nếu có)

#### Join rule (để khỏi dính lỗi dtype)

- Tất cả parquet/CSV dùng `node_id` nên thống nhất kiểu **string**.

### Protocol lock (để Person 2/3 không implement lệch nhau)

#### Evaluation setting (MAPR2026 v3: transductive)

- Graph là cố định; label IC chỉ có cho một tập node được sample.
- **Tất cả metric accuracy/ranking phải tính trên held-out labeled nodes (test mask)**. Không tính trên toàn bộ nodes.

**Lưu ý quan trọng về “scope” dữ liệu:**

- `ic_scores_primary.parquet` và `regression_targets.parquet` thường là **labeled subset** (do compute budget). Đây là đúng với plan v3.
- Proxies/baselines/surrogates có thể dự đoán cho full-graph để đo runtime, nhưng **khi tính metric thì chỉ dùng test mask trên labeled nodes**.

#### Metric definitions (đã lock tại M0 — không thay đổi)

- `spearman_rho`: Spearman correlation giữa `y_true` và `y_pred` trên **test labeled nodes** (sau khi apply test mask).
- `ndcg_at_10pct`: NDCG@k với $k=\lceil 0.10 \times n_{test}\rceil$; relevance lấy theo `y_true` (regression target = `log1p(ic_score_mean)`).
- `precision_at_10pct`: Precision@k với cùng k; “true top-k” định nghĩa theo top-k của `y_true` trên test set.

> **[M0-locked]** k được tính theo `n_test` (số test nodes), không phải tổng active nodes. Không thay đổi định nghĩa này mà không update `docs/m0_decisions.md`.

### Mock artifacts để không phải chờ nhau

- Nếu `ic_scores_primary.parquet` chưa có: dùng `data/processed/sis_table.parquet` (hoặc `pagerank`) làm nhãn tạm (`ic_score_mean ≈ sis_score`), để Person 2/3 viết pipeline và unit-test I/O.
- Nếu `graph_csr.npz` chưa có: Person 2/3 có thể tạm chạy trên một graph nhỏ (subgraph) được export từ `graph_active.edgelist` để validate logic.

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

# (1) Proxies placeholder (dry-run chỉ sample nhỏ; real mode sẽ chạy full graph)
python diffusion_proxies.py --dry-run --seed 42 --n-sample 2000

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

> **Nếu `split_masks.parquet` chưa có:** Person 3 nhờ Person 1 chạy lệnh (A.3) ở chế độ `--dry-run` và commit file `data/processed/split_masks.parquet`. Không tự tạo split thay thế.

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
2. **Dead account audit** (bắt buộc trước sampling — Section 13 plan v3)
   - Script: `src/data/dead_account_audit.py` (hoặc inline trong preprocessing)
   - Report: `n_dead`, `%dead`, `mean_degree_dead vs live`, `mean_views_dead vs live`
   - Output: ghi vào `outputs/stage0_data_quality/dead_account_report.json`
   - **Lý do:** stats phải có trong Section 5 (Limitations) của paper: "Dead accounts (X%) have lower degree and views than active accounts."
3. Day-1 benchmark scripts
   - `ic_runtime_benchmark.json`
   - `one_hop_correlation.json`
   - `docs/day1_decisions.md`
4. IC pilot + diagnostics (CV / non-degenerate checks)
5. IC primary labels + label stability (Jaccard top-decile across 3 MC seeds)
   - `ic_scores_primary.parquet`
   - `regression_targets.parquet`, `classification_labels.parquet`
   - **Bootstrap 95% CI** cho mỗi node: `n_bootstrap=1000`, lưu `ic_ci_lower`, `ic_ci_upper` (strongly recommended — ~30 phút implement)
6. **[M0-locked] Split mask** — tạo ngay sau khi có `ic_scores_primary.parquet`
   - `data/processed/split_masks.parquet`
   - Rule cứng: `test_frac=0.20`, `stratify=degree_quintile` (q=5), `seed=42`
   - Dùng flag `--test-frac 0.20 --seed 42` trong `ic_labels_primary.py`
   - Ghi số `n_train / n_test` vào `docs/day1_decisions.md` để team biết

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
Jaccard threshold : 0.85 (nếu thấp hơn → tăng n_runs)
```

> **⚠ CRITICAL — worker_seed cho stability check phải KHÁC với primary run:**
> - Primary: `worker_seed = 42 + node` → mỗi node có RNG riêng, tất cả runs dùng cùng seed
> - Stability: `worker_seed = mc_seed * 10000 + node` → mc_seed=0,1,2 → 3 *genuinely independent* MC experiments
> - Nếu dùng cùng seed cho cả 3 MC experiments → Jaccard sẽ = 1.0 (artificial, không phải ổn định thật sự)

**Pilot diagnostics — phải report đủ 6 metrics (v3 Section 4.1), pilot = 200 nodes × 50 runs:**

| Metric | Threshold | Ý nghĩa |
|---|---|---|
| `mean_reach` | report | mean single-seed reach |
| `median_reach` | < 5% LCC | nếu cao hơn → cascade explosive |
| `iqr_reach` | report | spread của distribution |
| `top10_to_median_ratio` | >> 1 | nếu ≈ 1 → ranking vô nghĩa |
| `rank_stability` (Spearman giữa MC seeds) | report | |
| `cv_score` | **> 0.3** | nếu thấp → cascade chết quá nhanh |

> **`cv_noise_threshold = 0.50`** (v3 Section 15): node nào có per-node CV > 0.50 là "high-variance node" → exclude khỏi stability metrics, ghi lại count. Cascade quá noisy trên node đó → cần tăng n_runs.

**Stratified sampling với KS check (v3 Section 4.4):**
- Dùng `pd.qcut(degree, q=5)` để stratify (same rule với split mask)
- Sau sampling: chạy KS test trên `degree`, `kshell`, `pagerank` — warn nếu KS stat > 0.10
- Ghi `ks_results` vào stability report

**Definition of Done (DoD) cho Track A:**

- CSR mapping deterministic (rerun ra đúng mapping, `degrees[i] == indptr[i+1]-indptr[i]`).
- Dead account report tồn tại: `outputs/stage0_data_quality/dead_account_report.json` có `n_dead`, `pct_dead`, `mean_degree_dead`, `mean_degree_live`.
- Day-1 artifacts sinh ra được: `ic_runtime_benchmark.json` và `one_hop_correlation.json`.
- IC pilot cho ra đủ 6 diagnostics metrics, CV > 0.3.
- Jaccard stability ≥ 0.85 across 3 MC seeds.
- Bootstrap 95% CI đã tính (nếu thời gian cho phép): cột `ic_ci_lower`, `ic_ci_upper` trong `ic_scores_primary.parquet`.
- `split_masks.parquet` tồn tại, schema đúng, coverage = 100% labeled nodes, test_frac ≈ 0.20.

**Gợi ý entrypoint:**

- `src/mapr2026_v3/export_csr.py`
- `src/mapr2026_v3/day1_benchmark.py`
- `src/mapr2026_v3/ic_labels_primary.py`

**Runbook tối thiểu:**

- Unblock team ngay: chạy 3 lệnh ở Mục 2.1 (A).
- Real mode: implement IC (primary: worker_seed=42+node; stability: mc_seed*10000+node), pilot diagnostics 6 metrics, KS check, stability 3 MC seeds (n_runs=150 each).

---

### Person 2 — Track B: Divergence analysis (typology IC×views + proxies + null model)

**Mục tiêu:** Task B (views vs IC typology) + baseline Group 3 (one-hop/two-hop) + null-model typology comparison.

**Có thể làm trước khi IC labels xong** bằng mock nhãn (SIS/pagerank) để hoàn thiện pipeline.

**Deliverables (theo thứ tự dependency):**

1. **[BẮT BUỘC] Community detection + cross-community features** (v3 Section 6)
   - Input: `data/processed/graph_active.edgelist` (hoặc graph_csr.npz)
   - Script: `src/graph/community.py` (đã có sẵn) — Louvain, `resolution=1.0`, `seed=42`
   - **Library:** `python-louvain` (`community.best_partition(G_nx, resolution=1.0, random_state=42)`) hoặc `cdlib` — KHÔNG dùng NetworkX Louvain (không support `random_state`)
   - Output thêm vào `node_attributes.parquet` (hoặc file riêng `community_features.parquet`):
     - `community_id`: int (Louvain partition)
     - `cross_community_edge_fraction`: float (fraction neighbors in different community)
   - Scope: ALL active nodes (phủ 100%)
   - **Lý do bắt buộc:** structural profiling claim "Hidden nodes are cross-community bridges" cần `cross_community_edge_fraction` — không có thì không support được finding này.

2. Diffusion proxies (Group 3) — **scope: FULL active graph** (M0-locked)
   - Input: `graph_csr.npz`
   - Output: `data/processed/diffusion_proxies.parquet` (ALL active nodes)
   - Output phụ: `outputs/mapr2026_v3_results/runtime_breakdown.csv` (`model_name`, `inference_sec_full_graph`)
   - **Lưu ý:** Không filter — Person 3 apply test mask khi tính metrics
   - Two-hop complexity: O(deg²) per node — estimate runtime trước khi chạy full graph

3. Typology IC×views (2×2 quadrant) + quadrant sizing
   - Input: `ic_scores_primary.parquet` + `node_attributes.parquet` (có `community_id`, `cross_community_edge_fraction`)
   - Output: `data/processed/typology_labels_ic_views.parquet`
   - Threshold: top 10% cho cả IC và views (M0-locked)
   - **Two-sample strategy nếu Hidden quadrant < 150:** tăng `n_sample` lên **8.000–10.000 nodes** (từ mặc định 5.000), augment với Sample B (high-betweenness + low-views nodes từ full graph). Sample B chỉ dùng cho typology analysis, KHÔNG dùng để train GNN. Candidates: `betweenness > quantile(0.70)` AND `views < quantile(0.30)` AND chưa có trong Sample A.

4. Structural profiling — Hidden vs Overrated (v3 Section 11)
   - Columns cần: `degree`, `pagerank`, `kshell`, `betweenness`, **`cross_community_edge_fraction`**, `life_time`
   - Method: MWU + Cliff's delta (Δ ≥ 0.20 là effect size meaningful)
   - BH-FDR correction trên tất cả p-values (Section 8.4)
   - Expected: Hidden → higher betweenness + cross_community_fraction; Overrated → higher degree + views

5. **life_time external validation của typology** (v3 Section 10 — quy tắc quan trọng)
   - IC labels KHÔNG dùng `life_time` → genuinely independent → valid external corroboration
   - Method 1: Partial Spearman (IC rank vs life_time | degree controlled)
   - Method 2: Stratified MWU by degree quintile, BH-FDR corrected
   - **CẢNH BÁO:** KHÔNG dùng `life_time` để validate GNN-full predictions (GNN-full đã thấy life_time trong features)

6. Null model comparison (configuration model) trên typology (v3 Section 5)
   - **Spec cụ thể:** 500 nodes × **3 realizations** × **100 runs/node**
   - So sánh TYPOLOGY QUADRANT (không chỉ rank correlation) giữa real graph và null
   - Câu hỏi: "Nếu null cũng có Hidden quadrant với betweenness cao → typology là degree-distribution artifact"
   - Output: `null_model_typology_summary.json` (rho_mean±std, hidden_betweenness_null_mean)

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

**DoD cho Track B:**

- `community_id` và `cross_community_edge_fraction` có trong `node_attributes.parquet` (hoặc file riêng), phủ 100% active nodes.
- `community_id` và `cross_community_edge_fraction` phủ 100% active nodes, dùng `python-louvain` với `resolution=1.0, random_state=42`.
- Proxies chạy xong trên FULL active graph (missing = 0), `runtime_breakdown.csv` có `inference_sec_full_graph`.
- Typology: mỗi quadrant ≥ 150 nodes (nếu không → apply two-sample strategy), có JSON report.
- Structural profiling: MWU + Cliff's delta (`threshold Δ ≥ 0.20`) + BH-FDR cho 6 columns, kết quả ghi ra `structural_profiling.csv`.
- `life_time` validation: chạy được cả 2 methods, ghi p_corrected (không phải p_raw). **Success target: ≥ 3/5 degree quintiles significant** (`cliffs_delta_threshold=0.20, fdr_alpha=0.05`).
- Null model: 3 realizations × 500 nodes × 100 runs, output `null_model_typology_summary.json`.

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

2. Baselines (tất cả filter qua test mask trước khi tính metrics):
   - **Group 1 — Raw features O(1):** `rank(views)`, `rank(views/life_time)`, `rank(degree)`
   - **Group 2 — Centrality O(N log N → NE):** PageRank (α=0.85), k-shell, Betweenness (NetworKit `ApproxBetweenness2`, `epsilon=0.10`, `delta=0.10`) — reuse artifacts Stage 1–2
   - **Group 3 — Diffusion proxies O(E):** one-hop + two-hop từ `diffusion_proxies.parquet` (full graph, filter test mask)

3. **Group 4 — Shallow Embedding Baselines** (v3 Section 7 Group 4 — ghi vào `baseline_ranking_metrics.csv`, KHÔNG phải surrogate CSV):
   - **Node2Vec + LR:** `dim=64, walks=20` (⚠ KHÔNG phải 200 — 10x chậm hơn), `walk_len=20`, LR regression trên embedding
   - **MLP raw attributes:** 2-layer MLP, features = `[views_log, views/day, life_time]`
   - **Lưu ý naming:** Master plan v3 gọi đây là "Group 4 Baselines" (không phải "surrogates"). Kết quả phải vào `baseline_ranking_metrics.csv` cùng với Group 1–3 để so sánh đầy đủ trong Table 2 của paper.

4. **Group 5 — GNN — 4 ablation variants** (v3 Section 7 Group 5, chỉ làm nếu Day-1 branch viable):

   | Variant | Features (in_dim) | Role |
   |---|---|---|
   | **GNN-raw-attr** | `views_log_norm, views_per_day_norm, life_time_norm` (3) | **Primary proposed** |
   | GNN-graph-only | `degree_norm` only (1) | Ablation: topology without attributes |
   | GNN-centrality | `degree_norm, pagerank_norm, kshell_norm` (3) | Ablation: hand-crafted features |
   | GNN-full | all 6 features (normalized) | Supplementary upper bound |

   > **Feature normalization bắt buộc**: tất cả features phải normalize trước khi vào GNN (min-max hoặc z-score). Column names trong experiment.yaml là `*_norm`. Không dùng raw values trực tiếp.

   Architecture: GraphSAGE, `hidden_dim=128`, `n_layers=2`, `dropout=0.3`, Huber Loss (`delta=1.0`), `lr=0.001`, `epochs=200`.
   Framework: **PyTorch Geometric (PyG) ≥ 2.5**, `torch ≥ 2.0`. Hardware yêu cầu: GPU ≥ 8GB VRAM (RTX 3080 / A100). Fallback nếu không có GPU: DGL + CPU (chậm hơn ~5×).

   Ablation story:
   - GNN-raw-attr vs MLP-raw-attr → giá trị của message passing
   - GNN-raw-attr vs GNN-graph-only → giá trị của attributes
   - GNN-raw-attr vs Group 2 baselines → giá trị của learned representations

5. Repeated training seeds + BH-FDR (v3 Sections 8.4–8.5):
   - **5 seeds:** `[42, 123, 456, 789, 1024]` → report `mean ± std`
   - **BH-FDR correction** cho tất cả MWU p-values (dùng `statsmodels.multipletests` method='fdr_bh') — report `p_corrected`, KHÔNG phải `p_raw`

6. Runtime table (v3 Section 9.3):

   | Component | Metric | Notes |
   |---|---|---|
   | Feature precompute (degree, PR, kshell) | time | Centrality baselines only |
   | MC IC labeling (N_seeds × N_runs) | time | One-time cost — từ Person 1 |
   | GNN training (5 seeds) | time | With GPU |
   | **GNN inference (168,114 nodes)** | **runtime_sec** | Full active graph |
   | Node2Vec training | time | |
   | Speedup: MC IC vs GNN inference | **Zx** | Key claim cho paper |

   `runtime_sec` trong CSV = **inference only** (không tính load + precompute).

**Runtime rule (để so sánh fair):** khuyến nghị log riêng 3 phần (precompute / train / inference). Trong `baseline_ranking_metrics.csv` có thể để `runtime_sec` là inference time trên full active nodes, và ghi chi tiết breakdown ở file phụ `outputs/mapr2026_v3_results/runtime_breakdown.csv` (nếu team đồng ý ở M0).

> **QUAN TRỌNG (v3 Section 9.3):** Nếu GNN-raw-attr là primary, **không cần tính centrality precompute time** (degree/PR/kshell) vào runtime GNN — centrality chỉ cần cho GNN-centrality và GNN-full. Việc loại bỏ centrality precompute khỏi primary GNN pipeline làm runtime so sánh **fair hơn** (và là một điểm mạnh của GNN-raw-attr: không cần expensive precompute).

**Gợi ý entrypoint (để review dễ):**

- `src/mapr2026_v3/eval_ranking_harness.py`
- `src/mapr2026_v3/run_baselines.py`
- `src/mapr2026_v3/run_surrogates.py`

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
- GNN-raw-attr chạy được 5 seeds, `surrogate_ranking_metrics.csv` có mean±std (M5).
- Runtime table có `Speedup: MC IC vs GNN inference` được tính (M5).
- Tất cả MWU p-values đã BH-FDR corrected, không report p_raw.
- `runtime_sec` = full-graph inference time (đo `time.time()` bao toàn bộ forward pass, không tính file load).

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

---

### Milestone M1 — Unblock song song (6/4 buổi chiều → 7/4)

**Mục tiêu:** tạo placeholder artifacts đúng schema để 3 track không bị block nhau.

| Person | Việc phải merge | Artifact tạo ra |
|---|---|---|
| Person 1 | `export_csr.py` (small mode) | `graph_csr.npz` |
| Person 1 | `ic_labels_primary.py --dry-run` | `split_masks.parquet` (mock) |
| Person 2 | `diffusion_proxies.py --dry-run` | `diffusion_proxies.parquet` (header) |
| Person 3 | `eval_ranking_harness.py` skeleton | import `load_split_mask()` chạy được |

**Done khi:** Person 3 có thể import `load_split_mask(PATHS.split_masks)` mà không lỗi (mock file tồn tại).

---

### Milestone M2 — Day-1 decisions (7/4, cuối ngày)

**Mục tiêu:** lock compute budget và GNN narrative — không implement surrogate trước milestone này.

| Person | Việc | Output |
|---|---|---|
| Person 1 | Chạy `day1_benchmark.py` (real mode, 100 nodes × 50 runs) | `ic_runtime_benchmark.json` |
| Person 1 | Chạy one-hop ρ check (200 pilot nodes × 50 runs) | `one_hop_correlation.json` |
| Cả team | Họp online 30 phút, đọc kết quả | Điền Phần 3 của `docs/m0_decisions.md` |

**Decision gate:**
- `projected_hours < 4h` → N_seeds=5000, N_runs=200
- `4h–8h` → N_seeds=3000, N_runs=150
- `> 8h` → N_seeds=2000, N_runs=100 (ghi limitation)
- `one_hop_rho < 0.8` → GNN là primary contribution
- `one_hop_rho 0.8–0.9` → GNN + 2-hop baseline head-to-head
- `one_hop_rho > 0.9` → **restructure:** proxies là primary, GNN là secondary; title changes

**Prepared fallback narratives (ghi vào `docs/day1_decisions.md` khi biết kết quả):**

| Tình huống | Narrative |
|---|---|
| `one_hop_rho > 0.9` | "We find that one-hop analytical spread (O(E)) achieves ρ > 0.9 with full MC IC scores — expensive simulation can be approximated by a local formula. GNN surrogate provides gains in divergence analysis and handles evolution." |
| `GNN-raw-attr ≤ two-hop` | "2-hop analytical approximation O(E²) achieves ρ ≈ X with MC IC, closely matching GNN surrogate — weighted-cascade dynamics are well-approximated by local structural summaries. GNN's value lies in efficient inference as graph evolves." |
| `views/IC ρ > 0.8` | "We find high popularity-diffusion agreement (ρ > 0.8) on Twitch's dense graph. The small divergent subset (Hidden influencers) shows systematically higher betweenness and cross-community connectivity." |

**Done khi:** `docs/m0_decisions.md` Phần 3 được commit với N_seeds, N_runs, narrative_branch.

---

### Milestone M3 — IC labels primary (8–12/4)

**Mục tiêu:** IC labels thật → unblock toàn bộ pipeline.

| Person | Việc | Deadline gợi ý |
|---|---|---|
| Person 1 | MC IC simulation (full N_seeds × N_runs từ M2) | 10/4 |
| Person 1 | Label stability check (Jaccard ≥ 0.85) + split_masks thật | 10/4 |
| Person 2 | Build typology IC×views (real IC labels) | 12/4 |
| Person 3 | Chạy baseline ranking thật (Group 1–2) | 12/4 |

**Done khi:** `baseline_ranking_metrics.csv` có ít nhất Group 1–2 rows với real IC labels.

---

### Milestone M4 — Full pipeline (12–22/4)

| Person | Việc | Deadline gợi ý |
|---|---|---|
| Person 2 | **Community detection** (Louvain + cross_community_edge_fraction) | 10/4 |
| Person 2 | Structural profiling (MWU + Cliff's Δ + BH-FDR) | 18/4 |
| Person 2 | life_time validation (partial Spearman + stratified MWU) | 18/4 |
| Person 2 | Null model comparison (500 nodes × 3 × 100) + summary JSON | 18/4 |
| Person 3 | Group 3 baselines (one-hop/two-hop từ proxies full graph) | 15/4 |
| Person 3 | Node2Vec (`dim=64, walks=20`) + LR + MLP raw attr | 18/4 |
| Person 3 | GNN-raw-attr + 3 ablation variants (nếu branch viable) | 22/4 |
| Person 3 | Runtime table + Speedup calculation | 22/4 |

---

### Milestone M5 — Integration + paper hand-off (22–27/4)

- Tất cả artifacts gom vào `outputs/mapr2026_v3_results/`
- Final `baseline_ranking_metrics.csv` (Groups 1–5) + `surrogate_ranking_metrics.csv` hoàn chỉnh
- `runtime_breakdown.csv` hoàn chỉnh (precompute / train / inference riêng biệt)
- Bàn giao cho người viết paper: bảng kết quả + plots chính

---

## 4b) Risk Management (v3 Section 19)

| Rủi ro | Xác suất | Impact | Action |
|---|---|---|---|
| `one_hop_rho > 0.9` → GNN story invalid | Trung bình | **Critical** | M2: check trước; dùng prepared narrative (Mục 4 M2) |
| IC runtime > 8h | Trung bình | **Critical** | M2: reduce 2k seeds, 100 runs; ghi limitation |
| GNN không beat cheap proxies | Trung bình | Thấp | Prepared narrative "negative result" vẫn publishable |
| Hidden quadrant < 150 nodes | Trung bình | Cao | Expand 8–10k sample + Sample B strategy (Person 2 Mục 3) |
| `views/IC ρ > 0.8` | Trung bình | Thấp | Prepared narrative "high agreement" (Mục 4 M2) |
| loky OOM với full graph | Thấp | Cao | Reduce `n_jobs`; monitor RAM ≥ 32 GB khi chạy |
| PyG installation issues | Thấp | Trung bình | Setup M0; fallback: DGL nếu PyG fail |
| Paper > 6 trang | Trung bình | **Blocker** | Cắt theo bảng dưới |

## 4c) Scope Reduction — Cắt khi cần (v3 Section 16)

Nếu timeline tight, cắt theo thứ tự này (an toàn nhất trước):

| Có thể cắt | Phải giữ bắt buộc |
|---|---|
| Uniform-p sensitivity variant | Weighted cascade IC (primary) |
| Graph perturbation test | Label stability (Jaccard ≥ 0.85) |
| 5% / 15% thresholds (chỉ giữ 10%) | Null model (3 realizations × 500 nodes × 100 runs) |
| Eigenvector/betweenness trong GNN features | One-hop + two-hop proxies (Group 3) |
| GNN-full variant | Community detection (Louvain + cross_comm_fraction) |
| Bootstrap CI (strongly recommended nhưng optional) | GNN-raw-attr (primary) + GNN-graph-only (ablation) |
| Detailed betweenness profiling | BH-FDR correction cho tất cả MWU |
| Secondary metrics (P@10%) | life_time validation của typology |

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

## 8) Checklist nhanh theo người (tóm tắt 1 trang)

### Person 1 — Phạm Quốc Vĩnh

| # | Việc | Script | Artifact output | Deadline |
|---|---|---|---|---|
| 1 | CSR export | `export_csr.py` | `graph_csr.npz` | M1 (6/4) |
| 2 | **Dead account audit** | `src/data/dead_account_audit.py` | `outputs/stage0_data_quality/dead_account_report.json` | M0 (6/4) |
| 3 | Day-1 benchmark | `day1_benchmark.py` | `ic_runtime_benchmark.json` | M2 (7/4) |
| 4 | One-hop ρ check | `day1_benchmark.py` | `one_hop_correlation.json` | M2 (7/4) |
| 5 | IC pilot + stability (+ bootstrap CI) | `ic_labels_primary.py` | stability report + bootstrap CIs | 9/4 |
| 6 | IC labels (full N×R) | `ic_labels_primary.py` | `ic_scores_primary.parquet`, `regression_targets.parquet`, `classification_labels.parquet` | 10/4 |
| 7 | **Split mask** [M0-locked] | `ic_labels_primary.py` | `split_masks.parquet` (cùng lúc #6) | 10/4 |
| 8 | Ghi `day1_decisions.md` | manual | `docs/day1_decisions.md` | M2 (7/4) |

### Person 2 — Trần Hùng Vĩ

| # | Việc | Script | Artifact output | Deadline |
|---|---|---|---|---|
| 1 | Proxies skeleton (dry-run) | `diffusion_proxies.py --dry-run` | placeholder | M1 (7/4) |
| 2 | **[BẮT BUỘC] Community detection** | `src/graph/community.py` | `community_id` + `cross_community_edge_fraction` trong `node_attributes.parquet` | 10/4 |
| 3 | **Proxies thật (full graph)** | `diffusion_proxies.py` | `diffusion_proxies.parquet` + `runtime_breakdown.csv` | 15/4 |
| 4 | Typology IC×views | `typology_ic_views.py --pct 0.10` | `typology_labels_ic_views.parquet` + quadrant JSON | 12/4 |
| 5 | Structural profiling (MWU + Cliff's Δ + BH-FDR) | `typology_ic_views.py` | profiling report CSV | 18/4 |
| 6 | **life_time validation typology** | `typology_ic_views.py` | partial Spearman + stratified MWU kết quả | 18/4 |
| 7 | Null model (500 nodes × 3 × 100 runs) | `null_model_typology.py` | `null_model_typology_summary.json` | 18/4 |

> Chú ý: **Bước 4 cần `ic_scores_primary.parquet`** (~10/4). Trong khi chờ: dùng `sis_table.parquet` làm mock.
> **Bước 2 không phụ thuộc IC labels** — có thể làm ngay từ đầu song song với bước 1.

### Person 3 — Trần Quốc Hải

| # | Việc | Script | Artifact output | Deadline |
|---|---|---|---|---|
| 1 | Harness skeleton | `eval_ranking_harness.py` | `load_split_mask()` + `compute_metrics()` OK | M1 (7/4) |
| 2 | Baselines Group 1–2 (mock labels) | `run_baselines.py` | `baseline_ranking_metrics.csv` (mock) | 9/4 |
| 3 | **Baselines Group 1–2 (real IC)** | `run_baselines.py` | CSV real (Group 1: views/views_day/degree, Group 2: PR/kshell/betweenness) | 12/4 |
| 4 | Baselines Group 3 (proxies) | `run_baselines.py` | CSV + one-hop/two-hop rows | 15/4 |
| 5 | **Group 4 — Node2Vec + LR** (`dim=64, walks=20`) | `run_baselines.py` | `baseline_ranking_metrics.csv` (thêm rows Group 4) | 18/4 |
| 6 | **Group 4 — MLP raw attr** (`views_log, views/day, life_time`) | `run_baselines.py` | cập nhật `baseline_ranking_metrics.csv` | 18/4 |
| 7 | **Group 5 — GNN-raw-attr** (nếu branch viable, 5 seeds) | `run_surrogates.py` | `surrogate_ranking_metrics.csv` (mean±std) | 22/4 |
| 8 | **Group 5 — GNN ablation** (graph-only, centrality, full) | `run_surrogates.py` | cập nhật `surrogate_ranking_metrics.csv` | 22/4 |
| 9 | Runtime table + Speedup MC vs GNN | manual/script | `runtime_breakdown.csv` hoàn chỉnh | 22/4 |

> **Tất cả bước 2–9:** load `split_masks.parquet` → `apply_test_mask()` → `compute_metrics()`. Không tự tạo split.
> **Group 4 vs Group 5:** Node2Vec+LR và MLP vào `baseline_ranking_metrics.csv` (comparable với Group 1–3). GNN variants vào `surrogate_ranking_metrics.csv` (với mean±std vì 5 seeds).
> **BH-FDR:** tất cả MWU p-values phải corrected bằng `multipletests(method='fdr_bh')` — report p_corrected.
