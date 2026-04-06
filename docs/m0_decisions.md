# MAPR2026 v3 — M0 Decision Record

> **Điền file này tại buổi kick-off (6/4/2026) và commit trước khi ai bắt đầu code.**
> Nếu cần thay đổi một quyết định đã lock, phải cập nhật file này, thông báo cả team, và update tất cả script liên quan.

---

## Phần 1: Quyết định đã lock (không thay đổi trừ khi có đồng thuận)

| Quyết định | Giá trị | Arg/param | Ai chịu trách nhiệm |
|---|---|---|---|
| `test_frac` | **0.20** | `--test-frac 0.20` | Person 1 |
| `stratify_by` | **degree_quintile** (pd.qcut q=5, duplicates='drop') | trong `_create_split_mask()` | Person 1 |
| `split_seed` | **42** | `--seed 42` | Person 1 |
| `classification_threshold` | **top 10%** (quantile 0.90) | hardcoded trong `ic_labels_primary.py` | Person 1 |
| `min_quadrant_size` | **150** | trong `typology_ic_views.py` | Person 2 |
| `n_sample` (default) | **5.000** | `--n-sample 5000` | Person 1 |
| Proxies scope | **FULL active graph** | không filter trong `diffusion_proxies.py` | Person 2 |
| `runtime_sec` definition | **full-graph inference only** | log trong `runtime_breakdown.csv` | Person 3 |
| Split mask owner | **Person 1** | `ic_labels_primary.py --out-mask` | Person 1 |
| Split mask consumer rule | **load, không tự tạo** | `load_split_mask()` + `apply_test_mask()` | Person 3 |

---

## Phần 2: Quyết định defer đến M2 (sau Day-1 benchmark)

Điền sau khi có kết quả từ `day1_benchmark.py`:

| Quyết định | Giá trị | Điều kiện |
|---|---|---|
| `N_seeds × N_runs` | **5000 × 200** | `projected_runtime < 4h` |
| GNN narrative branch | **viable_gnn** | one-hop `ρ = 0.7392` (< 0.8) |
| Uniform-p sensitivity | **Có (optional, chạy sau nếu còn thời gian)** | Không ảnh hưởng quyết định Day-1 |

---

## Phần 3: Kết quả Day-1 benchmark (điền tại M2)

```
per_sim_ms           : 0.5507069587707519
projected_total_hours: 0.15297415521409777
decision             : N_seeds=5000, N_runs=200
one_hop_rho          : 0.7391903714947583
narrative_branch     : viable_gnn
```

---

## Phần 3b: Critical Note (M2 post-check) + Freeze Benchmark Config

### A) Critical note về độ ổn định runtime

- Đã chạy lặp Day-1 nhiều lần để kiểm tra stability:
	- `outputs/day1_critical/run1_seed42/`
	- `outputs/day1_critical/run2_seed42/`
	- `outputs/day1_critical/run3_seed123/`
	- `outputs/day1_critical/run4_seed777/`
- Runtime per simulation biến thiên đáng kể theo run:
	- `per_sim_ms` min: `0.4758413314819336`
	- `per_sim_ms` max: `2.068499279022217`
	- `cv_per_sim_ms`: `0.7809127962112249`
- **Planning rule (freeze):** dùng **median runtime** cho ước lượng compute budget, không dùng 1 lần chạy đơn lẻ.
	- `median_per_sim_ms`: `0.6423945903778076`
	- `median_projected_hours (5000×200)`: `0.17844294177161318`

### B) Critical note về one-hop ρ / narrative branch

- Qua nhiều seed benchmark (`42`, `123`, `777`), `spearman_rho` nằm trong `[0.7345, 0.7563]`.
- Branch không đổi giữa các run: `viable_gnn`.
- Quy tắc diễn giải: nếu future rerun đưa `ρ` vào vùng sát ngưỡng (`0.78–0.82` hoặc `0.88–0.92`) thì đánh dấu "cần review", chưa chốt narrative ngay.

### C) Freeze benchmark config (để so sánh công bằng các lần sau)

Giữ cố định cấu hình sau cho mọi lần benchmark Day-1 chính thức:

- `--bench-nodes 100`
- `--bench-runs 50`
- `--pilot-nodes 200`
- `--pilot-runs 50`
- `--target-n-sample 5000`
- `--n-jobs -1`
- `--seed 42` (run chính để log M2); run phụ stability dùng seed `{123, 777}`
- Graph input: `data/processed/graph_csr.npz` (không regenerate giữa các run so sánh)
- IC model: `weighted_cascade` (không đổi p-model khi benchmark Day-1)
- Sampling: `degree_quintile_stratified`

Lệnh chuẩn run chính (M2 log):

```powershell
python src/mapr2026_v3/day1_benchmark.py --n-jobs -1 --seed 42 --bench-nodes 100 --bench-runs 50 --pilot-nodes 200 --pilot-runs 50 --target-n-sample 5000 --out-dir outputs/day1_benchmark
```

Lệnh run stability (không thay config, chỉ đổi seed):

```powershell
python src/mapr2026_v3/day1_benchmark.py --n-jobs -1 --seed 123 --bench-nodes 100 --bench-runs 50 --pilot-nodes 200 --pilot-runs 50 --target-n-sample 5000 --out-dir outputs/day1_critical/run_seed123
python src/mapr2026_v3/day1_benchmark.py --n-jobs -1 --seed 777 --bench-nodes 100 --bench-runs 50 --pilot-nodes 200 --pilot-runs 50 --target-n-sample 5000 --out-dir outputs/day1_critical/run_seed777
```

---

## Phần 4: Xác nhận artifact tồn tại trước khi code (M0 checklist)

Đánh dấu [x] sau khi verify từng artifact:

- [x] `data/processed/graph_active.edgelist` — đọc được bằng NetworkX/igraph
- [x] `data/processed/node_attributes.parquet` — có cột `node_id`, `views`, `life_time`, `degree`
- [x] `data/processed/centrality_table.parquet` — có `node_id`, `degree`, `pagerank`, `betweenness`, `kshell`
- [x] `data/processed/community_labels.parquet` — phủ 100% active nodes
- [x] `data/processed/sis_table.parquet` — dùng làm mock IC trong dry-run
- [x] `outputs/stage0_data_quality/dead_account_report.json` — pre-Day1 audit bắt buộc
- [x] `outputs/stage0_data_quality/lcc_report.json` — pre-Day1 LCC check bắt buộc
- [x] `data/processed/graph_csr.npz` — input bắt buộc cho Day-1 benchmark

---

## Phần 5: Kết quả split mask (điền tại M3, sau khi Person 1 tạo)

```
n_labeled (tổng IC nodes): 5000
n_train                  : 4000
n_test                   : 1000
actual_test_frac         : 0.20
stratified_by_degree     : True
artifact_path            : data/processed/split_masks.parquet
split_sha256             : 005de40762f6c75e4df66a53efeaa883d126d52abd5c4af0224d736992362104
freeze_manifest          : outputs/day1_benchmark/split_freeze_manifest.json
versioned_handoff_dir    : outputs/handoffs/person1_day1_20260406_p1_day1_v1
created_at               : see `outputs/day1_benchmark/split_freeze_manifest.json` -> `timestamp`
```

## Phần 6: P0 pre-handoff quality gate status (provisional/final)

Artifacts mới đã tạo:
- `outputs/day1_benchmark/ic_label_stability.json`
- `outputs/day1_benchmark/ic_label_uncertainty.json`
- `data/processed/ic_scores_primary_with_ci.parquet`
- `outputs/handoffs/person1_day1_20260406_p1_day1_v1/manifest.json`

Kết quả chính:
- Stability (3 MC seeds, 150 runs/seed):
	- `jaccard_mean`: `0.3069298298144156`
	- `jaccard_min`: `0.3020833333333333`
	- `jaccard_pass_threshold (>=0.85)`: `False`
- Uncertainty quanh top-10 threshold:
	- `boundary_ratio` (CI crosses threshold): `0.199`
	- `n_boundary_among_y_top10`: `415 / 500`
	- `ambiguous_ratio`: `0.155`

Trạng thái bàn giao cho downstream:
- Runtime benchmark + one-hop branch: **FINAL cho planning**
- `regression_targets.parquet`: **PROVISIONAL-USABLE**
- `classification_labels.parquet` (`y_top10`): **PROVISIONAL (high-noise)**

Rule truyền thông nội bộ (bắt buộc khi handoff):
- Person 2/3 phải dùng split từ artifact đã freeze (không tự split).
- Khi report kết quả từ `y_top10`, phải ghi rõ đây là provisional cho tới khi stability pass threshold hoặc protocol được cập nhật chính thức.
