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
| `N_seeds × N_runs` | ___(điền sau)___ | Dựa vào `per_sim_ms` từ benchmark |
| GNN narrative branch | ___(điền sau)___ | Dựa vào one-hop ρ: <0.8 / 0.8–0.9 / >0.9 |
| Uniform-p sensitivity | Có / Không | TBD |

---

## Phần 3: Kết quả Day-1 benchmark (điền tại M2)

```
per_sim_ms           : ___
projected_total_hours: ___
decision             : N_seeds=___, N_runs=___
one_hop_rho          : ___
narrative_branch     : ___  (viable_gnn / two_hop_primary / restructure)
```

---

## Phần 4: Xác nhận artifact tồn tại trước khi code (M0 checklist)

Đánh dấu [x] sau khi verify từng artifact:

- [ ] `data/processed/graph_active.edgelist` — đọc được bằng NetworkX/igraph
- [ ] `data/processed/node_attributes.parquet` — có cột `node_id`, `views`, `life_time`, `degree`
- [ ] `data/processed/centrality_table.parquet` — có `node_id`, `degree`, `pagerank`, `betweenness`, `kshell`
- [ ] `data/processed/community_labels.parquet` — phủ 100% active nodes
- [ ] `data/processed/sis_table.parquet` — dùng làm mock IC trong dry-run

---

## Phần 5: Kết quả split mask (điền tại M3, sau khi Person 1 tạo)

```
n_labeled (tổng IC nodes): ___
n_train                  : ___
n_test                   : ___
actual_test_frac         : ___  (phải ≈ 0.20)
stratified_by_degree     : True / False (True nếu CSR available)
artifact_path            : data/processed/split_masks.parquet
created_at               : ___
```
