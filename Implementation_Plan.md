# Implementation Plan chi tiết - Project SNA Twitch (Group 9)

## 1) Mục tiêu triển khai

Từ proposal hiện tại, kế hoạch implementation tập trung vào 4 trục chính:

1. Xây pipeline SNA có thể chạy lặp lại, tạo ra SIS và 2x2 typology.
2. Chứng minh được tính hợp lệ của Hidden Influencer bằng validation độc lập (single-seed IC).
3. Benchmark rõ ràng các chiến lược seeding trong multi-seed IC.
4. Kiểm tra khả năng detectability bằng surface metrics (`views`, `degree`) qua ML baseline.

Mọi bước đều phải có:

- Input rõ ràng.
- Script tái sử dụng.
- Output chuẩn hoá.
- Log và seed để reproducibility.

---

## 2) Kiến trúc triển khai tổng thể

Pipeline thực thi gồm 8 phase liên tục:

1. Environment setup và data intake.
2. Data quality + preprocessing graph active.
3. Centrality analysis (Degree, PageRank, Betweenness approximate).
4. Community + core structure (Louvain, k-shell).
5. SIS + typology + robustness.
6. Independent validation (single-seed IC theo nhóm).
7. Seeding benchmark (multi-seed IC).
8. ML detectability + report packaging.

Nguyên tắc xuyên suốt:

- Không chỉnh sửa dữ liệu gốc.
- Tách output theo từng stage.
- Mỗi stage có 1 file metrics tổng hợp và 1 file metadata/params.

---

## 3) Folder structure đề xuất

```text
Social/
  data/
    raw/
      twitch_edges.csv
      twitch_features.csv
      README_source.md
    interim/
      active_nodes.csv
      active_edges.csv
      node_index_map.parquet
    processed/
      graph_active.edgelist          # [CHANGED] thay vì .pkl
      node_attributes.parquet        # [NEW] attributes riêng
      centrality_table.parquet
      community_labels.parquet
      sis_table.parquet
      typology_labels.parquet

  notebooks/
    00_data_audit.ipynb
    01_centrality_analysis.ipynb
    02_community_core.ipynb
    03_sis_typology.ipynb
    04_ic_simulation.ipynb
    05_ml_detectability.ipynb

  src/
    config/
      base.yaml
      paths.yaml
      experiment.yaml

    data/
      load_raw.py
      preprocess_graph.py
      split_ml.py

    graph/
      build_graph.py
      centrality.py
      community.py
      kshell.py
      null_model.py                  # [NEW] configuration model comparison

    sis/
      compute_sis.py
      build_typology.py
      robustness.py

    simulation/
      ic_model.py
      ic_calibration.py              # [NEW] IC parameter calibration
      seed_strategies.py
      run_single_seed_ic.py
      run_multi_seed_ic.py

    ml/
      features_surface.py
      train_lr.py
      train_rf.py                    # [NEW] RandomForest
      shap_analysis.py               # [NEW] SHAP interpretation
      evaluate_metrics.py

    evaluation/
      stats_tests.py                 # [ENHANCED] Benjamini-Hochberg, Cliff's Delta
      power_analysis.py              # [NEW] power analysis
      ranking_overlap.py
      summary_tables.py

    utils/
      io_utils.py
      logging_utils.py
      seed_utils.py
      plot_utils.py
      parallel_utils.py              # [NEW] joblib wrapper

  # [CHANGED] Thay PowerShell bằng cross-platform scripts
  Makefile                           # [NEW] Make targets
  run_all.sh                         # [NEW] Bash script cross-platform
  run_all.py                         # [NEW] Python CLI entrypoint (Windows-friendly)
  requirements.txt                   # [ENHANCED] exact versions pinned
  requirements.in                    # [NEW] for pip-compile

  reports/
    final_report.md
    figures/
      fig_rank_divergence.png
      fig_typology_distribution.png
      fig_ic_strategy_comparison.png
      fig_hidden_vs_overrated_ic.png
      fig_sensitivity_heatmap.png    # [NEW]
      fig_shap_beeswarm.png          # [NEW]
      fig_confusion_matrix.png       # [NEW]
    tables/
      table_rq1_metrics.csv
      table_rq2_hidden_validation.csv
      table_rq3_ic_benchmark.csv
      table_rq4_detectability_report.csv
    drafts/
      report_outline.md

  outputs/
    stage0_data_quality/             # [NEW]
    stage1/
    stage2/
    stage3/
    stage3_ic_calibration/           # [NEW]
    stage4_single_seed/
    stage5_multi_seed/
    stage6_ml/

  logs/
    run_history/
    timing/
    errors/

  tests/
    test_preprocess.py
    test_sis.py
    test_ic.py
    test_ml_pipeline.py

  docs/
    implementation_notes.md          # [CRITICAL] SIS formula + citations
    assumptions_limitations.md       # [CRITICAL] graph static, undirected
    experiment_registry.md           # [CRITICAL] all config changes logged

  README.md
```

---

## 4) Quy ước dữ liệu và output

### 4.1 Quy ước tên cột chuẩn

- `node_id`: ID streamer.
- `views`, `degree`, `pagerank`, `betweenness`, `kshell`.
- `sis_score`, `sis_rank`.
- `views_group` (`high`/`low`), `sis_group` (`high`/`low`).
- `typology_label` (`true`, `hidden`, `overrated`, `non`).

### 4.2 Quy ước file metrics theo stage

Mỗi stage phải ghi:

- `metrics.json`: metric chính + timestamp + seed + config hash.
- `params.json`: toàn bộ tham số chạy.
- `artifact_index.csv`: danh sách artifact được tạo.

---

## 5) Kế hoạch implementation theo tuần (chi tiết) — Final Expert Review Version

> **Timeline Overview (10 tuần)**
>
> | Tuần | Nội dung chính                             | Deliverable chính                         | Quality Gate                            |
> | ---- | ------------------------------------------ | ----------------------------------------- | --------------------------------------- |
> | 1-2  | Setup + Data Audit + Null Model Prep       | `interim/*`, `null_model.py` skeleton     | No self-loop, no duplicate              |
> | 3-4  | Centrality + Community + k-shell           | `stage1/*`, `stage2/*`                    | No NA centrality                        |
> | 5    | SIS + Typology + Robustness + Null Model   | `sis_table.parquet`, `robustness_summary` | Jaccard >= 0.7 (target vận hành nội bộ) |
> | 6    | IC Calibration + Single-seed (RQ2)         | `stage4_single_seed/*`                    | 50 runs/seed theo thiết kế mẫu          |
> | 7    | Multi-seed Benchmark (RQ3) + Sensitivity p | `stage5_multi_seed/*`                     | Rank stability qua 3 giá trị p          |
> | 8-9  | ML Detectability (RQ4) + Ablation + SHAP   | `stage6_ml/*`, tables + figures           | No data leakage                         |
> | 10   | Final Report + Packaging + runners         | `final_report.md` + all artifacts         | End-to-end 1 lệnh                       |

**Phân lớp implementation:**

- **Must-have:** Các stage cốt lõi cho 4 RQ chính (SIS/typology, single-seed IC, multi-seed IC, LR detectability) + reproducibility.
- **Nice-to-have:** RandomForest, SHAP, Node2Vec ablation, power-analysis mở rộng nếu còn thời gian.
- Các ngưỡng như Jaccard, NMI, reach ratio được dùng như **target vận hành nội bộ**, không phải ngưỡng học thuật bắt buộc.

---

### 🔴 Week 1-2: Setup, Data Audit + Null Model Preparation

**Mục tiêu:**

- Chốt môi trường chạy thống nhất giữa các máy (Windows/Linux/Mac) với 1 cách gọi lệnh tương đương.
- **Định nghĩa chính thức SIS formula với literature grounding** để khóa thiết kế trước khi coding Stage 3.
- Xây xong data foundation cho toàn pipeline: raw -> interim -> processed (không dùng pickle).
- Hoàn tất data audit + graph integrity check + active-subgraph extraction với log đầy đủ.
- Chuẩn bị sẵn null model skeleton để Week 5 chỉ cần bổ sung logic so sánh.
- Chốt phạm vi dữ liệu: **Twitch Gamers bản global** (không trộn biến thể dataset khác).

**Yêu cầu kỹ thuật + thông số bắt buộc (để tránh mơ hồ khi implement):**

- **Input contract stage0:**
  - `twitch_edges.csv` bắt buộc có tối thiểu: `source`, `target`.
  - `twitch_features.csv` bắt buộc có tối thiểu: `node_id`, `views`, `dead_account`.
  - Kiểu dữ liệu tối thiểu: `node_id` nhất quán giữa 2 file; `views` numeric; `dead_account` binary (0/1).
- **Thông số mặc định phải khai báo trong `src/config/experiment.yaml`:**
  - `GLOBAL_SEED: 42`.
  - `graph_mode: undirected_simple`.
  - `filter_dead_account_value: 0`.
  - `allow_self_loops: false`.
  - `allow_duplicate_edges: false`.
- **Rule xử lý orphan nodes (để kết quả nhất quán):**
  - Mặc định giữ orphan trong `active_nodes.csv` nếu node hợp lệ theo filter.
  - `active_edges.csv` chỉ chứa cạnh hợp lệ sau khi remove self-loop/duplicate.
  - Mọi quyết định loại orphan đặc biệt phải ghi rõ trong `metrics.json` + `experiment_registry.md`.
- **Connected components + LCC decision rule (bắt buộc):**
  - Sau khi remove self-loop/duplicate phải chạy component analysis và ghi vào `outputs/stage0_data_quality/component_analysis.json`.
  - Ngưỡng quyết định mặc định:
    - `lcc_ratio >= 0.99` -> `keep_all`.
    - `0.95 <= lcc_ratio < 0.99` -> `keep_all` + warning trong `docs/assumptions_limitations.md`.
    - `lcc_ratio < 0.95` -> dừng để chốt nhóm `restrict_to_lcc` hoặc `keep_all` với rationale rõ.
  - Mọi quyết định LCC phải ghi vào `docs/experiment_registry.md` trước khi export `graph_active.edgelist`.

**Kế hoạch thực thi chi tiết (theo thứ tự chạy):**

1. **Kick-off kỹ thuật và freeze scope (Day 1 - buổi 1):**
   - Xác nhận source dữ liệu, checksum file raw, và đường dẫn chuẩn trong `src/config/paths.yaml`.
   - Chốt naming convention (snake_case) và định dạng artifact (`.csv`, `.parquet`, `.json`, `.edgelist`).
   - Chốt quy ước seed dùng chung:
     - `GLOBAL_SEED` cho toàn pipeline.
     - Seed riêng cho từng stage ghi trong `params.json`.
   - Chốt rule thay đổi config: mọi thay đổi phải ghi vào `docs/experiment_registry.md`.

2. **CRITICAL: Chuẩn hóa environment + runner (Day 1 - buổi 2):**
   - Tạo và khóa dependency:
     - Cập nhật `requirements.in`.
     - Bổ sung `networkit>=10.1` (preferred backend cho betweenness approximate; Windows có thể cần WSL2/conda hoặc build từ source).
     - Sinh `requirements.txt` bằng `pip-compile requirements.in > requirements.txt`.
   - Thiết lập 3 cách chạy tương đương:
     - `make stage0` (Linux/Mac hoặc môi trường có Make).
     - `./run_all.sh --stage stage0`.
     - `python run_all.py --stage stage0` (ưu tiên cho Windows).
   - Smoke test cài đặt:
     - Import test cho `networkx`, `pandas`, `scikit-learn`, `cdlib`, `statsmodels`, `joblib`.
     - Ghi kết quả vào `logs/run_history/week1_env_check.log`.

3. **CRITICAL: Viết `docs/implementation_notes.md` - SIS Formula Definition (Day 2):**

   ```latex
   SIS(v) = (rank_norm(PageRank) + rank_norm(Betweenness) + rank_norm(k-shell)) / 3

   rank_norm(x) = (rank(x) - 1) / (N - 1)   # percentile rank [0, 1]
   ```

   - **Citations bắt buộc**: Kitsak et al. (2010) k-shell, Li et al. (2021) hybrid centrality.

- **Notation policy (bắt buộc):** toàn bộ tài liệu/report dùng `rank_norm` trong miền `[0, 1]`; không dùng lẫn với rank thô khi diễn giải `sis_score`.
- **k-shell Rank Normalization policy (bắt buộc trong `docs/implementation_notes.md`):**
  - Sort multi-key: `kshell DESC -> degree DESC -> node_id ASC`.
  - Assign `rank` từ `1..N`, sau đó chuẩn hóa `rank_norm = (N - rank) / (N - 1)`.
  - Không dùng công thức cộng hybrid kiểu `kshell + degree/max_degree` để tie-break.
- Plan Eigenvector check:
  - Nếu `corr(Eigenvector, PageRank) > 0.8` -> redundant (không đưa vào SIS chính).
  - Nếu `corr <= 0.8` -> chạy sensitivity sweep `[0.3, 0.3, 0.2, 0.2]` ở tuần sau.
- Bổ sung mục "Out-of-scope" để tránh scope creep (ví dụ: temporal diffusion, edge-weight modeling).

4. **Data intake + audit notebook (Day 2-3):**
   - Hoàn thiện `src/data/load_raw.py`:
     - Validate schema bắt buộc của edges/features.
     - Validate kiểu dữ liệu cột chính (`numeric`, `categorical`, `binary`).
     - Validate missing ratio theo cột và ghi vào `outputs/stage0_data_quality/metrics.json`.
   - Tạo notebook `notebooks/00_data_audit.ipynb` với các biểu đồ tối thiểu:
     - Histogram/log-hist cho `views`.
     - Tỉ lệ `dead_account`.
     - Phân bố degree sơ bộ từ raw edge list.
   - Export report nhanh: `outputs/stage0_data_quality/data_audit_summary.md`.

5. **Graph preprocessing + integrity checks (Day 3-4):**
   - Hoàn thiện `src/data/preprocess_graph.py` với luồng chuẩn:
     - B1: Load raw edges/features.
     - B2: Lọc node `dead_account = 0`.
     - B3: Dựng active graph undirected.
     - B4: Remove self-loop và duplicate edges.
     - B4.5: Chạy `analyze_connected_components(G_active)` để lấy `num_components`, `lcc_size`, `lcc_ratio`, `isolated_nodes`, `small_components_le10`.
     - B4.6: Apply `apply_lcc_filter(G_active, lcc_nodes, decision)` theo quyết định đã log trong `docs/experiment_registry.md`.
     - B5: Kiểm tra node orphan (nếu có) và log quyết định giữ/bỏ.
   - Kiểm tra chất lượng bắt buộc:
     - `num_self_loops == 0`.
     - `num_duplicate_edges == 0`.
     - `num_nodes_active` và `num_edges_active` khớp giữa log và file xuất.
     - Có `lcc_ratio` và decision `keep_all/restrict_to_lcc` trong metrics + registry.
   - Lưu trung gian:
     - `data/interim/active_nodes.csv`.
     - `data/interim/active_edges.csv`.
     - `data/interim/node_index_map.parquet`.
     - `outputs/stage0_data_quality/component_analysis.json`.

6. **CRITICAL: Chuẩn hóa output processed (không pickle) (Day 4):**
   - Export graph: `data/processed/graph_active.edgelist`.
   - Export attributes: `data/processed/node_attributes.parquet`.
   - Cập nhật `artifact_index.csv` cho stage0, gồm checksum và timestamp.

7. **Null model skeleton + smoke execution (Day 5):**
   - Tạo `src/graph/null_model.py` skeleton với các hàm:
     - `build_configuration_model(degree_sequence, seed)`.
     - `sanitize_multigraph_to_simple_graph(G)`.
     - `summarize_null_graph_stats(G_null)`.
   - Chưa chạy full comparison ở Week 1-2, chỉ smoke test:
     - Sinh null graph trên sample nhỏ (1-5% nodes) để kiểm tra pipeline không vỡ.
   - Ghi kết quả smoke vào `outputs/stage0_data_quality/null_model_smoke.json`.

7b. **CRITICAL - Resource profiling + fallback compute (Day 5, chạy song song với null smoke):**

- Tạo section `Resource Profiling` trong `docs/implementation_notes.md` với benchmark tối thiểu:
  - Graph load runtime + memory peak.
  - Betweenness approximate runtime (ghi rõ backend + params thực tế).
  - Single IC run runtime (1 seed, full graph).
- Ghi profile vào `outputs/stage0_data_quality/resource_profile.json` với schema:
  - `graph_load_memory_gb`, `betweenness_backend`, `betweenness_runtime_sec`, `single_ic_run_runtime_sec`.
  - `betweenness_params` (ví dụ: `{epsilon, delta}` nếu NetworKit; hoặc `{k_pivots, seed}` nếu NetworkX).
  - `machine_ram_gb`, `cpu_cores`, `profiling_seed`.
- Lưu ý đo memory:
  - `tracemalloc` dùng cho baseline nhanh nhưng có thể thấp hơn RSS thực; nếu có thể ghi thêm RSS để tránh under-estimate.
- Quy tắc fallback (bắt buộc log vào `docs/experiment_registry.md` khi trigger):
  - Nếu NetworKit không available -> fallback NetworkX với `k_pivots = 2000` (không được < 1000 khi graph > 50k nodes) và bắt buộc log quyết định.
  - Nếu đang fallback NetworkX và `betweenness_runtime_sec > 1800` -> dừng để chuyển môi trường chạy betweenness sang Linux/WSL2 (ưu tiên) hoặc điều chỉnh kế hoạch runtime và ghi rõ limitation.
  - `graph_load_memory_gb > 8` -> ưu tiên cách dựng graph giảm overhead (`from_edgelist`) hoặc sparse/CSR path.
  - `single_ic_run_runtime_sec > 30` -> giữ pilot `runs_per_seed=50` trước khi scale full benchmark.

8. **Definition of Ready cho Week 3 (cuối Week 2):**
   - Runner hoạt động bằng ít nhất 1 lệnh trên máy chính (ưu tiên `python run_all.py --stage stage0`).
   - Stage0 artifacts đầy đủ, đọc được, và tái tạo được với cùng seed.

- Có `resource_profile.json` + quyết định fallback (nếu trigger threshold) đã log vào registry.
- Có `component_analysis.json` + LCC decision entry trước khi khóa `graph_active.edgelist`.
- `docs/implementation_notes.md` và `docs/experiment_registry.md` đã có entry đầu tiên.
- Tạo issue list cho Week 3: bottleneck dự kiến (betweenness runtime, memory usage, I/O speed).

**Execution runbook tối thiểu (để team chạy thống nhất):**

1. Cài môi trường và lock dependencies:

- `pip-compile requirements.in > requirements.txt`
- `pip install -r requirements.txt`
- `networkit` là preferred backend cho betweenness approximate; nếu không cài được (thường gặp trên Windows), fallback NetworkX phải dùng `k_pivots = 2000` và log quyết định.

2. Chạy stage0 bằng runner chính của máy:

- Linux/Mac: `make stage0` hoặc `./run_all.sh --stage stage0`
- Windows: `python run_all.py --stage stage0`

3. Verify nhanh sau chạy:

- Có `outputs/stage0_data_quality/metrics.json` và `artifact_index.csv`.
- Có `outputs/stage0_data_quality/resource_profile.json` và `outputs/stage0_data_quality/component_analysis.json`.
- Có `data/processed/graph_active.edgelist` và `data/processed/node_attributes.parquet`.
- Log có ghi seed + config hash.

**Fail-fast criteria (dừng và xử lý ngay):**

- Thiếu bất kỳ input raw file hoặc mismatch schema bắt buộc.
- `num_self_loops > 0` hoặc `num_duplicate_edges > 0` sau preprocess.
- Runner fail liên tiếp 2 lần với cùng config/seed.
- `implementation_notes.md` chưa có công thức SIS + citations.
- Chưa có `outputs/stage0_data_quality/resource_profile.json` trước khi sang Week 3.
- Chưa có `outputs/stage0_data_quality/component_analysis.json` hoặc thiếu LCC decision entry trong `docs/experiment_registry.md` trước khi export graph processed.
- Trigger fallback threshold nhưng không có decision log trong `docs/experiment_registry.md`.

**Handover package Week 1-2 (bắt buộc trước khi sang Week 3):**

- `outputs/stage0_data_quality/metrics.json`, `artifact_index.csv`, `data_audit_summary.md`.
- `outputs/stage0_data_quality/resource_profile.json`, `outputs/stage0_data_quality/component_analysis.json`.
- `data/interim/active_nodes.csv`, `data/interim/active_edges.csv`, `data/interim/node_index_map.parquet`.
- `data/processed/graph_active.edgelist`, `data/processed/node_attributes.parquet`.
- `docs/implementation_notes.md`, `docs/experiment_registry.md` (đã có entry scope + seed + config).

**Deliverable:**

- `docs/implementation_notes.md` (SIS formula + citations complete).
- `data/interim/*`, `data/processed/graph_active.edgelist`, `data/processed/node_attributes.parquet`.
- `requirements.txt` (exact pinned), `Makefile`, `run_all.sh`, `run_all.py`.
- `src/graph/null_model.py` skeleton.
- `outputs/stage0_data_quality/metrics.json`.
- `outputs/stage0_data_quality/resource_profile.json`, `outputs/stage0_data_quality/component_analysis.json`.

**Quality Gate:**

- Không có self-loop, duplicate edge trong active graph.
- Number of node/edge active khớp log preprocessing.
- SIS formula có citation + decision rule rõ trong `implementation_notes.md`.
- Resource profiling đã chạy và có fallback decision log nếu vượt ngưỡng runtime/memory.
- LCC decision (`keep_all`/`restrict_to_lcc`) đã được chốt và log trước khi freeze dữ liệu processed.
- `make stage0`, `./run_all.sh --stage stage0`, và `python run_all.py --stage stage0` có ít nhất 1 cách chạy PASS trên máy triển khai.
- Tất cả artifact stage0 có mặt trong `artifact_index.csv` và có timestamp + config hash.

---

### 🔵 Week 3-4: Stage 1 + Stage 2 (SNA Core) + Eigenvector + Louvain Stability

**Mục tiêu:**

- Hoàn thành đầy đủ Stage 1 (centrality) và Stage 2 (community + core) trên active graph đã chuẩn hóa từ Week 1-2.
- Đảm bảo mọi signal cấu trúc chính có thể tái tạo với cùng seed và có log runtime để ước lượng chi phí cho các tuần sau.
- **Khóa quyết định Eigenvector redundancy + Louvain stability** làm đầu vào cho Stage 3.
- Tạo bảng phân tích RQ1 sơ bộ với artifact rõ ràng để review nội bộ trước khi sang SIS/typology.

**Yêu cầu kỹ thuật + thông số bắt buộc (để chạy đồng nhất giữa các máy):**

- **Stage 1 (centrality) parameters mặc định:**
  - `pagerank_alpha: 0.85`, `pagerank_tol: 1e-6`, `pagerank_max_iter: 100`.
  - `eigenvector_tol: 1e-6`, `eigenvector_max_iter: 1000`.
  - Betweenness approximate (preferred): `backend=networkit`, `epsilon=0.10`, `delta=0.10`.
  - Betweenness fallback (chỉ khi NetworKit không available): `backend=networkx`, `k_pivots=2000`, `seed=42`.
  - Guard: nếu `num_nodes > 50k` thì `k_pivots < 1000` là **hard error** (pipeline dừng ngay).

  Ví dụ cấu hình trong `src/config/experiment.yaml`:

  ```yaml
  centrality:
    betweenness:
      backend: "networkit" # "networkit" | "networkx"
      # NetworKit params (dùng khi backend=networkit)
      epsilon: 0.10
      delta: 0.10
      # NetworkX fallback params (dùng khi backend=networkx)
      k_pivots: 2000 # KHÔNG đặt < 1000 cho graph > 50k nodes
      normalized: true
      seed: 42
      # Guard
      min_k_large_graph: 1000
  ```

- **Stage 1 fallback rule (platform/runtime control):**
  - Nếu NetworKit không available (ví dụ Windows không build/cài được) -> bắt buộc fallback NetworkX `k_pivots=2000` và ghi quyết định vào `docs/experiment_registry.md`.
- **Stage 2 (community) parameters mặc định:**
  - `louvain_runs: 10`.
  - `louvain_resolution: 1.0` (trừ khi có issue log chính thức).
  - `louvain_seed_list` phải được ghi rõ trong `params.json`.
- **Attribute diagnostics (bổ sung để khớp proposal):**
  - Community-language alignment bắt buộc report bằng `NMI` và `purity_score`.
  - Core-periphery theo `kshell` phải so sánh `life_time` bằng Mann-Whitney U + effect size.
- **Output schema tối thiểu để downstream không lỗi:**
  - Stage1 table phải có: `node_id`, `degree`, `pagerank`, `eigenvector`, `betweenness`, `views`.
  - Stage2 table phải có: `node_id`, `community_id`, `kshell`, `inter_community_edge_ratio`.
  - k-shell rank cho SIS phải theo rule deterministic: sort `kshell DESC -> degree DESC -> node_id ASC`.
  - Stage2 output nên có thêm `kshell_rank`, `kshell_rank_norm` để trace tie-breaking khi cần audit.
  - Stage2 diagnostics phải có: `attribute_community_analysis.csv`, `core_lifetime_tests.csv`.

**Kế hoạch thực thi chi tiết (theo thứ tự chạy):**

1. **Pre-flight check cho Stage 1/2 (Day 1 - buổi 1):**
   - Xác nhận input tồn tại và đọc được:
     - `data/processed/graph_active.edgelist`.
     - `data/processed/node_attributes.parquet`.
   - Validate config chạy:
     - Seed của stage1/stage2 trong `src/config/experiment.yaml`.
     - Betweenness backend + params (NetworKit: `epsilon/delta`; NetworkX: `k_pivots/seed`).
     - Số lần chạy Louvain = 10.
   - Tạo thư mục output nếu chưa có:
     - `outputs/stage1/`.
     - `outputs/stage2/`.

2. **Stage 1A - Tính centrality cốt lõi (Day 1 - buổi 2 đến Day 2):**
   - Chạy `src/graph/centrality.py` để tính:
     - Degree.
     - PageRank.
     - Eigenvector centrality.
   - Tính Betweenness approximate:
     - Thứ tự ưu tiên:
       - NetworKit backend (preferred): ApproxBetweenness2 với `epsilon=0.10`, `delta=0.10`.
       - NetworkX fallback (chỉ khi NetworKit không available): `nx.betweenness_centrality(G, k=2000, seed=42)`.
     - Bắt buộc ghi vào `outputs/stage1/centrality_metrics.json`:
       - `betweenness_backend`: `networkit` | `networkx`.
         - `betweenness_k_or_epsilon`: giá trị thực tế dùng (NetworkX: `k_pivots`; NetworKit: `epsilon`).
       - `betweenness_runtime_sec`.
         - `betweenness_cv_estimate` (rough heuristic; chỉ áp dụng cho NetworkX): `1 / sqrt(k_pivots)`.
     - Nếu NetworKit unavailable và dùng fallback NetworkX: bắt buộc log decision + impact vào `docs/experiment_registry.md`.
   - Chuẩn hóa tên cột đầu ra vào `centrality_table.parquet`:
     - `degree`, `pagerank`, `eigenvector`, `betweenness`.

3. **Stage 1B - Phân tích divergence với views (Day 2):**
   - Merge centrality table với `views` từ attributes.
   - Tính Spearman:
     - `rho(degree, views)`.
     - `rho(pagerank, views)`.
     - `rho(betweenness, views)`.
     - `rho(eigenvector, views)`.
   - Tính top-20% overlap (Jaccard) cho từng cặp ranking với views.
   - Xuất artifact:
     - `outputs/stage1/centrality_table.parquet`.
     - `outputs/stage1/centrality_metrics.json` (kèm runtime, seed, config hash).
     - `outputs/stage1/rq1_ranking_metrics.csv` (bảng tóm tắt phục vụ RQ1).

4. **CRITICAL - Eigenvector redundancy decision (cuối Day 2):**
   - Tính `Spearman(Eigenvector, PageRank)`.
   - Quy tắc quyết định:
     - Nếu `corr > 0.8` -> đánh dấu redundant, không đưa vào SIS chính.
     - Nếu `corr <= 0.8` -> đánh dấu non-redundant, chuẩn bị sensitivity sweep 4-trọng số ở Week 5.
   - Ghi quyết định vào:
     - `docs/experiment_registry.md`.
     - `outputs/stage1/centrality_metrics.json`.

5. **Stage 2A - Louvain multi-run stability (Day 3):**
   - Chạy Louvain 10 lần với seed khác nhau (dùng `joblib.Parallel(n_jobs=-1)`).
   - Mỗi run lưu:
     - Community labels.
     - Modularity Q.
     - Runtime từng run.
   - Tính NMI pairwise giữa 10 partitions và lấy:
     - `nmi_mean`, `nmi_std`, `nmi_min`, `nmi_max`.
   - Chọn best partition theo modularity lớn nhất để làm partition chính.

6. **Stage 2B - Core structure + brokerage + attribute diagnostics (Day 3-4):**

- Chạy k-shell decomposition, thêm cột `kshell` vào bảng trung tâm.
- Chuẩn hóa rank k-shell cho SIS bằng multi-key sort (không dùng hybrid cộng điểm):
  - Sort `kshell DESC`, tie-break `degree DESC`, tie-break cuối `node_id ASC`.
  - Assign `kshell_rank` (1..N) và `kshell_rank_norm = (N - rank) / (N - 1)`.
  - Guard case `N=1`: gán `kshell_rank_norm = 1.0` để tránh chia 0.
- Tính brokerage metric `inter_community_edge_ratio` cho từng node.
- Phân tích community-language alignment:
  - Tính `NMI(community_id, language)`.
  - Tính `purity_score` theo community.
- Phân tích core-periphery theo `life_time`:
  - Định nghĩa core/periphery theo ngưỡng `kshell` đã chốt.
  - So sánh `life_time` bằng Mann-Whitney U + Cliff's Delta.
- Xuất artifact Stage 2:
  - `outputs/stage2/community_labels.parquet` (best partition).
  - `outputs/stage2/louvain_stability_report.json` (Q + NMI + runtime).
  - `outputs/stage2/community_metrics.csv` (tổng hợp cộng đồng).
  - `outputs/stage2/attribute_community_analysis.csv`.
  - `outputs/stage2/core_lifetime_tests.csv`.

7. **Data integrity + consistency checks (Day 4):**
   - Kiểm tra coverage:
     - Centrality/community/kshell phủ 100% active nodes.
   - Kiểm tra NA bắt buộc:
     - Không có NA trong `degree`, `pagerank`, `betweenness`, `kshell`.
   - Kiểm tra khóa join:
     - `node_id` unique trong mọi bảng stage1/stage2.
   - Cập nhật:
     - `artifact_index.csv`, `params.json`, `metrics.json` cho cả stage1 và stage2.

8. **Review nội bộ + Definition of Ready cho Week 5 (cuối Day 5):**
   - Tổ chức review nhanh 30-45 phút:
     - Check logic centrality divergence.
     - Check Louvain stability và lựa chọn partition chính.
   - Chốt các đầu vào bắt buộc cho Week 5:
     - `centrality_table.parquet` đã có `degree/pagerank/betweenness/kshell`.
     - `community_labels.parquet` và `inter_community_edge_ratio` sẵn sàng cho typology/brokerage test.
   - Tạo risk note:
     - Runtime betweenness nếu vượt SLA nội bộ.
     - NMI thấp cần cảnh báo là target vận hành nội bộ, không phải ngưỡng học thuật bắt buộc.

**Execution runbook tối thiểu (để team chạy thống nhất):**

1. Chạy Stage 1 theo thứ tự:

- `python run_all.py --stage stage1` (hoặc target tương đương trong Make/bash).
- Verify có `outputs/stage1/centrality_table.parquet`, `outputs/stage1/centrality_metrics.json`, `outputs/stage1/rq1_ranking_metrics.csv`.

2. Chạy Stage 2:

- `python run_all.py --stage stage2`.
- Verify có `outputs/stage2/community_labels.parquet`, `outputs/stage2/louvain_stability_report.json`, `outputs/stage2/community_metrics.csv`.

3. Chạy kiểm tra integrity cuối tuần:

- Không NA ở cột bắt buộc.
- 100% node coverage cho centrality + community + kshell.
- `node_id` unique trong các bảng chính.

**Fail-fast criteria (dừng và xử lý ngay):**

- `betweenness_backend` không được ghi trong `outputs/stage1/centrality_metrics.json`.
- NetworKit unavailable nhưng không có fallback decision entry trong `docs/experiment_registry.md`.
- Nếu NetworkX fallback được dùng trên graph lớn (num_nodes > 50k):
  - `k_pivots < 2000` -> fail-fast.
  - `k_pivots < 1000` -> hard error (guard bắt buộc).
- Betweenness runtime vượt ngân sách nội bộ mà chưa có risk note/log giải thích.
- Louvain chưa đủ 10 runs hoặc thiếu báo cáo stability (Q + NMI).
- Không ghi decision Eigenvector redundancy vào `experiment_registry.md`.
- K-shell tie-breaking triển khai sai rule (dùng hybrid score thay vì multi-key sort deterministic).
- Bảng stage1/stage2 thiếu `metrics.json` hoặc `params.json` hoặc `artifact_index.csv`.

**Handover package Week 3-4 (bắt buộc trước khi sang Week 5):**

- `outputs/stage1/centrality_table.parquet`, `outputs/stage1/centrality_metrics.json`, `outputs/stage1/rq1_ranking_metrics.csv`.
- `outputs/stage2/community_labels.parquet`, `outputs/stage2/louvain_stability_report.json`, `outputs/stage2/community_metrics.csv`, `outputs/stage2/attribute_community_analysis.csv`, `outputs/stage2/core_lifetime_tests.csv`.
- Kết quả test tie-breaking k-shell trong `tests/test_sis.py` (pass) + note quy tắc trong `docs/implementation_notes.md`.
- Risk note runtime/NMI + decision Eigenvector redundancy đã ghi trong `docs/experiment_registry.md`.

**Deliverable:**

- `outputs/stage1/centrality_metrics.json`, `outputs/stage1/centrality_table.parquet`.
- `outputs/stage1/rq1_ranking_metrics.csv`.
- `outputs/stage2/community_labels.parquet`, `outputs/stage2/louvain_stability_report.json`.
- `outputs/stage2/community_metrics.csv`.
- `outputs/stage2/attribute_community_analysis.csv`, `outputs/stage2/core_lifetime_tests.csv`.
- Bảng metrics RQ1 sơ bộ + log quyết định Eigenvector redundancy.

**Quality Gate:**

- Không có NA trong centrality columns.
- Community label phủ 100% active nodes.
- Eigenvector redundancy decision + NMI stability ghi trong `experiment_registry.md`.
- Có đủ `metrics.json`, `params.json`, `artifact_index.csv` cho cả stage1 và stage2.
- Louvain chạy đủ 10 seeds; nếu `NMI < 0.85` phải có cảnh báo rõ trong report (target vận hành nội bộ).

---

### 🟡 Week 5: Stage 3 (SIS + Typology + Robustness + Null Model Comparison)

**Mục tiêu:**

- Sinh SIS chính thức theo công thức equal-weight đã khóa từ Week 1-2.
- Tạo nhãn typology 2×2 ổn định, có thể tái tạo, và sẵn sàng cho validation ở Week 6.
- **Validate typology against null model** để kiểm tra nguy cơ artifact từ degree distribution.
- Hoàn tất robustness/sensitivity để lượng hóa độ bền của Hidden Influencer set.
- Chuẩn bị power analysis và sampling plan làm input trực tiếp cho single-seed IC ở Week 6.

**Yêu cầu kỹ thuật + thông số bắt buộc (để implement nhất quán):**

- **Stage 3 parameters mặc định trong `src/config/experiment.yaml`:**
  - `topk_threshold: 0.20` (sensitivity: `[0.15, 0.20, 0.25]`).
  - `weight_variants`: equal-weight `[1/3, 1/3, 1/3]` + PR-heavy + Bet-heavy + KS-heavy.
  - `rank_norm_method: percentile_rank` với miền chuẩn hóa `[0, 1]`.
- **Rule gán nhóm typology (tránh sai lệch do tie):**
  - Dùng percentile-rank theo `sis_rank` và `views` thay vì raw value.
  - Tie-break theo `node_id` tăng dần khi cần cắt top-k.
  - **Canonical split**: typology luôn cắt theo `topk SIS-rank` x `topk views-rank`.
  - Cụm từ "innermost k-shell threshold" trong proposal được operationalize thành SIS-rank (vì k-shell là 1 thành phần trong SIS), và phải log trong `docs/experiment_registry.md`.
- **Null-model control parameters:**
  - `null_model_type: configuration_model`.
  - `null_sample_ratio: 0.20` (hoặc full graph nếu runtime cho phép).
  - `null_seed` phải ghi rõ trong `params.json`.
  - `null_sample_repr_check: required` trước khi chạy so sánh null vs real.
- **Power analysis defaults:**
  - `alpha: 0.05`, `power_target: 0.80`.
  - Effect-size grid tối thiểu: `[0.2, 0.5, 0.8]` để chốt khoảng sample khả thi.
  - Không dùng trực tiếp power của t-test cho kết luận MWU nếu chưa có note hiệu chỉnh.
- **Schema tối thiểu cho outputs Stage 3:**
  - `sis_table.parquet`: `node_id`, `sis_score`, `sis_rank`, `sis_group`.
  - `typology_labels.parquet`: `node_id`, `views_group`, `sis_group`, `typology_label`.
  - `robustness_summary.csv`: `variant_name`, `hidden_set_size`, `jaccard_vs_base`.
  - `null_model_comparison.csv`: `metric_name`, `real_value`, `null_value`, `delta`, `warning_flag`.
  - Rows tối thiểu trong `null_model_comparison.csv`:
    - `degree_ks_stat`, `degree_ks_p`, `clustering_coefficient`, `avg_path_length_estimate`, `pct_hidden_influencer`.

**Kế hoạch thực thi chi tiết (theo thứ tự chạy):**

1. **Pre-flight check Stage 3 (Day 1 - buổi 1):**
   - Xác nhận input từ Week 3-4 đã đủ:
     - `outputs/stage1/centrality_table.parquet`.
     - `outputs/stage2/community_labels.parquet`.
     - Cột bắt buộc: `node_id`, `pagerank`, `betweenness`, `kshell`, `views`.

- Xác nhận config Stage 3 trong `src/config/experiment.yaml`:
  - `topk_threshold` cho typology (mặc định 20%).
  - `weight_variants` cho sensitivity.
  - `seed_stage3`.
- Tạo thư mục output:
  - `outputs/stage3/`.
  - `data/processed/`.

2. **CRITICAL - Compute SIS (Day 1 - buổi 2):**

   ```python
   sis_score = (rank_norm(pagerank) + rank_norm(betweenness) + rank_norm(kshell)) / 3
   ```

   - Dùng rank normalization nhất quán toàn tập node active.
   - Tạo các cột tối thiểu:
     - `sis_score`, `sis_rank`, `sis_group`.
   - Export:
     - `data/processed/sis_table.parquet`.
     - `outputs/stage3/metrics.json` (kèm seed, runtime, config hash).

3. **Build 2×2 Typology (Day 2):**
   - Split top-20% SIS & top-20% views → 4 nhóm:
     - True Influencer (high SIS, high views)
     - Hidden Influencer (high SIS, low views)
     - Overrated (low SIS, high views)
     - Non-Influencer (low SIS, low views)
   - Kiểm tra phân phối nhóm:
     - % mỗi nhóm.
     - Tổng 4 nhóm = 100% node active.
   - Export:
     - `data/processed/typology_labels.parquet`.
     - `outputs/stage3/typology_distribution.csv`.

4. **Kiểm định cấu trúc typology (Day 2-3):**
   - So sánh Hidden vs Overrated trên:
     - `betweenness` rank.
     - `inter_community_edge_ratio`.
   - Test thống kê:
     - Mann-Whitney U.
     - Effect size (Rank-biserial, Cliff's Delta).
   - Export:
     - `outputs/stage3/typology_structural_tests.csv`.

5. **Robustness / Sensitivity Analysis (Day 3-4):**
   - PR-heavy `[0.5, 0.25, 0.25]`, Bet-heavy `[0.25, 0.5, 0.25]`, KS-heavy `[0.25, 0.25, 0.5]`.
   - Threshold sensitivity: 15% / 20% / 25%.
   - **+ Eigenvector variant** `[0.25, 0.25, 0.25, 0.25]` nếu không redundant.
   - Tính **Jaccard stability** của Hidden set giữa các biến thể.
   - Tổng hợp sensitivity matrix:
     - Trục 1: weight variants.
     - Trục 2: threshold variants.
   - Export:
     - `outputs/stage3/robustness_summary.csv`.
     - `outputs/stage3/sensitivity_matrix.csv`.

6. **CRITICAL - Null model comparison (Day 4):**
   - Dùng `src/graph/null_model.py` để tạo configuration-model graph từ degree sequence.
   - Trước khi chạy null branch với sample 20%, bắt buộc kiểm tra representativeness:
     - So sánh degree distribution sample vs full (KS distance + quantile check).
     - Nếu không đạt tiêu chí, tăng `null_sample_ratio` hoặc chuyển full-graph null run.
   - Chạy lại pipeline SIS/typology trên null graph (có thể sample 20% nodes để tối ưu thời gian).
   - So sánh null vs real:
     - Degree distribution (KS stat + p-value).
     - Global clustering/transitivity (real vs null).
     - Average path length estimate (sampling-based, nêu rõ phạm vi tính trên graph khả dụng/LCC).
     - `% Hidden Influencers`.
     - Phân phối group imbalance.
   - Quy tắc cảnh báo:
     - Nếu `degree_ks_stat > 0.1` -> warning `HIGH_KS_DISTANCE`.
     - Nếu `clustering_delta < 0.05` -> warning `LOW_CLUSTERING_DELTA`.
     - Nếu `|apl_real_estimate - apl_null_estimate| < 0.5` -> warning `SIMILAR_APL`.
     - Nếu `|pct_hidden_real - pct_hidden_null| < 0.02` -> warning `possible degree-distribution artifact`.
   - Export:
     - `outputs/stage3/null_sample_representativeness.csv`.
     - `outputs/stage3/null_model_comparison.csv`.
     - `outputs/stage3/null_model_warnings.md`.

7. **Power analysis + sampling plan cho Week 6 (Day 5):**
   - Dùng 1 trong 2 cách cho Mann-Whitney U:
     - Simulation-based power với dữ liệu pilot distribution, hoặc
     - TTestIndPower + ARE factor `0.955` như bound gần đúng.
   - Bắt buộc ghi rõ phương pháp chọn power trong `outputs/stage3/power_assumption_note.md` và `docs/assumptions_limitations.md`.
   - Chốt phạm vi sample mỗi nhóm cho single-seed IC:
     - Mục tiêu 100-200 node/nhóm (điều chỉnh theo power và ngân sách runtime).
   - Export:
     - `outputs/stage3/power_analysis.csv`.
     - `outputs/stage3/power_assumption_note.md`.
     - Cập nhật kế hoạch vào `outputs/stage3/metrics.json` và `docs/experiment_registry.md`.

8. **Review nội bộ + Definition of Ready cho Week 6 (cuối Day 5):**
   - Review 30-45 phút với checklist:
     - SIS reproducibility.
     - Typology distribution hợp lý.
     - Kết quả robustness và null-model đã có kết luận rõ.
   - Chốt input bàn giao cho Week 6:
     - `typology_labels.parquet`.
     - `robustness_summary.csv`.
     - `null_model_comparison.csv`.
     - `power_analysis.csv` + sampling plan.
   - Nếu có warning artifact/null-model, phải ghi rõ trong note để tránh overclaim ở RQ2.

**Deliverable:**

- `data/processed/sis_table.parquet`, `data/processed/typology_labels.parquet`.
- `outputs/stage3/robustness_summary.csv` (Jaccard stability).
- `outputs/stage3/sensitivity_matrix.csv`.
- `outputs/stage3/null_sample_representativeness.csv`.
- `outputs/stage3/null_model_comparison.csv`.
- `outputs/stage3/null_model_warnings.md`.
- `outputs/stage3/power_analysis.csv`.
- `outputs/stage3/power_assumption_note.md`.
- Power analysis + sampling plan ghi trong `outputs/stage3/metrics.json`.

**Acceptance Criteria (Week 5):**

- SIS reproducible khi chạy lại cùng seed.
- Typology labels đầy đủ cho 100% node active, không missing group.
- Null model analysis hoàn tất, có kết luận và warning (nếu có) trong log/report.
- Null model có đủ kiểm tra cấu trúc: degree KS + clustering delta + APL estimate.
- Null sample representativeness check pass hoặc có lý do tăng sample/full-run đã log.
- Jaccard stability >= 0.7 giữa variants (target vận hành nội bộ, không phải ngưỡng học thuật bắt buộc).
- Power analysis hoàn tất và có sample-size plan khả thi cho Week 6.
- Phương pháp power cho MWU đã documented (simulation hoặc ARE-adjusted bound).
- Có đủ `metrics.json`, `params.json`, `artifact_index.csv` cho stage3.

**Definition of Ready cho Week 6:**

- `outputs/stage3_ic_calibration/` đã sẵn cấu trúc thư mục, config p-grid đã khai báo trong `src/config/experiment.yaml`.
- `typology_labels.parquet` và sampling frame sẵn sàng để rút mẫu stratified theo nhóm.
- `power_analysis.csv` đã chốt cỡ mẫu mục tiêu và giới hạn runtime chấp nhận được.
- Risk note đã ghi rõ: cảnh báo null-model artifact (nếu có), và cách diễn giải RQ2 tương ứng.

**Execution runbook tối thiểu (để team chạy thống nhất):**

1. Chạy Stage 3 theo đúng thứ tự:

- `python run_all.py --stage stage3`.
- Verify có `data/processed/sis_table.parquet`, `data/processed/typology_labels.parquet`, `outputs/stage3/robustness_summary.csv`, `outputs/stage3/sensitivity_matrix.csv`.

2. Chạy null-model branch:

- `python run_all.py --stage stage3 --mode null_model` (hoặc lệnh tương đương theo runner).
- Verify có `outputs/stage3/null_model_comparison.csv`.

3. Chạy power-analysis:

- `python run_all.py --stage stage3 --mode power_analysis` (hoặc lệnh tương đương).
- Verify có `outputs/stage3/power_analysis.csv` và sampling plan đã ghi trong `metrics.json`.

**Fail-fast criteria (dừng và xử lý ngay):**

- Typology không phủ đủ 100% active nodes hoặc tổng tỉ lệ 4 nhóm khác 100%.
- `sis_score` hoặc `sis_rank` có NA/inf.
- Null-model comparison không tái tạo được hoặc không có kết luận cảnh báo/không cảnh báo rõ ràng.
- Jaccard stability quá thấp trên đa số variant và chưa có risk note giải thích.
- Thiếu `metrics.json` hoặc `params.json` hoặc `artifact_index.csv` cho stage3.

**Handover package Week 5 (bắt buộc trước khi sang Week 6):**

- `data/processed/sis_table.parquet`, `data/processed/typology_labels.parquet`.
- `outputs/stage3/robustness_summary.csv`, `outputs/stage3/sensitivity_matrix.csv`, `outputs/stage3/null_model_comparison.csv`.
- `outputs/stage3/power_analysis.csv`, `outputs/stage3/metrics.json` (kèm sampling plan).
- Entry trong `docs/experiment_registry.md` cho decision Eigenvector variant và null-model warning status.

---

### 🟢 Week 6: IC Calibration + Single-seed Validation (RQ2)

**Mục tiêu:**

- **Calibrate IC parameters tối ưu** trước khi chạy validation để tránh ceiling/floor effect.
- Hoàn tất single-seed IC validation cho các nhóm typology theo sampling plan đã chốt ở Week 5.
- Trả lời RQ2 bằng evidence đầy đủ: mean reach, kiểm định thống kê, multiple-testing correction, effect size.
- Đảm bảo pipeline validation có thể tái tạo với seed cố định và runtime hợp lý.

**Yêu cầu kỹ thuật + thông số bắt buộc (để implement nhất quán):**

- **Calibration parameters mặc định:**
  - `p_grid: [0.01, 0.03, 0.05, 0.08]`.
  - `pilot_subgraph_ratio: 0.20`.
  - `calibration_runs_per_seed: 30` (tối thiểu) cho pilot.
- **Rule chọn `p_calibrated` (deterministic):**
  - Ưu tiên `reach_ratio` trong `[0.08, 0.25]`.
  - Nếu nhiều ứng viên: chọn `std_reach` thấp hơn.
  - Nếu vẫn hòa: chọn `p` nhỏ hơn để tránh overspread.
  - **Bổ sung κ-target method (song song, literature-grounded):**
    - Load `mean_degree` từ `outputs/stage0_data_quality/metrics.json`.
    - Với mỗi `κ ∈ {1, 2, 3}` tính `p_kappa = κ / mean_degree` (clamp `[0.001, 0.5]`).
    - So sánh `p_kappa` với candidates từ reach-range pilot.
    - Selection rule: ưu tiên intersection (nếu có), nếu không thì fallback reach-range và ghi rõ rationale.

  Ví dụ cấu hình trong `src/config/experiment.yaml`:

  ```yaml
  ic_calibration:
    # Reach-range heuristic
    reach_range_low: 0.08
    reach_range_high: 0.25
    p_grid: [0.01, 0.03, 0.05, 0.08]
    pilot_subgraph_ratio: 0.20
    calibration_runs_per_seed: 30

    # κ-target method (Guille et al. 2013)
    kappa_target_list: [1, 2, 3]
    p_selection_rule: "intersection_prefer_kappa"
  ```

- **Single-seed validation parameters:**
  - `runs_per_seed: 50` (bắt buộc).
  - Sampling stratified theo `typology_label`, bám power-plan (mục tiêu 100-200 node/nhóm).
  - `parallel_backend: joblib` với seed cố định theo run.
- **Statistical defaults:**
  - `alpha: 0.05`, BH-FDR `q: 0.05`.
  - Report bắt buộc: p-value đã chỉnh + effect size (rank-biserial, Cliff's delta).
- **Schema tối thiểu cho outputs Stage 4:**
  - `sampling_frame.csv`: `node_id`, `typology_label`, `kshell`, `sampling_seed`, `inclusion_flag`.
  - `single_seed_node_summary.csv`: `node_id`, `mean_reach`, `std_reach`, `ci95_low`, `ci95_high`.
  - `rq2_stats_tests.csv`: `comparison`, `test_name`, `p_raw`, `p_bh`, `effect_size`, `effect_size_type`.

**Kế hoạch thực thi chi tiết (theo thứ tự chạy):**

1. **Pre-flight check cho Week 6 (Day 1 - buổi 1):**
   - Xác nhận input bắt buộc từ Week 5:
     - `data/processed/typology_labels.parquet`.
     - `outputs/stage3/power_analysis.csv`.
     - `outputs/stage3/metrics.json` (sampling plan + seed info).

- Xác nhận config calibration/validation trong `src/config/experiment.yaml`:
  - `p_grid = {0.01, 0.03, 0.05, 0.08}`.
  - `runs_per_seed = 50`.
  - `sample_per_group` theo power plan.
- Tạo thư mục output:
  - `outputs/stage3_ic_calibration/`.
  - `outputs/stage4_single_seed/`.

2. **CRITICAL - IC calibration (Day 1 - buổi 2):**
   - Chạy pilot IC trên subgraph đại diện (20% nodes).
   - Với mỗi `p` trong grid, tính:
     - `mean_reach`, `std_reach`, `reach_ratio = mean_reach / N`.
     - Runtime trung bình mỗi run.
   - Quy tắc chọn `p_calibrated`:
     - Ưu tiên `reach_ratio` nằm trong `[8%, 25%]` (target vận hành nội bộ).
     - Nếu nhiều `p` hợp lệ, chọn `p` có variance ổn định hơn (std thấp hơn) và runtime hợp lý hơn.
   - **2b. THÊM — κ-target computation (chạy song song, không tốn thêm pilot runtime):**
     - Load `mean_degree` từ `outputs/stage0_data_quality/metrics.json`.
     - Tính `p_kappa = κ / mean_degree` cho `κ ∈ {1, 2, 3}`.
     - Nếu `p_kappa` gần một `p_grid` candidate nằm trong reach-range (khoảng cách < 0.02): chọn candidate đó và note “intersection — validated by κ-target”.
     - Nếu không: dùng reach-range heuristic và ghi rõ vì sao κ-target không áp dụng được.
   - Export:
     - `outputs/stage3_ic_calibration/calibration_results.csv`.
     - `outputs/stage3_ic_calibration/calibration_summary.json`.
     - `calibration_summary.json` phải gồm thêm: `kappa_target_map`, `p_kappa_computed`, `method_used`, `selection_rationale`, `mean_degree_used`.
   - Ghi rationale chọn `p` vào `docs/experiment_registry.md`.

3. **Sampling stratified cho single-seed validation (Day 2):**
   - Rút mẫu theo từng nhóm typology dựa trên power plan:
     - Mục tiêu 100-200 node/nhóm (hoặc theo giới hạn đã chốt).
   - Ghi lại sampling frame:
     - Danh sách node được chọn theo nhóm.
     - Phân bố `kshell` theo từng typology group để kiểm tra independence condition.
     - Seed sampling.
   - Export:
     - `outputs/stage4_single_seed/sampling_frame.csv`.

4. **Single-seed IC execution (Day 2-3):**
   - Chạy single-seed IC cho từng node mẫu với `runs_per_seed = 50`.
   - Dùng `joblib.Parallel(n_jobs=-1)` để tối ưu runtime.
   - Thu thập per-node metrics:
     - `mean_reach`, `std_reach`, `ci95_low`, `ci95_high`.
   - Export raw results:
     - `outputs/stage4_single_seed/single_seed_raw_runs.parquet`.
     - `outputs/stage4_single_seed/single_seed_node_summary.csv`.

5. **So sánh nhóm + kiểm định thống kê (Day 3-4):**
   - So sánh chính:
     - Hidden vs Overrated.
     - Hidden vs True.
     - True vs Overrated.
   - Kiểm định:
     - Mann-Whitney U.
     - Benjamini-Hochberg correction cho multiple comparisons.
   - Effect size:
     - Rank-biserial.
     - Cliff's Delta.
   - Export:
     - `outputs/stage4_single_seed/rq2_hidden_validation.csv`.
     - `outputs/stage4_single_seed/rq2_stats_tests.csv`.

6. **Runtime + reproducibility audit (Day 4):**
   - Kiểm tra điều kiện tái lập:
     - Re-run kiểm thử trên một sample nhỏ với cùng seed, so sánh sai lệch cho phép.
   - Ghi log runtime:
     - Runtime baseline (không parallel).
     - Runtime parallel.
   - Export:
     - `logs/timing/week6_single_seed_timing.csv`.
     - `outputs/stage4_single_seed/repro_check.json`.

7. **Review nội bộ + Definition of Ready cho Week 7 (cuối Day 5):**
   - Review 30-45 phút với checklist:
     - Calibration rationale rõ và nhất quán.
     - Kết luận RQ2 có đủ p-value đã chỉnh và effect size.
     - Kết quả không overclaim khi effect nhỏ hoặc chồng lấn CI lớn.
   - Chốt input bàn giao cho Week 7:
   - `p_calibrated` đã khóa trong `src/config/experiment.yaml`.
   - File kết quả single-seed hoàn chỉnh và reproducible.
   - Risk note về runtime ngân sách cho multi-seed 900+ runs.

**Deliverable:**

- `outputs/stage3_ic_calibration/calibration_results.csv`.
- `outputs/stage3_ic_calibration/calibration_summary.json`.
- `outputs/stage4_single_seed/rq2_hidden_validation.csv` (mean reach + effect sizes).
- `outputs/stage4_single_seed/sampling_frame.csv`.
- `outputs/stage4_single_seed/single_seed_node_summary.csv`.
- `outputs/stage4_single_seed/rq2_stats_tests.csv`.
- Timing log (runtime comparison baseline vs parallelized).

**Acceptance Criteria (Week 6):**

- Calibration p ghi vào `src/config/experiment.yaml`.
- 50 runs per seed x sample size hợp lệ + seed logged.
- Có đủ kết quả cho tất cả so sánh nhóm chính (Hidden/Overrated/True).
- Multiple testing correction (BH) đã áp dụng và report cùng effect size.
- `rq2_hidden_validation.csv` có thể truy vết về raw runs và sampling frame.
- Repro check pass trên sample test với cùng seed.
- Có đủ `metrics.json`, `params.json`, `artifact_index.csv` cho `stage3_ic_calibration` và `stage4_single_seed`.

**Definition of Ready cho Week 7:**

- `p_calibrated` đã khóa, không thay đổi trong benchmark multi-seed trừ khi có issue log chính thức.
- Nếu `p_calibrated != 0.01`, bắt buộc tạo protocol-deviation note (so với proposal gốc) trong `docs/experiment_registry.md` trước khi chạy Week 7.
- Budget runtime cho Week 7 đã ước lượng từ timing Week 6.
- Bộ seed strategies và config `k=50` đã có skeleton chạy thử thành công.
- Artifact Week 6 đầy đủ: calibration + validation + stats + timing + reproducibility.

**Execution runbook tối thiểu (để team chạy thống nhất):**

1. Chạy calibration trước:

- `python run_all.py --stage stage3_ic_calibration`.
- Verify có `outputs/stage3_ic_calibration/calibration_results.csv`, `outputs/stage3_ic_calibration/calibration_summary.json` và `p_calibrated` đã ghi vào config/registry.

2. Chạy single-seed validation:

- `python run_all.py --stage stage4_single_seed`.
- Verify có `outputs/stage4_single_seed/sampling_frame.csv`, `outputs/stage4_single_seed/single_seed_node_summary.csv`, `outputs/stage4_single_seed/rq2_hidden_validation.csv`, `outputs/stage4_single_seed/rq2_stats_tests.csv`.

3. Chạy reproducibility check:

- Re-run sample test với cùng seed.
- Verify có `outputs/stage4_single_seed/repro_check.json` và timing log `logs/timing/week6_single_seed_timing.csv`.

**Fail-fast criteria (dừng và xử lý ngay):**

- `p_calibrated` chưa được chốt nhưng đã chạy full validation.
- `outputs/stage3_ic_calibration/calibration_summary.json` thiếu các fields: `kappa_target_map`, `method_used`, `selection_rationale`.
- `method_used = reach_range` nhưng không có note giải thích vì sao κ-target không áp dụng được trong `docs/experiment_registry.md`.
- Số run thực tế nhỏ hơn `runs_per_seed = 50` cho bất kỳ group/seed bucket nào.
- Sampling frame không khớp power plan hoặc mất cân bằng nghiêm trọng không có note giải thích.
- Thiếu BH correction hoặc thiếu effect size trong báo cáo RQ2.
- Kết quả `rq2_hidden_validation.csv` không truy vết được về raw runs.

**Handover package Week 6 (bắt buộc trước khi sang Week 7):**

- `outputs/stage3_ic_calibration/calibration_results.csv`, `outputs/stage3_ic_calibration/calibration_summary.json`.
- `outputs/stage4_single_seed/sampling_frame.csv`, `outputs/stage4_single_seed/single_seed_node_summary.csv`.
- `outputs/stage4_single_seed/rq2_hidden_validation.csv`, `outputs/stage4_single_seed/rq2_stats_tests.csv`.
- `logs/timing/week6_single_seed_timing.csv`, `outputs/stage4_single_seed/repro_check.json`.
- Entry trong `docs/experiment_registry.md` về rationale chọn `p_calibrated`, protocol-deviation status so với `p=0.01` của proposal, và risk note runtime cho Week 7.

---

### 🔵 Week 7: Multi-seed IC Benchmark (RQ3) + IC Sensitivity Analysis

**Mục tiêu:**

- Benchmark 6 seeding strategies chính trong cùng một protocol chuẩn để so sánh công bằng.
- **Test robustness qua 3 giá trị p** nhằm đánh giá độ ổn định thứ hạng chiến lược.
- Trả lời RQ3 bằng evidence đầy đủ: mean reach, CI, kiểm định thống kê giữa các chiến lược.
- Khóa bộ kết quả benchmark để bàn giao cho phần báo cáo và ML narrative ở Week 8-9.

**Yêu cầu kỹ thuật + thông số bắt buộc (để implement nhất quán):**

- **Benchmark parameters mặc định trong `src/config/experiment.yaml`:**
  - `k_seeds: 50`.
  - `runs_main: 300`.
  - `runs_sensitivity: 50`.
  - `p_main: p_calibrated`.
  - `runs_main` là hằng số benchmark, không được inherit từ `runs_per_seed` của single-seed stage.
- **Sensitivity configuration mặc định:**
  - `p_sensitivity_grid: [p_calibrated - 0.01, p_calibrated, p_calibrated + 0.01]`.
  - Nếu biên dưới < 0 thì clamp về 0.01 và bắt buộc log quyết định.
- **Protocol alignment với proposal:**
- Proposal baseline benchmark dùng `p=0.01`.
- Nếu `p_main != 0.01`, bắt buộc tạo `outputs/stage5_multi_seed/benchmark_protocol_note.md` + entry trong `docs/experiment_registry.md` với các trường: `proposal_value`, `implemented_value`, `rationale`, `comparability_impact`.
- **Rule chọn seed set (deterministic):**
  - Top-k theo score giảm dần cho từng strategy.
  - Tie-break theo `node_id` tăng dần.
  - Random strategy phải có `random_seed_strategy` riêng và log trong `params.json`.
- **Statistical defaults cho RQ3:**
  - Global test: Kruskal-Wallis.
  - Post-hoc: Dunn + correction BH-FDR `q=0.05`.
  - Effect size phải report cho cặp trọng tâm.
- **Schema tối thiểu cho outputs Stage 5:**
  - `rq3_strategy_benchmark.csv`: `strategy`, `mean_reach`, `std_reach`, `ci95_low`, `ci95_high`, `runtime_sec`.
  - `rq3_sensitivity_p.csv`: `strategy`, `p_value`, `mean_reach`, `rank`.
  - `ablation_kshell.csv` (nice-to-have): `strategy`, `mean_reach`, `std_reach`, `ci95_low`, `ci95_high`, `delta_vs_kshell_seeding`, `note`.
  - `rq3_rank_stability.csv`: `strategy`, `rank_main`, `rank_p_minus`, `rank_p_plus`, `max_rank_shift`.
  - `rq3_stats_tests.csv`: `comparison`, `test_name`, `p_raw`, `p_bh`, `effect_size`, `effect_size_type`.

**Kế hoạch thực thi chi tiết (theo thứ tự chạy):**

1. **Pre-flight check cho Week 7 (Day 1 - buổi 1):**
   - Xác nhận input từ Week 6:
   - `p_calibrated` đã khóa trong `src/config/experiment.yaml`.
   - `outputs/stage4_single_seed/rq2_hidden_validation.csv` (để tham chiếu consistency).
   - Timing budget từ `logs/timing/week6_single_seed_timing.csv`.
   - Xác nhận tham số benchmark:
     - `k_seeds = 50`.
     - `runs_main = 300`.
     - `runs_sensitivity = 50`.
     - `runs_main` không inherit từ single-seed config.
     - Protocol alignment với proposal (`p=0.01`) đã được ghi nhận nếu có lệch.
   - Tạo thư mục output:
     - `outputs/stage5_multi_seed/`.
     - `logs/timing/` (nếu chưa có).

2. **Implement và freeze seed strategies (Day 1 - buổi 2):**
   - Hoàn thiện `src/simulation/seed_strategies.py` với 6 chiến lược chính:
     - Random.
     - Views-based.
     - Degree-based.
     - PageRank-based.
     - Betweenness-based.
     - k-shell-based.
   - Nice-to-have ablation (tách riêng, không tính vào 6 chiến lược chính):
     - `SIS_PR_Bet` (SIS từ PageRank + Betweenness, không dùng k-shell).
   - Chuẩn hóa rule chọn seed:
     - Top-k theo score giảm dần.
     - Tie-break bằng `node_id` tăng dần để tái lập.
   - Export danh sách seed để audit:
     - `outputs/stage5_multi_seed/seed_sets_k50.csv`.

3. **Main benchmark run (Day 2):**
   - Chạy multi-seed IC với:
     - `p = p_calibrated`.
     - `k = 50`.
     - `runs = 300` cho mỗi strategy.
   - Thu thập metrics theo strategy:
     - `mean_reach`, `std_reach`, `ci95_low`, `ci95_high`.
     - Runtime tổng và runtime trung bình mỗi run.
   - Export:
     - `outputs/stage5_multi_seed/rq3_strategy_benchmark.csv`.
     - `outputs/stage5_multi_seed/rq3_main_raw_runs.parquet`.

4. **Sensitivity analysis theo p (Day 3):**
   - Chạy IC với 3 giá trị p:
     - `p_calibrated`.
     - `p_calibrated - 0.01`.
     - `p_calibrated + 0.01`.
   - Cấu hình:
     - 6 strategies x 50 runs x 3 p = 900 runs total.
   - Tính rank theo từng p và so sánh:
     - Rank shift mỗi strategy.
     - Rank correlation giữa các cấu hình p.
   - Export:
     - `outputs/stage5_multi_seed/rq3_sensitivity_p.csv`.
     - `outputs/stage5_multi_seed/rq3_rank_stability.csv`.

5. **Statistical inference cho RQ3 (Day 3-4):**
   - So sánh toàn cục:
     - Kruskal-Wallis trên reach distributions.
   - So sánh cặp:
     - Post-hoc Dunn test (có correction).
   - Effect size cho cặp trọng tâm:
     - k-shell vs random.
     - k-shell vs views-based.
   - Export:
     - `outputs/stage5_multi_seed/rq3_stats_tests.csv`.

6. **Visualization + reporting artifacts (Day 4):**
   - Tạo hình bắt buộc:
     - `fig_ic_strategy_comparison.png` (box/violin + CI).
     - `fig_sensitivity_heatmap.png` (rank stability theo p).
   - Tạo bảng tóm tắt cho report:
     - `reports/tables/table_rq3_ic_benchmark.csv`.

7. **Runtime + reproducibility audit (Day 4-5):**
   - Kiểm tra replay trên sample nhỏ để xác nhận tính tái lập với seed cố định.
   - Ghi timing:
     - Main benchmark timing.
     - Sensitivity timing.
   - Export:
     - `logs/timing/week7_multi_seed_timing.csv`.
     - `outputs/stage5_multi_seed/repro_check.json`.

8. **Review nội bộ + Definition of Ready cho Week 8-9 (cuối Day 5):**
   - Review 30-45 phút với checklist:
     - RQ3 evidence đủ mạnh (mean/CI + test + effect size).
     - Rank stability đã được giải thích khi p thay đổi.
   - Nếu có chạy ablation `SIS_PR_Bet`, chỉ report appendix/footnote; không dùng thay thế kết luận RQ3 chính.
   - Không overclaim khi chênh lệch nhỏ hoặc CI chồng lấn.
   - Chốt bàn giao cho Week 8-9:
     - Bảng benchmark chính đã cố định.
     - Kết quả sensitivity và risk note về phụ thuộc p.
     - Figure/table đã sẵn để viết report.

**Deliverable:**

- `outputs/stage5_multi_seed/rq3_strategy_benchmark.csv`.
- `outputs/stage5_multi_seed/rq3_sensitivity_p.csv`.
- `outputs/stage5_multi_seed/ablation_kshell.csv` (nếu timeline cho phép, chỉ appendix).
- `outputs/stage5_multi_seed/rq3_rank_stability.csv`.
- `outputs/stage5_multi_seed/rq3_stats_tests.csv`.
- `outputs/stage5_multi_seed/seed_sets_k50.csv`.
- `outputs/stage5_multi_seed/benchmark_protocol_note.md` (nếu `p_main != 0.01`).
- Figures: `fig_ic_strategy_comparison.png`, `fig_sensitivity_heatmap.png`.
- Timing + reproducibility: `logs/timing/week7_multi_seed_timing.csv`, `outputs/stage5_multi_seed/repro_check.json`.

**Acceptance Criteria (Week 7):**

- 300 runs (main) + 900 runs (sensitivity) hoàn tất + seed logged.
- `p_calibrated` được giữ cố định cho main benchmark, mọi thay đổi p chỉ nằm trong sensitivity branch.
- `runs_main = 300` được giữ cố định cho benchmark report và không bị override bởi config stage khác.
- Rank stability đã được định lượng và có file giải thích rank shift.
- Nếu chạy ablation `SIS_PR_Bet`, kết quả được tách riêng appendix và không ghi đè narrative kết luận chính.
- Có đủ kiểm định thống kê (global + post-hoc) và effect size cho cặp trọng tâm.
- Timing và reproducibility check đã ghi đầy đủ.
- Có đủ `metrics.json`, `params.json`, `artifact_index.csv` cho `stage5_multi_seed`.

**Definition of Ready cho Week 8-9:**

- Bảng kết quả RQ3 đã khóa phiên bản để không ảnh hưởng narrative RQ4.
- Figure/table cho RQ3 đã sẵn sàng tích hợp vào report.
- Risk note về sensitivity theo p và protocol-deviation status (nếu có) đã ghi vào `docs/experiment_registry.md`.
- Không còn blocker kỹ thuật từ simulation pipeline sang ML pipeline.

**Execution runbook tối thiểu (để team chạy thống nhất):**

1. Chạy benchmark main branch:

- `python run_all.py --stage stage5_multi_seed --mode main`.
- Verify có `outputs/stage5_multi_seed/rq3_strategy_benchmark.csv`, `outputs/stage5_multi_seed/rq3_main_raw_runs.parquet`, `outputs/stage5_multi_seed/seed_sets_k50.csv`.

2. Chạy sensitivity branch:

- `python run_all.py --stage stage5_multi_seed --mode sensitivity`.
- Verify có `outputs/stage5_multi_seed/rq3_sensitivity_p.csv`, `outputs/stage5_multi_seed/rq3_rank_stability.csv`.

3. Chạy statistical + reproducibility checks:

- Tạo `rq3_stats_tests.csv`, timing log, và `repro_check.json`.
- Verify `reports/tables/table_rq3_ic_benchmark.csv` đã cập nhật theo kết quả khóa phiên bản.

**Fail-fast criteria (dừng và xử lý ngay):**

- Số run thực tế không đạt `runs_main = 300` hoặc `runs_sensitivity = 50`/config p.
- Main benchmark dùng p khác `p_calibrated` mà không có issue log chính thức.
- Rank stability output thiếu hoặc không giải thích được rank shift bất thường.
- Thiếu statistical test toàn cục/post-hoc hoặc thiếu effect size cho cặp trọng tâm.
- Thiếu timing log hoặc thiếu reproducibility check cho stage5.

**Handover package Week 7 (bắt buộc trước khi sang Week 8-9):**

- `outputs/stage5_multi_seed/rq3_strategy_benchmark.csv`, `outputs/stage5_multi_seed/rq3_sensitivity_p.csv`.
- `outputs/stage5_multi_seed/rq3_rank_stability.csv`, `outputs/stage5_multi_seed/rq3_stats_tests.csv`.
- `outputs/stage5_multi_seed/seed_sets_k50.csv`, `outputs/stage5_multi_seed/repro_check.json`.
- `outputs/stage5_multi_seed/ablation_kshell.csv` (nếu chạy nice-to-have).
- `logs/timing/week7_multi_seed_timing.csv`, `reports/tables/table_rq3_ic_benchmark.csv`.
- `outputs/stage5_multi_seed/benchmark_protocol_note.md` (nếu lệch proposal baseline).
- Entry trong `docs/experiment_registry.md` ghi rõ sensitivity risk note, protocol alignment, và version khóa của RQ3 outputs.

---

### 🟡 Week 8-9: ML Detectability (RQ4) + Ablation + SHAP Analysis

**Mục tiêu:**

- Answer RQ4 với pipeline ML tái lập được, không leakage, và có khả năng diễn giải.
- Đánh giá rõ mức detectability của Hidden Influencer từ surface metrics.
- Giữ phân lớp rõ ràng giữa must-have và nice-to-have để kiểm soát scope.
- Chuẩn bị đầy đủ bảng/hình cho report và bàn giao packaging ở Week 10.

**Yêu cầu kỹ thuật + thông số bắt buộc (để implement nhất quán):**

- **Data split + CV defaults:**
  - `train_test_split: 80/20` (stratified theo `typology_label`).
  - `cv_folds: 5` (StratifiedKFold, shuffle bật).
  - `ml_seed` phải được dùng nhất quán cho split và training.
- **Must-have model set (không được thiếu):**
  - LR views-only.
  - LR degree-only.
  - LR views+degree.
  - Majority-class baseline.
- **Nice-to-have gating rule:**
  - Chỉ chạy RF/SHAP/Node2Vec khi must-have đã pass đầy đủ quality checks.
  - Không dùng kết quả nice-to-have để thay thế kết luận chính nếu must-have chưa đạt.
- **Leakage control requirements:**
  - Cấm trùng `node_id` giữa train/test.
  - Mọi transform fit trên train và apply cho test (không fit trên full data).
  - Lưu metadata split vào artifact để truy vết (`split_seed`, class distribution).
- **Schema tối thiểu cho outputs Stage 6:**
  - `rq4_detectability_report.csv`: `model_name`, `macro_f1`, `weighted_f1`, `hidden_f1`, `hidden_precision`, `hidden_recall`, `hidden_effect_size`, `hidden_effect_size_type`, `top_permutation_feature`.
  - `rq4_cv_summary.csv`: `model_name`, `cv_macro_f1_mean`, `cv_macro_f1_std`, `cv_hidden_f1_mean`.
  - `rq4_metrics_detailed.csv`: `model_name`, `class_label`, `precision`, `recall`, `f1`, `support`.
  - `perm_importance_lr_views_only.csv`, `perm_importance_lr_degree_only.csv`, `perm_importance_lr_views_degree.csv`.
  - `repro_check.json`: `split_seed`, `train_size`, `test_size`, `leakage_check_pass`, `rerun_delta_summary`.

**Kế hoạch thực thi chi tiết (theo thứ tự chạy):**

1. **Pre-flight check cho Week 8-9 (Day 1 - buổi 1):**
   - Xác nhận input từ các tuần trước:
     - `data/processed/typology_labels.parquet`.

- `outputs/stage1/centrality_table.parquet` hoặc bảng features tương đương.
  - Kết quả RQ2/RQ3 đã khóa để không thay đổi nhãn nền.
- Xác nhận config ML trong `src/config/experiment.yaml`:
  - Split `train/test = 80/20` (stratified).
  - `cv_folds = 5` (StratifiedKFold).
  - Seed cho split và training.
- Tạo thư mục output:
  - `outputs/stage6_ml/`.
  - `reports/figures/`.

2. **Feature engineering + dataset assembly (Day 1 - buổi 2):**
   - Hoàn thiện `src/ml/features_surface.py` với feature set:
     - Must-have: `views`, `degree`.
     - Optional: `views_per_degree`, `log_views` (không làm thay đổi baseline chính).
   - Build dataset cuối cho ML:
     - Merge features với typology labels.
     - Validate class distribution trước split.
   - Export:
     - `outputs/stage6_ml/ml_dataset_snapshot.parquet`.

3. **Must-have ML pipeline (Day 2):**
   - Split 80/20 stratified theo typology.
   - Huấn luyện và đánh giá 3 mô hình LR bắt buộc:
     - LR (views-only).
     - LR (degree-only).
     - LR (views + degree).
   - Baseline floor:
     - Majority-class classifier.
   - CV strategy:
     - StratifiedKFold(n_splits=5) trên train split cho tuning nhẹ.
   - Export:
     - `outputs/stage6_ml/rq4_detectability_report.csv`.
     - `outputs/stage6_ml/rq4_cv_summary.csv`.

4. **Evaluation + diagnostics bắt buộc (Day 2-3):**
   - Tính metrics:
     - F1 per class, macro-F1, weighted-F1.
     - Precision/recall theo lớp.
     - Confusion matrix.
   - Thêm thống kê cho Hidden class:
     - Rank-biserial, Cliff's Delta (nếu có so sánh phù hợp).
   - Kiểm tra leakage:
     - Không trùng node giữa train/test.
     - Stratification đúng theo nhãn typology.
   - Chạy permutation importance trên test set cho cả 3 LR models (must-have diagnostics, không phụ thuộc SHAP/RF).
   - Nếu có SHAP từ nhánh nice-to-have, đối chiếu hướng quan trọng feature giữa SHAP và permutation; nếu lệch lớn phải ghi note điều tra.
   - Export:
     - `outputs/stage6_ml/rq4_metrics_detailed.csv`.
     - `outputs/stage6_ml/perm_importance_lr_views_only.csv`.
     - `outputs/stage6_ml/perm_importance_lr_degree_only.csv`.
     - `outputs/stage6_ml/perm_importance_lr_views_degree.csv`.
     - `reports/figures/fig_confusion_matrix.png`.

5. **Nice-to-have branch (Day 3-4, chỉ khi must-have đã pass):**
   - RandomForest (views + degree) để tham chiếu non-linear upper-bound.
   - SHAP analysis trên best model của nhánh RF:
     - SHAP summary values.
     - Beeswarm/importance plot.
   - Nếu RF skip do time/resource, chạy SHAP fallback trên LR (`LinearExplainer`) để vẫn giữ interpretability cho RQ4.
   - Node2Vec ablation (appendix):
     - So sánh LR(views+degree) với LR(Node2Vec).
   - Export:
     - `outputs/stage6_ml/shap_analysis.csv`.
     - `outputs/stage6_ml/shap_summary_plot.png`.
     - `reports/figures/fig_shap_beeswarm.png`.
     - `outputs/stage6_ml/shap_linear_lr_summary.csv` (nếu RF skip và dùng LR SHAP fallback).
     - `outputs/stage6_ml/rq4_ablation_node2vec.csv`.

6. **Test + reproducibility audit (Day 4):**
   - Chạy unit tests `tests/test_ml_pipeline.py`.
   - Re-run một cấu hình ML với cùng seed để xác minh tái lập.
   - Ghi timing huấn luyện và inference:
     - `logs/timing/week8_9_ml_timing.csv`.
   - Export reproducibility check:
     - `outputs/stage6_ml/repro_check.json`.

7. **Review nội bộ + Definition of Ready cho Week 10 (Day 5):**
   - Review 30-45 phút với checklist:
     - Kết luận RQ4 có bám success criterion cho Hidden class.
     - Must-have và nice-to-have được tách rõ trong report text.
     - Không overclaim nếu kết quả vượt kỳ vọng.
   - Chốt bàn giao cho Week 10:
     - Bộ bảng/hình RQ4 đã khóa.
     - Mapping artifact -> claim đã hoàn chỉnh.
     - README draft cho phần chạy ML đã cập nhật.

**Deliverable:**

- `outputs/stage6_ml/rq4_detectability_report.csv` (all metrics + effect sizes).
- `outputs/stage6_ml/rq4_cv_summary.csv`.
- `outputs/stage6_ml/rq4_metrics_detailed.csv`.
- `outputs/stage6_ml/perm_importance_lr_views_only.csv`, `outputs/stage6_ml/perm_importance_lr_degree_only.csv`, `outputs/stage6_ml/perm_importance_lr_views_degree.csv`.
- `outputs/stage6_ml/shap_analysis.csv`, `outputs/stage6_ml/shap_summary_plot.png`.
- `reports/figures/fig_confusion_matrix.png`, `reports/figures/fig_shap_beeswarm.png`.
- `outputs/stage6_ml/repro_check.json`.
- `tests/test_ml_pipeline.py` passing.

**Acceptance Criteria (Week 8-9):**

- No data leakage (verified by test).
- Split 80/20 stratified được áp dụng đúng và có thể tái lập.
- Must-have models (3 LR + majority baseline) chạy hoàn chỉnh và có report đầy đủ.
- Hidden class metrics được report rõ (F1, precision/recall, confusion behavior).
- Permutation importance cho 3 LR models đã có và được dùng làm interpretation baseline độc lập với SHAP.
- Effect sizes/statistical diagnostics được ghi rõ ở các so sánh đã định nghĩa.
- Nhánh nice-to-have không làm thay đổi kết luận chính của RQ4.
- Có đủ `metrics.json`, `params.json`, `artifact_index.csv` cho `stage6_ml`.

**Definition of Ready cho Week 10:**

- Bộ artifact RQ1-RQ4 đã đủ và khóa phiên bản.
- Figure/tables cho report đã sẵn (bao gồm RQ4 confusion + optional SHAP).
- README chạy pipeline đã có hướng dẫn stage-level và run-all.
- Không còn blocker kỹ thuật cho bước packaging/repro end-to-end.

**Execution runbook tối thiểu (để team chạy thống nhất):**

1. Chạy nhánh must-have trước:

- `python run_all.py --stage stage6_ml --mode must_have`.
- Verify có `outputs/stage6_ml/rq4_detectability_report.csv`, `outputs/stage6_ml/rq4_cv_summary.csv`, `outputs/stage6_ml/rq4_metrics_detailed.csv`, `reports/figures/fig_confusion_matrix.png`.

2. Chạy nhánh nice-to-have (chỉ khi must-have PASS):

- `python run_all.py --stage stage6_ml --mode nice_to_have`.
- Verify có `outputs/stage6_ml/shap_analysis.csv`, `outputs/stage6_ml/shap_summary_plot.png` (và `outputs/stage6_ml/rq4_ablation_node2vec.csv` nếu bật ablation).

3. Chạy test + reproducibility:

- Chạy `tests/test_ml_pipeline.py`.
- Re-run 1 cấu hình cùng seed và verify `outputs/stage6_ml/repro_check.json` + `logs/timing/week8_9_ml_timing.csv`.

**Fail-fast criteria (dừng và xử lý ngay):**

- Phát hiện leakage (node overlap train/test hoặc stratification sai) ở bất kỳ checkpoint nào.
- Baseline must-have chưa pass nhưng đã dùng kết quả nice-to-have để kết luận chính.
- Hidden class metrics không đủ để diễn giải (thiếu F1/precision/recall/confusion behavior).
- Thiếu log seed/split hoặc không tái tạo được kết quả test split 80/20.
- Báo cáo RQ4 có overclaim so với evidence metric/effect size hiện có.

**Handover package Week 8-9 (bắt buộc trước khi sang Week 10):**

- `outputs/stage6_ml/rq4_detectability_report.csv`, `outputs/stage6_ml/rq4_cv_summary.csv`, `outputs/stage6_ml/rq4_metrics_detailed.csv`.
- `outputs/stage6_ml/perm_importance_lr_views_only.csv`, `outputs/stage6_ml/perm_importance_lr_degree_only.csv`, `outputs/stage6_ml/perm_importance_lr_views_degree.csv`.
- `reports/figures/fig_confusion_matrix.png`, `outputs/stage6_ml/repro_check.json`, `logs/timing/week8_9_ml_timing.csv`.
- `outputs/stage6_ml/shap_analysis.csv`, `outputs/stage6_ml/shap_summary_plot.png` (nếu nhánh nice-to-have được bật).
- `outputs/stage6_ml/rq4_ablation_node2vec.csv` (nếu chạy ablation).
- Entry trong `docs/experiment_registry.md` ghi rõ model set dùng cho kết luận chính (must-have) và phần mở rộng (nice-to-have).

---

### 🟢 Week 10: Final Report + Packaging + End-to-End Reproducibility

**Mục tiêu:**

- Finalize report, artifact packaging.
- **Single command reproducibility**: `make run_all` hoặc `./run_all.sh` hoặc `python run_all.py`.

**Yêu cầu kỹ thuật + thông số bắt buộc (để implement nhất quán):**

- **Release freeze parameters:**
  - `release_version` phải chốt theo format `vX.Y-rcZ` (ví dụ `v2.1-rc1`).
  - `release_tag_date` theo định dạng `YYYY-MM-DD`.
  - `config_hash` của bản nộp phải ghi trong `docs/experiment_registry.md` và `outputs/final_repro_check.json`.
- **E2E execution policy:**
  - Runner chính bắt buộc chạy full pipeline 1 lần hoàn chỉnh.
  - Runner phụ bắt buộc smoke tối thiểu 1 lần cho claim cross-platform.
  - `max_retry_same_config: 1` (nếu fail lần 2 với cùng config thì dừng và mở issue).
- **Artifact integrity checks:**
  - Mọi artifact được dẫn trong report phải tồn tại vật lý và có đường dẫn đúng.
  - Bắt buộc có checksum (khuyến nghị `sha256`) cho các file tổng hợp chính trong `artifact_index.csv`.
  - Không được thay đổi nội dung artifact sau khi freeze release trừ khi tăng `rc`.
- **Schema tối thiểu cho `outputs/final_repro_check.json`:**
  - `release_version`, `timestamp`, `config_hash`.
  - `primary_runner`, `secondary_runner`, `primary_status`, `secondary_status`.
  - `stage_status` (stage0..stage6), `timing_summary`, `open_high_issues_count`.
  - `rq_artifact_mapping_check_pass`, `cross_platform_claim_pass`.
- **Final sign-off rule:**
  - Chỉ được mark PASS khi đồng thời đạt: E2E pass runner chính, smoke pass runner phụ, mapping RQ->artifact pass, và `open_high_issues_count = 0`.

**Kế hoạch thực thi chi tiết (theo thứ tự chạy):**

1. **Pre-flight check cho Week 10 (Day 1 - buổi 1):**

- Xác nhận bộ artifact đầu vào đã đủ theo mapping RQ1-RQ4.
- Freeze config cuối cùng:
  - Seed, thresholds, `p_calibrated`, split ML.
- Xác nhận versioning:
  - Cập nhật changelog trong `docs/experiment_registry.md`.

2. **Finalize report draft (Day 1 - buổi 2 đến Day 2):**

- Hoàn thiện `reports/final_report.md`:
  - Mỗi RQ có: kết quả chính, bảng/figure tham chiếu, file evidence.
  - Nêu rõ assumptions và limitations.
  - Tránh overclaim khi effect nhỏ hoặc CI chồng lấn.
- Hoàn thiện `reports/drafts/report_outline.md` thành final narrative.

3. **Documentation completion (Day 2):**

- Hoàn thiện `docs/assumptions_limitations.md`.
  - Bắt buộc nêu: (1) limitation power-approximation cho MWU (nếu dùng ARE-adjusted bound), (2) circular validation risk giữa SIS(k-shell) và benchmark k-shell seeding.
  - Bổ sung section bắt buộc: `Circular Validation Control` (independence condition của single-seed IC + giới hạn diễn giải causal).
- Hoàn thiện `docs/experiment_registry.md`:
  - Timestamp, config hash, mọi thay đổi tham số quan trọng.
  - Nếu benchmark lệch proposal baseline (`p=0.01`), bắt buộc có entry protocol-deviation với rationale và comparability impact.
- Cập nhật `README.md`:
  - Setup, run theo stage, run end-to-end.
  - Troubleshooting ngắn cho Windows/Linux/Mac.

4. **Code review + evidence audit (Day 3):**

- Mỗi stage owner review chéo tối thiểu 1 stage khác.
- Checklist audit:
  - Claim nào trong report cũng có artifact đi kèm.
  - Không có mismatch tên file giữa report và outputs.
- Xuất biên bản:
  - `docs/code_review_notes.md`.

5. **Packaging + end-to-end run (Day 3-4):**

- Kiểm tra targets `Makefile`: `setup`, `stage0..stage6`, `run_all`.
- Chạy E2E bằng ít nhất 1 runner chính:
  - `make run_all` hoặc `./run_all.sh` hoặc `python run_all.py`.
- Khuyến nghị smoke E2E trên runner thứ 2 để xác nhận claim cross-platform.
- Ghi timing và kết quả:
  - `logs/timing/week10_e2e_timing.csv`.
  - `outputs/final_repro_check.json`.

6. **Final QA + release candidate (Day 5):**

- Chạy lại QA checklist Week/DoD mapping.
- Verify không thiếu artifact quan trọng trong `reports/figures`, `reports/tables`, `outputs/`.
- Đóng gói bản nộp nội bộ (release candidate):
  - Report, docs, config, scripts, outputs index.

**Deliverable:**

- `reports/final_report.md` (complete).
- `docs/assumptions_limitations.md`, `docs/experiment_registry.md` (complete).
- `Makefile` + `run_all.sh` + `run_all.py` tested end-to-end.
- `docs/code_review_notes.md`.
- `outputs/final_repro_check.json`.
- `logs/timing/week10_e2e_timing.csv`.
- All scripts tested + code review documented.

**Acceptance Criteria (Week 10):**

- Ít nhất 1 runner thực hiện thành công pipeline end-to-end từ raw data.
- Mapping RQ -> artifact -> claim khớp 100%.
- Report không có claim thiếu evidence, và assumptions/limitations đã nêu rõ.
- README đủ để người khác trong nhóm tái chạy mà không cần hướng dẫn miệng.
- Cross-platform claim có bằng chứng tối thiểu từ runner chính + smoke runner phụ.
- Có đủ `metrics.json`, `params.json`, `artifact_index.csv` cho toàn bộ stage0..stage6 và thông tin release trong `outputs/final_repro_check.json`.

**Definition of Ready cho Final Submission:**

- Report và toàn bộ artifact đã khóa phiên bản nộp.
- QA checklist Week/DoD pass hoàn toàn.
- Không còn blocker kỹ thuật/open issue mức High.
- Có release candidate nội bộ sẵn để nộp hoặc demo.

**Execution runbook tối thiểu (để team chạy thống nhất):**

1. Freeze bản nộp trước khi chạy E2E:

- Chốt `src/config/experiment.yaml`, khóa seed/config hash, và ghi entry cuối vào `docs/experiment_registry.md`.
- Tag nội bộ bản candidate (ví dụ theo ngày/phiên bản) để tránh trộn artifact giữa các lần chạy.

2. Chạy E2E bằng runner chính:

- `python run_all.py` hoặc `make run_all` hoặc `./run_all.sh`.
- Verify hoàn tất stage0 -> stage6 không lỗi, sinh đủ artifact index/timing.

3. Chạy smoke bằng runner phụ:

- Chọn runner còn lại (Make hoặc bash hoặc Python CLI) để smoke cross-platform claim.
- Ghi rõ pass/fail và môi trường chạy vào `outputs/final_repro_check.json`.

4. Final QA đóng gói:

- Cross-check RQ -> artifact -> claim.
- Chốt `reports/final_report.md`, `docs/code_review_notes.md`, `logs/timing/week10_e2e_timing.csv`.

**Fail-fast criteria (dừng và xử lý ngay):**

- E2E fail ở bất kỳ stage nào nhưng vẫn tiếp tục đóng gói release candidate.
- Mismatch giữa tên artifact trong report và tên file thực tế trong `outputs/`, `reports/tables/`, `reports/figures/`.
- Cross-platform claim chưa có bằng chứng runner phụ nhưng đã ghi PASS final QA.
- `final_repro_check.json` thiếu thông tin seed/config hash/timestamp của lần chạy cuối.
- Còn open issue mức High liên quan reproducibility nhưng chưa có mitigation note.

**Handover package Week 10 (final submission bundle):**

- `reports/final_report.md`, `reports/tables/*`, `reports/figures/*` đã dùng trong report.
- `docs/assumptions_limitations.md`, `docs/experiment_registry.md`, `docs/code_review_notes.md`.
- `outputs/final_repro_check.json`, `logs/timing/week10_e2e_timing.csv`, các `artifact_index.csv` theo stage.
- Runner scripts: `Makefile`, `run_all.sh`, `run_all.py` (kèm hướng dẫn trong `README.md`).
- QA evidence: mapping pass theo mục 12 và checklist release candidate nội bộ.

---

## 6) Mapping RQ -> script -> output

| RQ  | Script chính                                                                                            | Output chính                                                                                                                          |
| --- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| RQ1 | `src/graph/centrality.py`, `src/evaluation/ranking_overlap.py`, `src/evaluation/attribute_alignment.py` | `outputs/stage1/rq1_ranking_metrics.csv`, `outputs/stage2/attribute_community_analysis.csv`, `outputs/stage2/core_lifetime_tests.csv` |
| RQ2 | `src/sis/build_typology.py`, `src/simulation/run_single_seed_ic.py`                                     | `outputs/stage4_single_seed/rq2_hidden_validation.csv`                                                                                |
| RQ3 | `src/simulation/run_multi_seed_ic.py`                                                                   | `outputs/stage5_multi_seed/rq3_strategy_benchmark.csv`                                                                                |
| RQ4 | `src/ml/train_lr.py`, `src/ml/evaluate_metrics.py`                                                      | `outputs/stage6_ml/rq4_detectability_report.csv`                                                                                      |

---

## 7) Kế hoạch phân công nhân sự

Đề xuất dựa trên timeline proposal:

- Duy + Hải:
  - Data preprocess, IC simulation, run scripts automation.
- Vĩ + Vĩnh:
  - Centrality/community/kshell, SIS robustness, ML evaluation.
- All:
  - Stage 3 decision threshold, đọc kết quả, viết report.

Cơ chế làm việc:

- Họp kỹ thuật 2 lần/tuần (30-45 phút).
- Mỗi tuần chốt 1 milestone artifact (không chỉ chốt code).
- Mọi thay đổi config phải ghi vào `docs/experiment_registry.md`.

---

## 8) Quản lý rủi ro và phương án dự phòng

1. Betweenness quá chậm.

- Giảm sample pivot, hoặc chạy theo batch và lưu checkpoint.

2. IC runtime quá lớn.

- Song song hoá runs, hoặc giảm sample cho single-seed trước khi scale full.

3. Hidden set không ổn định khi đổi trọng số.

- Chuyển framing sang "structural candidate group" và báo rõ sensitivity.

4. ML kết quả quá tốt trên Hidden.

- Diễn giải theo hướng "partial detectability", không ép kết luận cực đoan.

5. Kết quả trái hypothesis.

- Giữ nguyên protocol, báo trung thực, bám backup plan trong proposal.

6. Lệch protocol benchmark so với proposal (`p=0.01`).

- Bắt buộc tạo protocol-deviation note (proposal value vs implemented value), nêu impact comparability trong report.

7. Circular validation risk (SIS chứa k-shell, RQ3 benchmark k-shell seeding).

- Acknowledge rõ trong assumptions/limitations; diễn giải RQ3 như benchmark hiệu năng strategy, không coi là proof độc lập duy nhất cho SIS.

8. Resource/runtime risk vượt ngưỡng profiling.

- Trigger fallback betweenness backend (NetworKit -> NetworkX), graph representation, hoặc pilot IC policy theo ngưỡng đã chốt; mọi trigger phải có log rationale và impact.

9. Disconnected components gây lệch diễn giải centrality.

- Bắt buộc component analysis + LCC decision trước khi khóa graph processed; nếu `lcc_ratio < 0.95` phải dừng để chốt quyết định nhóm.

---

## 9) Definition of Done (DoD) — Enhanced

Project được xem là hoàn tất khi đạt đủ:

### Core Requirements:

1. Toàn bộ scripts chính chạy thành công từ raw data đến final tables.
2. Có artifact đầy đủ cho 4 RQ (file + figure + metric).
3. Có log seed/params cho tất cả thí nghiệm chính.
4. Report cuối dẫn chiếu đúng tới artifact, không có claim không có bằng chứng.
5. **Single command** (`make run_all` hoặc `./run_all.sh` hoặc `python run_all.py`) tái tạo pipeline end-to-end.

### Scientific Rigor (NEW):

6. **SIS formula** có literature grounding (citations documented).
7. **Null model comparison** hoàn tất (typology validity checked).
8. **Statistical tests** có multiple testing correction + effect sizes.
9. **IC calibration** documented với rationale.
10. **Attribute diagnostics** (community-language + core-periphery life_time) được report đầy đủ.
11. **Protocol alignment/deviation log** được ghi rõ khi benchmark khác proposal baseline.
12. **Method limitations** (MWU power approximation + circular validation risk) được ghi trong assumptions/limitations.

### Reproducibility (ENHANCED):

13. `requirements.txt` có exact versions pinned.
14. **Cross-platform**: Pipeline chạy được trên Linux/Mac/Windows (qua Makefile + bash + Python CLI tương đương).
15. **No pickle files** - tất cả data lưu dạng parquet/edgelist.
16. **Resource profiling + fallback governance** hoàn tất trước Stage 1 (`resource_profile.json` + decision log nếu trigger).
17. **Component analysis + LCC decision governance** hoàn tất trước khi freeze `graph_active.edgelist`.

---

## 10) Priority Checklist — Làm trong 3 ngày đầu tiên

> **Mục tiêu**: Hoàn thành các task critical trong tuần 1 để foundation vững chắc.

### Day 1: Foundation Setup

- [ ] Tạo toàn bộ folder structure như mục 3.
- [ ] Tạo/chuẩn hóa `requirements.in`, sau đó sinh `requirements.txt` bằng `pip-compile` (giữ exact versions):
  ```
  networkit==10.1
  networkx==3.3
  pandas==2.2.0
  numpy==1.26.0
  scikit-learn==1.5.0
  cdlib==0.3.0
  statsmodels==0.14.0
  shap==0.44.0
  joblib==1.4.0
  pyarrow==15.0.0
  scipy==1.12.0
  matplotlib==3.8.0
  seaborn==0.13.0
  ```
- [ ] Ghi chú rõ betweenness backend mặc định là NetworKit; nếu không cài được (thường gặp trên Windows) thì fallback NetworkX bắt buộc `k_pivots = 2000` và phải log quyết định.
- [ ] Tạo `Makefile` với targets: `setup`, `stage0`, `stage1`, ..., `run_all`.
- [ ] Tạo `run_all.sh` (bash cross-platform).
- [ ] Tạo `run_all.py` (Python CLI/task runner tương đương cho Windows).

### Day 2: SIS Formula + Data Foundation

- [ ] **Viết `docs/implementation_notes.md`** với:
  - SIS formula định nghĩa toán học.
  - Literature citations (Kitsak 2010, Li 2021).
  - Eigenvector redundancy check plan.
- [ ] Hoàn thiện `src/data/load_raw.py` và `src/data/preprocess_graph.py`.
- [ ] Chạy data audit: xác nhận graph **undirected**, check self-loop, duplicate.

### Day 3: Critical Infrastructure

- [ ] Tạo skeleton `src/graph/null_model.py` (configuration model).
- [ ] Tạo skeleton `src/simulation/ic_calibration.py`.
- [ ] Thêm unit tests (khi bắt đầu code repo):
  - `tests/test_betweenness.py`: hard error nếu `num_nodes > 50k` mà `k_pivots < min_k_large_graph`; kiểm tra fallback NetworkX dùng `k_pivots=2000` khi NetworKit unavailable.
  - `tests/test_ic.py`: `p_kappa = κ/mean_degree` có clamp `[0.001, 0.5]` và selection rule trả về `method_used/selection_rationale`.
- [ ] Tạo `src/evaluation/stats_tests.py` với:
  - `multipletests` (Benjamini-Hochberg).
  - Cliff's Delta function.
  - Power analysis wrapper.
- [ ] Tạo file config gốc `src/config/base.yaml`.
- [ ] Export graph: `data/processed/graph_active.edgelist` + `data/processed/node_attributes.parquet` (**NO pickle**).
- [ ] Ghi metrics stage0: `outputs/stage0_data_quality/metrics.json`.

### Validation Checklist (End of Day 3):

- [ ] `make setup` chạy thành công (virtualenv + dependencies).
- [ ] Graph không có self-loop, duplicate edge.
- [ ] `docs/implementation_notes.md` có SIS formula + citations.
- [ ] Chốt ngưỡng tài nguyên máy cho IC và Betweenness.
- [ ] Ghi chú rõ: các ngưỡng calibration/stability là target vận hành nội bộ, không phải ngưỡng học thuật bắt buộc.

---

## 11) Summary: Key Changes from Expert Review

| Aspect            | Before            | After                                             |
| ----------------- | ----------------- | ------------------------------------------------- |
| SIS Formula       | Implicit          | Explicit formula + citations                      |
| Null Model        | Missing           | Configuration model comparison (Week 5)           |
| IC Calibration    | Missing           | Reach-range pilot + κ-target rationale (Week 6)   |
| Statistical Tests | p-value only      | Benjamini-Hochberg + Effect sizes                 |
| ML Pipeline       | LR only           | LR baselines (must-have) + RF/SHAP (nice-to-have) |
| Data Format       | pickle            | edgelist + parquet                                |
| Scripts           | PowerShell        | Makefile + bash + Python CLI (cross-platform)     |
| Timeline          | Week 7-8 overload | Balanced 10-week schedule                         |
| Louvain           | Single run        | 10 runs + NMI stability                           |
| Requirements      | Version ranges    | Exact versions pinned                             |
| Betweenness       | NX k=500 approx   | NetworKit preferred; NX fallback k=2000 + guards  |

---

## 12) QA Checklist (Week/DoD 1-1 Mapping) — Pre v2.4 Sign-off

_Checklist này dùng để QA tính nhất quán tài liệu (document-level). Không thay thế việc chạy thực nghiệm thực tế._

| Week | Artifact đầu ra bắt buộc                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Mục DoD đối chiếu                                                       | Trạng thái mapping |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------- | ------------------ |
| 1-2  | `outputs/stage0_data_quality/metrics.json`, `outputs/stage0_data_quality/resource_profile.json`, `outputs/stage0_data_quality/component_analysis.json`, `data/processed/graph_active.edgelist`, `data/processed/node_attributes.parquet`, `Makefile`, `run_all.sh`, `run_all.py`                                                                                                                                                                                                                                                                 | Core #1, Core #5, Repro #13, Repro #14, Repro #15, Repro #16, Repro #17 | PASS               |
| 3-4  | `outputs/stage1/centrality_metrics.json`, `outputs/stage1/centrality_table.parquet`, `outputs/stage1/rq1_ranking_metrics.csv`, `outputs/stage2/community_labels.parquet`, `outputs/stage2/louvain_stability_report.json`, `outputs/stage2/attribute_community_analysis.csv`, `outputs/stage2/core_lifetime_tests.csv`                                                                                                                                                                                                                            | Core #1, Core #2, Core #3, Scientific #10                               | PASS               |
| 5    | `data/processed/sis_table.parquet`, `data/processed/typology_labels.parquet`, `outputs/stage3/robustness_summary.csv`, `outputs/stage3/null_model_comparison.csv`, `outputs/stage3/null_sample_representativeness.csv`, `outputs/stage3/null_model_warnings.md`, `outputs/stage3/power_assumption_note.md`                                                                                                                                                                                                                                       | Scientific #6, Scientific #7, Scientific #12, Core #2, Core #3          | PASS               |
| 6    | `outputs/stage3_ic_calibration/calibration_results.csv`, `outputs/stage4_single_seed/rq2_hidden_validation.csv`, `outputs/stage4_single_seed/sampling_frame.csv`                                                                                                                                                                                                                                                                                                                                                                                 | Scientific #8, Scientific #9, Core #2, Core #3                          | PASS               |
| 7    | `outputs/stage5_multi_seed/rq3_strategy_benchmark.csv`, `outputs/stage5_multi_seed/rq3_sensitivity_p.csv`, `outputs/stage5_multi_seed/ablation_kshell.csv` (nếu chạy), `outputs/stage5_multi_seed/rq3_rank_stability.csv`, `outputs/stage5_multi_seed/rq3_stats_tests.csv`, `outputs/stage5_multi_seed/repro_check.json`, `outputs/stage5_multi_seed/benchmark_protocol_note.md` (nếu lệch proposal), `logs/timing/week7_multi_seed_timing.csv`                                                                                                  | Core #2, Core #3, Core #4, Scientific #8, Scientific #11                | PASS               |
| 8-9  | `outputs/stage6_ml/rq4_detectability_report.csv`, `outputs/stage6_ml/rq4_cv_summary.csv`, `outputs/stage6_ml/rq4_metrics_detailed.csv`, `outputs/stage6_ml/perm_importance_lr_views_only.csv`, `outputs/stage6_ml/perm_importance_lr_degree_only.csv`, `outputs/stage6_ml/perm_importance_lr_views_degree.csv`, `outputs/stage6_ml/repro_check.json`, `reports/figures/fig_confusion_matrix.png`, `outputs/stage6_ml/shap_analysis.csv`, `outputs/stage6_ml/shap_summary_plot.png`, `outputs/stage6_ml/shap_linear_lr_summary.csv` (nếu RF skip) | Core #2, Core #3, Scientific #8, Repro #13                              | PASS               |
| 10   | `reports/final_report.md`, `docs/assumptions_limitations.md`, `docs/experiment_registry.md`, `docs/code_review_notes.md`, `outputs/final_repro_check.json`, `logs/timing/week10_e2e_timing.csv`, runners chạy end-to-end                                                                                                                                                                                                                                                                                                                         | Core #4, Core #5, Scientific #12, Repro #14                             | PASS               |

**RQ -> Artifact cross-check (1-1):**

- RQ1 -> `outputs/stage1/rq1_ranking_metrics.csv`
- RQ2 -> `outputs/stage4_single_seed/rq2_hidden_validation.csv`
- RQ3 -> `outputs/stage5_multi_seed/rq3_strategy_benchmark.csv`
- RQ4 -> `outputs/stage6_ml/rq4_detectability_report.csv`

**QA kết luận trước khi chốt 2.4:**

- Các yêu cầu bắt buộc đã đồng bộ: SIS equal-weight, ML split 80/20, Python CLI/task runner, must-have vs nice-to-have, target vận hành nội bộ, dataset global.
- Tên artifact giữa Week/Deliverable/Mapping/DoD đã khớp.
- Resource profiling + component/LCC governance đã được đưa vào gate bắt buộc trước Stage 1.
- Không còn mâu thuẫn tham số cũ (0.4/0.4/0.2, 70/10/20, DE-only scope).
- Protocol alignment với proposal và các deviation (nếu có) đã được log đầy đủ trong `docs/experiment_registry.md`.
- Limitation quan trọng (MWU power approximation, circular validation risk) đã được khai báo rõ trong `docs/assumptions_limitations.md`.

---

_Last updated: Per Final Expert Review (March 2026)_
_Document version: 2.4_
