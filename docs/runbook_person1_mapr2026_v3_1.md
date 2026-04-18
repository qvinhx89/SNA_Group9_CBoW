# Runbook — Person 1 (MAPR2026 v3.1)

Mục tiêu: chạy đúng thứ tự các bước của Person 1 để tạo **artifact contracts** (unblock Person 2/3), đồng thời có **checkpoint/gates** để biết khi nào cần chuyển sang Option B hoặc tạo “stability explanation”.

Nguồn gốc kế hoạch:
- `docs/Plan flow.md`
- `docs/MAPR2026_v3_team_parallel_coding_plan.md` (Section 8 — Person 1)

> OS note: runbook viết theo PowerShell (Windows). Nếu bạn dùng bash, chỉ cần bỏ `powershell` fencing.

---

## 0) Preflight nhanh (2–5 phút)

### 0.1 Environment

```powershell
# (Khuyến nghị theo plan)
conda activate sna_group9_cbow_py312
python --version
pip install -r requirements.txt
```

### 0.2 Base inputs phải có (Stage 0 cũ)

Tối thiểu:
- `data/processed/graph_active.edgelist`
- `data/processed/node_attributes.parquet` (có `node_id`, `views`)

Nếu thiếu:

```powershell
python run_all.py --stage 0
```

---

## 1) PHASE 0 — QA + CSR export (unblock P2/P3) [MAPR-MUST]

Ghi chú đồng bộ plan:
- `docs/Plan flow.md` có vài tên output Stage 0 kiểu cũ (ví dụ `dead_accounts.json`, `lcc_audit.json`). Repo hiện tại dùng **canonical contracts** là `dead_account_report.json` và `lcc_report.json` (đúng theo script trong `src/data/`). Runbook này bám theo canonical names.

### 1.1 Dead account audit

```powershell
python src/data/dead_account_audit.py
```

Expected output:
- `outputs/stage0_data_quality/dead_account_report.json`

### 1.2 LCC audit

```powershell
python src/data/lcc_audit.py
```

Expected output:
- `outputs/stage0_data_quality/lcc_report.json`

> Nếu `pct_lcc < 90%`: báo team ngay; về nguyên tắc sampling/IC nên restrict LCC (ghi quyết định vào `docs/m0_decisions.md`).

### 1.3 Export CSR (deterministic mapping)

```powershell
python src/mapr2026_v3/export_csr.py --run
```

Expected output:
- `data/processed/graph_csr.npz`

Quick sanity (không bắt buộc): chạy lại lần 2 phải ra y hệt (determinism). Nếu thấy mapping drift → fix `export_csr.py` (sort `node_id` trước khi build CSR).

Handoff ngay sau bước này (để P2/P3 chạy song song):
- `data/processed/graph_csr.npz`

---

## 2) PHASE 1A — Day-1 benchmark + pilot diagnostics [MAPR-MUST]

### 2.1 Day-1 benchmark (runtime + one-hop correlation)

Command chuẩn đã ghi trong `docs/day1_decisions.md`:

```powershell
python src/mapr2026_v3/day1_benchmark.py --n-jobs -1 --seed 42 --bench-nodes 100 --bench-runs 50 --pilot-nodes 200 --pilot-runs 50 --target-n-sample 5000 --out-dir outputs/day1_benchmark
```

Expected outputs:
- `outputs/day1_benchmark/ic_runtime_benchmark.json`
- `outputs/day1_benchmark/one_hop_correlation.json`

Action:
- Copy các số “Locked decision” và “decision_branch” vào `docs/day1_decisions.md` (nếu chưa có).

### 2.2 IC pilot diagnostics (cv_score + jaccard stability pilot)

```powershell
python src/mapr2026_v3/ic_pilot_diagnostics.py
```

Expected output:
- `outputs/day1_benchmark/ic_pilot_diagnostics.json`

Interpret nhanh:
- `cv_score > 0.30` là regression-ready gate theo plan.
- `jaccard_stability` thấp không block regression target, nhưng báo hiệu binary top-10 noisy.

---

## 3) PHASE 1A — Primary IC labels A0 + split mask [MAPR-MUST]

### 3.1 Run primary labeling (A0: weighted cascade)

```powershell
python src/mapr2026_v3/ic_labels_primary.py --n-runs 200 --n-sample 5000 --n-jobs -1
```

Expected outputs (contracts):
- `data/processed/ic_scores_primary.parquet`
- `data/processed/regression_targets.parquet`
- `data/processed/classification_labels.parquet`
- `data/processed/split_masks.parquet`

Notes:
- Split là **M0-locked**: 80/20, degree-quintile stratified, seed=42. Person 3 **không được tự tạo split**.
- Script sẽ cố update section M3 views/IC alignment trong `docs/day1_decisions.md` (nếu fail thì chạy bước 3.2).

### 3.2 (M3) Views/IC alignment update-only (nếu cần)

```powershell
python src/mapr2026_v3/ic_labels_primary.py --update-m3-only
```

Expected side-effect:
- cập nhật section “M3 Views/IC Alignment Check” trong `docs/day1_decisions.md`.

---

## 4) Post-label quality evidence + handoff freeze (để P2/P3 consume an toàn)

### 4.1 Label uncertainty (boundary nodes + CI parquet)

```powershell
python src/mapr2026_v3/ic_label_uncertainty.py
```

Expected outputs:
- `outputs/day1_benchmark/ic_label_uncertainty.json`
- `data/processed/ic_scores_primary_with_ci.parquet`

### 4.2 Label stability (cross-seed)

```powershell
python src/mapr2026_v3/ic_label_stability.py --n-runs 150 --mc-seeds 0,1,2 --n-jobs -1 --jaccard-threshold 0.85
```

Expected output:
- `outputs/day1_benchmark/ic_label_stability.json`

### 4.3 (Triggered) Stability explanation artifact

Điều kiện trigger:
- `ic_label_stability.json` có `jaccard_mean < 0.85` (hoặc `jaccard_pass_threshold=false`).

Chạy protocol:

```powershell
python src/mapr2026_v3/ic_feasibility_protocol.py --out-dir outputs/ic_feasibility
```

Protocol outputs (nguồn để extract):
- `outputs/ic_feasibility/phase1_community_overlap.json`
- `outputs/ic_feasibility/phase2_threshold_analysis.json`
- `outputs/ic_feasibility/pivot_decision_report.json`

Sau đó tạo **artifact bắt buộc** (manual extract theo schema plan):
- `outputs/day1_benchmark/stability_explanation.json`

Gợi ý (theo `docs/day1_decisions.md` mục 12):
- `pct_communities_spanning_boundary`
- `mean_gap_to_noise`
- `n_thresholds_tested`
- `interpretation`

### 4.4 Freeze + version hóa package handoff (khuyến nghị)

Chế độ strict (final) sẽ fail nếu gate fail; dùng provisional để vẫn bàn giao nhưng giữ trạng thái gate.

```powershell
# strict
python src/mapr2026_v3/freeze_day1_handoff.py --version-tag 20260418_p1_day1_v3_final

# nếu gate fail nhưng cần unblock P2/P3
python src/mapr2026_v3/freeze_day1_handoff.py --version-tag 20260418_p1_day1_v3_provisional --quality-mode provisional
```

Expected outputs:
- `outputs/day1_benchmark/quality_gate_report.json`
- `outputs/day1_benchmark/split_freeze_manifest.json`
- `outputs/handoffs/person1_day1_<version_tag>/manifest.json`

Handoff cho Person 2/3 (tối thiểu):
- `data/processed/graph_csr.npz`
- `data/processed/ic_scores_primary.parquet`
- `data/processed/regression_targets.parquet`
- `data/processed/classification_labels.parquet`
- `data/processed/split_masks.parquet`
- `outputs/day1_benchmark/*.json` (benchmark + pilot + stability + uncertainty + gate report)

---

## 5) I-A branch (attribute-informed) — pilot MUST, full labeling conditional

### 5.1 Pre-register (MUST trước khi chạy pilot)

Edit `docs/experiment_registry.md` và thêm statement:

- `H-IA: Under I-A labels, GATv2 sẽ outperform degree (degree blind to row-norm IC); FAIL → A0-only narrative.`

### 5.2 I-A pilot gate (unconditional MUST)

```powershell
python src/mapr2026_v3/ic_pilot_ia.py --n-jobs -1
```

Expected output:
- `outputs/mapr2026_v3_results/ia_pilot_diagnostics.json`

Decision rule (PASS nếu đủ cả 3):
- `cv_across_nodes > 0.30`
- `abs(spearman_ic_vs_degree) < 0.75`
- `abs(spearman_ic_vs_nbr_views_mean_proxy) < 0.85`

### 5.3 I-A full labeling (CHỈ chạy nếu pilot PASS)

```powershell
python src/mapr2026_v3/ic_labels_attribute_ia.py --n-jobs -1
```

Nếu bạn cần **GPU CUDA-only** (Windows):

```powershell
python src/mapr2026_v3/ic_labels_attribute_ia_cuda.py --progress-every 50
```

Ghi chú: command này **abort** nếu `torch.cuda.is_available()==False`.

Expected outputs:
- `outputs/mapr2026_v3_results/ic_scores_ia.parquet`
- `data/processed/regression_targets_ia.parquet`

---

## 6) PHASE 2 — C1 Degree-controlled IC variance (MAPR-MUST)

> Theo plan, C1 nên dùng thêm `one_hop_spread` từ `data/processed/diffusion_proxies.parquet` (P2) để có tier-2 residual evidence.

Chạy C1:

```powershell
python src/mapr2026_v3/degree_controlled_variance.py
```

Expected output:
- `outputs/mapr2026_v3_results/degree_controlled_ic_variance.json`

Ghi chú:
- Script sẽ tự động **bật tier-2** nếu thấy `data/processed/diffusion_proxies.parquet` đầy đủ cho labeled nodes.
- Nếu muốn chạy chỉ tier-1 (within-degree CV): thêm `--disable-tier2`.

---

## 7) A2 sensitivity (BOOST — làm sau khi critical path xong)

Hiện repo có diagnostics nhanh cho A2 (không phải full A2 labeling):

```powershell
python src/mapr2026_v3/a2_quick_diagnostics.py
```

Nếu team quyết định chạy full A2 labels theo plan, cần implement/enable A2 variant trong labeling pipeline (vì `ic_labels_primary.py` hiện chưa có flag variant A2).

---

## 8) Lock config v3.1 (paper-facing, manual) [MAPR-MUST]

Cập nhật `src/config/experiment.yaml` để “construct validity” không bị trôi (nội dung theo checklist row 16):
- `graph_directed: false` (+ note)
- `calibration_mode: variance_check`
- `p_primary: weighted_cascade`
- `ic_backend: csr_numpy`
- `ic_parallel: joblib_loky`

---

## 9) Minimal “Done” checklist (để không block người khác)

Trước khi ping Person 2/3 “có thể chạy thật”:
- CSR: `data/processed/graph_csr.npz`
- Labels: `data/processed/ic_scores_primary.parquet` + `regression_targets.parquet`
- Split: `data/processed/split_masks.parquet`
- Day1: `outputs/day1_benchmark/ic_runtime_benchmark.json` + `one_hop_correlation.json`
- Quality evidence: `ic_pilot_diagnostics.json` + `ic_label_stability.json` + `ic_label_uncertainty.json`
- (Nếu triggered) `outputs/day1_benchmark/stability_explanation.json`
- (Khuyến nghị) handoff folder `outputs/handoffs/person1_day1_<version_tag>/` có manifest SHA256
