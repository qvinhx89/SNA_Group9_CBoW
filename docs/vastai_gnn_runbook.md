# Vast.ai GPU runbook — MAPR2026 v3.2 (A0 + HSCC) GNN surrogates

Mục tiêu: chạy các thí nghiệm GNN (SAGE/GCN/GIN/GAT/APPNP) để xấp xỉ IC labels theo **dual-operationalization** (`A0` + `HSCC`) và xuất artifact theo plan v3.2.

Artifacts liên quan
- Labels/targets:
  - A0: data/processed/regression_targets_a0.parquet
  - HSCC: data/processed/regression_targets_hscc_refined.parquet
  - Sensitivity A2: data/processed/regression_targets_a2.parquet

- Split mask (M0-locked): data/processed/split_masks.parquet
- Graph edgelist: data/processed/graph_active.edgelist
- Node attrs: data/processed/node_attributes.parquet
- Centrality table: data/processed/centrality_table.parquet

## 0) Chọn image / môi trường
Khuyến nghị: chọn image có sẵn CUDA + PyTorch.

Yêu cầu: cài được các package sau trong môi trường Python:
- torch
- torch_geometric (và các dependency torch-scatter/torch-sparse/… tương ứng)

Lưu ý: nếu cài torch_geometric lỗi do Python 3.12, cân nhắc dùng Python 3.11 trong container.

## 1) Setup repo
```bash
git clone <your-repo-url>
cd sna_twitch_influencer_project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Sau đó cài PyTorch + PyG phù hợp CUDA của instance.
(Phần này phụ thuộc CUDA version của image; dùng hướng dẫn chính thức của PyTorch/PyG để chọn đúng wheel.)

Quick check:
```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import torch_geometric; print('pyg ok')"
python -c "from torch_geometric.nn import APPNP; print('APPNP ok')"
```

## 1.5) Local sanity check (CPU, chạy nhanh)
Lưu ý: training full graph bằng CPU có thể rất chậm. Để “test tín hiệu” nhanh trước khi thuê GPU, chạy trên induced subgraph của 5k labeled nodes:

```bash
python src/mapr2026_v3/run_surrogates.py \
  --only-edge-only \
  --node-scope labeled \
  --seeds 42 \
  --max-epochs 20 \
  --early-stop --patience 5 \
  --out-csv outputs/mapr2026_v3_results/surrogate_edge_only_quick.csv
```

Khi chạy trên vast.ai GPU để lấy kết quả “chuẩn”, để mặc định `--node-scope all` (hoặc không truyền flag).

## 2) (Tuỳ chọn) Tạo hybrid IC targets (CPU)
Chỉ cần làm nếu bạn muốn chạy surrogate trên target hybrid.

```bash
python src/mapr2026_v3/ic_labels_hybrid_views.py --n-jobs -1 --alpha 0.85 --p-max 0.5
```

Output:
- outputs/mapr2026_v3_results/ic_scores_hybrid_views.parquet
- data/processed/regression_targets_hybrid_views.parquet

## 3) Chạy GNN surrogates theo plan v3.2

Gợi ý: chạy riêng từng regime và xuất ra **cùng 1 file** (script sẽ ghi cột `label_regime`) hoặc xuất ra các file riêng tuỳ workflow.

### 3.0 Baselines (khuyến nghị chạy trước để có comparator HSCC)
```bash
python src/mapr2026_v3/run_baselines.py \
  --targets-path data/processed/regression_targets_a0.parquet \
  --out-csv outputs/mapr2026_v3_results/baseline_ranking_metrics.csv

python src/mapr2026_v3/run_baselines.py \
  --targets-path data/processed/regression_targets_hscc_refined.parquet \
  --out-csv outputs/mapr2026_v3_results/baseline_ranking_metrics.csv
```
### 3.1 Edge-only (graph-only strict, x=1)
```bash
python src/mapr2026_v3/run_surrogates.py \
  --only-edge-only \
  --early-stop --patience 20 \
  --targets-path data/processed/regression_targets_a0.parquet \
  --out-csv outputs/mapr2026_v3_results/surrogate_ranking_metrics.csv
```

### 3.2 C2 architecture comparison (raw_attr)
```bash
python src/mapr2026_v3/run_surrogates.py \
  --include-c2-arch \
  --early-stop --patience 20 \
  --targets-path data/processed/regression_targets_hscc_refined.parquet \
  --out-csv outputs/mapr2026_v3_results/surrogate_ranking_metrics.csv
```

### 3.3 C2 trên A2 targets (H2: GNN–A2 alignment)
```bash
python src/mapr2026_v3/run_surrogates.py \
  --targets-path data/processed/regression_targets_a2.parquet \
  --include-c2-arch \
  --early-stop --patience 20 \
  --out-csv outputs/mapr2026_v3_results/surrogate_ranking_metrics.csv
```

### 3.4 C2 trên hybrid targets
```bash
python src/mapr2026_v3/run_surrogates.py \
  --targets-path data/processed/regression_targets_hybrid_views.parquet \
  --include-c2-arch \
  --early-stop --patience 20 \
  --out-csv outputs/mapr2026_v3_results/surrogate_c2_hybrid_raw_attr.csv
```

## 4) Kéo kết quả về local
Kéo thư mục outputs/mapr2026_v3_results/ (các file CSV) về máy.

## 5) Checklist sanity trước khi báo cáo
- Mỗi file surrogate_*.csv có đủ các cột mean/std + runtime/train
- Có ít nhất các rows C2: gcn_raw_attr, gin_raw_attr, gat_raw_attr, **appnp_raw_attr**
- So sánh với baseline: outputs/mapr2026_v3_results/baseline_ranking_metrics.csv
- Báo cáo thêm: speedup (runtime_breakdown.csv) nếu cần
