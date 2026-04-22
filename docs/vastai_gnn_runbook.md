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
Khuyến nghị: chọn image có sẵn CUDA + Conda/Miniconda để dễ quản lý binary PyTorch.

Máy thuê (ví dụ của bạn): Tesla V100-SXM2-32GB (driver ~580.x, CUDA driver reported 13.0). Với GPU loại này, an toàn nhất là cài PyTorch qua Conda (`pytorch-cuda=12.8` hoặc `pytorch-cuda=11.8`) vì các package Conda thường bao gồm kernel binary cho nhiều compute capability (V100 = CC 7.0).

Yêu cầu: môi trường Python phải cài được:
- `torch` (thông qua Conda)
- `torch_geometric` và các dependency native (`pyg-lib`, `torch-scatter`, `torch-sparse`, `torch-cluster`, `torch-spline-conv`)

Lưu ý: nếu image mặc định dùng Python 3.12 và bạn gặp lỗi khi cài `torch_geometric`, tạo env với Python 3.10 hoặc 3.11 (khuyến nghị 3.10 cho tính tương thích với các wheels hiện thời).

Thêm: luôn kiểm tra `nvidia-smi` và `python` trong cùng env trước khi cài. Xem phần "Quick checks" bên dưới.

## 1) Setup repo + Conda env (khuyến nghị)
Sử dụng Conda (Miniconda) giúp cài các binary PyTorch/CUDA tương thích cho GPU V100.

```bash
# clone repo (nếu chưa có)
git clone <your-repo-url> ~/SNA_Group9_CBoW
cd ~/SNA_Group9_CBoW

# (Nếu chưa có) cài Miniconda (chỉ cần chạy lần đầu trên instance)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p /opt/conda
export PATH=/opt/conda/bin:$PATH
hash -r

# tạo và kích hoạt env khuyến nghị (Python 3.10)
conda create -n mapr python=3.10 -y
conda activate mapr

# cập nhật pip/tools
python -m pip install -U pip setuptools wheel

# cài requirements cơ bản (sẽ cài PyTorch/PyG riêng theo below)
python -m pip install -r requirements.txt || true
```

Sau đó, cài PyTorch + pytorch-cuda qua Conda (thích hợp cho V100). Xem phần "Install PyTorch & PyG".

### Quick checks (trước khi cài)
```bash
nvidia-smi
which python
python - <<'PY'
import torch,sys
print('python', sys.executable)
print('torch', getattr(torch,'__version__',None))
print('torch.version.cuda', getattr(torch.version,'cuda',None))
print('cuda_available', torch.cuda.is_available())
try:
  p = torch.cuda.get_device_properties(0)
  print('device:', p.name, 'CC=', p.major, p.minor, 'memGB=', round(p.total_memory/1024**3,2))
except Exception as e:
  print('device info error:', e)
PY
```

### Install PyTorch & PyG (Conda-based recommended for V100)
1) Cài PyTorch + CUDA runtime qua `conda` (chọn 12.8 nếu driver mới, fallback 11.8):

```bash
# Prefer pytorch-cuda=12.8 (if driver supports it)
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.8 -c pytorch -c nvidia || \
  conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Verify torch
python - <<'PY'
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'ok', torch.cuda.is_available())
try:
  p = torch.cuda.get_device_properties(0)
  print('device', p.name, 'CC', p.major, p.minor)
except Exception as e:
  print('device info error', e)
PY
```

2) Sau khi `torch` đã sẵn sàng trong env `mapr`, cài các native wheels PyG bằng `pip` (dùng wheel index tương thích với `torch.__version__`):

```bash
TORCH_VER=$(python -c "import torch;print(torch.__version__)")
WHEEL_URL="https://data.pyg.org/whl/torch-${TORCH_VER}.html"
echo "Using wheel index: $WHEEL_URL"

python -m pip install -f "$WHEEL_URL" pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv
python -m pip install --no-deps --force-reinstall torch-geometric

# Verify pyg + APPNP
python - <<'PY'
import torch, torch_geometric
from torch_geometric.nn import APPNP
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cuda_ok', torch.cuda.is_available())
print('pyg', torch_geometric.__version__)
print('APPNP OK')
PY
```

Ghi chú:
- Luôn chạy `TORCH_VER` lấy trực tiếp từ `python` trong cùng env để tránh mismatch.
- Nếu thấy lỗi kiểu `no kernel image is available for execution on the device`, cài lại PyTorch qua `conda` với `pytorch-cuda=11.8` hoặc `=12.8` (thử cả hai nếu cần) rồi cài lại PyG wheels.

## 1.5) Local sanity check (CPU, chạy nhanh)
Lưu ý: training full graph bằng CPU có thể rất chậm. Để “test tín hiệu” nhanh trước khi chạy GPU, chạy trên induced subgraph của 5k labeled nodes (hoặc dùng `--node-scope labeled`):

```bash
python src/mapr2026_v3/run_surrogates.py \
  --only-edge-only \
  --node-scope labeled \
  --seeds 42 \
  --max-epochs 20 \
  --early-stop --patience 5 \
  --out-csv outputs/mapr2026_v3_results/surrogate_edge_only_quick.csv
```

Khi chạy trên instance V100 (GPU) để lấy kết quả “chuẩn”, để mặc định `--node-scope all` (hoặc không truyền flag). Trên GPU, trên các lệnh baseline/gpu khuyến nghị thêm `--skip-node2vec` để tránh phải cài node2vec native wheel.

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
  --label-regime a0 \
  --out-csv outputs/mapr2026_v3_results/baseline_ranking_metrics.csv

python src/mapr2026_v3/run_baselines.py \
  --targets-path data/processed/regression_targets_hscc_refined.parquet \
  --label-regime hscc \
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
  --label-regime hscc \
  --early-stop --patience 20 \
  --targets-path data/processed/regression_targets_hscc_refined.parquet \
  --out-csv outputs/mapr2026_v3_results/surrogate_ranking_metrics.csv
```

### HSCC quickstart (khuyến nghị chạy theo thứ tự này)
Chạy trong `tmux` để không mất job khi rớt SSH. Trước khi chạy, đảm bảo bạn đã `conda activate mapr` và `torch` + `pyg` import OK.

```bash
apt-get update && apt-get install -y tmux
tmux new -s hscc -d

# (1) Baselines HSCC (skip Node2Vec on GPU)
tmux send-keys -t hscc "python src/mapr2026_v3/run_baselines.py \
  --targets-path data/processed/regression_targets_hscc_refined.parquet \
  --label-regime hscc \
  --skip-node2vec \
  --out-csv outputs/mapr2026_v3_results/baseline_ranking_metrics.csv 2>&1 | tee outputs/mapr2026_v3_results/baseline_hscc_gpu.log" C-m

# (2) GNN surrogates HSCC (APPNP + edge-only + C2)
tmux send-keys -t hscc "python src/mapr2026_v3/run_surrogates.py \
  --targets-path data/processed/regression_targets_hscc_refined.parquet \
  --label-regime hscc \
  --include-c2-arch \
  --include-edge-only \
  --early-stop --patience 20 \
  --out-csv outputs/mapr2026_v3_results/surrogate_ranking_metrics.csv 2>&1 | tee outputs/mapr2026_v3_results/surrogate_hscc_gpu.log" C-m

# attach to watch logs:
tmux attach -t hscc
# or tail logs
tail -n 200 -f outputs/mapr2026_v3_results/baseline_hscc_gpu.log
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

## Troubleshooting (những lỗi hay gặp)
- "no kernel image is available for execution on the device": thường do PyTorch binary không chứa kernel cho CC của GPU (V100 CC=7.0). Hãy cài lại PyTorch bằng `conda install pytorch-cuda=12.8` hoặc `=11.8`, rồi cài lại PyG wheels.
- Nếu `TORCH_VER` rỗng khi cài PyG wheels (ví dụ `https://data.pyg.org/whl/torch-.html`), đó là dấu hiệu `torch` chưa cài trong cùng env — chạy `python -c 'import torch; print(torch.__version__)'` trước.
- Nếu gặp lỗi khi cài native wheels (ví dụ `pyg-lib` không tìm thấy), kiểm tra rằng `TORCH_VER` và Python ABI tương thích với các wheels trên trang PyG; dùng Conda-built `torch` thường giảm thiểu vấn đề.

Nếu muốn, mình sẽ commit thêm một file `scripts/setup_gpu_env.sh` với các lệnh Conda/pip đã dùng để bạn chạy tự động trên instance.
