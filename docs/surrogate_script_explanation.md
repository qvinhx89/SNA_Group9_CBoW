# Gốc Rễ và Cấu Trúc Script `run_surrogates.py`

Script `run_surrogates.py` đóng vai trò là "Cỗ Máy ML" chính trong Tracking C của đồ án MAPR2026. Nó đảm nhiệm việc đánh giá và đo lường năng lực của các mô hình Hồi Quy thay thế (Surrogate Regressors) trong việc dự báo nhãn Xếp Hạng lan truyền (IC Scores) từ đầu vào là các đặc trưng đồ thị.

## 1. I/O Pipeline & Quản Lý Dữ Liệu
Dữ liệu được quản lý thông qua lớp `SurrogateDataBundle`. Việc nạp dữ liệu được tinh chỉnh qua cơ chế `extract_subgraph`, cho phép filter tập trung vào node cần thiết, giảm chi phí VRAM.

## 2. Kiến Trúc Mô Hình (Memory & Performance)
Script triển khai 5 cấu trúc Graph Neural Network cơ bản: SAGE, GCN, GIN, GAT, và APPNP.
Đáng lưu ý:
- SAGE và GCN hoạt động tốt nhờ cơ chế tổng hợp mảng (mean aggregation).
- GAT (Graph Attention Network) đặc biệt "ăn" VRAM do tính chất `concat=True` trên số lượng lớn heads.

## 3. Cơ Chế 5-Seed Ensemble
Mỗi mô hình được huẩn luyện với 5 trạng thái khởi tạo ngẫu nhiên (5 seeds). Kỹ thuật này giúp loại bỏ sự biến động gradient và đem lại kết quả `mean ± std` cực kỳ ổn định.

## 4. Hàm Loss Mục Tiêu (Huber vs Rankloss)
Script khởi khởi động với hàm Huber Loss (giảm nhạy cảm nhiễu) nhưng sau đó chuyển sang Pairwise Margin Rank Loss trong C3 nếu bật flag. Khung Ranking này tối ưu trực tiếp cho bài toán NDCG trên Sub-Network.
