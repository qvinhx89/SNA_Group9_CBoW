# Vận hành Tiềm năng Ảnh hưởng qua IC Weighted-Cascade và Học thay thế trên Twitch

Phiên bản: paper_v1_vietnamese  
Ngày cập nhật số liệu: 2026-04-13

Lưu ý quan trọng: Trong tài liệu này, từ groundtruth được dùng theo nghĩa groundtruth vận hành của Track A (IC-based operational groundtruth) do nhóm tự xây dựng từ pipeline mô phỏng và các artifact kiểm định. Tài liệu không giả định có groundtruth hành vi quan sát trực tiếp ngoài dữ liệu hiện có.

## Tóm tắt
Báo cáo này vận hành tiềm năng ảnh hưởng trên đồ thị Twitch bằng mô phỏng Independent Cascade (IC) weighted-cascade phi tham số, sau đó chuyển trọng tâm sang mục tiêu hồi quy khi nhãn nhị phân top-10 cho thấy nhiễu biên cao. Trên đồ thị 168114 nút và 6797557 cạnh, kết quả cho thấy mức phân kỳ đáng kể giữa độ phổ biến bề mặt (views) và ảnh hưởng cấu trúc (IC). Spearman giữa views và IC đạt 0.468860 (p-value = 9.17e-272, n = 5000), xác nhận chỉ tương quan trung bình thay vì đồng thuận cao. Đồng thời, các kiểm định ổn định nhãn cho thấy jaccard top-decile trung bình 0.306930 (min 0.302083), thấp hơn ngưỡng gate 0.85, cùng boundary_ratio 0.199 và ambiguous_ratio 0.155. Từ đó, chiến lược an toàn và có thể bảo vệ được là: dùng IC continuous làm target chính cho regression; giữ nhãn nhị phân ở vai trò phụ trợ có gắn uncertainty.

## Groundtruth dùng trong báo cáo này (tổng hợp từ Track A)
1. Groundtruth continuous chính: ic_score_mean từ IC weighted-cascade trên 5000 nút gán nhãn (artifact: data/processed/ic_scores_primary.parquet).
2. Groundtruth ổn định nhãn: báo cáo pairwise seed cho top-decile và rank consistency (artifact: outputs/day1_benchmark/ic_label_stability.json).
3. Groundtruth bất định nhãn: phân tích CI crossing, boundary ratio, ambiguous ratio (artifact: outputs/day1_benchmark/ic_label_uncertainty.json và outputs/day1_benchmark/ic_label_uncertainty_consensus.json).
4. Groundtruth tính khả thi vận hành: benchmark runtime, quality gate, stability sweep, và policy comparison của Track A (artifact trong outputs/day1_benchmark/).

Diễn giải thống nhất: mọi kết luận về noisy binary và quyết định chuyển sang regression đều bám trên bộ groundtruth Track A nêu trên.

## I. Câu chuyện phương pháp bạn sẽ trình bày
Mạch nói khuyến nghị cho buổi trình bày:
1. Thiết kế IC như groundtruth vận hành liên tục của Track A: mô phỏng weighted-cascade để đo reach kỳ vọng theo Monte Carlo.
2. Chạy IC ra kết quả nhưng nhãn nhị phân bị noisy ở vùng biên: gate fail trên jaccard và cv_score.
3. Chứng minh noisy là đặc tính cấu trúc, không phải bug cài đặt: kiểm định cộng đồng, threshold sweep, uncertainty CI và kiểm tra dữ liệu nền.
4. Kết hợp các bằng chứng để chuyển sang regression: giữ tín hiệu liên tục IC làm mục tiêu chính, binary chỉ bổ trợ.

## II. Thiết kế IC (IC được thiết kế như thế nào)
- Mô hình lan truyền: weighted-cascade, xác suất kích hoạt cạnh (u,v) = 1/degree(v).
- Dữ liệu nền: active graph sau tiền xử lý gồm 168114 nút, 6797557 cạnh; LCC = 100%.
- Thiết kế benchmark runtime:
  - bench_nodes = 100 (degree-quintile stratified)
  - bench_runs = 50
  - cấu hình mục tiêu: 5000 seeds x 200 runs
- Thiết kế pilot correlation:
  - pilot_nodes = 200, pilot_runs = 50
  - so sánh one-hop với IC để quyết định nhánh mô hình.

## III. Chạy IC ra kết quả như thế nào
### 1) Tính khả thi compute
- per_sim_ms = 0.480275
- projected_total_hours (5000x200) = 0.133410 giờ
- decision_action = proceed_as_planned

### 2) Kết quả tương quan one-hop vs IC (pilot)
- spearman_rho = 0.739190
- p_value = 7.8149e-36
- jaccard_at_10pct = 0.111111 (tương ứng overlap 4/20 ở top-k)
- ndcg_at_10pct = 0.329979
- decision_branch = viable_gnn

## IV. Chứng minh nhãn IC binary bị noisy như thế nào
### 1) Gate ổn định thất bại trên nhãn nhị phân
- cv_score = 0.210879 (< 0.3)
- jaccard_mean = 0.306930 (< 0.85)
- jaccard_min = 0.302083 (< 0.8)
- pass_all = false, quality_mode = provisional

### 2) Nhiễu biên đo được trực tiếp từ uncertainty
- boundary_ratio = 0.199
- ambiguous_ratio = 0.155
- n_boundary_ci_crossing_threshold = 995/5000

### 3) Chính sách nhãn xác nhận vấn đề nằm ở binary boundary
So sánh policy labels trên cùng 5000 node:
- A_hard_top10: positive_ratio = 0.1000, boundary_ratio = 0.83, ambiguous_ratio = 0.684
- B_consensus_top10: positive_ratio = 0.0752, boundary_ratio = 0.6197, ambiguous_ratio = 0.4362
=> Dù consensus giảm nhiễu, nhiễu biên vẫn còn cao, cho thấy vấn đề không biến mất khi chỉ đổi ngưỡng gán nhãn.

## V. Chứng minh đây không phải lỗi pipeline đơn lẻ
### 1) Kiểm chứng lớp dữ liệu nền: đầu vào không cho thấy dấu hiệu lỗi ETL nghiêm trọng
- LCC đạt 100% (n_nodes_total = 168114, n_nodes_lcc = 168114, n_components = 1), nên không có hiện tượng graph vỡ cụm gây sai số do thiếu kết nối toàn cục.
- Dead account chỉ 3.0688%, và nhóm dead có degree/views thấp hơn rõ rệt so với live; điều này cho thấy nhiễu metadata có tồn tại nhưng không phải nguồn chi phối tín hiệu IC.
- Kết luận lớp này: dữ liệu nền đủ sạch để chạy suy luận IC; chưa có bằng chứng lỗi do hỏng input contract.

### 2) Kiểm chứng lớp cài đặt nhãn: nhiễu không biến mất khi đổi policy, nên không phải bug if-else đơn giản
- Với cùng tập 5000 node, đổi từ A_hard_top10 sang B_consensus_top10 có giảm nhiễu nhưng không triệt tiêu:
  - boundary_ratio: 0.83 -> 0.6197
  - ambiguous_ratio: 0.684 -> 0.4362
- Nếu đây là lỗi code nhãn thuần túy (ví dụ sai ngưỡng, sai join), khi sửa policy hợp lý thì nhiễu thường sụt mạnh về mức thấp. Ở đây nhiễu vẫn cao, nghĩa là vấn đề nằm ở bản chất phân phối điểm gần biên.

### 3) Kiểm chứng lớp cấu trúc: có bằng chứng dương tính cho giả thuyết "structural boundary mixing"
- pct_communities_spanning_boundary = 0.842: phần lớn cộng đồng đi qua ranh giới phân loại, tức biên top-10 không tách gọn theo community block.
- mean_gap_to_noise = 0.002393: khoảng cách tín hiệu-biên rất nhỏ, nên dao động Monte Carlo nhẹ cũng đủ làm đổi nhãn ở vùng rìa.
- interpretation = structural từ báo cáo giải thích ổn định: kết luận chính thức của artifact cũng nghiêng về nguyên nhân cấu trúc.

### 4) Kiểm chứng phản chứng: nếu là bug code thì sẽ thấy mẫu khác
Nếu là bug triển khai, thường xuất hiện ít nhất một trong các dấu hiệu sau:
- Metric dao động vô quy luật khi tăng n_runs.
- Kết quả đảo chiều lớn khi thay nhẹ threshold/policy.
- Nhiều chỉ số nền (LCC, dead-audit, join-coherence) đồng thời bất thường.

Những gì quan sát thực tế trong Track A lại ngược lại:
- Spearman stability cho regression tăng dần theo n_runs (xấp xỉ 0.685 -> 0.827), tức hành vi có quy luật hội tụ thay vì ngẫu loạn do bug.
- Threshold robustness của typology vẫn đạt Jaccard cao với base (0.747 ở ngưỡng 0.15 và 0.793 ở ngưỡng 0.25), cho thấy mẫu cấu trúc ổn định tương đối dưới perturbation hợp lý.

### 5) Kết luận defensive để trả lời giảng viên
Kết luận hợp lý nhất từ chuỗi bằng chứng là: bất ổn của nhãn binary top-10 chủ yếu đến từ phân phối điểm IC dày ở vùng biên và trộn cấu trúc liên cộng đồng, không phải lỗi code pipeline đơn lẻ. Vì vậy quyết định chuyển mục tiêu chính sang regression (IC continuous) là quyết định phương pháp luận, không phải "né lỗi triển khai".

### 6) Câu trả lời ngắn (20-30 giây) khi bị hỏi gắt
"Nếu do bug code, chúng em kỳ vọng thấy dấu hiệu hỏng contract hoặc hành vi ngẫu loạn khi thay n_runs/policy. Nhưng thực tế LCC và audit nền đều ổn, nhiễu giảm nhưng không mất khi đổi policy, đồng thời các chỉ số cấu trúc cho thấy 84.2% cộng đồng băng qua biên và gap-to-noise rất nhỏ. Nghĩa là nhiễu nằm ở bản chất boundary của đồ thị, nên em chọn regression continuous làm target chính để giảm rủi ro nhị phân hóa cưỡng bức." 

## VI. Kết hợp yếu tố khác để chuyển sang regression
### 1) Vì sao regression hợp lý hơn ở giai đoạn này
- Báo cáo stability sweep nêu rõ khuyến nghị: tiếp tục dùng regression là mục tiêu chính, binary chỉ bổ sung.
- Spearman stability của regression tăng theo n_runs (từ ~0.685 ở 150 runs lên ~0.827 ở 1200 runs), dù chưa đạt ngưỡng 0.9 nghiêm ngặt.

### 2) Liên hệ trực tiếp với câu chuyện views vs IC
- Trên tập đã gán IC (n = 5000): Spearman(views, ic_score_mean) = 0.468860, p-value = 9.17e-272.
- Hàm ý: views không đại diện đủ cho ảnh hưởng cấu trúc; cần học surrogate theo target IC continuous thay vì target nhị phân cứng.

### 3) Quyết định mô hình cho vòng tiếp theo
- Mục tiêu chính: hồi quy ic_score_mean.
- Mục tiêu phụ: nhãn binary chỉ dùng cho phân tích bổ trợ/risk reporting, bắt buộc đi kèm uncertainty.
- Rule vận hành: công bố rõ quality_mode = provisional và tránh overclaim trên binary top-10.

## VII. Bảng số liệu mới nhất (để nói nhanh)
| Nhóm | Chỉ số | Giá trị |
| --- | --- | --- |
| Dữ liệu nền | n_nodes_active_graph | 168114 |
| Dữ liệu nền | n_edges_active_graph | 6797557 |
| Dữ liệu nền | n_missing_views_filled | 14 |
| Runtime IC | per_sim_ms | 0.480275 |
| Runtime IC | projected_total_hours (5000x200) | 0.133410 |
| Pilot IC | spearman_rho (one-hop vs IC) | 0.739190 |
| Pilot IC | jaccard_at_10pct | 0.111111 |
| Gate | cv_score | 0.210879 |
| Gate | jaccard_mean | 0.306930 |
| Gate | jaccard_min | 0.302083 |
| Uncertainty | boundary_ratio | 0.199 |
| Uncertainty | ambiguous_ratio | 0.155 |
| Divergence | spearman(views, IC) | 0.468860 |

## VIII. Nguồn số liệu groundtruth Track A
- outputs/stage0_data_quality/metrics.json
- outputs/stage0_data_quality/dead_account_report.json
- outputs/stage0_data_quality/lcc_report.json
- outputs/day1_benchmark/ic_runtime_benchmark.json
- outputs/day1_benchmark/one_hop_correlation.json
- outputs/day1_benchmark/quality_gate_report.json
- outputs/day1_benchmark/ic_label_stability.json
- outputs/day1_benchmark/ic_label_uncertainty.json
- outputs/day1_benchmark/stability_explanation.json
- outputs/day1_benchmark/ic_regression_stability.json
- outputs/day1_benchmark/policy_compare/policy_comparison_summary.csv
- outputs/stage3/typology_summary.json
