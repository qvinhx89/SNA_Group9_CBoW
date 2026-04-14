# Track A Presentation Playbook (Person 1)

Mục tiêu tài liệu này: giúp bạn báo cáo đúng flow bắt buộc cho buổi trình bày:
1) IC chạy cấu hình như nào.
2) IC chạy ra kết quả như nào.
3) Từ kết quả đó vì sao chuyển qua regression.
4) Vì sao fail là do cấu trúc mạng chứ không phải pipeline.
5) Trả lời RQ1 bằng groundtruth đã chạy.

## 1) Flow 1 câu để mở đầu

Em đã vận hành IC theo weighted-cascade trên tập labeled 5000 node, đo được continuous signal đủ mạnh nhưng binary top-10 bất ổn ở boundary; từ bằng chứng đó em chọn regression làm mục tiêu chính và chứng minh bất ổn đến từ cấu trúc mạng, từ đó trả lời RQ1 theo kết luận có điều kiện.

## 2) IC cấu hình chạy như nào

### 2.1 Dữ liệu và mô hình
- Active graph: 168114 node, 6797557 cạnh.
- Biểu diễn: CSR deterministic để đảm bảo tái lập.
- Mô hình diffusion: weighted-cascade, p(u,v) = 1/degree(v).

### 2.2 Cấu hình benchmark trước khi chạy full
- bench_nodes = 100, bench_runs = 50.
- target cấu hình chính = 5000 seeds x 200 runs.
- n_jobs = -1.

### 2.3 Cấu hình pilot kiểm tra proxy
- pilot_nodes = 200, pilot_runs = 50.
- sampling: degree-quintile stratified.

### 2.4 Cấu hình full labeling
- n_labeled_nodes = 5000.
- n_runs_per_node = 200.
- Output đồng bộ 3 artifact: ic_scores_primary, regression_targets, classification_labels.

## 3) IC chạy kết quả như nào

### 3.1 Runtime và tính khả thi
- per_sim_ms = 0.480275.
- projected_total_hours (5000 x 200) = 0.133410 giờ.
- decision_action = proceed_as_planned.

Kết luận: IC labeling chạy được trong budget Day-1, không bị nghẽn compute.

### 3.2 Kết quả pilot one-hop vs IC
- Spearman rho = 0.739190, p = 7.8149e-36.
- Jaccard top-10% = 0.111111.
- NDCG top-10% = 0.329979.
- decision_branch = viable_gnn.

Kết luận: one-hop có liên hệ với IC ở mức global, nhưng hụt rõ ở vùng top-k nên không thay được IC.

### 3.3 Kết quả full label và gate ổn định
- Full label: 5000 node, y_top10 = 500 (10%).
- quality gate: cv_score = 0.210879, jaccard_mean = 0.306930, jaccard_min = 0.302083, pass_all = false.
- uncertainty: boundary_ratio = 0.199, ambiguous_ratio = 0.155, n_boundary_ci_crossing_threshold = 995/5000.

Kết luận: binary top-10 không ổn định đủ để làm mục tiêu chính.

## 4) Từ IC result vì sao chuyển sang regression

Quyết định chuyển sang regression dựa trên 3 ý:

1. Continuous IC vẫn giàu thông tin (không degenerate)
- top10_to_median_ratio = 57.528963.

2. Ổn định ranking continuous tăng theo budget
- Spearman mean theo n_runs: 0.6854 (150) -> 0.7180 (300) -> 0.7503 (500) -> 0.7868 (800) -> 0.8267 (1200).

3. Binary top-10 bất ổn ở vùng biên
- boundary và ambiguous cao, nên threshold cứng làm mất ổn định nhãn.

Kết luận vận hành: regression target là nhánh chính; binary top-10 chỉ giữ vai trò phụ trợ kèm uncertainty.

## 5) Chứng minh fail do cấu trúc mạng, không phải pipeline

## 5.1 Bằng chứng cấu trúc trực tiếp
- pct_communities_spanning_boundary = 0.842 (16/19 cộng đồng chạm boundary).
- mean_gap_to_noise = 0.002392857.
- n_thresholds_tested = 28.
- interpretation = structural.

Ý nghĩa: cộng đồng trộn qua vùng biên mạnh và khoảng cách biên nhỏ hơn cỡ nhiễu, nên nhãn nhị phân khó ổn định là hệ quả topology.

### 5.2 Bằng chứng phản chứng với policy
- A_hard_top10: boundary_ratio = 0.83, ambiguous_ratio = 0.684.
- B_consensus_top10: boundary_ratio = 0.6197, ambiguous_ratio = 0.4362.

Ý nghĩa: đổi policy có giảm nhiễu nhưng không triệt tiêu nhiễu. Nếu là lỗi pipeline đơn giản, nhiễu thường phải sụt về mức thấp sau khi sửa policy.

### 5.3 Bằng chứng lớp nền không hỏng contract
- LCC = 100%.
- dead account = 3.0688% và profile yếu hơn live (không phải nguồn chi phối).
- Runtime và output schema đồng bộ qua các artifact chính.

Ý nghĩa: không có dấu hiệu lỗi ETL/pipeline đủ mạnh để giải thích toàn bộ instability quan sát được.

## 6) Trả lời RQ1 bằng groundtruth đã chạy

RQ1: IC operationalization có tạo được ranking influence đủ phân biệt và đủ ổn định để dùng làm surrogate target không?

Trả lời chuẩn để báo cáo:
- Đúng ở tầng continuous: IC tạo được groundtruth usable cho regression (discriminative + có xu hướng ổn định theo n_runs + khả thi compute).
- Không đủ đúng ở tầng binary cứng: top-10 unstable tại boundary nên chỉ dùng bổ trợ.

Kết luận RQ1 theo chuẩn học thuật:
RQ1 được chấp nhận theo dạng có điều kiện, và đây là kết luận tích cực vì nó cho phép pipeline downstream đi tiếp hợp lệ với regression target thay vì dừng nghiên cứu.

## 7) Script thuyết trình 5 phút theo flow mới

### Mở đầu (30-40 giây)
Em báo cáo Track A theo 5 bước: cấu hình IC, kết quả IC, quyết định regression, chứng minh nguyên nhân fail, và chốt trả lời RQ1 bằng groundtruth.

### Bước 1: Cấu hình IC (1 phút)
Em chạy weighted-cascade với p(u,v)=1/degree(v) trên graph active 168114 node, 6797557 cạnh. Sau benchmark 100x50 và pilot 200x50, em chốt cấu hình full 5000 node nhân 200 runs để tạo groundtruth Day-1.

### Bước 2: Kết quả IC (1 phút 20 giây)
Runtime projected chỉ 0.13341 giờ nên khả thi. Pilot one-hop vs IC có rho 0.739 nhưng top-k Jaccard chỉ 0.111, nghĩa là proxy cục bộ không thay được IC ở vùng quan trọng nhất. Khi chạy full labels, gate ổn định binary fail: cv_score 0.210879, jaccard_mean 0.306930, jaccard_min 0.302083.

### Bước 3: Vì sao chuyển regression (50 giây)
Em không dừng vì continuous IC vẫn usable: top10_to_median_ratio 57.53 và Spearman stability tăng dần theo n_runs đến 0.8267 ở 1200 runs. Vì vậy em chuyển mục tiêu chính sang regression và giữ binary ở mức provisional có gắn uncertainty.

### Bước 4: Vì sao fail là do cấu trúc mạng (1 phút)
Em có 3 bằng chứng: một là 84.2% cộng đồng băng qua boundary; hai là gap-to-noise cực nhỏ 0.00239 qua 28 ngưỡng; ba là đổi policy từ hard sang consensus chỉ giảm chứ không xóa nhiễu. Chuỗi này cho thấy nguyên nhân chính là structural boundary mixing, không phải pipeline bug đơn lẻ.

### Bước 5: Chốt RQ1 (30-40 giây)
RQ1 đúng theo nghĩa có điều kiện: đúng mạnh cho continuous surrogate target, chưa đạt cho binary top-10. Groundtruth Day-1 vì vậy được sử dụng hợp lệ cho regression downstream.

## 8) Q&A ngắn khi bị hỏi gắt

- Hỏi: Gate fail sao vẫn đi tiếp?
  Trả lời: Fail ở binary threshold, không fail ở continuous signal. Regression là quyết định phương pháp luận, không phải né lỗi.

- Hỏi: Sao khẳng định không phải pipeline bug?
  Trả lời: Vì có bằng chứng cấu trúc độc lập gồm community overlap cao, gap-to-noise thấp, và policy đổi mà nhiễu vẫn còn.

- Hỏi: RQ1 cuối cùng là đúng hay sai?
  Trả lời: Đúng có điều kiện: đúng cho continuous regression target, không đủ đúng cho binary top-10.

## 9) Danh sách số cần thuộc để nói không nhìn giấy

- 168114 node, 6797557 cạnh.
- Runtime projected 0.133410 giờ (5000 x 200).
- Pilot rho 0.739190, top-k Jaccard 0.111111.
- Gate: cv_score 0.210879, jaccard_mean 0.306930, jaccard_min 0.302083.
- Uncertainty: boundary_ratio 0.199, ambiguous_ratio 0.155, boundary_count 995/5000.
- Structural: 16/19 cộng đồng span boundary, pct = 0.842, mean_gap_to_noise = 0.002392857.
- Regression stability tăng đến Spearman mean 0.8267 ở n_runs = 1200.
