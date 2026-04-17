# Track A Report - Person 1 (M0 -> M3)

Ngày cập nhật: 2026-04-13  
Phạm vi: Tổng hợp đầy đủ 11 task của Person 1 theo parallel coding plan v3, dùng số liệu thật từ artifacts đang có trong workspace.
Ghi chú snapshot thời gian cho Task 4-5: report cập nhật ngày 13/04, nhưng artifact benchmark đang dùng được sinh ngày 07/04 (theo timestamp trong ic_runtime_benchmark.json và one_hop_correlation.json).

Cập nhật theo feedback trước present (13/04): đã rollback 3 artifact Stage 0 (node_attributes.parquet, metrics.json, preprocess_report.json) về baseline an toàn để không phá vỡ contract downstream của Person 2/3; đồng thời giữ nguyên code fix đo lường n_missing_views_filled trong src/data/preprocess_graph.py và toàn bộ chỉnh sửa narrative Task 1-5 trong report này.

## Bộ câu hỏi nghiên cứu chính của project (v3)

1. RQ1: IC operationalization có tạo được ranking influence đủ phân biệt và đủ ổn định để dùng làm surrogate target không?  
   Hypothesis H1: Weighted-cascade IC tạo tín hiệu continuous có ý nghĩa cho regression (discriminative, không degenerate), dù nhãn binary top-10 có thể bất ổn ở boundary. `cv_score` có thể không vượt ngưỡng heuristic `cv_adequate=0.3` trong regime near-critical; readiness cho regression được đánh giá chủ yếu qua độ ổn định thứ hạng (Spearman) và bằng chứng “structural boundary” khi Jaccard nhãn thấp (không quy kết đơn giản là sampling error).

2. RQ2: Views (popularity) đồng thuận tới mức nào với IC-based influence ranking?  
   Hypothesis H2: Mức divergence được khóa theo narrative-tier dựa trên rho(views, IC): strong divergence, moderate divergence, hoặc high agreement. Dù thuộc tier nào, typology Hidden và Overrated vẫn là khung phân tích chính để kiểm định bất đồng rank.

3. RQ2b: Tương quan giữa IC với views, centrality, one-hop, two-hop là gì (global và theo regime)?  
   Hypothesis H3: One-hop và các structural metric sẽ tương quan khá cao với IC, nhưng không thay thế hoàn toàn IC. Two-hop có thể tương quan cao hơn one-hop (gợi ý có thêm thông tin từ 2-hop); đồng thời pattern tương quan có thể không đồng đều giữa các vùng cấu trúc (cần stratified analysis để khóa claim này).

4. RQ3: GNN surrogate có xấp xỉ IC tốt hơn cheap proxies không, và lợi ích tính toán là bao nhiêu?  
   Hypothesis H4: GNN có thể đạt chất lượng ranking cạnh tranh hoặc cao hơn một phần proxy rẻ; lợi thế tốc độ chính là so với MC IC labeling, không mặc định nhanh hơn mọi analytical proxy. Nếu không vượt proxy, kết luận vẫn publishable theo hướng proxy địa phương đã đủ mạnh.

5. RQ3b: Node type nào khó dự đoán nhất bằng mô hình rẻ?  
   Hypothesis H5: Hidden được kỳ vọng là nhóm khó dự đoán nhất (Spearman thấp hơn, MAE cao hơn), vì đây là outlier cấu trúc mà raw popularity khó nắm bắt. Claim chỉ khóa mức kết luận cuối khi đã có per-group metrics đầy đủ cho các model chính theo plan.

6. RQ4: Đặc trưng cấu trúc nào phân biệt nhóm có rank views và rank IC bất đồng?  
   Hypothesis H6: Hidden được kỳ vọng có tín hiệu cấu trúc nổi bật hơn Overrated (đặc biệt ở cross-community connectivity và các chỉ số lõi), còn Overrated nghiêng về popularity/surface signal hơn là vị trí lan truyền hiệu quả. Diễn giải tổng quát phải đi kèm caveat null-model để tránh overclaim structural uniqueness.

---

## Task 1 - CSR export

### File liên quan trực tiếp để kiểm tra

- Code chính: src/mapr2026_v3/export_csr.py
- Input chính: data/processed/graph_active.edgelist
- Output/check chính: data/processed/graph_csr.npz, outputs/stage0_data_quality/metrics.json

### Bối cảnh task (1 câu)

Task 1 thuộc milestone M1 (6/4), là hạ tầng bắt buộc cho RQ1 và RQ3 vì mọi bước IC simulation và benchmark runtime đều chạy trên biểu diễn CSR.

### Mục đích khoa học + hypothesis liên quan

Đây là task chuẩn bị kỹ thuật, nhưng có ý nghĩa khoa học gián tiếp rất quan trọng: nếu mapping node không deterministic thì mọi so sánh ổn định (H1) và so sánh surrogate-vs-IC (H4) đều mất tính hợp lệ. Nói cách khác, task này kiểm định điều kiện đo lường trước khi kiểm định giả thuyết. Nó bảo vệ luận điểm rằng divergence hay instability quan sát được về sau là hiện tượng của mạng, không phải lỗi do thay đổi thứ tự node giữa các lần chạy.

### Thiết kế và protocol
Script đọc edge list active, dựng đồ thị vô hướng bằng cách thêm cả hai chiều cạnh, rồi sắp xếp theo thứ tự lexicographic để khóa tính tái lập. Sau đó script xuất contract CSR gồm indptr, indices, degrees, node_ids vào graph_csr.npz. Rule pass thực thi gồm: file tồn tại, đủ 4 key schema, và quy mô khớp với dữ liệu preprocess. Rule kiểm chứng bổ sung: nnz trong CSR phải khớp đúng với 2 nhân số cạnh active vì biểu diễn vô hướng lưu cả hai chiều. Artifact này được các script day1_benchmark và ic_labels_primary dùng trực tiếp.

### Kết quả chính
CSR hiện tại có 168114 nodes và 13595114 non-zero entries. Degree mean là 80.8684, median là 32, min là 1, max là 35279, cho thấy phân phối bậc lệch mạnh. Trong metrics stage0, n_edges_active là 6797557, và nnz đúng bằng 2 nhân 6797557. Điều này xác nhận export vô hướng đang đúng contract.

### Phân tích diễn giải

Kết quả ủng hộ mục tiêu tái lập của Task 1: nền tảng đo lường đã ổn định, nên các kết luận ở Task 5-8 có thể diễn giải theo ý nghĩa học thuật thay vì nghi ngờ lỗi dữ liệu. Với đồ thị cỡ 168k node, CSR cũng giúp giảm overhead truy cập lân cận cho mô phỏng cascade. Về học thuật, task này không xác nhận H1/H4 trực tiếp, nhưng là điều kiện cần để mọi kiểm định sau đó có giá trị.

### Rủi ro/giới hạn
Script hiện vẫn là bản dựng adjacency trực tiếp, có thể gây áp lực RAM nếu quy mô graph tăng thêm. Ngoài ra, chưa có artifact benchmark riêng cho thời gian export CSR. Rủi ro governance còn lại là rerun input mà không khóa version tag mới.

### Kết luận hành động

Trạng thái: done.  
Quyết định tiếp theo: tiếp tục freeze theo checksum mỗi lần rerun để giữ reproducibility.

---

## Task 2 - Dead account audit

### File liên quan trực tiếp để kiểm tra

- Code chính: src/data/dead_account_audit.py
- Input chính: data/raw/large_twitch_features.csv, data/raw/large_twitch_edges.csv
- Output/check chính: outputs/stage0_data_quality/dead_account_report.json

### Bối cảnh task (1 câu)

Task 2 thuộc milestone M0 (6/4), phục vụ kiểm soát bias cho RQ1-RQ4 bằng cách kiểm toán mức độ nhiễu do dead account.

### Mục đích khoa học + hypothesis liên quan

Task này kiểm định một giả định quan trọng trước phân tích influence: dead account có đủ lớn để làm méo tín hiệu cấu trúc và popularity hay không. Nó không kiểm định trực tiếp H1-H6, nhưng đóng vai trò kiểm định chất lượng bối cảnh thực nghiệm để tránh overclaim ở RQ4. Nếu dead account có degree hoặc views bất thường, kết quả diffusion có thể phản ánh artifact dữ liệu hơn là cơ chế lan truyền thực.

### Thiết kế và protocol
Script đọc raw features và edges, tự suy luận tên cột đầu-cuối cạnh theo các pattern phổ biến, sau đó tính degree bằng tần suất xuất hiện node trên cả hai đầu cạnh. Dữ liệu được chuẩn hóa node_id thành string và merge degree vào bảng node. Script tách hai nhóm dead và live theo cột dead_account, rồi báo cáo số lượng và trung bình views, degree theo từng nhóm. Rule pass: không thiếu cột bắt buộc numeric_id, dead_account, views; report JSON ghi đủ key contract.

### Kết quả chính

Kết quả cho thấy n_dead = 5159 và n_live = 162955, tương ứng pct_dead = 3.0688%. Mean degree dead = 17.4582 thấp hơn mạnh so với live = 82.8759. Mean views dead = 2555.61 trong khi live = 194037.88. Tức là nhóm dead vừa ít về số lượng, vừa yếu về kết nối và tín hiệu popularity.

### Phân tích diễn giải

Kết quả nghiêng về việc dead account không phải động lực chính tạo ra tín hiệu influence quan sát được. Điều này giúp củng cố diễn giải rằng instability top-10 ở các task sau khó quy hoàn toàn cho lỗi vệ sinh dữ liệu. Về học thuật, đây là phần caveat bắt buộc để diễn giải divergence có trách nhiệm: ta có bằng chứng định lượng rằng nguồn nhiễu dead account tồn tại, nhưng không phải thành phần chi phối cấu trúc.

### Rủi ro/giới hạn
Định nghĩa dead_account phụ thuộc metadata tại thời điểm crawl, có thể lệch theo thời gian. Báo cáo chưa phân rã theo community hoặc degree regime nên chưa biết dead account tập trung ở vùng cấu trúc nào. Nếu raw schema đổi, cơ chế suy luận cột cạnh cần kiểm tra lại.

### Kết luận hành động

Trạng thái: done.  
Quyết định tiếp theo: giữ report này như caveat bắt buộc trong mọi bản thảo kết quả.

---

## Task 3 - LCC check

### File liên quan trực tiếp để kiểm tra

- Code chính: src/data/lcc_audit.py
- Input chính: data/processed/graph_active.edgelist
- Output/check chính: outputs/stage0_data_quality/lcc_report.json

### Bối cảnh task (1 câu)

Task 3 thuộc milestone M0 (6/4), phục vụ RQ1 và RQ3 bằng cách xác nhận đồ thị active đủ liên thông để mô phỏng lan truyền có ý nghĩa toàn cục.

### Mục đích khoa học + hypothesis liên quan

Mục đích là kiểm định tính hợp lệ của không gian thực nghiệm: nếu đồ thị bị tách nhiều thành phần, ranking IC sẽ trộn lẫn hiệu ứng thành phần và hiệu ứng diffusion, làm mờ diễn giải cho H1/H4. Vì vậy task này kiểm định điều kiện nền, không chỉ kiểm tra kỹ thuật. Khi LCC cao, ta có quyền diễn giải các kết quả ổn định/bất ổn sau đó ở cấp mạng thống nhất.

### Thiết kế và protocol

Script đọc edge list active bằng networkx, tính connected components, lấy số node toàn graph, số node của largest connected component, và tỷ lệ phần trăm LCC. Rule pass thực tế Day-1 là pct_lcc rất cao (định hướng >= 95%) để xem các mô hình lan truyền có cùng không gian hoạt động. Artifact JSON lưu đầy đủ n_nodes_total, n_nodes_lcc, pct_lcc, n_components, kèm timestamp để truy vết.

### Kết quả chính

Kết quả hiện tại: n_nodes_total = 168114, n_nodes_lcc = 168114, pct_lcc = 100.0, n_components = 1. Nghĩa là toàn bộ đồ thị active nằm trong một thành phần liên thông duy nhất. Không có island component cần xử lý riêng.

### Phân tích diễn giải
Kết quả 100% LCC ủng hộ mạnh tính hợp lệ nội bộ cho pipeline IC và surrogate. Khi binary top-10 bất ổn, ta không cần ưu tiên giả thuyết lỗi do graph rời rạc. Về mặt học thuật, điều này giúp các so sánh one-hop, IC, GNN nằm trên cùng một nền topological, tăng độ tin cậy cho diễn giải chênh lệch performance.

### Rủi ro/giới hạn
Kết quả phụ thuộc snapshot active hiện tại; nếu preprocess thay đổi phải rerun. Báo cáo LCC không phản ánh các đặc tính hình học khác như đường kính hay modularity. Script đọc toàn bộ graph vào memory nên cần theo dõi khi data tăng.

### Kết luận hành động

Trạng thái: done.  
Quyết định tiếp theo: rerun LCC audit khi có thay đổi preprocess lớn.

---

## Task 4 - Day-1 benchmark runtime

### File liên quan trực tiếp để kiểm tra
- Code chính: src/mapr2026_v3/day1_benchmark.py  
- Input chính: data/processed/graph_csr.npz  
- Output/check chính: outputs/day1_benchmark/ic_runtime_benchmark.json, outputs/mapr2026_v3_results/runtime_breakdown.csv

### Bối cảnh task (1 câu)

Task 4 thuộc milestone M2 (7/4), phục vụ RQ3 và H4 bằng cách đo chi phí thực của MC IC labeling trước khi so với surrogate/proxy.

### Mục đích khoa học + hypothesis liên quan

Task này kiểm định tính khả thi tính toán cho nhánh nghiên cứu surrogate: liệu có đủ budget để dùng IC làm mốc so sánh hay không. H4 nói về lợi ích tốc độ của surrogate so với MC labeling, vì vậy phải có baseline runtime định lượng và tái lập. Đây không phải task chứng minh model nào tốt hơn, mà là task khóa điều kiện thử nghiệm.

### Thiết kế và protocol
Protocol benchmark dùng 100 node (degree-quintile stratified), chạy 50 mô phỏng mỗi node để ước lượng per_sim_ms. Từ per_sim_ms, script nội suy projected_total_hours cho cấu hình chính 5000 x 200 và hai cấu hình giảm tải 3000 x 150, 2000 x 100. Rule quyết định khóa theo plan: projected < 4h thì proceed_as_planned; 4-8h giảm tải; >8h tối thiểu hóa budget. Kết quả được ghi vào artifact JSON và thêm hàng mc_ic_labeling trong runtime_breakdown.csv để so với các baseline khác.

### Kết quả chính
per_sim_ms = 0.480275 và projected_total_hours cho 5000 x 200 là 0.13341 giờ. Hai phương án dự phòng lần lượt là 0.06003 giờ (3000 x 150) và 0.02668 giờ (2000 x 100). Decision action là proceed_as_planned. Trong runtime_breakdown.csv, mc_ic_labeling được ghi là 480.275 giây full-graph-equivalent.

### Phân tích diễn giải

Kết quả ủng hộ điều kiện vận hành của H4: MC IC labeling không bị compute bottleneck trên hệ hiện tại. Vì vậy so sánh surrogate-vs-IC có thể triển khai thực nghiệm thay vì chỉ mang tính ý tưởng. Về học thuật, baseline runtime định lượng giúp diễn giải lợi ích tính toán theo ratio minh bạch, tránh kết luận cảm tính.

### Rủi ro/giới hạn

Runtime nhạy với phần cứng và tải máy, nên không xem như hằng số tuyệt đối. Sample benchmark chỉ 100 node, có thể chưa phản ánh đầy đủ tail behavior của hub cực lớn. Tài liệu quyết định hiện có nhiều con số runtime theo ngữ cảnh khác nhau, cần đồng bộ định nghĩa khi viết báo cáo cuối.

### Kết luận hành động

Trạng thái: done.  
Quyết định tiếp theo: giữ cấu hình 5000 x 200 làm baseline chính thức trong chu kỳ hiện tại.

---

## Task 5 - One-hop rho check

### File liên quan trực tiếp để kiểm tra

- Code chính: src/mapr2026_v3/day1_benchmark.py
- Input chính: data/processed/graph_csr.npz
- Output/check chính: outputs/day1_benchmark/one_hop_correlation.json

### Bối cảnh task (1 câu)

Task 5 thuộc milestone M2 (7/4), phục vụ RQ2b và RQ3 bằng cách kiểm tra one-hop có đủ thay thế IC hay vẫn còn khoảng trống cho surrogate.

### Mục đích khoa học + hypothesis liên quan

Task này kiểm định phần trọng tâm của H3: one-hop có thể tương quan cao với IC nhưng không thay thế hoàn toàn. Đồng thời, nó là gate thực nghiệm cho H4: nếu one-hop quá sát IC, động lực học surrogate sẽ giảm. Vì vậy task đóng vai trò phân nhánh phương pháp, không chỉ báo cáo tương quan.

### Thiết kế và protocol
Script lấy 200 pilot nodes theo stratified degree, chạy IC 50 runs/node để lấy ic_mean, đồng thời tính one-hop spread bằng tổng 1/degree của hàng xóm. Các chỉ số báo cáo gồm Spearman rho toàn cục, p-value, Jaccard@10% cho overlap top-k, và NDCG@10% cho chất lượng ranking top-k. Rule nhánh: rho < 0.8 thì viable_gnn; 0.8-0.9 thì two_hop_primary; >0.9 mới cân nhắc restructure.

### Kết quả chính
spearman_rho = 0.739190, p_value = 7.8149e-36, jaccard_at_10pct = 0.111111, ndcg_at_10pct = 0.329979. Decision branch là viable_gnn. Số node hợp lệ trong pilot là 200.

### Phân tích diễn giải

Kết quả vừa ủng hộ vừa giới hạn H3: liên hệ thứ hạng có thật và có ý nghĩa thống kê, nhưng overlap top-k rất thấp. Điều này cho thấy one-hop là proxy tốt ở mức tổng quát, nhưng mất thông tin ở vùng ảnh hưởng cao nhất. Về học thuật, đây là bằng chứng thực nghiệm quan trọng để giữ hướng surrogate thay vì dừng ở baseline rẻ.

### Rủi ro/giới hạn
Pilot cỡ 200 node có thể dao động ở top-k do tail distribution nặng. Kết quả phụ thuộc seed và stochastic cascade dù đã cố định quy tắc seed. One-hop là metric cục bộ nên không phản ánh đầy đủ tương tác đa-bước.

### Kết luận hành động

Trạng thái: done.  
Quyết định tiếp theo: giữ nhánh viable_gnn và tiếp tục so sánh proxy/surrogate ở các stage sau.

---

## Task 6 - IC pilot + stability gate

### File liên quan trực tiếp để kiểm tra

- Code chính: src/mapr2026_v3/ic_pilot_diagnostics.py, src/mapr2026_v3/ic_label_stability.py, src/mapr2026_v3/freeze_day1_handoff.py
- Input chính: data/processed/graph_csr.npz, data/processed/ic_scores_primary.parquet, data/processed/centrality_table.parquet
- Output/check chính: outputs/day1_benchmark/ic_pilot_diagnostics.json, outputs/day1_benchmark/ic_label_stability.json, outputs/day1_benchmark/quality_gate_report.json

### Bối cảnh task (1 câu)

Task 6 thuộc nhóm quality gate cuối M2 (deadline 9/4), phục vụ trực tiếp RQ1/H1 bằng cách kiểm định độ ổn định của tín hiệu IC trước khi handoff.

### Mục đích khoa học + hypothesis liên quan

Task này kiểm định đồng thời hai giả thiết thành phần của H1: continuous IC có degenerate hay không, và binary top-10 có ổn định hay không. Điểm quan trọng là tách bạch hai tầng kết luận thay vì gom chung. Nếu binary thất bại nhưng continuous còn tốt, nghiên cứu vẫn có đường đi theo regression surrogate.

### Thiết kế và protocol

Pilot diagnostics chạy 200 node, 50 runs/node với hai nhánh seed A/B để đo rank_stability và jaccard_stability. CV được tính theo per-node std/mean; cv_score là trung bình CV trên phần node không vượt noise threshold 0.5. Bảng per_quintile_cv được bổ sung để nhìn theo regime degree. Song song, ic_label_stability chạy 5000 node với 3 MC seeds (0,1,2), 150 runs/seed, tính pairwise Jaccard top-decile và Spearman rank. Rule pass của quality gate: cv_score > 0.3, jaccard_mean >= 0.85, jaccard_min >= 0.80.

### Kết quả chính

Pilot: mean_reach = 26.2124, median_reach = 3.28, top10_to_median_ratio = 57.529, cv_score = 0.210879, cv_noise_count = 166. Pilot jaccard_stability = 0.142857, rank_stability = 0.673199. Full stability: jaccard_mean = 0.306930, jaccard_min = 0.302083, spearman_mean = 0.685384. quality_gate_report ghi pass_all = false (quality_mode provisional).

### Phân tích diễn giải

Kết quả ủng hộ một phần H1: tín hiệu continuous không sụp (top10_to_median_ratio cao), nhưng binary top-10 quá nhạy ở boundary nên thất bại gate ổn định. Về học thuật, điều này rất quan trọng vì nó cho phép kết luận có điều kiện thay vì kết luận trắng-đen. Option B vì vậy không phải né lỗi, mà là cách quản trị uncertainty minh bạch khi tiếp tục nghiên cứu.

### Rủi ro/giới hạn

Ngưỡng Jaccard hiện hành khá chặt cho đồ thị có boundary dày, có thể dẫn tới fail lặp lại. cv_score phụ thuộc định nghĩa noise nên nhạy với rule cắt. Pilot 200 node vẫn có sai số lấy mẫu dù đã stratified.

### Kết luận hành động

Trạng thái: partial (gate fail nhưng đã governance theo Option B).  
Quyết định tiếp theo: dùng regression target làm nhánh chính, binary ở mức provisional có kèm uncertainty.

---

## Task 7 - IC labels full N x R

### File liên quan trực tiếp để kiểm tra

- Code chính: src/mapr2026_v3/ic_labels_primary.py
- Input chính: data/processed/graph_csr.npz
- Output/check chính: data/processed/ic_scores_primary.parquet, data/processed/regression_targets.parquet, data/processed/classification_labels.parquet

### Bối cảnh task (1 câu)

Task 7 (deadline 10/4) là đầu ra lõi của cuối M2, phục vụ trực tiếp RQ1 và làm input bắt buộc cho RQ2-RQ4.

### Mục đích khoa học + hypothesis liên quan

Task này kiểm định năng lực operationalization của H1 ở cấp artifact chính thức: có tạo được target continuous giàu thông tin và nhãn top-10 theo quy tắc lock hay không. Không có task này thì mọi so sánh model ở downstream đều mất nền tham chiếu. Đây là bước biến mô phỏng thành dữ liệu huấn luyện/đánh giá có thể tái sử dụng.

### Thiết kế và protocol

Script lấy mẫu 5000 node theo degree-quintile stratified (seed 42), chạy weighted-cascade IC với n_runs = 200 và worker_seed = seed + node_row. Xuất 3 artifact: ic_scores_primary (mean/std/runs/model), regression_targets với y = log1p(ic_score_mean), và classification_labels với y_top10 theo quantile 90%. Rule pass gồm schema đầy đủ và đồng bộ node_id giữa 3 file. Sau đó uncertainty pipeline tạo CI để đo mức nhạy biên threshold.

### Kết quả chính

Cả 3 artifact đều có 5000 dòng, n_runs_unique = [200], y_top10 có đúng 500 node (10%). Phân phối ic_score_mean: mean = 31.0962, median = 6.2525, p90 = 77.67, max = 2317.71. Uncertainty báo cáo boundary CI crossing là 995/5000 (19.9%), ambiguous là 775/5000 (15.5%).

### Phân tích diễn giải

Kết quả ủng hộ mạnh phần continuous của H1: signal có độ phân biệt rộng và không degenerate, phù hợp làm surrogate regression target. Ngược lại, nhãn binary top-10 mang uncertainty cao gần ngưỡng cắt, giải thích nhất quán với fail stability ở Task 6. Về học thuật, đây là kết quả publishable kiểu two-layer claim: mạnh ở continuous, thận trọng ở binary.

### Rủi ro/giới hạn

Labeling chỉ trên 5000/168114 node nên vẫn có rủi ro sampling bias. Quy tắc top-10 là ngưỡng cắt rời rạc, nhạy với biến động nhỏ quanh threshold. Khi đổi budget hoặc p_model, phải freeze version mới thay vì ghi đè.

### Kết luận hành động

Trạng thái: done (kèm caveat cho binary).  
Quyết định tiếp theo: regression target là nhánh tiêu thụ chính cho downstream.

---

## Task 8 - Stability explanation (bắt buộc khi Jaccard thấp)

### File liên quan trực tiếp để kiểm tra

- Code chính: quy trình extract feasibility + freeze lockstep (tích hợp trong freeze pipeline)
- Input chính: data/processed/ic_scores_primary.parquet, data/processed/community_labels.parquet
- Output/check chính: outputs/ic_feasibility/phase1_community_overlap.json, outputs/ic_feasibility/phase2_threshold_analysis.json, outputs/day1_benchmark/stability_explanation.json

### Bối cảnh task (1 câu)

Task 8 thuộc mốc 10/4, là task bắt buộc theo plan khi stability binary thấp, nhằm trả lời nguyên nhân khoa học thay vì chỉ báo cáo thất bại kỹ thuật.

### Mục đích khoa học + hypothesis liên quan

Task này kiểm định giả thuyết phụ của H1: Jaccard thấp có thể là hệ quả cấu trúc graph và độ dày boundary, không phải lỗi sampling đơn thuần. Nếu đúng, kết luận về continuous target vẫn đứng vững và narrative Option B có cơ sở học thuật. Đây là bước biến tín hiệu fail thành insight có thể diễn giải.

### Thiết kế và protocol

Phase 1 đo overlap community giữa top-10 và boundary band 10-20%, báo cáo community-set Jaccard và tỷ lệ community span cả hai vùng. Rule pivot phase 1: overlap cao vượt ngưỡng feasibility thì nghiêng về structural mixing. Phase 2 sweep 28 ngưỡng từ 3% đến 30%, tính estimated_jaccard và gap_to_noise để xem có vùng threshold nào đạt mức khả thi >= 0.7 không. stability_explanation.json tổng hợp 3 chỉ số khóa và kết luận interpretation.

### Kết quả chính

Phase 1: community_overlap_jaccard = 0.842 và 16/19 community span boundary (84.21%). Phase 2: n_thresholds_tested = 28, best_threshold_pct = 0.30, best_estimated_jaccard = 0.6566, target_threshold_10pct_jaccard = 0.5149. summary mean_gap_to_noise = 0.002392857. stability_explanation kết luận interpretation = structural.

### Phân tích diễn giải

Kết quả ủng hộ mạnh hướng structural explanation: top và cận biên cùng chia sẻ tầng community, nên cắt nhị phân theo top-k khó ổn định về bản chất. Về học thuật, đây là điểm rất quan trọng vì nó bảo toàn giá trị nghiên cứu khi binary gate fail. Thay vì coi đó là lỗi pipeline, ta có thể công bố như một phát hiện về topology-induced instability.

### Rủi ro/giới hạn

estimated_jaccard trong phase 2 là thước suy luận, không thay thế hoàn toàn full rerun cho mọi threshold. Kết quả phụ thuộc community_labels hiện hành, có thể đổi khi đổi thuật toán partition. Cần khóa rõ nguồn dữ liệu để tránh circular traceability.

### Kết luận hành động

Trạng thái: done.  
Quyết định tiếp theo: giữ narrative structural divergence và tiếp tục nhánh regression.

---

## Task 9 - Split mask M0-locked

### File liên quan trực tiếp để kiểm tra

- Code chính: src/mapr2026_v3/ic_labels_primary.py, src/mapr2026_v3/freeze_day1_handoff.py
- Input chính: data/processed/ic_scores_primary.parquet, data/processed/graph_csr.npz
- Output/check chính: data/processed/split_masks.parquet, outputs/day1_benchmark/split_freeze_manifest.json

### Bối cảnh task (1 câu)

Task 9 thuộc deadline 10/4 nhưng là rule M0-locked, phục vụ RQ3-RQ4 bằng cách khóa công bằng train/test cho mọi consumer.

### Mục đích khoa học + hypothesis liên quan

Task này không kiểm định giả thuyết nội dung mà kiểm định tính hợp lệ so sánh thực nghiệm. Nếu mỗi nhóm tự chia split khác nhau, kết luận H4-H6 có nguy cơ bị split variance che lấp. Vì vậy đây là tầng kiểm soát phương pháp để bảo vệ tính công bằng giữa baseline/proxy/surrogate.

### Thiết kế và protocol

Split được tạo trên đúng tập labeled 5000 node với test_frac = 0.20, stratify theo degree_quintile, seed = 42. Output split_masks gồm node_id và split train/test. Freeze pipeline tính SHA256, size, lock rule, và ghi consumer_rule cấm local re-splitting vào split_freeze_manifest. Rule pass gồm: số dòng bằng số labeled, tỷ lệ 80/20 đúng, checksum được lock trong manifest và handoff version.

### Kết quả chính

split_masks có 5000 dòng, trong đó train = 4000, test = 1000. split_freeze_manifest ghi SHA256 = 005de40762f6c75e4df66a53efeaa883d126d52abd5c4af0224d736992362104, size = 35759 bytes. Rule lock cũng ghi rõ stratify_by degree_quintile và seed 42. versioned_handoff_dir trỏ tới gói v3i optionB lockstep.

### Phân tích diễn giải

Kết quả này củng cố độ tin cậy của mọi benchmark downstream vì cùng dùng một split bất biến. Về học thuật, đây là tiền đề để diễn giải chênh lệch model theo năng lực thực, không phải theo may rủi chia dữ liệu. Stratify theo degree cũng làm giảm rủi ro mất cân bằng cấu trúc giữa train/test.

### Rủi ro/giới hạn

Split không stratify trực tiếp theo nhãn top-10 hay typology, nên đánh giá theo nhóm hiếm vẫn có thể dao động. Nếu phạm vi labeled node thay đổi, split cũ không còn hợp lệ. Rủi ro lớn nhất là vi phạm governance khi consumer tự tách dữ liệu lại.

### Kết luận hành động

Trạng thái: done.  
Quyết định tiếp theo: bắt buộc toàn team tiêu thụ đúng split đã freeze trong package active.

---

## Task 10 - Ghi và khóa day1_decisions

### File liên quan trực tiếp để kiểm tra

- Code chính: docs/day1_decisions.md (governance artifact), hỗ trợ tự động tại src/mapr2026_v3/ic_labels_primary.py
- Input chính: outputs/day1_benchmark/_.json, outputs/ic_feasibility/_.json
- Output/check chính: docs/day1_decisions.md, outputs/handoffs/person1_day1_20260409_p1_day1_v3i_optionB_lockstep/manifest.json

### Bối cảnh task (1 câu)

Task 10 thuộc M2 nhưng được cập nhật xuyên mốc để khóa quyết định vận hành và narrative nhất quán cho toàn bộ RQ.

### Mục đích khoa học + hypothesis liên quan

Task này kiểm định tính minh bạch quy trình nghiên cứu: khi gate fail, nhóm có ghi nhận trung thực và khóa quyết định tiêu thụ dữ liệu rõ ràng hay không. Nó không kiểm định nội dung H1-H6 trực tiếp, nhưng là điều kiện để các kết luận học thuật có thể audit và tái lập. Không có task này, downstream dễ rơi vào tình trạng mỗi nhánh hiểu một policy khác nhau.

### Thiết kế và protocol

Tài liệu day1_decisions tổng hợp lệnh chạy chính thức, kết quả benchmark, ngưỡng gate, trạng thái pass/fail, mode provisional/final, và quy tắc consume split/labels. Khi stability fail, section về explanation và Option B phải được cập nhật đồng bộ với artifacts tương ứng. M3 section cũng được gắn cơ chế auto-update từ script để tránh ghi tay lệch số. Rule pass là: tài liệu đủ chứng cứ tái lập và khớp với manifest handoff active.

### Kết quả chính

day1_decisions hiện ghi rõ gate fail với cv_score = 0.2108788620, jaccard_mean = 0.3069298298, jaccard_min = 0.3020833333. Tài liệu khóa package active là person1_day1_20260409_p1_day1_v3i_optionB_lockstep. Handoff manifest ghi n_files = 27, quality_gate_pass_all = false, khớp với quality_gate_report. Section M3 đã có số rho views-IC và tier tương ứng.

### Phân tích diễn giải

Task đạt mục tiêu governance: kết quả không đẹp vẫn được báo cáo đầy đủ, không bị chỉnh sửa theo hướng thuận lợi giả tạo. Về học thuật, đây là thực hành quan trọng để kết luận có trách nhiệm trong bối cảnh uncertainty cao. Nó cũng là cầu nối vận hành giữa Person 1 và Person 2/3, giảm mạnh rủi ro lệch protocol tiêu thụ artifact.

### Rủi ro/giới hạn

Tài liệu dài nên có nguy cơ stale nếu không cập nhật đồng bộ mỗi lần rerun. Một số con số runtime có thể khác nhau theo ngữ cảnh đo (single run, median planning, benchmark artifact), cần nêu rõ định nghĩa khi trích dẫn. Governance vẫn phụ thuộc kỷ luật tuân thủ của team.

### Kết luận hành động

Trạng thái: done.  
Quyết định tiếp theo: duy trì pattern lockstep, mọi thay đổi phải tạo version tag mới thay vì ghi đè.

---

## Task 11 - M3 views/IC alignment check

### File liên quan trực tiếp để kiểm tra

- Code chính: src/mapr2026_v3/ic_labels_primary.py (compute + update M3 section)
- Input chính: data/processed/ic_scores_primary.parquet, data/processed/node_attributes.parquet
- Output/check chính: section M3 trong docs/day1_decisions.md

### Bối cảnh task (1 câu)

Task 11 thuộc milestone M3, phục vụ trực tiếp RQ2/H2 bằng cách khóa narrative tier giữa popularity và diffusion influence.

### Mục đích khoa học + hypothesis liên quan

Task này kiểm định trực tiếp H2: views và IC thuộc tier đồng thuận nào trong chu kỳ hiện tại. Mục tiêu khoa học không phải chứng minh nhân quả, mà là đặt nền diễn giải cho typology Hidden/Overrated dựa trên bằng chứng rank divergence. Kết quả của task quyết định giọng điệu học thuật cho các phân tích RQ2 và RQ4.

### Thiết kế và protocol

Script join node_id giữa IC scores và node attributes, chuẩn hóa views thành numeric và loại missing trước khi tính Spearman. Sau đó map rho vào narrative tier bằng ngưỡng đã khóa: rho < 0.70 là strong_divergence, 0.70-0.85 là moderate, >0.85 là high_agreement. Cơ chế update tự động section M3 giúp giảm lỗi copy số thủ công. Rule pass: đủ overlap node, rho và p-value hữu hạn, section được cập nhật đúng format.

### Kết quả chính

Trên 5000 node overlap, spearmanr(views, ic_score_mean) = 0.46886009249660393 và p_value = 9.170499016140683e-272. Theo lookup table, tier là strong_divergence. day1_decisions section M3 đã phản ánh đúng kết quả này.

### Phân tích diễn giải

Kết quả ủng hộ H2 theo nhánh divergence: popularity không thể thay thế diffusion potential ở cấp rank. Điều này giải thích vì sao cần giữ khung Hidden/Overrated cho phân tích downstream. Về học thuật, p-value rất nhỏ xác nhận liên hệ thống kê tồn tại, nhưng độ lớn rho 0.469 cho thấy mức đồng thuận chỉ vừa phải, không đủ để đồng nhất hai khái niệm influence và popularity.

### Rủi ro/giới hạn

Spearman chỉ đo tính đơn điệu, không phản ánh cấu trúc phi tuyến hoặc dị thường cục bộ. Kết quả có thể nhạy với xử lý outlier views cực lớn. Đây là kết quả global, chưa thay thế phân tích theo regime degree/community của RQ2b.

### Kết luận hành động

Trạng thái: done.  
Quyết định tiếp theo: giữ narrative strong_divergence cho chu kỳ hiện tại và tiếp tục kiểm định theo nhóm cấu trúc ở các stage sau.

---

## Tổng kết trạng thái 11 task Person 1

- Done: Task 1, 2, 3, 4, 5, 7, 8, 9, 10, 11.
- Partial: Task 6 (quality gate fail theo ngưỡng cứng, nhưng đã được quản trị theo Option B).
- Blocked: Không có task bị blocked kỹ thuật trong checklist Person 1.

Kết luận vận hành: Person 1 đã hoàn thành trọn gói Day-1 lockstep v3i với chuỗi artifact có thể truy vết, kiểm chứng và bàn giao; điểm cần thận trọng còn lại là instability của nhãn binary top-10, đã được lượng hóa và diễn giải minh bạch.

---

## Chứng minh RQ1 (dùng khi bảo vệ)

### Phát biểu cần chứng minh
RQ1: IC operationalization có tạo được ranking influence đủ phân biệt và đủ ổn định để dùng làm surrogate target hay không?

Theo H1 đã khóa ở đầu báo cáo, đây là mệnh đề hai tầng:
- Tầng A (continuous): tín hiệu IC phải đủ phân biệt, không degenerate, dùng được cho regression.
- Tầng B (binary): nhãn top-10 có thể bất ổn ở boundary, nhưng bất ổn đó phải được lượng hóa và diễn giải minh bạch.

### Chuỗi bằng chứng định lượng

1) Tầng A đạt yêu cầu về độ phân biệt tín hiệu
- Trong pilot diagnostics: mean_reach = 26.2124, median_reach = 3.28, top10_to_median_ratio = 57.528963.
- Trong full labeling 5000 node: ic_score_mean có mean = 31.0962, median = 6.2525, p90 = 77.67, max = 2317.71.

Diễn giải: phân phối có biên độ rộng, không sụp về một cụm hẹp, nên đủ thông tin để làm regression target.

2) Tầng A có tính ổn định theo hướng hội tụ khi tăng budget
- Stability sweep cho Spearman mean theo n_runs: 150 -> 0.6854, 300 -> 0.7180, 500 -> 0.7503, 800 -> 0.7868, 1200 -> 0.8267.

Diễn giải: xu hướng tăng đơn điệu theo budget cho thấy tín hiệu continuous có trật tự hội tụ, không phải nhiễu ngẫu nhiên do pipeline lỗi.

3) Tầng A khả thi vận hành
- per_sim_ms = 0.480275, projected_total_hours cho cấu hình chính 5000x200 chỉ 0.13341 giờ, decision_action = proceed_as_planned.

Diễn giải: target IC continuous không chỉ đúng về mặt thống kê mà còn triển khai được trong ngân sách chạy thực tế.

4) Tầng B thất bại có kiểm soát (đúng như caveat của H1)
- quality gate: cv_score = 0.210879 (<0.3), jaccard_mean = 0.306930 (<0.85), jaccard_min = 0.302083 (<0.8), pass_all = false.
- uncertainty: boundary_ratio = 0.199, ambiguous_ratio = 0.155, n_boundary_ci_crossing_threshold = 995/5000.

Diễn giải: binary top-10 không ổn định ở vùng biên, nên không được dùng làm target chính.

5) Bằng chứng nguyên nhân cấu trúc (không phải bug đơn lẻ)
- pct_communities_spanning_boundary = 0.842 (16/19 community span boundary).
- mean_gap_to_noise = 0.002392857, n_thresholds_tested = 28, interpretation = structural.
- Đổi policy A -> B chỉ giảm nhiễu nhưng không triệt tiêu (boundary và ambiguous vẫn cao).

Diễn giải: bất ổn nhị phân là hệ quả boundary mixing + threshold sensitivity của cấu trúc mạng, không phải lỗi if-else đơn giản trong gán nhãn.

### Kết luận logic
RQ1 được chấp nhận theo dạng có điều kiện, phù hợp đúng định nghĩa H1:
- Đúng ở tầng continuous: IC operationalization tạo được surrogate target cho regression (discriminative + có xu hướng ổn định + khả thi compute).
- Không đúng ở tầng binary cứng: top-10 unstable tại boundary, nên chỉ giữ vai trò phụ trợ kèm uncertainty.

Nói ngắn gọn khi bảo vệ: "RQ1 đúng cho mục tiêu continuous regression và sai có kiểm soát cho binary top-10; đây là kết luận phương pháp luận chặt chẽ, không phải né kết quả xấu."
