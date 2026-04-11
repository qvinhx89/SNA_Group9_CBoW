### Person 2 — Track B: Divergence analysis (typology IC×views + proxies + null model)

Bộ câu hỏi nghiên cứu chính của project (phiên bản v3 hiện tại) gồm 6 câu, với hypotheses đi kèm như sau.

1. RQ1: IC operationalization có tạo được ranking influence đủ phân biệt và đủ ổn định để dùng làm surrogate target không?
Hypothesis H1:
Weighted-cascade IC tạo tín hiệu continuous có ý nghĩa cho regression (discriminative, không degenerate), dù nhãn binary top-10 có thể bất ổn ở boundary.
Điểm kỳ vọng:
cv_score vượt ngưỡng readiness cho regression; nếu Jaccard nhãn thấp thì đó là đặc tính cấu trúc của graph hơn là lỗi sampling thuần túy.

2. RQ2: Views (popularity) đồng thuận tới mức nào với IC-based influence ranking?
Hypothesis H2:
Có divergence đáng kể giữa popularity và diffusion influence, tức tồn tại nhóm Hidden (IC cao, views thấp) và Overrated (views cao, IC thấp) với quy mô có ý nghĩa.

3. RQ2b: Tương quan giữa IC với views, centrality, one-hop, two-hop là gì (global và theo regime)?
Hypothesis H3:
One-hop và các structural metric sẽ tương quan khá cao với IC, nhưng không thay thế hoàn toàn IC.
Two-hop bổ sung thêm biến thiên beyond one-hop, và pattern tương quan không đồng đều giữa các vùng cấu trúc.

4. RQ3: GNN surrogate có xấp xỉ IC tốt hơn cheap proxies không, và lợi ích tính toán là bao nhiêu?
Hypothesis H4:
GNN có thể đạt chất lượng ranking cạnh tranh hoặc cao hơn proxy rẻ, đồng thời inference nhanh hơn rất nhiều so với chạy MC IC lặp lại.
Nếu không vượt proxy, kết luận vẫn publishable theo hướng “proxy địa phương đã đủ mạnh”.

5. RQ3b: Node type nào khó dự đoán nhất bằng mô hình rẻ?
Hypothesis H5:
Hidden là nhóm khó dự đoán nhất (Spearman thấp hơn, MAE cao hơn), vì chúng là outlier cấu trúc mà raw popularity khó nắm bắt.

6. RQ4: Đặc trưng cấu trúc nào phân biệt nhóm có rank views và rank IC bất đồng?
Hypothesis H6:
Hidden có xu hướng bridge hơn (betweenness và cross-community connectivity cao hơn), còn Overrated nghiêng về popularity/surface signal hơn là vị trí lan truyền hiệu quả.

---

Dưới đây là báo cáo 2 task theo cấu trúc 7 phần, bám vào plan, code và artifacts hiện có.

**Task 2: Community detection + cross-community fraction**

1. Bối cảnh task  
Task này là hạng mục MUST trong Track B tại MAPR2026_v3_team_parallel_coding_plan.md, trực tiếp phục vụ RQ2 và RQ4 trong MAPR2026_Implementation_Plan_v3.md và MAPR2026_Implementation_Plan_v3.md.

2. Mục đích khoa học + hypothesis  
Mục tiêu là đưa meso-level structure vào phân tích divergence: node nào đóng vai trò bắc cầu giữa community. Hypothesis cơ chế cấu trúc là nhóm Hidden có tín hiệu bridge mạnh hơn nhóm Overrated; nếu không có biến cross-community thì claim này không kiểm định được như plan nêu ở MAPR2026_v3_team_parallel_coding_plan.md.

3. Thiết kế và protocol  
Code chạy Louvain 10 seed, chọn partition theo modularity cao nhất, tính NMI pairwise để kiểm tra ổn định tại community.py. Sau đó tính cross-community edge fraction cho từng node ở community.py, ghi contract artifact community_features ở community.py, và xuất metrics gồm mean_nmi_louvain + stability_warning ở community.py.

4. Kết quả chính  
Run hiện tại cho n_nodes = 168114, n_communities = 21, best_modularity = 0.42268, mean_nmi = 0.70089, stability_warning = false trong metrics.json. Báo cáo chi tiết 10 run và NMI pairwise nằm ở louvain_stability_report.json. Preflight xác nhận schema đúng, không missing, phủ toàn bộ active nodes tại preflight_person2_latest.txt.

5. Phân tích diễn giải  
Task này hoàn thành tốt vai trò “structural backbone” cho RQ2/RQ4: bạn đã có biến community_id và cross_community_edge_fraction để chuyển từ mô tả divergence sang giải thích cơ chế. Tuy nhiên mức ổn định community chỉ vừa chạm ngưỡng (mean_nmi khoảng 0.701), nên bằng chứng cơ chế hiện là đủ dùng nhưng chưa thật mạnh theo tiêu chuẩn reviewer khắt khe.

6. Rủi ro và giới hạn  
Rủi ro chính là sensitivity theo resolution/seed có thể ảnh hưởng kết luận bridge nếu chỉ bám một cấu hình. Ngoài ra, community structure trên đồ thị dense có thể bị over-merge, nên nếu không có sensitivity bổ sung thì lập luận RQ4 dễ bị hỏi thêm.

7. Kết luận hành động  
Task 2 hiện đạt trạng thái completion-ready cho pipeline, và đủ điều kiện làm input cho structural profiling. Về học thuật, nên xem đây là bằng chứng “đạt ngưỡng” và cần kèm sensitivity note khi viết kết luận RQ4.

---

**Task 3: Proxies thật full graph + runtime**

1. Bối cảnh task  
Task này là Group 3 baseline trong Track B tại MAPR2026_v3_team_parallel_coding_plan.md, kết nối trực tiếp RQ2b (metric correlation) và RQ3 (proxy vs surrogate) tại MAPR2026_Implementation_Plan_v3.md và MAPR2026_Implementation_Plan_v3.md.

2. Mục đích khoa học + hypothesis  
Mục tiêu là tạo baseline diffusion rẻ để kiểm tra proxy utility: one-hop/two-hop có thể xấp xỉ IC đến đâu trước khi cần GNN. Hypothesis là proxy tương quan đáng kể với IC, và giúp định lượng trade-off chất lượng so với tốc độ.

3. Thiết kế và protocol  
Script đọc CSR, kiểm tra contract bidirectional ở diffusion_proxies.py, tính one-hop và two-hop ở diffusion_proxies.py, ép full coverage + no NaN, rồi upsert runtime vào runtime_breakdown tại diffusion_proxies.py và diffusion_proxies.py.

4. Kết quả chính  
Artifact proxies đang ở mode real_full_graph, rows = 168114, inference_sec_full_graph = 0.82755 trong diffusion_proxies_status.json. Runtime row đã ghi ở runtime_breakdown.csv. Về hiệu năng ranking: one_hop Spearman = 0.6877, two_hop Spearman = 0.5239 trong baseline_ranking_metrics.csv và baseline_ranking_metrics.csv. Correlation matrix cũng cho IC-one_hop khoảng 0.717 và IC-two_hop khoảng 0.523 ở metric_correlation_matrix.json.

5. Phân tích diễn giải  
Task 3 hoàn thành mạnh về operational objective: full-graph nhanh, reproducible, usable cho RQ2b/RQ3. Về khoa học, one-hop có utility rõ ràng; two-hop hiện thấp hơn kỳ vọng nên chưa chứng minh được “higher-order proxy tốt hơn” như narrative kỳ vọng. Điều này vẫn hợp lệ cho RQ3 vì nó giúp phân định khi nào proxy đủ, khi nào surrogate cần thiết.

6. Rủi ro và giới hạn  
Rủi ro lớn nhất là khả năng lệch công thức two-hop giữa plan và implementation. Code hiện tính dạng tổng láng giềng bậc hai không nhân hệ số p(u,v) như mô tả expected-spread trong plan; điểm này nằm ở diffusion_proxies.py. Nếu không làm rõ, reviewer có thể coi đây là mismatch phương pháp.

7. Kết luận hành động  
Task 3 đã đạt chuẩn pipeline và cung cấp bằng chứng định lượng hữu ích cho RQ2b/RQ3. Tuy nhiên, để defensible hơn về methodology, cần khóa rõ định nghĩa two-hop đang dùng (hoặc chỉnh công thức để khớp plan) trước khi chốt claim cuối.

---

**Task 4: Tạo typology IC×views (True/Hidden/Overrated/Non)**

1. Bối cảnh task  
Task này là lõi của divergence analysis trong Track B, được mô tả ở MAPR2026_v3_team_parallel_coding_plan.md, và gắn trực tiếp với RQ2, đồng thời là đầu vào cho RQ4 và RQ3b trong MAPR2026_Implementation_Plan_v3.md.

2. Mục đích khoa học + hypothesis  
Mục tiêu là kiểm định divergence hypothesis: popularity (views) không đồng nhất với diffusion influence (IC). Nếu không có typology 2x2, bạn không thể tách nhóm Hidden/Overrated để kiểm tra cơ chế cấu trúc hay độ khó dự đoán theo nhóm.

3. Thiết kế và protocol  
Code merge IC score với views theo node_id, đặt ngưỡng top 10% cho cả IC và views tại typology_ic_views.py và typology_ic_views.py, rồi gán nhãn typology tại typology_ic_views.py. Báo cáo quadrant được build với điều kiện min_quadrant_ok tại typology_ic_views.py và typology_ic_views.py. Gate ép lỗi nếu yêu cầu min quadrant mà fail ở typology_ic_views.py.

4. Kết quả chính  
Artifact hiện có cho thấy n_total = 5000, Hidden = 285 (5.7%), Overrated = 285 (5.7%), min_quadrant_ok = true, two_sample_applied = false trong typology_quadrant_report.json, typology_quadrant_report.json, typology_quadrant_report.json, typology_quadrant_report.json, typology_quadrant_report.json.

5. Phân tích diễn giải  
Task 4 đã hoàn thành đúng vai trò trung tâm cho RQ2: có divergence rõ (không collapse về một trục popularity), đồng thời đủ cỡ mẫu cho Hidden/Overrated để chạy Task 5 và downstream RQ3b. Việc không cần two-sample cho thấy setup hiện tại có power cơ bản ổn cho so sánh nhóm.

6. Rủi ro và giới hạn  
Typology phụ thuộc ngưỡng top 10%, nên kết luận có sensitivity theo threshold. Ngoài ra typology hiện là phân lớp cắt ngưỡng, chưa phản ánh uncertainty quanh boundary nếu chỉ nhìn nhãn cứng.

7. Kết luận hành động  
Task 4 đạt chuẩn completion-ready cho mục tiêu divergence. Có thể dùng trực tiếp làm nền cho structural profiling (Task 5) và per-group prediction difficulty (RQ3b), với lưu ý phải báo cáo rõ tính phụ thuộc ngưỡng.

---

**Task 5: So sánh Hidden vs Overrated bằng MWU + Cliff’s Delta + BH-FDR**

1. Bối cảnh task  
Task 5 là bước biến divergence từ mô tả thành bằng chứng thống kê cho RQ4 và tăng lực diễn giải cho RQ2, theo thiết kế ở MAPR2026_v3_team_parallel_coding_plan.md.

2. Mục đích khoa học + hypothesis  
Hypothesis cơ chế cấu trúc là Hidden khác Overrated theo các đặc trưng bridge/structural, không chỉ khác views bề mặt. Task này kiểm định giả thuyết bằng effect size và multiple testing control thay vì chỉ nhìn mean.

3. Thiết kế và protocol  
Pipeline join typology với node attributes, centrality, community features; fail-fast nếu thiếu giá trị ở typology_ic_views.py. Thống kê dùng Mann-Whitney hai phía trên 6 đặc trưng tại typology_ic_views.py, Cliff’s delta tính từ U-stat tại typology_ic_views.py, BH-FDR áp trên toàn bộ p-values tại typology_ic_views.py, tiêu chí significant là p_corrected < 0.05 và |delta| >= ngưỡng.

4. Kết quả chính  
Kết quả trong structural_profiling.csv cho thấy 5/6 biến đạt ý nghĩa và effect size thực dụng: degree, pagerank, kshell, betweenness, cross_community_edge_fraction đều significant ở structural_profiling.csv, structural_profiling.csv, structural_profiling.csv, structural_profiling.csv, structural_profiling.csv. life_time không đạt tiêu chí (delta nhỏ, không significant theo ngưỡng effect) ở structural_profiling.csv.

5. Phân tích diễn giải  
Bằng chứng hiện tại ủng hộ structural mechanism hypothesis: Hidden nổi bật ở các chỉ số cấu trúc và bridge, phù hợp claim cơ chế hơn là chỉ popularity mismatch. Việc cross_community_edge_fraction có delta vượt ngưỡng là điểm quan trọng vì nó kết nối trực tiếp với luận điểm bridge trong RQ4.

6. Rủi ro và giới hạn  
So sánh là univariate theo từng feature; chưa phải mô hình nhân quả đa biến. Một số chỉ số cấu trúc có tương quan cao, nên cần cẩn trọng khi diễn giải “vai trò riêng” của từng biến. life_time yếu cũng nhắc rằng không nên overclaim external corroboration từ biến này.

7. Kết luận hành động  
Task 5 hiện đạt mức bằng chứng thống kê tốt cho RQ4 và củng cố RQ2. Bạn có thể dùng bảng này như evidence chính, đồng thời ghi rõ giới hạn đồng biến và tính quan sát khi viết kết luận học thuật.

---

Dưới đây là báo cáo chi tiết cho Task 6 và Task 7 theo đúng cấu trúc 7 phần, bám vào code và artifact hiện có.

**Task 6: External corroboration bằng life_time**

1. Bối cảnh task  
Task này nằm trong Track B để bảo vệ construct validity của typology (IC x views), được mô tả trong MAPR2026_v3_team_parallel_coding_plan.md. Nó phục vụ trực tiếp RQ2 về độ tin cậy ngoài các chỉ số cấu trúc thuần trong MAPR2026_Implementation_Plan_v3.md.

2. Mục đích khoa học + hypothesis liên quan  
Mục tiêu là kiểm định robustness/validity hypothesis: nếu typology thật sự mang ý nghĩa influence potential, thì phải có dấu hiệu corroboration từ biến exogenous như life_time sau khi kiểm soát degree. Đây là kiểm định “độc lập tương đối” vì IC labels không dùng life_time trong quá trình tạo nhãn.

3. Thiết kế và protocol  
Code dùng 2 lớp kiểm định ở typology_ic_views.py.  
- Partial Spearman: residualized rank để tính Spearman(IC, life_time | degree) tại typology_ic_views.py.  
- Stratified MWU theo degree quintile + BH-FDR tại typology_ic_views.py, typology_ic_views.py.  
Tiêu chí thành công là n_quintiles_significant >= 3 tại typology_ic_views.py. Nếu fail, kích hoạt fallback language corroboration ở typology_ic_views.py.

4. Kết quả chính  
Kết quả thực tế trong lifetime_validation.json: partial_spearman_rho = -0.020, p = 0.1566, n_quintiles_significant = 0, success = false tại lifetime_validation.json, lifetime_validation.json.  
Hai quintile đầu quá ít Hidden (2 và 7) nên gần như không có power, thấy ở lifetime_validation.json, lifetime_validation.json.  
Fallback language đã được trigger và có tín hiệu mạnh trong language_validation.json, language_validation.json, language_validation.json, nhưng file cũng ghi rõ chỉ là bổ sung tại language_validation.json.

5. Phân tích diễn giải  
Task 6 hiện không xác nhận được validity theo life_time (theo định nghĩa gate của plan), nên robustness claim cho RQ2 phải giữ mức thận trọng. Điểm tốt là pipeline xử lý đúng IF PROBLEM logic và không “che” fail bằng fallback. Preflight xác nhận cơ chế này đã chạy đúng tại preflight_person2_latest.txt.

6. Rủi ro và giới hạn  
Rủi ro chính là power bất cân bằng theo quintile (Hidden quá ít ở low-degree bins), khiến kiểm định dễ fail dù có tín hiệu thật yếu-vừa. Ngoài ra life_time có thể phản ánh tenure hơn là influence, nên thất bại ở Task 6 không bác bỏ hoàn toàn typology, nhưng làm suy yếu external corroboration. Hạn chế này đã được ghi nhận trong assumptions_limitations.md.

7. Kết luận hành động  
Task 6 hoàn thành đúng protocol nhưng outcome là inconclusive theo gate đã lock. Khi báo cáo RQ2 nên ghi rõ: lifetime validation fail, language validation chỉ supplementary, không thay thế evidence chính từ typology/structural/null package.

---

**Task 7: Configuration null để kiểm tra degree artifact**

1. Bối cảnh task  
Task này là kiểm định null-model cốt lõi cho độ vững diễn giải của RQ2, nhằm tách signal divergence thật khỏi artifact do degree sequence, theo mô tả trong MAPR2026_v3_team_parallel_coding_plan.md.

2. Mục đích khoa học + hypothesis liên quan  
Hypothesis cần kiểm định: nếu typology Hidden chỉ là hệ quả cơ học của degree distribution, thì trên configuration null sẽ tái tạo pattern tương tự. Ngược lại, nếu real graph khác null rõ rệt thì có cơ sở nói divergence mang nội dung cấu trúc thật.

3. Thiết kế và protocol  
Script thực thi contract 500 nodes x 3 realizations x 100 runs/node ở null_model_typology.py, null_model_typology.py, null_model_typology.py.  
Nó tạo null bằng nx.configuration_model tại null_model_typology.py, chạy IC weighted-cascade trên null, rồi so rho(real IC, null IC) và hidden betweenness real vs null. Rule diễn giải tự động nằm ở null_model_typology.py.

4. Kết quả chính  
Artifact cuối cùng trong null_model_typology_summary.json cho thấy n_nodes = 500, n_realizations = 3, n_runs_per_node = 100 tại null_model_typology_summary.json, null_model_typology_summary.json, null_model_typology_summary.json.  
rho_mean = 0.441 và hidden_bet_real thấp hơn null mean, dẫn đến diễn giải “comparable to configuration null; potential degree artifact” tại null_model_typology_summary.json, null_model_typology_summary.json, null_model_typology_summary.json. Preflight xác nhận null package đầy đủ Task 7/8/9 tại preflight_person2_latest.txt.

5. Phân tích diễn giải  
Với kết quả hiện tại, Task 7 không ủng hộ mạnh “structural uniqueness” của Hidden theo tiêu chí configuration null; ngược lại nó buộc narrative RQ2 phải thận trọng hơn và nghiêng về khả năng degree-distribution artifact một phần. Điểm tích cực là đây là bằng chứng robustness trung thực, giúp tránh overclaim divergence.

6. Rủi ro và giới hạn  
Rủi ro phương pháp đáng chú ý là ngưỡng diễn giải dùng max(0.05, null_std) ở null_model_typology.py, trong khi thang betweenness đang rất nhỏ (cỡ e-05) ở null_model_typology_summary.json. Ngưỡng tuyệt đối 0.05 có thể quá thô theo scale dữ liệu, làm rule phân loại thiên về “comparable”. Đây là điểm nên ghi rõ khi thảo luận limitation.

7. Kết luận hành động  
Task 7 đã hoàn thành contract kỹ thuật và cung cấp bằng chứng robustness quan trọng cho RQ2, nhưng kết quả hiện chưa cho phép claim mạnh rằng Hidden pattern vượt xa null theo tiêu chí đang dùng. Khuyến nghị báo cáo theo hướng: divergence tồn tại ở dữ liệu quan sát, nhưng một phần cấu trúc có thể giải thích bởi degree sequence; do đó cần đặt claim ở mức vừa phải và dựa đồng thời vào package Task 5 + Task 8/9.

---

Dưới đây là báo cáo Task 8 và Task 9 theo cấu trúc 7 phần, dựa trên code và artifact hiện có.

**Task 8: Views-permutation null (phá views, giữ IC/graph)**

1. Bối cảnh task  
Task 8 là nhánh MUST của B5 core theo MAPR2026_v3_team_parallel_coding_plan.md, phục vụ RQ2 và RQ2b như một kiểm định mechanism-specific cho robustness.

2. Mục đích khoa học + hypothesis  
Mục tiêu là tách đóng góp riêng của views vào divergence: nếu chỉ xáo views mà cấu trúc divergence thay đổi theo hướng random, thì phần divergence quan sát được có thể là non-random mechanism thay vì nhiễu.

3. Thiết kế và protocol  
Hàm `_compute_views_permutation_null` giữ IC scores cố định, permute views qua labeled nodes, rebuild typology nhiều lần ở typology_ic_views.py.  
Các thống kê chính: hidden_count, overrated_count, agreement/divergence rate, empirical p-values.  
Rule diễn giải được mã hóa trong hàm: “significantly higher than permutation null” ở typology_ic_views.py, hoặc “within permutation-null range” ở typology_ic_views.py.  
Task được gọi tự động từ pipeline main tại typology_ic_views.py, ghi artifact tại typology_ic_views.py.

4. Kết quả chính  
Artifact ở views_permutation_null_summary.json:  
- n_nodes_labeled = 5000, n_permutations = 200.  
- Real: agreement_rate = 0.886, divergence_rate = 0.114.  
- Null mean: agreement_rate_mean = 0.8200, divergence_rate_mean = 0.1800.  
- Empirical p cho agreement_rate_ge_real = 0.004975 (ý nghĩa).  
Diễn giải tự động: observed agreement cao hơn null, divergence pattern non-random.

5. Phân tích diễn giải  
Kết quả ủng hộ robustness theo hướng mechanism-specific: khi phá views, mức alignment thực nghiệm giữa views-IC bị giảm đáng kể về null baseline. Điều này cho thấy pattern quan sát không phải do ngẫu nhiên thuần từ thresholding.

6. Rủi ro và giới hạn  
Test này kiểm định ngẫu nhiên hóa theo hoán vị, nhưng chưa mô hình hóa phụ thuộc theo cấu trúc cộng đồng/degree strata trong phép hoán vị. Ngoài ra, nó chứng minh “non-random” chứ không tự nó chứng minh cơ chế nhân quả đầy đủ.

7. Kết luận hành động  
Task 8 hoàn thành tốt và là bằng chứng mạnh bổ sung cho RQ2/RQ2b. Nên dùng như “robustness-evidence layer” kèm Task 4/5, không dùng độc lập để kết luận cơ chế.

---

**Task 9: IC-permutation null (phá IC, giữ views/graph)**

1. Bối cảnh task  
Task 9 là nhánh MUST còn lại của B5 core theo MAPR2026_v3_team_parallel_coding_plan.md, có execution lock phải chạy đủ cả 2 permutation trước sign-off tại MAPR2026_v3_team_parallel_coding_plan.md.

2. Mục đích khoa học + hypothesis  
Mục tiêu là tách đóng góp riêng của IC ranking vào divergence: nếu phá IC mà pattern alignment giảm về null, thì divergence thực nghiệm có thành phần thông tin thực từ IC chứ không chỉ từ views.

3. Thiết kế và protocol  
Hàm `_compute_ic_permutation_null` giữ views cố định, permute ic_score_mean qua labeled nodes ở typology_ic_views.py.  
Thống kê và rule diễn giải tương tự Task 8; nhánh “significantly higher than IC-permutation null” nằm ở typology_ic_views.py, còn nhánh inconclusive ở typology_ic_views.py.  
Pipeline gọi và ghi artifact tại typology_ic_views.py, typology_ic_views.py.

4. Kết quả chính  
Artifact ở ic_permutation_null_summary.json:  
- n_nodes_labeled = 5000, n_permutations = 200.  
- Real agreement_rate = 0.886, null agreement_rate_mean = 0.8199.  
- Empirical p agreement_rate_ge_real = 0.004975 (ý nghĩa).  
- Hidden real = 285 thấp hơn null mean ~450.255.  
Diễn giải tự động: observed agreement cao hơn IC-permutation null, divergence pattern non-random.

5. Phân tích diễn giải  
Task 9 cho kết luận cùng chiều với Task 8, tạo đối xứng kiểm định: phá IC hay phá views đều làm pattern tiến về null distribution. Điều này tăng độ tin cậy rằng divergence quan sát là cấu trúc tín hiệu thật của cặp (IC, views), không phải “ảnh ảo” do một phía đơn lẻ.

6. Rủi ro và giới hạn  
Giống Task 8, đây là bằng chứng ngẫu nhiên hóa mạnh nhưng vẫn là kiểm định thống kê quan sát. Ngoài ra, vì top_pct cố định và thresholding cứng, kết quả có thể nhạy với percentile lock trong trường hợp phân phối sát ngưỡng.

7. Kết luận hành động  
Task 9 đạt yêu cầu contract và củng cố robustness claim cho RQ2/RQ2b theo hướng mechanism-specific. Cùng với Task 8, đây là cặp evidence rất giá trị để bảo vệ trước phản biện “divergence chỉ là ngẫu nhiên”.

---

**Task 10: Backup residual divergence khi min_quadrant_ok vẫn fail sau two-sample**

1. Bối cảnh task  
Task 10 là nhánh contingency trong plan, chỉ kích hoạt khi min_quadrant_ok vẫn false sau two-sample, theo MAPR2026_v3_team_parallel_coding_plan.md và bảng task MAPR2026_v3_team_parallel_coding_plan.md. Nó phục vụ RQ2 theo đường fallback, không phải main path.

2. Mục đích khoa học + hypothesis liên quan  
Mục tiêu không phải tạo hypothesis mới, mà bảo toàn khả năng kiểm định divergence khi top-10 typology thiếu power. Cụ thể, plan yêu cầu residual score z(rank(IC)) - z(rank(views)) để vẫn định lượng Hidden-like và Overrated-like trong tình huống khó.

3. Thiết kế và protocol  
Theo plan, phải chạy sau hai điều kiện nối tiếp:  
- Bước 1: min_quadrant_ok false sau typology ban đầu.  
- Bước 2: đã áp dụng two-sample strategy mà vẫn fail.  
Khi đó mới sinh artifact fallback ở MAPR2026_v3_team_parallel_coding_plan.md.

4. Kết quả chính (trạng thái thực thi hiện tại)  
Hiện tại trigger chưa xảy ra vì min_quadrant_ok true ở typology_quadrant_report.json, và two_sample_applied false ở typology_quadrant_report.json.  
Ngoài ra, không có artifact residual file trong thư mục kết quả mapr2026_v3_results và truy vấn theo tên file không thấy kết quả.  
Quan trọng hơn: trong implementation hiện tại, cờ two_sample_applied đang hard-code false ở typology_ic_views.py, nghĩa là logic two-sample/residual chưa được hiện thực đầy đủ trong cùng script.

5. Phân tích diễn giải  
Về mặt governance: Task 10 chưa cần chạy là đúng vì contingency chưa trigger.  
Về mặt readiness: fallback path trên giấy có, nhưng execution path trong code còn thiếu mảnh chính (two-sample + residual report generation), nên nếu sau này gặp min_quadrant_ok false thật thì pipeline có nguy cơ không có “đường thoát” đúng contract.

6. Rủi ro và giới hạn  
Rủi ro là latent implementation gap: plan yêu cầu fallback cụ thể nhưng code hiện chỉ cảnh báo “hãy chạy two-sample” ở typology_ic_views.py và typology_ic_views.py, chưa tự sinh residual artifact. Nếu data shift làm quadrant nhỏ trong future run, đây sẽ là blocker phương pháp.

7. Kết luận hành động  
Task 10 hiện ở trạng thái not-triggered và not-fully-implemented. Trước sign-off nên quyết định một trong hai:  
1. Implement đầy đủ two-sample + residual artifact path trong script.  
2. Hoặc lock rõ trong tài liệu rằng contingency này chỉ là manual protocol, không auto trong code.

---

**Task 11: Ma trận tương quan global 8x8 + p corrected**

1. Bối cảnh task  
Task 11 là MUST theo plan tại MAPR2026_v3_team_parallel_coding_plan.md, trả lời trực tiếp RQ2b và cung cấp context cho RQ3.

2. Mục đích khoa học + hypothesis liên quan  
Mục tiêu là lượng hóa IC giống/khác các metric rẻ (views, degree, pagerank, kshell, betweenness, one-hop, two-hop) ở mức hệ thống. Đây là kiểm định trực tiếp proxy utility hypothesis và một phần divergence hypothesis.

3. Thiết kế và protocol  
Code dựng frame 8 metric, kiểm tra coverage/NA/duplicate chặt ở typology_ic_views.py.  
Sau đó tính Spearman full pairwise, BH-FDR trên upper triangle, xuất p_matrix_corrected đối xứng ở typology_ic_views.py.  
Nhánh rho_by_degree_quintile là tùy chọn IF TIME tại typology_ic_views.py.  
Artifact được gọi và ghi trong main flow ở typology_ic_views.py đến typology_ic_views.py.

4. Kết quả chính  
Artifact đầy đủ ở metric_correlation_matrix.json với n_rows_expected 5000 và coverage_ok true tại metric_correlation_matrix.json và metric_correlation_matrix.json.  
Một số điểm nổi bật:  
- IC-views rho khoảng 0.4689 ở metric_correlation_matrix.json.  
- IC-one-hop rho khoảng 0.7171 ở metric_correlation_matrix.json.  
- IC-two-hop rho khoảng 0.5234 ở metric_correlation_matrix.json.  
- Có column_mapping chuẩn hóa betweenness ở metric_correlation_matrix.json và metric_correlation_matrix.json.  
- Nhánh rho_by_degree_quintile đã có dữ liệu ở metric_correlation_matrix.json, ví dụ Q3 views rất thấp metric_correlation_matrix.json, Q4 degree cao hơn rõ metric_correlation_matrix.json.

5. Phân tích diễn giải  
Task 11 đang làm rất tốt vai trò RQ2b: nó cho thấy views không đại diện tốt cho IC toàn cục, trong khi one-hop gần IC hơn two-hop trong run hiện tại. Đồng thời, breakdown theo quintile cho thấy quan hệ IC-views/IC-degree không đồng nhất theo regime, rất hữu ích để giải thích vì sao baseline khó ở vài vùng cấu trúc.

6. Rủi ro và giới hạn  
Ma trận tương quan là bằng chứng liên hệ, không phải nhân quả. Ngoài ra, kết quả phụ thuộc chất lượng upstream features; nếu centrality/proxies drift theo version thì matrix cũng đổi. Dù vậy pipeline đã có guard coverage và schema khá chặt.

7. Kết luận hành động  
Task 11 đã completion-ready cả về kỹ thuật và giá trị học thuật. Có thể dùng làm bảng trụ cột cho RQ2b và làm ngữ cảnh bắt buộc cho phần thảo luận RQ3 baseline difficulty. Preflight cũng xác nhận Task 11 đạt chuẩn blocking tại preflight_person2_latest.txt.



