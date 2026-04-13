### Person 2 — Track B: Divergence analysis (typology IC×views + proxies + null model)

Bộ câu hỏi nghiên cứu chính của project (phiên bản v3 hiện tại) gồm 6 câu, với hypotheses đi kèm như sau.

1. RQ1: IC operationalization có tạo được ranking influence đủ phân biệt và đủ ổn định để dùng làm surrogate target không?
   Hypothesis H1:
   Weighted-cascade IC tạo tín hiệu continuous có ý nghĩa cho regression (discriminative, không degenerate), dù nhãn binary top-10 có thể bất ổn ở boundary.
   Điểm kỳ vọng:
   cv_score vượt ngưỡng readiness cho regression; nếu Jaccard nhãn thấp thì đó là đặc tính cấu trúc của graph hơn là lỗi sampling thuần túy.

2. RQ2: Views (popularity) đồng thuận tới mức nào với IC-based influence ranking?
   Hypothesis H2:
   Mức divergence được khóa theo narrative-tier của plan dựa trên rho(views, IC): strong divergence / moderate divergence / high agreement.
   Dù thuộc tier nào, typology Hidden (IC cao, views thấp) và Overrated (views cao, IC thấp) vẫn là khung phân tích chính để kiểm định bất đồng rank.

3. RQ2b: Tương quan giữa IC với views, centrality, one-hop, two-hop là gì (global và theo regime)?
   Hypothesis H3:
   One-hop và các structural metric sẽ tương quan khá cao với IC, nhưng không thay thế hoàn toàn IC.
   Two-hop bổ sung thêm biến thiên beyond one-hop, và pattern tương quan không đồng đều giữa các vùng cấu trúc.

4. RQ3: GNN surrogate có xấp xỉ IC tốt hơn cheap proxies không, và lợi ích tính toán là bao nhiêu?
   Hypothesis H4:
   GNN có thể đạt chất lượng ranking cạnh tranh hoặc cao hơn một phần proxy rẻ; lợi thế tốc độ chính là so với MC IC labeling, không mặc định nhanh hơn mọi analytical proxy.
   Nếu không vượt proxy, kết luận vẫn publishable theo hướng “proxy địa phương đã đủ mạnh”.

5. RQ3b: Node type nào khó dự đoán nhất bằng mô hình rẻ?
   Hypothesis H5:
   Hidden được kỳ vọng là nhóm khó dự đoán nhất (Spearman thấp hơn, MAE cao hơn), vì đây là outlier cấu trúc mà raw popularity khó nắm bắt.
   Claim này chỉ được khóa ở mức kết luận cuối khi đã có per-group metrics đầy đủ cho các model chính theo plan.

6. RQ4: Đặc trưng cấu trúc nào phân biệt nhóm có rank views và rank IC bất đồng?
   Hypothesis H6:
   Hidden được kỳ vọng có tín hiệu cấu trúc nổi bật hơn Overrated (đặc biệt ở cross-community connectivity và các chỉ số lõi), còn Overrated nghiêng về popularity/surface signal hơn là vị trí lan truyền hiệu quả.
   Diễn giải ở mức tổng quát phải đi kèm caveat null-model (configuration null) để tránh overclaim structural uniqueness.

---

Dưới đây là báo cáo các task theo cấu trúc 7 phần, bám vào plan, code và artifacts hiện có.

**Task 2: Community detection + cross-community fraction**

File liên quan trực tiếp (để kiểm tra):

- Code chính: src/graph/community.py
- Input chính: data/processed/graph_active.edgelist
- Output/check chính: data/processed/community_features.parquet, outputs/stage2/metrics.json, outputs/stage2/louvain_stability_report.json, outputs/mapr2026_v3_results/louvain_resolution_sensitivity.json, outputs/mapr2026_v3_results/preflight_person2_latest.txt

1. Bối cảnh task  
   Task này là hạng mục MUST trong Track B tại docs/MAPR2026_v3_team_parallel_coding_plan.md, trực tiếp phục vụ RQ2 và RQ4 trong MAPR2026_Implementation_Plan_v3.md.

2. Mục đích khoa học + hypothesis  
   Mục tiêu là đưa meso-level structure vào phân tích divergence: node nào có tín hiệu kết nối liên cộng đồng và vị trí cấu trúc nổi bật. Hypothesis cơ chế cấu trúc là nhóm Hidden có connectivity signal mạnh hơn nhóm Overrated; nếu không có biến cross-community thì claim này không kiểm định được như plan nêu ở docs/MAPR2026_v3_team_parallel_coding_plan.md.

3. Thiết kế và protocol  
   Diễn giải phương pháp (chi tiết): Task 2 được thiết kế theo hướng tách rõ "chất lượng partition" và "giá trị đặc trưng downstream". Trước hết pipeline chạy Louvain nhiều seed độc lập để giảm rủi ro lệ thuộc khởi tạo ngẫu nhiên, sau đó chọn partition có modularity cao nhất làm kết quả chính. Tiếp theo, độ ổn định cấu trúc được lượng hóa bằng NMI pairwise giữa các lần chạy để kiểm tra xem kết quả có nhất quán hay không (thay vì chỉ nhìn modularity một lần). Chỉ sau khi qua bước kiểm tra ổn định, hệ thống mới tính cross-community edge fraction cho từng node và xuất đồng thời cả bảng đặc trưng lẫn báo cáo ổn định để đảm bảo các phân tích RQ2/RQ4 có nền tảng phương pháp rõ ràng.
   Trong detect_communities(), pipeline chạy run_louvain_single() theo 10 seed, chọn partition có modularity cao nhất, rồi tính NMI pairwise bằng compute_nmi_between_partitions() để kiểm tra ổn định. Tín hiệu bridge được tính bằng compute_cross_community_edge_fraction(), sau đó ghi community_features.parquet và xuất metrics.json (mean_nmi_louvain, stability_warning) cùng louvain_stability_report.json.
   Ghi chú reproducibility cho NMI: compute_nmi_between_partitions() gọi sklearn.metrics.normalized_mutual_info_score(labels1, labels2) mà không truyền tham số average_method — tức là dùng giá trị mặc định của sklearn là 'arithmetic'. Khi viết phần Methods trong paper cần ghi rõ: "NMI computed using sklearn's normalized_mutual_info_score with average_method='arithmetic' (sklearn default)". Nếu so sánh với bài khác dùng average_method='geometric' hoặc 'min', giá trị NMI sẽ khác — cần lưu ý khi đọc literature.
   Công thức tường minh: `cross_community_edge_fraction(u) = |{v ∈ N(u) : community(v) ≠ community(u)}| / deg(u)`. Isolated nodes (deg=0) nhận giá trị 0.0. Best partition được chọn theo modularity cao nhất trong 10 seed — đây là thực hành chuẩn cho Louvain vì modularity là hàm mục tiêu tối ưu hóa trực tiếp.

4. Kết quả chính  
   Run hiện tại (resolution = 1.1, nmi_threshold = 0.67 được khóa trong base.yaml) cho n_nodes = 168,114, n_communities = 24 (best_seed = 6), best_modularity = 0.42313, mean_nmi = 0.69968, std_nmi = 0.02463, min_nmi = 0.65146, stability_warning = false trong metrics.json. pct_top3 = 49.23% (3 community lớn nhất gộp 82,765/168,114 nodes). cross_community_edge_fraction: mean = 0.3918, std = 0.2160, min = 0.0, max = 0.9643. over_merge_warning = false. Báo cáo chi tiết 10 run và NMI pairwise nằm ở louvain_stability_report.json; n_communities biến thiên 19–26 giữa các seed (run4=19, run5/run8=26), best partition được chọn theo modularity cao nhất. Preflight xác nhận schema đúng, không missing, phủ toàn bộ active nodes tại outputs/mapr2026_v3_results/preflight_person2_latest.txt.
   Full sweep 9 mức resolution (0.5, 1.0, 1.1, 1.3, 1.4, 1.5, 1.55, 1.6, 2.0; mỗi mức 10 seed) nằm ở louvain_resolution_sensitivity.json. Evaluation framework dùng hard gate: mean_nmi_min = 0.67, min_nmi_min = 0.60, modularity_retain_ratio = 0.95. Kết quả hard_gate_pass: resolution 0.5 và 2.0 fail; tất cả từ 1.0 đến 1.6 pass. Composite scoring (weights: stability=0.35, modularity=0.25, structure=0.20, downstream=0.0) xếp hạng: 1.1 đạt 0.9588 (stability=0.957, modularity=1.000, structure=0.910); 1.0 đạt 0.9005 (stability=0.882, modularity=0.993, structure=0.818); 1.3 đạt 0.697; 1.6 đạt 0.627. Recommended_resolution từ framework = 1.1.
   Bảng tóm tắt sweep: 1.0 — 21 comm, pct_top3=55.93%, NMI=0.70089, over_merge=true; 1.1 — 24 comm, pct_top3=49.23%, NMI=0.69968, over_merge=false; 1.3 — 26 comm, pct_top3=40.08%, NMI=0.68761; 1.4 — 29 comm, pct_top3=37.20%, NMI=0.67998; 1.5 — 26 comm, pct_top3=33.03%, NMI=0.68539; 1.55 — 32 comm, pct_top3=32.61%, NMI=0.67788; 1.6 — 34 comm, pct_top3=31.99%, NMI=0.69054.
   Downstream validation so sánh trực tiếp resolution 1.1 và 1.6 cho Task 5: kết luận Hidden-vs-Overrated nhất quán giữa hai resolution — cross_community_edge_fraction của Hidden luôn cao hơn Overrated, Cliff's delta đều meaningful (1.1: δ=0.2062, p=2.45×10⁻⁵; 1.6: δ=0.2082, p=2.05×10⁻⁵), significance không bị đảo chiều.

5. Phân tích diễn giải  
   Task này hoàn thành tốt vai trò “structural backbone” cho RQ2/RQ4: biến community_id và cross_community_edge_fraction đã sẵn sàng để chuyển từ mô tả divergence sang giải thích cơ chế. Resolution 1.1 (hiện là primary trong base.yaml) cải thiện rõ nét so với resolution 1.0 trên hai tiêu chí quan trọng cho downstream analysis: (1) pct_top3 giảm từ 55.93% xuống 49.23%, qua ngưỡng 50% của evaluation framework — over_merge_warning từ true xuống false; (2) n_communities tăng từ 21 lên 24, cho partition đều hơn và cross_community_edge_fraction có variance tốt hơn để phân biệt Hidden vs Overrated. Modularity vẫn cao (mean=0.42113 vs 0.42098 ở res=1.0). Composite score xác nhận res=1.1 vượt res=1.0 (0.9588 vs 0.9005), lợi thế chủ yếu đến từ structure component (0.910 vs 0.818) và stability component (0.957 vs 0.882). NMI mean 0.69968 vượt hard gate (0.67) với đệm 0.030; min_nmi = 0.65146 > min_nmi_min = 0.60. Bằng chứng cơ chế hiện mạnh hơn trước: community structure không bị cảnh báo over-merge, phân phối community cân bằng hơn, và downstream effect size nhất quán.

6. Rủi ro và giới hạn  
   Rủi ro sensitivity theo resolution/seed đã được kiểm tra toàn diện qua sweep 9 mức và validation downstream cho Task 5; quyết định cấu hình chính đã được khóa ở resolution = 1.1 (nmi_threshold = 0.67) trong base.yaml. Giới hạn kỹ thuật còn lại: (1) n_communities biến thiên 19–26 giữa các seed tại res=1.1 — phản ánh inherent stochasticity của Louvain trên dense graph; artifact được tạo từ best partition theo modularity nên variance này không ảnh hưởng kết quả, nhưng reviewer khắt khe có thể hỏi; (2) min_nmi = 0.65146 (cặp run1 vs run4) là điểm thấp nhất trong 45 cặp — vẫn trên min_nmi_min = 0.60 của hard gate, nhưng chênh lệch đệm chỉ 0.051; (3) sensitivity artifact dùng nmi_threshold = 0.70 để tính stability_warning flag (kế thừa từ version cũ), trong khi base.yaml hiện dùng 0.67 — cần lưu ý khi đọc stability_warning trong louvain_resolution_sensitivity.json (flag ở file đó phản ánh ngưỡng cũ 0.70, không phải ngưỡng production 0.67).

7. Kết luận hành động  
   Task 2 đã hoàn thành — artifacts production hiện tại (community_features.parquet, louvain_stability_report.json, metrics.json) phản ánh primary resolution = 1.1 được khóa trong base.yaml. Pipeline sẵn sàng làm input cho Task 5 (structural profiling) với 24 communities, stability_warning=false, over_merge_warning=false, cross_community_edge_fraction đủ variance (std=0.2160). k-shell cũng hoàn thành trong cùng Stage 2: kshell_max=149, kshell_mean=40.84, phủ toàn bộ 168,114 nodes.
   Phân công sensitivity: resolution 1.6 (34 communities, NMI=0.6905, composite_score=0.6265) là sensitivity branch chính — kết luận Hidden-vs-Overrated nhất quán (δ: 1.1=0.2062 vs 1.6=0.2082). Resolution 1.0 có thể dùng làm cross-check bổ sung nếu reviewer yêu cầu (composite_score=0.9005, NMI=0.70089, nhưng over_merge_warning=true).
   Paper narrative được đề xuất: "Louvain community detection at resolution γ=1.1 (mean NMI=0.700 across 10 seeds, 24 communities, pct_top3=49.2%) identified structurally coherent groups. Sensitivity analysis at γ=1.6 (34 communities) confirmed consistent Hidden–Overrated gap (Cliff's δ∈[0.206, 0.208], p<3×10⁻⁵ both settings), supporting robustness to resolution choice."
   Ghi chú về rounding: mean NMI thực tế = 0.69968 → paper report là "0.700" (làm tròn 3 chữ số thập phân). Nếu reviewer kiểm tra artifact sẽ thấy 0.69968 trong metrics.json — nên note tường minh trong supplementary hoặc footnote để tránh bị hỏi.

---

**Task 3: Proxies thật full graph + runtime**

File liên quan trực tiếp (để kiểm tra):

- Code chính: src/mapr2026_v3/diffusion_proxies.py
- Input chính: data/processed/graph_csr.npz
- Output/check chính: data/processed/diffusion_proxies.parquet, outputs/mapr2026_v3_results/diffusion_proxies_status.json, outputs/mapr2026_v3_results/runtime_breakdown.csv, outputs/mapr2026_v3_results/baseline_ranking_metrics.csv, outputs/mapr2026_v3_results/metric_correlation_matrix.json

1. Bối cảnh task  
   Task này là Group 3 baseline trong Track B tại docs/MAPR2026_v3_team_parallel_coding_plan.md, kết nối trực tiếp RQ2b (metric correlation) và RQ3 (proxy vs surrogate) tại MAPR2026_Implementation_Plan_v3.md.

2. Mục đích khoa học + hypothesis  
   Mục tiêu là tạo baseline diffusion rẻ để kiểm tra proxy utility: one-hop/two-hop có thể xấp xỉ IC đến đâu trước khi cần GNN. Hypothesis là proxy tương quan đáng kể với IC, và giúp định lượng trade-off chất lượng so với tốc độ.

3. Thiết kế và protocol  
   Diễn giải phương pháp (chi tiết): Task 3 bám nguyên tắc "contract-first" để tránh sai số lan xuống toàn bộ baseline benchmarking. Pipeline bắt đầu từ kiểm tra cấu trúc CSR (đặc biệt tính đối xứng vô hướng hai chiều) vì công thức two-hop tối ưu phụ thuộc trực tiếp vào giả định này. Sau đó hệ thống tính one-hop và two-hop trên toàn bộ active graph theo cùng một mapping node_id để không phát sinh lệch chỉ số khi join với các artifact khác. Trước khi ghi kết quả, pipeline áp dụng các guard bắt buộc (duy nhất node_id, phủ đủ số node, không NaN) để đảm bảo artifact hợp lệ cho eval harness. Cuối cùng runtime được ghi riêng cho full-graph inference nhằm giữ công bằng khi so sánh với các baseline/surrogate khác trong cùng bảng runtime.
   Trong main(), script load CSR và gọi \_assert_csr_bidirectional() để kiểm tra contract vô hướng hai chiều. Proxy được tính bằng \_compute_one_hop() và \_compute_two_hop(), sau đó validate unique node_id + full coverage + no NaN trước khi ghi diffusion_proxies.parquet. Runtime được cập nhật qua \_upsert_runtime_row() vào runtime_breakdown.csv, đồng thời status được ghi qua write_json() vào diffusion_proxies_status.json.

4. Kết quả chính  
   Artifact proxies đang ở mode real_full_graph, rows = 168,114 trong diffusion_proxies_status.json. Runtime theo runtime_breakdown.csv: one_hop = 0.003815s, two_hop = 0.004734s (inference per full graph theo bảng hiện tại). Lưu ý: `diffusion_proxies` row trong runtime_breakdown = 1.089s là runtime của toàn bộ artifact (cả hai proxy cùng lúc + overhead I/O), không phải inference của riêng two_hop.
   Hiệu năng ranking (baseline_ranking_metrics.csv):

   | Model        | Spearman   | NDCG@10%   | Precision@10% | Inference (s) |
   | ------------ | ---------- | ---------- | ------------- | ------------- |
   | degree       | 0.8263     | 0.8815     | 0.60          | 0.006169      |
   | pagerank     | 0.8241     | 0.8568     | 0.56          | 0.005640      |
   | kshell       | 0.8159     | 0.6771     | 0.50          | 0.000351      |
   | node2vec_lr  | 0.8108     | 0.8573     | 0.592         | 0.021849      |
   | **two_hop**  | **0.8039** | **0.8478** | **0.55**      | **0.004734**  |
   | one_hop      | 0.6877     | 0.8329     | 0.52          | 0.003815      |
   | betweenness  | 0.7164     | 0.7348     | 0.42          | 0.000270      |
   | mlp_raw_attr | 0.4353     | 0.6012     | 0.424         | 0.000473      |
   | views        | 0.4261     | 0.6098     | 0.45          | 0.006650      |
   | views_day    | 0.3655     | 0.5795     | 0.41          | 0.002651      |

   GNN models (surrogate_ranking_metrics.csv): gnn_centrality Spearman = 0.8172 (inference = 0.048s, train = 17.4s); gnn_full = 0.8130 (inference = 0.050s, train = 17.8s); gnn_graph_only = 0.4724; gnn_raw_attr = 0.5357. MC IC labeling = 480s.
   Caveat cho reviewer: row `gnn_random` hiện có runtime/train time rất lớn (4.43s / 2132.7s), không đồng pha với các model còn lại (~0.048s / ~17s). Đây là sanity row đặc biệt và không nên dùng làm mốc chính cho so sánh hiệu năng nếu chưa xác nhận lại cấu hình run.
   Correlation matrix (metric_correlation_matrix.json): IC-one_hop = 0.717, IC-two_hop = 0.815.
   Speedup context: `diffusion_proxies.py` (compute cả hai proxies từ CSR từ đầu) = 1.089s → **480s / 1.089s ≈ 441× nhanh hơn MC IC**. Với Spearman chỉ kém gnn_centrality 0.013 (0.804 vs 0.817).
   Lưu ý kỹ thuật về runtime tables: `runtime_sec` cho `one_hop` (0.003815s) và `two_hop` (0.004734s) trong `baseline_ranking_metrics.csv` và `runtime_breakdown.csv` là thời gian **convert preloaded values sang PyTorch tensor** trong `eval_heuristic()` (run_baselines.py:283-306) — KHÔNG phải thời gian compute từ CSR. Thời gian compute thực sự từ scratch được ghi ở row `diffusion_proxies` (1.089s). Khi viết paper về runtime, dùng 1.089s (cả hai proxies) làm baseline, không dùng các số một-digit milliseconds từ bảng baseline.

5. Phân tích diễn giải  
   Task 3 hoàn thành mạnh về cả operational objective lẫn scientific utility sau khi fix công thức. Full-graph inference vẫn nhanh và reproducible, đồng thời two-hop hiện thể hiện tín hiệu mạnh hơn one-hop trên các chỉ số ranking chính. Quan trọng hơn, two-hop đã tiệm cận nhóm baseline mạnh (degree/pagerank) và khá gần gnn_centrality về Spearman, cho thấy proxy giải tích O(E) có thể nén phần lớn tín hiệu cấu trúc liên quan IC trong setting hiện tại. Kết quả này hỗ trợ tốt hơn cho RQ2b/RQ3: cheap diffusion proxies có thể đạt mức tương quan cao với IC khi định nghĩa đúng theo weighted-cascade.

6. Rủi ro và giới hạn  
   Rủi ro mismatch công thức giữa plan và implementation đã được xử lý: two-hop hiện bám công thức weighted-cascade theo plan. Giới hạn còn lại là đây vẫn là proxy cục bộ 2-hop (không mô phỏng đầy đủ cascade nhiều bước), nên không nên diễn giải như thay thế hoàn toàn cho IC Monte Carlo ở mọi mục tiêu phân tích.

7. Kết luận hành động  
   Task 3 đã đạt chuẩn pipeline và hiện cũng nhất quán phương pháp với plan. Có thể chốt claim ở mức mạnh hơn trước: two-hop proxy (định nghĩa đúng) là baseline cạnh tranh cho RQ2b/RQ3, đồng thời vẫn giữ ghi chú rằng IC Monte Carlo là chuẩn tham chiếu cuối cùng. Với RQ3, framing nên là: GNN có lợi thế rõ khi so với MC IC; còn so với analytical proxies, lợi thế chính hiện nghiêng về tính mô-đun/khả năng mở rộng pipeline hơn là tốc độ thuần. Về runtime, cần viết chính xác rằng một số baseline riêng lẻ (degree/pagerank/one-hop/two-hop) nhanh hơn GNN, trong khi artifact tổng hợp diffusion_proxies full-graph có thể chậm hơn inference của một model GNN đơn lẻ.

---

**Task 4: Tạo typology IC×views (True/Hidden/Overrated/Non)**

File liên quan trực tiếp (để kiểm tra):

- Code chính: src/mapr2026_v3/typology_ic_views.py
- Input chính: data/processed/ic_scores_primary.parquet, data/processed/node_attributes.parquet
- Output/check chính: data/processed/typology_labels_ic_views.parquet, outputs/mapr2026_v3_results/typology_quadrant_report.json

1. Bối cảnh task  
   Task này là lõi của divergence analysis trong Track B, được mô tả ở docs/MAPR2026_v3_team_parallel_coding_plan.md, và gắn trực tiếp với RQ2, đồng thời là đầu vào cho RQ4 và RQ3b trong MAPR2026_Implementation_Plan_v3.md.

2. Mục đích khoa học + hypothesis  
   Mục tiêu là kiểm định divergence hypothesis: popularity (views) không đồng nhất với diffusion influence (IC). Nếu không có typology 2x2, bạn không thể tách nhóm Hidden/Overrated để kiểm tra cơ chế cấu trúc hay độ khó dự đoán theo nhóm.

3. Thiết kế và protocol  
   Diễn giải phương pháp (chi tiết): Task 4 đóng vai trò chuẩn hóa định nghĩa divergence thành một thủ tục gán nhãn có thể tái lập. Sau khi hợp nhất IC score và thuộc tính node theo node_id, pipeline dùng cùng một quy tắc percentile (top 10%) cho cả trục IC và trục views để tạo bốn nhóm True/Hidden/Overrated/Non bằng logic điều kiện tường minh. Thiết kế này giúp tránh việc chọn ngưỡng tùy tiện theo từng lần chạy. Song song đó, hệ thống sinh quadrant report với số lượng/tỷ lệ từng nhóm và cờ min_quadrant_ok để đánh giá ngay khả năng thống kê của downstream tests; tức là chất lượng phân nhóm không chỉ được nhìn qua "có tạo được 4 nhóm" mà còn qua "có đủ cỡ mẫu để kiểm định hay không".
   Trong main(), df_ic được merge với node_attributes theo node_id; ngưỡng top 10% cho IC và views lấy bằng quantile. Nhãn typology được gán qua \_assign_typology_labels() (dựa trên \_assign_typology_label()), còn báo cáo quadrant được tạo bởi \_build_quadrant_report() với các cờ min_quadrant_ok và two_sample_applied. Nếu bật --require-min-quadrant và min_quadrant_ok=false thì script fail-fast bằng ValueError.

4. Kết quả chính  
   Artifact typology_quadrant_report.json: n_total = 5,000 (IC simulation sample — ~3% của 168,114 active nodes; IC chỉ được tính trên sample do chi phí MC), ic_threshold = 77.67 (top-10% quantile của ic_score_mean), views_threshold = 66,147.20 (top-10% quantile của views).
   Phân bố: True = 215 (4.3%), Hidden = 285 (5.7%), Overrated = 285 (5.7%), Non = 4,215 (84.3%). min_quadrant_ok = true, two_sample_applied = false.
   Logic symmetry: top-10% IC = 500 nodes, top-10% views = 500 nodes. Overlap (True) = 215 → Hidden = 500 − 215 = 285; Overrated = 500 − 215 = 285. Sự đối xứng này là hệ quả của định nghĩa percentile cứng — không phải coincidence. True nhỏ (215 < 500) phản ánh divergence cao giữa IC và views (IC-views ρ = 0.469).

5. Phân tích diễn giải  
   Task 4 đã hoàn thành đúng vai trò trung tâm cho RQ2: True=215 (4.3%) thấp hơn kỳ vọng nếu IC-views tương quan hoàn toàn (khi đó True ≈ 500), cho thấy divergence rõ và không collapse. Với Hidden=Overrated=285, MWU test (Task 5) có power tốt cho so sánh nhóm đối xứng. Giá trị ic_threshold=77.67 và views_threshold=66,147 là các con số cụ thể phản ánh phân phối thực tế của Twitch EN dataset — nên được report trong paper để đảm bảo reproducibility.
   Lưu ý scope: typology được tạo trên sample 5,000 nodes (~3% graph). Kết luận về Hidden/Overrated là valid trong scope sample này, nhưng không tự động suy rộng cho toàn bộ 168k nodes. Đây là limitation cần acknowledge trong paper.

6. Rủi ro và giới hạn  
   Typology phụ thuộc ngưỡng top 10% — nhưng Stage 3 robustness_summary.json cho thấy SIS-based typology (toàn graph) có Jaccard 0.747/1.000/0.793 tại threshold 15%/20%/25%, all_pass=true. Tuy nhiên đây là robustness của SIS typology, KHÔNG phải IC×views typology. Hiện chưa có threshold sensitivity test tương đương cho typology_labels_ic_views. Đây là gap cần ghi nhận — nếu reviewer yêu cầu threshold sensitivity cho IC×views, sẽ cần chạy thêm. Ngoài ra typology là hard-cutoff label, chưa phản ánh boundary uncertainty với nodes sát ngưỡng.

7. Kết luận hành động  
   Task 4 đạt chuẩn completion-ready cho mục tiêu divergence. Có thể dùng trực tiếp làm nền cho structural profiling (Task 5) và per-group prediction difficulty (RQ3b), với lưu ý phải báo cáo rõ tính phụ thuộc ngưỡng.

---

**Task 5: So sánh Hidden vs Overrated bằng MWU + Cliff’s Delta + BH-FDR**

File liên quan trực tiếp (để kiểm tra):

- Code chính: src/mapr2026_v3/typology_ic_views.py
- Input chính: data/processed/typology_labels_ic_views.parquet, data/processed/node_attributes.parquet, data/processed/centrality_table.parquet, data/processed/kshell_table.parquet, data/processed/community_features.parquet
- Output/check chính: outputs/mapr2026_v3_results/structural_profiling.csv

1. Bối cảnh task  
   Task 5 là bước biến divergence từ mô tả thành bằng chứng thống kê cho RQ4 và tăng lực diễn giải cho RQ2, theo thiết kế ở docs/MAPR2026_v3_team_parallel_coding_plan.md.

2. Mục đích khoa học + hypothesis  
   Hypothesis cơ chế cấu trúc là Hidden khác Overrated theo các đặc trưng bridge/structural, không chỉ khác views bề mặt. Task này kiểm định giả thuyết bằng effect size và multiple testing control thay vì chỉ nhìn mean.

3. Thiết kế và protocol  
   Diễn giải phương pháp (chi tiết): Task 5 được thiết kế như một gói kiểm định đầy đủ, không dừng ở p-value đơn lẻ. Pipeline trước hết dựng một analysis frame nhất quán từ typology + structural features và fail-fast nếu thiếu dữ liệu quan trọng để tránh thiên lệch ngầm. Với mỗi đặc trưng, hệ thống dùng Mann-Whitney hai phía (phù hợp dữ liệu lệch/phân phối không chuẩn) để kiểm tra khác biệt Hidden-vs-Overrated. Song song đó Cliff's delta được tính để phản ánh cỡ hiệu ứng thực tế, tránh trường hợp "có ý nghĩa thống kê nhưng hiệu ứng nhỏ". Cuối cùng BH-FDR được áp dụng trên toàn bộ tập kiểm định để kiểm soát false discovery khi test nhiều biến đồng thời, rồi mới gắn nhãn significant theo cả tiêu chí p_corrected và |delta|.
   Pipeline tạo frame bằng \_build_structural_frame() (join typology + node attributes + centrality/kshell + community và fail-fast nếu thiếu giá trị). Thống kê nằm trong \_compute_structural_profiling(): Mann-Whitney hai phía trên 6 đặc trưng, Cliff’s delta qua \_cliffs_delta_from_u(), rồi BH-FDR bằng multipletests; tiêu chí significant là p_corrected < 0.05 và |delta| >= delta_threshold.

4. Kết quả chính  
   Primary artifact structural_profiling.csv được tạo từ partition resolution = 1.1 (24 communities, khóa trong base.yaml). Kết quả 6 features (MWU + BH-FDR, tiêu chí: p_corrected < 0.05 AND |δ| ≥ 0.20):

   | Feature                       | Hidden mean | Overrated mean | Cliff's δ | p_corrected | Significant |
   | ----------------------------- | ----------- | -------------- | --------- | ----------- | ----------- | --- | ------- |
   | degree                        | 197.84      | 99.53          | 0.4549    | 1.69e-20    | ✓           |
   | pagerank                      | 1.195e-5    | 7.782e-6       | 0.3524    | 5.01e-13    | ✓           |
   | kshell                        | 96.05       | 54.80          | 0.5759    | 7.23e-32    | ✓           |
   | betweenness                   | 1.628e-5    | 5.785e-6       | 0.3592    | 2.35e-13    | ✓           |
   | cross_community_edge_fraction | 0.4169      | 0.3423         | 0.2062    | 2.45e-5     | ✓           |
   | life_time                     | 1740.78     | 1881.43        | −0.1069   | 0.0273      | ✗ (         | δ   | < 0.20) |

   5/6 biến đạt ý nghĩa (n_hidden=285, n_overrated=285). Effect size category theo Vargha & Delaney (2000): kshell δ=0.576 = **large**; degree δ=0.455, betweenness δ=0.359, pagerank δ=0.352 = **medium**; cross_community_edge_fraction δ=0.206 = **small-medium** (trên threshold 0.147 của "small"). Ngưỡng δ ≥ 0.20 dùng trong pipeline nằm giữa small và medium theo convention này — conservative nhưng hợp lý cho claim "meaningful practical difference".
   BH-FDR với 6 tests: với 5 p-value rất nhỏ (≤ 5×10⁻¹³), correction không làm đổi kết quả. life_time p_raw=0.0273 = p_corrected (rank 6/6, correction factor 6/6=1.0) — fail vì |δ|=0.107 < 0.20, không phải vì p.
   Đáng chú ý: Hidden có life_time NGẮN hơn Overrated (1741 vs 1881 ngày). Điều này nhất quán với narrative "Hidden = newer/niche streamers chưa tích lũy được views audience" — không mâu thuẫn với claim influence. Phù hợp với lifetime_validation fail ở Task 6.
   Sensitivity (task5_resolution_sensitivity_comparison.json): cross_community_edge_fraction nhất quán ở res=1.1 (δ=0.2062, p=2.45e-5) và res=1.6 (δ=0.2082, p=2.05e-5) — same_direction, both meaningful, significance not flipped.

5. Phân tích diễn giải  
   Bằng chứng hiện tại ủng hộ structural mechanism hypothesis ở mức so sánh nhóm: Hidden nổi bật hơn Overrated ở các chỉ số cấu trúc và connectivity, phù hợp claim cơ chế hơn là chỉ popularity mismatch. Việc cross_community_edge_fraction có delta vượt ngưỡng là điểm quan trọng cho RQ4. Tuy nhiên cần giữ caveat rằng đây là bằng chứng within-sample Hidden-vs-Overrated, chưa tự nó khẳng định “structural uniqueness beyond null model”; đặc biệt diễn giải bridge nên tránh tuyệt đối hóa khi null-model chưa ủng hộ mạnh.

6. Rủi ro và giới hạn  
   So sánh là univariate theo từng feature; chưa phải mô hình nhân quả đa biến. Tương quan cao giữa các structural metrics (degree-kshell ρ=0.993, degree-pagerank ρ=0.981 từ metric_correlation_matrix.json) nghĩa là 4 trong 5 significant features phần lớn đo cùng tín hiệu “network position”. Tín hiệu độc lập thực sự trong 5 biến significant là: (1) network position / node centrality (được đo bởi degree/kshell/pagerank/betweenness — redundant nhưng nhất quán); (2) cross_community_edge_fraction — feature duy nhất đến từ meso-level community structure, và là feature duy nhất có tín hiệu độc lập với network position. Reviewer có thể hỏi: “nếu control for degree, cross_community_edge_fraction vẫn significant không?” — đây là câu hỏi hợp lý cần chuẩn bị caveat. Scope: so sánh dựa trên 285 vs 285 nodes từ sample 5,000 (~3% graph).

7. Kết luận hành động  
   Task 5 đạt mức bằng chứng thống kê tốt cho RQ4 và củng cố RQ2. Primary artifact structural_profiling.csv đã được tạo từ partition res=1.1 (locked trong base.yaml) — không cần thay đổi cấu hình. Sensitivity branches đã validate: res=1.6 cho cùng direction, δ trong khoảng [0.206, 0.208], significance không flip — claim là robust theo resolution choice. kshell cho δ cao nhất (0.576), tức Hidden nổi bật nhất ở core position; cross_community_edge_fraction có δ thấp nhất trong 5 biến significant (0.206) nhưng đủ threshold và là feature duy nhất đến từ community structure (Task 2 output). Khi viết kết luận học thuật: ghi rõ giới hạn univariate comparison và tính quan sát; tránh overclaim "structural uniqueness" đặc biệt khi null model (Task 7) cho kết quả thận trọng.

---

Dưới đây là báo cáo chi tiết cho Task 6 và Task 7 theo đúng cấu trúc 7 phần, bám vào code và artifact hiện có.

**Task 6: External corroboration bằng life_time**

File liên quan trực tiếp (để kiểm tra):

- Code chính: src/mapr2026_v3/typology_ic_views.py
- Input chính: data/processed/typology_labels_ic_views.parquet, data/processed/node_attributes.parquet, data/processed/community_features.parquet
- Output/check chính: outputs/mapr2026_v3_results/lifetime_validation.json, outputs/mapr2026_v3_results/language_validation.json, docs/assumptions_limitations.md, outputs/mapr2026_v3_results/preflight_person2_latest.txt

1. Bối cảnh task  
   Task này nằm trong Track B để bảo vệ construct validity của typology (IC x views), được mô tả trong docs/MAPR2026_v3_team_parallel_coding_plan.md. Nó phục vụ trực tiếp RQ2 về độ tin cậy ngoài các chỉ số cấu trúc thuần trong MAPR2026_Implementation_Plan_v3.md.

2. Mục đích khoa học + hypothesis liên quan  
   Mục tiêu là kiểm định robustness/validity hypothesis: nếu typology thật sự mang ý nghĩa influence potential, thì phải có dấu hiệu corroboration từ biến exogenous như life_time sau khi kiểm soát degree. Đây là kiểm định “độc lập tương đối” vì IC labels không dùng life_time trong quá trình tạo nhãn.

3. Thiết kế và protocol  
   Diễn giải phương pháp (chi tiết): Task 6 áp dụng chiến lược kiểm định hai tầng để giảm nguy cơ kết luận sai do nhiễu degree. Tầng 1 dùng partial Spearman (IC với life_time, có kiểm soát degree) để đo liên hệ tổng quát sau khi đã triệt bớt ảnh hưởng của độ bậc. Tầng 2 chia node theo degree quintile và chạy MWU + BH-FDR trong từng strata để kiểm tra liệu tín hiệu có nhất quán theo các vùng cấu trúc khác nhau hay không. Gate thành công được định nghĩa rõ bằng số quintile có ý nghĩa thống kê. Nếu gate này không đạt, pipeline chuyển sang language fallback như một lớp corroboration phụ trợ; quan trọng là fallback được ghi nhận tường minh là supplementary chứ không thay thế kết quả lifetime.
   Code dùng 2 lớp kiểm định với vị trí rõ ràng trong typology_ic_views.py.
   1. Partial Spearman: \_partial_spearman_rho() được gọi trong \_compute_lifetime_validation() để tính Spearman(IC, life_time | degree) theo residualized ranks.
   2. Stratified MWU theo degree quintile + BH-FDR đều nằm trong \_compute_lifetime_validation().
      Tiêu chí thành công là n_quintiles_significant >= 3 ngay trong payload lifetime_validation; nếu fail theo điều kiện trigger trong code (partial_spearman_rho < 0.05 với rho có dấu OR n_quintiles_significant < 3) và không phải dry-run thì main() gọi \_compute_language_validation() và ghi language_validation.json như fallback bổ sung.

4. Kết quả chính  
   Kết quả thực tế trong lifetime_validation.json: partial_spearman_rho = −0.020, p = 0.157 (n.s.), n_quintiles_significant = 0/5, success = false. Phân bố Hidden theo quintile: Q0=2, Q1=7, Q2=20, Q3=46, Q4=210 node — power bất cân bằng nghiêm trọng ở Q0/Q1. Không có quintile nào đạt significant sau BH correction.
   Fallback language_validation.json đã được trigger (trigger_condition trong code là partial_spearman_rho < 0.05 OR n_quintiles_significant < 3). Kết quả language:
   - nmi_community_language_labeled = 0.4832 (community và ngôn ngữ không đồng nhất hoàn toàn)
   - neighbor_language_entropy: Hidden mean = 0.2063, Overrated mean = 0.4221
   - Cliff's δ = −0.4930, p = 2.20e-24 (highly significant)
   - Kết luận: Hidden có neighbor language diversity THẤP HƠN Overrated rõ rệt — tức Hidden tập trung trong cộng đồng ngôn ngữ đồng nhất hơn, Overrated có kết nối đa ngôn ngữ hơn.
     Fallback được ghi rõ là supplementary, không thay thế lifetime gate.

5. Phân tích diễn giải  
   Task 6 hiện không xác nhận được validity theo life_time (theo định nghĩa gate của plan), nên robustness claim cho RQ2 phải giữ mức thận trọng. Điểm tốt là pipeline xử lý đúng IF PROBLEM logic và không “che” fail bằng fallback. Preflight xác nhận cơ chế này đã chạy đúng tại outputs/mapr2026_v3_results/preflight_person2_latest.txt.

6. Rủi ro và giới hạn  
   Rủi ro chính là power bất cân bằng theo quintile (Hidden quá ít ở low-degree bins), khiến kiểm định dễ fail dù có tín hiệu thật yếu-vừa. Ngoài ra life_time có thể phản ánh tenure hơn là influence, nên thất bại ở Task 6 không bác bỏ hoàn toàn typology, nhưng làm suy yếu external corroboration. Hạn chế này đã được ghi nhận trong docs/assumptions_limitations.md.

7. Kết luận hành động  
   Task 6 hoàn thành đúng protocol nhưng outcome là inconclusive theo gate đã lock. Khi báo cáo RQ2 nên ghi rõ: lifetime validation fail, language validation chỉ supplementary, không thay thế evidence chính từ typology/structural/null package.

---

**Task 7: Configuration null để kiểm tra degree artifact**

File liên quan trực tiếp (để kiểm tra):

- Code chính: src/mapr2026_v3/null_model_typology.py
- Input chính: data/processed/typology_labels_ic_views.parquet, data/processed/graph_csr.npz
- Output/check chính: outputs/mapr2026_v3_results/null_model_typology_summary.json, outputs/mapr2026_v3_results/preflight_person2_latest.txt

1. Bối cảnh task  
   Task này là kiểm định null-model cốt lõi cho độ vững diễn giải của RQ2, nhằm tách signal divergence thật khỏi artifact do degree sequence, theo mô tả trong docs/MAPR2026_v3_team_parallel_coding_plan.md.

2. Mục đích khoa học + hypothesis liên quan  
   Hypothesis cần kiểm định: nếu typology Hidden chỉ là hệ quả cơ học của degree distribution, thì trên configuration null sẽ tái tạo pattern tương tự. Ngược lại, nếu real graph khác null rõ rệt thì có cơ sở nói divergence mang nội dung cấu trúc thật.

3. Thiết kế và protocol  
   Diễn giải phương pháp (chi tiết): Task 7 được xây theo logic "kiểm định vượt-degree-sequence". Từ cùng tập node mẫu, pipeline dựng subgraph thực rồi tạo nhiều configuration-model realization để giữ phân phối degree nhưng xáo trộn cấu trúc liên kết bậc cao. Trên mỗi realization null, IC được mô phỏng bằng đúng công thức weighted-cascade như ở graph thực để đảm bảo so sánh cùng propagation regime. Hệ thống đọc kết quả theo hai trục: (i) mức đồng thuận ranking giữa real IC và null IC (Spearman), (ii) chênh lệch betweenness của nhóm Hidden giữa real-subgraph và null-subgraph. Trục (ii) dùng rule scale-aware (gap chuẩn hóa theo sigma thích nghi) để tránh lỗi ngưỡng tuyệt đối không cùng thang đo, đồng thời thêm ngữ cảnh rho_mean để hạn chế diễn giải quá mức khi rank agreement còn yếu.
   Trong main(), script thực thi contract 500 nodes x 10 realizations x 100 runs/node (rerun latest): sample node từ typology, dựng subgraph thật bằng \_build_real_subgraph_from_csr(), rồi tạo null bằng nx.configuration_model theo từng realization. IC weighted-cascade trên null được tính qua \_simulate_ic_means(); sau đó so Spearman(real IC, null IC) và hidden betweenness real-vs-null (qua \_hidden_betweenness_mean()). Rule diễn giải tự động nằm trong \_build_null_interpretation().

4. Kết quả chính  
   Artifact cuối cùng trong null_model_typology_summary.json cho thấy n_nodes = 500, n_realizations = 10, n_runs_per_node = 100.  
   rho_mean = 0.4409 (rho_std = 0.00234); hidden_bet_real = 1.68e-05, hidden_bet_null_mean = 6.31e-05, hidden_bet_null_std = 3.74e-05, hidden_betweenness_gap = -4.63e-05, hidden_betweenness_gap_sigma = -1.24. Diễn giải hiện tại là “comparable to configuration null on this scale; potential degree-distribution artifact (rho_mean=0.441)” tại null_model_typology_summary.json. Preflight xác nhận null package đầy đủ Task 7/8/9 tại outputs/mapr2026_v3_results/preflight_person2_latest.txt.

5. Phân tích diễn giải  
   Với kết quả hiện tại, Task 7 không ủng hộ mạnh “structural uniqueness” của Hidden theo tiêu chí configuration null; ngược lại nó buộc narrative RQ2 phải thận trọng hơn và nghiêng về khả năng degree-distribution artifact một phần. Dấu gap chuẩn hóa hiện tại là âm (hidden_betweenness_gap_sigma = -1.24), tức Hidden-betweenness trên real-subgraph thấp hơn null-mean theo thang đo thích nghi. Điểm tích cực là đây là bằng chứng robustness trung thực, giúp tránh overclaim divergence.

6. Rủi ro và giới hạn  
   Rủi ro ngưỡng tuyệt đối 0.05 đã được xử lý: rule diễn giải hiện là scale-aware (dựa trên gap và sigma thích nghi) và có thêm ngữ cảnh rho_mean. Sau rerun 10 realizations, rho_std gần như không đổi so với run 3 realizations (0.00230 -> 0.00234), cho thấy ước lượng rho khá ổn định; tuy vậy kết luận null-model vẫn nên giữ ở mức thận trọng do scope 500-node subgraph và khác biệt topology giữa real-subgraph vs null-subgraph. Ngoài ra cần ghi rõ scope mismatch: so sánh betweenness đang là giữa real-subgraph và null-subgraph (cùng tập node mẫu nhưng khác edge topology), không tương đương trực tiếp với phát biểu full-graph role.

7. Kết luận hành động  
   Task 7 đã hoàn thành contract kỹ thuật với rule diễn giải đúng scale. Kết quả hiện vẫn không ủng hộ claim “Hidden vượt xa null”, nên narrative phù hợp là: divergence quan sát có thật, nhưng thành phần cấu trúc có thể bị degree-sequence giải thích một phần; cần tổng hợp cùng evidence từ Task 5 và package permutation null (Task 8/9).

---

Dưới đây là báo cáo Task 8 và Task 9 theo cấu trúc 7 phần, dựa trên code và artifact hiện có.

**Task 8: Views-permutation null (phá views, giữ IC/graph)**

File liên quan trực tiếp (để kiểm tra):

- Code chính: src/mapr2026_v3/typology_ic_views.py
- Input chính: data/processed/typology_labels_ic_views.parquet (được tạo từ ic_scores_primary + node_attributes)
- Output/check chính: outputs/mapr2026_v3_results/views_permutation_null_summary.json

1. Bối cảnh task  
   Task 8 là nhánh MUST của B5 core theo docs/MAPR2026_v3_team_parallel_coding_plan.md, phục vụ RQ2 và RQ2b như một kiểm định mechanism-specific cho robustness.

2. Mục đích khoa học + hypothesis  
   Mục tiêu là tách đóng góp riêng của views vào divergence: nếu chỉ xáo views mà cấu trúc divergence thay đổi theo hướng random, thì phần divergence quan sát được có thể là non-random mechanism thay vì nhiễu.

3. Thiết kế và protocol  
   Diễn giải phương pháp (chi tiết): Task 8 kiểm định vai trò riêng của views bằng permutation test có lặp. Thiết kế giữ cố định IC score và graph, chỉ hoán vị views trên cùng tập node để tạo null distribution cho các chỉ số typology (hidden/overrated counts, agreement/divergence rates). Nhờ lặp nhiều permutation, pipeline không dựa vào một lần xáo trộn duy nhất mà ước lượng được cả trung bình và độ lệch chuẩn của null. Kết quả thực sau đó được định vị trong phân phối này thông qua empirical p-values (tail probabilities), giúp trả lời câu hỏi liệu pattern quan sát có thực sự bất thường so với ngẫu nhiên hay không dưới giả thuyết "views assignment không mang tín hiệu".
   Hàm \_compute_views_permutation_null() giữ IC scores cố định, permute views qua labeled nodes và rebuild typology lặp lại n_permutations lần.  
   Các thống kê chính: hidden_count, overrated_count, agreement/divergence rate, empirical p-values.  
   Rule diễn giải được mã hóa trực tiếp trong các nhánh if/elif của \_compute_views_permutation_null() dựa trên agreement_rate thực nghiệm so với null mean và empirical p-value.  
   Task được gọi trong main() và ghi artifact qua write_json(views_permutation_json_path, views_perm_summary).

4. Kết quả chính  
   Artifact ở views_permutation_null_summary.json:
   1. n_nodes_labeled = 5000, n_permutations = 200.
   2. Real: agreement_rate = 0.886, divergence_rate = 0.114.
   3. Null mean: agreement_rate_mean = 0.8200, divergence_rate_mean = 0.1800.
   4. Empirical p cho agreement_rate_ge_real = 0.004975 (ý nghĩa).  
      Diễn giải tự động: observed agreement cao hơn null, divergence pattern non-random.
   5. Với n_permutations = 200 và công thức empirical p có +1 correction, p-value nhỏ nhất có thể đạt là 1/(200+1) = 0.004975; giá trị hiện tại chính là mức sàn này (0/200 permutation extreme hơn real).

5. Phân tích diễn giải  
   Kết quả ủng hộ robustness theo hướng mechanism-specific: khi phá views, mức alignment thực nghiệm giữa views-IC bị giảm đáng kể về null baseline. Điều này cho thấy pattern quan sát không phải do ngẫu nhiên thuần từ thresholding.

6. Rủi ro và giới hạn  
   Test này kiểm định ngẫu nhiên hóa theo hoán vị, nhưng chưa mô hình hóa phụ thuộc theo cấu trúc cộng đồng/degree strata trong phép hoán vị. Ngoài ra, nó chứng minh “non-random” chứ không tự nó chứng minh cơ chế nhân quả đầy đủ.

7. Kết luận hành động  
   Task 8 hoàn thành tốt và là bằng chứng mạnh bổ sung cho RQ2/RQ2b. Nên dùng như “robustness-evidence layer” kèm Task 4/5, không dùng độc lập để kết luận cơ chế.

---

**Task 9: IC-permutation null (phá IC, giữ views/graph)**

File liên quan trực tiếp (để kiểm tra):

- Code chính: src/mapr2026_v3/typology_ic_views.py
- Input chính: data/processed/typology_labels_ic_views.parquet (được tạo từ ic_scores_primary + node_attributes)
- Output/check chính: outputs/mapr2026_v3_results/ic_permutation_null_summary.json

1. Bối cảnh task  
   Task 9 là nhánh MUST còn lại của B5 core theo docs/MAPR2026_v3_team_parallel_coding_plan.md, có execution lock phải chạy đủ cả 2 permutation trước sign-off tại docs/MAPR2026_v3_team_parallel_coding_plan.md.

2. Mục đích khoa học + hypothesis  
   Mục tiêu là tách đóng góp riêng của IC ranking vào divergence: nếu phá IC mà pattern alignment giảm về null, thì divergence thực nghiệm có thành phần thông tin thực từ IC chứ không chỉ từ views.

3. Thiết kế và protocol  
   Diễn giải phương pháp (chi tiết): Task 9 là phép đối xứng cần thiết để tránh kết luận một chiều từ Task 8. Ở đây pipeline giữ nguyên views và graph, chỉ hoán vị IC score để kiểm định giả thuyết ngược lại: nếu IC không mang thông tin cấu trúc thực, pattern divergence hiện tại có thể tái tạo bằng ngẫu nhiên. Cách tính thống kê và empirical p-values tương thích với Task 8 để hai phép thử có thể đối chiếu trực tiếp. Khi cả hai permutation test đều chỉ ra kết quả thực khác null theo cùng hướng, mức tin cậy cho nhận định "divergence pattern non-random" được tăng đáng kể so với việc chỉ dựa vào một phía hoán vị.
   Hàm \_compute_ic_permutation_null() giữ views cố định, permute ic_score_mean qua labeled nodes.  
   Thống kê và rule diễn giải tương tự Task 8; nhánh kết luận được quyết định trong \_compute_ic_permutation_null() theo quan hệ giữa agreement_rate thực nghiệm, null mean và empirical p-value.  
   Pipeline gọi trong main() và ghi artifact qua write_json(ic_permutation_json_path, ic_perm_summary).

4. Kết quả chính  
   Artifact ở ic_permutation_null_summary.json:
   1. n_nodes_labeled = 5000, n_permutations = 200.
   2. Real agreement_rate = 0.886, null agreement_rate_mean = 0.8199.
   3. Empirical p agreement_rate_ge_real = 0.004975 (ý nghĩa).
   4. Hidden real = 285 thấp hơn null mean ~450.255.  
      Diễn giải tự động: observed agreement cao hơn IC-permutation null, divergence pattern non-random.
   5. Tương tự Task 8, với n_permutations = 200 thì p-value floor là 1/201 = 0.004975; kết quả hiện tại cũng chạm mức sàn.

5. Phân tích diễn giải  
   Task 9 cho kết luận cùng chiều với Task 8, tạo đối xứng kiểm định: phá IC hay phá views đều làm pattern tiến về null distribution. Điều này tăng độ tin cậy rằng divergence quan sát là cấu trúc tín hiệu thật của cặp (IC, views), không phải “ảnh ảo” do một phía đơn lẻ.

6. Rủi ro và giới hạn  
   Giống Task 8, đây là bằng chứng ngẫu nhiên hóa mạnh nhưng vẫn là kiểm định thống kê quan sát. Ngoài ra, vì top_pct cố định và thresholding cứng, kết quả có thể nhạy với percentile lock trong trường hợp phân phối sát ngưỡng.

7. Kết luận hành động  
   Task 9 đạt yêu cầu contract và củng cố robustness claim cho RQ2/RQ2b theo hướng mechanism-specific. Cùng với Task 8, đây là cặp evidence rất giá trị để bảo vệ trước phản biện “divergence chỉ là ngẫu nhiên”.

---

**Task 10: Backup residual divergence khi min_quadrant_ok vẫn fail sau two-sample**

File liên quan trực tiếp (để kiểm tra):

- Spec/plan chính: docs/MAPR2026_v3_team_parallel_coding_plan.md, MAPR2026_Implementation_Plan_v3.md
- Code liên quan hiện tại: src/mapr2026_v3/typology_ic_views.py
- Trigger/check chính: outputs/mapr2026_v3_results/typology_quadrant_report.json
- Artifact kỳ vọng khi trigger: outputs/mapr2026_v3_results/residual_divergence_report.json

1. Bối cảnh task  
   Task 10 là nhánh contingency trong plan, chỉ kích hoạt khi min_quadrant_ok vẫn false sau two-sample, theo docs/MAPR2026_v3_team_parallel_coding_plan.md. Nó phục vụ RQ2 theo đường fallback, không phải main path.

2. Mục đích khoa học + hypothesis liên quan  
   Mục tiêu không phải tạo hypothesis mới, mà bảo toàn khả năng kiểm định divergence khi top-10 typology thiếu power. Cụ thể, plan yêu cầu residual score z(rank(IC)) - z(rank(views)) để vẫn định lượng Hidden-like và Overrated-like trong tình huống khó.

3. Thiết kế và protocol  
   Diễn giải phương pháp (chi tiết): Task 10 là cơ chế contingency để duy trì khả năng kiểm định khi typology theo ngưỡng cứng bị thiếu power. Thiết kế theo gate tuần tự nhằm tránh lạm dụng fallback: chỉ khi nhánh chính thất bại về kích thước nhóm, và sau đó nhánh tăng cường two-sample cũng vẫn không đạt, mới chuyển sang residual divergence. Về bản chất, residual approach thay hard quadrant bằng độ lệch liên tục giữa thứ hạng IC và thứ hạng views, giúp giữ thông tin divergence ngay cả khi các ô Hidden/Overrated quá nhỏ để kiểm định ổn định. Vì vậy Task 10 không phải đường chính, mà là "safety valve" để bảo toàn tính kiểm định của RQ2 trong tình huống dữ liệu bất lợi.
   Theo plan, phải chạy sau hai điều kiện nối tiếp:
   1. Bước 1: min_quadrant_ok false sau typology ban đầu.
   2. Bước 2: đã áp dụng two-sample strategy mà vẫn fail.  
      Khi đó mới sinh artifact fallback outputs/mapr2026_v3_results/residual_divergence_report.json theo protocol trong docs/MAPR2026_v3_team_parallel_coding_plan.md.

4. Kết quả chính (trạng thái thực thi hiện tại)  
   Hiện tại trigger chưa xảy ra vì min_quadrant_ok true ở typology_quadrant_report.json, và two_sample_applied false ở typology_quadrant_report.json.  
   Ngoài ra, không có artifact residual file trong thư mục kết quả mapr2026_v3_results và truy vấn theo tên file không thấy kết quả.  
   Quan trọng hơn: trong implementation hiện tại, cờ two_sample_applied đang được set cố định trong \_build_quadrant_report(), nghĩa là logic two-sample/residual chưa được hiện thực đầy đủ trong cùng script.

5. Phân tích diễn giải  
   Về mặt governance: Task 10 chưa cần chạy là đúng vì contingency chưa trigger.  
   Về mặt readiness: fallback path trên giấy có, nhưng execution path trong code còn thiếu mảnh chính (two-sample + residual report generation), nên nếu sau này gặp min_quadrant_ok false thật thì pipeline có nguy cơ không có “đường thoát” đúng contract.

6. Rủi ro và giới hạn  
   Rủi ro là latent implementation gap: plan yêu cầu fallback cụ thể nhưng code hiện mới dừng ở nhánh cảnh báo trong main() khi min_quadrant_ok=false (chưa có bước sinh residual_divergence_report.json). Nếu data shift làm quadrant nhỏ trong future run, đây sẽ là blocker phương pháp.

7. Kết luận hành động  
   Task 10 hiện ở trạng thái not-triggered và not-fully-implemented. Trước sign-off nên quyết định một trong hai:
   1. Implement đầy đủ two-sample + residual artifact path trong script.
   2. Hoặc lock rõ trong tài liệu rằng contingency này chỉ là manual protocol, không auto trong code.

---

**Task 11: Ma trận tương quan global 8x8 + p corrected**

File liên quan trực tiếp (để kiểm tra):

- Code chính: src/mapr2026_v3/typology_ic_views.py
- Input chính: data/processed/ic_scores_primary.parquet, data/processed/node_attributes.parquet, data/processed/centrality_table.parquet, data/processed/kshell_table.parquet, data/processed/diffusion_proxies.parquet
- Output/check chính: outputs/mapr2026_v3_results/metric_correlation_matrix.json, outputs/mapr2026_v3_results/preflight_person2_latest.txt

1. Bối cảnh task  
   Task 11 là MUST theo plan tại docs/MAPR2026_v3_team_parallel_coding_plan.md, trả lời trực tiếp RQ2b và cung cấp context cho RQ3.

2. Mục đích khoa học + hypothesis liên quan  
   Mục tiêu là lượng hóa IC giống/khác các metric rẻ (views, degree, pagerank, kshell, betweenness_approx, one-hop, two-hop) ở mức hệ thống. Đây là kiểm định trực tiếp proxy utility hypothesis và một phần divergence hypothesis.

3. Thiết kế và protocol  
   Diễn giải phương pháp (chi tiết): Task 11 là lớp tổng hợp định lượng cho RQ2b với hai mục tiêu song song: đo tương quan toàn cục và kiểm tra tính dị thể theo regime cấu trúc. Pipeline trước hết dựng frame 8 metric trên cùng tập node hợp lệ với các guard coverage/NA/duplicate để bảo đảm ma trận tương quan không bị méo do lỗi join. Sau đó Spearman pairwise được tính cho toàn bộ cặp metric, và p-values được BH-correct để tránh overclaim khi số phép thử lớn. Ngoài ma trận tổng quát, hệ thống còn tách theo degree quintile để đọc được sự thay đổi của quan hệ IC-views/IC-structural metrics theo từng vùng degree, từ đó hỗ trợ diễn giải vì sao một số baseline mạnh ở global nhưng có thể yếu ở một số strata cụ thể.
   Frame 8 metric được dựng trong \_build_metric_correlation_frame() với các guard coverage/NA/duplicate.  
   Spearman full pairwise và BH-FDR trên upper triangle được tính trong \_compute_metric_correlation_payload(), rồi xuất rho_matrix và p_matrix_corrected đối xứng.  
   Nhánh rho_by_degree_quintile là tùy chọn qua include_rho_by_degree_quintile trong \_compute_metric_correlation_payload().  
   Artifact được tạo trong main() bằng write_json(metric_corr_json_path, metric_corr_payload).

4. Kết quả chính  
   Artifact ở metric_correlation_matrix.json có n_rows_expected = 5000 và coverage_ok = true; payload hiện có các khối metrics, rho_matrix, p_matrix_corrected, column_mapping. Nhánh rho_by_degree_quintile là tùy chọn và không xuất hiện trong artifact hiện tại (không bật include_rho_by_degree_quintile khi chạy).  
   Một số điểm nổi bật:
   1. IC-views rho khoảng 0.4689 ở metric_correlation_matrix.json.
   2. IC-one-hop rho khoảng 0.7171 ở metric_correlation_matrix.json.
   3. IC-two-hop rho khoảng 0.8153 ở metric_correlation_matrix.json.
   4. one-hop và two-hop tương quan cao với nhau (rho khoảng 0.9114), cho thấy hai proxy cùng phản ánh mạnh cấu trúc khuếch tán cục bộ sau khi chuẩn hóa đúng công thức two-hop.
   5. Có column_mapping chuẩn hóa betweenness ở metric_correlation_matrix.json; source hiện tại map từ cột betweenness và được canonical hóa thành betweenness_approx trong payload.
   6. Nhánh rho_by_degree_quintile chưa có trong artifact hiện tại (do không bật cờ include_rho_by_degree_quintile).

5. Phân tích diễn giải  
   Task 11 đang làm rất tốt vai trò RQ2b: nó cho thấy views không đại diện tốt cho IC toàn cục, trong khi sau khi sửa công thức thì two-hop gần IC hơn one-hop ở mức global correlation. Breakdown theo degree quintile là nhánh tùy chọn trong code và chưa được bật ở artifact hiện tại; nếu cần phân tích regime sâu hơn cho paper, có thể rerun với include_rho_by_degree_quintile để bổ sung bằng chứng phân tầng.

6. Rủi ro và giới hạn  
   Ma trận tương quan là bằng chứng liên hệ, không phải nhân quả. Ngoài ra, kết quả phụ thuộc chất lượng upstream features; nếu centrality/proxies drift theo version thì matrix cũng đổi. Dù vậy pipeline đã có guard coverage và schema khá chặt.

7. Kết luận hành động  
   Task 11 đã completion-ready cả về kỹ thuật và giá trị học thuật, đồng thời phản ánh đúng trạng thái mới sau rerun downstream của Task 3. Có thể dùng làm bảng trụ cột cho RQ2b và làm ngữ cảnh bắt buộc cho phần thảo luận RQ3 baseline difficulty. Preflight cũng xác nhận Task 11 đạt chuẩn blocking tại outputs/mapr2026_v3_results/preflight_person2_latest.txt.

---

**Cross-task synthesis: mức hoàn thiện và phần còn thiếu trước sign-off**

**Trạng thái artifact hiện tại (latest snapshot, gồm rerun Task 7 ngày 13/04/2026)**

| Task    | Artifact chính                                     | Trạng thái   | Ghi chú                                                |
| ------- | -------------------------------------------------- | ------------ | ------------------------------------------------------ |
| Task 2  | community_features.parquet, metrics.json           | ✓ Production | res=1.1, 24 comm, stability_warning=false              |
| Stage 3 | sis_metrics.json, kshell_table.parquet             | ✓ Updated    | kshell_max=149, kshell_mean=40.84                      |
| Task 4  | typology_quadrant_report.json                      | ✓ Current    | Hidden=285, Overrated=285, min_quadrant_ok=true        |
| Task 5  | structural_profiling.csv                           | ✓ Updated    | Primary uses res=1.1 partition                         |
| Task 5  | structural_profiling_res1_1.csv, \_res1_6.csv      | ✓ Updated    | Sensitivity confirmed                                  |
| Task 6  | lifetime_validation.json, language_validation.json | ✓ Current    | lifetime fail→language fallback triggered              |
| Task 7  | null_model_typology_summary.json                   | ✓ Current    | rho=0.441, sigma=−1.24 (n_realizations=10, thận trọng) |
| Task 8  | views_permutation_null_summary.json                | ✓ Current    | p=0.005, non-random confirmed                          |
| Task 9  | ic_permutation_null_summary.json                   | ✓ Current    | p=0.005, non-random confirmed                          |
| Task 11 | metric_correlation_matrix.json                     | ✓ Current    | IC-views=0.469, IC-two_hop=0.815                       |

1. Điểm đã đủ mạnh để chốt trong report
   - Community detection (Task 2) đã chuyển sang primary resolution=1.1: over_merge_warning cleared, pct_top3=49.23%, composite score 0.9588 vs 0.9005 cho res=1.0.
   - Task 5 bảng kết quả đầy đủ: 5/6 biến significant, kshell có δ cao nhất (0.576), cross_community_edge_fraction đủ threshold (δ=0.206) và robust theo sensitivity (res=1.6 cho δ=0.208).
   - Task 3 fix tạo thay đổi material: two-hop từ baseline yếu trở thành baseline cạnh tranh (Spearman=0.8039 vs IC), cần phản ánh nhất quán ở bảng so sánh RQ2b/RQ3.
   - Task 8/9 nhất quán theo cả hai hướng permutation, đủ làm lớp robustness cho divergence claim (p=0.005 cả hai).

2. Điểm cần viết thận trọng
   - Task 5 cho bằng chứng Hidden-vs-Overrated mạnh trong sample, nhưng Task 7 cho tín hiệu null-model thận trọng (rho=0.441, sigma=−1.24 vẫn chưa vượt ngưỡng mạnh); không nên claim “Hidden chắc chắn unique beyond degree-sequence null”.
   - Task 6: lifetime validation fail, language validation supplementary chỉ — robustness claim cho RQ2 external corroboration cần giữ mức thận trọng.
   - Language validation cho thấy Hidden có neighbor language entropy THẤP HƠN Overrated (δ=−0.493) — điều này có thể diễn giải là Hidden tập trung trong cộng đồng ngôn ngữ đồng nhất (“niche influencer”), nhưng cần caveat là đây là fallback supplementary.
   - Runtime framing cần tách bạch: GNN nhanh hơn nhiều so với MC IC; nhiều analytical baseline riêng lẻ nhanh hơn GNN, nhưng artifact tổng hợp diffusion_proxies full-graph không nhất thiết nhanh hơn inference của một model GNN đơn lẻ.

3. Gaps còn lại (nếu muốn completion-ready ở mức publication cao)
   - RQ3b hiện đã có artifact per-group đầy đủ cho các model GNN chính (gnn_raw_attr, gnn_graph_only, gnn_centrality, gnn_full) trong per_group_prediction_error.csv, và bằng chứng nhất quán cho thấy Hidden là nhóm khó dự đoán nhất (Spearman thấp nhất, MAE cao nhất). Có thể khóa H5 ở mức kết luận chính, kèm caveat scope test-split.
   - Cần ghi chú ngắn về `gnn_random`: vì runtime/train time hiện bất thường so với các model còn lại, nên xem đây là sanity row và không dùng làm mốc hiệu năng chính cho narrative.
   - Task 10 vẫn là contingency chưa implement đầy đủ đường auto (two-sample + residual artifact), cần quyết định rõ triển khai code hay lock manual protocol trong tài liệu.
