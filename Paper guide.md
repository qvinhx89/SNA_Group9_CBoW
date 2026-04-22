# Hướng dẫn Supervisor: Viết paper MAPR 2026

## Kế hoạch 9 ngày để hoàn thiện một paper dual-operationalization có thể defend

---

## Phần 1: Định danh paper - bài này thực chất là gì

Trước khi viết một dòng nào, cả team phải thống nhất thật rõ paper này đóng góp gì và không claim điều gì.

**Paper này là:** một bài comparative empirical study cho thấy hiệu quả của GNN surrogate learning trong việc xấp xỉ IC influence phụ thuộc mạnh vào cách operationalize IC. Dưới degree-coupled IC (A0), analytical baselines gần như tối ưu. Dưới attribute-community IC (HSCC), GNN message passing có thể khai thác các mẫu cross-community engagement mà flat baselines không truy cập được. Phần stability analysis cho thấy structural binary instability là đóng góp methodological phụ nhưng thực sự mới.

**Paper này không phải:** claim rằng GNN luôn vượt trội cho influence prediction. Đây cũng không phải claim rằng HSCC là diffusion model "đúng" cho Twitch. Và cũng không phải claim rằng MC-IC scores là real influence.

**Tiêu đề nên phản ánh contrast, không phải blanket GNN claim.** Ví dụ:
"When Does Graph Learning Outperform Analytical Baselines? A Comparative Study of IC Operationalizations for Influence Approximation"
hoặc ngắn gọn hơn:
"Regime-Dependent GNN Surrogate Learning for Monte Carlo Influence Estimation on Social Networks."

---

## Phần 2: Hướng dẫn viết theo từng mục / section

### Mục 1 / Section 1 - Introduction (0.5 trang, khoảng 400 từ)

**Đoạn 1 / Paragraph 1 (3-4 câu; có thể draft bằng tiếng Anh như sau).**
Identifying influential users in online social networks is critical for viral marketing, community management, and platform recommendation. Monte Carlo Independent Cascade (MC-IC) simulation provides a principled operationalization of influence potential grounded in the diffusion model of Kempe et al. (2003). However, MC-IC is computationally expensive — requiring hundreds of stochastic simulations per node — making it impractical for repeated evaluation on large-scale graphs.

**Đoạn 2 / Paragraph 2 (3-4 câu; có thể draft bằng tiếng Anh như sau).**
Graph Neural Networks (GNNs) offer a natural surrogate: learn to approximate IC scores from graph structure and node attributes, then deploy the trained model for fast inference. Prior work on GNN-based influence estimation (Kumar et al., 2022; Ling et al., 2023) has focused on single diffusion models, leaving open the question of when learned representations actually outperform simple analytical baselines such as degree centrality. On dense social networks where cascades die quickly, degree itself may capture most of the diffusion signal, leaving little room for GNN improvement.

**Đoạn 3 / Paragraph 3 (4-5 câu - contributions; có thể draft bằng tiếng Anh như sau).**
In this paper, we investigate how the choice of IC operationalization determines whether GNN surrogate learning adds value over analytical baselines. We compare two defensible operationalizations on the Twitch social network (168K nodes, 6.8M edges): (1) weighted cascade (A0), where transmission probability depends only on target degree, and (2) HSCC, a domain-informed variant incorporating source engagement velocity and cross-community amplification. Our contributions are threefold. First, we show that binary influence classification is structurally unstable on dense networks, motivating continuous regression as the principled prediction formulation. Second, we demonstrate that under degree-coupled IC, all GNN architectures converge to the degree centrality ceiling, confirming that the operationalization — not the model architecture — is the binding constraint. Third, under HSCC, GNN message passing captures cross-community engagement patterns that flat baselines cannot access, achieving Spearman ρ = [X] compared to [Y] for the strongest non-graph baseline.

### Mục 2 / Section 2 - Background (0.75 trang)

**2.1 IC và Weighted Cascade.** Ở phần này, hãy định nghĩa mô hình IC theo Kempe et al. (2003). Nêu rõ weighted cascade parameterization `p(u,v) = 1/deg(v)` và cite việc DeepIM (Ling et al., 2023) cũng dùng setup này. Giải thích rằng MC estimation cần `R` simulation runs cho mỗi seed node để tạo ra mean reach score. Sau đó chốt luôn surrogate learning problem: từ IC scores của một tập node con, học một hàm `f(G, X) -> IC scores` cho toàn bộ graph.

**2.2 GNN Architectures.** Viết một đoạn ngắn bao quát 5 architecture đã test, mỗi architecture một câu: GraphSAGE, GCN, GIN, GAT, và APPNP. Mục tiêu ở đây không phải dạy lại GNN, mà chỉ đủ để reviewer thấy vì sao shortlist này hợp lý và được chạy với cùng hyperparameter setting để đảm bảo fair comparison.

**2.3 Evaluation Protocol.** Nêu rõ đây là transductive setting. Định nghĩa Spearman `ρ` là metric chính, `NDCG@10%` là metric phụ. Nhấn mạnh rằng metrics chỉ được tính trên held-out labeled nodes; full-graph inference chỉ dùng để báo runtime. Đồng thời nhắc rõ practical equivalence bound đã pre-register là `|Δ Spearman| ≤ 0.02`.

### Mục 3 / Section 3 - MC-IC as Operational Metric (1.0 trang)

Phần này phải làm được 3 việc: biện minh vì sao IC là một metric hợp lý, trình bày phát hiện về stability, và giới thiệu rõ hai operationalization đang active là `A0` và `HSCC`.

**3.1 Construct Validity và thiết kế operationalization.** Có thể lấy gần nguyên văn đoạn ở `Implementation Plan` Mục 1.1 / Section 1.1 vì đoạn đó đã khá chắc. Sau đó giới thiệu `A0` và `HSCC` như hai operationalization đều defendable nhưng tạo ra hai learning regime khác nhau về bản chất. `A0` mô hình hóa attention dilution thông qua target degree; `HSCC` mô hình hóa source engagement velocity kết hợp cross-community bridging. Phải nói rất rõ rằng `HSCC` là domain-informed design, không phải claim về cơ chế diffusion thật của Twitch.

**3.2 Tính discriminative.** Trình bày IC reach distribution cho `A0` với các con số mean `31.1`, median `6.25`, top-10/median ratio khoảng `8x`. Sau đó giải thích rằng phân phối này heavy-tailed và phù hợp với trực giác influence dynamics ngoài đời: đa số node reach thấp, chỉ một nhóm nhỏ tạo cascade lớn. Với `HSCC`, báo mean reach `4.83` và `CV = 0.583`.

**3.3 Label Stability - một phát hiện mang tính cấu trúc.** Đây là điểm methodological mới nhất của paper. Nêu rõ Jaccard instability (`0.307` ở 150 runs, không vượt `0.68` ngay cả ở 1200 runs). Sau đó đưa nguyên nhân cấu trúc: `84.2%` Louvain communities span qua top-10% boundary và gap-to-noise ratio gần như bằng 0 ở mọi threshold đã test. Kết luận phần này nên viết rất dứt khoát bằng tiếng Anh nếu cần đưa vào paper:
"Binary influence classification is structurally unstable on dense social networks with heavy-tailed IC distributions. This instability is irreducible by increasing simulation runs and reflects a property of the graph topology rather than simulation variance."

**3.4 Formulation dạng regression.** Cần nói rằng stability analysis dẫn tới quyết định dùng continuous regression trên log-transformed IC scores như principled prediction formulation. Cite Spearman stability (`0.827` ở 1200 runs) để cho thấy rank ordering vẫn ổn định ngay cả khi binary top-k membership không ổn định. Cách viết phải thể hiện đây là positive design choice, không phải fallback.

### Mục 4 / Section 4 - GNN Surrogate Learning (2.0 trang)

**4.1 Experimental Setup (0.3 trang).** Trình bày ngắn gọn các thành phần cần có trong paper: Twitch MUSAE với `168,114` nodes và `6,797,557` edges, IC labeling trên `5,000` stratified nodes với `200` runs, split `80/20` degree-stratified dùng chung cho cả hai operationalization, GNN gồm 5 architectures với `hidden_dim=128`, `2` layers, `dropout=0.3`, `HuberLoss`, `200` epochs, `5` seeds; baseline stack gồm degree, PageRank, k-shell, one-hop, two-hop, LR variants và MLP.

**4.2 A0 Results — Structural Ceiling (0.5 trang).** Ở đây hãy trình bày bảng kết quả chính cho `A0`: degree đạt khoảng `0.826`, two-hop khoảng `0.804`, best GNN khoảng `0.82`. Phải nêu bootstrap CI. Nếu CI chứa 0 trong biên `±0.02`, có thể dùng nguyên câu tiếng Anh sau trong paper:
"Under A0, all GNN architectures achieve Spearman ρ statistically equivalent to degree centrality (bootstrap 95% CI: [X, Y]). This confirms that when IC transmission probability is a direct function of target degree, analytical baselines capture the dominant signal."

Đồng thời nêu thêm phát hiện `+0.099` về message passing (`GNN-raw-attr 0.534` so với `MLP 0.435`) như một bằng chứng rằng graph structure có signal, nhưng signal đó đã được precomputed centrality nắm gần hết.

**4.3 HSCC Results — Graph-Aware Regime (0.5 trang).** Hãy trình bày bảng baseline của `HSCC` lấy từ frozen regime-tagged outputs, không dùng ad hoc intermediate runs. Degree được kỳ vọng gần 0, `LR(life_time)` mạnh, `LR(views+life_time)` và `MLP(raw attrs)` mạnh hơn, và best GNN phải được so sánh với strongest flat baseline dưới matched feature access. Narrative chính là: dưới `HSCC`, degree không còn informative, nhưng source-side attributes đã là predictor mạnh; vì vậy mọi lợi thế của GNN phải được diễn giải như incremental value đến từ graph/community structure beyond flat baselines. Nếu frozen outputs cuối cho thấy margin nhỏ, cứ viết trung thực như vậy.

**Nếu GNN dùng pipeline `raw_attr` hiện tại của repo:** hãy mặc định `language` là có mặt whenever cột `language` tồn tại, vì code hiện tại tự mở rộng `lang_*` dummies vào `raw_attr`. Do đó, trừ khi team chủ động tắt và ghi lại quyết định đó trước final run, paper bắt buộc phải có fairness baselines `LR(views+life_time+language)` và `MLP(raw attrs+language)`. Nếu các fairness baselines này tiến rất gần best GNN, paper phải frame phần gain của GNN như residual message-passing component mà thôi.

**4.4 Contrast Analysis (0.4 trang).** Đây là phần cốt lõi về mặt intellectual contribution. Cần trả lời rõ: vì sao GNN thắng dưới `HSCC` nhưng không thắng dưới `A0`? Với `A0`, `R²(IC, degree) = 0.887`, tức degree giải thích gần `89%` variance; thêm attribute gần như không thêm signal đáng kể. Với `HSCC`, degree chỉ giải thích một phần rất nhỏ variance. IC score có thể được tách thành `phi(u)` (engagement velocity, flat attribute models có thể nắm) và `cross_community_fraction` (structural bridging, phải qua neighborhood aggregation mới recover tốt). `phi`, `lr_phi`, và `phi × cross_frac` chỉ nên dùng như diagnostic hoặc oracle-style interpretation tools trừ khi comparator policy được re-lock; comparator chính thức của HSCC trong MAPR paper vẫn là strongest flat baseline từ frozen fairness table với matched feature access.

**4.5 Runtime (0.3 trang).** Phần này nên giữ ngắn và thực dụng. Nêu `MC-IC labeling = 480s` cho `5,000` nodes, `GNN inference = 0.067s` cho `168,114` nodes, suy ra speedup khoảng `7,169x`. Cách frame đúng là: surrogate learning có giá trị thực tế vì nhanh hơn rất nhiều so với repeated MC simulation, bất kể GNN có vượt degree hay chỉ ngang degree.

### Mục 5 / Section 5 - Discussion and Limitations (0.5 trang)

**5.1 Khi nào GNN thực sự thêm giá trị?** Câu trả lời ở mức concept là: chỉ khi diffusion model mã hóa loại thông tin buộc phải đi qua neighborhood aggregation mới recover được, cụ thể là khi IC score phụ thuộc vào composition của neighborhood chứ không chỉ vào local degree. Đây là điều kiện cấu trúc của IC formula, không phải phẩm chất bẩm sinh của một GNN architecture nào.

**5.2 Limitations.** Phải nêu rõ ít nhất 4 ý: follower graph không phải observed diffusion; `A0` và `HSCC` chỉ là operationalization chứ không phải ground truth; `HSCC` là domain-informed formula mới nhưng chưa được validate như diffusion law thật; và `life_time` là một baseline predictor rất mạnh dưới `HSCC`, nên mọi claim về GNN đều phụ thuộc vào fairness của baseline feature access.

**5.3 Vì sao không học `p` trực tiếp từ dữ liệu?** Có thể chốt phần này bằng 1 câu tiếng Anh như sau:
"Learning edge-level transmission probabilities requires supervised cascade logs unavailable in this dataset; weighted cascade and HSCC provide principled zero-shot alternatives."

---

## Phần 3: Các paper nên đọc và cite

### Nhóm bắt buộc cite (core framework)

**Kempe, Kleinberg, and Tardos (2003).** "Maximizing the Spread of Influence through a Social Network." KDD. Đây là reference nền tảng cho mô hình IC và weighted cascade parameterization. Cần cite cho định nghĩa IC, tính NP-hard của influence maximization, và công thức weighted cascade `p(u,v) = 1/in-degree(v)`.

**Ling, Jiang, Wang, Thai, Xue, Song, Qiu, and Zhao (2023).** "Deep Graph Representation Learning and Optimization for Influence Maximization." ICML. Đây là paper DeepIM. Cần cite cho 2 việc: `(1)` weighted cascade experimental setup mà bạn đang bám theo, và `(2)` như một đại diện của learning-based influence maximization. Paper này giải bài toán seed-set optimization, khác task của bạn, nhưng diffusion setup thì tương thích.

**Hamilton, Ying, and Leskovec (2017).** "Inductive Representation Learning on Large Graphs." NeurIPS. Cite cho GraphSAGE architecture và inductive learning paradigm.

**Kipf and Welling (2017).** "Semi-Supervised Classification with Graph Convolutional Networks." ICLR. Cite cho GCN architecture và đặc biệt là symmetric normalization `D^{-1/2}AD^{-1/2}` có sự tương đồng cấu trúc với A2 sensitivity variant.

**Xu, Hu, Leskovec, and Jegelka (2019).** "How Powerful are Graph Neural Networks?" ICLR. Cite cho GIN architecture và lập luận WL-equivalent expressiveness.

**Veličković, Cucurull, Casanova, Romero, Liò, and Bengio (2018).** "Graph Attention Networks." ICLR. Cite cho attention-based aggregation và giả thuyết rằng attention có thể học weighting liên quan đến degree.

**Klicpera, Bojchevski, and Günnemann (2019).** "Predict Then Propagate: Graph Neural Networks Meet Personalized PageRank." ICLR. Cite cho APPNP architecture và Personalized PageRank propagation.

**Rozemberczki, Allen, and Sarkar (2021).** "Multi-Scale Attributed Node Embedding." Journal of Complex Networks. Đây là paper mô tả MUSAE Twitch dataset. Cần cite cho semantics của dataset, mutual-follow edges, và các node attributes như `views`, `life_time`, `language`, `dead_account`.

### Nhóm nên cite (giúp claim chắc hơn)

**Kitsak, Gallos, Havlin, Liljeros, Muchnik, Stanley, and Makse (2010).** "Identification of Influential Spreaders in Complex Networks." Nature Physics. Cite cho finding rằng k-shell coreness có thể dự báo spreading ability, từ đó biện minh vì sao structural centrality baselines là đối thủ mạnh dưới degree-coupled IC.

**Guille, Hacid, Favre, and Zighed (2013).** "Information Diffusion in Online Social Networks: A Survey." ACM SIGMOD Record. Nên cite Section 4 về khó khăn đánh giá khi không có behavioral ground truth; rất hợp cho construct validity discussion.

**Burt (1992).** "Structural Holes: The Social Structure of Competition." Harvard University Press. Cite cho structural holes theory dùng để biện minh về mặt domain cho thành phần cross-community amplification của HSCC.

**Benjamini and Hochberg (1995).** "Controlling the False Discovery Rate." Journal of the Royal Statistical Society Series B. Cite nếu bạn có dùng BH-FDR correction cho multiple testing.

**Blondel, Guillaume, Lambiotte, and Lefebvre (2008).** "Fast Unfolding of Communities in Large Networks." Journal of Statistical Mechanics. Cite cho Louvain community detection algorithm.

### Nhóm có thể cân nhắc cite nếu còn chỗ

**Chen, Wang, and Wang (2010).** "Scalable Influence Maximization for Prevalent Viral Marketing in Large-Scale Social Networks." KDD. Có thể cite cho PMIA và luận điểm influence decay theo hop distance.

**Kumar, Mallik, Khetarpal, and Panda (2022).** "Influence Maximization in Social Networks Using Graph Embedding and Graph Neural Network." Information Sciences. Dùng như một representative của hướng GNN-based influence methods huấn luyện trên IC-simulated labels.

**Aral and Walker (2012).** "Identifying Influential and Susceptible Members of Social Networks." Science. Có thể cite để hỗ trợ luận điểm social ties có liên quan tới influence pathways.

**Grover and Leskovec (2016).** "node2vec: Scalable Feature Learning for Networks." KDD. Cite nếu Node2Vec xuất hiện như baseline.

**Lü, Chen, Ren, Zhang, Zhang, and Zhou (2016).** "Vital Nodes Identification in Complex Networks." Physics Reports. Cite cho survey tổng quan về node importance metrics.

---

## Phần 4: Các việc quan trọng phải hoàn tất trong 9 ngày tới

### Ngày 21 (hôm nay) - khóa và xác minh

- **Người 1 / Person 1:** xác minh rằng `regression_targets_hscc_refined.parquet` tồn tại và đọc được. Nếu chưa có thì regenerate ngay từ HSCC IC scores hiện có. Đồng thời kiểm tra `experiment_registry.md` để chắc rằng formula lock của HSCC-refined đang khớp với codebase hiện tại; nếu thiếu hoặc lệch thì sửa trước final HSCC run.

- **Người 2 / Person 2:** xác nhận rằng `community_features.parquet` cover 100% active nodes với cả `community_id` lẫn `cross_community_edge_fraction`. Đây là upstream dependency bắt buộc để diễn giải HSCC.

- **Người 3 / Person 3:** xác minh rằng evaluation runners hiện tại đã emit `label_regime` cho cả baseline outputs lẫn surrogate outputs. Nếu các CSV đang có bị stale, bị trộn regime, hoặc thiếu HSCC rows thì phải regenerate lại frozen regime-tagged outputs.

### Ngày 22-23 - baseline fairness (HSCC) + train GNN cho cả hai regime

- **Người 3 / Person 3:** chạy HSCC flat baselines trước khi chạy GNN: `LR(life_time)`, `LR(views+life_time)`, `LR(degree+views+life_time)`, `MLP(views, life_time)`. Trong repo hiện tại, nếu dùng `raw_attr` và có cột `language` thì language dummies được thêm tự động; vì vậy `LR(views+life_time+language)` và `MLP(views+life_time+language)` là fairness baselines bắt buộc trừ khi team chủ động tắt `language` và ghi lại quyết định đó trước frozen run. Hãy lấy strongest flat baseline từ matched-access table này làm HSCC bootstrap comparator. `phi` và `lr_phi` chỉ nên xem như diagnostic interpretation rows, không phải default paper comparator, trừ khi plans được re-lock lại rõ ràng.

- **Người 3 / Person 3:** đồng thời train GNN architectures trên cả A0 labels và HSCC labels. Với `A0`, dùng raw_attr features `views_log`, `views_per_day`, `life_time`. Với `HSCC`, giả định cùng pipeline raw_attr và nhớ rằng `language` sẽ tự được thêm vào nếu có, trừ khi bị disable có chủ đích và có ghi lại. Tổng cộng `5 architectures × 5 seeds × 2 regimes = 50` runs. Với tốc độ khoảng `23s/run`, tổng thời gian trên GPU vào khoảng `20 phút`.

### Ngày 24 - bootstrap CI và khóa kết quả

- **Người 3 / Person 3:** chạy bootstrap CI cho cả hai regime. Với `A0`: so sánh best GNN với degree. Với `HSCC`: so sánh best GNN với strongest flat baseline từ frozen matched-access fairness table. Record cả Spearman lẫn NDCG CIs. Chỉ sau khi các frozen artifacts này tồn tại thì paper mới được phép dùng claim wording gắn với `gnn_vs_degree_bootstrap_ci_a0.json` hoặc `gnn_vs_baseline_bootstrap_ci_hscc.json`.

Sau ngày 24, toàn bộ experimental results phải được đóng băng. Không mở thêm experiment mới, không đổi parameter, không thêm IC formulation mới.

### Ngày 25-27 - viết paper

Ngày 25: draft Mục 1-2 / Sections 1-2 (`Introduction + Background`). Hai phần này không phụ thuộc vào số liệu cuối.

Ngày 26: draft Mục 3-4 / Sections 3-4 (`MC-IC Metric + GNN Results`) với số liệu thật từ frozen result tables. Đồng thời tạo Hình 1 / Figure 1 (pipeline diagram) và Hình 2 / Figure 2 (two-panel results cho contrast giữa `A0` và `HSCC`).

Ngày 27: draft Mục 5 / Section 5 (`Discussion + Limitations`), hoàn tất tất cả bảng, rồi làm internal review pass.

### Ngày 28-29 - polish và format

Ngày 28: xử lý IEEE formatting, kiểm tra double-blind compliance (xóa author names, affiliations, acknowledgments khỏi PDF), kiểm tra figure readability ở chế độ grayscale, và format reference list.

Ngày 29: cả team đọc lại lần cuối. Sửa mọi claim không được frozen results support. Đảm bảo abstract phản ánh đúng findings thực tế chứ không phải điều team mong sẽ thấy.

### Ngày 30 - nộp

Nộp bài. Không sửa kết quả hay claim vào phút chót.

---

## Phần 5: Ba claim paper bắt buộc phải chống đỡ được bằng evidence

Mỗi claim trong paper phải map được tới một frozen artifact cụ thể. Các mapping dưới đây là evidence requirements cho bản paper cuối, không phải assumption được viết trước khi outputs được khóa.

### Claim 1
**Có thể dùng nguyên tiếng Anh trong paper:** Binary influence classification is structurally unstable on dense social networks.

Evidence bắt buộc: `stability_explanation.json` (`pct_communities_spanning_boundary = 0.842`, `mean_gap_to_noise` gần 0), Jaccard stability sweep (`0.307 -> 0.682`, không bao giờ chạm `0.85`), và Spearman stability (`0.685 -> 0.827`, tức rank ordering ổn định dần trong khi binary membership thì không). Các artifacts này hiện đã có trong repo, nhưng khi viết paper vẫn phải cite đúng frozen copies được dùng cho bản nộp.

### Claim 2
**Có thể dùng nguyên tiếng Anh trong paper:** Under degree-coupled IC (A0), GNN is practically equivalent to degree centrality.

Evidence frozen bắt buộc: `gnn_vs_degree_bootstrap_ci_a0.json` (bootstrap CI chứa 0 trong pre-registered equivalence band), một frozen A0 results table cho thấy degree và best GNN trên cùng split, cùng diagnostic về message-passing contribution như `GNN-raw-attr vs MLP-raw-attr`. Không viết claim cuối với số cụ thể cho đến khi regime-tagged A0 tables và bootstrap JSON đã freeze.

### Claim 3
**Có thể dùng nguyên tiếng Anh trong paper:** Under attribute-community IC (HSCC), GNN outperforms flat baselines by capturing cross-community engagement structure.

Evidence frozen bắt buộc: `gnn_vs_baseline_bootstrap_ci_hscc.json` (bootstrap CI cho GNN so với strongest flat baseline lấy từ frozen HSCC fairness table dưới matched feature access), một frozen HSCC results table cho baseline stack và best GNN, cùng oracle-style decomposition (`phi`, `lr_phi`, `phi × cross_frac`) chỉ dùng để diễn giải signal decomposition. Không được coi `phi` hay `lr_phi` là comparator chính của paper nếu comparator policy chưa được re-lock trong plans.

Nếu một trong ba claim này không được kết quả thực tế support, paper phải được viết lại để phản ánh đúng dữ liệu. Contrast story giữa Claim 2 và Claim 3 vẫn publishable ngay cả khi margin của Claim 3 chỉ nhỏ, vì chính contrast đó mới là finding quan trọng.

---

## Phần 6: Reviewer sẽ hỏi gì và nên trả lời ra sao

**Reviewer hỏi: "Why not use real cascade data?"** Cách trả lời nên là: Twitch dataset không chứa behavioral cascade logs. MC-IC chỉ đóng vai trò như một simulation-based proxy có cơ sở phương pháp luận từ Kempe et al. (2003) và Ling et al. (2023). Mọi findings trong paper đều phải được hiểu là properties của simulation, không phải measurement của real influence. Điểm này phải được nói rõ ở Mục 3.1 / Section 3.1 và phần Limitations.

**Reviewer hỏi: "Why is HSCC a good diffusion model?"** Cách trả lời nên là: paper không claim `HSCC` là Twitch diffusion model thật. `HSCC` chỉ là một domain-informed operationalization được thiết kế để kiểm tra xem neighborhood composition có tạo thêm learnable value cho GNN hay không. Engagement velocity là proxy hợp lý cho content production rate, còn cross-community amplification được biện minh bởi structural holes theory của Burt (1992). Đóng góp của paper là comparative finding, không phải claim về realism của HSCC.

**Reviewer hỏi: "Why not compare against published influence maximization methods like DeepIM?"** Cách trả lời nên là: DeepIM và các phương pháp tương tự giải một bài toán khác, đó là chọn seed set tối ưu để tối đa total cascade reach. Task của paper này là node-level IC score regression, tức dự đoán influence potential của từng node riêng lẻ, không phải tối ưu một seed set tập thể. Vì vậy comparison trực tiếp không phù hợp, dù diffusion setup của paper có theo weighted cascade tradition giống DeepIM.

**Reviewer hỏi: "The Twitch dataset is from 2021. Is it still relevant?"** Cách trả lời nên là: MUSAE Twitch vẫn là benchmark chuẩn cho graph-level analysis và vẫn được cite trong nhiều công trình gần đây. Đóng góp của paper này là methodological - khi nào GNN thêm giá trị cho IC approximation - chứ không phải một kết luận đặc thù cho nền tảng Twitch ở thời điểm 2021.

**Reviewer hỏi: "Life_time dominates HSCC labels. Isn't GNN just learning life_time?"** Cách trả lời nên là: vấn đề này được xử lý bằng baseline fairness dưới matched feature access. Nếu pipeline `raw_attr` hiện tại có `language`, thì HSCC fairness table cũng phải có các LR/MLP baselines dùng `language` trước khi được phép claim lợi thế cho GNN. Chỉ phần residual margin sau matched baselines mới được diễn giải là graph/community message-passing gain. Khi đó, `phi`, `lr_phi`, và `phi × cross_frac` mới được dùng để giải thích tại sao residual structural component này tồn tại.

---

Dưới đây là hướng dẫn của mình với vai trò **supervisor SNA + AI/ML**, dựa trên:

- `MAPR2026_Implementation_Plan_v3.md`
- `docs/MAPR2026_v3_team_parallel_coding_plan.md`
- và một supervisor guide trước đó

Mình sẽ đi theo tinh thần: **practical, defendable, nộp được MAPR**, không lan man.

---

# 1. Đánh giá tổng quan về 2 plan mới nhất

## Verdict
**Hai plan mới nhất đã đủ tốt để viết một paper defensible cho MAPR 2026**, với điều kiện:

1. **đóng băng scope ngay bây giờ**,
2. **không mở thêm operationalization mới ngoài A0 + HSCC**,
3. **baseline fairness dưới HSCC phải đầy đủ**,
4. **paper phải được viết như một comparative operationalization study**, không phải một paper “GNN luôn tốt hơn”.

## Paper mạnh nhất lúc này là gì?
Không phải:
- “GNN beats all baselines”
- cũng không phải “HSCC là true Twitch diffusion model”

Mà là:

> **“Giá trị của GNN surrogate phụ thuộc mạnh vào IC operationalization. Dưới A0, analytical baselines gần như trần. Dưới HSCC, graph message passing có thể thêm giá trị vượt quá các flat baselines.”**

Đây là một câu chuyện:
- **honest**
- **đủ mới cho MAPR**
- và **rất khó bị reviewer đánh gục** nếu viết đúng.

---

# 2. Paper này thực chất là paper gì?

Đây là điểm phải chốt trước khi viết.

## Định danh / Identity của paper
**Paper này là một bài empirical comparative study về operationalization của influence và surrogate learning.**

### Nó KHÔNG phải:
- paper về “real influence”
- paper về “novel GNN architecture”
- paper về “new IC theory”
- paper về “influence maximization”

### Nó LÀ:
- paper về **simulation-defined influence approximation**
- paper về **regime-dependent value of GNNs**
- paper về **khi nào analytical baselines đủ, khi nào graph learning bắt đầu hữu ích**

---

# 3. Cấu trúc narrative nên giữ chặt

Mình đồng ý mạnh với hướng đi của supervisor guide, nhưng sẽ làm nó sắc hơn và practical hơn cho MAPR.

## Narrative trục chính
### Bước 1
**MC-IC là một operational metric hợp lý nhưng chỉ là proxy**

### Bước 2
**Binary top-k labels không ổn định → regression/ranking là formulation đúng**

### Bước 3
**Không có một IC operationalization nào “tự nhiên đúng” — operationalization choice quyết định learnability**

### Bước 4
**GNN không có giá trị cố hữu; giá trị của nó phụ thuộc regime của label**
- A0: structural ceiling, degree gần tối ưu
- HSCC: GNN có thể học thêm cross-community signal beyond flat baselines

---

# 4. Hướng dẫn viết paper theo từng mục / section

Mình sẽ cho luôn hướng viết cụ thể, gần như có thể dùng để drafting.

---

## Mục 1 / Section 1 - Introduction (0.5 trang)

## Mục tiêu
Trả lời 4 câu:
1. Bài toán là gì?
2. Vì sao khó?
3. Vì sao MC-IC cần surrogate?
4. Đóng góp cụ thể là gì?

## Cấu trúc đề xuất

### Đoạn 1 / Paragraph 1 - Problem
- identify influential users / power users trong static social graph
- thiếu behavioral cascade logs
- nên phải operationalize influence gián tiếp

### Đoạn 2 / Paragraph 2 - Tension
- MC-IC là proxy hợp lý nhưng đắt
- GNN là surrogate candidate
- nhưng chưa rõ khi nào GNN thực sự hơn các heuristic/baseline đơn giản

### Đoạn 3 / Paragraph 3 - Core idea
- compare two IC operationalizations:
  - A0: structural weighted cascade
  - HSCC: source-velocity + cross-community amplification
- show that operationalization choice changes the learning problem

### Contributions — 3 bullet là đẹp nhất
Mình khuyên viết như sau:

1. **We analyze Monte Carlo IC as a simulation-defined operational metric for influence potential on Twitch, and show that binary top-k labels are structurally unstable while continuous regression targets remain usable.**
2. **We compare two diffusion operationalizations, A0 and HSCC, and show that they induce qualitatively different approximation regimes: degree-dominated versus graph-aware attribute-community structure.**
3. **We benchmark analytical, flat, and GNN surrogates under both regimes, showing that GNN value is regime-dependent rather than universal, while retaining orders-of-magnitude speedups over repeated MC simulation.**

## Cần tránh
- “we identify real power users”
- “we propose a superior GNN”
- “we discover the correct Twitch diffusion model”

---

## Mục 2 / Section 2 - Background / Related Work (0.5–0.75 trang)

## Nên chia 3 cụm rất ngắn

### 2.1 Influence / IC / diffusion
- Kempe et al. (2003)
- weighted cascade setup
- DeepIM chỉ cite như example của learning-based IM dùng weighted cascade

### 2.2 Node importance / structural baselines
- degree / PageRank / k-shell
- Kitsak et al. (2010)
- optionally Lü et al. survey nếu còn reference budget

### 2.3 Graph surrogate / GNN
- GraphSAGE
- GCN
- GIN
- GAT
- APPNP
- surrogate / graph regression framing

## Lưu ý quan trọng
Không đi sâu influence maximization literature quá nhiều.  
Bài của bạn không optimize seed set; bài của bạn **approximate node-level IC scores**.

---

## Mục 3 / Section 3 - MC-IC as Operational Metric (đây là section quan trọng nhất)  
**~1.0–1.25 trang**

Mình đồng ý với supervisor guide: **Mục 3 / Section 3 phải được mở rộng hơn Mục 4 / Section 4 so với bản plan cũ**, vì đây mới là contribution methodological mạnh nhất.

## 3.1 Construct validity
Dùng gần nguyên văn đoạn bạn đã chuẩn bị:
- follower graph ≠ observed diffusion
- nhưng là structural substrate hợp lý trong absence of logs
- all findings are about the operationalization, not real influence

## 3.2 Operationalizations
Giới thiệu rất rõ:

### A0
\[
p(u,v)=1/\deg(v)
\]
- standard structural operationalization
- degree-coupled by design

### HSCC
\[
p(u,v)=\mathrm{clip}\left(\lambda \frac{\phi(u)}{\deg(u)}(1+\gamma \mathbf{1}[c_u\neq c_v]),0,p_{\max}\right)
\]
- domain-informed alternative
- source engagement velocity + community bridging
- not “true Twitch diffusion”, only a comparative operationalization

## 3.3 Discriminativeness
Ở đây nên có **1 bảng ngắn**:

| Regime | Mean reach | Median | CV | Comment |
|---|---:|---:|---:|---|
| A0 | ... | ... | ... | heavy-tailed, structural |
| HSCC | ... | ... | ... | selective, non-degenerate |

### Key writing point
- A0: broad but degree-dominated
- HSCC: smaller mean reach nhưng vẫn discriminative
- I-A: chỉ mention 1 câu trong Discussion/Appendix như negative archive, đừng đưa vào main text dài

## 3.4 Stability and regression
Đây là phần bạn phải viết thật crisp:

- binary top-k unstable
- community boundary spanning
- gap-to-noise gần 0
- instability is structural, not just Monte Carlo noise
- therefore:
  - **continuous regression is the principled formulation**
  - binary labels are only secondary/provisional

### Câu rất nên có
> We treat regression not as a fallback but as the natural formulation for a simulation-derived continuous influence target.

## 3.5 Why not degree?
Đây là gap phải fill.

### Mình khuyên viết như sau:
- Under A0, degree is indeed a very strong approximation.
- But the degree-controlled (and ideally one-hop-controlled) variance analysis shows whether IC retains variance beyond local connectivity.
- Under HSCC, degree collapses as a predictor almost entirely.

=> như vậy bạn không cần overclaim rằng “IC always goes beyond degree”.  
Bạn viết đúng hơn:
> The extent to which IC goes beyond degree depends on the operationalization.

Đây là câu rất mạnh.

---

## Mục 4 / Section 4 - Surrogate Learning Across Operationalizations  
**~2.0 trang**

Đây là section kết quả chính, nhưng phải viết gọn và có logic.

---

## 4.1 Setup
Ngắn gọn:
- Twitch MUSAE
- 168k nodes
- 5k labeled nodes
- transductive split
- 5 seeds
- metrics: Spearman, NDCG@10, P@10
- runtime reported separately

---

## 4.2 A0 results — “structural ceiling”
Đây phải là subsection riêng và **không được xem là thất bại**.

### Thông điệp chính
- degree / two-hop are already near-optimal
- GNNs converge toward that ceiling
- bootstrap comparator = degree

### Nếu CI cho practical equivalence:
Câu chuẩn:
> Under A0, the best GNN architecture is practically equivalent to degree centrality under the pre-registered equivalence bound, indicating that the limiting factor is the diffusion operationalization rather than the GNN architecture.

### Cần đưa:
- degree
- one-hop
- two-hop
- best flat baseline
- 4–5 GNN architectures (raw_attr)
- maybe best ablation result

---

## 4.3 HSCC results — “graph-aware regime”
Đây là subsection main-claim.

### Thông điệp chính
- degree không còn là comparator đúng
- strongest flat baseline mới là comparator thật
- GNN có thể thêm giá trị nếu học được phần graph/community structure beyond raw attrs

### Bắt buộc:
Bảng HSCC phải có:
- LR(life_time)
- LR(views + life_time)
- LR(degree + views + life_time)
- MLP(raw attrs)
- nếu GNN dùng pipeline `raw_attr` hiện tại:
  - LR(... + language)
  - MLP(... + language)

**Lưu ý cho codebase hiện tại:** `raw_attr` currently auto-includes `language` dummies when the column exists. Vì vậy, nếu team không tắt rõ ràng và không ghi lại quyết định đó trước run cuối, hãy mặc định rằng fairness baselines với `language` là bắt buộc.

### Nếu GNN thắng strongest flat baseline:
Câu chuẩn:
> Under HSCC, GNN message passing significantly improves over the strongest flat baseline, suggesting that neighborhood structure contributes information not recoverable from node-level attributes alone.

### Nếu GNN chỉ xấp xỉ / hơn rất ít:
Câu chuẩn:
> Under HSCC, flat attribute models explain most of the source-driven component, while GNN contributes a smaller but measurable improvement attributable to graph-mediated community structure.

### Nếu GNN thua strongest flat baseline:
Vẫn viết được:
> Under HSCC, most of the predictive signal is already captured by source-side engagement attributes, with only limited incremental benefit from graph message passing.

=> paper vẫn ổn nếu contrast A0 vs HSCC còn mạnh.

**Comparator lock cho bản MAPR:** `phi`, `lr_phi`, và `phi × (1+cross_frac)` chỉ dùng để giải thích cơ chế hoặc làm oracle-style upper bound. Comparator chính trong main paper vẫn là **strongest flat baseline từ frozen HSCC fairness table với matched feature access**.

---

## 4.4 Contrast analysis
Đây là trái tim của paper.

### Cần giải thích rõ:
#### A0
- label is degree-coupled
- analytical baselines strong
- GNN near ceiling

#### HSCC
- label decomposes into:
  - source engagement term
  - graph/community amplification term
- flat baselines capture mostly source term
- GNN can only add value on the structural amplification part

### Đây là nơi dùng:
- `phi`
- `phi × (1+cross_frac)`
- nhưng chỉ như **ceiling / interpretation**, không phải main baseline trừ khi comparator policy được re-lock rõ ràng trong 2 plan

### Câu rất hay để dùng:
> The contrast between A0 and HSCC shows that surrogate learnability is not a property of the model alone; it is jointly determined by the diffusion operationalization and the information already recoverable by simple baselines.

---

## 4.5 Runtime
Giữ ngắn và sạch:

- MC-IC labeling cost
- GNN training cost
- GNN inference cost
- analytical baseline cost (near-zero inference)

### Cách diễn đạt quan trọng
- speedup **vs MC-IC**
- not vs degree

---

## Mục 5 / Section 5 - Discussion & Limitations (0.5 trang)

Đây là nơi bạn “khóa” reviewer.

## 5.1 When does GNN help?
Câu trả lời:
- not universally
- only when target depends on graph-mediated information not already captured by strong flat baselines

## 5.2 Limitations
Tối thiểu nên có 4 ý:

1. follower graph ≠ observed diffusion path
2. A0 and HSCC are operationalizations, not ground truth
3. HSCC is novel and domain-informed, not empirically validated diffusion law
4. transductive evaluation only
5. small mean reach under HSCC should be interpreted as selective diffusion, not broad viral spread

## 5.3 Why not learn p from data?
Giữ câu này:
- no supervised diffusion logs
- hence zero-shot operationalizations

---

# 5. Hình và bảng nên có gì

Do MAPR chỉ có 6 trang, đừng tham.

## Hình bắt buộc nên có
### Hình 1 / Figure 1
Pipeline diagram:
- graph
- A0 / HSCC
- MC-IC labels
- regression target
- baselines + GNN surrogates

### Hình 2 / Figure 2
**Two-panel results figure**
- trái: A0
- phải: HSCC
- bar chart / dot plot with CI
- line reference:
  - A0: degree
  - HSCC: strongest flat baseline

Đây là figure quan trọng nhất của paper.

## Bảng bắt buộc nên có
### Bảng 1 / Table 1
Dataset + operationalizations
- nodes, edges
- A0 formula
- HSCC formula
- mean/median/CV

### Bảng 2 / Table 2
A0 results (main subset)
- degree, one-hop, two-hop, MLP, GNNs

### Bảng 3 / Table 3
HSCC results
- LR(life_time), LR(views+life_time), LR(deg+views+lt), MLP, GNNs

### Bảng 4 / Table 4 (nếu còn chỗ)
Runtime mini-table

Nếu thiếu chỗ:
- merge runtime vào main results table dưới dạng cột cuối

---

# 6. Những claim nào được phép và không được phép

## Được phép
- “MC-IC is a principled operational metric”
- “A0 and HSCC induce different approximation regimes”
- “binary labels are structurally unstable”
- “under A0, analytical baselines are near-optimal”
- “under HSCC, GNNs may add value beyond flat baselines”
- “surrogate value depends on operationalization”

## Không được phép
- “we identify real power users”
- “HSCC is the true Twitch diffusion model”
- “GNN always outperforms baselines”
- “MC-IC is ground truth”
- “practical equivalence” nếu chưa pre-register bound và chưa có CI phù hợp
- “GNN is feature-agnostic” theo nghĩa tuyệt đối

### Thay “feature-agnostic” bằng:
> **without precomputed structural summaries**

---

# 7. Kịch bản viết paper theo kết quả cuối

Đây là phần cực quan trọng.

---

## Kịch bản 1 / Case 1 - Tốt nhất
### A0: GNN ≈ degree  
### HSCC: GNN > strongest flat baseline

=> paper rất mạnh cho MAPR

### Abstract nên viết:
- A0 = structural ceiling
- HSCC = graph-aware regime
- GNN advantage is regime-dependent
- 7000x speedup over MC-IC

---

## Kịch bản 2 / Case 2 - A0: GNN ≈ degree  
### HSCC: GNN ≈ strongest flat baseline
=> paper vẫn publishable

### Framing:
- operationalization contrast is main contribution
- HSCC shows that source-side attributes dominate much of the signal
- graph message passing adds limited but interpretable value
- still useful as fast surrogate

---

## Kịch bản 3 / Case 3 - A0: GNN < degree  
### HSCC: GNN < strong flat baseline
=> vẫn cứu được paper nếu viết đúng

### Main contribution chuyển thành:
1. binary instability finding
2. operationalization contrast
3. analytical/flat baselines often suffice depending on the regime
4. GNN is not universally superior

Đây vẫn là một paper empirical negative-result tốt cho MAPR.

---

# 8. Những paper cần đọc và cite

Mình chia thành 3 nhóm:
- **must cite in main paper**
- **should read / cite if space allows**
- **read only, not necessarily cite**

---

## 8.1 Must cite (main paper)
### Influence / IC / operationalization
1. **Kempe, Kleinberg, Tardos (2003)**  
   *Maximizing the Spread of Influence through a Social Network*  
   → foundational IC model

2. **Ling et al. (2023), DeepIM**  
   *Deep Graph Representation Learning and Optimization for Influence Maximization*  
   → weighted cascade setup, learning-based IM context

3. **Rozemberczki et al. (2021)**  
   MUSAE / Twitch dataset paper  
   → dataset semantics

### GNN architectures
4. **Hamilton et al. (2017)** — GraphSAGE  
5. **Kipf & Welling (2017)** — GCN  
6. **Xu et al. (2019)** — GIN  
7. **Veličković et al. (2018)** — GAT  
8. **Klicpera et al. (2019)** — APPNP  

### Evaluation / construct validity / node importance
9. **Kitsak et al. (2010)**  
   → k-shell spreaders

10. **Guille et al. (2013)**  
   → evaluation without behavioral logs

11. **Aral & Walker (2012)**  
   → social ties and influence pathways

### Stats / community
12. **Benjamini & Hochberg (1995)**  
13. **Blondel et al. (2008)** — Louvain

### Nếu cần HSCC justification
14. **Burt (1992)** — Structural Holes

> Nếu reference budget thực sự phải ≤12, mình khuyên giữ:
- Kempe
- DeepIM
- Rozemberczki
- GraphSAGE
- GCN
- GIN
- GAT
- APPNP
- Kitsak
- Guille
- Burt
- Blondel  
và có thể bỏ Aral & Walker hoặc BH nếu space quá chặt (BH có thể chỉ mention methodologically).

---

## 8.2 Should read (có thể cite nếu cần mở rộng)
1. **Lü et al. (2016), Physics Reports**  
   survey về vital nodes  
2. **Chen, Wang & Wang (2010), KDD**  
   PMIA / local influence approximation  
3. **Grover & Leskovec (2016)**  
   Node2Vec  
4. **Brody et al. (2022)**  
   GATv2 — chỉ nếu GATv2 còn xuất hiện ở appendix/future work  
5. **Hu et al. (2019)**  
   GINE — chỉ nếu C5 được mention  
6. **Fosdick et al. (2018)**  
   configuration model/nulls, nếu bạn còn nói về nulls ở extended version

---

## 8.3 Read but probably not cite in MAPR version
- GCNII
- HGT
- GraphGPS
- fairness/per-group analysis papers
- strong journal-only methodological papers

Chúng tốt cho extended version, không cần nhồi vào MAPR 6 trang.

---

# 9. Hướng dẫn viết Abstract rất cụ thể

Mình khuyên abstract theo template này:

### Câu 1 / Sentence 1
Problem + difficulty:
> Identifying influential users on static social networks without behavioral cascade logs requires simulation-based operationalizations of influence, but the learnability of such operationalizations remains poorly understood.

### Câu 2 / Sentence 2
Method:
> We study two Monte Carlo Independent Cascade (MC-IC) operationalizations on the Twitch social network: a structural weighted-cascade regime (A0) and a domain-informed source-community regime (HSCC).

### Câu 3 / Sentence 3
Stability/regression:
> We show that binary top-k influence labels are structurally unstable, motivating continuous regression on simulation-derived influence scores.

### Câu 4 / Sentence 4
Main contrast:
> Under A0, analytical structural baselines are already near-optimal, whereas under HSCC the strongest baselines shift to flat source-attribute models.

### Câu 5 / Sentence 5
GNN result:
> Across GraphSAGE, GCN, GIN, GAT, and APPNP, GNN surrogates provide regime-dependent value, ranging from practical equivalence to structural baselines under A0 to measurable gains over flat baselines under HSCC.

### Câu 6 / Sentence 6
Runtime:
> In all cases, learned surrogates provide orders-of-magnitude faster inference than repeated MC simulation.

---

# 10. Những việc mình yêu cầu team làm ngay trước khi viết

## Must-do ngay hôm nay
### Người 1 / Person 1
- verify HSCC regression target file còn đúng và readable
- verify registry entry for HSCC vẫn khớp formula lock hiện tại
- freeze config

### Người 2 / Person 2
- verify community feature coverage
- ensure diffusion proxies file clean
- provide quick note on language-community alignment if already available

### Người 3 / Person 3
- implement **all HSCC fairness baselines**
- rerun or refresh regime-tagged output tables nếu file hiện có còn stale / thiếu regime rows
- ensure bootstrap comparators are regime-specific and tied to frozen outputs only

---

# 11. Kết luận cuối của supervisor

## Kết luận của supervisor
**Hai file plan hiện tại đã đủ tốt để viết một paper defensible cho MAPR, nếu bạn giữ đúng đường `A0 + HSCC` và không mở thêm scope.**

### Điểm mạnh nhất của paper:
- stability finding
- regime contrast
- honest demonstration rằng GNN value is conditional, not universal
- runtime story sạch

### Điểm dễ bị reviewer tấn công nhất:
- HSCC fairness baselines chưa đủ
- HSCC có nguy cơ bị xem là engineered nếu không frame đúng
- claim về GNN phải map đúng với bootstrap của từng regime

### Nếu sửa 3 điểm này tốt:
1. baseline fairness for HSCC
2. regime-specific comparator logic
3. tight writing with no extra branches

=> **paper hoàn toàn có thể defend được ở MAPR**.
