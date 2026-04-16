## Phân tích RQ1 (H1) chi tiết — Dành cho sinh viên CS/AI/ML

**Ghi chú về phạm vi số liệu (để tránh hiểu nhầm):** Các con số trong file này được trích từ artifacts trong `outputs/` và chủ yếu được tính trên **tập labeled nodes (n=5,000)** dùng để gán nhãn IC (không phải toàn bộ 168,114 nodes). Khi mình nói “top-10%”, mặc định là **top-10% trong 5,000 labeled nodes** trừ khi ghi rõ khác.

### Bộ Research Questions v3 (để map với Hypotheses)

1. **RQ1 (H1):** IC operationalization có tạo được ranking influence đủ phân biệt và đủ ổn định để dùng làm surrogate target không?
2. **RQ2 (H2):** Views (popularity) đồng thuận tới mức nào với IC-based influence ranking?
3. **RQ2b (H3):** Tương quan giữa IC với views, centrality, one-hop, two-hop là gì (global và theo regime)?
4. **RQ3 (H4):** GNN surrogate có xấp xỉ IC tốt hơn cheap proxies không, và lợi ích tính toán là bao nhiêu?
5. **RQ3b (H5):** Node type nào khó dự đoán nhất bằng mô hình rẻ?
6. **RQ4 (H6):** Đặc trưng cấu trúc nào phân biệt nhóm có rank views và rank IC bất đồng?

---

### Bài toán gốc: Chúng ta đang cố làm gì?

Mục tiêu của project là dự đoán ai có ảnh hưởng nhất trên mạng Twitch. Để huấn luyện model, ta cần **nhãn (labels)** — tức là phải biết "ground truth: node này ảnh hưởng bao nhiêu."

Nhãn này được tạo ra bằng **IC simulation (Information Cascade)**: giả lập quá trình tin tức lan truyền qua mạng. Chạy simulation → đếm xem node đó kích hoạt được bao nhiêu node khác → đó là "điểm ảnh hưởng" của nó.

**Vấn đề:** IC simulation là **stochastic** — mỗi lần chạy dùng random seed khác nhau, kết quả sẽ khác nhau. Câu hỏi RQ1 đặt ra là: _nhãn IC này có đủ ổn định và đủ phân biệt để dùng làm learning target không?_

---

### Tại sao IC không deterministic?

Trong weighted-cascade IC, mỗi cạnh (u → v) có xác suất truyền tin `p(u,v)`. Khi chạy simulation từ một node nguồn:

```
Với mỗi cạnh (u → v):
    tung đồng xu với xác suất p(u,v)
    nếu rơi "heads" → v bị kích hoạt và tiếp tục lan truyền
    nếu rơi "tails" → lan truyền dừng tại đây
```

Vì đây là quá trình xác suất, cùng một node chạy 100 lần sẽ cho 100 kết quả reach khác nhau. Ta lấy **trung bình** nhiều lần chạy để ước lượng ảnh hưởng thực sự.

Câu hỏi kỹ thuật: cần chạy **bao nhiêu lần** để kết quả ổn định?

---

### Hai cách đo stability — và tại sao chúng cho kết quả khác nhau

Project đo stability bằng hai metric:

**Metric 1 — Spearman correlation** (đo thứ hạng tổng thể):
Chạy simulation với seed A và seed B → lấy IC score của toàn bộ nodes → tính Spearman giữa hai bảng xếp hạng.

**Metric 2 — Jaccard similarity** (đo sự trùng khớp của top-10% list):
Chạy seed A → lấy danh sách top 10% nodes. Chạy seed B → lấy danh sách top 10% khác. Tính Jaccard = |A ∩ B| / |A ∪ B|.

Kết quả thực nghiệm (stability sweep từ 150 đến 1200 runs):

| Số runs        | Spearman  | Jaccard@10% |
| -------------- | --------- | ----------- |
| 150            | 0.685     | 0.307       |
| 1200           | **0.827** | **0.682**   |
| Ngưỡng cần đạt | —         | **0.85**    |

**Hai số đi theo hai hướng khác nhau.** Spearman tăng nhanh và đạt 0.827 ở 1200 runs — khá tốt. Nhưng Jaccard chỉ đạt 0.682 dù tăng gấp 8 lần số runs, vẫn xa ngưỡng 0.85.

Tại sao lại như vậy?

---

### Hiệu ứng "ranh giới lớp" — lý do Jaccard và Spearman diverge

Hãy dùng một analogy quen thuộc: **điểm thi và xếp loại.**

Giả sử lớp học có 100 sinh viên. Điểm của họ phân phối từ 0 đến 10. Bạn chạy "simulation" hai lần (ví dụ hai đợt thi khác nhau) và nhận được kết quả hơi khác nhau:

- **Spearman cao** vì: sinh viên giỏi vẫn giỏi, sinh viên kém vẫn kém. Thứ tự tương đối của 100 người không thay đổi nhiều.
- **Jaccard thấp** vì: những sinh viên nằm đúng ở ranh giới top-10 (vị trí thứ 9, 10, 11, 12) sẽ bị đẩy qua đẩy lại giữa hai lần thi chỉ vì điểm số chênh nhau 0.1–0.2.

Trong IC, vấn đề còn tệ hơn vì:

```
stability_explanation.json:
    84.2% of communities span the top-10% boundary
    mean gap-to-noise = 0.00239
```

**84.2% communities** có thành viên nằm cả hai phía ranh giới top-10%. Tức là ranh giới không cắt "sạch" theo community — nó xuyên qua giữa các nhóm nodes có IC score gần nhau.

**Gap-to-noise = 0.00239** ở đây là **tỷ số** (dimensionless), được tính theo dạng:

```
gap_to_noise = gap / (local_std / sqrt(n_runs))
```

Nó cho biết “khoảng cách giữa node thứ k và (k+1)” ở boundary nhỏ tới mức nào so với **standard error** quanh boundary. Giá trị 0.00239 nghĩa là boundary gap **nhỏ hơn rất nhiều** so với nhiễu ước lượng (xấp xỉ chỉ ~0.24% standard error) → top-10% list rất dễ bị “đảo chỗ” bởi noise.

Hệ quả: ngay cả khi tăng runs lên 1200, vẫn có hàng trăm nodes "ngồi đúng trên ranh giới" và không thể phân loại ổn định.

---

### Structural cause — vì sao đây là đặc tính của graph, không phải lỗi sampling

Có thể bạn sẽ hỏi: _"Sao không chạy thêm runs nữa? 10,000 runs thì Jaccard có đạt 0.85 không?"_

Từ các diagnostics hiện có, câu trả lời hợp lý là: **khó có khả năng** đạt 0.85 chỉ bằng cách tăng `n_runs` trong cùng formulation, vì nguyên nhân chính nằm ở **boundary đặt trong vùng mật độ rất dày** (rất nhiều nodes có IC gần nhau) — một thuộc tính gắn với cấu trúc graph và phân phối IC score.

```
pivot_decision_report.json:
    community-set Jaccard = 0.842 (cao)
    max estimated Jaccard across all thresholds = 0.657 (thấp)
```

**Community-set Jaccard = 0.842** cao — nghĩa là nếu bạn hỏi _"community nào nằm trong top ảnh hưởng?"_, câu trả lời ổn định. Nhưng nếu hỏi _"node cụ thể nào trong top 10%?"_, câu trả lời bất ổn.

Đây là đặc tính thường gặp ở **power-law networks** như mạng xã hội: phân phối IC score lệch mạnh — một vài hub có IC cực cao, còn lại co cụm dày ở vùng thấp/trung bình. Khi **boundary (top-k)** rơi vào vùng mật độ dày, binary labeling sẽ khó ổn định.

Lưu ý: `max estimated Jaccard = 0.657` là **ước lượng** từ protocol threshold sweep (dùng mean/std của IC để ước lượng Jaccard kỳ vọng theo CLT), không phải “định lý toán học” cho mọi `n_runs`. Nhưng nó là bằng chứng thực nghiệm mạnh rằng “tăng thêm runs” sẽ gặp **diminishing returns** nếu vẫn giữ cùng IC model + cùng cách cắt top-k.

---

### cv_score — tín hiệu thứ hai xác nhận vấn đề

```
ic_pilot_diagnostics.json (pilot: 200 nodes, 50 runs):
    cv_score = 0.211
    cv_noise_threshold = 0.5
    cv_noise_count = 166/200 nodes
```

Ở đây protocol dùng hai tín hiệu liên quan CV:

1. `cv_noise_count` (166/200 nodes) cho thấy **đa số nodes** có CV per-node vượt ngưỡng 0.5 ở pilot → IC reach của nhiều nodes còn rất nhiễu ở ngân sách pilot.
2. `cv_score = 0.211` là một **diagnostic scalar** được dùng trong feasibility protocol (tham chiếu thêm `pivot_decision_report.json`, nơi target heuristic là `cv_score > 0.3`). Giá trị 0.211 gợi ý dynamics đang ở regime “near-critical-moderate”, chưa phải “adequate-spread” theo tiêu chuẩn gate.

Điều này không có nghĩa là IC vô dụng. Nó có nghĩa là: nếu dùng IC score trực tiếp làm **binary label** (top-10% vs không), thì variance của estimate sẽ liên tục làm nodes nhảy qua nhảy lại ranh giới. Nhưng nếu dùng làm **continuous regression target**, variance là noise bình thường trong bài toán regression — model có thể học được trend dù từng điểm có nhiễu.

---

### Quyết định Pivot — Regression thay vì Classification

Kết quả của RQ1 dẫn đến quyết định thiết kế quan trọng nhất của project:

```
pivot_decision_report.json:
    VERDICT: PIVOT_CONFIRMED
    Option B: continuous regression on log-transformed IC scores
```

**Thay vì:** Dự đoán binary label "node này có trong top 10% không?" (Classification)

**Chuyển sang:** Dự đoán continuous score "node này có IC score bao nhiêu?" (Regression)

**Lý do kỹ thuật:**

|                 | Classification top-10%                       | Regression IC score                     |
| --------------- | -------------------------------------------- | --------------------------------------- |
| Stability       | Jaccard 0.682 ở 1200 runs, fail 0.85         | Spearman 0.827 ở 1200 runs, acceptable  |
| Sensitivity     | Rất nhạy với boundary noise                  | Robust với noise vì predict trend       |
| Metric đánh giá | Accuracy/F1 bị dominated bởi boundary cases  | Spearman/NDCG đo ranking quality        |
| Defensibility   | Reviewer có thể attack: "label không stable" | Reviewer khó attack: rank signal stable |

Đây là một pattern phổ biến trong ML: khi boundary của binary classification không ổn định do noise hoặc inherent uncertainty, **calibration uncertainty bằng continuous target thường tốt hơn** là cố gắng làm sắc nét ranh giới nhân tạo.

---

### Tóm tắt RQ1 theo một sơ đồ logic

```
IC simulation stochastic
        ↓
Chạy nhiều seeds → kết quả khác nhau
        ↓
Đo stability theo 2 chiều:
    Spearman (thứ hạng tổng thể): 0.685 → 0.827  ← ổn định
    Jaccard (top-10% list):       0.307 → 0.682  ← không ổn định
        ↓
Tại sao Jaccard fail?
    84.2% communities span top-10% boundary
    mean gap-to-noise = 0.00239  → structural cause, không phải sampling error
        ↓
Nếu tăng runs mãi có fix được không?
    Không — max estimated Jaccard = 0.657 < 0.7 (theoretical ceiling thấp)
        ↓
Kết luận: binary classification top-10% là formulation sai
        ↓
Pivot: continuous regression on IC score
    → Spearman stable → ranking quality metric NDCG/Spearman defensible
```

---

### Điều RQ1 KHÔNG nói

Một điểm dễ hiểu sai: RQ1 **không nói** "IC simulation vô dụng" hay "nhãn IC sai." Nó nói: _"IC tạo tín hiệu ranking có ý nghĩa (Spearman 0.827), nhưng binary top-10% không phải cách đóng gói tín hiệu đó một cách ổn định. Dùng continuous score thì tín hiệu vẫn ở đó và stable hơn."_

Đây là lý do tại sao toàn bộ pipeline đánh giá sau đó dùng Spearman và NDCG (continuous ranking metrics), không phải Accuracy hay F1 (binary metrics).

---

## Phân tích RQ2 (H2) và RQ2b (H3) — Dành cho sinh viên CS/AI/ML

---

### Bối cảnh: Hai cách đo "ảnh hưởng"

Project đang có **hai tín hiệu** để đo mức độ ảnh hưởng của một streamer:

- **Views** — tín hiệu _popularity_: người dùng xem nhiều = nổi tiếng. Dễ thu thập, không cần tính toán.
- **IC score** — tín hiệu _diffusion potential_: mô phỏng xem thông tin lan truyền bao xa nếu xuất phát từ node đó. Tốn kém, nhưng phản ánh cấu trúc mạng thực sự.

**RQ2 hỏi:** Hai tín hiệu này đồng ý với nhau đến mức nào? Khi nào chúng lệch nhau?

---

## RQ2 — Views vs IC divergence

### Kết quả cơ bản: Tương quan yếu, nhưng không phải random

```
Spearman(views, IC score) = 0.469
```

Spearman 0.469 là **tương quan đơn điệu mức vừa (moderate monotonic association)**: views và IC có liên quan, nhưng còn xa mới là “một cái thay thế cái kia”. Để so sánh:

```
Spearman(degree, IC score) = 0.826
Spearman(views,  IC score) = 0.469   ← yếu hơn gần gấp đôi
```

Degree (số lượng kết nối) đã giải thích IC tốt hơn views rất nhiều. Điều này cho thấy IC đang đo "vị trí trong mạng", không phải "độ nổi tiếng theo lượt xem."

---

### Framework Typology — Tư duy như một confusion matrix

Cách nhóm nghiên cứu phân tích divergence là chia nodes vào 4 nhóm dựa trên hai ngưỡng (top-10% IC và top-10% views):

```
                    IC score
                  Thấp    |    Cao
               ─────────────────────
Views  Cao  |  Overrated  |   True   |
       Thấp |    Non      |  Hidden  |
               ─────────────────────
```

Nếu bạn quen với confusion matrix trong ML, đây chính xác là cùng một cấu trúc — chỉ thay "predicted" và "actual" bằng "views rank" và "IC rank":

| Nhóm          | Ý nghĩa                                                     | Số nodes | %     |
| ------------- | ----------------------------------------------------------- | -------- | ----- |
| **Non**       | Thấp cả hai — "người dùng bình thường"                      | 4,215    | 84.3% |
| **Hidden**    | IC cao, views thấp — "influencer ẩn"                        | 285      | 5.7%  |
| **Overrated** | Views cao, IC thấp — "nổi tiếng nhưng không lan truyền tốt" | 285      | 5.7%  |
| **True**      | Cao cả hai — "influencer thực sự"                           | 215      | 4.3%  |

**Hidden và Overrated là hai nhóm nghiên cứu chính** vì chúng đại diện cho divergence — nơi views và IC "không đồng ý" với nhau rõ rệt nhất.

---

### Permutation Null — Chứng minh divergence không phải ngẫu nhiên

Đây là phần kỹ thuật quan trọng nhất của RQ2. Câu hỏi cần trả lời: _"285 Hidden nodes này có xuất hiện do ngẫu nhiên không, hay nó phản ánh cấu trúc thực sự của mạng?"_

**Cách test (đúng theo artifacts hiện có):**

- `views_permutation_null_summary.json`: giữ IC score cố định, **permute views** giữa các nodes (tức giả sử views không liên quan IC), rồi rebuild typology và đo divergence.
- `ic_permutation_null_summary.json`: giữ views cố định, **permute IC score** giữa các nodes, rồi rebuild typology.

Cả hai hướng permute đều nhằm kiểm tra cùng một null hypothesis:

```
Null hypothesis: IC và views hoàn toàn độc lập (random permutation)
```

Kết quả:

```
Real Hidden count:  285
Null mean:          449.92
Null std:           ~(tính từ 200 permutations)

Real agreement rate: 0.886
Null mean:           0.820
p-value:             0.00498 (minimum achievable với 200 perms)
```

**Cách đọc đúng** (điểm dễ nhầm):

Test này chứng minh rằng **agreement giữa views và IC cao hơn ngẫu nhiên** (real 0.886 > null 0.820). Dưới null hypothesis (permute một trong hai tín hiệu), bạn kỳ vọng sẽ có ~450 Hidden nodes — nhưng thực tế chỉ có 285. Điều này có nghĩa là:

- Views và IC **không độc lập** — chúng chia sẻ một phần signal chung (cả hai đều tương quan với degree)
- Nhưng vẫn còn 285 Hidden nodes — tức là **vẫn có divergence có cấu trúc thực sự**, không phải artifact của cắt ngưỡng ngẫu nhiên

Hình dung như sau:

```
Nếu IC = random shuffle:  ~450 Hidden (nhiều divergence → views và IC hoàn toàn khác nhau)
Thực tế:                   285 Hidden (ít hơn random → hai tín hiệu có correlation)
Kết luận:                  Divergence THỰC SỰ tồn tại, nhưng ít hơn worst case
                           và có cấu trúc (không random)
```

---

## RQ2b — Ma trận tương quan toàn bộ metrics

### 8 metrics, 1 bức tranh

RQ2b mở rộng câu hỏi: không chỉ so views vs IC mà còn so **tất cả các metrics** với nhau.

```
Metrics: ic_score_mean, views, degree, pagerank, kshell,
         betweenness_approx, one_hop_spread, two_hop_spread
```

Kết quả từ metric_correlation_matrix.json (Spearman, 5,000 nodes, BH-corrected):

**IC score tương quan với các metrics khác:**

```
degree    → IC:  0.828   ████████░░  mạnh
pagerank  → IC:  0.830   ████████░░  mạnh
kshell    → IC:  0.816   ████████░░  mạnh
two_hop   → IC:  0.815   ████████░░  mạnh
one_hop   → IC:  0.717   ███████░░░  khá mạnh
views     → IC:  0.469   ████░░░░░░  yếu
```

**Tương quan giữa các structural metrics với nhau:**

```
degree   ↔ kshell:   0.993   ████████████  gần như IDENTICAL
pagerank ↔ two_hop:  0.986   ████████████  gần như IDENTICAL
degree   ↔ pagerank: ~0.97   ████████████  rất cao
```

---

### Phát hiện quan trọng: Multicollinearity cực cao

Degree, kshell, pagerank, two_hop có tương quan rất cao với nhau (ví dụ: degree↔kshell = 0.993; pagerank↔two_hop = 0.986). Trong ML, khi các features có tương quan > 0.9 thì chúng thường đang đo **cùng một underlying factor**. Đây gọi là **multicollinearity**.

Hình dung trực quan:

```
Tất cả structural metrics ≈ "Tín hiệu X" (một underlying factor duy nhất)
    degree   = X + noise nhỏ
    kshell   = X + noise nhỏ
    pagerank = X + noise nhỏ
    two_hop  = X + noise nhỏ

IC score = f(X) + residual       → correlation ~0.82 với X
views    = g(X) + lớn residual   → correlation ~0.47 với X
```

Ghi chú thêm (để hợp với H3): `one_hop_spread` cũng tương quan dương với IC (0.717), nhưng thấp hơn rõ so với cluster centrality (~0.82). Điều này ủng hộ ý tưởng “one-hop có ích nhưng không thay thế hoàn toàn IC”, còn `two_hop_spread` tiệm cận cluster hơn.

Điều này giải thích một lúc hai thứ:

1. **Tại sao baselines không thua nhau nhiều** (degree 0.826, pagerank 0.824, kshell 0.816): vì chúng đang đo cùng một underlying factor.
2. **Tại sao GNN không vượt baselines**: GNN được cấp centrality features — tức cũng là signal X — nên không có thông tin mới để học thêm.

---

### Tại sao views lại "lệch" khỏi cluster structural?

Views có Spearman 0.469 với IC, trong khi cả cluster structural metrics đều ~0.82. Điều này có nghĩa là views chứa **một thành phần không có trong structural metrics** — và thành phần này không giúp dự đoán IC.

Intuition: Một streamer có thể có ít kết nối trong mạng Twitch (degree thấp → IC thấp) nhưng vẫn có hàng triệu views nhờ nội dung viral, được recommend bởi thuật toán, hoặc crossover từ nền tảng khác. Views đo "exposure" còn IC đo "network position" — hai khái niệm khác nhau về bản chất.

---

### Gap về regime-level analysis

**Những gì đã có (global):**

Ma trận tương quan 8×8 tính trên toàn bộ 5,000 nodes. Đây là global view — trung bình hóa mọi thứ.

**Những gì còn thiếu (regime-level):**

H3 nói: _"pattern tương quan không đồng đều giữa các vùng cấu trúc."_ Tức là câu hỏi: correlation giữa views và IC có thay đổi tùy theo degree của node không?

Hình dung:

```
Nodes degree thấp  (ngoài rìa):  Spearman(views, IC) = ???
Nodes degree trung bình:         Spearman(views, IC) = ???
Nodes degree cao   (hub):        Spearman(views, IC) = ???
```

Có thể views và IC tương quan tốt hơn ở nhóm hub (vì hub có nhiều views lẫn nhiều kết nối) nhưng tệ hơn ở nhóm periphery. Nếu đúng như vậy thì con số global 0.469 đang che giấu sự khác biệt theo regime.

Hiện tại chưa có artifact nào tính correlation theo từng regime, nên claim "pattern không đồng đều" trong H3 chưa có bằng chứng số liệu trực tiếp. Muốn khóa claim này cần tính stratified Spearman theo degree quartile hoặc kshell quintile.

---

### Tóm tắt RQ2 + RQ2b theo sơ đồ logic

```
RQ2: Views có đồng ý với IC không?
        ↓
Spearman(views, IC) = 0.469 → Liên quan, nhưng yếu
        ↓
Typology 4 quadrants: Hidden (IC↑ views↓), Overrated (IC↓ views↑)
        ↓
Permutation null:
    Real Hidden = 285 < Null mean = 450
    p = 0.00498 → Divergence có cấu trúc thực (không phải random threshold artifact)
        ↓
Kết luận RQ2: Views và IC không thay thế được nhau;
    divergence có tính chọn lọc cấu trúc

─────────────────────────────────────────────────────────

RQ2b: Cái gì tương quan với IC? Bao nhiêu?
        ↓
Ma trận 8×8:
    Structural cluster (degree/kshell/pagerank/two_hop): IC ~0.82
    Views: IC 0.469  → nằm tách biệt khỏi cluster
        ↓
Multicollinearity: Structural metrics ≈ một underlying factor X
    → IC = f(X) + noise
    → Views = g(X) + lớn noise + thành phần riêng
        ↓
Kết luận RQ2b (global): IC là hàm của vị trí mạng, không phải popularity
Gap: Regime-level analysis chưa có
```

---

### Một câu tóm gọn toàn bộ RQ2 + RQ2b

> **"Số lượt xem (views) và khả năng lan truyền (IC) đo hai thứ khác nhau: views đo exposure/popularity, IC đo vị trí cấu trúc trong mạng. Hai tín hiệu này có liên quan (Spearman 0.469, không phải 0) nhưng không đủ để thay thế nhau (285 Hidden influencers bị views đánh giá thấp một cách có hệ thống, không phải ngẫu nhiên). Mọi structural metric khác (degree, pagerank, kshell) giải thích IC tốt hơn views gần gấp đôi — và chúng đều đang đo cùng một underlying factor là vị trí trung tâm trong mạng."**

---

## Phân tích RQ3 (H4) và RQ3b (H5) — Dành cho sinh viên CS/AI/ML

---

### Bối cảnh: Tại sao cần surrogate model?

Nhắc lại từ RQ1: **IC simulation tốn 480 giây** cho một lần chạy full graph. Nếu muốn phân tích influence cho 168,114 nodes, hoặc muốn thử nhiều scenario khác nhau, chạy IC mỗi lần là không thực tế.

**Ý tưởng surrogate:** Thay vì chạy IC simulation tốn kém, huấn luyện một model học cách _xấp xỉ_ kết quả IC từ các features có sẵn. Sau đó dùng model đó để predict IC score nhanh.

Đây là một pattern quen thuộc trong ML:

```
Expensive oracle  →  Train surrogate  →  Fast approximation
(IC simulation)       (GNN/baseline)     (inference <1s)
```

**RQ3 hỏi hai thứ:**

1. Surrogate (GNN) có predict IC tốt hơn các baseline rẻ không?
2. Lợi ích tốc độ là bao nhiêu?

---

## Phần 1 — Taxonomy của các models được test

Trước khi đi vào kết quả, cần hiểu rõ từng nhóm model đang làm gì:

**Baselines — Analytical proxies (không cần training):**

| Model          | Cách tính                                                            | Chi phí                                  |
| -------------- | -------------------------------------------------------------------- | ---------------------------------------- |
| `degree`       | Đếm số neighbors                                                     | O(N)                                     |
| `pagerank`     | Random walk convergence                                              | O(N·iter)                                |
| `kshell`       | K-core decomposition                                                 | O(N+E)                                   |
| `one_hop`      | One-hop spread: $\sum_{v\in N(u)} 1/\deg(v)$                         | O(E) (linear)                            |
| `two_hop`      | Two-hop spread (weighted-cascade style), dùng thông tin 2-hop từ CSR | ~O(E)–O(E·\bar{d}) (phụ thuộc implement) |
| `mlp_raw_attr` | Neural net trên node attributes                                      | Train ~few sec                           |
| `node2vec_lr`  | Graph embedding + linear regression                                  | Train ~172s                              |

**Surrogates — GNN variants (cần training, được test với feature sets khác nhau):**

| Model            | Features được cấp                          | Mục đích                        |
| ---------------- | ------------------------------------------ | ------------------------------- |
| `gnn_centrality` | degree + pagerank + kshell                 | Best-case GNN                   |
| `gnn_full`       | centrality + raw attributes                | Full information                |
| `gnn_raw_attr`   | views + lifetime + views_log               | Chỉ node attributes             |
| `gnn_graph_only` | Không có node features, chỉ graph topology | Ablation: message passing thuần |
| `gnn_random`     | Random features (null baseline)            | Sanity check                    |

Thiết kế này là một **ablation study có hệ thống**: bằng cách thay đổi feature set, ta có thể tách biệt contribution của từng loại thông tin.

---

## Phần 2 — Kết quả head-to-head

### Bảng so sánh đầy đủ

```
Model              Spearman   NDCG@10%   Prec@10%   Training
─────────────────────────────────────────────────────────────
degree             0.8263     0.8815     0.600      0s (analytical)
pagerank           0.8241     0.8568     0.560      0s (analytical)
kshell             0.8159     ~0.778*    0.500      0s (analytical)
node2vec_lr        0.8090     0.8515     0.576      172s
two_hop            0.8039     0.8478     0.552      0s (analytical)
─────────────────────────────────────────────────────────────
gnn_centrality     0.8168     0.8597     0.572      23.5s
gnn_full           0.8134     0.8565     0.566      23.5s
─────────────────────────────────────────────────────────────
mlp_raw_attr       0.4350     0.6010     ~0.40      few sec
gnn_raw_attr       0.5341     0.6742     0.452      23.5s
gnn_graph_only     0.4703     0.8350     0.514      23.5s
─────────────────────────────────────────────────────────────
gnn_random         0.2750     0.2196     0.164      23.5s  (null)
─────────────────────────────────────────────────────────────
*kshell NDCG có tie-breaking issue chưa lock
```

---

### Finding 1: GNN không vượt best baseline — và đây là lý do tại sao

`gnn_centrality` (Spearman 0.8168) thua `degree` (Spearman 0.8263). Khoảng cách nhỏ (−0.010) nhưng chiều đi sai hướng so với kỳ vọng mặc định "deep learning > handcrafted features."

**Tại sao?** Đây là hệ quả trực tiếp của multicollinearity đã thấy ở RQ2b:

```
degree   ↔ kshell:    0.993  ┐
pagerank ↔ two_hop:   0.986  ├─ Tất cả ≈ một factor "network centrality"
degree   ↔ pagerank:  ~0.97  ┘

IC score ↔ factor này: ~0.826
```

Khi `gnn_centrality` được cấp `[degree, pagerank, kshell]` làm features, GNN đang học một hàm của **thông tin đã gần như đồng nhất** với nhau. `degree` analytical baseline cũng đang dùng thông tin tương tự, trực tiếp hơn, không có training overhead.

Hình dung bằng analogy ML quen thuộc:

> Giống như bạn train một neural network để predict house price với features `[area_m2, area_ft2, area_cm2]` — cả ba đều là cùng một thứ. Model có thể học được nhưng không tốt hơn linear regression trên `area_m2` đơn giản.

GNN có thể học **non-linear combinations** của features, nhưng khi underlying signal đã gần linear với target (Spearman 0.826 là correlation rất cao), non-linearity không add nhiều giá trị.

---

### Finding 2: Ablation sạch nhất — message passing có ích khi thiếu structural features

```
mlp_raw_attr    Spearman = 0.435   (MLP với views + lifetime)
gnn_raw_attr    Spearman = 0.534   (GNN với cùng features)
Δ = +0.099
```

Đây là controlled experiment rõ ràng nhất trong toàn bộ study: **cùng feature set, chỉ khác architecture** (MLP không dùng graph vs GNN dùng graph).

GNN tốt hơn MLP ~10 Spearman points → **message passing thực sự học được structural information** từ graph topology, ngay cả khi chỉ được cấp raw node attributes.

Nhưng dù vậy, `gnn_raw_attr` (0.534) vẫn kém xa `degree` (0.826). Giải thích: views và lifetime không encode vị trí mạng, và message passing với features yếu chỉ cải thiện được một phần.

---

### Finding 3: gnn_graph_only — anomaly cần giải thích

```
gnn_graph_only:  Spearman = 0.470   NDCG = 0.835
```

Spearman chỉ 0.470 (thứ hạng tổng thể tệ) nhưng NDCG@10% lại 0.835 (top-k quality cao bất thường). Đây là contradiction rõ ràng.

**Giải thích:** GNN chỉ dùng graph topology (không có node features) có thể học được **local neighborhood structure** — đủ để nhận ra "node này ở vùng dense, có thể là influencer" → NDCG top-k tốt. Nhưng không calibrate được **magnitude** của IC score → Spearman overall thấp vì thứ hạng ở vùng middle bị sai nhiều.

Điều này sẽ được thấy rõ hơn trong per-group analysis ở RQ3b (True group: gnn_graph_only có Spearman tốt nhưng MAE rất cao).

---

### Finding 4: Runtime story — lợi thế thực sự của surrogate

```
mc_ic_labeling:   480.3s    (tạo nhãn IC một lần)
node2vec train:   172.3s    (training)
gnn training:      23.5s    (training, average)
gnn inference:      0.067s  (predict toàn bộ graph)
degree inference:  ~0.005s  (analytical)
```

Speedup: 480s / 0.067s ≈ **7,169×** so với IC simulation.

**Cách đọc đúng về giá trị của surrogate:**

```
Scenario A — Không có surrogate:
    Muốn IC ranking → chạy IC simulation → 480s mỗi lần

Scenario B — Có surrogate (GNN):
    Lần đầu: chạy IC để tạo nhãn (480s) + train GNN (23.5s) = 503.5s
    Lần 2, 3, ..., n: chỉ cần inference → 0.067s mỗi lần

Break-even: sau 2 lần → tiết kiệm 480s × (n-1) - 23.5s
```

Đây không phải "GNN nhanh hơn baselines" (baselines analytical còn nhanh hơn GNN). Đây là "GNN nhanh hơn IC simulation gốc" — điều thực sự có ý nghĩa khi cần chạy divergence analysis nhiều lần hoặc trên nhiều subgraph khác nhau.

---

## Phần 3 — RQ3b: Per-group analysis

### Setup: Tại sao global metric không đủ?

Giả sử bạn train một model phân loại ảnh chó/mèo và đạt accuracy 95%. Nghe có vẻ tốt. Nhưng nếu dataset có 95% ảnh chó và 5% ảnh mèo, model có thể đạt 95% chỉ bằng cách **predict "chó" cho mọi ảnh** — không học được gì về mèo.

Trong project này, **Non chiếm 84.3%** của typology. Model đạt Spearman 0.817 global có thể đang học rất tốt trên Non (đa số) và hoàn toàn fail trên Hidden (5.7% nhưng quan trọng nhất).

Per-group analysis phá vỡ global metric thành từng nhóm để kiểm tra điều này.

---

### Kết quả per-group (GNN models trên test split ~1,000 nodes)

```
Nhóm       n      gnn_centrality           gnn_full
                  Spearman  MAE            Spearman  MAE
──────────────────────────────────────────────────────────
Non        853    0.784     0.547          0.783     0.545   ← tốt
Overrated   53    0.851     0.519          0.844     0.599   ← rất tốt
True        43    0.661     0.681          0.659     0.608   ← trung bình
Hidden      51    0.211     1.421          0.222     1.433   ← sụp đổ
```

Visualize pattern:

```
Spearman theo nhóm (gnn_centrality):

Overrated  ████████████████████████████████████████ 0.851
Non        ██████████████████████████████████████   0.784
True       █████████████████████████████████        0.661
Hidden     ██████████                               0.211
                                                ↑
                                         Sụp đổ hoàn toàn
```

---

### Tại sao Hidden là nhóm khó nhất?

Đây là câu hỏi quan trọng nhất của RQ3b. Hãy phân tích từng yếu tố:

**Hidden = IC cao + views thấp**. Để model predict đúng IC score của Hidden nodes, nó cần học: _"node này có views thấp nhưng IC cao — tại sao?"_

Câu trả lời từ structural_profiling (RQ4):

```
Hidden nodes:   degree trung bình = 197.8
Overrated nodes: degree trung bình = 99.5
```

Hidden influencers có **degree cao gần gấp đôi** Overrated. Nhưng views của họ thấp. Nghĩa là: họ có nhiều kết nối trong mạng (→ IC cao) nhưng không nổi tiếng theo lượt xem.

**Vấn đề cho model:** Hidden nodes phá vỡ trực giác “popularity cao thì influence cao”: views thấp nhưng IC cao. Vì views và IC chỉ tương quan mức vừa (Spearman 0.469), nên có một phần dữ liệu mà “views” trở thành feature gây nhiễu nếu model dựa vào nó quá nhiều.

Tuy nhiên, cần nói cẩn thận: Overrated (views cao nhưng IC thấp) chính là counterexample cho “views cao → IC cao”, nên đúng hơn là: model phải học một quan hệ **không thuần tuyến tính** giữa views và IC, và Hidden/Overrated là các vùng khó nhất của quan hệ đó.

Thêm vào đó, **n=51 Hidden nodes trong test set** — quá nhỏ để model học được pattern đặc thù. Trong training set cũng ít Hidden nodes tương tự.

---

### Overrated dễ hơn Hidden — tại sao?

```
Overrated: Spearman 0.851 (rất tốt)
Hidden:    Spearman 0.211 (sụp đổ)
```

Overrated nodes (views cao, IC thấp) dễ predict vì:

- Model đã thấy pattern "views cao → thường có IC cao" trong training
- Nhưng Overrated nodes có views cao **và** degree tương đối thấp (99.5 vs Hidden 197.8)
- → Model học được: nếu views cao nhưng degree thấp → IC thấp (không lan truyền tốt dù nổi tiếng)

Pattern này **consistent** và xuất hiện đủ nhiều trong training để model nắm được.

---

### Anomaly của gnn_graph_only trên True group

```
gnn_graph_only trên True group:
    Spearman = 0.752   (cao nhất trong các GNNs cho nhóm này!)
    MAE      = 2.554   (cao nhất — calibration tệ nhất!)
```

Đây là contradiction rõ ràng nhất trong toàn bộ dataset:

- **Spearman 0.752**: model biết ai trong True group có rank cao hơn ai (ordering đúng)
- **MAE 2.554**: nhưng giá trị predict bị sai lệch lớn về magnitude

**Giải thích:** True nodes (IC cao + views cao) thường là các hub trung tâm trong mạng. Graph topology (không có features) đủ để nhận ra "node này ở vùng dense → rank cao trong nhóm." Nhưng vì không có node attributes, model không thể calibrate đúng **giá trị tuyệt đối** của IC score → MAE cao.

Đây là ví dụ điển hình của **good ranking, poor calibration** — một vấn đề quan trọng nếu model được dùng không chỉ để xếp hạng mà còn để estimate actual influence magnitude.

---

### Gap của RQ3b: Thiếu per-group cho baselines

Hiện tại chỉ có per-group metrics cho GNN variants. Câu hỏi chưa được trả lời:

```
degree trên Hidden:    Spearman = ???
pagerank trên Hidden:  Spearman = ???
two_hop trên Hidden:   Spearman = ???
```

**Nhận định hợp lý (nhưng cần số liệu để chốt):** Baselines structural (degree/pagerank/kshell/2-hop) **có thể vẫn khá tốt** trên Hidden, vì Hidden được định nghĩa bởi (IC cao, views thấp) và các baseline này **không dùng views**. Do đó, nói “baseline fail trên Hidden” là không chắc và dễ bị reviewer bắt bẻ.

Đây là gap thực sự cần fill: **tính per-group evaluation cho baselines từ artifacts đã có** (không cần rerun model; chỉ cần join typology labels vào các cột prediction của baselines rồi tính Spearman/MAE theo nhóm).

---

### Tóm tắt RQ3 + RQ3b theo sơ đồ logic

```
RQ3: GNN có tốt hơn baselines không?
        ↓
Best GNN (gnn_centrality): Spearman 0.8168
Best baseline (degree):    Spearman 0.8263
→ GNN KHÔNG vượt best baseline
        ↓
Tại sao? Multicollinearity:
    GNN features (degree/pagerank/kshell) ≈ cùng signal với degree baseline
    → GNN không có thông tin mới để học thêm
        ↓
Bằng chứng message passing có ích:
    gnn_raw_attr (0.534) > mlp_raw_attr (0.435): +0.099 Spearman
    → GNN học được structure từ graph, nhưng không đủ để bắt kịp structural baselines
        ↓
Runtime lợi thế:
    IC simulation: 480s → GNN inference: 0.067s → speedup 7,169×
    Giá trị nằm ở "chạy IC một lần, dùng surrogate nhiều lần"
        ↓
Kết luận RQ3: GNN competitive không dominant; structural baselines gần trần
    Narrative: "Feature set quyết định chất lượng, không phải model architecture"

─────────────────────────────────────────────────────────────────

RQ3b: Nhóm nào khó dự đoán nhất?
        ↓
Per-group Spearman (gnn_centrality):
    Non:      0.784  ← đa số, dễ
    Overrated: 0.851  ← pattern consistent, dễ
    True:     0.661  ← trung bình
    Hidden:   0.211  ← sụp đổ hoàn toàn
        ↓
Tại sao Hidden khó?
    - Views thấp dù degree cao → phá vỡ majority pattern
    - n=51 trong test → không đủ để model học pattern đặc thù
    - Out-of-distribution: IC cao không đi cùng tín hiệu nào model đã thấy
        ↓
Global Spearman 0.817 che giấu failure hoàn toàn trên Hidden
→ Aggregate metric không đủ, cần per-group evaluation
        ↓
Gap: Chưa có per-group cho baselines → cần tính bổ sung
```

---

### Một câu tóm gọn RQ3 + RQ3b

> **"GNN đạt performance cạnh tranh với baselines tốt nhất (Spearman 0.817 vs 0.826) nhưng không vượt, vì cả hai đều bị giới hạn bởi cùng một ceiling: IC chủ yếu là hàm của network centrality, và baselines analytical đã encode thông tin đó trực tiếp. Giá trị thực của surrogate nằm ở tốc độ (7,169× nhanh hơn IC simulation), không phải accuracy. Quan trọng hơn, per-group analysis tiết lộ điều global metric che giấu: mọi model đều sụp đổ trên Hidden influencers (Spearman 0.211) — đúng nhóm mà popularity-based signals fail nhất — trong khi đạt gần hoàn hảo trên Overrated (0.851). Đây là bằng chứng rằng bottleneck không nằm ở model capacity mà nằm ở sự thiếu vắng features phân biệt được Hidden khỏi Non nodes."**

---

## Phân tích RQ4 (H6) — Dành cho sinh viên CS/AI/ML

---

### Bối cảnh: RQ4 hỏi câu gì?

Từ RQ2 ta biết: Hidden influencers (IC cao, views thấp) và Overrated nodes (IC thấp, views cao) là hai nhóm divergence chính. Từ RQ3b ta biết: mọi model đều fail trên Hidden.

**RQ4 hỏi tiếp:** _Tại sao? Hai nhóm này khác nhau về mặt cấu trúc mạng như thế nào? Feature nào phân biệt được họ?_

Đây là bước chuyển từ _"mô tả hiện tượng"_ sang _"giải thích cơ chế"_. Trong ML, đây tương đương với bài toán **feature attribution / interpretability**: thay vì hỏi "model predict đúng không?", ta hỏi "tại sao một số nodes khó predict hơn?"

---

### Setup: Structural Profiling là gì?

Với mỗi node trong Hidden và Overrated group, project đo 6 features cấu trúc:

```
1. degree                    — số lượng kết nối trực tiếp
2. pagerank                  — "tầm quan trọng" theo random walk
3. kshell (k-core number)    — node nằm ở "lớp lõi" thứ mấy của mạng
4. betweenness centrality    — node nằm trên bao nhiêu shortest path
5. cross_community_fraction  — tỷ lệ neighbors thuộc community khác
6. life_time                 — thời gian hoạt động (ngày)
```

Sau đó so sánh distribution của hai nhóm Hidden vs Overrated bằng statistical test + effect size.

---

### Kết quả: Hidden và Overrated khác nhau về mọi structural feature

```
Feature                    Hidden mean    Overrated mean    Significant?
───────────────────────────────────────────────────────────────────────
degree                     197.8          99.5              Có ✓
kshell                      96.05          54.8             Có ✓
pagerank                    1.19e-5         7.78e-6         Có ✓
betweenness                 1.63e-5         5.79e-6         Có ✓
cross_community_fraction     0.417           0.342          Có ✓
life_time                 1740.8          1881.4            KHÔNG ✗
```

Visualize chênh lệch:

```
degree:      Hidden ████████████████████ 197.8
             Overrated ██████████ 99.5          (Hidden gấp đôi)

kshell:      Hidden █████████████████ 96.05
             Overrated █████████ 54.8           (Hidden gấp 1.75×)

cross_comm:  Hidden ██████████████████ 0.417
             Overrated █████████████ 0.342      (Hidden cao hơn 22%)

life_time:   Hidden ████████████████ 1740.8
             Overrated █████████████████ 1881.4 (KHÔNG khác biệt)
```

**Kết luận trực tiếp:** Hidden influencers có vị trí mạng **mạnh hơn đáng kể** so với Overrated nodes trên các feature cấu trúc mạng (degree/kshell/pagerank/betweenness/cross-community). Nhưng thời gian hoạt động (life_time) **không phân biệt** hai nhóm một cách có ý nghĩa thống kê.

---

### Giải thích từng feature

**Degree (197.8 vs 99.5):**

Hidden influencers có trung bình gần **gấp đôi** số kết nối so với Overrated. Điều này giải thích trực tiếp tại sao IC của họ cao: trong weighted-cascade, node có nhiều neighbors có nhiều "đường" để lan truyền thông tin ra ngoài.

Overrated nodes có degree thấp hơn đáng kể — họ nổi tiếng (views cao) nhưng không kết nối rộng trong mạng, nên thông tin từ họ không lan truyền được xa.

---

**Kshell (96.05 vs 54.8) — "Depth in the network core":**

K-core decomposition là thuật toán bóc vỏ mạng dần dần:

```
Bước 1: Xóa tất cả nodes có degree < 2  → còn lại là "2-core"
Bước 2: Xóa tất cả nodes có degree < 3  → còn lại là "3-core"
...
Bước k: Xóa nodes có degree < k  → còn lại là "k-core"
```

Kshell của một node = giá trị k lớn nhất mà node đó còn tồn tại. Kshell cao → node nằm sâu trong "lõi" của mạng, kết nối với những nodes cũng có nhiều kết nối.

Hidden: kshell = 96 → nằm ở lõi thứ 96, sâu trong network core
Overrated: kshell = 55 → nằm ngoài rìa hơn đáng kể

Trong IC diffusion, nodes ở core sâu có khả năng reach ra toàn bộ mạng cao hơn vì họ kết nối với nhiều hub khác.

---

**Cross-community fraction (0.417 vs 0.342) — "Bridge giữa cộng đồng":**

Từ Task 2, mạng được chia thành 24 communities bằng Louvain algorithm. `cross_community_fraction` của một node = tỷ lệ neighbors của nó thuộc **community khác**:

```
cross_community_fraction(u) = |{v ∈ N(u) : community(v) ≠ community(u)}| / degree(u)
```

- Node có fraction = 0 → **pure insider**: chỉ kết nối trong cộng đồng của mình
- Node có fraction = 1 → **pure bridge**: mọi kết nối đều ra cộng đồng khác

Hidden: 0.417 → ~42% neighbors thuộc community khác
Overrated: 0.342 → ~34% neighbors thuộc community khác

Hidden influencers là **broker giữa cộng đồng** — họ kết nối nhiều community khác nhau, nên thông tin từ họ có thể "nhảy" sang nhiều vùng mạng, tăng reach của IC.

---

**Life_time — Finding KHÔNG significant:**

Thời gian hoạt động (số ngày account tồn tại) không phân biệt được Hidden và Overrated. Đây cũng là một finding có giá trị: _"tuổi tài khoản không giải thích divergence"_ — loại trừ một hypothesis thay thế.

---

### Phần quan trọng nhất: Null Model Caveat

Đây là điểm kỹ thuật tinh tế nhất của RQ4, và là chỗ dễ **overclaim** nhất nếu không cẩn thận.

**Vấn đề đặt ra:** Ta đã chứng minh Hidden có degree cao hơn Overrated (197.8 vs 99.5). Nhưng câu hỏi tiếp theo là: _"Hidden có structural position bất thường so với kỳ vọng từ degree của họ không?"_

Tại sao câu hỏi này quan trọng? Vì:

- Node có degree 200 **tự nhiên** sẽ có betweenness cao, kshell cao, pagerank cao
- Nếu Hidden chỉ đặc biệt vì họ có degree cao, thì "cấu trúc đặc biệt" của họ chỉ là hệ quả của degree — không phải independent finding

**Configuration Null Model** kiểm định chính xác điều này:

```
Null hypothesis: Hidden nodes có structural position
                 "bình thường" so với distribution của
                 nodes có cùng degree sequence
```

Cách thực hiện (đúng theo artifact hiện có): Tạo nhiều random graphs giữ nguyên degree sequence của mạng gốc (mỗi node giữ nguyên degree) nhưng **shuffle** ngẫu nhiên các kết nối. Sau đó đo **betweenness** của Hidden nodes trong các null graphs này.

```
null_model_typology_summary.json:
    rho_mean = 0.441
    hidden_betweenness_gap = -4.63e-05
    hidden_betweenness_gap_sigma = -1.24   ← KHÔNG significant (|z| < 2)
```

Lưu ý phạm vi: artifact null model này chạy trên **subgraph sample (n=500 nodes)** với **10 realizations**, nên kết luận cần được phát biểu như một “sanity check / caveat”, không phải khẳng định tuyệt đối cho toàn graph.

**Kết quả:** gap_sigma = −1.24, tức là betweenness của Hidden nodes chỉ thấp hơn null mean 1.24 standard deviations — **không đủ để reject null hypothesis** (thường cần |z| ≥ 2).

---

### Hai so sánh khác nhau — không mâu thuẫn

Đây là điểm dễ nhầm nhất. Kết quả trông có vẻ contradictory:

```
structural_profiling: Hidden betweenness (1.63e-5) >> Overrated (5.79e-6)  → SIGNIFICANT
null_model:           Hidden betweenness vs configuration null              → NOT SIGNIFICANT
```

Nhưng hai test này hỏi **hai câu hỏi hoàn toàn khác nhau**:

```
Test 1 (structural_profiling):
    "Hidden có khác Overrated về betweenness không?"
    → Có, significant
    → Hidden có betweenness gấp ~2.8× Overrated

Test 2 (null model):
    "Hidden có betweenness bất thường so với nodes có cùng degree không?"
    → Không, gap_sigma = -1.24, không significant
    → Betweenness của Hidden nằm trong expectation từ degree của họ
```

Dùng analogy ML: giống như so sánh feature importance theo hai baseline khác nhau:

- Test 1: Hidden vs Overrated → Hidden "thắng" vì có degree cao hơn nên betweenness cũng cao hơn
- Test 2: Hidden vs "đồng đẳng cùng degree" → Hidden không đặc biệt hơn các nodes khác cùng degree

**Kết luận đúng (nên nói đúng phạm vi):** Với **betweenness**, configuration null cho thấy mức betweenness của Hidden _không “bất thường”_ so với kỳ vọng từ degree sequence (z ≈ -1.24). Điều này ủng hộ diễn giải rằng _ít nhất với betweenness_, chênh lệch Hidden vs Overrated có thể bị chi phối mạnh bởi degree. Các feature khác (kshell/pagerank/cross-community) chưa được kiểm bằng configuration null trong artifact hiện tại, nên không nên khẳng định “mọi structural advantage đều chỉ do degree”.

---

### Tổng hợp: Câu chuyện cơ chế của RQ4

Kết hợp tất cả findings, một narrative coherent xuất hiện:

```
┌─────────────────────────────────────────────────────────────┐
│  Hidden influencers: IC cao, views thấp                     │
│                                                             │
│  Tại sao IC cao?                                            │
│    → Degree cao (197.8) → nhiều đường lan truyền            │
│    → Kshell cao (96) → nằm sâu trong network core           │
│    → Cross-community fraction cao (0.417) → bridge nhiều   │
│      cộng đồng → reach rộng                                │
│                                                             │
│  Tại sao views thấp dù IC cao?                              │
│    → Life_time không khác → không phải vì account mới      │
│    → Structural position không giải thích được views        │
│    → Likely: content type, platform algorithm, audience     │
│      niche → ngoài scope của structural analysis           │
│                                                             │
│  Structural uniqueness có "genuinely special" không?        │
│    → Null model: gap_sigma = -1.24 → KHÔNG                  │
│    → Hidden đặc biệt vì degree cao, không phải vì có        │
│      property cấu trúc độc lập ngoài degree                │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Overrated nodes: IC thấp, views cao                        │
│                                                             │
│  Tại sao IC thấp?                                           │
│    → Degree thấp hơn (99.5) → ít đường lan truyền          │
│    → Kshell thấp hơn (54.8) → ở ngoài rìa network          │
│    → Cross-community fraction thấp hơn (0.342) → ít bridge │
│      → lan truyền bị giới hạn trong một vùng               │
│                                                             │
│  Tại sao views cao dù IC thấp?                              │
│    → Content exposure, algorithm boost, external fame       │
│    → Popularity ≠ network diffusion capacity               │
└─────────────────────────────────────────────────────────────┘
```

---

### Hàm ý cho RQ3b: Tại sao model fail trên Hidden?

RQ4 giải thích retroactively tại sao RQ3b cho kết quả model fail trên Hidden:

Hidden nodes có **degree cao** nhưng **views thấp** — hai tín hiệu mâu thuẫn với majority pattern mà model học được. Trong training set:

- Nodes có degree cao thường có IC cao VÀ views cũng tương đối cao (True group)
- Hidden nodes phá vỡ điều này: degree cao → IC cao, nhưng views thấp

Model không có feature nào phân biệt "degree cao + views thấp" (Hidden) khỏi "degree cao + views cao" (True) nếu chỉ nhìn vào một số feature phổ biến như views/lifetime và vài centrality cơ bản. Một hướng defensible để đề xuất (nhưng vẫn cần thực nghiệm) là thêm các feature “bridging” như `cross_community_edge_fraction` vì nó phân biệt Hidden vs Overrated rõ rệt trong profiling.

---

### Tóm tắt RQ4 theo sơ đồ logic

```
RQ4: Feature cấu trúc nào phân biệt Hidden vs Overrated?
        ↓
Structural profiling (6 features, BH-corrected tests):
    degree:               Hidden 197.8  >>  Overrated 99.5   ✓ significant
    kshell:               Hidden 96.05  >>  Overrated 54.8   ✓ significant
    pagerank:             Hidden        >>  Overrated         ✓ significant
    betweenness:          Hidden        >>  Overrated         ✓ significant
    cross_comm_fraction:  Hidden 0.417  >   Overrated 0.342  ✓ significant
    life_time:            Hidden 1740   ≈   Overrated 1881   ✗ NOT significant
        ↓
H6 confirmed (đúng hơn): Hidden có structural profile mạnh hơn Overrated trên hầu hết metric cấu trúc mạng; life_time không significant
        ↓
Null model caveat:
    Configuration null (giữ nguyên degree, shuffle connections)
    hidden_betweenness_gap_sigma = -1.24  → NOT significant
        ↓
Interpretation:
    Hidden structural advantage largely explained by their higher degree
    → Không có "mysterious structural property" độc lập ngoài degree
        ↓
Kết luận RQ4:
    "Hidden influencers nổi bật vì degree cao + cross-community position
     → natural consequences of network position.
     Views thấp là bất thường (không giải thích được bằng structure)
     → divergence nằm ở popularity mechanism, không phải network mechanism"
```

---

### Một câu tóm gọn RQ4

> **"Hidden influencers có vị trí mạng mạnh hơn Overrated trên mọi chiều đo lường (degree gấp đôi, kshell gấp 1.75×, cross-community fraction cao hơn 22%) — điều này giải thích IC cao của họ. Nhưng configuration null model cho thấy structural advantage này là hệ quả tự nhiên của degree cao, không phải một property cấu trúc độc lập. Điều thực sự bất thường không phải là cấu trúc mạng của Hidden nodes, mà là sự vắng mặt của views tương xứng — một hiện tượng ngoài tầm giải thích của structural analysis thuần túy."**

---

## Appendix — Sensitivity của IC với công thức $p(u,v)$

Phần chính của project dùng IC với **weighted cascade**:

$$p(u,v) = \min\left(1, \frac{\kappa}{\deg(v)}\right),\quad \kappa = 1$$

Trong code/artefacts hiện tại, $p$ phụ thuộc vào **degree của node bị tác động** (target) $v$ (tức là node “dễ bị kích hoạt” hơn khi degree nhỏ). Trong feasibility protocol đã có “kappa sweep” để kiểm tra việc tăng/giảm cường độ xác suất có thay đổi kết luận hay chỉ thay đổi chế độ cascade.

Mục tiêu của appendix này là: nếu reviewer hỏi “kết luận RQ1–RQ4 có phụ thuộc vào cách chọn $p(u,v)$ không?”, ta có một **ablation/sensitivity plan** rõ ràng, rẻ, và diễn giải được.

### Nguyên tắc quan trọng nhất: tránh đổi _cascade regime_ một cách tầm thường

Nhiều công thức $p(u,v)$ “nghe hợp lý” nhưng nếu không chuẩn hoá thì sẽ đẩy xác suất lên quá cao (ví dụ nhiều cạnh có $p\approx 0.2$–$0.5$), làm cascade bùng nổ và IC scores **bão hoà**. Khi đó mọi thứ (ranking, correlation, typology) có thể thay đổi mạnh nhưng không phải vì “cơ chế đúng hơn”, mà vì ta đã đổi bài toán sang một regime khác.

Vì vậy, khi thử $p(u,v)$ mới, nên giữ “mức năng lượng” tương đương baseline, theo một trong hai cách:

- **Calibrate theo mean edge probability:** đặt $p'(u,v)=\mathrm{clip}(s\,p_\text{raw}(u,v),0,1)$, chọn $s$ sao cho $\mathbb{E}[p'] \approx \mathbb{E}[p_\text{WC}]$ trên sample cạnh.
- **Calibrate theo cascade size trên sample:** chọn tham số (hoặc hệ số $s$) sao cho mean cascade size / mean influence trên một sample node (ví dụ 200–400 nodes) gần baseline.

Nếu không calibrate, thí nghiệm sẽ khó bảo vệ vì “khác biệt” có thể chỉ là artefact của scaling.

### Phân loại các họ công thức $p(u,v)$ (và rủi ro)

Để thảo luận “đúng/sai” cho $p(u,v)$, cách sạch nhất là tách thành 2 tầng:

1. **Node susceptibility (target-based):** $p$ tăng/giảm theo đặc trưng của node $v$ (degree/kshell/pagerank/views/life_time).
2. **Edge affinity (edge-based):** $p$ phụ thuộc vào “độ liên quan” của cặp $(u,v)$ (cùng community, common neighbors, Adamic–Adar, Jaccard, v.v.).

Trong đồ án hiện tại, cấu trúc “target-based” là phù hợp vì:

- Dữ liệu cạnh không có trọng số tương tác theo thời gian, nên edge-affinity phức tạp dễ trở thành “tự chế” khó justify.
- Một số edge-affinity (CN/AA) có chi phí tính toán lớn trên graph lớn.

#### (A) Structural-only: an toàn, dễ defend

- **WC baseline:** $p=1/\deg(v)$.
- **Kappa-scaled WC:** $p=\min(1,\kappa/\deg(v))$ với $\kappa\in\{1.5,2.0,3.0\}$ (đã có khung trong feasibility protocol).
- **Degree exponent:** $p=\min(1,\kappa/\deg(v)^\alpha)$ với $\alpha\in\{0.5,1.0,1.5\}$.

Ý nghĩa khoa học: kiểm tra xem kết luận có “đứng vững” khi ta điều chỉnh mức penalty theo degree.

#### (B) Community/bridge modifiers: giá trị cao cho RQ4, nhưng phải giữ scale

Một modifier hợp lý (vì RQ4 nhấn mạnh “cross-community position”) là:

$$p'(u,v)=\mathrm{clip}\big(p_\text{base}(u,v)\cdot (1+\beta\,g(u,v)),0,1\big)$$

trong đó $g(u,v)$ có thể là:

- indicator “edge là cross-community” (cần community assignment có sẵn), hoặc
- hàm của node-level `cross_community_edge_fraction`.

Sau đó **calibrate lại** $\beta$ hoặc hệ số scale để giữ mean $p$ / cascade size tương đương baseline.

Ý nghĩa khoa học: nếu Hidden vẫn nổi bật khi ta thay đổi giả định “bridging edges mạnh hơn/yếu hơn”, thì narrative “Hidden là bridge/core” trở nên reviewer-proof hơn.

#### (C) Popularity-driven / Twitch signals: chỉ nên dùng như _stress test_

Các công thức dùng `views`/`life_time`/các tín hiệu Twitch để tăng $p(u,v)$ có thể hợp lý nếu ta coi đây là “propensity to influence” gắn với popularity. Tuy nhiên nó tạo rủi ro diễn giải:

- Nếu $p(u,v)$ phụ thuộc vào `views(v)` thì IC label sẽ “nhiễm” popularity ngay trong định nghĩa, làm RQ2/RQ3 dễ bị circular.

Vì vậy nếu thử, nên trình bày rõ đây là **sensitivity/stress test**: “Nếu ta giả định diffusion probability phụ thuộc popularity thì kết luận thay đổi như thế nào?” chứ không phải “model đúng hơn”.

#### (D) Similarity-based (CN/AA/Jaccard): có thể đúng về mặt ý tưởng, nhưng đắt

Các edge-affinity kiểu common neighbors / Adamic–Adar thường yêu cầu tính overlap lân cận cho rất nhiều cạnh → nặng trên graph lớn. Nếu muốn dùng, nên:

- chỉ chạy trên subgraph/sample,
- hoặc chỉ dùng cho một tập cạnh giới hạn,
- hoặc dùng approximation.

Nếu mục tiêu là “defend kết luận”, họ (A) và (B) thường đủ và rẻ hơn.

### Bộ thí nghiệm tối thiểu (khuyến nghị) để bổ sung robustness

Để vừa ít công, vừa trả lời reviewer một cách thuyết phục, đề xuất chạy 3 cấu hình $p(u,v)$:

1. **Baseline:** WC ($\kappa=1$).
2. **Scaling ablation:** WC với $\kappa=2$ (hoặc $\alpha=0.5$) đã **calibrate** để không bùng cascade.
3. **Bridge ablation:** WC $\times$ cross-community modifier (boost hoặc penalty), calibrate để giữ mean $p$/cascade size.

Tuỳ thời gian, thêm 1 stress test:

4. **Popularity hybrid (stress test):** convex blend giữa WC và một popularity score đã chuẩn hoá, rồi calibrate lại để tránh regime change.

### Cách báo cáo để “khóa” diễn giải RQ1–RQ4

Không cần thêm quá nhiều metric mới; chỉ cần vài kiểm tra chuẩn:

- **Stability của IC labels:** Spearman $\rho$ giữa IC scores (baseline vs variant) + overlap của top-k.
- **Stability của typology:** số lượng Hidden/Overrated/True/Non và Jaccard overlap của tập Hidden.
- **RQ2 invariance:** tương quan (IC vs one-hop/two-hop/kshell/views) có giữ thứ tự/độ lớn tương đối không.
- **RQ4 narrative check:** structural profiling Hidden vs Overrated có giữ dấu hiệu chính (degree/kshell/bridge) không.

Nếu các kết luận chính giữ nguyên dưới (1)–(3), ta có một câu trả lời rất mạnh: “kết luận chủ yếu đến từ cấu trúc mạng, không phụ thuộc vào một công thức $p(u,v)$ cụ thể”.
