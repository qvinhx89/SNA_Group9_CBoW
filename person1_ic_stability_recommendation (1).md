# Khuyen Nghi Chinh Thuc Gui Person 1

Date: 2026-04-06  
Prepared for: Person 1 (Track A - IC core)  
Prepared by: Team review support

## 1) Muc tieu van de can fix

Hoan tat Stage-4 quality gate theo dung contract v3 trong boi canh:

- Binary top-10 labels dang bat on manh o vung bien.
- Regression target van la huong chinh theo v3, can duoc lock bang bang chung ro rang.
- Mot artifact bat buoc cua Stage-4 hien chua co.

## 2) Hien trang da xac minh (artifact-grounded)

### 2.1 Stability artifact hien co

Nguon: outputs/day1_benchmark/ic_label_stability.json

- n_runs_per_seed: 150
- jaccard_mean: 0.3069298298144156
- jaccard_min: 0.3020833333333333
- jaccard_pass_threshold (>=0.85): false
- spearman_mean: 0.685383690615586

### 2.2 Uncertainty artifact hien co

Nguon: outputs/day1_benchmark/ic_label_uncertainty.json

- boundary_ratio: 0.199
- ambiguous_ratio: 0.155
- top10_ic_score_mean_threshold: 77.67000000000003

### 2.3 Day-1 benchmark branch

Nguon: outputs/day1_benchmark/one_hop_correlation.json

- spearman_rho: 0.7391903714947583
- decision_branch: viable_gnn

### 2.4 Thieu artifact bat buoc

Chua thay: outputs/day1_benchmark/ic_pilot_diagnostics.json

Theo contract Stage-4, file nay la bat buoc va phai co day du cac metrics pilot + KS + jaccard_stability.

## 3) Chan doan ky thuat

- Van de chinh KHONG phai pipeline sai.
- Van de chinh la hard threshold top-10 tren phan phoi continuous co duoi dai + nhieu node sat nguong.
- boundary_ratio ~ 20% va ambiguous_ratio ~ 15.5% xac nhan vung bien rong, gay dao dong label top-10.
- Tang n_runs co the cai thien mot phan nhung hieu qua bien giam dan, de vuot 0.85 thuong doi chi phi runtime khong xung dang.

## 4) Quyet dinh nen lock ngay (khong xung dot M0)

1. Giu nguyen M0 contracts:

- classification_threshold = top 10%
- typology axis van top 10%
- jaccard target contract van 0.85 (khong sua nguong khi chua re-lock M0)

2. Chuyen trong tam van hanh sang huong dung v3:

- Regression target y = log1p(ic_score_mean) la PRIMARY.
- Binary y_top10 la SECONDARY va uncertainty-aware.

3. Khong doi nghia artifact cu dang duoc consumer dung.

- Khong sua schema bat buoc cua classification_labels.parquet.
- Neu them consensus thi xuat them artifact moi, khong pha backward compatibility.

## 5) Ke hoach hanh dong uu tien cho Person 1

## P0 - Bat buoc trong ngay (blocker)

### P0.1 Tao ic_pilot_diagnostics.json dung schema contract

Can co toi thieu:

- n_pilot_nodes
- n_pilot_runs
- mean_reach
- median_reach
- iqr_reach
- top10_to_median_ratio
- cv_score
- rank_stability
- cv_noise_count
- jaccard_stability
- ks_results (degree, kshell, pagerank)

Yeu cau them:

- cv_noise_threshold=0.50 ap dung dung theo docs.
- jaccard_stability phai duoc ghi sau buoc 3 MC stability experiments.

### P0.2 Cap nhat day1_decisions.md

Them section "Label stability decision" voi noi dung:

- Binary top-10 hien provisional.
- Regression la target chinh de tiep tuc pipeline.
- Neu co consensus labels thi ghi ro la supplementary branch.

## P1 - Chot bang chung cho regression stability

Muc tieu: xac nhan regression target du on dinh de team tiep tuc Person 2/3.

De xuat output moi:

- outputs/day1_benchmark/ic_regression_stability.json

Noi dung khuyen nghi:

- Pairwise Spearman giua cac MC seeds tren toan bo labeled nodes.
- Bao cao mean/min Spearman theo tung muc n_runs (vi du: 150, 300, 600, 900, 1200).
- Neu mean Spearman >= 0.90 (hoac nguong team lock) thi cho phep tiep tuc voi regression target ma khong can doi binary Jaccard dat 0.85.

Ghi chu quan trong:

- Script ic_label_stability.py da tinh pairwise Spearman; can bo sung report rieng cho regression gate neu team muon gate tach biet.

## P2 - Binary supplementary theo huong consensus (khuyen nghi manh)

Khong thay artifact cu, them artifact moi de an toan:

- data/processed/classification_labels_consensus.parquet

Schema de xuat:

- node_id
- y_top10_consensus (0/1)
- is_uncertain (0/1)
- vote_count (0..3)
- p_above_top10_threshold (co the reuse tu uncertainty pipeline)

Policy de xuat:

- Positive neu vote_count >= 2
- Negative neu vote_count = 0
- Uncertain neu vote_count = 1 (co the include vote_count=2 as uncertain neu team muon strict hon, nhung phai lock ro)

Rule su dung cho downstream:

- Person 3 binary eval: loai uncertain khoi binary metrics.
- Person 2 typology: van dung top-10 continuous threshold theo M0 (khong doi sang top-20 neu chua re-lock).

## P3 - Hoan thien handoff package cho Person 2/3

Can them vao handoff sau khi P0-P2 xong:

- outputs/day1_benchmark/ic_pilot_diagnostics.json
- outputs/day1_benchmark/ic_regression_stability.json (neu tao)
- data/processed/classification_labels_consensus.parquet (neu tao)
- docs/day1_decisions.md (ban cap nhat)

## 6) Acceptance criteria (Definition of Done)

Person 1 duoc xem la fix xong khi dat tat ca:

1. Co file outputs/day1_benchmark/ic_pilot_diagnostics.json dung schema contract.
2. day1_decisions.md co section decision ro rang: regression primary, binary provisional/consensus.
3. Co bang chung regression stability (artifact + so lieu).
4. Neu trien khai consensus: co artifact moi va huong dan su dung cho Person 3.
5. Khong vi pham M0 locks (top-10, split rule, runtime semantics).

## 7) Risk neu khong fix

- Stage-4 contract tiep tuc fail do thieu artifact bat buoc.
- Team tiep tuc tranh luan tren binary threshold thay vi di tiep theo regression pipeline.
- Person 2/3 consume labels khong dong nhat, gay drift ket qua va kho defend trong paper.

## 8) Message ngan de gui Person 1 (copy/paste)

Person 1,

Can fix gap Stage-4 theo thu tu uu tien sau:

1. Tao outputs/day1_benchmark/ic_pilot_diagnostics.json dung schema contract.
2. Cap nhat docs/day1_decisions.md: regression target la primary; binary top-10 la provisional/consensus branch.
3. Chot bang chung regression stability (pairwise Spearman theo MC seeds, khuyen nghi bao cao theo n_runs sweep).
4. Neu dung consensus labels, xuat them classification_labels_consensus.parquet (khong sua schema classification_labels.parquet hien tai).
5. Handoff lai cho Person 2/3 voi artifact moi + ghi ro rule consume.

Muc tieu la unblock pipeline theo v3 ma khong vi pham M0 lock.
