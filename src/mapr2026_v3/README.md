# MAPR2026 v3 — Script Entry Points

Các file trong folder này được scaffold để 3 người có thể bắt đầu code song song mà ít conflict.

Artifacts/contracts: xem `docs/MAPR2026_v3_team_parallel_coding_plan.md`.

## Preflight (Person 2)

Chạy từ workspace root để kiểm tra prerequisite theo contract hiện tại:

- `python src/mapr2026_v3/preflight_person2.py`

Thêm `--json` nếu muốn output machine-readable.

## Quick dry-run (không chạy compute nặng)

1. Tạo placeholder Day-1 artifacts:

- `python src/mapr2026_v3/day1_benchmark.py --dry-run`

2. Tạo mock IC labels từ SIS (nếu đã có stage3 SIS), hoặc random:

- `python src/mapr2026_v3/ic_labels_primary.py --dry-run`

3. Tạo typology IC×views (dùng mock IC cũng được):

- `python src/mapr2026_v3/typology_ic_views.py --dry-run`

4. Tạo headers cho metrics outputs:

- `python src/mapr2026_v3/run_baselines.py --dry-run`
- `python src/mapr2026_v3/run_surrogates.py --dry-run`

## Real runs

Các file này cố tình để `NotImplementedError` ở chế độ real cho tới khi từng owner hoàn thiện thuật toán.

## Day1 quality-gate utilities (Person 1)

Sau khi đã có IC labels real-mode, chạy thêm bộ kiểm định chất lượng:

- `python src/mapr2026_v3/ic_pilot_diagnostics.py`
- `python src/mapr2026_v3/ic_label_stability.py --n-runs 150 --mc-seeds 0,1,2 --n-jobs -1 --jaccard-threshold 0.85`
- `python src/mapr2026_v3/ic_label_uncertainty.py`

Freeze handoff có hard gate tự động:

- Final mode (mặc định, fail nếu quality gate không đạt):
	- `python src/mapr2026_v3/freeze_day1_handoff.py --version-tag <tag>`
- Provisional mode (ghi gate report nhưng vẫn cho freeze):
	- `python src/mapr2026_v3/freeze_day1_handoff.py --version-tag <tag> --quality-mode provisional`

## P1 stability sweep (with paper evidence logs)

Chạy sweep theo grid n_runs và sinh log minh chứng đầy đủ:

- `python src/mapr2026_v3/ic_stability_sweep.py --run-grid 150,300,500,800,1200 --mc-seeds 0,1,2 --n-jobs -1`

Artifacts sinh ra:
- `outputs/day1_benchmark/stability_sweep/stability_sweep_events.jsonl`
- `outputs/day1_benchmark/stability_sweep/stability_sweep.log`
- `outputs/day1_benchmark/stability_sweep/n_runs_<N>.json`
- `outputs/day1_benchmark/stability_sweep/stability_sweep_summary.csv`
- `outputs/day1_benchmark/stability_sweep/stability_sweep_summary.json`

Gợi ý smoke test nhanh trước khi chạy full grid:

- `python src/mapr2026_v3/ic_stability_sweep.py --run-grid 20,40 --max-nodes 300 --n-jobs -1`

## P2 consensus supplementary branch

Giữ nguyên `classification_labels.parquet` (compatibility) và xuất thêm nhánh
consensus theo winner policy B:

- `python src/mapr2026_v3/ic_policy_compare.py --n-jobs -1`
- `python src/mapr2026_v3/ic_export_consensus_labels.py`

Định nghĩa lock để tránh drift:
- `y_top10_consensus == policy_b == (seed_vote_count >= consensus_k)`
- `vote_count` phải lấy từ `seed_vote_count` trong `policy_labels_abc.parquet`

Sinh các artifact:
- `data/processed/classification_labels_consensus.parquet`
- `outputs/day1_benchmark/policy_compare/classification_labels_consensus_report.json`

Để đo uncertainty theo nhãn consensus mà không ghi đè artifact cũ:

- `python src/mapr2026_v3/ic_label_uncertainty.py --cls data/processed/classification_labels_consensus.parquet --out-json outputs/day1_benchmark/ic_label_uncertainty_consensus.json --out-ic-ci data/processed/ic_scores_primary_with_ci_consensus.parquet`

## Regression stability report

Tạo report stability cho regression target từ sweep outputs:

- `python src/mapr2026_v3/ic_regression_stability.py --spearman-threshold 0.90`

Artifact:
- `outputs/day1_benchmark/ic_regression_stability.json`
