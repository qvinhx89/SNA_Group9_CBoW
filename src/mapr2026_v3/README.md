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
