# Day-1 Decisions (Official)

Date: 2026-04-06
Owner: Person 1

## 1) Official benchmark run

Command:

```powershell
python src/mapr2026_v3/day1_benchmark.py --n-jobs -1 --seed 42 --bench-nodes 100 --bench-runs 50 --pilot-nodes 200 --pilot-runs 50 --target-n-sample 5000 --out-dir outputs/day1_benchmark
```

Artifacts:
- outputs/day1_benchmark/ic_runtime_benchmark.json
- outputs/day1_benchmark/one_hop_correlation.json

## 2) Runtime benchmark result

- per_sim_ms: 0.5507069587707519
- projected_total_hours (5000x200): 0.15297415521409777
- Decision gate: projected runtime < 4h
- Locked decision: N_seeds = 5000, N_runs = 200

## 3) One-hop correlation result

- spearman_rho: 0.7391903714947583
- p_value: 7.814871409002748e-36
- Decision gate: rho < 0.8
- Locked narrative branch: viable_gnn

## 4) Critical note (post-check)

Stability runs were executed in:
- outputs/day1_critical/run1_seed42/
- outputs/day1_critical/run2_seed42/
- outputs/day1_critical/run3_seed123/
- outputs/day1_critical/run4_seed777/

Observed range:
- per_sim_ms: 0.4758413314819336 to 2.068499279022217
- spearman_rho: 0.7345352544434723 to 0.7563291838310282
- decision_branch across runs: viable_gnn (stable)

Planning rule:
- Use median runtime for planning, not a single run.
- median_per_sim_ms: 0.6423945903778076
- median_projected_hours (5000x200): 0.17844294177161318

## 5) Frozen benchmark config for fair future comparison

Keep fixed for Day-1 reruns:
- --bench-nodes 100
- --bench-runs 50
- --pilot-nodes 200
- --pilot-runs 50
- --target-n-sample 5000
- --n-jobs -1
- --seed 42 (official log run)

For stability checks, only change seed (recommended: 123, 777).

## 6) Notes for paper limitations

- LCC condition: active graph currently has pct_lcc = 100.0 (from outputs/stage0_data_quality/lcc_report.json).
- Dead-account audit available in outputs/stage0_data_quality/dead_account_report.json and must be reported in limitations section.

## 7) P0 pre-handoff quality gate (for Person 2/3 consume)

Run commands (official P0 checks):

```powershell
python src/mapr2026_v3/ic_label_uncertainty.py
python src/mapr2026_v3/ic_label_stability.py --n-runs 150 --mc-seeds 0,1,2 --n-jobs -1 --jaccard-threshold 0.85
python src/mapr2026_v3/freeze_day1_handoff.py --version-tag 20260406_p1_day1_v1
```

Generated artifacts:
- outputs/day1_benchmark/ic_label_uncertainty.json
- outputs/day1_benchmark/ic_label_stability.json
- outputs/day1_benchmark/split_freeze_manifest.json
- data/processed/ic_scores_primary_with_ci.parquet
- outputs/handoffs/person1_day1_20260406_p1_day1_v1/manifest.json

Observed results:
- Stability (3 independent MC seeds, 150 runs/seed, 5000 labeled nodes):
	- jaccard_mean: 0.3069298298144156
	- jaccard_min: 0.3020833333333333
	- jaccard_pass_threshold (>= 0.85): False
	- spearman_mean: 0.685383690615586
- Uncertainty around top-10 threshold (77.67000000000003):
	- boundary nodes (CI crosses threshold): 995 / 5000 (19.9%)
	- boundary among y_top10=1: 415 / 500
	- ambiguous nodes (0.1 < p_above_threshold < 0.9): 775 / 5000 (15.5%)

Interpretation:
- Day-1 runtime decision and one-hop branch are reproducible enough for planning.
- Binary label `y_top10` is currently high-noise and must be treated as provisional.
- Continuous target `y=log1p(ic_score_mean)` can be consumed now, but report uncertainty-aware interpretation.

## 8) Freeze + versioning status

- Split artifact is frozen via checksum manifest:
	- file: `data/processed/split_masks.parquet`
	- sha256: `005de40762f6c75e4df66a53efeaa883d126d52abd5c4af0224d736992362104`
	- rule: test_frac=0.20, degree_quintile stratified, seed=42
- Versioned handoff package:
	- `outputs/handoffs/person1_day1_20260406_p1_day1_v1/`
	- manifest includes SHA256 for all transferred files (12 files)

## 9) Handoff message for teammates (copy/paste)

Team update (Person 1):
- Day-1 planning artifacts are locked and versioned (`person1_day1_20260406_p1_day1_v1`).
- Please consume split strictly from `data/processed/split_masks.parquet` (no local re-splitting).
- Use `regression_targets.parquet` as primary target for current experiments.
- `classification_labels.parquet` (`y_top10`) is available but provisional due to low cross-seed stability; avoid making strong final claims from binary label alone.
- Uncertainty and stability evidence is attached in `outputs/day1_benchmark/ic_label_uncertainty.json` and `outputs/day1_benchmark/ic_label_stability.json`.
