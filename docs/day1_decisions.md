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
python src/mapr2026_v3/ic_pilot_diagnostics.py
python src/mapr2026_v3/ic_label_uncertainty.py
python src/mapr2026_v3/ic_label_stability.py --n-runs 150 --mc-seeds 0,1,2 --n-jobs -1 --jaccard-threshold 0.85
python src/mapr2026_v3/freeze_day1_handoff.py --version-tag 20260406_p1_day1_v2_final
# nếu quality gate fail nhưng vẫn cần bàn giao tạm:
python src/mapr2026_v3/freeze_day1_handoff.py --version-tag 20260406_p1_day1_v2_provisional --quality-mode provisional
```

Generated artifacts:
- outputs/day1_benchmark/ic_pilot_diagnostics.json
- outputs/day1_benchmark/ic_label_uncertainty.json
- outputs/day1_benchmark/ic_label_stability.json
- outputs/day1_benchmark/quality_gate_report.json
- outputs/day1_benchmark/split_freeze_manifest.json
- data/processed/ic_scores_primary_with_ci.parquet
- outputs/handoffs/person1_day1_20260406_p1_day1_v2_provisional/manifest.json

Observed results:
- Stability (3 independent MC seeds, 150 runs/seed, 5000 labeled nodes):
	- jaccard_mean: 0.3069298298144156
	- jaccard_min: 0.3020833333333333
	- jaccard_pass_threshold (>= 0.85): False
	- spearman_mean: 0.685383690615586
- Pilot diagnostics (200 nodes, 50 runs/node):
	- cv_score: 0.21087886200621908 (threshold > 0.3 => fail)
	- rank_stability: 0.673199023270778
	- jaccard_stability (pilot A vs B): 0.14285714285714285
- Uncertainty around top-10 threshold (77.67000000000003):
	- boundary nodes (CI crosses threshold): 995 / 5000 (19.9%)
	- boundary among y_top10=1: 415 / 500
	- ambiguous nodes (0.1 < p_above_threshold < 0.9): 775 / 5000 (15.5%)

Quality gate behavior (automated):
- `freeze_day1_handoff.py` ở mode `final` sẽ fail nếu không đạt ngưỡng:
	- `cv_score > 0.3`
	- `jaccard_mean >= 0.85`
	- `jaccard_min >= 0.80`
- Trạng thái fail được ghi vào `outputs/day1_benchmark/quality_gate_report.json`.

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
	- `outputs/handoffs/person1_day1_20260406_p1_day1_v2_provisional/`
	- manifest includes SHA256 + quality gate metadata (mode/provisional status)

## 9) Handoff message for teammates (copy/paste)

Team update (Person 1):
- Day-1 planning artifacts are locked and versioned (`person1_day1_20260406_p1_day1_v2_provisional`).
- Please consume split strictly from `data/processed/split_masks.parquet` (no local re-splitting).
- Use `regression_targets.parquet` as primary target for current experiments.
- `classification_labels.parquet` (`y_top10`) is available but provisional due to low cross-seed stability; avoid making strong final claims from binary label alone.
- Uncertainty and stability evidence is attached in `outputs/day1_benchmark/ic_label_uncertainty.json` and `outputs/day1_benchmark/ic_label_stability.json`.

## 10) Label Stability Decision (P2 consensus branch)

Context:
- Full sweep (150,300,500,800,1200) still fails hard Jaccard gate for binary top-10 labels.
- Policy comparison (A/B/C) selected `B_consensus_top10` as winner under noise + typology constraints.

Locked operating rule:
- Regression target (`regression_targets.parquet`) remains PRIMARY for downstream modeling.
- Binary top-10 remains SECONDARY/provisional.
- Keep canonical `classification_labels.parquet` unchanged for backward compatibility.
- Export consensus branch as supplementary artifact (do not overwrite canonical labels).
- Locked definition: `y_top10_consensus == policy_b == (seed_vote_count >= consensus_k)`.
- `vote_count` in consensus artifact must come from MC-seed voting (`seed_vote_count`), not `policy_a + policy_b + policy_c`.

Official commands:

```powershell
python src/mapr2026_v3/ic_policy_compare.py --n-jobs -1
python src/mapr2026_v3/ic_export_consensus_labels.py
python src/mapr2026_v3/ic_label_uncertainty.py --cls data/processed/classification_labels_consensus.parquet --out-json outputs/day1_benchmark/ic_label_uncertainty_consensus.json --out-ic-ci data/processed/ic_scores_primary_with_ci_consensus.parquet
python src/mapr2026_v3/ic_regression_stability.py --spearman-threshold 0.90
python src/mapr2026_v3/freeze_day1_handoff.py --version-tag 20260406_p1_day1_v3_provisional_consensusB --quality-mode provisional
```

Expected supplementary artifacts:
- `data/processed/classification_labels_consensus.parquet`
- `outputs/day1_benchmark/policy_compare/classification_labels_consensus_report.json`
- `outputs/day1_benchmark/ic_label_uncertainty_consensus.json`
- `outputs/day1_benchmark/ic_regression_stability.json`
- `outputs/day1_benchmark/policy_compare/policy_comparison_summary.csv`
- `outputs/day1_benchmark/policy_compare/policy_comparison_summary.json`

Consumer guidance:
- Person 3 binary evaluation: use consensus labels and exclude `is_uncertain=1` when reporting binary metrics.
- Person 2 typology axis: keep M0 top-10 rule on continuous IC+views thresholds (no M0 lock change).

## 11) Option B Execution Lock (Official from Person 1)

Status:
- Option B is activated for operational handoff.
- Meaning: keep truthful gate status (`pass_all=false`) while allowing downstream progress under fixed governance.
- Active lockstep package for current cycle: `person1_day1_20260408_p1_day1_v3g_optionB_lockstep`.

Additional alignment:
- `one_hop_correlation.json` now includes `jaccard_at_10pct` and `ndcg_at_10pct` for stricter top-k agreement reporting.

Non-negotiable lockstep rules for Person 2 and Person 3:
1. Use exactly one handoff package version per experiment cycle.
2. Do not re-split data locally.
3. Do not replace canonical labels with consensus labels in compatibility pipelines.
4. Report binary metrics with uncertainty handling (exclude uncertain nodes when claiming strict binary performance).

Mandatory consume protocol:
- Canonical compatibility branch:
	- `data/processed/classification_labels.parquet`
- Supplementary consensus branch:
	- `data/processed/classification_labels_consensus.parquet`
	- `outputs/day1_benchmark/policy_compare/classification_labels_consensus_report.json`
- Shared split (single source of truth):
	- `data/processed/split_masks.parquet`

Operational note:
- If any team member needs changes, create a new version tag and freeze a new package.
- Never overwrite an existing handoff directory.

## 12) Stability Explanation + Narrative Lock (Plan Update Sync)

New required artifact (triggered because `jaccard_stability < 0.85`):
- `outputs/day1_benchmark/stability_explanation.json`
- Sources used:
	- `outputs/ic_feasibility/phase1_community_overlap.json`
	- `outputs/ic_feasibility/phase2_threshold_analysis.json`
- Current extracted result:
	- `pct_communities_spanning_boundary = 0.842`
	- `mean_gap_to_noise = 0.0023928571428571436`
	- `n_thresholds_tested = 28`
	- `interpretation = structural`

Runtime table alignment for downstream speedup analysis:
- Added `mc_ic_labeling` row into `outputs/mapr2026_v3_results/runtime_breakdown.csv`
- Rule used: `inference_sec_full_graph = projected_total_hours * 3600` from `ic_runtime_benchmark.json`

Narrative lock (paper/report wording):
- Regression target (`y = log1p(ic_score_mean)`) is PRIMARY by formulation for simulation-derived continuous IC signal.
- Binary top-10 remains supplementary/provisional due to threshold sensitivity and uncertainty around boundary nodes.
- Option B is an operational governance mode, not a claim that regression is only a fallback.
