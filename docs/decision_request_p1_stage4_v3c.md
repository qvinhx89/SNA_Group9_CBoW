# Decision Request - Stage 4 Governance (Person 1)

- DR ID: DR-P1-20260406-01
- Date: 2026-04-06
- Owner: Person 1
- Scope: Day1 IC stability, quality gate acceptance, downstream consume rules
- Decision deadline requested: 2026-04-07 12:00

## 1) Decision Summary

Team can not close Stage 4 with hard-gate pass under current thresholds, even after reproducible reruns and policy cleanup.  
This request asks approvers to choose an official governance path so Person 2 and Person 3 can continue without contract drift.

Requested decision outcome:

1. Confirm which acceptance mode is official for this cycle (Option A/B/C below).
2. Confirm downstream label consume rules (canonical vs consensus branch).
3. Confirm whether quality gate remains strict or is accepted as provisional for this version.

## 2) Current Evidence (Pre-filled)

### 2.1 Quality gate status

- Observed: `pass_all = false`
- Source: `outputs/day1_benchmark/quality_gate_report.json`
- Freeze package status also records fail: `outputs/handoffs/person1_day1_20260406_p1_day1_v3c_provisional_consensusB/manifest.json`

### 2.2 Binary label stability status

- Observed: `jaccard_mean = 0.3069298298`, `jaccard_min = 0.3020833333`
- Target thresholds: `jaccard_mean >= 0.85`, `jaccard_min >= 0.80`
- Source: `outputs/day1_benchmark/stability_sweep/stability_sweep_summary.json`

### 2.3 Regression stability status

- Observed best: `spearman_mean = 0.8267109303`
- Target threshold used: `spearman_mean >= 0.90`
- Source: `outputs/day1_benchmark/ic_regression_stability.json`

### 2.4 Policy-definition drift has been fixed

- Locked definition: `policy_b == (seed_vote_count >= consensus_k)`
- Source: `outputs/day1_benchmark/policy_compare/policy_comparison_summary.json`
- Consensus export consistency report:
	`outputs/day1_benchmark/policy_compare/classification_labels_consensus_report.json`

## 3) Constraints To Respect

1. Do not silently change acceptance thresholds without approval.
2. Do not break canonical downstream compatibility.
3. Keep artifacts reproducible and versioned via immutable freeze package.
4. Keep explicit documentation for audit and final report defense.

## 4) Decision Options

## Option A - Keep strict hard-gate, continue compute until pass

Definition:

- Keep current strict thresholds exactly as-is.
- Continue increasing runs/seeds or changing simulation budget until pass.

Pros:

- No governance interpretation needed.
- Closest to original strict contract.

Cons:

- High runtime/cost with unclear probability of reaching thresholds.
- Blocks Person 2/3 pipeline progress.

Approval implications:

- Approvers accept delay and extra compute budget.

## Option B - Provisional acceptance with locked governance (RECOMMENDED)

Definition:

- Accept current package as `provisional` for this cycle.
- Set `regression` as primary target for modeling progress.
- Keep `binary` as supplementary branch with uncertainty-aware handling.
- Use locked consensus-B supplementary labels where needed.

Pros:

- Unblocks downstream work immediately.
- Preserves scientific honesty (no fake pass).
- Keeps all artifacts reproducible and auditable.

Cons:

- Requires explicit governance note and consume rules.

Approval implications:

- Approvers explicitly accept provisional mode for this version tag.
- Approvers accept documented consume rules for Person 2/3.

## Option C - Change thresholds now and declare pass

Definition:

- Lower thresholds (or alter gate criteria) immediately for this cycle.

Pros:

- Can produce pass status quickly.

Cons:

- Highest governance risk if not justified rigorously.
- Harder to defend if interpreted as moving goalposts.

Approval implications:

- Requires explicit threshold-change rationale and sign-off from all approvers.

## 5) Requested Approval Criteria

This request is approved only when all criteria are checked:

- [ ] One option is selected (A/B/C) with approver names.
- [ ] Downstream consume rules are accepted.
- [ ] Versioning rule is accepted (new version tag for any rerun).
- [ ] Documentation update requirement is accepted.

Mandatory consume-rule decision (pick one final wording):

- [ ] Canonical compatibility branch: use `data/processed/classification_labels.parquet`
- [ ] Supplementary analysis branch: use `data/processed/classification_labels_consensus.parquet`
- [ ] For strict binary eval: exclude uncertain nodes via `is_uncertain` or `vote_count`

## 6) Recommended Resolution (Person 1 proposal)

Person 1 recommends **Option B** for this cycle.

Reason:

1. Current hard thresholds are not met by measured evidence.
2. Technical drift issue has already been fixed and validated.
3. Provisional governance allows progress without falsifying gate status.
4. This preserves traceability and report defensibility.

## 7) Execution Plan After Approval

If Option B is approved, Person 1 will execute:

1. Update governance note in `docs/day1_decisions.md` with approved wording.
2. Publish consume rules to Person 2/3 using package:
	 `outputs/handoffs/person1_day1_20260406_p1_day1_v3c_provisional_consensusB`
3. If any requested adjustments are made, freeze new immutable package with new tag.

If Option A is approved, Person 1 will execute:

1. Submit compute plan (run-grid, expected runtime, budget).
2. Re-run sweeps and regenerate evidence.
3. Re-issue freeze package with new version tag.

If Option C is approved, Person 1 will execute:

1. Update threshold contract in docs with justification and approver signatures.
2. Recompute gate report under approved new thresholds.
3. Freeze new immutable package with threshold-change note.

## 8) Risk If No Decision

1. Team blocks continue due to unresolved acceptance criteria.
2. Person 2/3 may consume labels inconsistently.
3. Final report defense risk increases due to undocumented governance gap.


