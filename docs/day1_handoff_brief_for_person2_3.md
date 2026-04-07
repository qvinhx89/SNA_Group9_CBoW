# Day1 Handoff Brief (Person 2/3 Consumers)

Date: 2026-04-06
Owner: Person 1
Mode: Option B (provisional, governance-locked)
Active package baseline: person1_day1_20260407_p1_day1_v3e_optionB_lockstep

## 1) Why this mode exists

- Hard gate currently does not pass.
- This is not treated as a hidden pass.
- Team progress continues under explicit governance and immutable artifacts.

## 2) Lockstep rules (must-follow)

1. Use exactly one handoff package version per experiment cycle.
2. Do not create local train/test splits.
3. Keep canonical and supplementary branches separated.
4. Any change request must create a new version tag (no overwrite).

## 3) Mandatory data sources

- Shared split (single source of truth):
  - `data/processed/split_masks.parquet`
  - `outputs/day1_benchmark/split_freeze_manifest.json`
- Canonical compatibility labels:
  - `data/processed/classification_labels.parquet`
- Supplementary consensus labels:
  - `data/processed/classification_labels_consensus.parquet`
  - `outputs/day1_benchmark/policy_compare/classification_labels_consensus_report.json`
- Primary modeling target:
  - `data/processed/regression_targets.parquet`

## 4) Consumer guidance by role

Person 2:
- Keep typology axis by M0 rule on continuous thresholds.
- Do not replace canonical branch with consensus branch in compatibility outputs.

Person 3:
- Binary evaluation can use consensus branch.
- For strict binary claims, exclude uncertain nodes (`is_uncertain=1` or `vote_count=1`).
- Always state evaluation scope (all nodes vs non-uncertain subset).

## 5) Reporting language (recommended)

- "This cycle uses Option B provisional governance with immutable handoff artifacts."
- "Regression is treated as primary objective; binary is supplementary and uncertainty-aware."
- "All experiments reuse the frozen split and declared handoff version."

## 6) Team acknowledgement checklist

- [ ] I confirm using one declared handoff version only.
- [ ] I confirm not re-splitting locally.
- [ ] I confirm canonical vs consensus branches are not mixed silently.
- [ ] I confirm uncertainty handling is declared for binary metrics.
