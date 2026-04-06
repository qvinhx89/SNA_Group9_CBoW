# Day1 Handoff Brief (Person 2/3 Consumers)

Date: 2026-04-06
Owner: Person 1
Version: person1_day1_20260406_p1_day1_v1

## 1) What is frozen and safe to reuse now

- Split mask is frozen and checksum-locked:
  - path: `data/processed/split_masks.parquet`
  - sha256: `005de40762f6c75e4df66a53efeaa883d126d52abd5c4af0224d736992362104`
  - freeze manifest: `outputs/day1_benchmark/split_freeze_manifest.json`
- Versioned package manifest:
  - `outputs/handoffs/person1_day1_20260406_p1_day1_v1/manifest.json`

## 2) How to consume (must-follow)

1. Do not create your own split.
2. Always load split using the shared artifact (`split_masks.parquet`).
3. Use `regression_targets.parquet` as primary target for current model comparison.
4. Use `classification_labels.parquet` (`y_top10`) as provisional only.

## 3) Why y_top10 is provisional

Evidence from official P0 checks:
- Stability report: `outputs/day1_benchmark/ic_label_stability.json`
  - jaccard_mean = 0.3069298298144156
  - jaccard_min = 0.3020833333333333
  - threshold (required) = 0.85
- Uncertainty report: `outputs/day1_benchmark/ic_label_uncertainty.json`
  - boundary nodes (CI crosses top-10 threshold) = 995/5000
  - boundary among positive labels = 415/500

Interpretation:
- Binary top-10 label is sensitive to MC seed noise near threshold.
- Continuous IC target is more stable for immediate downstream experiments.

## 4) Recommended reporting language

- "Runtime decision and one-hop branch are locked for planning."
- "Regression metrics are computed on frozen split and are handoff-ready."
- "Classification metrics on y_top10 are provisional pending stronger cross-seed stability."
