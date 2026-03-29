# Predictions Before Running Experiments

Purpose: pre-register expected outcomes before seeing final results.
Rule: update this file only before a major run, and keep all old entries.

---

## Metadata

- Date: 2026-03-28
- Authors: Group 9
- Snapshot ID: StageC_PreRun_v1
- Config hash: pending
- Notes:
  - This file is locked before running Stage 4+ experiments.
  - Any post-run interpretation must be recorded in docs/live_interpretation_log.md.

---

## RQ1

- Expected:
  - Spearman(views, PageRank) > Spearman(views, Betweenness)
- Reasoning:
  - Views and PageRank both reflect popularity/prestige signals.
- Falsification condition:
  - If Spearman(views, Betweenness) >= Spearman(views, PageRank).
- If wrong, possible explanations to investigate:
  - PageRank captures recursive structure that views do not.
  - Data quality or preprocessing changed rank behavior.
  - Outlier nodes dominate one metric.

## RQ2

- Expected:
  - mean_reach(Hidden) > mean_reach(Overrated)
  - Effect size is medium or larger.
  - Rank-biserial correlation (effect_size_r) > 0.20.
  - Cliff's delta > 0.30.
- Reasoning:
  - Hidden nodes should have stronger structural spread potential than view-heavy but structurally weak nodes.
  - If SIS truly captures latent spreading potential, Hidden should outperform Overrated under the same IC settings.
- Falsification condition:
  - Hidden <= Overrated in mean reach.
  - Or 95% CI overlap with near-zero practical effect.
  - Or corrected p-value is non-significant and effect size is negligible.
- If wrong, possible explanations to investigate:
  - Mutual-friendship graph is weak proxy for diffusion pathways.
  - IC parameter choice creates floor or ceiling effects.
  - Typology threshold causes unstable group composition.
  - Hidden label may partially reflect k-shell mechanics without extra predictive value.

## RQ3

- Expected ranking:
  - k-shell >= Betweenness > PageRank > Degree > Views > Random
- Reasoning:
  - Core position and brokerage should better predict spread than surface popularity.
  - k-shell should remain competitive because core membership influences multi-hop reach.
- Falsification condition:
  - Views or random baseline is competitive with structural strategies.
  - Strategy ordering is unstable under robustness reruns.
- If wrong, possible explanations to investigate:
  - Chosen p value in IC is poorly calibrated.
  - Strategy implementation mismatch.
  - Graph sparsity and component structure dominate outcomes.
  - Mutual edges may not encode actual influence pathways on Twitch.

## Claim Readiness Rules (Pre-registered)

- RQ2 claim is promotable to abstract only if all are true:
  - mean_reach(Hidden) > mean_reach(Overrated)
  - p_corrected_bh < 0.05
  - effect_size_r >= 0.20
- RQ3 claim is promotable to abstract only if all are true:
  - Structural strategy ranking beats views and random by clear margin.
  - Ranking is directionally consistent in robustness rerun(s).
  - Confidence intervals do not collapse to near-identical performance.

## RQ4

- Expected:
  - Hidden class F1 remains low for all LR baselines.
- Reasoning:
  - Surface metrics should struggle to detect structurally hidden influence patterns.
- Falsification condition:
  - Hidden F1 becomes high and stable across splits.
- If wrong, possible explanations to investigate:
  - Feature leakage from labels.
  - Split/stratification artifact.
  - One metric proxy is stronger than expected.

---

## Integrity Checklist

- [ ] Predictions written before full experiment run.
- [ ] Falsification condition provided for each RQ.
- [ ] At least 2 alternative explanations written for each RQ.
- [ ] Config hash recorded.
- [ ] Date and authors recorded.
