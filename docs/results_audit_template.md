# Results Audit Template

Purpose: test each claim before writing final report text.
Rule: no claim enters abstract without a completed audit row.

---

## How To Use

1. Add one section per claim.
2. Fill exact evidence with artifact path and numbers.
3. State boundary conditions and alternative explanations.
4. Rate strength honestly: strong, moderate, weak.

---

## Claim Audit Entries

### CLAIM 01

- Claim: Hidden influencers achieve higher mean cascade reach than Overrated under IC simulation.
- Evidence artifact path:
	- outputs/stage4_single_seed/rq2_hidden_validation.csv
	- outputs/stage4_single_seed/rq2_hidden_validation.json
- Evidence numbers:
	- mean_reach_hidden =
	- mean_reach_overrated =
	- delta_mean =
	- ci95_hidden =
	- ci95_overrated =
- Statistical support:
	- p_raw =
	- p_corrected_bh =
	- effect_size_r =
	- cliffs_delta =
- Conditions where claim holds:
	- IC setting fixed at calibrated p.
	- Group definitions from typology threshold in config.
- Conditions where claim may fail:
	- Different graph snapshot or interaction-based graph.
	- Different diffusion model assumptions.
- Alternative explanation:
	- k-shell component in SIS may partly drive expected spread advantage.
- Strength rating: strong | moderate | weak
- Decision: abstract | main results | appendix only

### CLAIM 02

- Claim: Structural seed strategies (k-shell, betweenness, pagerank, degree) outperform views and random for multi-seed IC reach.
- Evidence artifact path:
	- outputs/stage5_multi_seed/rq3_strategy_benchmark.csv
	- outputs/stage5_multi_seed/rq3_strategy_benchmark.json
- Evidence numbers:
	- mean_reach_by_strategy =
	- rank_order =
	- delta_vs_views =
	- delta_vs_random =
- Statistical support:
	- pairwise p_corrected_bh =
	- effect_size_r by key pair =
- Conditions where claim holds:
	- Same k seeds and same calibrated p across strategies.
	- Same run budget and random seed policy.
- Conditions where claim may fail:
	- Under different p regimes or time-varying influence models.
	- Under behavioral interaction networks instead of mutual friendship.
- Alternative explanation:
	- Structural strategies may benefit from graph construction artifacts.
- Strength rating: strong | moderate | weak
- Decision: abstract | main results | appendix only

### CLAIM 03

- Claim:
- Evidence artifact path:
- Evidence numbers:
- Statistical support:
- Conditions where claim holds:
- Conditions where claim may fail:
- Alternative explanation:
- Strength rating: strong | moderate | weak
- Decision: abstract | main results | appendix only

---

## Final Gate

- [ ] Every abstract claim has strength strong or moderate.
- [ ] Every number is traceable to an artifact.
- [ ] No causal language used for correlational evidence.
- [ ] At least one plausible confound listed per claim.
