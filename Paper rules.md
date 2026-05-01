# Academic Writing Rules for MAPR 2026 Paper
## Canonical Writing Protocol
### Primary source for paper drafting and skeleton alignment

---

## How to Use This File

This is the **canonical writing style rulebook** for the paper — not the frozen-values reference.

**Two-file authority model:**

| Question type | Go to |
|---|---|
| What number/ρ/CI/comparator to use? | **`Paper guide.md`** — frozen values, paste-ready English sentences |
| How to write it (style, tone, structure)? | **This file (`Paper rules.md`)** — principles, language patterns, section rules |
| What claim is allowed? | Both files together — guide provides the artifact; rules govern the framing |

When the two files appear to conflict on a factual matter (e.g., a number), **`Paper guide.md` wins** — it carries the frozen artifacts. When they conflict on a style or framing principle, **this file wins**.

> **Note on embedded frozen examples in rules:** Some rules (E5, E6, F3, E12a, Section 11) contain paste-ready template sentences with actual frozen ρ/CI/timing values. This is intentional: style-sensitive contexts (contrast paragraphs, table captions, runtime comparisons) benefit from having the frozen wording immediately adjacent to the governing rule. These embedded values are **copies** of the frozen numbers in `Paper guide.md` — the authoritative source for all frozen numbers remains `Paper guide.md` Sections 11–13. If frozen values change, update `Paper guide.md` first, then update these rule examples to match.

`Paper skeleton.md` should follow these rules, not define them. Use the skeleton only as a drafting scaffold after it has been aligned to this rulebook.

Workflow:

1. Read `Paper guide.md` §1 (identity) and §4 (claims + frozen evidence) first to internalize the paper's core argument.
2. Then read this file's core philosophy (§0) and section rules (§4–§9) before drafting each section.
3. Treat every example sentence here as a template, not as a pre-committed claim.
4. Finalize the title, abstract, contribution bullets, and contrast paragraph only after the relevant artifacts are frozen — they are now frozen; see `Paper guide.md` §6.4.

This file exists to help the team write prose that is:

- natural rather than mechanical,
- calm rather than defensive,
- specific rather than vague,
- and reviewer-defensible rather than outcome-driven.

The paper should read as a comparative study, not a victory story.

---

# 0. Core Writing Philosophy

## Rule G0.1 - Write the paper as a comparative study

Do not write the paper as:

- "GNN is better than baselines."
- "HSCC is the correct diffusion model."
- "We found the true influencers."

Write it as:

> A comparative study of how IC operationalization changes surrogate learnability and baseline competitiveness.

Core identity:

- `A0` = structural reference regime
- `HSCC` = domain-informed source-community regime
- GNN value = regime-dependent, not universal

## Rule G0.2 - Every substantive claim must map to an artifact

Before writing any result sentence, ask:

> If a reviewer challenges this sentence, which file do we open?

If there is no artifact, the claim does not belong in the paper.

Typical mapping:

- descriptive claims -> tables / CSVs
- inferential claims -> bootstrap JSONs
- methodological claims -> formulas + protocol text
- scope / caveat claims -> construct-validity and limitation sections

## Rule G0.3 - Comparator choice must be regime-aware

Comparators are not fixed across all regimes.

- Under `A0`, the primary comparator is `degree`.
- Under `HSCC`, the primary comparator is the strongest flat non-graph baseline under matched feature access.

Do not write:

> GNN outperforms degree under HSCC

unless degree is being used only as contextual evidence of regime shift.

Preferred sentence:

> Under HSCC, degree is no longer the relevant comparator; the strongest matched flat baseline becomes the appropriate reference.

## Rule G0.4 - Regression is the primary formulation

Do not frame regression as a rescue move after classification failed.

Preferred framing:

> Regression is the principled formulation for a simulation-derived continuous target. Binary-label instability provides additional empirical justification.

## Rule G0.5 - Tone must be precise, calm, and non-defensive

Prefer:

- "we evaluate"
- "we compare"
- "we find"
- "our results suggest"
- "consistent with"
- "under A0"
- "under HSCC"

Avoid:

- "obviously"
- "clearly superior"
- "surprisingly"
- "groundbreaking"
- "proves"
- "state-of-the-art"

## Rule G0.6 - Separate claim state from claim content

Use the following provenance discipline throughout the paper. `Paper skeleton.md` should mirror it:

- `FROZEN` - safe for final paper claims
- `PRELIMINARY` - usable in working drafts with explicit hedging
- `PLACEHOLDER` - not for final prose

Practical rules:

- Abstract should use frozen numbers only. If frozen numbers are not ready, keep the abstract in draft mode rather than writing submission-ready prose from preliminary results.
- Body text may reference preliminary numbers in an internal draft, but must hedge them.
- Placeholders stay as author notes, never as final prose.

---

# 1. Global Style Rules

## Rule S1 - Use first person plural

Preferred:

- "we propose"
- "we evaluate"
- "we compare"
- "we find"

Avoid overusing passive voice such as:

- "it was found that"
- "it can be seen that"

## Rule S2 - Distinguish three levels of claim

### Observation

> Degree achieves Spearman rho = 0.826 under A0.

### Supported interpretation

> This suggests that A0 induces a degree-dominated approximation regime.

### General takeaway

> These results indicate that surrogate learnability depends on the diffusion operationalization rather than on architecture choice alone.

Do not jump from observation to sweeping conclusion in one step.

## Rule S3 - Keep quantitative formatting consistent

Recommended precision:

- Spearman / NDCG / P@10: `3` decimals
- runtime in seconds: `1` decimal if `>= 1 s`, up to `3` decimals if `< 1 s`
- speedup: integer
- sample size: exact integer

Examples:

- `rho = 0.826`
- `NDCG@10 = 0.857`
- `480.3 s`
- `0.086 s`  ← headline runtime uses `hscc,gnn_raw_attr` in `runtime_breakdown.csv`
- `5,590x`   ← speedup for the same row vs `mc_ic_labeling`; round to `5,500x` in paper prose
- `5,000 labeled nodes`

Never round in a direction that flatters the paper.

**Delta (Δ) and confidence interval (CI) notation — use these formats consistently:**

- Delta: `Δρ = +0.033` (sign always explicit; positive = GNN improvement over comparator)
- CI: `95% CI [+0.021, +0.044]` (square brackets; sign explicit; lower bound first)
- Combined: `Δρ = +0.033, 95% CI [+0.021, +0.044]`
- For negative results: `Δρ = −0.018, 95% CI [−0.029, −0.008]` (use − not -)
- Equivalence bound: `δ₀ = 0.02` (not δ0 or delta0)

**Equivalence vs significance interpretation — four cases (exhaustive):**

| Case | CI pattern | Example | Interpretation | Allowed wording |
|---|---|---|---|---|
| **Case 1** | Both bounds > 0 | [+0.021, +0.044] | significantly better | `significantly improves over` |
| **Case 2** | Both bounds < 0 | [−0.029, −0.008] | significantly worse | `remains statistically below` |
| **Case 3** | Spans zero, fully within [−δ₀, +δ₀] | [−0.015, +0.010] | practically equivalent | `practically equivalent under the pre-registered bound (δ₀=0.02)` |
| **Case 4** | Spans zero, partially outside [−δ₀, +δ₀] | [−0.025, +0.010] | **inconclusive** | `results are inconclusive; the CI spans zero but extends beyond the equivalence window` |

**Important:** Case 4 is NOT equivalent and NOT significant. Do not silently drop one tail and claim equivalence. Report it honestly as inconclusive.

**For this paper's frozen results:** A0 CI = [−0.029, −0.008] is **Case 2** (both bounds negative → significantly worse) → `gnn_significantly_worse`. This is NOT Case 3 or 4 — the CI does not span zero. The |−0.029| = 0.029 > δ₀ = 0.02 note is for explaining why equivalence is also not possible, but the primary determination is: CI is fully negative → Case 2 → significantly worse. HSCC CI = [+0.021, +0.044] is **Case 1** (both bounds positive → significantly better).

## Rule S4 - Open paragraphs with logic, not with naked numbers

Weak:

> Degree achieves 0.826 under A0.

Better:

> Under the structural weighted-cascade regime, analytical baselines already capture most of the available signal. Degree achieves Spearman rho = 0.826.

## Rule S5 - Keep one paragraph centered on one point

A paragraph should mainly do one job:

- report a result,
- justify a design choice,
- interpret a contrast,
- state a limitation,
- or bridge sections.

If one paragraph tries to do all of them, split it.

## Rule S6 - Vary sentence length naturally

Dense academic prose becomes hard to read when every sentence has the same rhythm.

Good default pattern:

- one short fact sentence,
- one longer explanatory sentence,
- one short implication or bridge sentence.

Use rhythm to improve readability, not to sound decorative.

## Rule S7 - Integrate citations where they matter

Preferred:

> following the weighted-cascade parameterization of Kempe et al. [1]

Less good:

> we use weighted cascade [1]

Group citations by function, not as a pile at the end of a sentence.

## Rule S8 - Reserve inferential words for inferential evidence

Words such as `significant`, `significantly`, `equivalent`, `practically equivalent`, `better`, and `worse` should only be used when the corresponding inferential evidence is available and clearly identified.

Preferred discipline:

- use `significantly` only with a stated bootstrap CI or formal inferential result,
- use `practically equivalent` only with the pre-registered equivalence logic,
- use neutral wording such as `higher`, `lower`, `suggests`, or `preliminary` when the evidence is not yet frozen.

---

# 2. Title Rules

## Rule T1 - The title should foreground contrast, not hype

Good title patterns:

- "When Does Graph Learning Add Value Beyond Strong Baselines?"
- "A Comparative Study of IC Operationalizations for Influence Approximation"
- "Regime-Dependent Surrogate Learning for IC Approximation"

Avoid:

- "A Novel GNN Framework for Power User Detection"
- "A Superior GNN Approach for Influence Prediction"
- "Graph Intelligence for Real Influence Discovery"

## Rule T2 - Avoid `power user` unless tightly defined

Safer alternatives:

- influence approximation
- simulation-defined influence potential
- IC surrogate learning
- operationalization-dependent learnability

---

# 3. Abstract Rules

## Rule A1 - Open with the problem, not with the method

Weak:

> We propose a GNN surrogate for IC simulation.

Better:

> Monte Carlo Independent Cascade (MC-IC) simulation provides a principled operationalization of influence potential but is computationally expensive at scale.

## Rule A2 - Mention dataset scale once

Use the dataset name and scale once in the abstract. Do not repeat it.

Example:

> on the Twitch Gamers social network (168K nodes, 6.8M edges)

## Rule A3 - Keep the abstract to five moves

The abstract should cover:

1. problem
2. why it is difficult
3. what is being compared
4. main contrast result
5. generalizable takeaway

## Rule A4 - Match claim strength to claim state

Conditional templates:

- If bootstrap is frozen and CI excludes zero positively:  
  `X significantly improves over Y`
- If bootstrap is frozen and CI lies within the equivalence bound:  
  `X is practically equivalent to Y`
- If bootstrap is frozen and CI is fully negative:  
  `X remains statistically below Y`
- If numbers are preliminary:  
  `preliminary results suggest ...`

Do not hard-code a final outcome before the rerun is frozen.

For the final paper, the abstract should not use preliminary inferential claims.

**FROZEN CASE — Active template for this paper (use verbatim or adapt):**

*A0 (gnn_significantly_worse):*

> Under degree-coupled IC (A0), the best GNN (GCN, raw node attributes) remains statistically below degree centrality (Δρ = −0.018, 95% CI [−0.029, −0.008]), consistent with a structural ceiling imposed by the degree-coupled operationalization. This result is not a failure — it confirms that analytical degree information already saturates most of the available surrogate signal under this regime.

*HSCC (gnn_significantly_better):*

> Under source-community IC (HSCC), the best GNN (SAGE) significantly outperforms the strongest matched flat baseline — LR(degree, views, life_time, language), ρ = 0.884 — achieving ρ = 0.915 (Δρ = +0.033, 95% CI [+0.021, +0.044]), suggesting that neighborhood message passing captures residual community-propagation structure beyond what node-level source attributes alone encode.

*C3 rankloss (inferential: vs comparator; descriptive: vs standard GNN):*

> Ranking-aware training (SAGE + combined Huber + pairwise ranking loss, α=0.5) achieves ρ = 0.924, a descriptive gain of +0.009 over standard Huber training, and significantly outperforms the flat comparator by Δρ = +0.041 (95% CI [+0.030, +0.053]).

Note: "+0.009 vs standard SAGE" is descriptive (no paired bootstrap). "+0.041 vs comparator" is inferential (CI from frozen JSON). Use "significantly" only for the comparator claim.

## Rule A5 - End with an insight beyond Twitch

The last sentence should tell the reader what they learn beyond this single dataset.

Good pattern:

> Our findings suggest that the usefulness of graph learning for influence approximation depends on the information structure induced by the diffusion operationalization, rather than on architecture choice alone.

## Rule A6 - Keep the abstract self-contained

Spell out the main acronyms on first use in the abstract even if they appear again in the body.

---

# 4. Section I - Introduction Rules

## Rule I1 - Open with application need, not textbook definition

Good opening themes:

- influence-aware recommendation
- social amplification
- simulation cost
- absence of cascade logs

Avoid:

- "Influence maximization is a classic problem..."
- "A social network is a graph..."

## Rule I2 - Express the gap as a question

Good:

> It remains unclear whether learned graph surrogates add value over the strongest valid non-graph baselines once the diffusion operationalization is fixed.

Weak:

> Prior work fails to compare against simple baselines.

## Rule I3 - Contribution bullets must be falsifiable

Good templates:

- `We show that binary top-k labels are structurally unstable (Jaccard=0.31 under A0; structural cause extends to HSCC via community topology invariance).`
- `We show that comparator choice changes across regimes.`
- `We evaluate whether GNNs add value beyond analytical and flat baselines under two operationalizations.`

Avoid contribution bullets that already assume the final sign of the result before numbers are frozen.

## Rule I4 - Do not overuse `novel`

If used at all, reserve it for one high-level contribution. In most cases the paper does not need the word.

## Rule I5 - Lock the paper identity early

The introduction should contain a sentence close to:

> This paper studies not whether GNNs are universally superior, but under which diffusion operationalizations graph message passing adds value beyond analytical and flat baselines.

## Rule I6 - Keep contributions compact

Three contributions is the right scale. More than that usually means the paper is trying to do too much.

---

# 5. Section II - Background / Related Work Rules

## Rule B1 - Define once, then escalate formality

Good practice:

- motivate lightly in the introduction,
- define formally in Section II,
- refer back consistently later.

Do not give two competing definitions of the same concept.

## Rule B2 - Every formula needs plain-language interpretation

After each equation, include one sentence explaining what it means in ordinary language.

Example after A0:

> This parameterization models diffusion attenuation through competing pathways: nodes with high in-degree receive proportionally lower transmission probability from each neighbor, reflecting that cascade reach is shared among multiple incoming alternatives — not an attention mechanism, but a structural normalization by the competing-path count.

Note: do NOT use the phrase "attention dilution" for A0. A0 (`p(u,v)=1/deg(v)`) is not an attention mechanism; it is a competing-pathway normalization. "Attention dilution" is technically incorrect and will invite reviewer pushback.

Example after HSCC:

> This operationalization combines source-side engagement intensity with a structural incentive for cross-community spread.

## Rule B3 - Describe GNN architectures comparatively

Do not write four disconnected mini-paragraphs.

Preferred style (four active architectures — GAT excluded):

> GraphSAGE uses mean aggregation, GCN applies symmetric normalization, GIN emphasizes multiset expressiveness through sum aggregation, and APPNP decouples transformation from propagation via K-step PPR.

If GAT must be mentioned in background context, include exactly one sentence:

> GAT was considered but excluded due to GPU memory constraints (OOM at hidden_dim=128); we therefore evaluate four architectures.

Do not present GAT as an evaluated architecture anywhere in the results, tables, or figures.

**APPNP evaluation note (frozen — required disclosure):** APPNP was run under both regimes but exhibited high seed variance (std=0.417 under A0; std=0.146 under HSCC), both exceeding the `--gnn-std-threshold 0.1` policy. APPNP results may appear in the results table for completeness, but **must not be selected as the best-architecture representative** for either regime and must be accompanied by a note that they are excluded from the best-arch pool due to instability. Preferred disclosure sentence:

> APPNP was excluded from the best-architecture comparison in both regimes due to seed variance exceeding the 0.1 threshold (std=0.417 under A0; std=0.146 under HSCC); its results are reported in the table for completeness only.

## Rule B4 - Pre-commit the evaluation protocol before main results

Make sure the reader sees the metrics, split logic, seeds, and equivalence bound before the interpretation of the main results begins.

## Rule B5 - Keep related work functional

Every sentence in Section II should either:

- define a concept needed for the paper, or
- justify an experimental choice.

If it does neither, cut it.

---

# 6. Section III - MC-IC as Comparative Operational Metric Rules

## Rule M1 - Construct validity must come first

State clearly at the start of the section:

- follower graph does not equal observed diffusion,
- the graph is used as a structural substrate,
- all claims concern simulation-defined influence approximation rather than real influence.

This must appear before the operationalization details, not buried in limitations.

## Rule M2 - Present A0 and HSCC as different assumptions, not old vs new

Do not write:

- `HSCC improves A0`
- `A0 fails and HSCC fixes it`

Preferred:

- `A0 and HSCC encode different assumptions about diffusion.`
- `A0 serves as a structural reference regime, whereas HSCC represents a domain-informed source-community regime.`

## Rule M3 - Justify each HSCC component and pre-empt the three reviewer questions

After the HSCC formula, include short justification for:

- `phi(u)` → engagement velocity / tenure-normalized activity (rank-normalized `log1p(views)/(1+life_time)`)
- community boost → cross-community exposure / structural holes
- clipping and constants → bounded stability, not calibrated to real logs

Keep these justifications short and transparent.

**Pre-empt the three reviewer questions that are nearly certain to appear:**

1. **"Why rank normalization instead of raw `views`?"**
   > `views` has a heavy-tailed distribution; rank normalization prevents a small number of extremely large accounts from dominating the source term and keeps the scale stable across reruns.

2. **"Why `log1p(views)/(1+life_time)` rather than raw `views`?"**
   > The term is a proxy for engagement velocity rather than cumulative popularity. `log1p` compresses outliers; dividing by `1+life_time` avoids rewarding accounts simply for existing longer.

3. **"Why fix `λ`, `γ`, `p_max` instead of tuning them?"**
   > HSCC is a comparative operationalization that is frozen to test regime learnability, not a parameter set optimized to maximize GNN advantage. Tuning the constants to favour GNN would conflate operationalization choice with architecture selection.

These three answers should appear in Section III of the paper (or in the Appendix if space is tight), not only in the rebuttal.

**Placement in the paper — two acceptable styles:**

*Option A (integrated prose):* Weave the answers into the HSCC formula paragraph itself, as a short parenthetical or follow-up sentence. Example: "The rank-based formulation is used because the raw views distribution is heavy-tailed — a small number of extremely active accounts would otherwise dominate the source term (see also Appendix §A)."

*Option B (compact FAQ block):* After the HSCC formula and its plain-language interpretation (Rule B2), add a 3-item bulleted block labeled "Operationalization design notes:" with one sentence per question. This is more reviewer-friendly but costs ~3–4 lines of space.

Avoid: writing the three answers only in the Conclusion or Discussion section — reviewers may question the method before reaching those sections.

## Rule M4 - Present label instability as a scientific finding when supported

If stability artifacts exist, frame instability as a methodological finding.

If they do not, use the available evidence honestly and hedge the claim.

Do not turn the section into a debugging narrative.

## Rule M5 - Use the word `structural` precisely

In this paper, `structural` can refer to:

- graph-topology structure,
- boundary instability induced by topology,
- or structure-driven signal in the labels.

When ambiguity is possible, disambiguate explicitly, for example:

- `graph-structural`
- `boundary-structural instability`
- `structure-driven signal`

## Rule M6 - Answer `why not degree?` differently by regime

Under `A0`:

- degree being strong is itself a result,
- and it helps define the structural ceiling.

Under `HSCC`:

- degree collapse marks a regime shift,
- so degree is not the meaningful main comparator.

Useful sentence:

> Whether IC adds information beyond degree depends on the operationalization.

## Rule M7 - Interpret residual variance honestly

If analysis shows only limited signal beyond local structure, say so directly.

Do not force a richer story than the numbers support.

## Rule M8 - Include a compact regime-contrast summary

This section should contain a small contrast table or equivalent summary comparing A0 and HSCC on:

- mean reach,
- variability,
- degree dependence,
- and strongest baseline behavior.

This compresses the paper's core setup into one high-value visual.

---

# 7. Section IV - Experiments / Results Rules

## Rule E1 - Setup must be complete and boring

The setup paragraph should simply list:

- dataset size,
- filtering,
- labeling budget,
- split,
- architectures,
- hyperparameters,
- seeds,
- metrics.

This paragraph is for reproducibility, not storytelling.

## Rule E2 - Separate A0 and HSCC clearly

Do not mix the two regimes in one results subsection.

Expected structure:

- `4.2` = A0
- `4.3` = HSCC
- `4.4` = regime contrast

## Rule E3 - Table first, interpretation second

Present numbers primarily through tables or figures, then interpret them in prose.

Do not turn the body text into a duplicate of the table.

## Rule E4 - CI before interpretation

Preferred:

> The 95% bootstrap CI for the Spearman difference is [...], indicating ...

Avoid:

> Model X is equivalent / better / worse. The CI is ...

The CI is the evidence; the interpretation follows from it.

## Rule E5 - Under HSCC, highlight comparator shift explicitly

At least one sentence in the paper must make the regime-shift explicit, naming the actual strongest comparator:

> Under HSCC, degree centrality drops from **rank 1 in A0** (ρ = 0.826, highest of all A0 baselines) to **near-last in HSCC** (ρ = −0.006, below random ranking); it is therefore included only as contextual evidence of regime shift, not as the primary HSCC comparator. The relevant comparator is `lr_degree_views_life_time_lang` (LR with degree, views, life_time, and language features; ρ = 0.884), the official comparator locked in the frozen bootstrap CI artifact. Note: `lr_views_life_time_lang` is within 0.001 ρ points (ρ = 0.88442 vs 0.88430) — a practical tie at the fourth decimal. The comparator choice is justified by pre-specification in the artifact, not by a clear point-estimate gap.

The sentence above is frozen — do not replace placeholders, they are already filled with actual artifact values.

Shorter version for constrained space:

> Degree centrality collapses from ρ = 0.826 (A0) to ρ = −0.006 (HSCC), marking a regime shift; the relevant HSCC comparator is LR(degree, views, life_time, language) at ρ = 0.884.

Do not write:
> Under HSCC, we compare GNN against degree.

If degree appears in the HSCC table, add a footnote: "Degree included for regime-contrast reference only (ρ = −0.006); it is not the primary HSCC comparator."

## Rule E6 - The contrast paragraph is the most important paragraph in the paper

It should explain:

- A0 is degree-coupled,
- HSCC adds source-side and graph/community structure,
- analytical baselines suffice in one regime but not necessarily in the other,
- therefore GNN value is operationalization-dependent.

Write this paragraph only after the numbers are stable.

**Frozen contrast paragraph template (paste-ready, adapt to available space):**

> Under degree-coupled IC (A0), degree centrality already captures most of the surrogate signal (ρ = 0.826), and the best GNN (GCN) remains statistically below this ceiling (ρ = 0.808, Δρ = −0.018). Under source-community IC (HSCC), degree collapses to near-random (ρ = −0.006), and the landscape restructures: source-side attributes dominate the flat baselines, with LR(degree, views, life_time, language) reaching ρ = 0.884. Graph message passing adds a further Δρ = +0.033 over this shifted comparator (SAGE, ρ = 0.915; 95% CI [+0.021, +0.044]), consistent with residual community-propagation structure that node-level attributes cannot encode. Together, these two results confirm that surrogate learnability is governed by the operationalization rather than by the architecture.

**Note on A0 result framing:** The A0 result (`gnn_significantly_worse`) is NOT a failure — it is a scientific finding. The correct framing is:

> Under A0, degree centrality functions as a near-perfect structural proxy, leaving no recoverable signal that graph learning can add — this confirms the structural ceiling hypothesis rather than indicating a model deficiency.

Do NOT frame this as "GNN failed under A0" or "A0 is a weak regime." Frame it as: "The degree-coupled operationalization imposes a structural ceiling that is already captured analytically; the interesting variation appears under HSCC."

## Rule E7 - Keep architecture comparison compact

This is not an architecture paper.

One short paragraph is enough to note:

- which architecture is strongest in each regime,
- which models were excluded and why (one sentence per exclusion),
- and whether the pattern fits the regime story.

Template for hardware exclusion (use verbatim or adapt):

> GAT was excluded from the main evaluation due to GPU memory constraints (OOM at hidden_dim=128 on our hardware); we therefore report results for four architectures: GraphSAGE, GCN, GIN, and APPNP.

Keep the exclusion explanation to one sentence. Do not turn it into a subsection or defensive discussion.

**Exception — Rule E12 interaction:** Rule E12 requires reporting the architecture-regime interaction (GCN@A0 vs SAGE@HSCC, GIN collapse) as a *finding*, not merely a footnote. This is compatible with Rule E7: keep the prose compact (one paragraph), but **do include the frozen values and the regime-alignment interpretation** that Rule E12 specifies. The one-paragraph limit of Rule E7 applies to the architecture discussion as a standalone section — if the architecture-regime interaction is integrated into the regime results section (§4.2 / §4.3), it can be woven into those result paragraphs without requiring a separate architecture section.

## Rule E8 - Report variance, but do not over-analyze it

Seed variance is result hygiene.

Suggested practice:

- if variance is small, mention stability briefly and move on
- if variance is large, note it explicitly and explain whether the model is excluded from main claims

## Rule E9 - Negative micro-results should be concise

If rankloss or another auxiliary variant does not materially change the regime-level story, report it briefly.

Example:

> Ranking-aware training did not materially alter the regime-level conclusion and is therefore omitted from the main discussion.

**Rankloss space-allocation hierarchy (for tight page limits):**

This paper's C3 rankloss IS frozen and significant vs comparator (CI [+0.030, +0.053]). If the paper is over the page limit, use this cut order — do NOT cut rankloss before cheaper alternatives:

1. First cut: oracle/phi decomposition rows from Table 3 (appendix or cut entirely)
2. Second cut: verbose discussion prose in §5 (compress to 2–3 sentences)
3. Third cut: Table 4 runtime mini-table (merge runtime column into Table 2/3)
4. Fourth cut: NDCG@10 and P@10 columns from results tables (keep Spearman only in main paper)
5. **Last resort only:** Move SAGE+rankloss row to appendix — acceptable if and only if there is literally no other cut available
6. **Never cut:** The rankloss bootstrap CI value from the main claim or abstract if it appears there; either include both the standard GNN and rankloss rows, or cut both

## Rule E10 - Runtime is a practical story, not the main contribution

Frame runtime as:

> once trained, the surrogate provides rapid full-graph inference compared with rerunning MC-IC

Do not let runtime become the central claim if the comparative story is weak.

## Rule E11 - Distinguish precomputed-embedding baselines from real-time baselines

If Node2Vec or similar embedding-based baselines appear in the results table, note clearly — either in the table caption or the experimental setup paragraph — that embedding generation required offline precomputation (approximately 2.5 minutes — 153 seconds — per regime, per frozen `runtime_breakdown.csv`) and is therefore not directly comparable to real-time analytical inference such as degree centrality or LR.

Preferred wording for setup section:

> Node2Vec embeddings were generated offline prior to evaluation (approximately 2.5 minutes — 153 seconds — per regime, per frozen runtime_breakdown.csv); the reported inference time reflects only the downstream LR prediction step (~0.04s); the LR model fit is bundled into the training time (~153s, together with embedding generation).

Do not frame Node2Vec inference speed as equivalent to degree centrality or LR(raw_attr) inference speed without this disclaimer.

In the results table, place Node2Vec in a distinct row group labeled **"Shallow Embedding"** (separate from analytical baselines and flat attribute models) to make the computational difference visually clear.

## Rule E12 - Report architecture-regime interaction as a finding, not a footnote

The choice of best architecture differs by regime and must be reported explicitly in Section IV.

**Frozen results (h=128 official rerun):**
- Under A0 (degree-coupled): **GCN** (symmetric normalization) is best raw_attr arch (ρ=0.808). SAGE performed poorly (ρ=0.534). GCN's symmetric normalization aligns naturally with degree-coupled propagation.
- Under HSCC (source-velocity): **SAGE** (mean aggregation) is best (ρ=0.915). GCN dropped to ρ=0.602. SAGE's mean aggregation suits source-side engagement signals.
- **GIN exhibited near-random performance under HSCC** (ρ=0.028; std=0.046, not excluded by threshold). Report as a finding: GIN's sum aggregation collapsed under source-velocity operationalization.
- **APPNP was excluded in both regimes** due to high seed variance (std=0.417 under A0; std=0.146 under HSCC), consistent with the `--gnn-std-threshold 0.1` policy.

Preferred framing for architecture comparison paragraph:

> The best-performing architecture differs by regime: GCN under A0 (ρ=0.808), SAGE under HSCC (ρ=0.915). This architecture-regime interaction is consistent with the hypothesis that operationalization choice drives inductive bias alignment — GCN's symmetric normalization suits degree-coupled propagation, while SAGE's mean aggregation suits source-velocity signal aggregation.

Do not frame this as "GCN is better than SAGE" — regime context must always accompany architecture comparisons.

**GIN collapse reporting requirement:** GIN's near-random HSCC result (ρ=0.028) must be reported explicitly in the paper — do not hide it or omit it from the results table. Preferred sentence:

> GIN exhibits near-random performance under HSCC (ρ = 0.028), consistent with the hypothesis that sum aggregation without normalization fails to recover source-velocity signals dominated by a heavy-tailed engagement distribution.

**GIN framing — this is a finding, not a model failure.** Do not write "GIN performed poorly." The correct framing treats the collapse as an empirical finding about architecture-operationalization fit, not as a deficiency of GIN in general. Preferred framing: "GIN's near-random result under HSCC reveals that sum aggregation without normalization does not align with source-velocity label structure..." — same scientific register used for the A0 structural ceiling result.

**CI equivalence bound note (avoid confusion):** When writing about A0 equivalence, use this precise logic: for a CI of [−0.029, −0.008], practical equivalence under δ₀=0.02 requires the *entire* CI to fall within [−0.02, +0.02]. Here, the CI lower bound −0.029 lies outside this window (|−0.029| = 0.029 > 0.02), so equivalence is NOT supported. This means the CI is fully negative and `gnn_significantly_worse` is the correct interpretation, not `practically equivalent`. Do not write "CI lower bound is outside the equivalence bound" without the absolute value clarification — the directional phrasing is ambiguous to non-statistician reviewers.

## Rule E13 - Binary instability claim: scope of evidence is A0-only; extension to HSCC must be framed as topology argument

The formal stability diagnostic (Jaccard, gap-to-noise sweep) was conducted under **A0 only** (weighted-cascade, using the frozen Person 1 A0 score artifact — `ic_scores_a0.parquet` in the current team handoff). Key artifacts:

| Artifact | Path | Scope |
|---|---|---|
| `stability_explanation.json` | `outputs/day1_benchmark/` | A0; community field is graph-topology |
| `ic_label_stability.json` | `outputs/day1_benchmark/` | A0 — Jaccard=0.307 across 3 MC seeds |
| `phase1_community_overlap.json` | `outputs/ic_feasibility/` | A0 scores; community structure regime-invariant |
| `phase2_threshold_analysis.json` | `outputs/ic_feasibility/` | A0 only |
| `pivot_decision_report.json` | `outputs/ic_feasibility/` | A0 only; has `evidence_statement_for_paper` |

**Do NOT write:** "Jaccard=0.31 under HSCC" — that is an A0 measurement only.

**DO write for HSCC:** One of the following scoped extensions:
- *"The community-overlap structural argument (84.2% communities spanning top-k boundary) is a graph-topology property invariant to IC model parameterization, and therefore extends directly to HSCC."*
- *"Under HSCC, degree collapse (ρ = −0.006) renders any degree-anchored binary threshold meaningless independently of Jaccard stability."*

**Contribution bullet scope correction:** When listing contributions, use:
> "We show that binary top-k IC labels are structurally unstable (formal diagnostic under A0; structural cause extends to HSCC via community topology)."

This avoids overclaiming while preserving the general regression motivation for both regimes.

---

# 8. Section V - Discussion and Limitations Rules

## Rule D1 - Start the discussion with the broad insight

A good opening sentence should generalize beyond this single dataset.

Example:

> Our results suggest that the usefulness of graph learning for IC approximation is governed by the information structure of the operationalization rather than by architecture choice alone.

## Rule D2 - State limitations as facts, not apologies

Good:

> HSCC is a domain-informed comparative operationalization rather than a validated generative law.

Weak:

> Unfortunately, HSCC may not fully capture real Twitch diffusion.

## Rule D3 - Keep limitations compact without becoming robotic

Each limitation should usually fit in one compact sentence, or one sentence plus a short clarifying clause when needed.

Do not turn limitations into defensive mini-essays.

## Rule D4 - Future work must be specific

Weak:

- `future work includes more datasets`

Better:

- `evaluating the contrast on a dataset with behavioral cascades, such as Higgs, would test whether the regime-dependent finding persists under empirical diffusion data`

## Rule D5 - Do not introduce new numbers in discussion

All quantitative evidence should already appear in the results section, tables, or figures before the discussion interprets it.

## Rule E14 - NDCG@10% and Precision@10% are secondary — Spearman drives inferential claims

**Primary metric:** Spearman ρ — only metric backed by frozen bootstrap CI artifacts. All inferential claims (significantly better/worse/equivalent) derive from Spearman bootstrap CIs.

**Secondary metrics:** NDCG@10% and Precision@10% — reported in tables for completeness and reader reference, but:
- Do NOT make inferential claims based on NDCG or P@10 alone ("GNN significantly improves NDCG" is not allowed without a separate NDCG bootstrap CI artifact — which does not exist for this paper)
- Do NOT use NDCG trends to override a Spearman-based claim (e.g., "NDCG improves even though Spearman is below degree — therefore GNN is better" is invalid)
- If NDCG and Spearman disagree, acknowledge the divergence as descriptive: "NDCG@10% improves slightly, but the Spearman-based bootstrap test indicates GNN remains statistically below degree centrality"
- NDCG@10% in this paper's frozen HSCC CI JSON: delta=+0.074, CI=[+0.050, +0.099] — this value CAN be cited descriptively to corroborate Spearman finding, but the Spearman CI is the pre-registered primary claim

**Allowed:** "Table 2 reports Spearman ρ (primary), NDCG@10%, and Precision@10% for context."
**Forbidden:** "GNN significantly outperforms degree on NDCG@10%." (no CI artifact for this)

## Rule E12a - Cross-regime architecture comparison: framing order and language

When one architecture performs well in one regime but poorly in another, follow this discipline:

**Framing order for prose:**
1. State A0 winner with frozen ρ: "Under A0, GCN achieves the best raw_attr performance (ρ=0.808)"
2. State HSCC winner with frozen ρ: "Under HSCC, SAGE achieves the best performance (ρ=0.915)"
3. State the interaction: "This architecture-regime interaction is consistent with..."
4. Report GIN collapse: "GIN's near-random HSCC result (ρ=0.028) further illustrates this interaction"

**Forbidden framings:**
- "GCN is generally better than SAGE" — regime context is MANDATORY
- "SAGE is the best architecture" — only under HSCC
- "GIN performed poorly" — must say "GIN collapsed under HSCC" (not in A0)
- "All architectures performed similarly under A0" — GIN ρ=0.615 vs GCN ρ=0.808 is not similar

**Allowed:** "The best-performing architecture differs by regime: GCN under A0 (ρ=0.808), SAGE under HSCC (ρ=0.915)."

## Rule F7 - APPNP exclusion footnote: standardized format

APPNP appears in Table 2, Table 3, and Figure 2 but is excluded from the best-arch comparison. Standard treatment:

- **Symbol:** Use dagger † (U+2020) after "APPNP" in table rows and figure legend
- **Standard footnote text:** "†APPNP excluded from best-architecture comparison: seed standard deviation exceeded 0.1 in both regimes (A0: std=0.417; HSCC: std=0.146), per pre-registered exclusion criterion."
- **In table caption:** "(†excluded from best-architecture comparison; see footnote)"
- **In main text:** one sentence only: "APPNP was excluded from the best-arch selection in both regimes due to high seed variance (std=0.417 under A0; std=0.146 under HSCC), exceeding the pre-registered threshold of 0.1."

Do NOT: describe APPNP's exclusion as a failure or bug. Do NOT: omit APPNP from table entirely. Do NOT: use double-dagger ‡ unless already used for another annotation in the same table.

## Rule F4a - Model naming convention (addendum to Rule F4)

| Context | Correct name | Incorrect |
|---|---|---|
| First use in main text | "GraphSAGE (mean aggregation)" | "SAGE", "gnn_raw_attr" |
| Subsequent main text | "SAGE" or "GraphSAGE" | "gnn_raw_attr", "graph_sage" |
| Table row label | "SAGE" or "GNN (SAGE, raw_attr)" | "gnn_raw_attr" |
| Figure legend | "SAGE (mean agg.)" | "gnn_raw_attr", "SAGE-raw" |
| Artifact code reference (inline backtick) | `` `gnn_raw_attr` `` | "gnn raw attr", "gnn-raw-attr" |
| Method section | "GraphSAGE [Hamilton et al., 2017] with mean aggregation" | just "SAGE" without citation on first use |

Same convention applies to other architectures:
- "GCN" after first use as "Graph Convolutional Network (GCN) [Kipf & Welling, 2017]"
- "GIN" after first use as "Graph Isomorphism Network (GIN) [Xu et al., 2019]"
- "APPNP" after first use as "APPNP [Klicpera et al., 2019]"

---

## Quick Reference: Model-Specific Writing Rules

Writers looking for a specific model's constraints — find the relevant rule here:

| Model | Key writing constraint | See Rule(s) |
|-------|----------------------|-------------|
| **GCN** | "Graph Convolutional Network (GCN)" at first use; "GCN" thereafter; never "GCN surrogate" | Rule F4a |
| **SAGE / GraphSAGE** | Tables: "SAGE"; prose: "GraphSAGE" at first use, then "SAGE" or "GraphSAGE" (consistent per paper); NEVER "gnn_raw_attr" in prose | Rule F4a |
| **GIN** | Collapse under HSCC (ρ=0.028) is a **finding** — frame as "architecture-operationalization fit", not "GIN failed" or "GIN performed poorly" | Rule E12a |
| **APPNP** | Dagger (†) in table rows; one-sentence exclusion explanation in main text; report results in appendix for completeness; do NOT use double-dagger (‡) unless already reserved for another annotation | Rules E12a, F7 |
| **Node2Vec** | Separate "Shallow Embedding" group in tables; runtime MUST show precomp (~153s offline) separately from inference (~0.04s); NEVER merge precomp time with GNN inference time | Rule E11, Section 11 |
| **degree** | A0: primary comparator (analytical baseline); HSCC: contextual only — footnote "included for regime-contrast reference only; ρ = −0.006" — NOT the HSCC claim comparator | Rules E4, E5 |
| **`lr_degree_views_life_time_lang`** | HSCC primary comparator — LOCKED; name explicitly at least once in main text, table, or caption; mark with ★ or bold in table | Rule HSCC-F1 (guide), Rule F3 |
| **SAGE (all feats) / ‡ row** | Double-dagger (‡) in table; footnote "diagnostic row only; oracle feature access (full graph attributes) not available at inference time — excluded from main comparisons" | Rule F7 note |

Cross-reference: for full paste-ready sentences and frozen ρ values, see `Paper guide.md` Sections 11–12.

---

## Rule F8 — Abbreviation Consistency

Define at first use; abbreviate consistently thereafter. Do NOT alternate between full form and abbreviation within a section.

| Full form | Standard abbreviation | First definition | Notes |
|-----------|-----------------------|-----------------|-------|
| Independent Cascade | IC | §2.1 | |
| Monte Carlo Independent Cascade | MC-IC | §2.1 or §3 | Always hyphenated |
| Twitch Gamers graph | Twitch graph | §3.1 | Use the dataset name "Twitch Gamers" (Rozemberczki & Sarkar, 2021); do not label it as "EN". |
| Graph Neural Network | GNN | §1 or §2.3 | "GNNs" (plural OK) |
| Logistic Regression | LR | §4.1 | Tables: "LR(...)"; prose: spell out at first mention |
| Spearman rank correlation | ρ | §4.1 | Always lowercase Greek; define as "Spearman ρ" at first use |
| Normalized Discounted Cumulative Gain | NDCG@k | §4.1 | Include @k value (e.g., NDCG@10); define at first use in §4 |
| GraphSAGE | SAGE (tables only) | §4.1 | In prose: "GraphSAGE" at first mention; see Rule F4a for prose vs table rules |
| IC regime A0 | A0 | §3.2 | Define as "weighted cascade regime (A0)" |
| IC regime HSCC | HSCC | §3.2 | Define as "source-community regime (HSCC)"; expand acronym once |

**Do NOT abbreviate:** "degree centrality" (always spell out in results prose), "influence score" (always spell out), "community" (always spell out).

**Do NOT alternate:** If you use "IC" in §2, use "IC" (not "Independent Cascade") consistently through §4. Exception: "Monte Carlo IC simulation" is allowed in runtime discussion to distinguish simulation from model.

---

# 9. Section VI - Conclusion Rules

## Rule C1 - Conclusion should restate the comparative answer, not repeat the abstract

The conclusion should answer:

> When does graph learning add value beyond the strongest valid comparator in each regime?

It should not simply restate the whole pipeline or collapse regime-specific comparator logic into one baseline story.

## Rule C2 - End on the regime-dependent lesson

Good conclusion pattern:

- one sentence on the main contrast,
- one sentence on what this implies for surrogate design,
- one sentence on why operationalization matters more than architecture alone.

**Optional C3 mention (only if rankloss artifact is frozen and presented in Results):**

If the C3 rankloss artifact is frozen and the rankloss row is presented in Results, the conclusion may optionally add one sentence:

> Ranking-aware training provides a further marginal gain under HSCC, suggesting that explicit rank supervision is a viable extension when the operationalization is source-side-driven.

Do not add this sentence if the C3 artifact is not frozen or if the rankloss result is omitted from Results. If you mention the `+0.009` over standard SAGE, frame it as descriptive only, not as an inferential claim.

## Rule C3 - Do not add new caveats or new evidence here

The conclusion is for synthesis, not for fresh argument.

---

# 10. Figures and Tables Rules

## Rule F1 - A figure should be understandable in 10 seconds

Readers should instantly see:

- left panel = A0
- right panel = HSCC
- reference line = comparator
- focal markers or bars = GNN results

## Rule F2 - Prefer grayscale-safe figure logic

Do not rely on color alone.

Use:

- dashed or solid reference lines,
- marker shape differences,
- panel labels,
- ordering,
- and captions that state the comparator explicitly.

## Rule F3 - Table captions must be self-contained; comparator must be locked

A table caption should tell the reader:

- what setting the table covers,
- what the primary comparator is,
- and what averaging / seed logic applies.

**Comparator lock discipline:** Once a table's primary comparator is set (e.g. `degree` for A0; `LR(degree, views, life_time, language)` / `lr_degree_views_life_time_lang` for HSCC in this paper), do **not** change it in subsequent drafts. Changing the comparator mid-draft invalidates previously written claim sentences and makes reviewer responses inconsistent. If the comparator must change (e.g. because new frozen artifacts show a different baseline is strongest), update the claim sentences and the abstract at the same time — not independently.

**Worked examples of locked table captions (use or adapt):**

*Table 2 (A0 results):*
> Surrogate ranking performance under the degree-coupled IC regime (A0). Primary comparator: degree centrality (ρ = 0.826). All GNN results averaged over 5 random seeds; ± values are standard deviations. †APPNP excluded from best-arch comparison (seed std > 0.1 threshold). ‡Node2Vec: embedding precomputation ~153s offline; reported inference time is downstream LR only.

*Table 3 (HSCC results):*
> Surrogate ranking performance under the source-community IC regime (HSCC). Primary comparator: LR(degree, views, life_time, language) — `lr_degree_views_life_time_lang`, ρ = 0.884 — official comparator per frozen bootstrap CI artifact. Note: `lr_views_life_time_lang` (ρ = 0.88442) is within 0.001 points — a practical tie; comparator selection is justified by artifact pre-specification. Degree included for regime-contrast reference only (ρ = −0.006). All GNN results averaged over 5 random seeds. †APPNP excluded from best-arch comparison (seed std > 0.1). ‡Node2Vec: precomputation ~153s offline.

These captions are frozen — use them as the reference for all subsequent paper drafts. Do not silently change the comparator row or footnote content without updating claim sentences simultaneously.

**Table caption completeness checklist (verify before submission):**

Before finalizing any results table, confirm all 7 points are satisfied:

1. ☐ **Regime stated** — caption says "A0" or "HSCC" (not just "Surrogate Ranking Results")
2. ☐ **Primary comparator named** — caption identifies the locked comparator by name with its ρ value (not just "strongest baseline")
3. ☐ **Metric defined** — caption states "Spearman ρ" and any secondary metric (NDCG@10, P@10) appearing in columns
4. ☐ **Seed protocol stated** — caption includes "5 random seeds; ± values are standard deviations" (or abbreviated equivalent)
5. ☐ **Abbreviations expanded** — LR, GNN, MC-IC each defined at first table appearance (or in a shared table notes block)
6. ☐ **Dagger (†) explained** — APPNP row footnote: "†APPNP excluded from best-architecture comparison: seed std > 0.1 threshold (A0: std=0.417; HSCC: std=0.146)"
7. ☐ **Double-dagger (‡) explained** (if present) — SAGE (all feats) row footnote: "‡Diagnostic row only; oracle feature access not available at inference time — excluded from main comparisons"

These 7 checks apply to Table 2 (A0 results), Table 3 (HSCC results), and any appendix table containing GNN or baseline ρ values.

**Comparator notation — both forms are acceptable; pick one and keep it consistent:**
- Code-style (for methods section): `` `lr_degree_views_life_time_lang` ``
- Prose-style (for results text): `LR(degree, views, life_time, language)` or `LR(dvtl+lang)`
- Caption-style (for table captions): `LR(degree, views, life_time, language)` — spell out in full on first use

Do not mix forms within a single section. In tables, keep the same row label throughout all drafts (Rule F4).

## Rule F4 - Keep model naming consistent across text and tables

Pick one naming scheme and keep it everywhere.

If the table says `GCN (raw_attr)`, the prose should not drift between:

- `GCN raw`
- `GCN surrogate`
- `graph convolution model`

unless that naming scheme is made explicit and kept consistent.

## Rule F5 - Choose honestly between figure and table

Use tables when exact values matter.  
Use figures when shape, spread, or contrast matters more than exact values.

## Rule F6 - Keep oracle-style diagnostics out of the main comparative figure

If there is a phi-oracle or related upper-bound diagnostic, place it in:

- the appendix,
- a supplementary table,
- or a short discussion note.

Do not let it dominate the main results figure.

---

# 11. Language Patterns to Use and Avoid

## Preferred patterns

- `we evaluate`
- `we compare`
- `we find`
- `our results suggest`
- `under A0`
- `under HSCC`
- `consistent with`
- `practically equivalent` **(⚠️ only when the full CI ⊆ [−δ₀, +δ₀]; i.e., both bounds lie within the equivalence window. For A0 CI=[−0.029, −0.008]: NOT equivalent. See Rule S3 Case 2.)**
- `domain-informed operationalization`
- `matched feature access`

## Avoid these patterns

- `obviously`
- `clearly superior`
- `surprisingly`
- `proves`
- `state-of-the-art`
- `real influence`
- `ground-truth influence`
- `HSCC is better than A0`
- `GNN beats the baseline`
- `significant` or `significantly` without a stated inferential test or CI

## Better replacements

Instead of:

> HSCC is better than A0

Use:

> HSCC induces a qualitatively different approximation regime from A0.

Instead of:

> GNN beats the baseline

Use:

> GNN improves over the strongest matched flat baseline under HSCC.

Instead of:

> GNN is feature-agnostic

Use:

> GNN does not require precomputed structural summaries.

## Node2Vec language patterns (see also Rule E11)

**Preferred (use one of these when describing Node2Vec):**

> Node2Vec embeddings were generated offline prior to evaluation (approximately 2.5 minutes — 153 seconds — per regime); the reported inference time reflects only the downstream LR prediction step (~0.04s); the LR model fit is bundled into the training time (~153s, together with embedding generation).

> We include Node2Vec as a shallow-embedding baseline; embedding generation required approximately 2.5 minutes (153 seconds) per regime and was completed as a preprocessing step before evaluation.

**Avoid (will invite reviewer challenge):**

> Node2Vec is a fast baseline. ← wrong without qualifying the precomputation cost
> Node2Vec inference is comparable to degree centrality. ← wrong; degree has no offline precomputation step
> Node2Vec runs in O(1) per node at inference time. ← misleading; omits the precomputation cost entirely

If Node2Vec appears in a runtime table, its row must show **precomputation time separately** from downstream LR inference time. Do not merge these two into a single number.

**Frozen timing reference (from `runtime_breakdown.csv` — use these exact values):**
- Precomputation: **~153 seconds (~2.5 minutes)** per regime
- Downstream LR prediction (inference only): **~0.040 seconds** ← `inference_sec_full_graph` in CSV
- Note: `train_sec` = 153s bundles embedding generation + LR fit together; 0.040s is predict-only
- Do NOT write "30–60 minutes" — the actual frozen measurement is ~153s

## Ranking loss / rankloss language patterns (C3 BOOST — ✅ FROZEN)

**Preferred (use one of these when describing the rankloss variant):**

> Ranking-aware training (combined Huber + pairwise ranking loss, α=0.5) achieves ρ = 0.924 under HSCC, a descriptive gain of +0.009 over standard Huber training (ρ = 0.915). [Descriptive — no paired bootstrap between rankloss and standard SAGE.]

> The rankloss variant (SAGE, `loss_mode=rankloss_combined`, `rankloss_alpha=0.5`) achieves ρ = 0.924 under HSCC, outperforming the strongest flat baseline by Δρ = +0.041 (95% CI [+0.030, +0.053]).

> We include a ranking-aware training variant (combined Huber + pairwise ranking loss) as an optional BOOST path; when frozen, it achieves ρ = 0.924 and Δρ = +0.041 over the matched flat baseline under HSCC.

**Avoid:**

> The ranking loss model is clearly superior. ← overclaims; +0.009 over Huber-GNN is marginal
> Rankloss fixes the regression loss's deficiency. ← too strong; frame as complementary, not corrective
> We designed a novel ranking loss. ← this is a standard pairwise loss variant; do not overclaim novelty

**Placement guidance (main paper vs appendix):**

The rankloss result fits naturally as:
- A bolded row in Table 3 (HSCC results) labeled **"SAGE + rankloss (C3)"**, placed directly below the standard `gnn_raw_attr` row
- One short sentence in §4.3 prose noting the improvement: "A ranking-aware variant further improves to ρ = 0.924 (+0.009 over standard training, Δρ = +0.041 over comparator, 95% CI [+0.030, +0.053])."
- One optional sentence in the conclusion (see Rule C2)

Do NOT move rankloss entirely to the appendix — the artifact is frozen and the result is significant vs the comparator. Including it in the main table as a row strengthens the paper. Only move it to appendix if space absolutely requires cutting after all other cuts are made.

---

# 12. Outcome-Dependent Claim Guide

These are templates, not predictions.

**Quick navigation — find your template:**
- Evidence still preliminary → *"If evidence is still preliminary"* below
- A0 result (frozen: `gnn_significantly_worse`) → **"If A0 is significantly below degree ← ✅ ACTIVE"**
- HSCC result (frozen: `gnn_significantly_better`) → **"If HSCC significantly improves ← ✅ ACTIVE"**
- C3 rankloss (frozen: significant improvement) → **"C3 Rankloss Outcome Templates ← ✅ ACTIVE"**
- HSCC tie or loss (not active; archived) → *"If HSCC is approximately tied"* / *"If HSCC is significantly below"*

## If evidence is still preliminary or mixed

Use wording like:

> Preliminary results suggest a regime-dependent pattern, but the final comparative claim should be locked only after frozen bootstrap outputs are available.

## If A0 is practically equivalent to degree

> **[NOT ACTIVE for this paper — frozen result is `gnn_significantly_worse`, not equivalent. Use the "significantly below" template below.]**

Use wording like:

> Under A0, the best GNN is practically equivalent to degree under the pre-registered equivalence bound.

## If A0 significantly improves over degree

> **[NOT ACTIVE for this paper — frozen result is `gnn_significantly_worse`. Use the "significantly below" template below.]**

Use wording like:

> Under A0, the best GNN significantly improves over degree, indicating that the degree-coupled operationalization still leaves recoverable graph-structured signal beyond the analytical ceiling implied by simple centrality alone.

## If A0 is significantly below degree  ← ✅ ACTIVE TEMPLATE (frozen result)

Use wording like:

> Under A0, the best GNN (GCN, raw node attributes) remains statistically below degree centrality (Δρ = −0.018, 95% CI [−0.029, −0.008]), indicating that the degree-coupled operationalization imposes a structural learnability ceiling.

**Frozen values:** GCN ρ=0.808, degree ρ=0.826, delta=−0.018, CI=[−0.029, −0.008].

**Do NOT use "practically equivalent" for A0.** Practical equivalence requires the *entire* CI to lie within the pre-registered equivalence window [−δ₀, +δ₀] = [−0.02, +0.02]. The frozen CI = [−0.029, −0.008]: both bounds are negative and the lower bound |−0.029| = 0.029 exceeds δ₀ = 0.02. This means the CI is fully negative (GNN is below degree) AND the magnitude is too large for equivalence — the correct interpretation is `gnn_significantly_worse`, not `practically equivalent`.

**How to frame this positively for reviewers:** The A0 result is NOT a failure of the GNN. The correct framing is that the degree-coupled operationalization is analytically saturated — degree itself is the structural prior embedded in the label-generation mechanism, so no graph model can outperform it. This is the structural ceiling hypothesis confirmed, not a model deficiency.

## If HSCC significantly improves over the strongest flat baseline  ← ✅ ACTIVE TEMPLATE (frozen result)

Use wording like:

> Under HSCC, the best GNN (SAGE, raw node attributes including language) significantly outperforms the strongest matched flat baseline — LR(degree, views, life_time, language), ρ = 0.884 — achieving Δρ = +0.033 (95% CI [+0.021, +0.044]), consistent with residual neighborhood-structured signal beyond node-level source attributes alone.

**Frozen values:** SAGE ρ=0.915, `lr_degree_views_life_time_lang` ρ=0.884, delta=+0.033, CI=[+0.021, +0.044].
Comparator name for paper table: `lr_degree_views_life_time_lang`. Do NOT change this comparator in subsequent drafts (Rule F3 comparator lock).

## If HSCC is approximately tied with the strongest flat baseline

> **[NOT ACTIVE for this paper — frozen result is `gnn_significantly_better`. Archive only.]**

Use wording like:

> Under HSCC, [actual strongest matched flat baseline] already captures most of the source-side signal, leaving limited room for additional gains from message passing.

## If HSCC is significantly below the strongest flat baseline

> **[NOT ACTIVE for this paper — frozen result is `gnn_significantly_better`. Archive only.]**

Use wording like:

> Under HSCC, the best GNN remains statistically below [actual strongest matched flat baseline], suggesting that the operationalization is dominated by node-level source attributes rather than by additional neighborhood signal recoverable through message passing.

## If both regimes favor simpler baselines

Use wording like:

> The results suggest that graph learning is not universally advantageous for IC approximation and that baseline sufficiency depends strongly on the operationalization.

---

## C3 Rankloss Outcome Templates (BOOST path — ✅ FROZEN, confirmed)

**Claim precision note — two separate claims, different evidence levels:**

| Claim | Evidence | Allowed wording |
|---|---|---|
| Rankloss vs flat comparator | ✅ Inferential — `gnn_vs_rankloss_bootstrap_ci_hscc.json` Δρ=+0.041, CI=[+0.030, +0.053] | "significantly outperforms the flat comparator" |
| Rankloss vs standard Huber SAGE | ✅ Descriptive only — point estimate 0.924 − 0.915 = +0.009 | "achieves +0.009 Spearman points above standard training (descriptive)" |

Do **not** write "significantly improves over the standard Huber-trained GNN" — no paired bootstrap between rankloss and standard SAGE exists in the frozen artifacts. The +0.009 is a point estimate, not an inferential result.

### C3 rankloss vs comparator ← ✅ ACTIVE TEMPLATE (inferential, frozen)

Use wording like:

> Ranking-aware training (combined Huber + pairwise ranking loss, α=0.5) achieves ρ = 0.924 under HSCC — a descriptive gain of +0.009 over the standard Huber-trained GNN (ρ = 0.915) — and significantly outperforms the strongest flat baseline by Δρ = +0.041 (95% CI [+0.030, +0.053]), suggesting that explicit rank supervision provides additional benefit for top-influence-score recovery.

**Key word discipline:** "descriptive gain of +0.009" (no CI, no "significantly"); "significantly outperforms the flat baseline" with the CI explicitly stated.

**Status: ✅ FROZEN** — artifact verified: `loss_mode=rankloss_combined`, `rankloss_alpha=0.5`, best arch=SAGE (ρ=0.924). Inferential support is rankloss-vs-comparator only.

### If C3 rankloss does not materially improve over Huber GNN

> **[NOT ACTIVE — frozen result shows improvement. Archive only.]**

Use wording from Rule E9:

> Ranking-aware training did not materially alter the regime-level conclusion and is therefore omitted from the main discussion.

Note: C3 artifact is confirmed frozen. The conditional gate "if C3 was NOT run" no longer applies — C3 can be included in the paper.

---

# 13. Final Writing Workflow

## Freeze discipline

Do not finalize:

- title,
- abstract,
- contribution bullets,
- and the main contrast paragraph

until all of the following are frozen:

**✅ ALL FROZEN — Person 3 rerun complete (2026-04-28). Ready to finalize all of the above.**

1. ✅ A0 bootstrap: `gnn_vs_degree_bootstrap_ci_a0.json` — GCN, delta=−0.018, CI=[−0.029, −0.008], `gnn_significantly_worse`, h=128 confirmed
2. ✅ HSCC bootstrap: `gnn_vs_baseline_bootstrap_ci_hscc.json` — SAGE, delta=+0.033, CI=[+0.021, +0.044], `gnn_significantly_better`, h=128 confirmed
3. ✅ Official HSCC flat comparator in the frozen CI artifact: `lr_degree_views_life_time_lang` (ρ=0.884, include_language=true) — verified in `baseline_ranking_metrics_hscc_clean.csv`
4. ✅ Runtime table: `runtime_breakdown.csv` — MC-IC=480.3s; headline runtime uses `hscc,gnn_raw_attr` inference≈0.086s; speedup≈5,590× (round to ~5,500× in paper prose), Node2Vec precomp≈153s
5. ✅ **(C3 BOOST — confirmed frozen)** rankloss bootstrap: `gnn_vs_rankloss_bootstrap_ci_hscc.json` — SAGE, delta=+0.041, CI=[+0.030, +0.053]; `feature_policy.loss_mode="rankloss_combined"`, `feature_policy.rankloss_alpha=0.5` — both fields present and verified ✅

## Draft discipline

While results are still moving:

- keep claim strength moderate,
- prefer templates over locked wording,
- mark preliminary numbers clearly in internal drafts,
- and keep author notes out of final prose.

## Final quality bar

The final paper should read as:

- one coherent comparative argument,
- written in calm and natural prose,
- with each major claim traceable to an artifact,
- and with no section sounding like it was generated from a rigid checklist.

---

# 14. Closing Note

This file should function as a writing protocol, not as a pile of disconnected tips.

If the team follows it consistently, the paper will be:

- easier to read,
- harder to attack,
- less prone to overclaiming,
- and easier to keep consistent across `Paper skeleton.md` and later drafts because they should follow this rulebook.
