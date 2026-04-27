# Academic Writing Rules for MAPR 2026 Paper
## Canonical Writing Protocol
### Primary source for paper drafting and skeleton alignment

---

## How to Use This File

This is the single authoritative writing guide for the paper.

Use this file as the authority source. `Paper skeleton.md` should follow these rules, not define them.

Use the skeleton only as a drafting scaffold after it has been aligned to this rulebook:

1. Read the core philosophy first.
2. Draft each section in the same order as the skeleton.
3. Treat every example sentence here as a template, not as a pre-committed claim.
4. Finalize the title, abstract, contribution bullets, and contrast paragraph only after the relevant artifacts are frozen.

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
- `0.086 s`
- `7,169x`
- `5,000 labeled nodes`

Never round in a direction that flatters the paper.

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

- "When Does Graph Learning Outperform Analytical Baselines?"
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

> on the Twitch social network (168K nodes, 6.8M edges)

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

> It remains unclear whether learned graph surrogates add value over simple analytical baselines once the diffusion operationalization is fixed.

Weak:

> Prior work fails to compare against simple baselines.

## Rule I3 - Contribution bullets must be falsifiable

Good templates:

- `We show that binary top-k labels are structurally unstable under A0.`
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

> This parameterization models attention dilution: a node with many incoming alternatives receives less transmission probability from each individual neighbor.

Example after HSCC:

> This operationalization combines source-side engagement intensity with a structural incentive for cross-community spread.

## Rule B3 - Describe GNN architectures comparatively

Do not write five disconnected mini-paragraphs.

Preferred style:

> GraphSAGE uses mean aggregation, GCN applies symmetric normalization, GIN emphasizes multiset expressiveness through sum aggregation, GAT learns attention-weighted neighbor aggregation, and APPNP decouples transformation from propagation.

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

## Rule M3 - Justify each HSCC component briefly

After the HSCC formula, include short justification for:

- `phi(u)` -> engagement velocity / tenure-normalized activity
- community boost -> cross-community exposure / structural holes
- clipping and constants -> bounded stability, not calibrated to real logs

Keep these justifications short and transparent.

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

At least one sentence in the paper should make this explicit:

> Under HSCC, degree is included only as contextual evidence of regime shift; the relevant comparator is the strongest flat non-graph baseline under matched feature access.

## Rule E6 - The contrast paragraph is the most important paragraph in the paper

It should explain:

- A0 is degree-coupled,
- HSCC adds source-side and graph/community structure,
- analytical baselines suffice in one regime but not necessarily in the other,
- therefore GNN value is operationalization-dependent.

Write this paragraph only after the numbers are stable.

## Rule E7 - Keep architecture comparison compact

This is not an architecture paper.

One short paragraph is enough to note:

- which architecture is strongest in each regime,
- which models were unstable or excluded,
- and whether the pattern fits the regime story.

## Rule E8 - Report variance, but do not over-analyze it

Seed variance is result hygiene.

Suggested practice:

- if variance is small, mention stability briefly and move on
- if variance is large, note it explicitly and explain whether the model is excluded from main claims

## Rule E9 - Negative micro-results should be concise

If rankloss or another auxiliary variant does not materially change the regime-level story, report it briefly.

Example:

> Ranking-aware training did not materially alter the regime-level conclusion and is therefore omitted from the main discussion.

## Rule E10 - Runtime is a practical story, not the main contribution

Frame runtime as:

> once trained, the surrogate provides rapid full-graph inference compared with rerunning MC-IC

Do not let runtime become the central claim if the comparative story is weak.

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

## Rule F3 - Table captions must be self-contained

A table caption should tell the reader:

- what setting the table covers,
- what the primary comparator is,
- and what averaging / seed logic applies.

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
- `practically equivalent`
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

---

# 12. Outcome-Dependent Claim Guide

These are templates, not predictions.

## If evidence is still preliminary or mixed

Use wording like:

> Preliminary results suggest a regime-dependent pattern, but the final comparative claim should be locked only after frozen bootstrap outputs are available.

## If A0 is practically equivalent to degree

Use wording like:

> Under A0, the best GNN is practically equivalent to degree under the pre-registered equivalence bound.

## If A0 significantly improves over degree

Use wording like:

> Under A0, the best GNN significantly improves over degree, indicating that the degree-coupled operationalization still leaves recoverable graph-structured signal beyond the analytical ceiling implied by simple centrality alone.

## If A0 is significantly below degree

Use wording like:

> Under A0, the best GNN remains statistically below degree, indicating that the degree-coupled operationalization imposes a structural ceiling.

## If HSCC significantly improves over the strongest flat baseline

Use wording like:

> Under HSCC, graph message passing provides measurable gains over the strongest matched flat baseline, consistent with residual neighborhood-structured signal beyond node-level attributes alone.

## If HSCC is approximately tied with the strongest flat baseline

Use wording like:

> Under HSCC, the strongest flat baseline already captures most of the source-side signal, leaving limited room for additional gains from message passing.

## If HSCC is significantly below the strongest flat baseline

Use wording like:

> Under HSCC, the best GNN remains statistically below the strongest matched flat baseline, suggesting that the operationalization is dominated by node-level source attributes rather than by additional neighborhood signal recoverable through message passing.

## If both regimes favor simpler baselines

Use wording like:

> The results suggest that graph learning is not universally advantageous for IC approximation and that baseline sufficiency depends strongly on the operationalization.

---

# 13. Final Writing Workflow

## Freeze discipline

Do not finalize:

- title,
- abstract,
- contribution bullets,
- and the main contrast paragraph

until all of the following are frozen:

1. A0 bootstrap
2. HSCC bootstrap
3. strongest HSCC flat comparator
4. runtime table

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
