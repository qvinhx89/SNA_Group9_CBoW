# Stage C Sanity Protocol

Scope: Stage 0 to Stage 3 only.
Goal: catch scientific and implementation issues early before branch split.

---

## Stage 0 Sanity Questions

- Are node and edge counts plausible for the expected dataset?
- Are there duplicated node ids after preprocessing?
- Are missing values in key columns handled explicitly?
- Is the processed graph connected enough for centrality and IC assumptions?

## Stage 1 Sanity Questions

- Does centrality distribution look heavy-tailed as expected?
- Do top centrality nodes overlap with high-view nodes partially, not perfectly?
- Are Spearman correlations in a plausible range?
- Are betweenness settings recorded and reproducible?

## Stage 2 Sanity Questions

- Is mean_nmi_louvain above threshold in configuration?
- Is number of communities plausible, not degenerate?
- Is k-shell produced for all nodes required by SIS?
- Is there any obvious mismatch in node ids across files?

## Stage 3 Sanity Questions

- Is Hidden percentage in a usable range for downstream validation?
- Do typology counts sum to total node count?
- Are robustness thresholds stable enough for interpretation?
- Are null-model or control checks interpreted conservatively?

---

## Stop Conditions

Stop and investigate before continuing if any condition is true:

- Required artifact is missing.
- Any stage exits with non-zero code.
- Typology or SIS files cannot be read.
- Group size for key comparison is too small for planned test.
- Configuration changed without registry entry.

---

## Minimum Evidence Bundle For Stage C Gate

- Stage 0 output files exist and are readable.
- Stage 1 metrics and parameters file exist.
- Stage 2 stability and labels exist.
- Stage 3 SIS table exists.
- Stage C gate report exists under logs/run_history.
