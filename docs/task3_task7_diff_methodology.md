## Diff-Methodology (Task 3 & Task 7): Before vs After

This note documents what changed, why it changed, and what the changes do (and do not) affect.

## Task 3 (Two-hop diffusion proxy)

### Before

- Implementation effectively used an unweighted second-neighbor aggregate:
  - $\sum_{v\in N(u)}\sum_{w\in N(v), w\neq u} \frac{1}{\deg(w)}$
- This omitted first-hop activation weighting $p(u,v)$ required by weighted-cascade expectation.

### After

- Implementation is now plan-conformant with weighted-cascade expected spread:
  - $\text{two\_hop}(u)=\sum_{v\in N(u)} p(u,v)\left(1+\sum_{w\in N(v), w\neq u}p(v,w)\right)$
  - with $p(x,y)=1/\deg(y)$.
- Optimized $O(E)$ form used in code:
  - $\sum_{v\in N(u)} \text{inv\_deg}[v]\,(1+\text{one\_hop}[v]-\text{inv\_deg}[u])$.

### Assumptions behind the optimized form

- CSR graph must be stored as undirected-bidirectional adjacency (both $u\to v$ and $v\to u$ exist).
- This contract is now fail-fast validated by symmetry check before two-hop computation.

### Practical effect on downstream metrics

- `two_hop` baseline improved from prior formula-drifted run:
  - Spearman: $0.5239 \rightarrow 0.8039$
  - NDCG@10%: $0.5027 \rightarrow 0.8478$
  - Precision@10%: $0.25 \rightarrow 0.55$
- In correlation matrix, $\rho(\text{IC},\text{two\_hop})$ increased from $0.5234$ to $0.8153$.

## Task 7 (Null-model interpretation rule)

### Before

- Interpretation used a fixed absolute threshold:
  - compare real Hidden betweenness to $\mu_{null}+\max(0.05,\sigma_{null})$.
- This is scale-mismatched for this setup because Hidden betweenness is typically around $10^{-5}$.

### After

- Interpretation is now scale-aware:
  - compute gap $\Delta = b_{real}-\mu_{null}$,
  - normalize by adaptive sigma $\sigma^*=\max(\sigma_{null},\text{adaptive floor})$,
  - evaluate standardized gap $z=\Delta/\sigma^*$.
- Interpretation message now includes $\rho_{mean}$ context to avoid over-strong structural claims when rank agreement is weak.
- Artifact stores explicit diagnostics:
  - `hidden_betweenness_gap`
  - `hidden_betweenness_gap_sigma`

### Important non-impact clarification

- Task 7 change updates interpretation logic, not the underlying null simulation protocol.
- Core null protocol remains unchanged (500 nodes x 3 realizations x 100 runs/node).

## Scope impact summary

- Directly affected:
  - Task 3 proxy quality and all downstream comparisons using `two_hop`.
  - Task 7 narrative robustness for null-model interpretation.
- Not directly affected:
  - Day-1 IC quality gates (CV/Jaccard) and Option A/Option B decision path.
  - IC primary simulation probability model itself.

## Reporting note (recommended wording)

- "Task 3 now uses plan-conformant weighted-cascade two-hop expected spread, removing first-hop weighting drift from the earlier implementation."
- "Task 7 now uses a scale-aware null-gap interpretation with rho-context, reducing threshold-induced interpretation bias while preserving the same null simulation protocol."

## Audit references

- Task 3 implementation: `src/mapr2026_v3/diffusion_proxies.py`
- Task 7 interpretation logic: `src/mapr2026_v3/null_model_typology.py`
- Current baseline metrics: `outputs/mapr2026_v3_results/baseline_ranking_metrics.csv`
- Current correlation matrix: `outputs/mapr2026_v3_results/metric_correlation_matrix.json`
- Current null summary: `outputs/mapr2026_v3_results/null_model_typology_summary.json`
