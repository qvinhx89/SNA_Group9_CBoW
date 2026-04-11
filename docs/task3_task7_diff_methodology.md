## Diff-Methodology (Task 3 & Task 7): Before vs After

### Task 3 (Two-hop diffusion proxy)

- Before:
  - Implementation used an unweighted second-neighbor aggregate:
    - $\sum_{v\in N(u)}\sum_{w\in N(v), w\neq u} \frac{1}{\deg(w)}$
  - This omitted the first-hop activation weight $p(u,v)$ from weighted-cascade.

- After:
  - Implementation is now plan-conformant with weighted-cascade expected spread:
    - $\text{two\_hop}(u)=\sum_{v\in N(u)} p(u,v)\left(1+\sum_{w\in N(v), w\neq u}p(v,w)\right)$
    - with $p(x,y)=1/\deg(y)$.
  - Optimized O(E) form used in code:
    - $\sum_{v\in N(u)} \text{inv\_deg}[v]\,(1+\text{one\_hop}[v]-\text{inv\_deg}[u])$.

- Practical effect on downstream metrics:
  - `two_hop` baseline improved from:
    - Spearman $0.5239 \rightarrow 0.8039$
    - NDCG@10% $0.5027 \rightarrow 0.8478$
    - Precision@10% $0.25 \rightarrow 0.55$
  - In correlation matrix, $\rho(\text{IC},\text{two\_hop})$ increased from $0.5234$ to $0.8153$.

### Task 7 (Null-model interpretation rule)

- Before:
  - Interpretation used fixed absolute threshold:
    - compare real Hidden betweenness to $\mu_{null}+\max(0.05,\sigma_{null})$.
  - This was scale-mismatched because Hidden betweenness values are typically in $10^{-5}$ range.

- After:
  - Interpretation is now scale-aware:
    - compute gap $\Delta = b_{real}-\mu_{null}$,
    - normalize by adaptive sigma $\sigma^*=\max(\sigma_{null},\text{adaptive floor})$,
    - evaluate $z=\Delta/\sigma^*$.
  - Interpretation message now also includes $\rho_{mean}$ context to avoid over-strong claims when rank agreement is weak.
  - Artifact now stores two explicit diagnostics:
    - `hidden_betweenness_gap`
    - `hidden_betweenness_gap_sigma`

### Reporting note

- Claims are now method-consistent with MAPR2026 v3 plan:
  - Task 3: two-hop is no longer formula-drifted.
  - Task 7: interpretation no longer depends on non-scaled absolute cutoff.
- Recommended wording:
  - "Task 3 now uses plan-conformant weighted-cascade two-hop expected spread."
  - "Task 7 uses a scale-aware null-gap interpretation with rho-context, reducing threshold-induced bias."
