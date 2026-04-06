# Day-1 Decisions (MAPR2026 v3)

**Date**: 2026-04-06T04:04:30

## 1. IC Runtime Benchmark

**Per-simulation time**: 8.39 ms

**Projected total runtime** (selected config):
- N_seeds: 5,000
- N_runs: 200
- Projected: 2.3 hours

**Decision**: Use 5,000 seeds × 200 runs

**All tested configurations**:
- 5,000 seeds × 200 runs → 2.3h (<4h)
- 3,000 seeds × 150 runs → 1.1h (4-8h)
- 2,000 seeds × 100 runs → 0.5h (>8h)

## 2. One-Hop Baseline Reality Check

**Spearman ρ** (one-hop vs IC pilot): 0.800 (p=0.000000)

**Decision branch**: `gnn_primary`

**Narrative**: GNN story viable; proceed as planned

## 3. Locked Parameters for Downstream Stages

```yaml
# IC Labels (Stage 4)
n_seeds: 5000
n_runs: 200
p_model: weighted_cascade  # p(u,v) = 1/degree(v)

# GNN Narrative Branch
narrative_branch: gnn_primary
primary_baseline: gnn_raw_attr
```

## 4. Action Items

- [ ] Proceed with IC labels using 5,000 seeds × 200 runs
- [ ] Update all downstream scripts with locked decisions
