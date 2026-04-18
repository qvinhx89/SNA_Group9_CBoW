# Handoff message — Person 2/3 (Day1 package, MAPR2026 v3.1)

**Handoff tag**: `person1_day1_20260418_p1_day1_v3i_ia_cuda`

**What’s in this handoff (frozen + checksummed)**
- A0 (weighted cascade) contracts: `ic_scores_primary`, `regression_targets`, `classification_labels`, `split_masks`, + Day1 benchmark/evidence JSONs.
- I‑A (attribute‑informed; CUDA‑only run) contracts: `ic_scores_ia`, `regression_targets_ia`.
- Quality mode: **provisional / Option B lockstep** (binary top‑k unstable, regression OK).

**Where to consume**
- Manifest (paths + sha256): `outputs/handoffs/person1_day1_20260418_p1_day1_v3i_ia_cuda/manifest.json`

**Consumer rules (do not drift)**
1) **Do not re-split.** Always load `data/processed/split_masks.parquet` from the handoff.
2) Treat **regression target** as primary evaluation target for A0: `data/processed/regression_targets.parquet` (Option B).
3) Treat `data/processed/classification_labels.parquet` (top‑10%) as **supplementary only**, and use uncertainty/stability evidence when reporting.

**Recommendation: use I‑A vs A0**
- If you are training a surrogate/GNN model: **prefer I‑A labels** (`regression_targets_ia.parquet`) for training/ablation because pilot PASS and it is less degree/proxy‑entangled.
- Keep A0 labels as the baseline branch (for reproducibility and paper narrative), but expect top‑10% binary instability.

**Sanity check command (recommended before training)**
```powershell
.venv/Scripts/python.exe scripts/verify_handoff_package.py --manifest outputs/handoffs/person1_day1_20260418_p1_day1_v3i_ia_cuda/manifest.json
```

**Key evidence (quick pointers)**
- Quality gate report: `outputs/day1_benchmark/quality_gate_report.json` (pass_all=false, provisional)
- Stability/uncertainty: `outputs/day1_benchmark/ic_label_stability.json`, `outputs/day1_benchmark/ic_label_uncertainty.json`
- Stability explanation: `outputs/day1_benchmark/stability_explanation.json`
- I‑A pilot: `outputs/mapr2026_v3_results/ia_pilot_diagnostics.json` (PASS)
