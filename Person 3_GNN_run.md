# Person 3 — GNN / Baseline Rerun Guide

Muc tieu cua file nay la giup Person 3 rerun pipeline sau khi code da duoc sua theo `GNN_fix.md`, de cap nhat lai artifacts moi nhat va defensible hon.

## 1. Nguyen tac truoc khi chay

- Chay tu thu muc `src/mapr2026_v3`.
- A0 phai chay voi `include_language=False`.
- HSCC phai chay voi `include_language=True`.
- `in_dim=24` chi la expected value cua snapshot hien tai (`3 + 21 lang dummies`). Rule dung de verify van la:
  - A0: `in_dim = 3`
  - HSCC: `in_dim = 3 + n_lang_dummies`
- Mac dinh governance:
  - `gat_heads = 4`
  - `appnp_alpha = 0.15`
- Neu GAT bi OOM va phai fallback sang `--gat-heads 2`, thi bootstrap CI phai dung cung gia tri `--gat-heads 2`.

## 2. Mo terminal va vao dung thu muc

```powershell
cd "D:\UIT\Y3 - S2\Social Network Analysis\SNA_Group9_CBoW\src\mapr2026_v3"
```

## 3. Dry-run TRUOC KHI chay that (bat buoc)

Chay tat ca 5 lenh nay truoc de kiem tra code khong loi import/path. Khong ton GPU time.

```powershell
python run_baselines.py --dry-run --targets-path data/processed/regression_targets_a0.parquet --label-regime a0 --out-csv outputs/mapr2026_v3_results/baseline_a0_DRYRUN.csv

python run_baselines.py --dry-run --targets-path data/processed/regression_targets_hscc_refined.parquet --label-regime hscc --include-language --out-csv outputs/mapr2026_v3_results/baseline_hscc_DRYRUN.csv

python run_surrogates.py --dry-run --targets-path data/processed/regression_targets_a0.parquet --label-regime a0 --out-csv outputs/mapr2026_v3_results/surrogate_a0_DRYRUN.csv

python run_surrogates.py --dry-run --targets-path data/processed/regression_targets_hscc_refined.parquet --label-regime hscc --include-language --out-csv outputs/mapr2026_v3_results/surrogate_hscc_DRYRUN.csv

python bootstrap_ci.py --dry-run --surrogate-csv-a0 outputs/mapr2026_v3_results/surrogate_a0_DRYRUN.csv --surrogate-csv-hscc outputs/mapr2026_v3_results/surrogate_hscc_DRYRUN.csv --include-language-hscc
```

Tat ca 5 lenh phai in `[OK]` hoac hien paths dung. Neu co loi ImportError/FileNotFoundError thi fix truoc.

## 3b. Xoa per_group stale rows truoc khi chay

`per_group_prediction_error.csv` hien co 1 stale row voi `label_regime=NaN` (tu lan chay cu). Vi upsert key la 3 cot `[label_regime, model_name, typology_group]`, row NaN se KHONG bi overwrite boi rows moi. Phai xoa file de rerun tao moi sach:

```powershell
del "..\..\..\outputs\mapr2026_v3_results\per_group_prediction_error.csv"
```

Hoac dung Python:

```powershell
python -c "import pathlib; p = pathlib.Path('../../..') / 'outputs/mapr2026_v3_results/per_group_prediction_error.csv'; p.unlink(missing_ok=True); print('Deleted:', p)"
```

## 4. Debug rerun chinh

### Buoc 1 — A0 Baselines (khong node2vec, fast pass)

```powershell
python run_baselines.py --targets-path data/processed/regression_targets_a0.parquet --label-regime a0 --out-csv outputs/mapr2026_v3_results/baseline_ranking_metrics_a0_clean.csv --skip-node2vec
```

Verify:

- KHONG co row `lr_views_life_time_lang`
- KHONG co row `lr_degree_views_life_time_lang`
- file chi chua `label_regime = a0`

### Buoc 2 — HSCC Baselines (khong node2vec, fast pass)

```powershell
python run_baselines.py --targets-path data/processed/regression_targets_hscc_refined.parquet --label-regime hscc --include-language --out-csv outputs/mapr2026_v3_results/baseline_ranking_metrics_hscc_clean.csv --skip-node2vec
```

Verify:

- PHAI co row `lr_views_life_time_lang`
- PHAI co row `lr_degree_views_life_time_lang`
- file chi chua `label_regime = hscc`

### Buoc 3 — A0 GNN

```powershell
python run_surrogates.py --targets-path data/processed/regression_targets_a0.parquet --label-regime a0 --out-csv outputs/mapr2026_v3_results/surrogate_ranking_metrics_a0_clean.csv --include-c2-arch --include-c3-rankloss --gat-heads 4 --appnp-alpha 0.15
```

Verify trong terminal:

- `[FEATURE AUDIT] include_language=False`
- `feature_names` chi gom `views_log`, `views_per_day`, `life_time` cho `raw_attr`
- `in_dim=3` voi `raw_attr`

Neu GAT OOM:

```powershell
python run_surrogates.py --targets-path data/processed/regression_targets_a0.parquet --label-regime a0 --out-csv outputs/mapr2026_v3_results/surrogate_ranking_metrics_a0_clean.csv --include-c2-arch --include-c3-rankloss --gat-heads 2 --appnp-alpha 0.15
```

Khi do phai ghi note vao `docs/experiment_registry.md`.

### Buoc 4 — HSCC GNN

```powershell
python run_surrogates.py --targets-path data/processed/regression_targets_hscc_refined.parquet --label-regime hscc --include-language --out-csv outputs/mapr2026_v3_results/surrogate_ranking_metrics_hscc_clean.csv --include-c2-arch --include-c3-rankloss --gat-heads 4 --appnp-alpha 0.15
```

Verify trong terminal:

- `[FEATURE AUDIT] include_language=True`
- `in_dim = 3 + n_lang_dummies`
- snapshot hien tai expected la `24`
- `feature_names` co cac cot `lang_*`

Neu GAT OOM, rerun lai buoc nay voi `--gat-heads 2` va ghi registry note.

### Buoc 5 — Bootstrap CI

Neu Buoc 3 va Buoc 4 deu chay voi `--gat-heads 4`:

```powershell
python bootstrap_ci.py --surrogate-csv-a0 outputs/mapr2026_v3_results/surrogate_ranking_metrics_a0_clean.csv --surrogate-csv-hscc outputs/mapr2026_v3_results/surrogate_ranking_metrics_hscc_clean.csv --targets-a0 data/processed/regression_targets_a0.parquet --targets-hscc data/processed/regression_targets_hscc_refined.parquet --include-language-hscc --n-bootstrap 1000 --equivalence-bound 0.02 --gat-heads 4 --appnp-alpha 0.15 --out-a0 outputs/mapr2026_v3_results/gnn_vs_degree_bootstrap_ci_a0.json --out-hscc outputs/mapr2026_v3_results/gnn_vs_baseline_bootstrap_ci_hscc.json
```

Neu Buoc 3/4 phai fallback GAT sang 2 heads, thi doi bootstrap thanh:

```powershell
python bootstrap_ci.py --surrogate-csv-a0 outputs/mapr2026_v3_results/surrogate_ranking_metrics_a0_clean.csv --surrogate-csv-hscc outputs/mapr2026_v3_results/surrogate_ranking_metrics_hscc_clean.csv --targets-a0 data/processed/regression_targets_a0.parquet --targets-hscc data/processed/regression_targets_hscc_refined.parquet --include-language-hscc --n-bootstrap 1000 --equivalence-bound 0.02 --gat-heads 2 --appnp-alpha 0.15 --out-a0 outputs/mapr2026_v3_results/gnn_vs_degree_bootstrap_ci_a0.json --out-hscc outputs/mapr2026_v3_results/gnn_vs_baseline_bootstrap_ci_hscc.json
```

Verify:

- `gnn_vs_degree_bootstrap_ci_a0.json` co:
  - `feature_policy.include_language = false`
  - `surrogate_csv_used`
  - `interpretation`
- `gnn_vs_baseline_bootstrap_ci_hscc.json` co:
  - `feature_policy.include_language = true`
  - `surrogate_csv_used`
  - `interpretation`

## 4. Node2Vec truoc khi freeze-final

Hai buoc baseline o tren dang la fast debug pass vi co `--skip-node2vec`.

Truoc khi freeze bang baseline cuoi, Person 3 phai chot mot trong hai huong:

### Option A — Rerun Node2Vec

A0:

```powershell
python run_baselines.py --targets-path data/processed/regression_targets_a0.parquet --label-regime a0 --out-csv outputs/mapr2026_v3_results/baseline_ranking_metrics_a0_clean.csv
```

HSCC:

```powershell
python run_baselines.py --targets-path data/processed/regression_targets_hscc_refined.parquet --label-regime hscc --include-language --out-csv outputs/mapr2026_v3_results/baseline_ranking_metrics_hscc_clean.csv
```

Verify:

- co row `node2vec_lr` trong moi file clean

### Option B — Carry-forward Node2Vec

Chi duoc dung neu:

- co legacy row dung regime,
- provenance dang tin,
- row do duoc graft vao file clean cuoi,
- va ghi ro vao `docs/experiment_registry.md`.

Neu khong co trusted HSCC Node2Vec artifact rieng, khong nen assume carry-forward HSCC la hop le.

## 5. Cleanup shared artifacts (SAU khi Buoc 3 va 4 xong)

> **Luu y:** `runtime_breakdown.csv` da duoc cleanup truoc (0 null rows). Chay lenh nay them 1 lan nua sau khi GNN rerun ket thuc de dam bao khong co row null moi.

```powershell
python -c "import pandas as pd; rt = pd.read_csv('outputs/mapr2026_v3_results/runtime_breakdown.csv'); ANALYTICAL = {'betweenness','degree','kshell','pagerank','one_hop','two_hop','phi','life_time','views','views_day','lr_life_time','lr_views_life_time','lr_phi','lr_degree_views_life_time','mc_ic_labeling','diffusion_proxies'}; STALE_GNN = {'gnn_centrality','gnn_full','gnn_graph_only','gnn_random','gnn_raw_attr','mlp_raw_attr','node2vec_lr'}; null_mask = rt['label_regime'].isna() | (rt['label_regime'].astype(str).str.strip() == ''); rt.loc[null_mask & rt['model_name'].isin(ANALYTICAL), 'label_regime'] = 'analytical'; rt = rt[~(null_mask & rt['model_name'].isin(STALE_GNN))].reset_index(drop=True); rt.to_csv('outputs/mapr2026_v3_results/runtime_breakdown.csv', index=False); print(rt.groupby('label_regime')['model_name'].count().to_dict())"
```

Ket qua expected sau rerun: `{'a0': XX, 'hscc': XX, 'analytical': 16}` — khong co NaN.

## 6. Checklist truoc khi bao ket qua moi

- `baseline_ranking_metrics_a0_clean.csv`
  - khong co `*_lang`
  - co `label_regime = a0`
- `baseline_ranking_metrics_hscc_clean.csv`
  - co `lr_views_life_time_lang`
  - co `lr_degree_views_life_time_lang`
  - co `label_regime = hscc`
- `surrogate_ranking_metrics_a0_clean.csv`
  - co 9 models: `gnn_raw_attr, gnn_graph_only, gnn_centrality, gnn_full, gcn_raw_attr, gin_raw_attr, gat_raw_attr, appnp_raw_attr, best_arch_raw_attr_rankloss`
  - co `label_regime = a0`
- `surrogate_ranking_metrics_hscc_clean.csv`
  - co 9 models nhu tren
  - co `label_regime = hscc`
- `per_group_prediction_error.csv`
  - co cot `label_regime`
  - co rows cho `a0` va `hscc`
  - KHONG con stale null rows (da xoa file truoc khi chay theo Buoc 3b)
- `runtime_breakdown.csv`
  - khong con null `label_regime`
  - con row `mc_ic_labeling` voi `label_regime = analytical`
- 2 file bootstrap JSON
  - co `feature_policy`
  - co `surrogate_csv_used`
  - co `interpretation`
- `docs/experiment_registry.md`
  - cap nhat `a0_feature_policy`
  - cap nhat `hscc_feature_policy`
  - cap nhat `gat_actual_heads`
  - cap nhat `appnp_actual_alpha`
  - cap nhat `hscc_fairness_baselines_confirmed`
  - cap nhat `node2vec_status`

## 7. Bao cao ket qua cho team

Khi gui ket qua moi cho team, nen bao cao toi thieu:

- best A0 baseline va best A0 GNN
- best HSCC flat baseline va best HSCC GNN
- 2 ket luan bootstrap:
  - A0: GNN vs degree
  - HSCC: GNN vs strongest flat baseline
- governance note neu co fallback `gat_heads = 2`
- node2vec status: `rerun` / `carry_forward` / `de_scoped`
