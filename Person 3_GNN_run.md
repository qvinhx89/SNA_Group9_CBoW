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
  - `appnp_k = 10`
  - `hidden_channels = 128`
- **A100 default path (team da chot)**: Person 3 phai chay NHANH MAC DINH truoc, tuc la giu `hidden_channels=128`, `appnp_k=10`, KHONG them `--skip-gat`, KHONG them `--hidden-channels 64`. Chi xem cac nhanh fallback neu lenh mac dinh that su OOM / fail vi memory.
- **Neu GAT bi OOM**: KHONG dung `--gat-heads 2` (khong giam duoc VRAM). Dung `--hidden-channels 64` hoac `--skip-gat`. Xem chi tiet o Buoc 3.
- **Governance bat buoc**: bootstrap CI phai dung CUNG cau hinh voi run surrogate cho moi tham so co anh huong den retrain best C2 model: `--gat-heads`, `--appnp-alpha`, `--appnp-k`, `--hidden-channels`, va (neu dung paper filter) `--gnn-std-threshold`. Neu surrogate dung `--hidden-channels 64`, bootstrap cung phai co `--hidden-channels 64`; neu surrogate/paper workflow dung `--gnn-std-threshold 0.1`, bootstrap cung phai dung `--gnn-std-threshold 0.1`.

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

## 3b. Xoa stale CSV files truoc khi chay (BAT BUOC)

`run_surrogates.py` dung **upsert** (khong ghi de sach): neu file cu da co row `gat_raw_attr`, row do se con nguyen khi rerun voi `--skip-gat`. Bootstrap se thay row stale va co the chon GAT ngoai y muon. Phai xoa ca 3 file nay truoc khi chay GNN:

```powershell
python -c "
import pathlib
base = pathlib.Path('../..')
files = [
    base / 'outputs/mapr2026_v3_results/per_group_prediction_error.csv',
    base / 'outputs/mapr2026_v3_results/surrogate_ranking_metrics_a0_clean.csv',
    base / 'outputs/mapr2026_v3_results/surrogate_ranking_metrics_hscc_clean.csv',
]
for p in files:
    p.unlink(missing_ok=True)
    print('Deleted:', p)
"
```

**Tai sao phai xoa ca surrogate CSV:**
- `per_group_prediction_error.csv`: upsert key la 3 cot `[label_regime, model_name, typology_group]`; row NaN cu se KHONG bi overwrite.
- `surrogate_ranking_metrics_*_clean.csv`: upsert key la 2 cot `[label_regime, model_name]`; neu lan cu co `gat_raw_attr` cho cung regime va lan nay dung `--skip-gat`, row stale van con; bootstrap se thay row do va co the chon/rerun GAT ngoai y muon.

> Neu chi can xoa 1 file cu the, dung `missing_ok=True` — lenh khong bao loi neu file khong ton tai.

## 4. Debug rerun chinh

> **THU TU RA QUYET DINH CHO PERSON 3**
> 1. Luon chay **LENH MAC DINH** truoc.
> 2. Chi neu lenh mac dinh bi **OOM / CUDA out of memory / fail vi memory** thi moi chuyen sang **LENH FALLBACK A**.
> 3. Chi neu **LENH FALLBACK A** van OOM thi moi dung **LENH FALLBACK B (LAST RESORT)**.
> 4. Sau khi da dung fallback nao o surrogate run, bootstrap phai dung lai DUNG hyperparameters / rules tuong ung.

### Buoc 1 — A0 Baselines (khong node2vec, fast pass)

**LENH MAC DINH — chay truoc**

```powershell
python run_baselines.py --targets-path data/processed/regression_targets_a0.parquet --label-regime a0 --out-csv outputs/mapr2026_v3_results/baseline_ranking_metrics_a0_clean.csv --skip-node2vec
```

Verify:

- KHONG co row `lr_views_life_time_lang`
- KHONG co row `lr_degree_views_life_time_lang`
- file chi chua `label_regime = a0`

### Buoc 2 — HSCC Baselines (khong node2vec, fast pass)

**LENH MAC DINH — chay truoc**

```powershell
python run_baselines.py --targets-path data/processed/regression_targets_hscc_refined.parquet --label-regime hscc --include-language --out-csv outputs/mapr2026_v3_results/baseline_ranking_metrics_hscc_clean.csv --skip-node2vec
```

Verify:

- PHAI co row `lr_views_life_time_lang`
- PHAI co row `lr_degree_views_life_time_lang`
- file chi chua `label_regime = hscc`

### Buoc 3 — A0 GNN

**LENH MAC DINH — chay truoc tren A100-40GB**

```powershell
python run_surrogates.py --targets-path data/processed/regression_targets_a0.parquet --label-regime a0 --out-csv outputs/mapr2026_v3_results/surrogate_ranking_metrics_a0_clean.csv --include-c2-arch --include-c3-rankloss --gat-heads 4 --appnp-alpha 0.15
```

> **DAY LA LENH MAC DINH PHAI THU TRUOC TREN A100-40GB.**
> Khong nhay sang fallback neu lenh nay chua thuc su bi OOM / CUDA out of memory / fail vi memory.

Verify trong terminal:

- `[FEATURE AUDIT] include_language=False`
- `feature_names` chi gom `views_log`, `views_per_day`, `life_time` cho `raw_attr`
- `in_dim=3` voi `raw_attr`

## FALLBACK CHI DUNG NEU LENH MAC DINH O TREN THAT SU OOM

Neu GAT OOM — **`--gat-heads 2` KHONG phai fix**. Ly do: `hidden_channels=128` la constant, memory ∝ `E × hidden_channels` khong doi khi giam heads. Co 2 option dung:

**LENH FALLBACK A — Giam `hidden_channels` (chi dung neu LENH MAC DINH OOM):**

```powershell
python run_surrogates.py --targets-path data/processed/regression_targets_a0.parquet --label-regime a0 --out-csv outputs/mapr2026_v3_results/surrogate_ranking_metrics_a0_clean.csv --include-c2-arch --include-c3-rankloss --gat-heads 4 --appnp-alpha 0.15 --hidden-channels 64
```

**Luu y quan trong:** `--hidden-channels 64` anh huong den **TOAN BO GNN family** (SAGE, GCN, GIN, GAT, APPNP) — khong chi rieng GAT. Tat ca model trong run do deu dung hidden=64. Neu muon chi GAT dung 64 trong khi cac arch khac dung 128, can sửa code them `--gat-hidden-channels` rieng (chua co). Cach don gian nhat la `--skip-gat` (Option B). Neu dung Option A, ghi registry: `all_models_hidden=64` va nen dung cung gia tri khi compare.

**LENH FALLBACK B — Loai GAT hoan toan (LAST RESORT, chi neu FALLBACK A van OOM):**

> ⚠️ **TRUOC KHI chay Option B**: dam bao da xoa `surrogate_ranking_metrics_a0_clean.csv` theo Section 3b. Neu file cu da co `gat_raw_attr`, `--skip-gat` khong xoa no — bootstrap se thay row stale va rerun GAT ngoai y muon.

```powershell
python run_surrogates.py --targets-path data/processed/regression_targets_a0.parquet --label-regime a0 --out-csv outputs/mapr2026_v3_results/surrogate_ranking_metrics_a0_clean.csv --include-c2-arch --include-c3-rankloss --gat-heads 4 --appnp-alpha 0.15 --skip-gat
```

Khi dung Option B: CSV se co 8 models (thieu `gat_raw_attr`). Day la ket qua hop le — bootstrap van chay duoc mien la co it nhat 1 C2 model khac. Ghi registry: `gat_excluded_reason: OOM_A100_40GB_hidden128`.

### Buoc 4 — HSCC GNN

**LENH MAC DINH — chay truoc tren A100-40GB**

```powershell
python run_surrogates.py --targets-path data/processed/regression_targets_hscc_refined.parquet --label-regime hscc --include-language --out-csv outputs/mapr2026_v3_results/surrogate_ranking_metrics_hscc_clean.csv --include-c2-arch --include-c3-rankloss --gat-heads 4 --appnp-alpha 0.15
```

> **DAY LA LENH MAC DINH PHAI THU TRUOC TREN A100-40GB.**
> Khong nhay sang fallback neu lenh nay chua thuc su bi OOM / CUDA out of memory / fail vi memory.

Verify trong terminal:

- `[FEATURE AUDIT] include_language=True`
- `in_dim = 3 + n_lang_dummies`
- snapshot hien tai expected la `24`
- `feature_names` co cac cot `lang_*`

**CAC LENH FALLBACK CHI DUNG NEU LENH MAC DINH HSCC O TREN THAT SU OOM.**

**LENH FALLBACK A — dung cung cau truc nhu Buoc 3, nhung cho HSCC**

```powershell
python run_surrogates.py --targets-path data/processed/regression_targets_hscc_refined.parquet --label-regime hscc --include-language --out-csv outputs/mapr2026_v3_results/surrogate_ranking_metrics_hscc_clean.csv --include-c2-arch --include-c3-rankloss --gat-heads 4 --appnp-alpha 0.15 --hidden-channels 64
```

**LENH FALLBACK B — LAST RESORT, chi neu FALLBACK A van OOM**

> ⚠️ **TRUOC KHI chay LENH FALLBACK B**: dam bao da xoa `surrogate_ranking_metrics_hscc_clean.csv` theo Section 3b.

```powershell
python run_surrogates.py --targets-path data/processed/regression_targets_hscc_refined.parquet --label-regime hscc --include-language --out-csv outputs/mapr2026_v3_results/surrogate_ranking_metrics_hscc_clean.csv --include-c2-arch --include-c3-rankloss --gat-heads 4 --appnp-alpha 0.15 --skip-gat
```

Neu GAT OOM — thu `--hidden-channels 64` truoc, neu van OOM thi moi dung `--skip-gat`. Ghi registry tuong tu.

### Buoc 5 — Bootstrap CI

Bootstrap **se retrain lai best C2 model** tim duoc tu surrogate CSV de lay predictions cho CI computation. Neu best model la `gat_raw_attr`, no se rerun GAT. Vi vay bootstrap phai match day du cac hyperparameters/quy tac lua chon da dung khi tao surrogate CSV: `--gat-heads`, `--appnp-alpha`, `--appnp-k`, `--hidden-channels`, va `--gnn-std-threshold` (neu dung de loai model variance cao). Neu Buoc 3/4 dung `--skip-gat`, bootstrap se chon model khac (GCN/GIN/SAGE) va khong can chay GAT.

**LENH MAC DINH — dung khi A0 + HSCC deu chay mac dinh tren A100**

```powershell
python bootstrap_ci.py --surrogate-csv-a0 outputs/mapr2026_v3_results/surrogate_ranking_metrics_a0_clean.csv --surrogate-csv-hscc outputs/mapr2026_v3_results/surrogate_ranking_metrics_hscc_clean.csv --targets-a0 data/processed/regression_targets_a0.parquet --targets-hscc data/processed/regression_targets_hscc_refined.parquet --include-language-hscc --n-bootstrap 1000 --equivalence-bound 0.02 --gat-heads 4 --appnp-alpha 0.15 --gnn-std-threshold 0.1 --out-a0 outputs/mapr2026_v3_results/gnn_vs_degree_bootstrap_ci_a0.json --out-hscc outputs/mapr2026_v3_results/gnn_vs_baseline_bootstrap_ci_hscc.json
```

> **Lenh tren la cho truong hop DEFAULT (hidden=128, appnp-k=10).**
> - `--gnn-std-threshold 0.1`: tu dong loai model co `spearman_rho_std > 0.1` khi chon best C2 (dam bao APPNP std=0.697 bi loai khoi bootstrap, nhat quan voi checklist paper).
> - Neu Buoc 3/4 co doi `--gat-heads`, `--appnp-alpha`, `--appnp-k`, hoac `--hidden-channels`, PHAI them cung gia tri vao lenh nay de bootstrap retrain dung cau hinh.
> - Neu paper workflow dang dung `--gnn-std-threshold 0.1`, bootstrap cung phai giu dung nguong do; khong duoc de bootstrap chon best model bang mot nguong khac surrogate/paper table.
> - Neu Buoc 3/4 dung `--skip-gat`, bootstrap tu dong chon best model khac (GCN/GIN/SAGE).
> - Cach nho don gian: neu A0/HSCC deu chay MAC DINH tren A100, thi dung nguyen lenh bootstrap MAC DINH nay. Chi them flags vao bootstrap neu ban DA that su dung fallback khi train surrogate.

**LENH FALLBACK — khong co mot lenh co dinh rieng**

> Neu Person 3 da dung fallback o Buoc 3 hoac Buoc 4, thi KHONG copy may moc lenh mac dinh o tren.
> Hay lay LENH MAC DINH o tren va them/chinh lai DUNG nhung flags da dung trong surrogate run:
> - `--hidden-channels 64` neu da dung FALLBACK A
> - `--skip-gat` khong can them vao bootstrap, nhung bootstrap se tu chon model khac neu surrogate CSV khong con `gat_raw_attr`
> - `--appnp-k`, `--appnp-alpha`, `--gat-heads`, `--gnn-std-threshold` phai giu dung nhu run surrogate/paper workflow

Verify:

- `gnn_vs_degree_bootstrap_ci_a0.json` co:
  - `feature_policy.include_language = false`
  - `feature_policy.hidden_channels` (khop voi gia tri da dung o Buoc 3)
  - `feature_policy.appnp_k` (khop voi gia tri da dung o Buoc 3)
  - `feature_policy.gnn_std_threshold = 0.1`
  - `surrogate_csv_used`
  - `interpretation`
- `gnn_vs_baseline_bootstrap_ci_hscc.json` co:
  - `feature_policy.include_language = true`
  - `feature_policy.hidden_channels` (khop voi gia tri da dung o Buoc 4)
  - `feature_policy.gnn_std_threshold = 0.1`
  - `surrogate_csv_used`
  - `interpretation`

## 5. Node2Vec truoc khi freeze-final

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

## 6. Cleanup shared artifacts (SAU khi Buoc 3 va 4 xong)

> **Luu y:** `runtime_breakdown.csv` da duoc cleanup truoc (0 null rows). Chay lenh nay them 1 lan nua sau khi GNN rerun ket thuc de dam bao khong co row null moi.

```powershell
python -c "import pathlib, pandas as pd; p = pathlib.Path('../..') / 'outputs/mapr2026_v3_results/runtime_breakdown.csv'; rt = pd.read_csv(p); ANALYTICAL = {'betweenness','degree','kshell','pagerank','one_hop','two_hop','phi','life_time','views','views_day','lr_life_time','lr_views_life_time','lr_phi','lr_degree_views_life_time','mc_ic_labeling','diffusion_proxies'}; STALE_GNN = {'gnn_centrality','gnn_full','gnn_graph_only','gnn_random','gnn_raw_attr','mlp_raw_attr','node2vec_lr'}; null_mask = rt['label_regime'].isna() | (rt['label_regime'].astype(str).str.strip() == ''); rt.loc[null_mask & rt['model_name'].isin(ANALYTICAL), 'label_regime'] = 'analytical'; rt = rt[~(null_mask & rt['model_name'].isin(STALE_GNN))].reset_index(drop=True); rt.to_csv(p, index=False); print(rt.groupby('label_regime')['model_name'].count().to_dict()); print('Updated:', p.resolve())"
```

Ket qua expected sau rerun: `{'a0': XX, 'hscc': XX, 'analytical': 16}` — khong co NaN.

## 7. Checklist truoc khi bao ket qua moi

- `baseline_ranking_metrics_a0_clean.csv`
  - khong co `*_lang`
  - co `label_regime = a0`
- `baseline_ranking_metrics_hscc_clean.csv`
  - co `lr_views_life_time_lang`
  - co `lr_degree_views_life_time_lang`
  - co `label_regime = hscc`
- `surrogate_ranking_metrics_a0_clean.csv`
  - co **8 hoac 9 models**: `gnn_raw_attr, gnn_graph_only, gnn_centrality, gnn_full, gcn_raw_attr, gin_raw_attr, appnp_raw_attr, best_arch_raw_attr_rankloss` + `gat_raw_attr` (neu khong OOM)
  - neu thieu `gat_raw_attr`: OK — ghi registry `gat_excluded_reason`
  - kiem tra APPNP: neu `spearman_rho_std > 0.1` thi **khong dung row `appnp_raw_attr` trong paper table** VA bootstrap da tu dong loai no (do `--gnn-std-threshold 0.1`), ghi registry `appnp_excluded_reason: high_variance`
  - **KHONG co row stale tu lan chay cu** (file da bi xoa theo Section 3b)
  - co `label_regime = a0`
- `surrogate_ranking_metrics_hscc_clean.csv`
  - co **8 hoac 9 models** nhu tren (cung logic GAT/APPNP)
  - kiem tra APPNP std tuong tu
  - **KHONG co row stale tu lan chay cu** (file da bi xoa theo Section 3b)
  - co `label_regime = hscc`
- `per_group_prediction_error.csv`
  - co cot `label_regime`
  - co rows cho `a0` va `hscc`
  - KHONG con stale null rows (da xoa file truoc khi chay theo Buoc 3b)
- `runtime_breakdown.csv`
  - khong con null `label_regime`
  - con row `mc_ic_labeling` voi `label_regime = analytical`
- 2 file bootstrap JSON
  - co `feature_policy.include_language`
  - co `feature_policy.hidden_channels` (khop voi gia tri dung o Buoc 3/4)
  - co `feature_policy.appnp_k` (khop voi gia tri dung o Buoc 3/4)
  - co `feature_policy.gnn_std_threshold = 0.1`
  - co `surrogate_csv_used`
  - co `interpretation`
- `docs/experiment_registry.md`
  - cap nhat `a0_feature_policy`
  - cap nhat `hscc_feature_policy`
  - cap nhat `gat_actual_heads` (4 neu chay duoc; hoac `gat_excluded_reason` neu skip)
  - cap nhat `gat_hidden_channels` (128 neu default; 64 neu dung `--hidden-channels 64`)
  - cap nhat `appnp_actual_alpha`
  - cap nhat `appnp_actual_k` (10 default; hoac gia tri khac neu tune)
  - cap nhat `appnp_excluded_reason` neu std > 0.1
  - cap nhat `hscc_fairness_baselines_confirmed`
  - cap nhat `node2vec_status`

## 8. Bao cao ket qua cho team

Khi gui ket qua moi cho team, nen bao cao toi thieu:

- best A0 baseline va best A0 GNN
- best HSCC flat baseline va best HSCC GNN
- 2 ket luan bootstrap:
  - A0: GNN vs degree
  - HSCC: GNN vs strongest flat baseline
- governance note: GAT status (`chay duoc hidden=128` / `chay voi hidden=64` / `excluded_OOM`)
- governance note: APPNP status (`stable std<0.1` / `excluded_high_variance std=X`)
- node2vec status: `rerun` / `carry_forward` / `de_scoped`
