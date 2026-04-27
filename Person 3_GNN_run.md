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
  - `rankloss_alpha = 0.5` (chi can giu khop khi bat C3 rankloss/bootstrap)
- **Quyet dinh chinh thuc (30/4)**: GAT da bi drop chinh thuc do OOM tren A100-40GB o `hidden_channels=128`. LUON them `--skip-gat` vao lenh GNN. Ghi registry: `gat_excluded_reason: OOM_A100_40GB_hidden128`.
- **Cau hinh GNN chinh thuc**: `hidden_channels=128`, `appnp_k=10`, `--skip-gat`. Chi dung `--hidden-channels 64` neu cac model KHAC (SAGE/GCN/GIN/APPNP) van OOM sau khi da drop GAT.
- **Governance bat buoc**: bootstrap CI phai dung CUNG cau hinh voi run surrogate cho moi tham so co anh huong den retrain best C2 model: `--gat-heads`, `--appnp-alpha`, `--appnp-k`, `--hidden-channels`, va (neu dung paper filter) `--gnn-std-threshold`; neu bat `--include-rankloss-comparison` thi phai giu khop ca `--rankloss-alpha`. Neu surrogate dung `--hidden-channels 64`, bootstrap cung phai co `--hidden-channels 64`; neu surrogate/paper workflow dung `--gnn-std-threshold 0.1`, bootstrap cung phai dung `--gnn-std-threshold 0.1`.

## 2. Mo terminal va vao dung thu muc

```powershell
cd "D:\UIT\Y3 - S2\Social Network Analysis\SNA_Group9_CBoW\src\mapr2026_v3"
```

## 3. Dry-run TRUOC KHI chay that (bat buoc)

Chay tat ca 5 lenh nay truoc de kiem tra code khong loi import/path. Khong ton GPU time.

```powershell
python run_baselines.py --dry-run --targets-path data/processed/regression_targets_a0.parquet --label-regime a0 --out-csv outputs/mapr2026_v3_results/baseline_a0_DRYRUN.csv

python run_baselines.py --dry-run --targets-path data/processed/regression_targets_hscc_refined.parquet --label-regime hscc --include-language --out-csv outputs/mapr2026_v3_results/baseline_hscc_DRYRUN.csv

python run_surrogates.py --dry-run --targets-path data/processed/regression_targets_a0.parquet --label-regime a0 --out-csv outputs/mapr2026_v3_results/surrogate_a0_DRYRUN.csv --include-c2-arch --include-c3-rankloss --gat-heads 4 --appnp-alpha 0.15 --skip-gat

python run_surrogates.py --dry-run --targets-path data/processed/regression_targets_hscc_refined.parquet --label-regime hscc --include-language --out-csv outputs/mapr2026_v3_results/surrogate_hscc_DRYRUN.csv --include-c2-arch --include-c3-rankloss --gat-heads 4 --appnp-alpha 0.15 --skip-gat

python bootstrap_ci.py --dry-run --surrogate-csv-a0 outputs/mapr2026_v3_results/surrogate_a0_DRYRUN.csv --surrogate-csv-hscc outputs/mapr2026_v3_results/surrogate_hscc_DRYRUN.csv --include-language-hscc --include-rankloss-comparison --rankloss-alpha 0.5
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

## 4. Rerun chinh

> **THU TU RA QUYET DINH CHO PERSON 3**
> 1. LUON chay **LENH CHINH THUC** (co `--skip-gat`, hidden=128) — GAT da bi drop chinh thuc 30/4.
> 2. Chi neu SAGE/GCN/GIN/APPNP bi OOM sau khi da drop GAT thi moi dung **LENH FALLBACK** (`--skip-gat --hidden-channels 64`).
> 3. Sau khi da dung fallback, bootstrap phai them `--hidden-channels 64` tuong ung.

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

> ⚠️ **QUYET DINH CHINH THUC (30/4):** GAT da bi drop do OOM tren A100-40GB o hidden=128. LUON them `--skip-gat`. Ghi registry: `gat_excluded_reason: OOM_A100_40GB_hidden128`.

**LENH CHINH THUC**

```powershell
python run_surrogates.py --targets-path data/processed/regression_targets_a0.parquet --label-regime a0 --out-csv outputs/mapr2026_v3_results/surrogate_ranking_metrics_a0_clean.csv --include-c2-arch --include-c3-rankloss --gat-heads 4 --appnp-alpha 0.15 --skip-gat
```

Verify trong terminal:

- `[FEATURE AUDIT] include_language=False`
- `feature_names` chi gom `views_log`, `views_per_day`, `life_time` cho `raw_attr`
- `in_dim=3` voi `raw_attr`
- KHONG co dong `Training gat_raw_attr` trong output (xac nhan GAT da bi skip)

**LENH FALLBACK — chi dung neu SAGE/GCN/GIN/APPNP van OOM sau khi da drop GAT**

```powershell
python run_surrogates.py --targets-path data/processed/regression_targets_a0.parquet --label-regime a0 --out-csv outputs/mapr2026_v3_results/surrogate_ranking_metrics_a0_clean.csv --include-c2-arch --include-c3-rankloss --gat-heads 4 --appnp-alpha 0.15 --skip-gat --hidden-channels 64
```

Neu dung LENH FALLBACK nay: `--hidden-channels 64` anh huong den TOAN BO GNN con lai (SAGE/GCN/GIN/APPNP). Ghi registry: `all_models_hidden=64`. Bootstrap phai them `--hidden-channels 64` tuong ung.

### Buoc 4 — HSCC GNN

> ⚠️ **QUYET DINH CHINH THUC (30/4):** GAT da bi drop do OOM tren A100-40GB o hidden=128. LUON them `--skip-gat`. Ghi registry: `gat_excluded_reason: OOM_A100_40GB_hidden128`.

**LENH CHINH THUC**

```powershell
python run_surrogates.py --targets-path data/processed/regression_targets_hscc_refined.parquet --label-regime hscc --include-language --out-csv outputs/mapr2026_v3_results/surrogate_ranking_metrics_hscc_clean.csv --include-c2-arch --include-c3-rankloss --gat-heads 4 --appnp-alpha 0.15 --skip-gat
```

Verify trong terminal:

- `[FEATURE AUDIT] include_language=True`
- `in_dim = 3 + n_lang_dummies`
- snapshot hien tai expected la `24`
- `feature_names` co cac cot `lang_*`
- KHONG co dong `Training gat_raw_attr` trong output (xac nhan GAT da bi skip)

**LENH FALLBACK — chi dung neu SAGE/GCN/GIN/APPNP van OOM sau khi da drop GAT**

```powershell
python run_surrogates.py --targets-path data/processed/regression_targets_hscc_refined.parquet --label-regime hscc --include-language --out-csv outputs/mapr2026_v3_results/surrogate_ranking_metrics_hscc_clean.csv --include-c2-arch --include-c3-rankloss --gat-heads 4 --appnp-alpha 0.15 --skip-gat --hidden-channels 64
```

Neu dung LENH FALLBACK nay: `--hidden-channels 64` anh huong den TOAN BO GNN con lai (SAGE/GCN/GIN/APPNP). Ghi registry: `all_models_hidden=64`. Bootstrap phai them `--hidden-channels 64` tuong ung.

### Buoc 5 — Bootstrap CI

Bootstrap **se retrain lai best C2 model** tim duoc tu surrogate CSV de lay predictions cho CI computation. Vi Buoc 3/4 LUON dung `--skip-gat`, surrogate CSV khong co `gat_raw_attr` — bootstrap se tu dong chon best model trong {SAGE, GCN, GIN, APPNP}. Bootstrap phai match day du cac hyperparameters da dung khi tao surrogate CSV: `--gat-heads` (giu de consistency, du GAT khong duoc chon), `--appnp-alpha`, `--appnp-k`, `--hidden-channels`, `--gnn-std-threshold`, va (neu bat `--include-rankloss-comparison`) `--rankloss-alpha`.

**LENH CHINH THUC — GAT excluded; surrogate CSV chi co {SAGE, GCN, GIN, APPNP, non-C2 variants}**

```powershell
python bootstrap_ci.py --surrogate-csv-a0 outputs/mapr2026_v3_results/surrogate_ranking_metrics_a0_clean.csv --surrogate-csv-hscc outputs/mapr2026_v3_results/surrogate_ranking_metrics_hscc_clean.csv --targets-a0 data/processed/regression_targets_a0.parquet --targets-hscc data/processed/regression_targets_hscc_refined.parquet --include-language-hscc --n-bootstrap 1000 --equivalence-bound 0.02 --gat-heads 4 --appnp-alpha 0.15 --gnn-std-threshold 0.1 --include-rankloss-comparison --rankloss-alpha 0.5 --out-a0 outputs/mapr2026_v3_results/gnn_vs_degree_bootstrap_ci_a0.json --out-hscc outputs/mapr2026_v3_results/gnn_vs_baseline_bootstrap_ci_hscc.json --out-hscc-rankloss outputs/mapr2026_v3_results/gnn_vs_rankloss_bootstrap_ci_hscc.json
```

> **Lenh tren la cho truong hop DEFAULT (hidden=128, appnp-k=10, rankloss-alpha=0.5).**
> - `--gnn-std-threshold 0.1`: tu dong loai model co `spearman_rho_std > 0.1` khi chon best C2 (dam bao APPNP std=0.697 bi loai khoi bootstrap, nhat quan voi checklist paper).
> - `--include-rankloss-comparison`: chay them bootstrap CI cho `best_arch_raw_attr_rankloss` (C3 validation) vs strongest flat baseline HSCC; ket qua vao `gnn_vs_rankloss_bootstrap_ci_hscc.json`. Retrain lai best C2 arch bang `loss_mode=rankloss_combined` (alpha*Huber + (1-alpha)*PairwiseRank) — cung mode voi `run_surrogates.py --include-c3-rankloss`.
> - GAT da bi drop: `--gat-heads 4` van giu trong lenh de parameter consistency, nhung bootstrap se KHONG bao gio chon/retrain GAT (vi `gat_raw_attr` khong co trong surrogate CSV).
> - `--rankloss-alpha 0.5` la default va khop voi `run_surrogates.py` default. Neu da doi `--rankloss-alpha` khi chay Buoc 3/4, PHAI doi gia tri tuong ung o day.
> - Neu Buoc 3/4 co doi `--appnp-alpha`, `--appnp-k`, hoac `--hidden-channels`, PHAI them cung gia tri vao lenh nay de bootstrap retrain dung cau hinh.
> - Neu paper workflow dang dung `--gnn-std-threshold 0.1`, bootstrap cung phai giu dung nguong do; khong duoc de bootstrap chon best model bang mot nguong khac surrogate/paper table.
> - Vi Buoc 3/4 LUON dung `--skip-gat`, bootstrap da tu dong chon best model trong {SAGE, GCN, GIN, APPNP}.
> - Cach nho: neu A0/HSCC deu chay LENH CHINH THUC tren A100, dung nguyen lenh bootstrap CHINH THUC nay. Chi them `--hidden-channels 64` neu da dung LENH FALLBACK khi train surrogate.

**LENH FALLBACK bootstrap — chi dung neu da dung LENH FALLBACK (`--hidden-channels 64`) o Buoc 3/4**

> Lay LENH CHINH THUC o tren va them `--hidden-channels 64`. KHONG them `--skip-gat` vao bootstrap (no khong co tac dung o day — bootstrap tu chon best model tu CSV, GAT da vang mat tu truoc).
> `--appnp-k`, `--appnp-alpha`, `--gat-heads`, `--gnn-std-threshold`, va `--rankloss-alpha` giu nguyen.

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
- `gnn_vs_rankloss_bootstrap_ci_hscc.json` co:
  - `gnn_model` bat dau bang `best_arch_raw_attr_rankloss(` (vi du `best_arch_raw_attr_rankloss(sage)`)
  - `feature_policy.loss_mode = "rankloss_combined"`
  - `feature_policy.rankloss_alpha = 0.5` (hoac gia tri da truyen qua `--rankloss-alpha`)
  - `feature_policy.include_language = true`
  - `feature_policy.hidden_channels` (khop voi gia tri da dung o Buoc 4)
  - `feature_policy.gnn_std_threshold = 0.1`
  - `comparator_model` khop voi strongest flat baseline cua HSCC
  - `interpretation` (mot trong 4 gia tri: `gnn_significantly_better`, `gnn_significantly_worse`, `practically_equivalent`, `no_clear_superiority`)

## 5. Node2Vec — PHAI CHOT TRUOC KHI BAO CAO KET QUA

> ⚠️ **Quyet dinh bat buoc truoc Section 7**: Buoc 1/2 chay voi `--skip-node2vec` nen file baseline hien tai **khong co row `node2vec_lr`**. Phai chon 1 trong 3 option duoi day va ghi vao registry truoc khi bao ket qua cho team.

### Boi canh: tai sao phai quyet dinh?

- `node2vec_lr` la 1 baseline trong paper table (Group 4: Shallow Embedding).
- Neu khong lam gi them, file baseline se thieu row nay khi nop paper.
- Node2vec embedding training tren graph 168K nodes ton khoang 30–60 phut; LR fit tren embeddings co san chi ton ~2 phut.

---

### Option A — Rerun Node2Vec (RECOMMENDED neu con du thoi gian)

Chay lai Buoc 1/2 **khong co** `--skip-node2vec`. Node2vec se duoc train lai tu dau — can ~30–60 phut moi regime.

> **Preflight cho Option A:** Section 3 dry-run KHONG test Node2Vec path thuc te. Truoc khi commit rerun Node2Vec, chay import check sau:

```powershell
python -c "from torch_geometric.nn import Node2Vec; print('OK Node2Vec import')"
```

> Neu import fail, KHONG chon Option A cho den khi env duoc fix. Neu import pass nhung khi chay thuc te xuat hien `[WARN] Node2Vec dependency missing ...` hoac row `node2vec_lr` ra NaN, xem nhu Option A that bai va quay ve Option C (de-scope) hoac Option B neu provenance cu du tin cay.

A0:

```powershell
python run_baselines.py --targets-path data/processed/regression_targets_a0.parquet --label-regime a0 --out-csv outputs/mapr2026_v3_results/baseline_ranking_metrics_a0_clean.csv
```

HSCC:

```powershell
python run_baselines.py --targets-path data/processed/regression_targets_hscc_refined.parquet --label-regime hscc --include-language --out-csv outputs/mapr2026_v3_results/baseline_ranking_metrics_hscc_clean.csv
```

Verify: co row `node2vec_lr` trong ca 2 file **va cac metric chinh khong phai NaN**. Ghi registry: `node2vec_status: rerun_fresh`.

### Option B — Carry-forward tu artifact cu

Chi duoc dung neu **tat ca** dieu kien sau dung:

1. Co file artifact cu co row `node2vec_lr` cho DUNG regime (`a0` / `hscc`).
2. Row do duoc tao voi `include_language` khop (a0 = False, hscc = True).
3. Provenance ro rang (biet ro chay luc nao, voi config nao).

**KHONG dung Option B neu khong chac chan dieu kien 2** — HSCC node2vec voi `include_language=False` la sai cau hinh va ket qua se khong hop le.

Neu du dieu kien: graft thu cong row `node2vec_lr` tu artifact cu vao file clean. Ghi registry: `node2vec_status: carry_forward; source: <ten file cu>; verified: True`.

### Option C — De-scope Node2Vec (NHANH NHAT)

Neu khong co thoi gian va khong co trusted artifact: **bo node2vec khoi paper table**. Group 4 (Shallow Embedding) chi bao gom MLP hoac bo han. Ghi registry: `node2vec_status: de_scoped; reason: time_constraint`.

Khong can chay them gi. Update comment trong paper draft Section 4.1 (Table baseline) de bo dong node2vec_lr.

---

> **Ket luan khi chon xong**: Cap nhat `node2vec_status` trong `docs/experiment_registry.md` **truoc khi chay Section 7 checklist**.

## 6. Cleanup shared artifacts (SAU khi Buoc 3 va 4 xong)

> **Luu y:** `runtime_breakdown.csv` da duoc cleanup truoc (0 null rows). Chay lenh nay them 1 lan nua sau khi GNN rerun ket thuc de dam bao khong co row null moi.

```powershell
python -c "import pathlib, pandas as pd; p = pathlib.Path('../..') / 'outputs/mapr2026_v3_results/runtime_breakdown.csv'; rt = pd.read_csv(p); ANALYTICAL = {'betweenness','degree','kshell','pagerank','one_hop','two_hop','phi','life_time','views','views_day','lr_life_time','lr_views_life_time','lr_phi','lr_degree_views_life_time','mc_ic_labeling','diffusion_proxies'}; STALE_GNN = {'gnn_centrality','gnn_full','gnn_graph_only','gnn_random','gnn_raw_attr','mlp_raw_attr','node2vec_lr'}; null_mask = rt['label_regime'].isna() | (rt['label_regime'].astype(str).str.strip() == ''); rt.loc[null_mask & rt['model_name'].isin(ANALYTICAL), 'label_regime'] = 'analytical'; rt = rt[~(null_mask & rt['model_name'].isin(STALE_GNN))].reset_index(drop=True); rt = rt[rt['model_name'] != 'gat_raw_attr'].reset_index(drop=True); rt.to_csv(p, index=False); print(rt.groupby('label_regime')['model_name'].count().to_dict()); print('Updated:', p.resolve())"
```

> **Luu y ve GAT cleanup**: Lenh tren co them buoc xoa **tat ca** row `gat_raw_attr` (ca a0 lan hscc) vi GAT da bi drop chinh thuc. Buoc nay can thiet vi cac row GAT cu co `label_regime` hop le — cleanup null_mask cu khong xoa duoc chung.

Ket qua expected sau rerun: `{'a0': XX, 'hscc': XX, 'analytical': 16}` — khong co NaN, khong co `gat_raw_attr`.

## 7. Checklist truoc khi bao ket qua moi

- `baseline_ranking_metrics_a0_clean.csv`
  - khong co `*_lang`
  - co `label_regime = a0`
  - node2vec: co row `node2vec_lr` **va khong NaN** (neu chay Option A) HOAC confirmed carry-forward (Option B) HOAC khong co row (neu Option C de-scope) — phai khop voi quyet dinh o Section 5
- `baseline_ranking_metrics_hscc_clean.csv`
  - co `lr_views_life_time_lang`
  - co `lr_degree_views_life_time_lang`
  - co `label_regime = hscc`
  - node2vec: cung logic nhu tren — neu chay Option A thi row `node2vec_lr` cung phai khong NaN; phai khop quyet dinh Section 5
- `surrogate_ranking_metrics_a0_clean.csv`
  - co **8 models chinh thuc**: `gnn_raw_attr, gnn_graph_only, gnn_centrality, gnn_full, gcn_raw_attr, gin_raw_attr, appnp_raw_attr, best_arch_raw_attr_rankloss`
  - KHONG co `gat_raw_attr` (GAT da bi drop chinh thuc — ghi registry `gat_excluded_reason: OOM_A100_40GB_hidden128`)
  - kiem tra APPNP: neu `spearman_rho_std > 0.1` thi **khong dung row `appnp_raw_attr` trong paper table** VA bootstrap da tu dong loai no (do `--gnn-std-threshold 0.1`), ghi registry `appnp_excluded_reason: high_variance`
  - **KHONG co row stale tu lan chay cu** (file da bi xoa theo Section 3b)
  - co `label_regime = a0`
- `surrogate_ranking_metrics_hscc_clean.csv`
  - co **8 models chinh thuc** nhu tren
  - KHONG co `gat_raw_attr` (cung ly do — ghi registry tuong tu)
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
  - KHONG co row `gat_raw_attr` (da xoa boi Section 6 cleanup)
- 2 file bootstrap JSON chinh (C4)
  - co `feature_policy.include_language`
  - co `feature_policy.hidden_channels` (khop voi gia tri dung o Buoc 3/4)
  - co `feature_policy.appnp_k` (khop voi gia tri dung o Buoc 3/4)
  - co `feature_policy.gnn_std_threshold = 0.1`
  - co `surrogate_csv_used`
  - co `interpretation`
- `gnn_vs_rankloss_bootstrap_ci_hscc.json` (C3 rankloss bootstrap — HSCC only)
  - co `gnn_model` bat dau bang `best_arch_raw_attr_rankloss(`
  - co `feature_policy.loss_mode = "rankloss_combined"`
  - co `feature_policy.rankloss_alpha = 0.5` (hoac gia tri da truyen qua `--rankloss-alpha`)
  - co `feature_policy.include_language = true`
  - co `feature_policy.hidden_channels` khop voi Buoc 4
  - co `feature_policy.gnn_std_threshold = 0.1`
  - co `comparator_model` + `comparator_spearman_on_test`
  - co `interpretation` hop le
- `docs/experiment_registry.md`
  - cap nhat `a0_feature_policy`
  - cap nhat `hscc_feature_policy`
  - cap nhat `gat_excluded_reason: OOM_A100_40GB_hidden128` (GAT luon bi excluded — khong can `gat_actual_heads` hay `gat_hidden_channels`)
  - cap nhat `appnp_actual_alpha`
  - cap nhat `appnp_actual_k` (10 default; hoac gia tri khac neu tune)
  - cap nhat `appnp_excluded_reason` neu std > 0.1
  - cap nhat `hscc_fairness_baselines_confirmed`
  - cap nhat `node2vec_status`

## 8. Bao cao ket qua cho team

Khi gui ket qua moi cho team, nen bao cao toi thieu:

- best A0 baseline va best A0 GNN
- best HSCC flat baseline va best HSCC GNN
- 3 ket luan bootstrap:
  - A0 (C4): GNN vs degree
  - HSCC (C4): GNN vs strongest flat baseline
  - HSCC (C3): rankloss model vs strongest flat baseline
- governance note: GAT `excluded_OOM` chinh thuc (`gat_excluded_reason: OOM_A100_40GB_hidden128`). Neu co ket qua h=64 tu lan chay cu, co the ghi footnote trong paper voi ρ_A0=0.344, ρ_HSCC=0.513 nhung khong dua vao main table.
- governance note: APPNP status (`stable std<0.1` / `excluded_high_variance std=X`)
- node2vec status: `rerun` / `carry_forward` / `de_scoped`
