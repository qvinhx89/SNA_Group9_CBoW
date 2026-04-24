## I. VERIFICATION SUMMARY

| Finding | Kết luận | Hệ quả cho plan |
|---|---|---|
| `bootstrap_ci.py` đang dùng 1 surrogate CSV chung và chưa truyền feature policy theo regime | Đúng | Phải có F9 để tách A0/HSCC và truyền `include_language` cho HSCC |
| `--dry-run` có thể overwrite artifact thật | Đúng | Mọi dry-run phải ghi ra file riêng |
| `per_group_prediction_error.csv` thiếu `label_regime` | Đúng | Phải thêm `label_regime` + upsert key 3 cột |
| `runtime_breakdown.csv` không được fill null thành `"analytical"` hàng loạt | Đúng | Chỉ tag analytical rows; drop stale regime-dependent rows |
| Comparator HSCC trong bootstrap hiện có nguy cơ thiếu `language` | Đúng | Baseline/GNN/CI phải dùng cùng feature policy |
| `gat_heads` / `appnp_alpha` nên là override có governance, không phải knob tự do | Hợp lý | Default bám plan; override phải log rõ |
| Node2Vec không nên bị bỏ âm thầm khỏi baseline contract | Đúng | Có thể skip ở debug pass, nhưng freeze cuối phải rerun hoặc carry-forward có note |
| `HSCC in_dim = 24` chỉ là expected value của snapshot hiện tại, không phải invariant của plan | Đúng | Validation chính thức nên check `3 + n_lang_dummies`; số `24` chỉ dùng như reference snapshot |
| Artifacts của Person 1/2 đang unblock Person 3 theo nghĩa **data dependency**, không phải toàn bộ workflow | Đúng | Team có thể bắt đầu fix/rerun ngay, nhưng F1-F9 + registry + Node2Vec freeze rule vẫn là pre-freeze blockers |
| Carry-forward Node2Vec chỉ hợp lệ nếu có legacy row đúng regime và row đó được graft vào clean CSV cuối | Đúng | Không đủ chỉ “ghi note”; artifact freeze-final phải vẫn có `node2vec_lr` hoặc de-scope rõ ràng |
| `ic_scores_hscc_refined.parquet` hiện ở `outputs/`, không ở `data/processed/` | Đúng | Không block Person 3 vì rerun dùng `regression_targets_hscc_refined.parquet`; không nên hard-code path HSCC IC score mới nếu chưa normalize artifact location |

Tóm lại, technical diagnosis của plan là đúng; phần cần siết là **governance**, **artifact hygiene**, và **feature-policy parity**.

---

## II. KẾ HOẠCH SỬA REVISED — FINAL v2.0

### Tổng bảng thay đổi (revised)

| # | File | Thay đổi | Mức độ |
|---|---|---|---|
| F1 | `run_surrogates.py` | Xóa hack lines 1015–1016 | 🔴 |
| F2 | `run_surrogates.py` | `_derive_features()` + `load_surrogate_data_bundle()` + CLI `--include-language` | 🔴 |
| F3 | `run_baselines.py` | `_derive_features()` + `load_baseline_data_bundle()` + `collect_heuristic_rows()` + CLI `--include-language` | 🔴 |
| F4 | `run_surrogates.py` | GAT heads: expose param, fix architecture, add governance log | 🟡 |
| F5 | `run_surrogates.py` | APPNP alpha: expose param, fix architecture, add governance log | 🟡 |
| F6 | `run_surrogates.py` | `per_group`: thêm `label_regime`, upsert key 3-column, overwrite→upsert | 🟡 |
| F7 | `run_surrogates.py` | Log `[FEATURE AUDIT]` in_dim + feature names (thật, không chỉ feature count) | 🟡 |
| F8 | `runtime_breakdown.csv` | Selective cleanup: drop stale GNN rows, tag analytical rows | ⚪ |
| **F9** | **`bootstrap_ci.py`** | **Split surrogate CSV args + pass `include_language` per regime** | 🔴 |

---

## PHASE 0 — CODE CHANGES

### F1: Xóa hack (run_surrogates.py)

Xóa đúng 2 dòng 1015–1016. Không thay bằng gì.

```python
# XÓA hai dòng này:
    # Temporary hack to ONLY run gat_raw_attr
    model_specs = [("gat_raw_attr", "raw_attr", False, "gat")]
```

---

### F2-F3: Language policy control (`run_surrogates.py` + `run_baselines.py`)

Mục tiêu của cụm này là **ngăn auto-include `language` ở mọi regime** và biến feature policy thành một quyết định explicit.

**Common pattern cần áp dụng cho cả hai file:**

```python
def _derive_features(
    node_attributes: pd.DataFrame,
    include_language: bool = False,
) -> pd.DataFrame:
    ...
    features = pd.DataFrame(
        {
            "node_id": df["node_id"],
            "views_log": np.log1p(views_raw).astype(float),
            "views_per_day": (views_raw / life_time).astype(float),
            "life_time": life_time.astype(float),
        }
    )

    if include_language:
        lang_col = "language" if "language" in df.columns else ("lang" if "lang" in df.columns else None)
        if lang_col is not None:
            lang_series = df[lang_col].astype(str).fillna("unknown").replace({"nan": "unknown", "None": "unknown"})
            lang_dummies = pd.get_dummies(lang_series, prefix="lang", dtype=float)
            features = pd.concat([features, lang_dummies], axis=1)
        else:
            print("[WARN] --include-language set but no 'language'/'lang' column found in node_attributes.")

    return features
```

**Touchpoints bắt buộc:**

| File | Các chỗ phải sửa |
|---|---|
| `run_surrogates.py` | `_derive_features()`; `load_surrogate_data_bundle(..., include_language=False)`; `parse_args()` thêm `--include-language`; truyền flag ở vòng lặp `model_specs` và block C3 rankloss |
| `run_baselines.py` | `_derive_features()`; `load_baseline_data_bundle(..., include_language=False)`; `collect_heuristic_rows(..., include_language=False)`; `parse_args()` thêm `--include-language`; truyền flag trong `main()` |

**Quy tắc giữ nguyên:** phần lang-aware baselines trong `run_baselines.py` không cần viết lại; chúng chỉ nên auto-trigger khi `derived` thật sự có `lang_*` columns.

---

### F4-F5: GAT / APPNP config với governance log

Phần này chỉ cần một nguyên tắc rõ: **default phải khớp plan**, còn override chỉ là fallback có ghi log.

**Thay đổi chính trong `GNNSurrogateRegressor`:**

```python
class GNNSurrogateRegressor(nn.Module):
    def __init__(
        self,
        arch: str,
        in_channels: int,
        hidden_channels: int = 128,
        dropout: float = 0.3,
        gat_heads: int = 4,
        appnp_alpha: float = 0.15,
    ) -> None:
        ...
        elif self.arch == "gat":
            per_head_dim = max(1, hidden_channels // gat_heads)
            self.conv1 = GATConv(in_channels, per_head_dim, heads=gat_heads, concat=True, dropout=dropout)
            self.conv2 = GATConv(per_head_dim * gat_heads, per_head_dim, heads=gat_heads, concat=True, dropout=dropout)
            self.head = nn.Linear(per_head_dim * gat_heads, 1)
        elif self.arch == "appnp":
            self.lin1 = nn.Linear(in_channels, hidden_channels)
            self.lin2 = nn.Linear(hidden_channels, hidden_channels)
            self.propagation = APPNP(K=10, alpha=appnp_alpha, dropout=dropout)
            self.head = nn.Linear(hidden_channels, 1)
```

**Các điểm phải chạm thêm:**
- thêm `gat_heads` và `appnp_alpha` vào `train_surrogate_5seeds(...)`;
- in `[GOVERNANCE WARN]` nếu giá trị khác plan (`4` / `0.15`);
- thêm 2 CLI args `--gat-heads`, `--appnp-alpha`;
- truyền 2 giá trị này ở cả loop chính và block C3 rankloss.

---

### F6-F7: Audit outputs (`per_group_prediction_error.csv` + feature audit)

**`per_group_prediction_error.csv`**
- thêm `label_regime` vào signature, row dict và schema;
- đổi overwrite sang upsert;
- key mới: `["label_regime", "model_name", "typology_group"]`;
- truyền `label_regime` từ `main()` vào hàm ghi file.

Snippet cốt lõi:

```python
cols_order = ["label_regime", "model_name", "typology_group", "n_nodes", "spearman_rho", "mae"]
...
combined = combined.drop_duplicates(
    subset=["label_regime", "model_name", "typology_group"], keep="last"
)
```

**Feature audit**
- thêm `feature_names: list[str]` vào `SurrogateDataBundle`;
- trả `feature_names=list(feature_cols)` từ `load_surrogate_data_bundle()`;
- log ngay khi build bundle đầu tiên:

```python
print(
    f"[FEATURE AUDIT] model={model_name} | regime={label_regime} | "
    f"feature_mode={feature_mode} | include_language={bool(args.include_language)} | "
    f"in_dim={in_dim} | features={bundle.feature_names} | "
    f"n_nodes={n_nodes} | n_edges={n_edges}"
)
```

Nếu log quá dài, giữ thêm một dòng phụ:

```python
print(f"[FEATURE AUDIT] first_features={bundle.feature_names[:10]}")
```

Lưu ý để team implement đúng: `bootstrap_ci.py` gọi `load_surrogate_data_bundle()` qua `_predict_gnn_best()`. Vì vậy nếu `feature_names` được lưu trong `SurrogateDataBundle` và `[FEATURE AUDIT]` được emit ngay tại loader / ngay sau lần build bundle đầu tiên, thì bootstrap path sẽ **inherit audit này tự động**. Không cần tạo thêm một luồng `feature_names` riêng chỉ cho bootstrap; thứ bắt buộc là **propagate `include_language` đúng theo regime**.

---

### F8-F9: `bootstrap_ci.py` + runtime cleanup

**`bootstrap_ci.py` cần 3 thay đổi nguyên tắc:**

1. **Tách input surrogate theo regime**
   - thêm `--surrogate-csv-a0`
   - thêm `--surrogate-csv-hscc`
   - giữ `--surrogate-csv` cũ chỉ như fallback legacy
   - **quan trọng:** hai arg mới nên default `""` / `None`, không nên default cùng một path với `--surrogate-csv`, nếu không fallback branch sẽ thành dead code

2. **Truyền feature policy đúng theo regime**
   - A0: `include_language=False`
   - HSCC: `include_language=True` khi rerun surrogate dùng language
   - propagate flag này qua:
     - `_predict_gnn_best(...)`
     - `_build_linear_predictions(...)`
     - `_predict_mlp_raw_attr(...)`
     - `_select_strongest_flat_baseline_hscc(...)`

3. **Ghi trace vào JSON output**

```python
payload_hscc["feature_policy"] = {
    "include_language": include_language_hscc,
    "gnn_model": best_hscc_name,
}
payload_a0["feature_policy"] = {
    "include_language": False,
    "gnn_model": best_a0_name,
}
```

**Function-level spec tối thiểu để implement không bị mơ hồ:**

| Hàm | Thay đổi bắt buộc |
|---|---|
| `_predict_gnn_best(...)` | thêm `include_language: bool`; truyền xuống `load_surrogate_data_bundle(..., include_language=include_language)` |
| `_build_linear_predictions(...)` | thêm `include_language: bool`; gọi `derive_features(node_attributes, include_language=include_language)` |
| `_predict_mlp_raw_attr(...)` | thêm `include_language: bool`; gọi `derive_features(node_attributes, include_language=include_language)` |
| `_select_strongest_flat_baseline_hscc(...)` | thêm `include_language: bool`; propagate flag này xuống `_build_linear_predictions(...)`, `_predict_mlp_raw_attr(...)`, và `derive_features(...)` nội bộ |
| `parse_args()` | thêm `--surrogate-csv-a0`, `--surrogate-csv-hscc`, `--include-language-hscc`; để arg mới default rỗng để legacy fallback thực sự có tác dụng |
| `main()` | resolve `surrogate_csv_a0` / `surrogate_csv_hscc`; A0 luôn `include_language=False`; HSCC dùng `bool(args.include_language_hscc)`; ghi trace vào JSON output |

Gợi ý resolve logic trong `main()`:

```python
surrogate_csv_a0 = resolve_project_path(args.surrogate_csv_a0) if str(args.surrogate_csv_a0).strip() else resolve_project_path(args.surrogate_csv)
surrogate_csv_hscc = resolve_project_path(args.surrogate_csv_hscc) if str(args.surrogate_csv_hscc).strip() else resolve_project_path(args.surrogate_csv)
```

`_select_best_gnn_model_name()` không bắt buộc đổi signature nếu `main()` đã truyền đúng file per-regime vào nó.

Nên ghi thêm trace để debug/freeze rõ ràng hơn:

```python
payload_a0["surrogate_csv_used"] = str(surrogate_csv_a0)
payload_hscc["surrogate_csv_used"] = str(surrogate_csv_hscc)
```

**Runtime cleanup**

Thay vì fill null blanket, chỉ làm 2 việc:
- tag null rows thuộc analytical/pipeline thành `"analytical"`,
- drop null rows thuộc stale GNN/MLP/Node2Vec để chuẩn bị cho rerun sạch.

```python
null_mask = rt["label_regime"].isna() | (rt["label_regime"].astype(str).str.strip() == "")
rt.loc[null_mask & rt["model_name"].isin(ANALYTICAL_MODELS), "label_regime"] = "analytical"
rt = rt[~(null_mask & rt["model_name"].isin(STALE_REGIME_MODELS))].reset_index(drop=True)
```

Snapshot hiện tại của repo cho thấy null-regime rows đang rơi vào đúng 2 nhóm sau:

- **stale regime-dependent rows cần drop trước rerun sạch:** `gnn_centrality`, `gnn_full`, `gnn_graph_only`, `gnn_random`, `gnn_raw_attr`, `mlp_raw_attr`, `node2vec_lr`
- **analytical/pipeline rows có thể tag thành `analytical`:** `betweenness`, `degree`, `diffusion_proxies`, `kshell`, `life_time`, `lr_degree_views_life_time`, `lr_life_time`, `lr_phi`, `lr_views_life_time`, `mc_ic_labeling`, `one_hop`, `pagerank`, `phi`, `two_hop`, `views`, `views_day`

---

## PHASE 1 — LOCK FEATURE POLICY (Team decision — ghi vào experiment_registry.md)

```markdown
## Feature Policy Lock (v3.2 — post rerun)

| Regime | include_language | in_dim | Rationale |
|--------|-----------------|--------|-----------|
| A0     | False           | 3      | A0 IC = degree-proxy; language irrelevant |
| HSCC   | True            | `3 + n_lang_dummies` (current snapshot: 24) | HSCC IC = φ(u) × cross-community; language là proxy community mạnh trong snapshot hiện tại |

Fairness requirement: HSCC GNN với language → baselines phải có lr_views_life_time_lang + lr_degree_views_life_time_lang.
GAT actual heads: 4 (hoặc ghi 2 + lý do nếu VRAM fail)
APPNP actual alpha: 0.15
```

Ghi chú: con số `24` chỉ là **expected value trên snapshot hiện tại** của `node_attributes.parquet` (3 numeric + 21 language dummies). Validation thực tế nên dựa trên **feature audit output**, không hard-code tuyệt đối.

Wording defensible nên dùng khi share cho team: artifacts của Person 1/2 hiện **“unblocked for data dependency”**, không nên diễn đạt thành **“không còn blocker nào”**, vì blocker ở tầng code/governance trước freeze vẫn còn nằm ở F1-F9, Node2Vec completion rule, registry update, và bootstrap rerun sạch.

Suggested minimum registry payload sau khi rerun:

```markdown
- a0_feature_policy: include_language=False, expected_in_dim=3
- hscc_feature_policy: include_language=True, expected_in_dim=3+n_lang_dummies (snapshot at rerun time: <fill actual>)
- gat_actual_heads: 4 | <override value + reason>
- appnp_actual_alpha: 0.15 | <override value + reason>
- hscc_fairness_baselines_confirmed: yes/no + comparator rows listed
- node2vec_status: rerun | carry_forward | de_scoped
- node2vec_source_artifact: <path or legacy artifact id if carry-forward>
```

---

## PHASE 2 — RERUN COMMANDS (Safe, correct order)

Operational note: các command dưới đây giả định chạy từ `src/mapr2026_v3`. Các path kiểu `data/...` và `outputs/...` vẫn hợp lệ vì các script hiện tại resolve chúng theo `PROJECT_ROOT` qua `resolve_project_path()`.

### Bước 2.0 — Dry-run AN TOÀN (ghi sang file riêng, không đụng artifacts thật)

```bash
cd src/mapr2026_v3

# A0 baselines — dry-run an toàn
python run_baselines.py --dry-run \
  --targets-path data/processed/regression_targets_a0.parquet \
  --label-regime a0 \
  --out-csv outputs/mapr2026_v3_results/baseline_a0_DRYRUN.csv

# HSCC baselines — dry-run an toàn
python run_baselines.py --dry-run \
  --targets-path data/processed/regression_targets_hscc_refined.parquet \
  --label-regime hscc \
  --include-language \
  --out-csv outputs/mapr2026_v3_results/baseline_hscc_DRYRUN.csv

# A0 GNN — dry-run an toàn
python run_surrogates.py --dry-run \
  --targets-path data/processed/regression_targets_a0.parquet \
  --label-regime a0 \
  --out-csv outputs/mapr2026_v3_results/surrogate_a0_DRYRUN.csv

# HSCC GNN — dry-run an toàn
python run_surrogates.py --dry-run \
  --targets-path data/processed/regression_targets_hscc_refined.parquet \
  --label-regime hscc \
  --include-language \
  --out-csv outputs/mapr2026_v3_results/surrogate_hscc_DRYRUN.csv

# Bootstrap CI — dry-run
python bootstrap_ci.py --dry-run \
  --surrogate-csv-a0 outputs/mapr2026_v3_results/surrogate_a0_DRYRUN.csv \
  --surrogate-csv-hscc outputs/mapr2026_v3_results/surrogate_hscc_DRYRUN.csv \
  --include-language-hscc
```

### Node2Vec policy trước khi rerun baseline

`--skip-node2vec` được phép dùng ở **vòng rerun debug nhanh** để tập trung fix fairness/GNN trước. Tuy nhiên, trước khi freeze bảng baseline cuối, team phải chốt một trong hai hướng:

1. **rerun Node2Vec đúng regime** rồi append row tương ứng vào baseline CSV sạch, hoặc
2. **carry-forward Node2Vec row cũ** chỉ nếu graph/split/protocol không đổi và có note rõ trong registry rằng row này không bị ảnh hưởng bởi language-policy fix.

Khuyến nghị thực tế:
- vòng 1 dùng `--skip-node2vec` để lấy signal nhanh,
- vòng freeze cuối phải có Node2Vec row hoặc de-scope note rõ ràng.

Điều kiện carry-forward nên viết chặt để tránh hiểu sai:

- phải có **legacy row đúng regime** và provenance đáng tin;
- row đó phải được **graft vào file clean cuối** (không chỉ để nằm ở artifact legacy riêng lẻ);
- registry phải ghi rõ source artifact, lý do hợp lệ, và vì sao language-policy fix không ảnh hưởng Node2Vec path.

Snapshot hiện tại của repo:

- `baseline_ranking_metrics.csv` có `node2vec_lr` legacy row;
- `baseline_ranking_metrics_hscc.csv` **không** có `node2vec_lr`.

Vì vậy, với những artifact đang thấy trong repo hiện tại, **HSCC carry-forward không nên assume là available**. Safe default là: nếu team không có một trusted HSCC Node2Vec artifact khác, thì **HSCC Node2Vec phải rerun trước freeze**.

### Bước 2.1 — A0 Baselines (fast fairness/debug pass)

```bash
python run_baselines.py \
  --targets-path data/processed/regression_targets_a0.parquet \
  --label-regime a0 \
  --out-csv outputs/mapr2026_v3_results/baseline_ranking_metrics_a0_clean.csv \
  --skip-node2vec
# Sau khi xong, kiểm tra: không có lr_views_life_time_lang (đúng — A0 no language)
```

### Bước 2.2 — HSCC Baselines (fast fairness/debug pass)

```bash
python run_baselines.py \
  --targets-path data/processed/regression_targets_hscc_refined.parquet \
  --label-regime hscc \
  --include-language \
  --out-csv outputs/mapr2026_v3_results/baseline_ranking_metrics_hscc_clean.csv \
  --skip-node2vec
# Sau khi xong, kiểm tra: PHẢI có lr_views_life_time_lang và lr_degree_views_life_time_lang
```

### Bước 2.2b — Node2Vec completion rule trước freeze

Trước khi đóng băng artifact cuối:

- nếu **rerun Node2Vec**, dùng chính `run_baselines.py` **không** bật `--skip-node2vec` cho từng regime rồi xác nhận row `node2vec_lr` xuất hiện trong file clean;
- nếu **không rerun Node2Vec**, phải ghi note vào `experiment_registry.md` rằng:
  - row Node2Vec được carry-forward từ artifact cũ,
  - graph/split không đổi,
  - language-policy fix không ảnh hưởng Node2Vec path,
  - naming paper-facing sẽ được thống nhất là `Node2Vec + LR` (dù row code hiện tên `node2vec_lr`).

### Bước 2.3 — A0 GNN

```bash
python run_surrogates.py \
  --targets-path data/processed/regression_targets_a0.parquet \
  --label-regime a0 \
  --out-csv outputs/mapr2026_v3_results/surrogate_ranking_metrics_a0_clean.csv \
  --include-c2-arch \
  --include-c3-rankloss \
  --gat-heads 4 \
  --appnp-alpha 0.15
# Nếu VRAM < 14GB và GAT OOM: thêm --gat-heads 2, ghi vào experiment_registry.md
```

### Bước 2.4 — HSCC GNN

```bash
python run_surrogates.py \
  --targets-path data/processed/regression_targets_hscc_refined.parquet \
  --label-regime hscc \
  --out-csv outputs/mapr2026_v3_results/surrogate_ranking_metrics_hscc_clean.csv \
  --include-language \
  --include-c2-arch \
  --include-c3-rankloss \
  --gat-heads 4 \
  --appnp-alpha 0.15
# Kiểm tra [FEATURE AUDIT] log: include_language=True và in_dim = 3 + n_lang_dummies
```

### Bước 2.5 — Bootstrap CI

```bash
python bootstrap_ci.py \
  --surrogate-csv-a0  outputs/mapr2026_v3_results/surrogate_ranking_metrics_a0_clean.csv \
  --surrogate-csv-hscc outputs/mapr2026_v3_results/surrogate_ranking_metrics_hscc_clean.csv \
  --targets-a0  data/processed/regression_targets_a0.parquet \
  --targets-hscc data/processed/regression_targets_hscc_refined.parquet \
  --include-language-hscc \
  --n-bootstrap 1000 \
  --equivalence-bound 0.02 \
  --out-a0   outputs/mapr2026_v3_results/gnn_vs_degree_bootstrap_ci_a0.json \
  --out-hscc outputs/mapr2026_v3_results/gnn_vs_baseline_bootstrap_ci_hscc.json
```

### Bước 2.6 — Runtime cleanup + validation

```bash
python -c "
# Chạy cleanup script F8
import pandas as pd
rt = pd.read_csv('outputs/mapr2026_v3_results/runtime_breakdown.csv')
ANALYTICAL = {'betweenness','degree','kshell','pagerank','one_hop','two_hop',
              'phi','life_time','views','views_day','lr_life_time',
              'lr_views_life_time','lr_phi','lr_degree_views_life_time',
              'mc_ic_labeling','diffusion_proxies'}
STALE_GNN  = {'gnn_centrality','gnn_full','gnn_graph_only','gnn_random',
              'gnn_raw_attr','mlp_raw_attr','node2vec_lr'}
null_mask  = rt['label_regime'].isna() | (rt['label_regime'].astype(str).str.strip() == '')
rt.loc[null_mask & rt['model_name'].isin(ANALYTICAL), 'label_regime'] = 'analytical'
rt = rt[~(null_mask & rt['model_name'].isin(STALE_GNN))].reset_index(drop=True)
rt.to_csv('outputs/mapr2026_v3_results/runtime_breakdown.csv', index=False)
print('Done:', rt.groupby(\"label_regime\")[\"model_name\"].count().to_dict())
"
```

---

## PHASE 3 — VALIDATION CHECKLIST

```
FEATURE POLICY
□ [FEATURE AUDIT] log A0: in_dim=3, include_language=False
□ [FEATURE AUDIT] log A0: feature_names chỉ gồm views_log, views_per_day, life_time
□ [FEATURE AUDIT] log HSCC: include_language=True và in_dim = 3 + n_lang_dummies
□ [FEATURE AUDIT] log HSCC: feature_names có prefix lang_* đúng như snapshot hiện tại

BASELINE COMPLETENESS
□ baseline_ranking_metrics_a0_clean.csv: có cột `label_regime` và chỉ chứa giá trị `a0`
□ baseline_ranking_metrics_hscc_clean.csv: có cột `label_regime` và chỉ chứa giá trị `hscc`
□ baseline_ranking_metrics_a0_clean.csv: KHÔNG có lr_*_lang rows (correct)
□ baseline_ranking_metrics_hscc_clean.csv: CÓ lr_views_life_time_lang và lr_degree_views_life_time_lang
□ baseline CSV freeze-final: có row `node2vec_lr` tương ứng mỗi regime, hoặc de-scope rõ ràng
□ nếu carry-forward Node2Vec: row `node2vec_lr` thực sự đã được graft vào file clean tương ứng + registry có source artifact
□ với snapshot hiện tại: không dùng `baseline_ranking_metrics_hscc.csv` làm bằng chứng carry-forward HSCC vì file này không có `node2vec_lr`

SURROGATE COMPLETENESS
□ surrogate_ranking_metrics_a0_clean.csv: có cột `label_regime` và chỉ chứa giá trị `a0`
□ surrogate_ranking_metrics_hscc_clean.csv: có cột `label_regime` và chỉ chứa giá trị `hscc`
□ surrogate_ranking_metrics_a0_clean.csv: có 9 models (gnn_raw_attr, gnn_graph_only,
    gnn_centrality, gnn_full, gcn_raw_attr, gin_raw_attr, gat_raw_attr,
    appnp_raw_attr, best_arch_raw_attr_rankloss)
□ surrogate_ranking_metrics_hscc_clean.csv: cùng 9 models
□ Không có row nào có spearman_rho_std cực lớn (ví dụ > 0.05) mà không có explanation / governance note rõ

PER GROUP
□ per_group_prediction_error.csv: có cột label_regime
□ Có rows cho a0 VÀ hscc, nhiều hơn 1 model

BOOTSTRAP CI
□ bootstrap_ci --dry-run hiển thị đúng `surrogate_csv_a0` và `surrogate_csv_hscc` resolved paths
□ gnn_vs_degree_bootstrap_ci_a0.json: tồn tại, có "feature_policy.include_language=false"
□ gnn_vs_baseline_bootstrap_ci_hscc.json: tồn tại, có "feature_policy.include_language=true"
□ cả 2 JSON có `surrogate_csv_used`
□ Cả 2 file có "interpretation" field rõ ràng

RUNTIME
□ runtime_breakdown.csv không còn rows có label_regime=null
□ mc_ic_labeling còn tồn tại với label_regime="analytical"

GOVERNANCE
□ experiment_registry.md cập nhật: gat_actual_heads, appnp_actual_alpha,
    a0_feature_policy, hscc_feature_policy, hscc_fairness_baselines_confirmed
```