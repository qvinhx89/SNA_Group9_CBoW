# MAPR2026 v3.1 — Plan Flow (Task Dependencies + Parallelization)

Mục tiêu của file này: cho team nhìn **1 lần là hiểu**:

- Artifact nào **block** artifact nào (dependency DAG)
- Task nào của Person 1/2/3 **có thể chạy độc lập** để tối ưu thời gian
- Nhánh nào là **conditional** (I-A / A2) để không "kẹt scope"

Nguồn sự thật cho task list: `docs/MAPR2026_v3_team_parallel_coding_plan.md` (Section 8).

---

## Legend (đọc 15s)

- 🔴 **MAPR-MUST**: thiếu = paper không defensible
- 🟡 **BOOST**: làm sau khi xong toàn bộ 🔴
- 🔵 **FUTURE**: không làm cho MAPR
- **Artifact**: file output "contract" (người khác dùng)
- **Script**: command chạy để tạo artifact

**ASCII Symbol Guide:**

| Symbol   | Ý nghĩa                                               |
| -------- | ----------------------------------------------------- |
| `──▶`    | hard dependency (output phải xong trước mới chạy được) |
| `···▶`   | conditional dependency (chỉ chạy nếu điều kiện đúng) |
| `╔══╗`   | artifact file (output contract giữa các người)        |
| `[Px]`   | task owned by Person x (P1 / P2 / P3)                |
| `════`   | phase separator                                       |
| `┄┄`     | sub-section separator bên trong phase                 |

---

## Shared Artifacts (contract giữa 3 người)

| Artifact                                                         | Owner | Consumers | Notes                                                       |
| ---------------------------------------------------------------- | ----: | --------: | ----------------------------------------------------------- |
| `data/processed/graph_csr.npz`                                   |    P1 |  P1,P2,P3 | CSR + degrees + node_id mapping (nền tảng cho IC + proxies) |
| `data/processed/ic_scores_primary.parquet`                       |    P1 |     P2,P3 | IC labels primary (A0) trên labeled subset                  |
| `data/processed/split_masks.parquet`                             |    P1 |     P2,P3 | **Single source of truth** cho train/test split             |
| `data/processed/regression_targets.parquet`                      |    P1 |        P3 | IC score mean (float) — Y label cho GNN training; cùng lúc với `ic_scores_primary` |
| `data/processed/diffusion_proxies.parquet`                       |    P2 |     P1,P3 | one-hop/two-hop proxies trên FULL active graph              |
| `outputs/mapr2026_v3_results/metric_correlation_matrix.json`     |    P2 |       All | 8×8 Spearman + BH-FDR, chạy trên labeled node set          |
| `outputs/mapr2026_v3_results/degree_controlled_ic_variance.json` |    P1 |       All | C1 variance evidence (degree-banded + one-hop regression)   |
| `outputs/day1_benchmark/stability_explanation.json`              |    P1 |       All | **Conditional**: chỉ required nếu Jaccard stability < 0.85  |
| `outputs/mapr2026_v3_results/baseline_ranking_metrics.csv`       |    P3 |       All | Baselines Groups 1–4 metrics (test split only)              |
| `outputs/mapr2026_v3_results/surrogate_ranking_metrics.csv`      |    P3 |       All | GNN variants Group 5 metrics (mean±std across seeds)        |
| `outputs/mapr2026_v3_results/gnn_vs_degree_bootstrap_ci.json`    |    P3 |       All | C4 bootstrap CI (Spearman + NDCG@10%)                       |
| `outputs/mapr2026_v3_results/ic_bootstrap_ci.json`               |    P1 |       All | 🟡 BOOST: CI cho mean IC reach per node (MC noise); dùng Section 3.4 paper |
| `outputs/mapr2026_v3_results/runtime_breakdown.csv`              | P1+P2+P3 |    All | Runtime table: IC timing + proxy timing + GNN timing        |

---

## Dependency Flow (core A0 + optional branches)

Không cần GitHub/Obsidian để render — đọc trực tiếp trong bất kỳ text editor nào:

```text
════════════════════════════════════════════════════════════════════════════════════════
 PHASE 0 — Setup & QA                                         🔴 MAPR-MUST (all tasks)
════════════════════════════════════════════════════════════════════════════════════════

  data/raw/*.csv
  ├──▶ [P1] dead_account_audit.py  ──▶  outputs/stage0_data_quality/dead_accounts.json
  ├──▶ [P1] lcc_audit.py           ──▶  outputs/stage0_data_quality/lcc_audit.json
  └──▶ [P1] export_csr.py --run
                │
                ▼
       ╔═══════════════════════════════════╗
       ║  data/processed/graph_csr.npz    ║  ← P1 owns │ consumed by: P1, P2, P3
       ╚═══════════════════════════════════╝    CSR + degrees + node_id mapping
                │
                │  ◄── SAU KHI CÓ graph_csr.npz: P1 và P2 bắt đầu SONG SONG
                │
        ┌───────┴────────────────────────────────┐
        │                                         │
        ▼                                         ▼
  [P1 track → Phase 1A + 1B]          [P2 track → Phase 1C]


════════════════════════════════════════════════════════════════════════════════════════
 PHASE 1A — IC Simulation [P1]                                🔴 MAPR-MUST
════════════════════════════════════════════════════════════════════════════════════════

  graph_csr.npz
  ├──▶ [P1] day1_benchmark.py  ──▶  outputs/day1_benchmark/*
  │         (timing sanity check trước khi chạy IC thật)
  │
  └──▶ [P1] ic_pilot_diagnostics.py  ──▶  ic_pilot_stats.json
            (IC convergence check + runtime estimate, ~2 phút)

  [P1] ic_labels_primary.py   A0: p(u,v) = 1/deg(v), weighted cascade
       --n-runs 200 --n-sample 5000
       ├──▶ ╔════════════════════════════════════════════╗
       │    ║  data/processed/ic_scores_primary.parquet  ║  ← P1 owns │ P2, P3 consume
       │    ╚════════════════════════════════════════════╝    IC labels A0 (full N×R distribution)
       │
       ├──▶ ╔════════════════════════════════════════════╗
       │    ║  data/processed/regression_targets.parquet ║  ← P1 owns │ P3 consume
       │    ╚════════════════════════════════════════════╝    IC score mean (float Y label cho GNN training)
       │
       └──▶ ╔════════════════════════════════════════════╗
            ║  data/processed/split_masks.parquet        ║  ← P1 owns │ P2, P3 consume
            ╚════════════════════════════════════════════╝    [M0-locked: single source of truth]

  ic_scores_primary.parquet + split_masks.parquet
  ├──▶ [P1] ic_feasibility_protocol.py  (Jaccard stability check)
  │         └···▶ [ONLY IF Jaccard < 0.85]   🔴 triggered
  │               ╔══════════════════════════════════════════════════════════╗
  │               ║  outputs/day1_benchmark/stability_explanation.json      ║  CONDITIONAL
  │               ╚══════════════════════════════════════════════════════════╝
  │
  └──▶ [P1] bootstrap_ci_ic()   🟡 BOOST  (~15 phút, Strongly Recommended)
            (CI cho mean IC reach per node — đo MC simulation noise của label)
            └──▶ ╔══════════════════════════════════════════════════════════╗
                 ║  outputs/mapr2026_v3_results/ic_bootstrap_ci.json       ║
                 ╚══════════════════════════════════════════════════════════╝
                 [n_bootstrap=1000; fields: mean, ci_lower, ci_upper per node]
                 [dùng trong Section 3.4 paper để show label reliability]
                 [KHÁC với C4 bootstrap CI — C4 là GNN vs degree, đây là label quality]

  graph_csr.npz  (chạy SONG SONG với ic_labels_primary, không phụ thuộc nhau)
  └──▶ [P1] ic_pilot_ia.py   🔴 PILOT UNCONDITIONAL (~20 min, không cần đợi A0)
            └──▶ ╔══════════════════════════════════════════════════════════╗
                 ║  outputs/mapr2026_v3_results/ia_pilot_diagnostics.json  ║
                 ╚══════════════════════════════════════════════════════════╝
                 [→ PASS/FAIL gates Phase 3 I-A branch]


════════════════════════════════════════════════════════════════════════════════════════
 PHASE 1C — Diffusion Proxies [P2]   (song song với 1A, chỉ cần graph_csr)   🟡 BOOST
════════════════════════════════════════════════════════════════════════════════════════

  graph_csr.npz  (không cần IC labels → P2 bắt đầu ngay sau Phase 0)
  └──▶ [P2] diffusion_proxies.py
            ├──▶ ╔══════════════════════════════════════════════╗
            │    ║  data/processed/diffusion_proxies.parquet    ║  ← P2 owns │ P1[C1], P3 consume
            │    ╚══════════════════════════════════════════════╝    one-hop/two-hop, full active graph
            │
            └──▶ [proxy timing log → assembled vào runtime_breakdown.csv ở Phase 2]


════════════════════════════════════════════════════════════════════════════════════════
 PHASE 2 — Analysis Sprint            🔴 blocked by: ic_scores_primary + split_masks
════════════════════════════════════════════════════════════════════════════════════════

  ┄┄ C1 Degree-Controlled Variance [P1] 🔴 MAPR-MUST ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄

  ic_scores_primary.parquet  ──▶┐
  diffusion_proxies.parquet  ───┤  [P1] degree_controlled_variance.py
  (one_hop_spread required)     │  [two-tier: degree-band CV + one-hop regression residual]
                                └──▶ ╔══════════════════════════════════════════════════════╗
                                     ║  outputs/mapr2026_v3_results/                       ║
                                     ║    degree_controlled_ic_variance.json               ║
                                     ╚══════════════════════════════════════════════════════╝
  NOTE: P2 phải xong diffusion_proxies.parquet trước khi P1 chạy C1


  ┄┄ Metric Correlation Matrix [P2] 🟡 BOOST ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄

  ic_scores_primary.parquet  ──▶┐
  diffusion_proxies.parquet  ───┤  [P2] metric_correlation_matrix.py
  split_masks.parquet        ───┘  (8×8 Spearman + BH-FDR, labeled node set)
                                   └──▶ ╔══════════════════════════════════════════════════╗
                                        ║  outputs/mapr2026_v3_results/                   ║
                                        ║    metric_correlation_matrix.json               ║
                                        ╚══════════════════════════════════════════════════╝


  ┄┄ Baselines [P3] 🔴 MAPR-MUST (Groups 1–3) / 🟡 BOOST (Group 4) ┄┄┄┄┄┄┄┄┄┄┄┄┄┄

  eval_ranking_harness.py  ← code dependency only (determinism/ordering contract)
                             import bởi run_baselines.py và run_surrogates.py

  ic_scores_primary.parquet  ──▶┐
  split_masks.parquet        ───┤  [P3] run_baselines.py
  diffusion_proxies.parquet  ───┘
    ├─ Group 1: Degree / PageRank / Core-number       (analytic, no IC training target) 🔴
    ├─ Group 2: Community-based (SBM-Bernoulli, Louvain) (no IC training target)        🔴
    ├─ Group 3: Diffusion proxies (1-hop, 2-hop spread)  (needs diffusion_proxies)      🔴
    └─ Group 4: Node2Vec + LR        (IC labels dùng làm training target)               🟡 BOOST
                                     └──▶ ╔═══════════════════════════════════════════════╗
                                          ║  outputs/mapr2026_v3_results/                ║
                                          ║    baseline_ranking_metrics.csv              ║
                                          ╚═══════════════════════════════════════════════╝


  ┄┄ C2 GNN Surrogates [P3] 🔴 MAPR-MUST ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄

  ic_scores_primary.parquet  ──▶┐
  split_masks.parquet        ───┘  [P3] run_surrogates.py   (raw_attr features)
                                        5 archs: SAGE / GCN / GIN / GAT / APPNP
                                        × 5 seeds each = 25 runs total
                                        (requires PyG ≥ 2.3 for APPNP)
                                   ├──▶ ╔═══════════════════════════════════════════════╗
                                   │    ║  outputs/mapr2026_v3_results/                ║
                                   │    ║    surrogate_ranking_metrics.csv             ║
                                   │    ╚═══════════════════════════════════════════════╝
                                   │         [mean±std Spearman + NDCG@10%, per arch]
                                   │
                                   └──▶ [P3] C4 Bootstrap CI   🔴 MAPR-MUST
                                             1000 resamplings, SESOI ±0.02
                                             Spearman + NDCG@10% (best GNN vs degree baseline)
                                             └──▶ ╔══════════════════════════════════════════╗
                                                  ║  outputs/mapr2026_v3_results/           ║
                                                  ║    gnn_vs_degree_bootstrap_ci.json      ║
                                                  ╚══════════════════════════════════════════╝


  ┄┄ Runtime Breakdown Assembly [P1 + P2 + P3] ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄

  [P1] IC simulation timing     (ic_labels_primary.py log)   ──────────▶┐
  [P2] proxy timing             (diffusion_proxies.py log)   ──────────▶┤  assemble
  [P3] GNN inference timing     (run_surrogates.py log)      ──────────▶┤
  [P3] Node2Vec timing          (run_baselines.py Group 4)   ──────────▶┘
                                                              └──▶ ╔════════════════════════════════════╗
                                                                   ║  outputs/mapr2026_v3_results/     ║
                                                                   ║    runtime_breakdown.csv          ║
                                                                   ╚════════════════════════════════════╝


════════════════════════════════════════════════════════════════════════════════════════
 PHASE 3 — Boost / Future Branches       (chỉ bắt đầu SAU KHI toàn bộ 🔴 xong)
════════════════════════════════════════════════════════════════════════════════════════

  ┄┄ A2 Sensitivity Branch [P1 → P3] 🟡 BOOST ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄

  graph_csr.npz
  └──▶ [P1] ic_labels_primary.py --variant a2
            A2: p(u,v) = 1/√(deg_u × deg_v)    (structurally analogous to GCN D^{-1/2}AD^{-1/2})
            └──▶ ╔════════════════════════════════════════════════════════════╗
                 ║  data/processed/ic_scores_sensitivity_a2.parquet          ║
                 ╚════════════════════════════════════════════════════════════╝
                           │
                           └──▶ [P3] C2-A2 hypothesis test   🟡 BOOST
                                     split_masks.parquet + ic_scores_sensitivity_a2.parquet
                                     GCN only × 5 seeds  (targeted test, không phải full 5-arch sweep)
                                     └──▶ surrogate_ranking_metrics_a2.csv


  ┄┄ I-A Conditional Branch [P1 → P3] 🟡 BOOST  (pilot gate là 🔴 MUST) ┄┄┄┄┄┄┄┄┄

  ia_pilot_diagnostics.json  (produced in Phase 1A, unconditional ~20 min)
  │
  │  ⚠ PRE-REGISTRATION bắt buộc TRƯỚC KHI chạy pilot (5 phút, 🔴 MUST):
  │     Ghi vào docs/experiment_registry.md:
  │     "H-IA: Under I-A labels, GATv2 sẽ outperform degree (degree blind to row-norm IC);
  │      FAIL → A0-only narrative; không thay đổi hypothesis sau khi có kết quả."
  │
  ├─ CHECK 3 conditions:
  │   (1) CV > 0.3          ← I-A scores có đủ variance không?
  │   (2) |ρ_deg| < 0.75    ← I-A không quá correlated với degree?
  │   (3) |ρ_proxy| < 0.85  ← I-A không quá correlated với proxies?
  │
  ├···▶ [PASS: tất cả 3 điều kiện met]
  │     [P1] ic_labels_attribute_ia.py   🟡 BOOST
  │          ├──▶ ╔═══════════════════════════════════════════════════════╗
  │          │    ║  outputs/mapr2026_v3_results/ic_scores_ia.parquet    ║
  │          │    ╚═══════════════════════════════════════════════════════╝
  │          └──▶ ╔═══════════════════════════════════════════════════════╗
  │               ║  outputs/mapr2026_v3_results/                        ║
  │               ║    regression_targets_ia.parquet                     ║
  │               ╚═══════════════════════════════════════════════════════╝
  │                         │
  │               ic_scores_ia.parquet + split_masks.parquet
  │                         │
  │                         ├──▶ [P3] C2-I-A   🟡 BOOST
  │                         │         GATv2 + SAGE/GCN/GIN/APPNP × 5 seeds
  │                         │         └──▶ surrogate_ranking_metrics_ia.csv
  │                         │                         │
  │                         │                         └──▶ [P3] C4-I-A Bootstrap CI   🟡 BOOST
  │                         │                                   1000 resamplings, SESOI ±0.02
  │                         │                                   └──▶ bootstrap_ci_ia.json
  │                         │
  │                         └──▶ [P3] C3 Ranking Loss   🟡 BOOST
  │                                   combined Huber + top-k pairwise margin
  │                                   └──▶ surrogate_ranking_metrics_c3.csv
  │
  └···▶ [FAIL: bất kỳ condition nào không met]
        → 🔵 FUTURE:TKDE/WWW2027 — skip toàn bộ I-A track cho MAPR
```

---

## Parallelization Notes (để "song song thật")

1. **P1 và P2 bắt đầu SONG SONG ngay sau Phase 0**

   - P1: `dead_account_audit` → `lcc_audit` → `export_csr` → (song song) `ic_labels_primary` + `ic_pilot_ia`
   - P2: `diffusion_proxies.py` chỉ cần `graph_csr.npz` → không phụ thuộc IC labels → bắt đầu ngay

2. **P3 bị BLOCK cho đến khi P1 xong `ic_scores_primary` + `split_masks` + `regression_targets`**

   - Nhưng P3 chuẩn bị trước: kiểm tra harness import, dry-run config, `pytest -q` determinism tests
   - P3 cũng nên verify PyG ≥ 2.3 (bắt buộc cho APPNP): `python -c "import torch_geometric as pyg; from torch_geometric.nn import APPNP; print(pyg.__version__)"`
   - **P3 GNN execution order:** SAGE baseline (**17/4**) → C2 full 5-arch sweep (**19/4**) → C3+C4 song song (**21/4**) → ablation graph-only/centrality (**22/4**, dùng best_arch từ C2)

3. **C1 (P1) bị BLOCK bởi cả hai: `ic_scores_primary` (P1) + `diffusion_proxies` (P2)**

   - P1 không thể chạy `degree_controlled_variance.py` cho đến khi P2 xong proxies
   - Lý do: C1 cần `one_hop_spread` từ `diffusion_proxies.parquet` (two-tier test: degree-band CV + one-hop regression residual)

4. **Metric correlation matrix (P2) bị BLOCK bởi cả hai**

   - `ic_scores_primary.parquet` (P1) **VÀ** `diffusion_proxies.parquet` (P2) phải xong trước
   - Chạy cuối Phase 2, không block GNN training

5. **I-A pilot (`ic_pilot_ia.py`) là 🔴 UNCONDITIONAL**

   - P1 chạy ngay song song với `ic_labels_primary` (không cần đợi A0 xong)
   - **⚠ Pre-register TRƯỚC KHI chạy:** ghi hypothesis vào `docs/experiment_registry.md`
   - Kết quả `ia_pilot_diagnostics.json` quyết định có tiếp tục I-A full labels không
   - Full labels (`ic_labels_attribute_ia.py`) chỉ chạy nếu **cả 3** conditions đều PASS

5b. **IC Bootstrap CI (`bootstrap_ci_ic()`) là 🟡 BOOST — chạy song song sau khi có IC labels**

   - Đo MC simulation noise của label: CI cho mean IC reach per node
   - **Khác với C4** (GNN vs degree CI) — đây là về label quality, không phải GNN
   - Output: `ic_bootstrap_ci.json` — dùng trong Section 3.4 paper để justify label reliability

6. **A2 sensitivity là 🟡 BOOST — không block core pipeline**

   - Chỉ chạy sau khi toàn bộ 🔴 xong và còn thời gian trước 30/4
   - C2-A2 chỉ dùng GCN (không full 5-arch sweep) để tiết kiệm compute

7. **Runtime assembly cần đợi tất cả các track**

   - P1 IC timing + P2 proxy timing + P3 GNN timing + P3 Node2Vec timing
   - P2 hoặc P3 assemble cuối cùng vào `runtime_breakdown.csv`
