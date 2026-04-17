from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from _shared import PATHS, ensure_dir, load_csr_npz, require_columns


@dataclass(frozen=True)
class FormulaConfig:
    name: str
    alpha: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0
    mix_lambda: float = 0.0
    mix_w_sender: float = 0.0
    mix_w_receiver: float = 0.0
    mix_w_same_comm: float = 0.0


@dataclass(frozen=True)
class RaceStage:
    n_sample: int
    n_runs: int
    mc_seeds: list[int]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IC formula shootout benchmark")
    p.add_argument("--csr", default=PATHS.csr_npz)
    p.add_argument("--node-attrs", default=PATHS.node_attributes)
    p.add_argument("--centrality", default="data/processed/centrality_table.parquet")
    p.add_argument("--community", default="data/processed/community_labels.parquet")
    p.add_argument("--n-sample", type=int, default=1000)
    p.add_argument("--n-runs", type=int, default=20)
    p.add_argument("--mc-seeds", default="0,1,2")
    p.add_argument(
        "--formulas",
        default="all",
        help=(
            "Comma-separated formula names to run, or 'all'. "
            "Supported: weighted_cascade,hybrid_centered,sender_boost,sender_receiver,convex_mixture,community_boost,"
            "symmetric_deg,source_budget,receiver_budget_views"
        ),
    )
    p.add_argument("--hybrid-gamma", type=float, default=0.1)
    p.add_argument(
        "--budget-gamma",
        type=float,
        default=1.0,
        help=(
            "Gamma for receiver_budget_views (views-augmented weighted cascade preserving receiver budget). "
            "gamma=0 reduces exactly to weighted_cascade."
        ),
    )
    p.add_argument("--seed-multiplier", type=int, default=10000)
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--top-pct", type=float, default=0.10)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--out-dir", default="outputs/day1_benchmark/formula_shootout")
    p.add_argument("--auto-race", action="store_true")
    p.add_argument(
        "--race-stages",
        default="200:20:0,1;500:60:0,1,2,3;1000:150:0,1,2,3,4,5",
        help="Semicolon-separated stages n_sample:n_runs:seed_list",
    )
    p.add_argument("--race-keep-frac", type=float, default=0.4)
    p.add_argument("--race-min-keep", type=int, default=2)
    p.add_argument("--race-lambda", type=float, default=0.05)
    p.add_argument("--race-delta", type=float, default=0.01)
    p.add_argument("--race-ci-bootstrap", type=int, default=400)
    return p.parse_args()


def _parse_int_list(raw: str) -> list[int]:
    out = []
    for t in raw.split(","):
        t = t.strip()
        if t:
            out.append(int(t))
    if not out:
        raise ValueError("Empty integer list")
    return out


def _parse_race_stages(raw: str) -> list[RaceStage]:
    stages: list[RaceStage] = []
    for token in str(raw).split(";"):
        token = token.strip()
        if not token:
            continue
        parts = [p.strip() for p in token.split(":")]
        if len(parts) != 3:
            raise ValueError(f"Invalid stage token: {token}")
        n_sample = int(parts[0])
        n_runs = int(parts[1])
        seeds = _parse_int_list(parts[2])
        stages.append(RaceStage(n_sample=n_sample, n_runs=n_runs, mc_seeds=seeds))
    if not stages:
        raise ValueError("--race-stages is empty")
    return stages


def _ensure_node_id_str(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["node_id"] = out["node_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    return out


def _robust_scale(x: np.ndarray, q_low: float = 0.05, q_high: float = 0.95) -> np.ndarray:
    lo, hi = np.quantile(x, [q_low, q_high])
    den = float(hi - lo)
    if (not np.isfinite(den)) or den <= 0:
        den = 1.0
    return np.clip((x - float(lo)) / den, 0.0, 1.0)


def _neighbor_weight_sum(indptr: np.ndarray, indices: np.ndarray, sender_w: np.ndarray) -> np.ndarray:
    """Compute per-node sum of sender weights over its neighbors.

    For undirected graphs with symmetric CSR, this equals:
        sum_w[v] = \sum_{u in N(v)} w[u]

    Used by receiver_budget_views so that:
        p(u,v) = w[u] / sum_w[v]
    which preserves the weighted-cascade receiver budget and guarantees 0<=p<=1.
    """
    try:
        from scipy.sparse import csr_matrix
    except Exception as ex:  # pragma: no cover
        raise ImportError("scipy is required for receiver_budget_views") from ex

    n = int(len(sender_w))
    data = np.ones(int(len(indices)), dtype=np.float32)
    a = csr_matrix((data, indices.astype(np.int64, copy=False), indptr.astype(np.int64, copy=False)), shape=(n, n))
    out = a.T.dot(sender_w.astype(np.float64, copy=False))
    return np.asarray(out).reshape(-1).astype(float, copy=False)


def _sample_rows(degrees: np.ndarray, n_sample: int, seed: int) -> np.ndarray:
    n = int(len(degrees))
    if n_sample <= 0 or n_sample >= n:
        return np.arange(n, dtype=np.int64)

    idx = np.arange(n, dtype=np.int64)
    quint = pd.qcut(pd.Series(degrees.astype(float)), q=5, labels=False, duplicates="drop").to_numpy()
    train_idx, sample_idx = train_test_split(idx, test_size=int(n_sample), random_state=seed, stratify=quint)
    _ = train_idx
    return np.sort(sample_idx.astype(np.int64))


def _top_set(scores: np.ndarray, top_pct: float) -> set[int]:
    thr = float(np.quantile(scores, 1.0 - float(top_pct)))
    return set(np.where(scores >= thr)[0].tolist())


def _jaccard(a: set[int], b: set[int]) -> float:
    u = a | b
    if not u:
        return 1.0
    return float(len(a & b) / len(u))


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(pd.Series(x).corr(pd.Series(y), method="spearman"))


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _stable_rank(scores: np.ndarray, ids: np.ndarray | None = None) -> np.ndarray:
    if ids is None:
        ids = np.arange(len(scores), dtype=np.int64)
    # Primary = score desc, tie-break = id asc
    return np.lexsort((ids.astype(np.int64), -scores.astype(float)))


def _precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, top_pct: float) -> float:
    n = int(len(y_true))
    if n <= 0:
        return 0.0
    k = max(1, int(np.ceil(float(top_pct) * n)))
    ids = np.arange(n, dtype=np.int64)
    top_true = set(_stable_rank(y_true, ids)[:k].tolist())
    top_pred = set(_stable_rank(y_pred, ids)[:k].tolist())
    return float(len(top_true & top_pred) / float(k))


def _ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, top_pct: float) -> float:
    n = int(len(y_true))
    if n <= 0:
        return 0.0
    k = max(1, int(np.ceil(float(top_pct) * n)))
    ids = np.arange(n, dtype=np.int64)
    rank_pred = _stable_rank(y_pred, ids)[:k]
    rank_true = _stable_rank(y_true, ids)[:k]

    rel = y_true.astype(float)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(rel[rank_pred] * discounts))
    idcg = float(np.sum(rel[rank_true] * discounts))
    return float(dcg / idcg) if idcg > 0.0 else 0.0


def _edge_probability(
    cfg: FormulaConfig,
    inv_deg_u: float,
    inv_deg_v: float,
    sender_u: float,
    receiver_v: float,
    same_comm: float,
) -> float:
    if cfg.name == "weighted_cascade":
        p = inv_deg_v
    elif cfg.name == "symmetric_deg":
        # p(u,v) = 1/sqrt(deg(u)*deg(v)) = sqrt(inv_deg(u)*inv_deg(v))
        p = math.sqrt(max(0.0, inv_deg_u) * max(0.0, inv_deg_v))
    elif cfg.name == "source_budget":
        # p(u,v) = 1/deg(u)
        p = inv_deg_u
    elif cfg.name == "hybrid_centered":
        p = inv_deg_v * (1.0 + cfg.gamma * sender_u)
    elif cfg.name == "sender_boost":
        p = inv_deg_v * (1.0 + cfg.alpha * sender_u)
    elif cfg.name == "sender_receiver":
        p = inv_deg_v * (1.0 + cfg.alpha * sender_u - cfg.beta * receiver_v)
    elif cfg.name == "convex_mixture":
        z = cfg.mix_w_sender * sender_u - cfg.mix_w_receiver * receiver_v + cfg.mix_w_same_comm * same_comm
        learned = _sigmoid(float(z))
        p = cfg.mix_lambda * inv_deg_v + (1.0 - cfg.mix_lambda) * learned
    elif cfg.name == "community_boost":
        p = inv_deg_v * (1.0 + cfg.alpha * sender_u) * (1.0 + cfg.gamma * same_comm)
    elif cfg.name == "receiver_budget_views":
        # receiver_v is interpreted as sum_w[v] = sum_{u in N(v)} w[u]
        den = float(receiver_v)
        if den <= 0.0:
            p = 0.0
        else:
            p = float(sender_u) / den
    else:
        raise ValueError(f"Unknown formula: {cfg.name}")

    return float(np.clip(p, 0.0, 1.0))


def _simulate_once(
    source: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    inv_degrees: np.ndarray,
    sender_strength: np.ndarray,
    receiver_resistance: np.ndarray,
    community_ids: np.ndarray,
    cfg: FormulaConfig,
    rng: np.random.Generator,
) -> int:
    activated = {int(source)}
    frontier = [int(source)]

    while frontier:
        nxt: list[int] = []
        for u in frontier:
            st = int(indptr[u])
            en = int(indptr[u + 1])
            inv_deg_u = float(inv_degrees[u])
            for v_raw in indices[st:en]:
                v = int(v_raw)
                if v in activated:
                    continue
                same = 1.0 if (community_ids[u] >= 0 and community_ids[u] == community_ids[v]) else 0.0
                p = _edge_probability(
                    cfg=cfg,
                    inv_deg_u=inv_deg_u,
                    inv_deg_v=float(inv_degrees[v]),
                    sender_u=float(sender_strength[u]),
                    receiver_v=float(receiver_resistance[v]),
                    same_comm=float(same),
                )
                if rng.random() <= p:
                    activated.add(v)
                    nxt.append(v)
        frontier = nxt

    return int(len(activated))


def _simulate_node_mean(
    source: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    inv_degrees: np.ndarray,
    sender_strength: np.ndarray,
    receiver_resistance: np.ndarray,
    community_ids: np.ndarray,
    cfg: FormulaConfig,
    n_runs: int,
    worker_seed: int,
) -> float:
    rng = np.random.default_rng(worker_seed)
    vals = np.empty(int(n_runs), dtype=np.int32)
    for i in range(int(n_runs)):
        vals[i] = _simulate_once(
            source=source,
            indptr=indptr,
            indices=indices,
            inv_degrees=inv_degrees,
            sender_strength=sender_strength,
            receiver_resistance=receiver_resistance,
            community_ids=community_ids,
            cfg=cfg,
            rng=rng,
        )
    return float(np.mean(vals))


def _compute_scores_by_seed(
    rows: np.ndarray,
    csr: dict[str, np.ndarray],
    sender_strength: np.ndarray,
    receiver_resistance: np.ndarray,
    community_ids: np.ndarray,
    cfg: FormulaConfig,
    n_runs: int,
    mc_seeds: list[int],
    seed_multiplier: int,
    n_jobs: int,
) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for mc_seed in mc_seeds:
        seed_off = int(mc_seed) * int(seed_multiplier)

        def _worker(row: int) -> float:
            return _simulate_node_mean(
                source=int(row),
                indptr=csr["indptr"],
                indices=csr["indices"],
                inv_degrees=csr["inv_degrees"],
                sender_strength=sender_strength,
                receiver_resistance=receiver_resistance,
                community_ids=community_ids,
                cfg=cfg,
                n_runs=int(n_runs),
                worker_seed=int(seed_off + int(row)),
            )

        means = Parallel(n_jobs=int(n_jobs), backend="loky")(delayed(_worker)(int(r)) for r in rows)
        out[int(mc_seed)] = np.asarray(means, dtype=float)
    return out


def _stability_from_scores(scores_by_seed: dict[int, np.ndarray], top_pct: float) -> dict[str, float]:
    seeds = sorted(scores_by_seed.keys())
    jac_vals: list[float] = []
    rho_vals: list[float] = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            a = scores_by_seed[seeds[i]]
            b = scores_by_seed[seeds[j]]
            jac_vals.append(_jaccard(_top_set(a, top_pct), _top_set(b, top_pct)))
            rho_vals.append(_spearman(a, b))
    return {
        "jaccard_mean": float(np.mean(jac_vals)),
        "jaccard_min": float(np.min(jac_vals)),
        "spearman_mean": float(np.mean(rho_vals)),
        "spearman_min": float(np.min(rho_vals)),
    }


def _learnability(
    y: np.ndarray,
    graph_feats: np.ndarray,
    full_feats: np.ndarray,
    degrees_sample: np.ndarray,
    top_pct: float,
    seed: int = 42,
) -> dict[str, float]:
    idx = np.arange(len(y), dtype=np.int64)
    quint = pd.qcut(pd.Series(degrees_sample.astype(float)), q=5, labels=False, duplicates="drop").to_numpy()
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=int(seed), stratify=quint)

    def _fit_eval(x: np.ndarray) -> tuple[float, float, float, float]:
        m = RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=int(seed), n_jobs=-1)
        m.fit(x[train_idx], y[train_idx])
        pred = m.predict(x[test_idx])
        rho = _spearman(y[test_idx], pred)
        ndcg = _ndcg_at_k(y[test_idx], pred, top_pct)
        prec = _precision_at_k(y[test_idx], pred, top_pct)
        jac = _jaccard(_top_set(y[test_idx], top_pct), _top_set(pred, top_pct))
        return float(rho), float(ndcg), float(prec), float(jac)

    g_rho, g_ndcg, g_prec, g_jac = _fit_eval(graph_feats)
    f_rho, f_ndcg, f_prec, f_jac = _fit_eval(full_feats)
    return {
        "graph_only_spearman": g_rho,
        "graph_only_ndcg": g_ndcg,
        "graph_only_precision": g_prec,
        "graph_only_jaccard": g_jac,
        "graph_plus_views_spearman": f_rho,
        "graph_plus_views_ndcg": f_ndcg,
        "graph_plus_views_precision": f_prec,
        "graph_plus_views_jaccard": f_jac,
        "delta_spearman": float(f_rho - g_rho),
        "delta_ndcg": float(f_ndcg - g_ndcg),
        "delta_precision": float(f_prec - g_prec),
        "delta_jaccard": float(f_jac - g_jac),
    }


def _minmax(s: pd.Series) -> pd.Series:
    lo = float(s.min())
    hi = float(s.max())
    if hi <= lo:
        return pd.Series(np.ones(len(s), dtype=float), index=s.index)
    return (s - lo) / (hi - lo)


def _bootstrap_ci(samples: np.ndarray, n_bootstrap: int, alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    if len(samples) == 0:
        return float("nan"), float("nan")
    if len(samples) == 1:
        x = float(samples[0])
        return x, x
    rng = np.random.default_rng(int(seed))
    n = int(len(samples))
    means = np.empty(int(n_bootstrap), dtype=float)
    for i in range(int(n_bootstrap)):
        take = rng.integers(0, n, size=n)
        means[i] = float(np.mean(samples[take]))
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi


def _compute_race_samples(
    scores_by_seed: dict[int, np.ndarray],
    graph_feats: np.ndarray,
    full_feats: np.ndarray,
    sampled_degrees: np.ndarray,
    top_pct: float,
    sample_seed: int,
) -> np.ndarray:
    seeds = sorted(scores_by_seed.keys())
    if not seeds:
        return np.empty(0, dtype=float)
    anchor = seeds[0]
    out: list[float] = []
    top_anchor = _top_set(scores_by_seed[anchor], top_pct)
    for s in seeds:
        arr = scores_by_seed[s]
        if s == anchor:
            rho_anchor = 1.0
            jac_anchor = 1.0
        else:
            rho_anchor = _spearman(arr, scores_by_seed[anchor])
            jac_anchor = _jaccard(_top_set(arr, top_pct), top_anchor)

        y = np.log1p(arr)
        learn = _learnability(
            y=y,
            graph_feats=graph_feats,
            full_feats=full_feats,
            degrees_sample=sampled_degrees,
            top_pct=top_pct,
            seed=int(sample_seed + 997 * int(s)),
        )
        out.append(
            0.5 * float(rho_anchor)
            + 0.3 * float(learn["graph_plus_views_ndcg"])
            + 0.2 * float(learn["graph_plus_views_precision"])
        )
    return np.asarray(out, dtype=float)


def _evaluate_formula(
    cfg: FormulaConfig,
    rows: np.ndarray,
    csr: dict[str, np.ndarray],
    sender_strength: np.ndarray,
    sender_strength_views_centered: np.ndarray,
    receiver_resistance: np.ndarray,
    sender_budget_w: np.ndarray | None,
    receiver_budget_sum: np.ndarray | None,
    comm_ids: np.ndarray,
    n_runs: int,
    mc_seeds: list[int],
    seed_multiplier: int,
    n_jobs: int,
    top_pct: float,
    graph_feats: np.ndarray,
    full_feats: np.ndarray,
    sampled_degrees: np.ndarray,
    sample_seed: int,
) -> tuple[dict[str, float | str], dict[int, np.ndarray]]:
    t0 = time.perf_counter()
    if cfg.name == "hybrid_centered":
        sender_used = sender_strength_views_centered
        receiver_used = receiver_resistance
    elif cfg.name == "receiver_budget_views":
        if sender_budget_w is None or receiver_budget_sum is None:
            raise ValueError("receiver_budget_views requires precomputed sender_budget_w and receiver_budget_sum")
        sender_used = sender_budget_w
        receiver_used = receiver_budget_sum
    else:
        sender_used = sender_strength
        receiver_used = receiver_resistance

    scores_by_seed = _compute_scores_by_seed(
        rows=rows,
        csr=csr,
        sender_strength=sender_used,
        receiver_resistance=receiver_used,
        community_ids=comm_ids,
        cfg=cfg,
        n_runs=int(n_runs),
        mc_seeds=mc_seeds,
        seed_multiplier=int(seed_multiplier),
        n_jobs=int(n_jobs),
    )
    elapsed = float(time.perf_counter() - t0)

    stab = _stability_from_scores(scores_by_seed=scores_by_seed, top_pct=float(top_pct))
    y = np.mean(np.stack([scores_by_seed[s] for s in sorted(scores_by_seed)], axis=0), axis=0)
    y = np.log1p(y)

    learn = _learnability(
        y=y,
        graph_feats=graph_feats,
        full_feats=full_feats,
        degrees_sample=sampled_degrees,
        top_pct=float(top_pct),
        seed=int(sample_seed),
    )

    row = {
        "formula": cfg.name,
        "runtime_sec": elapsed,
        "jaccard_mean": stab["jaccard_mean"],
        "jaccard_min": stab["jaccard_min"],
        "spearman_mean": stab["spearman_mean"],
        "spearman_min": stab["spearman_min"],
        "graph_only_spearman": learn["graph_only_spearman"],
        "graph_only_ndcg": learn["graph_only_ndcg"],
        "graph_only_precision": learn["graph_only_precision"],
        "graph_plus_views_spearman": learn["graph_plus_views_spearman"],
        "graph_plus_views_ndcg": learn["graph_plus_views_ndcg"],
        "graph_plus_views_precision": learn["graph_plus_views_precision"],
        "delta_spearman": learn["delta_spearman"],
        "delta_ndcg": learn["delta_ndcg"],
        "delta_precision": learn["delta_precision"],
        "graph_only_jaccard": learn["graph_only_jaccard"],
        "graph_plus_views_jaccard": learn["graph_plus_views_jaccard"],
        "delta_jaccard": learn["delta_jaccard"],
        "n_sample": int(len(rows)),
        "n_runs": int(n_runs),
        "mc_seeds": ",".join(str(s) for s in mc_seeds),
        "alpha": cfg.alpha,
        "beta": cfg.beta,
        "gamma": cfg.gamma,
        "mix_lambda": cfg.mix_lambda,
        "mix_w_sender": cfg.mix_w_sender,
        "mix_w_receiver": cfg.mix_w_receiver,
        "mix_w_same_comm": cfg.mix_w_same_comm,
    }
    return row, scores_by_seed


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)

    csr_raw = load_csr_npz(args.csr)
    node_ids = csr_raw["node_ids"].astype(str)
    degrees = csr_raw["degrees"].astype(float)

    inv_degrees = np.zeros_like(degrees, dtype=float)
    mask = degrees > 0
    inv_degrees[mask] = 1.0 / degrees[mask]

    csr = {
        "indptr": csr_raw["indptr"],
        "indices": csr_raw["indices"],
        "inv_degrees": inv_degrees,
    }

    node_attrs = _ensure_node_id_str(pd.read_parquet(args.node_attrs))
    centrality = _ensure_node_id_str(pd.read_parquet(args.centrality))
    community = _ensure_node_id_str(pd.read_parquet(args.community))

    require_columns(node_attrs, ["node_id", "views", "life_time"], "node_attributes")
    require_columns(centrality, ["node_id", "degree", "pagerank", "kshell"], "centrality")
    require_columns(community, ["node_id", "community_id", "cross_community_edge_fraction"], "community")

    idx = pd.Index(node_ids)

    views = pd.to_numeric(node_attrs.set_index("node_id").reindex(idx)["views"], errors="coerce").fillna(0.0).to_numpy()
    life_time = pd.to_numeric(node_attrs.set_index("node_id").reindex(idx)["life_time"], errors="coerce").fillna(1.0).clip(lower=1.0).to_numpy()
    views_log = np.log1p(np.maximum(views, 0.0))
    views_per_day = views / life_time

    pagerank = pd.to_numeric(centrality.set_index("node_id").reindex(idx)["pagerank"], errors="coerce").fillna(0.0).to_numpy()
    kshell = pd.to_numeric(centrality.set_index("node_id").reindex(idx)["kshell"], errors="coerce").fillna(0.0).to_numpy()
    degree_cent = pd.to_numeric(centrality.set_index("node_id").reindex(idx)["degree"], errors="coerce").fillna(0.0).to_numpy()

    cross_comm = pd.to_numeric(
        community.set_index("node_id").reindex(idx)["cross_community_edge_fraction"], errors="coerce"
    ).fillna(0.0).to_numpy()
    comm_ids = pd.to_numeric(community.set_index("node_id").reindex(idx)["community_id"], errors="coerce").fillna(-1).astype(np.int64).to_numpy()

    s_views = _robust_scale(views_log)
    s_pr = _robust_scale(pagerank)
    s_ks = _robust_scale(kshell)
    sender_strength = (s_views + s_pr + s_ks) / 3.0
    sender_strength_views_centered = (s_views - float(np.mean(s_views))).astype(float, copy=False)

    r_deg = _robust_scale(degree_cent)
    r_cross = _robust_scale(cross_comm)
    receiver_resistance = (r_deg + r_cross) / 2.0

    mc_seeds = _parse_int_list(args.mc_seeds)

    configs_all = [
        FormulaConfig(name="weighted_cascade"),
        FormulaConfig(name="symmetric_deg"),
        FormulaConfig(name="source_budget"),
        FormulaConfig(name="hybrid_centered", gamma=float(args.hybrid_gamma)),
        FormulaConfig(name="receiver_budget_views", gamma=float(args.budget_gamma)),
        FormulaConfig(name="sender_boost", alpha=0.10),
        FormulaConfig(name="sender_receiver", alpha=0.15, beta=0.10),
        FormulaConfig(name="convex_mixture", mix_lambda=0.95, mix_w_sender=1.0, mix_w_receiver=0.8, mix_w_same_comm=0.3),
        FormulaConfig(name="community_boost", alpha=0.10, gamma=0.15),
    ]

    if str(args.formulas).strip().lower() == "all":
        configs = configs_all
    else:
        selected = {x.strip() for x in str(args.formulas).split(",") if x.strip()}
        configs = [c for c in configs_all if c.name in selected]
        if not configs:
            raise ValueError("No valid formulas selected via --formulas")

    if args.auto_race:
        stages = _parse_race_stages(args.race_stages)
        current_cfgs = list(configs)
        all_stage_rows: list[dict[str, float | str | int]] = []

        needs_budget = any(c.name == "receiver_budget_views" for c in current_cfgs)
        sender_budget_w: np.ndarray | None = None
        receiver_budget_sum: np.ndarray | None = None
        if needs_budget:
            if float(args.budget_gamma) < 0:
                raise ValueError("--budget-gamma must be >= 0")
            sender_budget_w = (1.0 + float(args.budget_gamma) * s_views).astype(float, copy=False)
            receiver_budget_sum = _neighbor_weight_sum(csr_raw["indptr"], csr_raw["indices"], sender_budget_w)

        for stage_idx, stage in enumerate(stages, start=1):
            rows = _sample_rows(degrees=degrees, n_sample=int(stage.n_sample), seed=int(args.sample_seed + stage_idx * 13))
            sampled_degrees = degree_cent[rows]
            graph_feats = np.column_stack([degree_cent[rows], pagerank[rows], kshell[rows]]).astype(float)
            full_feats = np.column_stack([
                degree_cent[rows],
                pagerank[rows],
                kshell[rows],
                views_log[rows],
                views_per_day[rows],
                life_time[rows],
            ]).astype(float)

            stage_rows: list[dict[str, float | str | int]] = []
            print(
                f"[RACE] Stage {stage_idx}/{len(stages)}: n_sample={stage.n_sample}, n_runs={stage.n_runs}, "
                f"seeds={stage.mc_seeds}, candidates={len(current_cfgs)}"
            )

            for cfg in current_cfgs:
                row, scores_by_seed = _evaluate_formula(
                    cfg=cfg,
                    rows=rows,
                    csr=csr,
                    sender_strength=sender_strength,
                    sender_strength_views_centered=sender_strength_views_centered,
                    receiver_resistance=receiver_resistance,
                    sender_budget_w=sender_budget_w,
                    receiver_budget_sum=receiver_budget_sum,
                    comm_ids=comm_ids,
                    n_runs=int(stage.n_runs),
                    mc_seeds=stage.mc_seeds,
                    seed_multiplier=int(args.seed_multiplier),
                    n_jobs=int(args.n_jobs),
                    top_pct=float(args.top_pct),
                    graph_feats=graph_feats,
                    full_feats=full_feats,
                    sampled_degrees=sampled_degrees,
                    sample_seed=int(args.sample_seed),
                )
                score_samples = _compute_race_samples(
                    scores_by_seed=scores_by_seed,
                    graph_feats=graph_feats,
                    full_feats=full_feats,
                    sampled_degrees=sampled_degrees,
                    top_pct=float(args.top_pct),
                    sample_seed=int(args.sample_seed),
                )
                runtime_penalty = float(args.race_lambda) * math.log1p(float(row["runtime_sec"]))
                score_mean = float(np.mean(score_samples) - runtime_penalty)
                ci_lo, ci_hi = _bootstrap_ci(
                    samples=score_samples,
                    n_bootstrap=int(args.race_ci_bootstrap),
                    alpha=0.05,
                    seed=int(args.sample_seed + stage_idx),
                )
                ci_lo -= runtime_penalty
                ci_hi -= runtime_penalty
                row_stage = dict(row)
                row_stage.update(
                    {
                        "stage": int(stage_idx),
                        "race_score_mean": score_mean,
                        "race_score_ci_low": float(ci_lo),
                        "race_score_ci_high": float(ci_hi),
                        "race_samples_n": int(len(score_samples)),
                    }
                )
                stage_rows.append(row_stage)
                print(
                    f"[RACE] {cfg.name}: score={score_mean:.4f}, CI=[{ci_lo:.4f}, {ci_hi:.4f}], "
                    f"rho={float(row['spearman_mean']):.4f}, ndcg={float(row['graph_plus_views_ndcg']):.4f}, p@k={float(row['graph_plus_views_precision']):.4f}, runtime={float(row['runtime_sec']):.1f}s"
                )

            stage_df = pd.DataFrame(stage_rows).sort_values("race_score_mean", ascending=False).reset_index(drop=True)
            all_stage_rows.extend(stage_rows)

            # Write stage checkpoint for long runs
            stage_csv = Path(out_dir) / f"ic_formula_race_stage_{stage_idx}.csv"
            stage_df.to_csv(stage_csv, index=False)
            stage_json = Path(out_dir) / f"ic_formula_race_stage_{stage_idx}.json"
            with stage_json.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "stage": int(stage_idx),
                        "n_sample": int(stage.n_sample),
                        "n_runs": int(stage.n_runs),
                        "mc_seeds": stage.mc_seeds,
                        "candidates": [c.name for c in current_cfgs],
                        "rows": stage_df.to_dict(orient="records"),
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            print(f"[RACE] Wrote stage checkpoint: {stage_csv}")

            if stage_idx < len(stages):
                base_keep = max(int(args.race_min_keep), int(math.ceil(len(stage_df) * float(args.race_keep_frac))))
                base_keep = min(base_keep, len(stage_df))
                kept = stage_df.head(base_keep).copy()
                best_low = float(kept.iloc[0]["race_score_ci_low"])
                kept = kept[kept["race_score_ci_high"] >= (best_low - float(args.race_delta))]
                if kept.empty:
                    kept = stage_df.head(base_keep)
                current_names = set(kept["formula"].astype(str).tolist())
                current_cfgs = [c for c in current_cfgs if c.name in current_names]
                print(f"[RACE] Stage {stage_idx} keep={len(current_cfgs)} -> {sorted(current_names)}")

        out = pd.DataFrame(all_stage_rows)
        final_stage = int(out["stage"].max())
        final_df = out[out["stage"] == final_stage].sort_values("race_score_mean", ascending=False).reset_index(drop=True)

        csv_stage = Path(out_dir) / "ic_formula_race_stages.csv"
        out.to_csv(csv_stage, index=False)
        csv_final = Path(out_dir) / "ic_formula_race_final.csv"
        final_df.to_csv(csv_final, index=False)

        payload = {
            "config": {
                "auto_race": True,
                "race_stages": [
                    {"n_sample": s.n_sample, "n_runs": s.n_runs, "mc_seeds": s.mc_seeds}
                    for s in stages
                ],
                "race_keep_frac": float(args.race_keep_frac),
                "race_min_keep": int(args.race_min_keep),
                "race_lambda": float(args.race_lambda),
                "race_delta": float(args.race_delta),
                "race_ci_bootstrap": int(args.race_ci_bootstrap),
            },
            "winner": final_df.iloc[0].to_dict() if len(final_df) else None,
            "final_rows": final_df.to_dict(orient="records"),
            "all_stage_rows": out.to_dict(orient="records"),
        }
        json_path = Path(out_dir) / "ic_formula_race_summary.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"[OK] Wrote: {csv_stage}")
        print(f"[OK] Wrote: {csv_final}")
        print(f"[OK] Wrote: {json_path}")
    else:
        rows = _sample_rows(degrees=degrees, n_sample=int(args.n_sample), seed=int(args.sample_seed))
        sampled_degrees = degree_cent[rows]
        graph_feats = np.column_stack([degree_cent[rows], pagerank[rows], kshell[rows]]).astype(float)
        full_feats = np.column_stack([
            degree_cent[rows],
            pagerank[rows],
            kshell[rows],
            views_log[rows],
            views_per_day[rows],
            life_time[rows],
        ]).astype(float)

        needs_budget = any(c.name == "receiver_budget_views" for c in configs)
        sender_budget_w: np.ndarray | None = None
        receiver_budget_sum: np.ndarray | None = None
        if needs_budget:
            if float(args.budget_gamma) < 0:
                raise ValueError("--budget-gamma must be >= 0")
            sender_budget_w = (1.0 + float(args.budget_gamma) * s_views).astype(float, copy=False)
            receiver_budget_sum = _neighbor_weight_sum(csr_raw["indptr"], csr_raw["indices"], sender_budget_w)

        rows_out: list[dict[str, float | str]] = []

        for cfg in configs:
            row, _ = _evaluate_formula(
                cfg=cfg,
                rows=rows,
                csr=csr,
                sender_strength=sender_strength,
                sender_strength_views_centered=sender_strength_views_centered,
                receiver_resistance=receiver_resistance,
                sender_budget_w=sender_budget_w,
                receiver_budget_sum=receiver_budget_sum,
                comm_ids=comm_ids,
                n_runs=int(args.n_runs),
                mc_seeds=mc_seeds,
                seed_multiplier=int(args.seed_multiplier),
                n_jobs=int(args.n_jobs),
                top_pct=float(args.top_pct),
                graph_feats=graph_feats,
                full_feats=full_feats,
                sampled_degrees=sampled_degrees,
                sample_seed=int(args.sample_seed),
            )
            rows_out.append(row)
            print(
                f"[OK] {cfg.name}: stability_j={float(row['jaccard_mean']):.4f}, stability_rho={float(row['spearman_mean']):.4f}, "
                f"learn_rho_graph={float(row['graph_only_spearman']):.4f}, learn_rho_full={float(row['graph_plus_views_spearman']):.4f}, "
                f"runtime={float(row['runtime_sec']):.1f}s"
            )

        out = pd.DataFrame(rows_out)
        out["score_stability"] = 0.6 * _minmax(out["jaccard_mean"]) + 0.4 * _minmax(out["spearman_mean"])
        out["score_learnability"] = (
            0.5 * _minmax(out["graph_plus_views_spearman"])
            + 0.3 * _minmax(out["graph_plus_views_ndcg"])
            + 0.2 * _minmax(out["graph_plus_views_precision"])
        )
        out["score_runtime"] = 1.0 - _minmax(out["runtime_sec"])
        out["score_total"] = 0.45 * out["score_stability"] + 0.40 * out["score_learnability"] + 0.15 * out["score_runtime"]
        out = out.sort_values("score_total", ascending=False).reset_index(drop=True)

        csv_path = Path(out_dir) / "ic_formula_shootout_summary.csv"
        out.to_csv(csv_path, index=False)

        payload = {
            "config": {
                "n_sample": int(args.n_sample),
                "n_runs": int(args.n_runs),
                "mc_seeds": mc_seeds,
                "top_pct": float(args.top_pct),
                "n_jobs": int(args.n_jobs),
                "sample_seed": int(args.sample_seed),
            },
            "winner": out.iloc[0].to_dict() if len(out) else None,
            "rows": out.to_dict(orient="records"),
        }
        json_path = Path(out_dir) / "ic_formula_shootout_summary.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"[OK] Wrote: {csv_path}")
        print(f"[OK] Wrote: {json_path}")


if __name__ == "__main__":
    main()
