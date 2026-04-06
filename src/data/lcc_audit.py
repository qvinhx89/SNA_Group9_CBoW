"""LCC audit for MAPR2026 v3 pre-Day1 prerequisites.

Contract output:
- outputs/stage0_data_quality/lcc_report.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import networkx as nx


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate LCC report from active graph")
    p.add_argument("--edgelist", default="data/processed/graph_active.edgelist")
    p.add_argument("--out", default="outputs/stage0_data_quality/lcc_report.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    edgelist_path = Path(args.edgelist)
    out_path = Path(args.out)

    if not edgelist_path.exists():
        raise FileNotFoundError(f"Missing edgelist: {edgelist_path}")

    graph = nx.read_edgelist(edgelist_path, nodetype=str)
    n_nodes_total = graph.number_of_nodes()

    if n_nodes_total == 0:
        raise ValueError("Graph appears empty")

    components = list(nx.connected_components(graph))
    n_components = len(components)
    n_nodes_lcc = max(len(c) for c in components)
    pct_lcc = (n_nodes_lcc / n_nodes_total) * 100.0

    report = {
        "timestamp": datetime.now().isoformat(),
        "n_nodes_total": int(n_nodes_total),
        "n_nodes_lcc": int(n_nodes_lcc),
        "pct_lcc": float(pct_lcc),
        "n_components": int(n_components),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[OK] Wrote LCC report: {out_path}")


if __name__ == "__main__":
    main()
