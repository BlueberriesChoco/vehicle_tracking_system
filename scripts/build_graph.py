#!/usr/bin/env python
"""Build a vehicle relation graph from behavior vector CSVs.

Outputs a JSON file containing:
  - nodes: vehicle key, community id, graph anomaly index G, degree, clustering;
  - communities: size / internal weight / density (for gang review);
  - edges: co-occurrence / following / path-similarity relations.

This is the first stage of research point 3 (relation graph + community discovery).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.graph import (
    RelationConfig,
    build_graph_payload,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_groups(value: str):
    return [item.strip() for item in value.split(",") if item.strip()]


from src.utils.vectors_io import load_vector_rows as load_rows


def main():
    parser = argparse.ArgumentParser(description="Build vehicle relation graph and graph anomaly index G")
    parser.add_argument("--input", required=True, help="A vec_*.csv file or directory containing vector CSVs")
    parser.add_argument("--output", required=True, help="Output graph JSON path")
    parser.add_argument("--min-overlap-sec", type=float, default=1.0, help="Co-occurrence minimum time overlap (s)")
    parser.add_argument("--min-overlap-ratio", type=float, default=0.5, help="Co-occurrence minimum overlap ratio")
    parser.add_argument("--follow-gap-sec", type=float, default=10.0, help="Following maximum enter-time gap (s)")
    parser.add_argument("--path-threshold", type=float, default=0.85, help="Trajectory similarity threshold")
    parser.add_argument("--min-points", type=int, default=8, help="Minimum trajectory points for similarity")
    parser.add_argument("--top-k", type=int, default=3, help="Keep top-k strongest neighbors per node (0 = no limit)")
    parser.add_argument("--resolution", type=float, default=1.0, help="Louvain resolution")
    parser.add_argument("--gae", action="store_true", help="(Deprecated) same as --method gae")
    parser.add_argument("--method", choices=["svd", "gae"], default="svd", help="Graph anomaly method (default: svd)")
    parser.add_argument("--gae-epochs", type=int, default=150, help="GAE training epochs")
    parser.add_argument("--include-freight", action="store_true", help="Include non ordinary vehicles")
    args = parser.parse_args()

    rows = load_rows(args.input)
    if not rows:
        logger.error("No vector rows found: %s", args.input)
        sys.exit(1)

    config = RelationConfig(
        co_occurrence_min_overlap_sec=args.min_overlap_sec,
        co_occurrence_min_overlap_ratio=args.min_overlap_ratio,
        follow_max_gap_sec=args.follow_gap_sec,
        path_similarity_threshold=args.path_threshold,
        min_trajectory_points=args.min_points,
        top_k=args.top_k,
    )
    method = "gae" if (args.method == "gae" or args.gae) else "svd"
    payload = build_graph_payload(
        rows,
        config=config,
        method=method,
        resolution=args.resolution,
        gae_epochs=args.gae_epochs,
        only_analysis_target=not args.include_freight,
    )
    if payload is None:
        logger.error("No analysable vehicles found.")
        sys.exit(1)

    node_count = payload["node_count"]
    edge_count = payload["edge_count"]
    community_count = len(payload["communities"])
    q = payload["modularity"]

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info(
        "Graph: %s nodes, %s edges, %s communities, modularity=%.4f",
        node_count, edge_count, community_count, q,
    )
    print(f"Graph saved to: {args.output}")
    print(f"Nodes: {node_count}, edges: {edge_count}, communities: {community_count}, modularity: {q:.4f}")


if __name__ == "__main__":
    main()
