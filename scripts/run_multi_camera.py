#!/usr/bin/env python
"""Multi-camera orchestration: per-camera detection/graph + cross-camera stitching.

Loads already-produced behavior vectors for several cameras, runs anomaly
detection and relation graphs per camera, then stitches adjacent cameras into
global vehicle ids.

Usage:
    python scripts/run_multi_camera.py \
        --cameras "admin_in=data/outputs/vectors/admin_in,admin_out=data/outputs/vectors/admin_out" \
        --pairs "admin_in:admin_out" \
        --output data/outputs/multi_camera.json \
        --no-detect
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.anomaly import AnomalyDetector, IsolationForestDetector, TrajectoryPredictionDetector
from src.graph import (
    RelationConfig,
    build_vehicle_graph,
    community_summary,
    graph_anomaly_index,
    louvain_partition,
    modularity,
)
from src.trajectory.stitching import CrossCameraStitcher
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_cameras(value: str):
    cameras = {}
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        camera_id, directory = token.split("=", 1)
        cameras[camera_id.strip()] = directory.strip()
    return cameras


def parse_pairs(value: str):
    pairs = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        a, b = token.split(":", 1)
        pairs.append((a.strip(), b.strip()))
    return pairs


from src.utils.vectors_io import load_vector_rows as load_vectors


def analysis_targets(rows):
    return [
        row for row in rows
        if str(row.get("analysis_target", "")).strip() in {"1", "true", "yes"}
    ]


def graph_summary(rows, top_k=3):
    config = RelationConfig(top_k=top_k)
    graph = build_vehicle_graph(rows, config=config)
    if len(graph.nodes) == 0:
        return {"node_count": 0, "edge_count": 0, "community_count": 0, "modularity": 0.0}
    membership = louvain_partition(graph.adjacency)
    scores = graph_anomaly_index(graph.adjacency, community=membership)
    summaries = community_summary(graph.adjacency, membership)
    q = modularity(graph.adjacency, membership)
    return {
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "community_count": len(summaries),
        "modularity": round(float(q), 4),
        "top_graph_anomaly": [
            {
                "node": graph.nodes[index],
                "graph_anomaly_index": round(float(scores["graph_anomaly_index"][index]), 4),
            }
            for index in list(np.argsort(-scores["graph_anomaly_index"]))[:3]
        ],
    }


def detect_rows(rows, camera_id):
    targets = analysis_targets(rows)
    if len(targets) < 4:
        logger.warning("[%s] too few targets for detection: %s", camera_id, len(targets))
        return rows, {"trained": False, "targets": len(targets)}

    iforest = IsolationForestDetector().fit(targets)
    predictor = TrajectoryPredictionDetector(
        past_len=5, future_len=5, epochs=20
    ).fit(targets)
    detector = AnomalyDetector(iforest=iforest, trajectory_predictor=predictor)
    detected = detector.detect(rows)
    anomalies = sum(1 for row in detected if int(row.get("is_anomaly", 0) or 0) == 1)
    return detected, {"trained": True, "targets": len(targets), "anomalies": anomalies}


def main():
    parser = argparse.ArgumentParser(description="Multi-camera orchestration")
    parser.add_argument("--cameras", required=True, help='e.g. "camA=dirA,camB=dirB"')
    parser.add_argument("--pairs", default="", help='e.g. "camA:camB,camB:camC"')
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--threshold", type=float, default=0.7, help="ReID cosine threshold")
    parser.add_argument("--top-k", type=int, default=3, help="Graph top-k")
    parser.add_argument("--no-detect", action="store_true", help="Skip unsupervised detection")
    args = parser.parse_args()

    cameras = parse_cameras(args.cameras)
    pairs = parse_pairs(args.pairs) if args.pairs else []

    vectors = {camera_id: load_vectors(directory) for camera_id, directory in cameras.items()}
    for camera_id, rows in vectors.items():
        logger.info("[%s] vectors: %s", camera_id, len(rows))

    detected = {}
    detection_summary = {}
    for camera_id, rows in vectors.items():
        if args.no_detect:
            detected[camera_id] = rows
            detection_summary[camera_id] = {"trained": False, "targets": len(analysis_targets(rows))}
        else:
            detected[camera_id], summary = detect_rows(rows, camera_id)
            detection_summary[camera_id] = summary

    graphs = {camera_id: graph_summary(rows, top_k=args.top_k) for camera_id, rows in detected.items()}

    stitch_results = {}
    global_ids = {}
    for from_camera, to_camera in pairs:
        if from_camera not in vectors or to_camera not in vectors:
            logger.warning("Pair skipped (missing camera): %s -> %s", from_camera, to_camera)
            continue
        stitcher = CrossCameraStitcher(similarity_threshold=args.threshold)
        matches = stitcher.match(vectors[from_camera], vectors[to_camera], from_camera, to_camera)
        nodes = [stitcher.passage_key(from_camera, row) for row in vectors[from_camera]]
        nodes += [stitcher.passage_key(to_camera, row) for row in vectors[to_camera]]
        mapping = stitcher.assign_global_ids(nodes, matches)
        key = f"{from_camera}->{to_camera}"
        stitch_results[key] = {
            "match_count": len(matches),
            "matches": matches,
            "global_vehicle_count": len(set(mapping.values())),
        }
        for node_key, gid in mapping.items():
            global_ids[str(node_key)] = gid

    payload = {
        "cameras": {camera_id: len(rows) for camera_id, rows in vectors.items()},
        "detection": detection_summary,
        "graphs": graphs,
        "stitching": stitch_results,
        "global_vehicle_ids": global_ids,
    }

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "cameras": payload["cameras"],
        "detection": payload["detection"],
        "stitching": {key: value["match_count"] for key, value in stitch_results.items()},
    }, ensure_ascii=False, indent=2))
    logger.info("Saved: %s", args.output)


if __name__ == "__main__":
    main()
