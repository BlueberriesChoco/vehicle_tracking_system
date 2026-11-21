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
import csv
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
from src.pipeline.global_identity import (
    annotate_global_identity,
    apply_graph_by_identity,
    build_global_graph,
    identity_summary,
    merge_passages,
    merge_risk_records,
)
from src.risk import EvidenceBuilder, RiskFusion, TimeFrequencyScorer, apply_passage_counts
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


def write_csv(rows, path):
    """写出向量 CSV（键取并集，缺字段留空）。"""
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def write_back(directory, detected, stitch_payload, top_k):
    """把跨机位身份回灌到向量、全局图与风险报告。"""
    annotated = annotate_global_identity(detected, stitch_payload)
    flat = [row for rows in annotated.values() for row in rows]

    graph = build_global_graph(merge_passages(flat), top_k=top_k)
    flat = apply_graph_by_identity(flat, graph)
    # 多机位场景下身份键覆盖率高得多（有 global_vehicle_id 兜底），
    # 频次通常可用；仍按同一套可用性逻辑决定是否剔除 T。
    flat, freq_summary = apply_passage_counts(flat)
    flat = TimeFrequencyScorer().score(flat)
    flat = RiskFusion().fuse(flat)
    flat = EvidenceBuilder().build_many(flat)
    merged = merge_risk_records(flat)

    by_camera = {}
    for row in flat:
        by_camera.setdefault(str(row.get("camera_id", "") or ""), []).append(row)
    for camera_id, rows in by_camera.items():
        write_csv(rows, os.path.join(directory, "vectors", f"{camera_id}.csv"))

    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "graph_global.json"), "w", encoding="utf-8") as handle:
        json.dump(graph or {}, handle, ensure_ascii=False, indent=2)
    with open(os.path.join(directory, "risk_global.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "vehicle_count": len(merged),
                "high_risk": sum(1 for row in merged if row.get("risk_level") == "high"),
                "medium_risk": sum(1 for row in merged if row.get("risk_level") == "medium"),
                "vehicles": merged,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    summary = {
        "identity": identity_summary(annotated, merged),
        "graph": {
            "node_count": (graph or {}).get("node_count", 0),
            "edge_count": (graph or {}).get("edge_count", 0),
            "community_count": len((graph or {}).get("communities", [])),
            "modularity": (graph or {}).get("modularity", 0.0),
        },
        "risk": {
            "high": sum(1 for row in merged if row.get("risk_level") == "high"),
            "medium": sum(1 for row in merged if row.get("risk_level") == "medium"),
            "evidence_total": sum(len(row.get("evidence", []) or []) for row in merged),
        },
    }
    with open(os.path.join(directory, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    logger.info(
        "Write-back: %s passages -> %s vehicles (%s with global id), graph nodes=%s",
        summary["identity"]["passages"],
        summary["identity"]["vehicles_after_merge"],
        summary["identity"]["passages_with_global_id"],
        summary["graph"]["node_count"],
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Multi-camera orchestration")
    parser.add_argument("--cameras", required=True, help='e.g. "camA=dirA,camB=dirB"')
    parser.add_argument("--pairs", default="", help='e.g. "camA:camB,camB:camC"')
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--threshold", type=float, default=0.7, help="ReID cosine threshold")
    parser.add_argument("--top-k", type=int, default=3, help="Graph top-k")
    parser.add_argument("--no-detect", action="store_true", help="Skip unsupervised detection")
    parser.add_argument(
        "--write-back",
        default="",
        help="Directory to write global-identity vectors, global graph and risk report",
    )
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

    if args.write_back:
        summary = write_back(args.write_back, detected, payload, args.top_k)
        print(json.dumps({"write_back": summary}, ensure_ascii=False, indent=2))

    logger.info("Saved: %s", args.output)


if __name__ == "__main__":
    main()
