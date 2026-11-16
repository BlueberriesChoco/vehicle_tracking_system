#!/usr/bin/env python
"""Evaluate measurable technical indicators of the prototype.

Computes:
- vehicle / analysis-target / alert counts and alert rate (proxy for FPR);
- evidence-chain trace completeness (share of evidence items with camera/track/source);
- graph size, community count, modularity;
- risk score distribution.

Usage:
    python scripts/eval_system.py \
        --vectors data/outputs/vectors/admin_in/20260901 \
        --alerts data/outputs/alerts/alerts_admin_in_20260901.csv \
        --risk data/outputs/risk/risk_admin_in_20260901.json \
        --graph data/outputs/graphs/graph_admin_in_20260901.json \
        --output data/outputs/eval/eval_admin_in_20260901.json
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.logger import get_logger

logger = get_logger(__name__)


from src.utils.vectors_io import load_vector_rows as load_vectors


def load_csv(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def evaluate(args):
    result = {}

    vectors = load_vectors(args.vectors)
    targets = [row for row in vectors if str(row.get("analysis_target", "")).strip() in {"1", "true", "yes"}]
    result["vehicle_count"] = len(vectors)
    result["analysis_targets"] = len(targets)

    if args.alerts and os.path.exists(args.alerts):
        alerts = load_csv(args.alerts)
        flagged = [row for row in alerts if str(row.get("is_anomaly", "0")) in {"1", "true", "True"}]
        result["alert_count"] = len(flagged)
        result["alert_rate"] = round(len(flagged) / max(len(targets), 1), 4)
        result["alert_rate_percent"] = round(100.0 * len(flagged) / max(len(targets), 1), 2)

    if args.risk and os.path.exists(args.risk):
        with open(args.risk, "r", encoding="utf-8") as f:
            risk = json.load(f)
        vehicles = risk.get("vehicles", [])
        result["risk_vehicle_count"] = len(vehicles)
        result["risk_high"] = sum(1 for v in vehicles if v.get("risk_level") == "high")
        result["risk_medium"] = sum(1 for v in vehicles if v.get("risk_level") == "medium")
        scores = [float(v.get("risk_score_100", 0)) for v in vehicles]
        result["risk_max"] = round(max(scores), 1) if scores else 0.0
        result["risk_mean"] = round(sum(scores) / len(scores), 1) if scores else 0.0

        total_items = 0
        traceable_items = 0
        for vehicle in vehicles:
            for item in vehicle.get("evidence", []):
                total_items += 1
                trace = item.get("trace", {})
                if trace.get("camera_id") and trace.get("track_id") != "" and trace.get("source_file"):
                    traceable_items += 1
        result["evidence_items"] = total_items
        result["evidence_traceable"] = traceable_items
        result["evidence_trace_completeness"] = (
            round(traceable_items / total_items, 4) if total_items else 0.0
        )

    if args.graph and os.path.exists(args.graph):
        with open(args.graph, "r", encoding="utf-8") as f:
            graph = json.load(f)
        result["graph_nodes"] = graph.get("node_count", 0)
        result["graph_edges"] = graph.get("edge_count", 0)
        result["graph_communities"] = len(graph.get("communities", []))
        result["graph_modularity"] = graph.get("modularity", 0.0)

    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate measurable prototype indicators")
    parser.add_argument("--vectors", required=True)
    parser.add_argument("--alerts", default=None)
    parser.add_argument("--risk", default=None)
    parser.add_argument("--graph", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result = evaluate(args)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info("Evaluation saved to: %s", args.output)


if __name__ == "__main__":
    main()
