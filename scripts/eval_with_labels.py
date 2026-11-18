#!/usr/bin/env python
"""Evaluate anomaly detection and risk scoring against labeled cases.

Joins annotated vectors with a label CSV (see ``config/label_template.csv``)
by ``passage_id`` and reports precision / recall / F1 and ROC-AUC for the
behavior anomaly index, the graph index and the fused risk score.

Usage:
    python scripts/eval_with_labels.py \
        --vectors data/outputs/full_rerun/alerts005/vectors.csv \
        --labels config/label_template.csv \
        --output data/outputs/full_rerun/label_eval.json
"""

import argparse
import csv
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.anomaly.metrics import precision_at_threshold, precision_recall_f1, roc_auc
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_csv(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_label(value) -> int:
    flag = str(value).strip().lower()
    if flag in {"1", "true", "yes"}:
        return 1
    if flag in {"0", "false", "no"}:
        return 0
    raise ValueError(f"Expected a binary label, got {value!r}")


def index_passages(rows, name, *, require_id=False):
    indexed = {}
    for row in rows:
        passage_id = (row.get("passage_id") or "").strip()
        if not passage_id:
            if require_id:
                raise ValueError(f"{name}: missing passage_id")
            continue
        if passage_id in indexed:
            raise ValueError(f"{name}: duplicate passage_id {passage_id!r}")
        indexed[passage_id] = row
    return indexed


def main():
    parser = argparse.ArgumentParser(description="Evaluate with labels")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--vectors", help="Annotated vectors CSV")
    inputs.add_argument("--risk-report", help="Full risk report JSON with vehicle records")
    parser.add_argument("--labels", required=True, help="Label CSV (passage_id + is_violation)")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    if args.risk_report:
        with open(args.risk_report, encoding="utf-8") as handle:
            vectors = json.load(handle)["vehicles"]
    else:
        vectors = load_csv(args.vectors)
    labels = load_csv(args.labels)
    try:
        vector_map = index_passages(vectors, "vectors")
        label_map = index_passages(labels, "labels", require_id=True)
        truth = {key: parse_label(row.get("is_violation")) for key, row in label_map.items()}
        matched = [(row, truth[key]) for key, row in vector_map.items() if key in truth]
        y_pred = [parse_label(row.get("is_anomaly")) for row, _ in matched]
        y_true = [label for _, label in matched]
        model_metrics = {}
        for field, available in (
            ("iforest_anomaly", "iforest_available"), ("traj_anomaly", "trajectory_available")
        ):
            scored = [(label, parse_label(row[field])) for row, label in matched
                      if row.get(available) not in (None, "") and parse_label(row[available])]
            model_metrics[field] = (
                {"scored": len(scored), **precision_recall_f1(*zip(*scored))}
                if scored else None
            )
    except ValueError as exc:
        parser.error(str(exc))

    if not matched:
        logger.error("No overlapping passage_id between vectors and labels")
        sys.exit(1)

    # 标注里如果有拼错/日期不对的 passage_id，会被静默丢弃；
    # 这里显式统计并告警，避免"标了 100 辆，实际只算到 30 辆"却无人察觉。
    unmatched = len(label_map) - len(matched)
    if unmatched > 0:
        ratio = unmatched / max(len(label_map), 1)
        logger.warning(
            "%d / %d labels did not match any vector passage_id (%.1f%%) — 检查 passage_id 拼写或日期",
            unmatched, len(label_map), 100.0 * ratio,
        )

    def collect_scores(field, availability=None):
        scores = []
        for row, label in matched:
            available = row.get(availability) if availability else None
            if available not in (None, "") and not parse_label(available):
                continue
            value = row.get(field)
            if value not in (None, ""):
                score = float(value)
                if not math.isfinite(score):
                    raise ValueError(f"{field} must be finite")
                scores.append((label, score))
        return tuple(zip(*scores)) if scores else ((), ())

    try:
        score_groups = {
            "risk_score": collect_scores("risk_score_100"),
            "iforest_score": collect_scores("iforest_score", "iforest_available"),
            "graph_anomaly_index": collect_scores("graph_anomaly_index", "graph_available"),
        }
    except ValueError as exc:
        parser.error(str(exc))
    risk_labels, risk_scores = score_groups["risk_score"]

    result = {
        "labeled_vehicles": len(matched),
        "labels_provided": len(label_map),
        "labels_unmatched": unmatched,
        "label_match_rate": round(len(matched) / len(label_map), 4),
        "vector_label_coverage": round(len(matched) / len(vectors), 4),
        "positives": sum(y_true),
        "negatives": len(y_true) - sum(y_true),
        "alert": precision_recall_f1(y_true, y_pred),
        "models": model_metrics,
        "auc": {key: roc_auc(*group) for key, group in score_groups.items()},
        "score_sample_counts": {key: len(group[0]) for key, group in score_groups.items()},
        "high_risk": precision_at_threshold(risk_labels, risk_scores, threshold=80.0) if risk_scores else None,
        "high_risk_count": sum(1 for score in risk_scores if score >= 80.0) if risk_scores else None,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info("Label evaluation saved to: %s", args.output)


if __name__ == "__main__":
    main()
