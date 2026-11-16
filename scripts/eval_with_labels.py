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
    return 1 if str(value).strip() in {"1", "True", "true", "yes"} else 0


def to_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main():
    parser = argparse.ArgumentParser(description="Evaluate with labels")
    parser.add_argument("--vectors", required=True, help="Annotated vectors CSV")
    parser.add_argument("--labels", required=True, help="Label CSV (passage_id + is_violation)")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    vectors = load_csv(args.vectors)
    labels = load_csv(args.labels)
    label_map = {
        str(row.get("passage_id", "")).strip(): row
        for row in labels
        if str(row.get("passage_id", "")).strip()
    }

    matched = []
    for row in vectors:
        passage_id = str(row.get("passage_id", "")).strip()
        label = label_map.get(passage_id)
        if label is None:
            continue
        matched.append((row, parse_label(label.get("is_violation"))))

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

    y_true = [label for _, label in matched]
    y_pred = [parse_label(row.get("is_anomaly")) for row, _ in matched]
    risk_scores = [to_float(row.get("risk_score_100")) for row, _ in matched]
    iforest_scores = [to_float(row.get("iforest_score")) for row, _ in matched]
    graph_scores = [to_float(row.get("graph_anomaly_index")) for row, _ in matched]

    result = {
        "labeled_vehicles": len(matched),
        "labels_provided": len(label_map),
        "labels_unmatched": unmatched,
        "positives": sum(y_true),
        "negatives": len(y_true) - sum(y_true),
        "alert": precision_recall_f1(y_true, y_pred),
        "auc": {
            "risk_score": roc_auc(y_true, risk_scores),
            "iforest_score": roc_auc(y_true, iforest_scores),
            "graph_anomaly_index": roc_auc(y_true, graph_scores),
        },
        "high_risk": precision_at_threshold(y_true, risk_scores, threshold=80.0),
        "high_risk_count": sum(1 for score in risk_scores if score >= 80.0),
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
