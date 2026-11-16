#!/usr/bin/env python
"""Synthetic anomaly evaluation for known anomaly types.

Because no labelled violation cases are available, this script injects known
anomalies into copies of normal behavior vectors and measures detection recall
per type plus the false-positive rate on the untouched normal rows.

Anomaly types:
- long_dwell   : 超长停留（max_dwell_sec / dwell_ratio / stop_count）
- snake        : 蛇形行驶（轨迹中段加横向正弦扰动）
- night_high_freq : 夜间高频（is_night / night_ratio / freq_count_24h）
- speeding     : 超速（avg_speed_ms / max_speed_ms）

Important: the model is trained only on the ORIGINAL normal rows; injected rows
are never used for training.

``auc`` 的含义要小心：它衡量「行为异常指数 A 能否把注入样本排到正常样本前面」。
如果某特征在训练集上恒定（本数据 ``max_dwell_sec`` / ``stop_count`` / ``path_deviation``
在 432 辆正常车中**全为 0**），树的切分点会退化，注入的正数与原值走同一分支，
A 完全不变——此时 AUC 会接近甚至低于 0.5，
**说明的是"注入对 A 不可见"，而不是"模型差"**。
同理，被 ``speed_reliable`` / ``path_reliable`` 门控掉的字段（未标定摄像头）不参与打分。

Usage:
    python scripts/synthetic_anomaly_eval.py \
        --input data/outputs/vectors/admin_in/20260901 \
        --output data/outputs/eval/synthetic_eval.json \
        --per-type 30
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.anomaly import AnomalyDetector, IsolationForestDetector, TrajectoryPredictionDetector
from src.anomaly.metrics import roc_auc
from src.utils.logger import get_logger

logger = get_logger(__name__)


from src.utils.vectors_io import load_vector_rows as load_rows


def _parse_points(text: str):
    points = []
    for token in str(text).split(";"):
        parts = token.split(",")
        if len(parts) == 2:
            try:
                points.append([float(parts[0]), float(parts[1])])
            except ValueError:
                pass
    return points


def _format_points(points):
    return ";".join(f"{x:.1f},{y:.1f}" for x, y in points)


def inject_long_dwell(row: dict) -> dict:
    record = dict(row)
    record["max_dwell_sec"] = max(float(record.get("max_dwell_sec") or 0), 600.0)
    record["dwell_ratio"] = max(float(record.get("dwell_ratio") or 0), 0.8)
    record["stop_count"] = max(int(float(record.get("stop_count") or 0)), 3)
    return record


def inject_night_high_freq(row: dict) -> dict:
    record = dict(row)
    record["is_night"] = 1
    record["night_ratio"] = 1.0
    record["freq_count_24h"] = max(int(float(record.get("freq_count_24h") or 0)), 12)
    record["freq_index"] = max(float(record.get("freq_index") or 0), 3.0)
    return record


def inject_speeding(row: dict) -> dict:
    record = dict(row)
    record["avg_speed_ms"] = max(float(record.get("avg_speed_ms") or 0), 30.0)
    record["max_speed_ms"] = max(float(record.get("max_speed_ms") or 0), 40.0)
    # 真实数据未标定时 speed_reliable=0，超速规则会被正确跳过；
    # 合成评估要验证“标定后规则是否生效”，因此显式置为 1。
    record["speed_reliable"] = 1
    return record


def inject_snake(row: dict) -> dict:
    record = dict(row)
    points = _parse_points(record.get("trajectory_points", ""))
    if len(points) >= 6:
        # 在轨迹中段叠加垂直于主方向的横向正弦扰动
        sx, sy = points[0]
        ex, ey = points[-1]
        dx, dy = ex - sx, ey - sy
        length = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / length, dx / length
        for index in range(1, len(points) - 1):
            t = index / (len(points) - 1)
            amplitude = 60.0 * math.sin(t * math.pi) * math.sin(t * math.pi * 6)
            points[index][0] += nx * amplitude
            points[index][1] += ny * amplitude
        record["trajectory_points"] = _format_points(points)
        record["path_deviation"] = max(float(record.get("path_deviation") or 0), 8.0)
        # 蛇形 = 平均转角大（见 PathReference.compute_smoothness“值越大越犹豫”），
        # 不是小；这里必须注入高值，否则合成样本和规则方向相反。
        record["path_smoothness"] = max(float(record.get("path_smoothness") or 0), 0.8)
    return record


def main():
    parser = argparse.ArgumentParser(description="Synthetic anomaly recall evaluation")
    parser.add_argument("--input", required=True, help="Normal vector CSV or directory")
    parser.add_argument("--output", required=True, help="Evaluation JSON path")
    parser.add_argument("--per-type", type=int, default=30, help="Injected samples per anomaly type")
    parser.add_argument("--epochs", type=int, default=30, help="Transformer training epochs")
    parser.add_argument("--contamination", type=float, default=0.05, help="Expected anomaly fraction for IF and trajectory residual")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_rows(args.input)
    normal = [row for row in rows if str(row.get("analysis_target", "")).strip() in {"1", "true", "yes"}]
    if len(normal) < 20:
        logger.error("Not enough normal rows: %s", len(normal))
        sys.exit(1)

    # 只用原始正常样本训练
    iforest = IsolationForestDetector(contamination=args.contamination).fit(normal)
    predictor = TrajectoryPredictionDetector(
        past_len=5, future_len=5, epochs=args.epochs,
        contamination=args.contamination, random_state=args.seed,
    ).fit(normal)
    detector = AnomalyDetector(iforest=iforest, trajectory_predictor=predictor)

    injectors = {
        "long_dwell": inject_long_dwell,
        "snake": inject_snake,
        "night_high_freq": inject_night_high_freq,
        "speeding": inject_speeding,
    }

    sample = normal[: args.per_type]
    injected = []
    labels = []
    for name, injector in injectors.items():
        for row in sample:
            injected.append(injector(row))
            labels.append(name)

    mixed = [dict(row, synthetic_label="normal") for row in normal] + [
        dict(row, synthetic_label=label) for row, label in zip(injected, labels)
    ]
    detected = detector.detect(mixed)

    per_type = {}
    for name in injectors:
        indices = [i for i, row in enumerate(detected) if row.get("synthetic_label") == name]
        flagged = [i for i in indices if int(row_flag(detected[i])) == 1]
        per_type[name] = {
            "injected": len(indices),
            "detected": len(flagged),
            "recall": round(len(flagged) / max(len(indices), 1), 4),
        }

    normal_indices = [i for i, row in enumerate(detected) if row.get("synthetic_label") == "normal"]
    false_positives = [i for i in normal_indices if int(row_flag(detected[i])) == 1]
    fpr = len(false_positives) / max(len(normal_indices), 1)

    # AUC：用行为异常指数 A（iforest_score）排序，衡量“未知异常”的区分能力。
    # IF 只见过正常样本，因此这里每个异常类型对它都是“未知类型”。
    def _score(index: int) -> float:
        try:
            return float(detected[index].get("iforest_score") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    normal_scores = [_score(i) for i in normal_indices]
    all_anomaly_scores, all_labels = [], []
    for name in injectors:
        indices = [i for i, row in enumerate(detected) if row.get("synthetic_label") == name]
        anomaly_scores = [_score(i) for i in indices]
        auc = roc_auc([1] * len(anomaly_scores) + [0] * len(normal_scores),
                      anomaly_scores + normal_scores)
        per_type[name]["auc"] = None if auc is None else round(float(auc), 4)
        all_anomaly_scores.extend(anomaly_scores)
        all_labels.extend([1] * len(anomaly_scores))

    overall_auc = roc_auc(all_labels + [0] * len(normal_scores), all_anomaly_scores + normal_scores)

    result = {
        "normal_count": len(normal_indices),
        "injected_per_type": args.per_type,
        "per_type": per_type,
        "overall_auc": None if overall_auc is None else round(float(overall_auc), 4),
        "false_positives": len(false_positives),
        "false_positive_rate": round(fpr, 4),
        "false_positive_rate_percent": round(100.0 * fpr, 2),
    }

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    logger.info("Synthetic evaluation saved to: %s", args.output)


def row_flag(row: dict) -> int:
    try:
        return int(row.get("is_anomaly", 0) or 0)
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    main()
