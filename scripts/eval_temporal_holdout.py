#!/usr/bin/env python
"""时间留出评估：回答「模型是不是在自己训练过的数据上打分」。

`run_full_pipeline.py` 会把当天这批车**先训练再打分**（样本内）。这个脚本改用
时间切分：前 60% 的通行记录训练，后 40% 打分，得到样本外的告警率。

用法：
    python scripts/eval_temporal_holdout.py \
        --vectors data/outputs/e2e/alerts/vectors_admin_in.csv \
        --train-ratio 0.6 --output data/outputs/eval/temporal_holdout.json

注意：这是**同一天、同一机位**的时间留出，不能替代跨天评估。
跨天漂移需要至少两天的数据（见 DATA_REQUIREMENTS.md）。
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.anomaly import IsolationForestDetector, TrajectoryPredictionDetector
from src.utils.logger import get_logger
from src.utils.vectors_io import load_vector_rows

logger = get_logger(__name__)


def _rate(rows, key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if int(row.get(key, 0) or 0) == 1) / len(rows)


def _stats(rows, field: str) -> dict:
    values = sorted(float(row.get(field, 0) or 0) for row in rows)
    if not values:
        return {}
    return {
        "mean": round(sum(values) / len(values), 2),
        "p95": round(values[int(len(values) * 0.95)], 2),
        "max": round(values[-1], 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Temporal holdout evaluation")
    parser.add_argument("--vectors", required=True, help="Annotated vectors CSV")
    parser.add_argument("--train-ratio", type=float, default=0.6, help="Fraction of earliest rows used for training")
    parser.add_argument("--contamination", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    rows = [r for r in load_vector_rows(args.vectors)
            if str(r.get("analysis_target", "")).strip() in {"1", "true", "yes"}]
    if len(rows) < 20:
        logger.error("Not enough analysis targets: %s", len(rows))
        sys.exit(1)
    rows.sort(key=lambda r: r.get("enter_time") or "")

    split = int(len(rows) * args.train_ratio)
    train, test = rows[:split], rows[split:]
    logger.info("train=%d test=%d", len(train), len(test))

    iforest = IsolationForestDetector(contamination=args.contamination).fit(train)
    if_train = iforest.score(train)
    if_test = iforest.score(test)

    predictor = TrajectoryPredictionDetector(
        past_len=5, future_len=5, epochs=args.epochs,
        contamination=args.contamination, random_state=42,
    ).fit(train)
    tf_train = predictor.score(train)
    tf_test = predictor.score(test)

    result = {
        "train_count": len(train),
        "test_count": len(test),
        "train_window": [train[0].get("enter_time"), train[-1].get("enter_time")],
        "test_window": [test[0].get("enter_time"), test[-1].get("enter_time")],
        "contamination": args.contamination,
        "iforest": {
            "in_sample_rate": round(_rate(if_train, "iforest_anomaly"), 4),
            "out_sample_rate": round(_rate(if_test, "iforest_anomaly"), 4),
            "train_scores": _stats(if_train, "iforest_score"),
            "test_scores": _stats(if_test, "iforest_score"),
        },
        "transformer": {
            "in_sample_rate": round(_rate(tf_train, "traj_anomaly"), 4),
            "out_sample_rate": round(_rate(tf_test, "traj_anomaly"), 4),
            "train_residual": _stats(tf_train, "traj_residual"),
            "test_residual": _stats(tf_test, "traj_residual"),
        },
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        logger.info("Temporal holdout saved to: %s", args.output)


if __name__ == "__main__":
    main()
