#!/usr/bin/env python
"""抗规避测试：把车的行为轻微改一改，看它会不会掉出前 K 名。

为什么需要：违规样本极少、人工查验有固定规律，对手摸清规律后只要稍微改变行为
就能躲开。如果系统对 5% 的扰动极其敏感，说明它抓的是表面阈值，同样好躲。

做法：加载已训练的 Isolation Forest，对每个候选施加「现实中司机能做到的」微扰，
重新打分，比较前 K 名名单的重叠率。

    重叠率高（≈1）→ 抓住的是复合模式，改动一点仍在名单里
    重叠率低（<0.5）→ 一改就掉出名单，容易被规避

用法：
    python scripts/eval_perturbation.py \
        --vectors data/outputs/e2e_acceptance_demo/alerts/vectors_admin_in.csv \
        --iforest data/outputs/e2e_acceptance_demo/references/iforest_admin_in.json \
        --top-k 20 --output data/outputs/review/perturbation.json
"""

import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.anomaly.baseline import DEFAULT_FEATURES
from src.anomaly.isolation_forest import IsolationForestDetector
from src.utils.logger import get_logger
from src.utils.vectors_io import load_vector_rows

logger = get_logger(__name__)


def feature_health(rows) -> list:
    """检查每个特征是否真的可用：常量、被可靠性门控、或几乎全是零。

    没有这一步，很容易出现「模型在跑，其实只用了 1 个特征」的假象。
    """
    report = []
    for spec in DEFAULT_FEATURES:
        values = []
        gated = 0
        for row in rows:
            flag = getattr(spec, "reliability_field", None)
            if flag and str(row.get(flag, 1)).strip() in {"0", "false", "no"}:
                gated += 1
                continue
            try:
                values.append(float(row.get(spec.name)))
            except (TypeError, ValueError):
                continue
        unique = len({round(value, 6) for value in values})
        if not values:
            status = "全部被门控"
        elif unique <= 1:
            status = "常量"
        elif sum(1 for value in values if value != 0) / len(values) < 0.05:
            status = "几乎全零"
        else:
            status = "可用"
        report.append({
            "feature": spec.name,
            "status": status,
            "values": len(values),
            "gated": gated,
            "unique": unique,
        })
    return report

# 每一项都是「司机现实中做得到」的微调，不是任意数值噪声
PERTURBATIONS = [
    ("停留时长 -20%", {"max_dwell_sec": 0.8, "dwell_ratio": 0.8}),
    ("停留时长 -10%", {"max_dwell_sec": 0.9, "dwell_ratio": 0.9}),
    ("停留次数 -1", {"stop_count": ("minus", 1)}),
    ("平均速度 +5%", {"avg_speed_ms": 1.05, "max_speed_ms": 1.05}),
    ("平均速度 -5%", {"avg_speed_ms": 0.95, "max_speed_ms": 0.95}),
    ("速度波动 -20%", {"speed_variance": 0.8}),
    ("路径偏离 -20%", {"path_deviation": 0.8}),
    ("轨迹平滑度 +10%", {"path_smoothness": 1.1}),
    ("通行时长 -10%", {"duration_sec": 0.9}),
    ("聚集指数 -20%", {"aggregation_index": 0.8}),
]


def _apply(rows, changes):
    perturbed = copy.deepcopy(rows)
    for row in perturbed:
        for field, rule in changes.items():
            if field not in row:
                continue
            try:
                value = float(row.get(field) or 0.0)
            except (TypeError, ValueError):
                continue
            if isinstance(rule, tuple) and rule[0] == "minus":
                value = max(0.0, value - float(rule[1]))
            else:
                value = value * float(rule)
            row[field] = value
    return perturbed


def _top_k(rows, detector, k):
    scored = detector.score(rows)
    ranked = sorted(scored, key=lambda row: -float(row.get("iforest_score", 0) or 0))
    return [str(row.get("passage_id", "")) for row in ranked[:k]], scored


def evaluate(rows, detector, top_k=20):
    baseline_keys, baseline_scored = _top_k(rows, detector, top_k)
    baseline_set = set(baseline_keys)
    baseline_rank = {}
    ranked = sorted(baseline_scored, key=lambda row: -float(row.get("iforest_score", 0) or 0))
    for position, row in enumerate(ranked):
        baseline_rank[str(row.get("passage_id", ""))] = position

    results = []
    for name, changes in PERTURBATIONS:
        perturbed = _apply(rows, changes)
        keys, _ = _top_k(perturbed, detector, top_k)
        overlap = len(baseline_set & set(keys)) / max(len(baseline_set), 1)
        shifts = []
        ranked2 = _top_k(perturbed, detector, len(perturbed))[0]
        for position, key in enumerate(ranked2):
            if key in baseline_rank:
                shifts.append(abs(position - baseline_rank[key]))
        results.append({
            "perturbation": name,
            "top_k_overlap": round(overlap, 3),
            "mean_rank_shift": round(sum(shifts) / len(shifts), 1) if shifts else None,
        })

    results.sort(key=lambda item: item["top_k_overlap"])
    robustness = round(sum(item["top_k_overlap"] for item in results) / len(results), 3) if results else None
    health = feature_health(rows)
    usable = [item["feature"] for item in health if item["status"] == "可用"]

    return {
        "top_k": top_k,
        "vehicles": len(rows),
        "baseline_top_k": baseline_keys,
        "robustness": robustness,
        "results": results,
        "easiest_to_evade": results[0]["perturbation"] if results else None,
        "feature_health": health,
        "usable_features": usable,
        "usable_feature_count": len(usable),
    }


def main():
    parser = argparse.ArgumentParser(description="Perturbation robustness test for anomaly ranking")
    parser.add_argument("--vectors", required=True, help="Annotated vectors CSV")
    parser.add_argument("--iforest", required=True, help="Trained Isolation Forest JSON")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output", default="", help="Optional JSON output path")
    args = parser.parse_args()

    rows = load_vector_rows(args.vectors)
    detector = IsolationForestDetector.load(args.iforest)
    report = evaluate(rows, detector, top_k=args.top_k)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        directory = os.path.dirname(args.output)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        logger.info("Saved: %s", args.output)


if __name__ == "__main__":
    main()
