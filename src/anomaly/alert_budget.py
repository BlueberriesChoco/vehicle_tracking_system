"""告警预算：把告警数量限制在目标比例以内。

无监督模型在分布漂移时会产出超过目标的告警率。实际监管场景里通常用
"告警预算"控制人工复核量：按异常分数排序，只保留预算内的告警；规则
命中（确定性证据）优先保留，不参与裁撤。
"""

from __future__ import annotations

from typing import List


def _flag(value) -> bool:
    return str(value).strip() in {"1", "True", "true", "yes"}


def apply_alert_budget(
    rows: List[dict],
    target_rate: float = 0.15,
    score_field: str = "anomaly_score",
    reason_field: str = "alert_reason",
) -> List[dict]:
    """把 ``is_anomaly`` 的数量限制在 ``target_rate * len(rows)`` 以内。

    - 规则命中行优先保留；
    - 其余按 ``score_field`` 降序补足预算；
    - 被裁撤的行标记 ``alert_capped=1``，原始模型标记（iforest/traj）保持不变，
      便于事后分析"模型本来报了但被预算压掉"的情况。
    """
    if not rows:
        return rows

    total = len(rows)
    budget = max(0, int(round(total * target_rate)))

    rule_indices = [
        index for index, row in enumerate(rows)
        if str(row.get(reason_field, "") or "").strip()
    ]
    scored_indices = sorted(
        range(total),
        key=lambda index: float(rows[index].get(score_field, 0) or 0),
        reverse=True,
    )

    keep = set()
    for index in rule_indices:
        if len(keep) >= budget:
            break
        keep.add(index)
    for index in scored_indices:
        if len(keep) >= budget:
            break
        keep.add(index)

    result = []
    for index, row in enumerate(rows):
        record = dict(row)
        was_alert = _flag(row.get("is_anomaly", 0))
        record["is_anomaly"] = 1 if index in keep else 0
        record["alert_capped"] = 1 if (was_alert and index not in keep) else 0
        result.append(record)
    return result


def choose_contamination(trials: List[dict], target_rate: float):
    """从标定结果中选告警率不超过目标的最大污染率。

    没有满足目标的 trial 时，退化为“最接近目标告警率”的那一项。
    """
    if not trials:
        return None
    within = [trial for trial in trials if trial.get("alert_rate", 1.0) <= target_rate]
    if within:
        return max(within, key=lambda trial: trial.get("contamination", 0.0))
    return min(trials, key=lambda trial: abs(trial.get("alert_rate", 1.0) - target_rate))
