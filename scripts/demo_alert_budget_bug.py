#!/usr/bin/env python
"""演示：告警预算写成"取分数最高的 N 条"会凭空造出告警。

背景：无监督模型输出的是分数，不是结论。为了把人工复核量控制在预算内，
需要从模型的告警里挑一部分留下。这里有两种写法：

  旧写法：在**全表**里取 anomaly_score 最高的 N 条，标成告警。
  现行写法：只在**模型原本就报了告警**的行里挑，且规则命中优先。

差别只在模型报得比预算少的时候暴露：旧写法凑不满 N 条，就会把分数很低、
甚至为 0 的普通车也提上来，标成"异常"却没有给任何理由。这类告警进了报告，
复核的人查不出问题，也说不清为什么被报。

用法: python scripts/demo_alert_budget_bug.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.anomaly.alert_budget import apply_alert_budget


def old_logic(rows, target_rate):
    """旧写法：全表按分数排序取前 N 条，不看模型原本报了什么。"""
    budget = max(0, int(round(len(rows) * target_rate)))
    ranked = sorted(range(len(rows)), key=lambda i: -float(rows[i]["anomaly_score"] or 0))
    keep = set(ranked[:budget])
    out = []
    for i, row in enumerate(rows):
        record = dict(row)
        record["is_anomaly"] = 1 if i in keep else 0
        out.append(record)
    return out


def make_rows():
    """10 条记录，模型只报了 2 条告警，但预算允许 3 条。"""
    return [
        {"track_id": "T01", "anomaly_score": 0.86, "is_anomaly": 1, "alert_reason": "停留 400 秒，超过 120 秒上限"},
        {"track_id": "T02", "anomaly_score": 0.71, "is_anomaly": 1, "alert_reason": ""},
        {"track_id": "T03", "anomaly_score": 0.34, "is_anomaly": 0, "alert_reason": ""},
        {"track_id": "T04", "anomaly_score": 0.22, "is_anomaly": 0, "alert_reason": ""},
        {"track_id": "T05", "anomaly_score": 0.15, "is_anomaly": 0, "alert_reason": ""},
        {"track_id": "T06", "anomaly_score": 0.09, "is_anomaly": 0, "alert_reason": ""},
        {"track_id": "T07", "anomaly_score": 0.05, "is_anomaly": 0, "alert_reason": ""},
        {"track_id": "T08", "anomaly_score": 0.02, "is_anomaly": 0, "alert_reason": ""},
        {"track_id": "T09", "anomaly_score": 0.01, "is_anomaly": 0, "alert_reason": ""},
        {"track_id": "T10", "anomaly_score": 0.00, "is_anomaly": 0, "alert_reason": ""},
    ]


def show(title, rows, note=""):
    alerts = [r for r in rows if r["is_anomaly"] == 1]
    print(f"\n{title}")
    if note:
        print(f"  {note}")
    print(f"  {'轨迹':<6}{'模型分':>7}  {'标成告警':<8}{'原判定':<8}理由")
    for row in rows:
        flag = "是" if row["is_anomaly"] == 1 else "—"
        origin = "异常" if row.get("_origin", row["is_anomaly"]) == 1 else "正常"
        reason = row.get("alert_reason") or "(无)"
        mark = "  <== 凭空造出来" if row["is_anomaly"] == 1 and row.get("_origin", 1) == 0 else ""
        print(f"  {row['track_id']:<6}{float(row['anomaly_score']):>7.2f}  {flag:<8}{origin:<8}{reason}{mark}")
    print(f"  合计告警 {len(alerts)} 条，预算允许 3 条")


def main():
    rows = make_rows()
    for row in rows:
        row["_origin"] = row["is_anomaly"]

    print("=" * 72)
    print("场景：10 辆车，模型自己只报了 2 条异常，但按 30% 的预算可以留 3 条")
    print("      （模型报得比预算少，是常态，训练集小的时候尤其常见）")

    old = old_logic(rows, target_rate=0.3)
    for row, origin in zip(old, rows):
        row["_origin"] = origin["_origin"]
    show("旧写法：全表取分数最高的 3 条", old)

    new = apply_alert_budget(rows, target_rate=0.3)
    for row, origin in zip(new, rows):
        row["_origin"] = origin["_origin"]
    show("现行写法：只在模型已报的行里挑，规则命中优先", new)

    promoted = [r for r, o in zip(old, rows) if r["is_anomaly"] == 1 and o["_origin"] == 0]
    print("\n结论")
    if promoted:
        print(f"  旧写法多报了 {len(promoted)} 条：" +
              "、".join(f"{r['track_id']}（分 {r['anomaly_score']}，无理由）" for r in promoted))
        print("  这些车分数低、没有命中任何规则，被报出来之后复核人员查不出问题。")
    print("  现行写法：告警数不超预算，且每一条都能追到模型判定或规则命中。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
