#!/usr/bin/env python
"""检查行为向量表的列质量：找出整列为空、整列为零的列。

为什么需要它：训练异常检测之前，如果某一列整列为空（字段没取到）或整列为零
（机位没标定、开关没打开），模型会把它当成"有效信号"。整列为零的列方差为 0，
任何偏离都会被放大；整列为空的列会被中位数填补成一个常数，等于给所有车加了
同一个偏移。两种情况的共同后果是：告警看起来很有道理，实际全是假的。

用法:
    python scripts/audit_vector_columns.py --dir data/outputs/e2e/vectors
    python scripts/audit_vector_columns.py --dir data/outputs/e2e/vectors --verbose
"""

import argparse
import csv
import glob
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 这些字段是文本型，全空属于"本次没启用"，不算数据缺陷
TEXT_FIELDS = {"plate_number", "plate_hash", "alert_reason", "trajectory_points"}

# 检测阶段才写入的字段：向量表刚产出时它们必然是空的，不是缺陷
DERIVED_FIELDS = {
    "anomaly_score",
    "is_anomaly",
    "iforest_score",
    "iforest_anomaly",
    "traj_residual",
    "traj_anomaly",
    "rule_hits",
}

# 可靠性开关：整列为 0 表示该机位没做几何标定，对应的速度/路径/聚集特征不可用
RELIABILITY_FIELDS = {"speed_reliable", "path_reliable", "aggregation_reliable", "passage_reliable"}

# 这几列整列为 0 通常有合理解释（白天时段、这批车都没停），需要人工确认而不是直接判为缺陷
CONTEXTUAL_ZERO = {
    "is_night": "这批录像可能在白天",
    "night_ratio": "同 is_night",
    "max_dwell_sec": "可能确实没有车停留",
    "stop_count": "同上",
    "dwell_ratio": "同上",
    "freq_index": "频次特征可能未计算",
}


def load_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def audit(rows, verbose=False):
    if not rows:
        return None
    total = len(rows)
    columns = list(rows[0].keys())

    empty_cols, zero_cols, detail = [], [], []
    for col in columns:
        values = [(row.get(col) or "").strip() for row in rows]
        n_empty = sum(1 for v in values if v == "")
        n_zero = sum(1 for v in values if v not in ("",) and _is_zero(v))
        sample = next((v for v in values if v != ""), "")
        detail.append((col, total - n_empty, n_zero, sample))

        if n_empty == total:
            empty_cols.append((col, n_empty))
        elif n_zero == total - n_empty:
            zero_cols.append((col, n_zero))

    return {
        "total": total,
        "columns": columns,
        "empty": empty_cols,
        "zero": zero_cols,
        "detail": detail,
    }


def _is_zero(value):
    try:
        return float(value) == 0.0
    except (TypeError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser(description="检查行为向量表的列质量")
    parser.add_argument("--dir", required=True, help="包含 vec_*.csv 的目录")
    parser.add_argument("--verbose", action="store_true", help="打印逐列明细")
    parser.add_argument("--max-files", type=int, default=3, help="最多检查几个文件")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, "**", "vec_*.csv"), recursive=True))
    if not paths:
        print(f"没找到 vec_*.csv：{args.dir}")
        return 1

    for path in paths[: args.max_files]:
        rows = load_rows(path)
        result = audit(rows, args.verbose)
        if result is None:
            print(f"{path}: 空文件，跳过")
            continue

        print("=" * 72)
        print(f"文件: {os.path.relpath(path)}")
        print(f"行数: {result['total']}    列数: {len(result['columns'])}")

        blank_defect = [c for c in result["empty"] if c[0] not in TEXT_FIELDS and c[0] not in DERIVED_FIELDS]
        blank_derived = [c for c in result["empty"] if c[0] in DERIVED_FIELDS]
        blank_text = [c for c in result["empty"] if c[0] in TEXT_FIELDS]

        zero_reliability = [c for c in result["zero"] if c[0] in RELIABILITY_FIELDS]
        zero_context = [c for c in result["zero"] if c[0] in CONTEXTUAL_ZERO]
        zero_derived = [c for c in result["zero"] if c[0] in DERIVED_FIELDS]
        zero_other = [
            c for c in result["zero"]
            if c[0] not in RELIABILITY_FIELDS and c[0] not in CONTEXTUAL_ZERO and c[0] not in DERIVED_FIELDS
        ]

        def line(col, n, note=""):
            print(f"  {col:<26} {n}/{result['total']}  {note}")

        if blank_defect:
            print("\n[整列为空·缺陷] 上游没产出，喂进模型就是假装有数")
            for col, n in blank_defect:
                line(col, n, "字段没取到")
        if zero_reliability:
            print("\n[整列为零·机位未标定] 这三列是可靠性开关，为 0 说明速度/路径/聚集特征不可用")
            for col, n in zero_reliability:
                line(col, n, "开关为 0")
        if zero_other:
            print("\n[整列为零·需人工确认] 可能是真实取值，也可能是没算")
            for col, n in zero_other:
                line(col, n, "方差为 0")
        if zero_context:
            print("\n[整列为零·大概率正常] 结合录像时段判断，不算缺陷")
            for col, n in zero_context:
                line(col, n, CONTEXTUAL_ZERO[col])
        if blank_derived or zero_derived:
            print("\n[检测阶段字段] 这批还没跑异常检测，空着是正常的")
            for col, n in blank_derived + zero_derived:
                line(col, n, "未运行检测")
        if blank_text:
            print("\n[文本字段] 本次未启用")
            for col, n in blank_text:
                line(col, n, "文本字段")

        if args.verbose:
            print("\n[逐列明细]")
            print(f"  {'列名':<26}{'非空':>6}{'零值':>6}  示例")
            for col, n_valid, n_zero, sample in result["detail"]:
                shown = sample[:24] if sample else "(空)"
                print(f"  {col:<26}{n_valid:>6}{n_zero:>6}  {shown}")

        problems = len(blank_defect) + len(zero_reliability)
        if problems:
            print(f"\n结论: {len(blank_defect)} 列上游未产出，{len(zero_reliability)} 列机位未标定。"
                  f"这两类必须先处理，否则告警不可信。")
        else:
            print("\n结论: 未发现上游缺列或未标定的问题列。")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
