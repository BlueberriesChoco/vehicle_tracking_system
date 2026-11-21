#!/usr/bin/env python
"""对比两种融合口径下「多信号」车辆的处置结果。

背景：默认权重下 A 的最大贡献只有 0.4 × 0.822 ≈ 0.33，个体异常再强也够不到
中风险线；而中风险主要靠图异常 G 撑起。于是出现一个反直觉的现象：被多个
独立信号同时命中的车，反而落在低风险。本脚本把这件事打出来，便于截图复核。

用法:
    python scripts/show_multi_signal.py --input data/outputs/e2e/alerts/vectors_admin_in.csv \
        --graph data/outputs/e2e/graphs/graph_admin_in.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.risk import RiskFusion, TimeFrequencyScorer
from src.risk.merge import apply_graph
from src.utils.vectors_io import load_vector_rows


def main():
    parser = argparse.ArgumentParser(description="多信号车辆在两种融合口径下的对比")
    parser.add_argument("--input", required=True, help="行为向量 CSV")
    parser.add_argument("--graph", required=True, help="图谱 JSON")
    parser.add_argument("--min-signals", type=int, default=2, help="至少命中几个独立信号")
    args = parser.parse_args()

    rows = load_vector_rows(args.input)
    if not rows:
        print(f"没有读到数据：{args.input}")
        return 1

    with open(args.graph, "r", encoding="utf-8") as f:
        graph = json.load(f)
    rows = apply_graph(rows, graph)
    rows = TimeFrequencyScorer().score(rows)

    fused_default = RiskFusion().fuse(rows)
    fused_corr = {
        (r.get("camera_id"), str(r.get("track_id"))): r
        for r in RiskFusion(corroboration=True).fuse(rows)
    }

    picked = []
    for row in fused_default:
        signals = RiskFusion.count_independent_signals(row)
        if signals >= args.min_signals:
            other = fused_corr.get((row.get("camera_id"), str(row.get("track_id"))), {})
            picked.append((row, signals, other))

    print(f"车辆总数: {len(rows)}    命中 ≥{args.min_signals} 个独立信号: {len(picked)}")
    print()
    print(f"  {'轨迹':<6}{'信号':>4}{'A':>7}{'G':>7}{'默认 R':>9}{'默认等级':>10}{'印证 R':>9}{'印证等级':>10}")
    for row, signals, other in sorted(picked, key=lambda item: -item[0]["risk_score_100"]):
        print(
            f"  {str(row.get('track_id')):<6}{signals:>4}"
            f"{row['risk_components']['A']:>7.2f}{row['risk_components']['G']:>7.2f}"
            f"{row['risk_score_100']:>9.1f}{row['risk_level']:>10}"
            f"{other.get('risk_score_100', 0):>9.1f}{other.get('risk_level', '-'):>10}"
        )

    low = sum(1 for row, _, _ in picked if row["risk_level"] == "low")
    print()
    print(f"结论: 默认口径下这 {len(picked)} 辆里有 {low} 辆是低风险；")
    print("      打开「多信号印证」后，它们的分数被抬到中风险线（50 分）以上。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
