#!/usr/bin/env python
"""Collect the headline metrics of a full-pipeline run into one Markdown block.

Docs used to drift because the same numbers (vehicles / alert rate / risk
levels / evidence / graph stats) were copied by hand into several files.
Run this after ``run_full_pipeline.py`` and paste the output where needed.

Usage:
    python scripts/report_metrics.py --root data/outputs/e2e --output data/outputs/e2e/metrics.md
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.report.metrics import collect_metrics, graph_layer_is_dead, render_markdown
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Collect headline metrics into Markdown")
    parser.add_argument("--root", required=True, help="Pipeline output root (e.g. data/outputs/e2e)")
    parser.add_argument("--output", default="", help="Optional Markdown output path")
    parser.add_argument("--json", default="", help="Optional JSON output path")
    parser.add_argument(
        "--benchmark",
        default="",
        help="Optional benchmark_throughput.py JSON to include throughput in the summary",
    )
    parser.add_argument(
        "--require-graph",
        action="store_true",
        help="Fail when the graph anomaly median is 0 (graph layer silently dead)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        logger.error("Not a directory: %s", args.root)
        sys.exit(1)

    metrics = collect_metrics(args.root, benchmark_path=args.benchmark)
    markdown = render_markdown(metrics)
    print(markdown)

    if args.require_graph and graph_layer_is_dead(metrics):
        # G 为 0 意味着图谱层没有贡献任何信号：要么没构建，要么节点键与向量键不匹配。
        # 这种“静默失效”不会报错，只能靠断言兜住。
        graph_median = (metrics.get("component_medians") or {}).get("G")
        logger.error(
            "图谱异常指数中位数为 %s，图谱层可能失效（未构建或节点键不匹配）",
            graph_median,
        )
        sys.exit(2)

    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(markdown)
        logger.info("Metrics written to: %s", args.output)
    if args.json:
        output_dir = os.path.dirname(args.json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        logger.info("Metrics JSON written to: %s", args.json)


if __name__ == "__main__":
    main()
