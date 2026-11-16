#!/usr/bin/env python
"""Build a one-click demo package from a full-pipeline output root.

Produces, under ``demo/``:
    risk_sample.md   - report of the highest-risk vehicle
    gang_sample.md   - densest high-anomaly community ("蚂蚁搬家" candidate)
    SUMMARY.md       - headline metrics + pointers

Usage:
    python scripts/build_demo_pack.py --root data/outputs/e2e --demo-dir demo
"""

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.report.metrics import collect_metrics, pick_top_risk, render_markdown
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _load_risk(root: str) -> dict:
    risk_dir = os.path.join(root, "risk")
    if not os.path.isdir(risk_dir):
        return {}
    for name in sorted(os.listdir(risk_dir)):
        if name.startswith("risk") and name.endswith(".json"):
            with open(os.path.join(risk_dir, name), "r", encoding="utf-8") as f:
                return json.load(f)
    return {}


def _write_risk_sample(path: str, record: dict, root: str):
    vehicle = f"{record.get('camera_id', '')}:{record.get('track_id', '')}"
    report = record.get("report_text") or "（该车辆没有生成报告文本）"
    content = "\n".join(
        [
            "# 单车风险报告样例",
            "",
            f"> 车辆：`{vehicle}`　综合风险 R = {record.get('risk_score_100', 0)} 分"
            f"（{record.get('risk_level', '')}）",
            f"> 数据来源：`{root}`　后端：`{record.get('report_backend', 'template')}`",
            "> 由 `scripts/build_demo_pack.py` 自动生成。",
            "",
            report,
            "",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _write_summary(path: str, metrics: dict, risk: dict, root: str, demo_dir: str):
    top = pick_top_risk(risk)
    content = "\n".join(
        [
            "# 演示包总览",
            "",
            "> 由 `scripts/build_demo_pack.py` 从一次完整链路输出自动生成。",
            f"> 数据来源：`{root}`",
            "",
            "## 一、关键指标",
            "",
            render_markdown(metrics).strip(),
            "",
            "## 二、本包内容",
            "",
            "| 文件 | 内容 |",
            "|---|---|",
            "| [`risk_sample.md`](risk_sample.md) | 单车风险报告（综合风险最高的那辆） |",
            "| [`gang_sample.md`](gang_sample.md) | 多车结伴社区案例（含“可能是高峰车流”的误报提示） |",
            "| [`README.md`](README.md) | 端到端演示步骤与产物说明 |",
            "",
            "## 三、怎么复现",
            "",
            "```powershell",
            "# 1) 全链路",
            ".venv\\python.exe scripts\\run_full_pipeline.py --input <录像> --camera admin_in --output data/outputs/e2e",
            "",
            "# 2) 一键打包（本目录）",
            ".venv\\python.exe scripts\\build_demo_pack.py --root data/outputs/e2e --demo-dir demo",
            "```",
            "",
            "## 四、这张表说明什么",
            "",
            "- **误报率**低于 15% 的目标（真实数据实测见上表），说明告警量在人工可复核范围内；",
            "- **证据可追溯率 100%** 表示每条证据都能回到录像分段，可作为执法参考；",
            "- **中风险/高风险**数量反映系统的“排查线索”规模，不是违规判定；",
            "- 若 **G 中位数为 0**，说明图谱层失效（键不匹配），`report_metrics.py --require-graph` 会直接报错。",
            "",
            f"最高风险车辆：`{top.get('camera_id', '')}:{top.get('track_id', '')}`，"
            f"R = {top.get('risk_score_100', 0)} 分。" if top else "",
            "",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser(description="Build the demo package")
    parser.add_argument("--root", required=True, help="Pipeline output root (e.g. data/outputs/e2e)")
    parser.add_argument("--demo-dir", default="demo")
    parser.add_argument("--benchmark", default="", help="Optional throughput JSON for the summary")
    parser.add_argument("--skip-gang", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        logger.error("Not a directory: %s", args.root)
        sys.exit(1)
    os.makedirs(args.demo_dir, exist_ok=True)

    risk = _load_risk(args.root)
    record = pick_top_risk(risk)
    if not record:
        logger.error("No risk records under %s", args.root)
        sys.exit(1)

    metrics = collect_metrics(args.root, benchmark_path=args.benchmark)

    risk_path = os.path.join(args.demo_dir, "risk_sample.md")
    _write_risk_sample(risk_path, record, args.root)
    logger.info("Risk sample written to: %s", risk_path)

    if not args.skip_gang:
        vectors = os.path.join(args.root, "alerts", "vectors_admin_in.csv")
        if not os.path.isfile(vectors):
            candidates = sorted(
                os.path.join(args.root, "alerts", name)
                for name in os.listdir(os.path.join(args.root, "alerts"))
                if name.startswith("vectors") and name.endswith(".csv")
            )
            vectors = candidates[0] if candidates else ""
        risk_json = ""
        risk_dir = os.path.join(args.root, "risk")
        for name in sorted(os.listdir(risk_dir)):
            if name.startswith("risk") and name.endswith(".json"):
                risk_json = os.path.join(risk_dir, name)
                break

        if vectors:
            command = [
                sys.executable,
                os.path.join(os.path.dirname(__file__), "export_gang_case.py"),
                "--vectors",
                vectors,
                "--risk",
                risk_json,
                "--output",
                os.path.join(args.demo_dir, "gang_sample.md"),
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning("Gang case export failed: %s", result.stderr.strip()[-300:])
            else:
                logger.info("Gang case written to: %s", os.path.join(args.demo_dir, "gang_sample.md"))

    summary_path = os.path.join(args.demo_dir, "SUMMARY.md")
    _write_summary(summary_path, metrics, risk, args.root, args.demo_dir)
    logger.info("Summary written to: %s", summary_path)


if __name__ == "__main__":
    main()
