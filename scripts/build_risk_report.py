#!/usr/bin/env python
"""Build multi-dimensional risk reports (research point 4).

Combines:
- A: Isolation Forest behavior anomaly index (``iforest_score``);
- G: graph anomaly index (``graph_anomaly_index`` from build_graph.py);
- T: time-frequency anomaly index (computed here);

into R = αA + βG + γT, then writes a structured evidence chain and a
human-readable Markdown report per vehicle.

Usage:
    python scripts/build_risk_report.py \
        --input data/outputs/alerts/vectors_admin_in_20260901.csv \
        --graph data/outputs/graphs/graph_admin_in_20260901.json \
        --output data/outputs/risk/risk_admin_in_20260901.json \
        --markdown data/outputs/risk/risk_admin_in_20260901.md
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.risk import (
    EvidenceBuilder,
    LocalLLMReportGenerator,
    ReportService,
    RiskFusion,
    TimeFrequencyScorer,
    TemplateReportGenerator,
)
from src.risk.merge import apply_graph, apply_stitch
from src.utils.logger import get_logger

logger = get_logger(__name__)


from src.utils.vectors_io import load_vector_rows as load_rows


def load_graph(path: str):
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_stitch(path: str):
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Build multi-dimensional risk reports")
    parser.add_argument("--input", required=True, help="Vector CSV or directory (annotated vectors preferred)")
    parser.add_argument("--graph", default=None, help="Graph JSON from build_graph.py")
    parser.add_argument("--stitch", default=None, help="Stitch JSON from run_multi_camera.py")
    parser.add_argument("--output", required=True, help="Risk report JSON path")
    parser.add_argument("--markdown", default=None, help="Risk report Markdown path")
    parser.add_argument("--alpha", type=float, default=0.4, help="Weight of A")
    parser.add_argument("--beta", type=float, default=0.35, help="Weight of G")
    parser.add_argument("--gamma", type=float, default=0.25, help="Weight of T")
    parser.add_argument("--freq-threshold", type=float, default=6.0, help="24h frequency threshold for T")
    parser.add_argument("--llm-endpoint", default=None, help="OpenAI-compatible local LLM endpoint")
    parser.add_argument("--llm-model", default="qwen2.5-7b-instruct", help="Local LLM model name")
    parser.add_argument("--top", type=int, default=0, help="Only report top-N vehicles (0 = all)")
    parser.add_argument(
        "--corroboration", action="store_true",
        help="Enable multi-signal corroboration: >=2 independent signals raise risk to at least medium",
    )
    args = parser.parse_args()

    rows = load_rows(args.input)
    if not rows:
        logger.error("No vector rows found: %s", args.input)
        sys.exit(1)

    graph = load_graph(args.graph)
    rows = apply_graph(rows, graph)

    stitch = load_stitch(args.stitch)
    rows = apply_stitch(rows, stitch)

    scorer = TimeFrequencyScorer(freq_threshold=args.freq_threshold)
    rows = scorer.score(rows)

    fusion = RiskFusion(alpha=args.alpha, beta=args.beta, gamma=args.gamma,
                        corroboration=args.corroboration)
    rows = fusion.fuse(rows)

    evidence_builder = EvidenceBuilder()
    rows = evidence_builder.build_many(rows)

    llm = LocalLLMReportGenerator(endpoint=args.llm_endpoint, model=args.llm_model)
    service = ReportService(template=TemplateReportGenerator(), llm=llm)

    # 先按风险排序并截断，再生成报告，避免对全部车辆调用大模型
    rows.sort(key=lambda item: -float(item.get("risk_score_100", 0)))
    if args.top > 0:
        rows = rows[: args.top]

    backends = {}
    for row in rows:
        report = service.generate(row, row.get("evidence", []))
        row["report_text"] = report["text"]
        row["report_backend"] = report["backend"]
        row["report_validation"] = report.get("validation", {})
        backends[report["backend"]] = backends.get(report["backend"], 0) + 1

    high = sum(1 for row in rows if row.get("risk_level") == "high")
    medium = sum(1 for row in rows if row.get("risk_level") == "medium")

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "version": 1,
                "kind": "phase4_risk_report",
                "weights": {"alpha": args.alpha, "beta": args.beta, "gamma": args.gamma},
                "vehicle_count": len(rows),
                "high_risk": high,
                "medium_risk": medium,
                "vehicles": rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    if args.markdown:
        md_dir = os.path.dirname(args.markdown)
        if md_dir:
            os.makedirs(md_dir, exist_ok=True)
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write("# 行政通道车辆风险排查报告\n\n")
            f.write(f"- 车辆总数：{len(rows)}\n")
            f.write(f"- 高风险：{high}，中风险：{medium}\n")
            f.write(f"- 融合权重：α={args.alpha}，β={args.beta}，γ={args.gamma}\n\n")
            for row in rows:
                f.write(row.get("report_text", "") + "\n\n")

    logger.info("Risk report: %s vehicles, high=%s, medium=%s", len(rows), high, medium)
    print(f"Risk JSON saved to: {args.output}")
    if args.markdown:
        print(f"Risk Markdown saved to: {args.markdown}")
    print(f"Vehicles: {len(rows)}, high risk: {high}, medium risk: {medium}")


if __name__ == "__main__":
    main()
