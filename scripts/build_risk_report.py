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
    apply_passage_counts,
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


def _is_analysis_target(record: dict) -> bool:
    """是否为监管对象。缺字段时按"是"处理，兼容旧数据。

    注意不能写 ``record.get(...) or ""``：``analysis_target=0`` 里的 0 是 falsy，
    会被 ``or`` 吞成空串，而空串不在"非监管"集合里，于是货车被当成监管对象——
    这正是本次要修的 R3 缺陷的翻版。
    """
    value = record.get("analysis_target", None)
    if value is None or value == "":
        return True
    return str(value).strip() not in {"0", "False", "false", "0.0"}


def _split_non_target(rows):
    """拆出 analysis_target=0 的非监管对象，并给它们一个如实的状态说明。"""
    kept, excluded = [], []
    for row in rows:
        if _is_analysis_target(row):
            kept.append(row)
        else:
            record = dict(row)
            record["risk_level"] = "not_evaluated"
            record["risk_score_100"] = ""
            record["risk_score"] = ""
            record["exclusion_reason"] = "非监管对象（车型不属于行政通道普通车辆），未参与风险评分"
            record["report_text"] = (
                f"### 车辆 {record.get('camera_id', '')}:{record.get('track_id', '')}（非监管对象）\n\n"
                f"- 车型分组：{record.get('vehicle_group', '')}\n"
                f"- 说明：该车不属于行政通道普通车辆，"
                f"未进入行为基线、图谱与风险评分流程，因此没有风险分。\n"
                f"- 处理：保留通行记录，仅供流量统计；如需纳入监管，请调整车型分类规则。\n"
            )
            excluded.append(record)
    return kept, excluded


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
    parser.add_argument(
        "--freq-min-coverage", type=float, default=0.5,
        help="身份键（车牌哈希等）覆盖率低于该值时，频次统计视为不可用（默认 0.5）",
    )
    parser.add_argument(
        "--no-renormalize", action="store_true",
        help="不做分量可用性重归一化（保留固定 α/β/γ，R 上限会回到 75 分）",
    )
    parser.add_argument(
        "--include-non-target", action="store_true",
        help="把 analysis_target=0 的车辆也放进风险评分（默认分离到 non_target_vehicles）",
    )
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

    # 跨窗口通行频次：T 分量的频次那一路此前恒为 1（计数器只在单段内累加）。
    # 这里按稳定身份键重算真实次数；身份键覆盖率不足时如实标记不可用，
    # 由 RiskFusion 剔除该分量并把权重放大，而不是把"无数据"当成"无异常"。
    rows, freq_summary = apply_passage_counts(rows, min_coverage=args.freq_min_coverage)
    logger.info(
        "Passage frequency: coverage=%.2f%% level=%s distinct=%s applied=%s",
        freq_summary["coverage"] * 100,
        freq_summary["key_level"],
        freq_summary["distinct"],
        freq_summary["applied"],
    )

    scorer = TimeFrequencyScorer(freq_threshold=args.freq_threshold)
    rows = scorer.score(rows)

    fusion = RiskFusion(alpha=args.alpha, beta=args.beta, gamma=args.gamma,
                        corroboration=args.corroboration,
                        renormalize=not args.no_renormalize)
    available = fusion.available_components(rows)
    weights = fusion.effective_weights(available)
    ceiling = fusion.score_ceiling(available) if not args.no_renormalize else 100.0
    if not available.get("T"):
        logger.warning(
            "T 分量不可用（身份键覆盖率 %.2f%%，阈值 %.2f）："
            "权重已重归一化为 α=%.3f β=%.3f γ=%.3f，R 上限恢复至 %.0f 分",
            freq_summary["coverage"] * 100,
            args.freq_min_coverage,
            weights["A"], weights["G"], weights["T"], ceiling,
        )
    if ceiling < 100.0:
        logger.warning(
            "仅 %d 路分量可用（A=%s G=%s T=%s）：分数上限封顶至 %.0f 分，"
            "高风险（≥80）需至少两路独立证据",
            sum(1 for name in ("A", "G", "T") if available.get(name)),
            available.get("A"), available.get("G"), available.get("T"), ceiling,
        )
    rows = fusion.fuse(rows, weights=weights)

    evidence_builder = EvidenceBuilder()
    rows = evidence_builder.build_many(rows)

    llm = LocalLLMReportGenerator(endpoint=args.llm_endpoint, model=args.llm_model)
    service = ReportService(template=TemplateReportGenerator(), llm=llm)

    # 非监管对象（analysis_target=0，如货车/两轮车）不进风险评分体系。
    # 它们没有图谱节点、没跑 IsolationForest，A/G/T 全为 0；若留在主列表，
    # 报告会写"综合风险 0.0 分……未发现异常，可作常规抽查对象"，
    # 把"未参与评估"错述成"已评估且正常"。这里单独归档并如实标注。
    excluded = []
    if not args.include_non_target:
        kept, excluded = _split_non_target(rows)
        if excluded:
            logger.info(
                "非监管对象 %s 辆（analysis_target=0）已从风险评分中分离，"
                "单列于 non_target_vehicles", len(excluded),
            )
        rows = kept

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
                "version": 2,
                "kind": "phase4_risk_report",
                "weights": {
                    "nominal": {"alpha": args.alpha, "beta": args.beta, "gamma": args.gamma},
                    "effective": {
                        "alpha": round(weights["A"], 4),
                        "beta": round(weights["G"], 4),
                        "gamma": round(weights["T"], 4),
                        "renormalized": not args.no_renormalize,
                    },
                },
                "components_available": available,
                "score_ceiling": ceiling,
                "score_capped_count": sum(
                    1 for row in rows if row.get("risk_score_capped")
                ),
                "passage_frequency": freq_summary,
                "vehicle_count": len(rows),
                "high_risk": high,
                "medium_risk": medium,
                "non_target_count": len(excluded),
                "non_target_vehicles": excluded,
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
            f.write(f"- 监管对象：{len(rows)}\n")
            f.write(f"- 高风险：{high}，中风险：{medium}\n")
            f.write(
                f"- 融合权重（名义）：α={args.alpha}，β={args.beta}，γ={args.gamma}\n"
            )
            f.write(
                f"- 融合权重（实际）：α={weights['A']:.3f}，β={weights['G']:.3f}，"
                f"γ={weights['T']:.3f}"
            )
            if not available.get("T"):
                f.write(
                    f"　（T 分量不可用：身份键覆盖率 {freq_summary['coverage']:.1%}，"
                    f"低于阈值 {args.freq_min_coverage:.0%}。权重已重归一化，"
                    f"否则 γ 会白占 {args.gamma / (args.alpha + args.beta + args.gamma):.2f} 权重）"
                )
            f.write("\n")
            if ceiling < 100.0:
                capped_n = sum(1 for row in rows if row.get("risk_score_capped"))
                f.write(
                    f"- 分数上限：{ceiling:.0f} 分（仅 "
                    f"{sum(1 for name in ('A', 'G', 'T') if available.get(name))} "
                    f"路分量可用；高风险需至少两路独立证据）"
                )
                if capped_n:
                    f.write(f"，本批 {capped_n} 辆触顶")
                f.write("\n")
            if excluded:
                f.write(f"- 非监管对象：{len(excluded)} 辆（未参与评分，见文末）\n")
            f.write("\n")
            for row in rows:
                f.write(row.get("report_text", "") + "\n\n")
            if excluded:
                f.write("## 非监管对象（未参与风险评分）\n\n")
                f.write(
                    f"以下 {len(excluded)} 辆车的车型分组不属于行政通道普通车辆，"
                    f"未进入行为基线、图谱与风险评分流程，因此**没有风险分**。\n"
                    f"保留记录仅供流量统计，不应据此判断其是否正常。\n\n"
                )
                for row in excluded:
                    f.write(row.get("report_text", "") + "\n\n")

    logger.info(
        "Risk report: %s vehicles, high=%s, medium=%s, non_target=%s",
        len(rows), high, medium, len(excluded),
    )
    print(f"Risk JSON saved to: {args.output}")
    if args.markdown:
        print(f"Risk Markdown saved to: {args.markdown}")
    print(f"Vehicles: {len(rows)}, high risk: {high}, medium risk: {medium}, non-target: {len(excluded)}")


if __name__ == "__main__":
    main()
