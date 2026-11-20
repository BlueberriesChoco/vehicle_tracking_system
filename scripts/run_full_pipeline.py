#!/usr/bin/env python
"""End-to-end prototype pipeline (research points 1-4 in one command).

Input: a video file (.mp4 / .h264 / .mbf) or a directory.
Output: behavior vectors, anomaly alerts, relation graph, risk JSON + Markdown report.

Usage:
    python scripts/run_full_pipeline.py \
        --input "F:/A区录像 新/行政车道进/xxx.mbf" \
        --camera admin_in \
        --output data/outputs \
        --max-frames 6000

Steps:
    1. .mbf -> .h264 (if needed)
    2. detection / tracking / trajectory / behavior vectors
    3. Load explicitly supplied Isolation Forest and Transformer models
    4. anomaly detection (rules + baseline + A + trajectory residual)
    5. relation graph + Louvain + graph anomaly index G
    6. risk fusion R = αA + βG + γT + evidence chain + report
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.anomaly import AnomalyDetector, IsolationForestDetector, TrajectoryPredictionDetector
from src.anomaly.alert_budget import apply_alert_budget
from src.anomaly.config import load_anomaly_config
from src.graph import RelationConfig, build_graph_payload
from src.output import AlertWriter, VectorWriter
from src.pipeline.batch_processor import BatchProcessor
from src.risk import (
    EvidenceBuilder,
    LocalLLMReportGenerator,
    ReportService,
    RiskFusion,
    TimeFrequencyScorer,
)
from src.risk.merge import apply_graph
from src.utils.logger import get_logger
from src.utils.mbf_extract import extract_mbf_file

logger = get_logger(__name__)


def ensure_h264(video_path: str) -> str:
    """把 .mbf 转成 .h264；其他格式原样返回。"""
    if video_path.lower().endswith(".mbf"):
        output = video_path[:-4] + ".h264"
        if not os.path.exists(output):
            extract_mbf_file(video_path, output)
            logger.info("Extracted H.264: %s", output)
        return output
    return video_path




from src.utils.vectors_io import load_vector_rows as load_vectors


def build_graph_and_index(rows, output_path, top_k=3, gae_model=None):
    payload = build_graph_payload(rows, config=RelationConfig(top_k=top_k), gae_model=gae_model)
    if payload is None:
        return None
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def main():
    parser = argparse.ArgumentParser(description="End-to-end prototype pipeline")
    parser.add_argument("--input", required=True, help="Video file (.mp4/.h264/.mbf) or directory")
    parser.add_argument("--camera", default="admin_in", help="Camera config id")
    parser.add_argument("--config", default="config/default.yaml", help="Global config")
    parser.add_argument("--output", default="data/outputs", help="Output root")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames per video (0 = all)")
    parser.add_argument("--device", type=int, default=0, help="GPU id (-1 = CPU)")
    parser.add_argument("--retrain", action="store_true", help="Demo only: train on the scoring batch")
    parser.add_argument("--iforest", help="Pretrained Isolation Forest JSON")
    parser.add_argument("--trajectory-model", help="Pretrained Transformer checkpoint prefix")
    parser.add_argument("--gae-model", help="GAE checkpoint prefix trained on an independent baseline graph")
    parser.add_argument("--contamination", type=float, default=None,
                        help="Expected anomaly fraction; default comes from camera config, else 0.05")
    parser.add_argument("--max-alert-rate", type=float, default=0.0,
                        help="Cap alert rate (e.g. 0.15); 0 disables the alert budget")
    parser.add_argument("--top-k", type=int, default=3, help="Graph top-k neighbors")
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--beta", type=float, default=0.35)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--llm-endpoint", default=None, help="Optional local LLM endpoint")
    args = parser.parse_args()
    if args.retrain and (args.iforest or args.trajectory_model):
        parser.error("--retrain cannot be combined with pretrained individual anomaly models")

    outputs = args.output
    references = os.path.join(outputs, "references")
    os.makedirs(references, exist_ok=True)

    # 污染率优先级：命令行 > 摄像头配置 > 全局异常配置 > 0.05
    anomaly_config = load_anomaly_config()
    contamination = args.contamination
    if contamination is None:
        camera_config_path = os.path.join("config", "camera", f"{args.camera}.yaml")
        camera_config = {}
        if os.path.exists(camera_config_path):
            import yaml
            with open(camera_config_path, "r", encoding="utf-8") as f:
                camera_config = yaml.safe_load(f) or {}
        fallback = (anomaly_config.get("isolation_forest") or {}).get("contamination", 0.05)
        contamination = float((camera_config.get("anomaly") or {}).get("contamination", fallback))
    logger.info("Contamination: %s (max alert rate: %s)", contamination, args.max_alert_rate)

    # ---- 1. 收集视频 ----
    if not os.path.exists(args.input):
        # 与"目录存在但没有视频"区分开，否则会被当成单个视频文件，抛出底层解码堆栈
        logger.error("Input path does not exist: %s", args.input)
        sys.exit(2)
    if os.path.isdir(args.input):
        videos = []
        for pattern in ("*.mp4", "*.h264", "*.mbf"):
            videos.extend(glob.glob(os.path.join(args.input, pattern)))
        videos = sorted(videos)
    else:
        videos = [args.input]
    if not videos:
        logger.error("No videos found: %s", args.input)
        sys.exit(1)
    logger.info("Videos: %s", len(videos))

    # ---- 2. 向量化 ----
    processor = BatchProcessor(
        config_path=args.config,
        camera_config_path=f"config/camera/{args.camera}.yaml",
    )
    if args.max_frames > 0:
        processor.max_frames = args.max_frames
    if args.device != 0:
        processor.detector.device = args.device

    # 同一小时的多段录像会写到同一个 vec_/events_ 文件名，后一段覆盖前一段；
    # 先按 (日期, 小时) 分组，重复的才加序号后缀，单段时文件名保持不变。
    sources = [ensure_h264(video) for video in videos]
    hour_keys = [processor._parse_video_datetime(Path(source).stem) for source in sources]
    hour_counts = {}
    for key in hour_keys:
        hour_counts[key] = hour_counts.get(key, 0) + 1
    seen = {}
    rows = []
    for source, key in zip(sources, hour_keys):
        suffix = ""
        if hour_counts[key] > 1:
            seen[key] = seen.get(key, 0) + 1
            suffix = f"{seen[key]:02d}"
        vector_path = processor.process_video(video_path=source, output_dir=outputs, name_suffix=suffix)
        rows.extend(load_vectors(vector_path))
    if not rows:
        logger.error("No vectors produced.")
        sys.exit(1)

    target_rows = [row for row in rows if str(row.get("analysis_target", "")).strip() in {"1", "true", "yes"}]
    logger.info("Vectors: %s total, %s analysis targets", len(rows), len(target_rows))

    # ---- 3. 训练无监督模型 ----
    # 短片段可能凑不够训练样本，此时跳过模型、仅保留规则检测，避免整条链路失败
    can_train = args.retrain and len(target_rows) >= 3
    if args.retrain and not can_train:
        logger.warning(
            "Only %s analysis targets; skipping IF/Transformer training (rule engine only)",
            len(target_rows),
        )

    iforest_path = os.path.join(references, f"iforest_{args.camera}.json")
    traj_prefix = os.path.join(references, f"traj_{args.camera}")
    iforest = IsolationForestDetector.load(args.iforest) if args.iforest else None
    trajectory_model = TrajectoryPredictionDetector.load(args.trajectory_model) if args.trajectory_model else None
    if can_train:
        logger.warning("Demo training uses the scoring batch; results are in-sample")
        try:
            model = IsolationForestDetector(contamination=contamination).fit(target_rows)
            model.save(iforest_path)
            iforest = model
        except ValueError as exc:
            logger.warning("Isolation Forest training failed (%s); falling back to rules", exc)
        try:
            model = TrajectoryPredictionDetector(
                past_len=5, future_len=5, epochs=30, contamination=contamination
            ).fit(target_rows)
            model.save(traj_prefix)
            trajectory_model = model
        except ValueError as exc:
            logger.warning("Transformer training failed (%s); falling back to rules", exc)

    # ---- 4. 异常检测 ----
    detector = AnomalyDetector(
        iforest=iforest,
        trajectory_predictor=trajectory_model,
        thresholds=anomaly_config.get("rules"),
    )
    detected = detector.detect(rows)
    if args.max_alert_rate and args.max_alert_rate > 0:
        before = sum(1 for row in detected if str(row.get("is_anomaly", 0)) in {"1", "True", "true"})
        detected = apply_alert_budget(detected, target_rate=args.max_alert_rate)
        after = sum(1 for row in detected if str(row.get("is_anomaly", 0)) in {"1", "True", "true"})
        logger.info("Alert budget: %s -> %s (target rate %.2f)", before, after, args.max_alert_rate)
    alerts_path = os.path.join(outputs, "alerts", f"alerts_{args.camera}.csv")
    AlertWriter().write(detected, alerts_path, anomaly_only=False)
    VectorWriter().write(detected, os.path.join(outputs, "alerts", f"vectors_{args.camera}.csv"))

    # ---- 5. 关系图谱 ----
    graph_payload = build_graph_and_index(
        detected, os.path.join(outputs, "graphs", f"graph_{args.camera}.json"),
        top_k=args.top_k, gae_model=args.gae_model,
    )
    detected = apply_graph(detected, graph_payload)

    # ---- 6. 风险融合 + 报告 ----
    scored = TimeFrequencyScorer().score(detected)
    fused = RiskFusion(alpha=args.alpha, beta=args.beta, gamma=args.gamma).fuse(scored)
    enriched = EvidenceBuilder().build_many(fused)
    service = ReportService(llm=LocalLLMReportGenerator(endpoint=args.llm_endpoint))
    for record in enriched:
        report = service.generate(record, record.get("evidence", []))
        record["report_text"] = report["text"]
        record["report_backend"] = report["backend"]
    enriched.sort(key=lambda item: -float(item.get("risk_score_100", 0)))

    risk_dir = os.path.join(outputs, "risk")
    os.makedirs(risk_dir, exist_ok=True)
    with open(os.path.join(risk_dir, f"risk_{args.camera}.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "kind": "phase4_risk_report",
                "vehicle_count": len(enriched),
                "high_risk": sum(1 for r in enriched if r["risk_level"] == "high"),
                "medium_risk": sum(1 for r in enriched if r["risk_level"] == "medium"),
                "vehicles": enriched,
            },
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    with open(os.path.join(risk_dir, f"risk_{args.camera}.md"), "w", encoding="utf-8") as f:
        f.write(f"# 行政通道车辆风险排查报告（{args.camera}）\n\n")
        f.write(f"- 车辆总数：{len(enriched)}\n")
        f.write(f"- 高风险：{sum(1 for r in enriched if r['risk_level'] == 'high')}\n")
        f.write(f"- 中风险：{sum(1 for r in enriched if r['risk_level'] == 'medium')}\n\n")
        for record in enriched:
            f.write(record["report_text"] + "\n\n")

    anomaly_count = sum(1 for r in enriched if int(r.get("is_anomaly", 0) or 0) == 1)

    # ---- 8. 运行清单：给控制台/后续对比一个稳定契约 ----
    # 模型改进后会产生新的 run，旧 run 靠这份清单仍可回放，界面不会因为字段变化而返工。
    models = []
    if iforest is not None:
        models.append({"name": "iforest", "path": iforest_path})
    if trajectory_model is not None:
        models.append({"name": "trajectory", "path": traj_prefix + ".pt"})
    has_iforest = iforest is not None
    has_traj = trajectory_model is not None
    model_version = "v1-iforest+traj" if (has_iforest and has_traj) else (
        "v1-iforest-only" if has_iforest else ("v1-traj-only" if has_traj else "v1-rules-only")
    )
    manifest = {
        "run_id": os.path.basename(os.path.abspath(outputs)),
        "camera": args.camera,
        "input": os.path.abspath(args.input),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_version": model_version,
        "models": models,
        "params": {
            "contamination": contamination,
            "retrain": bool(args.retrain),
            "trained": bool(has_iforest or has_traj),
            "max_alert_rate": args.max_alert_rate,
        },
        "counts": {
            "vehicles": len(enriched),
            "analysis_targets": len(target_rows),
            "anomalies": anomaly_count,
            "graph_nodes": graph_payload["node_count"] if graph_payload else 0,
            "communities": len(graph_payload["communities"]) if graph_payload else 0,
        },
    }
    with open(os.path.join(outputs, "run.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)

    logger.info(
        "Done. vectors=%s, anomalies=%s, graph_nodes=%s, communities=%s",
        len(enriched),
        anomaly_count,
        graph_payload["node_count"] if graph_payload else 0,
        len(graph_payload["communities"]) if graph_payload else 0,
    )
    print(f"Alerts:  {alerts_path}")
    print(f"Graph:   {os.path.join(outputs, 'graphs', f'graph_{args.camera}.json')}")
    print(f"Risk:    {os.path.join(risk_dir, f'risk_{args.camera}.json')}")
    print(f"Report:  {os.path.join(risk_dir, f'risk_{args.camera}.md')}")


if __name__ == "__main__":
    main()
