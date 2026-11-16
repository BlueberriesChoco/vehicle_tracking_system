#!/usr/bin/env python
"""GAE 迁移评估：基线图训练、目标图评分，并用结构注入测检出能力（指标 11）。

回答两个问题：

1. **能不能迁移**：模型在「前一段时间」的图上训练，给「后一段时间」的图打分——
   两图节点数不同（本数据前/后半段节点数一般不等），这一步不行就只能单图自监督。
2. **检出能力多少**：图异常没有真值，用**结构注入**构造已知异常节点，再算 AUC：
   - ``hub``：给节点额外连 15 条边（度远超该段正常范围）——模拟"与大量车辆建立关系"的车；
   - ``rewire``：把邻居整体换到远端节点、度数不变——模拟"换了一拨关系人"的车。

同时给出对照：``same_graph``（在目标图自己身上训练再打分，即原来的做法）。

用法：
    python scripts/eval_gae_transfer.py \
        --vectors data/outputs/e2e/alerts/vectors_admin_in.csv \
        --seeds 5 --victims 25 --output data/outputs/eval/gae_transfer.json

结论口径：AUC 只说明"注入的这一类异常能否被排序靠前"，不等于真实违规检出率；
指标 11 的门槛 85% 按 AUC 对比，达标与否逐类型列出。
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.anomaly.metrics import roc_auc
from src.graph.gae import GraphAutoencoderDetector
from src.graph.relations import RelationConfig, build_vehicle_graph, parse_time
from src.utils.logger import get_logger
from src.utils.vectors_io import load_vector_rows

logger = get_logger(__name__)

ANOMALY_KINDS = ("hub", "rewire")


def split_by_time(rows, train_ratio: float):
    """按进入时间切分，保证基线在前、目标在后（时间留出，不是随机切分）。"""
    timed = [row for row in rows if parse_time(row.get("enter_time"))]
    timed.sort(key=lambda row: parse_time(row.get("enter_time")))
    cut = max(1, int(len(timed) * train_ratio))
    return timed[:cut], timed[cut:]


def inject_anomalies(adjacency: np.ndarray, kind: str, victims, rng) -> np.ndarray:
    """对指定节点注入结构异常，返回扰动后的邻接矩阵。"""
    perturbed = np.array(adjacency, dtype=np.float64, copy=True)
    n = perturbed.shape[0]
    for victim in victims:
        if kind == "hub":
            candidates = [node for node in rng.choice(n, 15, replace=False) if node != victim]
            for node in candidates:
                perturbed[victim, node] = perturbed[node, victim] = 1.0
        elif kind == "rewire":
            for node in np.nonzero(perturbed[victim])[0]:
                perturbed[victim, node] = perturbed[node, victim] = 0.0
            candidates = [node for node in rng.choice(n, 8, replace=False) if node != victim][:3]
            for node in candidates:
                perturbed[victim, node] = perturbed[node, victim] = 1.0
        else:  # pragma: no cover
            raise ValueError(f"unknown anomaly kind: {kind}")
    return perturbed


def run_protocol(train_adjacency, target_adjacency, kind, seeds, victims, detector_kwargs):
    """跑指定协议：每个种子注入一次、训练一次、算一次 AUC。"""
    aucs = []
    for seed in range(seeds):
        rng = np.random.RandomState(seed)
        n = target_adjacency.shape[0]
        chosen = rng.choice(n, min(victims, n), replace=False)
        labels = np.zeros(n, dtype=int)
        labels[chosen] = 1
        perturbed = inject_anomalies(target_adjacency, kind, chosen, rng)

        detector = GraphAutoencoderDetector(random_state=seed, **detector_kwargs)
        detector.fit(train_adjacency)
        scores = detector.score(perturbed)["reconstruction_error"]
        value = roc_auc(labels, scores)
        if value is not None:
            aucs.append(float(value))
    return {
        "auc_mean": round(float(np.mean(aucs)), 4) if aucs else None,
        "auc_std": round(float(np.std(aucs)), 4) if aucs else None,
        "auc_per_seed": [round(value, 4) for value in aucs],
    }


def main():
    parser = argparse.ArgumentParser(description="GAE transfer evaluation (indicator 11)")
    parser.add_argument("--vectors", required=True, help="Behavior vector CSV")
    parser.add_argument("--train-ratio", type=float, default=0.5, help="Fraction of earliest passages for the baseline graph")
    parser.add_argument("--seeds", type=int, default=5, help="Random seeds per protocol")
    parser.add_argument("--victims", type=int, default=25, help="Injected anomaly nodes per run")
    parser.add_argument("--top-k", type=int, default=3, help="Graph sparsification (same as pipeline)")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    rows = [row for row in load_vector_rows(args.vectors)
            if str(row.get("analysis_target", "1")).strip() not in {"0", "false", "False"}]
    if len(rows) < 40:
        logger.error("Not enough analysis targets: %s", len(rows))
        sys.exit(1)

    baseline_rows, target_rows = split_by_time(rows, args.train_ratio)
    config = RelationConfig(top_k=args.top_k)
    baseline_graph = build_vehicle_graph(baseline_rows, config=config)
    target_graph = build_vehicle_graph(target_rows, config=config)

    logger.info(
        "Baseline graph: %s nodes / %s edges | Target graph: %s nodes / %s edges",
        baseline_graph.adjacency.shape[0], len(baseline_graph.edges),
        target_graph.adjacency.shape[0], len(target_graph.edges),
    )

    detector_kwargs = {
        "hidden_dim": args.hidden_dim,
        "latent_dim": args.latent_dim,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
    }

    results = {}
    for kind in ANOMALY_KINDS:
        results[kind] = {}
        for protocol, train_adjacency in (
            ("transfer", baseline_graph.adjacency),
            ("same_graph", target_graph.adjacency),
        ):
            results[kind][protocol] = run_protocol(
                train_adjacency, target_graph.adjacency, kind, args.seeds, args.victims, detector_kwargs
            )

    # 干净图（未注入）上的分数分布，用于说明阈值该取在哪
    clean_detector = GraphAutoencoderDetector(random_state=0, **detector_kwargs)
    clean_detector.fit(baseline_graph.adjacency)
    clean_scores = clean_detector.score(target_graph.adjacency)["reconstruction_error"]
    clean_degree = (target_graph.adjacency > 0).sum(axis=1)
    clean_stats = {
        "median": round(float(np.median(clean_scores)), 4),
        "p90": round(float(np.percentile(clean_scores, 90)), 4),
        "p99": round(float(np.percentile(clean_scores, 99)), 4),
        "max": round(float(clean_scores.max()), 4),
        "degree_correlation": round(float(np.corrcoef(clean_scores, clean_degree)[0, 1]), 4),
    }

    payload = {
        "kind": "gae_transfer_eval",
        "vectors": args.vectors,
        "train_ratio": args.train_ratio,
        "seeds": args.seeds,
        "victims": args.victims,
        "top_k": args.top_k,
        "detector": detector_kwargs,
        "baseline_graph": {
            "nodes": int(baseline_graph.adjacency.shape[0]),
            "edges": len(baseline_graph.edges),
        },
        "target_graph": {
            "nodes": int(target_graph.adjacency.shape[0]),
            "edges": len(target_graph.edges),
        },
        "anomaly_injection": {kind: results[kind] for kind in ANOMALY_KINDS},
        "clean_target_scores": clean_stats,
        "threshold_85_percent": 0.85,
    }

    if args.output:
        directory = os.path.dirname(args.output)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        logger.info("Saved %s", args.output)

    print()
    print(f"基线图 {payload['baseline_graph']['nodes']} 节点 / {payload['baseline_graph']['edges']} 边"
          f"　→　目标图 {payload['target_graph']['nodes']} 节点 / {payload['target_graph']['edges']} 边")
    print(f"{'注入类型':<10}{'协议':<12}{'AUC 均值':<12}{'标准差':<10}{'达标(≥0.85)'}")
    for kind in ANOMALY_KINDS:
        for protocol in ("transfer", "same_graph"):
            stats = results[kind][protocol]
            mean = stats["auc_mean"]
            verdict = "是" if (mean is not None and mean >= 0.85) else "否"
            label = "基线训练" if protocol == "transfer" else "同图训练"
            print(f"{kind:<12}{label:<12}{mean:<12}{stats['auc_std']:<10}{verdict}")
    print(f"\n干净目标图分数：中位 {clean_stats['median']} / P90 {clean_stats['p90']} / "
          f"P99 {clean_stats['p99']} / max {clean_stats['max']}（与度相关 r={clean_stats['degree_correlation']}）")


if __name__ == "__main__":
    main()
