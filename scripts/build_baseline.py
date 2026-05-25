#!/usr/bin/env python
"""正常行为基线构建脚本。

收集多日的正常通行行为向量，构建统计基线：
  1. 按摄像头聚合历史行为向量
  2. 拟合 Z-score 标准化参数
  3. 训练孤立森林（Isolation Forest）异常检测模型
  4. 输出基线文件到 data/references/

用法:
    python scripts/build_baseline.py \\
        --vector-dir data/outputs/vectors/cam01/20260520 \\
        --output data/references/baseline_cam01

    python scripts/build_baseline.py \\
        --vector-dir data/outputs/vectors/cam01 \\
        --days 7 \\
        --output data/references/baseline_cam01
"""

import argparse
import sys
import os
import glob
import csv
import json
from typing import List
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.behavior.feature_normalizer import FeatureNormalizer
from src.utils.logger import get_logger

logger = get_logger(__name__)

FEATURE_COLUMNS = [
    "avg_speed_ms", "max_speed_ms", "speed_variance",
    "max_dwell_sec", "stop_count", "dwell_ratio",
    "path_deviation", "path_smoothness",
    "is_night", "night_ratio",
    "freq_index", "aggregation_index",
]


def load_vectors_from_dir(vector_dir: str, recursive: bool = True) -> List[dict]:
    """从一个目录加载所有行为向量 CSV。"""
    pattern = os.path.join(vector_dir, "**", "vec_*.csv") if recursive else os.path.join(vector_dir, "vec_*.csv")
    files = sorted(glob.glob(pattern, recursive=recursive))

    all_vectors = []
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_vectors.append(row)

    logger.info(f"Loaded {len(all_vectors)} vectors from {len(files)} files")
    return all_vectors


def extract_numeric_matrix(vectors: List[dict]) -> np.ndarray:
    """从向量字典列表提取数值特征矩阵 (N, D)。"""
    rows = []
    for v in vectors:
        row = [float(v.get(col, 0.0)) for col in FEATURE_COLUMNS]
        rows.append(row)
    return np.array(rows, dtype=np.float64)


def compute_baseline_stats(matrix: np.ndarray) -> dict:
    """计算基线统计量（均值、标准差、分位数）。"""
    stats = {}
    for i, col in enumerate(FEATURE_COLUMNS):
        col_data = matrix[:, i]
        stats[col] = {
            "mean": float(np.mean(col_data)),
            "std": float(np.std(col_data)),
            "median": float(np.median(col_data)),
            "q1": float(np.percentile(col_data, 25)),
            "q3": float(np.percentile(col_data, 75)),
            "min": float(np.min(col_data)),
            "max": float(np.max(col_data)),
        }
    return stats


def save_baseline(
    output_prefix: str,
    normalizer: FeatureNormalizer,
    stats: dict,
    model=None,
):
    """保存基线文件。"""
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

    # 1. 标准化参数
    normalizer.save(f"{output_prefix}_normalizer.json")

    # 2. 统计量
    with open(f"{output_prefix}_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # 3. 孤立森林模型（Phase 3）
    if model is not None:
        import joblib
        joblib.dump(model, f"{output_prefix}_isolation_forest.pkl")

    logger.info(f"Baseline saved to: {output_prefix}_*")


def main():
    parser = argparse.ArgumentParser(description="Build normal behavior baseline")
    parser.add_argument("--vector-dir", required=True, help="Directory containing vector CSVs")
    parser.add_argument("--output", required=True, help="Output prefix for baseline files")
    parser.add_argument("--days", type=int, default=7, help="Number of days to use for baseline")
    args = parser.parse_args()

    # 加载向量
    all_vectors = load_vectors_from_dir(args.vector_dir)

    if not all_vectors:
        logger.error("No vectors found. Run pipeline first.")
        sys.exit(1)

    # 过滤异常数据（如果有标注）→ 只留正常通行
    normal_vectors = [v for v in all_vectors if v.get("is_anomaly", "0") == "0"]
    logger.info(f"Normal vectors: {len(normal_vectors)} / {len(all_vectors)}")

    # 提取数值矩阵
    matrix = extract_numeric_matrix(normal_vectors if normal_vectors else all_vectors)

    # 1. 计算统计基线
    stats = compute_baseline_stats(matrix)
    logger.info("=== Baseline Statistics ===")
    for col, s in stats.items():
        logger.info(f"  {col}: mean={s['mean']:.3f}, std={s['std']:.3f}")

    # 2. 拟合标准化器
    normalizer = FeatureNormalizer(method="zscore")
    vectors_list = [matrix[i] for i in range(matrix.shape[0])]
    normalizer.fit(vectors_list)

    # 3. 训练孤立森林 (Phase 3: 可选)
    model = None
    try:
        from sklearn.ensemble import IsolationForest
        model = IsolationForest(
            n_estimators=100,
            contamination=0.05,  # 预期 5% 异常率
            random_state=42,
        )
        model.fit(matrix)
        logger.info("Isolation Forest model trained.")
    except ImportError:
        logger.info("scikit-learn not available, skipping Isolation Forest.")

    # 保存
    save_baseline(args.output, normalizer, stats, model)

    print(f"\nBaseline saved to: {args.output}_*")
    print(f"  - {args.output}_normalizer.json")
    print(f"  - {args.output}_stats.json")
    if model:
        print(f"  - {args.output}_isolation_forest.pkl")


if __name__ == "__main__":
    main()
