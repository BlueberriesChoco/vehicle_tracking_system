"""异常检测配置读取（``config/anomaly_config.yaml``）。

此前该配置文件没有任何代码读取，属于“装饰性配置”；这里收敛为统一入口，
供 `run_anomaly_detection.py` 与 `run_full_pipeline.py` 使用。
"""

from __future__ import annotations

import os
from typing import Optional

import yaml

DEFAULT_PATH = "config/anomaly_config.yaml"


def load_anomaly_config(path: Optional[str] = DEFAULT_PATH) -> dict:
    """读取配置；文件不存在时返回空字典。"""
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
