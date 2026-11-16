"""统一的向量 CSV 读取工具。

多个脚本此前各自实现了一份 `load_rows` / `load_vectors`，细节（是否写
`_source_file`、是否递归 glob）已经出现分歧。这里收敛为唯一实现。
"""

from __future__ import annotations

import csv
import glob
import os
from pathlib import Path


def load_vector_rows(input_path: str, pattern: str = "vec_*.csv") -> list:
    """读取单个 CSV，或目录下所有匹配的 CSV，并给每行写入 ``_source_file``。"""
    path = Path(input_path)
    if path.is_file():
        files = [str(path)]
    else:
        files = sorted(glob.glob(os.path.join(str(input_path), "**", pattern), recursive=True))

    rows = []
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["_source_file"] = file_path
                rows.append(row)
    return rows
