#!/usr/bin/env python
"""把 web/roi_editor 导出的 roi_marks.json 写进 config/camera/*.yaml。

只覆盖标记过的字段：
    - roi        -> roi.polygon
    - entry_line -> entry_line.p1/p2
    - exit_line  -> exit_line.p1/p2
其余键（calibration、anomaly、camera_name 等）原样保留。

用法：
    python scripts/import_roi_marks.py --marks roi_marks.json --config-dir config/camera
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import yaml


def apply_marks(marks: dict, config_dir: str) -> Dict[str, int]:
    stats = {"updated": 0, "created": 0, "skipped": 0}
    for item in marks.get("cameras", []):
        camera = (item.get("camera") or "").strip()
        if not camera:
            stats["skipped"] += 1
            continue
        path = os.path.join(config_dir, f"{camera}.yaml")
        existed = os.path.isfile(path)
        config = {}
        if existed:
            with open(path, encoding="utf-8") as handle:
                config = yaml.safe_load(handle) or {}

        config.setdefault("camera_id", camera)
        config.setdefault("camera_name", camera)

        roi: List = item.get("roi") or []
        if len(roi) >= 3:
            config["roi"] = {"polygon": [[int(round(x)), int(round(y))] for x, y in roi]}
        if len(item.get("entry_line") or []) == 2:
            config["entry_line"] = {
                "p1": [int(round(v)) for v in item["entry_line"][0]],
                "p2": [int(round(v)) for v in item["entry_line"][1]],
            }
        if item.get("bidirectional"):
            config["bidirectional"] = True
        if len(item.get("exit_line") or []) == 2:
            config["exit_line"] = {
                "p1": [int(round(v)) for v in item["exit_line"][0]],
                "p2": [int(round(v)) for v in item["exit_line"][1]],
            }
        # 多条线：写 entry_lines / exit_lines，并移除单条键避免重复
        for plural, single in (("entry_lines", "entry_line"), ("exit_lines", "exit_line")):
            lines = [line for line in (item.get(plural) or []) if len(line) == 2]
            if lines:
                config[plural] = [
                    {"p1": [int(round(v)) for v in line[0]],
                     "p2": [int(round(v)) for v in line[1]]}
                    for line in lines
                ]
                config.pop(single, None)
        config.setdefault("calibration", {"px_per_meter": 0.0})
        config.setdefault("anomaly", {"contamination": 0.05})

        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
        stats["updated" if existed else "created"] += 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marks", required=True)
    parser.add_argument("--config-dir", default="config/camera")
    args = parser.parse_args()

    with open(args.marks, encoding="utf-8") as handle:
        marks = json.load(handle)
    stats = apply_marks(marks, args.config_dir)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
