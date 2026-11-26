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


class _ConfigDumper(yaml.SafeDumper):
    """让 [x, y] 这样的坐标点保持单行写法，polygon 仍然是多行。"""


def _represent_list(dumper, data):
    inline = all(isinstance(item, int) for item in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=inline)


_ConfigDumper.add_representer(list, _represent_list)


EDGE_MARGIN_RATIO = 0.04


def _clamp_point(point, width, height):
    """把线端点限制在画面内侧，避免车辆刚出现/即将消失时跨线被漏判。"""
    if not width or not height:
        return point
    mx, my = width * EDGE_MARGIN_RATIO, height * EDGE_MARGIN_RATIO
    x = min(max(float(point[0]), mx), width - mx)
    y = min(max(float(point[1]), my), height - my)
    return [int(round(x)), int(round(y))]


def _clamp_line(line, width, height, counter=None):
    p1 = line.get("p1") or line[0]
    p2 = line.get("p2") or line[1]
    new_p1 = _clamp_point(p1, width, height)
    new_p2 = _clamp_point(p2, width, height)
    if counter is not None and (list(new_p1) != list(p1) or list(new_p2) != list(p2)):
        counter[0] += 1
    return {"p1": new_p1, "p2": new_p2}


def apply_marks(marks: dict, config_dir: str, sizes: Dict[str, dict] | None = None,
                clamp_edge: bool = True) -> Dict[str, int]:
    stats = {"updated": 0, "created": 0, "skipped": 0, "clamped": 0}
    for item in marks.get("cameras", []):
        camera = (item.get("camera") or "").strip()
        if not camera:
            stats["skipped"] += 1
            continue
        path = os.path.join(config_dir, f"{camera}.yaml")
        existed = os.path.isfile(path)
        config = {}
        if existed:
            _clamp_counter = [0]
        with open(path, encoding="utf-8") as handle:
                config = yaml.safe_load(handle) or {}

        config.setdefault("camera_id", camera)
        config.setdefault("camera_name", camera)

        roi: List = item.get("roi") or []
        if len(roi) >= 3:
            config["roi"] = {"polygon": [[int(round(x)), int(round(y))] for x, y in roi]}
        size = (sizes or {}).get(camera, {})
        width, height = size.get("width"), size.get("height")
        if len(item.get("entry_line") or []) == 2:
            line = {"p1": item["entry_line"][0], "p2": item["entry_line"][1]}
            if clamp_edge:
                line = _clamp_line(line, width, height, _clamp_counter)
            config["entry_line"] = line
        if item.get("bidirectional"):
            config["bidirectional"] = True
        if len(item.get("exit_line") or []) == 2:
            line = {"p1": item["exit_line"][0], "p2": item["exit_line"][1]}
            if clamp_edge:
                line = _clamp_line(line, width, height, _clamp_counter)
            config["exit_line"] = line
        # 多条线：写 entry_lines / exit_lines，并移除单条键避免重复
        for plural, single in (("entry_lines", "entry_line"), ("exit_lines", "exit_line")):
            lines = [line for line in (item.get(plural) or []) if len(line) == 2]
            if lines:
                built = [{"p1": line[0], "p2": line[1]} for line in lines]
                if clamp_edge:
                    built = [_clamp_line(line, width, height, _clamp_counter) for line in built]
                config[plural] = built
                config.pop(single, None)
        config.setdefault("calibration", {"px_per_meter": 0.0})
        config.setdefault("anomaly", {"contamination": 0.05})

        with open(path, "w", encoding="utf-8") as handle:
            yaml.dump(config, handle, Dumper=_ConfigDumper, allow_unicode=True, sort_keys=False)
        stats["updated" if existed else "created"] += 1
        stats["clamped"] += _clamp_counter[0]
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marks", required=True)
    parser.add_argument("--config-dir", default="config/camera")
    parser.add_argument("--manifest", default="data/outputs/roi_frames/manifest.json",
                        help="参考帧清单，用于把贴边的线端点往画面内收")
    parser.add_argument("--no-clamp", action="store_true", help="不自动内收贴边的线")
    args = parser.parse_args()

    with open(args.marks, encoding="utf-8") as handle:
        marks = json.load(handle)
    sizes = {}
    if os.path.isfile(args.manifest):
        with open(args.manifest, encoding="utf-8") as handle:
            for item in json.load(handle).get("cameras", []):
                sizes[item["camera"]] = {"width": item.get("width"), "height": item.get("height")}
    stats = apply_marks(marks, args.config_dir, sizes=sizes, clamp_edge=not args.no_clamp)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
