#!/usr/bin/env python
"""清空所有机位配置里的 ROI，并把人工标定结果备份一份。

用途：交由人工在 web/roi_editor 里重新圈定 ROI 之前，先把自动推导的结果清掉，
避免标定工具里显示旧 ROI 干扰判断。

只删除 `roi` 字段，其余键（entry_line / exit_line / calibration / anomaly）原样保留。

用法：
    python scripts/reset_camera_rois.py                # 备份 + 清空
    python scripts/reset_camera_rois.py --dry-run      # 只看会改哪些文件
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from glob import glob

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default="config/camera")
    parser.add_argument("--backup-dir", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = args.backup_dir or os.path.join("data/work", f"camera_configs_backup_{stamp}")
    files = sorted(glob(os.path.join(args.config_dir, "*.yaml")))

    touched, skipped = [], []
    for path in files:
        with open(path, encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        if "roi" not in config:
            skipped.append(os.path.basename(path))
            continue
        touched.append(os.path.basename(path))
        if args.dry_run:
            continue
        os.makedirs(backup_dir, exist_ok=True)
        shutil.copy2(path, os.path.join(backup_dir, os.path.basename(path)))
        config.pop("roi", None)
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)

    summary = {
        "configs": len(files),
        "cleared": len(touched),
        "already_empty": len(skipped),
        "backup_dir": None if args.dry_run else backup_dir,
        "dry_run": args.dry_run,
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
