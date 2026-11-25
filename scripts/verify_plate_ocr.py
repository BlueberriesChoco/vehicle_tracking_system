#!/usr/bin/env python
"""车牌识别模块验证：分别测量「检测」「OCR」「端到端」三个环节。

为什么必须分开测：端到端识别率低，可能是检测没找到车牌，也可能是检测对了但
OCR 读错。两者的修法完全不同——前者要扩车牌模型，后者要换 OCR 引擎。
只看端到端总数会得出"车牌识别不行"这种无用结论。

用法：
    # 全部环节（需要 <crop>/*.jpg 与 <crop>/*_plate.jpg 两类产物）
    python scripts/verify_plate_ocr.py --crops data/outputs/e2e_acceptance_demo/crops

    # 只测 OCR 上限（只依赖 *_plate.jpg）
    python scripts/verify_plate_ocr.py --crops <dir> --mode ocr

    # 只测端到端（只依赖车辆裁剪）
    python scripts/verify_plate_ocr.py --crops <dir> --mode e2e --limit 50

产物：控制台摘要 + 可选 JSON（--output-json）。
"""

import argparse
import collections
import glob
import json
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.feature_extraction.plate_detector import PlateDetector
from src.feature_extraction.plate_ocr import PlateOCR, hyperlpr_model_dir

#: 新能源车牌（8 位，省份+字母+D/F+5 位）。用于区分格式命中率。
NEW_ENERGY = re.compile(
    r"[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤川青藏琼宁][A-HJ-NP-Z][DF][A-HJ-NP-Z0-9][0-9]{4}"
)


def percentiles(values):
    if not values:
        return {}
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "min": round(float(arr.min()), 3),
        "p50": round(float(np.percentile(arr, 50)), 3),
        "p90": round(float(np.percentile(arr, 90)), 3),
        "max": round(float(arr.max()), 3),
        "mean": round(float(arr.mean()), 3),
    }


def collect_pairs(crops_dir):
    """返回 [(passage_id, 车辆裁剪路径, 车牌裁剪路径或 None)]。

    车牌裁剪缺失是常态（检测失败就不落盘），不能因此跳过该车，
    否则端到端通过率会被系统性高估。
    """
    vehicles = {}
    for path in sorted(glob.glob(os.path.join(crops_dir, "*.jpg"))):
        name = os.path.basename(path)
        if name.endswith("_plate.jpg"):
            pid = name[: -len("_plate.jpg")]
            vehicles.setdefault(pid, {})["plate"] = path
        else:
            pid = name[: -len(".jpg")]
            vehicles.setdefault(pid, {})["vehicle"] = path
    return [
        (pid, item.get("vehicle"), item.get("plate"))
        for pid, item in sorted(vehicles.items())
        if item.get("vehicle")
    ]


def main():
    parser = argparse.ArgumentParser(description="Verify plate detection/OCR stages")
    parser.add_argument("--crops", required=True, help="车辆裁剪目录")
    parser.add_argument("--mode", choices=["all", "ocr", "e2e"], default="all")
    parser.add_argument("--limit", type=int, default=0, help="仅处理前 N 辆（0 = 全部）")
    parser.add_argument("--min-conf", type=float, default=0.7, help="OCR 采纳阈值")
    parser.add_argument("--min-plate-width", type=int, default=24, help="车牌最小像素宽度")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    model_dir = hyperlpr_model_dir()
    model_ready = model_dir.is_dir() and any(model_dir.glob("*.onnx"))
    ocr = PlateOCR(engine="hyperlpr3", min_confidence=args.min_conf)
    detector = PlateDetector(min_plate_width=args.min_plate_width)

    print("=" * 62)
    print(f"模型目录      : {model_dir}")
    print(f"模型文件      : {'就绪' if model_ready else '缺失'}")
    print(f"OCR 引擎      : {ocr.engine}{'' if ocr.enabled else '（不可用）'}")
    print(f"专用车牌检测器: {'YOLO' if detector.use_yolo else '形态学回退（models/license_plate_yolov8.pt 缺失）'}")
    print(f"采纳阈值      : {args.min_conf}")
    print("=" * 62)

    pairs = collect_pairs(args.crops)
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"车辆数: {len(pairs)}\n")

    ocr_stats = {"total": 0, "ok": 0, "empty": 0, "conf_reject": 0, "scores": [], "formats": collections.Counter(),
                 "failures": []}
    e2e_stats = {"total": 0, "detected": 0, "ok": 0, "no_plate_crop": 0, "scores": [],
                 "failures": []}

    for index, (pid, vehicle_path, plate_path) in enumerate(pairs):
        # ---- OCR 环节：给定精确车牌区域，测识别上限 ----
        if args.mode in ("all", "ocr"):
            ocr_stats["total"] += 1
            if plate_path is None:
                ocr_stats["empty"] += 1
                if len(ocr_stats["failures"]) < 8:
                    ocr_stats["failures"].append((pid, "无车牌裁剪（检测环节未产出）"))
            else:
                image = cv2.imread(plate_path)
                plate, conf = ocr.recognize(image) if image is not None else ("", 0.0)
                if plate:
                    ocr_stats["ok"] += 1
                    ocr_stats["scores"].append(conf)
                    ocr_stats["formats"]["新能源" if NEW_ENERGY.fullmatch(plate) else "普通"] += 1
                else:
                    if conf >= args.min_conf:
                        ocr_stats["empty"] += 1
                        reason = "格式校验不通过"
                    else:
                        ocr_stats["conf_reject"] += 1
                        reason = f"置信度 {conf:.3f} < {args.min_conf}"
                    if len(ocr_stats["failures"]) < 8:
                        ocr_stats["failures"].append((pid, reason))

        # ---- 端到端：车辆裁剪 → 检测 → OCR ----
        if args.mode in ("all", "e2e"):
            e2e_stats["total"] += 1
            vehicle = cv2.imread(vehicle_path)
            if vehicle is None:
                continue
            found = detector.detect(vehicle)
            if found is None:
                e2e_stats["no_plate_crop"] += 1
                if len(e2e_stats["failures"]) < 8:
                    e2e_stats["failures"].append((pid, "检测器未定位到车牌"))
            else:
                e2e_stats["detected"] += 1
                plate, conf = ocr.recognize(found[0])
                if plate:
                    e2e_stats["ok"] += 1
                    e2e_stats["scores"].append(conf)
                elif len(e2e_stats["failures"]) < 8:
                    e2e_stats["failures"].append((pid, f"检测到但 OCR 失败（conf={conf:.3f}）"))

        if (index + 1) % 50 == 0:
            print(f"  已处理 {index + 1}/{len(pairs)}", flush=True)

    payload = {"crops": args.crops, "model_dir": str(model_dir), "model_ready": model_ready,
               "engine": ocr.engine, "min_conf": args.min_conf}

    if args.mode in ("all", "ocr"):
        rate = ocr_stats["ok"] / ocr_stats["total"] * 100 if ocr_stats["total"] else 0.0
        print("\n[OCR 环节] 给定精确车牌区域")
        print(f"  样本        : {ocr_stats['total']}")
        print(f"  识别成功    : {ocr_stats['ok']} ({rate:.1f}%)")
        print(f"  置信度不足  : {ocr_stats['conf_reject']}")
        print(f"  无车牌裁剪  : {ocr_stats['empty']}")
        print(f"  置信度分布  : {percentiles(ocr_stats['scores'])}")
        print(f"  格式分布    : {dict(ocr_stats['formats'])}")
        for pid, reason in ocr_stats["failures"][:5]:
            print(f"    - {pid}: {reason}")
        payload["ocr"] = {"total": ocr_stats["total"], "ok": ocr_stats["ok"], "success_rate": round(rate, 2),
                          "conf_reject": ocr_stats["conf_reject"], "no_plate_crop": ocr_stats["empty"],
                          "confidence": percentiles(ocr_stats["scores"]),
                          "formats": dict(ocr_stats["formats"])}

    if args.mode in ("all", "e2e"):
        rate = e2e_stats["ok"] / e2e_stats["total"] * 100 if e2e_stats["total"] else 0.0
        det_rate = e2e_stats["detected"] / e2e_stats["total"] * 100 if e2e_stats["total"] else 0.0
        print("\n[端到端] 车辆裁剪 → 检测 → OCR")
        print(f"  样本        : {e2e_stats['total']}")
        print(f"  检测到车牌  : {e2e_stats['detected']} ({det_rate:.1f}%)")
        print(f"  最终识别成功: {e2e_stats['ok']} ({rate:.1f}%)")
        print(f"  置信度分布  : {percentiles(e2e_stats['scores'])}")
        for pid, reason in e2e_stats["failures"][:5]:
            print(f"    - {pid}: {reason}")
        payload["e2e"] = {"total": e2e_stats["total"], "detected": e2e_stats["detected"],
                          "detection_rate": round(det_rate, 2), "ok": e2e_stats["ok"],
                          "success_rate": round(rate, 2),
                          "confidence": percentiles(e2e_stats["scores"])}

    if args.output_json:
        out_dir = os.path.dirname(args.output_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n已写入: {args.output_json}")


if __name__ == "__main__":
    main()
