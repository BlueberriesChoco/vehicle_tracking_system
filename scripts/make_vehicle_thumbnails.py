#!/usr/bin/env python
"""为一次 run 里的每辆车截一张「行政通道截图」，供控制台详情页展示。

流水线里 `best_crop` 只存在内存（用于 Re-ID/颜色），跑完就丢了。本脚本按
风险报告里的 `enter_time` 回到源视频取帧、检测并裁出车辆，落盘到
`<run>/crops/<passage_id>.jpg`，同时写 `index.json`。

做法：
- 车辆按"取图时刻"排序，视频只顺序解码一遍（用 grab 跳过非目标帧）；
- 取图时刻取轨迹中段（enter + 0.6 × 全程），此时车辆离镜头更近、画面更完整；
- 在该帧上检测，挑中心最接近轨迹点的框；匹配不到就退化为以轨迹点为中心的固定裁剪。

用法：
    python scripts/make_vehicle_thumbnails.py \
        --run data/outputs/e2e_acceptance_demo \
        --video data/outputs/h264/admin_in/Download_....h264 \
        --width 360
"""

import argparse
import json
import os
import sys
from datetime import datetime

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.detection.detector import VehicleDetector
from src.utils.logger import get_logger

logger = get_logger(__name__)

MATCH_RADIUS_PX = 300      # 检测框中心与轨迹点的最大匹配距离
FALLBACK_BOX = (320, 220)  # 匹配不到时的固定裁剪尺寸
PADDING = 0.12             # 裁剪时向外扩的比例


def _parse_time(value):
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _trajectory_points(value):
    points = []
    for token in str(value or "").split(";"):
        token = token.strip()
        if not token or "," not in token:
            continue
        try:
            x, y = token.split(",", 1)
            points.append((float(x), float(y)))
        except ValueError:
            continue
    return points


def _video_start(video_path):
    """从文件名（…_20260901080110_20260901090712）解析起始时间。

    文件名里有两个时间戳（起、止），必须取**较早**的那个——之前按反序取到了结束时间，
    导致所有车辆的帧号算成负数、一张图都出不来。
    """
    stem = os.path.splitext(os.path.basename(video_path))[0]
    stamps = []
    for token in stem.split("_"):
        if len(token) == 14 and token.isdigit():
            try:
                stamps.append(datetime.strptime(token, "%Y%m%d%H%M%S"))
            except ValueError:
                continue
    return min(stamps) if stamps else None


def plan_targets(vehicles, start_time, fps):
    """算出每辆车的取图帧号与目标点。"""
    targets = []
    for record in vehicles:
        enter = _parse_time(record.get("enter_time"))
        exit_ = _parse_time(record.get("exit_time"))
        points = _trajectory_points(record.get("trajectory_points"))
        if enter is None or start_time is None or not points:
            continue
        span = (exit_ - enter).total_seconds() if exit_ else 0.0
        target_time = enter.timestamp() + max(0.0, span) * 0.6
        frame_index = int(round((target_time - start_time.timestamp()) * fps))
        if frame_index < 0:
            continue
        point = points[min(len(points) - 1, int(len(points) * 0.6))]
        targets.append(
            {
                "passage_id": str(record.get("passage_id", "")),
                "frame": frame_index,
                "point": point,
                "enter_time": record.get("enter_time"),
            }
        )
    targets.sort(key=lambda item: item["frame"])
    return targets


def crop_vehicle(frame, detections, point):
    """优先用检测框，匹配不到就退回固定尺寸裁剪。"""
    height, width = frame.shape[:2]
    best = None
    best_distance = MATCH_RADIUS_PX
    for det in detections or []:
        bbox = det.get("bbox") or []
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        distance = ((cx - point[0]) ** 2 + (cy - point[1]) ** 2) ** 0.5
        if distance < best_distance:
            best_distance = distance
            best = (x1, y1, x2, y2)

    if best is None:
        box_w, box_h = FALLBACK_BOX
        cx, cy = point
        best = (cx - box_w / 2, cy - box_h / 2, cx + box_w / 2, cy + box_h / 2)

    x1, y1, x2, y2 = best
    pad_w = (x2 - x1) * PADDING
    pad_h = (y2 - y1) * PADDING
    x1 = max(0, int(x1 - pad_w))
    y1 = max(0, int(y1 - pad_h))
    x2 = min(width, int(x2 + pad_w))
    y2 = min(height, int(y2 + pad_h))
    if x2 <= x1 or y2 <= y1:
        return None, best
    return frame[y1:y2, x1:x2].copy(), best


def save_plate_strip(crop, passage_id, crops_dir, reader=None):
    """从原始分辨率车辆裁剪里截出车牌条并放大，供人工读取。

    为什么不用车牌检测器：`models/license_plate_yolov8.pt` 未提供，形态学兜底在
    这条车道上会把草丛误检成车牌。这里按车辆下部固定比例取条，宁可多带一点背景，
    也要保证车牌完整落在画面里——**给人看的截图不需要精确框**。

    OCR 只作提示：1080p 下车牌高度约 60px，easyocr 会把 D 读成 0、Z 读成 2，
    所以结果低置信度时界面只展示、不当作车牌号使用。
    """
    height, width = crop.shape[:2]
    band = crop[int(height * 0.66): int(height * 0.96), int(width * 0.18): int(width * 0.72)]
    if band.size == 0:
        return None, "", 0.0
    big = cv2.resize(band, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    plate_name = f"{passage_id}_plate.jpg"
    ok, buffer = cv2.imencode(".jpg", big, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if ok:
        buffer.tofile(os.path.join(crops_dir, plate_name))

    text, conf = "", 0.0
    if reader is not None:
        try:
            for _, detected, score in reader.readtext(big):
                cleaned = "".join(ch for ch in str(detected).upper() if ch.isalnum())
                if 5 <= len(cleaned) <= 8 and float(score) >= 0.25:
                    if float(score) > conf:
                        text, conf = cleaned, float(score)
        except Exception:
            text, conf = "", 0.0
    return plate_name, text, round(conf, 3)


def main():
    parser = argparse.ArgumentParser(description="Extract per-vehicle thumbnails from the source video")
    parser.add_argument("--run", required=True, help="Run directory containing risk/risk_*.json")
    parser.add_argument("--video", default="", help="Source video (default: read from run.json 'input')")
    parser.add_argument("--width", type=int, default=360, help="Output image width in pixels")
    parser.add_argument("--max", type=int, default=0, help="Only process the first N vehicles (0 = all)")
    parser.add_argument("--no-plate", action="store_true", help="Skip the enlarged plate strip")
    parser.add_argument("--ocr", action="store_true", help="Also run easyocr on the plate strip (slow, low accuracy)")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run)
    risk_files = sorted(
        os.path.join(run_dir, "risk", name)
        for name in os.listdir(os.path.join(run_dir, "risk"))
        if name.startswith("risk_") and name.endswith(".json")
    ) if os.path.isdir(os.path.join(run_dir, "risk")) else []
    if not risk_files:
        logger.error("找不到风险报告：%s/risk/risk_*.json", run_dir)
        sys.exit(2)

    with open(risk_files[0], encoding="utf-8") as handle:
        vehicles = json.load(handle).get("vehicles", [])

    video = args.video
    if not video:
        manifest_path = os.path.join(run_dir, "run.json")
        if os.path.isfile(manifest_path):
            with open(manifest_path, encoding="utf-8") as handle:
                video = json.load(handle).get("input", "")
    if not video or not os.path.isfile(video):
        logger.error("找不到源视频，请用 --video 指定")
        sys.exit(2)

    start_time = _video_start(video)
    if start_time is None:
        logger.error("文件名里没有 14 位时间戳，无法对齐帧号：%s", os.path.basename(video))
        sys.exit(2)

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    targets = plan_targets(vehicles, start_time, fps)
    if args.max > 0:
        targets = targets[: args.max]
    logger.info("视频 %.1f fps，候选车辆 %d 辆", fps, len(targets))

    detector = VehicleDetector(model_path="models/yolov8s.pt")
    crops_dir = os.path.join(run_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    reader = None
    if args.ocr:
        try:
            import easyocr

            reader = easyocr.Reader(["ch_sim", "en"], gpu=False, verbose=False)
            logger.info("easyocr 已加载（仅作提示，识别率有限）")
        except Exception as exc:  # OCR 只是锦上添花，失败不阻断
            logger.warning("easyocr 不可用，跳过 OCR：%s", exc)

    index = {}
    frame_index = 0
    target_pos = 0
    saved = 0
    while target_pos < len(targets):
        if not cap.grab():
            break
        if frame_index == targets[target_pos]["frame"]:
            ok, frame = cap.retrieve()
            if ok:
                target = targets[target_pos]
                crop, bbox = crop_vehicle(frame, detector.detect(frame), target["point"])
                if crop is not None and crop.size:
                    height, width = crop.shape[:2]
                    scale = args.width / float(width)
                    thumb = cv2.resize(crop, (args.width, max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
                    filename = f"{target['passage_id']}.jpg"
                    path = os.path.join(crops_dir, filename)
                    ok2, buffer = cv2.imencode(".jpg", thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                    if ok2:
                        buffer.tofile(path)
                        saved += 1
                        entry = {
                            "file": filename,
                            "frame": target["frame"],
                            "bbox": [round(float(v), 1) for v in bbox],
                            "enter_time": target["enter_time"],
                        }
                        if not args.no_plate:
                            plate_file, plate_text, plate_conf = save_plate_strip(
                                crop, target["passage_id"], crops_dir, reader
                            )
                            entry.update(
                                {"plate_file": plate_file, "plate_text": plate_text, "plate_conf": plate_conf}
                            )
                        index[target["passage_id"]] = entry
            target_pos += 1
        frame_index += 1
    cap.release()

    with open(os.path.join(crops_dir, "index.json"), "w", encoding="utf-8") as handle:
        json.dump(index, handle, ensure_ascii=False, indent=2)

    logger.info("已生成 %d / %d 张车辆截图 → %s", saved, len(targets), crops_dir)
    print(json.dumps({"run": run_dir, "targets": len(targets), "saved": saved, "crops_dir": crops_dir},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
