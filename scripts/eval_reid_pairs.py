#!/usr/bin/env python
"""Evaluate the vehicle Re-ID backend by auto-generating same/different pairs.

No manual labels needed: vehicles are tracked across frames inside one video;
crops of the same track (with a time gap) are positives, crops of different
tracks are negatives. Reports ROC-AUC, EER and the best threshold.

Usage:
    python scripts/eval_reid_pairs.py \
        --video data/outputs/h264/admin_in/clip_08_short.h264 \
        --camera admin_in --frames 1500 --step 5 \
        --output data/outputs/eval/reid_pairs.json
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.anomaly.metrics import roc_auc
from src.detection.detector import VehicleDetector
from src.feature_extraction.color_recognizer import ColorRecognizer
from src.feature_extraction.reid_extractor import ReIDExtractor
from src.feature_extraction.vehicle_classifier import VehicleClassifier
from src.trajectory.stitching import _color_group
from src.utils.logger import get_logger

logger = get_logger(__name__)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Auto-evaluate vehicle Re-ID pairs")
    parser.add_argument("--video", required=True)
    parser.add_argument("--camera", default="admin_in")
    parser.add_argument("--frames", type=int, default=1500)
    parser.add_argument("--step", type=int, default=5, help="Sample every N frames")
    parser.add_argument("--min-crop-height", type=int, default=80, help="Skip tiny crops")
    parser.add_argument("--max-crop-height", type=int, default=100000, help="Skip oversized crops")
    parser.add_argument("--max-per-track", type=int, default=6)
    parser.add_argument("--min-gap", type=int, default=25,
                        help="Minimum frame gap between two crops of the same track")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    detector = VehicleDetector(model_path="models/yolov8s.pt", device=0)
    extractor = ReIDExtractor(backend="onnx", weights_path="models/vehicle_reid.onnx")
    logger.info("ReID backend: %s (dim %s)", extractor.active_backend, extractor.embedding_dim)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        logger.error("Cannot open video: %s", args.video)
        sys.exit(1)

    # 极简 IoU 跟踪，只为把同一辆车的裁剪归到同一条轨迹
    tracks = {}
    next_id = 0
    last_boxes = {}

    frame_idx = 0
    while frame_idx < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % args.step == 0:
            detections = detector.detect(frame)
            boxes = [d["bbox"] for d in detections]
            assigned = {}
            for det_index, box in enumerate(boxes):
                best, best_iou = -1, 0.3
                for track_id, last_box in last_boxes.items():
                    score = iou(box, last_box)
                    if score > best_iou:
                        best, best_iou = track_id, score
                if best == -1:
                    best = next_id
                    tracks[best] = []
                    next_id += 1
                assigned[best] = box
                x1, y1, x2, y2 = box
                height = y2 - y1
                if (
                    args.min_crop_height <= height <= args.max_crop_height
                    and (x2 - x1) > 20
                ):
                    crop = frame[max(0, y1):y2, max(0, x1):x2].copy()
                    if crop.size:
                        tracks[best].append((frame_idx, crop))
            last_boxes = assigned
        frame_idx += 1
    cap.release()

    usable = {tid: crops for tid, crops in tracks.items() if len(crops) >= 2}
    logger.info("Tracks: %s total, %s usable (>=2 crops)", len(tracks), len(usable))
    if len(usable) < 2:
        logger.error("Not enough tracks for evaluation")
        sys.exit(1)

    def embed(crops):
        return [(frame_idx, extractor.extract(crop)) for frame_idx, crop in crops]

    def sample_evenly(crops, limit):
        if len(crops) <= limit:
            return crops
        positions = np.linspace(0, len(crops) - 1, limit).astype(int)
        return [crops[index] for index in positions]

    embeddings = {
        tid: embed(sample_evenly(crops, args.max_per_track)) for tid, crops in usable.items()
    }

    positives, negatives = [], []
    track_ids = list(embeddings.keys())

    # 每条轨迹的属性（取最大裁剪估计），用于评估属性先验
    color_recognizer = ColorRecognizer()
    classifier = VehicleClassifier()
    attributes = {}
    for tid, crops in usable.items():
        largest = max(crops, key=lambda item: item[1].shape[0])[1]
        try:
            attributes[tid] = (classifier.classify(largest, None), color_recognizer.recognize(largest))
        except Exception:
            attributes[tid] = ("", "")

    def attribute_penalty(a, b):
        type_a, color_a = attributes.get(a, ("", ""))
        type_b, color_b = attributes.get(b, ("", ""))
        penalty = 0.0
        if type_a and type_b and type_a != type_b:
            penalty += 0.05
        if _color_group(color_a) and _color_group(color_b) and _color_group(color_a) != _color_group(color_b):
            penalty += 0.05
        return penalty

    for tid in track_ids:
        items = embeddings[tid]
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                # 同车正样本要求足够的时间间隔，避免“几乎同一帧”抬高相似度
                if items[j][0] - items[i][0] < args.min_gap:
                    continue
                positives.append(float(np.dot(items[i][1], items[j][1])))
    for a in range(len(track_ids)):
        for b in range(a + 1, len(track_ids)):
            penalty = attribute_penalty(track_ids[a], track_ids[b])
            for _, ea in embeddings[track_ids[a]]:
                for _, eb in embeddings[track_ids[b]]:
                    negatives.append((float(np.dot(ea, eb)), penalty))

    if not positives or not negatives:
        logger.error("Need both positive and negative pairs")
        sys.exit(1)

    y_true = [1] * len(positives) + [0] * len(negatives)

    pos = np.array(positives)
    neg_sim = np.array([item[0] for item in negatives])
    neg_penalty = np.array([item[1] for item in negatives])
    neg = neg_sim - neg_penalty

    auc_plain = roc_auc(y_true, pos.tolist() + neg_sim.tolist())
    auc_prior = roc_auc(y_true, pos.tolist() + neg.tolist())

    thresholds = np.unique(np.round(np.concatenate([pos, neg]), 4))
    best_f1 = None
    best_eer = None
    at_one_percent = None
    for threshold in thresholds:
        tp = float(np.sum(pos >= threshold))
        fp = float(np.sum(neg >= threshold))
        recall = tp / len(pos)
        fpr = fp / len(neg)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        item = {
            "threshold": float(threshold),
            "recall": round(recall, 4),
            "fpr": round(fpr, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
        }
        if best_f1 is None or item["f1"] > best_f1["f1"]:
            best_f1 = item
        # EER：召回率与 (1 - 误报率) 最接近的点
        gap = abs(recall - (1.0 - fpr))
        if best_eer is None or gap < best_eer["gap"]:
            best_eer = dict(item, gap=round(gap, 4))
        # 常用工作点：误报率 <= 1% 时能取到的最高召回
        if fpr <= 0.01:
            if at_one_percent is None or recall > at_one_percent["recall"]:
                at_one_percent = item

    payload = {
        "video": args.video,
        "tracks": len(usable),
        "min_crop_height": args.min_crop_height,
        "max_crop_height": args.max_crop_height,
        "min_gap_frames": args.min_gap,
        "positive_pairs": len(positives),
        "negative_pairs": len(negatives),
        "positive_mean": round(float(pos.mean()), 4),
        "negative_mean": round(float(neg_sim.mean()), 4),
        "auc_reid_only": round(float(auc_plain), 4) if auc_plain is not None else None,
        "auc_with_attribute_prior": round(float(auc_prior), 4) if auc_prior is not None else None,
        "negative_pairs_penalized": int(np.sum(neg_penalty > 0)),
        "eer_point": best_eer,
        "at_fpr_1pct": at_one_percent,
        "best_f1_point": best_f1,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info("ReID pair evaluation saved to: %s", args.output)


if __name__ == "__main__":
    main()
