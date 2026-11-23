from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import cv2

from ..detection.detector import VehicleDetector
from ..detection.roi_filter import ROIFilter
from ..tracking.bytetrack_tracker import ByteTrackTracker
from ..tracking.tracklet import Tracklet
from ..trajectory.trajectory_extractor import TrajectoryExtractor
from ..behavior.behavior_vector import BehaviorVectorizer


class FramePipeline:
    """单帧处理管线。

    串联：检测 → ROI过滤 → 跟踪 → 保存最佳裁剪 → 轨迹进出判定 → 行为向量化。
    每帧调用一次 `process_frame()`。
    """

    def __init__(
        self,
        detector: VehicleDetector,
        roi_filter: ROIFilter,
        tracker: ByteTrackTracker,
        trajectory_extractor: TrajectoryExtractor,
        behavior_vectorizer: BehaviorVectorizer,
        frame_rate: int = 25,
    ):
        self.detector = detector
        self.roi_filter = roi_filter
        self.tracker = tracker
        self.trajectory_extractor = trajectory_extractor
        self.vectorizer = behavior_vectorizer
        self.frame_rate = frame_rate

        self._frame_idx: int = 0
        self._start_time: Optional[datetime] = None
        self._completed_vectors: List[dict] = []
        self.identity_crop_scale = (1.0, 1.0)

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: Optional[datetime] = None,
    ) -> Tuple[np.ndarray, List[dict]]:
        """处理单帧（检测 + 跟踪 + 向量化）。"""
        detections = self.detector.detect(frame)
        return self._process_with_detections(frame, detections, timestamp)

    def process_frames(
        self,
        frames: List[np.ndarray],
        timestamps: Optional[List[datetime]] = None,
        on_frame=None,
    ) -> List[Tuple[np.ndarray, List[dict]]]:
        """批量检测 + 逐帧跟踪/向量化。

        批处理只合并“检测”这一步（把多帧送进同一次 YOLO 前向），
        跟踪、轨迹完成判定和行为向量化仍按帧顺序执行，保证时序正确。

        ``on_frame(frame, active_tracklets)`` 会在每帧处理完时回调，
        供标注视频使用——必须在“当时”取轨迹快照，批处理结束后再取会拿到
        整批的最终状态。
        """
        if not frames:
            return []

        detections_batch = self.detector.detect_batch(frames)
        outputs = []
        for index, frame in enumerate(frames):
            timestamp = timestamps[index] if timestamps is not None and index < len(timestamps) else None
            outputs.append(
                self._process_with_detections(frame, detections_batch[index], timestamp, on_frame)
            )
        return outputs

    def _process_with_detections(
        self,
        frame: np.ndarray,
        detections: List[dict],
        timestamp: Optional[datetime] = None,
        on_frame=None,
    ) -> Tuple[np.ndarray, List[dict]]:
        if self._start_time is None:
            # 固定占位时间：用 datetime.now() 会让结果随运行时刻变化，不可复现
            self._start_time = timestamp or datetime(2000, 1, 1, 12, 0, 0)

        if timestamp is None:
            timestamp = self._start_time + timedelta(
                seconds=self._frame_idx / self.frame_rate
            )

        # Step 2: ROI 过滤
        detections = self.roi_filter.filter_detections(detections)

        # Step 3: 跟踪，负责给同一辆车维持稳定 track_id
        self.tracker.update(detections, self._frame_idx, timestamp)

        # Step 3.5: 保存每辆车的最佳裁剪（用于 Phase 2 特征提取）
        self._save_best_crops(frame)

        # Step 4: 轨迹完成判定；无进出线时使用“出现→消失”的自动完成逻辑
        all_tracklets = self.tracker.get_all_tracklets()
        newly_exited = self.trajectory_extractor.process_frame(
            {t.track_id: t for t in all_tracklets},
            self._frame_idx,
        )

        # Step 5: 只对本帧刚完成的轨迹构建行为向量，避免重复输出
        new_vectors = []
        active_tracklets = self.tracker.get_active_tracklets()
        all_positions = self._get_active_positions(active_tracklets)
        position_histories = self._get_position_histories(all_tracklets) if newly_exited else {}

        for tracklet in newly_exited:
            vec = self.vectorizer.vectorize(tracklet, all_positions, position_histories)
            new_vectors.append(vec)
            tracklet.identity_crops.clear()

        for tracklet in all_tracklets:
            if tracklet.state == "finished" and (
                not tracklet.has_entered
                or tracklet.trajectory_length < self.trajectory_extractor.min_trajectory_length
            ):
                tracklet.identity_crops.clear()

        self._completed_vectors.extend(new_vectors)
        self._frame_idx += 1

        if on_frame is not None:
            on_frame(frame, active_tracklets)

        return frame, new_vectors

    def _save_best_crops(self, frame: np.ndarray):
        """为每个活跃轨迹保存置信度最高的检测裁剪图和最大尺寸裁剪图。

        - best_crop: 最高置信度（用于颜色、车型识别）
        - largest_crop: 最大 bbox 面积（车辆最近时，用于车牌 OCR）
        """
        for tracklet in self.tracker.get_active_tracklets():
            if not tracklet.bboxes or tracklet.last_seen_frame != self._frame_idx:
                continue

            x1, y1, x2, y2 = tracklet.bboxes[-1]
            h, w = frame.shape[:2]
            x1_c = max(0, int(x1))
            y1_c = max(0, int(y1))
            x2_c = min(w, int(x2))
            y2_c = min(h, int(y2))
            if not (x2_c > x1_c and y2_c > y1_c):
                continue

            bbox_area = (x2_c - x1_c) * (y2_c - y1_c)
            crop = frame[y1_c:y2_c, x1_c:x2_c]

            latest_conf = tracklet.confidences[-1] if tracklet.confidences else 0
            if latest_conf >= tracklet.best_confidence:
                tracklet.best_confidence = latest_conf
                tracklet.best_crop = crop.copy()
                tracklet.best_crop_bbox = [x1_c, y1_c, x2_c, y2_c]

            # 保存最大裁剪（车辆最近时）
            if bbox_area > tracklet.largest_bbox_area:
                tracklet.largest_bbox_area = bbox_area
                tracklet.largest_crop = crop.copy()
                tracklet.largest_crop_bbox = [x1_c, y1_c, x2_c, y2_c]

            if not tracklet.has_exited and (
                getattr(self.vectorizer, "plate_ocr", None) is not None
                or getattr(self.vectorizer, "reid_extractor", None) is not None
            ):
                self._save_identity_crop(tracklet, crop, latest_conf, (x1_c, y1_c, x2_c, y2_c), (h, w))

    def _save_identity_crop(self, tracklet, crop, confidence, box, frame_size):
        if min(crop.shape[:2]) < 12:
            return
        x1, y1, x2, y2 = box
        h, w = frame_size
        sx, sy = self.identity_crop_scale
        native_size = (round((x2 - x1) * sx), round((y2 - y1) * sy))
        if min(native_size) < 12:
            return
        coverage = 0.5 if x1 == 0 or y1 == 0 or x2 == w or y2 == h else 1.0
        score = native_size[0] * native_size[1] * confidence * coverage
        bucket = self._frame_idx // max(2, round(self.frame_rate * 0.2))
        crops = tracklet.identity_crops
        if bucket in crops and crops[bucket][0] >= score:
            return
        if bucket not in crops and len(crops) >= 5:
            weakest = min(crops, key=lambda key: crops[key][0])
            if crops[weakest][0] >= score:
                return
            del crops[weakest]
        if native_size != (crop.shape[1], crop.shape[0]):
            crop = cv2.resize(crop, native_size, interpolation=cv2.INTER_AREA)
        else:
            crop = crop.copy()
        crops[bucket] = (score, crop)

    def finalize(self) -> List[dict]:
        """处理结束后，对未退出但已进入的轨迹也构建向量。"""
        remaining = []
        all_tracklets = self.tracker.get_all_tracklets()
        all_positions = self._get_all_last_positions(all_tracklets)
        position_histories = self._get_position_histories(all_tracklets)

        # 视频结束时，仍在画面内但已经满足进入条件的轨迹也要收尾输出
        for tracklet in all_tracklets:
            if tracklet.is_complete or tracklet.has_entered:
                already_done = any(
                    v["track_id"] == tracklet.track_id for v in self._completed_vectors
                )
                if not already_done and tracklet.trajectory_length >= self.trajectory_extractor.min_trajectory_length:
                    vec = self.vectorizer.vectorize(tracklet, all_positions, position_histories)
                    remaining.append(vec)
                    tracklet.identity_crops.clear()

        self._completed_vectors.extend(remaining)
        return remaining

    @property
    def frame_idx(self) -> int:
        return self._frame_idx

    @property
    def completed_vectors(self) -> List[dict]:
        return self._completed_vectors

    def reset(self):
        """Reset independent video state without reloading inference models."""
        self.tracker.reset()
        self.trajectory_extractor.reset()
        self.vectorizer.reset()
        self._frame_idx = 0
        self._start_time = None
        self._completed_vectors.clear()

    @staticmethod
    def _get_active_positions(active_tracklets: List[Tracklet]) -> Dict[int, Tuple[float, float]]:
        positions = {}
        for t in active_tracklets:
            if t.centers:
                positions[t.track_id] = t.centers[-1]
        return positions

    @staticmethod
    def _get_all_last_positions(all_tracklets: List[Tracklet]) -> Dict[int, Tuple[float, float]]:
        positions = {}
        for t in all_tracklets:
            if t.centers:
                positions[t.track_id] = t.centers[-1]
        return positions

    @staticmethod
    def _get_position_histories(
        all_tracklets: List[Tracklet],
    ) -> Dict[int, Dict[int, Tuple[float, float]]]:
        return {
            tracklet.track_id: dict(zip(tracklet.frame_indices, tracklet.centers))
            for tracklet in all_tracklets
        }
