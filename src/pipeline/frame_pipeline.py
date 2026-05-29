from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np

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

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: Optional[datetime] = None,
    ) -> Tuple[np.ndarray, List[dict]]:
        """处理单帧。

        Args:
            frame: BGR 图像 (H, W, 3)
            timestamp: 帧时间戳

        Returns:
            (标注后的帧, 新完成的行为向量列表)
        """
        if self._start_time is None:
            self._start_time = timestamp or datetime.now()

        if timestamp is None:
            timestamp = self._start_time + timedelta(
                seconds=self._frame_idx / self.frame_rate
            )

        # Step 1: 检测
        detections = self.detector.detect(frame)

        # Step 2: ROI 过滤
        detections = self.roi_filter.filter_detections(detections)

        # Step 3: 跟踪
        self.tracker.update(detections, self._frame_idx, timestamp)

        # Step 3.5: 保存每辆车的最佳裁剪（用于 Phase 2 特征提取）
        self._save_best_crops(frame)

        # Step 4: 轨迹进出判定（传所有 tracklet，不活跃的用于 auto-exit 检测）
        all_tracklets = self.tracker.get_all_tracklets()
        newly_exited = self.trajectory_extractor.process_frame(
            {t.track_id: t for t in all_tracklets},
            self._frame_idx,
        )

        # Step 5: 对刚完成穿越的轨迹构建行为向量
        new_vectors = []
        active_tracklets = self.tracker.get_active_tracklets()
        all_positions = self._get_active_positions(active_tracklets)

        for tracklet in newly_exited:
            vec = self.vectorizer.vectorize(tracklet, all_positions)
            new_vectors.append(vec)

        self._completed_vectors.extend(new_vectors)
        self._frame_idx += 1

        return frame, new_vectors

    def _save_best_crops(self, frame: np.ndarray):
        """为每个活跃轨迹保存置信度最高的检测裁剪图。"""
        for tracklet in self.tracker.get_active_tracklets():
            if not tracklet.bboxes:
                continue

            latest_conf = tracklet.confidences[-1] if tracklet.confidences else 0
            if latest_conf > tracklet.best_confidence:
                tracklet.best_confidence = latest_conf
                x1, y1, x2, y2 = tracklet.bboxes[-1]
                h, w = frame.shape[:2]
                x1_c = max(0, int(x1))
                y1_c = max(0, int(y1))
                x2_c = min(w, int(x2))
                y2_c = min(h, int(y2))
                if x2_c > x1_c and y2_c > y1_c:
                    tracklet.best_crop = frame[y1_c:y2_c, x1_c:x2_c].copy()
                    tracklet.best_crop_bbox = [x1_c, y1_c, x2_c, y2_c]

    def finalize(self) -> List[dict]:
        """处理结束后，对未退出但已进入的轨迹也构建向量。"""
        remaining = []
        all_tracklets = self.tracker.get_all_tracklets()
        all_positions = self._get_all_last_positions(all_tracklets)

        for tracklet in all_tracklets:
            if tracklet.is_complete or tracklet.has_entered:
                already_done = any(
                    v["track_id"] == tracklet.track_id for v in self._completed_vectors
                )
                if not already_done and tracklet.trajectory_length >= self.trajectory_extractor.min_trajectory_length:
                    vec = self.vectorizer.vectorize(tracklet, all_positions)
                    remaining.append(vec)

        self._completed_vectors.extend(remaining)
        return remaining

    @property
    def frame_idx(self) -> int:
        return self._frame_idx

    @property
    def completed_vectors(self) -> List[dict]:
        return self._completed_vectors

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
