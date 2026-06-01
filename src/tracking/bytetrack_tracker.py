from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta

from .tracklet import Tracklet


class ByteTrackTracker:
    """ByteTrack 跟踪器封装。

    直接基于 IoU + 卡尔曼滤波实现 ByteTrack 核心逻辑，不依赖 boxmot 等外部库。
    核心思想：高分检测先匹配 → 低分检测二次关联 → 未匹配轨迹更新状态。
    """

    def __init__(
        self,
        track_buffer: int = 30,
        match_thresh: float = 0.3,
        frame_rate: int = 25,
        min_hits: int = 3,
        camera_id: str = "cam01",
    ):
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        self.min_hits = min_hits
        self.camera_id = camera_id

        # 轨迹存储：track_id -> Tracklet
        self.tracklets: Dict[int, Tracklet] = {}
        self._next_id: int = 0
        self._frame_idx: int = 0
        self._start_time: Optional[datetime] = None

        # 卡尔曼滤波器：track_id -> KalmanState
        self._kalman_states: Dict[int, np.ndarray] = {}

    def update(
        self,
        detections: List[dict],
        frame_idx: int,
        timestamp: Optional[datetime] = None,
    ) -> Dict[int, Tracklet]:
        """用当前帧的检测结果更新所有轨迹。

        Args:
            detections: 检测列表 [{"bbox": [x1,y1,x2,y2], ...}, ...]
            frame_idx: 当前帧号
            timestamp: 当前帧时间戳

        Returns:
            当前活跃的轨迹字典
        """
        if self._start_time is None:
            self._start_time = timestamp or datetime.now()

        if timestamp is None:
            timestamp = self._start_time + timedelta(seconds=frame_idx / self.frame_rate)

        self._frame_idx = frame_idx

        # Step 1: 将检测分为高分和低分两组
        high_dets, low_dets = self._split_by_confidence(detections)

        # Step 2: 使用卡尔曼预测当前帧位置
        predictions = self._predict_all()

        # Step 3: 已确认轨迹与高分检测做第一次关联（IoU）
        # Keep recently lost tracks eligible for matching. A single missed
        # detection must not permanently split one vehicle into two IDs.
        all_active_ids = [
            tid for tid, t in self.tracklets.items()
            if t.is_active or (t.is_lost and t.lost_frames <= self.track_buffer)
        ]
        matched, unmatched_tracks, unmatched_high = self._match_iou(
            all_active_ids, high_dets, predictions
        )

        # Step 4: 低分检测与未匹配轨迹做第二次关联
        matched_low, unmatched_tracks, unmatched_low = self._match_iou(
            unmatched_tracks, low_dets, predictions
        )
        matched.update(matched_low)

        # Step 5: 对仍未匹配的轨迹标记丢失
        for tid in unmatched_tracks:
            if tid in self.tracklets:
                self.tracklets[tid].mark_lost()
                self._update_kalman(tid, None)

        # Step 6: 未匹配的高分检测 → 新建轨迹
        for det in unmatched_high:
            self._create_tracklet(det, frame_idx, timestamp)

        # Step 7: 更新已匹配轨迹
        for tid, det_idx in matched.items():
            det = detections[det_idx]
            bbox = det["bbox"]
            center = ((bbox[0] + bbox[2]) / 2, bbox[3])  # 底部中心点
            self.tracklets[tid].add_detection(frame_idx, timestamp, bbox, center, det["confidence"])
            self.tracklets[tid].mark_reactivated(frame_idx)
            self._update_kalman(tid, self._bbox_to_z(bbox))

        # Step 8: 删除失效轨迹
        self._remove_dead_tracks()

        return self.tracklets

    def _split_by_confidence(self, detections: List[dict]) -> Tuple[List[dict], List[dict]]:
        """将检测分为高分(>=0.5)和低分两组。"""
        high, low = [], []
        for i, det in enumerate(detections):
            if det["confidence"] >= 0.5:
                high.append((i, det))
            else:
                low.append((i, det))
        return high, low

    def _predict_all(self) -> Dict[int, List[float]]:
        """对所有活跃轨迹做卡尔曼预测。"""
        predictions = {}
        for tid in list(self._kalman_states.keys()):
            if tid in self.tracklets and self.tracklets[tid].is_lost:
                # 丢失轨迹不做预测，维持原框
                last_bbox = self.tracklets[tid].bboxes[-1] if self.tracklets[tid].bboxes else None
                if last_bbox:
                    predictions[tid] = last_bbox
            else:
                state = self._kalman_states.get(tid)
                if state is not None:
                    predictions[tid] = self._kalman_predict(state)
        return predictions

    def _match_iou(
        self,
        track_ids: List[int],
        dets: List[Tuple[int, dict]],
        predictions: Dict[int, List[float]],
    ) -> Tuple[Dict[int, int], List[int], List[dict]]:
        """基于 IoU 的匈牙利匹配（贪心近似）。

        Returns:
            matched: {track_id: det_idx_in_full_list}
            unmatched_tracks: 未匹配的 track_id 列表
            unmatched_dets: (det_idx_in_full_list, det) 列表
        """
        if not track_ids or not dets:
            return {}, list(track_ids), [d for _, d in dets]

        # 构建 IoU 矩阵
        iou_matrix = np.zeros((len(track_ids), len(dets)))
        for i, tid in enumerate(track_ids):
            pred_bbox = predictions.get(tid)
            if pred_bbox is None:
                continue
            for j, (det_idx, det) in enumerate(dets):
                iou_matrix[i, j] = self._iou(pred_bbox, det["bbox"])

        matched = {}
        matched_track_set = set()
        matched_det_set = set()

        # 贪心匹配：优先匹配 IoU 最高的对
        while True:
            if iou_matrix.size == 0:
                break
            max_iou = iou_matrix.max()
            if max_iou < self.match_thresh:
                break
            track_i, det_j = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            tid = track_ids[track_i]
            det_idx = dets[det_j][0]
            matched[tid] = det_idx
            matched_track_set.add(tid)
            matched_det_set.add(det_j)
            iou_matrix[track_i, :] = 0
            iou_matrix[:, det_j] = 0

        unmatched_tracks = [tid for tid in track_ids if tid not in matched_track_set]
        unmatched_dets = [det for j, (idx, det) in enumerate(dets) if j not in matched_det_set]

        return matched, unmatched_tracks, unmatched_dets

    def _get_confirmed_track_ids(self) -> List[int]:
        """获取已确认的轨迹（连续检测次数 >= min_hits）。"""
        return [
            tid for tid, t in self.tracklets.items()
            if t.is_active and t.trajectory_length >= self.min_hits
        ]

    def _create_tracklet(self, det: dict, frame_idx: int, timestamp: datetime):
        """创建新轨迹。"""
        tid = self._next_id
        self._next_id += 1

        bbox = det["bbox"]
        center = ((bbox[0] + bbox[2]) / 2, bbox[3])

        tracklet = Tracklet(
            track_id=tid,
            camera_id=self.camera_id,
            class_id=det["class_id"],
            class_name=det["class_name"],
            first_seen_frame=frame_idx,
            first_seen_time=timestamp,
        )
        tracklet.add_detection(frame_idx, timestamp, bbox, center, det["confidence"])
        self.tracklets[tid] = tracklet
        self._init_kalman(tid, bbox)

    def _remove_dead_tracks(self):
        """删除失效轨迹（丢失超过 buffer 或已完成的）。"""
        to_remove = []
        for tid, t in self.tracklets.items():
            if t.lost_frames > self.track_buffer:
                t.mark_finished()
                to_remove.append(tid)

        for tid in to_remove:
            # finished 的保留在字典中供后续查询，但不再活跃
            self.tracklets[tid].state = "finished"
            if tid in self._kalman_states:
                del self._kalman_states[tid]

    # ---- 卡尔曼滤波（简化实现）----

    def _init_kalman(self, tid: int, bbox: List[int]):
        """初始化卡尔曼状态: [cx, cy, s, r, vcx, vcy, vs]"""
        x1, y1, x2, y2 = bbox
        w = max(x2 - x1, 1)
        h = max(y2 - y1, 1)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        s = w * h
        r = w / h
        state = np.array([cx, cy, s, r, 0, 0, 0], dtype=np.float64)
        self._kalman_states[tid] = state

    def _kalman_predict(self, state: np.ndarray) -> List[float]:
        """预测下一帧的 bbox。恒定速度模型。"""
        # 状态转移
        predicted = state.copy()
        predicted[0] += predicted[4]  # cx += vcx
        predicted[1] += predicted[5]  # cy += vcy
        predicted[2] += predicted[6]  # s += vs
        # 转换回 bbox
        return self._z_to_bbox(predicted)

    def _update_kalman(self, tid: int, z: Optional[np.ndarray]):
        """用观测值更新卡尔曼状态。z = [cx, cy, s, r]"""
        if tid not in self._kalman_states:
            return
        state = self._kalman_states[tid]
        if z is not None:
            # 简单指数平滑更新
            alpha = 0.7
            state[:4] = alpha * z + (1 - alpha) * state[:4]
            # 更新速度
            prev_cx, prev_cy = state[0], state[1]
            state[4] = z[0] - state[0]  # vcx
            state[5] = z[1] - state[1]  # vcy
            state[6] = z[2] - state[2]  # vs
            state[0], state[1], state[2] = z[0], z[1], z[2]
        self._kalman_states[tid] = state

    @staticmethod
    def _bbox_to_z(bbox: List[int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        w = max(x2 - x1, 1)
        h = max(y2 - y1, 1)
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2, w * h, w / h])

    @staticmethod
    def _z_to_bbox(z: np.ndarray) -> List[float]:
        cx, cy, s, r, *_ = z
        s = max(s, 1)
        r = max(r, 0.1)
        w = np.sqrt(s * r)
        h = s / w
        return [float(cx - w / 2), float(cy - h / 2), float(cx + w / 2), float(cy + h / 2)]

    @staticmethod
    def _iou(bbox_a: List[float], bbox_b: List[int]) -> float:
        """计算两个 bbox 的 IoU。"""
        xa = max(bbox_a[0], bbox_b[0])
        ya = max(bbox_a[1], bbox_b[1])
        xb = min(bbox_a[2], bbox_b[2])
        yb = min(bbox_a[3], bbox_b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
        area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def get_active_tracklets(self) -> List[Tracklet]:
        return [t for t in self.tracklets.values() if t.is_active]

    def get_finished_tracklets(self) -> List[Tracklet]:
        return [t for t in self.tracklets.values() if t.state == "finished"]

    def get_all_tracklets(self) -> List[Tracklet]:
        return list(self.tracklets.values())

    def get_tracklet(self, track_id: int) -> Optional[Tracklet]:
        return self.tracklets.get(track_id)

    def reset(self):
        """Clear per-video tracking state while keeping configuration."""
        self.tracklets.clear()
        self._kalman_states.clear()
        self._next_id = 0
        self._frame_idx = 0
        self._start_time = None
