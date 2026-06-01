from typing import Tuple, Optional, List
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Tracklet:
    """车辆轨迹数据结构。

    维护一辆车从出现到消失的完整跟踪状态，累积所有帧的位置记录和时间戳。
    """

    track_id: int
    camera_id: str
    class_id: int
    class_name: str

    # 生命周期
    state: str = "active"           # active / lost / finished
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    first_seen_time: Optional[datetime] = None
    last_seen_time: Optional[datetime] = None
    total_frames: int = 0
    lost_frames: int = 0            # 连续丢失帧数

    # 轨迹数据：并行数组
    frame_indices: List[int] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)
    bboxes: List[List[int]] = field(default_factory=list)     # [x1,y1,x2,y2]
    centers: List[Tuple[float, float]] = field(default_factory=list)  # 底部中心点
    confidences: List[float] = field(default_factory=list)

    # 通道事件
    has_entered: bool = False       # 是否已穿越入口线
    has_exited: bool = False        # 是否已穿越出口线
    enter_frame: Optional[int] = None
    exit_frame: Optional[int] = None

    # 特征（Phase 2 填充）
    best_confidence: float = 0.0
    avg_bbox_area: float = 0.0
    best_crop: object = None              # 最佳检测帧裁剪（用于 Phase 2 特征提取）
    best_crop_bbox: List[int] = field(default_factory=list)
    largest_crop: object = None           # 最大尺寸裁剪（车辆最近时，用于车牌OCR）
    largest_crop_bbox: List[int] = field(default_factory=list)
    largest_bbox_area: float = 0.0        # 记录最大 bbox 面积，用于比较
    plate_number: str = ""                # 车牌号
    plate_hash: str = ""                  # 车牌哈希
    reid_embedding: object = None         # ReID 特征向量 (np.ndarray)

    def add_detection(
        self,
        frame_idx: int,
        timestamp: datetime,
        bbox: List[int],
        center: Tuple[float, float],
        confidence: float,
    ):
        """追加一帧的检测关联记录。"""
        self.frame_indices.append(frame_idx)
        self.timestamps.append(timestamp)
        self.bboxes.append(bbox)
        self.centers.append(center)
        self.confidences.append(confidence)

        self.last_seen_frame = frame_idx
        self.last_seen_time = timestamp
        self.total_frames += 1

        if confidence > self.best_confidence:
            self.best_confidence = confidence

        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        if self.avg_bbox_area == 0:
            self.avg_bbox_area = area
        else:
            self.avg_bbox_area = 0.9 * self.avg_bbox_area + 0.1 * area

    def mark_lost(self):
        """标记当前帧丢失。"""
        self.lost_frames += 1
        if self.state == "active":
            self.state = "lost"

    def mark_reactivated(self, frame_idx: int):
        """丢失后重新关联。"""
        self.lost_frames = 0
        self.state = "active"

    def mark_finished(self):
        """标记轨迹结束。"""
        self.state = "finished"

    def mark_entered(self, frame_idx: int):
        self.has_entered = True
        self.enter_frame = frame_idx

    def mark_exited(self, frame_idx: int):
        self.has_exited = True
        self.exit_frame = frame_idx

    @property
    def trajectory_length(self) -> int:
        """轨迹点数。"""
        return len(self.centers)

    @property
    def is_complete(self) -> bool:
        """是否为完整穿越通道的轨迹。"""
        return self.has_entered and self.has_exited

    @property
    def is_active(self) -> bool:
        return self.state == "active"

    @property
    def is_lost(self) -> bool:
        return self.state == "lost"

    @property
    def duration_seconds(self) -> Optional[float]:
        """穿越通道耗时（秒）。"""
        if self.first_seen_time is None or self.last_seen_time is None:
            return None
        return (self.last_seen_time - self.first_seen_time).total_seconds()

    def get_bbox_at(self, frame_idx: int) -> Optional[List[int]]:
        """获取指定帧的检测框。"""
        for i, fidx in enumerate(self.frame_indices):
            if fidx == frame_idx:
                return self.bboxes[i]
        return None

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "camera_id": self.camera_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "state": self.state,
            "first_seen_frame": self.first_seen_frame,
            "last_seen_frame": self.last_seen_frame,
            "total_frames": self.total_frames,
            "has_entered": self.has_entered,
            "has_exited": self.has_exited,
            "enter_frame": self.enter_frame,
            "exit_frame": self.exit_frame,
            "trajectory_length": self.trajectory_length,
            "is_complete": self.is_complete,
        }
