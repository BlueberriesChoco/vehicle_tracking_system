from typing import List, Dict, Optional, Tuple
import numpy as np
from datetime import datetime

from ..tracking.tracklet import Tracklet
from .scene_geometry import SceneGeometry
from .trajectory_smoother import TrajectorySmoother


class TrajectoryExtractor:
    """轨迹提取器。

    从 ByteTrack 跟踪结果中提取完整轨迹，判断进出事件，
    平滑去噪，并标记是否为完整穿越通道的有效轨迹。
    """

    def __init__(
        self,
        scene_geometry: SceneGeometry,
        smoother: TrajectorySmoother,
        min_trajectory_length: int = 10,
        enter_min_frames: int = 5,
    ):
        self.scene = scene_geometry
        self.smoother = smoother
        self.min_trajectory_length = min_trajectory_length
        self.enter_min_frames = enter_min_frames

        # 已进入的轨迹（等待穿越出口）
        self._entered_tracks: Dict[int, Tracklet] = {}

    def process_frame(
        self,
        tracklets: Dict[int, Tracklet],
        frame_idx: int,
    ) -> List[Tracklet]:
        """逐帧处理：检测进出事件。

        优先使用标定的进出线判定穿越；
        若进出线未配置（退化线段），则自动判定：
          - enter: 轨迹点数达到 enter_min_frames 视为进入
          - exit:  轨迹变为非活跃（lost/finished）视为离开

        Returns:
            当前帧刚完成 exit 的轨迹列表
        """
        newly_exited: List[Tracklet] = []
        entry_configured = self.scene.has_valid_entry_line()
        exit_configured = self.scene.has_valid_exit_line()

        for tid, tracklet in tracklets.items():
            # ---- 进入判定 ----
            if tracklet.is_active and not tracklet.has_entered:
                entered = False
                if entry_configured and tracklet.trajectory_length >= 2:
                    p_prev = tracklet.centers[-2]
                    p_curr = tracklet.centers[-1]
                    if self.scene.check_line_crossing(p_prev, p_curr, "entry"):
                        entered = True
                elif tracklet.trajectory_length >= self.enter_min_frames:
                    entered = True

                if entered:
                    tracklet.mark_entered(frame_idx)
                    self._entered_tracks[tid] = tracklet

            # ---- 离开判定 ----
            if tracklet.has_entered and not tracklet.has_exited:
                exited = False
                if exit_configured and tracklet.trajectory_length >= 2:
                    p_prev = tracklet.centers[-2]
                    p_curr = tracklet.centers[-1]
                    if self.scene.check_line_crossing(p_prev, p_curr, "exit"):
                        exited = True
                elif not tracklet.is_active:
                    exited = True

                if exited:
                    tracklet.mark_exited(frame_idx)
                    newly_exited.append(tracklet)
                    if tid in self._entered_tracks:
                        del self._entered_tracks[tid]

        return newly_exited

    def extract_valid_trajectories(
        self,
        all_tracklets: List[Tracklet],
    ) -> List[Tracklet]:
        """从所有轨迹中筛选出有效的完整穿越轨迹。

        筛选条件：
        1. 轨迹点数 >= min_trajectory_length
        2. has_entered == True
        3. has_exited == True (除非视频中途截断)
        """
        valid = []
        for t in all_tracklets:
            if t.trajectory_length < self.min_trajectory_length:
                continue
            if not t.has_entered:
                continue
            valid.append(t)

        return valid

    def get_smoothed_trajectory(
        self,
        tracklet: Tracklet,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """获取平滑后的轨迹坐标数组。"""
        return self.smoother.smooth_single_track(tracklet.centers)

    def build_trajectory_summary(self, tracklet: Tracklet) -> dict:
        """构建轨迹摘要字典。"""
        xs_smooth, ys_smooth = self.get_smoothed_trajectory(tracklet)

        trajectory_length_px = 0.0
        for i in range(1, len(xs_smooth)):
            dx = xs_smooth[i] - xs_smooth[i - 1]
            dy = ys_smooth[i] - ys_smooth[i - 1]
            trajectory_length_px += np.sqrt(dx * dx + dy * dy)

        trajectory_length_m = self.scene.pixel_to_world(trajectory_length_px)

        enter_time = tracklet.first_seen_time
        exit_time = tracklet.last_seen_time
        if tracklet.enter_frame is not None and tracklet.enter_frame < len(tracklet.timestamps):
            enter_time = tracklet.timestamps[tracklet.enter_frame]
        if tracklet.exit_frame is not None and tracklet.exit_frame < len(tracklet.timestamps):
            exit_time = tracklet.timestamps[tracklet.exit_frame]

        return {
            "track_id": tracklet.track_id,
            "camera_id": tracklet.camera_id,
            "class_name": tracklet.class_name,
            "trajectory_length_m": round(trajectory_length_m, 2),
            "num_points": len(xs_smooth),
            "enter_time": enter_time.isoformat() if enter_time else None,
            "exit_time": exit_time.isoformat() if exit_time else None,
            "duration_sec": tracklet.duration_seconds,
            "is_complete": tracklet.is_complete,
            "xs_smooth": xs_smooth.tolist(),
            "ys_smooth": ys_smooth.tolist(),
        }
