from typing import List, Dict, Tuple
import numpy as np

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
          - exit:  轨迹超过丢失缓冲期（finished）视为离开

        Returns:
            当前帧刚完成 exit 的轨迹列表
        """
        newly_exited: List[Tracklet] = []
        entry_configured = self.scene.has_valid_entry_line()
        exit_configured = self.scene.has_valid_exit_line()

        for tid, tracklet in tracklets.items():
            # ---- 进入判定：有标定线用过线判断，无标定线用最小轨迹长度近似 ----
            if tracklet.is_active and not tracklet.has_entered:
                entered = False
                if entry_configured and tracklet.trajectory_length >= 2:
                    p_prev = tracklet.centers[-2]
                    p_curr = tracklet.centers[-1]
                    crossed_entry = self.scene.check_line_crossing(p_prev, p_curr, "entry")
                    # 双向车道：先跨哪条线都算进入，记录先跨的是哪一条
                    if self.scene.bidirectional and exit_configured:
                        crossed_exit = self.scene.check_line_crossing(p_prev, p_curr, "exit")
                        if crossed_entry or crossed_exit:
                            entered = True
                            tracklet.entry_side = "entry" if crossed_entry else "exit"
                    elif crossed_entry:
                        entered = True
                        tracklet.entry_side = "entry"
                elif tracklet.trajectory_length >= self.enter_min_frames:
                    entered = True

                if entered:
                    tracklet.mark_entered(frame_idx)
                    self._entered_tracks[tid] = tracklet

            # ---- 离开判定：有出口线用过线判断，无出口线用轨迹失活判断 ----
            if tracklet.has_entered and not tracklet.has_exited:
                exited = False
                if exit_configured and tracklet.trajectory_length >= 2:
                    p_prev = tracklet.centers[-2]
                    p_curr = tracklet.centers[-1]
                    if self.scene.bidirectional and entry_configured:
                        # 双向：跨过另一条线才算离开，避免同一条线重复计数
                        if tracklet.entry_side == "entry":
                            exited = self.scene.check_line_crossing(p_prev, p_curr, "exit")
                        else:
                            exited = self.scene.check_line_crossing(p_prev, p_curr, "entry")
                    elif self.scene.check_line_crossing(p_prev, p_curr, "exit"):
                        exited = True
                    # 双向车道兜底：朝镜头方向驶来的车，可能在画面边缘跨出口线时
                    # 轨迹还不足 2 点（跨线事件漏检），于是只记到一次跨线。
                    # 轨迹结束时补记离开，避免整条被丢弃。
                    if (not exited and self.scene.bidirectional
                            and tracklet.state == "finished"
                            and tracklet.trajectory_length >= self.min_trajectory_length):
                        exited = True
                        tracklet.exit_by_track_end = True
                elif tracklet.state == "finished":
                    exited = True

                if exited:
                    tracklet.mark_exited(tracklet.last_seen_frame)
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

        # 用逐点地面距离累加，才能吃到 scene 的单应矩阵；
        # pixel_to_world 只按常数 px_per_meter 换算，标定后仍然会算错。
        centers = [(float(xs_smooth[i]), float(ys_smooth[i])) for i in range(len(xs_smooth))]
        trajectory_length_m = self.scene.trajectory_length_meters(centers)

        # enter/exit frame 是全局帧号，必须先定位到轨迹自身的 frame_indices 再取时间戳。
        enter_time = tracklet.first_seen_time
        exit_time = tracklet.last_seen_time
        if tracklet.enter_frame in tracklet.frame_indices:
            enter_time = tracklet.timestamps[tracklet.frame_indices.index(tracklet.enter_frame)]
        if tracklet.exit_frame in tracklet.frame_indices:
            exit_time = tracklet.timestamps[tracklet.frame_indices.index(tracklet.exit_frame)]

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

    def reset(self):
        """Clear per-video trajectory lifecycle state."""
        self._entered_tracks.clear()
