from typing import List, Tuple
import numpy as np
import cv2

from ..tracking.tracklet import Tracklet
from ..trajectory.scene_geometry import SceneGeometry


# 车型颜色映射
CLASS_COLORS = {
    "car": (0, 255, 0),          # 绿色
    "truck": (255, 0, 0),        # 蓝色
    "bus": (0, 255, 255),        # 黄色
    "motorcycle": (255, 0, 255), # 紫色
    "unknown": (128, 128, 128),  # 灰色
}


class Visualizer:
    """可视化绘制器。

    在帧上绘制检测框、轨迹ID、轨迹尾迹、出入口线、ROI区域。
    """

    def __init__(
        self,
        scene_geometry: SceneGeometry,
        draw_trajectory: bool = True,
        trajectory_tail: int = 50,
        box_thickness: int = 2,
        font_scale: float = 0.5,
    ):
        self.scene = scene_geometry
        self.draw_trajectory = draw_trajectory
        self.trajectory_tail = trajectory_tail
        self.box_thickness = box_thickness
        self.font_scale = font_scale

    def draw_frame(
        self,
        frame: np.ndarray,
        tracklets: List[Tracklet],
    ) -> np.ndarray:
        """在单帧上绘制所有可视化元素。"""
        vis = frame.copy()

        # 绘制出入口线
        self._draw_line(vis, self.scene.entry_p1, self.scene.entry_p2, (0, 255, 0), "ENTRY")
        self._draw_line(vis, self.scene.exit_p1, self.scene.exit_p2, (0, 0, 255), "EXIT")

        # 绘制每个活跃轨迹
        for tracklet in tracklets:
            if not tracklet.centers:
                continue

            color = CLASS_COLORS.get(tracklet.class_name, (128, 128, 128))

            # 检测框
            if tracklet.bboxes:
                x1, y1, x2, y2 = tracklet.bboxes[-1]
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, self.box_thickness)

            # 轨迹尾迹
            if self.draw_trajectory and len(tracklet.centers) >= 2:
                tail = tracklet.centers[-self.trajectory_tail:]
                for i in range(1, len(tail)):
                    pt1 = (int(tail[i - 1][0]), int(tail[i - 1][1]))
                    pt2 = (int(tail[i][0]), int(tail[i][1]))
                    alpha = 0.3 + 0.7 * (i / len(tail))
                    faded = tuple(int(c * alpha) for c in color)
                    cv2.line(vis, pt1, pt2, faded, 1)

            # 标签: ID + 类型
            label = f"ID:{tracklet.track_id} {tracklet.class_name}"
            if tracklet.bboxes:
                x1, y1, x2, y2 = tracklet.bboxes[-1]
                self._draw_label(vis, label, (x1, y1 - 5), color)

            # 如果已进入/退出，标记
            if tracklet.has_entered:
                cv2.circle(vis, (int(tracklet.centers[tracklet.enter_frame - tracklet.frame_indices[0]][0]),
                                 int(tracklet.centers[tracklet.enter_frame - tracklet.frame_indices[0]][1])
                                 if tracklet.enter_frame else (0, 0)),
                           5, (0, 255, 0), -1)

        return vis

    def _draw_line(
        self,
        frame: np.ndarray,
        p1: np.ndarray,
        p2: np.ndarray,
        color: Tuple[int, int, int],
        label: str,
    ):
        """绘制线段并标注。"""
        pt1 = (int(p1[0]), int(p1[1]))
        pt2 = (int(p2[0]), int(p2[1]))
        cv2.line(frame, pt1, pt2, color, 2)
        mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        self._draw_label(frame, label, mid, color)

    def _draw_label(
        self,
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int],
    ):
        """绘制带背景的文字标签。"""
        (tw, th), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 2
        )
        x, y = position
        y = max(y, th + 5)
        cv2.rectangle(frame, (x, y - th - 5), (x + tw, y + baseline), (0, 0, 0), -1)
        cv2.putText(
            frame, text, (x, y - 3),
            cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, color, 2,
        )
