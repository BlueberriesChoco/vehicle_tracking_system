from typing import List, Tuple
import numpy as np
import cv2


class ROIFilter:
    """ROI 区域过滤器。

    只保留位于行政通道 ROI 多边形内的检测框，过滤掉通道外的干扰目标
    （如远处道路、建筑旁的车辆）。
    """

    def __init__(self, roi_polygon: List[List[int]]):
        """
        Args:
            roi_polygon: ROI 多边形顶点 [[x1,y1], [x2,y2], ...]
        """
        if len(roi_polygon) < 3:
            raise ValueError("ROI polygon must have at least 3 vertices")
        self.polygon = np.array(roi_polygon, dtype=np.int32)

    def contains_point(self, x: float, y: float) -> bool:
        return cv2.pointPolygonTest(self.polygon, (float(x), float(y)), False) >= 0

    def filter_detections(
        self,
        detections: List[dict],
        use_center: bool = True,
    ) -> List[dict]:
        """过滤检测结果，仅保留 ROI 内的目标。

        Args:
            detections: 检测结果列表
            use_center: True=用检测框底部中心点判定，False=用检测框中心点判定

        Returns:
            过滤后的检测列表
        """
        filtered = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            if use_center:
                cx, cy = (x1 + x2) / 2, y2  # 底部中心（接地点）
            else:
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            if self.contains_point(cx, cy):
                filtered.append(det)

        return filtered

    def draw_roi(
        self,
        frame: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """在帧上绘制 ROI 多边形。"""
        vis = frame.copy()
        cv2.polylines(vis, [self.polygon], isClosed=True, color=color, thickness=thickness)
        return vis
