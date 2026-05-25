from typing import Tuple, Dict, Optional
import numpy as np
import cv2


class ColorRecognizer:
    """车身颜色识别器。

    基于 HSV 空间颜色聚类的方法：
    1. 分离高/低饱和度像素
    2. 低饱和度 + 高亮度 → 白色，低饱和度 + 低亮度 → 黑色，其余 → 灰色
    3. 高饱和度像素对色相(H)做 K-Means 聚类，主色调映射到颜色标签
    """

    # (H_min, H_max, S_min, S_max, V_min, V_max) — OpenCV HSV 范围
    COLOR_RANGES: Dict[str, Tuple[int, int, int, int, int, int]] = {
        "red":    (0, 10, 50, 255, 50, 255),
        "orange": (11, 25, 50, 255, 50, 255),
        "yellow": (26, 35, 50, 255, 50, 255),
        "green":  (36, 85, 50, 255, 50, 255),
        "cyan":   (86, 100, 50, 255, 50, 255),
        "blue":   (101, 130, 50, 255, 50, 255),
        "purple": (131, 155, 50, 255, 50, 255),
        "pink":   (156, 170, 50, 255, 50, 255),
    }

    # 红色的第二段 (H 环绕 180°)
    RED_RANGE_2 = (170, 180, 50, 255, 50, 255)

    SATURATION_THRESH = 40    # 低于此值视为无色彩（白/黑/灰）
    VALUE_WHITE_THRESH = 200  # V > 此值且 S 低 → 白色
    VALUE_BLACK_THRESH = 60   # V < 此值且 S 低 → 黑色

    def __init__(self, method: str = "hsv_clustering"):
        self.method = method
        self.enabled = True

    def recognize(self, vehicle_crop: np.ndarray) -> str:
        """识别车辆主色调。

        Args:
            vehicle_crop: BGR 车辆裁剪区域

        Returns:
            颜色标签: white/black/grey/red/blue/green/yellow/orange/cyan/purple/pink/unknown
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return "unknown"

        hsv = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        total_pixels = h.size

        # 1. 统计白/黑/灰（低饱和度像素）
        low_sat_mask = s < self.SATURATION_THRESH
        low_sat_pixels = np.sum(low_sat_mask)

        if low_sat_pixels > 0:
            v_low_sat = v[low_sat_mask]
            white_pixels = np.sum(v_low_sat > self.VALUE_WHITE_THRESH)
            black_pixels = np.sum(v_low_sat < self.VALUE_BLACK_THRESH)
            grey_pixels = low_sat_pixels - white_pixels - black_pixels

            white_ratio = white_pixels / total_pixels
            black_ratio = black_pixels / total_pixels
            grey_ratio = grey_pixels / total_pixels

            if white_ratio > 0.35:
                return "white"
            if black_ratio > 0.35:
                return "black"
            if grey_ratio > 0.40:
                return "grey"

        # 2. 对高饱和度像素做色相投票
        high_sat_mask = s >= self.SATURATION_THRESH
        high_sat_pixels = np.sum(high_sat_mask)

        if high_sat_pixels < total_pixels * 0.05:
            # 几乎没有彩色像素，根据亮度判定
            mean_v = np.mean(v)
            if mean_v > self.VALUE_WHITE_THRESH:
                return "white"
            elif mean_v < self.VALUE_BLACK_THRESH:
                return "black"
            else:
                return "grey"

        h_high = h[high_sat_mask]

        # 对每个颜色范围计算匹配像素比例
        color_scores = {}
        for color_name, (h_min, h_max, s_min, s_max, v_min, v_max) in self.COLOR_RANGES.items():
            score = self._count_in_range(hsv, high_sat_mask, h_min, h_max, s_min, s_max, v_min, v_max)
            color_scores[color_name] = score

        # 红色特殊处理（两段H范围）
        red_score = self._count_in_range(
            hsv, high_sat_mask, self.COLOR_RANGES["red"][0], self.COLOR_RANGES["red"][1],
            *self.COLOR_RANGES["red"][2:]
        )
        red_score += self._count_in_range(hsv, high_sat_mask, *self.RED_RANGE_2)
        color_scores["red"] = red_score

        # 返回得分最高的颜色
        if color_scores:
            best_color = max(color_scores, key=color_scores.get)
            best_score = color_scores[best_color] / high_sat_pixels
            if best_score > 0.15:
                return best_color

        # 退化到亮度判定
        mean_v = np.mean(v)
        if mean_v > 160:
            return "white"
        elif mean_v < 80:
            return "black"
        else:
            return "grey"

    def _count_in_range(
        self,
        hsv: np.ndarray,
        mask: np.ndarray,
        h_min: int, h_max: int,
        s_min: int, s_max: int,
        v_min: int, v_max: int,
    ) -> int:
        """统计在给定HSV范围内的像素数。"""
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        range_mask = (
            (h >= h_min) & (h <= h_max) &
            (s >= s_min) & (s <= s_max) &
            (v >= v_min) & (v <= v_max) &
            mask
        )
        return int(np.sum(range_mask))

    def recognize_from_bbox(
        self,
        frame: np.ndarray,
        bbox: list,
    ) -> str:
        """从帧+检测框直接识别颜色。"""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            return "unknown"
        crop = frame[y1:y2, x1:x2]
        return self.recognize(crop)
