from typing import Optional, Tuple, List
import numpy as np
import cv2


class PlateDetector:
    """车牌区域检测器。

    方法一（默认）：形态学 + 轮廓检测
      - 转为灰度 → Sobel 边缘 → 二值化 → 形态学闭运算 → 找矩形轮廓
      - 适合中国车牌（蓝牌/绿牌/黄牌）的矩形特征

    方法二（可选）：YOLOv8 小模型
      - 专用车牌检测模型，更鲁棒但需要额外模型文件
    """

    # 中国车牌宽高比约 3.14:1 (440×140mm)
    PLATE_ASPECT_MIN = 2.0
    PLATE_ASPECT_MAX = 5.5
    PLATE_AREA_MIN = 500      # 最小像素面积
    PLATE_AREA_MAX = 50000    # 最大像素面积

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.use_yolo = model_path is not None

        if self.use_yolo:
            self._load_yolo(model_path)

    def _load_yolo(self, model_path: str):
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
        except ImportError:
            self.use_yolo = False

    def detect(self, vehicle_crop: np.ndarray) -> Optional[Tuple[np.ndarray, list]]:
        """在车辆裁剪区域内检测车牌。

        Args:
            vehicle_crop: BGR 车辆裁剪区域

        Returns:
            (plate_image, corners) 或 None
            plate_image: 透视校正后的车牌图像
            corners: 车牌四个角点 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return None

        if self.use_yolo and self.model is not None:
            return self._detect_with_yolo(vehicle_crop)

        return self._detect_with_morphology(vehicle_crop)

    def _detect_with_morphology(self, crop: np.ndarray) -> Optional[Tuple[np.ndarray, list]]:
        """基于形态学操作的车牌检测。"""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # 直方图均衡化增强对比度
        gray = cv2.equalizeHist(gray)

        # Sobel 垂直边缘（车牌有丰富的垂直边缘）
        sobel_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        sobel_x = cv2.convertScaleAbs(sobel_x)

        # 二值化
        _, binary = cv2.threshold(sobel_x, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

        # 形态学闭运算 — 连接字符区域
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_h)

        # 找轮廓
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_plate = None
        best_score = 0.0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.PLATE_AREA_MIN or area > self.PLATE_AREA_MAX:
                continue

            # 最小外接矩形
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)

            w = rect[1][0]
            h = rect[1][1]
            if min(w, h) == 0:
                continue

            aspect_ratio = max(w, h) / min(w, h)
            if not (self.PLATE_ASPECT_MIN <= aspect_ratio <= self.PLATE_ASPECT_MAX):
                continue

            # 评分：面积 + 宽高比接近标准3.14
            ar_score = 1.0 - min(abs(aspect_ratio - 3.14) / 2.0, 1.0)
            area_score = min(area / 2000.0, 1.0)
            score = 0.5 * ar_score + 0.5 * area_score

            if score > best_score:
                best_score = score
                best_plate = self._order_corners(box)

        if best_plate is None:
            return None

        # 透视校正
        plate_img = self._perspective_correct(crop, best_plate)
        return plate_img, best_plate.tolist()

    def _detect_with_yolo(self, crop: np.ndarray) -> Optional[Tuple[np.ndarray, list]]:
        """使用 YOLO 模型检测车牌。"""
        results = self.model(crop, conf=0.3, verbose=False)

        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            if len(boxes) > 0:
                x1, y1, x2, y2 = boxes[0].astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(crop.shape[1], x2), min(crop.shape[0], y2)
                plate_img = crop[y1:y2, x1:x2]
                corners = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                return plate_img, corners

        return None

    def _order_corners(self, pts: np.ndarray) -> np.ndarray:
        """将四个角点按 (左上, 右上, 右下, 左下) 排序。"""
        rect = np.zeros((4, 2), dtype=np.float32)

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # 左上：和最小
        rect[2] = pts[np.argmax(s)]  # 右下：和最大

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # 右上：差最小
        rect[3] = pts[np.argmax(diff)]  # 左下：差最大

        return rect

    def _perspective_correct(
        self, crop: np.ndarray, corners: np.ndarray, output_size: Tuple[int, int] = (220, 70)
    ) -> np.ndarray:
        """对车牌区域做透视变换校正。"""
        dst_pts = np.array([
            [0, 0],
            [output_size[0] - 1, 0],
            [output_size[0] - 1, output_size[1] - 1],
            [0, output_size[1] - 1],
        ], dtype=np.float32)

        src_pts = corners.astype(np.float32)
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(crop, M, output_size)
        return warped
