"""检测器单元测试（Phase 4 启用）。"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest


class TestVehicleDetector:
    """测试 YOLOv8 检测器封装。"""

    def test_bbox_center(self):
        from src.detection.detector import VehicleDetector
        center = VehicleDetector.bbox_center([0, 0, 100, 200])
        assert center == (50, 200)

    def test_bbox_area(self):
        from src.detection.detector import VehicleDetector
        area = VehicleDetector.bbox_area([0, 0, 100, 200])
        assert area == 20000


class TestROIFilter:
    """测试 ROI 过滤器。"""

    def test_contains_point(self):
        from src.detection.roi_filter import ROIFilter
        roi = ROIFilter([[0, 0], [100, 0], [100, 100], [0, 100]])
        assert roi.contains_point(50, 50) is True
        assert roi.contains_point(150, 50) is False

    def test_empty_polygon_raises(self):
        from src.detection.roi_filter import ROIFilter
        with pytest.raises(ValueError):
            ROIFilter([[0, 0]])
