from typing import List, Tuple
import numpy as np
from ultralytics import YOLO


class VehicleDetector:
    """YOLOv8 车辆检测器封装。

    输入图像帧，输出车辆检测框列表，过滤仅保留车辆类别。
    支持 FP16 推理加速。
    """

    VEHICLE_CLASSES = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    def __init__(
        self,
        model_path: str = "models/yolov8s.pt",
        img_size: int = 640,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        classes: List[int] = None,
        device: int = 0,
        half: bool = True,
    ):
        self.model = YOLO(model_path)
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes if classes is not None else [2, 3, 5, 7]
        self.device = device
        self.half = half

        if device >= 0:
            self.model.to(f"cuda:{device}")
            if half:
                self.model.model.half()

    def detect(self, frame: np.ndarray) -> List[dict]:
        """对单帧执行车辆检测。

        Returns:
            List[dict]: [{"bbox": [x1,y1,x2,y2], "confidence": float, "class_id": int, "class_name": str}, ...]
        """
        results = self.model(
            frame,
            imgsz=self.img_size,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            device=self.device,
            half=self.half,
            verbose=False,
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)

            for bbox, conf, cls_id in zip(boxes, confs, cls_ids):
                detections.append({
                    "bbox": bbox.astype(int).tolist(),
                    "confidence": float(conf),
                    "class_id": int(cls_id),
                    "class_name": self.VEHICLE_CLASSES.get(cls_id, "unknown"),
                })

        return detections

    @staticmethod
    def bbox_center(bbox: List[int]) -> Tuple[float, float]:
        """计算检测框的底部中心点（车辆接地点的合理近似）。"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, y2)

    @staticmethod
    def bbox_area(bbox: List[int]) -> float:
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)
