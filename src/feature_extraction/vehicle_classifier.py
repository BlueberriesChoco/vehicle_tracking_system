from typing import Tuple, Optional
import numpy as np
import cv2


class VehicleClassifier:
    """车型细粒度分类器。

    类别:
      sedan/suv/mpv/car - 普通行政通道车辆
      truck - 货车
      bus - 大巴/公交
      motorcycle - 摩托车/电瓶车
      tricycle - 三轮车
      special - 特种车辆
    """

    # 无专用分类模型时，使用检测框宽高比做粗分；正式场景应以本地样本微调模型替代。
    ASPECT_RATIO_RANGES = {
        "motorcycle": (0.4, 0.9),
        "car": (0.9, 1.6),
        "truck": (1.6, 2.4),
        "bus": (2.2, 3.5),
    }

    ORDINARY_TYPES = {"car", "sedan", "suv", "mpv"}
    FREIGHT_TYPES = {"truck"}
    TWO_WHEELER_TYPES = {"motorcycle", "tricycle"}
    LARGE_PASSENGER_TYPES = {"bus"}

    def __init__(self, model_path: Optional[str] = None, img_size: int = 224):
        self.model_path = model_path
        self.img_size = img_size
        self.model = None
        self.enabled = True

        if model_path is not None:
            self._load_model(model_path)

    def _load_model(self, model_path: str):
        """加载分类模型（MobileNetV3 / EfficientNet）。"""
        try:
            import torch
            import torchvision.models as models

            num_classes = 8
            self.model = models.mobilenet_v3_small(weights=None, num_classes=num_classes)

            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()

            if torch.cuda.is_available():
                self.model = self.model.cuda()

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            pass
        except Exception:
            self.model = None

    def classify(self, vehicle_crop: np.ndarray, bbox: Optional[list] = None) -> str:
        """识别车型。"""
        if self.model is not None and vehicle_crop is not None and vehicle_crop.size > 0:
            result = self._classify_with_model(vehicle_crop)
            if result != "unknown":
                return result

        if bbox is not None:
            return self._classify_by_geometry(bbox)

        return "car"

    def _classify_with_model(self, crop: np.ndarray) -> str:
        """用 CNN 模型分类。"""
        import torch
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        try:
            tensor = transform(crop).unsqueeze(0).to(self._device)
            with torch.no_grad():
                output = self.model(tensor)
                pred = output.argmax(dim=1).item()
            types = ["sedan", "suv", "mpv", "truck", "bus", "motorcycle", "tricycle", "special"]
            return types[pred] if pred < len(types) else "unknown"
        except Exception:
            return "unknown"

    def _classify_by_geometry(self, bbox: list) -> str:
        """基于检测框的宽高比和面积做启发式分类。"""
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            return "car"

        aspect_ratio = w / h
        area = w * h

        if area < 2000:
            return "car"

        if aspect_ratio >= 2.2 and area >= 15000:
            return "bus"
        if aspect_ratio >= 1.6 and area >= 15000:
            return "truck"
        if aspect_ratio < 0.9:
            return "motorcycle"
        if aspect_ratio > 1.4:
            return "suv"
        if aspect_ratio > 0.95:
            return "sedan"
        return "mpv"

    def classify_from_yolo(self, yolo_class_name: str, bbox: list) -> str:
        """从 YOLOv8 检测结果映射到细粒度类型。"""
        yolo_map = {
            "car": "car",
            "motorcycle": "motorcycle",
            "bus": "bus",
            "truck": "truck",
        }
        mapped = yolo_map.get(yolo_class_name, "car")

        # YOLO 已识别为 truck/bus/motorcycle 时直接信任粗类别；只有 car 再细分。
        if mapped == "car" and bbox:
            return self._classify_by_geometry(bbox)

        return mapped

    @classmethod
    def group_vehicle_type(cls, vehicle_type: str) -> str:
        """把细粒度车型归并为业务研判分组。"""
        normalized = (vehicle_type or "unknown").strip().lower()
        if normalized in cls.ORDINARY_TYPES:
            return "ordinary"
        if normalized in cls.FREIGHT_TYPES:
            return "freight"
        if normalized in cls.TWO_WHEELER_TYPES:
            return "two_wheeler"
        if normalized in cls.LARGE_PASSENGER_TYPES:
            return "large_passenger"
        return "unknown"

    @classmethod
    def is_ordinary_vehicle(cls, vehicle_type: str) -> bool:
        return cls.group_vehicle_type(vehicle_type) == "ordinary"
