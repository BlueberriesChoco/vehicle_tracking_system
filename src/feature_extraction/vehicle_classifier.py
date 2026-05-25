from typing import Tuple, Optional
import numpy as np
import cv2


class VehicleClassifier:
    """车型细粒度分类器。

    结合几何特征（宽高比、面积）和可选的 CNN 模型做车型分类。
    无模型时使用启发式规则：基于检测框的宽高比和相对大小推断车型。

    类别:
      car - 小汽车（轿车/SUV）
      truck - 货车
      bus - 大巴/公交
      motorcycle - 摩托车/电瓶车
      tricycle - 三轮车
      special - 特种车辆
    """

    # 几何先验（基于典型车辆宽高比）
    ASPECT_RATIO_RANGES = {
        "motorcycle": (0.4, 0.9),   # 摩托车/电瓶车：高度 > 宽度
        "car":        (0.9, 1.6),   # 小汽车：宽高比 ~1.0-1.5
        "truck":      (1.2, 2.2),   # 货车：宽高比较大
        "bus":        (1.8, 3.5),   # 大巴：宽高比最大
    }

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
            pass  # torch 不可用时退化到几何方法
        except Exception:
            self.model = None

    def classify(self, vehicle_crop: np.ndarray, bbox: Optional[list] = None) -> str:
        """识别车型。

        Args:
            vehicle_crop: BGR 车辆裁剪区域
            bbox: [x1, y1, x2, y2] 检测框（用于几何推断）

        Returns:
            车型标签
        """
        # 优先使用 CNN 模型
        if self.model is not None and vehicle_crop is not None and vehicle_crop.size > 0:
            result = self._classify_with_model(vehicle_crop)
            if result != "unknown":
                return result

        # 退化到几何启发式
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

        # 非常小的框 → 远处车辆，保持 car
        if area < 2000:
            return "car"

        for vtype, (ar_min, ar_max) in self.ASPECT_RATIO_RANGES.items():
            if ar_min <= aspect_ratio <= ar_max:
                # 进一步用面积区分 truck vs bus
                if vtype in ("truck", "bus") and area < 15000:
                    return "car"  # 小面积大宽高比 → 可能是角度问题
                return vtype

        return "car"

    def classify_from_yolo(self, yolo_class_name: str, bbox: list) -> str:
        """从 YOLOv8 检测结果映射到细粒度类型。

        结合 YOLO 的粗分类和几何推断。
        """
        yolo_map = {
            "car": "car",
            "motorcycle": "motorcycle",
            "bus": "bus",
            "truck": "truck",
        }
        mapped = yolo_map.get(yolo_class_name, "car")

        # 对 car 类别用几何进一步区分轿车/SUV/MPV
        if mapped == "car" and bbox:
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            ar = w / h if h > 0 else 1.0
            if ar > 1.4:
                return "suv"
            elif ar > 0.95:
                return "sedan"
            else:
                return "mpv"

        return mapped
