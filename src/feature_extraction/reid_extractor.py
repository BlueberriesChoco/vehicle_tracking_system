from typing import Optional, Dict, List, Tuple
import numpy as np
import cv2


class ReIDExtractor:
    """车辆重识别特征提取器。

    提取车辆外观特征向量，用于：
    1. 跨视频段轨迹拼接（同一车辆在不同小时段视频中的关联）
    2. 断联轨迹重关联（遮挡后同一车辆再出现）

    支持两种后端：
      - torchreid (OSNet): 专用车辆 ReID 模型，效果最好
      - torchvision backbone: 无额外依赖时的备选（ResNet50 去掉分类头）
    """

    def __init__(
        self,
        model_name: str = "osnet_x1_0",
        weights_path: Optional[str] = None,
    ):
        self.model_name = model_name
        self.weights_path = weights_path
        self.model = None
        self.embedding_dim = 512
        self.enabled = False
        self._device = "cpu"

        if weights_path is not None:
            self._load_model(weights_path, model_name)

    def _load_model(self, weights_path: str, model_name: str):
        """尝试加载 ReID 模型。"""
        try:
            import torch

            if torch.cuda.is_available():
                self._device = "cuda"
        except ImportError:
            self.enabled = False
            return

        # 尝试 torchreid
        try:
            import torchreid
            self.model = torchreid.models.build_model(
                name=model_name,
                num_classes=1,
                pretrained=False,
            )
            import torch
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state_dict, strict=False)
            self.model = self.model.to(self._device)
            self.model.eval()
            self.embedding_dim = 512
            self.enabled = True
            return
        except (ImportError, Exception):
            pass

        # 尝试 fast-reid
        try:
            self._load_fastreid(weights_path)
            if self.enabled:
                return
        except Exception:
            pass

        # 退化：用 torchvision ResNet50 提取通用特征
        try:
            self._load_torchvision_backbone()
        except Exception:
            self.enabled = False

    def _load_torchvision_backbone(self):
        """使用 torchvision ResNet50 去掉分类头作为特征提取器。"""
        import torch
        import torchvision.models as models
        from torchvision import transforms

        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model = torch.nn.Sequential(*list(backbone.children())[:-1])
        self.model = self.model.to(self._device)
        self.model.eval()
        self.embedding_dim = 2048
        self.enabled = True

        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_fastreid(self, weights_path: str):
        """尝试加载 fast-reid 模型。"""
        pass  # fast-reid 加载逻辑，需要时实现

    def extract(self, vehicle_crop: np.ndarray) -> np.ndarray:
        """提取车辆外观特征向量。

        Args:
            vehicle_crop: BGR 车辆裁剪区域

        Returns:
            (embedding_dim,) 维特征向量（L2归一化后）
        """
        if not self.enabled or self.model is None:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        if vehicle_crop is None or vehicle_crop.size == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        try:
            return self._extract_internal(vehicle_crop)
        except Exception:
            return np.zeros(self.embedding_dim, dtype=np.float32)

    def _extract_internal(self, crop: np.ndarray) -> np.ndarray:
        """内部推理。"""
        import torch
        from torchvision import transforms

        # 推断使用的 transform
        if not hasattr(self, "_transform"):
            self._transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        tensor = self._transform(crop).unsqueeze(0).to(self._device)

        with torch.no_grad():
            features = self.model(tensor)
            if features.dim() > 2:
                features = features.squeeze(-1).squeeze(-1)

        vec = features.squeeze(0).cpu().numpy()

        # L2 归一化
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec.astype(np.float32)

    def extract_from_bbox(self, frame: np.ndarray, bbox: list) -> np.ndarray:
        """从帧和检测框直接提取特征。"""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        crop = frame[y1:y2, x1:x2]
        return self.extract(crop)

    def compute_similarity(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """计算两个特征向量的余弦相似度。

        Returns:
            [0, 1] 之间的相似度（1=完全相同）
        """
        if emb_a is None or emb_b is None:
            return 0.0
        if np.linalg.norm(emb_a) == 0 or np.linalg.norm(emb_b) == 0:
            return 0.0
        return float(np.dot(emb_a, emb_b))

    def match(
        self,
        query_emb: np.ndarray,
        gallery: Dict[int, np.ndarray],
        threshold: float = 0.7,
    ) -> Optional[int]:
        """在特征库中查找最佳匹配。

        Args:
            query_emb: 查询特征向量
            gallery: {track_id: embedding} 候选库
            threshold: 匹配阈值（低于此值视为不匹配）

        Returns:
            匹配的 track_id 或 None
        """
        best_id = None
        best_score = 0.0

        for tid, emb in gallery.items():
            score = self.compute_similarity(query_emb, emb)
            if score > best_score:
                best_score = score
                best_id = tid

        if best_score >= threshold:
            return best_id
        return None
