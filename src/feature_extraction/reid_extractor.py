from typing import List, Optional, Dict
import os
import numpy as np
import cv2


class ReIDExtractor:
    """车辆重识别特征提取器。

    优先级：
      1. torchreid/OSNet 权重（适合正式跨摄像头 ReID）
      2. torchvision backbone（可作为通用外观特征）
      3. HSV 颜色直方图 + 形状特征（无模型 fallback，适合先跑通流程）
    """

    def __init__(
        self,
        model_name: str = "osnet_x1_0",
        weights_path: Optional[str] = None,
        backend: str = "auto",
        allow_torchvision_fallback: bool = False,
    ):
        self.model_name = model_name
        self.weights_path = weights_path or ""
        self.backend = backend
        self.allow_torchvision_fallback = allow_torchvision_fallback
        self.model = None
        self.embedding_dim = 74
        self.enabled = True
        self._device = "cpu"
        self._active_backend = "histogram"

        try:
            import torch
            if torch.cuda.is_available():
                self._device = "cuda"
        except ImportError:
            pass

        self._init_backend()

    @property
    def active_backend(self) -> str:
        return self._active_backend

    def _init_backend(self):
        # 车辆专用 ONNX ReID（如 VeRi/CLIP-ReID）优先：它是唯一实测有区分度的后端
        if self.backend in ("auto", "onnx") and self.weights_path.endswith(".onnx") and os.path.exists(self.weights_path):
            if self._load_onnx(self.weights_path):
                return

        if self.backend in ("auto", "torchreid") and self.weights_path and os.path.exists(self.weights_path):
            self._load_model(self.weights_path, self.model_name)
            if self.enabled and self.model is not None:
                return

        if self.backend in ("auto", "torchvision") and self.allow_torchvision_fallback:
            try:
                self._load_torchvision_backbone()
                return
            except Exception:
                pass

        self.model = None
        self.embedding_dim = 74
        self.enabled = True
        self._active_backend = "histogram"

    def _load_onnx(self, weights_path: str) -> bool:
        """加载车辆 ReID ONNX 模型（输入 [B,3,S,S]，输出 [B,D]）。"""
        try:
            import onnxruntime as ort
        except ImportError:
            return False
        try:
            providers = ["CPUExecutionProvider"]
            self._session = ort.InferenceSession(weights_path, providers=providers)
            input_meta = self._session.get_inputs()[0]
            shape = input_meta.shape
            size = shape[2] if isinstance(shape[2], int) else 256
            self._input_name = input_meta.name
            self._output_name = self._session.get_outputs()[0].name
            self._onnx_size = int(size)
            self._onnx_mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
            self._onnx_std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
            self.embedding_dim = 512
            self.enabled = True
            self._active_backend = "onnx"
            return True
        except Exception:
            return False

    def _extract_onnx(self, crop: np.ndarray) -> np.ndarray:
        size = self._onnx_size
        image = cv2.resize(crop, (size, size))[:, :, ::-1].astype(np.float32) / 255.0
        image = (image - self._onnx_mean) / self._onnx_std
        tensor = image.transpose(2, 0, 1)[None]
        vec = self._session.run([self._output_name], {self._input_name: tensor})[0][0]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)

    def extract_batch(self, crops: List[np.ndarray]) -> np.ndarray:
        """批量提取特征（ONNX 后端：N 个裁剪拼成一个 batch，只推理一次）。

        非 ONNX 后端退化为逐个调用 ``extract``，结果一致、只是没有批量收益。
        批量推理是性能优化项 P0（见 PERFORMANCE.md）：单条 111ms，批量后
        单条分摊成本显著下降。
        """
        if not crops:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        if self._active_backend != "onnx":
            return np.stack([self.extract(crop) for crop in crops])

        size = self._onnx_size
        batch = np.empty((len(crops), 3, size, size), dtype=np.float32)
        for index, crop in enumerate(crops):
            image = cv2.resize(crop, (size, size))[:, :, ::-1].astype(np.float32) / 255.0
            image = (image - self._onnx_mean) / self._onnx_std
            batch[index] = image.transpose(2, 0, 1)

        vectors = self._session.run([self._output_name], {self._input_name: batch})[0]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vectors / norms).astype(np.float32)

    def _load_model(self, weights_path: str, model_name: str):
        """尝试加载专用 ReID 模型。"""
        try:
            import torch
        except ImportError:
            self.enabled = False
            return

        try:
            import torchreid
            self.model = torchreid.models.build_model(
                name=model_name,
                num_classes=1,
                pretrained=False,
            )
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
            # 预训练权重的分类头维度与原训练集 identity 数一致（如 MSMT17 为 4101），
            # 与本地 num_classes 不同；检索只需要特征，剔除 classifier.* 再加载。
            # 注意：strict=False 只忽略缺失/多余键，尺寸不匹配仍会报错。
            state_dict = {
                key: value for key, value in state_dict.items()
                if not key.startswith("classifier.")
            }
            self.model.load_state_dict(state_dict, strict=False)
            self.model = self.model.to(self._device)
            self.model.eval()
            self.embedding_dim = 512
            self.enabled = True
            self._active_backend = "torchreid"
            return
        except (ImportError, Exception):
            pass

        try:
            self._load_fastreid(weights_path)
        except Exception:
            self.enabled = False

    def _load_torchvision_backbone(self):
        """使用 torchvision ResNet50 去掉分类头作为通用特征提取器。"""
        import torch
        import torchvision.models as models
        from torchvision import transforms

        try:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            backbone = models.resnet50(weights=weights)
        except Exception:
            backbone = models.resnet50(weights=None)
        self.model = torch.nn.Sequential(*list(backbone.children())[:-1])
        self.model = self.model.to(self._device)
        self.model.eval()
        self.embedding_dim = 2048
        self.enabled = True
        self._active_backend = "torchvision"

        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_fastreid(self, weights_path: str):
        """尝试加载 fast-reid 模型。"""
        pass

    def extract(self, vehicle_crop: np.ndarray) -> np.ndarray:
        """提取车辆外观特征向量。"""
        if not self.enabled or vehicle_crop is None or vehicle_crop.size == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        try:
            if self._active_backend == "onnx":
                return self._extract_onnx(vehicle_crop)
            if self.model is None:
                return self._extract_histogram_features(vehicle_crop)
            return self._extract_internal(vehicle_crop)
        except Exception:
            return np.zeros(self.embedding_dim, dtype=np.float32)

    def extract_track(self, crops: List[np.ndarray]) -> tuple[np.ndarray, int]:
        """Fuse a bounded set of distinct observations into one embedding."""
        vectors = []
        limit = 3 if self.active_backend == "onnx" else 1
        for crop in crops[:limit]:
            vector = self.extract(crop)
            if np.all(np.isfinite(vector)) and np.linalg.norm(vector) > 0:
                vectors.append(vector)
        if not vectors:
            return np.zeros(0, dtype=np.float32), 0
        center = np.mean(vectors, axis=0)
        norm = np.linalg.norm(center)
        if norm <= 1e-8:
            return np.zeros(0, dtype=np.float32), 0
        return (center / norm).astype(np.float32), len(vectors)

    def _extract_histogram_features(self, crop: np.ndarray) -> np.ndarray:
        """无模型外观特征：HSV 直方图 + 检测框形状。"""
        resized = cv2.resize(crop, (128, 256))
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()

        h, w = crop.shape[:2]
        area = float(w * h)
        aspect = float(w / h) if h else 0.0
        mean = hsv.reshape(-1, 3).mean(axis=0) / np.array([180.0, 255.0, 255.0])
        std = hsv.reshape(-1, 3).std(axis=0) / np.array([180.0, 255.0, 255.0])
        shape = np.array([
            min(w / 1000.0, 1.0),
            min(h / 1000.0, 1.0),
            min(area / 500000.0, 1.0),
            min(aspect / 4.0, 1.0),
        ], dtype=np.float32)

        vec = np.concatenate([hist_h, hist_s, hist_v, mean, std, shape]).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def _extract_internal(self, crop: np.ndarray) -> np.ndarray:
        """内部推理。"""
        import torch
        from torchvision import transforms

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
        """计算两个特征向量的余弦相似度。"""
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
        """在特征库中查找最佳匹配。"""
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

