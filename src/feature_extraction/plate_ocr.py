from typing import Tuple
import numpy as np
import cv2
import hashlib
import logging
import os
from pathlib import Path
import re
import unicodedata


logger = logging.getLogger(__name__)
HYPERLPR_MODEL_VERSION = "20230229"
HYPERLPR_MODELS = {
    "y5fu_640x_sim.onnx": "0306de937471b87f56eb3f5620815e7e7058f8ab7428e0734fc64627cc4d716c",
    "rpv3_mdict_160_r3.onnx": "35521a46fd8beaa6fd942f969bc666f4ef5c8433d2082eae002e6c94c5d307a2",
    "litemodel_cls_96x_r1.onnx": "fe123688d3bf08b9ef029fcd57f3ac4644ac2e5ef8b9e7a676aacd5fad152142",
}


def hyperlpr_model_dir() -> Path:
    home = Path(os.environ.get("HOMEPATH", str(Path.home()))) if os.name == "nt" else Path.home()
    return (home / ".hyperlpr3").absolute() / HYPERLPR_MODEL_VERSION / "onnx"


class PlateOCR:
    """车牌字符识别器。

    支持 HyperLPR3 专用检测识别，以及 EasyOCR 和 PaddleOCR。

    识别结果清洗：过滤非车牌字符，格式化标准车牌号。
    """

    PLATE_PATTERN = re.compile(
        r"[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤川青藏琼宁][A-HJ-NP-Z]"
        r"(?:[A-HJ-NP-Z0-9]{4}[A-HJ-NP-Z0-9学警挂港澳]|"
        r"[DF][A-HJ-NP-Z0-9][0-9]{4}|[0-9]{5}[DF])"
    )

    def __init__(self, engine: str = "paddleocr", lang: str = "ch", min_confidence: float = 0.55):
        self.engine = engine
        self.lang = lang
        self.min_confidence = min_confidence
        self.ocr = None
        self.enabled = True
        self._init_engine()
        self.enabled = self.ocr is not None
        if not self.enabled:
            logger.warning("Plate OCR backend unavailable: %s (%s)", self.engine, self.lang)

    def _init_engine(self):
        """初始化 OCR 引擎。"""
        if self.engine == "paddleocr":
            self._init_paddleocr()
        elif self.engine == "easyocr":
            self._init_easyocr()
        elif self.engine == "hyperlpr3":
            try:
                # HyperLPR3 0.1.3 downloads from an HTTP endpoint at import time.
                model_dir = hyperlpr_model_dir()
                if not all((model_dir / name).is_file() for name in HYPERLPR_MODELS):
                    raise FileNotFoundError(model_dir)
                import hyperlpr3
                self.ocr = hyperlpr3.LicensePlateCatcher(
                    folder=str(model_dir.parents[1]), detect_level=hyperlpr3.DETECT_LEVEL_HIGH,
                )
            except Exception as exc:
                logger.warning("HyperLPR3 unavailable, trying EasyOCR: %s", exc)
                self.engine = "easyocr"
                self._init_easyocr()

    def _init_paddleocr(self):
        try:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(
                lang=self.lang,
                use_textline_orientation=True,
            )
        except ImportError:
            self.ocr = None
        except Exception:
            self.ocr = None

    def _init_easyocr(self):
        try:
            import easyocr
            import torch
            use_gpu = torch.cuda.is_available()
            languages = ["ch_sim", "en"] if self.lang in ("ch", "ch_sim") else [self.lang]
            self.ocr = easyocr.Reader(languages, gpu=use_gpu)
        except ImportError:
            self.ocr = None
        except Exception:
            self.ocr = None

    def recognize(self, plate_image: np.ndarray) -> Tuple[str, float]:
        """识别车牌字符。

        Args:
            plate_image: 车牌区域图像（已校正）

        Returns:
            (plate_number, confidence): 车牌号字符串和置信度
        """
        if plate_image is None or plate_image.size == 0:
            return "", 0.0

        height, width = plate_image.shape[:2]
        if width < 160:
            plate_image = cv2.resize(plate_image, (160, round(height * 160 / width)),
                                     interpolation=cv2.INTER_CUBIC)
        text, confidence = self._recognize_internal(plate_image)
        plate = self._clean_plate(text)
        if plate and confidence >= self.min_confidence:
            return plate, round(confidence, 3)
        enhanced_text, enhanced_conf = self._recognize_internal(self.enhance_plate_image(plate_image))
        enhanced_plate = self._clean_plate(enhanced_text)
        if enhanced_plate and enhanced_conf >= self.min_confidence:
            return enhanced_plate, round(enhanced_conf, 3)
        return "", round(max(confidence, enhanced_conf), 3)

    def _recognize_internal(self, plate_image: np.ndarray) -> Tuple[str, float]:
        """调用 OCR 引擎识别。"""
        if self.engine == "paddleocr" and self.ocr is not None:
            return self._recognize_paddle(plate_image)
        elif self.engine == "easyocr" and self.ocr is not None:
            return self._recognize_easy(plate_image)
        elif self.engine == "hyperlpr3" and self.ocr is not None:
            return self._recognize_hyperlpr(plate_image, 0)
        return "", 0.0

    def _recognize_paddle(self, plate_image: np.ndarray) -> Tuple[str, float]:
        try:
            if hasattr(self.ocr, "predict"):
                fragments = []
                for page in self.ocr.predict(plate_image):
                    fragments.extend(zip(page.get("rec_texts", []), page.get("rec_scores", [])))
                return self._best_reading(fragments)
            result = self.ocr.ocr(plate_image, cls=True)
            if result and result[0]:
                return self._best_reading([line[1] for line in result[0] if line and len(line) >= 2])
        except Exception:
            pass
        return "", 0.0

    def _recognize_easy(self, plate_image: np.ndarray) -> Tuple[str, float]:
        try:
            result = self.ocr.readtext(plate_image)
            result.sort(key=lambda item: min(point[0] for point in item[0]))
            return self._best_reading([(text, conf) for _, text, conf in result])
        except Exception:
            pass
        return "", 0.0

    def _best_reading(self, fragments) -> Tuple[str, float]:
        if not fragments:
            return "", 0.0
        candidates = list(fragments)
        if len(fragments) > 1:
            candidates.append(("".join(text for text, _ in fragments), min(score for _, score in fragments)))
        valid = [(self._clean_plate(text), float(score)) for text, score in candidates]
        return max(((text, score) for text, score in valid if text), key=lambda pair: pair[1], default=("", 0.0))

    @classmethod
    def _clean_plate(cls, raw: str) -> str:
        cleaned = re.sub(r"[\s·.\-]+", "", unicodedata.normalize("NFKC", raw or "").upper())
        return cleaned if cls.PLATE_PATTERN.fullmatch(cleaned) else ""

    def recognize_track(self, crops, detector) -> Tuple[str, float, int]:
        """Require agreement between independently sampled frames for weak readings."""
        readings = {}
        for crop in crops:
            if self.engine == "hyperlpr3":
                plate, confidence = self._recognize_hyperlpr(crop, detector.min_plate_width)
            else:
                result = detector.detect(crop)
                plate, confidence = self.recognize(result[0]) if result is not None else ("", 0.0)
                if not plate:
                    height, width = crop.shape[:2]
                    band = crop[int(height * 0.55):, int(width * 0.10):int(width * 0.90)]
                    if band.shape[1] >= detector.min_plate_width and band.shape[0] >= 12:
                        plate, confidence = self.recognize(band)
            if plate:
                readings.setdefault(plate, []).append(confidence)
        if not readings:
            return "", 0.0, 0
        ranked = sorted(readings.items(), key=lambda pair: len(pair[1]), reverse=True)
        plate, scores = ranked[0]
        if len(ranked) > 1 and len(scores) <= len(ranked[1][1]):
            return "", 0.0, 0
        if len(scores) == 1 and scores[0] < 0.94:
            return "", 0.0, 0
        return plate, round(sum(scores) / len(scores), 3), len(scores)

    def _recognize_hyperlpr(self, crop: np.ndarray, min_width: int) -> Tuple[str, float]:
        height, width = crop.shape[:2]
        if width < min_width or height < 12:
            return "", 0.0
        scale = min(3.0, max(1.0, 320.0 / width))
        image = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale > 1 else crop
        readings = []
        try:
            for text, confidence, _, box in self.ocr(image):
                x1, y1, x2, y2 = box
                original_width = (min(x2, image.shape[1]) - max(x1, 0)) / scale
                original_height = (min(y2, image.shape[0]) - max(y1, 0)) / scale
                plate = self._clean_plate(text)
                if plate and confidence >= self.min_confidence and original_width >= min_width and original_height >= 9:
                    readings.append((plate, float(confidence)))
        except Exception as exc:
            logger.debug("HyperLPR3 inference failed: %s", exc)
        return max(readings, key=lambda pair: pair[1], default=("", 0.0))

    def hash_plate(self, plate_number: str) -> str:
        """对车牌号做哈希，保护隐私。

        Returns:
            空字符串或8位16进制哈希
        """
        if not plate_number:
            return ""
        return hashlib.sha256(plate_number.encode()).hexdigest()[:8]

    @staticmethod
    def enhance_plate_image(plate_img: np.ndarray) -> np.ndarray:
        """增强车牌图像对比度以提升 OCR 准确率。"""
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        # CLAHE 自适应直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 回到 BGR
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

