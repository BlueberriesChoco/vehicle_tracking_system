from typing import Tuple, Optional
import numpy as np
import cv2
import hashlib
import re


class PlateOCR:
    """车牌字符识别器。

    支持两种后端：
      - paddleocr: PaddleOCR（推荐，中文场景最优）
      - easyocr: EasyOCR（备选，纯Python无GPU也可用）

    识别结果清洗：过滤非车牌字符，格式化标准车牌号。
    """

    # 中国车牌号正则（支持蓝牌/绿牌/黄牌/白牌）
    PLATE_PATTERN = re.compile(
        r"([京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤川青藏琼宁][A-HJ-NP-Z])"
        r"([A-HJ-NP-Z0-9]{5,6})"
        r"|([A-HJ-NP-Z]{1}[A-HJ-NP-Z0-9]{4,5}[A-HJ-NP-Z]{1})"
    )

    def __init__(self, engine: str = "paddleocr", lang: str = "ch"):
        self.engine = engine
        self.lang = lang
        self.ocr = None
        self.enabled = True
        self._init_engine()

    def _init_engine(self):
        """初始化 OCR 引擎。"""
        if self.engine == "paddleocr":
            self._init_paddleocr()
        elif self.engine == "easyocr":
            self._init_easyocr()

    def _init_paddleocr(self):
        try:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang=self.lang,
                use_gpu=False,
                show_log=False,
            )
        except ImportError:
            self.ocr = None
        except Exception:
            self.ocr = None

    def _init_easyocr(self):
        try:
            import easyocr
            self.ocr = easyocr.Reader([self.lang], gpu=False)
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
        if not self.enabled or plate_image is None or plate_image.size == 0:
            return "", 0.0

        text, confidence = self._recognize_internal(plate_image)

        text = self._clean_plate(text)
        return text, round(confidence, 3)

    def _recognize_internal(self, plate_image: np.ndarray) -> Tuple[str, float]:
        """调用 OCR 引擎识别。"""
        if self.engine == "paddleocr" and self.ocr is not None:
            return self._recognize_paddle(plate_image)
        elif self.engine == "easyocr" and self.ocr is not None:
            return self._recognize_easy(plate_image)
        return "", 0.0

    def _recognize_paddle(self, plate_image: np.ndarray) -> Tuple[str, float]:
        from paddleocr import PaddleOCR
        try:
            result = self.ocr.ocr(plate_image, cls=True)
            if result and result[0]:
                texts = []
                confs = []
                for line in result[0]:
                    if line and len(line) >= 2:
                        texts.append(line[1][0])
                        confs.append(line[1][1])
                if texts:
                    combined = "".join(texts)
                    avg_conf = sum(confs) / len(confs)
                    return combined, avg_conf
        except Exception:
            pass
        return "", 0.0

    def _recognize_easy(self, plate_image: np.ndarray) -> Tuple[str, float]:
        try:
            result = self.ocr.readtext(plate_image)
            if result:
                texts = []
                confs = []
                for bbox, text, conf in result:
                    texts.append(text)
                    confs.append(conf)
                if texts:
                    combined = "".join(texts)
                    avg_conf = sum(confs) / len(confs)
                    return combined, avg_conf
        except Exception:
            pass
        return "", 0.0

    def _clean_plate(self, raw: str) -> str:
        """清洗 OCR 输出为合法车牌号格式。"""
        if not raw:
            return ""

        # 移除常见 OCR 杂讯
        cleaned = re.sub(r"[^京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤川青藏琼宁"
                         r"A-HJ-NP-Z0-9学警挂港澳领使]", "", raw.upper())

        # 纠正常见 OCR 错误
        corrections = {
            "0": "O",  # 车牌中不出现数字0，是字母O（仅限字母位）
            "1": "I",  # 同样
            "8": "B",
        }

        # 长度校验（标准车牌7~8位）
        cleaned = cleaned.strip()
        if 6 <= len(cleaned) <= 9:
            return cleaned

        return cleaned if len(cleaned) >= 5 else ""

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
