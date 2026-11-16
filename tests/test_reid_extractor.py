"""Re-ID 特征提取器的后端选择与回退测试。"""

import numpy as np

from src.feature_extraction.reid_extractor import ReIDExtractor


def test_missing_onnx_weights_fall_back_to_histogram():
    extractor = ReIDExtractor(backend="onnx", weights_path="models/not_present.onnx")

    assert extractor.active_backend == "histogram"
    embedding = extractor.extract(np.zeros((256, 128, 3), dtype=np.uint8))
    assert embedding.shape == (74,)


def test_histogram_embedding_is_normalized():
    extractor = ReIDExtractor(backend="histogram")
    embedding = extractor.extract(np.full((256, 128, 3), 128, dtype=np.uint8))

    assert abs(float(np.linalg.norm(embedding)) - 1.0) < 1e-4


def test_empty_crop_returns_zero_vector():
    extractor = ReIDExtractor(backend="histogram")
    embedding = extractor.extract(np.zeros((0, 0, 3), dtype=np.uint8))

    assert embedding.shape == (74,)
    assert float(np.linalg.norm(embedding)) == 0.0
