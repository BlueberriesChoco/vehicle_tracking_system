"""Re-ID 批量推理的正确性与性能说明测试。

`extract_batch` 的结果必须与逐条 `extract` 完全一致；
性能上本机 CPU 批量反而更慢（见 PERFORMANCE.md），因此默认不启用批量，
只为将来换 GPU provider 保留能力。
"""

import os

import numpy as np
import pytest

from src.feature_extraction.reid_extractor import ReIDExtractor

ONNX_WEIGHTS = "models/vehicle_reid.onnx"


def test_batch_matches_single_on_histogram_backend():
    """无 ONNX 权重时走直方图回退，批量只是逐个调用的封装，必须完全一致。"""
    extractor = ReIDExtractor(backend="histogram")
    crops = [
        np.full((64, 64, 3), value, dtype=np.uint8)
        for value in (30, 90, 150, 210)
    ]

    single = np.stack([extractor.extract(crop) for crop in crops])
    batch = extractor.extract_batch(crops)

    assert batch.shape == single.shape
    assert np.allclose(batch, single)


def test_empty_batch_returns_empty_matrix():
    extractor = ReIDExtractor(backend="histogram")

    result = extractor.extract_batch([])

    assert result.shape[0] == 0


@pytest.mark.skipif(not os.path.exists(ONNX_WEIGHTS), reason="ONNX ReID weights not present")
def test_onnx_batch_matches_single():
    extractor = ReIDExtractor(backend="onnx", weights_path=ONNX_WEIGHTS)
    rng = np.random.RandomState(0)
    crops = [rng.randint(0, 255, (120, 220, 3), dtype=np.uint8) for _ in range(3)]

    single = np.stack([extractor.extract(crop) for crop in crops])
    batch = extractor.extract_batch(crops)

    for index in range(len(crops)):
        assert float(np.dot(single[index], batch[index])) > 0.999
