from types import SimpleNamespace

import numpy as np

from src.feature_extraction.plate_ocr import PlateOCR


def test_plate_validation_requires_province_and_correct_new_energy_format():
    assert PlateOCR._clean_plate("\u82cf\uff26\u00b7\uff11\uff12\uff13\uff14\uff15") == "\u82cfF12345"
    assert PlateOCR._clean_plate("F12345") == ""
    assert PlateOCR._clean_plate("\u82cfFA123456") == ""
    assert PlateOCR._clean_plate("\u82cfFD12345") == "\u82cfFD12345"


def test_multiple_frames_confirm_weak_reading_and_reject_tie(monkeypatch):
    ocr = PlateOCR(engine="none")
    detector = SimpleNamespace(detect=lambda image: (image, []), min_plate_width=32)
    crops = [np.zeros((16, 40, 3), dtype=np.uint8) for _ in range(3)]
    a, b = ("\u82cfF12345", 0.82), ("\u82cfF12346", 0.89)

    def vote(readings):
        results = iter(readings)
        monkeypatch.setattr(ocr, "recognize", lambda image: next(results))
        return ocr.recognize_track(crops[:len(readings)], detector)

    assert vote([a]) == ("", 0.0, 0)
    assert vote([a, b]) == ("", 0.0, 0)
    assert vote([a, b, a]) == ("\u82cfF12345", 0.82, 2)


def test_specialized_ocr_checks_plate_size_before_upscaling():
    ocr = PlateOCR(engine="none")
    ocr.ocr = lambda image: [("\u82cfF12345", 0.98, 0, [0, 0, 60, 30])]

    crop = np.zeros((48, 80, 3), dtype=np.uint8)
    assert ocr._recognize_hyperlpr(crop, 24) == ("", 0.0)
    ocr.ocr = lambda image: [("\u82cfF12345", 0.98, 0, [0, 0, 75, 30])]
    assert ocr._recognize_hyperlpr(crop, 24) == ("\u82cfF12345", 0.98)


def test_model_dir_does_not_use_windows_homepath(monkeypatch):
    """模型目录必须基于 Path.home()，不能用 HOMEPATH。

    Windows 上 HOMEPATH 只有盘符之后的路径段，**不含盘符**；用它拼路径会按
    **当前工作盘**解析。实测把模型目录解析到 D 盘，而模型下载在 C 盘用户目录，
    于是 hyperlpr3 初始化抛 NoSuchFile、静默回退 EasyOCR，
    车牌识别率从 ~90% 掉到 0.45%，日志里只有一条 warning。

    这个断言防的就是那次回退：路径必须落在真实用户目录下。
    """
    from pathlib import Path

    from src.feature_extraction.plate_ocr import HYPERLPR_MODEL_VERSION, hyperlpr_model_dir

    monkeypatch.delenv("HYPERLPR_MODEL_ROOT", raising=False)
    monkeypatch.setenv("HOMEPATH", "\\Users\\someone_else")
    path = hyperlpr_model_dir()

    assert path.is_absolute()
    assert path == (Path.home() / ".hyperlpr3" / HYPERLPR_MODEL_VERSION / "onnx").absolute()
    assert "someone_else" not in str(path)


def test_model_root_can_be_overridden_for_deployment(monkeypatch, tmp_path):
    """部署时模型可能不在用户目录，需支持 HYPERLPR_MODEL_ROOT 指定根目录。"""
    from src.feature_extraction.plate_ocr import HYPERLPR_MODEL_VERSION, hyperlpr_model_dir

    monkeypatch.setenv("HYPERLPR_MODEL_ROOT", str(tmp_path))
    assert hyperlpr_model_dir() == (tmp_path / ".hyperlpr3" / HYPERLPR_MODEL_VERSION / "onnx").absolute()
