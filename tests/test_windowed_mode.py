"""分窗模式的跨窗轨迹连续性回归测试。

历史问题：分窗入口每个窗口都调用一次 `process_video`，而 `process_video`
开头会 `pipeline.reset()`，导致跨在窗口边界上的车辆被切成两条轨迹，
`track_id` 还会逐窗从 0 重新编号、与前一窗口撞号。

修复方式是把整段录像一次性解码、状态跨窗保留，只在开始时 reset 一次。
这些用例不依赖视频文件，用源码结构断言防止回退。
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BATCH_PROCESSOR = ROOT / "src" / "pipeline" / "batch_processor.py"
RUN_WINDOWED = ROOT / "scripts" / "run_windowed.py"


def _tree(path):
    return ast.parse(path.read_text(encoding="utf-8"))


def _function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def test_windowed_entry_uses_single_pass_api():
    """run_windowed 必须走 process_video_windows，不能逐窗调用 process_video。"""
    source = RUN_WINDOWED.read_text(encoding="utf-8")
    assert "process_video_windows" in source, "分窗入口应使用跨窗保持的 process_video_windows"

    called = set()
    for node in ast.walk(_tree(RUN_WINDOWED)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            called.add(node.func.attr)
    assert "process_video_windows" in called
    assert "process_video" not in called, "逐窗 process_video 会重置跟踪器，禁止回退"


def test_windowed_pipeline_resets_only_once():
    """_process_video_windows 全函数只能有一次 pipeline.reset()。"""
    node = _function(_tree(BATCH_PROCESSOR), "_process_video_windows")
    assert node is not None, "缺少 _process_video_windows 实现"

    resets = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if isinstance(func, ast.Attribute) and func.attr == "reset":
            value = func.value
            if isinstance(value, ast.Attribute) and value.attr == "pipeline":
                resets.append(child.lineno)
    assert len(resets) == 1, f"跨窗处理只应在开始时 reset 一次，实际 {len(resets)} 次：行 {resets}"


def test_public_window_api_exists():
    from src.pipeline.batch_processor import BatchProcessor

    assert hasattr(BatchProcessor, "process_video_windows")
    signature = ast.parse(
        Path(__file__).resolve().parents[1].joinpath("src/pipeline/batch_processor.py").read_text(encoding="utf-8")
    )
    node = _function(signature, "process_video_windows")
    names = [arg.arg for arg in node.args.args]
    assert names[:4] == ["self", "video_path", "output_dir", "window_frames"]
