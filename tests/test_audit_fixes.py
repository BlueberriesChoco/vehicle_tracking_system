"""代码审查（223d1c79）整改的回归测试。

每个用例对应报告里的一条发现，防止同类问题再次出现。
这些都是原有测试没有覆盖的路径，所以“全量通过”不能证明它们正确。
"""

import csv
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.anomaly.alert_budget import apply_alert_budget
from src.report.metrics import collect_metrics
from src.trajectory.scene_geometry import SceneGeometry
from src.utils.threaded_reader import ThreadedFrameReader


# ---------- P1-1 告警预算只能裁撤，不能提拔 ----------

def test_alert_budget_never_promotes_normal_rows():
    rows = [{"is_anomaly": 1 if i < 2 else 0, "anomaly_score": 1.0 if i < 2 else 0.0,
             "alert_reason": "rule" if i < 2 else ""} for i in range(100)]

    result = apply_alert_budget(rows, target_rate=0.30)

    flagged = [row for row in result if row["is_anomaly"] == 1]
    assert len(flagged) == 2, "预算高于实际告警数时不允许把普通行提升为告警"


def test_alert_budget_still_caps_when_over_budget():
    rows = [{"is_anomaly": 1, "anomaly_score": float(100 - i), "alert_reason": ""} for i in range(40)]
    rows += [{"is_anomaly": 0, "anomaly_score": 0.0, "alert_reason": ""} for _ in range(60)]

    result = apply_alert_budget(rows, target_rate=0.10)

    assert sum(1 for row in result if row["is_anomaly"] == 1) == 10
    assert sum(1 for row in result if row["alert_capped"] == 1) == 30


def test_alert_budget_keeps_rule_hits_first():
    rows = [{"is_anomaly": 1, "anomaly_score": 0.1, "alert_reason": "蛇形行驶"}]
    rows += [{"is_anomaly": 1, "anomaly_score": 0.9, "alert_reason": ""} for _ in range(9)]
    rows += [{"is_anomaly": 0, "anomaly_score": 0.0, "alert_reason": ""} for _ in range(90)]

    result = apply_alert_budget(rows, target_rate=0.01)

    assert result[0]["is_anomaly"] == 1, "规则命中必须优先保留"


# ---------- P1-2 队列满时不能丢帧 ----------

class _FakeCapture:
    def __init__(self, total):
        self.total = total
        self.index = 0

    def read(self):
        if self.index >= self.total:
            return False, None
        frame = np.full((2, 2, 3), self.index, dtype=np.uint8)
        self.index += 1
        return True, frame


def test_threaded_reader_does_not_drop_frames_when_consumer_is_slow():
    """消费端停顿必须触发“队列满”，否则测不到丢帧。

    注意：只加 20ms 的小延迟是不够的——生产者 0.5 秒的 put 超时不会触发，
    旧实现也能读完 20 帧，测试就变成了假绿。这里按审查复现条件停顿 1 秒。
    """
    total = 20
    reader = ThreadedFrameReader(_FakeCapture(total), queue_size=2)

    frames = []
    for index in range(total):
        if index == 2:
            time.sleep(1.0)  # 队列容量 2，停顿期间生产者必然阻塞
        ok, frame = reader.read()
        if not ok:
            break
        frames.append(int(frame[0, 0, 0]))
    reader.stop()

    assert len(frames) == total, f"队列满时丢了帧：只读到 {len(frames)}/{total}"
    assert frames == list(range(total)), "帧顺序必须保持"


# ---------- P1-3 空窗口不该让分窗任务崩掉 ----------

def _load_script(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_audit_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_count_vector_rows_treats_header_only_csv_as_empty(tmp_path):
    module = _load_script("run_windowed")
    header_only = tmp_path / "vec_a.csv"
    header_only.write_text("track_id,camera_id\n", encoding="utf-8")
    with_rows = tmp_path / "vec_b.csv"
    with_rows.write_text("track_id,camera_id\n1,admin_in\n2,admin_in\n", encoding="utf-8")

    assert module.count_vector_rows(str(header_only)) == 0
    assert module.count_vector_rows(str(with_rows)) == 2
    assert module.count_vector_rows("") == 0


# ---------- P2-1 短轨迹训练失败必须可捕获 ----------

def test_transformer_fit_raises_on_too_few_windows():
    from src.anomaly.trajectory_transformer import TrajectoryPredictionDetector

    short = ";".join(f"{500 + i},{100}" for i in range(10))
    rows = [{"track_id": str(i), "camera_id": "cam01", "analysis_target": "1",
             "trajectory_points": short} for i in range(3)]

    detector = TrajectoryPredictionDetector(past_len=5, future_len=5, epochs=1, device="cpu")
    with pytest.raises(ValueError):
        detector.fit(rows)


# ---------- P2-2 告警率分母必须是车辆总数 ----------

def test_metrics_alert_rate_uses_vehicle_total(tmp_path):
    alerts_dir = tmp_path / "alerts"
    alerts_dir.mkdir()
    # 向量文件 10 辆车
    with open(alerts_dir / "vectors.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["track_id", "is_anomaly"])
        for i in range(10):
            writer.writerow([i, 0])
    # 告警文件只含 2 条告警（run_anomaly_detection 的 anomaly_only 行为）
    with open(alerts_dir / "alerts.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["track_id", "is_anomaly", "alert_reason"])
        for i in range(2):
            writer.writerow([i, 1, "iforest_score=0.9"])

    metrics = collect_metrics(str(tmp_path))

    assert metrics["alert_rate_percent"] == pytest.approx(20.0), "分母必须是 10 辆车，不是 2 条告警"


# ---------- P2-3 轨迹长度必须使用单应矩阵 ----------

def test_trajectory_length_uses_homography():
    homography = np.array([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 1.0]])
    geometry = SceneGeometry(
        ((0, 0), (0, 0)), ((0, 0), (0, 0)),
        px_per_meter=10.0, homography=homography,
    )
    centers = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)]

    via_homography = geometry.trajectory_length_meters(centers)
    via_constant = geometry.pixel_to_world(sum(
        float(np.hypot(centers[i][0] - centers[i - 1][0], centers[i][1] - centers[i - 1][1]))
        for i in range(1, len(centers))
    ))

    assert via_homography != pytest.approx(via_constant), "两种算法应当给出不同结果（这正是 bug 所在）"
    assert via_homography == pytest.approx(10.0 + 20.0, rel=1e-6)


def test_extractor_trajectory_length_uses_homography():
    """真正出错的入口是 TrajectoryExtractor，必须直接验证它写出的字段。"""
    from src.tracking.tracklet import Tracklet
    from src.trajectory.trajectory_extractor import TrajectoryExtractor
    from src.trajectory.trajectory_smoother import TrajectorySmoother

    homography = np.array([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 1.0]])
    geometry = SceneGeometry(
        ((0, 0), (0, 0)), ((0, 0), (0, 0)),
        px_per_meter=10.0, homography=homography,
    )
    extractor = TrajectoryExtractor(geometry, TrajectorySmoother())

    tracklet = Tracklet(track_id=1, camera_id="admin_in", class_id=2, class_name="car")
    tracklet.frame_indices = [0, 1, 2]
    tracklet.centers = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)]
    tracklet.timestamps = [None, None, None]

    summary = extractor.build_trajectory_summary(tracklet)

    # 单应矩阵下：100px 横向 = 10 m，100px 纵向 = 20 m，合计 30 m；
    # 若仍走 pixel_to_world(10 px/m)，会得到 12.73 m。
    assert summary["trajectory_length_m"] == pytest.approx(30.0, rel=5e-3)


# ---------- P2-4 同小时多段录像不能互相覆盖 ----------

def test_output_filename_suffix_separates_same_hour_segments():
    # 函数内导入：便于在未包含该辅助函数的旧版本上单独运行其余用例
    from src.pipeline.batch_processor import build_event_filename, build_vector_filename

    plain = build_vector_filename("admin_in", "20260901", "08")
    first = build_vector_filename("admin_in", "20260901", "08", "01")
    second = build_vector_filename("admin_in", "20260901", "08", "02")

    assert plain == "vec_admin_in_20260901_08.csv"
    assert len({plain, first, second}) == 3
    assert build_event_filename("admin_in", "20260901", "08", "02") == "events_admin_in_20260901_08_02.csv"


# ---------- P3-1 标注视频必须用“每帧当时”的轨迹 ----------

def test_flush_batch_draws_per_frame_tracklets():
    from src.pipeline import batch_processor as module

    class _FakePipeline:
        def process_frames(self, frames, timestamps, on_frame=None):
            outputs = []
            for index, frame in enumerate(frames):
                if on_frame is not None:
                    on_frame(frame, [f"t{index}"])  # 每帧各自不同的活跃轨迹
                outputs.append((frame, []))
            return outputs

    class _FakeVisualizer:
        def draw_frame(self, frame, tracklets):
            return f"{int(frame[0, 0, 0])}:{','.join(tracklets)}"

    class _FakeWriter:
        def __init__(self):
            self.frames = []

        def write(self, frame):
            self.frames.append(frame)

    processor = module.BatchProcessor.__new__(module.BatchProcessor)
    processor.pipeline = _FakePipeline()
    processor.visualizer = _FakeVisualizer()

    writer = _FakeWriter()
    frames = [np.full((2, 2, 3), i, dtype=np.uint8) for i in range(3)]
    processor._flush_batch(frames, [None, None, None], writer)

    assert writer.frames == ["0:t0", "1:t1", "2:t2"], "每帧必须画当时的轨迹，而不是整批最终状态"
