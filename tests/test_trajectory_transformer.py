"""Transformer 轨迹预测与残差判异的回归测试。"""

import math

import numpy as np

from src.anomaly import TrajectoryPredictionDetector, make_windows, normalize_window
from src.anomaly.trajectory_transformer import TrajectoryTransformer


def _smooth_traj(n=30, vx=10.0, vy=2.0, curve=2.0, seed=0):
    rng = np.random.RandomState(seed)
    x, y = 0.0, 0.0
    points = []
    for i in range(n):
        x += vx + rng.normal(0, 0.2)
        y += vy + curve * math.sin(i / 4.0)
        points.append((x, y))
    return ";".join(f"{px:.1f},{py:.1f}" for px, py in points)


def _vec(points, target=1):
    return {
        "camera_id": "cam01",
        "track_id": 1,
        "analysis_target": target,
        "vehicle_group": "ordinary",
        "trajectory_points": points,
    }


def test_make_windows_shapes_and_count():
    points = np.arange(80, dtype=np.float64).reshape(-1, 2)  # 40 个点
    windows = make_windows(points, past_len=8, future_len=8, stride=1)

    assert len(windows) == 40 - 16 + 1
    past, future = windows[0]
    assert past.shape == (8, 2)
    assert future.shape == (8, 2)


def test_make_windows_returns_empty_when_too_short():
    points = np.zeros((10, 2))
    assert make_windows(points, past_len=8, future_len=8) == []


def test_normalize_window_centers_on_last_past_point():
    past = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    future = np.array([[30.0, 0.0], [40.0, 0.0]])

    past_n, future_n, scale = normalize_window(past, future)

    assert scale == 10.0
    np.testing.assert_allclose(past_n[-1], [0.0, 0.0])
    np.testing.assert_allclose(future_n[0], [1.0, 0.0])


def test_transformer_forward_output_shape():
    model = TrajectoryTransformer(past_len=8, future_len=6, d_model=32, nhead=4, num_layers=1)
    import torch

    output = model(torch.zeros(3, 8, 2))

    assert tuple(output.shape) == (3, 6, 2)


def test_training_reduces_loss_on_smooth_trajectories():
    vectors = [_vec(_smooth_traj(seed=i)) for i in range(30)]
    detector = TrajectoryPredictionDetector(
        past_len=6,
        future_len=6,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        epochs=12,
        batch_size=32,
        device="cpu",
        random_state=7,
    )

    detector.fit(vectors, validation_split=0.2)

    history = detector.training_history
    assert history[-1]["train_loss"] < history[0]["train_loss"]


def test_anomalous_trajectory_has_higher_residual():
    vectors = [_vec(_smooth_traj(seed=i)) for i in range(40)]
    detector = TrajectoryPredictionDetector(
        past_len=6,
        future_len=6,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        epochs=15,
        batch_size=32,
        device="cpu",
        random_state=11,
    )
    detector.fit(vectors, validation_split=0.0)

    normal = _vec(_smooth_traj(seed=99))
    # 异常轨迹：最后一小段突然反向大幅偏移
    anomaly_points = _smooth_traj(seed=100).split(";")
    anomaly_points = anomaly_points[:20] + ["-500.0,900.0", "-520.0,920.0", "-540.0,940.0",
                                            "-560.0,960.0", "-580.0,980.0", "-600.0,1000.0"]
    anomalous = _vec(";".join(anomaly_points))

    scored = detector.score([normal, anomalous])

    assert scored[1]["traj_residual"] > scored[0]["traj_residual"]


def test_save_and_load_roundtrip(tmp_path):
    vectors = [_vec(_smooth_traj(seed=i)) for i in range(20)]
    detector = TrajectoryPredictionDetector(
        past_len=5,
        future_len=5,
        d_model=16,
        nhead=2,
        num_layers=1,
        dim_feedforward=32,
        epochs=3,
        device="cpu",
        random_state=3,
    )
    detector.fit(vectors, validation_split=0.0)

    prefix = str(tmp_path / "traj_model")
    detector.save(prefix)
    loaded = TrajectoryPredictionDetector.load(prefix)

    normal = _vec(_smooth_traj(seed=77))
    original_score = detector.score([normal])[0]["traj_residual"]
    loaded_score = loaded.score([normal])[0]["traj_residual"]

    assert abs(original_score - loaded_score) < 1e-6


def test_fit_reuses_existing_weights_for_incremental_update():
    vectors = [_vec(_smooth_traj(seed=i)) for i in range(24)]
    detector = TrajectoryPredictionDetector(
        past_len=5,
        future_len=5,
        d_model=16,
        nhead=2,
        num_layers=1,
        dim_feedforward=32,
        epochs=3,
        device="cpu",
        random_state=5,
    )
    detector.fit(vectors, validation_split=0.0)
    model_before = detector.model

    detector.epochs = 2
    detector.fit(vectors, validation_split=0.0, reuse_existing=True)

    # 复用同一模型对象，不重新初始化
    assert detector.model is model_before
    assert len(detector.training_history) == 2


def test_fit_tolerates_vectors_that_were_already_scored():
    """重训场景：输入向量已带 traj_residual，但轨迹太短不产生窗口。

    旧实现会在 record["traj_residual_zscore"] 上抛 KeyError，
    导致「用上次的输出重训」这条正常路径直接崩溃。
    """
    from src.anomaly.trajectory_transformer import TrajectoryPredictionDetector

    def row(i, points):
        return {
            "track_id": str(i),
            "camera_id": "cam01",
            "analysis_target": "1",
            "trajectory_points": points,
            # 模拟上一次运行写入的残差
            "traj_residual": 12.5,
        }

    long_track = ";".join(f"{500 + 10 * s},{100}" for s in range(40))
    short_track = ";".join(f"{500 + 10 * s},{100}" for s in range(3))
    rows = [row(i, long_track) for i in range(8)] + [row(100 + i, short_track) for i in range(3)]

    detector = TrajectoryPredictionDetector(past_len=5, future_len=5, epochs=1, device="cpu")
    detector.fit(rows)  # 不应抛异常

    assert detector.model is not None

