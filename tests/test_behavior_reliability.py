from datetime import datetime, timedelta

import numpy as np

from src.behavior.features.aggregation_feature import AggregationFeatureExtractor
from src.behavior.features.speed_feature import SpeedFeatureExtractor
from src.pipeline.frame_pipeline import FramePipeline
from src.pipeline.daily_aggregator import DailyAggregator
from src.pipeline.batch_processor import BatchProcessor
from src.tracking.bytetrack_tracker import ByteTrackTracker
from src.tracking.tracklet import Tracklet
from src.trajectory.scene_geometry import SceneGeometry
from src.trajectory.trajectory_extractor import TrajectoryExtractor
from src.trajectory.trajectory_smoother import TrajectorySmoother


def _scene(px_per_meter=10.0):
    return SceneGeometry(((0, 0), (0, 0)), ((0, 0), (0, 0)), px_per_meter)


def _detection(bbox=(0, 0, 100, 100), confidence=0.9):
    return {
        "bbox": list(bbox),
        "confidence": confidence,
        "class_id": 2,
        "class_name": "car",
    }


def test_tracker_reactivates_recently_lost_track():
    tracker = ByteTrackTracker(track_buffer=3, match_thresh=0.5)
    start = datetime(2026, 6, 1, 8, 0, 0)

    tracker.update([_detection()], 0, start)
    tracker.update([], 1, start + timedelta(seconds=1))
    assert tracker.get_tracklet(0).is_lost

    tracker.update([_detection()], 2, start + timedelta(seconds=2))

    assert tracker.get_tracklet(0).is_active
    assert tracker.get_tracklet(0).trajectory_length == 2
    assert len(tracker.get_all_tracklets()) == 1


def test_aggregation_excludes_current_vehicle():
    extractor = AggregationFeatureExtractor(_scene(), aggregation_radius_m=5)

    result = extractor.extract([(10, 10)], {7: (10, 10)}, current_track_id=7)

    assert result["aggregation_index"] == 0.0
    assert result["nearest_vehicle_m"] == -1.0


def test_aggregation_uses_positions_from_matching_frames():
    extractor = AggregationFeatureExtractor(_scene(), aggregation_radius_m=5)
    histories = {
        7: {1: (0, 0), 2: (10, 0)},
        8: {1: (100, 0), 2: (100, 0)},
    }

    result = extractor.extract(
        [(0, 0), (10, 0)],
        {7: (10, 0), 8: (10, 0)},
        current_track_id=7,
        frame_indices=[1, 2],
        position_histories=histories,
    )

    assert result["aggregation_index"] == 0.0
    assert result["nearest_vehicle_m"] == 9.0


def test_speed_uses_measured_timestamp_interval():
    extractor = SpeedFeatureExtractor(_scene(px_per_meter=10), frame_rate=25)
    timestamps = [
        datetime(2026, 6, 1, 8, 0, 0),
        datetime(2026, 6, 1, 8, 0, 2),
    ]

    result = extractor.extract(np.array([0.0, 20.0]), np.array([0.0, 0.0]), timestamps)

    assert result["avg_speed_ms"] == 1.0


def test_trajectory_summary_uses_track_local_timestamp_index():
    extractor = TrajectoryExtractor(_scene(), TrajectorySmoother())
    start = datetime(2026, 6, 1, 8, 0, 0)
    tracklet = Tracklet(1, "cam01", 2, "car")
    tracklet.add_detection(100, start, [0, 0, 10, 10], (5, 10), 0.9)
    tracklet.add_detection(101, start + timedelta(seconds=1), [1, 0, 11, 10], (6, 10), 0.9)
    tracklet.mark_entered(101)

    summary = extractor.build_trajectory_summary(tracklet)

    assert summary["enter_time"] == (start + timedelta(seconds=1)).isoformat()


class _Resettable:
    def __init__(self):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1


def test_frame_pipeline_reset_clears_video_state():
    tracker = _Resettable()
    trajectory = _Resettable()
    vectorizer = _Resettable()
    pipeline = FramePipeline(None, None, tracker, trajectory, vectorizer)
    pipeline._frame_idx = 12
    pipeline._start_time = datetime(2026, 6, 1)
    pipeline._completed_vectors.append({"track_id": 1})

    pipeline.reset()

    assert tracker.reset_count == 1
    assert trajectory.reset_count == 1
    assert vectorizer.reset_count == 1
    assert pipeline.frame_idx == 0
    assert pipeline.completed_vectors == []


def test_daily_aggregator_parses_time_and_csv_numeric_values():
    aggregator = DailyAggregator()
    group = [
        {
            "camera_id": "cam01",
            "vehicle_type": "car",
            "vehicle_color": "white",
            "enter_time": "2026-06-01T08:00:00",
            "exit_time": "2026-06-01T08:00:02",
            "duration_sec": "2.0",
            "trajectory_length_m": "4.0",
            "avg_speed_ms": "2.0",
            "max_speed_ms": "3.0",
            "speed_variance": "0.2",
            "max_dwell_sec": "0",
            "stop_count": "1",
            "dwell_ratio": "0.25",
            "path_deviation": "0.1",
            "path_smoothness": "0.2",
            "aggregation_index": "0.4",
            "nearest_vehicle_m": "2.5",
        },
        {
            "camera_id": "cam01",
            "vehicle_type": "car",
            "vehicle_color": "white",
            "enter_time": "2026-06-01T09:00:00",
            "exit_time": "2026-06-01T09:00:03",
            "duration_sec": "3.0",
            "trajectory_length_m": "6.0",
            "avg_speed_ms": "2.0",
            "max_speed_ms": "4.0",
            "speed_variance": "0.4",
            "max_dwell_sec": "1",
            "stop_count": "2",
            "dwell_ratio": "0.5",
            "path_deviation": "0.2",
            "path_smoothness": "0.4",
            "aggregation_index": "0.6",
            "nearest_vehicle_m": "1.5",
        },
    ]

    merged = aggregator._merge_group(3, group)

    assert aggregator._parse_time("2026-06-01T08:00:00") == datetime(2026, 6, 1, 8)
    assert merged["duration_sec"] == 5.0
    assert merged["trajectory_length_m"] == 10.0
    assert merged["stop_count"] == 3
    assert merged["aggregation_index"] == 0.6


def test_daily_aggregator_keys_include_video_segment():
    aggregator = DailyAggregator()

    first = aggregator._get_track_key({"_segment_id": "08.csv", "track_id": "1"})
    second = aggregator._get_track_key({"_segment_id": "09.csv", "track_id": "1"})

    assert first != second
    assert aggregator._parse_embedding("[0.1, 0.2]").tolist() == [0.1, 0.2]


def test_daily_aggregator_requires_identity_evidence_for_cross_segment_merge():
    aggregator = DailyAggregator()
    previous = {
        "_segment_id": "08.csv",
        "track_id": "1",
        "exit_time": "2026-06-01T08:59:59",
        "trajectory_points": "0,0;10,10",
    }
    current = {
        "_segment_id": "09.csv",
        "track_id": "1",
        "enter_time": "2026-06-01T09:00:01",
        "trajectory_points": "11,11;20,20",
    }
    previous_key = aggregator._get_track_key(previous)
    current_key = aggregator._get_track_key(current)
    id_map = {previous_key: 3}

    aggregator._match_across_segments([previous], [current], id_map)
    assert current_key not in id_map

    previous["plate_hash"] = "same"
    current["plate_hash"] = "same"
    aggregator._match_across_segments([previous], [current], id_map)
    assert id_map[current_key] == 3


def test_parse_video_start_time_from_hourly_filename():
    parsed = BatchProcessor._parse_video_start_time("ch01_20260520_080000_090000")

    assert parsed == datetime(2026, 5, 20, 8, 0, 0)
