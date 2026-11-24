import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.merge_camera_layouts import merge_layouts  # noqa: E402


def _layout(cameras, roads=None):
    return {"map": {"width": 1280, "height": 1809}, "cameras": cameras, "roads": roads or []}


def test_cameras_are_unioned_by_id():
    merged, conflicts = merge_layouts([
        _layout([{"id": "a", "x": 100, "y": 100}]),
        _layout([{"id": "b", "x": 200, "y": 200}]),
    ])

    assert {c["id"] for c in merged["cameras"]} == {"a", "b"}
    assert conflicts == []


def test_later_position_wins_and_conflict_is_reported():
    merged, conflicts = merge_layouts([
        _layout([{"id": "a", "x": 100, "y": 100}]),
        _layout([{"id": "a", "x": 130, "y": 100}]),
    ])

    assert merged["cameras"][0]["x"] == 130
    assert len(conflicts) == 1


def test_small_position_drift_is_not_reported():
    """同一机位两次点击差 1-2 px 属正常，不算冲突。"""
    merged, conflicts = merge_layouts([
        _layout([{"id": "a", "x": 100, "y": 100}]),
        _layout([{"id": "a", "x": 101, "y": 101}]),
    ])

    assert conflicts == []
    assert merged["cameras"][0]["x"] == 101


def test_roads_are_deduplicated():
    road = [{"x": 0, "y": 0}, {"x": 100, "y": 0}]
    merged, _ = merge_layouts([
        _layout([], [road]),
        _layout([], [list(road)]),
    ])

    assert len(merged["roads"]) == 1


def test_result_is_sorted_for_stable_diff():
    merged, _ = merge_layouts([
        _layout([{"id": "围网2", "x": 1, "y": 1}, {"id": "围网1", "x": 2, "y": 2}]),
    ])

    assert [c["id"] for c in merged["cameras"]] == ["围网1", "围网2"]
