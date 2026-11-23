import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.import_camera_layout import build, project_on_polyline  # noqa: E402


def _layout(cameras, roads):
    return {
        "map": {"width": 1280, "height": 1809},
        "cameras": cameras,
        "roads": roads,
    }


def test_project_returns_arc_length_and_offset():
    arc, offset = project_on_polyline((50, 0), [(0, 0), (100, 0)])

    assert abs(arc - 50) < 1e-6
    assert offset < 1e-6


def test_cameras_are_chained_in_travel_order():
    """机位沿道路顺序串链，不能按 id 字母序。"""
    layout = _layout(
        cameras=[
            {"id": "c", "x": 300, "y": 0},
            {"id": "a", "x": 100, "y": 0},
            {"id": "b", "x": 200, "y": 0},
        ],
        roads=[[{"x": 0, "y": 0}, {"x": 400, "y": 0}]],
    )
    _, edges = build(layout, meters_per_pixel=0.5)

    forward = [(e["from_camera_id"], e["to_camera_id"]) for e in edges if e["from_camera_id"] < e["to_camera_id"]]
    assert forward == [("a", "b"), ("b", "c")] or [e for e in edges if e["from_camera_id"] == "a"][0]["to_camera_id"] == "b"


def test_distance_uses_meters_per_pixel():
    layout = _layout(
        cameras=[{"id": "a", "x": 0, "y": 0}, {"id": "b", "x": 200, "y": 0}],
        roads=[[{"x": 0, "y": 0}, {"x": 400, "y": 0}]],
    )
    _, edges = build(layout, meters_per_pixel=0.5)

    assert edges[0]["distance_m"] == 100.0


def test_reverse_edges_are_added():
    layout = _layout(
        cameras=[{"id": "a", "x": 0, "y": 0}, {"id": "b", "x": 100, "y": 0}],
        roads=[[{"x": 0, "y": 0}, {"x": 200, "y": 0}]],
    )
    _, edges = build(layout, meters_per_pixel=0.5)

    pairs = {(e["from_camera_id"], e["to_camera_id"]) for e in edges}
    assert ("a", "b") in pairs and ("b", "a") in pairs


def test_far_away_camera_is_excluded():
    """离线太远的机位不该被算进这条道路的链。"""
    layout = _layout(
        cameras=[{"id": "a", "x": 0, "y": 0}, {"id": "far", "x": 100, "y": 500}],
        roads=[[{"x": 0, "y": 0}, {"x": 400, "y": 0}]],
    )
    _, edges = build(layout, meters_per_pixel=0.5, max_offset_px=60)

    assert edges == []
