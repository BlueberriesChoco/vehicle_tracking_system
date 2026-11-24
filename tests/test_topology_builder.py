import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_topology import (  # noqa: E402
    build_topology,
    connectivity_report,
    derive_window,
)


def test_derive_window_uses_distance():
    """没填时间窗时，按距离与车速推算，并保证下限 20 秒。"""
    low, high = derive_window(85.0, None, None)

    assert low == 20
    assert high > low


def test_explicit_window_is_kept():
    """显式填写的时间窗优先，不被推算覆盖。"""
    low, high = derive_window(85.0, 20, 10800)

    assert (low, high) == (20, 10800)


def test_duplicate_edge_is_reported():
    rows = [
        {"from_camera_id": "a", "to_camera_id": "b", "distance_m": "30"},
        {"from_camera_id": "a", "to_camera_id": "b", "distance_m": "30"},
    ]
    edges, problems = build_topology(rows, {"a", "b"})

    assert len(edges) == 1
    assert any("重复边" in item for item in problems)


def test_unknown_camera_is_reported():
    rows = [{"from_camera_id": "a", "to_camera_id": "ghost", "distance_m": "30"}]
    edges, problems = build_topology(rows, {"a"})

    assert len(edges) == 1
    assert any("不在 cameras.yaml" in item for item in problems)


def test_connectivity_report_finds_isolated_and_endpoints():
    edges = [
        {"from_camera_id": "a", "to_camera_id": "b", "min_travel_sec": 20, "max_travel_sec": 100},
    ]
    report = connectivity_report(edges, {"a", "b", "lonely"})

    assert "lonely" in report["isolated"]
    assert report["sources"] == ["a"]
    assert report["sinks"] == ["b"]


def test_generated_file_round_trips():
    """生成的 yaml 必须能被拓扑加载器读回。"""
    from src.site.topology import SiteTopology

    path = Path("config/site/topology.yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["edges"], "topology.yaml 至少要有一条边"

    topology = SiteTopology("config/site")
    assert getattr(topology, "edges", None)


def test_dwell_allowance_extends_max_window():
    """未填 max 时要加上场区内停留预算，否则真实长停留链路会被剪掉。"""
    low, high = derive_window(30.0, None, None, dwell_sec=1800)

    assert low == 20
    assert high >= 1800


def test_override_beats_derived_window():
    """人工覆盖表优先级最高（例如行政通道 3 小时停留）。"""
    from scripts.build_topology import load_overrides

    overrides = load_overrides("config/site/topology_overrides.csv")

    assert ("admin_in", "admin_out") in overrides
    assert overrides[("admin_in", "admin_out")]["max_travel_sec"] == 10800