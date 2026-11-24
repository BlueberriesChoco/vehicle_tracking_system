from pathlib import Path

from src.site.topology import SiteTopology


def test_site_topology_loads_default_config():
    topology = SiteTopology("config/site")

    assert topology.site["site_id"] == "bonded_zone_demo"
    assert topology.validate() == []
    # 生产 cameras.yaml 已由落图工具导出的真实机位替换，演示用的 cam01 不再存在
    assert topology.cameras
    assert topology.get_camera("admin_in") is not None
    assert topology.get_camera("admin_out") is not None


def test_site_topology_reports_missing_references(tmp_path: Path):
    (tmp_path / "site.yaml").write_text("site_id: test\n", encoding="utf-8")
    (tmp_path / "zones.yaml").write_text("zones: []\n", encoding="utf-8")
    (tmp_path / "roads.yaml").write_text(
        "roads:\n  - road_id: road_a\n    zone_id: missing_zone\n",
        encoding="utf-8",
    )
    (tmp_path / "cameras.yaml").write_text(
        "cameras:\n  - camera_id: cam_x\n    zone_id: missing_zone\n    road_id: missing_road\n",
        encoding="utf-8",
    )
    (tmp_path / "topology.yaml").write_text(
        "edges:\n  - from_camera_id: cam_x\n    to_camera_id: missing_cam\n    min_travel_sec: 20\n    max_travel_sec: 10\n",
        encoding="utf-8",
    )
    (tmp_path / "routes.yaml").write_text("routes: []\n", encoding="utf-8")

    errors = SiteTopology(str(tmp_path)).validate()

    assert "road road_a references missing zone missing_zone" in errors
    assert "camera cam_x references missing zone missing_zone" in errors
    assert "camera cam_x references missing road missing_road" in errors
    assert "edge 0 references missing to_camera_id missing_cam" in errors
    assert "edge 0 has min_travel_sec greater than max_travel_sec" in errors
