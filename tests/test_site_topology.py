from pathlib import Path

from src.site.topology import SiteTopology


def test_site_topology_loads_default_config():
    topology = SiteTopology("config/site")

    assert topology.site["site_id"] == "bonded_zone_demo"
    assert topology.get_camera("cam01")["road_id"] == "zone_a_nmv_entrance_lane"
    assert topology.get_camera_road("cam01")["zone_id"] == "zone_a_gate_nmv_entrance"
    assert topology.get_camera_zone("cam01")["type"] == "gate"
    assert topology.validate() == []


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
