"""风险合并工具（图谱 / 跨机位）的回归测试。"""

import logging

from src.risk.merge import apply_graph, apply_stitch, node_key


def _row(track, camera="admin_in", passage_id="", source=""):
    row = {"track_id": track, "camera_id": camera, "analysis_target": "1"}
    if passage_id:
        row["passage_id"] = passage_id
    if source:
        row["_source_file"] = source
    return row


def test_node_key():
    assert node_key({"camera_id": "admin_in", "track_id": 7}) == "admin_in:7"


def test_node_key_prefers_passage_id():
    row = {"camera_id": "admin_in", "track_id": 7, "passage_id": "admin_in_2026090108_7"}
    assert node_key(row) == "admin_in_2026090108_7"


def test_apply_graph_enriches_nodes():
    graph = {
        "nodes": [{"node": "admin_in:1", "graph_anomaly_index": 0.9, "community_id": 3}],
        "communities": [{"community_id": 3, "size": 5, "density": 0.8}],
    }

    rows = apply_graph([_row("1")], graph)

    assert rows[0]["graph_anomaly_index"] == 0.9
    assert rows[0]["community_id"] == 3
    assert rows[0]["community_size"] == 5
    assert rows[0]["community_density"] == 0.8


def test_apply_graph_defaults_for_unknown_node():
    rows = apply_graph([_row("99")], {"nodes": [], "communities": []})
    assert rows[0]["graph_anomaly_index"] == 0.0


def _stitch():
    return {
        "global_vehicle_ids": {
            "admin_in_20260901080110_1": 7,
            "admin_out_20260901080200_11": 7,
        },
        "stitching": {
            "admin_in->admin_out": {
                "matches": [
                    {
                        "from_camera": "admin_in",
                        "from_track": "1",
                        "to_camera": "admin_out",
                        "to_track": "11",
                        "from_key": "admin_in_20260901080110_1",
                        "to_key": "admin_out_20260901080200_11",
                        "similarity": 1.0,
                    }
                ]
            }
        },
    }


def test_apply_stitch_matches_by_passage_id():
    rows = apply_stitch(
        [_row("1", passage_id="admin_in_20260901080110_1"), _row("2", passage_id="admin_in_20260901080110_2")],
        _stitch(),
    )

    assert rows[0]["global_vehicle_id"] == 7
    assert rows[0]["cross_camera_partners"] == [
        {"camera": "admin_out", "track": "11", "similarity": 1.0}
    ]
    # 没有匹配上的车辆不应被标注
    assert "global_vehicle_id" not in rows[1]
    assert "cross_camera_partners" not in rows[1]


def test_apply_stitch_deduplicates_partners():
    stitch = _stitch()
    rows = apply_stitch(
        [_row("1", passage_id="admin_in_20260901080110_1", source="vec_admin_in_20260901_08.csv")],
        stitch,
    )
    partners = rows[0]["cross_camera_partners"]
    assert len(partners) == 1


def test_apply_stitch_falls_back_to_source_key():
    stitch = {
        "global_vehicle_ids": {"admin_in:vec_admin_in_20260901_08.csv:1": 9},
        "stitching": {
            "admin_in->admin_out": {
                "matches": [
                    {
                        "from_camera": "admin_in",
                        "from_track": "1",
                        "to_camera": "admin_out",
                        "to_track": "11",
                        "similarity": 0.9,
                    }
                ]
            }
        },
    }

    rows = apply_stitch(
        [_row("1", source="vec_admin_in_20260901_08.csv")],
        stitch,
    )

    assert rows[0]["global_vehicle_id"] == 9
    assert rows[0]["cross_camera_partners"][0]["camera"] == "admin_out"

def test_apply_graph_warns_when_node_keys_do_not_match(caplog):
    """旧版图谱用 camera:track 键、向量用 passage_id 时，G 会静默变 0——必须告警。"""
    rows = [_row(1, passage_id="admin_in_20260901080110_1")]
    graph = {"nodes": [{"node": "admin_in:1", "graph_anomaly_index": 0.9}], "communities": []}

    with caplog.at_level(logging.WARNING):
        enriched = apply_graph(rows, graph)

    assert enriched[0]["graph_anomaly_index"] == 0.0
    assert any("完全不匹配" in record.message for record in caplog.records)


def test_apply_graph_applies_score_when_keys_match():
    rows = [_row(1, passage_id="admin_in_20260901080110_1")]
    graph = {
        "nodes": [{"node": "admin_in_20260901080110_1", "graph_anomaly_index": 0.9, "community_id": 0}],
        "communities": [{"community_id": 0, "size": 3, "density": 0.5}],
    }

    enriched = apply_graph(rows, graph)

    assert enriched[0]["graph_anomaly_index"] == 0.9
    assert enriched[0]["community_size"] == 3

def test_apply_graph_marks_unavailable_nodes():
    """无图谱节点应标记 graph_available=0，而不是只把 G 写成 0。"""
    rows = [_row(1, passage_id="admin_in_20260901080110_1")]
    graph = {"nodes": [{"node": "admin_in:1", "graph_anomaly_index": 0.9}], "communities": []}

    enriched = apply_graph(rows, graph)

    assert enriched[0]["graph_available"] == 0


def test_apply_graph_marks_available_nodes():
    rows = [_row(1, passage_id="admin_in_20260901080110_1")]
    graph = {"nodes": [{"node": "admin_in_20260901080110_1", "graph_anomaly_index": 0.4}], "communities": []}

    enriched = apply_graph(rows, graph)

    assert enriched[0]["graph_available"] == 1
