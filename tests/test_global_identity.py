"""跨机位全局身份回灌的单元测试。

覆盖：只给真正匹配上的通行打全局号、一车一节点建图、G 回填到每次通行、
按身份合并风险记录（证据去重、伙伴并集）、覆盖统计。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.global_identity import (
    annotate_global_identity,
    apply_graph_by_identity,
    build_global_graph,
    identity_key,
    identity_summary,
    merge_passages,
    merge_risk_records,
)


def _row(camera, track, enter, exit_, duration, points=12, passage=None):
    return {
        "passage_id": passage or f"{camera}_{track}",
        "camera_id": camera,
        "track_id": track,
        "analysis_target": "1",
        "is_anomaly": 0,
        "enter_time": f"2026-09-01T08:{enter}:00",
        "exit_time": f"2026-09-01T08:{exit_}:00",
        "duration_sec": duration,
        "trajectory_points": ";".join(f"{100 + i},{200 + i}" for i in range(points)),
    }


def _stitch(matches):
    """构造最小拼接载荷：global_vehicle_ids + 每个配对的 matches。"""
    global_ids = {}
    pairs = {}
    for index, (a, b, similarity) in enumerate(matches):
        gid = f"G{index + 1}"
        global_ids[a] = gid
        global_ids[b] = gid
        pairs.setdefault("camA->camB", {"matches": []})["matches"].append(
            {
                "from_camera": a.split("_")[0],
                "from_track": a.split("_")[1],
                "to_camera": b.split("_")[0],
                "to_track": b.split("_")[1],
                "from_key": a,
                "to_key": b,
                "similarity": similarity,
            }
        )
    return {"global_vehicle_ids": global_ids, "stitching": pairs}


def test_only_matched_rows_get_global_id():
    rows = {"camA": [_row("camA", "1", "01", "05", 240), _row("camA", "2", "10", "14", 240)],
            "camB": [_row("camB", "9", "02", "06", 240)]}
    stitch = _stitch([("camA_1", "camB_9", 0.91)])

    annotated = annotate_global_identity(rows, stitch)

    assert annotated["camA"][0]["global_vehicle_id"] == "G1"
    assert annotated["camB"][0]["global_vehicle_id"] == "G1"
    assert "global_vehicle_id" not in annotated["camA"][1] or not annotated["camA"][1]["global_vehicle_id"]
    # 伙伴关系也要写回，供证据链使用
    assert annotated["camA"][0]["cross_camera_partners"][0]["camera"] == "camB"


def test_merge_passages_collapses_one_vehicle_to_one_row():
    rows = [
        {**_row("camA", "1", "01", "05", 240), "global_vehicle_id": "G1"},
        {**_row("camB", "9", "07", "09", 120), "global_vehicle_id": "G1"},
        _row("camA", "2", "10", "14", 240),
    ]

    merged = merge_passages(rows)

    assert len(merged) == 2, "两次通行应合并为一行，未拼接的车独立"
    vehicle = next(row for row in merged if row["identity_key"] == "gv:G1")
    assert vehicle["passage_count"] == 2
    assert vehicle["camera_ids"] == "camA,camB"
    assert vehicle["enter_time"] == "2026-09-01T08:01:00"
    assert vehicle["exit_time"] == "2026-09-01T08:09:00"
    assert vehicle["duration_sec"] == 360


def test_global_graph_uses_identity_as_node():
    rows = [
        {**_row("camA", "1", "01", "05", 240), "global_vehicle_id": "G1"},
        {**_row("camB", "9", "02", "06", 240), "global_vehicle_id": "G1"},
    ]

    graph = build_global_graph(merge_passages(rows))

    assert graph is not None
    assert graph["node_count"] == 1, "同一辆车跨机位只应有一个节点"
    assert graph["nodes"][0]["node"] == "gv:G1"


def test_apply_graph_by_identity_fills_every_passage():
    rows = [
        {**_row("camA", "1", "01", "05", 240), "global_vehicle_id": "G1"},
        {**_row("camB", "9", "02", "06", 240), "global_vehicle_id": "G1"},
    ]
    graph = build_global_graph(merge_passages(rows))

    enriched = apply_graph_by_identity(rows, graph)

    assert all(row["graph_available"] == 1 for row in enriched)
    assert enriched[0]["graph_anomaly_index"] == enriched[1]["graph_anomaly_index"]


def test_apply_graph_by_identity_raises_when_nothing_matches():
    graph = {"nodes": [{"node": "gv:OTHER", "graph_anomaly_index": 0.1, "community_id": 0}],
             "communities": []}
    rows = [{**_row("camA", "1", "01", "05", 240), "global_vehicle_id": "G1"}]

    try:
        apply_graph_by_identity(rows, graph)
    except ValueError as exc:
        assert "完全不匹配" in str(exc)
    else:
        raise AssertionError("键完全不匹配时必须报错，否则 G 会静默为 0")


def test_merge_risk_records_agrees_on_max_risk_and_unions_evidence():
    base = {
        "risk_score_100": 62.0,
        "risk_level": "medium",
        "evidence": [{"evidence_type": "graph_anomaly", "summary": "图异常"}],
        "cross_camera_partners": [{"camera": "camB", "track": "9", "similarity": 0.91}],
    }
    rows = [
        {**_row("camA", "1", "01", "05", 240), "global_vehicle_id": "G1", **base},
        {
            **_row("camB", "9", "07", "09", 120),
            "global_vehicle_id": "G1",
            "risk_score_100": 41.0,
            "risk_level": "low",
            "evidence": [{"evidence_type": "graph_anomaly", "summary": "图异常"},
                         {"evidence_type": "rule", "summary": "蛇形行驶"}],
            "cross_camera_partners": [{"camera": "camA", "track": "1", "similarity": 0.91}],
        },
    ]

    merged = merge_risk_records(rows)

    assert len(merged) == 1
    record = merged[0]
    assert record["risk_score_100"] == 62.0
    assert record["passage_count"] == 2
    assert record["camera_ids"] == "camA,camB"
    assert len(record["evidence"]) == 2, "同一段证据应去重，新增证据保留"
    assert len(record["cross_camera_partners"]) == 2


def test_unstiched_passages_stay_separate():
    rows = [
        {**_row("camA", "1", "01", "05", 240), "risk_score_100": 10.0},
        {**_row("camA", "2", "10", "14", 240), "risk_score_100": 20.0},
    ]

    merged = merge_risk_records(rows)

    assert len(merged) == 2, "没有全局身份的通行不能被合并"
    assert identity_key(rows[0]) == "camA_1"


def test_identity_summary_reports_coverage():
    annotated = {
        "camA": [{**_row("camA", "1", "01", "05", 240), "global_vehicle_id": "G1"},
                 _row("camA", "2", "10", "14", 240)],
        "camB": [{**_row("camB", "9", "02", "06", 240), "global_vehicle_id": "G1"}],
    }
    merged = [{"camera_ids": "camA,camB"}, {"camera_ids": "camA"}]

    summary = identity_summary(annotated, merged)

    assert summary["passages"] == 3
    assert summary["passages_with_global_id"] == 2
    assert summary["global_identity_rate"] == round(2 / 3, 4)
    assert summary["vehicles_after_merge"] == 2
    assert summary["multi_camera_vehicles"] == 1


def _load_script(name):
    import importlib.util

    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_gi_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_write_back_creates_all_outputs(tmp_path):
    """脚本级：write_back 必须落盘向量、全局图、风险报告与汇总。"""
    module = _load_script("run_multi_camera")
    detected = {
        "camA": [_row("camA", "1", "01", "05", 240), _row("camA", "2", "10", "14", 240)],
        "camB": [_row("camB", "9", "02", "06", 240)],
    }
    for rows in detected.values():
        for row in rows:
            row["is_anomaly"] = "0"
            row["iforest_score"] = "0.5"

    summary = module.write_back(str(tmp_path), detected, _stitch([("camA_1", "camB_9", 0.9)]), 3)

    for name in ("graph_global.json", "risk_global.json", "summary.json"):
        assert (tmp_path / name).exists(), f"缺少 {name}"
    assert (tmp_path / "vectors" / "camA.csv").exists()
    assert (tmp_path / "vectors" / "camB.csv").exists()
    assert summary["identity"]["passages"] == 3
    assert summary["identity"]["passages_with_global_id"] == 2
    assert summary["identity"]["vehicles_after_merge"] == 2
