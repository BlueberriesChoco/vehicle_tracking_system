"""指标汇总工具的测试。"""

import csv
import json

from src.report.metrics import collect_metrics, render_markdown


def _write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_collect_metrics_from_minimal_run(tmp_path):
    alerts_dir = tmp_path / "alerts"
    graphs_dir = tmp_path / "graphs"
    risk_dir = tmp_path / "risk"
    for directory in (alerts_dir, graphs_dir, risk_dir):
        directory.mkdir()

    _write_csv(
        alerts_dir / "vectors_cam.csv",
        [{"track_id": "1"}, {"track_id": "2"}, {"track_id": "3"}],
        ["track_id"],
    )
    _write_csv(
        alerts_dir / "alerts_cam.csv",
        [
            {"track_id": "1", "is_anomaly": "1", "alert_reason": "iforest_score=0.8;蛇形行驶(平均转角0.55)"},
            {"track_id": "2", "is_anomaly": "1", "alert_reason": "iforest_score=0.7"},
            {"track_id": "3", "is_anomaly": "0", "alert_reason": ""},
        ],
        ["track_id", "is_anomaly", "alert_reason"],
    )
    with open(graphs_dir / "graph_cam.json", "w", encoding="utf-8") as f:
        json.dump({"node_count": 3, "edge_count": 2, "communities": [{"community_id": 0}], "modularity": 0.5}, f)
    with open(risk_dir / "risk_cam.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "high_risk": 0,
                "medium_risk": 1,
                "vehicles": [
                    {
                        "risk_score_100": 55.0,
                        "iforest_score": 0.8,
                        "graph_anomaly_index": 0.4,
                        "time_freq_index": 0.0,
                        "evidence": [{"trace": {"source_file": "a.csv"}}, {"trace": {"source_file": ""}}],
                    },
                    {"risk_score_100": 10.0},
                ],
            },
            f,
        )

    metrics = collect_metrics(str(tmp_path))

    assert metrics["vehicles"] == 3
    assert metrics["alerts"] == 2
    assert metrics["alert_rate_percent"] == 66.67
    assert metrics["risk_medium"] == 1
    assert metrics["risk_max"] == 55.0
    assert metrics["evidence_total"] == 2
    assert metrics["evidence_traced"] == 1
    assert metrics["evidence_trace_rate_percent"] == 50.0
    assert metrics["graph_nodes"] == 3
    assert metrics["alert_reasons"] == {"iforest_score": 2, "蛇形行驶": 1}


def test_render_markdown_contains_headline_numbers():
    metrics = {
        "vehicles": 444,
        "alerts": 41,
        "alert_rate_percent": 9.23,
        "risk_high": 0,
        "risk_medium": 8,
        "risk_min": 0.0,
        "risk_median": 35.5,
        "risk_max": 59.0,
        "component_medians": {"A": 0.43, "G": 0.509, "T": 0.0},
        "evidence_total": 1320,
        "evidence_traced": 1320,
        "evidence_trace_rate_percent": 100.0,
        "graph_nodes": 432,
        "graph_edges": 1427,
        "graph_communities": 105,
        "graph_modularity": 0.7348,
        "alert_reasons": {"iforest_score": 22},
    }

    markdown = render_markdown(metrics)

    assert "**444**" in markdown
    assert "**41 / 9.23%**" in markdown
    assert "**8 / 0**" in markdown
    assert "**1320 / 100.0%**" in markdown
    assert "432 / 1427 / 105 / 0.7348" in markdown
    assert "iforest_score 22" in markdown


def test_graph_layer_is_dead_detects_zero_median():
    from src.report.metrics import graph_layer_is_dead

    assert graph_layer_is_dead({"component_medians": {"G": 0.0}}) is True
    assert graph_layer_is_dead({"component_medians": {"G": 0.4}}) is False
    assert graph_layer_is_dead({"component_medians": {}}) is True


def test_pick_top_risk_returns_highest_scoring_vehicle():
    from src.report.metrics import pick_top_risk

    payload = {
        "vehicles": [
            {"track_id": "1", "risk_score_100": 10.0},
            {"track_id": "2", "risk_score_100": 59.0},
            {"track_id": "3", "risk_score_100": 35.0},
        ]
    }

    top = pick_top_risk(payload)

    assert top["track_id"] == "2"


def test_pick_top_risk_handles_empty_payload():
    from src.report.metrics import pick_top_risk

    assert pick_top_risk({}) == {}


def test_collect_metrics_includes_benchmark(tmp_path):
    """传入吞吐基准 JSON 时，汇总结果应带上 FPS 与路数。"""
    import json

    from src.report.metrics import collect_metrics, render_markdown

    bench = tmp_path / "bench.json"
    with open(bench, "w", encoding="utf-8") as f:
        json.dump({"pipeline_fps": 125.6, "estimated_streams_at_target_fps": 8.4}, f)

    metrics = collect_metrics(str(tmp_path), benchmark_path=str(bench))
    markdown = render_markdown(metrics)

    assert metrics["pipeline_fps"] == 125.6
    assert metrics["streams"] == 8.4
    assert "125.6 FPS" in markdown
    assert "8.4 路" in markdown


# -- 图谱层健康度：口径随 G 的标定方式变更（2026-11-20）--------------------------
#
# G 从 min-max 改为分位锚点标定后（P50→0、P99→1），中位数**按定义就是 0**。
# 旧判据"中位数为 0 即失效"会恒定误报，CI 的 --require-graph 会因此在正常产物上失败。
# 新判据看"分布是否有区分度"。


def test_graph_median_zero_is_normal_under_anchor_calibration():
    """标定后中位数为 0 是预期结果，不能算失效。"""
    from src.report.metrics import graph_layer_is_dead

    metrics = {
        "component_medians": {"G": 0.0},
        "graph_score_max": 1.0,
        "graph_available_nodes": 432,
    }

    assert graph_layer_is_dead(metrics) is False


def test_graph_is_dead_when_all_scores_collapse_to_zero():
    """所有节点 G 都为 0（最大值也 0）才是真的失效。"""
    from src.report.metrics import graph_layer_is_dead

    metrics = {
        "component_medians": {"G": 0.0},
        "graph_score_max": 0.0,
        "graph_available_nodes": 432,
    }

    assert graph_layer_is_dead(metrics) is True


def test_graph_is_dead_when_no_available_nodes():
    """图谱节点键与向量键完全不匹配时，可用节点数为 0。"""
    from src.report.metrics import graph_layer_is_dead

    metrics = {
        "component_medians": {"G": 0.0},
        "graph_score_max": None,
        "graph_available_nodes": 0,
    }

    assert graph_layer_is_dead(metrics) is True


def test_graph_health_falls_back_to_median_for_legacy_payload():
    """旧产物没有 graph_score_max 字段时，沿用旧判据，避免误报失效。"""
    from src.report.metrics import graph_layer_is_dead

    assert graph_layer_is_dead({"component_medians": {"G": 0.4}}) is False
    assert graph_layer_is_dead({"component_medians": {}}) is True


# -- 图谱层健康度：口径随 G 的标定方式变更（2026-11-20）--------------------------
#
# G 从 min-max 改为分位锚点标定后（P50→0、P99→1），中位数**按定义就是 0**。
# 旧判据"中位数为 0 即失效"会恒定误报，CI 的 --require-graph 会因此在正常产物上失败。
# 新判据看"分布是否有区分度"。


def test_graph_median_zero_is_normal_under_anchor_calibration():
    """标定后中位数为 0 是预期结果，不能算失效。"""
    from src.report.metrics import graph_layer_is_dead

    metrics = {
        "component_medians": {"G": 0.0},
        "graph_score_max": 1.0,
        "graph_available_nodes": 432,
    }

    assert graph_layer_is_dead(metrics) is False


def test_graph_is_dead_when_all_scores_collapse_to_zero():
    """所有节点 G 都为 0（最大值也 0）才是真的失效。"""
    from src.report.metrics import graph_layer_is_dead

    metrics = {
        "component_medians": {"G": 0.0},
        "graph_score_max": 0.0,
        "graph_available_nodes": 432,
    }

    assert graph_layer_is_dead(metrics) is True


def test_graph_is_dead_when_no_available_nodes():
    """图谱节点键与向量键完全不匹配时，可用节点数为 0。"""
    from src.report.metrics import graph_layer_is_dead

    metrics = {
        "component_medians": {"G": 0.0},
        "graph_score_max": None,
        "graph_available_nodes": 0,
    }

    assert graph_layer_is_dead(metrics) is True


def test_graph_health_falls_back_to_median_for_legacy_payload():
    """旧产物没有 graph_score_max 字段时，沿用旧判据，避免误报失效。"""
    from src.report.metrics import graph_layer_is_dead

    assert graph_layer_is_dead({"component_medians": {"G": 0.4}}) is False
    assert graph_layer_is_dead({"component_medians": {}}) is True
