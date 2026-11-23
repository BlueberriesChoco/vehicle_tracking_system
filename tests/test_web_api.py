"""控制台后端与待核查清单的单测。

只构造假 run 目录，不依赖真实产物；另有一条 HTTP 冒烟（随机端口）验证路由可用。
"""

import csv
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.report.review_list import build_rows, rank_vehicles
from src.web import server as web


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fake_root(tmp_path: Path) -> Path:
    root = tmp_path / "outputs"
    run = root / "demo_run"
    _write_json(
        run / "risk" / "risk_admin_in.json",
        {
            "vehicle_count": 2,
            "medium_risk": 1,
            "high_risk": 0,
            "vehicles": [
                {
                    "passage_id": "p_low",
                    "camera_id": "admin_in",
                    "enter_time": "2026-09-01T08:01:00",
                    "duration_sec": 2.0,
                    "risk_score_100": 10.0,
                    "risk_level": "low",
                    "risk_components": {"A": 0.2, "G": 0.1, "T": 0.0},
                    "evidence": [{"evidence_type": "graph_anomaly", "summary": "图异常偏低",
                                  "trace": {"source_file": "vec.csv", "camera_id": "admin_in", "track_id": "1"}}],
                },
                {
                    "passage_id": "p_high",
                    "camera_id": "admin_in",
                    "enter_time": "2026-09-01T08:05:00",
                    "duration_sec": 3.0,
                    "risk_score_100": 58.0,
                    "risk_level": "medium",
                    "risk_components": {"A": 0.8, "G": 0.9, "T": 0.0},
                    "evidence": [
                        {"evidence_type": "graph_anomaly", "summary": "图异常偏高"},
                        {"evidence_type": "rule", "summary": "蛇形行驶"},
                    ],
                },
            ],
        },
    )
    _write_csv(
        run / "alerts" / "vectors_admin_in.csv",
        [{"passage_id": "p_low", "is_anomaly": "0"}, {"passage_id": "p_high", "is_anomaly": "1"}],
        ["passage_id", "is_anomaly"],
    )
    _write_csv(
        run / "alerts" / "alerts_admin_in.csv",
        [{"passage_id": "p_high", "is_anomaly": "1"}],
        ["passage_id", "is_anomaly"],
    )
    _write_json(
        run / "graphs" / "graph_admin_in.json",
        {"node_count": 2, "edge_count": 1, "modularity": 0.5,
         "communities": [{"community_id": 0, "size": 2}],
         "graph_anomaly_method": "low_rank_svd"},
    )
    (run / "vectors").mkdir(exist_ok=True)
    return root


def test_review_list_ranks_by_score():
    vehicles = [{"passage_id": "a", "risk_score_100": 10}, {"passage_id": "b", "risk_score_100": 90}]

    rows = build_rows(rank_vehicles(vehicles, budget=2))

    assert [row["passage_id"] for row in rows] == ["b", "a"]
    assert rows[0]["rank"] == 1


def test_review_list_budget_zero_returns_all():
    vehicles = [{"passage_id": str(i), "risk_score_100": i} for i in range(5)]
    assert len(rank_vehicles(vehicles, budget=0)) == 5


def test_list_runs_reports_counts(tmp_path):
    root = _fake_root(tmp_path)

    runs = web.list_runs(root)

    assert len(runs) == 1
    run = runs[0]
    assert run["vehicles"] == 2
    assert run["alerts"] == 1
    assert run["medium"] == 1
    assert run["model_version"].startswith("inferred:")
    assert run["model_declared"] is False


def test_run_detail_includes_graph_and_checks(tmp_path):
    root = _fake_root(tmp_path)
    run_id = web.list_runs(root)[0]["id"]

    detail = web.run_detail(root, run_id)

    assert detail["graph"]["nodes"] == 2
    assert detail["graph"]["method"] == "low_rank_svd"
    assert detail["checks"]["model_declared"] is False
    assert detail["checks"]["calibrated"] in (True, False)


def test_review_rows_respects_budget(tmp_path):
    root = _fake_root(tmp_path)
    run_id = web.list_runs(root)[0]["id"]

    rows = web.review_rows(root, run_id, budget=1)

    assert len(rows) == 1
    assert rows[0]["passage_id"] == "p_high"


def test_vehicle_record_found_and_missing(tmp_path):
    root = _fake_root(tmp_path)
    run_id = web.list_runs(root)[0]["id"]

    assert web.vehicle_record(root, run_id, "p_high")["risk_score_100"] == 58.0
    assert web.vehicle_record(root, run_id, "nope") is None


def test_artifact_path_blocks_traversal(tmp_path):
    root = _fake_root(tmp_path)
    run_id = web.list_runs(root)[0]["id"]

    inside = web.artifact_path(root, run_id, "graphs/graph_admin_in.json")
    assert inside is not None and inside.is_file()
    assert web.artifact_path(root, run_id, "../../config/default.yaml") is None


def test_unknown_run_returns_none(tmp_path):
    root = _fake_root(tmp_path)
    assert web.run_detail(root, "does_not_exist") is None
    assert web.review_rows(root, "does_not_exist") is None


def test_http_smoke(tmp_path):
    root = _fake_root(tmp_path)
    server = web.serve("127.0.0.1", 0, root, allow_runs=False)
    port = server.server_address[1]
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=5) as response:
            health = json.loads(response.read())
        assert health["ok"] is True

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/runs", timeout=5) as response:
            runs = json.loads(response.read())
        assert len(runs) == 1

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/models", timeout=5) as response:
            models = json.loads(response.read())
        assert "versions" in models

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as response:
            assert b"<title>" in response.read()
    finally:
        server.shutdown()


def test_post_runs_rejected_when_read_only(tmp_path):
    root = _fake_root(tmp_path)
    server = web.serve("127.0.0.1", 0, root, allow_runs=False)
    port = server.server_address[1]
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/runs",
            data=json.dumps({"input": "x.h264"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(request, timeout=5)
        except urllib.error.HTTPError as error:
            assert error.code == 403
        else:
            raise AssertionError("只读模式应返回 403")
    finally:
        server.shutdown()


def test_thumbnail_endpoint_serves_crop(tmp_path):
    root = _fake_root(tmp_path)
    run_id = web.list_runs(root)[0]["id"]
    crops = root / "demo_run" / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    (crops / "p_high.jpg").write_bytes(b"\xff\xd8\xff\xd9")

    server = web.serve("127.0.0.1", 0, root, allow_runs=False)
    port = server.server_address[1]
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/runs/{run_id}/vehicles/p_high/thumbnail", timeout=5
        ) as response:
            assert response.status == 200
            assert response.headers["Content-Type"] == "image/jpeg"
            assert response.read().startswith(b"\xff\xd8")
    finally:
        server.shutdown()


def test_thumbnail_missing_returns_404(tmp_path):
    root = _fake_root(tmp_path)
    run_id = web.list_runs(root)[0]["id"]

    server = web.serve("127.0.0.1", 0, root, allow_runs=False)
    port = server.server_address[1]
    try:
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/runs/{run_id}/vehicles/unknown/thumbnail", timeout=5
            )
        except urllib.error.HTTPError as error:
            assert error.code == 404
        else:
            raise AssertionError("没有截图时应返回 404")
    finally:
        server.shutdown()


def test_plate_endpoint_and_display_fields(tmp_path):
    root = _fake_root(tmp_path)
    run_id = web.list_runs(root)[0]["id"]
    crops = root / "demo_run" / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    (crops / "p_high_plate.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    (crops / "index.json").write_text(
        json.dumps({"p_high": {"file": "p_high.jpg", "plate_file": "p_high_plate.jpg",
                               "plate_text": "A37622", "plate_conf": 0.9}}, ensure_ascii=False),
        encoding="utf-8",
    )

    record = web.vehicle_record(root, run_id, "p_high")
    assert record["display"]["plate_text"] == "A37622"
    assert record["display"]["has_plate_image"] is True

    server = web.serve("127.0.0.1", 0, root, allow_runs=False)
    port = server.server_address[1]
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/runs/{run_id}/vehicles/p_high/plate", timeout=5
        ) as response:
            assert response.status == 200
            assert response.headers["Content-Type"] == "image/jpeg"
    finally:
        server.shutdown()


def test_plate_missing_returns_404(tmp_path):
    root = _fake_root(tmp_path)
    run_id = web.list_runs(root)[0]["id"]

    server = web.serve("127.0.0.1", 0, root, allow_runs=False)
    port = server.server_address[1]
    try:
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/runs/{run_id}/vehicles/p_high/plate", timeout=5
            )
        except urllib.error.HTTPError as error:
            assert error.code == 404
        else:
            raise AssertionError("没有车牌条时应返回 404")
    finally:
        server.shutdown()


def test_plate_override_file_and_post(tmp_path):
    root = _fake_root(tmp_path)
    run_id = web.list_runs(root)[0]["id"]
    run_dir = root / "demo_run"

    assert web.save_plate_override(run_dir, "p_high", "苏F·A37622") is True
    assert web.load_plate_overrides(run_dir)["p_high"] == "苏F·A37622"
    record = web.vehicle_record(root, run_id, "p_high")
    assert record["display"]["plate"] == "苏F·A37622"
    assert record["display"]["plate_confirmed"] is True

    server = web.serve("127.0.0.1", 0, root, allow_runs=True)
    port = server.server_address[1]
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/runs/{run_id}/vehicles/p_high/plate",
            data=json.dumps({"plate": "苏F·B88888"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read())
        assert payload["ok"] is True
        assert web.load_plate_overrides(run_dir)["p_high"] == "苏F·B88888"
    finally:
        server.shutdown()


def test_plate_post_rejected_when_read_only(tmp_path):
    root = _fake_root(tmp_path)
    run_id = web.list_runs(root)[0]["id"]
    server = web.serve("127.0.0.1", 0, root, allow_runs=False)
    port = server.server_address[1]
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/runs/{run_id}/vehicles/p_high/plate",
            data=json.dumps({"plate": "苏F·C00000"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(request, timeout=5)
        except urllib.error.HTTPError as error:
            assert error.code == 403
        else:
            raise AssertionError("只读模式应拒绝写车牌")
    finally:
        server.shutdown()


def test_review_sort_modes(tmp_path):
    root = _fake_root(tmp_path)
    run_id = web.list_runs(root)[0]["id"]

    by_risk = web.review_rows(root, run_id, budget=2, sort="risk")
    by_behavior = web.review_rows(root, run_id, budget=2, sort="behavior")

    assert by_risk[0]["passage_id"] == "p_high"
    assert by_behavior[0]["passage_id"] == "p_high"
    assert "parts" in by_risk[0]


# -- 控制台读旧口径产物（2026-11-20）--------------------------------------------
#
# 三个缺陷，共同点是"界面显示的东西和磁盘上最新的产物不一致"：
# 1. `_first()` 按文件名字母序取第一个 → `risk_admin_in.json` 排在
#    `risk_admin_in_v2.json` 前面，控制台永远显示旧口径；
# 2. 只匹配 `risk_*.json` → `risk.json`（无下划线）读不到，整批显示为空；
# 3. 清单直接遍历 risk.vehicles，未排除 non_target_vehicles。


def _write_risk(run: Path, name: str, payload: dict, mtime: float) -> Path:
    path = run / "risk" / name
    _write_json(path, payload)
    import os
    os.utime(path, (mtime, mtime))
    return path


def _risk_payload(scores, medium=None, high=0, extra_vehicles=None):
    vehicles = [
        {
            "passage_id": f"p{i}",
            "camera_id": "admin_in",
            "analysis_target": "1",
            "risk_score_100": score,
            "risk_level": "medium" if score >= 50 else "low",
            "risk_components": {"A": 0.5, "G": 0.5, "T": 0.0},
        }
        for i, score in enumerate(scores)
    ]
    payload = {
        "version": 2,
        "vehicle_count": len(vehicles),
        "medium_risk": medium if medium is not None else sum(1 for s in scores if s >= 50),
        "high_risk": high,
        "vehicles": vehicles,
    }
    if extra_vehicles:
        payload["vehicles"].extend(extra_vehicles)
    return payload


def _bare_run(tmp_path: Path, name: str = "run") -> Path:
    run = tmp_path / "outputs" / name
    (run / "risk").mkdir(parents=True, exist_ok=True)
    (run / "alerts").mkdir(exist_ok=True)
    (run / "graphs").mkdir(exist_ok=True)
    return run


def test_first_picks_newest_file_not_alphabetical(tmp_path):
    """`_v2` 字母序在后，但更新的那个必须被选中。

    这是控制台长期显示旧口径的直接原因：实测 full_rerun 批次磁盘上已有
    `risk_admin_in_v2.json`（38 中风险 / 最高 80.2），界面却一直显示
    `risk_admin_in.json` 的（15 中风险 / 最高 61.3）。
    """
    run = _bare_run(tmp_path)
    _write_risk(run, "risk_admin_in.json", _risk_payload([61.3], medium=15), mtime=1_700_000_000)
    newest = _write_risk(run, "risk_admin_in_v2.json", _risk_payload([80.2], medium=38), mtime=1_700_000_900)

    assert web._first(run / "risk", "risk*.json") == newest


def test_first_ties_are_deterministic(tmp_path):
    """mtime 相同时按名称兜底，避免同一份数据两次调用给出不同文件。"""
    run = _bare_run(tmp_path)
    _write_risk(run, "risk_a.json", _risk_payload([1.0]), mtime=1_700_000_000)
    _write_risk(run, "risk_b.json", _risk_payload([2.0]), mtime=1_700_000_000)

    first = web._first(run / "risk", "risk*.json")
    second = web._first(run / "risk", "risk*.json")

    assert first == second
    assert first.name == "risk_b.json"


def test_risk_json_without_camera_suffix_is_read(tmp_path):
    """`risk.json`（无下划线）必须能读到——旧冒烟产物就是这个命名。"""
    run = _bare_run(tmp_path, "smoke")
    _write_risk(run, "risk.json", _risk_payload([75.0], medium=3), mtime=1_700_000_000)

    assert web._risk_path(run) is not None
    assert web._risk_path(run).name == "risk.json"

    runs = web.list_runs(tmp_path / "outputs")
    assert runs[0]["medium"] == 3
    assert runs[0]["risk_max"] == 75.0


def test_review_rows_excludes_non_target_vehicles(tmp_path):
    """非监管对象不进「待核查车辆」——它们没有风险分，列出会被读成"0 分正常车"。"""
    run = _bare_run(tmp_path)
    _write_json(
        run / "risk" / "risk_admin_in.json",
        {
            "version": 2,
            "vehicle_count": 2,
            "medium_risk": 1,
            "high_risk": 0,
            "vehicles": [
                {"passage_id": "p_ok", "analysis_target": "1",
                 "risk_score_100": 60.0, "risk_components": {"A": 0.5, "G": 0.5, "T": 0.0}},
                {"passage_id": "p_truck", "analysis_target": "0",
                 "risk_score_100": 0.0, "risk_components": {"A": 0.0, "G": 0.0, "T": 0.0}},
            ],
        },
    )

    rows = web.review_rows(tmp_path / "outputs", "run", budget=0)

    assert [row["passage_id"] for row in rows] == ["p_ok"]


def test_analysis_target_zero_is_not_falsy_trap():
    """`analysis_target=0` 必须判为 False。

    写成 ``str(value or "")`` 会把整数 0 吞成空串，于是货车被当成监管对象——
    这是 `build_risk_report.py` 里踩过的同一个坑，控制台不能重犯。
    """
    assert web._is_analysis_target({"analysis_target": 0}) is False
    assert web._is_analysis_target({"analysis_target": "0"}) is False
    assert web._is_analysis_target({"analysis_target": 0.0}) is False
    assert web._is_analysis_target({"analysis_target": "False"}) is False
    assert web._is_analysis_target({"analysis_target": 1}) is True
    assert web._is_analysis_target({"analysis_target": "1"}) is True
    # 字段缺失 = 旧产物，所有车都参与过评分
    assert web._is_analysis_target({}) is True


def test_summarize_run_counts_non_target(tmp_path):
    run = _bare_run(tmp_path)
    _write_json(
        run / "risk" / "risk_admin_in.json",
        {
            "version": 2,
            "vehicle_count": 2,
            "medium_risk": 1,
            "high_risk": 0,
            "non_target_count": 1,
            "non_target_vehicles": [{"passage_id": "p_truck", "analysis_target": "0"}],
            "vehicles": [
                {"passage_id": "p_ok", "analysis_target": "1", "risk_score_100": 60.0},
                {"passage_id": "p_ok2", "analysis_target": "1", "risk_score_100": 20.0},
            ],
        },
    )

    summary = web.summarize_run(run, tmp_path / "outputs")

    assert summary["vehicles"] == 2
    assert summary["non_target_count"] == 1
    assert summary["total_vehicles"] == 3
    assert summary["risk_max"] == 60.0  # 非监管对象的 0 分不拉低/不参与


def test_summarize_run_detects_non_target_in_legacy_payload(tmp_path):
    """旧产物没有 non_target_vehicles 字段时，按 analysis_target 现算。"""
    run = _bare_run(tmp_path)
    _write_json(
        run / "risk" / "risk_admin_in.json",
        {
            "vehicle_count": 2,
            "medium_risk": 0,
            "high_risk": 0,
            "vehicles": [
                {"passage_id": "p_ok", "analysis_target": "1", "risk_score_100": 30.0},
                {"passage_id": "p_truck", "analysis_target": "0", "risk_score_100": 0.0},
            ],
        },
    )

    summary = web.summarize_run(run, tmp_path / "outputs")

    assert summary["non_target_count"] == 1
    # 旧产物 vehicle_count=2 含非监管对象，不再重复加
    assert summary["total_vehicles"] == 3


def test_score_parts_uses_zero_gamma_not_nominal():
    """重归一化后 γ=0.0 不能被 `or 0.25` 回填成名义权重。

    修前：界面显示"频次占 25%"，同时又说"频次无数据"，自相矛盾。
    """
    record = {
        "risk_score_100": 80.2,
        "risk_components": {"A": 0.6, "G": 1.0, "T": 0.0,
                            "alpha": 0.5333, "beta": 0.4667, "gamma": 0.0},
        "risk_component_available": {"A": 1, "G": 1, "T": 0},
    }

    parts = web.score_parts(record)

    assert parts["weights"]["T"] == 0.0
    assert parts["points"]["T"] == 0.0
    assert parts["unavailable"] == ["T"]
    assert parts["available_count"] == 2


def test_score_parts_total_matches_reported_score():
    """拆分后的总分必须等于融合层写回的 R，否则列表与详情显示不一致。"""
    record = {
        "risk_score_100": 80.2,
        "risk_components": {"A": 0.6, "G": 1.0, "T": 0.0,
                            "alpha": 0.5333, "beta": 0.4667, "gamma": 0.0},
        "risk_component_available": {"A": 1, "G": 1, "T": 0},
    }

    parts = web.score_parts(record)

    # 三项相加会得到 80.3（四舍五入误差），必须以 R 为准
    assert parts["total"] == 80.2


def test_score_parts_reports_capping():
    """封顶信息要透到前端，否则 R=75 会被当成原始分。"""
    record = {
        "risk_score_100": 75.0,
        "risk_score_raw_100": 100.0,
        "risk_score_ceiling": 75.0,
        "risk_score_capped": 1,
        "risk_components": {"A": 0.0, "G": 1.0, "T": 0.0,
                            "alpha": 0.0, "beta": 1.0, "gamma": 0.0},
        "risk_component_available": {"A": 0, "G": 1, "T": 0},
    }

    parts = web.score_parts(record)

    assert parts["capped"] is True
    assert parts["ceiling"] == 75.0
    assert parts["raw_total"] == 100.0
    assert parts["available_count"] == 1


def test_score_parts_legacy_record_has_no_capping():
    """旧产物没有封顶字段：不能误报成"已封顶"。"""
    record = {
        "risk_score_100": 59.0,
        "risk_components": {"A": 0.62, "G": 0.97, "T": 0.0,
                            "alpha": 0.4, "beta": 0.35, "gamma": 0.25},
    }

    parts = web.score_parts(record)

    assert parts["capped"] is False
    assert parts["ceiling"] == 100.0
    assert parts["raw_total"] is None


def test_score_parts_shares_use_raw_base_under_capping():
    """封顶时份额的分母要用原始加权分，不能除以封顶后的总分。

    单路分量贡献 100 分、被压到 75 分后，若按 100/75 算份额会得到 133%，
    界面出现"关系 100（133%）"这种不可能的占比。
    """
    record = {
        "risk_score_100": 75.0,
        "risk_score_raw_100": 100.0,
        "risk_score_ceiling": 75.0,
        "risk_score_capped": 1,
        "risk_components": {"A": 0.0, "G": 1.0, "T": 0.0,
                            "alpha": 0.0, "beta": 1.0, "gamma": 0.0},
        "risk_component_available": {"A": 0, "G": 1, "T": 0},
    }

    parts = web.score_parts(record)

    assert parts["shares"]["G"] == 100
    assert sum(parts["shares"].values()) == 100


def test_score_parts_shares_sum_to_hundred():
    """三路分量齐全时份额必须正好 100%，四舍五入的尾差要收掉。"""
    record = {
        "risk_score_100": 80.2,
        "risk_components": {"A": 0.6, "G": 1.0, "T": 0.0,
                            "alpha": 0.5333, "beta": 0.4667, "gamma": 0.0},
        "risk_component_available": {"A": 1, "G": 1, "T": 1},
    }

    parts = web.score_parts(record)

    assert all(0 <= value <= 100 for value in parts["shares"].values())
    assert sum(parts["shares"].values()) == 100


def test_run_detail_infers_comparability_for_legacy_graph(tmp_path):
    """旧图谱没有 comparable 字段时按 method 推断，避免详情页漏提示。"""
    root = _fake_root(tmp_path)
    run_id = web.list_runs(root)[0]["id"]

    detail = web.run_detail(root, run_id)

    assert detail["graph"]["comparable"] is False
    assert detail["graph"]["self_referential"] is True


def test_run_detail_reads_declared_comparability(tmp_path):
    """新版图谱显式声明 gae_baseline 时，详情页必须显示为可比。"""
    run = _bare_run(tmp_path, "baseline_run")
    _write_json(run / "risk" / "risk_admin_in.json", _risk_payload([10.0]))
    _write_json(
        run / "graphs" / "graph_admin_in.json",
        {
            "node_count": 2,
            "edge_count": 1,
            "communities": [],
            "graph_anomaly_method": "gae_baseline",
            "graph_anomaly_comparable": True,
            "graph_anomaly_self_referential": False,
        },
    )

    detail = web.run_detail(tmp_path / "outputs", "baseline_run")

    assert detail["graph"]["comparable"] is True
    assert detail["graph"]["self_referential"] is False


def test_journeys_endpoint(tmp_path):
    """跨机位行程接口：无文件时返回空，有文件时原样返回。"""
    root = _fake_root(tmp_path)
    server = web.serve("127.0.0.1", 0, root, allow_runs=False)
    port = server.server_address[1]
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/journeys", timeout=5) as response:
            empty = json.loads(response.read())
        assert empty["journey_count"] == 0

        payload = {
            "journey_count": 1,
            "journeys": [
                {"global_id": "G0001", "travel_sec": 90.0, "similarity": 0.9,
                 "legs": [{"camera": "admin_in", "passage_id": "a"},
                          {"camera": "admin_out", "passage_id": "b"}]}
            ],
        }
        (root / "journeys.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/journeys", timeout=5) as response:
            loaded = json.loads(response.read())
        assert loaded["journey_count"] == 1
        assert loaded["journeys"][0]["legs"][1]["camera"] == "admin_out"
    finally:
        server.shutdown()