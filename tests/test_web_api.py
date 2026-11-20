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
