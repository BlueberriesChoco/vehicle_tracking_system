"""极少样本场景下新增工具的单测。

覆盖：待核查清单排序、特征可用性诊断、微扰施加、盲评抽样与不泄漏分数。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.anomaly.baseline import DEFAULT_FEATURES


def _load(name):
    import importlib.util

    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_rt_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _vehicle(pid, score, duration=2.0, dwell=0.0, evidence=1):
    return {
        "passage_id": pid,
        "camera_id": "admin_in",
        "enter_time": "2026-09-01T08:01:00",
        "duration_sec": duration,
        "risk_score_100": score,
        "evidence": [{"summary": f"证据{i}"} for i in range(evidence)],
        "max_dwell_sec": dwell,
    }


def test_review_list_ranks_and_truncates():
    module = _load("build_review_list")
    vehicles = [_vehicle("a", 10), _vehicle("b", 90), _vehicle("c", 50)]

    picked = module.rank_vehicles(vehicles, budget=2)

    assert [item["passage_id"] for item in picked] == ["b", "c"]


def test_review_list_has_no_anomaly_flag():
    module = _load("build_review_list")
    rows = module.build_rows(module.rank_vehicles([_vehicle("a", 10)], budget=1))

    assert "is_anomaly" not in rows[0], "清单是线索排序，不应带「异常」判定"
    assert rows[0]["rank"] == 1
    assert rows[0]["evidence_count"] == 1


def test_feature_health_distinguishes_gated_constant_and_usable():
    module = _load("eval_perturbation")
    rows = []
    for index in range(20):
        row = {
            "duration_sec": 1.0 + index * 0.1,
            "max_dwell_sec": 0.0,
            "max_concurrent_vehicles": float(index % 3),
            "speed_reliable": 0,
        }
        rows.append(row)

    health = {item["feature"]: item for item in module.feature_health(rows)}

    assert health["avg_speed_ms"]["status"] in {"全部被门控", "常量"}
    assert health["max_dwell_sec"]["status"] == "常量"
    assert health["duration_sec"]["status"] == "可用"


def test_apply_perturbation_respects_floor():
    module = _load("eval_perturbation")
    rows = [{"stop_count": 0.0, "duration_sec": 10.0}]

    result = module._apply(rows, {"stop_count": ("minus", 1), "duration_sec": 0.9})

    assert result[0]["stop_count"] == 0.0, "减一不能出现负的停留次数"
    assert result[0]["duration_sec"] == 9.0
    assert rows[0]["duration_sec"] == 10.0, "原始数据不能被就地修改"


def test_evaluate_reports_structure():
    module = _load("eval_perturbation")

    class StubDetector:
        def score(self, rows):
            return [{**row, "iforest_score": float(row.get("duration_sec", 0))} for row in rows]

    rows = [_vehicle(f"v{index}", 0, duration=1.0 + index * 0.5) for index in range(30)]
    report = module.evaluate(rows, StubDetector(), top_k=5)

    assert report["top_k"] == 5
    assert len(report["baseline_top_k"]) == 5
    assert 0.0 <= report["robustness"] <= 1.0
    assert report["results"], "至少要有一组扰动结果"
    assert all(0.0 <= item["top_k_overlap"] <= 1.0 for item in report["results"])


def test_blind_review_hides_scores_and_is_deterministic():
    module = _load("export_blind_review")
    vehicles = [_vehicle(f"v{index}", index * 2.5) for index in range(40)]

    first = module.sample_cases(vehicles, n_random=10, n_top=10, seed=7)
    second = module.sample_cases(vehicles, n_random=10, n_top=10, seed=7)

    assert [row["passage_id"] for row, _ in first] == [row["passage_id"] for row, _ in second]

    blind = module.to_blind_row(1, first[0][0])
    assert "risk_score_100" not in blind
    assert "risk_level" not in blind
    assert set(blind) == set(module.BLIND_COLUMNS)


def test_random_group_excludes_system_top():
    module = _load("export_blind_review")
    vehicles = [_vehicle(f"v{index}", index * 2.5) for index in range(40)]

    cases = module.sample_cases(vehicles, n_random=10, n_top=10, seed=3)
    top_ids = {row["passage_id"] for row, group in cases if group == "system_top"}
    random_ids = {row["passage_id"] for row, group in cases if group == "random"}

    assert not (top_ids & random_ids), "随机组不应与系统高分重复，否则基线被污染"


def test_default_features_are_documented():
    """保底检查：特征表存在且非空，供 feature_health 使用。"""
    assert len(DEFAULT_FEATURES) >= 10
    assert all(spec.name for spec in DEFAULT_FEATURES)
