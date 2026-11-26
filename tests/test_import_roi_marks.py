import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.import_roi_marks import apply_marks  # noqa: E402


def _marks(camera, roi=None, entry=None, exit_=None):
    item = {"camera": camera}
    if roi:
        item["roi"] = roi
    if entry:
        item["entry_line"] = entry
    if exit_:
        item["exit_line"] = exit_
    return {"cameras": [item]}


def test_roi_and_lines_are_written(tmp_path):
    stats = apply_marks(
        _marks("cam_a", [[10, 10], [200, 10], [200, 150], [10, 150]],
               [[5, 100], [250, 100]], [[5, 140], [250, 140]]),
        str(tmp_path),
    )

    assert stats["created"] == 1 and stats["updated"] == 0 and stats["skipped"] == 0
    data = yaml.safe_load((tmp_path / "cam_a.yaml").read_text(encoding="utf-8"))
    assert data["camera_id"] == "cam_a"
    assert data["roi"]["polygon"][0] == [10, 10]
    assert data["entry_line"]["p1"] == [5, 100]
    assert data["exit_line"]["p2"] == [250, 140]


def test_existing_calibration_is_preserved(tmp_path):
    original = {
        "camera_id": "admin_in",
        "camera_name": "综保A区行政车道进",
        "calibration": {"homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "px_per_meter": 70.0},
        "anomaly": {"contamination": 0.02},
    }
    (tmp_path / "admin_in.yaml").write_text(
        yaml.safe_dump(original, allow_unicode=True), encoding="utf-8")

    apply_marks(_marks("admin_in", [[0, 0], [10, 0], [10, 10]]), str(tmp_path))

    data = yaml.safe_load((tmp_path / "admin_in.yaml").read_text(encoding="utf-8"))
    assert data["calibration"]["px_per_meter"] == 70.0
    assert data["calibration"]["homography"][0] == [1, 0, 0]
    assert data["anomaly"]["contamination"] == 0.02
    assert data["camera_name"] == "综保A区行政车道进"


def test_two_point_roi_is_ignored(tmp_path):
    """ROI 少于 3 点说明没画完，不应写进配置。"""
    apply_marks(_marks("cam_b", [[10, 10], [20, 20]]), str(tmp_path))

    data = yaml.safe_load((tmp_path / "cam_b.yaml").read_text(encoding="utf-8"))
    assert "roi" not in data


def test_entry_line_keeps_existing_exit(tmp_path):
    (tmp_path / "cam_c.yaml").write_text(
        yaml.safe_dump({"camera_id": "cam_c", "exit_line": {"p1": [1, 2], "p2": [3, 4]}}),
        encoding="utf-8")

    apply_marks(_marks("cam_c", [[0, 0], [9, 0], [9, 9]], [[1, 1], [9, 1]]), str(tmp_path))

    data = yaml.safe_load((tmp_path / "cam_c.yaml").read_text(encoding="utf-8"))
    assert data["exit_line"]["p1"] == [1, 2]
    assert data["entry_line"]["p2"] == [9, 1]


def test_bidirectional_flag_is_written(tmp_path):
    """双向车道标记要写进配置，pipeline 才能用双向判定。"""
    marks = {"cameras": [{"camera": "cam_d", "roi": [[0, 0], [9, 0], [9, 9]], "bidirectional": True}]}

    apply_marks(marks, str(tmp_path))

    data = yaml.safe_load((tmp_path / "cam_d.yaml").read_text(encoding="utf-8"))
    assert data["bidirectional"] is True