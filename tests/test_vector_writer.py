"""向量写出器的字段契约测试。"""

import csv

from src.output.vector_writer import VectorWriter


def test_csv_columns_include_passage_id():
    assert "passage_id" in VectorWriter.CSV_COLUMNS


def test_writer_preserves_passage_id(tmp_path):
    vector = {
        "track_id": 1,
        "passage_id": "admin_in_20260901080110_1",
        "camera_id": "admin_in",
        "analysis_target": 1,
    }
    path = tmp_path / "vec_test.csv"

    VectorWriter().write([vector], str(path))

    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["passage_id"] == "admin_in_20260901080110_1"
    assert rows[0]["track_id"] == "1"


def test_writer_handles_missing_passage_id(tmp_path):
    vector = {"track_id": 2, "camera_id": "admin_in", "analysis_target": 1}
    path = tmp_path / "vec_test.csv"

    VectorWriter().write([vector], str(path))

    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["passage_id"] == ""
