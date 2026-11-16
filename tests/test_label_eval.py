"""标注回填链路的护栏测试。

用**合成的假标签**验证「标注到位后指标能算出来」这条通路，不产出任何真实指标结论。
重点是两个容易翻车的边界：

1. 只标注了一部分车辆时，未标注的不能被当成负样本；
2. 标注里 passage_id 写错时，必须显式报出"有多少条没对上"，不能静默丢弃。
"""

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VECTORS = ROOT / "data/outputs/e2e/alerts/vectors_admin_in.csv"


def _write_labels(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["passage_id", "is_violation", "violation_type"])
        for pid, flag in rows:
            writer.writerow([pid, flag, "synthetic_test"])


def _read_vectors(limit):
    with open(VECTORS, encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if (r.get("passage_id") or "").strip()]
    return rows[:limit]


def _run(vectors, labels, output):
    return subprocess.run(
        [sys.executable, "scripts/eval_with_labels.py",
         "--vectors", str(vectors), "--labels", str(labels), "--output", str(output)],
        cwd=str(ROOT), capture_output=True, text=True, timeout=300,
    )


def test_partial_labels_do_not_count_unlabeled_as_negatives(tmp_path):
    rows = _read_vectors(40)
    label_path = tmp_path / "partial.csv"
    _write_labels(label_path, [(r["passage_id"], 1 if i < 6 else 0) for i, r in enumerate(rows)])

    result = _run(VECTORS, label_path, tmp_path / "out.json")

    assert result.returncode == 0, (result.stdout + result.stderr)[-600:]
    payload = (tmp_path / "out.json").read_text(encoding="utf-8")
    import json

    data = json.loads(payload)
    assert data["labeled_vehicles"] == 40
    assert data["labels_unmatched"] == 0
    assert data["positives"] + data["negatives"] == 40


def test_unmatched_labels_are_reported_not_silently_dropped(tmp_path):
    rows = _read_vectors(12)
    labels = []
    for i, row in enumerate(rows):
        pid = row["passage_id"] + ("_TYPO" if i % 3 == 0 else "")
        labels.append((pid, 1 if i < 4 else 0))
    label_path = tmp_path / "broken.csv"
    _write_labels(label_path, labels)

    result = _run(VECTORS, label_path, tmp_path / "out.json")

    assert result.returncode == 0, (result.stdout + result.stderr)[-600:]
    assert "did not match" in (result.stdout + result.stderr)

    import json

    data = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert data["labels_provided"] == 12
    assert data["labels_unmatched"] == 4
    assert data["labeled_vehicles"] == 8
