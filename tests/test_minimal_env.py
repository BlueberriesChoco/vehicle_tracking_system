"""最小依赖验证：分析层（无监督检测/图谱/风险/报告）不应隐式依赖 OpenCV 或 ultralytics。

动机：部署时可能只跑分析层（不接视频），如果 risk/graph 模块在顶层 import 了 cv2，
就会白白拖入整个 OpenCV + ultralytics 依赖。

实现方式：起一个独立子进程，把 cv2/ultralytics 置为 None（等效"未安装"），
再跑一遍 IF → 图谱 → 融合 → 证据 → 报告的完整流程。
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PIPELINE = r"""
import sys
for name in ("cv2", "ultralytics"):
    sys.modules[name] = None
sys.path.insert(0, ".")

from src.anomaly.isolation_forest import IsolationForestDetector
from src.graph.payload import build_graph_payload
from src.graph.relations import RelationConfig
from src.risk.evidence import EvidenceBuilder
from src.risk.fusion import RiskFusion
from src.risk.merge import apply_graph
from src.risk.report import TemplateReportGenerator

rows = [
    {
        "track_id": str(i),
        "camera_id": "cam01",
        "analysis_target": "1",
        "vehicle_group": "ordinary",
        "passage_id": "cam01_20260901080000_%d" % i,
        "enter_time": "2026-09-01T08:%02d:00" % i,
        "exit_time": "2026-09-01T08:%02d:03" % i,
        "trajectory_points": ";".join("%d,%d" % (500 + 3 * s, 100 + 40 * s) for s in range(12)),
        "avg_speed_ms": "%.2f" % (8 + 0.1 * i),
        "path_smoothness": "%.3f" % (0.1 + 0.005 * i),
        "path_deviation": "0.0",
    }
    for i in range(30)
]

detector = IsolationForestDetector(n_estimators=30, max_samples=32, random_state=42)
detector.fit(rows)
scored = detector.score(rows)
payload = build_graph_payload(scored, RelationConfig(top_k=3))
fused = RiskFusion().fuse(apply_graph(scored, payload))
top = max(fused, key=lambda row: row.get("risk_score_100", 0))
builder = EvidenceBuilder()
text = TemplateReportGenerator().generate(top, builder.build(top))

assert payload["node_count"] == 30, payload["node_count"]
assert top["risk_score_100"] > 0
assert len(text.splitlines()) > 5
print("OK")
"""


def test_analysis_layer_does_not_require_opencv_or_ultralytics():
    result = subprocess.run(
        [sys.executable, "-c", PIPELINE],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, (result.stdout[-400:] + "\n" + result.stderr[-800:])
    assert "OK" in result.stdout
