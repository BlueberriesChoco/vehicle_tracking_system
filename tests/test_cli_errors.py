"""入口脚本的失败路径。

路径不存在时应该给出清晰提示并以非零码退出，而不是抛 Python 堆栈——
否则运维看到的是一屏 traceback，而不是"路径写错了"。
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CASES = [
    (
        "scripts/extract_mbf.py",
        ["--input", "data/outputs/definitely_missing.mbf", "--output", "data/outputs/x"],
    ),
    (
        "scripts/run_full_pipeline.py",
        [
            "--input", "data/outputs/definitely_missing_dir",
            "--camera", "admin_in",
            "--output", "data/outputs/x",
        ],
    ),
]


def test_missing_input_reports_clearly_instead_of_traceback():
    for script, args in CASES:
        result = subprocess.run(
            [sys.executable, script, *args],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = (result.stdout or "") + (result.stderr or "")

        assert result.returncode == 2, f"{script} 退出码应为 2，实际 {result.returncode}\n{output[-500:]}"
        assert "does not exist" in output, f"{script} 未给出清晰提示\n{output[-500:]}"
        assert "Traceback" not in output, f"{script} 仍在抛堆栈\n{output[-500:]}"
