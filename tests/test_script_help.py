"""脚本可发现性的静态检查。

每个 `scripts/*.py` 都应有模块级 docstring；使用 argparse 的还应提供 `description=`，
否则 `--help` 只输出一行 usage，用户不知道这个脚本是干什么的。

这里只做静态解析（毫秒级），不实际执行脚本——执行 32 个脚本的 `--help`
需要几十秒，不适合放进单元测试。
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((ROOT / "scripts").glob("*.py"))


def test_scripts_have_docstrings():
    missing = [p.name for p in SCRIPTS if not ast.get_docstring(ast.parse(p.read_text(encoding="utf-8")))]
    assert not missing, f"以下脚本缺少模块 docstring：{missing}"


def test_argparse_scripts_describe_themselves():
    """用 argparse 的脚本必须给 --help 提供 description，否则帮助信息只有 usage。"""
    offenders = []
    for script in SCRIPTS:
        source = script.read_text(encoding="utf-8")
        if "argparse" not in source:
            continue
        if "description=" not in source and "__doc__" not in source:
            offenders.append(script.name)
    assert not offenders, f"以下脚本的 --help 没有描述：{offenders}"


def test_scripts_directory_is_not_empty():
    assert SCRIPTS, "scripts/ 目录下没有 Python 脚本"
    assert len(SCRIPTS) >= 30, f"脚本数量异常减少：{len(SCRIPTS)}"
