#!/usr/bin/env python
"""Pre-delivery checklist: verify the repo is complete and self-consistent.

Checks:
  1. required top-level files and directories exist;
  2. every YAML under config/ parses;
  3. every script referenced in the docs exists;
  4. every document linked from README exists;
  5. no stray virtualenv / temp directories are present.

Usage:
    python scripts/delivery_check.py
"""

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

REQUIRED_FILES = [
    "README.md",
    "requirements.txt",
    "Dockerfile",
    "docker-compose.yml",
    ".gitignore",
    "config/default.yaml",
    "config/anomaly_config.yaml",
    "config/model_config.yaml",
]

REQUIRED_DIRS = ["src", "scripts", "tests", "config", "demo", "docs"]

REQUIRED_DOCS = [
    "START_HERE.md", "GOAL_STATUS.md", "AUDIT_REPORT.md", "README.md", "DELIVERY.md", "ACCEPTANCE_REPORT.md", "METRICS_TRACEABILITY.md",
    "OPERATIONS.md", "PERFORMANCE.md", "BENEFITS.md", "ANOMALY_SIGNATURES.md",
    "DATA_REQUIREMENTS.md", "FEASIBILITY_REPORT.md", "CODE_STRUCTURE.md", "CHANGELOG.md",
]

STRAY_PATTERNS = [".venv_", "node_modules", "__pycache__"]


def check_files():
    missing = [name for name in REQUIRED_FILES if not os.path.isfile(os.path.join(ROOT, name))]
    missing += [name for name in REQUIRED_DIRS if not os.path.isdir(os.path.join(ROOT, name))]
    missing += [name for name in REQUIRED_DOCS if not os.path.isfile(os.path.join(ROOT, name))]
    return missing


def check_yaml():
    try:
        import yaml
    except ImportError:
        return ["PyYAML 未安装，跳过 YAML 校验"]

    problems = []
    for current, _, files in os.walk(os.path.join(ROOT, "config")):
        for name in files:
            if not name.endswith((".yaml", ".yml")):
                continue
            path = os.path.join(current, name)
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    yaml.safe_load(handle)
            except Exception as exc:  # noqa: BLE001 - 交付检查需要捕获所有解析错误
                problems.append(f"{os.path.relpath(path, ROOT)}: {exc}")
    return problems


# 文档里的占位示例，不是真实脚本引用
PLACEHOLDER_SCRIPTS = {"xxx.py", "foo.py", "bar.py", "example.py", "script.py"}


def check_script_references():
    """文档里 `scripts/xxx.py` 形式的引用必须真实存在（占位符除外）。"""
    pattern = re.compile(r"scripts[/\\]([A-Za-z0-9_]+\.py)")
    seen = {}
    for name in os.listdir(ROOT):
        if not name.endswith(".md"):
            continue
        with open(os.path.join(ROOT, name), "r", encoding="utf-8") as handle:
            for match in pattern.finditer(handle.read()):
                seen.setdefault(match.group(1), set()).add(name)
    return [
        f"scripts/{script}（被 {sorted(docs)} 引用）"
        for script, docs in sorted(seen.items())
        if script not in PLACEHOLDER_SCRIPTS
        and not os.path.isfile(os.path.join(ROOT, "scripts", script))
    ]


def check_readme_links():
    path = os.path.join(ROOT, "README.md")
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    missing = []
    for target in re.findall(r"\]\(([^)#]+\.md)\)", text):
        if target.startswith("http"):
            continue
        if not os.path.isfile(os.path.join(ROOT, target)):
            missing.append(target)
    return sorted(set(missing))


def check_stray_dirs():
    found = []
    for name in os.listdir(ROOT):
        full = os.path.join(ROOT, name)
        if not os.path.isdir(full):
            continue
        if any(name.startswith(prefix) or name == prefix for prefix in STRAY_PATTERNS):
            if name == "__pycache__":
                continue
            found.append(name)
    return found


def main():
    results = [
        ("必需文件与目录", check_files()),
        ("YAML 配置解析", check_yaml()),
        ("文档引用的脚本存在", check_script_references()),
        ("README 链接有效", check_readme_links()),
        ("无遗留临时目录", check_stray_dirs()),
    ]

    failed = 0
    for label, problems in results:
        if problems:
            failed += 1
            print(f"FAIL  {label}")
            for item in problems:
                print(f"        - {item}")
        else:
            print(f"OK    {label}")

    print()
    if failed:
        print(f"交付检查未通过：{failed}/{len(results)} 项有问题")
        return 1

    print(f"交付检查通过：{len(results)}/{len(results)} 项")
    return 0


if __name__ == "__main__":
    sys.exit(main())
