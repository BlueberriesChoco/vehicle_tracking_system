#!/usr/bin/env python
"""检查当前环境是否与 requirements.lock.txt 一致，并校验 GPU/torch 组合是否可用。

为什么需要：
    venv 不能跨机器拷贝，torch 的构建又必须按显卡选（sm_120 需要 cu128+）。
    换机器后跑这个脚本，能一次看出「版本漂移」和「torch 选错」两类问题。

用法：
    .venv\\python.exe scripts/check_environment.py
退出码：
    0 = 全部一致且 torch 能用当前 GPU
    1 = 有版本不一致或 torch/GPU 不匹配
"""

from __future__ import annotations

import importlib
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK = os.path.join(ROOT, "requirements.lock.txt")
EXPECTED_PYTHON = (3, 12)


def parse_lock(path: str) -> dict:
    expected = {}
    if not os.path.isfile(path):
        return expected
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s#]+)", line)
        if match:
            expected[match.group(1).lower().replace("_", "-")] = match.group(2)
    return expected


def installed_version(name: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def main() -> int:
    problems = []
    print("=" * 60)
    print("Python")
    print("=" * 60)
    py = sys.version_info
    print(f"  当前: {py.major}.{py.minor}.{py.micro}")
    if (py.major, py.minor) != EXPECTED_PYTHON:
        problems.append(f"Python 版本应为 {EXPECTED_PYTHON[0]}.{EXPECTED_PYTHON[1]}，实际 {py.major}.{py.minor}")

    print()
    print("=" * 60)
    print("依赖版本")
    print("=" * 60)
    expected = parse_lock(LOCK)
    for name, want in sorted(expected.items()):
        have = installed_version(name)
        if have is None:
            problems.append(f"{name} 未安装（期望 {want}）")
            print(f"  [缺失] {name:<22} 期望 {want}")
        elif have != want:
            problems.append(f"{name} 版本 {have} != {want}")
            print(f"  [不一致] {name:<20} 实际 {have}  期望 {want}")
        else:
            print(f"  [OK]   {name:<22} {have}")

    print()
    print("=" * 60)
    print("torch / GPU")
    print("=" * 60)
    try:
        import torch
    except Exception as error:
        print(f"  [失败] 无法导入 torch: {error}")
        problems.append("torch 未安装")
        torch = None

    if torch is not None:
        print(f"  torch        {torch.__version__}")
        print(f"  CUDA runtime {torch.version.cuda}")
        print(f"  CUDA 可用    {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            capability = torch.cuda.get_device_capability(0)
            print(f"  GPU          {name}")
            print(f"  计算能力     {capability[0]}.{capability[1]} (sm_{capability[0]}{capability[1]})")
            cuda_version = float(torch.version.cuda or "0")
            if capability[0] >= 12 and cuda_version < 12.8:
                problems.append(
                    f"GPU 是 sm_{capability[0]}{capability[1]}（50 系），但 torch 是 cu{cuda_version} 构建，"
                    "缺少对应内核，请改装 cu128 及以上的 torch"
                )
            # 真正跑一次算子，才能确认内核存在
            try:
                x = torch.ones(64, 64, device="cuda")
                (x @ x).sum().item()
                print("  GPU 算子自检 通过")
            except Exception as error:
                problems.append(f"GPU 算子执行失败: {error}")
                print(f"  GPU 算子自检 失败: {error}")
        else:
            problems.append("CUDA 不可用（只能 CPU 运行，速度会慢很多）")

    print()
    print("=" * 60)
    if problems:
        print(f"发现 {len(problems)} 个问题：")
        for item in problems:
            print(f"  - {item}")
        return 1
    print("环境一致，torch 可用。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
