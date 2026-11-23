"""依赖声明检查：代码里用到的第三方包必须在 requirements.txt 中声明。

历史问题：`psutil` 在 soak_test.py 里是硬导入，却没有写进 requirements，
换台机器跑浸泡测试会直接 ImportError。这个测试防止同类问题再发生。

可选后端（缺失会自动回退）不要求声明，见 OPTIONAL_MODULES。
"""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# import 名 -> PyPI 包名（不一致的需要映射）
IMPORT_TO_PACKAGE = {
    "cv2": "opencv-python",
    "yaml": "pyyaml",
    "PIL": "pillow",
    "docx": "python-docx",
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
}

# 有 try/except 回退的可选后端，不要求出现在 requirements.txt
OPTIONAL_MODULES = {
    "hyperlpr3",
    "easyocr",
    "paddleocr",
    "onnxruntime",
    "torchreid",
    "boxmot",
    "tensorrt",
}

LOCAL_MODULES = {"src", "scripts", "tests"}


def _declared_packages():
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    declared = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split(">=")[0].split("==")[0].split("[")[0].strip().lower()
        if name:
            declared.add(name)
    return declared


def _imported_modules():
    stdlib = set(sys.stdlib_module_names)
    found = {}
    for folder in ("src", "scripts"):
        for path in (ROOT / folder).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    modules = [node.module.split(".")[0]]
                for module in modules:
                    if module in stdlib or module in LOCAL_MODULES:
                        continue
                    found.setdefault(module, set()).add(str(path.relative_to(ROOT)))
    return found


def test_all_required_third_party_imports_are_declared():
    declared = _declared_packages()
    missing = []
    for module, files in _imported_modules().items():
        if module in OPTIONAL_MODULES:
            continue
        package = IMPORT_TO_PACKAGE.get(module, module).lower()
        if package not in declared:
            missing.append(f"{module} (used in {sorted(files)[0]})")

    assert not missing, "以下第三方依赖未在 requirements.txt 声明：\n" + "\n".join(missing)


def test_requirements_file_is_parseable():
    declared = _declared_packages()

    for expected in ("ultralytics", "opencv-python", "numpy", "scipy", "pyyaml", "torch", "pytest"):
        assert expected in declared
