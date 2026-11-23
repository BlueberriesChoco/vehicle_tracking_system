#!/usr/bin/env python
"""Download the pinned HyperLPR3 ONNX weights and verify their SHA-256."""

import hashlib
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.feature_extraction.plate_ocr import HYPERLPR_MODELS, hyperlpr_model_dir

WEIGHT_URL = (
    "https://raw.githubusercontent.com/szad670401/HyperLPR/"
    "58ad4ef736fec5289e8c8cb842e066e70c4672c3/resource/models/onnx/"
)


def main():
    model_dir = hyperlpr_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, digest in HYPERLPR_MODELS.items():
        path = model_dir / name
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
            print(f"Verified: {path}")
            continue
        with urlopen(WEIGHT_URL + name, timeout=30) as response:
            data = response.read(20 * 1024 * 1024 + 1)
        if hashlib.sha256(data).hexdigest() != digest:
            raise SystemExit(f"Unexpected model SHA-256: {name}")
        with NamedTemporaryFile(dir=model_dir, delete=False) as handle:
            temp = Path(handle.name)
            handle.write(data)
        try:
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
        print(f"Installed: {path}")


if __name__ == "__main__":
    main()
