"""MBF 录像格式解析。

南京南通海关综保区导出的 ``.mbf`` 录像是一种私有封装，内部实际是标准
H.264 Annex B 裸流。本模块把它还原为可直接解码的 H.264 字节流。

格式结构（逆向分析 + 实测验证）：

- 文件头（偏移 0）::

    4d 42 46 30 | 00 00 02 00 | 00 00 20 00 | ...
    "MBF0"        header_size   block_size
                  =131072       =2097152

- 块 0（0 ~ block_size-1）：索引/元数据区，不含视频流。
- 块 1..N：每块 block_size 字节，块头为 ``MBFT2`` + 31 字节保留，
  随后是 H.264 Annex B 裸流。流在块边界处被任意截断，去掉块头后
  按顺序拼接即可还原连续码流。

该结构已用 ``行政车道进`` 真实录像验证：还原出 1920x1080 / 25fps /
29704 帧，与录像时长一致。
"""

from __future__ import annotations

import os
from typing import Optional

MBF_MAGIC = b"MBF0"
MBF_BLOCK_MARKER = b"MBFT2"
MBF_DEFAULT_BLOCK_SIZE = 2097152  # 2 MB
MBF_BLOCK_HEADER_SIZE = 36        # "MBFT2"(5) + 31 字节保留


def parse_mbf_header(data: bytes) -> dict:
    """解析 MBF0 文件头。"""
    if not data.startswith(MBF_MAGIC):
        raise ValueError("不是 MBF0 文件")
    if len(data) < 12:
        raise ValueError("MBF0 文件头不完整")
    return {
        "version": int.from_bytes(data[4:8], "little"),
        "header_size": int.from_bytes(data[4:8], "little"),
        "block_size": int.from_bytes(data[8:12], "little") or MBF_DEFAULT_BLOCK_SIZE,
    }


def extract_h264(data: bytes, block_size: Optional[int] = None) -> bytes:
    """从 MBF 字节流中还原 H.264 Annex B 裸流。"""
    header = parse_mbf_header(data)
    if block_size is None or block_size <= 0:
        block_size = header["block_size"]

    out = bytearray()
    for offset in range(block_size, len(data), block_size):
        segment = data[offset:offset + block_size]
        if segment.startswith(MBF_BLOCK_MARKER):
            out.extend(segment[MBF_BLOCK_HEADER_SIZE:])
        else:
            out.extend(segment)
    return bytes(out)


def extract_mbf_file(input_path: str, output_path: Optional[str] = None) -> str:
    """读取 ``.mbf`` 文件，写出 ``.h264``，返回输出路径。"""
    with open(input_path, "rb") as f:
        data = f.read()
    stream = extract_h264(data)

    if output_path is None:
        if input_path.lower().endswith(".mbf"):
            output_path = input_path[:-4] + ".h264"
        else:
            output_path = input_path + ".h264"

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(stream)
    return output_path
