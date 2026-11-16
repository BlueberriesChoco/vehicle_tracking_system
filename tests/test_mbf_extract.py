"""MBF 录像格式解析的回归测试（合成数据，不依赖真实录像）。"""

import pytest

from src.utils.mbf_extract import (
    MBF_BLOCK_HEADER_SIZE,
    MBF_BLOCK_MARKER,
    extract_h264,
    parse_mbf_header,
)


def _make_mbf(payload_segments, block_size=256):
    """构造一个最小 MBF 文件：块 0 为头，块 1..N 为 MBFT2 + 31 字节 + 载荷。"""
    header = b"MBF0" + (128).to_bytes(4, "little") + block_size.to_bytes(4, "little") + b"\x00" * 8
    data = bytearray(header.ljust(block_size, b"\x00"))

    payload_size = block_size - MBF_BLOCK_HEADER_SIZE
    for segment in payload_segments:
        assert len(segment) == payload_size
        block = MBF_BLOCK_MARKER + b"\x00" * (MBF_BLOCK_HEADER_SIZE - len(MBF_BLOCK_MARKER)) + segment
        data.extend(block)

    return bytes(data)


def test_parse_mbf_header():
    data = _make_mbf([], block_size=256)
    header = parse_mbf_header(data)

    assert header["block_size"] == 256
    assert header["header_size"] == 128


def test_extract_h264_concatenates_blocks():
    payload_size = 256 - MBF_BLOCK_HEADER_SIZE
    first = bytes([1]) * payload_size
    second = bytes([2]) * payload_size

    data = _make_mbf([first, second], block_size=256)
    stream = extract_h264(data)

    assert stream == first + second


def test_extract_h264_rejects_non_mbf():
    with pytest.raises(ValueError):
        extract_h264(b"NOT_AN_MBF" + b"\x00" * 64)


def test_extract_h264_skips_non_marker_blocks():
    payload_size = 256 - MBF_BLOCK_HEADER_SIZE
    first = bytes([3]) * payload_size
    second = bytes([4]) * payload_size

    data = _make_mbf([first, second], block_size=256)
    # 把第二个块标记去掉，模拟无标记的块：应原样拼接
    marker_pos = 256 + 256
    tampered = data[:marker_pos] + b"ZZZZZ" + data[marker_pos + 5:]

    stream = extract_h264(tampered)

    assert stream[:payload_size] == first
    assert stream[payload_size:payload_size + 5] == b"ZZZZZ"
