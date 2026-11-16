"""预取帧读取器的测试：顺序与直接读取必须完全一致。"""

import queue

import numpy as np

from src.utils.threaded_reader import ThreadedFrameReader


class _FakeCapture:
    def __init__(self, frames):
        self._frames = frames
        self._index = 0

    def read(self):
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame


def _frames(n=10):
    return [np.full((4, 4, 3), value, dtype=np.uint8) for value in range(n)]


def test_reader_preserves_order_and_values():
    frames = _frames()
    reader = ThreadedFrameReader(_FakeCapture(frames), queue_size=3)

    received = []
    while True:
        ok, frame = reader.read()
        if not ok:
            break
        received.append(int(frame[0, 0, 0]))

    reader.stop()

    assert received == list(range(len(frames)))


def test_reader_rejects_invalid_queue_size():
    try:
        ThreadedFrameReader(_FakeCapture(_frames(1)), queue_size=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for queue_size=0")


def test_reader_returns_false_after_stop():
    reader = ThreadedFrameReader(_FakeCapture(_frames(2)), queue_size=2)
    reader.stop()

    ok, frame = reader.read()

    assert ok is False
    assert frame is None
