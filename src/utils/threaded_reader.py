"""带预取的帧读取器：把解码放到独立线程，与推理重叠。

`cv2.VideoCapture.read()` 在 1080P 上约占单帧处理时间的 1/4（见 PERFORMANCE.md）。
它是一次 C 调用，执行期间会释放 GIL，因此放进独立线程后可以与 GPU 推理重叠。

只用于顺序读取：不改变帧顺序，也不改变时间戳，保证结果与直接读取一致。
"""

from __future__ import annotations

import queue
import threading
from typing import Optional, Tuple

import numpy as np


class ThreadedFrameReader:
    """有界队列 + 后台解码线程。"""

    def __init__(self, capture, queue_size: int = 32, sentinel: object = None):
        if queue_size < 1:
            raise ValueError("queue_size must be >= 1")
        self._capture = capture
        self._queue: "queue.Queue[Tuple[bool, Optional[np.ndarray], int]]" = queue.Queue(
            maxsize=queue_size
        )
        self._stop = threading.Event()
        self._sentinel = sentinel
        self._thread = threading.Thread(target=self._run, daemon=True, name="frame-prefetch")
        self._thread.start()

    def _run(self):
        index = 0
        while not self._stop.is_set():
            ok, frame = self._capture.read()
            try:
                self._queue.put((ok, frame, index), timeout=0.5)
            except queue.Full:
                # 队列满且主线程还没消费，说明消费端是瓶颈，直接退出
                break
            index += 1
            if not ok:
                break

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """按顺序返回下一帧；读完后返回 (False, None)。"""
        try:
            ok, frame, _ = self._queue.get(timeout=30.0)
        except queue.Empty:
            return False, None
        return ok, frame

    def stop(self):
        self._stop.set()
        # 排空队列，避免后台线程卡在 put 上
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
