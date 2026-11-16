"""带预取的帧读取器：把解码放到独立线程，与推理重叠。

`cv2.VideoCapture.read()` 在 1080P 上约占单帧处理时间的 1/4（见 PERFORMANCE.md）。
它是一次 C 调用，执行期间会释放 GIL，因此放进独立线程后可以与 GPU 推理重叠。

只用于顺序读取：不改变帧顺序，也不改变时间戳，保证结果与直接读取一致。
"""

from __future__ import annotations

import logging
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

            # 队列满表示消费端是瓶颈，只能阻塞等待，绝不能丢帧：
            # 旧实现在 0.5 秒超时后直接 break，会把剩余视频静默截断，
            # 而上层只看到 (False, None)，误当成“视频结束”。
            delivered = False
            while not self._stop.is_set():
                try:
                    self._queue.put((ok, frame, index), timeout=0.2)
                    delivered = True
                    break
                except queue.Full:
                    continue
            if not delivered:  # 已被 stop() 中断
                return

            index += 1
            if not ok:
                break

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """按顺序返回下一帧；读完后返回 (False, None)。"""
        try:
            ok, frame, _ = self._queue.get(timeout=30.0)
        except queue.Empty:
            # 超时又发生在解码线程仍存活时，说明不是“读完”，而是真的卡住了；
            # 至少留下日志，避免静默当作视频结束。
            if self._thread.is_alive() and not self._stop.is_set():
                logging.getLogger(__name__).warning(
                    "Frame prefetch timed out after 30s while the decode thread is still alive; "
                    "treating as end-of-stream (frames may be incomplete)"
                )
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
