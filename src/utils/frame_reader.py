import cv2
from typing import Optional, Tuple


class FrameReader:
    """视频帧读取器。

    支持视频文件、RTSP 流和 USB 摄像头三种输入源。
    提供统一的 open/read/release 接口。
    """

    @staticmethod
    def open(source: str) -> cv2.VideoCapture:
        """打开视频源。

        Args:
            source: 视频文件路径 / RTSP URL / 摄像头索引(0, 1, ...)

        Returns:
            cv2.VideoCapture 对象
        """
        # 判断是否为整数（摄像头）
        try:
            cam_idx = int(source)
            cap = cv2.VideoCapture(cam_idx)
        except ValueError:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        return cap

    @staticmethod
    def get_properties(cap: cv2.VideoCapture) -> dict:
        """获取视频属性。"""
        return {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }

    @staticmethod
    def read_frame(cap: cv2.VideoCapture) -> Optional[Tuple[bool, object]]:
        """读取一帧。"""
        ret, frame = cap.read()
        if not ret:
            return None
        return ret, frame

    @staticmethod
    def release(cap: cv2.VideoCapture):
        cap.release()
