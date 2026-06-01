import os
import cv2
import yaml
from typing import Optional, List
from datetime import datetime, timedelta
from pathlib import Path

from ..detection.detector import VehicleDetector
from ..detection.roi_filter import ROIFilter
from ..tracking.bytetrack_tracker import ByteTrackTracker
from ..trajectory.scene_geometry import SceneGeometry
from ..trajectory.trajectory_smoother import TrajectorySmoother
from ..trajectory.trajectory_extractor import TrajectoryExtractor
from ..trajectory.path_reference import PathReference
from ..behavior.behavior_vector import BehaviorVectorizer
from ..feature_extraction.color_recognizer import ColorRecognizer
from ..feature_extraction.vehicle_classifier import VehicleClassifier
from ..feature_extraction.plate_detector import PlateDetector
from ..feature_extraction.plate_ocr import PlateOCR
from ..feature_extraction.reid_extractor import ReIDExtractor
from ..output.vector_writer import VectorWriter
from ..visualization.draw import Visualizer
from ..utils.frame_reader import FrameReader
from ..utils.logger import get_logger

from .frame_pipeline import FramePipeline


class BatchProcessor:
    """批量视频处理器。

    对一段视频执行完整的检测→跟踪→轨迹提取→行为向量化流水线，
    输出行为向量 CSV 和可选的标注视频。
    """

    def __init__(
        self,
        config_path: str = "config/default.yaml",
        camera_config_path: Optional[str] = None,
    ):
        self.logger = get_logger(__name__)

        # 加载配置
        self.config = self._load_config(config_path)
        self.camera_config = {}
        if camera_config_path and os.path.exists(camera_config_path):
            self.camera_config = self._load_config(camera_config_path)

        # 模型配置（Phase 2 特征提取器开关）
        model_config_path = os.path.join(os.path.dirname(config_path), "model_config.yaml")
        self.model_config = self._load_config(model_config_path) if os.path.exists(model_config_path) else {}

        self._init_components()

    def _load_config(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _init_components(self):
        """初始化所有组件。"""
        cfg = self.config
        cam_cfg = self.camera_config

        # 检测器
        det_cfg = cfg.get("detection", {})
        pipeline_cfg = cfg.get("pipeline", {})
        device = pipeline_cfg.get("device", 0)

        self.detector = VehicleDetector(
            model_path=det_cfg.get("model_path", "models/yolov8s.pt"),
            img_size=det_cfg.get("img_size", 640),
            conf_threshold=det_cfg.get("conf_threshold", 0.35),
            iou_threshold=det_cfg.get("iou_threshold", 0.45),
            classes=det_cfg.get("classes", [2, 3, 5, 7]),
            device=device,
            half=det_cfg.get("half", True),
        )

        # ROI 过滤器
        roi_polygon = cam_cfg.get("roi", {}).get("polygon", [])
        if not roi_polygon:
            self.logger.warning("ROI polygon not configured, using full frame")
            roi_polygon = [[0, 0], [1920, 0], [1920, 1080], [0, 1080]]
        self.roi_filter = ROIFilter(roi_polygon)

        # 跟踪器
        track_cfg = cfg.get("tracking", {})
        camera_id = cam_cfg.get("camera_id", "cam01")
        self.frame_rate = track_cfg.get("frame_rate", 25)

        self.tracker = ByteTrackTracker(
            track_buffer=track_cfg.get("track_buffer", 30),
            match_thresh=track_cfg.get("match_thresh", 0.8),
            frame_rate=self.frame_rate,
            min_hits=track_cfg.get("min_hits", 3),
            camera_id=camera_id,
        )

        # 场景几何
        entry_line = cam_cfg.get("entry_line", {})
        exit_line = cam_cfg.get("exit_line", {})
        calib = cam_cfg.get("calibration", {})
        px_per_meter = calib.get("px_per_meter", 20.0)
        if px_per_meter <= 0 and calib.get("ref_length_m", 0) > 0:
            ref_px = self._compute_ref_length_px(calib)
            if ref_px > 0:
                px_per_meter = ref_px / calib["ref_length_m"]

        self.scene_geometry = SceneGeometry(
            entry_line=(
                tuple(entry_line.get("p1", [0, 0])),
                tuple(entry_line.get("p2", [0, 0])),
            ),
            exit_line=(
                tuple(exit_line.get("p1", [0, 0])),
                tuple(exit_line.get("p2", [0, 0])),
            ),
            px_per_meter=px_per_meter,
        )

        # 轨迹平滑器
        traj_cfg = cfg.get("trajectory", {})
        self.smoother = TrajectorySmoother(
            window_size=traj_cfg.get("smoothing_window", 5),
            method="savgol",
        )

        # 轨迹提取器
        self.trajectory_extractor = TrajectoryExtractor(
            scene_geometry=self.scene_geometry,
            smoother=self.smoother,
            min_trajectory_length=traj_cfg.get("min_trajectory_length", 10),
            enter_min_frames=traj_cfg.get("enter_min_frames", 5),
        )

        # 参考路径
        self.path_reference = PathReference()
        ref_path_file = cam_cfg.get("reference_path_file", "")
        if ref_path_file and os.path.exists(ref_path_file):
            self.path_reference.load(ref_path_file)

        # 行为向量化器
        behavior_cfg = cfg.get("behavior", {})
        speed_cfg = behavior_cfg.get("speed", {})
        night_cfg = behavior_cfg.get("night", {})
        # Phase 2: 初始化车辆属性特征提取器
        color_recognizer = self._init_color_recognizer(self.model_config)
        vehicle_classifier = self._init_vehicle_classifier(self.model_config)
        plate_detector = self._init_plate_detector(self.model_config)
        plate_ocr = self._init_plate_ocr(self.model_config)
        reid_extractor = self._init_reid_extractor(self.model_config)

        self.vectorizer = BehaviorVectorizer(
            scene_geometry=self.scene_geometry,
            trajectory_extractor=self.trajectory_extractor,
            smoother=self.smoother,
            path_reference=self.path_reference,
            frame_rate=self.frame_rate,
            night_start_hour=night_cfg.get("start_hour", 22),
            night_end_hour=night_cfg.get("end_hour", 6),
            dwell_threshold_ms=speed_cfg.get("dwell_threshold_ms", 0.5),
            dwell_min_duration_sec=speed_cfg.get("dwell_min_duration_sec", 3.0),
            color_recognizer=color_recognizer,
            vehicle_classifier=vehicle_classifier,
            plate_detector=plate_detector,
            plate_ocr=plate_ocr,
            reid_extractor=reid_extractor,
        )

        # 管线
        self.pipeline = FramePipeline(
            detector=self.detector,
            roi_filter=self.roi_filter,
            tracker=self.tracker,
            trajectory_extractor=self.trajectory_extractor,
            behavior_vectorizer=self.vectorizer,
            frame_rate=self.frame_rate,
        )

        # 可视化
        vis_cfg = cfg.get("visualization", {})
        self.visualizer = Visualizer(
            scene_geometry=self.scene_geometry,
            draw_trajectory=vis_cfg.get("draw_trajectory", True),
            trajectory_tail=vis_cfg.get("trajectory_tail", 50),
            box_thickness=vis_cfg.get("box_thickness", 2),
            font_scale=vis_cfg.get("font_scale", 0.5),
        )

        # 输出
        output_cfg = cfg.get("output", {})
        self.vector_writer = VectorWriter(
            output_format=output_cfg.get("format", "csv"),
        )

        self.frame_skip = pipeline_cfg.get("frame_skip", 1)
        self.max_frames = pipeline_cfg.get("max_frames", 0)
        self.output_video_enabled = pipeline_cfg.get("output_video", False)

    def process_video(
        self,
        video_path: str,
        output_dir: str = "data/outputs",
        output_video_path: Optional[str] = None,
    ) -> str:
        """处理一段视频文件并输出行为向量。

        Args:
            video_path: 输入视频路径
            output_dir: 输出目录
            output_video_path: 标注视频输出路径（可选）

        Returns:
            输出的行为向量 CSV 文件路径
        """
        self.logger.info(f"Processing video: {video_path}")
        self.pipeline.reset()

        # 解析输出文件路径
        video_name = Path(video_path).stem
        camera_id = self.tracker.camera_id

        # 从视频文件名推断日期和小时
        date_str, hour_str = self._parse_video_datetime(video_name)

        vector_output_dir = os.path.join(
            output_dir, "vectors", camera_id, date_str
        )
        os.makedirs(vector_output_dir, exist_ok=True)

        # 打开视频
        cap = FrameReader.open(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        source_fps = cap.get(cv2.CAP_PROP_FPS) or self.frame_rate
        video_start_time = self._parse_video_start_time(video_name) or datetime.now()
        self.logger.info(f"Total frames: {total_frames}")

        # 视频写入器
        video_writer = None
        if self.output_video_enabled and output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_writer = cv2.VideoWriter(output_video_path, fourcc, self.frame_rate, (w, h))

        # 逐帧处理
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 跳帧
            if frame_count % self.frame_skip != 0:
                continue

            # 最大帧数限制
            if self.max_frames > 0 and frame_count > self.max_frames:
                break

            # 处理帧
            timestamp = video_start_time + timedelta(
                seconds=(frame_count - 1) / source_fps
            )
            annotated_frame, new_vectors = self.pipeline.process_frame(frame, timestamp)

            # 写标注视频
            if video_writer is not None:
                active_tracklets = self.tracker.get_active_tracklets()
                vis_frame = self.visualizer.draw_frame(annotated_frame, active_tracklets)
                video_writer.write(vis_frame)

            # 进度
            if frame_count % 100 == 0:
                self.logger.info(
                    f"Frame {frame_count}/{total_frames}, "
                    f"completed vectors: {len(self.pipeline.completed_vectors)}"
                )

        # 收尾
        remaining = self.pipeline.finalize()
        self.logger.info(f"Finalized {len(remaining)} remaining trajectories")

        cap.release()
        if video_writer is not None:
            video_writer.release()

        # 写出行为向量
        all_vectors = self.pipeline.completed_vectors
        csv_filename = f"vec_{camera_id}_{date_str}_{hour_str}.csv"
        csv_path = os.path.join(vector_output_dir, csv_filename)

        self.vector_writer.write(all_vectors, csv_path)
        self.logger.info(
            f"Output: {csv_path} ({len(all_vectors)} vehicle vectors)"
        )

        return csv_path

    # ---- Phase 2 extractor initializers ----

    @staticmethod
    def _init_color_recognizer(model_cfg: dict):
        color_cfg = model_cfg.get("color", {}) if model_cfg else {}
        if color_cfg.get("enabled", False):
            return ColorRecognizer(method=color_cfg.get("method", "hsv_clustering"))
        return None

    @staticmethod
    def _init_vehicle_classifier(model_cfg: dict):
        classifier_cfg = model_cfg.get("classifier", {}) if model_cfg else {}
        if classifier_cfg.get("enabled", False):
            return VehicleClassifier(model_path=classifier_cfg.get("weights"))
        return VehicleClassifier()

    @staticmethod
    def _init_plate_detector(model_cfg: dict):
        plate_cfg = model_cfg.get("plate", {}) if model_cfg else {}
        if plate_cfg.get("enabled", False):
            return PlateDetector(model_path=plate_cfg.get("detector_weights"))
        return None

    @staticmethod
    def _init_plate_ocr(model_cfg: dict):
        plate_cfg = model_cfg.get("plate", {}) if model_cfg else {}
        if plate_cfg.get("enabled", False):
            return PlateOCR(engine=plate_cfg.get("ocr_engine", "paddleocr"))
        return None

    @staticmethod
    def _init_reid_extractor(model_cfg: dict):
        reid_cfg = model_cfg.get("reid", {}) if model_cfg else {}
        if reid_cfg.get("enabled", False):
            return ReIDExtractor(
                model_name=reid_cfg.get("model_name", "osnet_x1_0"),
                weights_path=reid_cfg.get("weights"),
            )
        return None

    @staticmethod
    def _compute_ref_length_px(calib: dict) -> float:
        """计算参考线段在图像中的像素长度。"""
        p1 = calib.get("ref_point_1", [0, 0])
        p2 = calib.get("ref_point_2", [0, 0])
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return (dx * dx + dy * dy) ** 0.5

    @staticmethod
    def _parse_video_datetime(video_name: str) -> tuple:
        """从视频文件名解析日期和小时。

        期望命名格式: ch01_20260520_080000_090000.mp4
        返回: ("20260520", "08")
        """
        parts = video_name.split("_")
        date_str = "unknown"
        hour_str = "00"

        for i, part in enumerate(parts):
            if len(part) == 8 and part.isdigit():
                date_str = part
                if i + 1 < len(parts) and len(parts[i + 1]) >= 2:
                    hour_str = parts[i + 1][:2]
                break

        return date_str, hour_str

    @staticmethod
    def _parse_video_start_time(video_name: str) -> Optional[datetime]:
        """Parse the recording start timestamp from an hourly video filename."""
        parts = video_name.split("_")
        for index, part in enumerate(parts):
            if len(part) == 8 and part.isdigit() and index + 1 < len(parts):
                time_part = parts[index + 1]
                if len(time_part) >= 6 and time_part[:6].isdigit():
                    try:
                        return datetime.strptime(part + time_part[:6], "%Y%m%d%H%M%S")
                    except ValueError:
                        return None
        return None
