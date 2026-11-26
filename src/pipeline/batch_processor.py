import os
import cv2
import numpy as np
import yaml
from typing import Optional
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
from ..output.event_writer import EventWriter
from ..visualization.draw import Visualizer
from ..utils.frame_reader import FrameReader
from ..utils.threaded_reader import ThreadedFrameReader
from ..utils.logger import get_logger

from .frame_pipeline import FramePipeline


def _segment_suffix(name_suffix: str) -> str:
    return f"_{name_suffix}" if name_suffix else ""


def build_vector_filename(camera_id: str, date_str: str, hour_str: str, name_suffix: str = "") -> str:
    """向量 CSV 文件名。

    同一小时可能有多段录像，``name_suffix`` 用于区分，否则后一段会覆盖前一段。
    """
    return f"vec_{camera_id}_{date_str}_{hour_str}{_segment_suffix(name_suffix)}.csv"


def build_event_filename(camera_id: str, date_str: str, hour_str: str, name_suffix: str = "") -> str:
    """通行事件 CSV 文件名，后缀规则与向量 CSV 一致。"""
    return f"events_{camera_id}_{date_str}_{hour_str}{_segment_suffix(name_suffix)}.csv"


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

        # 加载配置：default.yaml 控制通用参数，camera/*.yaml 覆盖单摄像头 ROI/标定
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
        self.reference_resolution = tuple(cam_cfg.get("reference_resolution") or ())
        if self.reference_resolution and (
            len(self.reference_resolution) != 2
            or any(type(value) is not int or value <= 0 for value in self.reference_resolution)
        ):
            raise ValueError("reference_resolution must contain two positive integers [width, height]")

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
        has_configured_roi = bool(roi_polygon)
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
            match_thresh=track_cfg.get("match_thresh", 0.3),
            frame_rate=self.frame_rate,
            min_hits=track_cfg.get("min_hits", 3),
            camera_id=camera_id,
            high_conf_threshold=track_cfg.get("high_conf_threshold", 0.5),
        )

        # 场景几何：决定速度、路径偏离、聚集距离等字段是否可靠
        entry_line = cam_cfg.get("entry_line", {})
        exit_line = cam_cfg.get("exit_line", {})
        calib = cam_cfg.get("calibration", {})
        px_per_meter = calib.get("px_per_meter", 20.0)
        if px_per_meter <= 0 and calib.get("ref_length_m", 0) > 0:
            ref_px = self._compute_ref_length_px(calib)
            if ref_px > 0:
                px_per_meter = ref_px / calib["ref_length_m"]

        homography = self._parse_homography(calib)
        geometry_level = self._infer_geometry_level(
            has_configured_roi=has_configured_roi,
            entry_line=entry_line,
            exit_line=exit_line,
            px_per_meter=px_per_meter,
            calibration=calib,
            has_homography=homography is not None,
        )

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
            geometry_level=geometry_level,
            has_roi=has_configured_roi,
            homography=homography,
            bidirectional=bool(cam_cfg.get("bidirectional", False)),
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

        # 行为向量化器：在轨迹完成后一次性汇总 Phase1/2 所有字段
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

        # 输出：向量 CSV 给特征分析使用，事件 CSV 给后续告警/跨摄像头使用
        output_cfg = cfg.get("output", {})
        self.vector_writer = VectorWriter(
            output_format=output_cfg.get("format", "csv"),
        )
        self.event_writer = EventWriter()
        self.event_output_enabled = output_cfg.get("save_events", True)

        self.frame_skip = pipeline_cfg.get("frame_skip", 1)
        self.max_frames = pipeline_cfg.get("max_frames", 0)
        self.output_video_enabled = pipeline_cfg.get("output_video", False)
        # 批处理推理：把多帧合并成一次 YOLO 前向，摊薄每帧 Python 开销
        self.batch_size = max(1, int(pipeline_cfg.get("batch_size", 1)))
        # 解码预取：>0 时用独立线程解码，与推理重叠（见 PERFORMANCE.md P1）
        self.prefetch_frames = max(0, int(pipeline_cfg.get("prefetch_frames", 0)))

    def _flush_batch(self, frames, timestamps, video_writer=None):
        """处理缓冲区中的一批帧并清空。"""
        if not frames:
            return

        # 标注视频必须在“每帧当时”取轨迹快照：批处理结束后再调
        # get_active_tracklets() 拿到的是整批的最终状态，会把最后一帧的框
        # 画到前 15 帧上，甚至画出本批稍后才创建的轨迹。
        def _on_frame(frame, active_tracklets):
            if video_writer is not None:
                video_writer.write(self.visualizer.draw_frame(frame, active_tracklets))

        self.pipeline.process_frames(frames, timestamps, on_frame=_on_frame if video_writer is not None else None)
        frames.clear()
        timestamps.clear()

    def process_video(
        self,
        video_path: str,
        output_dir: str = "data/outputs",
        output_video_path: Optional[str] = None,
        start_frame: int = 0,
        name_suffix: str = "",
    ) -> str:
        """处理一段视频文件并输出行为向量。

        Args:
            video_path: 输入视频路径
            output_dir: 输出目录
            output_video_path: 标注视频输出路径（可选）
            start_frame: 起始帧（用于分窗处理长视频）
            name_suffix: 追加到输出文件名后缀，用于区分同一小时的多段录像
                （否则后一段会覆盖前一段的向量与事件 CSV）

        Returns:
            输出的行为向量 CSV 文件路径
        """
        for _, csv_path in self._process_video_windows(
            video_path, output_dir, output_video_path, start_frame, name_suffix
        ):
            return csv_path
        raise RuntimeError(f"No vector output for {video_path}")

    def process_video_windows(
        self, video_path: str, output_dir: str, window_frames: int, max_windows: int = 0,
    ):
        """Yield (source start frame, CSV path) for each window of one recording."""
        if window_frames <= 0 or max_windows < 0:
            raise ValueError("Expected positive window_frames and non-negative max_windows")
        yield from self._process_video_windows(
            video_path, output_dir, None, 0, "", window_frames, max_windows
        )

    def _process_video_windows(
        self, video_path, output_dir, output_video_path, start_frame, name_suffix,
        window_frames=0, max_windows=0,
    ):
        self.logger.info(f"Processing video: {video_path}")
        self.pipeline.reset()

        # 解析输出文件路径
        video_name = Path(video_path).stem
        camera_id = self.tracker.camera_id

        # 从视频文件名推断日期和小时，用于稳定生成输出目录和文件名
        date_str, hour_str = self._parse_video_datetime(video_name)

        # 打开视频
        cap = FrameReader.open(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        source_fps = cap.get(cv2.CAP_PROP_FPS) or self.frame_rate
        parsed_start = self._parse_video_start_time(video_name)
        time_known = parsed_start is not None
        if not time_known:
            # 不能用 datetime.now()：否则同一段录像在白天/晚上跑会得到不同的夜间判定，
            # 结果不可复现。改用固定占位时间，并把 time_known 置 0（禁用夜间相关判断）。
            parsed_start = datetime(2000, 1, 1, 12, 0, 0)
            self.logger.warning(
                "文件名未包含时间戳（%s），使用占位时间 %s；夜间相关判断已禁用",
                video_name,
                parsed_start.isoformat(),
            )
        video_start_time = parsed_start
        self.logger.info(f"Total frames: {total_frames}")

        # 分窗处理：从指定帧开始
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
            self.logger.info(f"Seek to frame {start_frame}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        reference_size = self.reference_resolution
        resize_to_reference = bool(reference_size and (width, height) != reference_size)
        self.pipeline.identity_crop_scale = (1.0, 1.0)
        if resize_to_reference:
            ref_width, ref_height = reference_size
            if width <= 0 or height <= 0 or abs(width * ref_height / (height * ref_width) - 1) > 0.02:
                cap.release()
                raise ValueError("Video aspect ratio differs from camera reference_resolution")
            self.logger.info("Resizing %sx%s frames to camera reference %sx%s", width, height, ref_width, ref_height)
            self.pipeline.identity_crop_scale = (width / ref_width, height / ref_height)

        # 视频写入器
        video_writer = None
        if self.output_video_enabled and output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            output_size = reference_size if resize_to_reference else (width, height)
            video_writer = cv2.VideoWriter(output_video_path, fourcc, self.frame_rate, output_size)

        # 逐帧处理
        frame_count = 0
        batch_frames = []
        batch_timestamps = []
        batch_size = self.batch_size
        # 解码预取：独立线程解码，与推理重叠；不改变帧顺序与时间戳
        reader = None
        if self.prefetch_frames > 0:
            reader = ThreadedFrameReader(cap, queue_size=self.prefetch_frames)
            self.logger.info(f"Decode prefetch enabled (queue={self.prefetch_frames})")
        window_index = 0
        emitted = 0
        frame_limit = window_frames * max_windows if window_frames else self.max_frames

        def write_window():
            nonlocal emitted
            vectors = self.pipeline.completed_vectors[emitted:]
            directory = os.path.join(output_dir, f"window_{window_index:03d}") if window_frames else output_dir
            path = self._write_video_outputs(
                vectors, directory, camera_id, date_str, hour_str,
                name_suffix, video_start_time, time_known,
            )
            emitted += len(vectors)
            return start_frame + window_index * window_frames, path

        try:
            while not frame_limit or frame_count < frame_limit:
                ret, frame = reader.read() if reader is not None else cap.read()
                if not ret:
                    break
                # Read ahead by one frame so an exact-boundary EOF finalizes the last window.
                if window_frames and frame_count and frame_count % window_frames == 0:
                    self._flush_batch(batch_frames, batch_timestamps, video_writer)
                    yield write_window()
                    window_index += 1

                frame_count += 1
                if frame_count % self.frame_skip != 0:
                    continue
                timestamp = video_start_time + timedelta(
                    seconds=(start_frame + frame_count - 1) / source_fps
                )
                if resize_to_reference:
                    frame = cv2.resize(frame, reference_size, interpolation=cv2.INTER_LINEAR)
                batch_frames.append(frame)
                batch_timestamps.append(timestamp)
                if len(batch_frames) >= batch_size:
                    self._flush_batch(batch_frames, batch_timestamps, video_writer)
                if frame_count % 100 == 0:
                    self.logger.info(
                        f"Frame {frame_count}/{total_frames}, "
                        f"completed vectors: {len(self.pipeline.completed_vectors)}"
                    )

            self._flush_batch(batch_frames, batch_timestamps, video_writer)
            remaining = self.pipeline.finalize()
            self.logger.info(f"Finalized {len(remaining)} remaining trajectories")
        finally:
            if reader is not None:
                reader.stop()
            cap.release()
            if video_writer is not None:
                video_writer.release()
        yield write_window()

    def _write_video_outputs(
        self, all_vectors, output_dir, camera_id, date_str, hour_str,
        name_suffix, video_start_time, time_known,
    ):
        # 给每条通行记录一个全局唯一 ID：摄像头 + 录像段起始时间 + track_id
        segment_start = (
            video_start_time.strftime("%Y%m%d%H%M%S") if video_start_time else "unknown"
        )
        for vector in all_vectors:
            vector["passage_id"] = f"{camera_id}_{segment_start}_{vector.get('track_id')}"
            vector["time_known"] = 1 if time_known else 0
            if not time_known:
                # 时间未知时不能声称“夜间”，否则夜间规则会误报
                vector["is_night"] = 0
                vector["night_ratio"] = 0.0

        csv_filename = build_vector_filename(camera_id, date_str, hour_str, name_suffix)
        vector_output_dir = os.path.join(output_dir, "vectors", camera_id, date_str)
        os.makedirs(vector_output_dir, exist_ok=True)
        csv_path = os.path.join(vector_output_dir, csv_filename)

        self.vector_writer.write(all_vectors, csv_path)
        self.logger.info(
            f"Output: {csv_path} ({len(all_vectors)} vehicle vectors)"
        )

        # 写出通行事件：每条完成轨迹转换为一条 vehicle_passage，供 Phase3/4 复用
        if self.event_output_enabled:
            event_output_dir = os.path.join(output_dir, "events", camera_id, date_str)
            event_filename = build_event_filename(camera_id, date_str, hour_str, name_suffix)
            event_path = os.path.join(event_output_dir, event_filename)
            self.event_writer.write(all_vectors, event_path)
            self.logger.info(
                f"Events: {event_path} ({len(all_vectors)} vehicle passage events)"
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
            return PlateDetector(model_path=plate_cfg.get("detector_weights"), conf_threshold=plate_cfg.get("detector_conf", 0.35), min_plate_width=plate_cfg.get("min_plate_width", 80))
        return None

    @staticmethod
    def _init_plate_ocr(model_cfg: dict):
        plate_cfg = model_cfg.get("plate", {}) if model_cfg else {}
        if plate_cfg.get("enabled", False):
            return PlateOCR(engine=plate_cfg.get("ocr_engine", "easyocr"), lang=plate_cfg.get("ocr_lang", "ch_sim"), min_confidence=plate_cfg.get("ocr_min_conf", 0.55))
        return None

    @staticmethod
    def _init_reid_extractor(model_cfg: dict):
        reid_cfg = model_cfg.get("reid", {}) if model_cfg else {}
        if reid_cfg.get("enabled", False):
            return ReIDExtractor(
                model_name=reid_cfg.get("model_name", "osnet_x1_0"),
                weights_path=reid_cfg.get("weights"),
                backend=reid_cfg.get("backend", "auto"),
                allow_torchvision_fallback=reid_cfg.get("allow_torchvision_fallback", False),
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
    def _infer_geometry_level(
        has_configured_roi: bool,
        entry_line: dict,
        exit_line: dict,
        px_per_meter: float,
        calibration: dict,
        has_homography: bool = False,
    ) -> str:
        """Infer which geometry-dependent behavior fields are trustworthy."""
        entry_valid = entry_line.get("p1", [0, 0]) != entry_line.get("p2", [0, 0])
        exit_valid = exit_line.get("p1", [0, 0]) != exit_line.get("p2", [0, 0])
        calibration_valid = (
            px_per_meter > 0
            and calibration.get("ref_length_m", 0) > 0
            and calibration.get("ref_point_1", [0, 0]) != calibration.get("ref_point_2", [0, 0])
        )

        # 有单应矩阵时，速度/路径/聚集距离可用透视关系换算
        if has_configured_roi and has_homography:
            return "calibrated"
        if has_configured_roi and entry_valid and exit_valid and calibration_valid:
            return "calibrated"
        if has_configured_roi:
            return "roi"
        return "none"

    @staticmethod
    def _parse_homography(calibration: dict):
        """解析 3x3 单应矩阵；缺失或格式错误时返回 None。"""
        raw = calibration.get("homography")
        if not raw:
            return None
        try:
            matrix = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if matrix.size != 9:
            return None
        return matrix.reshape(3, 3)

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
            if len(part) >= 14 and part[:14].isdigit():
                return part[:8], part[8:10]
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
            if len(part) >= 14 and part[:14].isdigit():
                try:
                    return datetime.strptime(part[:14], "%Y%m%d%H%M%S")
                except ValueError:
                    return None
            if len(part) == 8 and part.isdigit() and index + 1 < len(parts):
                time_part = parts[index + 1]
                if len(time_part) >= 6 and time_part[:6].isdigit():
                    try:
                        return datetime.strptime(part + time_part[:6], "%Y%m%d%H%M%S")
                    except ValueError:
                        return None
        return None
