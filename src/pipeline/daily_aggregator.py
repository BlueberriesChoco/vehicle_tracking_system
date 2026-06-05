"""日轨迹聚合器。

将同一摄像头24小时分段的轨迹按车辆外观特征（ReID embedding）
和时空约束进行跨段拼接，解决同一车辆在不同小时段视频中被分配
不同 track_id 的问题。

核心逻辑：
  1. 读入当天所有小时段的行为向量 CSV
  2. 按时间排序，对相邻视频段边界的轨迹做 ReID 匹配
  3. 匹配成功的合并为同一车辆，统一 global_vehicle_id
  4. 输出日聚合 CSV
"""

import os
import csv
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np


class DailyAggregator:
    """日轨迹聚合器。

    Attributes:
        merge_window_sec: 跨段拼接时间窗口（秒），默认60秒——
            前一段最后出现和后一段首次出现的时间差在此窗口内才尝试匹配
        similarity_threshold: ReID 余弦相似度阈值
        spatio_radius_m: 空间匹配半径（米），ReID 不可用时退化到空间匹配
    """

    def __init__(
        self,
        merge_window_sec: float = 60.0,
        similarity_threshold: float = 0.7,
        spatio_radius_m: float = 10.0,
    ):
        self.merge_window_sec = merge_window_sec
        self.similarity_threshold = similarity_threshold
        self.spatio_radius_m = spatio_radius_m

    def aggregate(
        self,
        vector_dir: str,
        output_path: str,
        reid_extractor=None,
    ) -> str:
        """对一天内所有小时段向量进行聚合。

        Args:
            vector_dir: 包含所有小时段 CSV 的目录
            output_path: 日聚合输出 CSV 路径
            reid_extractor: ReID 特征提取器（可选，用于跨段匹配）

        Returns:
            日聚合 CSV 文件路径
        """
        # 1. 加载所有小时段向量
        hourly_files = sorted([
            f for f in os.listdir(vector_dir)
            if f.endswith(".csv") and f.startswith("vec_")
        ])

        if not hourly_files:
            return ""

        all_vectors: List[dict] = []
        for fname in hourly_files:
            filepath = os.path.join(vector_dir, fname)
            vectors = self._load_csv(filepath)
            for vec in vectors:
                vec["_segment_id"] = fname
            all_vectors.extend(vectors)

        if len(all_vectors) <= 1:
            self._write_csv(all_vectors, output_path)
            return output_path

        # 2. 按进入时间排序
        all_vectors.sort(key=lambda v: v.get("enter_time", ""))

        # 3. 构建跨段匹配图
        # 策略：对相邻向量做时空约束 + ReID 相似度匹配
        global_id_map: Dict[Tuple[str, str], int] = {}
        next_global_id = 0
        global_groups: Dict[int, List[dict]] = defaultdict(list)

        # 按小时分组
        hourly_groups: Dict[str, List[dict]] = defaultdict(list)
        for vec in all_vectors:
            hour_key = self._get_hour_key(vec.get("enter_time", ""))
            hourly_groups[hour_key].append(vec)

        sorted_hours = sorted(hourly_groups.keys())

        # 逐小时处理并跨段匹配
        prev_boundary_vectors: List[dict] = []  # 前一段末尾的车辆

        for hour in sorted_hours:
            hour_vectors = hourly_groups[hour]
            if not hour_vectors:
                continue

            # 当前段的起始车辆（前5分钟进入的）
            current_head = self._get_boundary_vectors(hour_vectors, head=True)

            # 与前一段末尾做匹配
            if prev_boundary_vectors and current_head:
                self._match_across_segments(
                    prev_boundary_vectors, current_head,
                    global_id_map, reid_extractor,
                )

            # 分配 global_id 给当前段尚未分配的向量
            for vec in hour_vectors:
                track_key = self._get_track_key(vec)
                if track_key not in global_id_map:
                    global_id_map[track_key] = next_global_id
                    next_global_id += 1

                gid = global_id_map[track_key]
                global_groups[gid].append(vec)

            # 当前段末尾作为下一次匹配的前一段
            prev_boundary_vectors = self._get_boundary_vectors(hour_vectors, head=False)

        # 4. 合并同组向量
        merged_vectors = []
        for gid, group in global_groups.items():
            merged = self._merge_group(gid, group)
            merged_vectors.append(merged)

        # 5. 重新计算高频通行指数
        merged_vectors = self._recompute_freq_index(merged_vectors)

        # 6. 写出
        self._write_csv(merged_vectors, output_path)
        return output_path

    def _match_across_segments(
        self,
        prev_vectors: List[dict],
        curr_vectors: List[dict],
        global_id_map: Dict[Tuple[str, str], int],
        reid_extractor=None,
    ):
        """在前后两段的边界车辆之间做匹配。

        匹配条件：
        1. 时间差 < merge_window_sec
        2. 空间距离 < spatio_radius_m（若有轨迹点）
        3. ReID 相似度 > similarity_threshold（若有 ReID 模型）
        """
        matched_current_keys = set()
        for pv in prev_vectors:
            p_exit = self._parse_time(pv.get("exit_time", ""))
            if p_exit is None:
                continue

            p_endpoint = self._get_last_trajectory_point(pv)

            for cv in curr_vectors:
                c_key = self._get_track_key(cv)
                if c_key in matched_current_keys:
                    continue
                c_enter = self._parse_time(cv.get("enter_time", ""))
                if c_enter is None:
                    continue

                time_diff = abs((c_enter - p_exit).total_seconds())
                if time_diff > self.merge_window_sec:
                    continue

                c_startpoint = self._get_first_trajectory_point(cv)

                # 空间距离
                spatial_ok = True
                if p_endpoint and c_startpoint:
                    dx = p_endpoint[0] - c_startpoint[0]
                    dy = p_endpoint[1] - c_startpoint[1]
                    dist_px = (dx * dx + dy * dy) ** 0.5
                    # 粗略判断（像素距离 < 合理值）
                    if dist_px > 150:  # 像素
                        spatial_ok = False

                if not spatial_ok:
                    continue

                # ReID 匹配（Phase 2：需要存储 embedding）
                identity_match = False
                p_plate = pv.get("plate_hash", "")
                c_plate = cv.get("plate_hash", "")
                if p_plate and c_plate and p_plate == c_plate:
                    identity_match = True
                if reid_extractor is not None:
                    p_emb = self._parse_embedding(pv.get("reid_embedding", []))
                    c_emb = self._parse_embedding(cv.get("reid_embedding", []))
                    if len(p_emb) > 0 and len(c_emb) > 0:
                        sim = reid_extractor.compute_similarity(p_emb, c_emb)
                        if sim >= self.similarity_threshold:
                            identity_match = True

                if not identity_match:
                    continue

                # 匹配成功：共享同一 global_id
                p_key = self._get_track_key(pv)
                c_key = self._get_track_key(cv)
                if p_key in global_id_map:
                    global_id_map[c_key] = global_id_map[p_key]
                elif c_key in global_id_map:
                    global_id_map[p_key] = global_id_map[c_key]
                else:
                    # 新建共享 ID
                    pass  # 在后续统一分配时处理
                matched_current_keys.add(c_key)
                break

    def _merge_group(self, global_id: int, group: List[dict]) -> dict:
        """将同一车辆的多段轨迹合并为一条记录。"""
        if len(group) == 1:
            vec = dict(group[0])
            vec["global_vehicle_id"] = global_id
            vec["segment_count"] = 1
            return vec

        # 按时间排序
        group.sort(key=lambda v: v.get("enter_time", ""))

        first = group[0]
        last = group[-1]

        # 聚合统计量
        total_duration = sum(self._as_float(v.get("duration_sec", 0)) for v in group)
        total_length = sum(self._as_float(v.get("trajectory_length_m", 0)) for v in group)
        avg_speeds = [
            self._as_float(v.get("avg_speed_ms", 0))
            for v in group if self._as_float(v.get("avg_speed_ms", 0)) > 0
        ]
        max_dwells = [self._as_float(v.get("max_dwell_sec", 0)) for v in group]
        stop_counts = sum(self._as_int(v.get("stop_count", 0)) for v in group)

        merged = {
            "track_id": global_id,
            "global_vehicle_id": global_id,
            "camera_id": first.get("camera_id", ""),
            "vehicle_type": self._majority_vote(group, "vehicle_type"),
            "vehicle_color": self._majority_vote(group, "vehicle_color"),
            "plate_number": first.get("plate_number", ""),
            "plate_hash": first.get("plate_hash", ""),
            "geometry_level": first.get("geometry_level", "none"),
            "speed_reliable": first.get("speed_reliable", 0),
            "path_reliable": first.get("path_reliable", 0),
            "aggregation_reliable": first.get("aggregation_reliable", 0),
            "passage_reliable": first.get("passage_reliable", 0),
            "enter_time": first.get("enter_time", ""),
            "exit_time": last.get("exit_time", ""),
            "duration_sec": round(total_duration, 2),
            "trajectory_length_m": round(total_length, 2),
            "avg_speed_ms": round(np.mean(avg_speeds), 3) if avg_speeds else 0,
            "max_speed_ms": max(self._as_float(v.get("max_speed_ms", 0)) for v in group),
            "speed_variance": round(np.mean([self._as_float(v.get("speed_variance", 0)) for v in group]), 3),
            "max_dwell_sec": max(max_dwells) if max_dwells else 0,
            "stop_count": stop_counts,
            "dwell_ratio": round(
                sum(self._as_float(v.get("dwell_ratio", 0)) * self._as_float(v.get("duration_sec", 0)) for v in group) /
                total_duration if total_duration > 0 else 0, 4
            ),
            "path_deviation": max(self._as_float(v.get("path_deviation", 0)) for v in group),
            "path_smoothness": round(np.mean([self._as_float(v.get("path_smoothness", 0)) for v in group]), 4),
            "is_night": first.get("is_night", 0),
            "night_ratio": first.get("night_ratio", 0),
            "freq_index": first.get("freq_index", 0),
            "freq_count_24h": first.get("freq_count_24h", 0),
            "aggregation_index": max(self._as_float(v.get("aggregation_index", 0)) for v in group),
            "nearest_vehicle_m": min(self._as_float(v.get("nearest_vehicle_m", -1)) for v in group),
            "trajectory_points": first.get("trajectory_points", ""),
            "reid_embedding": first.get("reid_embedding", []),
            "segment_count": len(group),
            "anomaly_score": first.get("anomaly_score", 0),
            "is_anomaly": first.get("is_anomaly", 0),
            "alert_reason": first.get("alert_reason", ""),
        }

        return merged

    def _recompute_freq_index(self, vectors: List[dict]) -> List[dict]:
        """用日聚合后的计数重新计算高频通行指数。"""
        plate_counts = defaultdict(int)
        for v in vectors:
            ph = v.get("plate_hash", "")
            if ph:
                plate_counts[ph] += 1

        if len(plate_counts) <= 1:
            return vectors

        all_counts = list(plate_counts.values())
        mean = np.mean(all_counts)
        std = np.std(all_counts)
        if std == 0:
            std = 1.0

        for v in vectors:
            ph = v.get("plate_hash", "")
            count = plate_counts.get(ph, 1) if ph else 1
            v["freq_count_24h"] = count
            v["freq_index"] = round((count - mean) / std, 3)

        return vectors

    @staticmethod
    def _get_hour_key(time_str: str) -> str:
        """从 ISO 时间戳提取小时键。"""
        if len(time_str) >= 13:
            return time_str[:13]  # "2026-05-20T08"
        return time_str[:10] if time_str else "unknown"

    @staticmethod
    def _get_boundary_vectors(
        vectors: List[dict], head: bool = True, window_min: int = 5
    ) -> List[dict]:
        """获取一段视频的边界轨迹（开头或末尾）。"""
        if not vectors:
            return []

        timed_vectors = []
        for v in vectors:
            parsed = DailyAggregator._parse_time(v.get("enter_time", ""))
            if parsed is not None:
                timed_vectors.append((v, parsed))
        if not timed_vectors:
            return []

        sorted_vectors = sorted(timed_vectors, key=lambda item: item[1])
        window = timedelta(minutes=window_min)
        if head:
            ref_time = sorted_vectors[0][1]
            return [v for v, timestamp in sorted_vectors if timestamp <= ref_time + window]

        ref_time = sorted_vectors[-1][1]
        return [v for v, timestamp in sorted_vectors if timestamp >= ref_time - window]

    @staticmethod
    def _get_last_trajectory_point(vec: dict) -> Optional[Tuple[float, float]]:
        pts_str = vec.get("trajectory_points", "")
        if not pts_str:
            return None
        parts = pts_str.split(";")
        if not parts:
            return None
        last = parts[-1]
        coords = last.split(",")
        if len(coords) == 2:
            return float(coords[0]), float(coords[1])
        return None

    @staticmethod
    def _get_first_trajectory_point(vec: dict) -> Optional[Tuple[float, float]]:
        pts_str = vec.get("trajectory_points", "")
        if not pts_str:
            return None
        parts = pts_str.split(";")
        if not parts:
            return None
        first = parts[0]
        coords = first.split(",")
        if len(coords) == 2:
            return float(coords[0]), float(coords[1])
        return None

    @staticmethod
    def _parse_time(time_str: str) -> Optional[datetime]:
        if not time_str:
            return None
        try:
            return datetime.fromisoformat(time_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _get_track_key(vec: dict) -> Tuple[str, str]:
        return str(vec.get("_segment_id", "")), str(vec.get("track_id", "-1"))

    @staticmethod
    def _parse_embedding(value) -> np.ndarray:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (TypeError, ValueError):
                return np.array([])
        return np.array(value if value is not None else [], dtype=np.float64)

    @staticmethod
    def _as_float(value) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _as_int(value) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _majority_vote(group: List[dict], key: str) -> str:
        """对分类标签做多数投票。"""
        votes = defaultdict(int)
        for v in group:
            val = v.get(key, "")
            if val and val != "unknown":
                votes[val] += 1
        if votes:
            return max(votes, key=votes.get)
        return group[0].get(key, "unknown")

    @staticmethod
    def _load_csv(filepath: str) -> List[dict]:
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)

    @staticmethod
    def _write_csv(vectors: List[dict], filepath: str):
        if not vectors:
            return
        from ..output.vector_writer import VectorWriter
        writer = VectorWriter()
        writer.write(vectors, filepath)
