"""跨摄像头同车关联与轨迹拼接（研究要点四/Phase 4）。

思路：
1. 先用摄像头拓扑边给出的“通行时间窗口”筛选候选（A 车离开后，多久可能出现在 B）；
2. 使用可信车牌或 ReID 外观相似度做一对一匹配；
3. 用并查集把匹配对合并成 `global_vehicle_id`，形成跨摄像头车辆级身份。

"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..feature_extraction.plate_ocr import PlateOCR


_COLOR_GROUPS = {
    "white": "light", "silver": "light", "gray": "light", "grey": "light",
    "black": "dark", "dark": "dark",
    "red": "chromatic", "blue": "chromatic", "green": "chromatic",
    "yellow": "chromatic", "orange": "chromatic", "cyan": "chromatic",
    "purple": "chromatic", "brown": "chromatic",
}
SITE_TIMEZONE = timezone(timedelta(hours=8))


def _color_group(value) -> str:
    return _COLOR_GROUPS.get(str(value or "").strip().lower(), "")


def parse_time(value) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        # Older vector CSVs use local site time (UTC+8) without an offset.
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=SITE_TIMEZONE)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def parse_embedding(value) -> np.ndarray:
    if value is None:
        return np.zeros(0, dtype=np.float64)
    if isinstance(value, np.ndarray):
        return value.astype(np.float64).reshape(-1)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return np.zeros(0, dtype=np.float64)
    try:
        return np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return np.zeros(0, dtype=np.float64)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0 or a.shape != b.shape or not (np.isfinite(a).all() and np.isfinite(b).all()):
        return 0.0
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / norm)


def trusted_plate(row: dict) -> str:
    try:
        confidence = float(row.get("plate_confidence") or 0)
        votes = int(row.get("plate_votes") or 0)
    except (TypeError, ValueError, OverflowError):
        return ""
    if not np.isfinite(confidence) or confidence > 1 or confidence < (0.70 if votes >= 2 else 0.94):
        return ""
    return PlateOCR._clean_plate(str(row.get("plate_number") or ""))


class CrossCameraStitcher:
    """把相邻摄像头的通行记录匹配成同一辆车。"""

    def __init__(
        self,
        topology=None,
        similarity_threshold: float = 0.7,
        default_min_travel_sec: float = 5.0,
        default_max_travel_sec: float = 600.0,
        attribute_penalty: float = 0.05,
        ambiguity_margin: float = 0.0,
    ):
        self.topology = topology
        self.similarity_threshold = similarity_threshold
        self.default_min_travel_sec = default_min_travel_sec
        self.default_max_travel_sec = default_max_travel_sec
        # 车型/颜色不一致时对 ReID 分数做小幅惩罚（先验，不是硬过滤）
        self.attribute_penalty = attribute_penalty
        self.ambiguity_margin = ambiguity_margin

    def _attribute_penalty_for(self, source: dict, target: dict) -> float:
        penalty = 0.0
        source_type = str(source.get("vehicle_type", "") or "").strip().lower()
        target_type = str(target.get("vehicle_type", "") or "").strip().lower()
        if source_type and target_type and source_type != target_type:
            penalty += self.attribute_penalty

        source_color = _color_group(source.get("vehicle_color"))
        target_color = _color_group(target.get("vehicle_color"))
        if source_color and target_color and source_color != target_color:
            penalty += self.attribute_penalty

        return penalty

    def _time_window(self, from_camera: str, to_camera: str) -> Tuple[float, float]:
        if self.topology is not None:
            edge = self.topology.get_edge(from_camera, to_camera)
            if edge:
                return (
                    float(edge.get("min_travel_sec", self.default_min_travel_sec)),
                    float(edge.get("max_travel_sec", self.default_max_travel_sec)),
                )
        return self.default_min_travel_sec, self.default_max_travel_sec

    @staticmethod
    def passage_key(camera_id: str, passage: dict) -> str:
        """生成跨分段唯一的通行键。

        优先使用流水线写入的 ``passage_id``；其次 ``global_vehicle_id``；
        最后退化为 ``camera:源文件:track_id``，避免同一机位不同录像段的
        track_id 撞车。
        """
        passage_id = str(passage.get("passage_id", "") or "").strip()
        if passage_id:
            return passage_id

        global_id = str(passage.get("global_vehicle_id", "") or "").strip()
        if global_id and global_id not in {"0", "nan", "None"}:
            return f"gv:{global_id}"
        source = os.path.basename(str(passage.get("_source_file", "") or ""))
        if source:
            return f"{camera_id}:{source}:{passage.get('track_id')}"
        return f"{camera_id}:{passage.get('track_id')}"

    def match(
        self,
        source_passages: Sequence[dict],
        target_passages: Sequence[dict],
        from_camera: str,
        to_camera: str,
    ) -> List[dict]:
        """在 source（上游摄像头）与 target（下游摄像头）之间做一对一匹配。"""
        min_travel, max_travel = self._time_window(from_camera, to_camera)
        if not source_passages or not target_passages or not np.isfinite([min_travel, max_travel]).all():
            return []

        def observations(rows, time_field):
            return [
                (
                    parse_time(row.get(time_field))
                    if str(row.get("time_known", "")).strip().lower() not in {"0", "0.0", "false"} else None,
                    trusted_plate(row),
                    parse_embedding(row.get("reid_embedding")),
                    str(row.get("reid_backend") or ""),
                )
                for row in rows
            ]

        sources = observations(source_passages, "exit_time")
        targets = observations(target_passages, "enter_time")
        scores = np.full((len(sources), len(targets)), -np.inf)
        candidates = {}
        for i, (source_exit, source_plate, source_embedding, source_backend) in enumerate(sources):
            if source_exit is None:
                continue
            source = source_passages[i]
            for j, (target_enter, target_plate, target_embedding, target_backend) in enumerate(targets):
                if target_enter is None:
                    continue
                travel = (target_enter - source_exit).total_seconds()
                if travel < min_travel or travel > max_travel:
                    continue
                target = target_passages[j]
                if source_plate and target_plate and source_plate != target_plate:
                    continue
                comparable = (
                    source_backend in {"", "onnx"} and target_backend in {"", "onnx"}
                    and source_embedding.size != 74 and target_embedding.size != 74
                )
                similarity = cosine_similarity(source_embedding, target_embedding) if comparable else 0.0
                penalty = self._attribute_penalty_for(source, target)
                plate_match = bool(source_plate and source_plate == target_plate)
                score = similarity - penalty
                if not plate_match and similarity <= 0:
                    continue
                # A plate match ranks above any directly competing appearance match.
                scores[i, j] = 2.0 + max(0.0, similarity) * 0.1 if plate_match else score
                if not plate_match and score < self.similarity_threshold:
                    continue
                candidates[i, j] = {
                    "from_camera": from_camera,
                    "from_track": source.get("track_id"),
                    "to_camera": to_camera,
                    "to_track": target.get("track_id"),
                    "from_key": self.passage_key(from_camera, source),
                    "to_key": self.passage_key(to_camera, target),
                    "similarity": round(similarity, 4),
                    "attribute_penalty": round(penalty, 4),
                    "score": 1.0 if plate_match else round(score, 4),
                    "travel_sec": round(travel, 2),
                    "match_method": "plate" if plate_match else "reid",
                }

        if not candidates:
            return []
        best_rows = np.max(scores, axis=1)
        best_cols = np.max(scores, axis=0)
        # Dummy columns permit rows with insufficient evidence to stay unmatched.
        costs = np.zeros((len(sources), len(targets) + len(sources)))
        costs[:, :len(targets)] = 1e6
        for (i, j), match in candidates.items():
            score = scores[i, j]
            if self.ambiguity_margin > 0:
                other_row = max(scores[i, :j].max(initial=-np.inf), scores[i, j + 1:].max(initial=-np.inf))
                other_col = max(scores[:i, j].max(initial=-np.inf), scores[i + 1:, j].max(initial=-np.inf))
                if score < max(best_rows[i], best_cols[j]) or score - max(other_row, other_col) < self.ambiguity_margin:
                    continue
            costs[i, j] = -score
        matches = [
            candidates[i, j] for i, j in zip(*linear_sum_assignment(costs))
            if j < len(targets) and costs[i, j] < 0
        ]
        return matches

    def select_chain_matches(self, rows_by_camera: Dict[str, List[dict]], matches: Sequence[dict]) -> List[dict]:
        """Prevent forks and conflicting plates across multiple camera pairs."""
        plates = {
            self.passage_key(camera, row): trusted_plate(row)
            for camera, rows in rows_by_camera.items() for row in rows
        }
        successor = {}
        predecessor = {}
        accepted = []
        for match in sorted(matches, key=lambda row: (row["match_method"] == "plate", row["score"]), reverse=True):
            source, target = match["from_key"], match["to_key"]
            if source == target or source in successor or target in predecessor:
                continue
            head, tail = source, target
            while head in predecessor:
                head = predecessor[head]
            while tail in successor:
                tail = successor[tail]
            if head == target or tail == source:
                continue
            a, b = plates.get(head), plates.get(target)
            if a and b and a != b:
                continue
            plates[head] = a or b
            successor[source] = target
            predecessor[target] = source
            accepted.append(match)
        return accepted

    def assign_global_ids(self, nodes: Sequence, matches: Sequence[dict]) -> Dict:
        """用并查集把匹配对合并为全局车辆 ID。

        ``nodes`` 可以是 ``(camera, track)`` 元组，也可以是 ``passage_key`` 字符串；
        返回值的键与输入节点类型一致。
        """

        def normalize(node) -> str:
            if isinstance(node, tuple):
                return f"{node[0]}:{node[1]}"
            return str(node)

        keys = [normalize(node) for node in nodes]
        parent: Dict[str, str] = {key: key for key in keys}

        def find(node: str) -> str:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(a: str, b: str):
            root_a, root_b = find(a), find(b)
            if root_a != root_b:
                parent[root_b] = root_a

        for match in matches:
            a = match.get("from_key") or normalize((match["from_camera"], match["from_track"]))
            b = match.get("to_key") or normalize((match["to_camera"], match["to_track"]))
            if a in parent and b in parent:
                union(a, b)

        roots: Dict[str, int] = {}
        mapping = {}
        for node in nodes:
            root = find(normalize(node))
            if root not in roots:
                roots[root] = len(roots)
            mapping[node] = roots[root]
        return mapping
