"""跨摄像头同车关联与轨迹拼接（研究要点四/Phase 4）。

思路：
1. 先用摄像头拓扑边给出的“通行时间窗口”筛选候选（A 车离开后，多久可能出现在 B）；
2. 再用 ReID 外观相似度做一对一匹配（贪心，按相似度降序）；
3. 用并查集把匹配对合并成 `global_vehicle_id`，形成跨摄像头车辆级身份。

说明：当前 ReID 使用 HSV 直方图 fallback，区分度有限；接入正式 ReID 模型后，
同一套匹配逻辑可直接复用。
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


_COLOR_GROUPS = {
    "white": "light", "silver": "light", "gray": "light", "grey": "light",
    "black": "dark", "dark": "dark",
    "red": "chromatic", "blue": "chromatic", "green": "chromatic",
    "yellow": "chromatic", "orange": "chromatic", "cyan": "chromatic",
    "purple": "chromatic", "brown": "chromatic",
}


def _color_group(value) -> str:
    return _COLOR_GROUPS.get(str(value or "").strip().lower(), "")


def parse_time(value) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
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
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return 0.0
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / norm)


class CrossCameraStitcher:
    """把相邻摄像头的通行记录匹配成同一辆车。"""

    def __init__(
        self,
        topology=None,
        similarity_threshold: float = 0.7,
        default_min_travel_sec: float = 5.0,
        default_max_travel_sec: float = 600.0,
        attribute_penalty: float = 0.05,
    ):
        self.topology = topology
        self.similarity_threshold = similarity_threshold
        self.default_min_travel_sec = default_min_travel_sec
        self.default_max_travel_sec = default_max_travel_sec
        # 车型/颜色不一致时对 ReID 分数做小幅惩罚（先验，不是硬过滤）
        self.attribute_penalty = attribute_penalty

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

        candidates = []
        for source in source_passages:
            source_exit = parse_time(source.get("exit_time"))
            if source_exit is None:
                continue
            source_embedding = parse_embedding(source.get("reid_embedding"))
            for target in target_passages:
                target_enter = parse_time(target.get("enter_time"))
                if target_enter is None:
                    continue
                travel = (target_enter - source_exit).total_seconds()
                if travel < min_travel or travel > max_travel:
                    continue
                similarity = cosine_similarity(
                    source_embedding, parse_embedding(target.get("reid_embedding"))
                )
                penalty = self._attribute_penalty_for(source, target)
                score = similarity - penalty
                if score >= self.similarity_threshold:
                    candidates.append((score, travel, source, target, similarity, penalty))

        # 贪心一对一：综合分高优先，避免一辆车被重复匹配
        candidates.sort(key=lambda item: -item[0])
        used_source = set()
        used_target = set()
        matches = []
        for score, travel, source, target, similarity, penalty in candidates:
            source_key = self.passage_key(from_camera, source)
            target_key = self.passage_key(to_camera, target)
            if source_key in used_source or target_key in used_target:
                continue
            used_source.add(source_key)
            used_target.add(target_key)
            matches.append(
                {
                    "from_camera": from_camera,
                    "from_track": source.get("track_id"),
                    "to_camera": to_camera,
                    "to_track": target.get("track_id"),
                    "from_key": source_key,
                    "to_key": target_key,
                    "similarity": round(similarity, 4),
                    "attribute_penalty": round(penalty, 4),
                    "score": round(score, 4),
                    "travel_sec": round(travel, 2),
                }
            )
        return matches

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
