"""从行为向量中抽取“车辆—车辆”时空关系，构建关系图谱。

关系类型（对应研究要点三）：
- ``co_occurrence``：两车通行时间段重叠（同屏/同时段出现）。
- ``following``：两车进入时间接近、路径形状相似（结伴/尾随）。
- ``path_similarity``：两车轨迹形状高度相似（同路线）。

说明：
- 轨迹坐标为摄像头像素坐标，跨摄像头不可直接比较，因此默认只在本摄像头内建边
  （``same_camera_only=True``）。跨摄像头关系属于 Phase 4。
- 轨迹相似度做了起点平移 + 路径长度归一化，只比较“形状”，不比较绝对尺度。
- 邻接矩阵权重为各关系强度的累加，范围大致在 [0, 3]。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    relation: str
    weight: float


@dataclass
class VehicleGraph:
    """车辆关系图谱。"""

    nodes: List[str] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)
    adjacency: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    node_index: Dict[str, int] = field(default_factory=dict)

    def edge_list(self) -> List[Tuple[str, str, str, float]]:
        return [(e.source, e.target, e.relation, e.weight) for e in self.edges]

    def neighbors(self, node: str) -> List[str]:
        index = self.node_index.get(node)
        if index is None:
            return []
        return [
            self.nodes[j] for j in np.nonzero(self.adjacency[index])[0] if j != index
        ]


# ---------------------------------------------------------------------------
# 基础解析
# ---------------------------------------------------------------------------


def parse_time(value) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def parse_trajectory(points) -> np.ndarray:
    """把 ``"x1,y1;x2,y2;..."`` 解析为 (N, 2) 数组。"""
    if points is None:
        return np.empty((0, 2), dtype=np.float64)
    if isinstance(points, np.ndarray):
        return np.asarray(points, dtype=np.float64).reshape(-1, 2)

    coords: List[Tuple[float, float]] = []
    for token in str(points).split(";"):
        token = token.strip()
        if not token:
            continue
        parts = token.split(",")
        if len(parts) != 2:
            continue
        try:
            coords.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    return np.asarray(coords, dtype=np.float64) if coords else np.empty((0, 2), dtype=np.float64)


def resample_trajectory(points: np.ndarray, k: int = 16) -> np.ndarray:
    """按弧长把轨迹重采样为 k 个点。"""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if points.shape[0] == 0:
        return points
    if points.shape[0] == 1 or k <= 1:
        return np.repeat(points[:1], max(k, 1), axis=0)

    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg)])
    total = cumulative[-1]
    if total <= 0:
        return np.repeat(points[:1], k, axis=0)

    targets = np.linspace(0.0, total, k)
    index = np.searchsorted(cumulative, targets, side="right") - 1
    index = np.clip(index, 0, len(points) - 2)
    frac = (targets - cumulative[index]) / np.maximum(seg[index], 1e-9)
    return points[index] + frac[:, None] * (points[index + 1] - points[index])


def _normalized_shape(points: np.ndarray, k: int = 16) -> np.ndarray:
    sampled = resample_trajectory(points, k)
    if sampled.shape[0] == 0:
        return sampled
    shifted = sampled - sampled[0]
    length = float(np.linalg.norm(np.diff(shifted, axis=0), axis=1).sum())
    if length <= 0:
        return np.zeros_like(shifted)
    return shifted / length


def trajectory_similarity(a, b, k: int = 16) -> float:
    """轨迹形状相似度，范围 [0, 1]。"""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 2)
    if a.shape[0] < 2 or b.shape[0] < 2:
        return 0.0
    pa = _normalized_shape(a, k)
    pb = _normalized_shape(b, k)
    if pa.shape[0] != pb.shape[0] or pa.shape[0] == 0:
        return 0.0
    mean_distance = float(np.mean(np.linalg.norm(pa - pb, axis=1)))
    return float(np.clip(1.0 - mean_distance, 0.0, 1.0))


def time_overlap_seconds(a_enter, a_exit, b_enter, b_exit) -> float:
    if a_enter is None or b_enter is None:
        return 0.0
    if a_exit is None:
        a_exit = a_enter
    if b_exit is None:
        b_exit = b_enter
    start = max(a_enter, b_enter)
    end = min(a_exit, b_exit)
    return max(0.0, (end - start).total_seconds())


def _node_key(vector: dict, index: int, use_global_vehicle_id: bool) -> str:
    if use_global_vehicle_id:
        global_id = str(vector.get("global_vehicle_id", "") or "").strip()
        if global_id:
            return f"gv:{global_id}"
    # passage_id 是流水线写入的全局唯一通行 ID，避免同一机位不同录像段的
    # track_id 撞车（track_id 只在段内唯一）
    passage_id = str(vector.get("passage_id", "") or "").strip()
    if passage_id:
        return passage_id
    camera = str(vector.get("camera_id", "") or "")
    track = str(vector.get("track_id", "") or index)
    return f"{camera}:{track}"


def _is_analysis_target(vector: dict) -> bool:
    target = str(vector.get("analysis_target", "")).strip().lower()
    if target in {"0", "false", "no"}:
        return False
    if target in {"1", "true", "yes"}:
        return True
    # 旧 CSV 没有该字段时，回退到 vehicle_group
    group = str(vector.get("vehicle_group", "ordinary")).strip().lower()
    return group in {"ordinary", ""}


# ---------------------------------------------------------------------------
# 图构建
# ---------------------------------------------------------------------------


class RelationConfig:
    def __init__(
        self,
        co_occurrence_min_overlap_sec: float = 1.0,
        co_occurrence_min_overlap_ratio: float = 0.5,
        follow_max_gap_sec: float = 10.0,
        path_similarity_threshold: float = 0.85,
        min_trajectory_points: int = 8,
        same_camera_only: bool = True,
        use_global_vehicle_id: bool = False,
        top_k: int = 0,
    ):
        self.co_occurrence_min_overlap_sec = co_occurrence_min_overlap_sec
        self.co_occurrence_min_overlap_ratio = co_occurrence_min_overlap_ratio
        self.follow_max_gap_sec = follow_max_gap_sec
        self.path_similarity_threshold = path_similarity_threshold
        self.min_trajectory_points = min_trajectory_points
        self.same_camera_only = same_camera_only
        self.use_global_vehicle_id = use_global_vehicle_id
        # 单车道场景下轨迹普遍相似，用 top-k 邻居把图变稀疏（0 表示不限制）
        self.top_k = top_k


def build_vehicle_graph(
    vectors: Sequence[dict],
    config: Optional[RelationConfig] = None,
    only_analysis_target: bool = True,
) -> VehicleGraph:
    """从行为向量构建车辆关系图谱。

    ``config.top_k > 0`` 时，只保留每个节点权重最高的 k 个邻居（取并集），
    避免单车道场景下轨迹普遍相似导致图退化为完全图。
    """
    config = config or RelationConfig()

    records = []
    for index, vector in enumerate(vectors):
        if only_analysis_target and not _is_analysis_target(vector):
            continue
        records.append(
            {
                "key": _node_key(vector, index, config.use_global_vehicle_id),
                "camera_id": str(vector.get("camera_id", "") or ""),
                "enter": parse_time(vector.get("enter_time")),
                "exit": parse_time(vector.get("exit_time")),
                "duration": float(vector.get("duration_sec") or 0.0),
                "trajectory": parse_trajectory(vector.get("trajectory_points")),
            }
        )

    nodes = [record["key"] for record in records]
    n = len(nodes)
    node_index = {key: i for i, key in enumerate(nodes)}
    adjacency = np.zeros((n, n), dtype=np.float64)
    edges: List[GraphEdge] = []

    # 先收集所有候选关系，便于之后按 top-k 裁剪
    pair_data: Dict[Tuple[int, int], dict] = {}
    for i in range(n):
        for j in range(i + 1, n):
            a, b = records[i], records[j]
            if config.same_camera_only and a["camera_id"] != b["camera_id"]:
                continue

            pair_edges: List[GraphEdge] = []
            weights: List[float] = []

            # 1) 共现：按较短通行时长的重叠比例归一化
            overlap = time_overlap_seconds(a["enter"], a["exit"], b["enter"], b["exit"])
            if overlap >= config.co_occurrence_min_overlap_sec:
                shorter = max(min(a["duration"], b["duration"]), 1e-9)
                ratio = float(np.clip(overlap / shorter, 0.0, 1.0))
                if ratio >= config.co_occurrence_min_overlap_ratio:
                    pair_edges.append(GraphEdge(a["key"], b["key"], "co_occurrence", round(ratio, 4)))
                    weights.append(ratio)

            # 2) 路径相似：过短的轨迹形状不可靠，不参与相似度判断
            if (
                a["trajectory"].shape[0] >= config.min_trajectory_points
                and b["trajectory"].shape[0] >= config.min_trajectory_points
            ):
                similarity = trajectory_similarity(a["trajectory"], b["trajectory"])
            else:
                similarity = 0.0
            if similarity >= config.path_similarity_threshold:
                pair_edges.append(GraphEdge(a["key"], b["key"], "path_similarity", round(similarity, 4)))
                weights.append(similarity)

            # 3) 跟随：进入时间接近 + 路径相似
            if a["enter"] is not None and b["enter"] is not None:
                gap = abs((a["enter"] - b["enter"]).total_seconds())
                if gap <= config.follow_max_gap_sec and similarity >= config.path_similarity_threshold:
                    pair_edges.append(GraphEdge(a["key"], b["key"], "following", round(similarity, 4)))
                    weights.append(similarity)

            if weights:
                pair_data[(i, j)] = {"weight": float(sum(weights)), "edges": pair_edges}

    # top-k 裁剪：保留每个节点最强的 k 条边（并集）
    if config.top_k and config.top_k > 0 and pair_data:
        incident: Dict[int, List[Tuple[float, Tuple[int, int]]]] = {i: [] for i in range(n)}
        for (i, j), data in pair_data.items():
            incident[i].append((data["weight"], (i, j)))
            incident[j].append((data["weight"], (i, j)))
        keep = set()
        for items in incident.values():
            items.sort(key=lambda item: -item[0])
            for _, key in items[: config.top_k]:
                keep.add(key)
    else:
        keep = set(pair_data.keys())

    for (i, j), data in pair_data.items():
        if (i, j) not in keep:
            continue
        adjacency[i, j] += data["weight"]
        adjacency[j, i] += data["weight"]
        edges.extend(data["edges"])

    return VehicleGraph(nodes=nodes, edges=edges, adjacency=adjacency, node_index=node_index)
