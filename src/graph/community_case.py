"""团伙案例：从图谱社区中挑选最值得核查的子图。"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Tuple


def rank_communities(nodes: List[dict], min_size: int = 3) -> List[Tuple[int, List[dict], float]]:
    """按“规模 × 平均图异常”给社区排序。

    - 规模小于 ``min_size`` 的社区不参与（单车不成团）；
    - 排序分 = 平均 G × sqrt(规模)，兼顾“异常程度”和“团伙规模”。

    返回 ``[(community_id, member_nodes, mean_g), ...]``，已按分数降序。
    """
    members = defaultdict(list)
    for node in nodes:
        members[node["community_id"]].append(node)

    ranked = []
    for community_id, items in members.items():
        if len(items) < min_size:
            continue
        mean_g = sum(item["graph_anomaly_index"] for item in items) / len(items)
        ranked.append((community_id, items, mean_g))

    ranked.sort(key=lambda entry: -(entry[2] * (len(entry[1]) ** 0.5)))
    return ranked
