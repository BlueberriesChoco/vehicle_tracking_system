"""跨窗口通行频次统计（时间频率分量 T 的真实输入）。

## 为什么需要这个模块

T 分量的频次那一路原本取 ``freq_count_24h``，但这个字段由
``FrequencyFeatureExtractor`` 产生，而该提取器的计数器**只在单段视频内累加**，
每段结束就 ``reset()``。单窗口运行时每辆车的 ``passage_id`` 唯一，
于是 ``get_count`` 恒返回 1，``excess = max(1-1, 0) = 0``，频次分量恒为 0。

``DailyAggregator._recompute_freq_index`` 里其实有一份"按 plate_hash 统计当日次数"
的正确逻辑，但 ``DailyAggregator`` 是整目录级的重操作（要读全部小时段、做 ReID 跨段拼接），
从未被任何脚本接入，因此 T 一直拿不到真值。

本模块只做其中「数次数」这一件事，轻量、无副作用、可单独测试。

## 身份键的层级（这是本模块的核心约束）

统计"同一辆车当天来了几次"必须有一个跨窗口稳定的身份键。按可靠性排序：

1. ``plate_hash``：**唯一可靠**的身份键，来自车牌识别；
2. ``global_vehicle_id``：跨机位 ReID 拼接的结果，覆盖同机位跨窗口也有效；
3. ``plate_number``：原始车牌号，需配合 camera 才能避免不同机位串号。

三者都缺失时**无法判断是否同一辆车**，此时必须如实返回"不可用"，
而不是把每辆车都当成只来了一次——后者会让 T 恒为 0，把"没有数据"
伪装成"时间上很干净"。本项目的实测就是这种情况：444 辆车里只有 2 辆有
``plate_hash``（0.45%），所以 T 在实际数据上必然不可用。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple


#: 身份键覆盖率达到该比例才认为频次统计有效。
DEFAULT_MIN_COVERAGE: float = 0.5


def identity_key(record: dict) -> Optional[Tuple[str, str]]:
    """返回 ``(层级名, 键值)``；无法确定身份时返回 None。

    层级名写进结果便于排查"这批频次是基于什么算出来的"。
    """
    plate_hash = str(record.get("plate_hash", "") or "").strip()
    if plate_hash:
        return ("plate_hash", plate_hash)

    global_id = str(record.get("global_vehicle_id", "") or "").strip()
    if global_id:
        return ("global_vehicle_id", global_id)

    plate_number = str(record.get("plate_number", "") or "").strip()
    if plate_number:
        camera = str(record.get("camera_id", "") or "").strip()
        return ("plate_number", f"{camera}:{plate_number}")

    return None


def count_passages(records: Sequence[dict]) -> Dict[str, object]:
    """统计每条记录所属车辆的通行次数。

    Returns:
        ``{
            "counts": [...与输入等长的次数（无法确定身份时为 0）],
            "coverage": 可确定身份的记录占比,
            "key_level": 主要使用的身份键层级（或 "none"）,
            "distinct": 可确定身份的不同车辆数,
            "available": 覆盖率是否达到阈值,
        }``

    ``counts`` 中无法确定身份的记录填 **0** 而不是 1：
    0 表示"未知"，1 表示"确认只来一次"，两者语义不同，
    混用会让下游把"没识别出车牌"当成"确认只来一次"。
    """
    keys: List[Optional[Tuple[str, str]]] = [identity_key(record) for record in records]

    counters: Dict[Tuple[str, str], int] = {}
    for key in keys:
        if key is not None:
            counters[key] = counters.get(key, 0) + 1

    counts = [counters.get(key, 0) if key is not None else 0 for key in keys]

    identified = sum(1 for key in keys if key is not None)
    total = len(records) or 1
    coverage = identified / total

    level_counts: Dict[str, int] = {}
    for key in keys:
        if key is not None:
            level_counts[key[0]] = level_counts.get(key[0], 0) + 1
    dominant = max(level_counts, key=level_counts.get) if level_counts else "none"

    return {
        "counts": counts,
        "coverage": round(coverage, 4),
        "key_level": dominant,
        "distinct": len(counters),
        "available": coverage >= DEFAULT_MIN_COVERAGE,
    }


def apply_passage_counts(
    records: Sequence[dict],
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> Tuple[List[dict], Dict[str, object]]:
    """把统计出的次数写回 ``freq_count_24h``，并返回统计摘要。

    覆盖率不足时**不改写** ``freq_count_24h``（保留原值并标记不可用），
    避免用不可靠的身份键造出一堆假的"来了很多次"。

    Returns:
        ``(enriched_records, summary)``，summary 里含 ``applied`` 标记。
    """
    stats = count_passages(records)
    available = stats["coverage"] >= min_coverage

    enriched: List[dict] = []
    for record, count in zip(records, stats["counts"]):
        item = dict(record)
        item["freq_identity_coverage"] = stats["coverage"]
        item["freq_identity_level"] = stats["key_level"]
        if available and count > 0:
            item["freq_count_24h"] = count
            item["freq_count_source"] = "identity_key"
        else:
            item["freq_count_source"] = "segment_local"
        enriched.append(item)

    summary = {
        **stats,
        "applied": bool(available),
        "min_coverage": min_coverage,
    }
    return enriched, summary
