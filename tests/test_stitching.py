"""跨摄像头同车关联与轨迹拼接的回归测试。"""

import numpy as np
import pytest

from src.trajectory.stitching import (
    CrossCameraStitcher,
    cosine_similarity,
    parse_embedding,
)


def _passage(track, enter, exit_, embedding):
    return {
        "track_id": track,
        "enter_time": enter,
        "exit_time": exit_,
        "reid_embedding": list(embedding),
    }


def _base_time(minute=0, second=0):
    return f"2026-09-01T10:{minute:02d}:{second:02d}"


def test_cosine_similarity():
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(1.0)
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_parse_embedding_from_json_string():
    embedding = parse_embedding("[0.1, 0.2, 0.3]")
    np.testing.assert_allclose(embedding, [0.1, 0.2, 0.3])


def test_match_three_vehicles_across_cameras():
    source = [
        _passage(1, _base_time(0, 0), _base_time(0, 0), [1.0, 0.0, 0.0]),
        _passage(2, _base_time(1, 0), _base_time(1, 0), [0.0, 1.0, 0.0]),
        _passage(3, _base_time(2, 0), _base_time(2, 0), [0.0, 0.0, 1.0]),
    ]
    target = [
        _passage(11, _base_time(0, 30), _base_time(0, 30), [1.0, 0.0, 0.0]),
        _passage(12, _base_time(1, 30), _base_time(1, 30), [0.0, 1.0, 0.0]),
        _passage(13, _base_time(2, 30), _base_time(2, 30), [0.0, 0.0, 1.0]),
    ]

    matches = CrossCameraStitcher(similarity_threshold=0.7).match(source, target, "camA", "camB")

    assert len(matches) == 3
    assert {match["from_track"]: match["to_track"] for match in matches} == {1: 11, 2: 12, 3: 13}


def test_match_rejects_outside_time_window():
    source = [_passage(1, _base_time(0, 0), _base_time(0, 0), [1.0, 0.0])]
    # 1 小时后才出现，超出默认 600 秒窗口
    target = [_passage(11, "2026-09-01T11:00:00", "2026-09-01T11:00:00", [1.0, 0.0])]

    matches = CrossCameraStitcher().match(source, target, "camA", "camB")

    assert matches == []


def test_match_rejects_low_similarity():
    source = [_passage(1, _base_time(0, 0), _base_time(0, 0), [1.0, 0.0])]
    target = [_passage(11, _base_time(0, 30), _base_time(0, 30), [0.0, 1.0])]

    matches = CrossCameraStitcher(similarity_threshold=0.7).match(source, target, "camA", "camB")

    assert matches == []


def test_match_is_one_to_one():
    source = [
        _passage(1, _base_time(0, 0), _base_time(0, 0), [1.0, 0.0]),
        _passage(2, _base_time(0, 1), _base_time(0, 1), [1.0, 0.0]),
    ]
    target = [_passage(11, _base_time(0, 30), _base_time(0, 30), [1.0, 0.0])]

    matches = CrossCameraStitcher().match(source, target, "camA", "camB")

    assert len(matches) == 1


def test_assign_global_ids_merges_matched_pairs():
    nodes = [("camA", "1"), ("camB", "11"), ("camB", "12")]
    matches = [
        {"from_camera": "camA", "from_track": 1, "to_camera": "camB", "to_track": 11},
    ]

    mapping = CrossCameraStitcher().assign_global_ids(nodes, matches)

    assert mapping[("camA", "1")] == mapping[("camB", "11")]
    assert mapping[("camB", "12")] != mapping[("camA", "1")]


def test_assign_global_ids_with_passage_id_nodes():
    """节点用 passage_id 字符串时，匹配对里的 from_key/to_key 必须能对上。"""
    nodes = ["admin_in_20260901090712_4", "admin_out_20260901093242_4"]
    matches = [
        {
            "from_camera": "admin_in",
            "from_track": "4",
            "to_camera": "admin_out",
            "to_track": "4",
            "from_key": "admin_in_20260901090712_4",
            "to_key": "admin_out_20260901093242_4",
            "similarity": 0.89,
        }
    ]

    mapping = CrossCameraStitcher().assign_global_ids(nodes, matches)

    assert mapping["admin_in_20260901090712_4"] == mapping["admin_out_20260901093242_4"]


def test_passage_key_is_unique_across_source_files():
    first = {"track_id": "1", "_source_file": "vec_cam_20260901_08.csv"}
    second = {"track_id": "1", "_source_file": "vec_cam_20260901_18.csv"}

    key_first = CrossCameraStitcher.passage_key("camA", first)
    key_second = CrossCameraStitcher.passage_key("camA", second)

    assert key_first != key_second
    assert key_first.endswith("vec_cam_20260901_08.csv:1")


def test_passage_key_prefers_global_vehicle_id():
    passage = {"track_id": "1", "global_vehicle_id": "42", "_source_file": "x.csv"}
    assert CrossCameraStitcher.passage_key("camA", passage) == "gv:42"


def test_passage_key_prefers_passage_id():
    passage = {
        "track_id": "1",
        "passage_id": "admin_in_20260901080110_1",
        "_source_file": "x.csv",
    }
    assert CrossCameraStitcher.passage_key("admin_in", passage) == "admin_in_20260901080110_1"


def test_attribute_penalty_demotes_color_mismatch():
    source = _passage(1, _base_time(0, 0), _base_time(0, 0), [1.0, 0.0])
    source["vehicle_type"] = "sedan"
    source["vehicle_color"] = "white"
    target = _passage(11, _base_time(0, 30), _base_time(0, 30), [1.0, 0.0])
    target["vehicle_type"] = "sedan"
    target["vehicle_color"] = "blue"

    matches = CrossCameraStitcher(similarity_threshold=0.7).match([source], [target], "camA", "camB")

    assert len(matches) == 1
    # 颜色分组不同 -> 分数被惩罚，但仍在阈值以上
    assert matches[0]["similarity"] > matches[0]["score"]
    assert matches[0]["attribute_penalty"] > 0


def test_attribute_penalty_zero_for_consistent_attributes():
    source = _passage(1, _base_time(0, 0), _base_time(0, 0), [1.0, 0.0])
    source["vehicle_type"] = "sedan"
    source["vehicle_color"] = "white"
    target = _passage(11, _base_time(0, 30), _base_time(0, 30), [1.0, 0.0])
    target["vehicle_type"] = "sedan"
    target["vehicle_color"] = "white"

    matches = CrossCameraStitcher(similarity_threshold=0.7).match([source], [target], "camA", "camB")

    assert matches[0]["attribute_penalty"] == 0.0
    assert matches[0]["score"] == matches[0]["similarity"]
