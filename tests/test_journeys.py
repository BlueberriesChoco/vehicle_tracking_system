import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_journeys import assign_pairs, build_chain_journeys, build_journeys  # noqa: E402


def _row(passage, enter, exit_, embedding, camera="x"):
    return {
        "passage_id": passage,
        "camera_id": camera,
        "enter_time": f"2026-09-01T{enter}",
        "exit_time": f"2026-09-01T{exit_}",
        "reid_embedding": embedding,
    }


def test_travel_window_is_enforced():
    """穿行低于 20 秒的配对必须被排除（拓扑下限）。"""
    rows_a = [_row("a1", "08:00:00", "08:00:05", "[1.0, 0.0]")]
    rows_b = [_row("b1", "08:00:06", "08:00:10", "[1.0, 0.0]")]

    assert build_journeys(rows_a, rows_b) == []


def test_one_to_one_assignment():
    """两辆外观相近的车不能同时抢同一个下游记录。"""
    rows_a = [
        _row("a1", "08:00:00", "08:00:05", "[1.0, 0.0]"),
        _row("a2", "08:00:20", "08:00:25", "[0.99, 0.01]"),
    ]
    rows_b = [_row("b1", "08:02:00", "08:02:05", "[1.0, 0.0]")]

    pairs = assign_pairs(rows_a, rows_b)

    assert len(pairs) == 1
    assert list(pairs.values())[0]["to"] == "b1"


def test_chain_journey_composes_three_legs():
    """A→B→C 应合成一条三段行程。"""
    camera_a = [_row("a1", "08:00:00", "08:00:05", "[1.0, 0.0, 0.0]", "A")]
    camera_b = [_row("b1", "08:02:00", "08:02:05", "[1.0, 0.0, 0.0]", "B")]
    camera_c = [_row("c1", "08:05:00", "08:05:05", "[1.0, 0.0, 0.0]", "C")]

    journeys = build_chain_journeys(
        {"A": camera_a, "B": camera_b, "C": camera_c},
        [("A", "B"), ("B", "C")],
    )

    assert len(journeys) == 1
    legs = journeys[0]["legs"]
    assert [leg["camera"] for leg in legs] == ["A", "B", "C"]
    assert journeys[0]["travel_sec"] > 0


def test_chain_breaks_at_missing_link():
    """中间机位没有匹配时，链路应停在能连上的部分，而不是硬连。"""
    camera_a = [
        _row("a1", "08:00:00", "08:00:05", "[1.0, 0.0]", "A"),
        _row("a2", "08:10:00", "08:10:05", "[0.0, 1.0]", "A"),
    ]
    camera_b = [_row("b1", "08:02:00", "08:02:05", "[1.0, 0.0]", "B")]

    journeys = build_chain_journeys({"A": camera_a, "B": camera_b}, [("A", "B")])

    assert len(journeys) == 1
    assert journeys[0]["legs"][0]["passage_id"] == "a1"
