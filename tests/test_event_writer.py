import csv

from src.output.event_writer import EventWriter


def test_event_writer_builds_passage_event_from_vector():
    vector = {
        "camera_id": "cam01",
        "track_id": 7,
        "global_vehicle_id": "veh_7",
        "enter_time": "2026-06-01T08:00:00",
        "exit_time": "2026-06-01T08:00:10",
        "duration_sec": 10.0,
        "vehicle_type": "car",
        "vehicle_color": "white",
        "plate_number": "A12345",
        "plate_hash": "hash1",
        "plate_confidence": 0.88,
        "geometry_level": "calibrated",
        "speed_reliable": 1,
        "path_reliable": 1,
        "aggregation_reliable": 1,
        "passage_reliable": 1,
        "avg_speed_ms": 2.0,
        "max_speed_ms": 3.0,
        "reid_embedding": [0.1, 0.2],
    }

    event = EventWriter().build_events([vector])[0]

    assert event["event_type"] == "vehicle_passage"
    assert event["event_time"] == "2026-06-01T08:00:10"
    assert event["reid_available"] == 1
    assert event["plate_confidence"] == 0.88
    assert event["event_id"] == "cam01_7_2026-06-01T08_00_10"


def test_event_writer_writes_header_for_empty_events(tmp_path):
    path = tmp_path / "events.csv"

    EventWriter().write([], str(path))

    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    assert rows == [EventWriter.CSV_COLUMNS]
