"""Build an offline topology explorer from the current repository files."""

import argparse
import base64
import csv
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "config/site"
FILES = (
    "cameras.yaml", "topology.yaml", "topology_edges.csv",
    "topology_overrides.csv", "camera_scope.csv", "camera_aliases.csv",
)


def build_page() -> str:
    sources = {name: (SITE / name).read_text(encoding="utf-8-sig") for name in FILES}
    scope = {row["camera_id"]: row for row in csv.DictReader(sources["camera_scope.csv"].splitlines())}
    aliases = list(csv.DictReader(sources["camera_aliases.csv"].splitlines()))
    overrides = {
        (row["from_camera_id"], row["to_camera_id"])
        for row in csv.DictReader(sources["topology_overrides.csv"].splitlines())
    }
    cameras = yaml.safe_load(sources["cameras.yaml"])["cameras"]
    edges = yaml.safe_load(sources["topology.yaml"])["edges"]
    layout = json.loads((ROOT / "web/camera_placer/camera_layout_merged.json").read_text(encoding="utf-8"))
    image = (ROOT / "web/map_editor/assets/zone-a-base-map.jpg").read_bytes()
    payload = {
        "cameras": cameras, "edges": edges, "scope": scope, "aliases": aliases,
        "overrides": sorted(overrides), "map": layout["map"], "roads": layout.get("roads", []),
        "sources": sources,
        "image": "data:image/jpeg;base64," + base64.b64encode(image).decode("ascii"),
    }
    template = (ROOT / "web/topology_viewer/template.html").read_text(encoding="utf-8")
    return template.replace("__TOPOLOGY_DATA__", json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/validation/topology_viewer/index.html")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_page(), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
