from pathlib import Path
from typing import Dict, List, Optional

import yaml


class SiteTopology:
    """Load and validate site-level semantic topology config.

    This layer binds camera outputs to roads and zones so later phases can
    reason about route validity, cross-camera travel windows, and business risk.
    """

    def __init__(self, config_dir: str = "config/site"):
        self.config_dir = Path(config_dir)
        self.site = self._load_yaml("site.yaml")
        self.zones = self._index_by_id(self._load_yaml("zones.yaml").get("zones", []), "zone_id")
        self.roads = self._index_by_id(self._load_yaml("roads.yaml").get("roads", []), "road_id")
        self.cameras = self._index_by_id(self._load_yaml("cameras.yaml").get("cameras", []), "camera_id")
        self.edges = self._load_yaml("topology.yaml").get("edges", [])
        self.routes = self._load_yaml("routes.yaml").get("routes", [])

    def _load_yaml(self, filename: str) -> dict:
        path = self.config_dir / filename
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _index_by_id(items: List[dict], id_field: str) -> Dict[str, dict]:
        indexed = {}
        for item in items:
            item_id = item.get(id_field)
            if item_id:
                indexed[item_id] = item
        return indexed

    def get_camera(self, camera_id: str) -> Optional[dict]:
        return self.cameras.get(camera_id)

    def get_camera_road(self, camera_id: str) -> Optional[dict]:
        camera = self.get_camera(camera_id)
        if not camera:
            return None
        return self.roads.get(camera.get("road_id"))

    def get_camera_zone(self, camera_id: str) -> Optional[dict]:
        camera = self.get_camera(camera_id)
        if not camera:
            return None
        return self.zones.get(camera.get("zone_id"))

    def validate(self) -> List[str]:
        """Return config consistency errors; empty list means valid."""
        errors = []

        for zone_id, zone in self.zones.items():
            parent_id = zone.get("parent_zone_id")
            if parent_id and parent_id not in self.zones:
                errors.append(f"zone {zone_id} references missing parent zone {parent_id}")

        for road_id, road in self.roads.items():
            zone_id = road.get("zone_id")
            if zone_id and zone_id not in self.zones:
                errors.append(f"road {road_id} references missing zone {zone_id}")
            for adjacent_id in road.get("adjacent_road_ids", []) or []:
                if adjacent_id not in self.roads:
                    errors.append(f"road {road_id} references missing adjacent road {adjacent_id}")

        for camera_id, camera in self.cameras.items():
            zone_id = camera.get("zone_id")
            road_id = camera.get("road_id")
            if zone_id and zone_id not in self.zones:
                errors.append(f"camera {camera_id} references missing zone {zone_id}")
            if road_id and road_id not in self.roads:
                errors.append(f"camera {camera_id} references missing road {road_id}")

        for index, edge in enumerate(self.edges):
            from_camera = edge.get("from_camera_id")
            to_camera = edge.get("to_camera_id")
            if from_camera not in self.cameras:
                errors.append(f"edge {index} references missing from_camera_id {from_camera}")
            if to_camera not in self.cameras:
                errors.append(f"edge {index} references missing to_camera_id {to_camera}")
            min_sec = edge.get("min_travel_sec")
            max_sec = edge.get("max_travel_sec")
            if min_sec is not None and max_sec is not None and min_sec > max_sec:
                errors.append(f"edge {index} has min_travel_sec greater than max_travel_sec")

        return errors
