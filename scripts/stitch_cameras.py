#!/usr/bin/env python
"""Match vehicles across two cameras and assign global vehicle IDs.

Reads two vector CSV sets (one per camera), uses the camera topology time
window plus ReID cosine similarity to match passages, then merges matches into
global vehicle ids.

Usage:
    python scripts/stitch_cameras.py \
        --from-vectors data/outputs/vectors/cam_up \
        --to-vectors data/outputs/vectors/cam_down \
        --from-camera cam_up \
        --to-camera cam_down \
        --output data/outputs/stitch/cam_up_to_down.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.site.topology import SiteTopology
from src.trajectory.stitching import CrossCameraStitcher
from src.utils.logger import get_logger

logger = get_logger(__name__)


from src.utils.vectors_io import load_vector_rows as load_rows


def main():
    parser = argparse.ArgumentParser(description="Cross-camera vehicle stitching")
    parser.add_argument("--from-vectors", required=True)
    parser.add_argument("--to-vectors", required=True)
    parser.add_argument("--from-camera", required=True)
    parser.add_argument("--to-camera", required=True)
    parser.add_argument("--topology", default="config/site", help="Site topology config dir")
    parser.add_argument("--threshold", type=float, default=0.7, help="ReID cosine similarity threshold")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    source = load_rows(args.from_vectors)
    target = load_rows(args.to_vectors)
    if not source or not target:
        logger.error("No vectors: source=%s target=%s", len(source), len(target))
        sys.exit(1)

    topology = SiteTopology(args.topology) if os.path.isdir(args.topology) else None
    stitcher = CrossCameraStitcher(topology=topology, similarity_threshold=args.threshold)
    matches = stitcher.match(source, target, args.from_camera, args.to_camera)

    # 节点键必须与匹配对里的 from_key/to_key 同源（优先 passage_id），
    # 否则 assign_global_ids 找不到节点、匹配对不会被合并。
    nodes = [stitcher.passage_key(args.from_camera, row) for row in source]
    nodes += [stitcher.passage_key(args.to_camera, row) for row in target]
    global_ids = stitcher.assign_global_ids(nodes, matches)

    payload = {
        "from_camera": args.from_camera,
        "to_camera": args.to_camera,
        "source_count": len(source),
        "target_count": len(target),
        "match_count": len(matches),
        "matches": matches,
        "global_ids": {str(node): gid for node, gid in global_ids.items()},
    }

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info("Matched %s pairs between %s and %s", len(matches), args.from_camera, args.to_camera)
    print(f"Matches: {len(matches)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
