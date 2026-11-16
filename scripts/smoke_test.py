#!/usr/bin/env python
"""End-to-end smoke test for the modular script chain.

Runs the five separate CLIs in order and verifies each expected output exists:

    1. run_pipeline.py            -> raw behavior vectors
    2. run_anomaly_detection.py   -> alerts + annotated vectors
    3. build_graph.py             -> vehicle graph
    4. build_risk_report.py       -> risk report (JSON + Markdown)
    5. eval_system.py             -> evaluation summary

Usage:
    python scripts/smoke_test.py --video data/outputs/h264/admin_in/clip.h264 --max-frames 1500
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

WORK_ROOT = os.path.join("data", "outputs", "smoke")


def run_stage(name, command, expected_path):
    started = time.perf_counter()
    print(f"\n[{name}] {' '.join(command)}", flush=True)
    completed = subprocess.run(command, capture_output=True, text=True)
    elapsed = time.perf_counter() - started

    if completed.returncode != 0:
        print(completed.stdout[-2000:])
        print(completed.stderr[-2000:])
        return {"stage": name, "ok": False, "elapsed_sec": round(elapsed, 2),
                "reason": f"exit code {completed.returncode}"}

    matches = expected_path if isinstance(expected_path, list) else [expected_path]
    found = []
    for pattern in matches:
        hits = sorted(glob.glob(pattern, recursive=True))
        if not hits:
            return {"stage": name, "ok": False, "elapsed_sec": round(elapsed, 2),
                    "reason": f"missing output: {pattern}"}
        found.extend(hits)

    return {"stage": name, "ok": True, "elapsed_sec": round(elapsed, 2), "outputs": found}


def main():
    parser = argparse.ArgumentParser(description="End-to-end smoke test")
    parser.add_argument("--video", required=True, help="Input .h264 / .mp4 file")
    parser.add_argument("--camera", default="admin_in")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--max-frames", type=int, default=1500)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--keep", action="store_true", help="Keep intermediate outputs")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Video not found: {args.video}")
        sys.exit(2)

    if os.path.isdir(WORK_ROOT) and not args.keep:
        shutil.rmtree(WORK_ROOT, ignore_errors=True)
    os.makedirs(WORK_ROOT, exist_ok=True)

    python = sys.executable
    # run_pipeline.py 的 --output 是输出根目录，内部会再拼 vectors/
    vectors_root = WORK_ROOT
    vectors_camera_dir = os.path.join(WORK_ROOT, "vectors", args.camera)
    alerts_dir = os.path.join(WORK_ROOT, "alerts")
    graph_path = os.path.join(WORK_ROOT, "graphs", "graph.json")
    risk_json = os.path.join(WORK_ROOT, "risk", "risk.json")
    risk_md = os.path.join(WORK_ROOT, "risk", "risk.md")
    eval_path = os.path.join(WORK_ROOT, "eval", "eval.json")

    results = []

    results.append(run_stage(
        "1/5 run_pipeline",
        [python, "scripts/run_pipeline.py", "--video", args.video, "--camera", args.camera,
         "--config", args.config, "--output", vectors_root, "--max-frames", str(args.max_frames),
         "--device", str(args.device)],
        os.path.join(vectors_camera_dir, "**", "vec_*.csv"),
    ))
    if not results[-1]["ok"]:
        return finish(results, args.keep)

    results.append(run_stage(
        "2/5 run_anomaly_detection",
        [python, "scripts/run_anomaly_detection.py",
         "--input", vectors_camera_dir,
         "--output", os.path.join(alerts_dir, "alerts.csv"),
         "--annotated-vectors", os.path.join(alerts_dir, "vectors.csv")],
        [os.path.join(alerts_dir, "alerts.csv"), os.path.join(alerts_dir, "vectors.csv")],
    ))
    if not results[-1]["ok"]:
        return finish(results, args.keep)

    results.append(run_stage(
        "3/5 build_graph",
        [python, "scripts/build_graph.py",
         "--input", os.path.join(alerts_dir, "vectors.csv"),
         "--output", graph_path, "--top-k", "3"],
        graph_path,
    ))
    if not results[-1]["ok"]:
        return finish(results, args.keep)

    results.append(run_stage(
        "4/5 build_risk_report",
        [python, "scripts/build_risk_report.py",
         "--input", os.path.join(alerts_dir, "vectors.csv"),
         "--graph", graph_path,
         "--output", risk_json, "--markdown", risk_md],
        [risk_json, risk_md],
    ))
    if not results[-1]["ok"]:
        return finish(results, args.keep)

    results.append(run_stage(
        "5/5 eval_system",
        [python, "scripts/eval_system.py",
         "--vectors", os.path.join(alerts_dir, "vectors.csv"),
         "--alerts", os.path.join(alerts_dir, "alerts.csv"),
         "--risk", risk_json, "--graph", graph_path, "--output", eval_path],
        eval_path,
    ))

    return finish(results, args.keep)


def finish(results, keep):
    print("\n" + "=" * 62)
    for item in results:
        status = "OK  " if item["ok"] else "FAIL"
        extra = "" if item["ok"] else f"  ({item['reason']})"
        print(f"{status}  {item['stage']:<28} {item['elapsed_sec']:>7.1f}s{extra}")
    print("=" * 62)

    if os.path.exists(os.path.join(WORK_ROOT, "eval", "eval.json")):
        with open(os.path.join(WORK_ROOT, "eval", "eval.json"), encoding="utf-8") as f:
            summary = json.load(f)
        print(json.dumps({
            "vehicles": summary.get("vehicle_count"),
            "alerts": summary.get("alert_count"),
            "evidence_trace": summary.get("evidence_trace_completeness"),
            "risk_max": summary.get("risk_max"),
        }, ensure_ascii=False))

    print(f"(intermediate outputs at {WORK_ROOT}; delete the directory to clean up)")

    passed = all(item["ok"] for item in results)
    print("\nSMOKE TEST PASSED" if passed else "\nSMOKE TEST FAILED")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
