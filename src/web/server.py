"""只读控制台后端：把已有的 run 产物组织成 JSON API，供 `web/console` 使用。

设计原则：
- **只读产物，不碰算法**。pipeline 一行不改，这里只扫描目录、读 JSON/CSV。
- **零第三方依赖**（标准库 `http.server`），不引入 FastAPI，避免改动运行环境。
- **按 run 归档**：每次跑一个目录，`run.json` 记录模型版本；模型改进后产生新 run，
  旧 run 仍可回放对比——这是「先搭框架、模型后改」能成立的前提。

启动：
    python scripts/serve_console.py --port 8765
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = ROOT / "web" / "console"

SKIP_DIR_NAMES = {"h264", "probe", "calib_frames", "x", "__pycache__", "media", "assets"}
MAX_DEPTH = 2

_cache: Dict[str, dict] = {}
_jobs: Dict[str, dict] = {}


# ---------------------------------------------------------------------------
# 发现与摘要
# ---------------------------------------------------------------------------


def _candidate_run_dirs(root: Path) -> List[Path]:
    """找出含 risk/ 或 alerts/ 或 vectors/ 的目录，作为一次 run。"""
    found: List[Path] = []
    root = Path(root)
    if not root.exists():
        return found
    for dirpath, dirnames, _ in os.walk(root):
        current = Path(dirpath)
        depth = len(current.relative_to(root).parts)
        if depth > MAX_DEPTH:
            dirnames[:] = []
            continue
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIR_NAMES]
        if any((current / name).is_dir() for name in ("risk", "alerts", "vectors")):
            found.append(current)
    return found


def run_id_for(path: Path, root: Path) -> str:
    rel = path.relative_to(root).as_posix()
    return rel.replace("/", "__") or "root"


def _first(directory: Path, pattern: str) -> Optional[Path]:
    if not directory.is_dir():
        return None
    matches = sorted(directory.glob(pattern))
    return matches[0] if matches else None


def _read_csv_rows(path: Optional[Path]) -> List[dict]:
    if not path or not path.is_file():
        return []
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Optional[Path]) -> dict:
    if not path or not path.is_file():
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}


def _flag(value) -> bool:
    return str(value).strip() in {"1", "True", "true", "yes"}


def camera_calibrated(camera: str) -> bool:
    """摄像头配置里是否有未注释的 homography。"""
    config = ROOT / "config" / "camera" / f"{camera}.yaml"
    if not config.is_file():
        return False
    for line in config.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if stripped.startswith("homography:"):
            return True
    return False


def _model_version(run_dir: Path, manifest: dict) -> dict:
    if manifest.get("model_version"):
        return {
            "model_version": manifest["model_version"],
            "models": manifest.get("models", []),
            "declared": True,
        }
    references = run_dir / "references"
    iforest = sorted(references.glob("iforest_*.json")) if references.is_dir() else []
    traj = sorted(references.glob("traj_*.pt")) if references.is_dir() else []
    models = [{"name": "iforest", "path": path.as_posix()} for path in iforest]
    models += [{"name": "trajectory", "path": path.as_posix()} for path in traj]
    if iforest and traj:
        version = "inferred:iforest+traj"
    elif iforest or traj:
        version = "inferred:partial"
    else:
        version = "inferred:rules-only"
    return {"model_version": version, "models": models, "declared": False}


def summarize_run(run_dir: Path, root: Path) -> dict:
    """轻量摘要（列表页用），按文件 mtime 缓存。"""
    risk_path = _first(run_dir / "risk", "risk_*.json")
    alerts_path = _first(run_dir / "alerts", "alerts*.csv")
    vectors_path = _first(run_dir / "alerts", "vectors*.csv")
    graph_path = _first(run_dir / "graphs", "graph*.json")
    manifest_path = run_dir / "run.json"

    stamps = tuple(
        int(path.stat().st_mtime) if path and path.is_file() else 0
        for path in (risk_path, alerts_path, vectors_path, graph_path, manifest_path)
    )
    key = f"{run_dir}::{stamps}"
    if key in _cache:
        return _cache[key]

    risk = _read_json(risk_path)
    alerts = _read_csv_rows(alerts_path)
    vectors = _read_csv_rows(vectors_path)
    manifest = _read_json(manifest_path)
    version = _model_version(run_dir, manifest)

    vehicles = risk.get("vehicle_count") or len(vectors) or len(alerts)
    alert_count = sum(1 for row in alerts if _flag(row.get("is_anomaly")))
    camera = manifest.get("camera") or (risk.get("vehicles") or [{}])[0].get("camera_id", "")

    summary = {
        "id": run_id_for(run_dir, root),
        "name": run_dir.relative_to(root).as_posix() or ".",
        "camera": camera,
        "vehicles": vehicles,
        "alerts": alert_count,
        "alert_rate": round(100.0 * alert_count / vehicles, 2) if vehicles else None,
        "medium": risk.get("medium_risk"),
        "high": risk.get("high_risk"),
        "evidence_total": sum(len(row.get("evidence", []) or []) for row in risk.get("vehicles", [])),
        "model_version": version["model_version"],
        "model_declared": version["declared"],
        "calibrated": camera_calibrated(camera) if camera else False,
        "updated_at": datetime.fromtimestamp(max(stamps)).isoformat(timespec="seconds") if max(stamps) else None,
    }
    _cache[key] = summary
    return summary


def list_runs(root: Path) -> List[dict]:
    runs = [summarize_run(path, root) for path in _candidate_run_dirs(root)]
    runs.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
    return runs


def find_run_dir(root: Path, run_id: str) -> Optional[Path]:
    for path in _candidate_run_dirs(root):
        if run_id_for(path, root) == run_id:
            return path
    return None


# ---------------------------------------------------------------------------
# 详情 / 清单 / 单车
# ---------------------------------------------------------------------------


def run_detail(root: Path, run_id: str) -> Optional[dict]:
    run_dir = find_run_dir(root, run_id)
    if run_dir is None:
        return None
    summary = summarize_run(run_dir, root)
    risk = _read_json(_first(run_dir / "risk", "risk_*.json"))
    graph = _read_json(_first(run_dir / "graphs", "graph*.json"))
    manifest = _read_json(run_dir / "run.json")

    detail = dict(summary)
    detail["parameters"] = manifest.get("params", {})
    detail["graph"] = {
        "nodes": graph.get("node_count"),
        "edges": graph.get("edge_count"),
        "communities": len(graph.get("communities", []) or []),
        "modularity": graph.get("modularity"),
        "method": graph.get("graph_anomaly_method"),
    }
    detail["checks"] = {
        "calibrated": summary["calibrated"],
        "in_sample": bool(manifest.get("params", {}).get("retrain")) and bool(manifest.get("params", {}).get("trained", False)),
        "model_declared": summary["model_declared"],
    }
    return detail


def review_rows(root: Path, run_id: str, budget: int = 20) -> Optional[List[dict]]:
    from src.report.review_list import build_rows, rank_vehicles

    run_dir = find_run_dir(root, run_id)
    if run_dir is None:
        return None
    risk = _read_json(_first(run_dir / "risk", "risk_*.json"))
    vehicles = risk.get("vehicles", []) if isinstance(risk, dict) else []
    return build_rows(rank_vehicles(vehicles, budget=budget))


def vehicle_record(root: Path, run_id: str, passage_id: str) -> Optional[dict]:
    run_dir = find_run_dir(root, run_id)
    if run_dir is None:
        return None
    risk = _read_json(_first(run_dir / "risk", "risk_*.json"))
    for record in risk.get("vehicles", []) or []:
        if str(record.get("passage_id", "")) == passage_id:
            return record
    return None


def artifact_path(root: Path, run_id: str, relative: str) -> Optional[Path]:
    """把 run 内的相对路径解析成真实文件，拒绝目录穿越。"""
    run_dir = find_run_dir(root, run_id)
    if run_dir is None:
        return None
    candidate = (run_dir / relative).resolve()
    try:
        candidate.relative_to(run_dir.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def models_list(root: Path) -> dict:
    local = []
    models_dir = ROOT / "models"
    if models_dir.is_dir():
        for path in sorted(models_dir.iterdir()):
            if path.is_file():
                local.append({"name": path.name, "path": path.as_posix(), "size_mb": round(path.stat().st_size / 1e6, 1)})
    versions = {}
    for run in list_runs(root):
        versions.setdefault(run["model_version"], []).append(run["id"])
    return {"files": local, "versions": versions}


# ---------------------------------------------------------------------------
# 启动 run（异步子进程）
# ---------------------------------------------------------------------------


def start_run(root: Path, payload: dict, python: str = sys.executable) -> dict:
    run_id = payload.get("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:6]
    output = Path(payload.get("output") or (Path(root) / "runs" / run_id))
    command = [
        python,
        "scripts/run_full_pipeline.py",
        "--input", payload["input"],
        "--camera", payload.get("camera", "admin_in"),
        "--output", str(output),
    ]
    if payload.get("retrain"):
        command.append("--retrain")
    if payload.get("max_frames"):
        command += ["--max-frames", str(int(payload["max_frames"]))]

    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "run.log"
    handle = open(log_path, "w", encoding="utf-8")
    process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {
        "id": job_id,
        "run_id": run_id,
        "output": str(output),
        "log": str(log_path),
        "command": command,
        "proc": process,
        "started_at": time.time(),
        "status": "running",
    }
    return {"job_id": job_id, "run_id": run_id, "output": str(output), "command": command}


_PROGRESS = re.compile(r"Frame\s+(\d+)/(\d+)")


def job_status(job_id: str) -> Optional[dict]:
    job = _jobs.get(job_id)
    if job is None:
        return None
    process = job["proc"]
    code = process.poll()
    if code is not None and job["status"] == "running":
        job["status"] = "done" if code == 0 else "failed"
        job["returncode"] = code
    progress = None
    try:
        with open(job["log"], encoding="utf-8", errors="replace") as handle:
            for line in handle:
                match = _PROGRESS.search(line)
                if match:
                    current, total = int(match.group(1)), int(match.group(2))
                    progress = {"current": current, "total": total,
                                "percent": round(100.0 * current / total, 1) if total else None}
    except OSError:
        pass
    return {
        "id": job_id,
        "run_id": job["run_id"],
        "status": job["status"],
        "returncode": job.get("returncode"),
        "progress": progress,
        "log": job["log"],
        "elapsed_sec": round(time.time() - job["started_at"], 1),
    }


# ---------------------------------------------------------------------------
# HTTP 层
# ---------------------------------------------------------------------------


class ConsoleHandler(BaseHTTPRequestHandler):
    root: Path = ROOT / "data" / "outputs"
    allow_runs: bool = False
    server_version = "ZoneAConsole/0.1"

    def log_message(self, fmt, *args):  # 静默，避免刷屏
        return

    # -- helpers ----------------------------------------------------------
    def _json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path):
        suffix = path.suffix.lower()
        content_type = {
            ".html": "text/html; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".csv": "text/csv; charset=utf-8",
            ".md": "text/markdown; charset=utf-8",
            ".png": "image/png",
            ".jpg": "image/jpeg",
        }.get(suffix, "application/octet-stream")
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # -- routes -----------------------------------------------------------
    def do_GET(self):
        parsed = urlparse(self.path)
        parts = [part for part in parsed.path.split("/") if part]
        query = parse_qs(parsed.query)

        if not parts:
            return self._file(WEB_DIR / "index.html")
        if parts[0] in {"app.js", "style.css"}:
            return self._file(WEB_DIR / parts[0])

        if parts[:2] == ["api", "health"]:
            return self._json({"ok": True, "root": str(self.root), "allow_runs": self.allow_runs})

        if parts[:2] == ["api", "runs"]:
            if len(parts) == 2:
                return self._json(list_runs(self.root))
            run_id = parts[2]
            if len(parts) == 3:
                detail = run_detail(self.root, run_id)
                return self._json(detail) if detail else self._json({"error": "run not found"}, 404)
            if len(parts) == 4 and parts[3] == "review":
                budget = int(query.get("budget", ["20"])[0])
                rows = review_rows(self.root, run_id, budget=budget)
                return self._json(rows) if rows is not None else self._json({"error": "run not found"}, 404)
            if len(parts) == 5 and parts[3] == "vehicles":
                record = vehicle_record(self.root, run_id, parts[4])
                return self._json(record) if record else self._json({"error": "vehicle not found"}, 404)
            if len(parts) == 4 and parts[3] == "artifact":
                relative = query.get("path", [""])[0]
                path = artifact_path(self.root, run_id, relative)
                return self._file(path) if path else self._json({"error": "artifact not found"}, 404)

        if parts[:2] == ["api", "models"]:
            return self._json(models_list(self.root))

        if parts[:2] == ["api", "jobs"] and len(parts) == 3:
            status = job_status(parts[2])
            return self._json(status) if status else self._json({"error": "job not found"}, 404)

        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/runs":
            return self._json({"error": "not found"}, 404)
        if not self.allow_runs:
            return self._json({"error": "启用了只读模式；启动服务时加 --allow-runs 才能创建 run"}, 403)
        length = int(self.headers.get("Content-Length", "0") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._json({"error": "invalid json"}, 400)
        if not payload.get("input"):
            return self._json({"error": "input is required"}, 400)
        try:
            return self._json(start_run(self.root, payload))
        except Exception as exc:  # 启动失败要回传原因，而不是静默
            return self._json({"error": str(exc)}, 500)


def serve(host: str = "127.0.0.1", port: int = 8765, root: Optional[Path] = None, allow_runs: bool = False):
    ConsoleHandler.root = Path(root) if root else (ROOT / "data" / "outputs")
    ConsoleHandler.allow_runs = allow_runs
    server = ThreadingHTTPServer((host, port), ConsoleHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
