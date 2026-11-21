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
    """目录里最新的匹配文件。

    **必须按 mtime 而不是文件名字母序。** 同名产物常有多个版本
    （``risk_admin_in.json`` 与重跑后的 ``risk_admin_in_v2.json``），
    字母序下 ``_v2`` 排在最后，``matches[0]`` 会永远返回旧版本——
    实测控制台因此长期显示「15 中风险 / 最高 61.3 分」的旧口径数字，
    而磁盘上更新的 ``_v2`` 已是「38 中风险 / 最高 80.2」。

    平手时按名称兜底，保证结果可复现。
    """
    if not directory.is_dir():
        return None
    matches = sorted(directory.glob(pattern))
    if not matches:
        return None
    return max(matches, key=lambda path: (path.stat().st_mtime, path.name))


def _risk_path(run_dir: Path) -> Optional[Path]:
    """风险报告文件：兼容 ``risk_<camera>.json`` 与 ``risk.json`` 两种命名。"""
    return _first(run_dir / "risk", "risk*.json")


def _graph_path(run_dir: Path) -> Optional[Path]:
    return _first(run_dir / "graphs", "graph*.json")


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


def _to_float(value, default=None):
    """宽松转 float；无法转换时返回 ``default``（默认 None，用于区分"缺字段"与"值为 0"）。"""
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_analysis_target(record: dict) -> bool:
    """该记录是否属于「监管对象」（``analysis_target != 0``）。

    注意不能写成 ``str(value or "")``——``0`` 是 falsy，会被 ``or`` 吞成空串，
    于是 ``analysis_target=0`` 的货车/两轮车被误判为监管对象。这正是
    ``build_risk_report.py`` 里踩过的坑，这里用同样显式的写法。

    字段缺失时视为监管对象：兼容旧产物（旧版没有这一列，所有车都参与了评分）。
    """
    value = record.get("analysis_target", None)
    if value is None or value == "":
        return True
    return str(value).strip() not in {"0", "False", "false", "0.0"}


def camera_display_name(camera: str) -> str:
    """机位中文名（来自 config/camera/<id>.yaml 的 camera_name）。"""
    if not camera:
        return ""
    config = ROOT / "config" / "camera" / f"{camera}.yaml"
    if not config.is_file():
        return camera
    for line in config.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("camera_name:"):
            return stripped.split(":", 1)[1].strip().strip('"').strip("'")
    return camera


def build_headline(record: dict) -> str:
    """一句话说明"为什么排在这里"。

    注意 G 的阈值语义：G 现在是**相对基线分布的锚点标定值**（P50→0、P99→1），
    不是图内 min-max。旧口径下每张图必然产出一个 1.0，于是"本批最异常者"
    恒得满分、必然触发"关系明显异常"的措辞，哪怕这批车全都正常。
    因此这里还要求 ``graph_available`` 为真，避免把"无图谱数据"说成"关系正常/异常"。
    """
    components = record.get("risk_components") or {}
    graph = float(components.get("G", 0) or 0)
    anomaly = float(components.get("A", 0) or 0)
    residual = float(record.get("traj_residual", 0) or 0)
    graph_available = str(record.get("graph_available", 1)).strip() not in {"0", "False", "false", ""}

    if graph_available and graph >= 0.8:
        size = record.get("community_size")
        try:
            size_value = int(size)
        except (TypeError, ValueError):
            size_value = 0
        if size_value > 1:
            return (
                f"主要因为它与该时段另外 {size_value - 1} 辆车形成紧密关联"
                f"（同组共 {size_value} 辆），关联异常指数 {graph:.2f}。"
            )
        return f"主要因为它与同批车辆的关系明显异常（关联指数 {graph:.2f}）。"
    if residual >= 20:
        return f"主要因为行驶轨迹偏离预测，偏差约 {residual:.0f} 像素。"
    if anomaly >= 0.6:
        return f"主要因为速度、时长等行为组合偏离常态（异常指数 {anomaly:.2f}）。"
    rule = str(record.get("alert_reason", "") or "").strip()
    if rule:
        return f"主要因为触发监管规则：{rule.split(';')[0]}。"
    return "综合多项弱信号，建议例行核查。"


def build_reasons(record: dict) -> List[dict]:
    """把结构化证据翻译成给人看的原因（保留原文用于反查）。"""
    from src.report.review_list import reasons_with_labels

    components = record.get("risk_components") or {}
    graph = float(components.get("G", 0) or 0)
    anomaly = float(components.get("A", 0) or 0)
    residual = float(record.get("traj_residual", 0) or 0)
    rule = str(record.get("alert_reason", "") or "").strip()

    plain = {
        "graph_anomaly": f"与同批其他车辆的关系明显异常（关联指数 {graph:.2f}）",
        "transformer_residual": f"行驶轨迹与预测偏差约 {residual:.0f} 像素",
        "isolation_forest": f"速度、时长等行为组合偏离常态（异常指数 {anomaly:.2f}）",
        "rule": f"触发监管规则：{rule.split(';')[0]}" if rule else "",
    }

    output = []
    for item in reasons_with_labels(record):
        text = plain.get(item.get("signal", "")) or item.get("text", "")
        output.append({**item, "plain": text})
    return output


def score_parts(record: dict) -> dict:
    """把综合分拆成 A/G/T 三个来源，供界面显示"分从哪来"。

    权重取自记录的 ``risk_components``（融合层已按分量可用性重归一化并写回），
    因此这里显示的是**实际参与计算**的权重，而不是名义 α/β/γ。
    若某分量不可用（如单窗口运行下的 T），会在 ``unavailable`` 里列出，
    前端据此显示"无数据"而不是"0 分"。

    **不要用 ``components.get("gamma", 0.25) or 0.25``**：重归一化后 γ 恰好是
    ``0.0``，而 ``0.0`` 是 falsy，会被 ``or`` 回填成名义值 0.25——
    界面上就变成"频次占 25% 权重"，与"该分量未参与"完全矛盾。
    这与 ``build_risk_report.py`` 里 ``analysis_target=0`` 的坑是同一种错误。
    """
    components = record.get("risk_components") or {}

    def _weight(name: str, nominal: float) -> float:
        value = components.get(name, nominal)
        try:
            return float(value) if value is not None and value != "" else float(nominal)
        except (TypeError, ValueError):
            return float(nominal)

    weights = {"A": _weight("alpha", 0.4), "G": _weight("beta", 0.35), "T": _weight("gamma", 0.25)}
    values = {"A": float(components.get("A", 0) or 0),
              "G": float(components.get("G", 0) or 0),
              "T": float(components.get("T", 0) or 0)}
    points = {key: round(100 * weights[key] * values[key], 1) for key in values}

    available_flags = record.get("risk_component_available") or {}
    unavailable = [
        key for key in ("A", "G", "T")
        if str(available_flags.get(key, 1)).strip() in {"0", "False", "false"}
    ]

    # 总分优先用融合层写回的 risk_score_100：它才是排序和分档的依据。
    # 自己把三项加回去会因四舍五入产生 0.1 的偏差（实测 80.3 vs 80.2），
    # 列表和详情显示的数字对不上，一线会怀疑哪一个是错的。
    reported = _to_float(record.get("risk_score_100"), None)
    total = round(reported, 1) if reported is not None else round(sum(points.values()), 1)
    shares = {key: (round(100 * points[key] / total) if total else 0) for key in points}

    # 证据封顶：只剩一路分量可用时，融合层把分数压到 75 分（低于 80 分档）。
    # 界面必须说清"这是被压过的分"，否则一线看到 R=75 会当成原始分，
    # 进而误判这辆车"离高风险只差 5 分"。
    ceiling = _to_float(record.get("risk_score_ceiling"), 100.0)
    capped = str(record.get("risk_score_capped", 0)).strip() in {"1", "True", "true"}
    raw = _to_float(record.get("risk_score_raw_100"), None)

    return {
        "points": points,
        "shares": shares,
        "total": total,
        "values": values,
        "weights": {key: round(weights[key], 4) for key in weights},
        "unavailable": unavailable,
        "available_count": 3 - len(unavailable),
        "ceiling": ceiling,
        "capped": capped,
        "raw_total": round(raw, 1) if raw is not None else None,
    }


def load_plate_overrides(run_dir: Path) -> dict:
    """人工确认的车牌（crops/plates.csv：passage_id,plate）。

    机器 OCR 在这条车道上不可靠（车牌约 90px），因此留一个「人工确认一次、
    系统记住」的入口：填了这张表，界面就显示真实车牌，并可继续用作身份键。
    """
    path = run_dir / "crops" / "plates.csv"
    if not path.is_file():
        return {}
    mapping = {}
    with open(path, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = str(row.get("passage_id", "") or "").strip()
            plate = str(row.get("plate", "") or "").strip()
            if key and plate:
                mapping[key] = plate
    return mapping


def save_plate_override(run_dir: Path, passage_id: str, plate: str) -> bool:
    """把人工确认的车牌写入 crops/plates.csv（就地更新已有行）。"""
    plate = str(plate or "").strip()
    if not passage_id or not plate:
        return False
    path = run_dir / "crops" / "plates.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    if path.is_file():
        with open(path, encoding="utf-8-sig", newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if row.get("passage_id")]
    replaced = False
    for row in rows:
        if row["passage_id"] == passage_id:
            row["plate"] = plate
            replaced = True
            break
    if not replaced:
        rows.append({"passage_id": passage_id, "plate": plate, "note": "人工确认"})

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["passage_id", "plate", "note"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in writer.fieldnames})
    return True


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
    risk_path = _risk_path(run_dir)
    alerts_path = _first(run_dir / "alerts", "alerts*.csv")
    vectors_path = _first(run_dir / "alerts", "vectors*.csv")
    graph_path = _graph_path(run_dir)
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
    camera_name = camera_display_name(camera)

    first_enter = ""
    for row in risk.get("vehicles", []) or []:
        if row.get("enter_time"):
            first_enter = str(row["enter_time"])
            break
    date_part = first_enter[:10] if first_enter else ""

    # 非监管对象（货车/两轮车）单列在 non_target_vehicles，不计入 vehicle_count。
    # 这里把两类车辆都报出来：只报监管对象会让人以为"这批复核只处理了 N 辆"，
    # 而现场看到的实际过车数更多。
    scored = risk.get("vehicles", []) or []
    non_target = risk.get("non_target_vehicles", []) or []
    non_target_count = risk.get("non_target_count")
    if non_target_count is None:
        # 旧产物没有分离：从 vehicles 里按 analysis_target 现算
        non_target_count = sum(1 for row in scored if not _is_analysis_target(row))
    total_vehicles = vehicles + (int(non_target_count) if non_target_count else 0)

    summary = {
        "id": run_id_for(run_dir, root),
        "name": run_dir.relative_to(root).as_posix() or ".",
        "camera": camera,
        "camera_name": camera_name,
        "label": " · ".join(part for part in (camera_name, date_part, f"{vehicles} 辆") if part),
        "vehicles": vehicles,
        "non_target_count": int(non_target_count or 0),
        "non_target_listed": len(non_target),
        "total_vehicles": total_vehicles,
        "alerts": alert_count,
        "alert_rate": round(100.0 * alert_count / vehicles, 2) if vehicles else None,
        "medium": risk.get("medium_risk"),
        "high": risk.get("high_risk"),
        "evidence_total": sum(len(row.get("evidence", []) or []) for row in scored),
        "risk_max": max(
            (float(row.get("risk_score_100", 0) or 0) for row in scored if _is_analysis_target(row)),
            default=None,
        ),
        # 分量可用性与分数上限：告诉界面这次评分用了哪几路证据。
        # 缺这一层，用户会把"只用了 1 路"当成"3 路都很干净"。
        "components_available": risk.get("components_available"),
        "score_ceiling": risk.get("score_ceiling"),
        "score_capped_count": risk.get("score_capped_count"),
        "risk_version": risk.get("version"),
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
    risk = _read_json(_risk_path(run_dir))
    graph = _read_json(_graph_path(run_dir))
    manifest = _read_json(run_dir / "run.json")

    detail = dict(summary)
    detail["parameters"] = manifest.get("params", {})
    # G 能不能跨批次比较：训练图 == 评分图时只能当本批内部排序看。
    # 旧图谱没有 graph_anomaly_self_referential 字段，按 method 回退推断：
    # 只有 gae_baseline（加载了独立基线模型）才谈得上跨批次可比。
    method = graph.get("graph_anomaly_method")
    comparable = graph.get("graph_anomaly_comparable")
    self_referential = graph.get("graph_anomaly_self_referential")
    if self_referential is None and method:
        self_referential = method != "gae_baseline"
    if comparable is None and self_referential is not None:
        comparable = not self_referential

    detail["graph"] = {
        "nodes": graph.get("node_count"),
        "edges": graph.get("edge_count"),
        "communities": len(graph.get("communities", []) or []),
        "modularity": graph.get("modularity"),
        "method": method,
        "comparable": comparable,
        "self_referential": self_referential,
        "calibration": graph.get("graph_anomaly_calibration"),
    }
    detail["checks"] = {
        "calibrated": summary["calibrated"],
        "in_sample": bool(manifest.get("params", {}).get("retrain")) and bool(manifest.get("params", {}).get("trained", False)),
        "model_declared": summary["model_declared"],
    }
    return detail


def review_rows(root: Path, run_id: str, budget: int = 20, sort: str = "risk") -> Optional[List[dict]]:
    from src.report.review_list import build_rows, rank_vehicles

    run_dir = find_run_dir(root, run_id)
    if run_dir is None:
        return None
    risk = _read_json(_risk_path(run_dir))
    vehicles = risk.get("vehicles", []) if isinstance(risk, dict) else []

    # 非监管对象（货车/两轮车等 analysis_target=0）不参与行为基线、图谱与风险评分，
    # 它们的 A/G/T 全为 0、没有 risk_score_100。若混进「待核查车辆」，
    # 界面会显示成「可疑度 0 的正常车」，把"未评估"错述成"已评估且正常"。
    # 新版风险报告已把它们分离到 non_target_vehicles，这里再做一层防御：
    # 读到旧产物（未分离）时同样剔除，避免同一缺陷从另一个入口回来。
    vehicles = [record for record in vehicles if _is_analysis_target(record)]

    # 三种排序口径：综合 R（默认）、个体行为 A、轨迹预测残差
    if sort == "behavior":
        ordered = sorted(vehicles, key=lambda r: -float((r.get("risk_components") or {}).get("A", 0) or 0))
    elif sort == "trajectory":
        ordered = sorted(vehicles, key=lambda r: -float(r.get("traj_residual", 0) or 0))
    else:
        ordered = rank_vehicles(vehicles, budget=0)
    selected = ordered[:budget] if budget > 0 else ordered

    rows = build_rows(selected)

    # 列表里的「主要原因」用一句话结论；同时附上分数构成
    plain = {}
    parts = {}
    for record in selected:
        key = str(record.get("passage_id", ""))
        plain[key] = build_headline(record)
        parts[key] = score_parts(record)
    for row in rows:
        key = str(row.get("passage_id", ""))
        if key in plain:
            row["reason"] = plain[key]
        if key in parts:
            row["parts"] = parts[key]
    return rows


def vehicle_record(root: Path, run_id: str, passage_id: str) -> Optional[dict]:
    from src.report.review_list import display_name, reasons

    run_dir = find_run_dir(root, run_id)
    if run_dir is None:
        return None
    risk = _read_json(_risk_path(run_dir))
    overrides = load_plate_overrides(run_dir)
    for record in risk.get("vehicles", []) or []:
        if str(record.get("passage_id", "")) == passage_id:
            enriched = dict(record)
            camera = str(record.get("camera_id", "") or "")
            plate = str(record.get("plate_number", "") or "").strip()
            confirmed = overrides.get(passage_id, "")

            crop_entry = {}
            index_path = run_dir / "crops" / "index.json"
            if index_path.is_file():
                try:
                    with open(index_path, encoding="utf-8") as handle:
                        crop_entry = json.load(handle).get(passage_id, {}) or {}
                except (json.JSONDecodeError, OSError):
                    crop_entry = {}

            display_name_value = confirmed or plate or display_name(record)
            enriched["display"] = {
                "name": display_name_value,
                "plate": confirmed or plate,
                "plate_confirmed": bool(confirmed),
                "plate_text": crop_entry.get("plate_text", ""),
                "plate_conf": crop_entry.get("plate_conf", 0),
                "has_plate_image": bool(crop_entry.get("plate_file")),
                "vehicle_type": record.get("vehicle_type", ""),
                "vehicle_color": record.get("vehicle_color", ""),
                "camera": camera,
                "camera_name": camera_display_name(camera),
                "headline": build_headline(record),
                "reasons": build_reasons(record),
                "reason_texts": reasons(record),
                "parts": score_parts(record),
            }
            return enriched
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
                sort = query.get("sort", ["risk"])[0]
                rows = review_rows(self.root, run_id, budget=budget, sort=sort)
                return self._json(rows) if rows is not None else self._json({"error": "run not found"}, 404)
            if len(parts) == 5 and parts[3] == "vehicles":
                record = vehicle_record(self.root, run_id, parts[4])
                return self._json(record) if record else self._json({"error": "vehicle not found"}, 404)
            if len(parts) == 6 and parts[3] == "vehicles" and parts[5] in {"thumbnail", "plate"}:
                suffix = "" if parts[5] == "thumbnail" else "_plate"
                path = artifact_path(self.root, run_id, f"crops/{parts[4]}{suffix}.jpg")
                if path is None and parts[5] == "plate":
                    return self._json({"error": "plate image not found"}, 404)
                return self._file(path) if path else self._json({"error": "thumbnail not found"}, 404)
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
        parts = [part for part in parsed.path.split("/") if part]

        # 人工确认车牌：POST /api/runs/{id}/vehicles/{pid}/plate  {"plate": "苏F·DZ8777"}
        if len(parts) == 6 and parts[:2] == ["api", "runs"] and parts[3] == "vehicles" and parts[5] == "plate":
            if not self.allow_runs:
                return self._json({"error": "只读模式；启动时加 --allow-runs 才能保存"}, 403)
            length = int(self.headers.get("Content-Length", "0") or 0)
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return self._json({"error": "invalid json"}, 400)
            run_dir = find_run_dir(self.root, parts[2])
            if run_dir is None:
                return self._json({"error": "run not found"}, 404)
            ok = save_plate_override(run_dir, parts[4], payload.get("plate", ""))
            return self._json({"ok": ok, "plate": str(payload.get("plate", "")).strip()}) if ok \
                else self._json({"error": "plate is required"}, 400)

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
