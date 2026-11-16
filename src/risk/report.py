"""可解释报告生成（研究要点四）。

两种后端：
- ``TemplateReportGenerator``：完全由结构化证据填充，确定性、零幻觉，作为默认；
- ``LocalLLMReportGenerator``：调用本地/内网 OpenAI 兼容接口，把结构化证据交给
  大模型生成自然语言建议。接口不可用时自动回退到模板后端。

“安全、可控”的落实方式：
- 输入只允许结构化证据，不允许自由检索；
- 提示词明确要求“只能基于给定证据”，不得编造；
- 输出保留证据链原文，便于人工核对。
"""

from __future__ import annotations

import json
import re
import urllib.request
from datetime import datetime
from typing import List, Optional

_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")
_VEHICLE_ID_PATTERN = re.compile(r"[A-Za-z0-9_\u4e00-\u9fa5]+:\d+")
# 日期与时间：不应参与“编造数字/车辆编号”的校验。
# 用负向后顾避免把 cam02:99 里的 02:99 误当成时间剥掉。
_TIME_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:"
    r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?)?"
    r"|\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日?"
    r"|\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?"
    r")"
)
# 常见通用数字（时间、路数、天数等），不视为编造
_GENERIC_INTEGERS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 24, 30, 40, 50, 60, 80, 100, 365}

# 证据信号 -> 面向一线的业务措辞（模板报告与大模型提示词共用）
BUSINESS_EVIDENCE_TEXT = {
    "isolation_forest": "车辆行为模式与常态不符",
    "transformer_residual": "轨迹走势不规律、偏离预测路线",
    "graph_anomaly": "与其他车辆通行模式高度相似，疑似结伴",
    "time_frequency": "通行时段或频次异常",
    "rule_engine": "命中预设监管规则",
    "cross_camera": "在其他机位再次出现",
}


def _format_time(value) -> str:
    """把 ISO 时间格式化成 HH:MM:SS（去微秒）。"""
    if not value:
        return ""
    try:
        return datetime.fromisoformat(str(value)).strftime("%H:%M:%S")
    except (TypeError, ValueError):
        return str(value)


def _allowed_numbers(record: dict, evidence: List[dict]) -> List[float]:
    """收集记录与证据中真实存在的数值，用于校验大模型是否编造数字。"""
    values: List[float] = []
    for key in (
        "risk_score",
        "risk_score_100",
        "iforest_score",
        "traj_residual",
        "graph_anomaly_index",
        "time_freq_index",
        "freq_count_24h",
        "night_ratio",
        "track_id",
        "community_id",
        "community_size",
        "duration_sec",
        "stop_count",
        "max_concurrent_vehicles",
    ):
        try:
            values.append(float(record.get(key)))
        except (TypeError, ValueError):
            pass
    components = record.get("risk_components", {}) or {}
    for key in ("A", "G", "T"):
        try:
            values.append(float(components.get(key)))
        except (TypeError, ValueError):
            pass
    for item in evidence:
        try:
            values.append(float(item.get("value")))
        except (TypeError, ValueError):
            pass
    return values


def validate_response(text: str, record: dict, evidence: List[dict]) -> dict:
    """校验大模型输出是否只引用了证据中存在的数字与车辆。

    - 小数或非通用的 >=10 整数必须在证据数值中找到（容差 5%）；
    - 若出现 ``camera:track`` 形式的车辆编号，必须与当前记录一致；
    - 过短或空输出直接判为无效。
    """
    issues: List[str] = []
    if not text or len(text.strip()) < 10:
        issues.append("empty_or_too_short")
        return {"valid": False, "issues": issues}

    allowed = _allowed_numbers(record, evidence)
    # 时间戳里的 2026 / 09:09 不应被当成编造数字或车辆编号
    cleaned = _TIME_PATTERN.sub(" ", text)
    for token in _NUMBER_PATTERN.findall(cleaned):
        number = float(token)
        if "." not in token and (number < 10 or int(number) in _GENERIC_INTEGERS):
            continue  # 忽略“3 句话”“24 小时”这类通用整数
        grounded = any(
            abs(number - value) <= max(0.05 * max(abs(value), 1.0), 0.05)
            for value in allowed
        )
        if not grounded:
            issues.append(f"ungrounded_number:{token}")

    vehicle_id = f"{record.get('camera_id', '')}:{record.get('track_id', '')}"
    for match in _VEHICLE_ID_PATTERN.findall(cleaned):
        if match != vehicle_id:
            issues.append(f"foreign_vehicle_id:{match}")

    return {"valid": not issues, "issues": issues}


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class TemplateReportGenerator:
    """确定性的模板报告：每个结论都能对应到证据条目。"""

    def _vehicle_line(self, record: dict) -> str:
        parts = [
            f"摄像头 {record.get('camera_id', '') or '-'}",
            f"轨迹 {record.get('track_id', '') or '-'}",
        ]
        if record.get("plate_number"):
            parts.append(f"车牌 {record['plate_number']}")
        if record.get("vehicle_type"):
            parts.append(f"车型 {record['vehicle_type']}")
        if record.get("vehicle_color"):
            parts.append(f"颜色 {record['vehicle_color']}")
        return "，".join(parts)

    def _suggestions(self, record: dict, evidence: List[dict]) -> List[str]:
        signals = {item["signal"] for item in evidence}
        components = record.get("risk_components", {})
        a = _to_float(components.get("A"))
        g = _to_float(components.get("G"))
        t = _to_float(components.get("T"))
        residual = _to_float(record.get("traj_residual"))
        level = str(record.get("risk_level", "low"))

        # 低风险车辆的结论是“未发现需要优先处置的异常”，
        # 排查建议不能反过来写“偏离正常分布、建议重点核对”，否则自相矛盾。
        if level == "low":
            if "rule_engine" in signals:
                return ["该车命中规则阈值，建议按规则类型现场核查。"]
            return ["风险信号较弱，可作为常规抽查对象。"]

        suggestions = []
        if "isolation_forest" in signals and a >= 0.6:
            suggestions.append("该车行为特征偏离正常分布，建议调取视频复核其通行过程。")
        if "transformer_residual" in signals and residual >= 20.0:
            suggestions.append("该车轨迹预测残差偏大，存在急转、异常变道或轨迹异常，建议重点核对。")
        if "graph_anomaly" in signals and g >= 0.8:
            suggestions.append("该车处于连接异常密集的子图，建议连同同社区车辆一并核查是否结伴/团伙通行。")
        if "time_frequency" in signals and t >= 0.5:
            suggestions.append("该车存在高频或夜间通行特征，建议核对通行事由与申报记录。")
        if "rule_engine" in signals:
            suggestions.append("该车命中规则阈值，建议按规则类型现场核查。")
        if not suggestions:
            suggestions.append("风险信号较弱，可作为常规抽查对象。")
        return suggestions

    def generate(self, record: dict, evidence: Optional[List[dict]] = None) -> str:
        evidence = evidence if evidence is not None else record.get("evidence", [])
        level = record.get("risk_level", "low")
        level_label = {"high": "高风险", "medium": "中风险", "low": "低风险"}.get(level, level)
        components = record.get("risk_components", {})

        lines = [
            f"### 车辆 {record.get('camera_id', '')}:{record.get('track_id', '')}（{level_label}）",
            "",
            f"**一句话结论**：{self._conclusion(record, evidence, level_label)}",
            "",
        ]

        time_line = self._time_line(record)
        if time_line:
            lines.append(f"- 通行时间：{time_line}")
        lines.append(f"- 通行地点：{record.get('camera_id', '')} 行政车道")
        # graph_available=0 表示该车没有图谱节点（例如图谱与向量键不一致），
        # 此时不能写成“图 0.00”，否则会被误读为“图异常很低”。
        graph_available = str(record.get("graph_available", 1)).strip() not in {"0", "False", "false"}
        graph_text = f"图 {_to_float(components.get('G')):.2f}" if graph_available else "图 无数据"
        lines.append(
            f"- 风险构成：行为 {_to_float(components.get('A')):.2f} / "
            f"{graph_text} / "
            f"时间频率 {_to_float(components.get('T')):.2f}，"
            f"综合 R = {record.get('risk_score_100', 0):.1f} 分"
        )
        if record.get("_source_file"):
            lines.append(f"- 数据来源：`{record['_source_file']}`")
        if record.get("global_vehicle_id") not in (None, ""):
            lines.append(f"- 跨机位关联：全局车辆 ID {record['global_vehicle_id']}")

        lines.append("")
        level = str(record.get("risk_level", "low"))
        lines.append("**指标读数（未达告警阈值）：**" if level == "low" else "**证据链：**")
        if evidence:
            for index, item in enumerate(evidence, start=1):
                trace = item.get("trace", {})
                location = f"{trace.get('camera_id', '')}:{trace.get('track_id', '')}"
                if level == "low":
                    # 低风险车辆只陈述读数，不下“不符/偏离”的判断，避免与结论自相矛盾
                    lines.append(f"{index}. {item.get('description', '')}（{location}）")
                else:
                    business, detail = self._business_evidence(item)
                    if detail:
                        lines.append(f"{index}. {business}（{detail}；{location}）")
                    else:
                        lines.append(f"{index}. {business}（{location}）")
        else:
            lines.append("1. 无显著异常证据。")

        lines.append("")
        lines.append("**排查建议：**")
        for suggestion in self._suggestions(record, evidence):
            lines.append(f"- {suggestion}")

        return "\n".join(lines)

    def _time_line(self, record: dict) -> Optional[str]:
        enter = str(record.get("enter_time", "") or "").strip()
        exit_ = str(record.get("exit_time", "") or "").strip()
        if not enter and not exit_:
            return None

        def _fmt(value: str) -> str:
            if not value:
                return "?"
            try:
                return datetime.fromisoformat(value).strftime("%H:%M:%S")
            except (TypeError, ValueError):
                return value

        duration = _to_float(record.get("duration_sec"))
        return f"{_fmt(enter)} — {_fmt(exit_)}（约 {duration:.0f} 秒）"

    # 把不同类型的证据归一化到 [0,1]，用于挑“最强信号”作为主要疑点。
    # 规则命中是明确违规，直接给最高分。
    _SEVERITY = {
        "rule_engine": lambda value: 1.0,
        "graph_anomaly": lambda value: value,
        "isolation_forest": lambda value: value,
        "transformer_residual": lambda value: min(value / 100.0, 1.0),
    }

    def _key_issue(self, evidence: List[dict]) -> tuple:
        """选出最强信号作为主要疑点。

        只有达到业务阈值的条目才会给出“疑点”措辞；否则返回空，
        由 ``_conclusion`` 退化为“多项指标偏高但未触单项阈值”的说法。
        """
        best_text = ""
        best_score = -1.0
        for item in evidence:
            scorer = self._SEVERITY.get(item.get("signal", ""))
            if scorer is None:
                continue
            try:
                value = float(item.get("value"))
            except (TypeError, ValueError):
                value = 0.0
            score = scorer(value)
            if score <= best_score:
                continue
            business, _ = self._business_evidence(item)
            # 未达阈值时 _business_evidence 返回的是原始读数，不能当作“疑点”
            if business == item.get("description", ""):
                continue
            best_score = score
            best_text = business
        return best_text, best_score

    def _conclusion(self, record: dict, evidence: List[dict], level_label: str) -> str:
        time_line = self._time_line(record)
        when = f"于 {time_line.split('（')[0]} " if time_line else ""
        key_issue, _ = self._key_issue(evidence)

        risk = f"{record.get('risk_score_100', 0):.1f}"
        level = str(record.get("risk_level", "low"))
        # 风险较低的车辆不应声称存在“主要疑点”，否则会误导一线人员
        if level not in {"high", "medium"}:
            return (
                f"该车{when}通过行政通道，综合风险 {risk} 分"
                f"（{level_label}），风险较低，未发现需要优先处置的异常，可作为常规抽查对象。"
            )
        if not key_issue:
            return (
                f"该车{when}通过行政通道，综合风险 {risk} 分（{level_label}），"
                f"多项指标同时偏高但均未单独触及阈值，建议结合录像综合判断。"
            )
        return (
            f"该车{when}通过行政通道，综合风险 {risk} 分"
            f"（{level_label}），主要疑点是「{key_issue}」，建议按下方清单核查。"
        )

    _BUSINESS_TEXT = BUSINESS_EVIDENCE_TEXT

    # 只有达到这些量级，才把指标读数写成“疑点”；否则只陈述读数。
    # 否则 A=0.49 也会被写成“行为模式与常态不符”，与“低风险”结论矛盾。
    _BUSINESS_MIN = {
        "isolation_forest": 0.60,
        "graph_anomaly": 0.80,
        "transformer_residual": 20.0,
        "time_frequency": 0.50,
    }

    def _business_evidence(self, item: dict) -> tuple:
        signal = item.get("signal", "")
        business = self._BUSINESS_TEXT.get(signal)
        detail = item.get("description", "")
        if business is None:
            return detail, ""

        threshold = self._BUSINESS_MIN.get(signal)
        if threshold is not None:
            try:
                value = float(item.get("value"))
            except (TypeError, ValueError):
                value = None
            if value is not None and value < threshold:
                # 未达疑点量级：只陈述读数，不下判断
                return detail, ""

        return business, detail


class LocalLLMReportGenerator:
    """调用 OpenAI 兼容的本地/内网大模型接口。"""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        model: str = "qwen2.5-7b-instruct",
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def available(self) -> bool:
        return bool(self.endpoint)

    def _prompt(self, record: dict, evidence: List[dict]) -> str:
        business_lines = []
        for index, item in enumerate(evidence, start=1):
            label = BUSINESS_EVIDENCE_TEXT.get(item.get("signal", ""))
            detail = item.get("description", "")
            business_lines.append(
                f"{index}. {label}（{detail}）" if label else f"{index}. {detail}"
            )

        level_label = {"high": "高风险", "medium": "中风险", "low": "低风险"}.get(
            str(record.get("risk_level", "")), str(record.get("risk_level", ""))
        )
        return (
            "你是综合保税区行政通道的监管辅助助手。请只依据下面的结构化证据，"
            "用面向一线执法人员的业务语言写 3~5 句话：\n"
            "1) 第一句给结论：这辆车什么时候通过、风险高低、最需要注意什么；\n"
            "2) 后面 2~3 句给可执行的核查建议（查什么、查哪里）；\n"
            "3) 只能引用证据里出现的数字，不得编造车牌、时间或结论；\n"
            "4) 不要使用“异常指数 A=0.45”这类术语，改说“行为模式与常态不符”。\n\n"
            f"车辆：{record.get('camera_id', '')}:{record.get('track_id', '')}\n"
            f"通行时间：{_format_time(record.get('enter_time'))} ~ {_format_time(record.get('exit_time'))}\n"
            f"综合风险：{record.get('risk_score_100', 0):.1f} 分（{level_label}）\n"
            "证据：\n" + "\n".join(business_lines)
        )

    def generate(self, record: dict, evidence: Optional[List[dict]] = None) -> str:
        if not self.available():
            raise RuntimeError("未配置本地大模型接口")
        evidence = evidence if evidence is not None else record.get("evidence", [])
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是海关综保区监管辅助助手，只依据给定证据回答。"},
                {"role": "user", "content": self._prompt(record, evidence)},
            ],
            "temperature": 0.1,
        }
        data = json.dumps(payload).encode("utf-8")

        last_error: Optional[Exception] = None
        # 首次调用常因模型冷启动较慢，重试一次
        for _ in range(2):
            try:
                request = urllib.request.Request(
                    self.endpoint,
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                if self.api_key:
                    request.add_header("Authorization", f"Bearer {self.api_key}")
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"].strip()
            except Exception as exc:  # noqa: BLE001 - 重试后仍失败则交给上层回退
                last_error = exc
        raise last_error if last_error else RuntimeError("大模型调用失败")


class ReportService:
    """优先使用大模型，失败时回退模板。"""

    def __init__(
        self,
        template: Optional[TemplateReportGenerator] = None,
        llm: Optional[LocalLLMReportGenerator] = None,
    ):
        self.template = template or TemplateReportGenerator()
        self.llm = llm

    def generate(self, record: dict, evidence: Optional[List[dict]] = None) -> dict:
        evidence = evidence if evidence is not None else record.get("evidence", [])
        template_text = self.template.generate(record, evidence)
        if self.llm is not None and self.llm.available():
            try:
                text = self.llm.generate(record, evidence)
                validation = validate_response(text, record, evidence)
                if validation["valid"]:
                    # 模板保留结构化证据与风险分，大模型只补充自然语言建议
                    combined = f"{template_text}\n\n**大模型排查建议（本地模型）：**\n{text}"
                    return {"text": combined, "backend": "llm", "validation": validation}
                # 大模型引用了证据外的数字/车辆，回退到确定性模板
                return {
                    "text": template_text,
                    "backend": "template_fallback",
                    "validation": validation,
                }
            except Exception:
                # 大模型不可用时静默回退，保证报告始终可产出
                pass
        return {
            "text": template_text,
            "backend": "template",
            "validation": {"valid": True, "issues": []},
        }
