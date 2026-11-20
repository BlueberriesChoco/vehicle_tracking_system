"use strict";

const state = {
  runs: [],
  activeRun: null,
  activeVehicle: null,
  detail: null,
};

const $ = (id) => document.getElementById(id);

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

function fmtTime(value) {
  const text = String(value || "");
  if (!text) return "—";
  return text.replace("T", " ").slice(0, 19);
}

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) throw new Error(`${response.status}`);
  return response.json();
}

function setBanner(message) {
  const banner = $("banner");
  if (!message) {
    banner.classList.add("hidden");
    banner.textContent = "";
    return;
  }
  banner.classList.remove("hidden");
  banner.textContent = message;
}

function setSysStatus(parts) {
  $("sysStatus").textContent = parts.length ? parts.join("　·　") : "";
}

function renderRuns() {
  const list = $("runList");
  list.innerHTML = "";
  $("runCount").textContent = state.runs.length ? `${state.runs.length} 批` : "";

  if (!state.runs.length) {
    list.appendChild(el("div", "empty", "还没有录像批次"));
    return;
  }

  for (const run of state.runs) {
    const item = el("div", "run-item" + (state.activeRun === run.id ? " active" : ""));
    item.appendChild(el("div", "name", run.label || run.name));
    const meta = [];
    if (run.medium) meta.push(`${run.medium} 辆中风险`);
    meta.push(`最高 ${run.risk_max ?? "—"}`);
    item.appendChild(el("div", "meta", meta.join("　·　")));
    item.addEventListener("click", () => selectRun(run.id));
    list.appendChild(item);
  }
}

function renderKpis(detail) {
  const items = [
    { label: "通过车辆", value: detail.vehicles ?? "—" },
    { label: "中风险", value: detail.medium ?? "—" },
    { label: "高风险", value: detail.high ?? "—" },
    { label: "证据可追溯", value: detail.evidence_total ?? "—" },
  ];
  const wrap = $("kpis");
  wrap.innerHTML = "";
  for (const item of items) {
    const cell = el("div", "kpi");
    cell.appendChild(el("div", "value", String(item.value)));
    cell.appendChild(el("div", "label", item.label));
    wrap.appendChild(cell);
  }
}

function renderReview(rows) {
  const tbody = $("reviewTable").querySelector("tbody");
  tbody.innerHTML = "";
  $("reviewCount").textContent = rows.length ? `${rows.length} 辆` : "";

  if (!rows.length) {
    const tr = el("tr");
    const td = el("td", "dim", "本批次没有可核查车辆");
    td.colSpan = 6;
    tr.appendChild(td);
    tbody.appendChild(tr);
    return;
  }

  for (const row of rows) {
    const tr = el("tr");
    tr.dataset.pid = row.passage_id;
    if (state.activeVehicle === row.passage_id) tr.classList.add("active");

    const cells = [
      ["#", row.rank, "dim"],
      ["车辆", row.display_name, "vehicle"],
      ["通过时间", fmtTime(row.enter_time), "dim"],
      ["时长", row.duration_sec === "" ? "—" : `${row.duration_sec}s`, "dim"],
      ["可疑度", row.risk_score_100, "score"],
      ["主要原因", row.reason || "—", "reason"],
    ];
    for (const [label, value, cls] of cells) {
      const td = el("td", cls, value === undefined || value === null || value === "" ? "—" : String(value));
      td.title = label;
      tr.appendChild(td);
    }
    tr.addEventListener("click", () => selectVehicle(row.passage_id));
    tbody.appendChild(tr);
  }
}

function renderDetail(record) {
  $("detailEmpty").classList.add("hidden");
  const body = $("detailBody");
  body.classList.remove("hidden");
  body.innerHTML = "";

  const display = record.display || {};
  const score = Number(record.risk_score_100 || 0);

  // 1) 截图
  const shot = el("div", "vehicle-shot-wrap");
  const image = el("img", "vehicle-shot");
  image.alt = "车辆截图";
  image.src = `/api/runs/${encodeURIComponent(state.activeRun)}/vehicles/${encodeURIComponent(record.passage_id)}/thumbnail?t=${Date.now()}`;
  image.addEventListener("error", () => {
    shot.innerHTML = "";
    const placeholder = el("div", "vehicle-shot-empty", "该车暂无截图");
    placeholder.appendChild(el("span", "", "可在后端回填后刷新"));
    shot.appendChild(placeholder);
  });
  shot.appendChild(image);
  body.appendChild(shot);

  // 2) 基本信息（不含算法参数）
  const head = el("div", "vehicle-head");
  head.appendChild(el("div", "vehicle-name", display.name || "未识别车牌"));
  const facts = el("div", "vehicle-facts");
  facts.appendChild(el("span", "", display.camera_name || record.camera_id || "—"));
  facts.appendChild(el("span", "", fmtTime(record.enter_time)));
  facts.appendChild(el("span", "", `停留 ${record.duration_sec ?? "—"} 秒`));
  head.appendChild(facts);
  body.appendChild(head);

  // 3) 为什么列为可疑
  body.appendChild(el("h3", "", "为什么列为可疑"));
  const headline = el("div", "headline", display.headline || "综合多项信号。");
  body.appendChild(headline);

  const reasonList = el("div", "reason-list");
  for (const reason of display.reasons || []) {
    const item = el("div", "reason-item");
    item.appendChild(el("span", "reason-label", reason.label || "信号"));
    item.appendChild(el("span", "", reason.plain || reason.text || ""));
    reasonList.appendChild(item);
  }
  if (!reasonList.children.length) {
    reasonList.appendChild(el("div", "dim", "暂无结构化原因"));
  }
  body.appendChild(reasonList);

  // 4) 反查依据（折叠）
  const evidence = record.evidence || [];
  if (evidence.length) {
    const details = document.createElement("details");
    details.appendChild(el("summary", "", `反查依据（${evidence.length} 条）`));
    const traceList = el("div", "trace-list");
    for (const item of evidence) {
      const trace = item.trace || {};
      const line = el("div", "trace-item");
      line.appendChild(el("span", "trace-time", trace.timestamp || "—"));
      line.appendChild(el("span", "", String(trace.source_file || "").split(/[\\/]/).pop() || "—"));
      line.appendChild(el("span", "dim", trace.track_id ? `轨迹 ${trace.track_id}` : ""));
      traceList.appendChild(line);
    }
    details.appendChild(traceList);
    body.appendChild(details);
  }

  // 5) 排查建议（折叠）
  if (record.report_text) {
    const details = document.createElement("details");
    details.appendChild(el("summary", "", "排查建议"));
    const pre = el("div", "report-text", record.report_text);
    details.appendChild(pre);
    body.appendChild(details);
  }
}

async function selectRun(runId) {
  state.activeRun = runId;
  state.activeVehicle = null;
  renderRuns();
  $("detailBody").classList.add("hidden");
  $("detailEmpty").classList.remove("hidden");

  const detail = await api(`/api/runs/${encodeURIComponent(runId)}`);
  state.detail = detail;
  renderKpis(detail);

  const status = [];
  if (detail.checks && !detail.checks.calibrated) status.push("未标定");
  if (detail.checks && detail.checks.in_sample) status.push("样本内训练");
  setSysStatus(status);
  setBanner(status.length ? "提示：本批次为" + status.join("、") + "，结果仅供排查参考，不作为考核结论。" : "");

  const budget = Number($("budget").value || 20);
  const rows = await api(`/api/runs/${encodeURIComponent(runId)}/review?budget=${budget}`);
  renderReview(rows);
}

async function selectVehicle(passageId) {
  state.activeVehicle = passageId;
  const rows = await api(`/api/runs/${encodeURIComponent(state.activeRun)}/review?budget=${Number($("budget").value || 20)}`);
  renderReview(rows);
  const record = await api(`/api/runs/${encodeURIComponent(state.activeRun)}/vehicles/${encodeURIComponent(passageId)}`);
  renderDetail(record);
}

async function boot() {
  await api("/api/health");
  state.runs = await api("/api/runs");
  renderRuns();

  $("refresh").addEventListener("click", async () => {
    state.runs = await api("/api/runs");
    renderRuns();
    if (state.activeRun) await selectRun(state.activeRun);
  });

  $("budget").addEventListener("change", async () => {
    if (state.activeRun) await selectRun(state.activeRun);
  });

  if (state.runs.length) await selectRun(state.runs[0].id);
}

boot().catch((error) => setBanner(`加载失败：${error.message}`));
