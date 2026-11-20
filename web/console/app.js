"use strict";

const state = {
  runs: [],
  activeRun: null,
  activeVehicle: null,
  detail: null,
};

const $ = (id) => document.getElementById(id);

function badge(text, cls) {
  const span = document.createElement("span");
  span.className = "badge" + (cls ? " " + cls : "");
  span.textContent = text;
  return span;
}

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    throw new Error(`${response.status} ${await response.text()}`);
  }
  return response.json();
}

function setBanner(messages) {
  const banner = $("banner");
  if (!messages.length) {
    banner.classList.add("hidden");
    banner.textContent = "";
    return;
  }
  banner.classList.remove("hidden");
  banner.textContent = messages.join("　｜　");
}

function renderRuns() {
  const list = $("runList");
  list.innerHTML = "";
  $("runCount").textContent = state.runs.length ? `${state.runs.length} 个` : "";

  if (!state.runs.length) {
    list.innerHTML = '<div class="empty">还没有 run。<br />用 run_full_pipeline.py 跑一次，或加 --allow-runs 启动服务。</div>';
    return;
  }

  for (const run of state.runs) {
    const item = document.createElement("div");
    item.className = "run-item" + (state.activeRun === run.id ? " active" : "");
    item.dataset.id = run.id;

    const name = document.createElement("div");
    name.className = "name";
    name.textContent = run.name;

    const meta = document.createElement("div");
    meta.className = "meta";
    meta.appendChild(badge(run.model_version, run.model_declared ? "ok" : "warn"));
    meta.appendChild(badge(run.calibrated ? "已标定" : "未标定", run.calibrated ? "ok" : "warn"));
    meta.appendChild(document.createTextNode(`${run.vehicles || 0} 车 / ${run.alerts || 0} 告警`));

    item.appendChild(name);
    item.appendChild(meta);
    item.addEventListener("click", () => selectRun(run.id));
    list.appendChild(item);
  }
}

function renderKpis(detail) {
  const items = [
    { label: "车辆数", value: detail.vehicles ?? "—" },
    { label: "告警数", value: detail.alerts ?? "—" },
    { label: "中风险", value: detail.medium ?? "—" },
    { label: "证据可追溯", value: detail.evidence_total ? `${detail.evidence_total}` : "—" },
  ];
  const wrap = $("kpis");
  wrap.innerHTML = "";
  for (const item of items) {
    const cell = document.createElement("div");
    cell.className = "kpi";
    cell.innerHTML = `<div class="value">${item.value}</div><div class="label">${item.label}</div>`;
    wrap.appendChild(cell);
  }
}

function renderReview(rows) {
  const tbody = $("reviewTable").querySelector("tbody");
  tbody.innerHTML = "";
  $("reviewCount").textContent = rows.length ? `${rows.length} 辆` : "";

  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="6" class="dim">该 run 没有可核查记录</td></tr>';
    return;
  }

  for (const row of rows) {
    const tr = document.createElement("tr");
    tr.dataset.pid = row.passage_id;
    if (state.activeVehicle === row.passage_id) tr.classList.add("active");

    const cells = [
      ["#", row.rank, "dim"],
      ["车辆", row.passage_id, ""],
      ["进入时间", (row.enter_time || "").replace("T", " ").slice(0, 19), "dim"],
      ["时长", row.duration_sec, "dim"],
      ["可疑度", row.risk_score_100, "score"],
      ["证据", row.top_evidence || "—", "evidence"],
    ];
    for (const [label, value, cls] of cells) {
      const td = document.createElement("td");
      td.textContent = value === "" || value === undefined || value === null ? "—" : value;
      if (cls) td.className = cls;
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

  const score = Number(record.risk_score_100 || 0);
  const components = record.risk_components || {};

  // 车辆截图：来自 make_vehicle_thumbnails.py 回填的 crops/<passage_id>.jpg
  const shot = document.createElement("div");
  shot.className = "vehicle-shot-wrap";
  const image = document.createElement("img");
  image.className = "vehicle-shot";
  image.alt = "行政通道车辆截图";
  image.src = `/api/runs/${encodeURIComponent(state.activeRun)}/vehicles/${encodeURIComponent(record.passage_id)}/thumbnail?t=${Date.now()}`;
  image.addEventListener("error", () => {
    shot.classList.add("missing");
    shot.innerHTML = '<div class="vehicle-shot-empty">该车暂无截图<br /><span>先跑 scripts/make_vehicle_thumbnails.py 回填</span></div>';
  });
  shot.appendChild(image);
  body.appendChild(shot);

  const head = document.createElement("div");
  head.innerHTML = `
    <div class="kv">
      <div class="k">车辆</div><div class="v">${record.passage_id || "—"}</div>
      <div class="k">进入时间</div><div class="v">${(record.enter_time || "").replace("T", " ").slice(0, 19) || "—"}</div>
      <div class="k">通行时长</div><div class="v">${record.duration_sec ?? "—"} 秒</div>
      <div class="k">风险等级</div><div class="v">${record.risk_level || "—"}</div>
      <div class="k">可疑度</div><div class="v">${score.toFixed(1)}</div>
    </div>
    <div class="scorebar"><span style="width:${Math.min(100, score)}%"></span></div>
  `;
  body.appendChild(head);

  const compTitle = document.createElement("h3");
  compTitle.textContent = `分量 A=${(components.A ?? 0)}　G=${(components.G ?? 0)}　T=${(components.T ?? 0)}`;
  body.appendChild(compTitle);

  const evidenceTitle = document.createElement("h3");
  evidenceTitle.textContent = `证据（${(record.evidence || []).length} 条）`;
  body.appendChild(evidenceTitle);

  const list = document.createElement("div");
  list.className = "evidence-list";
  for (const item of record.evidence || []) {
    const box = document.createElement("div");
    box.className = "evidence-item";
    const type = document.createElement("span");
    type.className = "type";
    type.textContent = item.category || item.signal || item.evidence_type || "证据";
    box.appendChild(type);
    box.appendChild(document.createTextNode(item.description || item.summary || ""));
    const trace = item.trace || {};
    if (trace.source_file) {
      const traceLine = document.createElement("div");
      traceLine.className = "dim";
      traceLine.style.marginTop = "3px";
      traceLine.textContent = `溯源：${trace.camera_id || ""} / ${trace.track_id || ""} / ${String(trace.source_file).split(/[\\/]/).pop()}`;
      box.appendChild(traceLine);
    }
    list.appendChild(box);
  }
  if (!list.children.length) {
    list.innerHTML = '<div class="dim">该车辆没有结构化证据</div>';
  }
  body.appendChild(list);

  if (record.report_text) {
    const reportTitle = document.createElement("h3");
    reportTitle.textContent = "排查报告";
    body.appendChild(reportTitle);
    const pre = document.createElement("div");
    pre.className = "report-text";
    pre.textContent = record.report_text;
    body.appendChild(pre);
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

  const warnings = [];
  if (!detail.calibrated) warnings.push("该机位未标定：速度/路径/聚集特征被门控，模型实际只用了少数特征");
  if (detail.checks && detail.checks.in_sample) warnings.push("样本内训练（--retrain）：指标不能当作泛化结果");
  if (detail.checks && !detail.checks.model_declared) warnings.push("未声明模型版本（缺 run.json），仅供演示");
  setBanner(warnings);

  const budget = Number($("budget").value || 20);
  const rows = await api(`/api/runs/${encodeURIComponent(runId)}/review?budget=${budget}`);
  renderReview(rows);
}

async function selectVehicle(passageId) {
  state.activeVehicle = passageId;
  renderReview(await api(`/api/runs/${encodeURIComponent(state.activeRun)}/review?budget=${Number($("budget").value || 20)}`));
  const record = await api(`/api/runs/${encodeURIComponent(state.activeRun)}/vehicles/${encodeURIComponent(passageId)}`);
  renderDetail(record);
}

async function boot() {
  const health = await api("/api/health");
  $("rootHint").textContent = health.root;
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

  if (state.runs.length) {
    await selectRun(state.runs[0].id);
  }
}

boot().catch((error) => {
  setBanner([`加载失败：${error.message}`]);
});
