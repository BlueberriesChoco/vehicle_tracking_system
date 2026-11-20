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

  // 1.5) 车牌条：1080p 下车牌只有 ~60px 高，放大给人看；机器识读仅作参考
  const plateBlock = el("div", "plate-block");
  const plateImg = el("img", "plate-strip");
  plateImg.alt = "车牌放大";
  plateImg.src = `/api/runs/${encodeURIComponent(state.activeRun)}/vehicles/${encodeURIComponent(record.passage_id)}/plate?t=${Date.now()}`;
  plateImg.addEventListener("error", () => plateBlock.remove());
  plateBlock.appendChild(plateImg);

  let caption;
  if (display.plate) {
    caption = `车牌：${display.plate}`;
  } else if (display.plate_text) {
    const conf = Number(display.plate_conf || 0).toFixed(2);
    caption = `机器识读：${display.plate_text}（置信度 ${conf}，仅供参考，请以放大图为准）`;
  } else {
    caption = "车牌未识别，请对照上方放大图人工核对";
  }
  plateBlock.appendChild(el("div", "plate-caption", caption));

  // 没有车牌时给一个人工确认入口：机器读不出，人读一次即可，系统记住
  if (!display.plate) {
    const editor = el("div", "plate-edit");
    const input = document.createElement("input");
    input.type = "text";
    input.placeholder = "对照上图填写车牌，如 苏F·DZ8777";
    input.maxLength = 12;
    const save = el("button", "plate-save", "保存");
    save.type = "button";
    save.addEventListener("click", async () => {
      const value = input.value.trim();
      if (!value) return;
      save.disabled = true;
      save.textContent = "保存中";
      try {
        const response = await fetch(
          `/api/runs/${encodeURIComponent(state.activeRun)}/vehicles/${encodeURIComponent(record.passage_id)}/plate`,
          { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ plate: value }) }
        );
        if (response.status === 403) {
          save.textContent = "只读模式";
          save.title = "启动服务时加 --allow-runs 才能保存";
        } else if (response.ok) {
          await selectVehicle(record.passage_id);
        } else {
          save.textContent = "保存失败";
        }
      } catch (error) {
        save.textContent = "保存失败";
      } finally {
        save.disabled = false;
        if (save.textContent === "保存中") save.textContent = "保存";
      }
    });
    editor.appendChild(input);
    editor.appendChild(save);
    plateBlock.appendChild(editor);
  }

  body.appendChild(plateBlock);

  // 2) 基本信息（不含算法参数）
  const head = el("div", "vehicle-head");
  head.appendChild(el("div", "vehicle-name", display.plate || display.plate_text || display.name || "未识别车牌"));
  const facts = el("div", "vehicle-facts");
  facts.appendChild(el("span", "", display.camera_name || record.camera_id || "—"));
  facts.appendChild(el("span", "", fmtTime(record.enter_time)));
  facts.appendChild(el("span", "", `停留 ${record.duration_sec ?? "—"} 秒`));
  const look = [display.vehicle_color, display.vehicle_type].filter(Boolean).join(" ");
  if (look) facts.appendChild(el("span", "", look));
  facts.appendChild(el("span", "dim", `轨迹 ${record.track_id ?? "—"}`));
  head.appendChild(facts);
  body.appendChild(head);

  // 2.5) 排序分构成：说明分数从哪来，避免把"关系分"当成违规证据
  const parts = display.parts;
  if (parts) {
    const comp = el("div", "score-parts");
    comp.appendChild(
      el(
        "div",
        "score-parts-title",
        `排序分 ${parts.total}　=　关系 ${parts.points.G}（${parts.shares.G}%）　+　行为 ${parts.points.A}（${parts.shares.A}%）　+　频次 ${parts.points.T}（${parts.shares.T}%）`
      )
    );
    if (state.detail && state.detail.checks && !state.detail.checks.calibrated) {
      comp.appendChild(
        el(
          "div",
          "score-caveat",
          "未标定：速度、路径、聚集特征不可用。「关系分」反映的是与同时段车流的连接结构，不等于违规证据。"
        )
      );
    }
    body.appendChild(comp);
  }

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
  const rows = await api(`/api/runs/${encodeURIComponent(runId)}/review?budget=${budget}&sort=${sortValue()}`);
  renderReview(rows);
}

async function selectVehicle(passageId) {
  state.activeVehicle = passageId;
  const rows = await api(`/api/runs/${encodeURIComponent(state.activeRun)}/review?budget=${Number($("budget").value || 20)}&sort=${sortValue()}`);
  renderReview(rows);
  const record = await api(`/api/runs/${encodeURIComponent(state.activeRun)}/vehicles/${encodeURIComponent(passageId)}`);
  renderDetail(record);
}

function sortValue() {
  const select = $("sort");
  return select ? select.value : "risk";
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

  const sortSelect = $("sort");
  if (sortSelect) {
    sortSelect.addEventListener("change", async () => {
      if (state.activeRun) await selectRun(state.activeRun);
    });
  }

  if (state.runs.length) await selectRun(state.runs[0].id);
}

boot().catch((error) => setBanner(`加载失败：${error.message}`));
