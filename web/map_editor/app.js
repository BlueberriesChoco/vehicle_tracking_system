(() => {
  "use strict";

  const STORAGE_KEY = "vehicle-tracking-zone-map-v1";
  const MAP_WIDTH = 1280;
  const MAP_HEIGHT = 1809;
  const SVG_NS = "http://www.w3.org/2000/svg";

  const initialItems = [
    camera("G-01", 307, 576, "西侧边界出入口", "规划", 0),
    camera("G-02", 1114, 563, "东北侧边界出入口", "规划", 180),
    camera("G-03", 550, 1203, "南侧行政通道候选入口", "待核实", 270),
    camera("J-01", 666, 384, "北区西侧首个分叉", "规划", 90),
    camera("J-02", 563, 678, "中央西侧汇合点", "规划", 45),
    camera("J-03", 922, 678, "东侧中部三岔口", "规划", 225),
    camera("J-04", 614, 922, "南区中央分叉", "规划", 90),
    camera("J-05", 1075, 883, "东侧道路折转点", "规划", 180),
    camera("J-06", 448, 1050, "西南片区转向点", "规划", 0),
    camera("J-07", 883, 397, "北区内部主路口", "规划", 180),
    camera("R-01", 832, 1165, "东南长路段中继", "规划", 315),
    camera("R-02", 435, 461, "西北入口连接路中继", "规划", 45),
    camera("R-03", 602, 806, "中央南北道路中继", "规划", 90),
    camera("R-04", 998, 730, "东侧纵向道路中继", "规划", 270)
  ];

  const state = {
    items: loadItems(),
    selectedId: null,
    tool: "select",
    zoom: 1,
    drag: null,
    roadDraft: null,
    history: [],
    future: []
  };

  const overlay = document.getElementById("map-overlay");
  const stage = document.getElementById("map-stage");
  const baseMap = document.getElementById("base-map");
  const itemForm = document.getElementById("item-form");
  const emptyInspector = document.getElementById("empty-inspector");
  const itemList = document.getElementById("item-list");
  const deleteButton = document.getElementById("delete-button");
  const cameraTooltip = document.getElementById("camera-tooltip");
  const saveState = document.getElementById("save-state");
  const pointerPosition = document.getElementById("pointer-position");
  const toolHint = document.getElementById("tool-hint");

  overlay.setAttribute("viewBox", `0 0 ${MAP_WIDTH} ${MAP_HEIGHT}`);
  overlay.setAttribute("preserveAspectRatio", "xMidYMid meet");

  function makeId() {
    if (crypto.randomUUID) return crypto.randomUUID();
    return `item-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }

  function camera(cameraId, x, y, name, status, direction) {
    return {
      id: makeId(), type: "camera", x, y, name, cameraId,
      ip: "", direction, status, cameraType: "视频跟踪", notes: ""
    };
  }

  function deepCopy(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function loadItems() {
    try {
      const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
      return Array.isArray(saved?.items) ? saved.items : deepCopy(initialItems);
    } catch {
      return deepCopy(initialItems);
    }
  }

  function saveItems() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({version: 1, items: state.items}));
    saveState.textContent = `已自动保存 · ${new Date().toLocaleTimeString("zh-CN", {hour12: false})}`;
  }

  function snapshot() {
    state.history.push(deepCopy(state.items));
    if (state.history.length > 60) state.history.shift();
    state.future = [];
    updateHistoryButtons();
  }

  function restore(items) {
    state.items = deepCopy(items);
    state.selectedId = null;
    state.roadDraft = null;
    saveItems();
    render();
  }

  function undo() {
    if (!state.history.length) return;
    state.future.push(deepCopy(state.items));
    restore(state.history.pop());
    updateHistoryButtons();
  }

  function redo() {
    if (!state.future.length) return;
    state.history.push(deepCopy(state.items));
    restore(state.future.pop());
    updateHistoryButtons();
  }

  function updateHistoryButtons() {
    document.getElementById("undo-button").disabled = !state.history.length;
    document.getElementById("redo-button").disabled = !state.future.length;
  }

  function svgElement(name, attributes = {}) {
    const element = document.createElementNS(SVG_NS, name);
    Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
    return element;
  }

  function addText(parent, x, y, value, className, anchor = "middle") {
    const text = svgElement("text", {x, y, class: className, "text-anchor": anchor});
    text.textContent = value;
    parent.appendChild(text);
  }

  function render() {
    hideCameraTooltip();
    overlay.replaceChildren();
    const visible = {
      enterprise: document.getElementById("enterprise-layer").checked,
      road: document.getElementById("road-layer").checked,
      camera: document.getElementById("camera-layer").checked
    };

    state.items.forEach(item => {
      if (!visible[item.type]) return;
      const group = svgElement("g", {
        class: `map-item ${item.type} ${item.id === state.selectedId ? "selected" : ""}`,
        "data-id": item.id
      });

      if (item.type === "enterprise") renderEnterprise(group, item);
      if (item.type === "road") renderRoad(group, item);
      if (item.type === "camera") renderCamera(group, item);
      overlay.appendChild(group);
    });

    if (state.roadDraft?.points.length) {
      overlay.appendChild(svgElement("polyline", {
        class: "draft-road",
        points: state.roadDraft.points.map(point => `${point.x},${point.y}`).join(" ")
      }));
    }

    renderInspector();
    renderItemList();
    deleteButton.disabled = !state.selectedId;
  }

  function renderEnterprise(group, item) {
    group.appendChild(svgElement("rect", {
      x: item.x - item.width / 2,
      y: item.y - item.height / 2,
      width: item.width,
      height: item.height,
      rx: 4,
      class: "enterprise-shape"
    }));
    group.appendChild(svgElement("rect", {
      x: item.x - item.width / 2 - 8,
      y: item.y - item.height / 2 - 8,
      width: item.width + 16,
      height: item.height + 16,
      rx: 6,
      class: "selection-ring"
    }));
    addText(group, item.x, item.y + 7, item.name || "未命名企业", "map-label");
    const corners = [
      ["nw", item.x - item.width / 2, item.y - item.height / 2],
      ["ne", item.x + item.width / 2, item.y - item.height / 2],
      ["sw", item.x - item.width / 2, item.y + item.height / 2],
      ["se", item.x + item.width / 2, item.y + item.height / 2]
    ];
    corners.forEach(([handle, x, y]) => {
      group.appendChild(svgElement("rect", {
        x: x - 7, y: y - 7, width: 14, height: 14, rx: 2,
        class: "resize-handle", "data-handle": handle
      }));
    });
  }

  function renderRoad(group, item) {
    const pointString = item.points.map(point => `${point.x},${point.y}`).join(" ");
    group.appendChild(svgElement("polyline", {points: pointString, class: "road-hit"}));
    group.appendChild(svgElement("polyline", {points: pointString, class: "road-shape"}));
    const middle = item.points[Math.floor(item.points.length / 2)];
    if (middle) addText(group, middle.x, middle.y - 14, item.name || "未命名道路", "map-label");
  }

  function renderCamera(group, item) {
    const angle = Number(item.direction || 0) * Math.PI / 180;
    const dx = Math.cos(angle) * 24;
    const dy = -Math.sin(angle) * 24;
    group.appendChild(svgElement("line", {x1: item.x, y1: item.y, x2: item.x + dx, y2: item.y + dy, class: "camera-direction"}));
    group.appendChild(svgElement("circle", {cx: item.x, cy: item.y, r: 8, class: "camera-shape"}));
    group.appendChild(svgElement("circle", {cx: item.x, cy: item.y, r: 14, class: "selection-ring"}));
    addText(group, item.x + 13, item.y - 10, item.cameraId || "CAM", "map-label camera-id-label", "start");
    group.addEventListener("pointerenter", event => showCameraTooltip(event, item));
    group.addEventListener("pointermove", event => positionCameraTooltip(event));
    group.addEventListener("pointerleave", hideCameraTooltip);
  }

  function showCameraTooltip(event, item) {
    cameraTooltip.replaceChildren();
    const heading = document.createElement("strong");
    heading.textContent = `${item.cameraId || "CAM"} · ${item.name || "未命名摄像头"}`;
    const list = document.createElement("dl");
    [["IP", item.ip || "未填写"], ["类型", item.cameraType || "未填写"], ["朝向", `${Number(item.direction || 0)}°`], ["状态", item.status || "未填写"]].forEach(([label, value]) => {
      const term = document.createElement("dt");
      const description = document.createElement("dd");
      term.textContent = label;
      description.textContent = value;
      list.append(term, description);
    });
    cameraTooltip.append(heading, list);
    cameraTooltip.hidden = false;
    positionCameraTooltip(event);
  }

  function positionCameraTooltip(event) {
    const stageRect = stage.getBoundingClientRect();
    const tooltipWidth = cameraTooltip.offsetWidth || 210;
    const tooltipHeight = cameraTooltip.offsetHeight || 100;
    let left = event.clientX - stageRect.left + 14;
    let top = event.clientY - stageRect.top + 14;
    if (left + tooltipWidth > stageRect.width) left -= tooltipWidth + 28;
    if (top + tooltipHeight > stageRect.height) top -= tooltipHeight + 28;
    cameraTooltip.style.left = `${Math.max(6, left)}px`;
    cameraTooltip.style.top = `${Math.max(6, top)}px`;
  }

  function hideCameraTooltip() {
    cameraTooltip.hidden = true;
  }

  function renderInspector() {
    const item = selectedItem();
    emptyInspector.hidden = Boolean(item);
    itemForm.hidden = !item;
    if (!item) {
      itemForm.replaceChildren();
      return;
    }

    const common = `
      ${field("name", "名称", item.name)}
      <div class="field-row">
        ${field("x", "X 坐标", Math.round(item.x), "number")}
        ${field("y", "Y 坐标", Math.round(item.y), "number")}
      </div>`;

    if (item.type === "enterprise") {
      itemForm.innerHTML = `${common}
        ${field("category", "企业类型", item.category)}
        <div class="field-row">
          ${field("width", "区域宽度", item.width, "number")}
          ${field("height", "区域高度", item.height, "number")}
        </div>
        ${textarea("notes", "备注", item.notes)}`;
    } else if (item.type === "camera") {
      itemForm.innerHTML = `${field("cameraId", "摄像头 ID", item.cameraId)}${common}
        ${field("ip", "IP 地址", item.ip)}
        ${selectField("cameraType", "摄像头类型", item.cameraType, ["视频跟踪", "车牌抓拍", "车辆全景", "围网监控", "其他"])}
        <div class="field-row">
          ${field("direction", "朝向角度", item.direction, "number")}
          ${selectField("status", "状态", item.status, ["现有", "规划", "待核实", "停用"])}
        </div>
        ${textarea("notes", "备注", item.notes)}`;
    } else {
      itemForm.innerHTML = `${common}
        ${selectField("roadClass", "道路类型", item.roadClass, ["主路", "支路", "行政通道", "货运通道", "内部道路"])}
        ${textarea("notes", "备注", item.notes)}`;
    }
  }

  function field(name, label, value = "", type = "text") {
    return `<div class="field"><label for="field-${name}">${label}</label><input id="field-${name}" name="${name}" type="${type}" value="${escapeHtml(value)}"></div>`;
  }

  function textarea(name, label, value = "") {
    return `<div class="field"><label for="field-${name}">${label}</label><textarea id="field-${name}" name="${name}">${escapeHtml(value)}</textarea></div>`;
  }

  function selectField(name, label, value, options) {
    const optionHtml = options.map(option => `<option value="${escapeHtml(option)}" ${option === value ? "selected" : ""}>${escapeHtml(option)}</option>`).join("");
    return `<div class="field"><label for="field-${name}">${label}</label><select id="field-${name}" name="${name}">${optionHtml}</select></div>`;
  }

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"]/g, character => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"})[character]);
  }

  function renderItemList() {
    itemList.replaceChildren();
    const names = {enterprise: "企业", road: "道路", camera: "摄像头"};
    state.items.forEach(item => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = item.id === state.selectedId ? "active" : "";
      const name = document.createElement("span");
      name.textContent = item.type === "camera" ? `${item.cameraId} ${item.name}` : item.name;
      const kind = document.createElement("span");
      kind.className = "item-kind";
      kind.textContent = names[item.type];
      button.append(name, kind);
      button.addEventListener("click", () => selectItem(item.id));
      itemList.appendChild(button);
    });
  }

  function selectedItem() {
    return state.items.find(item => item.id === state.selectedId) || null;
  }

  function selectItem(id) {
    state.selectedId = id;
    setTool("select");
    render();
  }

  function setTool(tool) {
    state.tool = tool;
    if (tool !== "road") state.roadDraft = null;
    document.querySelectorAll("[data-tool]").forEach(button => button.classList.toggle("active", button.dataset.tool === tool));
    const hints = {
      select: "选择并拖动对象，右侧修改属性",
      enterprise: "点击地图添加企业区域",
      camera: "点击地图添加摄像头",
      road: "连续点击绘制道路，双击或按 Enter 结束"
    };
    toolHint.textContent = hints[tool];
    render();
  }

  function mapPoint(event) {
    const point = overlay.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    return point.matrixTransform(overlay.getScreenCTM().inverse());
  }

  function addEnterprise(point) {
    snapshot();
    const index = state.items.filter(item => item.type === "enterprise").length + 1;
    const item = {
      id: makeId(), type: "enterprise", x: point.x, y: point.y,
      width: 180, height: 100, name: `企业 ${index}`, category: "", notes: ""
    };
    state.items.push(item);
    state.selectedId = item.id;
    saveItems();
    setTool("select");
  }

  function addCamera(point) {
    snapshot();
    const index = state.items.filter(item => item.type === "camera").length + 1;
    const item = camera(`CAM-${String(index).padStart(2, "0")}`, point.x, point.y, "新摄像头", "规划", 0);
    state.items.push(item);
    state.selectedId = item.id;
    saveItems();
    setTool("select");
  }

  function addRoadPoint(point) {
    if (!state.roadDraft) {
      snapshot();
      state.roadDraft = {points: []};
    }
    state.roadDraft.points.push({x: point.x, y: point.y});
    render();
  }

  function finishRoad() {
    if (!state.roadDraft) return;
    if (state.roadDraft.points.length >= 2) {
      const index = state.items.filter(item => item.type === "road").length + 1;
      const item = {
        id: makeId(), type: "road", points: state.roadDraft.points,
        x: state.roadDraft.points[0].x, y: state.roadDraft.points[0].y,
        name: `道路 ${index}`, roadClass: "内部道路", notes: ""
      };
      state.items.push(item);
      state.selectedId = item.id;
      saveItems();
    }
    state.roadDraft = null;
    setTool("select");
  }

  function moveItem(item, dx, dy, original) {
    if (item.type === "road") {
      item.points = original.points.map(point => ({x: point.x + dx, y: point.y + dy}));
      item.x = original.x + dx;
      item.y = original.y + dy;
    } else {
      item.x = original.x + dx;
      item.y = original.y + dy;
    }
  }

  overlay.addEventListener("pointerdown", event => {
    const itemNode = event.target.closest(".map-item");
    const resizeHandle = event.target.closest(".resize-handle");
    if (state.tool === "select" && resizeHandle && itemNode) {
      const item = state.items.find(candidate => candidate.id === itemNode.dataset.id);
      state.selectedId = item.id;
      snapshot();
      state.resize = {id: item.id, handle: resizeHandle.dataset.handle, original: deepCopy(item)};
      overlay.setPointerCapture(event.pointerId);
      render();
      return;
    }
    if (state.tool === "select" && itemNode) {
      const item = state.items.find(candidate => candidate.id === itemNode.dataset.id);
      state.selectedId = item.id;
      snapshot();
      state.drag = {id: item.id, start: mapPoint(event), original: deepCopy(item)};
      overlay.setPointerCapture(event.pointerId);
      render();
      return;
    }
    if (state.tool === "enterprise") addEnterprise(mapPoint(event));
    if (state.tool === "camera") addCamera(mapPoint(event));
    if (state.tool === "road") addRoadPoint(mapPoint(event));
    if (state.tool === "select" && !itemNode) {
      state.selectedId = null;
      render();
    }
  });

  overlay.addEventListener("pointermove", event => {
    const point = mapPoint(event);
    pointerPosition.textContent = `X: ${Math.round(point.x)}, Y: ${Math.round(point.y)}`;
    if (state.resize) {
      const item = state.items.find(candidate => candidate.id === state.resize.id);
      resizeEnterprise(item, point, state.resize.handle, state.resize.original);
      render();
      return;
    }
    if (!state.drag) return;
    const item = state.items.find(candidate => candidate.id === state.drag.id);
    moveItem(item, point.x - state.drag.start.x, point.y - state.drag.start.y, state.drag.original);
    render();
  });

  overlay.addEventListener("pointerup", event => {
    if (state.resize) {
      state.resize = null;
      if (overlay.hasPointerCapture(event.pointerId)) overlay.releasePointerCapture(event.pointerId);
      saveItems();
      render();
      return;
    }
    if (!state.drag) return;
    state.drag = null;
    if (overlay.hasPointerCapture(event.pointerId)) overlay.releasePointerCapture(event.pointerId);
    saveItems();
    render();
  });

  function resizeEnterprise(item, point, handle, original) {
    const minimumWidth = 50;
    const minimumHeight = 35;
    let left = original.x - original.width / 2;
    let right = original.x + original.width / 2;
    let top = original.y - original.height / 2;
    let bottom = original.y + original.height / 2;

    if (handle.includes("w")) left = Math.min(point.x, right - minimumWidth);
    if (handle.includes("e")) right = Math.max(point.x, left + minimumWidth);
    if (handle.includes("n")) top = Math.min(point.y, bottom - minimumHeight);
    if (handle.includes("s")) bottom = Math.max(point.y, top + minimumHeight);

    item.x = (left + right) / 2;
    item.y = (top + bottom) / 2;
    item.width = right - left;
    item.height = bottom - top;
  }

  overlay.addEventListener("dblclick", event => {
    if (state.tool === "road") {
      event.preventDefault();
      finishRoad();
    }
  });

  itemForm.addEventListener("change", event => {
    const item = selectedItem();
    if (!item || !event.target.name) return;
    snapshot();
    const numericFields = new Set(["x", "y", "width", "height", "direction"]);
    item[event.target.name] = numericFields.has(event.target.name) ? Number(event.target.value) : event.target.value;
    if (item.type === "road" && (event.target.name === "x" || event.target.name === "y")) {
      const dx = item.x - item.points[0].x;
      const dy = item.y - item.points[0].y;
      item.points = item.points.map(point => ({x: point.x + dx, y: point.y + dy}));
    }
    saveItems();
    render();
  });

  document.querySelectorAll("[data-tool]").forEach(button => button.addEventListener("click", () => setTool(button.dataset.tool)));
  document.querySelectorAll(".layer-toggle input").forEach(input => input.addEventListener("change", render));

  deleteButton.addEventListener("click", () => {
    if (!state.selectedId) return;
    snapshot();
    state.items = state.items.filter(item => item.id !== state.selectedId);
    state.selectedId = null;
    saveItems();
    render();
  });

  document.getElementById("undo-button").addEventListener("click", undo);
  document.getElementById("redo-button").addEventListener("click", redo);

  document.getElementById("reset-button").addEventListener("click", () => {
    if (!confirm("恢复初始规划点位？当前本地编辑内容将被替换。")) return;
    snapshot();
    restore(initialItems);
  });

  function setZoom(nextZoom) {
    state.zoom = Math.min(2.2, Math.max(.55, nextZoom));
    stage.style.width = `${Math.round(Math.min(850, document.getElementById("map-viewport").clientWidth - 40) * state.zoom)}px`;
  }

  document.getElementById("zoom-in-button").addEventListener("click", () => setZoom(state.zoom + .15));
  document.getElementById("zoom-out-button").addEventListener("click", () => setZoom(state.zoom - .15));
  document.getElementById("zoom-reset-button").addEventListener("click", () => {
    state.zoom = 1;
    stage.style.width = "min(100%, 850px)";
  });

  document.getElementById("export-json-button").addEventListener("click", () => {
    downloadBlob("zone-a-map-data.json", JSON.stringify({version: 1, map: {width: MAP_WIDTH, height: MAP_HEIGHT}, items: state.items}, null, 2), "application/json");
  });

  document.getElementById("import-json-input").addEventListener("change", async event => {
    const file = event.target.files[0];
    if (!file) return;
    try {
      const data = JSON.parse(await file.text());
      if (!Array.isArray(data.items)) throw new Error("缺少 items 数组");
      snapshot();
      restore(data.items);
    } catch (error) {
      alert(`导入失败：${error.message}`);
    } finally {
      event.target.value = "";
    }
  });

  document.getElementById("export-png-button").addEventListener("click", exportPng);

  function downloadBlob(filename, content, type) {
    const url = URL.createObjectURL(new Blob([content], {type}));
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
  }

  async function exportPng() {
    const canvas = document.createElement("canvas");
    canvas.width = MAP_WIDTH;
    canvas.height = MAP_HEIGHT;
    const context = canvas.getContext("2d");
    context.drawImage(baseMap, 0, 0, MAP_WIDTH, MAP_HEIGHT);
    context.font = "600 23px Microsoft YaHei";
    context.textBaseline = "middle";
    state.items.forEach(item => drawCanvasItem(context, item));
    const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/png"));
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "zone-a-map.png";
    link.click();
    URL.revokeObjectURL(url);
  }

  function drawCanvasItem(context, item) {
    context.save();
    if (item.type === "enterprise") {
      context.fillStyle = "rgb(47 125 79 / 22%)";
      context.strokeStyle = "#2f7d4f";
      context.lineWidth = 3;
      context.fillRect(item.x - item.width / 2, item.y - item.height / 2, item.width, item.height);
      context.strokeRect(item.x - item.width / 2, item.y - item.height / 2, item.width, item.height);
      drawCanvasLabel(context, item.name, item.x, item.y, "center");
    } else if (item.type === "road") {
      context.strokeStyle = "#b56a16";
      context.lineWidth = 7;
      context.lineCap = "round";
      context.lineJoin = "round";
      context.beginPath();
      item.points.forEach((point, index) => index ? context.lineTo(point.x, point.y) : context.moveTo(point.x, point.y));
      context.stroke();
      const middle = item.points[Math.floor(item.points.length / 2)];
      drawCanvasLabel(context, item.name, middle.x, middle.y - 16, "center");
    } else {
      context.fillStyle = "#c83f34";
      context.strokeStyle = "#ffffff";
      context.lineWidth = 3;
      context.beginPath();
      context.arc(item.x, item.y, 13, 0, Math.PI * 2);
      context.fill();
      context.stroke();
      drawCanvasLabel(context, `${item.cameraId} ${item.name}`, item.x + 20, item.y - 10, "left");
    }
    context.restore();
  }

  function drawCanvasLabel(context, text, x, y, align) {
    context.textAlign = align;
    context.lineWidth = 5;
    context.strokeStyle = "#ffffff";
    context.fillStyle = "#17212b";
    context.strokeText(text || "", x, y);
    context.fillText(text || "", x, y);
  }

  document.addEventListener("keydown", event => {
    if (event.key === "Enter" && state.tool === "road") finishRoad();
    if (event.key === "Escape") setTool("select");
    if (event.key === "Delete" && state.selectedId && !["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) deleteButton.click();
    if (event.ctrlKey && event.key.toLowerCase() === "z") { event.preventDefault(); undo(); }
    if (event.ctrlKey && event.key.toLowerCase() === "y") { event.preventDefault(); redo(); }
  });

  baseMap.addEventListener("load", render);
  updateHistoryButtons();
  render();
})();
