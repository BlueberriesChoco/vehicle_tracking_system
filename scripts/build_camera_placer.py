"""生成 web/camera_placer/index.html：在园区底图上点选 142 个机位位置。

机位名直接取 F:\\A区录像 新 下的目录名，避免手抄出错。
"""

import json
import pathlib

SOURCE_DIR = pathlib.Path(r"F:\A区录像 新")
OUTPUT = pathlib.Path("web/camera_placer/index.html")

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<title>机位落图工具</title>
<style>
  * { box-sizing: border-box; }
  body { margin:0; font-family:"Microsoft YaHei",system-ui,sans-serif; background:#0f172a; color:#e2e8f0; height:100vh; display:flex; flex-direction:column; }
  header { padding:8px 14px; background:#1e3a5f; display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
  header strong { font-size:15px; }
  header .hint { font-size:12px; color:#a9c4de; }
  button { padding:5px 11px; border:1px solid #64748b; background:#334155; color:#fff; border-radius:4px; cursor:pointer; font-size:13px; }
  button:hover { background:#475569; }
  button.active { background:#0ea5e9; border-color:#0ea5e9; }
  main { flex:1; display:grid; grid-template-columns: 1fr 300px; min-height:0; }
  #stage { overflow:auto; background:#050a14; position:relative; }
  canvas { display:block; cursor:crosshair; }
  aside { border-left:1px solid #334155; display:flex; flex-direction:column; min-height:0; }
  .search { padding:8px; border-bottom:1px solid #334155; }
  .search input { width:100%; padding:6px 8px; background:#0b1220; color:#e2e8f0; border:1px solid #334155; border-radius:4px; }
  #list { flex:1; overflow:auto; }
  .item { padding:5px 10px; font-size:13px; cursor:pointer; border-bottom:1px solid #1e293b; display:flex; justify-content:space-between; }
  .item:hover { background:#1e293b; }
  .item.sel { background:#0c4a6e; }
  .item.done { color:#4ade80; }
  .item .dot { font-size:11px; color:#64748b; }
  footer { padding:8px 12px; border-top:1px solid #334155; font-size:12px; color:#94a3b8; }
  #out { width:100%; height:150px; background:#0b1220; color:#86efac; border:1px solid #334155; border-radius:4px; font-size:11px; display:none; }
</style>
</head>
<body>
<header>
  <strong>机位落图工具</strong>
  <span class="hint">右侧选机位 → 图上点位置；点住可拖动微调</span>
  <label class="hint">缩放
    <select id="zoom">
      <option value="0.6">60%</option>
      <option value="0.8">80%</option>
      <option value="1" selected>100%</option>
      <option value="1.5">150%</option>
      <option value="2">200%</option>
    </select>
  </label>
  <button id="roadBtn">画道路</button>
  <button id="undo">撤销</button>
  <button id="export">导出 JSON</button>
</header>
<main>
  <div id="stage"><canvas id="cv"></canvas></div>
  <aside>
    <div class="search"><input id="q" placeholder="搜索机位名称…" /></div>
    <div id="list"></div>
  </aside>
</main>
<footer id="status">未选中机位</footer>
<textarea id="out"></textarea>

<script id="camera-data" type="application/json">__CAMERAS__</script>
<script>
const MAP_SRC = "../map_editor/assets/zone-a-base-map.jpg";
const CAMS = JSON.parse(document.getElementById("camera-data").textContent);
const cv = document.getElementById("cv");
const ctx = cv.getContext("2d");
const img = new Image();
let scale = 1;
let placed = [];
let roads = [];
let selected = null;
let roadMode = false;
let roadDraft = [];
let dragging = null;

img.onload = () => { resize(); draw(); };
img.src = MAP_SRC;

function resize() {
  cv.width = img.naturalWidth * scale;
  cv.height = img.naturalHeight * scale;
  cv.style.width = cv.width + "px";
  cv.style.height = cv.height + "px";
}

function draw() {
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.drawImage(img, 0, 0, cv.width, cv.height);
  ctx.setTransform(scale, 0, 0, scale, 0, 0);

  ctx.lineWidth = 3;
  roads.forEach((road, i) => {
    ctx.strokeStyle = i % 2 ? "#38bdf8" : "#f59e0b";
    ctx.beginPath();
    road.forEach((p, k) => k ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y));
    ctx.stroke();
  });
  if (roadDraft.length) {
    ctx.strokeStyle = "#22d3ee";
    ctx.setLineDash([8, 6]);
    ctx.beginPath();
    roadDraft.forEach((p, k) => k ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y));
    ctx.stroke();
    ctx.setLineDash([]);
  }

  ctx.font = "13px Microsoft YaHei";
  for (const p of placed) {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 7, 0, Math.PI * 2);
    ctx.fillStyle = p.id === selected ? "#f43f5e" : "#22c55e";
    ctx.fill();
    ctx.strokeStyle = "#0f172a"; ctx.lineWidth = 2; ctx.stroke();
    ctx.fillStyle = "#f8fafc";
    ctx.fillText(p.name, p.x + 10, p.y + 4);
  }
}

function toMap(ev) {
  const rect = cv.getBoundingClientRect();
  return { x: Math.round((ev.clientX - rect.left) / scale), y: Math.round((ev.clientY - rect.top) / scale) };
}

function nearest(ev) {
  const p = toMap(ev);
  let best = null, bestD = 14;
  for (const q of placed) {
    const d = Math.hypot(q.x - p.x, q.y - p.y);
    if (d < bestD) { bestD = d; best = q; }
  }
  return best;
}

cv.addEventListener("mousedown", (ev) => {
  if (roadMode) return;
  const hit = nearest(ev);
  if (hit) { dragging = hit; return; }
  if (!selected) { setStatus("先在右侧选一个机位"); return; }
  const p = toMap(ev);
  placed = placed.filter((q) => q.id !== selected);
  placed.push({ id: selected, name: selected, x: p.x, y: p.y });
  renderList(); draw();
  setStatus("已放置 " + selected + " (" + p.x + ", " + p.y + ")");
});

cv.addEventListener("mousemove", (ev) => {
  if (!dragging) return;
  const p = toMap(ev);
  dragging.x = p.x; dragging.y = p.y;
  draw();
});

window.addEventListener("mouseup", () => {
  if (dragging) { dragging = null; setStatus("位置已更新"); }
});

cv.addEventListener("click", (ev) => {
  if (!roadMode) return;
  roadDraft.push(toMap(ev));
  draw();
});

cv.addEventListener("dblclick", () => {
  if (roadMode && roadDraft.length >= 2) {
    roads.push(roadDraft.slice());
    roadDraft = [];
    draw();
    setStatus("道路已添加");
  }
});

cv.addEventListener("contextmenu", (ev) => {
  ev.preventDefault();
  const hit = nearest(ev);
  if (hit) {
    placed = placed.filter((q) => q !== hit);
    renderList(); draw();
    setStatus("已删除 " + hit.name);
  }
});

function renderList() {
  const q = document.getElementById("q").value.trim();
  const box = document.getElementById("list");
  box.innerHTML = "";
  const done = new Set(placed.map((p) => p.id));
  for (const name of CAMS) {
    if (q && name.indexOf(q) < 0) continue;
    const row = document.createElement("div");
    row.className = "item" + (name === selected ? " sel" : "") + (done.has(name) ? " done" : "");
    row.innerHTML = "<span>" + name + "</span><span class='dot'>" + (done.has(name) ? "已放" : "") + "</span>";
    row.addEventListener("click", () => {
      selected = name;
      renderList();
      setStatus("已选 " + name + "，在图上点位置");
    });
    box.appendChild(row);
  }
}

function setStatus(text) {
  document.getElementById("status").textContent =
    text + "　|　已放 " + placed.length + "/" + CAMS.length + " 个，道路 " + roads.length + " 条";
}

document.getElementById("q").addEventListener("input", renderList);
document.getElementById("zoom").addEventListener("change", (e) => {
  scale = Number(e.target.value); resize(); draw();
});
document.getElementById("roadBtn").addEventListener("click", (e) => {
  roadMode = !roadMode;
  e.target.classList.toggle("active", roadMode);
  setStatus(roadMode ? "画道路：依次点顶点，双击结束" : "退出画道路");
});
document.getElementById("undo").addEventListener("click", () => {
  if (roadMode && roadDraft.length) { roadDraft.pop(); draw(); }
  else if (placed.length) { const last = placed.pop(); renderList(); draw(); setStatus("撤销 " + last.name); }
});
document.getElementById("export").addEventListener("click", () => {
  const payload = {
    map: { width: img.naturalWidth, height: img.naturalHeight, source: "zone-a-base-map.jpg" },
    cameras: placed.slice().sort((a, b) => a.id.localeCompare(b.id, "zh")),
    roads: roads,
  };
  const text = JSON.stringify(payload, null, 2);
  const blob = new Blob([text], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "camera_layout.json";
  a.click();
  const out = document.getElementById("out");
  out.style.display = "block";
  out.value = text;
});

renderList();
setStatus("未选中机位");
</script>
</body>
</html>
"""


def main() -> None:
    names = sorted([p.name for p in SOURCE_DIR.iterdir() if p.is_dir()])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    html = HTML.replace("__CAMERAS__", json.dumps(names, ensure_ascii=False))
    OUTPUT.write_text(html, encoding="utf-8")
    print(f"写入 {OUTPUT}（{OUTPUT.stat().st_size} 字节，{len(names)} 个机位）")


if __name__ == "__main__":
    main()
