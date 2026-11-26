"""生成 web/roi_editor/index.html：在参考帧上人工标定 ROI 与出入口线。

把 manifest 内联进页面，双击即可打开（不需要起服务）。
"""

import json
import pathlib

MANIFEST = pathlib.Path("data/outputs/roi_frames/manifest.json")
OUTPUT = pathlib.Path("web/roi_editor/index.html")
FRAME_PREFIX = "../../data/outputs/roi_frames/"

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<title>机位 ROI 标定</title>
<style>
  * { box-sizing: border-box; }
  body { margin:0; font-family:"Microsoft YaHei",system-ui,sans-serif; background:#0f172a; color:#e2e8f0; height:100vh; display:flex; flex-direction:column; }
  header { padding:8px 12px; background:#1e3a5f; display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
  header strong { font-size:15px; }
  .hint { font-size:12px; color:#a9c4de; }
  button { padding:4px 10px; border:1px solid #64748b; background:#334155; color:#fff; border-radius:4px; cursor:pointer; font-size:13px; }
  button:hover { background:#475569; }
  button.active { background:#0ea5e9; border-color:#0ea5e9; }
  main { flex:1; display:grid; grid-template-columns:1fr 290px; min-height:0; }
  #stage { overflow:auto; background:#050a14; }
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
  footer { padding:6px 12px; border-top:1px solid #334155; font-size:12px; color:#94a3b8; display:flex; gap:14px; flex-wrap:wrap; }
</style>
</head>
<body>
<header>
  <strong>机位 ROI 标定</strong>
  <button id="mRoi" class="active">1 ROI</button>
  <button id="mEntry">2 入口线</button>
  <button id="mExit">3 出口线</button>
  <button id="finish">完成(Enter)</button>
  <button id="undo">撤销点</button>
  <button id="clear">清空当前</button>
  <button id="prev">上一台(P)</button>
  <button id="next">下一台(N)</button>
  <label class="hint"><input type="checkbox" id="bothWay" /> 双向车道</label>
  <button id="export">导出 JSON</button>
  <span class="hint">点图加点；ROI 至少 3 点；线只需 2 点</span>
</header>
<main>
  <div id="stage"><canvas id="cv"></canvas></div>
  <aside>
    <div class="search"><input id="q" placeholder="搜索机位…" /></div>
    <div id="list"></div>
  </aside>
</main>
<footer>
  <span id="status">未选择机位</span>
  <span id="progress"></span>
</footer>

<script id="manifest" type="application/json">__MANIFEST__</script>
<script>
const FRAME_PREFIX = "__FRAME_PREFIX__";
const CAMS = JSON.parse(document.getElementById("manifest").textContent).cameras;
const KEY = "roi-editor-v2";  // v2：清空旧 ROI 后重新标定，避免读到历史数据
const cv = document.getElementById("cv");
const ctx = cv.getContext("2d");
const img = new Image();
let index = -1;
let mode = "roi";            // roi | entry_line | exit_line
let draft = [];              // 当前正在画的点
let records = JSON.parse(localStorage.getItem(KEY) || "{}");

function cam() { return CAMS[index]; }
function rec() { return records[cam().camera] || {}; }
function save() { records[cam().camera] = rec(); localStorage.setItem(KEY, JSON.stringify(records)); }
function syncBothWay() { const r = rec(); document.getElementById("bothWay").checked = !!r.bidirectional; }

function ensureLoaded() {
  const c = cam();
  if (!c) return;
  draft = [];
  if (!c.frame) { setStatus("该机位没有参考帧"); ctx.clearRect(0,0,cv.width,cv.height); return; }
  img.onload = () => { cv.width = img.naturalWidth; cv.height = img.naturalHeight; draw(); };
  img.src = FRAME_PREFIX + encodeURIComponent(c.frame);
}

function draw() {
  if (!img.complete || !img.naturalWidth) return;
  cv.width = img.naturalWidth; cv.height = img.naturalHeight;
  ctx.drawImage(img, 0, 0);
  const c = cam();
  const r = rec();
  // 已有配置
  if ((r.roi && r.roi.length >= 3) || (c.roi && c.roi.length >= 3 && !r.roi)) {
    strokePoly((r.roi || c.roi), "#fbbf24", true);
  }
  if ((r.entry_line && r.entry_line.length === 2) || (c.entry_line && !r.entry_line)) {
    strokePoly((r.entry_line || c.entry_line), "#22c55e", false, "入口");
  }
  if ((r.exit_line && r.exit_line.length === 2) || (c.exit_line && !r.exit_line)) {
    strokePoly((r.exit_line || c.exit_line), "#ef4444", false, "出口");
  }
  // 正在画的
  if (draft.length) {
    ctx.strokeStyle = "#38bdf8"; ctx.lineWidth = 2;
    ctx.beginPath();
    draft.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
    ctx.stroke();
    draft.forEach((p) => { ctx.beginPath(); ctx.arc(p[0], p[1], 4, 0, Math.PI*2); ctx.fillStyle = "#38bdf8"; ctx.fill(); });
  }
}

function strokePoly(points, color, close, label) {
  ctx.strokeStyle = color; ctx.lineWidth = 3;
  ctx.beginPath();
  points.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
  if (close) ctx.closePath();
  ctx.stroke();
  if (label) {
    ctx.fillStyle = color; ctx.font = "16px Microsoft YaHei";
    ctx.fillText(label, points[0][0] + 6, points[0][1] - 8);
  }
}

function toPoint(ev) {
  const rect = cv.getBoundingClientRect();
  return [Math.round((ev.clientX - rect.left) * cv.width / rect.width),
          Math.round((ev.clientY - rect.top) * cv.height / rect.height)];
}

cv.addEventListener("click", (ev) => {
  if (!cam()) return;
  draft.push(toPoint(ev));
  draw();
  setStatus(`当前 ${mode} 已加 ${draft.length} 点`);
});

function finish() {
  if (!cam()) return;
  const need = mode === "roi" ? 3 : 2;
  if (draft.length < need) { setStatus(`${mode} 至少要 ${need} 个点`); return; }
  const r = rec();
  if (mode === "roi") r.roi = draft.slice();
  else r[mode] = [draft[0], draft[draft.length - 1]];
  draft = [];
  save(); draw(); renderList();
  setStatus(`${cam().camera} 的 ${mode} 已保存`);
}

function setStatus(text) { document.getElementById("status").textContent = text; updateProgress(); }
function updateProgress() {
  const done = CAMS.filter((c) => records[c.camera] && records[c.camera].roi).length;
  document.getElementById("progress").textContent = `已完成 ${done}/${CAMS.length}`;
}

function renderList() {
  const q = document.getElementById("q").value.trim();
  const box = document.getElementById("list");
  box.innerHTML = "";
  CAMS.forEach((c, i) => {
    if (q && c.camera.indexOf(q) < 0) return;
    const row = document.createElement("div");
    const hasRoi = (records[c.camera] || {}).roi || c.roi;
    row.className = "item" + (i === index ? " sel" : "") + (hasRoi ? " done" : "");
    row.innerHTML = `<span>${c.camera}</span><span class="dot">${hasRoi ? "已有ROI" : ""}</span>`;
    row.onclick = () => { index = i; ensureLoaded(); renderList(); syncBothWay(); setStatus(c.camera); };
    box.appendChild(row);
  });
}

function go(delta) {
  if (!CAMS.length) return;
  finishIfDraft();
  index = Math.max(0, Math.min(CAMS.length - 1, index + delta));
  ensureLoaded(); renderList(); syncBothWay(); setStatus(cam().camera);
}
function finishIfDraft() {
  if (draft.length >= (mode === "roi" ? 3 : 2)) finish(); else draft = [];
}

document.getElementById("mRoi").onclick = () => setMode("roi");
document.getElementById("mEntry").onclick = () => setMode("entry_line");
document.getElementById("mExit").onclick = () => setMode("exit_line");
function setMode(m) {
  mode = m; draft = [];
  for (const [id, key] of [["mRoi","roi"],["mEntry","entry_line"],["mExit","exit_line"]]) {
    document.getElementById(id).classList.toggle("active", key === m);
  }
  setStatus(`模式：${m}`);
}
document.getElementById("finish").onclick = finish;
document.getElementById("undo").onclick = () => { draft.pop(); draw(); };
document.getElementById("clear").onclick = () => {
  const r = rec();
  if (mode === "roi") delete r.roi; else delete r[mode];
  draft = []; save(); draw(); renderList();
};
document.getElementById("prev").onclick = () => go(-1);
document.getElementById("next").onclick = () => go(1);
document.getElementById("q").addEventListener("input", renderList);
document.getElementById("bothWay").addEventListener("change", (e) => {
  const r = rec(); r.bidirectional = e.target.checked; save();
  setStatus(e.target.checked ? "已标记为双向车道" : "单向车道");
});
document.getElementById("export").onclick = () => {
  const payload = { version: 1, cameras: [] };
  for (const c of CAMS) {
    const r = records[c.camera];
    if (!r) continue;
    const item = { camera: c.camera, width: c.width, height: c.height };
    if (r.roi) item.roi = r.roi;
    if (r.entry_line) item.entry_line = r.entry_line;
    if (r.exit_line) item.exit_line = r.exit_line;
    if (r.bidirectional) item.bidirectional = true;
    payload.cameras.push(item);
  }
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "roi_marks.json";
  a.click();
};
window.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter") { ev.preventDefault(); finish(); }
  else if (ev.key === "Backspace") { ev.preventDefault(); draft.pop(); draw(); }
  else if (ev.key === "n" || ev.key === "N") { ev.preventDefault(); go(1); }
  else if (ev.key === "p" || ev.key === "P") { ev.preventDefault(); go(-1); }
  else if (ev.key === "1") { setMode("roi"); }
  else if (ev.key === "2") { setMode("entry_line"); }
  else if (ev.key === "3") { setMode("exit_line"); }
});

renderList();
updateProgress();
if (CAMS.length) { index = 0; ensureLoaded(); syncBothWay(); setStatus(CAMS[0].camera); }
</script>
</body>
</html>
"""


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    html = HTML.replace("__MANIFEST__", json.dumps(manifest, ensure_ascii=False))
    html = html.replace("__FRAME_PREFIX__", FRAME_PREFIX)
    OUTPUT.write_text(html, encoding="utf-8")
    ready = sum(1 for c in manifest["cameras"] if c.get("frame"))
    print(f"写入 {OUTPUT}（{OUTPUT.stat().st_size} 字节，{len(manifest['cameras'])} 台，其中 {ready} 台有参考帧）")


if __name__ == "__main__":
    main()
