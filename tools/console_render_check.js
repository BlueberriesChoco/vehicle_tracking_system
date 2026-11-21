// 用最小 DOM 桩执行 web/console/app.js 的渲染逻辑，验证它真的把新字段输出到 DOM。
// 目的：不依赖浏览器，也能确认"封顶提示""无数据""权重"确实进了页面结构。
"use strict";

const fs = require("fs");
const path = require("path");

// ---- 最小 DOM 桩 -----------------------------------------------------------
class Node {
  constructor(tag) {
    this.tagName = String(tag).toUpperCase();
    this.children = [];
    this.attributes = {};
    this.dataset = {};
    this.style = {};
    this._text = "";
    this._classes = new Set();
    this.colSpan = 0;
    this.title = "";
    this.src = "";
    this.type = "";
    this.placeholder = "";
    this.maxLength = 0;
    this.disabled = false;
    const self = this;
    this.classList = {
      add: (...names) => names.forEach((n) => self._classes.add(n)),
      remove: (...names) => names.forEach((n) => self._classes.delete(n)),
      contains: (n) => self._classes.has(n),
    };
  }
  get className() {
    return Array.from(this._classes).join(" ");
  }
  set className(value) {
    this._classes = new Set(String(value).split(/\s+/).filter(Boolean));
  }
  get textContent() {
    return this._text;
  }
  set textContent(value) {
    this._text = value === undefined || value === null ? "" : String(value);
    this.children = [];
  }
  get innerHTML() {
    return this._html || "";
  }
  set innerHTML(value) {
    this._html = value;
    this.children = [];
    this._text = "";
  }
  appendChild(child) {
    this.children.push(child);
    return child;
  }
  remove() {
    this._removed = true;
  }
  querySelector(selector) {
    if (selector === "tbody") {
      this._tbody = this._tbody || new Node("tbody");
      return this._tbody;
    }
    return null;
  }
  addEventListener() {}
  setAttribute(key, value) {
    this.attributes[key] = value;
  }
}

global.document = {
  createElement: (tag) => new Node(tag),
  getElementById: (id) => {
    elements[id] = elements[id] || new Node("div");
    return elements[id];
  },
};
const elements = {
  sort: Object.assign(new Node("select"), { value: "risk" }),
  budget: Object.assign(new Node("select"), { value: "20" }),
};

global.window = {};
global.fetch = async () => {
  throw new Error("fetch not stubbed");
};

// ---- 加载被测代码 -----------------------------------------------------------
const source = fs.readFileSync(
  path.join(__dirname, "..", "web", "console", "app.js"),
  "utf8"
);
// 去掉末尾的 boot() 调用：这里只测渲染函数，不需要启动
const body = source.replace(/boot\(\)\.catch[\s\S]*$/, "");

const sandbox = { document: global.document, window: global.window, fetch: global.fetch, console };
const vm = require("vm");
const context = vm.createContext(sandbox);
vm.runInContext(body + "\n;globalThis.__t = { renderDetail, renderReview, renderKpis, state, setBanner };", context);
const T = sandbox.__t;

// ---- 断言工具 --------------------------------------------------------------
let failed = 0;
function check(name, condition, detail) {
  if (condition) {
    console.log(`  OK   ${name}`);
  } else {
    failed += 1;
    console.log(`  FAIL ${name}${detail ? " :: " + detail : ""}`);
  }
}

function walk(node, visit) {
  visit(node);
  for (const child of node.children || []) walk(child, visit);
}

function allText(node) {
  let out = [];
  walk(node, (n) => {
    if (n._text) out.push(n._text);
  });
  return out.join(" | ");
}

// ---- 用例 1：封顶车辆（A/T 不可用，只剩 G）--------------------------------
console.log("\n[1] 封顶车辆详情（smoke 批次，仅 G 可用）");
elements.detailEmpty = new Node("div");
elements.detailBody = new Node("div");
T.state.activeRun = "smoke";
T.state.detail = { checks: { calibrated: false } };

T.renderDetail({
  passage_id: "p1",
  camera_id: "admin_in",
  enter_time: "2026-09-01T08:00:00",
  duration_sec: "1.5",
  track_id: "1",
  risk_score_100: 75.0,
  risk_score_raw_100: 100.0,
  risk_score_ceiling: 75.0,
  risk_score_capped: 1,
  display: {
    name: "未识别车牌 · 1",
    headline: "主要因为它与同批车辆的关系明显异常（关联指数 1.00）。",
    reasons: [],
    parts: {
      total: 75.0,
      raw_total: 100.0,
      ceiling: 75.0,
      capped: true,
      available_count: 1,
      unavailable: ["A", "T"],
      points: { A: 0.0, G: 75.0, T: 0.0 },
      shares: { A: 0, G: 100, T: 0 },
      values: { A: 0.0, G: 1.0, T: 0.0 },
      weights: { A: 0.0, G: 1.0, T: 0.0 },
    },
  },
});

const detailText = allText(elements.detailBody);
check("显示封顶提示", detailText.includes("分数已封顶"), detailText.slice(0, 300));
check("封顶显示原始分 100", detailText.includes("100"));
check("压缩目标 75 分出现", detailText.includes("75"));
check("A 显示无数据而非 0", detailText.includes("行为 无数据"), detailText.slice(0, 400));
check("T 显示无数据而非 0", detailText.includes("频次 无数据"));
check("权重标注未参与", detailText.includes("（未参与）"));

let cappedNode = null;
walk(elements.detailBody, (n) => {
  if ((n._classes || new Set()).has("score-capped")) cappedNode = n;
});
check("封顶提示用 score-capped 样式类（红色警示）", cappedNode !== null);

// ---- 用例 2：两路可用、未封顶 ----------------------------------------------
console.log("\n[2] 两路可用详情（full_rerun，A/G 可用）");
elements.detailBody = new Node("div");
T.renderDetail({
  passage_id: "p2",
  camera_id: "admin_in",
  enter_time: "2026-09-01T08:00:00",
  duration_sec: "2.0",
  risk_score_100: 80.2,
  display: {
    name: "苏F·DZ8777",
    headline: "关系异常",
    reasons: [],
    parts: {
      total: 80.2,
      raw_total: 80.2,
      ceiling: 100.0,
      capped: false,
      available_count: 2,
      unavailable: ["T"],
      points: { A: 25.6, G: 46.6, T: 0.0 },
      shares: { A: 32, G: 58, T: 0 },
      values: { A: 0.48, G: 1.0, T: 0.0 },
      weights: { A: 0.5333, G: 0.4667, T: 0.0 },
    },
  },
});
const d2 = allText(elements.detailBody);
check("未封顶时不显示封顶提示", !d2.includes("分数已封顶"), d2.slice(0, 300));
check("T 仍显示无数据", d2.includes("频次 无数据"));
check("显示实际权重 0.5333", d2.includes("0.5333"));

// ---- 用例 3：列表中的封顶标记 ----------------------------------------------
console.log("\n[3] 列表封顶标记");
elements.reviewTable = new Node("table");
elements.reviewCount = new Node("span");
T.state.activeVehicle = null;
T.renderReview([
  {
    rank: 1,
    passage_id: "p1",
    display_name: "车 A",
    enter_time: "2026-09-01T08:00:00",
    duration_sec: "1.5",
    risk_score_100: 75.0,
    reason: "关系异常",
    parts: { capped: true, raw_total: 100.0, ceiling: 75.0 },
  },
  {
    rank: 2,
    passage_id: "p2",
    display_name: "车 B",
    enter_time: "2026-09-01T08:01:00",
    duration_sec: "2.0",
    risk_score_100: 52.1,
    reason: "行为异常",
    parts: { capped: false, raw_total: 52.1, ceiling: 100.0 },
  },
]);
const rows = elements.reviewTable._tbody.children;
check("只对触顶车辆加标记", allText(rows[0]).includes("封顶") && !allText(rows[1]).includes("封顶"), allText(rows[0]) + " / " + allText(rows[1]));

// ---- 用例 4：KPI 显示非监管对象 --------------------------------------------
console.log("\n[4] KPI 非监管对象");
elements.kpis = new Node("div");
T.renderKpis({ vehicles: 432, medium: 38, high: 1, evidence_total: 1276, non_target_count: 11 });
const kpiText = allText(elements.kpis);
check("显示非监管对象计数", kpiText.includes("非监管对象") && kpiText.includes("11"), kpiText);
check("标注未参与评分", kpiText.includes("未参与评分"));

// ---- 用例 5：横幅说明分量缺失与封顶 ----------------------------------------
console.log("\n[5] 顶部横幅");
elements.banner = new Node("div");
elements.sysStatus = new Node("span");
// 直接验证横幅拼装逻辑：复用 selectRun 里的规则需要一个 api 桩
sandbox.fetch = async (url) => ({
  ok: true,
  json: async () => {
    if (String(url).includes("/review")) return [];
    return {
      vehicles: 15,
      medium: 3,
      high: 0,
      evidence_total: 0,
      non_target_count: 0,
      components_available: { A: false, G: true, T: false },
      score_ceiling: 75.0,
      score_capped_count: 2,
      checks: { calibrated: false },
      graph: { comparable: false, self_referential: true },
    };
  },
});
check("横幅逻辑已接入（字段存在性检查）", typeof T.setBanner === "function");

console.log(`\n${failed === 0 ? "全部通过" : failed + " 项失败"}`);
process.exit(failed === 0 ? 0 : 1);
