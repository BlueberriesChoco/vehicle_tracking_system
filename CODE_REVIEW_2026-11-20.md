# 代码核查报告（2026-11-20）

> 范围：`src/`（56 个模块）、`scripts/`（48 个脚本）、`tests/`（44 个测试文件），共 22753 行。
> 方式：**只读核查**，本轮未修改任何文件。
> 基线：`pytest -q` → **309 passed**（37.9s）；分支 `acceptance-fixes`，HEAD `c228056`。

---

## 0. 结论

代码整体健康度高：无 TODO/FIXME、无裸 `except:`、无可变默认参数、无 `src/` 层 `print` 残留、
无全局可变状态、GAE 改造成果已入库且测试覆盖到位。

但有 **3 个实质缺陷** 会直接影响对外报数口径，都属于「分数语义」层面，不是崩溃型 bug：

| 编号 | 缺陷 | 严重度 | 影响指标 |
|---|---|---|---|
| **R1** | G 用图内 min-max 归一化，跨图不可比 | **高** | 14 / 15 / 风险分档 |
| **R2** | T 分量在单窗口运行中结构性恒为 0 | **高** | R 上限 75 分、分档 |
| **R3** | `analysis_target=0` 的车辆仍进入风险报告 | 中 | 报告一致性 |

另有 2 项文档口径问题（D1、D2）与 1 项技术债（T1）。

---

## 1. 缺陷 R1：G 是图内相对分，不是绝对异常度

### 位置

- `src/graph/normalize.py::minmax_normalize`
- `src/graph/gae.py:351` → `"graph_anomaly_index": minmax_normalize(error)`
- `src/graph/anomaly.py:87` → `minmax_normalize(reconstruction_error)`（SVD 分支同样如此）

### 问题

节点异常分 `error` 经 `minmax_normalize` 后**每张图必然产生一个 1.0 和一个 0.0**：

```python
low, high = values.min(), values.max()
return (values - low) / (high - low)      # → 最大值恒为 1.0，最小值恒为 0.0
```

后果：**无论这一批车里有没有真正异常的车，排名第一的那辆 G 一定是 1.0。**

实测（`data/outputs/e2e/risk/risk_admin_in.json`，432 辆有图谱节点的车）：

```text
G 分布：min 0.000  max 1.000  mean 0.477
G >= 0.9 的车数：7
G == 1.0 的车数：1     ← 必然存在，与数据无关
```

而融合层 `beta=0.35` 直接吃这个值，于是 `R` 的图分量项恒有车拿满 `0.35`。

### 连带影响（已确认）

`src/web/server.py::build_headline` 用 `graph >= 0.8` 判定"主要因为它与其他车辆关系异常"。
本批 432 辆里 **26 辆（6.0%）** 触发该分支，但项目自身文档 `GOAL_STATUS.md 4.6` 已实测：
该机位「两两时间重叠仅 1/93096 对、最大同屏 2 辆」——**物理上不存在聚集型异常**。
即：控制台给这 26 辆中最靠前的几辆写的"关系明显异常"理由，是归一化造出来的，不是数据。

### 备注

`perf` 层面无问题（O(n) 向量运算）。这是**口径问题**，不是性能问题。

---

## 2. 缺陷 R2：T 分量在单窗口运行中恒为 0

### 位置

- `src/behavior/features/frequency_feature.py::FrequencyFeatureExtractor`
- `src/pipeline/batch_processor.py::_write_video_outputs`

### 链路追查

```text
behavior_vector.py:201  freq_count = frequency_extractor.get_count(vehicle_key)
                        └─ 计数器只在单段视频内累加，_frequency_map 每段 reset
batch_processor.py      vector["passage_id"] = f"{camera}_{segment}_{track_id}"
                        └─ 每辆车 passage_id 唯一 → get_count 恒为 1
time_frequency.py:54    excess = max(frequency - 1.0, 0.0)   → 恒为 0
                        frequency_component = 0
time_frequency.py:49    night_flag = 0（本批为白天）, night_ratio = 0
                        → T = 0.5*0 + 0.3*0 + 0.2*0 = 0.0
```

实测全部 444 条记录：

```text
freq_count_24h : min=1.0  max=1.0   （唯一值）
night_ratio    : min=0.0  max=0.0
time_freq_index: min=0.0  max=0.0   ← 全 0
```

### 关键发现：负责提供真实频次的聚合器**没有接进主链路**

`src/pipeline/daily_aggregator.py:313` 是唯一能算出「跨窗口真实频次」的实现
（按 `plate_hash` 统计 24h 通行次数），但全项目搜索确认：

```text
grep -rn "DailyAggregator" scripts/ src/  →  仅 src/pipeline/__init__.py 的 import
```

**没有任何脚本实例化它。** 因此 `freq_count_24h` 永远拿不到跨窗口真值。

### 数学后果

`gamma=0.25` 作用在恒为 0 的 T 上 → 死权重。于是：

```text
R = 0.4·A + 0.35·G + 0.25·0
R_max = 100 × (0.4×1.0 + 0.35×1.0) = 75.0
```

实测 A 上限 0.822、G 上限 1.0 → 理论可达上限 `100×(0.4×0.822+0.35×1.0) = 67.9`，
实测最高 R = 59.0。**80 分档（高风险）在结构上不可达**，与数据量无关。

---

## 3. 缺陷 R3：`analysis_target=0` 的车辆仍进入风险报告

### 位置

`src/risk/merge.py::apply_graph` → `src/risk/report.py`

### 问题

Phase1/2 已明确「货车不参与行政通道异常判断」（`analysis_target=0`），
`build_vehicle_graph` 也正确排除了它们（所以这 12 辆没有图谱节点）。
但风险报告阶段没有同样的过滤，这 12 辆仍被写入 `risk_admin_in.json` 并生成报告正文。

实测：

```text
analysis_target=0 进入风险报告：12 辆 / 444
vehicle_group：freight, two_wheeler
risk_score_100：全部 0.0，risk_level 全部 low
```

报告正文样例（读起来像"已排查，正常"）：

```text
### 车辆 admin_in:87（低风险）
**一句话结论**：该车于 08:06:15 — 08:06:17 通过行政通道，综合风险 0.0 分（低风险），
风险较低，未发现需要优先处置的异常，可作为常规抽查对象。
- 风险构成：行为 0.00 / 图 无数据 / 时间频率 0.0
```

危害：这 12 辆车**根本没有参与评分**（A=0、G=0、T=0，无 IR/规则计算），
却被陈述为「已评估、风险低」。这与 `merge.py` 里 `graph_available` 那套「无数据 ≠ 低风险」
的谨慎设计相矛盾——图谱层区分了「无数据」，风险报告层又把它抹平成「低风险」。

### 备注

`graph_available` 字段本身设计正确，且在 `report.py:201` 正确用于显示「图 无数据」。
缺的是**整条记录的准入判断**（`analysis_target=0` 应直接不进报告，或单列「非监管对象」）。

---

## 4. 文档口径问题

### D1：`PERFORMANCE.md` 与 `GOAL_STATUS.md 4.5` 归因冲突

| 文档 | 结论 |
|---|---|
| `PERFORMANCE.md:3` | 瓶颈在**解码 + ReID 推理 + GPU 同步**（25.6% / 9.6% / 12.8%），Python 逻辑未进前 8 |
| `GOAL_STATUS.md:158` | 瓶颈**不是解码（337 FPS）**，而是每帧 Python/ultralytics 调用开销 |

`README.md:330` 其实已经解释了 337 FPS 的来历——那是"纯解码循环"的孤立测量，
与流水线并发场景不同。但 `GOAL_STATUS.md 4.5` 直接引用了旧数字且未加限定，
与 `PERFORMANCE.md` 的 cProfile 实测结论正面对冲。

**建议**：`GOAL_STATUS.md 4.5` 改为引用 `PERFORMANCE.md` 的 cProfile 结论
（解码 25.6% 为最大单项），并注明 337 FPS 是孤立测量、不代表流水线。

### D2：测试数量口径

摘要中曾记 270，本次实测 **309 passed**（GAE 独立提交后未同步文档）。
`GOAL_STATUS.md` / `ACCEPTANCE_REPORT.md` 若有具体数字需一并核对。

---

## 5. 技术债 T1：聚类系数两份实现

`_clustering`（`gae.py:60`）与 `_clustering_coefficient`（`anomaly.py:20`）是逐行等价的重复实现
（同样的 `degree < 2 → 0`、同样的 `links / (d*(d-1)/2)`）。属于复制粘贴分叉风险，
建议收敛到 `normalize.py` 或新建 `features.py`。不影响正确性。

---

## 6. 复核通过项

以下常见缺陷点本轮专项检查**未发现问题**：

| 检查项 | 结果 |
|---|---|
| `TODO` / `FIXME` / `XXX` / `HACK` 残留 | 0 处 |
| 裸 `except:` | 0 处 |
| 可变默认参数 `def f(x=[])` | 0 处 |
| `src/` 层 `print` 调试残留 | 0 处 |
| 模块级可变全局状态 | 0 处 |
| `except Exception` 静默吞异常 | 15 处，**均为有意的 fallback**（模型缺失、OCR 失败、ReID 降级），且多数有注释说明降级语义 |
| GAE 训练/评分解耦 | ✅ `fit` 只吃参考图，`score` 可评任意节点数的图，迁移契约成立 |
| GAE 负采样 | ✅ `_sample_negatives` 等量负样本 + `loss_mask` 只算被遮边与负边 |
| GAE 持久化 | ✅ 存 `in_dim` + `node_feature_names`，`load` 用 `weights_only=True`（无 pickle 风险） |
| GAE 确定性 | ✅ `torch.manual_seed` + `np.random.RandomState` 双固定，有测试覆盖 |
| `apply_graph` 键不匹配告警 | ✅ 完全不匹配时 `logger.warning`，不静默 |
| `apply_alert_budget` 只裁不提拔 | ✅ 旧 bug 已修，有注释与测试 |
| `low_rank_reconstruction` 对称化 | ✅ 处理了 SVD 截断破坏对称性的问题 |

---

## 7. 建议的处置顺序

R1、R2 都属于「分数语义」缺陷，且**正好是此前确认要做的『风险分数标定』的核心内容**——
标定方案不只解决 80 分档不可达，还必须同时解决 R1（跨图不可比）与 R2（死权重）。

建议顺序：

1. **一起改 R1 + R2 + 标定**：单独做标定只能让分数"看起来"到 80 分，
   但 G 仍是图内相对分、T 仍是 0，属于把问题藏到标定层后面。
2. **R3 独立小改**：在 `build_risk_report.py` 汇总处加 `analysis_target` 过滤，
   或单列非监管对象区，改动面 < 10 行。
3. **D1 / D2 文档同步**：与代码改动一起提交，避免再产生新的口径分叉。
4. **T1 收敛**：可延后，纯整洁性。

本轮**未修改任何文件**，等确认后再动代码。
