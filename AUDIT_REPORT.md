# 代码审查整改报告

> **审查来源**：`223d1c79-67fd-45e2-a336-39898aa3d5c7`
> **审查基线**：HEAD `be9c9aa`（当时工作区只有 docx 变更，无代码 diff）
> **整改完成**：2026-11-16
> **结论**：8 条发现全部整改，7 条完全修复、1 条按语义保留并加了告警日志。
> 新增 11 个回归测试，全量 **266 passed**，CI 6 步全绿。

---

## 一、结论先说

审查的总体判断是「当前代码不能视为正确」，其中 3 条 P1 会在默认或常用配置下
造成**错误告警、静默丢帧、任务崩溃**。这个判断成立，我逐条复核后确认全部真存在。

| 项目 | 整改前 | 整改后 |
|---|---|---|
| 全量测试 | 255 passed | **266 passed**（+11 回归） |
| 存在的 P1 缺陷 | 3 | **0** |
| 存在的 P2 缺陷 | 4 | **0** |
| 存在的 P3 缺陷 | 1 | **0** |
| CI 6 步 | 全绿 | **全绿** |
| 端到端回归（demo 片段） | 7 车 / 1 中风险 / R=62.6 | **完全一致** |

**为什么"255 项通过"没能挡住这些缺陷**：它们都在测试覆盖之外——
预算逻辑只在 `--max-alert-rate` 时触发、预取只在消费端慢时丢帧、
分窗只在首窗为空时崩、告警率只在 `anomaly_only` 输出时算错。
本轮为每一条补了针对性用例。

---

## 二、整改总表

| # | 级别 | 问题 | 位置（整改后） | 修复方式 | 回归测试 |
|---|---|---|---|---|---|
| 1 | P1 | 告警预算把正常车提升为告警 | [alert_budget.py:40](src/anomaly/alert_budget.py#L40-L44) | 候选集从 `range(total)` 改为「原本就是告警的行」 | `test_alert_budget_never_promotes_normal_rows` |
| 2 | P1 | 预取队列满时静默丢帧 | [threaded_reader.py:42](src/utils/threaded_reader.py#L42-L58) | `queue.Full` 改为阻塞重试，不再 `break` | `test_threaded_reader_does_not_drop_frames_when_consumer_is_slow` |
| 3 | P1 | 空首窗导致分窗任务崩溃 | [run_windowed.py:50](scripts/run_windowed.py#L50-L59) / [:121](scripts/run_windowed.py#L121-L142) | 按 CSV 数据行数判断，样本不足跳过训练；训练失败可回退 | `test_count_vector_rows_treats_header_only_csv_as_empty` |
| 4 | P2 | 短片段 Transformer 训练抛异常中断 | [run_full_pipeline.py:183](scripts/run_full_pipeline.py#L183-L195) | 两处训练分别 try/except，失败只降级到规则 | `test_transformer_fit_raises_on_too_few_windows` |
| 5 | P2 | 仅含告警行时告警率算成 100% | [metrics.py:74](src/report/metrics.py#L74) | 分母固定用车辆总数 | `test_metrics_alert_rate_uses_vehicle_total` |
| 6 | P2 | 轨迹长度未使用单应矩阵 | [trajectory_extractor.py:120](src/trajectory/trajectory_extractor.py#L120) | 改用 `scene.trajectory_length_meters()` | `test_extractor_trajectory_length_uses_homography` |
| 7 | P2 | 同小时多段录像互相覆盖 | [batch_processor.py:36](src/pipeline/batch_processor.py#L36-L46) / [run_full_pipeline.py:149](scripts/run_full_pipeline.py#L149-L159) | 文件名加分段后缀，仅在同小时重复时启用 | `test_output_filename_suffix_separates_same_hour_segments` |
| 8 | P3 | 标注视频用整批最终轨迹状态 | [frame_pipeline.py:49](src/pipeline/frame_pipeline.py#L49-L70) / [batch_processor.py:248](src/pipeline/batch_processor.py#L248-L257) | 新增 `on_frame` 回调，按帧取轨迹快照 | `test_flush_batch_draws_per_frame_tracklets` |

---

## 三、逐条说明与前后对比

### 1. [P1] 告警预算会把正常车辆提升为告警

**原问题**：`scored_indices` 用 `range(total)` 覆盖全部行，按分数补满预算；
当实际告警数少于 `target_rate × total` 时，普通行被写成 `is_anomaly=1`，
且 `anomaly_score=0`、`alert_reason` 为空——凭空产生无证据告警。

**根因**：把"预算"理解成了"凑满 N 条"，而它的语义应该是"最多 N 条"。

**修复**：候选集改为 `alert_indices`（`is_anomaly=1` 的行），规则命中在其中优先保留。
预算从此**只能裁撤，不能提拔**。

**前后对比**（100 行 / 实际 2 条告警 / 预算 30）：

```text
修复前  is_anomaly=1 → 30 条   （凭空多出 28 条）
修复后  is_anomaly=1 →  2 条
```

---

### 2. [P1] 解码预取线程在队列满时静默丢弃剩余视频

**原问题**：`_run` 在 `queue.Full` 超时 0.5 秒后直接 `break` 结束线程；
`read()` 把剩余帧读完后返回 `(False, None)`，上层当成"视频结束"。

**根因**：把"消费端慢"误判为"生产结束"，且失败是静默的。

**修复**：队列满时**阻塞重试**（0.2 秒一轮，`stop()` 可中断），绝不丢帧；
另在 `read()` 的 30 秒超时分支加告警日志，避免"卡住"被静默当成结束。

**前后对比**（20 帧、队列 2、消费端第 3 帧停 1 秒）：

```text
修复前  读到 4/20 帧，丢失 16 帧
修复后  读到 20/20 帧，顺序连续
```

> 说明：30 秒超时后仍返回 `(False, None)`，这是既有 API 契约，本轮只加了日志、未改语义
> ——见第六节「遗留」。

---

### 3. [P1] 分窗模式遇到空的首个窗口会直接崩溃

**原问题**：空窗口仍会写出只有表头的 `vec_*.csv`，`find_vectors` 因此返回非 None，
`if not vector_csv` 永不触发；随后 `train_iforest.py` 在空 CSV 上失败，
`run_command` 抛 `RuntimeError` 终止整个任务。

**根因**：用"文件是否存在"代替了"是否有样本"。

**修复**：新增 `count_vector_rows()` 按数据行数判断；首窗样本 < 3 时跳过训练、
后续窗口只用规则与统计基线；训练子进程失败也只降级，不再中断。
调用异常检测时按模型文件是否存在动态拼参数。

---

### 4. [P2] 短片段仍可能因 Transformer 训练抛异常而中断

**原问题**：`can_train` 只检查 `target_rows >= 3`，但 `TrajectoryPredictionDetector.fit`
需要至少 4 个 `past_len + future_len` 窗口，3 条短轨迹会抛 `ValueError`，
整条链路直接失败——与"短片段跳过模型"的注释相反。

**修复**：IF 与 Transformer 的训练分别 try/except，失败记录警告并降级到规则引擎。
补充了一个测试明确 `fit()` 在窗口不足时确实会抛 `ValueError`，把这个前提固化下来。

---

### 5. [P2] 仅含告警行的 alerts CSV 会把告警率算成 100%

**原问题**：`denominator = len(alerts) or len(vectors)`。`run_anomaly_detection.py`
默认 `anomaly_only=True`，此时 `alerts.csv` 只有异常行，分母等于告警数 → 恒为 100%。

**影响**：`OPERATIONS.md` 与 `ci.ps1` 正是用它判断 5%–15% 告警率。

**修复**：`denominator = len(vectors) or len(alerts)`，分母始终是车辆总数。

**前后对比**（10 辆车 / 2 条告警）：

```text
修复前  100.0%   （分母=2）
修复后   20.0%   （分母=10）
```

---

### 6. [P2] 轨迹长度字段未使用新增的单应矩阵

**原问题**：`build_trajectory_summary` 先累加像素长度，再用 `pixel_to_world()` 换算；
后者只用常数 `px_per_meter`，完全忽略 `homography`。标定完成后
`trajectory_length_m` 仍然算错。

**修复**：改用 `self.scene.trajectory_length_meters(centers)`，逐点走地面距离。

**验证**：单应矩阵 `x/0.1, y/0.2` + `px_per_meter=10`，同一组点：

```text
pixel_to_world 旧路径  12.73 m
trajectory_length_meters 新路径  30.00 m
```

回归测试直接走 `TrajectoryExtractor.build_trajectory_summary`，断言写出 30.00 m。

---

### 7. [P2] 同一小时的多段录像会互相覆盖输出

**原问题**：输出名只含 `camera + date + hour`，同小时第二段会覆盖第一段的
`vec_*.csv` 与 `events_*.csv`，前一段车辆静默丢失（`run_batch_day.py` 早已用
suffix 规避，`run_full_pipeline.py` 没有）。

**修复**：`process_video` 新增 `name_suffix` 参数；文件名构造抽成
`build_vector_filename` / `build_event_filename` 两个函数便于测试；
`run_full_pipeline` 先按 `(日期, 小时)` 分组，**只有重复时才加 `_01`/`_02` 后缀**，
单段录像的文件名保持不变（避免影响既有产物与文档）。

---

### 8. [P3] 批处理输出的标注视频会使用错误的轨迹状态

**原问题**：`_flush_batch` 先处理完整批 16 帧，再对每帧调
`tracker.get_active_tracklets()`，拿到的是整批结束时的状态——
前 15 帧会被画上最后一帧的位置，甚至出现本批稍后才创建的轨迹。

**修复**：`process_frames` 新增可选回调 `on_frame(frame, active_tracklets)`，
在**每帧处理完的当下**取快照；`_flush_batch` 用它写标注视频。
未开启 `output_video` 时行为完全不变。

---

## 四、回归测试清单

新增 `tests/test_audit_fixes.py`（11 个用例，5.9 秒）：

| 用例 | 对应发现 |
|---|---|
| `test_alert_budget_never_promotes_normal_rows` | 1 |
| `test_alert_budget_still_caps_when_over_budget` | 1（不回归：超额仍要裁撤） |
| `test_alert_budget_keeps_rule_hits_first` | 1（不回归：规则优先） |
| `test_threaded_reader_does_not_drop_frames_when_consumer_is_slow` | 2 |
| `test_count_vector_rows_treats_header_only_csv_as_empty` | 3 |
| `test_transformer_fit_raises_on_too_few_windows` | 4 |
| `test_metrics_alert_rate_uses_vehicle_total` | 5 |
| `test_trajectory_length_uses_homography` | 6（几何层） |
| `test_extractor_trajectory_length_uses_homography` | 6（真正出错的入口） |
| `test_output_filename_suffix_separates_same_hour_segments` | 7 |
| `test_flush_batch_draws_per_frame_tracklets` | 8 |

此外为便于测试做了一处无行为变更的重构：把输出文件名构造从
`process_video` 内联字符串抽成模块级函数（`build_vector_filename` /
`build_event_filename`），生成结果与原来逐字符一致。

### 4.1 这些测试到底测的是什么规则

它们锁定的不是"物理真理"，而是**从审查意见与既有文档推导出的行为约定**。
按来源分三类：

| 类型 | 用例 | 规则来源 | 强度 |
|---|---|---|---|
| **契约型** | 预取不丢帧、告警率分母、轨迹长度、文件命名、标注视频、表头不算样本 | 模块 docstring / 指标定义 / 文件系统语义 | 强——违反即 bug |
| **业务语义** | 预算不提拔、超额仍裁撤、规则优先 | 审查意见 + 函数注释里"只保留预算内的告警" | 中——是**选择**，需业务确认 |
| **角色记录** | `test_transformer_fit_raises_on_too_few_windows` | 描述现状（`fit` 会抛异常），用来解释为何需要 try/except | 弱——不判断对错 |

### 4.2 测试有效性验证：在旧代码上跑一遍

写完测试不等于测试有效。用 `git worktree` 检出 HEAD `be9c9aa`（整改前），
把同一份测试文件放进去跑，看哪些会**变红**：

| 用例 | 旧代码结果 | 结论 |
|---|---|---|
| `never_promotes_normal_rows` | **FAILED** | 能抓到 P1-1 |
| `still_caps_when_over_budget` | PASSED | 防回归（旧代码本来就对） |
| `keeps_rule_hits_first` | PASSED | 防回归 |
| `threaded_reader_..._slow` | **FAILED** | 能抓到 P1-2（**改过一轮才成立**，见下） |
| `count_vector_rows_...` | **FAILED** | 能抓到 P1-3 |
| `transformer_fit_raises` | PASSED | 角色记录，不区分版本 |
| `metrics_alert_rate_...` | **FAILED** | 能抓到 P2-2 |
| `trajectory_length_uses_homography` | PASSED | 几何层旧代码已有该方法，不区分 |
| `extractor_trajectory_length_...` | **FAILED** | 能抓到 P2-3 |
| `output_filename_suffix_...` | **FAILED** | 能抓到 P2-4 |
| `flush_batch_draws_per_frame_...` | **FAILED** | 能抓到 P3-1 |

**7 红 / 4 绿**。4 条绿的都不是缺陷捕获器，而是"防过度修复"和"记录现状"。

> **这个实验还抓到我自己一条假绿测试**：`threaded_reader` 第一版用 20ms 轮询，
> 触发不了生产者 0.5 秒的 put 超时，旧代码也能读完 20 帧 —— 测试在新旧代码上都绿，
> 等于没测。改成审查报告给出的"消费端停 1 秒"后才在旧代码上变红。

---

## 五、端到端回归

```text
demo 片段 3000 帧全链路
  车辆 7 / 中风险 1 / 高风险 0 / 最高 R=62.6
  与整改前基线完全一致（无行为回归）

pytest -q          266 passed
scripts/ci.ps1     CI PASSED（6 步）
```

---

## 六、遗留与边界（未在本轮改动）

| 项 | 说明 | 为什么没改 |
|---|---|---|
| 预取 30 秒超时仍返回 `(False, None)` | 卡住的流会被当成"视频结束"；已加告警日志 | 改语义会影响所有调用方，需要单独设计"超时即报错"的契约 |
| 分窗首窗为空时直接跳过该窗 | 不会回填上一窗口的模型 | 首窗无车时也确实没有可训练样本；跳窗更符合"不制造结论"的原则 |
| 同小时后缀只在 `run_full_pipeline` 里自动分组 | 直接调 `BatchProcessor.process_video` 需自行传 `name_suffix` | 保持库层不做隐式命名推断 |
| 标注视频逐帧回调带来轻微开销 | 仅 `output_video=True` 时启用 | 该功能本就是调试用途，正确性优先 |

---

## 七、复现

```powershell
# 回归测试
.venv\python.exe -m pytest tests\test_audit_fixes.py -v

# 全量 + CI
.venv\python.exe -m pytest -q
.\scripts\ci.ps1

# 端到端（demo 片段，约 30 秒）
.venv\python.exe scripts\run_full_pipeline.py `
  --input data\outputs\demo\h264 --camera admin_in `
  --output data\outputs\audit_rerun --max-frames 3000
```

---

## 八、结论

审查提出的 8 条问题全部成立，均已整改并配了针对性回归测试。
其中 3 条 P1（错误告警、静默丢帧、任务崩溃）都属于"默认配置下会发生"，
不影响历史产物（本轮端到端回归与整改前完全一致），但会影响日常使用与验收，
建议把 `tests/test_audit_fixes.py` 纳入 CI 作为固定门禁。
