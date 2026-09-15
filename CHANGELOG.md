# 项目变更日志

本文档用于持续记录车辆行为分析系统的代码、配置、数据验证和已知限制。

## 维护规则

1. 每次修改代码、配置或数据处理逻辑时，必须同步更新本文档。
2. 每条记录应说明修改原因、涉及文件、具体改动、验证方式和已知限制。
3. 运行生成的临时 CSV、截图和本地开发工具配置不作为正式变更提交。
4. 对行为向量定义、风险评分逻辑或摄像头标定的修改，应明确说明对历史数据兼容性的影响。

## 当前状态

- 当前阶段：车辆检测、跟踪、轨迹提取和行为向量可靠性优化。
- 已具备能力：YOLOv8 车辆检测、ROI 过滤、简化 ByteTrack 跟踪、轨迹平滑、速度和停留分析、路径特征、时间特征、聚集特征、颜色识别、启发式车型识别、车牌区域检测、OCR 接口、ReID 接口、CSV 输出和日聚合骨架。
- 尚未完成：中央路北摄像头专属 ROI、进出线和透视标定；稳定车牌识别；正式 ReID 模型；异常检测基线训练；关系图谱和风险融合评分。

## 2026-09-15

### 新增代码结构说明文档和核心中文注释

#### 修改原因

项目已经包含 Phase 1/2/3 多个模块，后续维护需要清晰了解每个目录和文件的职责，同时关键业务逻辑需要中文注释辅助理解。

#### 具体改动

- 新增 `CODE_STRUCTURE.md`，按目录说明每个模块、每个文件的作用和主数据流。
- 在批处理、帧流水线、跟踪器、轨迹提取、行为向量、车型分组和 Phase 3 基线/异常检测代码中补充中文注释。
- 注释重点覆盖车辆出现到消失判定、行政通道普通车分析范围、可靠性字段过滤、向量/事件/告警输出边界。

#### 验证

```text
compileall: OK
37 passed
```
## 2026-09-14

### 限定行政通道普通车辆分析范围

#### 修改原因

项目暂不使用业务数据，当前判断对象限定为行政通道普通车辆；货运通道车辆不纳入异常研判。但模型仍必须识别货车，避免货车轨迹污染普通车辆基线和告警结果。

#### 具体改动

- `VehicleClassifier` 增加车辆分组能力，将 `sedan/suv/mpv/car` 归为 `ordinary`，将 `truck` 归为 `freight`。
- 行为向量新增 `vehicle_group` 和 `analysis_target` 字段。
- 向量 CSV、通行事件 CSV 和告警 CSV 均输出车辆分组和分析对象标记。
- Phase 3 基线构建默认只使用 `ordinary` 车辆，可通过 `--target-groups` 调整。
- Phase 3 异常检测默认跳过 `analysis_target=0` 的货车等非研判对象。
- 新增回归测试覆盖车型分组、CSV 字段和货车过滤。

#### 验证

```text
compileall: OK
37 passed
Phase3 CLI 临时样本：混入货车时，基线只使用普通车辆；货车保留输出但不触发行政通道告警。
```

#### 已知限制

- 当前无专用车型分类模型时，货车/普通车区分主要依赖 YOLO 粗类别和检测框几何启发式；远景、遮挡和斜向车身可能误分。
- 要稳定区分行政通道普通车和货车，后续应采集本地摄像头样本，训练或微调轻量车型分类模型。
### 搭建 Phase 3 基线和告警框架

#### 修改原因

Phase 1/2 已能稳定输出行为向量和车辆通行事件，下一步需要基于多天历史数据建立正常行为范围，并对新数据输出可复核的异常告警。Phase 3 框架应复用既有 CSV，不重复执行视频检测。

#### 具体改动

- 新增 `src/anomaly/baseline.py`，提供正常行为基线构建、保存、加载和 Z-score 评分。
- 升级 `src/anomaly/anomaly_detector.py`，支持规则告警和统计基线联合判定。
- 升级 `src/anomaly/rule_engine.py`，按可靠性字段跳过未标定速度和路径规则。
- 新增 `src/output/alert_writer.py`，输出可复核的告警 CSV。
- 重写 `scripts/build_baseline.py`，生成标准 Phase 3 JSON 基线文件。
- 新增 `scripts/run_anomaly_detection.py`，从行为向量 CSV 离线生成告警 CSV。
- 新增 `config/anomaly_config.yaml`，记录 Phase 3 基线、规则和告警输出默认配置。
- 新增 `tests/test_phase3_anomaly.py`，覆盖基线构建、异常评分、可靠性过滤和告警输出。
- README 增加 Phase 3 基线构建和告警检测使用说明。

#### 验证

```text
compileall: OK
33 passed
Phase3 CLI 临时样本：build_baseline.py 成功生成 JSON 基线，run_anomaly_detection.py 成功输出 1 条告警
```

#### 已知限制

- 当前统计基线是轻量 JSON 基线，适合先做离线筛查；后续可在样本充足后接入 IsolationForest、LOF 或时序模型。
- 正常行为基线质量依赖多天真实数据，短样本只能验证流程，不能代表正式阈值。
- 告警结果目前是离线 CSV，还未接入可视化复核和业务关系数据。
### 增加车辆通行事件输出

#### 修改原因

Phase 1/2 的行为向量适合做特征分析，但后续跨摄像头轨迹拼接、规则告警和业务风险融合更需要“一辆车一次通行一条记录”的事件表。本次新增事件输出，作为 Phase 2 到 Phase 3/4 的衔接数据。

#### 具体改动

- 新增 `src/output/event_writer.py`，从已完成行为向量生成 `vehicle_passage` 通行事件。
- 批处理完成后同步输出 `data/outputs/events/<camera_id>/<date>/events_<camera_id>_<date>_<hour>.csv`。
- `config/default.yaml` 增加 `output.save_events` 开关，默认启用。
- 新增事件输出回归测试，覆盖事件字段映射和空事件表头输出。
- README 增加通行事件输出路径和用途说明。

#### 验证

```text
compileall: OK
28 passed
真实视频 1200 帧短样本：向量 CSV 和事件 CSV 均已生成，当前样本段未形成完整轨迹，因此事件表仅含表头
```
### 增加空间语义配置骨架

#### 修改原因

后续要从单摄像头行为向量升级到多摄像头路径分析，需要先建立园区、区域、道路、摄像头和拓扑关系的配置结构。否则道路摄像头数据只能作为独立视频处理，无法判断路径是否合理，也无法设置跨摄像头通行时间窗口。

#### 具体改动

- 新增 `config/site/site.yaml`：园区基础信息。
- 新增 `config/site/zones.yaml`：区域和卡口定义。
- 新增 `config/site/roads.yaml`：道路和车道定义。
- 新增 `config/site/cameras.yaml`：摄像头到道路/区域的绑定。
- 新增 `config/site/topology.yaml`：跨摄像头/道路边关系骨架。
- 新增 `config/site/routes.yaml`：业务合法路线骨架。
- 新增 `src/site/topology.py`，提供配置读取和引用校验。
- 新增 `tests/test_site_topology.py`，覆盖默认配置加载和缺失引用报错。
- `.gitignore` 增加 `data/outputs/`，避免提交本地批处理结果。

#### 验证

```text
compileall: OK
26 passed
```

#### 数据使用说明

- Phase 1/2 只需要每个摄像头少量短样本，用于画面质量、ROI、跟踪和属性提取验证。
- Phase 3 需要多天历史道路摄像头数据，用于训练正常行为基线。
- Phase 4 需要多摄像头连续视频或结构化事件，用于跨摄像头全域轨迹拼接。
### 稳定 Phase 2 车辆属性提取

#### 修改原因

车牌 OCR 在全景监控视频中会出现低置信误识别；同时 ReID 依赖外部 OSNet 权重，缺失权重时无法输出外观特征。为让 Phase 2 在没有专用模型权重时也能稳定输出，本次增加车牌检测模型缺失回退、OCR 置信度过滤和无模型 ReID fallback。

#### 具体改动

- `PlateDetector` 支持专用 YOLO 车牌模型路径、检测置信度和最小车牌宽度配置。
- 专用车牌模型缺失时自动回退到形态学检测，不中断流水线。
- `PlateOCR` 增加 `min_confidence`，低于阈值时清空 `plate_number`，但保留置信度证据。
- CSV 新增 `plate_confidence` 字段。
- `ReIDExtractor` 增加无模型 fallback：HSV 颜色直方图 + 车辆框形状特征，输出 74 维归一化 embedding。
- `model_config.yaml` 启用 ReID fallback，并配置预留车牌 YOLO 权重路径 `models/license_plate_yolov8.pt`。
- 增加回归测试覆盖车牌模型缺失回退、ReID fallback、OCR 低置信过滤和 CSV 字段。

#### 验证

```text
compileall: OK
24 passed
短样本真实视频 1200 帧：输出 3 条车辆向量，reid_embedding 已写入，低置信 OCR 被过滤。
```

#### 已知限制

- 当前仓库还没有 `models/license_plate_yolov8.pt`，车牌检测仍会回退到形态学方法。
- 全景视频中车牌较小时，OCR 结果只能作为低置信候选，不应作为正式身份依据。
- 无模型 ReID fallback 适合先做跨段候选召回，不等同于正式车辆 ReID 模型。
## 2026-06-01

### 重写 README 使用说明

提交：本次提交

#### 修改原因

原 README 仍停留在早期“YOLOv8 + ByteTrack 车辆异常检测系统”说明阶段，未反映当前项目已经形成的行为向量可靠性字段、几何配置等级、真实视频测试结论、海康边缘主机接入方向和空间语义拓扑建设路线。

#### 具体改动

- 重写项目定位，明确当前系统是“数据采集层和行为向量化层”，还不是完整业务违规自动判定系统。
- 增加四层总体架构说明：
  - 视频行为向量化
  - 行为异常检测
  - 空间语义与摄像头拓扑
  - 风险融合与解释
- 更新当前能力和当前限制。
- 更新项目结构说明。
- 补充单视频运行、批量运行、输出路径和参数说明。
- 更新行为向量 CSV 字段说明，加入几何可靠性字段。
- 增加摄像头配置等级说明：
  - `none`
  - `roi`
  - `calibrated`
- 增加真实视频测试结论。
- 增加海康超脑边缘主机接入方向。
- 增加空间语义和拓扑建设方向。
- 增加测试命令和当前测试覆盖重点。
- 明确后续优先事项。

#### 涉及文件

- `README.md`
- `CHANGELOG.md`

#### 影响

- README 现在作为项目入口文档使用。
- 对外说明更明确地区分了“当前已实现能力”和“最终科研目标”。

### 增加几何配置等级与可靠性标记

提交：本次提交

#### 修改原因

后续会接入多个摄像头，不能要求每个摄像头一开始都完成精细标定；但如果完全不区分配置能力，未标定摄像头产生的速度、路径偏离和空间聚集特征会污染异常检测。为支持“先接入、后标定”的部署方式，本次增加几何能力分级和字段可靠性标记。

#### 具体改动

- 在 `SceneGeometry` 中增加 `geometry_level`：

  ```text
  none       未配置 ROI，几何相关字段不可靠
  roi        已配置 ROI，但未完成完整进出线和参考标定
  calibrated 已配置 ROI、进出线和参考距离，几何相关字段可参与风险评分
  ```

- 在行为向量中新增可靠性字段：

  ```text
  geometry_level
  speed_reliable
  path_reliable
  aggregation_reliable
  passage_reliable
  ```

- `BatchProcessor` 根据摄像头配置自动推断几何等级。
- CSV 输出和日聚合结果保留上述可靠性字段。
- 新增测试覆盖未配置、ROI 配置和完整标定三种模式。

#### 涉及文件

- `src/trajectory/scene_geometry.py`
- `src/pipeline/batch_processor.py`
- `src/behavior/behavior_vector.py`
- `src/output/vector_writer.py`
- `src/pipeline/daily_aggregator.py`
- `tests/test_behavior_reliability.py`

#### 验证

```text
20 passed
compileall: OK
git diff --check: OK
```

#### 影响

- 未标定摄像头仍可输出检测、跟踪、时间、频次、颜色和车型等基础字段。
- 下游异常检测应只使用 `*_reliable = 1` 的几何字段参与速度、路径偏离和聚集类风险评分。
- 这会改变 CSV 表头；旧 CSV 与新 CSV 在字段数量上不完全一致。

### 根据实测优化视频解析与轨迹关联

提交：`30dbf6e`

#### 修改原因

使用 `F:\test_videos\centerlode_north` 下的真实道路监控视频进行测试后，发现下载视频文件名格式与原解析器不一致，且当前简化 IoU 跟踪器使用 `0.8` 阈值时容易将同一车辆拆分为多个轨迹。

#### 具体改动

- 兼容监控下载文件名中的连续时间戳：

  ```text
  Download_CH05..._20260422080046_20260422093204_...
  ```

- 正确解析录像日期、开始时间和输出小时目录。
- 将简化 IoU 关联阈值从 `0.8` 调整为 `0.3`。
- 新增检测框发生合理抖动后仍保持同一轨迹的回归测试。
- 新增下载视频时间戳解析测试。
- 将 `.pytest_cache/` 和 `data/validation/` 加入 `.gitignore`。

#### 涉及文件

- `.gitignore`
- `config/default.yaml`
- `src/pipeline/batch_processor.py`
- `src/tracking/bytetrack_tracker.py`
- `tests/test_behavior_reliability.py`

#### 实测结果

- 测试视频共 `13` 段，覆盖 `2026-04-22 00:00:00` 至 `19:43:21`。
- 前 `11` 段可以读取，规格为 `1280x720 / 25 FPS`。
- 最后 `2` 段损坏，FFmpeg 报错：`moov atom not found`。
- `08:00` 时段约 `60` 秒样本：

  ```text
  输出轨迹数：28 -> 18
  最大局部 ID：69 -> 17
  ```

- `15:00` 时段约 `60` 秒样本：

  ```text
  输出轨迹数：32 -> 15
  轨迹时长中位数：2.18s -> 14.12s
  零距离聚集记录：0
  ```

#### 验证

```text
16 passed
compileall: OK
git diff --check: OK
```

#### 已知限制

- 中央路北摄像头尚未配置专属 ROI，路边停放车辆仍会进入分析范围。
- 当前使用单一 `px_per_meter`，未考虑道路透视关系，速度绝对值不能直接用于正式风险评分。

### 提升行为向量可靠性

提交：`633a638`

#### 修改原因

代码审查发现轨迹丢失恢复、跨视频状态隔离、跳帧速度、进出时间、聚集行为和日聚合逻辑会影响行为向量可信度。

#### 具体改动

- 允许短暂丢失轨迹继续参与关联，减少同一车辆被拆分为多个 `track_id`。
- 每段视频开始时清理轨迹、帧号、频次统计和已完成向量，同时保留模型实例。
- 使用真实时间戳计算速度，支持跳帧处理。
- 修复进出通道时间使用全局帧号索引轨迹内部时间数组的问题。
- 聚集特征排除当前车辆自身，并按相同帧号对齐其他车辆位置。
- OCR 完成后再登记高频通行特征，使新识别车牌立即参与统计。
- CSV 增加 `global_vehicle_id`、`segment_count` 和 `reid_embedding`。
- 日聚合使用“视频段 + 局部 ID”作为轨迹键。
- 跨视频合并必须存在相同车牌哈希或可靠 ReID 证据。
- 修复日聚合读取 CSV 后字符串参与数值计算的问题。
- 新增行为特征可靠性回归测试。

#### 涉及文件

- `src/behavior/behavior_vector.py`
- `src/behavior/features/aggregation_feature.py`
- `src/behavior/features/frequency_feature.py`
- `src/behavior/features/speed_feature.py`
- `src/output/vector_writer.py`
- `src/pipeline/batch_processor.py`
- `src/pipeline/daily_aggregator.py`
- `src/pipeline/frame_pipeline.py`
- `src/tracking/bytetrack_tracker.py`
- `src/trajectory/trajectory_extractor.py`
- `tests/test_behavior_reliability.py`

#### 验证

```text
14 passed
compileall: OK
git diff --check: OK
```

## 2026-05-25 至 2026-06-01：早期开发记录

以下内容根据 Git 提交历史整理。早期提交记录较简略，因此保留提交摘要和主要作用。

| 日期 | 提交 | 摘要 | 主要内容 |
|---|---|---|---|
| 2026-05-25 | `82251ef` | 初始化车辆追踪系统项目 | 建立检测、跟踪、轨迹、行为特征、异常检测骨架、批处理脚本和测试目录。 |
| 2026-05-26 | `eadee68` | 添加了模型 | 加入 YOLOv8 模型权重文件。 |
| 2026-05-28 | `5937d05` | 创建 `.gitignore` | 排除虚拟环境、Python 缓存和 IDE 配置目录。 |
| 2026-05-29 | `7ef08bf` | 标定了初始帧 | 新增 `cam01` 摄像头配置、测试帧和初始 CSV。 |
| 2026-05-29 | `96e1b1e` | 修复轨迹提取管线 | 完善 ByteTrack 匹配和自动进出判定。 |
| 2026-05-29 | `c50cee1` | 第一次跑通后的 CSV | 保存首次完整运行结果。 |
| 2026-05-29 | `8fb8775` | OCR 代码运行验证 | 确认全景画面车牌像素不足，形态学定位和 OCR 效果受限。 |
| 2026-06-01 | `0c5dcec` | 车辆属性验证 | 确认颜色识别、车型粗分类、时间和轨迹长度等字段状态。 |
| 2026-06-01 | `e9ad1f4` | Phase 2：车辆属性特征提取 | 优化颜色、车型、车牌 OCR 和车辆裁剪逻辑。 |

## 后续优先事项

1. 为中央路北摄像头配置专属 ROI，排除道路两侧长期停放车辆。
2. 增加进出线并验证车辆通行方向。
3. 使用透视变换或分区域标定替代单一 `px_per_meter`。
4. 用较长时间窗口重新生成行为向量，并检查轨迹碎片率、速度分布和聚集指数。
5. 建立正常行为基线，再接入无监督异常检测。
6. 在车辆、司机、企业和时间维度形成结构化关联数据，为关系图谱和风险融合评分提供输入。
