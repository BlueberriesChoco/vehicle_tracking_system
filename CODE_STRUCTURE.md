# 代码结构说明

本文档按项目目录说明每个模块和文件的作用，帮助后续维护 Phase 1/2/3 的视频处理、车辆属性提取、行为向量、基线告警和园区拓扑能力。

## 总体数据流

```text
视频文件
  -> scripts/run_pipeline.py
  -> BatchProcessor
  -> FramePipeline
  -> VehicleDetector
  -> ROIFilter
  -> ByteTrackTracker
  -> TrajectoryExtractor
  -> BehaviorVectorizer
  -> VectorWriter / EventWriter
  -> scripts/build_baseline.py
  -> BaselineBuilder
  -> scripts/run_anomaly_detection.py
  -> AnomalyDetector / AlertWriter
```

当前判断对象限定为行政通道普通车辆：`vehicle_group = ordinary` 且 `analysis_target = 1` 的记录进入 Phase 3 基线和告警；货车等非行政通道研判对象保留输出，但默认不参与异常判断。

## 根目录文件

| 文件 | 作用 |
|---|---|
| `README.md` | 项目说明、运行方式、输出字段、Phase 3 使用说明和后续计划。 |
| `CHANGELOG.md` | 中文变更日志，记录代码、配置、数据验证和已知限制。 |
| `requirements.txt` | Python 依赖清单。当前核心依赖包括 YOLO、OpenCV、NumPy、SciPy、PyYAML、Torch、pytest。 |
| `config.yaml` | 早期或兼容配置文件，主要配置以 `config/` 目录为准。 |
| `yolov8s.pt` | 根目录下的 YOLO 权重副本；正式代码默认使用 `models/yolov8s.pt`。 |

## `scripts/` 命令入口

| 文件 | 作用 |
|---|---|
| `scripts/run_pipeline.py` | 单视频处理入口。读取视频、摄像头配置和全局配置，调用 `BatchProcessor` 输出行为向量和通行事件。 |
| `scripts/run_batch.py` | 批量处理入口。按目录/日期处理多段视频。 |
| `scripts/calibrate_camera.py` | 交互式摄像头标定工具，用于标 ROI、进出线和参考距离，输出到 `config/camera/<camera_id>.yaml`。 |
| `scripts/build_baseline.py` | Phase 3 基线构建入口。读取历史 `vec_*.csv`，默认只使用普通车辆生成 JSON 正常行为基线。 |
| `scripts/train_iforest.py` | Phase 3 无监督检测训练入口。用正常样本训练 Isolation Forest，输出模型 JSON。 |
| `scripts/train_transformer.py` | 训练 Transformer 轨迹预测模型，输出 `.pt` + `.json`。 |
| `scripts/run_anomaly_detection.py` | Phase 3 告警检测入口。读取向量 CSV、基线 JSON 和可选 Isolation Forest 模型，输出告警 CSV。 |
| `scripts/build_graph.py` | 研究要点三入口。构建车辆关系图谱，输出节点社区、图异常指数 G、边和社区摘要 JSON。 |
| `scripts/extract_mbf.py` | 把现场 `.mbf` 录像转换为可直接解码的 `.h264` 裸流。 |
| `scripts/calibrate_homography.py` | 由地面控制点求解单应矩阵，可写回摄像头配置。 |
| `scripts/render_camera_config.py` | 把 ROI / 进出线画到真实帧，目视校验标定。 |
| `scripts/stitch_cameras.py` | 跨摄像头同车关联，输出匹配对与全局车辆 ID。 |
| `scripts/run_multi_camera.py` | 多机位编排：逐机位检测/构图 + 跨机位拼接，输出全局 ID 汇总。 |
| `scripts/update_models.py` | 模型增量更新：用新数据重拟合 IF、微调 Transformer，记录耗时。 |
| `scripts/calibrate_alert_rate.py` | 告警率自适应标定：搜索满足目标告警率的最大污染率。 |
| `scripts/eval_with_labels.py` | 有标注时的评估：precision/recall/F1、ROC-AUC、高风险精确率。 |
| `scripts/soak_test.py` | 浸泡测试：同进程多轮处理，检查内存泄漏与 FPS 漂移。 |
| `scripts/run_windowed.py` | 分窗处理长视频，单车上报延迟与视频总长解耦。 |
| `scripts/audit_rules.py` | 规则触发率审计：命中数、可靠性标记、字段分布。 |
| `scripts/report_metrics.py` | 从全链路产物汇总关键指标为 Markdown，避免文档数字过期。 |
| `scripts/export_gang_case.py` | 从图谱中导出最值得核查的社区，生成团伙案例报告。 |
| `scripts/build_demo_pack.py` | 一键生成演示包：单车报告 + 团伙案例 + 指标总览。 |
| `scripts/delivery_check.py` | 交付门禁：文件齐备、YAML 可解析、文档引用有效、无临时目录。 |
| `scripts/start.ps1` | Windows 一键启动全链路。 |
| `scripts/smoke_test.py` | 端到端冒烟测试：串联五个 CLI 并校验输出。 |
| `scripts/ci.ps1` | 本地 CI：单测 + 冒烟 + 规则审计 + 指标汇总 + 演示包 + 交付清单（共 6 步）。 |
| `scripts/verify_plate_ocr.py` | 车牌识别分环节验证：分别测「OCR 上限」（给定精确车牌区域）、「端到端」（车辆裁剪 → 检测 → OCR）、「真实入口」（`recognize_track` 多帧投票）。用于区分瓶颈在检测还是 OCR，避免只看总数得出无效结论。 |
| `Dockerfile` / `docker-compose.yml` | 容器化部署入口，详见 `deploy/README.md`。 |
| `scripts/build_risk_report.py` | 研究要点四入口。融合 A/G/T 得到风险 R，输出证据链与 Markdown 排查报告。 |
| `scripts/run_full_pipeline.py` | 一键端到端原型：视频 → 向量 → 告警 → 图谱 → 风险报告。 |
| `scripts/run_batch_day.py` | 全天批量处理：遍历 `.mbf`、转 h264、跑向量流水线，支持断点续跑。 |
| `scripts/eval_system.py` | 技术指标自评：告警率、证据可追溯率、图谱规模、风险分布。 |
| `scripts/benchmark_throughput.py` | 吞吐基准：检测/端到端 FPS，推算可支撑视频路数。 |

## `config/` 配置目录

| 文件 | 作用 |
|---|---|
| `config/default.yaml` | 全局默认配置，包括检测、跟踪、轨迹、行为、输出和可视化开关。 |
| `config/model_config.yaml` | Phase 2 模型配置，控制颜色、车型、车牌、OCR、ReID 等提取器。 |
| `config/behavior_config.yaml` | 行为特征配置，声明速度、停留、路径、时间、频次、聚集等特征维度。 |
| `config/anomaly_config.yaml` | Phase 3 配置，声明行政通道分析范围、基线参数、规则阈值和告警输出目录。 |
| `config/camera/cam01.yaml` | `cam01` 摄像头 ROI、进出线和标定配置。 |
| `config/camera/camera_01.yaml` | 早期摄像头配置样例。 |

## `config/site/` 园区空间语义配置

| 文件 | 作用 |
|---|---|
| `config/site/site.yaml` | 园区基础信息。 |
| `config/site/zones.yaml` | 区域、卡口、功能区等语义区域定义。 |
| `config/site/roads.yaml` | 道路、车道和相邻道路定义。 |
| `config/site/cameras.yaml` | 摄像头与道路/区域/朝向的绑定关系。 |
| `config/site/topology.yaml` | 摄像头之间或道路之间的拓扑边，包含跨摄像头通行时间窗口。 |
| `config/site/routes.yaml` | 合法路线或业务路线骨架。当前未接入业务数据时主要作为后续预留。 |

## `src/detection/` 检测模块

| 文件 | 作用 |
|---|---|
| `src/detection/detector.py` | YOLOv8 车辆检测封装。输出 bbox、置信度、COCO 类别和车辆粗类别 `car/motorcycle/bus/truck`。 |
| `src/detection/roi_filter.py` | ROI 过滤器。根据多边形区域过滤检测框，减少无关道路、停车区和背景误检。 |
| `src/detection/__init__.py` | detection 包导出文件。 |

## `src/tracking/` 跟踪模块

| 文件 | 作用 |
|---|---|
| `src/tracking/bytetrack_tracker.py` | 简化 ByteTrack 实现。使用高分检测优先匹配、低分检测二次关联、IoU 和简化卡尔曼预测维护轨迹 ID。 |
| `src/tracking/tracklet.py` | 单条轨迹数据结构。保存检测框、中心点、时间戳、进入/离开状态、最佳裁剪、最大裁剪、车牌和 ReID 特征。 |
| `src/tracking/__init__.py` | tracking 包导出文件。 |

## `src/trajectory/` 轨迹模块

| 文件 | 作用 |
|---|---|
| `src/trajectory/scene_geometry.py` | 场景几何和可靠性判断。管理进出线、像素/米换算、轨迹长度、点线距离，以及 `geometry_level` 和 `*_reliable` 字段。 |
| `src/trajectory/trajectory_extractor.py` | 轨迹生命周期判断。支持标定进出线判定，也支持“车辆出现到消失”的自动完成判定。 |
| `src/trajectory/trajectory_smoother.py` | 轨迹平滑器，降低检测框抖动对速度、路径长度和路径平滑度的影响。 |
| `src/trajectory/path_reference.py` | 参考路径管理。用于计算轨迹相对正常路径的偏离和路径平滑度。 |
| `src/trajectory/homography.py` | 单应矩阵求解与应用，把像素坐标映射到地面米制坐标。 |
| `src/trajectory/stitching.py` | 跨摄像头同车关联：拓扑时间窗 + ReID 相似度 + 并查集全局 ID。 |
| `src/trajectory/__init__.py` | trajectory 包导出文件。 |

## `src/behavior/` 行为向量模块

| 文件 | 作用 |
|---|---|
| `src/behavior/behavior_vector.py` | 行为向量主入口。整合轨迹、速度、停留、路径、时间、频次、聚集、颜色、车型、车牌和 ReID 字段。 |
| `src/behavior/feature_normalizer.py` | 特征标准化器，支持 z-score 和 min-max，用于后续异常检测或模型训练。 |
| `src/behavior/__init__.py` | behavior 包导出文件。 |

## `src/behavior/features/` 行为特征子模块

| 文件 | 作用 |
|---|---|
| `src/behavior/features/speed_feature.py` | 基于平滑轨迹和真实时间戳计算瞬时速度、平均速度、最大速度和速度方差。 |
| `src/behavior/features/dwell_feature.py` | 根据速度阈值计算停留时长、停留次数和停留占比。 |
| `src/behavior/features/path_feature.py` | 计算路径偏离和路径平滑度。 |
| `src/behavior/features/temporal_feature.py` | 计算夜间通行标记和夜间占比。 |
| `src/behavior/features/frequency_feature.py` | 计算车辆出现频次和 24 小时频次索引。 |
| `src/behavior/features/aggregation_feature.py` | 计算同帧车辆聚集指数和最近车辆距离。 |
| `src/behavior/features/__init__.py` | 行为特征子包导出文件。 |

## `src/feature_extraction/` 车辆属性模块

| 文件 | 作用 |
|---|---|
| `src/feature_extraction/color_recognizer.py` | 颜色识别。当前基于 HSV 规则/聚类估计车身颜色。 |
| `src/feature_extraction/vehicle_classifier.py` | 车型分类。优先使用可选 CNN 模型，缺失时用 YOLO 粗类别和 bbox 几何启发式区分普通车、货车、大客车、两轮车等。 |
| `src/feature_extraction/plate_detector.py` | 车牌区域检测。优先使用专用 YOLO 车牌模型，缺失时回退形态学候选检测。 |
| `src/feature_extraction/plate_ocr.py` | 车牌 OCR 封装。支持 OCR 置信度过滤，低置信时清空 `plate_number` 但保留 `plate_confidence`。 |
| `src/feature_extraction/reid_extractor.py` | ReID 特征提取。正式模型缺失时使用 HSV 直方图和形状特征 fallback，输出外观 embedding。 |
| `src/feature_extraction/__init__.py` | feature_extraction 包导出文件。 |

## `src/pipeline/` 流水线模块

| 文件 | 作用 |
|---|---|
| `src/pipeline/batch_processor.py` | 单段视频批处理主控。加载配置、初始化检测/跟踪/轨迹/行为/输出组件，逐帧处理视频并写出向量和事件。 |
| `src/pipeline/frame_pipeline.py` | 单帧流水线。串联检测、ROI、跟踪、裁剪保存、轨迹完成判定和行为向量生成。 |
| `src/pipeline/daily_aggregator.py` | 日聚合模块。合并同一摄像头同一天多段视频结果，并尝试用车牌/ReID 证据做跨段同车关联。 |
| `src/pipeline/__init__.py` | pipeline 包导出文件。 |

## `src/output/` 输出模块

| 文件 | 作用 |
|---|---|
| `src/output/vector_writer.py` | 行为向量 CSV/JSON writer，定义 `vec_*.csv` 表头。 |
| `src/output/event_writer.py` | 通行事件 writer，从每条完成轨迹生成一条 `vehicle_passage` 事件。 |
| `src/output/alert_writer.py` | Phase 3 告警 writer，输出可人工复核的异常告警 CSV。 |
| `src/output/__init__.py` | output 包导出文件。 |

## `src/anomaly/` Phase 3 异常检测模块

| 文件 | 作用 |
|---|---|
| `src/anomaly/baseline.py` | 正常行为基线构建、保存、加载和 Z-score 评分。默认只对普通车辆建基线。 |
| `src/anomaly/isolation_forest.py` | 纯 NumPy 自包含 Isolation Forest，输出 [0,1] 异常指数 A；含行为向量适配器与保存/加载。 |
| `src/anomaly/trajectory_transformer.py` | Transformer 轨迹预测（过去 K 点→未来 M 点），用预测残差判异，输出轨迹异常指数。 |
| `src/anomaly/rule_engine.py` | 规则引擎。检查超速、长时间滞留、路径偏离、夜间频繁停留，并尊重可靠性字段。 |
| `src/anomaly/anomaly_detector.py` | 异常检测组合器。合并规则、统计基线和 Isolation Forest 结果，写入异常字段。 |
| `src/anomaly/alert_budget.py` | 告警预算：按分数把告警量限制在目标比例内，规则命中优先。 |
| `src/anomaly/metrics.py` | 评估指标：ROC-AUC、precision/recall/F1、阈值精度。 |
| `src/anomaly/config.py` | 读取 `config/anomaly_config.yaml`（规则阈值等）。 |
| `src/anomaly/__init__.py` | anomaly 包导出文件。 |

## `src/graph/` 车辆关系图谱模块（研究要点三）

| 文件 | 作用 |
|---|---|
| `src/graph/relations.py` | 关系抽取。轨迹解析/重采样、形状相似度、时间重叠，构建共现/跟随/路径相似三类边。 |
| `src/graph/payload.py` | 图谱载荷组装（节点/社区/边），脚本共用唯一实现。输出 `graph_anomaly_comparable` / `graph_anomaly_self_referential` 标注 G 能否跨批次比较——判据是「训练图是否就是评分图」，**不能**看 `calibration.source`（GAE 的 `fit` 总写锚点，同图训练时 source 也是 baseline，会误标成可比）。 |
| `src/graph/community.py` | 纯 NumPy Louvain 社区发现，附模块度和社区摘要。 |
| `src/graph/anomaly.py` | 图异常指数 G。邻接矩阵低秩 SVD 重构残差（线性图自编码器基线），输出度与聚类系数。G 用分位锚点标定而非 min-max。 |
| `src/graph/gae.py` | 图自编码器 GAE。**特征驱动 GCN**（4 维结构特征 + 归一化邻接传播，参数与节点数无关），边掩码 + 负采样训练；支持基线图训练、目标图评分。`fit` 阶段在基线图上取分位锚点并随模型持久化。可选 `--method gae`。 |
| `src/graph/calibration.py` | **分位锚点标定**。`quantile_anchors` 从参考分布取 P50/P99，`anchor_normalize` 映射到 [0,1]。锚点来自参考分布而非本批数据，因此跨批次可比；数值超出锚点则 clip，不放大。 |
| `src/graph/features.py` | 图谱结构特征（聚类系数、二值化）。原 `anomaly.py` 与 `gae.py` 各有一份逐行等价的实现，已收敛至此。 |
| `src/graph/__init__.py` | graph 包导出文件。 |

## `src/risk/` 风险融合与报告模块（研究要点四）

| 文件 | 作用 |
|---|---|
| `src/risk/time_frequency.py` | 时间频率异常指数 T（频次、夜间占比、夜间标记），并输出 `time_freq_available` 说明该分量是否真的算得出。 |
| `src/risk/repeat_frequency.py` | **跨窗口通行频次**。按稳定身份键（`plate_hash` → `global_vehicle_id` → `camera:plate_number`）统计次数；覆盖率不足时如实标记不可用，不改写 `freq_count_24h`。 |
| `src/risk/fusion.py` | 风险融合 `R = αA + βG + γT`，输出 0~100 分与风险等级。按批次检测分量可用性，剔除不可用分量后把剩余权重按比例放大（否则 γ 会成为死权重，R 上限只有 75 分）。**单路证据封顶**：只剩一路分量可用时分数压到 75 分（低于 80 分档），即"高风险必须至少两路独立证据"——否则该路权重被放大到 1.0，`R` 退化成它的原值，会出现伪 100 分。输出 `risk_score_ceiling` / `risk_score_raw_100` / `risk_score_capped` 以便回溯。 |
| `src/risk/evidence.py` | 结构化证据链，每条证据可回溯到具体向量记录。 |
| `src/risk/report.py` | 模板报告与本地大模型报告适配器，失败自动回退模板。 |
| `src/risk/merge.py` | 风险合并：图谱异常/社区信息、跨机位伙伴与全局 ID。 |
| `src/risk/__init__.py` | risk 包导出文件。 |

## `src/site/` 园区拓扑模块

| 文件 | 作用 |
|---|---|
| `src/site/topology.py` | 加载和校验 `config/site/` 下的园区、区域、道路、摄像头、拓扑边和路线配置。 |
| `src/site/__init__.py` | site 包导出文件。 |

## `src/visualization/` 可视化模块

| 文件 | 作用 |
|---|---|
| `src/visualization/draw.py` | 在视频帧上绘制检测框、轨迹、进出线和标签，用于调试和可视化复核。 |
| `src/visualization/__init__.py` | visualization 包导出文件。 |

## `src/utils/` 工具模块

| 文件 | 作用 |
|---|---|
| `src/utils/frame_reader.py` | OpenCV 视频读取封装，提供打开、读取、属性获取和释放接口。 |
| `src/utils/mbf_extract.py` | MBF 录像格式解析：剥离私有封装，还原 H.264 Annex B 裸流。 |
| `src/utils/logger.py` | 日志工具，统一控制台和文件日志格式。 |
| `src/utils/time_utils.py` | 时间工具，处理夜间判断、时间重叠、帧号转时间、小时桶等。 |
| `src/utils/vectors_io.py` | 向量 CSV 统一读取（单文件/目录递归，写入 `_source_file`）。 |
| `src/utils/__init__.py` | utils 包导出文件。 |

## `web/map_editor/` 园区地图编辑器

| 文件 | 作用 |
|---|---|
| `web/map_editor/index.html` | 可直接打开的地图编辑器页面，包含工具栏、地图画布和属性面板。 |
| `web/map_editor/styles.css` | 地图编辑器布局、图层和地图对象样式。 |
| `web/map_editor/app.js` | 企业、道路、摄像头编辑，本地保存、撤销重做及 JSON/PNG 导入导出逻辑。 |
| `web/map_editor/assets/zone-a-base-map.jpg` | 采用 ASCII 文件名保存的综保区 A 区围网底图。 |
| `web/map_editor/README.md` | 地图编辑器使用与坐标系说明。 |

## `tests/` 测试目录

| 文件 | 作用 |
|---|---|
| `tests/test_detector.py` | 检测框中心、面积和 ROI 基础测试。 |
| `tests/test_behavior_reliability.py` | 跟踪稳定性、轨迹时间、几何可靠性、车牌/ReID fallback、CSV 字段和车辆分组测试。 |
| `tests/test_event_writer.py` | 通行事件 writer 测试。 |
| `tests/test_phase3_anomaly.py` | Phase 3 基线、规则、告警和普通车/货车过滤测试。 |
| `tests/test_site_topology.py` | 园区空间语义配置加载和引用校验测试。 |
| `tests/test_risk.py` | 风险融合测试：分量可用性、权重重归一化、证据封顶、`analysis_target=0` 剔除。 |
| `tests/test_gae.py` | 图自编码器测试：训练收敛、锚点标定、对称图退化、跨批次同误差同分、无锚点回退自标定。 |
| `tests/test_graph_comparability.py` | 图谱可比性判据测试：同图训练 `comparable=false`、加载基线 `true`、SVD 分支 `false`。 |
| `tests/test_web_api.py` | 控制台接口测试：产物按 mtime 选取、`risk.json` 命名兼容、非监管对象排除、γ=0.0 不回填、封顶透传、旧产物兼容。 |
| `tests/__init__.py` | tests 包标记文件。 |

## `tools/` 开发期校验脚本

| 文件 | 作用 |
|---|---|
| `tools/console_render_check.js` | 控制台渲染校验。用 Node 内置 `vm` + 最小 DOM 桩执行真实 `web/console/app.js`，断言 20 项渲染结果（封顶红条、`无数据` 文案、KPI 卡片、封顶小标、横幅的五类提示文案）。不依赖浏览器或 playwright，可直接进 CI。 |

## `src/web/` 控制台服务

| 文件 | 作用 |
|---|---|
| `src/web/server.py` | 只读 HTTP 服务与目录扫描。按 `mtime` 选取最新产物、兼容 `risk*.json` 命名、分离非监管对象、透传封顶与图谱可比性字段。 |
| `web/console/index.html` / `app.js` / `style.css` | 排查控制台前端：批次概览、复核清单、单车详情、分数构成与证据链。 |

## 数据和模型目录

| 路径 | 作用 |
|---|---|
| `models/yolov8s.pt` | YOLOv8 车辆检测权重。 |
| `data/test_videos/frame_0001.jpg` | 摄像头标定或测试用样例帧。 |
| `data/outputs/` | 本地运行输出目录，包括向量、事件、告警等；应作为生成物忽略，不提交。 |

## 主要输出文件

| 输出 | 路径模式 | 用途 |
|---|---|---|
| 行为向量 | `data/outputs/vectors/<camera_id>/<date>/vec_<camera_id>_<date>_<hour>.csv` | 每条完成轨迹一行，保存速度、停留、路径、属性、ReID 和异常占位字段。 |
| 通行事件 | `data/outputs/events/<camera_id>/<date>/events_<camera_id>_<date>_<hour>.csv` | 每条完成轨迹一条 `vehicle_passage`，用于后续跨摄像头和告警处理。 |
| 正常基线 | `data/references/baseline_<camera_id>.json` | Phase 3 正常行为统计基线。 |
| 告警结果 | `data/outputs/alerts/alerts_<camera_id>.csv` | Phase 3 异常检测结果，供人工复核。 |

## 当前阶段边界

- Phase 1：车辆检测、ROI、跟踪、轨迹、基础行为向量。
- Phase 2：颜色、车型、车牌候选/OCR、ReID 外观特征、普通车/货车分组。
- Phase 3：基于历史向量建立正常行为基线，并输出规则/统计告警。
- Phase 4：跨摄像头同车关联、全域轨迹拼接和空间路径判断；当前已有配置骨架，但仍需要真实摄像头拓扑和多天连续数据。
