# 综保区车辆行为向量化与异常排查系统

本项目面向综合保税区监管场景，目标是把监控视频中的车辆通行过程转化为结构化行为数据，并为后续异常检测、空间语义拓扑分析、企业/车辆/司机关系风险研判提供数据基础。

当前系统处于**数据采集层和行为向量化层**建设阶段。它可以处理监控视频，输出车辆轨迹、时间、速度、停留、频次、聚集、车型、颜色等行为向量；但还不是完整的“自动判定违规车辆”系统。完整业务异常判断还需要接入园区地图、摄像头拓扑、企业/仓库/车库位置、申报单和车辆授权关系。

## 核心思路

项目最终要解决的问题不是简单检测车辆，而是回答：

```text
哪些车辆或司机需要重点关注？
为什么需要关注？
证据是什么？
应该如何排查？
```

因此系统分为四层：

1. 视频行为向量化
   从监控视频中提取车辆检测、跟踪、轨迹、停留、频次和车辆属性。

2. 行为异常检测
   基于正常行为分布和规则引擎识别停留过久、夜间异常、频繁出现、异常聚集等行为。

3. 空间语义与摄像头拓扑
   将摄像头绑定到道路、卡口、企业门口、仓库、停车区等语义区域，判断车辆路径是否合理。

4. 风险融合与解释
   融合行为异常、空间路径异常、企业/司机/车辆关系异常，输出可解释风险证据和排查建议。

当前代码主要完成第 1 层，并为第 2 层和第 3 层预留接口。

## 当前能力

- YOLOv8 车辆检测。
- ROI 过滤。
- 简化 ByteTrack 跟踪。
- 短暂丢失轨迹恢复。
- 轨迹提取和平滑。
- 行为向量计算：
  - 平均速度
  - 最大速度
  - 停留时长
  - 停留次数
  - 路径平滑度
  - 夜间通行
  - 高频通行指数
  - 聚集行为指数
- 车辆属性提取：
  - 车型粗分类
  - 颜色识别
  - 车牌区域检测
  - OCR 接口
  - ReID 接口
- 每段视频独立处理，避免跨视频状态污染。
- 支持从监控下载文件名解析录像时间。
- 支持几何配置等级和可靠性字段。
- 支持日聚合骨架。
- 支持中文变更日志。

## 当前限制

- 未配置 ROI 的摄像头会使用全画面，可能混入路边停车或无关道路车辆。
- 未透视标定的摄像头，速度、路径偏离、车辆间距不能作为正式风险依据。
- 远景监控下车牌通常像素不足，OCR 不稳定。
- `anomaly_score`、`is_anomaly`、`alert_reason` 当前仍主要是占位字段，规则告警和统计异常模型尚未正式接入主流程。
- 尚未接入企业、司机、仓库、申报单、车辆授权关系。
- 尚未建立园区空间语义地图和摄像头拓扑。

## 项目结构

```text
vehicle_tracking_system/
├── CHANGELOG.md                   # 中文项目变更日志
├── README.md                      # 项目说明和使用说明
├── requirements.txt               # Python 依赖
├── config/
│   ├── default.yaml               # 全局默认配置
│   ├── model_config.yaml          # 车辆属性模型配置
│   ├── behavior_config.yaml       # 行为特征配置
│   └── camera/                    # 单摄像头 ROI、进出线、标定配置
├── scripts/
│   ├── run_pipeline.py            # 单视频处理入口
│   ├── run_batch.py               # 批量处理入口
│   ├── calibrate_camera.py        # 摄像头标定工具
│   └── build_baseline.py          # 正常行为基线构建脚本
├── src/
│   ├── detection/                 # YOLO 检测和 ROI 过滤
│   ├── tracking/                  # 简化 ByteTrack 跟踪
│   ├── trajectory/                # 场景几何、轨迹提取和平滑
│   ├── behavior/                  # 行为向量特征
│   ├── feature_extraction/        # 颜色、车型、车牌、OCR、ReID
│   ├── anomaly/                   # 异常检测骨架
│   ├── pipeline/                  # 视频处理、日聚合
│   ├── output/                    # CSV/JSON 输出
│   ├── visualization/             # 可视化绘制
│   └── utils/                     # 工具模块
├── tests/                         # 回归测试
├── data/                          # 测试帧和输出目录
└── models/                        # YOLO 权重
```

## 环境准备

```bash
pip install -r requirements.txt
```

模型权重放在：

```text
models/yolov8s.pt
```

当前仓库中已有 YOLO 权重文件。后续如果迁移仓库，建议改为外部下载或 Git LFS 管理。

## 单视频运行

基本命令：

```bash
python scripts/run_pipeline.py \
  --video data/test_videos/test_cam01_5min.mp4 \
  --camera cam01 \
  --output data/outputs
```

处理外部视频示例：

```bash
python scripts/run_pipeline.py \
  --video "F:\test_videos\centerlode_north\Download_CH05综保区A区中央路北_20260422080046_20260422093204_202606010928376.mp4" \
  --camera centerlode_north \
  --output data/outputs \
  --max-frames 1500
```

参数说明：

| 参数 | 说明 |
|---|---|
| `--video` | 输入视频路径 |
| `--camera` | 摄像头 ID，对应 `config/camera/<camera_id>.yaml` |
| `--config` | 全局配置文件，默认 `config/default.yaml` |
| `--output` | 输出根目录 |
| `--output-video` | 可选，输出标注后视频 |
| `--max-frames` | 最大处理帧数，调试时使用，`0` 表示不限制 |
| `--device` | GPU 编号，`-1` 表示 CPU |

## 批量运行

```bash
python scripts/run_batch.py \
  --camera cam01 \
  --date 20260422 \
  --video-dir data/videos/camera_01 \
  --output data/outputs
```

批量处理会逐段视频运行。当前处理器会在每段视频开始前清空轨迹状态，避免上一段视频的 `track_id`、频次统计和已完成向量污染下一段。

## 输出结果

行为向量 CSV 输出位置：

```text
data/outputs/vectors/<camera_id>/<date>/vec_<camera_id>_<date>_<hour>.csv
```

示例：

```text
data/outputs/vectors/cam01/20260422/vec_cam01_20260422_08.csv
```

主要字段：

| 字段 | 说明 |
|---|---|
| `track_id` | 当前视频段内轨迹 ID |
| `global_vehicle_id` | 日聚合后的全局车辆 ID |
| `segment_count` | 日聚合合并的视频段数 |
| `camera_id` | 摄像头 ID |
| `vehicle_type` | 车辆类型 |
| `vehicle_color` | 车辆颜色 |
| `plate_number` | OCR 识别车牌号，可能为空 |
| `plate_hash` | 车牌哈希 |
| `geometry_level` | 摄像头几何配置等级 |
| `speed_reliable` | 速度字段是否可靠 |
| `path_reliable` | 路径偏离字段是否可靠 |
| `aggregation_reliable` | 空间聚集字段是否可靠 |
| `passage_reliable` | 进出通行判断是否可靠 |
| `enter_time` | 车辆首次进入/出现时间 |
| `exit_time` | 车辆离开/最后出现时间 |
| `duration_sec` | 持续时间，秒 |
| `trajectory_length_m` | 轨迹长度，米，未标定时仅为估算 |
| `avg_speed_ms` | 平均速度，m/s，未标定时不建议用于告警 |
| `max_speed_ms` | 最大速度，m/s，未标定时不建议用于告警 |
| `speed_variance` | 速度方差 |
| `max_dwell_sec` | 最大停留时间 |
| `stop_count` | 停留次数 |
| `dwell_ratio` | 停留占比 |
| `path_deviation` | 路径偏离度 |
| `path_smoothness` | 路径平滑度 |
| `is_night` | 是否夜间 |
| `night_ratio` | 夜间占比 |
| `freq_index` | 高频通行指数 |
| `freq_count_24h` | 24 小时频次计数 |
| `aggregation_index` | 聚集行为指数 |
| `nearest_vehicle_m` | 最近车辆距离，米 |
| `trajectory_points` | 轨迹点序列 |
| `reid_embedding` | ReID 特征向量 |
| `anomaly_score` | 异常分数，占位或待接入 |
| `is_anomaly` | 是否异常，占位或待接入 |
| `alert_reason` | 告警原因，占位或待接入 |

## 摄像头配置等级

系统支持多摄像头渐进接入，不要求所有摄像头一开始都完成精细标定。

| `geometry_level` | 含义 | 可用能力 |
|---|---|---|
| `none` | 未配置 ROI | 检测、跟踪、时间、频次、颜色、车型 |
| `roi` | 已配置 ROI，但未完整标定 | 可排除无关区域，通行行为更干净 |
| `calibrated` | ROI、进出线、参考距离齐全 | 速度、路径、聚集距离可参与风险评分 |

可靠性字段的使用原则：

```text
speed_reliable = 0      不使用速度做超速告警
path_reliable = 0       不使用路径偏离做风险判断
aggregation_reliable=0  不使用空间距离聚集做风险判断
passage_reliable = 0    不使用进出线判断通行方向
```

## 摄像头标定

首次配置摄像头时，可以运行：

```bash
python scripts/calibrate_camera.py \
  --image data/test_videos/frame_0001.jpg \
  --camera cam01
```

标定结果保存到：

```text
config/camera/<camera_id>.yaml
```

建议优先配置：

1. 道路 ROI
   排除路边停车、绿化带、无关道路和建筑区域。

2. 进出线
   用于判断车辆通行方向。

3. 参考距离
   用于估算速度和空间距离。

对大量摄像头，建议先做 ROI 级配置，再对重点通道做完整标定。

## 真实视频测试结论

已使用 `F:\test_videos\centerlode_north` 下的中央路北监控视频做过抽样测试。

结论：

- 前 11 段视频可读，规格为 `1280x720 / 25 FPS`。
- 最后 2 段视频文件损坏，FFmpeg 报 `moov atom not found`。
- 文件名中的连续时间戳已经支持解析。
- IoU 跟踪阈值从 `0.8` 调整为 `0.3` 后，轨迹碎片明显减少。
- 在未配置中央路北专属 ROI 和透视标定前，速度和聚集距离不能作为正式风险判断依据。

## 海康边缘主机接入方向

如果现场使用海康超脑，例如 `iDS-9664NX-I8R/X`，它可以作为结构化事件来源。

可利用能力包括：

- 车辆结构化
- 车辆抓拍
- 车牌、车型、颜色
- 目标频次
- 区域入侵、越界、进入区域、离开区域
- 录像回放关联

推荐架构：

```text
海康结构化事件
        +
自研行为向量
        +
园区空间语义拓扑
        +
企业/司机/车辆/申报单数据
        =
异常车辆排查与风险评分
```

后续可新增：

```text
src/integrations/hikvision/
  client.py
  event_parser.py
  schema.py
```

## 空间语义和拓扑建设方向

项目后续需要建立园区空间语义数据：

```text
config/site/
  site.yaml
  zones.yaml
  roads.yaml
  cameras.yaml
  topology.yaml
  routes.yaml
```

需要的数据包括：

- 摄像头清单
- 摄像头对应道路和区域
- 摄像头朝向
- 道路相邻关系
- 企业、仓库、车库、停车区位置
- 允许通行路径
- 业务申报目的地

没有这些数据时，系统只能判断“画面级行为异常”，不能判断“业务路径异常”。

## 测试

运行全部测试：

```bash
python -m pytest -q
```

静态编译检查：

```bash
python -m compileall -q src scripts tests
```

当前可靠性测试重点覆盖：

- 短暂丢失轨迹恢复
- 检测框抖动下轨迹稳定性
- 聚集特征排除自身
- 按同一帧计算车辆距离
- 跳帧速度时间戳
- 轨迹过线时间
- 批处理状态重置
- 日聚合数值转换和跨段合并
- 视频文件名时间解析
- 几何可靠性等级

## 变更日志

每次代码、配置或处理逻辑变更，都需要同步更新：

```text
CHANGELOG.md
```

运行生成的 CSV、验证截图和本地工具配置不作为正式变更提交。

## 下一步

建议按以下顺序继续：

1. 增加轨迹质量评估与清洗输出。
2. 为中央路北摄像头配置 ROI。
3. 建立 `config/site/` 空间语义配置骨架。
4. 将摄像头绑定到道路和区域。
5. 生成车辆通行事件。
6. 接入规则告警。
7. 训练正常行为基线。
8. 接入企业、司机、车辆和申报单数据。
