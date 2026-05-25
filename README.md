# 综保区行政通道车辆异常检测系统

基于 YOLOv8 + ByteTrack 的车辆检测与跟踪系统，面向综保区行政通道场景，
实现车辆行为向量化与异常检测。

## 项目结构

```
vehicle_tracking_system/
├── config/                        # 配置文件
│   ├── default.yaml               # 全局默认配置
│   ├── model_config.yaml          # 模型参数
│   ├── behavior_config.yaml       # 行为向量特征配置
│   └── camera/                    # 各摄像头专属配置
│       └── camera_01.yaml
├── src/                           # 源代码
│   ├── detection/                 # 检测模块 (YOLOv8 + ROI过滤)
│   ├── tracking/                  # 跟踪模块 (ByteTrack)
│   ├── trajectory/                # 轨迹提取模块
│   ├── behavior/                  # 行为向量化模块 (核心)
│   ├── feature_extraction/        # 车辆属性特征 (Phase 2)
│   ├── anomaly/                   # 异常检测模块 (Phase 3)
│   ├── pipeline/                  # 流水线编排
│   ├── output/                    # 结果输出
│   ├── visualization/             # 可视化
│   └── utils/                     # 工具函数
├── scripts/                       # 运行脚本
│   ├── run_pipeline.py            # 主运行入口
│   ├── calibrate_camera.py        # 摄像头标定工具
│   └── run_batch.py               # 批量处理
├── data/
│   ├── test_videos/               # 测试视频
│   ├── videos/                    # 原始视频
│   ├── outputs/                   # 输出结果
│   └── references/                # 参考数据
├── models/                        # 模型权重
├── tests/                         # 单元测试
└── requirements.txt
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载模型

将 YOLOv8 模型权重放入 `models/` 目录：
- `yolov8s.pt` (自动下载或手动放置)

### 3. 摄像头标定（首次使用）

```bash
# 从测试视频截取首帧
python scripts/calibrate_camera.py \
    --image data/test_videos/frame_0001.jpg \
    --camera cam01
```

按提示标注 ROI、出入口线、参照物，标定结果自动保存到 `config/camera/cam01.yaml`。

### 4. 运行流水线

```bash
# 处理测试视频
python scripts/run_pipeline.py \
    --video data/test_videos/test_cam01_5min.mp4 \
    --camera cam01 \
    --output data/outputs
```

### 5. 查看输出

行为向量 CSV 文件位于：
```
data/outputs/vectors/cam01/<date>/vec_cam01_<date>_<hour>.csv
```

## 行为向量字段说明

| 字段 | 说明 | 单位 |
|------|------|------|
| avg_speed_ms | 平均速度 | m/s |
| max_speed_ms | 最大速度 | m/s |
| speed_variance | 速度方差 | - |
| max_dwell_sec | 最大停留时间 | s |
| stop_count | 停留次数 | 次 |
| dwell_ratio | 停留时长占比 | [0,1] |
| path_deviation | 路径偏离度 | m |
| path_smoothness | 路径平滑度 | rad |
| is_night | 是否夜间 | 0/1 |
| night_ratio | 夜间时段占比 | [0,1] |
| freq_index | 高频通行指数 | Z-score |
| aggregation_index | 聚集行为指数 | [0,1] |
| anomaly_score | 异常分数 | [0,1] |
| is_anomaly | 是否异常 | 0/1 |
| alert_reason | 告警原因 | 文本 |

## 实施阶段

- **Phase 1** (当前): 基础管线 — 检测→跟踪→轨迹→行为向量→CSV输出
- **Phase 2**: 完整行为向量 — 车牌OCR、ReID、场景标定
- **Phase 3**: 异常检测 — 规则引擎 + 统计模型
- **Phase 4**: 全量部署 — 批量处理、性能优化、评估
