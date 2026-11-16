# 交付说明（先看这一页）

> 综保区行政通道智慧监管原型：**仅依赖监控视频**，从轨迹到风险报告的全链路。
> 输出的是**排查线索**，不是违规判定。

---

## 一、30 秒验证它能跑

```powershell
# 1) 环境自检（6 步，约 1 分钟）
.\scripts\ci.ps1

# 2) 跑一个真实片段（约 30 秒）
.venv\python.exe scripts\run_full_pipeline.py `
  --input data\outputs\demo\h264 --camera admin_in `
  --output data\outputs\demo\run --max-frames 3000

# 3) 一键出报告 + 指标
.venv\python.exe scripts\build_demo_pack.py --root data\outputs\demo\run --demo-dir demo
.venv\python.exe scripts\report_metrics.py --root data\outputs\demo\run
```

打开 `demo/SUMMARY.md` 看到结果；`demo/risk_sample.md` 是单车报告，`demo/gang_sample.md` 是多车案例。

---

## 二、实测数字（66 分钟完整段，444 辆车）

| 指标 | 实测值 |
|---|---|
| 车辆数 | 444 |
| 告警数 / 告警率 | 41 / **9.23%**（目标 ≤15%） |
| 中风险 / 高风险 | 8 / 0（R 最高 59.0） |
| 行为异常指数 A 中位数 | 0.43 || 图异常指数 G 中位数 | 0.509 |
| 时间频率指数 T 中位数 | 0.0（该时段无夜间/高频） |
| 证据数 / 可追溯率 | 1276 / **100%** |
| 图谱 | 432 节点 / 1427 边 / 105 社区 / Q=0.7348 |
| 单卡吞吐 | 端到端 **8.4 路**；`frame_skip=2` 时 **12.7 路** |
| 端到端延迟 | **23.1 秒**（1 分钟片段） |
| 增量更新 | **5.06 秒** |
| 带宽压缩比 | **462 倍**（结构化结果 gzip，0.216%） |
| 已知异常检出 | 长停留 / 蛇形 / 夜间高频 / 超速 **均 100%**（合成验证） |

> ⚠️ **"41 告警"和"8 中风险"是两批不同的车**：41 辆是「个体异常」（`is_anomaly=1`，
> 由 IF / 规则触发），8 辆是「融合风险中档」（由 R 决定）。**交集只有 3 辆**（29、421、549）；
> 另外 5 辆中风险车是个体正常、但图异常 G 很高（0.91–1.00）才进入中风险。
>
> 原因：`is_anomaly` 只看个体行为，R 还包含**图异常 G**（群体信号）。
> **排查时应以风险列表（按 R 排序）为准**，告警标记只作参考。

---

## 三、23 项技术指标的状态

| 状态 | 数量 | 说明 |
|---|---|---|
| ✅ 已实测达标 | 6 | 证据链、延迟、增量更新、带宽、已知异常检出、停留精度 |
| 🟡 待外部输入 | 10 | 需标注数据 / 现场标定 / 双机位真值 / 多日数据 |
| 🟠 未达标 | 2 | 20 路吞吐（8.4–12.7 路，需 CUDA 13）；Re-ID（实测 AUC 0.794，目标 90%） |
| 🔴 无数据 | 5 | 团伙、边准确率、R 的 AUC、R≥80 精确率、未知异常 AUC |

逐项证据与复现命令见 [`METRICS_TRACEABILITY.md`](METRICS_TRACEABILITY.md)。

---

## 四、必须知道的限制

1. **未做现场标定**：速度、路径偏离字段标记为不可靠（`speed_reliable=0`），
   对应规则自动跳过——这是设计行为，不是故障；
2. **车牌 OCR 不可用**：1080p 下车牌约 60px，`87316` 会被读成 `873061`，不作身份依据；
3. **跨机位 Re-ID 未达标**：AUC 0.794、EER 28%（含属性先验 0.827），远低于 90%；
4. **该车道夜间基本无车**：00:00 与 20:57 两段 ROI 内检出均为 0；
5. **20 路未达标**：本机 CUDA 11.8，GPU 版 ORT 与 TensorRT 均需 CUDA 13；
6. **7×24 只做了 300 轮浸泡**（25 分钟），未做完整 72 小时；
7. **边缘设备未验证**：Dockerfile 是 x86_64 + CUDA 11.8，Jetson（ARM64）需要重做构建链
   （代码可移植，但 torch/opencv/onnxruntime 要换 JetPack 版 ARM64 wheel）。
   实测资源占用很轻（RSS 1.97 GB、GPU 351 MB），门槛在构建而非算力。

---

## 五、文档地图

| 想知道什么 | 看哪份 |
|---|---|
| 项目整体思路与架构 | [`README.md`](README.md) |
| **目标完成情况与待办** | **[`GOAL_STATUS.md`](GOAL_STATUS.md)** |
| 23 项指标自评 | [`ACCEPTANCE_REPORT.md`](ACCEPTANCE_REPORT.md) |
| 每项指标怎么复现 | [`METRICS_TRACEABILITY.md`](METRICS_TRACEABILITY.md) |
| 日常怎么运维、出问题怎么查 | [`OPERATIONS.md`](OPERATIONS.md) |
| 性能瓶颈与优化路线 | [`PERFORMANCE.md`](PERFORMANCE.md) |
| 经济效益怎么测算 | [`BENEFITS.md`](BENEFITS.md) |
| 异常行为怎么定义 | [`ANOMALY_SIGNATURES.md`](ANOMALY_SIGNATURES.md) |
| 还要补什么数据 | [`DATA_REQUIREMENTS.md`](DATA_REQUIREMENTS.md) |
| 代码结构 | [`CODE_STRUCTURE.md`](CODE_STRUCTURE.md) |
| 每次改动记录 | [`CHANGELOG.md`](CHANGELOG.md) |
| 端到端演示 | [`demo/README.md`](demo/README.md) |

---

## 六、部署

Docker 与本地两种方式见 [`deploy/README.md`](deploy/README.md)。

```powershell
# 本地最小启动
python -m venv .venv
.venv\python.exe -m pip install -r requirements.txt
# 把 models/yolov8s.pt 放进 models/（必需）
# 可选：models/vehicle_reid.onnx（车辆 ReID，缺失则回退 HSV 直方图）
```

---

## 七、怎么验证"结果可信"

```powershell
# 同一份输入跑两次，结构完全一致（444 车 / 41 告警 / 8 中风险）
.venv\python.exe scripts\run_full_pipeline.py --input <录像> --camera admin_in --output data\outputs\run_a
.venv\python.exe scripts\run_full_pipeline.py --input <录像> --camera admin_in --output data\outputs\run_b
.venv\python.exe scripts\report_metrics.py --root data\outputs\run_a
.venv\python.exe scripts\report_metrics.py --root data\outputs\run_b
```

三次不确定性来源均已收口：**时间**（无时间戳时固定占位并禁用夜间判断）、
**随机**（模型/图算法固定种子，并有实际输出一致的测试）、**文件顺序**（全部 sorted）。
