# 交付说明

> 项目：综保区行政通道智慧监管系统原型
> 版本：2026-11-10
> 状态：四层算法链路已实现并跑通；部分技术指标依赖现场数据，尚待验证（详见 `ACCEPTANCE_REPORT.md`）

---

## 一、交付物总览

| 类别 | 交付物 | 位置 |
|---|---|---|
| 源代码 | 四层算法 + 跨摄像头 + 部署脚本 | `src/`、`scripts/` |
| 模型 | YOLOv8s 检测权重、IF/Transformer 训练产物 | `models/`、运行时 `references/` |
| 配置 | 全局、摄像头、站点拓扑、异常阈值 | `config/` |
| 文档 | 8 份（见第五节） | 仓库根目录 |
| 演示 | 从 `.mbf` 到报告的完整样例 | `demo/` |
| 测试 | 244 个单元测试 + 端到端冒烟 | `tests/`、`scripts/smoke_test.py` |
| 部署 | Dockerfile / compose / 启动脚本 | `Dockerfile`、`docker-compose.yml`、`scripts/start.ps1` |

---

## 二、环境要求

| 项 | 要求 |
|---|---|
| Python | 3.12（仓库内 `.venv`） |
| GPU | 可选；有 NVIDIA GPU 时自动用 CUDA |
| 依赖 | `requirements.txt` |
| 模型权重 | `models/yolov8s.pt`（约 22 MB） |
| 大模型（可选） | 本地 Ollama 或任意 OpenAI 兼容接口 |

安装：

```powershell
python -m venv .venv
.venv\python.exe -m pip install -r requirements.txt
```

---

## 三、快速开始

### 1. 一键验证链路

```powershell
.\scripts\ci.ps1
```

包含 6 步：单元测试 + 五个 CLI 的端到端冒烟 + 规则触发率审计 + 指标汇总（G 中位数为 0 即失败）+ 一键演示包 + 交付清单检查。

### 2. 跑一个真实片段

```powershell
.\scripts\start.ps1 -Input "data/outputs/h264/admin_in" -Camera admin_in -Output data/outputs/run
```

### 3. 完整演示（`.mbf` → 报告）

见 [`demo/README.md`](demo/README.md)，实际结果见 [`demo/risk_sample.md`](demo/risk_sample.md)。

---

## 四、代码结构

详见 [`CODE_STRUCTURE.md`](CODE_STRUCTURE.md)。核心分层：

```
src/
├── detection/      车辆检测（YOLOv8s，支持批处理）
├── tracking/       跟踪（ByteTrack）
├── trajectory/     轨迹提取、平滑、几何标定、跨机位拼接
├── behavior/       行为向量化（速度/停留/路径/频次/同屏）
├── anomaly/        无监督异常（Isolation Forest、Transformer、告警预算、标定、指标）
├── graph/          关系图谱、Louvain 社区、SVD / GAE 异常指数
├── risk/           风险融合、证据链、模板报告、大模型报告、合并工具
├── pipeline/       帧流水线与批处理
└── output/         向量/告警写出
```

---

## 五、文档索引

| 文档 | 内容 |
|---|---|
| [`README.md`](README.md) | 项目入口、运行方式、指标实测 |
| [`CODE_STRUCTURE.md`](CODE_STRUCTURE.md) | 代码结构与模块职责 |
| [`CHANGELOG.md`](CHANGELOG.md) | 全部变更记录（按日期/条目） |
| [`FEASIBILITY_REPORT.md`](FEASIBILITY_REPORT.md) | 前期可行性研究 |
| [`ANOMALY_SIGNATURES.md`](ANOMALY_SIGNATURES.md) | 异常行为签名定义 |
| [`ACCEPTANCE_REPORT.md`](ACCEPTANCE_REPORT.md) | 23 项技术指标逐条自评 |
| [`DATA_REQUIREMENTS.md`](DATA_REQUIREMENTS.md) | 现场数据采集清单（P0/P1/P2） |
| [`DELIVERY.md`](DELIVERY.md) | 本文件，交付总览 |

---

## 六、验收证据（可复现）

| 检查项 | 命令 | 结果 |
|---|---|---|
| 单元测试 | `pytest -q` | **244 passed** |
| 端到端冒烟 | `python scripts/smoke_test.py --video <h264> --max-frames 1500` | **5/5 阶段通过** |
| 全量重跑 | `run_full_pipeline.py`（66 分钟录像） | 444 车 / 1276 证据 / 可追溯 100% |
| 指标汇总 | `python scripts/report_metrics.py --root data/outputs/e2e` | 一键输出车辆数/告警率/风险/证据/图谱统计 |
| 演示包 | `python scripts/build_demo_pack.py --root data/outputs/e2e --demo-dir demo` | 生成 SUMMARY + 单车报告 + 团伙案例 |
| 一页纸入口 | [`START_HERE.md`](START_HERE.md) | 30 秒验证 + 实测数字 + 限制 + 文档地图 |
| 运维手册 | [`OPERATIONS.md`](OPERATIONS.md) | 巡检清单、指标判读、阈值调整、6 类故障排查 |
| 指标对照表 | [`METRICS_TRACEABILITY.md`](METRICS_TRACEABILITY.md) | 23 项指标 → 命令 → 产物，验收可逐项复核 |
| 效益测算 | [`BENEFITS.md`](BENEFITS.md) | 经济/社会效益的参数化模型、算例与敏感性分析 |
| 演示 | `demo/README.md` | `.mbf` 236MB → 7 车 / 1 中风险 |
| 端到端延迟 | 1 分钟片段 | **23.1s**（≤2 分钟） |
| 单车延迟（分窗） | `run_windowed.py --window-sec 60` | **18.6–27.8s/窗**，与视频总长解耦 |
| 增量更新 | `update_models.py` | **5.06s**（≤4 小时） |
| 吞吐 | `benchmark_throughput.py` | 纯处理 **9.25 路** / 端到端 **8.4 路**（解码预取已开；未开为 6.5 路；skip=2 为 12.7 路） |
| 运行稳定性 | `soak_test.py` | **300 轮** 0 崩溃，FPS 漂移 +7.0%，RSS +36.9 MB（20/100/300 轮持平，非泄漏） |
| 告警率可控 | `calibrate_alert_rate.py` | contamination 0.08 → 14.22% |

---

## 七、技术指标现状（摘要）

| 状态 | 数量 | 说明 |
|---|---|---|
| ✅ 已实测达标 | 5 | 证据链、延迟、增量更新、误报率、部分已知异常 |
| 🟡 已实现待验证 | 10 | 主要缺标注/标定/双机位 |
| 🟠 未达标 | 2 | 20 路吞吐、R≥80 精确率 |
| 🔴 无法验证 | 6 | 缺违规标注、团伙真值、多天气数据 |

完整逐条评估见 [`ACCEPTANCE_REPORT.md`](ACCEPTANCE_REPORT.md)。

---

## 八、已知限制

1. **20 路 1080P 未达标**：实测 **8.4 路**（解码预取已生效）；`cProfile` 显示瓶颈在解码/推理/同步，需 TensorRT 与 GPU 版 ReID。
2. **无标注数据**：召回率、AUC、精确率无法计算；`eval_with_labels.py` 已就绪，标注到位即可出数。
3. **未现场标定**：`admin_in` 无地面控制点，速度/路径字段标记为不可靠。
4. **ReID 为 HSV 直方图 fallback**：跨机位匹配框架完整，但真实准确率需正式 ReID 模型。
5. **大模型为 0.5B**：事实校验机制生效，但文本质量一般，建议换 7B+。
6. **单机位验证为主**：多机位编排与拼接已实现，但缺真实双机位数据验证。

---

## 九、后续工作清单

按优先级：

1. **现场采集**（6–7 人天，可解锁 5 项指标）→ 见 [`DATA_REQUIREMENTS.md`](DATA_REQUIREMENTS.md)
2. **性能优化**：TensorRT / 批处理 / C++ 管线，目标 20 路
3. **正式 ReID 模型**接入
4. **7×24 长稳测试**
5. **大模型升级**至 7B/14B 并复测

---

*本交付说明与 `ACCEPTANCE_REPORT.md`、`DATA_REQUIREMENTS.md` 互相引用；所有数字均来自仓库内可复现脚本。*
