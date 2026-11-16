# 运行维护手册（OPERATIONS）

面向**日常运维人员**：怎么启动、看什么指标、指标异常怎么判断、阈值怎么调、出问题怎么查。
部署与镜像构建见 [`deploy/README.md`](deploy/README.md)，指标体系与实测值见
[`ACCEPTANCE_REPORT.md`](ACCEPTANCE_REPORT.md)。

---

## 零、环境准备（新机器冷启动）

```powershell
# 1) 建虚拟环境并装依赖（约 10–20 分钟，torch/CUDA 体积较大）
python -m venv .venv
.venv\python.exe -m pip install -U pip
.venv\python.exe -m pip install -r requirements.txt

# 2) 放模型权重（不入库，需单独分发）
#    models/yolov8s.pt            必需（检测）
#    models/vehicle_reid.onnx     可选（车辆 ReID；缺失回退 HSV 直方图）

# 3) 自检（6 步必须全绿）
.\scripts\ci.ps1
#    含交付清单检查：文件齐备 / YAML 可解析 / 文档引用有效 / 无临时目录
```

依赖分两类，见 [`requirements.txt`](requirements.txt)：

| 类型 | 缺失后果 |
|---|---|
| **必需**（ultralytics / opencv / numpy / scipy / pyyaml / torch / pytest / psutil） | 直接跑不起来 |
| **可选**（easyocr / onnxruntime / torchreid / paddleocr） | 有 try/except 回退，只是车牌/ReID 能力退化 |

`tests/test_dependencies.py` 会检查"代码里硬导入的第三方包是否都已声明"，
防止换台机器才发现缺包。

`tests/test_minimal_env.py` 验证**分析层（检测/图谱/风险/报告）不依赖 OpenCV 与 ultralytics**：
把这两个模块屏蔽后，IF→图谱→融合→证据→报告仍能跑通。

> 一个结构约束：`src/anomaly/__init__.py` 会导入 Transformer，因此 import 任何 anomaly
> 子模块都会连带加载 `torch`。`torch` 本就属于必需依赖，不影响部署；若做"纯规则、无 torch"
> 的极简部署，需要把该 `__init__` 改成惰性导入。

---

## 一、日常运行

### 1.1 每日

```powershell
# 1) 处理当天录像（整段或分窗）
.venv\python.exe scripts\run_full_pipeline.py `
  --input "F:\A区录像 新\行政车道进" `
  --camera admin_in `
  --output data/outputs/daily

# 2) 汇总当天关键指标（含图谱层断言）
.venv\python.exe scripts\report_metrics.py --root data/outputs/daily --require-graph

# 3) 规则触发率审计（发现“规则写了但不工作”）
.venv\python.exe scripts\audit_rules.py --input data/outputs/daily/alerts/vectors_admin_in.csv
```

长录像（>2 小时）建议用分窗模式，单车上报延迟与视频总长解耦：

```powershell
.venv\python.exe scripts\run_windowed.py `
  --video <视频文件> --camera admin_in --window-sec 600 `
  --output data/outputs/windowed
```

### 1.2 每次改代码后

```powershell
.\scripts\ci.ps1
```

四步：单元测试 → 端到端冒烟 → 规则审计 → 指标汇总（G 中位数为 0 直接失败）。

---

## 二、健康检查清单

| 检查项 | 命令 | 正常范围 | 异常含义 |
|---|---|---|---|
| 车辆数 | `report_metrics.py` | 与历史同段相当 | 骤降 → 检测/ROI/录像问题 |
| 告警率 | 同上 | **5%–15%** | >15% 误报过高；0% 可能规则/模型失效 |
| 中/高风险 | 同上 | 中风险约占 2%–5% | 突然为 0 → 检查 G 与 T |
| **G 中位数** | 同上 | **> 0.2** | **= 0 → 图谱层失效（见 5.1）** |
| T 中位数 | 同上 | 白天通常 0 | 恒定非 0 → 时间频率口径异常 |
| 证据可追溯率 | 同上 | **100%** | <100% → 证据缺少来源字段 |
| 规则命中 | `audit_rules.py` | 视场景而定 | 长期 0 → 阈值或可靠性门槛问题 |
| 处理帧率 | 日志 | ≥15 FPS/路 | 低于则见 5.4 |

---

## 三、关键指标怎么读

### 3.1 告警率

- **定义**：`is_anomaly=1` 的车辆 / 总车辆；
- 由 `contamination`（Isolation Forest 预期异常比例）与告警预算共同决定；
- 参考实测（66 分钟、444 辆）：`contamination=0.05` → **9.23%**。

**告警率对 `contamination` 的实测曲线**（`scripts/calibrate_alert_rate.py`，444 辆车）：

| contamination | 告警数 | 告警率 | ≤15% 目标 |
|---|---|---|---|
| 0.01 | 10 | 2.25% | ✅ |
| 0.02 | 17 | 3.83% | ✅ |
| 0.03 | 25 | 5.63% | ✅ |
| **0.05（当前）** | **41** | **9.23%** | ✅ |
| 0.08 | 64 | 14.41% | ✅（已贴边） |
| 0.10 | 79 | 17.79% | ❌ |

**这意味着什么**：9.23% 不是"模型天生如此"，而是 `contamination=0.05` 的**直接结果**——
阈值是分位数，改参数就线性改告警率。

如果业务方希望多捞一些线索、且能接受 15% 的上限，可以把 `contamination` 调到 **0.08**
（告警从 41 增到 64，多 56%），代价是人工复核量同步上升。**0.10 起就会突破误报率指标**。

复现：

```powershell
.venv\python.exe scripts/calibrate_alert_rate.py `
  --input data/outputs/e2e/alerts/vectors_admin_in.csv `
  --grid "0.01,0.02,0.03,0.05,0.08,0.10" --epochs 30
```

### 3.2 行为异常指数 A

- 来源：Isolation Forest 归一化分数（0–1）；
- A 只反映“相对正常分布偏离程度”，**不是违规概率**；
- 中位数通常在 0.4–0.5，属正常。

### 3.3 图异常指数 G

- 来源：图谱节点重构误差（SVD / GAE）；
- **G 中位数为 0 = 图谱层没有贡献**，这是最需要警惕的静默故障；
- 健康值参考：0.4–0.6。

### 3.4 时间频率指数 T

- 由“超出基线的频次 + 夜间通行”合成；
- 单次白天通行 **T=0**（正常）；
- 夜间出现（`is_night=1`）至少 +0.2；
- 需要**多日数据**频次分量才有意义，单段视频通常为 0。

### 3.5 综合风险 R

- `R = 0.4A + 0.35G + 0.25T`，输出 0–100；
- ≥80 高风险、≥50 中风险；
- 真实数据目前**没有车辆达到 80 分**（最高约 59），这是阈值偏保守，不是故障。

**有效权重与标称权重不同**：A 的归一化上界是 0.822、G 是 1.0，所以
`0.4×0.822 = 0.329 < 0.35×1.0 = 0.350`——**G 的实际最大贡献更大**。
换句话说，当前融合是「图优先」的：进中风险主要靠 G。

**两种口径（默认群体优先）**：

| 口径 | 命令 | 中风险（444 辆实测） | 适用 |
|---|---|---|---|
| 群体优先（默认） | 直接跑 | 8 | 团伙 / 蚂蚁搬家 |
| 个体优先 | 加 `--corroboration` | 13 | 踩点车 / 单车异常 |

```powershell
# 个体优先：≥2 个独立信号（IF / Transformer / 规则）命中即至少给 medium
.venv\python.exe scripts\build_risk_report.py `
  --input data/outputs/e2e/alerts/vectors_admin_in.csv `
  --graph data/outputs/e2e/graphs/graph_admin_in.json `
  --output data/outputs/e2e/risk/risk_admin_in.json `
  --corroboration
```

> 切换口径后，**指标 7 / 14 / 15 的统计基线会变**，对外报数时必须注明用的是哪一种。

---

## 四、阈值调整

### 4.1 告警率太高

```powershell
# 方式一：降低 contamination
.venv\python.exe scripts\run_full_pipeline.py --contamination 0.02 ...

# 方式二：直接给告警率上限（推荐，结果可预期）
.venv\python.exe scripts\run_full_pipeline.py --max-alert-rate 0.10 ...
```

标定脚本可给出“contamination → 实际告警率”的曲线：

```powershell
.venv\python.exe scripts\calibrate_alert_rate.py --input <vectors.csv>
```

### 4.2 规则阈值

全部集中在 [`config/anomaly_config.yaml`](config/anomaly_config.yaml)：

| 键 | 默认 | 含义 | 调整建议 |
|---|---|---|---|
| `speed_max_kmh` | 40 | 超速阈值 | **需先完成标定**，否则规则被跳过 |
| `dwell_max_sec` | 120 | 长时间滞留 | 通行车道可适当降低到 60 |
| `deviation_max_m` | 3.0 | 路径偏离 | 需标定 |
| `snake_smoothness_max` | 0.5 | 平均转角上限 | 443 辆标定集分布：p90=0.28、p95=0.35，0.5 约 1.4%–1.8% 触发 |
| `night_stop_count` | 2 | 夜间停留次数 | — |
| `night_high_freq_count` | 6 | 夜间高频 | — |
| `night_any_passage` | 1 | 夜间出现即告警 | 该车道夜间基本无车，建议保持开启 |

> ⚠️ 改阈值后务必跑 `audit_rules.py` 确认规则**实际触发数**发生变化，
> 而不是改完就假设生效。

### 4.3 几何标定

未标定时：`speed_reliable=0`、`path_reliable=0`，超速与路径偏离规则**自动跳过**（这是正确行为，不是 bug）。

完成标定后（需要**像素坐标**与对应的**地面米制坐标**，至少 4 组）：

```powershell
.venv\python.exe scripts\calibrate_homography.py `
  --pixels "100,200;800,200;800,900;100,900" `
  --world  "0,0;10,0;10,15;0,15" `
  --camera admin_in `
  --update-config
```

不带 `--update-config` 时只计算不写回，便于先验证残差。

---

## 五、故障排查

### 5.1 G 全为 0 / 图谱层失效

**现象**：`report_metrics.py` 的 G 中位数为 0；`--require-graph` 退出码 2。

**原因**：图谱节点键与向量键不一致（历史事故：图谱用 `camera:track`，向量用 `passage_id`）。

**排查**：

```powershell
# 看两边键长什么样
.venv\python.exe -c "
import json,csv
g=json.load(open('data/outputs/daily/graphs/graph_admin_in.json',encoding='utf-8'))
print([n['node'] for n in g['nodes'][:3]])
rows=list(csv.DictReader(open('data/outputs/daily/alerts/vectors_admin_in.csv',encoding='utf-8')))
print([r.get('passage_id') for r in rows[:3]])
"
```

**处理**：用当前版本重建图谱（`build_graph.py` 或重跑全链路），**不要复用旧图谱文件**。
`apply_graph` 在完全不匹配时会打 WARNING。

### 5.2 车辆数为 0

按顺序排查：

1. **录像时段有没有车**：夜间该车道基本无车（00:00 与 20:57 两段 ROI 内检出均为 0），这是正常的；
2. **ROI 是否把车排除了**：抽样检测，看 `roi_filter` 前后数量；
3. **是否跨过入口线**：车辆若从画面中部出现、从未越过 entry 线，不会被计为一辆（设计如此）；
4. **轨迹是否太短**：`min_trajectory_length` 默认 10，短暂出现会被过滤。

### 5.3 规则命中数为 0

1. `audit_rules.py` 查看 `speed_reliable` / `path_reliable` 是否为 0（未标定 → 规则按设计跳过）；
2. 检查字段分布：例如 `max_dwell_sec` 全 0 说明该车道没有停留；
3. 确认阈值方向：**蛇形是“转角越大越可疑”**（`>= θ`），不要写成 `<=`。

### 5.4 处理帧率不足

按收益从高到低排查：

| 手段 | 效果 |
|---|---|
| **确认解码预取已开**：`pipeline.prefetch_frames`（默认 32） | **+27%**（98 → 125 FPS），设为 0 会明显变慢 |
| 跳帧：`pipeline.frame_skip: 2` | **+47%**（125 → 190 FPS），代价是轨迹点减半、速度 +1.7% |
| 提高批量：`batch_size` 16 → 32 | 开启预取后仅 +1%（瓶颈已不在批量） |
| 分窗处理 | 降低内存，不改变吞吐 |
| 降分辨率 / 减小 `img_size` | 明显提速，但影响小目标检测 |
| 换 C++/TensorRT 后端 | 可达 20 路，需 CUDA 13 + 重新编译（本机不可行，见 `PERFORMANCE.md` P2） |

先跑 `python scripts/benchmark_throughput.py --video <h264> --frames 400` 确认当前基线。

### 5.5 大模型报告失败

- 报告服务**自动回退模板**，不会阻塞链路；
- 检查本地 Ollama：`curl http://127.0.0.1:11434/api/tags`；
- 0.5B 小模型容易复述提示词，事实校验不过就会回退，属预期行为；
- 建议 7B 以上模型。

### 5.6 `.mbf` 文件处理报错

```powershell
.venv\python.exe scripts\extract_mbf.py --input <mbf目录> --output <h264目录>
```

注意 `--output` 必须指向**已存在的目录**，否则会被当成输出文件名。

---

## 六、模型更新

新增一周数据后的增量更新：

```powershell
# 重新拟合 Isolation Forest + 微调 Transformer（实测约 5 秒）
.venv\python.exe scripts\update_models.py `
  --new-vectors <新的 vectors.csv> `
  --output-dir data/outputs/references `
  --existing-iforest data/outputs/references/iforest_admin_in.json `
  --existing-traj data/outputs/references/traj_admin_in
```

（`--existing-iforest` / `--existing-traj` 省略时等价于从头训练。）

或分别全量重训：

```powershell
.venv\python.exe scripts\train_iforest.py `
  --input <vectors.csv> --output data/outputs/references/iforest_admin_in.json

.venv\python.exe scripts\train_transformer.py `
  --input <vectors.csv> --output data/outputs/references/traj_admin_in --epochs 30
```

> 重训后请重跑 `report_metrics.py`，确认告警率仍在 5%–15%。

---

## 六之二、工具箱（按需使用）

这几个脚本不在日常主流程里，但排查与调参时很有用。全部支持 `--help`。

| 脚本 | 用途 | 典型用法 |
|---|---|---|
| `run_multi_camera.py` | 多机位批量处理 + 跨机位匹配 | `--cameras admin_in,admin_out --pairs admin_in:admin_out --output <dir>` |
| `benchmark_batch.py` | 扫描不同 batch size 的检测吞吐，找最优值 | `--video <h264> --batch-sizes 1,8,16,32` |
| `render_camera_config.py` | 把 ROI / 进出线画到画面上，核对配置是否对准车道 | `--video <h264> --camera admin_in --frame 500 --output check.png` |
| `calibrate_camera.py` | 交互式标定摄像头（选 ROI、画进出线） | `--help` 查看步骤 |
| `calibrate_homography.py` | 由地面控制点算单应矩阵并写回配置 | 见第四节 |
| `calibrate_alert_rate.py` | 扫描 contamination → 实际告警率的曲线 | `--input <vectors.csv>` |
| `export_gang_case.py` | 导出团伙案例报告 | 见 `demo/README.md` |
| `build_demo_pack.py` | 一键生成演示包 | 见 `demo/README.md` |
| `delivery_check.py` | 交付门禁（文件/配置/引用/临时目录） | 无参数 |

> 新机位上线前建议先跑 `render_camera_config.py`，确认 ROI 和进出线真的压在车道上。
> 本项目就出现过"入口线只覆盖半幅车道"导致漏检的问题，肉眼核对最快。

---

## 七、数据与隐私

- 车牌默认**不入库明文**：只保存 `plate_hash`（SHA-256 前 8 位）；
- 录像与输出目录（`data/outputs/`、`data/validation/`）已在 `.gitignore` 中；
- 模型权重（`models/*.pt`、`*.onnx`）不入库，部署时单独分发；
- 输出中的每条证据都带 `source_file`，可回溯到具体录像分段。

---

## 八、升级与回滚

```powershell
# 升级前
.\scripts\ci.ps1                     # 必须全绿
.venv\python.exe -m pytest -q          # 244 passed
.venv\python.exe scripts\delivery_check.py   # 交付清单 5 项

# 升级后验证
.venv\python.exe scripts\report_metrics.py --root <新输出> --require-graph
```

若升级后关键指标异常（告警率突变、G=0、车辆数骤降），回滚代码并**用旧版本重建产物**
——注意图谱、向量、风险报告必须由**同一版本**生成，混用会出现键不匹配。
