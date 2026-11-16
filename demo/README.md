# 端到端演示：从现场 `.mbf` 到风险排查报告

> 一键重新生成本演示包（从本目录那次演示运行）：
> ```powershell
> .venv\python.exe scripts\build_demo_pack.py --root data/outputs/demo/run --demo-dir demo
> ```
> 会生成 `SUMMARY.md` / `risk_sample.md` / `gang_sample.md`。
> 想看早高峰完整段（444 辆）的版本，把 `--root` 换成 `data/outputs/e2e`。

本目录记录一次**完整链路**的实际运行结果，用于交付演示与回归核对。

## 一、输入

| 项 | 值 |
|---|---|
| 现场文件 | `F:\A区录像 新\行政车道进\Download_南京南通海关综保A区行政车道进_20260901090712_20260901093521.mbf` |
| 大小 | 236 MB |
| 时段 | 2026-09-01 09:07:12 – 09:35:21 |
| 机位配置 | `config/camera/admin_in.yaml` |

## 二、执行步骤

### 1. `.mbf` → `.h264`

```bash
python scripts/extract_mbf.py \
  --input "F:\A区录像 新\行政车道进\Download_..._20260901090712_20260901093521.mbf" \
  --output data/outputs/demo/h264
```

结果：`234 MB` 的 `.h264` 裸流（`.mbf` 是私有封装，块头 + H.264 Annex B）。

### 2. 全链路处理

```bash
python scripts/run_full_pipeline.py \
  --input data/outputs/demo/h264 \
  --camera admin_in \
  --output data/outputs/demo/run \
  --max-frames 3000 \
  --device 0
```

只取前 3000 帧（约 2 分钟）以便快速演示。链路依次完成：检测 → 跟踪 → 行为向量 → 无监督异常 → 关系图谱 → 风险融合 → 证据链 → 报告。

### 3. 产物

| 产物 | 说明 |
|---|---|
| `vectors/.../vec_admin_in_20260901_09.csv` | 行为向量（含 `passage_id`） |
| `alerts/alerts_admin_in.csv` | 告警明细 |
| `alerts/vectors_admin_in.csv` | 带异常字段的向量 |
| `graphs/graph_admin_in.json` | 车辆关系图谱（节点/边/社区） |
| `risk/risk_admin_in.json` | 风险报告（结构化） |
| `risk/risk_admin_in.md` | 风险报告（可读版） |

## 三、本次结果

```text
车辆：7
高风险：0    中风险：1    低风险：6
最高风险：admin_in:7，R = 62.6 分
```

该车被识别为「行为模式与常态不符 + 与其他车辆通行模式高度相似（疑似结伴）」，报告见 [`risk_sample.md`](risk_sample.md)。

> **注意样本量**：本段只有 7 辆车，Isolation Forest 就在这 7 条上训练，
> 告警率会偏高（实测 2/7 = 28.57%），**不代表真实业务告警率**。
> 早高峰完整段（444 辆）的告警率是 **9.23%**，见 `ACCEPTANCE_REPORT.md`。

本目录的四份文件由同一次演示运行（`data/outputs/demo/run`）生成：

| 文件 | 来源 |
|---|---|
| `SUMMARY.md` / `risk_sample.md` / `gang_sample.md` | `python scripts/build_demo_pack.py --root data/outputs/demo/run --demo-dir demo` |
| 本 README | 手工维护 |

## 四、完整复现

```powershell
# 1) 转换
.venv\python.exe scripts/extract_mbf.py --input "<mbf路径>" --output data/outputs/demo/h264

# 2) 全链路
.venv\python.exe scripts/run_full_pipeline.py --input data/outputs/demo/h264 `
  --camera admin_in --output data/outputs/demo/run --max-frames 3000 --device 0

# 3) 只重出报告（不重跑视频）
.venv\python.exe scripts/build_risk_report.py `
  --input data/outputs/demo/run/alerts/vectors_admin_in.csv `
  --graph data/outputs/demo/run/graphs/graph_admin_in.json `
  --output data/outputs/demo/run/risk/risk_admin_in.json `
  --markdown data/outputs/demo/run/risk/risk_admin_in.md
```

## 五、团伙案例（多车）

单车的报告见 [`risk_sample.md`](risk_sample.md)；**多车结伴**的案例见
[`gang_sample.md`](gang_sample.md)。

> **为什么团伙案例来自完整段而非演示片段**：演示片段只有 7 辆车，
> 社区规模太小（最大 3 辆），不足以说明"蚂蚁搬家"。
> 因此 `gang_sample.md` 取自早高峰完整段（444 辆）。

```powershell
.venv\python.exe scripts/export_gang_case.py `
  --vectors data/outputs/e2e/alerts/vectors_admin_in.csv `
  --risk    data/outputs/e2e/risk/risk_admin_in.json `
  --output  demo/gang_sample.md
```

该样例从早高峰完整段中挑出**社区 #30（10 辆车、38 条内部边、平均 G=0.673）**，
并**主动标注了“可能是高峰车流误报”**——10 辆车在 26 秒内连续通过，单看共现不足以判定团伙。

## 六、说明

- 本段只有 7 辆车、1 条中风险，属于**稀疏时段**；早高峰完整段（66 分钟）最新全链路实测为 444 辆车、8 条中风险，见 `ACCEPTANCE_REPORT.md`。
- 报告中的 `admin_in:7` 可直接用于回查视频轨迹（`vectors CSV` 里的 `passage_id` + 时间戳）。
