# 技术指标 —— 验证路径对照表

> 用途：验收时逐项复核。每项指标都给出**验证方式、复现命令、产物文件**。
> 指标定义与实测结论见 [`ACCEPTANCE_REPORT.md`](ACCEPTANCE_REPORT.md)；
> 运维操作见 [`OPERATIONS.md`](OPERATIONS.md)。

## 约定

- 全链路产物的根目录记作 `$ROOT`（例如 `data/outputs/e2e`）；
- 一键汇总：`python scripts/report_metrics.py --root $ROOT --require-graph`；
- 标注到位后的回填：`python scripts/eval_with_labels.py --vectors <csv> --labels <csv>`。

## 一、车辆检测与轨迹提取

| # | 指标 | 状态 | 验证方式 | 产物 / 命令 |
|---|---|---|---|---|
| 1 | 检测准确率≥95%、召回≥93% | 🟡 | 无检测真值，只能证明"检测稳定运行" | `$ROOT/alerts/vectors_*.csv`（车辆数）；`python scripts/benchmark_throughput.py`（57–64 FPS） |
| 2 | 跨摄像头 Re-ID≥90% | 🟠 | 自动构造同车/异车对（无需标注），算 AUC/EER | `python scripts/eval_reid_pairs.py --video <h264> --min-gap 25` → **AUC 0.794**（含属性先验 0.827） |
| 3 | 轨迹拼接完整率≥88% | 🟡 | 需要双机位同车真值 | `python scripts/stitch_cameras.py --from-vectors ... --to-vectors ...` → `match_count` |
| 4 | 速度误差≤±5 km/h | 🟡 | 需要地面控制点；当前 `speed_reliable=0` | `python scripts/calibrate_homography.py --pixels ... --world ...`；`python scripts/audit_rules.py` 看 `speed_reliable` |
| 5 | 停留时间误差≤±3 秒 | ✅ | 基于真实时间戳，与标定无关 | `tests/test_features.py::test_dwell_detected_for_long_stop`（80 帧 = 3.2s） |

## 二、个体行为异常检测

| # | 指标 | 状态 | 验证方式 | 产物 / 命令 |
|---|---|---|---|---|
| 6 | Transformer 3 秒位移误差≤2 米 | 🟡 | 按 50 px/m 估算各视野误差 | 0.2s→**0.21 m**；1.0s→**1.22 m**（81.6% ≤2m）；3s 无法评估（轨迹中位数 1.84s） |
| 7 | A 召回≥80%、误报≤15% | 🟡 | **误报率可算**：告警数 / 车辆数 | `python scripts/report_metrics.py --root $ROOT` → **9.23%**；合成 9.49% |
| 8 | 已知异常类型检出≥85% | ✅ | 合成注入四类异常 | `python scripts/synthetic_anomaly_eval.py --input <csv> --output <json>` → 四类均 **100%** |
| 9 | 未知异常泛化 AUC≥0.85 | 🔴 | 无异常标注 | 需 `eval_with_labels.py` + 真实违规标签 |

## 三、图神经网络异常检测

| # | 指标 | 状态 | 验证方式 | 产物 / 命令 |
|---|---|---|---|---|
| 10 | 边准确率≥85% | 🔴 | "真实交互关系"无客观定义 | `$ROOT/graphs/graph_*.json`（`edge_count`） |
| 11 | GAE 重构误差识别异常节点≥85% | 🟡 | 结构注入 AUC：度越界型 **0.963±0.045**（达标）；保度重连型 0.584（未达标） | `python scripts/eval_gae_transfer.py --vectors data/outputs/e2e/alerts/vectors_admin_in.csv --seeds 5` → `data/outputs/eval/gae_transfer.json`；`tests/test_gae.py` |
| 12 | 团伙检测准确率≥80% | 🔴 | 无团伙 ground truth | `$ROOT/graphs/graph_*.json`（`communities`） |
| 13 | 社区发现检出≥75% | 🟡 | Louvain 可用，缺已知团伙 | `python scripts/report_metrics.py` → **105 社区 / Q=0.7348** |

## 四、风险融合与系统性能

| # | 指标 | 状态 | 验证方式 | 产物 / 命令 |
|---|---|---|---|---|
| 14 | R 的 AUC≥0.90 | 🔴 | 无高风险标注 | `python scripts/report_metrics.py` → 中风险 8 / 高风险 0，R∈[0,59.0] |
| 15 | R≥80 精确率≥75% | 🔴 | 同上；真实数据未触达 80 分 | 同上（`risk_max`） |
| 16 | 证据链准确率≥90% | ✅ | 每条证据是否带 `trace.source_file` | `python scripts/report_metrics.py` → **1276/1276 = 100%** |
| 17 | LLM 事实还原≥95%、幻觉<2% | 🟡 | 数字白名单校验，不通过自动回退模板 | `tests/test_risk.py::test_report_service_falls_back_on_hallucinated_llm_output` |
| 18 | 端到端延迟≤2 分钟 | ✅ | 1 分钟片段全链路计时 | `python scripts/run_full_pipeline.py --input <60s片段>` → **23.1s**；分窗 18.6–27.8s |
| 19 | 20 路 1080P / 单 T4 | 🟠 | 单路吞吐 × 路数 | `benchmark_throughput.py --batch-size 32` → **138.7 FPS ≈ 9.25 路**（纯处理）；端到端 125.6 FPS ≈ 8.4 路；`frame_skip=2` 时 12.7 路 |
| 20 | 带宽压缩比≥200 | ✅ | 只上传结构化结果并 gzip | 66min 录像 1016 MB vs 向量 CSV 5.16 MB（197x）/ gzip 2.20 MB（**462x**，0.216%） |

## 五、系统稳定与适应性

| # | 指标 | 状态 | 验证方式 | 产物 / 命令 |
|---|---|---|---|---|
| 21 | 7×24 无宕机、单路≥15 FPS | 🟡 | 浸泡测试 20 轮（非 72h） | `python scripts/soak_test.py --iterations 300 --max-frames 500` → 0 崩溃、FPS 漂移 +7.0%、RSS +36.9 MB |
| 22 | 增量重训≤4 小时 | ✅ | 计时增量更新 | `python scripts/update_models.py --new-vectors ... --output-dir ...` → **5.06s** |
| 23 | 光照/雨雪下降≤5 点 | 🟡 | 合成压暗（真实夜间无车） | 30% 亮度 → 检出 -8.2%，置信度稳定；`data/outputs/night*` |

## 六、按状态汇总

| 状态 | 数量 | 指标编号 |
|---|---|---|
| ✅ 已实测达标 | 6 | 5、8、16、18、20、22 |
| 🟡 已实现、待外部输入 | 10 | 1、3、4、6、7、11、13、17、21、23 |
| 🟠 未达标 | 2 | 2、19 |
| 🔴 无数据无法验证 | 5 | 9、10、12、14、15 |

## 七、外部输入到位后的回填路径

| 缺什么 | 拿到后怎么补 |
|---|---|
| 检测标注 | 标注 500 帧 → 算 P/R → 更新第 1 项 |
| 双机位同车真值 | `stitch_cameras.py` + 人工核对 → 更新第 2、3 项 |
| 地面控制点 | `calibrate_homography.py --pixels --world --update-config` → 激活超速/偏离规则，更新第 4 项 |
| 违规标注 | `eval_with_labels.py` → 更新第 7、9、14、15 项 |
| 多日数据 | 频次特征生效，T 不再恒为 0 → 更新第 7、14 项 |
| 高性能后端 | TensorRT/C++ → 更新第 19 项 |

## 八、复现前提

```powershell
# 1) 环境自检（必须全绿）
.\scripts\ci.ps1

# 2) 全链路重跑
.venv\python.exe scripts/run_full_pipeline.py --input <录像> --camera admin_in --output data/outputs/e2e

# 3) 一键汇总（G 中位数为 0 会直接失败）
.venv\python.exe scripts/report_metrics.py --root data/outputs/e2e --require-graph
```

> 所有产物必须由**同一版本**生成；混用不同版本的图谱/向量会出现键不匹配（见 `OPERATIONS.md` 5.1）。
