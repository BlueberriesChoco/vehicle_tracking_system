# 下一步任务手册（在另一台机器执行）

三件事，按顺序做。每件都有明确的产出和判定标准。

前置：仓库已同步到最新提交，`.venv` 已按 cu128 建好，`pytest` 全绿。

```powershell
git pull
.\.venv\python.exe scripts\check_environment.py   # 应输出「环境一致，torch 可用」
```

---

## 任务 A：全量实测（先做这个）

**目的**：确认 100 台在范围内的机位，到底有多少台能真的产出通行记录。

静态校验只能发现几何问题；「标了但一条通行都出不来」只有跑真实录像才能暴露。

```powershell
.\.venv\python.exe scripts\batch_validate_cameras.py `
    --start "2026-09-01T09:00:00" --minutes 5 `
    --output data\outputs\camera_validation.csv
```

- 每台取同一时段（默认 09:00）的 5 分钟，跑完整流水线，统计通行数
- 单台约 1.5 分钟（含取片段 + 解码 + 跟踪）
- 100 台约 **1 小时**（5080）；笔记本显卡约 2.5 小时

**先试 5 台**确认脚本没跑偏：

```powershell
.\.venv\python.exe scripts\batch_validate_cameras.py --limit 5 --minutes 3
```

**产出与判定**

`data/outputs/camera_validation.csv`：

| camera | passages | exit_code | error |
|---|---|---|---|
| 广兴路西 | 46 | 0 | |
| 中央路北 | 2 | 0 | |

- `passages = 0` 的机位 → 标定或配置仍有问题（也可能是该时段真的没车）
- `exit_code != 0` → 流水线报错，看 error 列

**把 results 发回来**，我按结果分类：真没车 / 标定问题 / 配置问题。

---

## 任务 B：主干线串行程

**目的**：验证「同一辆车在两个机位之间能被串起来」，这是行程分析的基础。

选一条拓扑上真实连通的链路：

```
admin_in → admin_out → 广兴路东 → 中央路北
```

**步骤 1：给 4 台跑同一时段的数据**

```powershell
# admin_in 已有标定，直接跑 66 分钟窗口
.\.venv\python.exe scripts\run_full_pipeline.py `
    --input "F:\A区录像 新\行政车道进\<对应文件>.mbf" --camera admin_in `
    --output data\outputs\mc_admin_in --max-frames 99046

# 广兴路东 / 中央路北 同理（先解压 mbf）
.\.venv\python.exe scripts\extract_mbf.py --input "<广兴路东的 mbf>" --output data\outputs\h264\gx_east
.\.venv\python.exe scripts\run_full_pipeline.py `
    --input "data\outputs\h264\gx_east\<文件>.h264" --camera 广兴路东 `
    --output data\outputs\mc_gx_east --max-frames 99046
```

**步骤 2：串行程**

```powershell
.\.venv\python.exe scripts\build_journeys.py `
    --in-a data\outputs\mc_admin_in\vectors\admin_in\20260901 `
    --in-b data\outputs\mc_gx_east\vectors\广兴路东\20260901 `
    --output data\outputs\journeys_admin_to_gx.json `
    --min-travel-sec 30 --max-travel-sec 1800
```

**判定**：输出的 `journey_count` 和 `similarity` 分布。

- 相似度中位 ≥0.8、行程数 >10 → 串联可用
- 行程数为 0 或个位数 → 时间窗或 ReID 阈值要调

---

## 任务 C：训练路线模型（依赖 A、B 的结果）

**当前状态**：管线与工具已就绪，训练代码还没有——因为还缺「足够的行程样本」。

**训练对象**（重新明确，避免跑偏）：

> 只看从行政通道卡口进来的车，判断它**在区内的行程**是否偏离常态。
> 「去没去查验平台」不是判据——关员会坐小车去平台查货再回来，这类车是正常的。

**第 1 步：攒行程数据**

按任务 B 的方法，把这条链路上的机位各跑 1 天：

```
admin_in → admin_out → 广兴路东 → 中央路北 → 中央路南
```

5 台 × 1 天 × 2.1 小时（5080）≈ **10.5 小时**，产出约 3000–5000 条行程。

**第 2 步：特征**

| 特征 | 来源 |
|---|---|
| 机位序列 | 行程的 legs |
| 每段耗时 | 相邻 leg 的时间差 |
| 区内总时长 | 首 leg 进 → 末 leg 出 |
| 点位停留 | 同机位多次出现的时间差 |
| 折返次数 | 机位重复出现 |

**第 3 步：模型**

- 路线：变阶马尔可夫（学常见机位序列）
- 时长：分位数（学每段正常耗时范围）
- 异常分 = 路线罕见度 + 时长越界度

**先别做**：深度模型。样本量不足以训练序列模型，马尔可夫 + 分位数可解释、可验证。

---

## 时间预算汇总

| 任务 | 5080 耗时 | 产出 |
|---|---|---|
| A 全量实测 | 约 1 小时 | 每台通行量清单 |
| B 主线串行程 | 约 1.5 小时（4 台 × 66 分钟窗口） | 行程 JSON + 相似度分布 |
| C 训练数据 | 约 10.5 小时（5 台 × 1 天） | 3000–5000 条行程 |

---

## 常见问题

**Q: 为什么有的机位 passages=0？**

先看该机位在那个时段有没有车（可以用 `data/outputs/camera_survey.csv` 的检出一列参考）。没车就不是标定问题。

**Q: 时间对不齐怎么办？**

`batch_validate_cameras.py` 按文件时间与大小估算偏移，误差几十秒。用于「有没有通行」够用；要做精确跨机位串联时，用 `extract_mbf.py` 解完整片段再 `--start-frame` 对齐。

**Q: 内存不够？**

流水线按视频流式处理，单台峰值约 2–3 GB。批量脚本逐台串行，不会叠加。
