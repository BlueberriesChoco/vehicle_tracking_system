# 机位拓扑网：怎么建、怎么填、怎么验

拓扑回答一个问题：**车从 A 机位开到 B 机位，正常要多久？**

多机位联动、按需调度、错配剔除都依赖它。没有拓扑，就只能两两暴力比对，110 个机位是 5,995 对，跑不动也验不了。

---

## 一、数据结构

拓扑由两个文件组成：

| 文件 | 作用 | 谁来改 |
|---|---|---|
| `config/site/topology_edges.csv` | **人填的边表**（Excel 可编辑） | 运维/业务 |
| `config/site/topology.yaml` | **脚本生成的正式配置** | 不要手改 |

生成：

```powershell
.venv\python.exe scripts/build_topology.py
# 只检查不写文件
.venv\python.exe scripts/build_topology.py --check
```

### CSV 列

| 列 | 必填 | 说明 |
|---|---|---|
| `from_camera_id` | ✅ | 上游机位 id，必须与 `cameras.yaml` / `config/camera/*.yaml` 一致 |
| `to_camera_id` | ✅ | 下游机位 id |
| `distance_m` | 建议 | 两机位间的实际行车距离（米）。填了才能自动推时间窗 |
| `min_travel_sec` | 可选 | 最快通过时间。不填按 `距离 ÷ 20km/h × 0.8` 推 |
| `max_travel_sec` | 可选 | 最慢通过时间（**要含场区内停留**）。不填按 `距离 ÷ 3km/h × 1.5` 推 |
| `route_id` | 可选 | 路线标识，便于按路线聚合统计 |
| `one_way` | 可选 | `true` 表示单向，不自动补反向边 |
| `notes` | 可选 | 人类可读备注 |

### 生成的 YAML

```yaml
version: 1
generated_by: scripts/build_topology.py
edges:
- from_camera_id: admin_in
  to_camera_id: admin_out
  min_travel_sec: 20
  max_travel_sec: 10800
  distance_m: 85.0
  route_id: admin_lane_in_to_out
  one_way: true
  notes: 行政通道进→出；上限 3 小时含场区内办事停留
```

---

## 二、时间窗怎么定（最容易错的地方）

时间窗是**硬约束**：配对时超出窗口的组合直接丢弃。定错了会有两种后果：

| 定错方向 | 后果 |
|---|---|
| **太窄** | 真实链路被剪掉，匹配率虚低（看起来"系统找不到车"） |
| **太宽** | 错配增多（A 车进、B 车出也能配上） |

推荐做法：

```
min_travel_sec = 距离 ÷ 最高车速(20 km/h) × 0.8     ← 留 20% 余量
max_travel_sec = 距离 ÷ 最低车速(3 km/h) × 1.5 + 停留预算
```

**关键是"停留预算"**：行政通道这种车辆要办事的路线，实测有车进区后 1–3 小时才出来。
`admin_in → admin_out` 只有 85 米，纯行驶 20 秒，但实测最大 8.4 分钟、业务上要留到 3 小时——
所以这一条必须手填 `max_travel_sec: 10800`，不能靠推算。

判断规则：

| 路线类型 | max_travel_sec |
|---|---|
| 卡口→卡口（只路过，不停） | 距离 ÷ 3 km/h × 1.5 |
| 卡口→场区（可能装卸） | + 30 分钟 |
| 场区→出口（可能办事） | + 3 小时 |

---

## 三、110 个机位怎么填

### 第一步：确定机位位置

两个来源：

1. **地图编辑器**（`web/map_editor`）：已有 `G-01`～`R-02` 点位，但那是**规划点位**，不是实际机位
2. **实地/图纸**：把 110 个机位标到园区图上（这是最花时间的一步）

没有坐标也能填 CSV，但只能靠人判断"哪两个机位相邻"，无法批量生成。

### 第二步：按道路分组

同一条路上的机位按行驶方向串成链：

```
卡口1 → 行政车道进 → 行政车道出 → 查验平台东
```

每条链只需相邻两两建边，**不要全连接**——全连接会让错配暴增，也让链路失去方向性。

### 第三步：反过来补边

双向通行的路要建两条边（A→B、B→A），时间窗相同；
单向（如出区卡口）只建一条，并标 `one_way: true`。

### 第四步：跑检查

```powershell
.venv\python.exe scripts/build_topology.py --check
```

输出：

```yaml
connectivity:
  nodes: 3
  edges: 1
  isolated: [cam01]              # 孤立机位，既无进也无出
  sources: [admin_in]            # 只有出边：路线起点
  sinks: [admin_out]             # 只有入边：路线终点
  one_way_gaps: [admin_in->admin_out]   # 有 A->B 无 B->A，单向需确认
```

要看的三个信号：

| 信号 | 含义 | 处理 |
|---|---|---|
| `isolated` 非空 | 该机位不在任何链路上 | 补边，或确认它是独立监控点 |
| `sources` 太多 | 碎片化，链没串起来 | 检查是不是漏了上游边 |
| `one_way_gaps` 很多 | 双向路只建了单向 | 补反向边 |

---

## 四、用数据反向校验拓扑

拓扑是人填的，一定要拿真实数据验。方法：

1. 选一对已建边的机位，两侧各跑一段录像（窗口要够长，≥2 小时）
2. 导出事件 CSV，取每辆车的 `exit_time`（A）与 `enter_time`（B）
3. 统计实际穿行时间分布

```python
# 示意
gaps = [ (b.enter_time - a.exit_time).seconds for a, b in matched ]
print(min(gaps), median(gaps), max(gaps))
```

对照表：

| 现象 | 说明 |
|---|---|
| 实测最大值 > 配置 max | **max 定窄了**，会剪掉真实链路，要放宽 |
| 实测最小值 < 配置 min | min 定宽了，或存在重复检测，要收紧 |
| 很多车在 A 出现但 B 无对应 | 要么 B 漏检（覆盖率问题），要么拓扑漏了另一条出路 |

`admin_in → admin_out` 的实测（15 分钟窗口）：穿行 21–540 秒，中位 128 秒，
而配置是 20–10800 秒，说明**上限非常宽松**——这是对的，因为长停留的车占多数。

---

## 五、常见错误

| 错误 | 后果 | 正确做法 |
|---|---|---|
| 全连接（每个机位连所有机位） | 错配率暴涨，等于没拓扑 | 只连相邻机位 |
| max 只按车速算，忘了停留 | 真实长停留链路被剪 | 按路线类型加停留预算 |
| 双向路只建单向 | 一半车辆无法配对 | 补反向边 |
| 机位 id 与 `cameras.yaml` 不一致 | 脚本报"不在 cameras.yaml"，但边仍生成 | 先统一 id 命名 |
| 手改 `topology.yaml` | 下次生成被覆盖 | 改 CSV 重跑脚本 |

---

## 六、当前进度

| 项 | 状态 |
|---|---|
| 格式与生成器 | ✅ `scripts/build_topology.py` |
| 检查（孤立/源汇/单向缺口） | ✅ `--check` |
| 已有边 | 1 条（`admin_in → admin_out`） |
| 待建 | 140 个机位的边表 |

**下一步**要先把 110 个机位标到园区图上，才能按道路分组。这一步只能靠现场/图纸，脚本替代不了。
