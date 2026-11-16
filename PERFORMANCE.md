# 性能剖析与 20 路优化路线

> 结论先行：**当前 8.4 路（端到端 125.6 FPS）的瓶颈不在"算法慢"，而在"解码 + 单条 ReID 推理 + GPU 同步"三块。**
> 本文件记录 `cProfile` 实测数据与对应的优化项。

## 一、实测数据

环境：单卡（NVIDIA，CUDA 11.8），YOLOv8s + batch=16，1080P 输入。

```text
400 帧 / 4.618 秒 = 86.6 FPS（含 4 辆车的 ReID 推理）
```

### 热点分布（按 `tottime`）

| 排名 | 热点 | 耗时 | 占比 | 性质 |
|---|---|---|---|---|
| 1 | `cv2.VideoCapture.read` | 1.181s | **25.6%** | 视频解码 |
| 2 | `torch.cuda.synchronize` | 0.589s | 12.8% | GPU 同步等待 |
| 3 | **`onnxruntime.run`（ReID）** | 0.442s | **9.6%** | 每辆车一次，CPU 推理 |
| 4 | `torch.conv2d` | 0.427s | 9.2% | 检测网络 |
| 5 | `non_max_suppression` | 0.259s | 5.6% | 后处理 |
| 6 | `torchvision.nms` | 0.152s | 3.3% | 后处理 |
| 7 | `resize` | 0.241s | 5.2% | 预处理 |
| 8 | 其余（跟踪/特征/IO） | ≈1.33s | 28.7% | Python |

### 关键观察

1. **解码占 1/4 以上**：`cv2.VideoCapture.read` 是单线程同步调用，GPU 在此期间空转。
2. **ReID 单次 111ms**：4 辆车耗时 0.442s。按 444 辆/66 分钟折算，ReID 累计约 **49 秒**。
   它跑在 **CPU ONNX** 上（CUDA EP 需要 CUDA 13，本机是 11.8）。
3. **`cuda_synchronize` 有 0.589s**：说明 CPU 在等 GPU，pipeline 是"串行等待"而非重叠。
4. 跟踪、特征提取等 Python 逻辑**没有进入前 8**，说明之前"瓶颈在 Python"的判断需要修正——
   真正的瓶颈是 **I/O 与推理**。

## 二、离 20 路还差多少

目标：20 路 × 15 FPS = **300 FPS**。

现状（开启解码预取）：

```text
端到端 process_video（含向量写出）：124.5–125.6 FPS ≈ 8.3–8.4 路
纯处理 process_frames（benchmark）：138.7 FPS ≈ 9.25 路
frame_skip=2（process_video）：190.6 FPS ≈ 12.7 路
```

距 20 路目标：默认差 2.2–2.4 倍，skip=2 差 **1.6 倍**。

> 两个口径要分清：`benchmark_throughput.py` 只算推理与跟踪（不含 CSV 写出），
> 数字会比 `process_video` 高约 10%。对外报"可承载路数"时用端到端口径更保守。

## 三、优化路线（按性价比排序）

### P0：ReID 移到 GPU（**批量在本机 CPU 上反而更慢**）

已实现 `ReIDExtractor.extract_batch()` 并实测（12 条真实裁剪，CPU ONNX）：

| batch | 逐条 | 批量 | 加速比 |
|---|---|---|---|
| 1 | 104.0 ms/条 | 92.6 ms/条 | 1.12× |
| 4 | 92.1 ms/条 | 103.2 ms/条 | **0.89×** |
| 12 | 98.7 ms/条 | 117.1 ms/条 | **0.84×** |
| 32 | 98.3 ms/条 | 127.6 ms/条 | **0.77×** |

**结论：CPU 上批量不但不省，单条成本还随 batch 增大**（32 条时涨到 128ms）。
这个模型是 ViT/CLIP 结构，CPU 上受内存带宽限制，批量无法摊薄。

因此 P0 的有效做法只剩**换 GPU 执行提供者**：

```python
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]   # 需 onnxruntime-gpu + CUDA 12/13
```

本机 CUDA 11.8 与 onnxruntime-gpu 1.31 要求的 CUDA 13 不匹配，**当前无法启用**。
`extract_batch` 保留在代码里：一旦换成 GPU provider，批量才有意义（届时需重新测）。

> 教训：性能优化必须实测。批量化在 GPU 上通常是常识，在 CPU 上却可能是负优化。

### P1：解码与推理重叠 —— **已实现，实测 +27~29%**

`src/utils/threaded_reader.py::ThreadedFrameReader`：独立线程解码 + 有界队列，
主线程按原顺序取帧（不改变帧顺序与时间戳），配置项 `pipeline.prefetch_frames`（默认 32）。

400 帧实测（best of 2）：

| batch | prefetch | FPS | 路数 |
|---|---|---|---|
| 16 | 0 | 98.0 | 6.5 |
| 16 | 32 | **124.5** | 8.3 |
| 32 | 0 | 97.2 | 6.5 |
| 32 | 32 | **125.6** | **8.4** |

**输出逐字节一致**（对比开启/关闭预取的 `vec_*.csv`）。

结论：解码曾占 25.6%，把它藏到推理后面后吞吐 **+27~29%**，端到端从 6.5 路提升到 **8.4 路**。
另外 batch 16 与 32 在开启预取后差异很小——瓶颈已经从"批量推理"转移到别处。

### P2：TensorRT / GPU ReID —— **本机环境不可行（已取证）**

#### 实测证据

```text
GPU        : RTX 3060 Laptop, driver 581.80
torch      : 2.5.1+cu118  -> CUDA 运行时 11.8
onnxruntime: 1.31.1，get_available_providers() 列出 CUDA/TensorRT
  ├─ CUDAExecutionProvider    -> 实际失败：
  │    "Require cuDNN 9.* and CUDA 13.*" -> 回退 CPU
  └─ TensorrtExecutionProvider -> 无 tensorrt 模块 -> 回退 CPU

pip install tensorrt  -> 解析到 tensorrt_cu13==11.4（CUDA 13 变体）
```

结论：**ORT 1.31 的 GPU provider 需要 CUDA 13 + cuDNN 9，而本机 torch 运行时是 CUDA 11.8。**
TensorRT 也只有 CUDA 13 的轮子。

#### 可行的替代路径（均需额外投入）

| 路径 | 代价 | 风险 |
|---|---|---|
| 装 CUDA 13 + cuDNN 9 运行时 | 约 1–2 GB | 与 torch 2.5.1+cu118 并存需配 PATH |
| 降级到 `onnxruntime-gpu==1.18.x`（CUDA 11.8） | 可能不兼容 numpy 2.x | 会替换现有可用的 CPU 版本 |
| YOLO 导出 TensorRT engine（`tensorrt_cu13`） | 约 1 GB + 重新导出 | 需 CUDA 13 运行时 |
| 把 ReID 换成 PyTorch 权重（torch 已有 CUDA 11.8） | 需 open_clip/timm 依赖 | 架构还原风险 |

#### 即使 GPU ReID 生效，也到不了 20 路

按剖析数据，ReID 占单帧时间 **9.6%**；换 GPU 后约降到 1%，整体约 **+9%**：

```text
8.4 路 × 1.09 ≈ 9.2 路
```

检测段（`conv2d` 9.2% + NMS 8.9% ≈ 18%）才是下一个大头，它必须靠 TensorRT。

#### 结论

**在本机现有环境下（CUDA 11.8 + ORT 1.31），P2 不可行**；20 路需要
「CUDA 13 运行时 + TensorRT engine + 跳帧读取」的组合改造，属于工程投入而非算法问题。

### P3：跳帧 —— 实测 skip=2 是甜点，skip=3 无额外收益

`pipeline.frame_skip` 是**配置项**，不需要改代码。2000 帧实测：

| skip | FPS | 路数 | 车辆数 | 平均轨迹点 | 平均速度 (m/s) |
|---|---|---|---|---|---|
| 1 | 129.5 | 8.6 | 21 | 48.0 | 8.57 |
| **2** | **190.6** | **12.7** | 21 | 24.6 | 8.72（+1.7%） |
| 3 | 191.2 | 12.7 | 21 | 16.4 | 8.97（+4.7%） |

#### 三个结论

1. **skip=2 提速 +47%**，且**车辆数完全不变**（21 → 21），说明没有漏检；
2. **skip=3 几乎没有进一步收益**（190.6 → 191.2）——因为当前实现是"读完帧再跳"，
   **解码成本不随 skip 下降**，跳过 2/3 的推理后瓶颈重新回到解码；
3. 速度估计随跳帧轻微上偏（+1.7% / +4.7%），仍远小于 ±5 km/h 容差；
   轨迹点数减半/减到 1/3，再往上跳会逼近 `min_trajectory_length=10`。

#### 建议

- **默认 `frame_skip: 1`**（完整轨迹，用于取证与速度精度）；
- **需要吞吐时用 `frame_skip: 2`**（12.7 路，精度代价可接受）；
- 不建议 skip≥3：收益消失、轨迹变稀。
- 若要跳过解码本身，需要改 `ThreadedFrameReader` 做**跳帧读取**（`cap.set` 或用 NVDEC 的 skip），
  那是另一项改造。

### P4：多进程分片

- 单卡已达显存上限时，按摄像头分片到多进程/多卡。

## 四、当前结论

| 配置 | 吞吐 | 路数 | 精度代价 |
|---|---|---|---|
| 基线（无优化） | 97–98 FPS | 6.5 | — |
| + 解码预取 | 125 FPS | 8.4 | 无（输出逐字节一致） |
| + 跳帧 skip=2 | **190.6 FPS** | **12.7** | 轨迹点减半、速度 +1.7% |
| + skip=3 | 191.2 FPS | 12.7 | 无额外收益，不建议 |
| + GPU ReID（若可行） | ≈207 FPS | ≈13.8 | 需 CUDA 13 运行时 |
| 目标 | 300 FPS | 20 | 需 TensorRT engine |

- 已落地两项：**解码重叠（P1，+28%）**、**跳帧（P3，+47%）**；
- P0（ReID 批量）实测为**负优化**，已排除；
- P2（TensorRT/GPU ReID）**本机不可行**，已取证（见上）；
- 20 路需要 CUDA 13 + TensorRT engine + 跳帧读取的组合改造。

## 五、复现方式

```powershell
# 吞吐基准（不同 batch）
.venv\python.exe scripts/benchmark_throughput.py --video <h264> --frames 400 --batch-size 32

# 本文件的剖析数据
.venv\python.exe -c "
import cProfile, pstats, sys
sys.path.insert(0,'.')
from src.pipeline.batch_processor import BatchProcessor
p = BatchProcessor(config_path='config/default.yaml', camera_config_path='config/camera/admin_in.yaml')
p.max_frames = 400
cProfile.run('p.process_video(r\"<h264>\", \"data/outputs/prof\")', 'prof.out')
pstats.Stats('prof.out').sort_stats('tottime').print_stats(15)
"
```

---

## 六、边缘部署评估（目标要求"可部署于边缘计算节点"）

### 6.1 实测资源占用

| 项 | 实测 | 说明 |
|---|---|---|
| 峰值 RSS | **1.97 GB** | 400 帧运行期间的进程内存峰值 |
| 峰值 GPU | **351 MB**（reserved 402 MB） | YOLOv8s 检测 + ReID 推理 |
| 运行时依赖体积 | 7.5 GB | 其中 torch 3.16 GB、onnxruntime 232 MB、scipy 111 MB、cv2 109 MB |
| 必需模型 | 21.5 MB | `models/yolov8s.pt` |
| 可选模型 | 329 MB | `models/vehicle_reid.onnx`（缺失时回退 HSV 直方图） |

**算力与内存门槛不高**：GPU 只要 351 MB、内存 2 GB —— Jetson Orin Nano（8 GB）这一档就能承载。

### 6.2 真正的门槛是软件栈（x86 → ARM64）

当前 `Dockerfile` 基于 `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`，是 **x86_64**。
Jetson 是 ARM64，需要换整套运行时：

| 依赖 | x86 现状 | Jetson 需要 |
|---|---|---|
| CUDA / cuDNN | 镜像自带 11.8 | JetPack 自带的版本 |
| PyTorch | PyPI `cu118` wheel | NVIDIA 提供的 **ARM64** wheel（JetPack 配套） |
| OpenCV | `opencv-python` (109 MB) | 通常从源码编译或 `opencv-python-headless` |
| onnxruntime | `onnxruntime-gpu` x86 | ARM64 构建，或改用 TensorRT 后端 |

**结论：代码可移植，构建链需要重做**——不是改几行代码，而是换一套 wheel 源。

### 6.3 已验证 / 未验证

| 项 | 状态 |
|---|---|
| x86 服务器 + 单卡运行 | ✅ 本仓库全部数字都在该环境实测 |
| Docker 镜像构建 | 🟡 Dockerfile 已提供，但未在 CI 中构建验证 |
| ARM64 / Jetson 运行 | ❌ **未验证**（当前无设备） |

### 6.4 边缘部署建议

1. **边缘端只跑到向量与事件**：与带宽指标一致（结构化结果 gzip 后 462 倍压缩），原始视频不出机房；
2. **ReID 权重可选**：不装可省 329 MB 磁盘，代价是跨机位能力退化到 HSV；
3. **算力不足时用 `frame_skip=2`**：换 47% 吞吐，代价是轨迹点减半、速度 +1.7%；
4. 上边缘前先跑 `python scripts/delivery_check.py` + `ci.ps1`，确认依赖完整。

> 说明：目标里的"可部署于边缘计算节点"在**代码层面**成立（纯 Python + 无 x86 专属依赖），
> 但在**构建层面**尚未验证，需要一台 Jetson 做实机适配。

---

## 七、多进程横向扩展实测（负面结果）

### 7.1 动机

单进程已测到 125.6 FPS（约 8.4 路），距离 20 路还差 2.4 倍。
在不动硬件的前提下，最直接的思路是**横向加进程**：每个进程处理一路，
线性堆叠。本轮用 `scripts/benchmark_multiprocess.py` 实测验证这个假设。

### 7.2 结果：2 路并发反而更慢

同一段 1080p 视频，每进程 900 帧，GPU 同一张 RTX 3060 Laptop（6 GB）：

| 配置 | 耗时 | 聚合 FPS | 折合路数（@15 FPS） | 峰值总内存 |
|---|---|---|---|---|
| 1 进程 × 900 帧 | 17.5 s | **51.4** | **3.42** | 2013 MB |
| 2 进程 × 900 帧 | 42.6 s | **42.3** | **2.82** | 3878 MB |

**扩展比 = 42.3 / 51.4 = 0.82×**——加一倍进程，总吞吐反而下降 18%。

### 7.3 原因

| 约束 | 实测 | 说明 |
|---|---|---|
| 内存 | 单进程 RSS **2.0 GB**；2 进程峰值 **3.9 GB** | 本机可用内存仅 3.4–3.9 GB，2 进程已经贴到上限，开始换页 |
| GPU | 单卡 6 GB，单进程占 351 MB | 检测争用同一张卡，核函数排队 |
| CPU | 16 逻辑核 | ultralytics 自身多线程，2 进程互相抢核 |

内存是主要瓶颈：进程一多就进入"内存紧张 → 换页 → 变慢"的正反馈。

### 7.4 对 20 路的含义

按实测数字外推，若要用多进程撑 20 路：

```text
内存需求 = 20 × 2.0 GB = 40 GB（本机 16 GB）
GPU      = 20 路检测争用，单卡吞吐上限约 125 FPS（≈8 路）
```

**结论：多进程不是通往 20 路的可行路径。** 20 路需要的是：

1. **单进程吞吐提升**（TensorRT engine + C++ 前处理），把 125 FPS 推到 300+；
2. 或**多卡**（每卡 8 路，3 卡可到 24 路）；
3. 或**降指标**到 8–12 路（当前水平）。

### 7.5 复现

```powershell
.venv\python.exe scripts/benchmark_multiprocess.py `
  --video data/outputs/h264/admin_in/clip_08_short.h264 `
  --workers 1,2 --frames 900
```

> 注意：脚本会先检查内存，预计需求超过物理内存 80% 时直接跳过该档，
> 避免跑到一半被系统 OOM。首次运行含冷启动开销（模型/CUDA 初始化约 20–25 s），
> 比较扩展比时应保证各档帧数一致。