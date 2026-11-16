# 部署说明

## 一、前置条件

- Docker 24+（GPU 场景需安装 NVIDIA Container Toolkit）
- 模型权重：`models/yolov8s.pt`（约 22 MB）需在构建前放到 `models/` 目录
- GPU：NVIDIA 显卡，CUDA 11.8 驱动（镜像已内置 CUDA 运行时）

## 二、Docker 运行

```bash
# 构建
docker build -t vehicle-tracking-system:latest .

# 运行（把宿主机录像目录挂载到 /videos）
docker run --gpus all --rm \
  -v "$PWD/data:/app/data" \
  -v "/path/to/recordings:/videos:ro" \
  vehicle-tracking-system:latest \
  --input /videos --camera admin_in --output /app/data/outputs
```

使用 Compose（需 Docker Compose 插件）：

```bash
export VIDEO_DIR=/path/to/recordings
docker compose up --build
```

> **默认挂载路径**：`docker-compose.yml` 里是 `${VIDEO_DIR:-./data/test_videos}`。
> 该目录在仓库里自带一张测试帧（`frame_0001.jpg`），**但没有视频文件**——
> 直接 `docker compose up` 而不放视频，管道会报"没有找到输入视频"。
> 把录像放进 `data/test_videos/`，或用 `VIDEO_DIR` 指向别处即可。

或 Windows PowerShell：

```powershell
$env:VIDEO_DIR = "F:\A区录像 新"
docker compose up --build
```

> 当前环境若提示 `docker: unknown command: docker compose`，说明只装了 Docker CLI，
> 需要单独安装 Docker Compose 插件，或直接用上面的 `docker run`。

## 三、无 Docker 的本地运行

```powershell
.venv\python.exe scripts\run_full_pipeline.py `
  --input data/outputs/h264/admin_in `
  --camera admin_in `
  --output data/outputs/full_run
```

或使用包装脚本：

```powershell
.\scripts\start.ps1 -Input "data/outputs/h264/admin_in" -Camera admin_in
```

## 四、目录约定

| 容器路径 | 用途 |
|---|---|
| `/videos` | 输入录像（只读挂载） |
| `/app/data/outputs` | 输出：轨迹向量、告警、图谱、风险报告 |
| `/app/config` | 摄像头与站点配置 |

## 五、镜像内容说明（静态审查结论）

> **构建状态**：`Dockerfile` 与 `docker-compose.yml` 已提供，但**尚未实际构建验证**
> （开发机 Docker daemon 未运行）。下表是静态审查结论。

| 项 | 情况 |
|---|---|
| 基础镜像 | `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`（x86_64，只含运行时，不含 nvcc） |
| Python | 3.10（Ubuntu 22.04 自带） |
| torch | 单独先装 `2.5.1+cu118`，再装 requirements（`torch>=2.0.0` 已被满足，不会覆盖） |
| 系统库 | `ffmpeg`、`libgl1`、`libglib2.0-0`，覆盖 cv2 的运行依赖 |
| `models/yolov8s.pt` | ✅ **会打进镜像**（21.5 MB） |
| `models/vehicle_reid.onnx` | ❌ **不会打进镜像**——`.dockerignore` 排除了 `*.onnx` |
| `data/`、`tests/`、`web/` | ❌ 不打包（compose 用 volume 挂载 `data/`） |

### 重要：镜像内没有 ReID 权重

由于 `.dockerignore` 含 `*.onnx`，容器启动时 `models/vehicle_reid.onnx` 不存在，
系统会**自动回退到 HSV 直方图**（见 `reid_extractor.py`），跨机位 Re-ID 能力随之退化。

如果需要镜像内带 ReID：

```dockerfile
# 方案 A：改 .dockerignore，把 *.onnx 改成只忽略其他 onnx
!models/vehicle_reid.onnx

# 方案 B：构建时挂载/拷贝（镜像会大 329 MB）
docker build -t vts:reid .
docker run -v "$PWD/models:/app/models" ...
```

> 329 MB 的权重让镜像膨胀明显，因此默认不打包；生产部署建议用 volume 挂载。

## 六、常见问题

1. **`models/yolov8s.pt` 缺失**：ultralytics 会尝试联网下载；离线环境需提前放入。
2. **GPU 不可用**：确认 `docker run --gpus all` 与 NVIDIA Container Toolkit；CPU 模式可加 `--device -1`。
3. **显存不足**：把 `config/default.yaml` 的 `pipeline.batch_size` 从 16 降到 4–8。
4. **`.mbf` 录像**：先运行 `python scripts/extract_mbf.py` 转换为 `.h264` 再处理。
