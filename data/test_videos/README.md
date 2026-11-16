# 输入视频放置目录

`docker-compose.yml` 默认把这个目录挂载到容器的 `/videos`：

```yaml
- ${VIDEO_DIR:-./data/test_videos}:/videos:ro
```

## 使用方式

1. 把待处理的录像（`.h264` / `.mp4` / `.avi`）放进本目录；
2. 直接 `docker compose up --build`（无需改配置）；或
3. 用环境变量指向别处：`$env:VIDEO_DIR = "F:\A区录像 新"`。

## 注意

**仓库里只自带 `frame_0001.jpg` 一张测试帧，没有任何视频文件**——
这是为了让本目录在 `git clone` 后存在（否则挂载会失败）。

若直接 `docker compose up` 而不放视频，管道会报"没有找到输入视频"。

## 现场 `.mbf` 录像

本系统支持的 `.mbf` 是海康私有格式，需要先转换：

```powershell
.venv\python.exe scripts\extract_mbf.py --input "F:\A区录像 新\行政区车道进" --output data\outputs\h264\admin_in
```

转换出的 `.h264` 再放进本目录，或直接把 `VIDEO_DIR` 指向上面的输出目录。
