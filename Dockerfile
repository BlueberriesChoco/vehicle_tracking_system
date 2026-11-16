FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# 先装 CUDA 11.8 版 torch，避免 requirements 拉到 CUDA 12 版本
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu118 \
    && python -m pip install -r requirements.txt

COPY . .

# 模型权重需在构建前放到 models/ 下（不随镜像层缓存）
RUN test -f models/yolov8s.pt || echo "WARNING: models/yolov8s.pt missing"

ENTRYPOINT ["python", "scripts/run_full_pipeline.py"]
CMD ["--help"]
