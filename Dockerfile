FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/requirements.txt
COPY requirements.linux.txt /workspace/requirements.linux.txt

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r /workspace/requirements.txt && \
    python3 -m pip install -r /workspace/requirements.linux.txt

COPY . /workspace

CMD ["python3", "train.py", "--config", "configs/sft_baseline.yaml", "--dry-run", "--max-samples", "8"]
