FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        ca-certificates \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.13.0+cpu \
    torchvision==0.28.0+cpu
RUN python3 -m pip install \
    aimet-onnx==2.31.0 \
    numpy==1.26.4 \
    onnx==1.20.0 \
    onnxruntime==1.22.0 \
    onnxsim==0.4.36 \
    torch==2.13.0+cpu \
    torchvision==0.28.0+cpu

WORKDIR /workspace
ENV PYTHONPATH=/workspace/src
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=5).read()"

CMD ["python3", "-m", "model_pipeline.models.aimet_service", "--host", "0.0.0.0", "--port", "8080"]
