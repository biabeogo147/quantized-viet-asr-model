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
    aimet-onnx \
    numpy \
    onnx \
    onnxruntime \
    onnxsim

WORKDIR /workspace
ENV PYTHONPATH=/workspace/src
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=5).read()"

CMD ["python3", "-m", "quantize.aimet_service", "--host", "0.0.0.0", "--port", "8080"]
