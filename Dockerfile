FROM python:3.11-slim

# System deps for OpenCV + ONNX
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces requires user 1000
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    U2NET_HOME=/home/user/.u2net \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    MPLCONFIGDIR=/tmp/mpl

WORKDIR /home/user/app

# Install Python deps first (better Docker layer caching)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

# Copy app files
COPY --chown=user . .

# HF Spaces requires port 7860
EXPOSE 7860

# Tune env for HF free tier (16GB RAM, 2 vCPU)
ENV BG_MAX_CONCURRENT=2 \
    BG_MAX_LOADED_MODELS=3 \
    BG_PARALLEL_INFER=2 \
    BG_QUALITY=premium \
    OMP_NUM_THREADS=2 \
    ORT_NUM_THREADS=2

CMD ["uvicorn", "bg_remover_api_v5:app", "--host", "0.0.0.0", "--port", "7860"]