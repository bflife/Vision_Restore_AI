# Vision-Restore AI - Docker Image
# Multi-stage build for optimized image size

# ============================================================================
# Stage 1: Base Image with Python and Dependencies
# ============================================================================
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# ============================================================================
# Stage 2: Dependencies Installation
# ============================================================================
FROM base as dependencies

# Copy requirements
COPY requirements.txt .
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install fastapi uvicorn[standard] python-multipart

# ============================================================================
# Stage 3: Application
# ============================================================================
FROM dependencies as application

# Copy application code
COPY vision_restore/ /app/vision_restore/
COPY app/ /app/app/
COPY run_gui.py /app/

# Create directories for models and cache
RUN mkdir -p /app/models /app/cache /app/uploads /app/outputs

# Set permissions
RUN chmod -R 755 /app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Default command
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# ============================================================================
# Stage 4: GPU-enabled Image (CUDA)
# ============================================================================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as gpu

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

# Install Python 3.10 and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
COPY pyproject.toml .

RUN pip install --upgrade pip setuptools wheel && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 && \
    pip install -r requirements.txt && \
    pip install fastapi uvicorn[standard] python-multipart

# Copy application
COPY vision_restore/ /app/vision_restore/
COPY app/ /app/app/
COPY run_gui.py /app/

# Create directories
RUN mkdir -p /app/models /app/cache /app/uploads /app/outputs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
