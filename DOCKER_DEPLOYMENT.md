# Vision-Restore AI - Docker Deployment Guide

> **Professional Image Restoration & Upscaling as a Service**

This guide covers deploying Vision-Restore AI as a containerized REST API service using Docker and FastAPI.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Deployment Options](#deployment-options)
- [API Documentation](#api-documentation)
- [Usage Examples](#usage-examples)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

Vision-Restore AI can be deployed as a REST API service, providing image enhancement capabilities via HTTP endpoints. This enables:

- **Remote Processing**: Process images from any client (web, mobile, desktop)
- **Scalability**: Deploy multiple containers for load balancing
- **Integration**: Easy integration with existing systems
- **Centralized Resources**: Share expensive GPU resources across multiple clients

### Key Features

✅ **RESTful API** - Standard HTTP endpoints for image processing  
✅ **Docker Support** - Both CPU and GPU (NVIDIA CUDA) configurations  
✅ **Multi-Model Support** - HAT, SwinIR, Real-ESRGAN, GFPGAN, CodeFormer  
✅ **Batch Processing** - Process multiple images in one request  
✅ **Auto Documentation** - Interactive API docs at `/docs`  
✅ **Health Checks** - Built-in health monitoring for orchestration  

---

## 🏗️ Architecture

```
┌─────────────┐          ┌──────────────────┐         ┌──────────────┐
│   Client    │  HTTP    │   FastAPI App    │  Model  │  AI Engines  │
│  (Python,   │◄────────►│  (app/api.py)    │◄───────►│  (PyTorch)   │
│   curl,     │  POST    │                  │  Calls  │              │
│   Browser)  │  /enhance│  Docker Container│         │  GPU/CPU     │
└─────────────┘          └──────────────────┘         └──────────────┘
```

### Components

- **FastAPI Application** (`app/api.py`) - REST API server
- **Enhancement Orchestrator** (`vision_restore/`) - Core processing engine
- **Docker Container** - Isolated runtime environment
- **AI Models** - Downloaded on first use (cached in `/app/models`)

---

## 📦 Prerequisites

### Required

- **Docker** 20.10+ ([Install Docker](https://docs.docker.com/get-docker/))
- **Docker Compose** 1.29+ (included with Docker Desktop)
- **4GB+ RAM** (8GB+ recommended)
- **10GB+ Disk Space** (for models and cache)

### Optional (for GPU acceleration)

- **NVIDIA GPU** with CUDA 11.8+ support
- **nvidia-docker2** ([Install Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
- **NVIDIA Driver** 525+ ([Download Drivers](https://www.nvidia.com/download/index.aspx))

### Verify Installation

```bash
# Check Docker
docker --version
docker compose version

# Check NVIDIA GPU (if using GPU)
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

---

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/vision-restore-ai.git
cd vision-restore-ai
```

### 2. Build and Start Service

**Option A: CPU-only (works on any system)**

```bash
docker-compose --profile cpu up -d
```

**Option B: GPU-accelerated (requires NVIDIA GPU)**

```bash
docker-compose --profile gpu up -d
```

### 3. Verify Service

```bash
# Check container status
docker-compose ps

# Check logs
docker-compose logs -f

# Test health endpoint
curl http://localhost:8000/health
```

### 4. Access API Documentation

Open in your browser:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 5. Test Image Enhancement

```bash
# Using curl
curl -X POST "http://localhost:8000/api/v1/enhance" \
  -F "file=@input.jpg" \
  -F "scale=4" \
  -F "upscale_model=auto" \
  -o output.png

# Using Python client
python examples/client_example.py
```

---

## 🔧 Deployment Options

### Development Mode

Quick setup for testing and development:

```bash
# CPU mode with live logs
docker-compose --profile cpu up

# GPU mode
docker-compose --profile gpu up
```

### Production Mode

Optimized for production deployment:

```bash
# Build images
docker-compose build

# Start in detached mode
docker-compose --profile cpu up -d

# View logs
docker-compose logs -f vision-restore-cpu

# Stop services
docker-compose down
```

### Custom Configuration

Create a `.env` file for environment variables:

```bash
# .env
VISION_RESTORE_DEVICE=cuda
VISION_RESTORE_LOG_LEVEL=INFO
VISION_RESTORE_PORT=8000
VISION_RESTORE_WORKERS=1
```

Update `docker-compose.yml`:

```yaml
environment:
  - VISION_RESTORE_DEVICE=${VISION_RESTORE_DEVICE:-cpu}
  - VISION_RESTORE_LOG_LEVEL=${VISION_RESTORE_LOG_LEVEL:-INFO}
ports:
  - "${VISION_RESTORE_PORT:-8000}:8000"
```

### Scaling Multiple Instances

```bash
# Start 3 instances with load balancer
docker-compose --profile cpu up -d --scale vision-restore-cpu=3

# Use nginx for load balancing
docker-compose --profile cpu --profile proxy up -d
```

---

## 📚 API Documentation

### Base URL

```
http://localhost:8000
```

### Endpoints

#### 1. Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "vision-restore-ai"
}
```

#### 2. Service Status

```http
GET /api/v1/status
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "available_models": {
    "upscale_models": ["realesrgan", "swinir", "hat"],
    "face_models": ["gfpgan", "codeformer"]
  },
  "gpu_available": true,
  "gpu_device": "NVIDIA GeForce RTX 3090"
}
```

#### 3. Get Available Models

```http
GET /api/v1/models
```

**Response:**
```json
{
  "upscale_models": ["realesrgan", "swinir", "hat"],
  "face_models": ["gfpgan", "codeformer"]
}
```

#### 4. Enhance Single Image

```http
POST /api/v1/enhance
Content-Type: multipart/form-data
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| file | file | required | Image file (jpg, png, webp, etc.) |
| scale | int | 4 | Upscale factor (2, 4, or 8) |
| upscale_model | string | "auto" | Model: auto, realesrgan, swinir, hat |
| face_model | string | "auto" | Face model: auto, gfpgan, codeformer, none |
| face_enhance | bool | true | Enable face enhancement |
| codeformer_fidelity | float | 0.7 | CodeFormer fidelity (0-1) |
| enable_denoise | bool | true | Enable denoising |
| enable_sharpen | bool | true | Enable sharpening |
| denoise_strength | float | 0.3 | Denoise strength (0-1) |
| sharpen_amount | float | 0.2 | Sharpen amount (0-1) |
| output_format | string | "png" | Output format: png, jpg, webp |
| quality | int | 95 | Output quality (1-100) |
| return_base64 | bool | false | Return base64 instead of binary |

**Response (binary image):**
```
Content-Type: image/png
X-Processing-Time: 5.234
X-Output-Width: 2048
X-Output-Height: 2048
```

**Response (JSON with base64):**
```json
{
  "success": true,
  "message": "Image enhanced successfully",
  "data": "base64_encoded_image_data...",
  "output_size": {"width": 2048, "height": 2048},
  "processing_time": 5.234,
  "model_used": "swinir"
}
```

#### 5. Batch Enhance Images

```http
POST /api/v1/batch
Content-Type: multipart/form-data
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| files | file[] | required | Multiple image files |
| scale | int | 4 | Upscale factor |
| upscale_model | string | "auto" | Upscaling model |
| face_model | string | "auto" | Face restoration model |
| face_enhance | bool | true | Enable face enhancement |
| output_format | string | "png" | Output format |
| quality | int | 95 | Output quality |

**Response:**
```json
{
  "success": true,
  "message": "Processed 3/3 images successfully",
  "processing_time": 15.678,
  "results": [
    {
      "filename": "image1.jpg",
      "success": true,
      "data": "base64_encoded_image...",
      "output_size": {"width": 2048, "height": 2048}
    },
    {
      "filename": "image2.jpg",
      "success": true,
      "data": "base64_encoded_image...",
      "output_size": {"width": 1920, "height": 1080}
    }
  ]
}
```

---

## 💻 Usage Examples

### cURL Examples

**Basic Enhancement:**
```bash
curl -X POST "http://localhost:8000/api/v1/enhance" \
  -F "file=@photo.jpg" \
  -F "scale=4" \
  -o enhanced.png
```

**Portrait with CodeFormer:**
```bash
curl -X POST "http://localhost:8000/api/v1/enhance" \
  -F "file=@portrait.jpg" \
  -F "scale=4" \
  -F "upscale_model=realesrgan" \
  -F "face_model=codeformer" \
  -F "codeformer_fidelity=0.7" \
  -F "face_enhance=true" \
  -o portrait_enhanced.png
```

**Get Base64 Response:**
```bash
curl -X POST "http://localhost:8000/api/v1/enhance" \
  -F "file=@image.jpg" \
  -F "scale=2" \
  -F "return_base64=true" \
  | jq '.data' -r | base64 -d > output.png
```

**Batch Processing:**
```bash
curl -X POST "http://localhost:8000/api/v1/batch" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg" \
  -F "scale=4" \
  | jq '.'
```

### Python Client

See `examples/client_example.py` for complete examples:

```python
from examples.client_example import VisionRestoreClient

client = VisionRestoreClient("http://localhost:8000")

# Check status
status = client.check_status()
print(f"GPU Available: {status['gpu_available']}")

# Enhance image
client.enhance_image(
    image_path="input.jpg",
    output_path="output.png",
    scale=4,
    upscale_model="swinir",
    face_enhance=True,
)

# Batch processing
client.batch_enhance(
    image_paths=["img1.jpg", "img2.jpg", "img3.jpg"],
    output_dir="enhanced_images",
    scale=4,
)
```

### JavaScript/TypeScript

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('scale', '4');
formData.append('upscale_model', 'auto');

const response = await fetch('http://localhost:8000/api/v1/enhance', {
  method: 'POST',
  body: formData,
});

const blob = await response.blob();
const url = URL.createObjectURL(blob);
imageElement.src = url;
```

---

## ⚡ Performance Tuning

### GPU Optimization

**1. Adjust Tile Size** (lower = less VRAM, slower)

Edit `vision_restore/core/config.py`:
```python
tile_size: int = 512  # Default, try 256 for low VRAM or 1024 for high VRAM
```

**2. Batch Size**

For batch processing, process in chunks:
```python
# Process 5 images at a time instead of all at once
for i in range(0, len(images), 5):
    batch = images[i:i+5]
    client.batch_enhance(batch, ...)
```

**3. Model Selection**

- **Fastest**: `realesrgan` (1-2 seconds per 4x upscale)
- **Balanced**: `swinir` (3-5 seconds)
- **Highest Quality**: `hat` (5-10 seconds)

### CPU Optimization

**1. Disable Heavy Processing**
```bash
curl -X POST "http://localhost:8000/api/v1/enhance" \
  -F "file=@image.jpg" \
  -F "scale=2" \
  -F "face_enhance=false" \
  -F "enable_denoise=false" \
  -o output.png
```

**2. Use Real-ESRGAN Only**
```bash
-F "upscale_model=realesrgan"
-F "face_model=none"
```

### Scaling

**Horizontal Scaling:**
```bash
# Start multiple workers
docker-compose up -d --scale vision-restore-cpu=4

# Add nginx load balancer
docker-compose --profile proxy up -d
```

**Resource Limits:**

Update `docker-compose.yml`:
```yaml
deploy:
  resources:
    limits:
      cpus: '4.0'
      memory: 8G
    reservations:
      memory: 4G
```

---

## 🔍 Troubleshooting

### Common Issues

#### 1. Out of Memory (GPU)

**Symptoms:** CUDA OOM error, container crashes

**Solutions:**
- Reduce tile size in config
- Use smaller scale (2x instead of 4x)
- Process smaller images
- Add more GPU memory or use CPU mode

```bash
# Switch to CPU mode
docker-compose --profile cpu up -d
```

#### 2. Slow Processing (CPU Mode)

**Symptoms:** Very slow enhancement times

**Solutions:**
- Use GPU mode if available
- Disable face enhancement
- Use faster model (realesrgan)
- Process smaller scale (2x)

#### 3. Models Not Downloading

**Symptoms:** "Model not found" errors

**Solutions:**
- Check internet connection
- Manually download models to `./models/` directory
- Check disk space
- Verify file permissions

```bash
# Check model directory
docker-compose exec vision-restore-cpu ls -la /app/models/

# Download models manually
docker-compose exec vision-restore-cpu python -c "
from vision_restore.engine.orchestrator import EnhancementOrchestrator
orch = EnhancementOrchestrator()
orch.download_models(face_enhance=True)
"
```

#### 4. Container Won't Start

**Symptoms:** Container exits immediately

**Solutions:**
```bash
# Check logs
docker-compose logs vision-restore-cpu

# Test Python imports
docker-compose run --rm vision-restore-cpu python -c "import vision_restore"

# Rebuild image
docker-compose build --no-cache
docker-compose up -d
```

#### 5. API Returns 500 Error

**Symptoms:** HTTP 500 Internal Server Error

**Solutions:**
- Check logs: `docker-compose logs -f`
- Verify image format (jpg, png, webp)
- Check file size (< 50MB recommended)
- Verify parameters are valid

### Debugging

**Enable Debug Logging:**
```yaml
# docker-compose.yml
environment:
  - VISION_RESTORE_LOG_LEVEL=DEBUG
```

**Access Container Shell:**
```bash
docker-compose exec vision-restore-cpu bash

# Inside container
python -c "from vision_restore import __version__; print(__version__)"
ls -la /app/models/
```

**Monitor Resources:**
```bash
# CPU/Memory usage
docker stats

# GPU usage (if using GPU)
watch -n 1 nvidia-smi
```

---

## 📞 Support

- **Documentation**: [Full API Docs](http://localhost:8000/docs)
- **GitHub Issues**: [Report Issues](https://github.com/yourusername/vision-restore-ai/issues)
- **Examples**: See `examples/` directory

---

## 📄 License

MIT License - See [LICENSE](../LICENSE) file for details.

---

**Next Steps:**

1. ✅ Service running → Test with `examples/test_api.sh`
2. 🔧 Optimize settings → Adjust for your hardware
3. 🚀 Deploy to production → Use proper reverse proxy (nginx)
4. 📊 Monitor performance → Set up logging and metrics
5. 🔒 Secure API → Add authentication if needed
