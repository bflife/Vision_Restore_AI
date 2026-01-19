# Vision-Restore AI - Docker & FastAPI 快速开始

> **🚀 5分钟快速部署指南**

## 📚 完整文档

- **[Docker 部署指南](DOCKER_DEPLOYMENT.md)** ⭐ - 完整的 Docker 部署文档（655行）
- **[API 功能文档](API_FEATURES.md)** - API 功能概述
- **[项目总结](PROJECT_SUMMARY.md)** - 完整的项目分析和实现总结

---

## ⚡ 快速开始（3步）

### 1️⃣ 启动服务

```bash
# CPU 模式（适用于任何系统）
docker-compose --profile cpu up -d

# 或 GPU 模式（需要 NVIDIA GPU）
docker-compose --profile gpu up -d
```

### 2️⃣ 验证服务

```bash
# 检查服务状态
curl http://localhost:8000/health

# 访问 API 文档
# 浏览器打开: http://localhost:8000/docs
```

### 3️⃣ 测试图像增强

```bash
# 使用 curl 测试
curl -X POST "http://localhost:8000/api/v1/enhance" \
  -F "file=@input.jpg" \
  -F "scale=4" \
  -o output_enhanced.png

# 或使用 Python 客户端
python examples/client_example.py
```

---

## 📖 详细文档链接

### 🐳 [Docker 部署指南](DOCKER_DEPLOYMENT.md)

完整的 Docker 部署文档，包含：

- **环境准备** - 安装 Docker、配置 GPU 支持
- **部署选项** - 开发模式、生产模式、集群部署
- **API 文档** - 所有端点的详细说明
- **使用示例** - cURL、Python、JavaScript 客户端
- **性能调优** - GPU 优化、资源限制、扩展配置
- **故障排除** - 常见问题解决方案

### 📄 主要内容：

#### 1. 快速启动
```bash
docker-compose --profile cpu up -d
```

#### 2. API 端点
- `GET /health` - 健康检查
- `GET /api/v1/status` - 服务状态
- `POST /api/v1/enhance` - 图像增强
- `POST /api/v1/batch` - 批量处理

#### 3. Python 客户端
```python
from examples.client_example import VisionRestoreClient

client = VisionRestoreClient("http://localhost:8000")
client.enhance_image(
    image_path="input.jpg",
    output_path="output.png",
    scale=4,
)
```

#### 4. cURL 示例
```bash
# 基础增强
curl -X POST "http://localhost:8000/api/v1/enhance" \
  -F "file=@photo.jpg" \
  -F "scale=4" \
  -o enhanced.png

# 高级设置
curl -X POST "http://localhost:8000/api/v1/enhance" \
  -F "file=@portrait.jpg" \
  -F "scale=4" \
  -F "upscale_model=swinir" \
  -F "face_model=codeformer" \
  -F "codeformer_fidelity=0.7" \
  -o enhanced.png
```

---

## 🔧 配置选项

### 环境变量

创建 `.env` 文件：

```bash
VISION_RESTORE_DEVICE=cuda        # cuda 或 cpu
VISION_RESTORE_LOG_LEVEL=INFO     # DEBUG, INFO, WARNING, ERROR
VISION_RESTORE_PORT=8000          # API 端口
```

### Docker Compose 配置

```yaml
services:
  vision-restore-cpu:
    profiles: [cpu]
    ports: ["8000:8000"]
    environment:
      - VISION_RESTORE_DEVICE=cpu
```

---

## 📊 API 端点概览

| 端点 | 方法 | 功能 |
|------|------|------|
| `/` | GET | API 信息 |
| `/health` | GET | 健康检查 |
| `/api/v1/status` | GET | 服务状态和可用模型 |
| `/api/v1/models` | GET | 获取可用 AI 模型 |
| `/api/v1/enhance` | POST | 单图像增强 |
| `/api/v1/batch` | POST | 批量图像处理 |

---

## 🎯 使用场景

1. **远程图像处理** - 部署到云端，客户端通过 API 调用
2. **批量处理系统** - 自动化图像增强流程
3. **微服务架构** - 作为独立服务集成到现有系统
4. **SaaS 应用** - 为 Web/Mobile 应用提供后端 API

---

## 📈 性能参考

| 配置 | 输入尺寸 | 放大倍数 | 处理时间 |
|------|----------|----------|----------|
| CPU (8核) | 512×512 | 4x | ~30-60秒 |
| RTX 3090 | 512×512 | 4x | ~2-5秒 |
| RTX 3090 | 1920×1080 | 4x | ~10-20秒 |

---

## 🔍 目录结构

```
vision-restore-ai/
├── app/
│   ├── __init__.py           # API 包初始化
│   └── api.py                # FastAPI 主应用 (600+ 行)
├── examples/
│   ├── client_example.py     # Python 客户端示例
│   └── test_api.sh           # Bash API 测试脚本
├── Dockerfile                # Docker 多阶段构建
├── docker-compose.yml        # Docker Compose 配置
├── requirements-api.txt      # API 依赖
├── DOCKER_DEPLOYMENT.md      # ⭐ 完整部署指南 (655行)
├── API_FEATURES.md           # API 功能文档
└── QUICK_START.md            # 本文档
```

---

## 🆘 常见问题

### Q: 如何启动 GPU 模式？

```bash
# 确保安装了 nvidia-docker2
docker-compose --profile gpu up -d
```

### Q: 如何查看日志？

```bash
docker-compose logs -f
```

### Q: 如何访问 API 文档？

浏览器打开：http://localhost:8000/docs

### Q: 如何停止服务？

```bash
docker-compose down
```

### Q: 如何扩展多个实例？

```bash
docker-compose up -d --scale vision-restore-cpu=3
```

---

## 📞 获取帮助

- **完整文档**: [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md)
- **API 文档**: http://localhost:8000/docs
- **客户端示例**: [examples/](examples/)
- **GitHub Issues**: https://github.com/bflife/Vision_Restore_AI/issues

---

## ✅ 检查清单

部署前确认：

- [ ] 已安装 Docker 和 Docker Compose
- [ ] 如使用 GPU，已安装 nvidia-docker2
- [ ] 已克隆项目仓库
- [ ] 已阅读 [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md)

部署后验证：

- [ ] 容器正常运行 (`docker-compose ps`)
- [ ] 健康检查通过 (`curl http://localhost:8000/health`)
- [ ] API 文档可访问 (http://localhost:8000/docs)
- [ ] 能够成功处理测试图像

---

## 🎉 开始使用

```bash
# 1. 克隆项目
git clone https://github.com/bflife/Vision_Restore_AI.git
cd Vision_Restore_AI

# 2. 启动服务
docker-compose --profile cpu up -d

# 3. 测试 API
curl http://localhost:8000/health

# 4. 查看文档
# 浏览器打开: http://localhost:8000/docs

# 5. 处理图像
curl -X POST "http://localhost:8000/api/v1/enhance" \
  -F "file=@your_image.jpg" \
  -F "scale=4" \
  -o enhanced_image.png
```

**祝您使用愉快！** 🚀

---

**💡 提示**: 完整的部署指南、故障排除和高级配置请参考 [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md)
