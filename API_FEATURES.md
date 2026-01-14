# FastAPI API 功能要求文档

本文档添加了 FastAPI 远程 API 调用功能，使 Vision-Restore AI 可以作为远程服务部署。

## 新增文件

### 1. FastAPI 应用 (`app/api.py`)

完整的 REST API 实现，提供以下端点：

- **GET /** - 服务根端点，返回 API 信息
- **GET /health** - 健康检查端点（用于容器编排）
- **GET /api/v1/status** - 获取服务状态和可用模型
- **GET /api/v1/models** - 获取可用的 AI 模型列表
- **POST /api/v1/enhance** - 单图像增强处理
- **POST /api/v1/batch** - 批量图像处理

### 2. Docker 配置

- **Dockerfile** - 多阶段构建，支持 CPU 和 GPU 模式
- **docker-compose.yml** - Docker Compose 配置，包含：
  - CPU-only 服务配置
  - GPU 加速服务配置（NVIDIA CUDA）
  - 可选的 Nginx 反向代理

### 3. 客户端示例

- **examples/client_example.py** - Python 客户端示例
- **examples/test_api.sh** - Bash 测试脚本

### 4. 部署文档

- **DOCKER_DEPLOYMENT.md** - 完整的 Docker 部署指南

## 功能特性

### API 功能

1. **图像增强 API**
   - 支持多种上采样模型（HAT, SwinIR, Real-ESRGAN）
   - 支持面部修复模型（GFPGAN, CodeFormer）
   - 可配置的预处理和后处理
   - 支持 2x、4x、8x 放大倍数
   - 支持多种输出格式（PNG, JPG, WebP）

2. **批量处理**
   - 一次请求处理多张图片
   - 并行处理提高效率
   - Base64 编码返回结果

3. **服务监控**
   - 健康检查端点
   - 服务状态查询
   - GPU 可用性检测
   - 处理时间统计

### Docker 支持

1. **多模式部署**
   - CPU 模式（适用于任何系统）
   - GPU 模式（NVIDIA CUDA 加速）
   - 可扩展配置

2. **容器优化**
   - 多阶段构建减小镜像大小
   - 健康检查自动重启
   - 持久化模型缓存
   - 资源限制配置

### 集成特性

1. **RESTful API**
   - 标准 HTTP 端点
   - JSON 和二进制响应
   - CORS 支持
   - 自动 API 文档（Swagger UI）

2. **客户端支持**
   - Python 客户端类
   - cURL 示例
   - Shell 脚本测试工具

## 使用示例

### 快速启动

```bash
# CPU 模式
docker-compose --profile cpu up -d

# GPU 模式（需要 NVIDIA GPU）
docker-compose --profile gpu up -d

# 访问 API 文档
http://localhost:8000/docs
```

### API 调用示例

```bash
# 基础图像增强
curl -X POST "http://localhost:8000/api/v1/enhance" \
  -F "file=@input.jpg" \
  -F "scale=4" \
  -o output.png

# 高级设置
curl -X POST "http://localhost:8000/api/v1/enhance" \
  -F "file=@portrait.jpg" \
  -F "scale=4" \
  -F "upscale_model=swinir" \
  -F "face_model=codeformer" \
  -F "codeformer_fidelity=0.7" \
  -o enhanced.png
```

### Python 客户端

```python
from examples.client_example import VisionRestoreClient

client = VisionRestoreClient("http://localhost:8000")

# 增强图像
client.enhance_image(
    image_path="input.jpg",
    output_path="output.png",
    scale=4,
    upscale_model="auto",
)

# 批量处理
client.batch_enhance(
    image_paths=["img1.jpg", "img2.jpg"],
    output_dir="outputs",
    scale=4,
)
```

## 技术栈

- **Web 框架**: FastAPI 0.100+
- **ASGI 服务器**: Uvicorn
- **容器化**: Docker, Docker Compose
- **AI 框架**: PyTorch 2.0+
- **图像处理**: OpenCV, Pillow
- **API 文档**: Swagger UI, ReDoc

## 性能优化

1. **GPU 加速**: 使用 NVIDIA CUDA 提供 10-50x 性能提升
2. **模型缓存**: 自动缓存下载的 AI 模型
3. **智能切片**: 自动切片大图像防止内存溢出
4. **并行处理**: 批量请求并行处理
5. **水平扩展**: 支持多容器负载均衡

## 部署建议

### 开发环境
```bash
docker-compose --profile cpu up
```

### 生产环境
```bash
# 构建优化镜像
docker-compose build

# 后台运行
docker-compose --profile gpu up -d

# 配置 Nginx 反向代理
docker-compose --profile proxy up -d

# 水平扩展
docker-compose up -d --scale vision-restore-cpu=3
```

## 监控和日志

```bash
# 查看日志
docker-compose logs -f

# 查看容器状态
docker-compose ps

# 资源监控
docker stats

# GPU 监控（如使用 GPU）
watch -n 1 nvidia-smi
```

## 安全建议

1. **API 认证**: 在生产环境添加 API 密钥认证
2. **HTTPS**: 使用 SSL/TLS 加密通信
3. **速率限制**: 防止 API 滥用
4. **输入验证**: 验证上传文件大小和格式
5. **资源限制**: 设置容器 CPU/内存限制

## 故障排除

详细的故障排除指南请参考 `DOCKER_DEPLOYMENT.md`。

## 下一步

1. ✅ 部署服务
2. 🧪 测试 API 端点
3. 📊 监控性能
4. 🔒 配置安全策略
5. 🚀 生产环境部署

---

有关更多详细信息，请参阅：
- [Docker 部署指南](DOCKER_DEPLOYMENT.md)
- [API 文档](http://localhost:8000/docs)
- [客户端示例](examples/)
