# Inference Server

这是一个基于 TensorRT 的推理服务器，将 `infer` 模块包装成网络服务，提供 HTTP API 供外部调用。

## 文件结构

```
lmf/sever/
├── hpp/
│   ├── InferenceEngine.hpp    # 推理引擎头文件
│   └── InferenceServer.hpp    # 推理服务器头文件
├── cpp/
│   ├── InferenceEngine.cpp    # 推理引擎实现
│   ├── InferenceServer.cpp    # 推理服务器实现
│   ├── main.cpp               # 主程序
│   └── test_client.cpp        # 测试客户端
├── CMakeLists.txt             # 编译配置
└── README.md                  # 说明文档
```

## 主要功能

- **推理引擎**: 封装 TensorRT 推理功能，支持单张和批量图像推理
- **HTTP 服务器**: 基于 Drogon 的高性能 HTTP 服务器
- **RESTful API**: 提供推理、健康检查、状态查询等接口
- **性能监控**: 内置推理时间统计和请求计数
- **部署管理**: 提供完整的部署脚本和生命周期管理

## API 接口

### 1. 推理接口

**POST** `/inference`

执行图像推理，支持 multipart/form-data 格式上传图像。

**请求参数:**
- `image`: 图像文件 (JPEG, PNG, BMP 等)
- `confidence_threshold`: 置信度阈值 (可选，默认 0.5)

**响应示例:**
```json
{
  "status_code": 0,
  "status_message": "Success",
  "inference_time_ms": 45.2,
  "detections": [
    {
      "class_id": 0,
      "confidence": 0.95,
      "x": 100,
      "y": 150,
      "width": 200,
      "height": 300,
      "class_name": "person"
    }
  ],
  "detection_count": 1
}
```

### 2. 健康检查

**GET** `/health`

检查服务器健康状态。

**响应示例:**
```json
{
  "status": "healthy",
  "timestamp": 1640995200
}
```

### 3. 状态查询

**GET** `/status`

获取服务器运行状态和统计信息。

**响应示例:**
```json
{
  "status": "running",
  "timestamp": 1640995200,
  "model_loaded": true,
  "total_requests": 100,
  "model_info": {
    "input_width": 640,
    "input_height": 640,
    "input_channels": 3
  }
}
```

## 使用方法

### 1. 编译

```bash
# 创建构建目录
mkdir build && cd build

# 配置
cmake ..

# 编译
make -j$(nproc)
```

### 2. 启动服务器


# 或直接启动
./inference_server --model /path/to/model.trt --port 8080 --device 0 --threads 4
```

### 3. 测试服务

```bash
# 测试推理接口
./test_client http://localhost:8080 test.jpg 0.5

# 测试健康检查
curl http://localhost:8080/health

# 测试状态查询
curl http://localhost:8080/status
```

### 4. 停止服务器

```bash
./deploy.sh stop
```

## 部署脚本

`deploy.sh` 提供了完整的服务器管理功能：

```bash
# 启动服务器
./deploy.sh start

# 停止服务器
./deploy.sh stop

# 重启服务器
./deploy.sh restart

# 检查状态
./deploy.sh status

# 查看日志
./deploy.sh logs

# 显示帮助
./deploy.sh help
```

## 配置参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型文件路径 | 必需 |
| `--port` | 服务器端口 | 8080 |
| `--device` | GPU 设备 ID | 0 |
| `--threads` | 工作线程数 | 4 |

## 依赖要求

- **OpenCV 4.x**: 图像处理
- **CUDA**: GPU 加速
- **TensorRT**: 模型推理
- **Drogon**: HTTP 服务器框架
- **libcurl**: HTTP 客户端
- **jsoncpp**: JSON 处理
- **CMake 3.10+**: 构建系统

## 性能优化

### 1. GPU 配置
- 选择合适的 GPU 设备
- 调整 CUDA 内存分配
- 使用 TensorRT 优化

### 2. 并发处理
- 调整工作线程数量
- 使用异步推理
- 实现请求队列

### 3. 内存管理
- 复用图像缓冲区
- 及时释放推理结果
- 监控内存使用

## 监控和日志

### 1. 性能指标
- 推理时间统计
- 请求成功率
- 并发处理能力

### 2. 日志记录
- 请求日志
- 错误日志
- 性能日志

### 3. 健康监控
- 服务可用性
- 资源使用情况
- 异常告警

## 扩展功能

### 1. 模型管理
- 支持多模型加载
- 模型热更新
- 模型版本管理

### 2. 负载均衡
- 多实例部署
- 请求分发
- 故障转移

### 3. 缓存机制
- 结果缓存
- 模型缓存
- 配置缓存

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型文件路径
   - 验证 TensorRT 版本兼容性
   - 检查 GPU 内存

2. **推理性能差**
   - 检查 GPU 利用率
   - 调整批处理大小
   - 优化模型配置

3. **服务启动失败**
   - 检查端口占用
   - 验证依赖库
   - 查看错误日志

### 调试技巧

```bash
# 检查服务状态
./deploy.sh status

# 查看详细日志
tail -f server.log

# 测试网络连接
curl -v http://localhost:8080/health

# 检查进程
ps aux | grep inference_server
```

## 许可证

请参考项目的许可证文件。
