# API接口和配置参数详细文档

## 1. API接口系统概览

### 1.1 接口架构设计

**API分层结构**:
```
FastAPI Application
├── Routers (路由层)
│   ├── task.py - 任务管理API
│   ├── health.py - 健康检查API
│   ├── stream.py - 流管理API
│   ├── discovery.py - 服务发现API
│   └── task_video.py - 视频任务API
├── Services (服务层)
│   ├── ZeroCopyTaskService - 零拷贝任务服务
│   ├── VideoService - 视频处理服务
│   └── DiscoveryService - 服务发现服务
└── Models (模型层)
    ├── requests.py - 请求模型
    ├── responses.py - 响应模型
    └── task.py - 任务模型
```

### 1.2 API版本和路径规范

**基础路径**: `/api/v1/`

**路径规范**:
- 任务管理: `/api/v1/task/`
- 健康检查: `/api/v1/health/`
- 流管理: `/api/v1/stream/`
- 服务发现: `/api/v1/discovery/`
- 视频处理: `/api/v1/video/`

## 2. 任务管理API详解

### 2.1 启动分析任务

**接口**: `POST /api/v1/task/start`

**请求参数**:
```json
{
  "model_code": "model-gcc",              // 模型代码 (必填)
  "stream_url": "rtsp://example.com/stream", // 流地址 (必填)
  "task_name": "检测任务_001",             // 任务名称 (可选)
  "analysis_type": "detection",           // 分析类型 (可选，默认detection)
  "enable_callback": false,               // 是否启用回调 (可选，默认false)
  "callback_url": "http://callback.url", // 回调地址 (可选)
  "save_result": true,                    // 是否保存结果 (可选，默认false)
  "save_images": false,                   // 是否保存图像 (可选，默认false)
  "device": "auto",                       // 设备类型 (可选，auto/cpu/gpu)
  "analysis_interval": 1,                 // 分析间隔帧数 (可选，默认1)
  "enable_hardware_decode": false,        // 硬件解码 (可选，默认false)
  "low_latency": true,                    // 低延迟模式 (可选，默认false)
  "config": {                             // 分析配置 (可选)
    "confidence_threshold": 0.5,
    "nms_threshold": 0.4,
    "max_detections": 100
  }
}
```

**响应格式**:
```json
{
  "success": true,
  "message": "零拷贝任务启动成功",
  "task_id": "uuid-string",
  "zero_copy_enabled": true,
  "memory_pool_status": {
    "total_blocks": 1000,
    "free_blocks": 995,
    "memory_usage_mb": 1024
  }
}
```

### 2.2 停止任务

**接口**: `POST /api/v1/task/stop`

**请求参数**:
```json
{
  "task_id": "uuid-string"  // 任务ID (必填)
}
```

**响应格式**:
```json
{
  "success": true,
  "message": "任务停止成功",
  "task_id": "uuid-string",
  "stopped_at": "2025-01-02T10:30:00Z"
}
```

### 2.3 获取任务状态

**接口**: `GET /api/v1/task/status/{task_id}`

**响应格式**:
```json
{
  "task_id": "uuid-string",
  "status": "running",                    // running/stopped/error
  "start_time": "2025-01-02T10:00:00Z",
  "duration": 1800,                       // 运行时长(秒)
  "frames_processed": 54000,              // 已处理帧数
  "processing_fps": 30.0,                 // 处理帧率
  "zero_copy_stats": {
    "zero_copy_operations": 54000,
    "batch_operations": 6750,
    "avg_processing_time": 0.033
  },
  "stream_info": {
    "stream_url": "rtsp://example.com/stream",
    "resolution": "1920x1080",
    "fps": 30.0,
    "is_connected": true
  },
  "analysis_stats": {
    "total_detections": 12500,
    "avg_confidence": 0.85,
    "processing_latency": 25.5
  }
}
```

### 2.4 获取任务列表

**接口**: `GET /api/v1/task/list`

**查询参数**:
- `status`: 任务状态过滤 (可选)
- `limit`: 返回数量限制 (可选，默认50)
- `offset`: 偏移量 (可选，默认0)

**响应格式**:
```json
{
  "total": 100,
  "tasks": [
    {
      "task_id": "uuid-string",
      "task_name": "检测任务_001",
      "status": "running",
      "model_code": "model-gcc",
      "stream_url": "rtsp://example.com/stream",
      "start_time": "2025-01-02T10:00:00Z",
      "frames_processed": 54000,
      "zero_copy_enabled": true
    }
  ]
}
```

## 3. 健康检查API详解

### 3.1 基础健康检查

**接口**: `GET /api/v1/health/`

**响应格式**:
```json
{
  "status": "healthy",                    // healthy/unhealthy/degraded
  "timestamp": "2025-01-02T10:30:00Z",
  "uptime": 3600,                         // 运行时长(秒)
  "version": "1.0.0",
  "zero_copy_enabled": true
}
```

### 3.2 详细健康检查

**接口**: `GET /api/v1/health/detailed`

**响应格式**:
```json
{
  "status": "healthy",
  "timestamp": "2025-01-02T10:30:00Z",
  "components": {
    "memory_pool": {
      "status": "healthy",
      "total_blocks": 1000,
      "free_blocks": 850,
      "memory_usage_percent": 15.0,
      "allocation_success_rate": 99.8
    },
    "task_processor": {
      "status": "healthy",
      "active_tasks": 5,
      "total_tasks_processed": 1250,
      "avg_processing_time": 0.033
    },
    "stream_manager": {
      "status": "healthy",
      "active_streams": 5,
      "total_streams_processed": 1250,
      "connection_success_rate": 98.5
    },
    "analyzer_factory": {
      "status": "healthy",
      "loaded_models": ["model-gcc", "yolo_detection"],
      "model_load_success_rate": 100.0
    },
    "redis": {
      "status": "healthy",
      "connection": true,
      "response_time_ms": 2.5
    },
    "database": {
      "status": "healthy",
      "connection": true,
      "query_success_rate": 99.9
    }
  },
  "system_resources": {
    "cpu_usage_percent": 45.2,
    "memory_usage_percent": 68.5,
    "disk_usage_percent": 25.8,
    "gpu_usage_percent": 78.3
  }
}
```

## 4. 流管理API详解

### 4.1 订阅视频流

**接口**: `POST /api/v1/stream/subscribe`

**请求参数**:
```json
{
  "stream_id": "stream_001",
  "stream_url": "rtsp://example.com/stream",
  "subscriber_id": "task_001",
  "config": {
    "buffer_size": 10,
    "frame_timeout": 5.0,
    "reconnect_interval": 5.0,
    "max_reconnect_attempts": 5,
    "enable_hardware_decode": false,
    "low_latency": true
  }
}
```

**响应格式**:
```json
{
  "success": true,
  "message": "流订阅成功",
  "stream_id": "stream_001",
  "subscriber_id": "task_001",
  "zero_copy_enabled": true,
  "stream_info": {
    "resolution": "1920x1080",
    "fps": 30.0,
    "format": "BGR"
  }
}
```

### 4.2 取消流订阅

**接口**: `DELETE /api/v1/stream/unsubscribe`

**请求参数**:
```json
{
  "stream_id": "stream_001",
  "subscriber_id": "task_001"
}
```

### 4.3 获取流信息

**接口**: `GET /api/v1/stream/info/{stream_id}`

**响应格式**:
```json
{
  "stream_id": "stream_001",
  "stream_url": "rtsp://example.com/stream",
  "status": "connected",                  // connected/disconnected/error
  "subscribers": ["task_001", "task_002"],
  "stream_info": {
    "resolution": "1920x1080",
    "fps": 30.0,
    "format": "BGR",
    "bitrate": 5000000
  },
  "statistics": {
    "frames_received": 108000,
    "frames_dropped": 50,
    "reconnect_count": 2,
    "uptime": 3600,
    "avg_frame_rate": 29.8
  },
  "zero_copy_stats": {
    "memory_blocks_allocated": 1000,
    "frame_references_created": 108000,
    "memory_efficiency": 95.5
  }
}
```

## 5. 服务发现API详解

### 5.1 注册服务

**接口**: `POST /api/v1/discovery/register`

**请求参数**:
```json
{
  "service_name": "analysis_service",
  "service_id": "analysis_001",
  "host": "192.168.1.100",
  "port": 8010,
  "metadata": {
    "version": "1.0.0",
    "zero_copy_enabled": true,
    "supported_models": ["model-gcc", "yolo_detection"],
    "max_concurrent_tasks": 50
  }
}
```

### 5.2 发现服务

**接口**: `GET /api/v1/discovery/services`

**查询参数**:
- `service_name`: 服务名称过滤 (可选)
- `healthy_only`: 仅返回健康服务 (可选，默认false)

**响应格式**:
```json
{
  "services": [
    {
      "service_name": "analysis_service",
      "service_id": "analysis_001",
      "host": "192.168.1.100",
      "port": 8010,
      "status": "healthy",
      "last_heartbeat": "2025-01-02T10:30:00Z",
      "metadata": {
        "version": "1.0.0",
        "zero_copy_enabled": true,
        "current_load": 0.45
      }
    }
  ]
}
```

## 6. 配置参数体系

### 6.1 服务配置 (ServiceConfig)

```python
class ServiceConfig(BaseModel):
    project_name: str = "Skyeye Analysis Service"
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8010
    debug_enabled: bool = False
    workers: int = 1
    max_request_size: int = 100 * 1024 * 1024  # 100MB
    request_timeout: int = 300  # 5分钟
```

### 6.2 内存配置 (MemoryConfig)

```python
class MemoryConfig(BaseModel):
    total_memory_gb: int = 20
    allocation_strategy: str = "percentage"  # percentage/fixed
    memory_percentage: float = 0.75  # 75%系统内存
    supported_resolutions: List[Tuple[int, int]] = [
        (1920, 1080), (1280, 720), (640, 480), (3072, 1728)
    ]
    blocks_per_resolution: int = 100
    enable_auto_cleanup: bool = True
    cleanup_interval: int = 30  # 秒
    memory_pressure_threshold: float = 0.85
    force_cleanup_threshold: float = 0.95
    alignment: int = 64  # 内存对齐字节数
```

### 6.3 任务配置 (TaskConfig)

```python
class TaskConfig(BaseModel):
    max_concurrent_tasks: int = 50
    default_analysis_interval: int = 1
    enable_zero_copy: bool = True
    enable_batch_processing: bool = True
    default_batch_size: int = 8
    batch_timeout: float = 0.1  # 秒
    task_timeout: int = 3600  # 1小时
    enable_task_persistence: bool = True
    result_retention_hours: int = 24
```

### 6.4 流配置 (StreamingConfig)

```python
class StreamingConfig(BaseModel):
    default_buffer_size: int = 10
    frame_timeout: float = 5.0
    reconnect_interval: float = 5.0
    max_reconnect_attempts: int = 5
    enable_hardware_decode: bool = False
    default_low_latency: bool = True
    stream_health_check_interval: int = 30
    max_concurrent_streams: int = 100
```

### 6.5 分析配置 (AnalysisConfig)

```python
class AnalysisConfig(BaseModel):
    default_model: str = "model-gcc"
    model_load_timeout: int = 60
    inference_timeout: float = 5.0
    default_confidence_threshold: float = 0.5
    default_nms_threshold: float = 0.4
    max_detections: int = 100
    enable_model_caching: bool = True
    model_cache_size: int = 5
```

### 6.6 性能配置 (PerformanceConfig)

```python
class PerformanceConfig(BaseModel):
    enable_performance_monitoring: bool = True
    stats_collection_interval: int = 10  # 秒
    enable_profiling: bool = False
    max_cpu_usage_percent: float = 80.0
    max_memory_usage_percent: float = 85.0
    enable_auto_scaling: bool = False
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
```

### 6.7 日志配置 (LoggingConfig)

```python
class LoggingConfig(BaseModel):
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_dir: str = "logs"
    max_log_size_mb: int = 100
    backup_count: int = 5
    enable_console_output: bool = True
    enable_file_output: bool = True
    separate_error_log: bool = True
```

### 6.8 Redis配置 (RedisConfig)

```python
class RedisConfig(BaseModel):
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    connection_pool_size: int = 10
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    retry_on_timeout: bool = True
    health_check_interval: int = 30
```

## 7. 环境变量配置

### 7.1 核心环境变量

```bash
# 服务配置
ANALYSIS_SERVICE_HOST=0.0.0.0
ANALYSIS_SERVICE_PORT=8010
ANALYSIS_SERVICE_DEBUG=false

# 内存配置
MEMORY_TOTAL_GB=20
MEMORY_ALLOCATION_STRATEGY=percentage
MEMORY_PERCENTAGE=0.75

# Redis配置
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# 日志配置
LOG_LEVEL=INFO
LOG_DIR=logs

# 性能配置
MAX_CONCURRENT_TASKS=50
ENABLE_ZERO_COPY=true
ENABLE_BATCH_PROCESSING=true
```

### 7.2 配置文件优先级

1. 环境变量 (最高优先级)
2. 配置文件 (config.yaml/config.json)
3. 默认值 (最低优先级)

---

**API文档版本**: v1.0  
**创建时间**: 2025-01-02  
**适用版本**: Analysis Service 零拷贝架构版本
