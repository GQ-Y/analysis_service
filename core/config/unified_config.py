"""
统一配置管理模块
整合所有分散的配置到一个统一的架构中
"""
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import os
import logging
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from dotenv import load_dotenv

# 加载环境变量
load_dotenv(override=True)

# ============================================================================
# 枚举类型定义
# ============================================================================

# 零拷贝架构下不需要性能模式枚举

# ============================================================================
# 基础配置模型
# ============================================================================

class BaseConfigModel(BaseModel):
    """基础配置模型"""
    pass

# ============================================================================
# 服务配置
# ============================================================================

class ServiceConfig(BaseConfigModel):
    """服务基础配置"""
    project_name: str = "Skyeye AI Analysis Service"
    description: str = "Skyeye AI Analysis Service"
    version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    debug_enabled: bool = False
    environment: str = "production"
    host: str = "0.0.0.0"
    port: int = 8002

# ============================================================================
# 日志配置
# ============================================================================

class LoggingConfig(BaseConfigModel):
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_enabled: bool = True
    file_path: str = "logs/analysis.log"
    console_enabled: bool = True
    max_file_size: int = 10485760  # 10MB
    backup_count: int = 5

# ============================================================================
# Redis配置
# ============================================================================

class RedisConfig(BaseConfigModel):
    """Redis配置"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str = ""
    prefix: str = "analysis:"
    max_connections: int = 50
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    retry_on_timeout: bool = True
    max_retries: int = 3
    default_ttl: int = 3600
    task_result_ttl: int = 7200

# ============================================================================
# 任务管理配置
# ============================================================================

class TaskConfig(BaseConfigModel):
    """任务管理配置"""
    max_concurrent: int = 50
    max_queue_size: int = 1000
    timeout: int = 7200  # 2小时
    cleanup_interval: int = 180  # 3分钟
    max_retries: int = 3
    retry_delay: int = 5
    retry_backoff_factor: float = 2.0
    result_ttl: int = 3600

# ============================================================================
# ZLMediaKit配置
# ============================================================================

class ZLMConfig(BaseConfigModel):
    """ZLMediaKit配置"""
    server_address: str = "127.0.0.1"
    http_port: int = 8089
    rtsp_port: int = 554
    rtmp_port: int = 1935
    api_port: int = 8089
    api_secret: str = "Na3VmIbECZ4Nl7NHpz5XuPGWQelEFoSD"
    log_level: int = 1
    log_path: str = "logs/zlm"
    thread_num: int = 0
    max_retries: int = 5
    retry_interval: int = 2
    api_timeout: int = 10
    startup_wait: int = 3
    health_check_interval: int = 30
    health_check_timeout: int = 5
    stream_timeout: int = 15
    stream_buffer_size: int = 30

# ============================================================================
# 协议配置
# ============================================================================

class RTSPConfig(BaseConfigModel):
    """RTSP协议配置"""
    port: int = 554
    ssl_port: int = 322
    auth_enable: bool = False
    auth_user: str = ""
    auth_password: str = ""
    rtp_type: str = "tcp"
    max_buffer_ms: int = 2000

class WebRTCConfig(BaseConfigModel):
    """WebRTC协议配置"""
    enable_audio: bool = False
    video_codec: str = "H264"
    max_bitrate: int = 2000000
    force_tcp: bool = False
    local_tcp_port: int = 8189
    use_whip: bool = False
    use_whep: bool = False

class ONVIFConfig(BaseConfigModel):
    """ONVIF协议配置"""
    auth_enable: bool = True
    auth_username: str = "admin"
    auth_password: str = "admin"
    connection_timeout: int = 10000
    receive_timeout: int = 15000
    prefer_profile_type: str = "main"
    prefer_h264: bool = True
    prefer_tcp: bool = True
    buffer_size: int = 1

class GStreamerConfig(BaseConfigModel):
    """GStreamer配置"""
    enable: bool = True
    preferred_engine: str = "auto"
    hardware_decode: bool = True
    hardware_decoder: str = "auto"
    buffer_size: int = 200
    max_buffer_ms: int = 1000
    min_buffer_ms: int = 100
    rtsp_latency: int = 200
    drop_on_latency: bool = True
    network_timeout: int = 20
    debug_pipeline: bool = False
    log_level: str = "WARNING"

class ProtocolConfig(BaseConfigModel):
    """协议配置"""
    timeout: int = 10000
    retry_count: int = 3
    retry_interval: int = 5000
    rtsp: RTSPConfig = RTSPConfig()
    webrtc: WebRTCConfig = WebRTCConfig()
    onvif: ONVIFConfig = ONVIFConfig()
    gstreamer: GStreamerConfig = GStreamerConfig()

# ============================================================================
# 流媒体配置
# ============================================================================

class StreamingConfig(BaseConfigModel):
    """流媒体配置"""
    use_zlmediakit: bool = True
    reconnect_attempts: int = 3
    reconnect_delay: int = 5
    read_timeout: int = 30
    connect_timeout: int = 10
    max_consecutive_errors: int = 5
    frame_buffer_size: int = 30
    log_level: str = "INFO"

# ============================================================================
# 帧处理配置
# ============================================================================

class FrameProcessingConfig(BaseConfigModel):
    """帧处理配置"""
    buffer_size: int = 50
    target_fps: int = 25
    stall_threshold: float = 3.0
    stall_recovery_threshold: float = 1.0
    max_frame_skip: int = 5
    adaptive_quality: bool = True

# ============================================================================
# 分析配置
# ============================================================================

class AnalysisConfig(BaseConfigModel):
    """分析配置"""
    confidence: float = 0.5
    iou: float = 0.4
    max_det: int = 300
    device: str = "auto"
    analyze_interval: int = 100
    alarm_interval: int = 5000
    random_interval_min: int = 50
    random_interval_max: int = 150
    push_interval: int = 1000
    default_model: str = "yolov8n.pt"

# ============================================================================
# 回调配置
# ============================================================================

class CallbackConfig(BaseConfigModel):
    """回调配置"""
    # HTTP回调
    http_url: Optional[str] = None
    http_timeout: int = 10
    
    # Socket回调
    socket_enabled: bool = True
    socket_host: str = "localhost"
    socket_port: int = 8090
    socket_connect_timeout: int = 5
    socket_send_timeout: int = 10
    socket_max_connect_attempts: int = 3
    socket_connect_retry_delay: int = 5

# ============================================================================
# 存储配置
# ============================================================================

class StorageConfig(BaseConfigModel):
    """存储配置"""
    data_dir: str = "data"
    log_dir: str = "logs"
    temp_dir: str = "temp"
    result_dir: str = "results"
    save_images: bool = True
    image_quality: int = 95
    max_image_size: int = 1920
    cleanup_old_files: bool = True
    cleanup_days: int = 7

# ============================================================================
# 零拷贝内存配置
# ============================================================================

class MemoryConfig(BaseConfigModel):
    """零拷贝内存管理配置"""

    # 基础内存池配置
    enable_memory_pool: bool = Field(
        True,
        description="是否启用内存池管理"
    )

    max_memory_usage_percent: float = Field(
        50.0,  # 降低到50%以适应开发环境
        ge=10.0,
        le=90.0,
        description="最大使用系统内存的百分比（开发环境优化）"
    )

    min_free_memory_gb: float = Field(
        1.0,  # 降低到1GB以适应开发环境
        ge=0.5,
        le=8.0,
        description="至少保留的系统内存（GB，开发环境优化）"
    )

    # 支持的分辨率配置
    supported_resolutions: List[Tuple[int, int]] = Field(
        default=[(640, 480), (1280, 720), (1920, 1080), (320, 240), (3072, 1728)],
        description="支持的视频分辨率列表"
    )

    # 每种分辨率的内存块数量（调整为适合开发环境）
    blocks_per_resolution: Dict[str, int] = Field(
        default={
            "640x480": 300,   # 减少到300块 (约264MB)
            "1280x720": 100,  # 减少到100块 (约263MB)
            "1920x1080": 50,  # 减少到50块 (约280MB)
            "320x240": 400,   # 减少到400块 (约88MB)
            "3072x1728": 30   # 添加3072x1728支持 (约480MB)
        },
        description="每种分辨率预分配的内存块数量（开发环境优化）"
    )

    # 内存回收策略配置
    auto_cleanup_interval: int = Field(
        30,
        ge=5,
        le=300,
        description="自动清理间隔（秒）"
    )

    max_block_age: int = Field(
        300,
        ge=60,
        le=3600,
        description="内存块最大存活时间（秒）"
    )

    memory_pressure_threshold: float = Field(
        0.9,
        ge=0.5,
        le=0.99,
        description="内存压力阈值（0.5-0.99）"
    )

    force_cleanup_threshold: float = Field(
        0.95,
        ge=0.8,
        le=0.99,
        description="强制清理阈值（0.8-0.99）"
    )

    # 性能优化配置
    enable_memory_alignment: bool = Field(
        True,
        description="启用内存对齐优化"
    )

    alignment_bytes: int = Field(
        64,
        ge=16,
        le=256,
        description="内存对齐字节数"
    )

    enable_zero_copy: bool = Field(
        True,
        description="启用零拷贝优化"
    )

    enable_batch_processing: bool = Field(
        True,
        description="启用批处理优化"
    )

    max_batch_size: int = Field(
        8,
        ge=1,
        le=32,
        description="最大批处理大小"
    )

    # =========================================================================
    # 动态扩容配置
    # =========================================================================

    enable_dynamic_expansion: bool = Field(
        True,
        description="当分辨率空闲内存块耗尽时自动扩容"
    )

    dynamic_expand_block_count: int = Field(
        20,
        ge=1,
        le=1000,
        description="动态扩容时新增的内存块数量"
    )

    def validate_system_memory(self) -> bool:
        """
        验证系统内存是否满足配置要求

        Returns:
            bool: 是否满足要求
        """
        try:
            import psutil
            from ..memory.memory_utils import calculate_memory_requirements

            # 获取系统内存信息
            memory = psutil.virtual_memory()
            total_memory_gb = memory.total / (1024**3)
            available_memory_gb = memory.available / (1024**3)

            # 计算内存需求
            total_requirement_bytes = calculate_memory_requirements(self)
            total_requirement_gb = total_requirement_bytes / (1024**3)

            # 计算最大可用内存
            max_usable_gb = min(
                total_memory_gb * (self.max_memory_usage_percent / 100),
                available_memory_gb - self.min_free_memory_gb
            )

            # 检查是否满足要求
            return total_requirement_gb <= max_usable_gb

        except Exception:
            # 如果检查失败，返回True以允许继续
            return True

    def calculate_total_memory_requirement(self) -> int:
        """
        计算总内存需求（字节）

        Returns:
            int: 总内存需求（字节）
        """
        total_bytes = 0

        for width, height in self.supported_resolutions:
            res_key = f"{width}x{height}"
            block_count = self.blocks_per_resolution.get(res_key, 100)

            # 计算单个帧的内存需求（RGB 3通道）
            frame_size = width * height * 3

            # 考虑内存对齐
            if self.enable_memory_alignment:
                frame_size = ((frame_size + self.alignment_bytes - 1)
                             // self.alignment_bytes) * self.alignment_bytes

            # 计算该分辨率的总内存需求
            resolution_memory = frame_size * block_count
            total_bytes += resolution_memory

        return total_bytes

# ============================================================================
# 性能优化配置
# ============================================================================

class SystemConfig(BaseConfigModel):
    """系统配置（优化后的配置）"""
    
    # 数据库优化
    db_pool_size: int = 20
    db_max_overflow: int = 50
    db_pool_recycle: int = 1800
    db_pool_timeout: int = 30
    db_query_timeout: int = 30
    db_batch_size: int = 100
    
    # 网络优化
    http_timeout: int = 30
    http_max_retries: int = 3
    http_backoff_factor: float = 0.3
    http_pool_connections: int = 10
    http_pool_maxsize: int = 20
    discovery_timeout: int = 10
    discovery_max_devices: int = 100

# ============================================================================
# 统一配置类
# ============================================================================

class UnifiedSettings(BaseSettings):
    """统一配置设置类"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra='ignore',
        case_sensitive=False
    )
    
    # 各模块配置
    service: ServiceConfig = ServiceConfig()
    logging: LoggingConfig = LoggingConfig()
    redis: RedisConfig = RedisConfig()
    task: TaskConfig = TaskConfig()
    zlm: ZLMConfig = ZLMConfig()
    protocols: ProtocolConfig = ProtocolConfig()
    streaming: StreamingConfig = StreamingConfig()
    frame_processing: FrameProcessingConfig = FrameProcessingConfig()
    analysis: AnalysisConfig = AnalysisConfig()
    callback: CallbackConfig = CallbackConfig()
    storage: StorageConfig = StorageConfig()
    memory: MemoryConfig = MemoryConfig()
    system: SystemConfig = SystemConfig()
    
    # 通信模式
    communication_mode: str = "http"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_from_environment()
        # 零拷贝架构下不需要性能模式配置
        self._setup_logging()
    
    def _load_from_environment(self):
        """从环境变量加载配置"""
        # 服务配置
        if os.getenv("DEBUG_ENABLED"):
            self.service.debug_enabled = os.getenv("DEBUG_ENABLED", "false").lower() == "true"
        if os.getenv("ENVIRONMENT"):
            self.service.environment = os.getenv("ENVIRONMENT", "production")
        if os.getenv("SERVICES_HOST"):
            self.service.host = os.getenv("SERVICES_HOST", "0.0.0.0")
        if os.getenv("SERVICES_PORT"):
            self.service.port = int(os.getenv("SERVICES_PORT", "8002"))
        
        # Redis配置
        if os.getenv("REDIS_HOST"):
            self.redis.host = os.getenv("REDIS_HOST", "localhost")
        if os.getenv("REDIS_PORT"):
            self.redis.port = int(os.getenv("REDIS_PORT", "6379"))
        if os.getenv("REDIS_DB"):
            self.redis.db = int(os.getenv("REDIS_DB", "0"))
        if os.getenv("REDIS_PASSWORD"):
            self.redis.password = os.getenv("REDIS_PASSWORD", "")
        if os.getenv("REDIS_MAX_CONNECTIONS"):
            self.redis.max_connections = int(os.getenv("REDIS_MAX_CONNECTIONS", "50"))
        
        # ZLM配置
        if os.getenv("ZLM_SERVER_ADDRESS"):
            self.zlm.server_address = os.getenv("ZLM_SERVER_ADDRESS", "127.0.0.1")
        if os.getenv("ZLM_HTTP_PORT"):
            self.zlm.http_port = int(os.getenv("ZLM_HTTP_PORT", "8089"))
        if os.getenv("ZLM_API_SECRET"):
            self.zlm.api_secret = os.getenv("ZLM_API_SECRET", "Na3VmIbECZ4Nl7NHpz5XuPGWQelEFoSD")
        
        # 任务配置
        if os.getenv("TASK_MAX_CONCURRENT"):
            self.task.max_concurrent = int(os.getenv("TASK_MAX_CONCURRENT", "50"))
        if os.getenv("TASK_MAX_QUEUE_SIZE"):
            self.task.max_queue_size = int(os.getenv("TASK_MAX_QUEUE_SIZE", "1000"))
        
        # 分析配置
        if os.getenv("ANALYSIS_CONFIDENCE"):
            self.analysis.confidence = float(os.getenv("ANALYSIS_CONFIDENCE"))
        if os.getenv("ANALYSIS_IOU"):
            self.analysis.iou = float(os.getenv("ANALYSIS_IOU"))
        if os.getenv("ANALYSIS_DEVICE"):
            self.analysis.device = os.getenv("ANALYSIS_DEVICE")
        
        # 回调配置
        if os.getenv("CALLBACK_URL"):
            self.callback.http_url = os.getenv("CALLBACK_URL")
        if os.getenv("SOCKET_CALLBACK_HOST"):
            self.callback.socket_host = os.getenv("SOCKET_CALLBACK_HOST", "localhost")
        if os.getenv("SOCKET_CALLBACK_PORT"):
            self.callback.socket_port = int(os.getenv("SOCKET_CALLBACK_PORT", "8090"))
    

    
    def _setup_logging(self):
        """设置日志"""
        if os.getenv("LOG_LEVEL"):
            self.logging.level = os.getenv("LOG_LEVEL", "INFO")
        
        log_level = getattr(logging, self.logging.level.upper(), logging.INFO)
        logging.basicConfig(level=log_level, format=self.logging.format)

# ============================================================================
# 全局配置实例
# ============================================================================

# 创建统一配置实例
unified_settings = UnifiedSettings()

# 为了向后兼容，提供旧的访问方式
settings = unified_settings 