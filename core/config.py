"""
配置模块
管理应用的配置参数
"""
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
import os
import logging
from dotenv import load_dotenv

# 加载.env文件（如果存在）并强制覆盖已存在的环境变量
load_dotenv(override=True)

# 简单可序列化的数据模型
class BaseSettingsModel(BaseModel):
    """基础设置模型类"""
    pass

# 输出设置
class OutputSettings(BaseSettingsModel):
    """输出设置"""
    save_dir: str = "results"
    save_txt: bool = False
    save_conf: bool = False
    save_crop: bool = False
    save_masks: bool = False
    save_annotated: bool = True
    save_frames: bool = False
    frame_interval: int = 1  # 保存帧的间隔

# 日志设置
class LoggingSettings(BaseSettingsModel):
    """日志设置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    file: str = "logs/app.log"
    max_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    console: bool = True



# 流媒体设置
class StreamingSettings(BaseSettingsModel):
    """流媒体设置"""
    reconnect_attempts: int = 3
    reconnect_delay: int = 5
    read_timeout: int = 30
    connect_timeout: int = 10
    max_consecutive_errors: int = 5
    frame_buffer_size: int = 30
    log_level: str = "INFO"
    # ZLMediaKit配置
    use_zlmediakit: bool = os.getenv("STREAMING_USE_ZLMEDIAKIT", "true").lower() == "true"
    zlm_server_address: str = os.getenv("ZLM_SERVER_ADDRESS", "127.0.0.1")
    zlm_http_port: int = int(os.getenv("ZLM_HTTP_PORT", "8088"))
    zlm_rtsp_port: int = int(os.getenv("ZLM_RTSP_PORT", "554"))
    zlm_rtmp_port: int = int(os.getenv("ZLM_RTMP_PORT", "1935"))
    zlm_api_port: int = int(os.getenv("ZLM_API_PORT", "8088"))
    zlm_api_secret: str = os.getenv("ZLM_API_SECRET", "Na3VmIbECZ4Nl7NHpz5XuPGWQelEFoSD")
    zlm_thread_num: int = int(os.getenv("ZLM_THREAD_NUM", "0"))
    zlm_log_level: int = int(os.getenv("ZLM_LOG_LEVEL", "1"))
    zlm_log_path: str = os.getenv("ZLM_LOG_PATH", "logs/zlm")

# 协议配置设置
class ProtocolSettings(BaseSettingsModel):
    """协议配置设置"""
    
    # 通用协议配置
    timeout: int = int(os.getenv("PROTOCOL_TIMEOUT", "10000"))  # 毫秒
    retry_count: int = int(os.getenv("PROTOCOL_RETRY_COUNT", "3"))
    retry_interval: int = int(os.getenv("PROTOCOL_RETRY_INTERVAL", "5000"))  # 毫秒

    # RTSP协议配置
    rtsp_port: int = int(os.getenv("PROTOCOL_RTSP_PORT", "554"))
    rtsp_ssl_port: int = int(os.getenv("PROTOCOL_RTSP_SSL_PORT", "322"))
    rtsp_auth_enable: bool = os.getenv("PROTOCOL_RTSP_AUTH_ENABLE", "false").lower() == "true"
    rtsp_auth_user: str = os.getenv("PROTOCOL_RTSP_AUTH_USER", "")
    rtsp_auth_password: str = os.getenv("PROTOCOL_RTSP_AUTH_PASSWORD", "")
    rtsp_rtp_type: str = os.getenv("PROTOCOL_RTSP_RTP_TYPE", "tcp")  # tcp, udp
    rtsp_max_buffer_ms: int = int(os.getenv("PROTOCOL_RTSP_MAX_BUFFER_MS", "2000"))  # 毫秒

    # WebRTC协议配置
    webrtc_enable_audio: bool = os.getenv("PROTOCOL_WEBRTC_ENABLE_AUDIO", "false").lower() == "true"
    webrtc_video_codec: str = os.getenv("PROTOCOL_WEBRTC_VIDEO_CODEC", "H264")  # VP8, VP9, H264
    webrtc_max_bitrate: int = int(os.getenv("PROTOCOL_WEBRTC_MAX_BITRATE", "2000000"))  # bps
    webrtc_force_tcp: bool = os.getenv("PROTOCOL_WEBRTC_FORCE_TCP", "false").lower() == "true"
    webrtc_local_tcp_port: int = int(os.getenv("PROTOCOL_WEBRTC_LOCAL_TCP_PORT", "8189"))
    webrtc_use_whip: bool = os.getenv("PROTOCOL_WEBRTC_USE_WHIP", "false").lower() == "true"
    webrtc_use_whep: bool = os.getenv("PROTOCOL_WEBRTC_USE_WHEP", "false").lower() == "true"

    # ONVIF协议配置
    onvif_auth_enable: bool = os.getenv("PROTOCOL_ONVIF_AUTH_ENABLE", "true").lower() == "true"
    onvif_auth_username: str = os.getenv("PROTOCOL_ONVIF_AUTH_USERNAME", "admin")
    onvif_auth_password: str = os.getenv("PROTOCOL_ONVIF_AUTH_PASSWORD", "admin")
    onvif_connection_timeout: int = int(os.getenv("PROTOCOL_ONVIF_CONNECTION_TIMEOUT", "10000"))  # 毫秒
    onvif_receive_timeout: int = int(os.getenv("PROTOCOL_ONVIF_RECEIVE_TIMEOUT", "15000"))  # 毫秒
    onvif_prefer_profile_type: str = os.getenv("PROTOCOL_ONVIF_PREFER_PROFILE_TYPE", "main")  # main, sub
    onvif_prefer_h264: bool = os.getenv("PROTOCOL_ONVIF_PREFER_H264", "true").lower() == "true"
    onvif_prefer_tcp: bool = os.getenv("PROTOCOL_ONVIF_PREFER_TCP", "true").lower() == "true"
    onvif_buffer_size: int = int(os.getenv("PROTOCOL_ONVIF_BUFFER_SIZE", "1"))

    # GStreamer配置
    gstreamer_enable: bool = os.getenv("GSTREAMER_ENABLE", "true").lower() == "true"
    gstreamer_preferred_engine: str = os.getenv("GSTREAMER_PREFERRED_ENGINE", "auto")  # auto, gstreamer, opencv
    gstreamer_hardware_decode: bool = os.getenv("GSTREAMER_HARDWARE_DECODE", "true").lower() == "true"
    gstreamer_hardware_decoder: str = os.getenv("GSTREAMER_HARDWARE_DECODER", "auto")  # auto, nvdec, vaapi, qsv, none
    gstreamer_buffer_size: int = int(os.getenv("GSTREAMER_BUFFER_SIZE", "200"))
    gstreamer_max_buffer_ms: int = int(os.getenv("GSTREAMER_MAX_BUFFER_MS", "1000"))  # 毫秒
    gstreamer_min_buffer_ms: int = int(os.getenv("GSTREAMER_MIN_BUFFER_MS", "100"))  # 毫秒
    gstreamer_rtsp_latency: int = int(os.getenv("GSTREAMER_RTSP_LATENCY", "200"))  # 毫秒
    gstreamer_drop_on_latency: bool = os.getenv("GSTREAMER_DROP_ON_LATENCY", "true").lower() == "true"
    gstreamer_network_timeout: int = int(os.getenv("GSTREAMER_NETWORK_TIMEOUT", "20"))  # 秒
    gstreamer_debug_pipeline: bool = os.getenv("GSTREAMER_DEBUG_PIPELINE", "false").lower() == "true"
    gstreamer_log_level: str = os.getenv("GSTREAMER_LOG_LEVEL", "WARNING")  # DEBUG, INFO, WARNING, ERROR

# 存储配置
class StorageSettings(BaseSettingsModel):
    """存储设置"""
    base_dir: str = "data"
    model_dir: str = "models"
    temp_dir: str = "temp"
    max_size: int = 10 * 1024 * 1024 * 1024  # 10GB



# 分析设置
class AnalysisSettings(BaseSettingsModel):
    """分析设置"""
    confidence: float = 0.2
    iou: float = 0.45
    max_det: int = 300
    device: str = "auto"
    analyze_interval: int = 1
    alarm_interval: int = 60
    random_interval_min: int = 0
    random_interval_max: int = 0
    push_interval: int = 1

# 应用配置
class Settings(BaseSettings):
    """应用配置类"""
    # 模型配置
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra='ignore',  # 忽略额外的配置项
        case_sensitive=False
    )

    # 基础配置
    PROJECT_NAME: str = "Skyeye AI Analysis Service"
    DESCRIPTION: str = "Skyeye AI Analysis Service"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    DEBUG_ENABLED: bool = os.getenv("DEBUG_ENABLED", "false").lower() == "true"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "production")

    # 服务配置
    SERVICES_HOST: str = os.getenv("SERVICES_HOST", "0.0.0.0")
    SERVICES_PORT: int = int(os.getenv("SERVICES_PORT", "8002"))

    # Redis配置
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    REDIS_PREFIX: str = "analysis:"
    REDIS_MAX_CONNECTIONS: int = int(os.getenv("REDIS_MAX_CONNECTIONS", "50"))
    REDIS_SOCKET_TIMEOUT: int = int(os.getenv("REDIS_SOCKET_TIMEOUT", "5"))
    REDIS_RETRY_ON_TIMEOUT: bool = os.getenv("REDIS_RETRY_ON_TIMEOUT", "true").lower() == "true"

    # 任务队列配置
    TASK_QUEUE_MAX_SIZE: int = 1000
    TASK_QUEUE_MAX_CONCURRENT: int = 10
    TASK_QUEUE_RESULT_TTL: int = 3600  # 结果保留时间（秒）
    TASK_QUEUE_CLEANUP_INTERVAL: int = 300  # 清理间隔（秒）
    TASK_QUEUE_MAX_RETRIES: int = 3
    TASK_QUEUE_RETRY_DELAY: int = 5

    # 日志配置
    LOGGING: LoggingSettings = LoggingSettings()

    # 输出配置
    OUTPUT: OutputSettings = OutputSettings()

    # 流媒体配置
    STREAMING: StreamingSettings = StreamingSettings()

    # 协议配置
    PROTOCOLS: ProtocolSettings = ProtocolSettings()

    # 存储配置
    STORAGE: StorageSettings = StorageSettings()

    # 分析配置
    ANALYSIS: AnalysisSettings = AnalysisSettings()

    # 默认目标检测配置
    DEFAULT_DETECTION_MODEL: str = "yolov8n.pt"

    # 设置日志级别
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # 通信模式，只支持http模式
    COMMUNICATION_MODE: str = "http"

    # HTTP回调配置 (新增/确保存在)
    CALLBACK_URL: str | None = os.getenv("CALLBACK_URL", None)
    HTTP_CALLBACK_TIMEOUT: int = int(os.getenv("HTTP_CALLBACK_TIMEOUT", "10"))

    # Socket回调配置 (新增)
    SOCKET_CALLBACK_ENABLED: bool = os.getenv("SOCKET_CALLBACK_ENABLED", "true").lower() == "true"
    SOCKET_CALLBACK_HOST: str = os.getenv("SOCKET_CALLBACK_HOST", "localhost")
    SOCKET_CALLBACK_PORT: int = int(os.getenv("SOCKET_CALLBACK_PORT", "8089"))
    SOCKET_CONNECT_TIMEOUT: int = int(os.getenv("SOCKET_CONNECT_TIMEOUT", "5"))
    SOCKET_SEND_TIMEOUT: int = int(os.getenv("SOCKET_SEND_TIMEOUT", "10"))
    SOCKET_MAX_CONNECT_ATTEMPTS: int = int(os.getenv("SOCKET_MAX_CONNECT_ATTEMPTS", "3"))
    SOCKET_CONNECT_RETRY_DELAY: int = int(os.getenv("SOCKET_CONNECT_RETRY_DELAY", "5"))

# 创建设置实例
settings = Settings()



# 从环境变量加载分析配置
if os.getenv("ANALYSIS_CONFIDENCE"):
    settings.ANALYSIS.confidence = float(os.getenv("ANALYSIS_CONFIDENCE"))
if os.getenv("ANALYSIS_IOU"):
    settings.ANALYSIS.iou = float(os.getenv("ANALYSIS_IOU"))
if os.getenv("ANALYSIS_MAX_DET"):
    settings.ANALYSIS.max_det = int(os.getenv("ANALYSIS_MAX_DET"))
if os.getenv("ANALYSIS_DEVICE"):
    settings.ANALYSIS.device = os.getenv("ANALYSIS_DEVICE")
if os.getenv("ANALYSIS_ANALYZE_INTERVAL"):
    settings.ANALYSIS.analyze_interval = int(os.getenv("ANALYSIS_ANALYZE_INTERVAL"))
if os.getenv("ANALYSIS_ALARM_INTERVAL"):
    settings.ANALYSIS.alarm_interval = int(os.getenv("ANALYSIS_ALARM_INTERVAL"))
if os.getenv("ANALYSIS_RANDOM_INTERVAL_MIN"):
    settings.ANALYSIS.random_interval_min = int(os.getenv("ANALYSIS_RANDOM_INTERVAL_MIN"))
if os.getenv("ANALYSIS_RANDOM_INTERVAL_MAX"):
    settings.ANALYSIS.random_interval_max = int(os.getenv("ANALYSIS_RANDOM_INTERVAL_MAX"))
if os.getenv("ANALYSIS_PUSH_INTERVAL"):
    settings.ANALYSIS.push_interval = int(os.getenv("ANALYSIS_PUSH_INTERVAL"))

# 设置日志级别
log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(level=log_level)
