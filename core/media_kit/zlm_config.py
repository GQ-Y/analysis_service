"""
ZLMediaKit配置模块
定义ZLMediaKit相关配置
"""
import os
from dataclasses import dataclass, field
from typing import Dict, Any
import json

from shared.utils.logger import get_normal_logger, get_exception_logger

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)

@dataclass
class ZLMConfig:
    """ZLMediaKit配置类"""

    # ZLMediaKit服务器相关配置
    server_address: str = "127.0.0.1"  # ZLMediaKit服务器地址
    http_port: int = 8088              # HTTP端口
    rtsp_port: int = 554              # RTSP端口
    rtmp_port: int = 1935             # RTMP端口
    api_port: int = 8088              # API端口，默认与HTTP端口相同
    api_secret: str = "OOEV3gbdHQh4VngpRdNcCeANzy4OFB4u"  # API密钥

    # 媒体流相关配置
    enable_hls: bool = True           # 是否启用HLS
    enable_rtsp: bool = True          # 是否启用RTSP
    enable_rtmp: bool = True          # 是否启用RTMP

    # 日志相关配置
    log_level: int = 1                # 日志级别，0-4，0为最详细，1仅错误日志
    log_path: str = "logs/zlm"        # 日志路径

    # 系统相关配置
    thread_num: int = 0               # 线程数，0表示使用系统默认值

    # ZLMediaKit库路径
    zlm_lib_path: str = field(default_factory=lambda: os.environ.get("ZLM_LIB_PATH", "/usr/local/lib"))

    def __post_init__(self):
        """初始化后处理"""
        normal_logger.info(f"ZLMediaKit配置: {self.server_address}:{self.http_port}")

        # 从环境变量或配置文件加载配置
        self._load_from_env()
        self._load_from_settings()

        # 确保日志目录存在
        os.makedirs(self.log_path, exist_ok=True)

    def _load_from_env(self):
        """从环境变量加载配置"""
        env_prefix = "ZLM_"
        for key in self.__annotations__:
            env_key = f"{env_prefix}{key.upper()}"
            if env_key in os.environ:
                value = os.environ[env_key]
                field_type = self.__annotations__[key]

                # 根据字段类型转换值
                if field_type == bool:
                    setattr(self, key, value.lower() in ('true', '1', 'yes'))
                elif field_type == int:
                    setattr(self, key, int(value))
                elif field_type == str:
                    setattr(self, key, value)
                elif field_type == Dict[str, Any]:
                    try:
                        setattr(self, key, json.loads(value))
                    except json.JSONDecodeError:
                        normal_logger.warning(f"无法解析环境变量 {env_key} 的JSON值: {value}")

    def _load_from_settings(self):
        """从应用设置加载配置"""
        try:
            from core.config import settings
            # 从STREAMING配置中加载ZLMediaKit相关配置
            if hasattr(settings, "STREAMING"):
                streaming_settings = settings.STREAMING
                # 映射配置字段
                if hasattr(streaming_settings, "zlm_server_address"):
                    self.server_address = streaming_settings.zlm_server_address
                if hasattr(streaming_settings, "zlm_http_port"):
                    self.http_port = streaming_settings.zlm_http_port
                if hasattr(streaming_settings, "zlm_api_secret"):
                    self.api_secret = streaming_settings.zlm_api_secret
                if hasattr(streaming_settings, "zlm_api_port"):
                    self.api_port = streaming_settings.zlm_api_port
                if hasattr(streaming_settings, "zlm_rtsp_port"):
                    self.rtsp_port = streaming_settings.zlm_rtsp_port
                if hasattr(streaming_settings, "zlm_rtmp_port"):
                    self.rtmp_port = streaming_settings.zlm_rtmp_port
                if hasattr(streaming_settings, "zlm_log_level"):
                    self.log_level = streaming_settings.zlm_log_level
                if hasattr(streaming_settings, "zlm_log_path"):
                    self.log_path = streaming_settings.zlm_log_path
                if hasattr(streaming_settings, "zlm_thread_num"):
                    self.thread_num = streaming_settings.zlm_thread_num
                if hasattr(streaming_settings, "zlm_lib_path"):
                    self.zlm_lib_path = streaming_settings.zlm_lib_path
        except ImportError:
            normal_logger.warning("无法导入core.config.settings，使用默认配置")

    def to_mk_config(self) -> Dict[str, Any]:
        """转换为MK_CONFIG格式"""
        return {
            "thread_num": self.thread_num,
            "log_level": self.log_level,
            "log_path": self.log_path
        }

    def to_ini_config(self) -> Dict[str, Dict[str, Any]]:
        """转换为INI配置格式"""
        config = {
            "api": {
                "apiDebug": 1,
                "secret": self.api_secret,
                "port": self.api_port
            },
            "http": {
                "port": self.http_port
            },
            "rtsp": {
                "port": self.rtsp_port
            },
            "rtmp": {
                "port": self.rtmp_port
            },
            "hls": {
                "enable": int(self.enable_hls)
            },
            "general": {
                "enableVhost": 1,
                "mediaServerId": "analysis_service"
            }
        }

        return config

# 默认配置实例
zlm_config = ZLMConfig()