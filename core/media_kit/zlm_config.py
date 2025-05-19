"""
ZLMediaKit配置模块
定义ZLMediaKit相关配置
"""
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import json

from shared.utils.logger import get_normal_logger, get_exception_logger
from core.config import settings

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
    http_ssl_port: int = 443          # HTTPS端口
    rtsp_ssl_port: int = 322          # RTSPS端口
    rtmp_ssl_port: int = 19350        # RTMPS端口
    api_port: int = 8088              # API端口，默认与HTTP端口相同
    api_secret: str = "OOEV3gbdHQh4VngpRdNcCeANzy4OFB4u"  # API密钥
    
    # 媒体流相关配置
    enable_hls: bool = True           # 是否启用HLS
    enable_mp4: bool = False          # 是否启用MP4录制
    enable_rtsp: bool = True          # 是否启用RTSP
    enable_rtmp: bool = True          # 是否启用RTMP
    enable_ts: bool = False           # 是否启用TS
    enable_fmp4: bool = False         # 是否启用FMP4
    
    # 日志相关配置
    log_level: int = 1                # 日志级别，0-4，0为最详细，1仅错误日志
    log_path: str = "logs/zlm"        # 日志路径
    log_days: int = 7                 # 日志保留天数
    
    # SSL相关配置
    ssl_cert: Optional[str] = None    # SSL证书路径
    ssl_key: Optional[str] = None     # SSL密钥路径
    
    # 系统相关配置
    thread_num: int = 0               # 线程数，0表示使用系统默认值
    
    # ZLMediaKit库路径
    zlm_lib_path: str = field(default_factory=lambda: os.environ.get("ZLM_LIB_PATH", "/usr/local/lib"))
    
    # 自定义配置
    extra_config: Dict[str, Any] = field(default_factory=dict)
    
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
        if hasattr(settings, "zlmediakit"):
            zlm_settings = settings.zlmediakit
            for key in self.__annotations__:
                if hasattr(zlm_settings, key):
                    setattr(self, key, getattr(zlm_settings, key))
    
    def to_mk_config(self) -> Dict[str, Any]:
        """转换为MK_CONFIG格式"""
        return {
            "thread_num": self.thread_num,
            "log_level": self.log_level,
            "log_path": self.log_path,
            "log_days": self.log_days,
            "ssl_cert": self.ssl_cert,
            "ssl_key": self.ssl_key
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
                "port": self.http_port,
                "sslport": self.http_ssl_port
            },
            "rtsp": {
                "port": self.rtsp_port,
                "sslport": self.rtsp_ssl_port
            },
            "rtmp": {
                "port": self.rtmp_port,
                "sslport": self.rtmp_ssl_port
            },
            "hls": {
                "enable": int(self.enable_hls)
            },
            "mp4": {
                "enable": int(self.enable_mp4)
            },
            "general": {
                "enableVhost": 1,
                "mediaServerId": "analysis_service"
            }
        }
        
        # 添加额外配置
        for section, items in self.extra_config.items():
            if section not in config:
                config[section] = {}
            for key, value in items.items():
                config[section][key] = value
        
        return config

# 默认配置实例
zlm_config = ZLMConfig() 