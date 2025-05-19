"""
GB28181协议配置模块
定义GB28181相关配置
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

@dataclass
class Gb28181Config:
    """GB28181配置类"""
    
    # SIP服务器配置
    sip_id: str = "34020000002000000001"    # SIP服务器ID
    sip_domain: str = "3402000000"         # SIP域
    sip_host: str = "0.0.0.0"              # SIP服务器监听地址
    sip_port: int = 5060                   # SIP服务器监听端口
    sip_username: str = "admin"            # SIP用户名
    sip_password: str = "admin"            # SIP密码
    
    # 国标注册相关
    register_timeout: int = 3600           # 注册超时时间(秒)
    heartbeat_cycle: int = 60              # 心跳周期(秒)
    max_timeout_times: int = 3             # 最大超时次数
    
    # 媒体相关
    media_host: str = "0.0.0.0"            # 媒体服务器地址
    media_port_start: int = 10000          # 媒体端口起始值
    media_port_end: int = 11000            # 媒体端口结束值
    
    # ZLMediaKit相关配置
    zlm_host: str = "127.0.0.1"            # ZLMediaKit服务器地址
    zlm_http_port: int = 8088              # ZLMediaKit HTTP API端口
    zlm_rtsp_port: int = 554               # ZLMediaKit RTSP端口
    zlm_rtmp_port: int = 1935              # ZLMediaKit RTMP端口
    zlm_secret: str = ""                   # ZLMediaKit API密钥
    zlm_rtp_port: int = 0                  # ZLMediaKit RTP端口，0表示随机分配
    
    # 码流相关
    prefer_stream_type: str = "main"       # 首选码流类型 (main/sub)
    auto_switch_sub_stream: bool = True    # 自动切换至子码流
    
    # 重连配置
    retry_count: int = 3                   # 重试次数
    retry_interval: int = 5000             # 重试间隔(毫秒)
    
    # 缓存配置
    buffer_size: int = 8                   # 缓存帧数量
    
    # 自定义配置
    extra_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典
        
        Returns:
            Dict[str, Any]: 配置字典
        """
        return {
            "sip_id": self.sip_id,
            "sip_domain": self.sip_domain,
            "sip_host": self.sip_host,
            "sip_port": self.sip_port,
            "sip_username": self.sip_username,
            "sip_password": self.sip_password,
            "register_timeout": self.register_timeout,
            "heartbeat_cycle": self.heartbeat_cycle,
            "max_timeout_times": self.max_timeout_times,
            "media_host": self.media_host,
            "media_port_start": self.media_port_start,
            "media_port_end": self.media_port_end,
            "zlm_host": self.zlm_host,
            "zlm_http_port": self.zlm_http_port,
            "zlm_rtsp_port": self.zlm_rtsp_port,
            "zlm_rtmp_port": self.zlm_rtmp_port,
            "zlm_secret": self.zlm_secret,
            "zlm_rtp_port": self.zlm_rtp_port,
            "prefer_stream_type": self.prefer_stream_type,
            "auto_switch_sub_stream": self.auto_switch_sub_stream,
            "retry_count": self.retry_count,
            "retry_interval": self.retry_interval,
            "buffer_size": self.buffer_size,
            "extra_config": self.extra_config
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Gb28181Config':
        """从字典创建配置
        
        Args:
            config_dict: 配置字典
            
        Returns:
            Gb28181Config: 配置实例
        """
        # 提取已知字段
        kwargs = {}
        for field_name in [
            "sip_id", "sip_domain", "sip_host", "sip_port", "sip_username", "sip_password",
            "register_timeout", "heartbeat_cycle", "max_timeout_times",
            "media_host", "media_port_start", "media_port_end",
            "zlm_host", "zlm_http_port", "zlm_rtsp_port", "zlm_rtmp_port", "zlm_secret", "zlm_rtp_port",
            "prefer_stream_type", "auto_switch_sub_stream",
            "retry_count", "retry_interval", "buffer_size"
        ]:
            if field_name in config_dict:
                kwargs[field_name] = config_dict[field_name]
        
        # 提取额外配置
        extra_config = {}
        for key, value in config_dict.items():
            if key not in kwargs and key != "extra_config":
                extra_config[key] = value
        
        # 如果config_dict中有extra_config，合并它
        if "extra_config" in config_dict:
            extra_config.update(config_dict["extra_config"])
        
        kwargs["extra_config"] = extra_config
        
        return cls(**kwargs)

# 尝试读取环境变量或配置文件配置ZLM相关参数
def load_default_config() -> Gb28181Config:
    """加载默认配置，优先从环境变量读取"""
    config = Gb28181Config()
    
    # 从环境变量加载配置
    if os.environ.get("ZLM_HOST"):
        config.zlm_host = os.environ.get("ZLM_HOST")
    
    if os.environ.get("ZLM_HTTP_PORT"):
        try:
            config.zlm_http_port = int(os.environ.get("ZLM_HTTP_PORT"))
        except ValueError:
            pass
            
    if os.environ.get("ZLM_RTSP_PORT"):
        try:
            config.zlm_rtsp_port = int(os.environ.get("ZLM_RTSP_PORT"))
        except ValueError:
            pass
            
    if os.environ.get("ZLM_SECRET"):
        config.zlm_secret = os.environ.get("ZLM_SECRET")
    
    return config

# 默认配置实例
gb28181_config = load_default_config()
