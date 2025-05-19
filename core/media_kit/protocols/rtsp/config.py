"""
RTSP协议配置模块
提供RTSP相关配置
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

@dataclass
class RtspConfig:
    """RTSP配置类"""
    
    # RTSP服务器相关配置
    port: int = 554                 # RTSP端口
    ssl_port: int = 322             # RTSPS端口
    auth_enable: bool = False       # 是否启用认证
    auth_user: str = ""             # 认证用户名
    auth_password: str = ""         # 认证密码
    
    # RTSP客户端相关配置
    timeout: int = 10000            # 连接超时时间(毫秒)
    retry_count: int = 3            # 重试次数
    retry_interval: int = 5000      # 重试间隔(毫秒)
    rtp_type: str = "tcp"           # RTP传输类型，可选值: tcp, udp
    
    # 流相关配置
    enable_audio: bool = True       # 是否启用音频
    enable_video: bool = True       # 是否启用视频
    
    # 缓存配置
    max_buffer_ms: int = 2000       # 最大缓冲时间(毫秒)
    
    # 自定义配置
    extra_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典
        
        Returns:
            Dict[str, Any]: 配置字典
        """
        return {
            "port": self.port,
            "ssl_port": self.ssl_port,
            "auth_enable": self.auth_enable,
            "auth_user": self.auth_user,
            "auth_password": self.auth_password,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "retry_interval": self.retry_interval,
            "rtp_type": self.rtp_type,
            "enable_audio": self.enable_audio,
            "enable_video": self.enable_video,
            "max_buffer_ms": self.max_buffer_ms,
            "extra_config": self.extra_config
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RtspConfig':
        """从字典创建配置
        
        Args:
            config_dict: 配置字典
            
        Returns:
            RtspConfig: 配置实例
        """
        # 提取已知字段
        kwargs = {}
        for field_name in [
            "port", "ssl_port", "auth_enable", "auth_user", "auth_password",
            "timeout", "retry_count", "retry_interval", "rtp_type",
            "enable_audio", "enable_video", "max_buffer_ms"
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

# 默认配置实例
rtsp_config = RtspConfig()
