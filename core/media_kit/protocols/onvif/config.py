"""
ONVIF协议配置模块
定义ONVIF相关配置
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

@dataclass
class OnvifConfig:
    """ONVIF配置类"""
    
    # 认证相关
    auth_enable: bool = True            # 是否启用认证
    auth_username: str = "admin"        # 认证用户名
    auth_password: str = "admin"        # 认证密码
    
    # 连接超时设置
    connection_timeout: int = 10000     # 连接超时时间(毫秒)
    receive_timeout: int = 15000        # 接收超时时间(毫秒)
    
    # 视频配置
    profile_token: str = ""             # 要使用的Profile Token
    prefer_profile_type: str = "main"   # 优先使用的Profile类型 (main/sub)
    prefer_h264: bool = True            # 优先使用H264编码
    prefer_tcp: bool = True             # 优先使用TCP传输
    
    # 重连配置
    retry_count: int = 3                # 重试次数
    retry_interval: int = 5000          # 重试间隔(毫秒)
    
    # 缓存配置
    buffer_size: int = 8                # 缓存帧数量
    
    # 控制相关
    ptz_speed_x: float = 0.5            # PTZ X轴速度 (0.0-1.0)
    ptz_speed_y: float = 0.5            # PTZ Y轴速度 (0.0-1.0)
    ptz_speed_zoom: float = 0.5         # PTZ 缩放速度 (0.0-1.0)
    
    # 自定义配置
    extra_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典
        
        Returns:
            Dict[str, Any]: 配置字典
        """
        return {
            "auth_enable": self.auth_enable,
            "auth_username": self.auth_username,
            "auth_password": self.auth_password,
            "connection_timeout": self.connection_timeout,
            "receive_timeout": self.receive_timeout,
            "profile_token": self.profile_token,
            "prefer_profile_type": self.prefer_profile_type,
            "prefer_h264": self.prefer_h264,
            "prefer_tcp": self.prefer_tcp,
            "retry_count": self.retry_count,
            "retry_interval": self.retry_interval,
            "buffer_size": self.buffer_size,
            "ptz_speed_x": self.ptz_speed_x,
            "ptz_speed_y": self.ptz_speed_y,
            "ptz_speed_zoom": self.ptz_speed_zoom,
            "extra_config": self.extra_config
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'OnvifConfig':
        """从字典创建配置
        
        Args:
            config_dict: 配置字典
            
        Returns:
            OnvifConfig: 配置实例
        """
        # 提取已知字段
        kwargs = {}
        for field_name in [
            "auth_enable", "auth_username", "auth_password",
            "connection_timeout", "receive_timeout",
            "profile_token", "prefer_profile_type", "prefer_h264", "prefer_tcp",
            "retry_count", "retry_interval", "buffer_size",
            "ptz_speed_x", "ptz_speed_y", "ptz_speed_zoom"
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
onvif_config = OnvifConfig()
