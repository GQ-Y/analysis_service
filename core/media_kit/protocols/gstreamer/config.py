"""
GStreamer协议配置模块
提供GStreamer相关配置
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

@dataclass
class GStreamerConfig:
    """GStreamer配置类"""
    
    # 连接配置
    timeout: int = 10000                # 连接超时时间(毫秒)
    retry_count: int = 3                # 重试次数
    retry_interval: int = 5000          # 重试间隔(毫秒)
    
    # 缓冲配置
    buffer_size: int = 200              # 缓冲区大小(帧数)
    max_buffer_ms: int = 1000           # 最大缓冲时间(毫秒)
    min_buffer_ms: int = 100            # 最小缓冲时间(毫秒)
    
    # 传输配置
    rtsp_transport: str = "tcp"         # RTSP传输协议 tcp/udp
    rtsp_latency: int = 200             # RTSP延迟(毫秒)
    drop_on_latency: bool = True        # 高延迟时丢帧
    
    # 硬件加速配置
    enable_hardware_decode: bool = True  # 启用硬件解码
    hardware_decoder: str = "auto"       # 硬件解码器: auto/nvdec/vaapi/qsv/none
    
    # 网络配置
    network_timeout: int = 20           # 网络超时(秒)
    connection_speed: int = 0           # 连接速度限制(0=无限制)
    
    # 音频配置
    enable_audio: bool = False          # 是否启用音频
    
    # 调试配置
    debug_pipeline: bool = False        # 调试管道
    log_level: str = "WARNING"          # 日志级别
    
    # 自定义管道配置
    custom_pipeline: Optional[str] = None  # 自定义管道字符串
    
    # 特定协议配置
    rtsp_protocols: List[str] = field(default_factory=lambda: ["tcp", "udp-mcast", "udp"])
    user_agent: str = "GStreamer/1.0"
    
    # 认证配置
    auth_enable: bool = False
    auth_user: str = ""
    auth_password: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典
        
        Returns:
            Dict[str, Any]: 配置字典
        """
        return {
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "retry_interval": self.retry_interval,
            "buffer_size": self.buffer_size,
            "max_buffer_ms": self.max_buffer_ms,
            "min_buffer_ms": self.min_buffer_ms,
            "rtsp_transport": self.rtsp_transport,
            "rtsp_latency": self.rtsp_latency,
            "drop_on_latency": self.drop_on_latency,
            "enable_hardware_decode": self.enable_hardware_decode,
            "hardware_decoder": self.hardware_decoder,
            "network_timeout": self.network_timeout,
            "connection_speed": self.connection_speed,
            "enable_audio": self.enable_audio,
            "debug_pipeline": self.debug_pipeline,
            "log_level": self.log_level,
            "custom_pipeline": self.custom_pipeline,
            "rtsp_protocols": self.rtsp_protocols,
            "user_agent": self.user_agent,
            "auth_enable": self.auth_enable,
            "auth_user": self.auth_user,
            "auth_password": self.auth_password
        }

# 全局配置实例
gstreamer_config = GStreamerConfig() 