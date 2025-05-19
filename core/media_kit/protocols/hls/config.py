"""
HLS协议配置模块
提供HLS相关配置
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

@dataclass
class HlsConfig:
    """HLS配置类"""
    
    # HLS服务器相关配置
    segment_duration: int = 5       # 每个分片的时长（秒）
    playlist_size: int = 5          # 播放列表大小
    retry_count: int = 3            # 重试次数
    retry_interval: int = 5000      # 重试间隔(毫秒)
    timeout: int = 10000            # 连接超时时间(毫秒)
    
    # 流相关配置
    enable_audio: bool = True       # 是否启用音频
    enable_video: bool = True       # 是否启用视频
    
    # 缓存配置
    cache_segments: bool = True     # 是否缓存分片
    max_segments: int = 5           # 最大缓存分片数
    
    # 下载选项
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    
    # 自定义配置
    extra_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典
        
        Returns:
            Dict[str, Any]: 配置字典
        """
        return {
            "segment_duration": self.segment_duration,
            "playlist_size": self.playlist_size,
            "retry_count": self.retry_count,
            "retry_interval": self.retry_interval,
            "timeout": self.timeout,
            "enable_audio": self.enable_audio,
            "enable_video": self.enable_video,
            "cache_segments": self.cache_segments,
            "max_segments": self.max_segments,
            "user_agent": self.user_agent,
            "extra_config": self.extra_config
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'HlsConfig':
        """从字典创建配置
        
        Args:
            config_dict: 配置字典
            
        Returns:
            HlsConfig: 配置实例
        """
        # 提取已知字段
        kwargs = {}
        for field_name in [
            "segment_duration", "playlist_size", "retry_count", "retry_interval", "timeout",
            "enable_audio", "enable_video", "cache_segments", "max_segments", "user_agent"
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
hls_config = HlsConfig()
