"""
HTTP协议配置模块
定义HTTP流相关配置
"""

from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field

@dataclass
class HttpConfig:
    """HTTP协议配置"""
    
    # 重试配置
    retry_count: int = 3          # 最大重试次数
    retry_interval: int = 5000    # 重试间隔(毫秒)
    
    # 超时配置
    timeout: int = 10000          # 连接超时时间(毫秒)
    
    # 认证配置
    auth_enable: bool = False     # 是否启用认证
    auth_user: str = ""           # 用户名
    auth_password: str = ""       # 密码
    
    # HTTP特有配置
    user_agent: str = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
    headers: Dict[str, str] = field(default_factory=dict)  # 自定义HTTP头
    
    # 缓存配置
    use_cache: bool = True        # 是否使用缓存
    cache_timeout: int = 3600     # 缓存超时时间(秒)
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'HttpConfig':
        """从字典创建配置
        
        Args:
            config: 配置字典
            
        Returns:
            HttpConfig: 配置对象
        """
        # 创建默认配置
        result = cls()
        
        # 更新配置
        for key, value in config.items():
            if hasattr(result, key):
                setattr(result, key, value)
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典
        
        Returns:
            Dict[str, Any]: 配置字典
        """
        return {
            "retry_count": self.retry_count,
            "retry_interval": self.retry_interval,
            "timeout": self.timeout,
            "auth_enable": self.auth_enable,
            "auth_user": self.auth_user,
            "auth_password": self.auth_password,
            "user_agent": self.user_agent,
            "headers": self.headers,
            "use_cache": self.use_cache,
            "cache_timeout": self.cache_timeout
        }

# 默认全局配置实例
http_config = HttpConfig()
