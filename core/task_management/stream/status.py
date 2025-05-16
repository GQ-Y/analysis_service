"""
视频流状态定义
"""
from enum import IntEnum, Enum

class StreamStatus(IntEnum):
    """视频流状态枚举"""
    INITIALIZING = 0  # 初始化中
    ONLINE = 1        # 在线
    OFFLINE = -1      # 离线
    ERROR = -2        # 错误状态

class StreamHealthStatus(Enum):
    """流健康状态"""
    HEALTHY = "healthy"           # 健康
    DEGRADED = "degraded"         # 性能下降
    UNSTABLE = "unstable"         # 不稳定
    UNHEALTHY = "unhealthy"       # 不健康 