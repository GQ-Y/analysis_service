"""
视频流状态定义
"""
from enum import IntEnum, Enum

class StreamStatus(IntEnum):
    """视频流状态枚举"""
    INITIALIZING = 0  # 初始化中
    CONNECTING = 2    # 连接中
    ONLINE = 1        # 在线
    OFFLINE = -1      # 离线
    ERROR = -2        # 错误状态
    RUNNING = 3       # 运行中
    RECONNECTING = 4  # 重连中
    PAUSED = 5        # 已暂停
    STOPPED = 6       # 已停止
    UNKNOWN = 7       # 未知状态

class StreamHealthStatus(Enum):
    """流健康状态"""
    HEALTHY = "healthy"           # 健康
    GOOD = "good"                 # 良好
    DEGRADED = "degraded"         # 性能下降
    POOR = "poor"                 # 较差
    UNSTABLE = "unstable"         # 不稳定
    UNHEALTHY = "unhealthy"       # 不健康
    ERROR = "error"               # 错误
    OFFLINE = "offline"           # 离线
    UNKNOWN = "unknown"           # 未知状态 