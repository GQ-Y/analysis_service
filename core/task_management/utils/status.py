"""
任务状态定义
"""
from enum import IntEnum

class TaskStatus(IntEnum):
    """任务状态枚举"""
    WAITING = 0      # 等待中
    PROCESSING = 1   # 处理中
    COMPLETED = 2    # 已完成
    FAILED = -1      # 失败
    TIMEOUT = -2     # 超时
    CANCELLED = -3   # 已取消
    STOPPING = -4    # 停止中
    STOPPED = -5     # 已停止 