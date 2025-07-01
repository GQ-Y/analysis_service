# 接口模块
# 定义核心接口，避免循环依赖

from .stream_interface import IVideoStream, StreamStatus, StreamHealthStatus

__all__ = [
    "IVideoStream",
    "StreamStatus",
    "StreamHealthStatus"
] 