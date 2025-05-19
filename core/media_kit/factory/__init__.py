"""
媒体工具包工厂模块
提供流工厂和协议工厂，用于创建不同类型的流和协议处理器
"""

from .stream_factory import StreamFactory, stream_factory

__all__ = [
    'StreamFactory',
    'stream_factory'
]
