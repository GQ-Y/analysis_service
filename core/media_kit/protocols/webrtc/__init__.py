"""
WebRTC协议模块
提供WebRTC流处理功能
"""

from .handler import WebRTCStream
from .config import webrtc_config, WebRTCConfig

__all__ = ['WebRTCStream', 'webrtc_config', 'WebRTCConfig']
