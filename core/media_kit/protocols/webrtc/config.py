"""
WebRTC协议配置模块
定义WebRTC流相关配置
"""

from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field

@dataclass
class WebRTCConfig:
    """WebRTC协议配置"""

    # 重试配置
    retry_count: int = 3          # 最大重试次数
    retry_interval: int = 5000    # 重试间隔(毫秒)

    # 超时配置
    timeout: int = 10000          # 连接超时时间(毫秒)

    # ICE服务器配置
    ice_servers: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"urls": "stun:stun.l.google.com:19302"},
        {"urls": "stun:stun1.l.google.com:19302"}
    ])

    # SDP交换相关
    use_zlm_webrtc: bool = True   # 是否使用ZLMedia的WebRTC功能
    zlm_api_url: str = field(default_factory=lambda: WebRTCConfig._get_zlm_api_url())  # ZLMedia API地址

    # WebRTC优化参数
    enable_audio: bool = False    # 是否启用音频
    video_codec: str = "H264"     # 视频编码(VP8, VP9, H264)

    # RTP参数
    max_bitrate: int = 2000000    # 最大码率(bps)

    # 传输相关
    force_tcp: bool = False       # 是否强制使用TCP传输
    local_tcp_port: int = 8189    # 本地TCP监听端口
    additional_hosts: List[str] = field(default_factory=list)  # 额外的主机地址

    # WHIP/WHEP相关
    use_whip: bool = False        # 是否使用WHIP(WebRTC-HTTP Ingestion Protocol)
    use_whep: bool = False        # 是否使用WHEP(WebRTC-HTTP Egress Protocol)

    @staticmethod
    def _get_zlm_api_url() -> str:
        """获取ZLMediaKit API URL"""
        try:
            from core.config import settings
            return f"http://{settings.STREAMING.zlm_server_address}:{settings.STREAMING.zlm_api_port}"
        except ImportError:
            # 如果无法导入配置，使用默认值
            return "http://127.0.0.1:8088"

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'WebRTCConfig':
        """从字典创建配置

        Args:
            config: 配置字典

        Returns:
            WebRTCConfig: 配置对象
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
            "ice_servers": self.ice_servers,
            "use_zlm_webrtc": self.use_zlm_webrtc,
            "zlm_api_url": self.zlm_api_url,
            "enable_audio": self.enable_audio,
            "video_codec": self.video_codec,
            "max_bitrate": self.max_bitrate,
            "force_tcp": self.force_tcp,
            "local_tcp_port": self.local_tcp_port,
            "additional_hosts": self.additional_hosts,
            "use_whip": self.use_whip,
            "use_whep": self.use_whep
        }

# 默认全局配置实例
webrtc_config = WebRTCConfig()
