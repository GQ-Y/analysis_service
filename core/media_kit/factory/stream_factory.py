"""
流工厂模块
提供统一的流创建接口，支持多种协议和处理引擎
"""

from typing import Dict, Any, Optional, Type
from enum import Enum
import urllib.parse

from shared.utils.logger import get_normal_logger, get_exception_logger

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)

from ..base.base_stream import BaseStream

# 导入协议处理器
from ..protocols.rtsp.handler import RtspStream
from ..protocols.rtmp.handler import RtmpStream
from ..protocols.http.handler import HttpStream
from ..protocols.hls.handler import HlsStream
from ..protocols.onvif.handler import OnvifStream

# 导入GStreamer处理器（如果可用）
try:
    from ..protocols.gstreamer.handler import GStreamerStream
    GSTREAMER_AVAILABLE = True
except ImportError:
    GSTREAMER_AVAILABLE = False
    normal_logger.warning("GStreamer不可用，将使用OpenCV模式")

class StreamEngine(Enum):
    """流处理引擎枚举"""
    OPENCV = "opencv"
    GSTREAMER = "gstreamer"
    AUTO = "auto"

class StreamFactory:
    """流工厂类，负责创建不同类型的流处理器"""
    
    # 协议映射表（OpenCV模式）
    OPENCV_PROTOCOL_MAP = {
        "rtsp": RtspStream,
        "rtmp": RtmpStream,
        "http": HttpStream,
        "https": HttpStream,
        "hls": HlsStream,
        "onvif": OnvifStream,
    }
    
    @staticmethod
    def create_stream(config: Dict[str, Any], engine: StreamEngine = StreamEngine.AUTO) -> Optional[BaseStream]:
        """创建流处理器
        
        Args:
            config: 流配置
            engine: 流处理引擎
            
        Returns:
            Optional[BaseStream]: 流处理器实例，失败返回None
        """
        try:
            url = config.get("url", "")
            if not url:
                normal_logger.error("创建流失败：URL为空")
                return None
            
            # 解析协议类型
            protocol = StreamFactory._parse_protocol(url)
            if not protocol:
                normal_logger.error(f"创建流失败：不支持的协议 {url}")
                return None
            
            # 确定使用的引擎
            selected_engine = StreamFactory._select_engine(engine, protocol, config)
            
            # 创建流实例
            if selected_engine == StreamEngine.GSTREAMER:
                return StreamFactory._create_gstreamer_stream(config)
            else:
                return StreamFactory._create_opencv_stream(config, protocol)
                
        except Exception as e:
            exception_logger.exception(f"创建流异常: {str(e)}")
            return None
    
    @staticmethod
    def _parse_protocol(url: str) -> Optional[str]:
        """解析URL协议类型
        
        Args:
            url: 流URL
            
        Returns:
            Optional[str]: 协议类型，失败返回None
        """
        try:
            parsed = urllib.parse.urlparse(url.lower())
            scheme = parsed.scheme
            
            if scheme in ["rtsp", "rtsps"]:
                return "rtsp"
            elif scheme in ["rtmp", "rtmps"]:
                return "rtmp"
            elif scheme in ["http", "https"]:
                # 检查是否为HLS流
                if ".m3u8" in url.lower():
                    return "hls"
                else:
                    return "http"
            else:
                # 检查是否为ONVIF URL（包含某些特征）
                if "onvif" in url.lower() or "wsdl" in url.lower():
                    return "onvif"
                
                return None
                
        except Exception as e:
            normal_logger.error(f"解析URL协议失败: {str(e)}")
            return None
    
    @staticmethod
    def _select_engine(engine: StreamEngine, protocol: str, config: Dict[str, Any]) -> StreamEngine:
        """选择流处理引擎
        
        Args:
            engine: 用户指定的引擎
            protocol: 协议类型
            config: 流配置
            
        Returns:
            StreamEngine: 选择的引擎
        """
        # 如果用户明确指定引擎
        if engine != StreamEngine.AUTO:
            if engine == StreamEngine.GSTREAMER and not GSTREAMER_AVAILABLE:
                normal_logger.warning("GStreamer不可用，回退到OpenCV")
                return StreamEngine.OPENCV
            return engine
        
        # 自动选择模式
        
        # 优先级1：检查配置中的偏好
        preferred_engine = config.get("preferred_engine", "").lower()
        if preferred_engine == "gstreamer" and GSTREAMER_AVAILABLE:
            normal_logger.info("根据配置偏好选择GStreamer引擎")
            return StreamEngine.GSTREAMER
        elif preferred_engine == "opencv":
            normal_logger.info("根据配置偏好选择OpenCV引擎")
            return StreamEngine.OPENCV
        
        # 优先级2：根据协议特性自动选择
        if GSTREAMER_AVAILABLE:
            # GStreamer适合的场景
            gstreamer_preferred_protocols = ["rtsp", "rtmp", "hls"]
            if protocol in gstreamer_preferred_protocols:
                normal_logger.info(f"协议 {protocol} 推荐使用GStreamer引擎")
                return StreamEngine.GSTREAMER
        
        # 优先级3：检查是否需要硬件加速
        if GSTREAMER_AVAILABLE and config.get("enable_hardware_decode", False):
            normal_logger.info("需要硬件加速，选择GStreamer引擎")
            return StreamEngine.GSTREAMER
        
        # 优先级4：检查是否有特殊要求
        if config.get("low_latency", False) and GSTREAMER_AVAILABLE:
            normal_logger.info("需要低延迟，选择GStreamer引擎")
            return StreamEngine.GSTREAMER
        
        # 默认使用OpenCV（兼容性最好）
        normal_logger.info("使用默认OpenCV引擎")
        return StreamEngine.OPENCV
    
    @staticmethod
    def _create_gstreamer_stream(config: Dict[str, Any]) -> Optional[BaseStream]:
        """创建GStreamer流
        
        Args:
            config: 流配置
            
        Returns:
            Optional[BaseStream]: GStreamer流实例
        """
        if not GSTREAMER_AVAILABLE:
            normal_logger.error("GStreamer不可用")
            return None
        
        try:
            return GStreamerStream(config)
        except Exception as e:
            exception_logger.exception(f"创建GStreamer流失败: {str(e)}")
            return None
    
    @staticmethod
    def _create_opencv_stream(config: Dict[str, Any], protocol: str) -> Optional[BaseStream]:
        """创建OpenCV流
        
        Args:
            config: 流配置
            protocol: 协议类型
            
        Returns:
            Optional[BaseStream]: OpenCV流实例
        """
        try:
            stream_class = StreamFactory.OPENCV_PROTOCOL_MAP.get(protocol)
            if not stream_class:
                normal_logger.error(f"不支持的协议: {protocol}")
                return None
            
            return stream_class(config)
        except Exception as e:
            exception_logger.exception(f"创建OpenCV流失败: {str(e)}")
            return None
    
    @staticmethod
    def get_supported_protocols() -> Dict[str, list]:
        """获取支持的协议列表
        
        Returns:
            Dict[str, list]: 支持的协议字典
        """
        opencv_protocols = list(StreamFactory.OPENCV_PROTOCOL_MAP.keys())
        
        result = {
            "opencv": opencv_protocols,
            "gstreamer": ["rtsp", "rtmp", "http", "https", "hls", "rtp"] if GSTREAMER_AVAILABLE else []
        }
        
        return result
    
    @staticmethod
    def is_gstreamer_available() -> bool:
        """检查GStreamer是否可用
        
        Returns:
            bool: GStreamer是否可用
        """
        return GSTREAMER_AVAILABLE
    
    @staticmethod
    def get_recommended_engine(url: str, config: Dict[str, Any] = None) -> StreamEngine:
        """获取推荐的引擎
        
        Args:
            url: 流URL
            config: 流配置（可选）
            
        Returns:
            StreamEngine: 推荐的引擎
        """
        if config is None:
            config = {}
        
        protocol = StreamFactory._parse_protocol(url)
        return StreamFactory._select_engine(StreamEngine.AUTO, protocol, config)
