"""
视频服务
整合视频编码、直播推流功能的主服务类
"""
from typing import Dict, Any, Optional

from shared.utils.logger import get_normal_logger, get_exception_logger
from services.video.encoders.file_encoder import FileEncoder
from services.video.streaming.live_streamer import LiveStreamer

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)


class VideoService:
    """视频服务 - 整合各种视频处理功能"""
    
    def __init__(self):
        """初始化视频服务"""
        self.file_encoder = FileEncoder()
        self.live_streamer = LiveStreamer()
        normal_logger.info("视频服务初始化完成")
    
    async def start_encoding(self, task_id: str, task_manager, format: str = "mp4",
                           quality: int = 80, width: Optional[int] = None,
                           height: Optional[int] = None, fps: int = 15) -> Dict[str, Any]:
        """
        开启视频编码 - 将分析结果转为MP4或FLV格式

        Args:
            task_id: 任务ID
            task_manager: 任务管理器实例
            format: 视频格式，支持"mp4"或"flv"
            quality: 视频质量(1-100)
            width: 视频宽度，为空则使用原始宽度
            height: 视频高度，为空则使用原始高度
            fps: 视频帧率

        Returns:
            Dict[str, Any]: 编码结果，包含视频URL
        """
        return await self.file_encoder.start_encoding(
            task_id=task_id,
            task_manager=task_manager,
            format=format,
            quality=quality,
            width=width,
            height=height,
            fps=fps
        )
    
    async def stop_encoding(self, task_id: str) -> Dict[str, Any]:
        """
        停止视频编码

        Args:
            task_id: 任务ID

        Returns:
            Dict[str, Any]: 停止结果
        """
        return await self.file_encoder.stop_encoding(task_id)
    
    async def start_live_stream(self, task_id: str, task_manager, format: str = "rtmp",
                              quality: int = 80, width: Optional[int] = None,
                              height: Optional[int] = None, fps: int = 15, stream_type: str = "ffmpeg") -> Dict[str, Any]:
        """
        开启直播流编码 - 支持两种推流模式

        Args:
            task_id: 任务ID
            task_manager: 任务管理器实例
            format: 流格式，支持"rtmp"、"hls"、"flv"
            quality: 视频质量(1-100)
            width: 视频宽度，为空则使用原始宽度
            height: 视频高度，为空则使用原始高度
            fps: 视频帧率
            stream_type: 推流类型，"ffmpeg"=直接FFmpeg输出FLV流(默认), "zlm"=使用ZLMediaKit服务器

        Returns:
            Dict[str, Any]: 编码结果，包含流信息和播放地址
        """
        return await self.live_streamer.start_live_stream(
            task_id=task_id,
            task_manager=task_manager,
            format=format,
            quality=quality,
            width=width,
            height=height,
            fps=fps,
            stream_type=stream_type
        )
    
    async def stop_live_stream(self, task_id: str) -> Dict[str, Any]:
        """
        停止视频直播流

        Args:
            task_id: 任务ID

        Returns:
            Dict[str, Any]: 停止结果
        """
        return await self.live_streamer.stop_live_stream(task_id)
    
    async def check_stream_status(self, task_id: str) -> Dict[str, Any]:
        """
        检查直播流状态

        Args:
            task_id: 任务ID

        Returns:
            Dict[str, Any]: 状态信息
        """
        return await self.live_streamer.check_stream_status(task_id)
    
    def update_analysis_result(self, task_id: str, analysis_result: Dict[str, Any]):
        """
        被动接收分析结果的方法，同时更新到文件编码器和直播推流器

        Args:
            task_id: 任务ID
            analysis_result: 分析结果字典
        """
        detections_count = len(analysis_result.get("detections", []))
        normal_logger.info(f"VideoService: 接收到任务 {task_id} 的分析结果，检测数量: {detections_count}")
        
        # 更新到文件编码器
        self.file_encoder.update_analysis_result(task_id, analysis_result)
        
        # 更新到直播推流器
        self.live_streamer.update_analysis_result(task_id, analysis_result)
        
        normal_logger.debug(f"VideoService: 已将分析结果传递给文件编码器和直播推流器: {task_id}")

    def set_performance_mode(self, mode: str) -> Dict[str, Any]:
        """
        设置直播流性能模式
        
        Args:
            mode: 性能模式 - "high_quality", "balanced", "high_performance"
            
        Returns:
            Dict[str, Any]: 设置结果
        """
        try:
            self.live_streamer.set_performance_mode(mode)
            return {
                "success": True,
                "message": f"性能模式已设置为: {mode}",
                "current_mode": mode
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"设置性能模式失败: {str(e)}",
                "current_mode": None
            }

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        获取直播流性能统计信息
        
        Returns:
            Dict[str, Any]: 性能统计信息
        """
        try:
            return {
                "success": True,
                "stats": self.live_streamer.get_performance_stats()
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"获取性能统计失败: {str(e)}",
                "stats": None
            }

    def reset_performance_stats(self) -> Dict[str, Any]:
        """
        重置直播流性能统计信息
        
        Returns:
            Dict[str, Any]: 重置结果
        """
        try:
            self.live_streamer.reset_performance_stats()
            return {
                "success": True,
                "message": "性能统计已重置"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"重置性能统计失败: {str(e)}"
            } 