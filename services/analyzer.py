"""
分析服务工厂
根据配置创建合适的分析服务实例
"""
import os
import sys
import logging
from typing import Dict, Any, List, Optional, Union

# 添加父级目录到sys.path以允许导入core模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from core.config import settings
from shared.utils.logger import setup_logger
from services.base_analyzer import BaseAnalyzerService
from core.analyzer.detection.yolo_detector import YOLODetector
from core.analyzer.segmentation.yolo_segmentor import YOLOSegmentor
from core.task_manager import TaskManager
from core.task_processor import TaskProcessor
from core.task_queue import TaskQueue

logger = setup_logger(__name__)

def create_analyzer_service() -> BaseAnalyzerService:
    """
    创建分析服务实例

    Returns:
        BaseAnalyzerService: 分析服务实例
    """
    logger.info("创建HTTP模式分析服务")

    # 创建HTTP模式分析服务
    from services.http.http_analyzer import HTTPAnalyzerService
    return HTTPAnalyzerService()

# 获取当前服务模式
def get_service_mode() -> str:
    """
    获取当前服务模式

    Returns:
        str: 服务模式，'http'
    """
    return "http"

class Analyzer:
    def __init__(self):
        """初始化分析器"""
        self.task_queue = None
        self.task_processor = None
        self.task_manager = None
        self.detector = None
        self.segmentor = None

        # 配置流媒体相关参数
        self.streaming_config = {
            "reconnect_attempts": settings.STREAMING.reconnect_attempts,
            "reconnect_delay": settings.STREAMING.reconnect_delay,
            "read_timeout": settings.STREAMING.read_timeout,
            "connect_timeout": settings.STREAMING.connect_timeout,
            "max_consecutive_errors": settings.STREAMING.max_consecutive_errors,
            "frame_buffer_size": settings.STREAMING.frame_buffer_size,
            "log_level": settings.STREAMING.log_level
        }

    async def initialize(self):
        """初始化分析服务"""
        try:
            # 初始化任务队列
            self.task_queue = TaskQueue()

            # 初始化任务管理器
            self.task_manager = TaskManager()

            # 初始化任务处理器
            self.task_processor = TaskProcessor(self.task_manager)

            # 初始化检测器和分割器
            self.detector = YOLODetector()
            self.segmentor = YOLOSegmentor()

            logger.info("分析服务初始化完成，使用HTTP通信模式")
            return True

        except Exception as e:
            logger.error(f"初始化分析服务失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False