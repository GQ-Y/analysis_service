"""
视频编码器基类
提供视频编码的通用功能
"""
from typing import Dict, Any, Optional, List, Union
import os
import time
import asyncio
import threading
import subprocess
import uuid
import cv2
import numpy as np
from datetime import datetime
import logging

from core.config import settings
from core.task_management.utils.status import TaskStatus
from shared.utils.logger import get_normal_logger, get_exception_logger, get_test_logger
from services.video.utils.frame_dropper import SmartFrameDropper
from services.video.utils.ffmpeg_params import FFmpegParamsGenerator

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)
test_logger = get_test_logger()


class BaseEncoder:
    """视频编码器基类"""

    def __init__(self):
        """初始化基础编码器"""
        # 创建输出目录结构
        self.output_base_dir = os.path.join(os.getcwd(), "temp", "videos")
        os.makedirs(self.output_base_dir, exist_ok=True)
        normal_logger.info(f"视频输出基础目录: {self.output_base_dir}")

        # 存储FFmpeg进程
        self.ffmpeg_processes = {}

        # 存储编码线程
        self.encoding_threads = {}

        # 智能帧丢弃器
        self.frame_droppers = {}  # task_id -> SmartFrameDropper

        # 分析结果缓存，用于被动接收分析结果
        self.analysis_results_cache = {}  # {task_id: latest_analysis_result}
        self.analysis_results_lock = threading.Lock()  # 线程安全锁

        # 服务基础URL
        host = "localhost" if settings.SERVICES_HOST == "0.0.0.0" else settings.SERVICES_HOST
        self.base_url = f"http://{host}:{settings.SERVICES_PORT}"

        # 检查FFmpeg是否可用
        self.ffmpeg_available = self._check_ffmpeg()

    def _check_ffmpeg(self) -> bool:
        """检查FFmpeg是否可用"""
        try:
            # 获取FFmpeg版本信息
            result = subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            if result.returncode == 0:
                ffmpeg_version = result.stdout.split('\n')[0]
                normal_logger.info(f"FFmpeg可用: {ffmpeg_version}")
                return True
            else:
                normal_logger.warning(f"FFmpeg不可用，返回码: {result.returncode}")
                normal_logger.warning(f"错误信息: {result.stderr}")
                self._log_ffmpeg_install_guide()
                return False
        except FileNotFoundError:
            normal_logger.warning("FFmpeg未安装或未找到")
            self._log_ffmpeg_install_guide()
            return False
        except Exception as e:
            normal_logger.warning(f"FFmpeg检查失败: {str(e)}")
            self._log_ffmpeg_install_guide()
            return False

    def _log_ffmpeg_install_guide(self):
        """记录FFmpeg安装指导"""
        normal_logger.info("=" * 60)
        normal_logger.info("FFmpeg安装指导:")
        normal_logger.info("Windows系统:")
        normal_logger.info("1. 从 https://ffmpeg.org/download.html 下载FFmpeg")
        normal_logger.info("2. 解压到某个目录(如 C:\\ffmpeg)")
        normal_logger.info("3. 将FFmpeg的bin目录添加到系统PATH环境变量")
        normal_logger.info("4. 重启命令行工具")
        normal_logger.info("")
        normal_logger.info("或者使用包管理器安装:")
        normal_logger.info("- Chocolatey: choco install ffmpeg")
        normal_logger.info("- Scoop: scoop install ffmpeg")
        normal_logger.info("")
        normal_logger.info("注意: 缺少FFmpeg将无法使用视频编码功能，但不影响分析服务正常运行")
        normal_logger.info("=" * 60)

    def update_analysis_result(self, task_id: str, analysis_result: Dict[str, Any]):
        """
        被动接收分析结果的方法

        Args:
            task_id: 任务ID
            analysis_result: 分析结果字典
        """
        with self.analysis_results_lock:
            self.analysis_results_cache[task_id] = analysis_result
            normal_logger.debug(f"更新分析结果缓存: {task_id}, 检测数量: {len(analysis_result.get('detections', []))}")

    def get_cached_analysis_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        从缓存中获取分析结果

        Args:
            task_id: 任务ID

        Returns:
            Optional[Dict[str, Any]]: 分析结果，如果没有则返回None
        """
        with self.analysis_results_lock:
            return self.analysis_results_cache.get(task_id)

    def clear_analysis_result_cache(self, task_id: str):
        """
        清理指定任务的分析结果缓存

        Args:
            task_id: 任务ID
        """
        with self.analysis_results_lock:
            if task_id in self.analysis_results_cache:
                del self.analysis_results_cache[task_id]
                normal_logger.debug(f"清理分析结果缓存: {task_id}")

    def create_default_frame(self, width: int = 640, height: int = 480, message: str = "等待视频流...") -> np.ndarray:
        """
        创建默认的黑色帧
        
        Args:
            width: 帧宽度
            height: 帧高度
            message: 显示的消息
            
        Returns:
            np.ndarray: 默认帧
        """
        # 创建一个黑色的默认帧
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 导入帧渲染器来处理中文文本
        from services.video.utils.frame_renderer import FrameRenderer
        
        # 在帧上绘制文本（支持中文）
        frame = FrameRenderer._put_chinese_text(
            frame,
            message,
            (int(width/2) - 100, int(height/2)),
            24,
            (255, 255, 255)
        )
        
        return frame 