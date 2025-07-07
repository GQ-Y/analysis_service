#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基础结果处理器
定义结果处理器的接口
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from app.core.zero_copy.frame_buffer import FrameBuffer


class BaseResultProcessor(ABC):
    """基础结果处理器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        初始化基础处理器
        
        Args:
            logger: 日志记录器
        """
        self.logger = logger or logging.getLogger(__name__)
    
    @abstractmethod
    def process(self, frame_buffer: FrameBuffer, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理结果
        
        Args:
            frame_buffer: 帧缓冲区
            task_config: 任务配置
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        pass
    
    def initialize(self) -> bool:
        """
        初始化处理器
        
        Returns:
            bool: 是否初始化成功
        """
        return True
    
    def cleanup(self):
        """清理资源"""
        pass 