"""
基础分析器接口
定义所有分析器必须实现的接口
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import numpy as np

class BaseAnalyzer(ABC):
    """
    基础分析器接口
    所有分析器必须实现这个接口
    """
    
    @abstractmethod
    async def load_model(self, model_code: str) -> bool:
        """
        加载模型
        
        Args:
            model_code: 模型代码
            
        Returns:
            bool: 是否加载成功
        """
        pass
        
    @abstractmethod
    async def detect(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        执行分析
        
        Args:
            image: 输入图像
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        pass
        
    @abstractmethod
    async def process_video_frame(self, frame: np.ndarray, frame_index: int, **kwargs) -> Dict[str, Any]:
        """
        处理视频帧
        
        Args:
            frame: 视频帧
            frame_index: 帧索引
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        pass
        
    @property
    @abstractmethod
    def model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息
        """
        pass
        
    @abstractmethod
    def release(self) -> None:
        """
        释放资源
        """
        pass
