"""
零拷贝分析器基类
重构BaseAnalyzer支持内存块引用输入和原地分析
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import time
import numpy as np

from .base_analyzer import BaseAnalyzer
from ..frame.frame_reference import FrameReference
from ..memory.memory_block import MemoryBlock
from ..memory.memory_pool import MemoryPool

# 使用项目现有的日志系统
try:
    from shared.utils.logger import get_normal_logger, get_exception_logger
    normal_logger = get_normal_logger(__name__)
    exception_logger = get_exception_logger(__name__)
except ImportError:
    import logging
    normal_logger = logging.getLogger(__name__)
    exception_logger = logging.getLogger(__name__)


class ZeroCopyBaseAnalyzer(BaseAnalyzer):
    """
    零拷贝分析器基类
    扩展基础分析器，支持内存块引用输入和原地分析
    """
    
    def __init__(self, model_code: Optional[str] = None, device: str = "auto", **kwargs):
        """
        初始化零拷贝分析器
        
        Args:
            model_code: 模型代码
            device: 推理设备
            **kwargs: 其他参数
        """
        super().__init__(model_code, device, **kwargs)
        
        # 零拷贝相关配置
        self.enable_zero_copy = kwargs.get("enable_zero_copy", True)
        self.enable_in_place_analysis = kwargs.get("enable_in_place_analysis", True)
        self.enable_batch_processing = kwargs.get("enable_batch_processing", True)
        self.max_batch_size = kwargs.get("max_batch_size", 8)
        
        # 内存池引用
        self.memory_pool: Optional[MemoryPool] = None
        
        # 性能统计
        self._zero_copy_stats = {
            "zero_copy_operations": 0,
            "in_place_operations": 0,
            "batch_operations": 0,
            "memory_copy_avoided": 0,
            "total_processing_time": 0.0,
        }
        
        normal_logger.info(f"零拷贝分析器初始化: {self.__class__.__name__}")
    
    def set_memory_pool(self, memory_pool: MemoryPool) -> bool:
        """
        设置内存池
        
        Args:
            memory_pool: 内存池实例
            
        Returns:
            bool: 是否设置成功
        """
        try:
            self.memory_pool = memory_pool
            normal_logger.info(f"分析器 {self.__class__.__name__} 设置内存池成功")
            return True
        except Exception as e:
            exception_logger.exception(f"设置内存池失败: {str(e)}")
            return False
    
    async def process_frame_reference(self, frame_ref: FrameReference, **kwargs) -> Dict[str, Any]:
        """
        处理帧引用（零拷贝）
        
        Args:
            frame_ref: 帧引用
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        start_time = time.time()
        
        try:
            # 获取帧数据（零拷贝）
            frame_data = frame_ref.get_data()
            if frame_data is None:
                raise ValueError("无法从帧引用获取数据")
            
            # 获取帧元数据
            metadata = frame_ref.get_metadata()
            
            # 执行零拷贝分析
            if self.enable_in_place_analysis:
                analysis_result = await self.analyze_in_place(frame_data, metadata, **kwargs)
                self._zero_copy_stats["in_place_operations"] += 1
            else:
                analysis_result = await self.detect(frame_data, **kwargs)
            
            # 更新统计
            self._zero_copy_stats["zero_copy_operations"] += 1
            self._zero_copy_stats["memory_copy_avoided"] += frame_data.nbytes
            
            processing_time = time.time() - start_time
            self._zero_copy_stats["total_processing_time"] += processing_time
            
            # 构建结果
            result = {
                "analysis_result": analysis_result,
                "frame_metadata": metadata.to_dict(),
                "processing_time": processing_time,
                "zero_copy_enabled": True,
                "in_place_analysis": self.enable_in_place_analysis,
                "memory_saved_bytes": frame_data.nbytes,
            }
            
            return result
            
        except Exception as e:
            exception_logger.exception(f"零拷贝帧处理失败: {str(e)}")
            # 回退到传统处理
            frame_data = frame_ref.get_data()
            if frame_data is not None:
                return await self.process_video_frame(frame_data, **kwargs)
            else:
                raise e
    
    async def process_frame_references_batch(self, frame_refs: List[FrameReference], **kwargs) -> List[Dict[str, Any]]:
        """
        批量处理帧引用（零拷贝）
        
        Args:
            frame_refs: 帧引用列表
            **kwargs: 其他参数
            
        Returns:
            List[Dict[str, Any]]: 分析结果列表
        """
        start_time = time.time()
        
        try:
            if not self.enable_batch_processing or len(frame_refs) == 1:
                # 逐个处理
                results = []
                for frame_ref in frame_refs:
                    result = await self.process_frame_reference(frame_ref, **kwargs)
                    results.append(result)
                return results
            
            # 提取帧数据和元数据
            frame_data_list = []
            metadata_list = []
            
            for frame_ref in frame_refs:
                frame_data = frame_ref.get_data()
                if frame_data is not None:
                    frame_data_list.append(frame_data)
                    metadata_list.append(frame_ref.get_metadata())
            
            if not frame_data_list:
                raise ValueError("批次中无有效帧数据")
            
            # 执行批量分析
            if self.enable_in_place_analysis:
                analysis_results = await self.analyze_batch_in_place(frame_data_list, metadata_list, **kwargs)
            else:
                analysis_results = await self.detect_batch(frame_data_list, **kwargs)
            
            # 更新统计
            self._zero_copy_stats["batch_operations"] += 1
            self._zero_copy_stats["zero_copy_operations"] += len(frame_data_list)
            
            total_memory_saved = sum(data.nbytes for data in frame_data_list)
            self._zero_copy_stats["memory_copy_avoided"] += total_memory_saved
            
            processing_time = time.time() - start_time
            self._zero_copy_stats["total_processing_time"] += processing_time
            
            # 构建批量结果
            results = []
            for i, (analysis_result, metadata) in enumerate(zip(analysis_results, metadata_list)):
                result = {
                    "analysis_result": analysis_result,
                    "frame_metadata": metadata.to_dict(),
                    "processing_time": processing_time / len(analysis_results),
                    "zero_copy_enabled": True,
                    "batch_processing": True,
                    "batch_size": len(frame_refs),
                    "memory_saved_bytes": frame_data_list[i].nbytes,
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            exception_logger.exception(f"批量零拷贝帧处理失败: {str(e)}")
            # 回退到逐个处理
            results = []
            for frame_ref in frame_refs:
                try:
                    result = await self.process_frame_reference(frame_ref, **kwargs)
                    results.append(result)
                except Exception as inner_e:
                    exception_logger.exception(f"回退处理失败: {str(inner_e)}")
                    # 添加错误结果
                    results.append({
                        "analysis_result": None,
                        "error": str(inner_e),
                        "zero_copy_enabled": False,
                    })
            return results
    
    @abstractmethod
    async def analyze_in_place(self, frame_data: np.ndarray, metadata: Any, **kwargs) -> Dict[str, Any]:
        """
        原地分析（零拷贝）
        
        Args:
            frame_data: 帧数据（numpy数组视图）
            metadata: 帧元数据
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        pass
    
    async def analyze_batch_in_place(self, frame_data_list: List[np.ndarray], metadata_list: List[Any], **kwargs) -> List[Dict[str, Any]]:
        """
        批量原地分析（零拷贝）
        
        Args:
            frame_data_list: 帧数据列表
            metadata_list: 帧元数据列表
            **kwargs: 其他参数
            
        Returns:
            List[Dict[str, Any]]: 分析结果列表
        """
        # 默认实现：逐个处理
        results = []
        for frame_data, metadata in zip(frame_data_list, metadata_list):
            result = await self.analyze_in_place(frame_data, metadata, **kwargs)
            results.append(result)
        return results
    
    async def detect_batch(self, frame_data_list: List[np.ndarray], **kwargs) -> List[Dict[str, Any]]:
        """
        批量检测
        
        Args:
            frame_data_list: 帧数据列表
            **kwargs: 其他参数
            
        Returns:
            List[Dict[str, Any]]: 检测结果列表
        """
        # 默认实现：逐个处理
        results = []
        for frame_data in frame_data_list:
            result = await self.detect(frame_data, **kwargs)
            results.append(result)
        return results
    
    def get_zero_copy_stats(self) -> Dict[str, Any]:
        """
        获取零拷贝统计信息
        
        Returns:
            Dict[str, Any]: 零拷贝统计信息
        """
        return {
            "zero_copy_stats": self._zero_copy_stats.copy(),
            "zero_copy_enabled": self.enable_zero_copy,
            "in_place_analysis_enabled": self.enable_in_place_analysis,
            "batch_processing_enabled": self.enable_batch_processing,
            "max_batch_size": self.max_batch_size,
            "memory_pool_available": self.memory_pool is not None,
        }
    
    def get_analysis_type(self) -> str:
        """获取分析类型"""
        return "zero_copy_base"
