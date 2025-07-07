#!/usr/bin/env python3
"""
分析任务ROI检测中间件
用于根据任务配置过滤检测结果
"""

from typing import List, Dict, Any, Optional, Callable
import logging
from dataclasses import dataclass

from app.models.analysis_result import AnalysisResult


@dataclass
class ROIConfig:
    """ROI配置"""
    # ROI坐标 (x1, y1, x2, y2) 或者多边形坐标
    coordinates: List[List[int]]  # [[x1,y1,x2,y2]] 或 [[x1,y1],[x2,y2],[x3,y3],...]
    roi_type: str = "rectangle"  # rectangle 或 polygon
    enabled: bool = True


class DetectionFilter:
    """检测结果过滤中间件"""
    
    def __init__(
        self,
        roi_config: Optional[Dict[str, Any]] = None,
        target_classes: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化检测过滤器
        
        Args:
            roi_config: ROI区域配置
            target_classes: 目标检测类别列表
            logger: 日志记录器
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # 解析ROI配置
        self.roi_config = None
        if roi_config and roi_config.get("enabled", False):
            try:
                self.roi_config = ROIConfig(
                    coordinates=roi_config.get("coordinates", []),
                    roi_type=roi_config.get("type", "rectangle"),
                    enabled=True
                )
                self.logger.info(f"🎯 ROI过滤器已启用: {self.roi_config.roi_type} 区域")
            except Exception as e:
                self.logger.error(f"❌ ROI配置解析失败: {e}")
                self.roi_config = None
        
        # 目标类别过滤
        self.target_classes = set(target_classes) if target_classes else None
        if self.target_classes:
            self.logger.info(f"🎯 类别过滤器已启用: {self.target_classes}")
        
        # 统计信息
        self.total_processed = 0
        self.roi_filtered = 0
        self.class_filtered = 0
        self.passed_count = 0
    
    def filter_result(self, result: AnalysisResult) -> Optional[AnalysisResult]:
        """
        过滤分析结果
        
        Args:
            result: 原始分析结果
            
        Returns:
            过滤后的分析结果，如果被完全过滤则返回None
        """
        if not result or not result.detections:
            return result
        
        self.total_processed += 1
        
        # 过滤检测结果
        filtered_detections = []
        
        for detection in result.detections:
            # 1. ROI过滤
            if self.roi_config and not self._is_in_roi(detection):
                self.roi_filtered += 1
                continue
            
            # 2. 类别过滤
            if self.target_classes and detection.class_name not in self.target_classes:
                self.class_filtered += 1
                continue
            
            # 通过所有过滤条件
            filtered_detections.append(detection)
        
        # 如果所有检测都被过滤掉了
        if not filtered_detections:
            return None
        
        # 创建过滤后的结果
        filtered_result = AnalysisResult(
            frame_id=result.frame_id,
            timestamp=result.timestamp,
            model_name=result.model_name,
            image_shape=result.image_shape,
            detections=filtered_detections
        )
        
        self.passed_count += 1
        
        # 定期输出统计信息
        if self.total_processed % 100 == 0:
            self._log_statistics()
        
        return filtered_result
    
    def _is_in_roi(self, detection) -> bool:
        """检查检测结果是否在ROI区域内"""
        if not self.roi_config:
            return True
        
        try:
            # 获取检测框中心点
            x1, y1, x2, y2 = detection.bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            if self.roi_config.roi_type == "rectangle":
                return self._point_in_rectangle(center_x, center_y)
            elif self.roi_config.roi_type == "polygon":
                return self._point_in_polygon(center_x, center_y)
            else:
                self.logger.warning(f"⚠️ 不支持的ROI类型: {self.roi_config.roi_type}")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ ROI检查失败: {e}")
            return True
    
    def _point_in_rectangle(self, x: float, y: float) -> bool:
        """检查点是否在矩形内"""
        if not self.roi_config.coordinates:
            return True
        
        try:
            x1, y1, x2, y2 = self.roi_config.coordinates[0]
            return x1 <= x <= x2 and y1 <= y <= y2
        except Exception as e:
            self.logger.error(f"❌ 矩形ROI检查失败: {e}")
            return True
    
    def _point_in_polygon(self, x: float, y: float) -> bool:
        """检查点是否在多边形内（射线法）"""
        if not self.roi_config.coordinates:
            return True
        
        try:
            polygon = self.roi_config.coordinates
            n = len(polygon)
            inside = False
            
            p1x, p1y = polygon[0]
            for i in range(1, n + 1):
                p2x, p2y = polygon[i % n]
                if y > min(p1y, p2y):
                    if y <= max(p1y, p2y):
                        if x <= max(p1x, p2x):
                            if p1y != p2y:
                                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or x <= xinters:
                                inside = not inside
                p1x, p1y = p2x, p2y
            
            return inside
            
        except Exception as e:
            self.logger.error(f"❌ 多边形ROI检查失败: {e}")
            return True
    
    def _log_statistics(self):
        """输出过滤统计信息"""
        if self.total_processed > 0:
            roi_rate = (self.roi_filtered / self.total_processed) * 100
            class_rate = (self.class_filtered / self.total_processed) * 100
            pass_rate = (self.passed_count / self.total_processed) * 100
            
            self.logger.info(
                f"🔍 检测过滤统计: 总计{self.total_processed} "
                f"ROI过滤{self.roi_filtered}({roi_rate:.1f}%) "
                f"类别过滤{self.class_filtered}({class_rate:.1f}%) "
                f"通过{self.passed_count}({pass_rate:.1f}%)"
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取过滤统计信息"""
        return {
            "total_processed": self.total_processed,
            "roi_filtered": self.roi_filtered,
            "class_filtered": self.class_filtered,
            "passed_count": self.passed_count,
            "roi_filter_rate": (self.roi_filtered / max(1, self.total_processed)) * 100,
            "class_filter_rate": (self.class_filtered / max(1, self.total_processed)) * 100,
            "pass_rate": (self.passed_count / max(1, self.total_processed)) * 100
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self.total_processed = 0
        self.roi_filtered = 0
        self.class_filtered = 0
        self.passed_count = 0


class FilterPipeline:
    """过滤器管道"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.filters: List[DetectionFilter] = []
        self.next_processors: List[Callable] = []
    
    def add_filter(self, filter_instance: DetectionFilter):
        """添加过滤器"""
        self.filters.append(filter_instance)
        self.logger.info(f"➕ 添加过滤器: {filter_instance.__class__.__name__}")
    
    def add_processor(self, processor: Callable):
        """添加下游处理器"""
        self.next_processors.append(processor)
        self.logger.info(f"➕ 添加处理器: {processor.__name__ if hasattr(processor, '__name__') else str(processor)}")
    
    def process(self, result: AnalysisResult):
        """处理分析结果"""
        if not result:
            return
        
        # 依次通过所有过滤器
        filtered_result = result
        for filter_instance in self.filters:
            filtered_result = filter_instance.filter_result(filtered_result)
            if not filtered_result:
                # 被过滤掉了
                return
        
        # 传递给下游处理器
        for processor in self.next_processors:
            try:
                processor(filtered_result)
            except Exception as e:
                self.logger.error(f"❌ 处理器执行失败: {e}") 