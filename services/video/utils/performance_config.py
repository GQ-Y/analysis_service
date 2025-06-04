"""
直播流性能配置管理
提供统一的性能优化配置管理
"""
from typing import Dict, Any, Optional
import threading
from enum import Enum

from shared.utils.logger import get_normal_logger

normal_logger = get_normal_logger(__name__)


class PerformanceMode(Enum):
    """性能模式枚举"""
    HIGH_QUALITY = "high_quality"       # 高质量模式：最佳视觉效果，性能较低
    BALANCED = "balanced"               # 平衡模式：质量和性能的平衡
    HIGH_PERFORMANCE = "high_performance"  # 高性能模式：最佳性能，质量较低


class PerformanceConfig:
    """性能配置管理器"""
    
    def __init__(self):
        self.current_mode = PerformanceMode.BALANCED
        self.lock = threading.RLock()
        
        # 性能配置参数
        self.config = {
            PerformanceMode.HIGH_QUALITY: {
                "frame_cache": {
                    "max_size": 100,
                    "ttl_seconds": 1.0
                },
                "frame_skipper": {
                    "stability_threshold": 5,
                    "skip_ratio": 0.1
                },
                "renderer": {
                    "enable_text_rendering": True,
                    "enable_debug_info": True,
                    "max_detections_render": 100,
                    "thickness": 2,
                    "use_chinese_fonts": True
                }
            },
            PerformanceMode.BALANCED: {
                "frame_cache": {
                    "max_size": 50,
                    "ttl_seconds": 3.0
                },
                "frame_skipper": {
                    "stability_threshold": 3,
                    "skip_ratio": 0.3
                },
                "renderer": {
                    "enable_text_rendering": True,
                    "enable_debug_info": False,
                    "max_detections_render": 50,
                    "thickness": 2,
                    "use_chinese_fonts": False
                }
            },
            PerformanceMode.HIGH_PERFORMANCE: {
                "frame_cache": {
                    "max_size": 30,
                    "ttl_seconds": 5.0
                },
                "frame_skipper": {
                    "stability_threshold": 2,
                    "skip_ratio": 0.5
                },
                "renderer": {
                    "enable_text_rendering": False,
                    "enable_debug_info": False,
                    "max_detections_render": 30,
                    "thickness": 1,
                    "use_chinese_fonts": False
                }
            }
        }
        
        normal_logger.info(f"性能配置管理器初始化完成，当前模式: {self.current_mode.value}")
    
    def set_performance_mode(self, mode: PerformanceMode) -> bool:
        """
        设置性能模式
        
        Args:
            mode: 性能模式
            
        Returns:
            bool: 是否设置成功
        """
        with self.lock:
            try:
                old_mode = self.current_mode
                self.current_mode = mode
                
                # 应用配置到各个组件
                self._apply_config_to_components()
                
                normal_logger.info(f"性能模式已从 {old_mode.value} 切换到 {mode.value}")
                return True
                
            except Exception as e:
                normal_logger.error(f"设置性能模式失败: {str(e)}")
                return False
    
    def get_current_mode(self) -> PerformanceMode:
        """获取当前性能模式"""
        with self.lock:
            return self.current_mode
    
    def get_config(self, component: str) -> Dict[str, Any]:
        """
        获取指定组件的配置
        
        Args:
            component: 组件名称 (frame_cache, frame_skipper, renderer)
            
        Returns:
            Dict[str, Any]: 组件配置
        """
        with self.lock:
            return self.config[self.current_mode].get(component, {})
    
    def _apply_config_to_components(self):
        """将当前配置应用到各个组件"""
        try:
            # 应用到帧缓存
            cache_config = self.get_config("frame_cache")
            from services.video.utils.frame_cache import frame_cache
            frame_cache.set_ttl(cache_config["ttl_seconds"])
            # 注意：max_size需要重新初始化才能改变
            
            # 应用到智能帧跳过器
            skipper_config = self.get_config("frame_skipper")
            from services.video.utils.smart_frame_skipper import smart_frame_skipper
            smart_frame_skipper.set_skip_ratio(skipper_config["skip_ratio"])
            smart_frame_skipper.set_stability_threshold(skipper_config["stability_threshold"])
            
            # 应用到渲染器
            renderer_config = self.get_config("renderer")
            from services.video.utils.optimized_frame_renderer import optimized_renderer
            optimized_renderer.enable_text_rendering = renderer_config["enable_text_rendering"]
            optimized_renderer.enable_debug_info = renderer_config["enable_debug_info"]
            optimized_renderer.max_detections_render = renderer_config["max_detections_render"]
            optimized_renderer.thickness = renderer_config["thickness"]
            optimized_renderer.use_chinese_fonts = renderer_config["use_chinese_fonts"]
            
            normal_logger.debug(f"性能配置已应用到所有组件: {self.current_mode.value}")
            
        except Exception as e:
            normal_logger.error(f"应用性能配置失败: {str(e)}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        获取所有组件的性能统计
        
        Returns:
            Dict[str, Any]: 性能统计信息
        """
        stats = {
            "current_mode": self.current_mode.value,
            "components": {}
        }
        
        try:
            # 获取帧缓存统计
            from services.video.utils.frame_cache import frame_cache
            stats["components"]["frame_cache"] = frame_cache.get_stats()
            
            # 获取智能帧跳过器统计
            from services.video.utils.smart_frame_skipper import smart_frame_skipper
            stats["components"]["frame_skipper"] = smart_frame_skipper.get_stats()
            
            # 获取渲染器统计
            from services.video.utils.optimized_frame_renderer import optimized_renderer
            stats["components"]["renderer"] = optimized_renderer.get_performance_stats()
            
        except Exception as e:
            normal_logger.error(f"获取性能统计失败: {str(e)}")
            stats["error"] = str(e)
        
        return stats
    
    def reset_all_stats(self):
        """重置所有组件的统计信息"""
        try:
            from services.video.utils.frame_cache import frame_cache
            from services.video.utils.smart_frame_skipper import smart_frame_skipper
            
            frame_cache.clear()
            smart_frame_skipper.reset()
            
            normal_logger.info("所有性能统计已重置")
            
        except Exception as e:
            normal_logger.error(f"重置性能统计失败: {str(e)}")


# 全局性能配置管理器实例
performance_config = PerformanceConfig() 