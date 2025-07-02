"""
内存池初始化器
负责启动时的系统检查、内存池预分配和配置验证
"""
import os
import sys
import time
from typing import Dict, Any, Optional
import psutil

from ..memory.memory_pool import MemoryPool
from ..memory.memory_utils import check_system_memory, calculate_memory_requirements

# 使用项目现有的日志系统
try:
    from shared.utils.logger import get_normal_logger
    logger = get_normal_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class MemoryInitializer:
    """
    内存池初始化器
    负责系统检查、配置验证和内存池初始化
    """
    
    def __init__(self, config):
        """
        初始化内存初始化器

        Args:
            config: 内存配置（MemoryConfig类型）
        """
        self.config = config
        self.memory_pool = None
        self.initialization_start_time = None
        self.initialization_end_time = None
        
        logger.info("内存初始化器创建完成")
    
    def check_system_requirements(self) -> bool:
        """
        检查系统内存要求（75%规则）
        
        Returns:
            bool: 是否满足系统要求
        """
        try:
            logger.info("开始检查系统内存要求...")
            
            # 获取系统内存信息
            memory_info = check_system_memory()
            if not memory_info:
                logger.error("无法获取系统内存信息")
                return False
            
            total_memory = memory_info["total_memory_bytes"]
            available_memory = memory_info["available_memory_bytes"]
            used_memory = memory_info["used_memory_bytes"]
            memory_usage_percent = memory_info["memory_usage_percent"]
            
            # 记录系统信息
            logger.info(f"系统平台: {memory_info.get('platform', 'Unknown')}")
            logger.info(f"系统架构: {memory_info.get('architecture', 'Unknown')}")
            logger.info(f"总内存: {total_memory/1024/1024/1024:.2f}GB")
            logger.info(f"已用内存: {used_memory/1024/1024/1024:.2f}GB ({memory_usage_percent:.1f}%)")
            logger.info(f"可用内存: {available_memory/1024/1024/1024:.2f}GB")
            
            # 检查最小内存要求
            min_total_memory = 4 * 1024 * 1024 * 1024  # 4GB
            if total_memory < min_total_memory:
                logger.error(f"系统总内存不足: {total_memory/1024/1024/1024:.2f}GB < 4GB")
                return False
            
            # 检查当前内存使用率
            if memory_usage_percent > 90:
                logger.error(f"当前内存使用率过高: {memory_usage_percent:.1f}% > 90%")
                return False
            
            # 计算可用内存的75%（按配置的百分比）
            max_usable = available_memory * (self.config.max_memory_usage_percent / 100.0)
            
            # 确保至少保留配置的最小内存
            min_free_bytes = self.config.min_free_memory_gb * 1024 * 1024 * 1024
            if available_memory - max_usable < min_free_bytes:
                max_usable = available_memory - min_free_bytes
                if max_usable <= 0:
                    logger.error(f"可用内存不足，无法保留{self.config.min_free_memory_gb}GB")
                    return False
            
            # 计算内存需求
            requirements = self._calculate_memory_requirements()
            required_memory = requirements["total_bytes"]
            
            logger.info(f"内存分配策略:")
            logger.info(f"  最大使用比例: {self.config.max_memory_usage_percent}%")
            logger.info(f"  保留内存: {self.config.min_free_memory_gb}GB")
            logger.info(f"  最大可用: {max_usable/1024/1024/1024:.2f}GB")
            logger.info(f"  需要内存: {required_memory/1024/1024/1024:.2f}GB")
            
            # 检查内存是否足够
            if required_memory > max_usable:
                logger.error(f"内存不足: 需要{required_memory/1024/1024/1024:.2f}GB, "
                            f"可用{max_usable/1024/1024/1024:.2f}GB")
                return False
            
            # 计算内存使用率
            usage_ratio = required_memory / max_usable
            logger.info(f"预计内存使用率: {usage_ratio*100:.1f}%")
            
            # 警告高内存使用率
            if usage_ratio > 0.8:
                logger.warning("内存使用率较高，可能影响系统性能")
            elif usage_ratio > 0.9:
                logger.warning("内存使用率很高，建议减少内存块数量或增加系统内存")
            
            logger.info("系统内存要求检查通过")
            return True
            
        except Exception as e:
            logger.error(f"系统内存要求检查失败: {str(e)}")
            return False
    
    def _calculate_memory_requirements(self) -> Dict[str, Any]:
        """
        计算内存需求
        
        Returns:
            Dict[str, Any]: 内存需求详情
        """
        return calculate_memory_requirements(
            resolutions=self.config.supported_resolutions,
            blocks_per_resolution=self.config.blocks_per_resolution,
            channels=3,  # RGB
            alignment=self.config.alignment_bytes if self.config.enable_memory_alignment else 1
        )
    
    def validate_configuration(self) -> bool:
        """
        验证内存配置
        
        Returns:
            bool: 配置是否有效
        """
        try:
            logger.info("开始验证内存配置...")
            
            # 验证基本配置
            if not self.config.enable_memory_pool:
                logger.warning("内存池已禁用")
                return True
            
            # 验证分辨率配置
            if not self.config.supported_resolutions:
                logger.error("未配置支持的分辨率")
                return False
            
            # 验证内存块数量配置
            total_blocks = sum(self.config.blocks_per_resolution.values())
            if total_blocks == 0:
                logger.error("内存块总数为0")
                return False
            
            logger.info(f"配置验证:")
            logger.info(f"  支持分辨率: {len(self.config.supported_resolutions)}种")
            logger.info(f"  总内存块: {total_blocks}个")
            logger.info(f"  内存对齐: {self.config.enable_memory_alignment}")
            logger.info(f"  对齐字节: {self.config.alignment_bytes}")
            logger.info(f"  自动清理: {self.config.auto_cleanup_interval}秒")
            
            # 验证清理配置
            if self.config.auto_cleanup_interval > 0:
                if self.config.max_block_age <= self.config.auto_cleanup_interval:
                    logger.warning("内存块最大年龄小于等于清理间隔，可能导致频繁清理")
            
            # 验证内存压力阈值
            if self.config.memory_pressure_threshold >= self.config.force_cleanup_threshold:
                logger.error("内存压力阈值应小于强制清理阈值")
                return False
            
            logger.info("内存配置验证通过")
            return True
            
        except Exception as e:
            logger.error(f"内存配置验证失败: {str(e)}")
            return False
    
    def initialize_memory_pool(self) -> Optional[MemoryPool]:
        """
        初始化内存池
        
        Returns:
            Optional[MemoryPool]: 初始化的内存池，失败返回None
        """
        try:
            self.initialization_start_time = time.time()
            
            logger.info("开始初始化内存池...")
            
            # 1. 检查系统要求
            if not self.check_system_requirements():
                logger.error("系统要求检查失败")
                return None
            
            # 2. 验证配置
            if not self.validate_configuration():
                logger.error("配置验证失败")
                return None
            
            # 3. 创建内存池
            self.memory_pool = MemoryPool(self.config)
            
            # 4. 初始化内存池
            if not self.memory_pool.initialize():
                logger.error("内存池初始化失败")
                return None
            
            self.initialization_end_time = time.time()
            initialization_time = self.initialization_end_time - self.initialization_start_time
            
            logger.info(f"内存池初始化成功，耗时: {initialization_time:.2f}秒")
            
            # 记录初始化摘要
            self._log_initialization_summary()
            
            return self.memory_pool
            
        except Exception as e:
            logger.error(f"内存池初始化异常: {str(e)}")
            return None
    
    def _log_initialization_summary(self) -> None:
        """记录初始化摘要"""
        if not self.memory_pool:
            return
        
        try:
            stats = self.memory_pool.get_pool_stats()
            requirements = self._calculate_memory_requirements()
            
            logger.info("=== 内存池初始化摘要 ===")
            logger.info(f"初始化时间: {self.initialization_end_time - self.initialization_start_time:.2f}秒")
            logger.info(f"支持分辨率: {len(self.config.supported_resolutions)}种")
            
            # 按分辨率详细信息
            for resolution, info in requirements["resolutions"].items():
                logger.info(f"  {resolution}: {info['block_count']}个块, "
                           f"{info['total_memory_mb']:.1f}MB")
            
            logger.info(f"总内存块: {stats['block_stats']['total_blocks']}个")
            logger.info(f"总内存: {requirements['total_gb']:.2f}GB")
            logger.info(f"内存对齐: {self.config.alignment_bytes}字节")
            logger.info(f"清理间隔: {self.config.auto_cleanup_interval}秒")
            logger.info("========================")
            
        except Exception as e:
            logger.warning(f"记录初始化摘要失败: {str(e)}")
    
    def get_initialization_info(self) -> Dict[str, Any]:
        """
        获取初始化信息
        
        Returns:
            Dict[str, Any]: 初始化信息
        """
        info = {
            "initialized": self.memory_pool is not None and self.memory_pool.initialized,
            "start_time": self.initialization_start_time,
            "end_time": self.initialization_end_time,
            "duration_seconds": None,
            "config_summary": {
                "enable_memory_pool": self.config.enable_memory_pool,
                "max_memory_usage_percent": self.config.max_memory_usage_percent,
                "min_free_memory_gb": self.config.min_free_memory_gb,
                "supported_resolutions": len(self.config.supported_resolutions),
                "total_blocks": sum(self.config.blocks_per_resolution.values()),
            }
        }
        
        if self.initialization_start_time and self.initialization_end_time:
            info["duration_seconds"] = self.initialization_end_time - self.initialization_start_time
        
        if self.memory_pool:
            info["memory_pool_stats"] = self.memory_pool.get_pool_stats()
        
        return info
    
    def cleanup(self) -> None:
        """清理资源"""
        if self.memory_pool:
            self.memory_pool.cleanup()
            self.memory_pool = None
            logger.info("内存初始化器清理完成")


def create_memory_initializer(config=None) -> MemoryInitializer:
    """
    创建内存初始化器

    Args:
        config: 内存配置，为None时使用默认配置

    Returns:
        MemoryInitializer: 内存初始化器实例
    """
    if config is None:
        from core.config import settings
        config = settings.memory

    return MemoryInitializer(config)


async def initialize_memory_system(config=None) -> Optional[MemoryPool]:
    """
    初始化内存系统（便捷函数）

    Args:
        config: 内存配置，为None时使用统一配置中的内存配置

    Returns:
        Optional[MemoryPool]: 初始化的内存池，失败返回None
    """
    if config is None:
        from core.config import settings
        config = settings.memory

    initializer = create_memory_initializer(config)
    return initializer.initialize_memory_pool()
