"""
内存池管理器
提供内存池的初始化、分配、回收和监控功能
"""
import time
import threading
from typing import Dict, List, Optional, Tuple, Any

from .memory_block import MemoryBlock, MemoryBlockManager, MemoryBlockStatus
from .memory_utils import check_system_memory, calculate_memory_requirements
from ..config.memory_config import MemoryConfig

# 使用项目现有的日志系统
try:
    from shared.utils.logger import get_normal_logger
    logger = get_normal_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class MemoryPool:
    """
    内存池管理器
    负责内存池的初始化、内存块的分配和回收
    """
    
    def __init__(self, config: MemoryConfig):
        """
        初始化内存池
        
        Args:
            config: 内存配置
        """
        self.config = config
        self.block_manager = MemoryBlockManager()
        self.initialized = False
        self.initialization_time = None
        
        # 统计信息
        self.stats = {
            "total_allocations": 0,
            "total_deallocations": 0,
            "allocation_failures": 0,
            "peak_usage": 0,
            "current_usage": 0,
        }
        
        # 线程安全
        self.lock = threading.RLock()
        
        # 清理线程控制
        self.cleanup_thread = None
        self.cleanup_stop_event = threading.Event()
        
        logger.info("内存池管理器创建完成")
    
    def initialize(self) -> bool:
        """
        初始化内存池
        执行系统内存检查和内存块预分配
        
        Returns:
            bool: 是否初始化成功
        """
        if self.initialized:
            logger.warning("内存池已经初始化")
            return True
        
        try:
            logger.info("开始初始化内存池...")
            
            # 1. 检查系统内存
            if not self._check_system_memory():
                logger.error("系统内存检查失败")
                return False
            
            # 2. 验证配置
            if not self.config.validate_system_memory():
                logger.error("内存配置验证失败")
                return False
            
            # 3. 预分配内存块
            if not self._preallocate_memory_blocks():
                logger.error("内存块预分配失败")
                return False
            
            # 4. 启动清理线程
            if self.config.auto_cleanup_interval > 0:
                self._start_cleanup_thread()
            
            self.initialized = True
            self.initialization_time = time.time()
            
            logger.info("内存池初始化成功")
            self._log_initialization_summary()
            
            return True
            
        except Exception as e:
            logger.error(f"内存池初始化失败: {str(e)}")
            return False
    
    def _check_system_memory(self) -> bool:
        """
        检查系统内存（75%规则）
        
        Returns:
            bool: 是否满足要求
        """
        try:
            # 获取系统内存信息
            memory_info = check_system_memory()
            if not memory_info:
                logger.error("无法获取系统内存信息")
                return False
            
            total_memory = memory_info["total_memory_bytes"]
            available_memory = memory_info["available_memory_bytes"]
            
            # 计算可用内存的配置百分比
            max_usable = available_memory * (self.config.max_memory_usage_percent / 100.0)
            
            # 确保至少保留配置的最小内存
            min_free_bytes = self.config.min_free_memory_gb * 1024 * 1024 * 1024
            if available_memory - max_usable < min_free_bytes:
                max_usable = available_memory - min_free_bytes
                if max_usable < 0:
                    logger.error(f"可用内存不足，无法保留{self.config.min_free_memory_gb}GB")
                    return False
            
            # 计算需要的内存
            requirements = calculate_memory_requirements(
                self.config.supported_resolutions,
                self.config.blocks_per_resolution,
                channels=3,
                alignment=self.config.alignment_bytes if self.config.enable_memory_alignment else 1
            )
            
            required_memory = requirements["total_bytes"]
            
            logger.info(f"系统内存检查:")
            logger.info(f"  总内存: {total_memory/1024/1024/1024:.2f}GB")
            logger.info(f"  可用内存: {available_memory/1024/1024/1024:.2f}GB")
            logger.info(f"  最大可用: {max_usable/1024/1024/1024:.2f}GB ({self.config.max_memory_usage_percent}%)")
            logger.info(f"  需要内存: {required_memory/1024/1024/1024:.2f}GB")
            logger.info(f"  保留内存: {self.config.min_free_memory_gb}GB")
            
            if required_memory > max_usable:
                logger.error(f"内存不足: 需要{required_memory/1024/1024/1024:.2f}GB, "
                            f"可用{max_usable/1024/1024/1024:.2f}GB")
                return False
            
            # 计算内存使用率
            usage_ratio = required_memory / max_usable
            logger.info(f"内存使用率: {usage_ratio*100:.1f}%")
            
            if usage_ratio > 0.9:
                logger.warning("内存使用率较高，可能影响系统性能")
            
            return True
            
        except Exception as e:
            logger.error(f"系统内存检查异常: {str(e)}")
            return False
    
    def _preallocate_memory_blocks(self) -> bool:
        """
        预分配内存块
        
        Returns:
            bool: 是否预分配成功
        """
        try:
            logger.info("开始预分配内存块...")
            
            total_blocks = 0
            total_memory = 0
            
            # 为每种分辨率创建内存块池
            for width, height in self.config.supported_resolutions:
                resolution_key = f"{width}x{height}"
                block_count = self.config.blocks_per_resolution.get(resolution_key, 100)
                
                logger.info(f"预分配 {resolution_key}: {block_count}个块")
                
                success = self.block_manager.create_block_pool(
                    width=width,
                    height=height,
                    channels=3,  # RGB
                    count=block_count,
                    alignment=self.config.alignment_bytes if self.config.enable_memory_alignment else 1
                )
                
                if not success:
                    logger.error(f"预分配 {resolution_key} 失败")
                    return False
                
                # 计算内存使用量
                frame_size = width * height * 3
                if self.config.enable_memory_alignment:
                    frame_size = ((frame_size + self.config.alignment_bytes - 1) 
                                 // self.config.alignment_bytes) * self.config.alignment_bytes
                
                resolution_memory = frame_size * block_count
                total_memory += resolution_memory
                total_blocks += block_count
                
                logger.info(f"  {resolution_key}: {block_count}个块, "
                           f"每块{frame_size/1024/1024:.2f}MB, "
                           f"小计{resolution_memory/1024/1024:.2f}MB")
            
            logger.info(f"内存块预分配完成: 总计{total_blocks}个块, "
                       f"{total_memory/1024/1024/1024:.2f}GB")
            
            # 更新统计信息
            self.stats["current_usage"] = total_memory
            self.stats["peak_usage"] = total_memory
            
            return True
            
        except Exception as e:
            logger.error(f"内存块预分配失败: {str(e)}")
            return False
    
    def allocate_frame_block(self, width: int, height: int) -> Optional[MemoryBlock]:
        """
        分配帧内存块
        
        Args:
            width: 图像宽度
            height: 图像高度
        
        Returns:
            Optional[MemoryBlock]: 分配的内存块，失败返回None
        """
        if not self.initialized:
            logger.error("内存池未初始化")
            return None
        
        try:
            with self.lock:
                # 从内存块管理器获取内存块
                block = self.block_manager.get_block_by_resolution(width, height)
                
                if block:
                    self.stats["total_allocations"] += 1
                    logger.debug(f"分配内存块: {block.block_id} ({width}x{height})")
                else:
                    self.stats["allocation_failures"] += 1
                    logger.warning(f"分配内存块失败: {width}x{height}")
                
                return block
                
        except Exception as e:
            logger.error(f"分配内存块异常: {str(e)}")
            self.stats["allocation_failures"] += 1
            return None
    
    def deallocate_frame_block(self, block: MemoryBlock) -> bool:
        """
        释放帧内存块
        
        Args:
            block: 要释放的内存块
        
        Returns:
            bool: 是否释放成功
        """
        if not self.initialized:
            logger.error("内存池未初始化")
            return False
        
        try:
            with self.lock:
                # 释放内存块引用
                if block.release():
                    # 如果引用计数为0，归还到空闲池
                    if block.get_ref_count() == 0:
                        success = self.block_manager.return_block(block)
                        if success:
                            self.stats["total_deallocations"] += 1
                            logger.debug(f"释放内存块: {block.block_id}")
                        return success
                    else:
                        # 还有其他引用，只是减少计数
                        logger.debug(f"内存块 {block.block_id} 还有 {block.get_ref_count()} 个引用")
                        return True
                else:
                    logger.warning(f"释放内存块失败: {block.block_id}")
                    return False
                
        except Exception as e:
            logger.error(f"释放内存块异常: {str(e)}")
            return False
    
    def _start_cleanup_thread(self) -> None:
        """启动清理线程"""
        if self.cleanup_thread is None or not self.cleanup_thread.is_alive():
            self.cleanup_stop_event.clear()
            self.cleanup_thread = threading.Thread(
                target=self._cleanup_worker,
                name="MemoryPoolCleanup",
                daemon=True
            )
            self.cleanup_thread.start()
            logger.info("内存池清理线程已启动")
    
    def _cleanup_worker(self) -> None:
        """清理工作线程"""
        while not self.cleanup_stop_event.is_set():
            try:
                self._perform_cleanup()
                self.cleanup_stop_event.wait(self.config.auto_cleanup_interval)
            except Exception as e:
                logger.error(f"内存清理异常: {str(e)}")
                self.cleanup_stop_event.wait(5)  # 出错后等待5秒
    
    def _perform_cleanup(self) -> None:
        """执行内存清理"""
        current_time = time.time()
        cleaned_count = 0
        
        with self.lock:
            # 检查所有内存块
            for block in self.block_manager.blocks.values():
                # 清理超时的内存块
                if (block.status == MemoryBlockStatus.PENDING_FREE and 
                    current_time - block.freed_time > self.config.max_block_age):
                    
                    if self.block_manager.return_block(block):
                        cleaned_count += 1
        
        if cleaned_count > 0:
            logger.debug(f"清理了 {cleaned_count} 个超时内存块")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """
        获取内存池统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        with self.lock:
            block_stats = self.block_manager.get_stats()
            
            return {
                "initialized": self.initialized,
                "initialization_time": self.initialization_time,
                "uptime_seconds": time.time() - self.initialization_time if self.initialization_time else 0,
                "config": {
                    "max_memory_usage_percent": self.config.max_memory_usage_percent,
                    "min_free_memory_gb": self.config.min_free_memory_gb,
                    "supported_resolutions": len(self.config.supported_resolutions),
                    "auto_cleanup_interval": self.config.auto_cleanup_interval,
                },
                "memory_stats": self.stats.copy(),
                "block_stats": block_stats,
            }
    
    def _log_initialization_summary(self) -> None:
        """记录初始化摘要"""
        stats = self.get_pool_stats()
        
        logger.info("=== 内存池初始化摘要 ===")
        logger.info(f"支持分辨率: {stats['config']['supported_resolutions']}种")
        logger.info(f"总内存块: {stats['block_stats']['total_blocks']}个")
        logger.info(f"内存使用: {self.stats['current_usage']/1024/1024/1024:.2f}GB")
        logger.info(f"清理间隔: {stats['config']['auto_cleanup_interval']}秒")
        logger.info("========================")
    
    def cleanup(self) -> None:
        """清理内存池资源"""
        logger.info("开始清理内存池...")
        
        # 停止清理线程
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_stop_event.set()
            self.cleanup_thread.join(timeout=5)
            logger.info("清理线程已停止")
        
        # 强制释放所有内存块
        with self.lock:
            for block in self.block_manager.blocks.values():
                if block.status != MemoryBlockStatus.FREE:
                    block.force_free()
        
        self.initialized = False
        logger.info("内存池清理完成")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取内存池统计信息

        Returns:
            Dict[str, Any]: 内存池统计信息
        """
        if not self.initialized:
            return {
                "initialized": False,
                "error": "内存池未初始化"
            }

        with self.lock:
            # 基础统计
            total_blocks = len(self.block_manager.blocks)
            used_blocks = sum(1 for block in self.block_manager.blocks.values()
                            if block.status in [MemoryBlockStatus.ALLOCATED, MemoryBlockStatus.IN_USE])
            available_blocks = total_blocks - used_blocks

            # 按分辨率统计
            resolution_stats = {}
            for resolution_key, resolution_blocks in self.block_manager.blocks_by_resolution.items():
                resolution_used = sum(1 for block in resolution_blocks
                                    if block.status in [MemoryBlockStatus.ALLOCATED, MemoryBlockStatus.IN_USE])
                resolution_available = len(resolution_blocks) - resolution_used

                resolution_stats[resolution_key] = {
                    "total": len(resolution_blocks),
                    "used": resolution_used,
                    "available": resolution_available,
                    "usage_ratio": resolution_used / len(resolution_blocks) if resolution_blocks else 0.0
                }

            # 内存使用统计
            total_memory_bytes = sum(block.size for block in self.block_manager.blocks.values())
            used_memory_bytes = sum(block.size for block in self.block_manager.blocks.values()
                                  if block.status in [MemoryBlockStatus.ALLOCATED, MemoryBlockStatus.IN_USE])

            # 计算使用率
            usage_ratio = used_blocks / total_blocks if total_blocks > 0 else 0.0
            memory_usage_ratio = used_memory_bytes / total_memory_bytes if total_memory_bytes > 0 else 0.0

            return {
                "initialized": True,
                "total_blocks": total_blocks,
                "used_blocks": used_blocks,
                "available_blocks": available_blocks,
                "usage_ratio": usage_ratio,
                "total_memory_mb": total_memory_bytes / (1024 * 1024),
                "used_memory_mb": used_memory_bytes / (1024 * 1024),
                "available_memory_mb": (total_memory_bytes - used_memory_bytes) / (1024 * 1024),
                "memory_usage_ratio": memory_usage_ratio,
                "resolution_stats": resolution_stats,
                "cleanup_interval": self.config.auto_cleanup_interval,
                "last_cleanup": getattr(self, '_last_cleanup_time', None),
                # 操作统计
                "total_allocations": self.stats.get("total_allocations", 0),
                "total_deallocations": self.stats.get("total_deallocations", 0),
                "allocation_failures": self.stats.get("allocation_failures", 0),
            }

    def get_detailed_stats(self) -> Dict[str, Any]:
        """
        获取详细的内存池统计信息

        Returns:
            Dict[str, Any]: 详细统计信息
        """
        basic_stats = self.get_stats()
        if not basic_stats.get("initialized", False):
            return basic_stats

        with self.lock:
            # 内存块详细信息
            block_details = []
            for block_id, block in self.block_manager.blocks.items():
                block_details.append({
                    "block_id": block_id,
                    "resolution": f"{block.width}x{block.height}",
                    "size_mb": block.size / (1024 * 1024),
                    "status": block.status.value,
                    "ref_count": block.ref_count,
                    "created_time": block.created_time,
                    "last_accessed": block.last_accessed,
                })

            # 性能统计
            performance_stats = {
                "total_allocations": self.stats.get("total_allocations", 0),
                "total_deallocations": self.stats.get("total_deallocations", 0),
                "allocation_failures": self.stats.get("allocation_failures", 0),
                "cleanup_cycles": getattr(self, '_cleanup_cycles', 0),
            }

            basic_stats.update({
                "block_details": block_details,
                "performance_stats": performance_stats,
            })

            return basic_stats

    def __del__(self):
        """析构函数"""
        if self.initialized:
            self.cleanup()
