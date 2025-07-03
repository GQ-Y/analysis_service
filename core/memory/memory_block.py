"""
内存块定义和管理
提供内存块的创建、引用计数、状态管理等功能
"""
import time
import threading
import ctypes
from typing import Optional, Tuple, Any, Dict
from enum import Enum
import numpy as np

# 使用项目现有的日志系统
try:
    from shared.utils.logger import get_normal_logger
    logger = get_normal_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class MemoryBlockStatus(Enum):
    """内存块状态枚举"""
    FREE = "free"           # 空闲状态
    ALLOCATED = "allocated" # 已分配状态
    IN_USE = "in_use"      # 使用中状态
    PENDING_FREE = "pending_free"  # 待释放状态


class MemoryBlock:
    """
    内存块类
    管理单个内存块的生命周期、引用计数和数据访问
    """
    
    def __init__(self, block_id: int, ptr: ctypes.c_void_p, size: int,
                 width: int, height: int, channels: int = 3, manager=None):
        """
        初始化内存块

        Args:
            block_id: 内存块唯一ID
            ptr: 内存指针
            size: 内存大小（字节）
            width: 图像宽度
            height: 图像高度
            channels: 图像通道数，默认3（RGB）
            manager: 内存块管理器引用
        """
        self.block_id = block_id
        self.ptr = ptr
        self.size = size
        self.width = width
        self.height = height
        self.channels = channels
        self.manager = manager
        
        # 引用计数（原子操作）
        self._ref_count = 0
        self._ref_lock = threading.RLock()
        
        # 状态管理
        self.status = MemoryBlockStatus.FREE
        self._status_lock = threading.RLock()
        
        # 时间戳
        self.created_time = time.time()
        self.last_access_time = time.time()
        self.allocated_time = None
        self.freed_time = None
        
        # 元数据
        self.metadata = {}
        
        logger.debug(f"创建内存块 {block_id}: {width}x{height}x{channels}, {size}字节")
    
    def acquire(self) -> bool:
        """
        获取内存块引用（增加引用计数）
        
        Returns:
            bool: 是否成功获取引用
        """
        with self._ref_lock:
            if self.status == MemoryBlockStatus.FREE:
                # 从空闲状态转为已分配状态
                with self._status_lock:
                    if self.status == MemoryBlockStatus.FREE:
                        self.status = MemoryBlockStatus.ALLOCATED
                        self.allocated_time = time.time()
            
            if self.status in [MemoryBlockStatus.ALLOCATED, MemoryBlockStatus.IN_USE]:
                self._ref_count += 1
                self.last_access_time = time.time()
                
                # 如果是第一次引用，状态转为使用中
                if self._ref_count == 1 and self.status == MemoryBlockStatus.ALLOCATED:
                    with self._status_lock:
                        self.status = MemoryBlockStatus.IN_USE
                
                logger.debug(f"内存块 {self.block_id} 引用计数: {self._ref_count}")
                return True
            else:
                logger.warning(f"内存块 {self.block_id} 状态 {self.status} 不允许获取引用")
                return False
    
    def release(self) -> bool:
        """
        释放内存块引用（减少引用计数）
        当引用计数为0时自动回收到内存池

        Returns:
            bool: 是否成功释放引用
        """
        with self._ref_lock:
            if self._ref_count > 0:
                self._ref_count -= 1
                self.last_access_time = time.time()

                logger.debug(f"内存块 {self.block_id} 引用计数: {self._ref_count}")

                # 如果引用计数为0，自动回收到内存池
                if self._ref_count == 0:
                    with self._status_lock:
                        self.status = MemoryBlockStatus.PENDING_FREE
                        self.freed_time = time.time()

                    # 自动回收到内存池
                    if self.manager:
                        try:
                            success = self.manager.return_block(self)
                            if success:
                                logger.debug(f"内存块 {self.block_id} 自动回收到内存池")
                            else:
                                logger.warning(f"内存块 {self.block_id} 自动回收失败")
                        except Exception as e:
                            logger.error(f"内存块 {self.block_id} 自动回收异常: {str(e)}")

                return True
            else:
                logger.warning(f"内存块 {self.block_id} 引用计数已为0，无法继续释放")
                return False
    
    def force_free(self) -> bool:
        """
        强制释放内存块（忽略引用计数）
        
        Returns:
            bool: 是否成功释放
        """
        with self._ref_lock, self._status_lock:
            if self.status != MemoryBlockStatus.FREE:
                self._ref_count = 0
                self.status = MemoryBlockStatus.FREE
                self.freed_time = time.time()
                self.allocated_time = None

                # 统计强制释放
                if self.manager:
                    self.manager._record_force_free()

                logger.debug(f"强制释放内存块 {self.block_id}")
                return True
            return False
    
    def get_numpy_view(self) -> Optional[np.ndarray]:
        """
        获取内存块的numpy视图（零拷贝）
        
        Returns:
            Optional[np.ndarray]: numpy数组视图，失败返回None
        """
        if self.status not in [MemoryBlockStatus.IN_USE, MemoryBlockStatus.ALLOCATED]:
            logger.error(f"内存块 {self.block_id} 状态 {self.status} 不允许访问数据")
            return None
        
        try:
            # 创建numpy数组视图，不拷贝数据
            buffer = (ctypes.c_uint8 * self.size).from_address(self.ptr.value)
            array = np.frombuffer(buffer, dtype=np.uint8)
            
            # 重塑为图像形状
            if self.channels == 1:
                shaped_array = array.reshape((self.height, self.width))
            else:
                shaped_array = array.reshape((self.height, self.width, self.channels))
            
            self.last_access_time = time.time()
            return shaped_array
            
        except Exception as e:
            logger.error(f"创建内存块 {self.block_id} 的numpy视图失败: {str(e)}")
            return None
    
    def is_available(self) -> bool:
        """
        检查内存块是否可用
        
        Returns:
            bool: 是否可用
        """
        return self.status == MemoryBlockStatus.FREE
    
    def is_in_use(self) -> bool:
        """
        检查内存块是否正在使用
        
        Returns:
            bool: 是否正在使用
        """
        return self.status == MemoryBlockStatus.IN_USE and self._ref_count > 0
    
    def get_ref_count(self) -> int:
        """
        获取当前引用计数
        
        Returns:
            int: 引用计数
        """
        with self._ref_lock:
            return self._ref_count
    
    def get_age(self) -> float:
        """
        获取内存块年龄（秒）
        
        Returns:
            float: 年龄（秒）
        """
        return time.time() - self.created_time
    
    def get_idle_time(self) -> float:
        """
        获取内存块空闲时间（秒）
        
        Returns:
            float: 空闲时间（秒）
        """
        return time.time() - self.last_access_time
    
    def get_info(self) -> Dict[str, Any]:
        """
        获取内存块详细信息
        
        Returns:
            Dict[str, Any]: 内存块信息
        """
        with self._ref_lock, self._status_lock:
            return {
                "block_id": self.block_id,
                "size": self.size,
                "width": self.width,
                "height": self.height,
                "channels": self.channels,
                "status": self.status.value,
                "ref_count": self._ref_count,
                "created_time": self.created_time,
                "last_access_time": self.last_access_time,
                "allocated_time": self.allocated_time,
                "freed_time": self.freed_time,
                "age_seconds": self.get_age(),
                "idle_seconds": self.get_idle_time(),
                "metadata": self.metadata.copy(),
            }
    
    def set_metadata(self, key: str, value: Any) -> None:
        """
        设置元数据
        
        Args:
            key: 元数据键
            value: 元数据值
        """
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        获取元数据
        
        Args:
            key: 元数据键
            default: 默认值
        
        Returns:
            Any: 元数据值
        """
        return self.metadata.get(key, default)
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"MemoryBlock(id={self.block_id}, "
                f"size={self.size}, "
                f"shape={self.width}x{self.height}x{self.channels}, "
                f"status={self.status.value}, "
                f"refs={self._ref_count})")
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return self.__str__()


class MemoryBlockManager:
    """
    内存块管理器
    管理多个内存块的创建、分配、回收
    """
    
    def __init__(self):
        """初始化内存块管理器"""
        self.blocks = {}  # block_id -> MemoryBlock
        self.blocks_by_resolution = {}  # "widthxheight" -> [MemoryBlock]
        self.free_blocks = {}  # "widthxheight" -> [MemoryBlock]
        self.next_block_id = 1
        self.lock = threading.RLock()

        # 强制释放统计
        self.force_free_stats = {
            "total_count": 0,
            "last_report_time": time.time(),
            "report_interval": 60  # 每60秒报告一次
        }

        logger.info("内存块管理器初始化完成")
    
    def create_block_pool(self, width: int, height: int, channels: int, 
                         count: int, alignment: int = 64) -> bool:
        """
        创建指定分辨率的内存块池
        
        Args:
            width: 图像宽度
            height: 图像高度
            channels: 图像通道数
            count: 内存块数量
            alignment: 内存对齐字节数
        
        Returns:
            bool: 是否创建成功
        """
        resolution_key = f"{width}x{height}"
        
        try:
            with self.lock:
                if resolution_key not in self.blocks_by_resolution:
                    self.blocks_by_resolution[resolution_key] = []
                    self.free_blocks[resolution_key] = []
                
                # 计算内存块大小
                frame_size = width * height * channels
                aligned_size = ((frame_size + alignment - 1) // alignment) * alignment
                
                created_count = 0
                for i in range(count):
                    # 分配内存
                    buffer = (ctypes.c_uint8 * aligned_size)()
                    ptr = ctypes.cast(buffer, ctypes.c_void_p)
                    
                    # 创建内存块
                    block = MemoryBlock(
                        block_id=self.next_block_id,
                        ptr=ptr,
                        size=aligned_size,
                        width=width,
                        height=height,
                        channels=channels,
                        manager=self
                    )
                    
                    # 添加到管理器
                    self.blocks[self.next_block_id] = block
                    self.blocks_by_resolution[resolution_key].append(block)
                    self.free_blocks[resolution_key].append(block)
                    
                    self.next_block_id += 1
                    created_count += 1
                
                logger.info(f"创建内存块池: {resolution_key}, "
                           f"{created_count}个块, "
                           f"每块{aligned_size/1024/1024:.2f}MB")
                
                return True
                
        except Exception as e:
            logger.error(f"创建内存块池失败: {str(e)}")
            return False
    
    def get_block_by_resolution(self, width: int, height: int) -> Optional[MemoryBlock]:
        """
        根据分辨率获取可用的内存块

        Args:
            width: 图像宽度
            height: 图像高度

        Returns:
            Optional[MemoryBlock]: 可用的内存块，无可用块返回None
        """
        resolution_key = f"{width}x{height}"

        with self.lock:
            free_list = self.free_blocks.get(resolution_key, [])

            # 尝试获取可用的内存块，最多尝试列表中的所有块
            attempts = 0
            max_attempts = len(free_list)

            while free_list and attempts < max_attempts:
                # 获取第一个空闲块
                block = free_list.pop(0)

                # 检查块状态是否真的是空闲的
                if block.is_available() and block.acquire():
                    logger.debug(f"成功分配内存块 {block.block_id} ({resolution_key})")
                    return block
                else:
                    # 获取失败，说明这个块有问题，不要放回列表
                    logger.warning(f"内存块 {block.block_id} 状态异常: {block.status}, 引用计数: {block.get_ref_count()}")
                    attempts += 1

            logger.warning(f"没有可用的内存块: {resolution_key}, 空闲列表大小: {len(free_list)}")
            return None
    
    def return_block(self, block: MemoryBlock) -> bool:
        """
        归还内存块到空闲池
        
        Args:
            block: 要归还的内存块
        
        Returns:
            bool: 是否成功归还
        """
        resolution_key = f"{block.width}x{block.height}"
        
        with self.lock:
            if block.status == MemoryBlockStatus.PENDING_FREE:
                # 强制释放并标记为空闲
                block.force_free()

                # 添加到空闲列表
                if resolution_key in self.free_blocks:
                    self.free_blocks[resolution_key].append(block)
                    logger.debug(f"归还内存块 {block.block_id} 到空闲池")
                    return True
                else:
                    # 如果分辨率键不存在，创建新的空闲列表
                    self.free_blocks[resolution_key] = [block]
                    logger.debug(f"创建新的空闲列表并归还内存块 {block.block_id}")
                    return True

            logger.warning(f"内存块 {block.block_id} 状态 {block.status} 不允许归还")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取内存块管理器统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        with self.lock:
            stats = {
                "total_blocks": len(self.blocks),
                "resolutions": {},
                "status_counts": {status.value: 0 for status in MemoryBlockStatus},
            }
            
            # 按分辨率统计
            for resolution, blocks in self.blocks_by_resolution.items():
                free_count = len(self.free_blocks.get(resolution, []))
                stats["resolutions"][resolution] = {
                    "total": len(blocks),
                    "free": free_count,
                    "in_use": len(blocks) - free_count,
                }
            
            # 按状态统计
            for block in self.blocks.values():
                stats["status_counts"][block.status.value] += 1
            
            return stats

    def _record_force_free(self):
        """记录强制释放统计"""
        with self.lock:
            self.force_free_stats["total_count"] += 1
            current_time = time.time()

            # 检查是否需要报告统计信息
            if (current_time - self.force_free_stats["last_report_time"] >=
                self.force_free_stats["report_interval"]):

                count = self.force_free_stats["total_count"]
                interval = self.force_free_stats["report_interval"]
                rate = count / interval if interval > 0 else 0

                logger.debug(f"内存块强制释放统计: 过去{interval}秒内共{count}次, 平均{rate:.2f}次/秒")

                # 重置统计
                self.force_free_stats["total_count"] = 0
                self.force_free_stats["last_report_time"] = current_time
