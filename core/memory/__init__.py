"""
内存管理模块
提供零拷贝帧处理的内存管理功能
"""

from ..config.memory_config import MemoryConfig, default_memory_config
from .memory_block import MemoryBlock, MemoryBlockManager, MemoryBlockStatus
from .memory_pool import MemoryPool
from .memory_utils import (
    check_system_memory,
    calculate_memory_requirements,
    align_size,
    align_memory_address,
    benchmark_memory_operations,
    get_optimal_alignment,
    check_numa_topology,
    estimate_memory_pressure
)

__all__ = [
    # 配置
    "MemoryConfig",
    "default_memory_config",
    
    # 内存块
    "MemoryBlock",
    "MemoryBlockManager", 
    "MemoryBlockStatus",
    
    # 内存池
    "MemoryPool",
    
    # 工具函数
    "check_system_memory",
    "calculate_memory_requirements",
    "align_size",
    "align_memory_address",
    "benchmark_memory_operations",
    "get_optimal_alignment",
    "check_numa_topology",
    "estimate_memory_pressure",
]
