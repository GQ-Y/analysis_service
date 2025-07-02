"""
内存管理配置模块
提供内存池、帧缓冲、回收策略等配置管理
"""
from typing import List, Dict, Tuple, Any
from pydantic import BaseModel, Field, validator
import psutil

# 使用项目现有的日志系统
try:
    from shared.utils.logger import get_normal_logger
    logger = get_normal_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class MemoryConfig(BaseModel):
    """内存管理配置类"""
    
    # ============================================================================
    # 基础内存池配置
    # ============================================================================
    
    enable_memory_pool: bool = Field(
        True, 
        description="是否启用内存池管理"
    )
    
    max_memory_usage_percent: float = Field(
        75.0, 
        ge=10.0, 
        le=90.0,
        description="最大使用系统内存的百分比（10-90%）"
    )
    
    min_free_memory_gb: float = Field(
        2.0, 
        ge=0.5, 
        le=8.0,
        description="至少保留的系统内存（GB）"
    )
    
    # ============================================================================
    # 支持的分辨率配置
    # ============================================================================
    
    supported_resolutions: List[Tuple[int, int]] = Field(
        default=[
            (1920, 1080),  # 1080p
            (1280, 720),   # 720p
            (640, 480),    # VGA
            (320, 240),    # QVGA
            (3840, 2160),  # 4K
        ],
        description="支持的视频分辨率列表"
    )
    
    # 每种分辨率的内存块数量配置
    blocks_per_resolution: Dict[str, int] = Field(
        default={
            "1920x1080": 100,   # 1080p: 100个块
            "1280x720": 150,    # 720p: 150个块
            "640x480": 200,     # VGA: 200个块
            "320x240": 300,     # QVGA: 300个块
            "3840x2160": 50,    # 4K: 50个块
        },
        description="每种分辨率预分配的内存块数量"
    )
    
    # ============================================================================
    # 内存回收策略配置
    # ============================================================================
    
    auto_cleanup_interval: int = Field(
        30, 
        ge=5, 
        le=300,
        description="自动清理间隔（秒）"
    )
    
    max_block_age: int = Field(
        300, 
        ge=60, 
        le=3600,
        description="内存块最大存活时间（秒）"
    )
    
    memory_pressure_threshold: float = Field(
        0.9, 
        ge=0.5, 
        le=0.99,
        description="内存压力阈值（0.5-0.99）"
    )
    
    force_cleanup_threshold: float = Field(
        0.95, 
        ge=0.8, 
        le=0.99,
        description="强制清理阈值（0.8-0.99）"
    )
    
    # ============================================================================
    # 性能优化配置
    # ============================================================================
    
    enable_memory_alignment: bool = Field(
        True,
        description="启用内存对齐优化"
    )
    
    alignment_bytes: int = Field(
        64, 
        description="内存对齐字节数（SIMD优化）"
    )
    
    enable_numa_awareness: bool = Field(
        True,
        description="启用NUMA感知优化"
    )
    
    prefault_pages: bool = Field(
        True,
        description="预分配页面，避免运行时页面错误"
    )
    
    # ============================================================================
    # 监控和调试配置
    # ============================================================================
    
    enable_memory_monitoring: bool = Field(
        True,
        description="启用内存使用监控"
    )
    
    monitoring_interval: int = Field(
        10, 
        ge=1, 
        le=60,
        description="内存监控间隔（秒）"
    )
    
    enable_memory_debug: bool = Field(
        False,
        description="启用内存调试模式"
    )
    
    # ============================================================================
    # 验证器
    # ============================================================================
    
    @validator('blocks_per_resolution')
    def validate_blocks_per_resolution(cls, v, values):
        """验证分辨率块数量配置"""
        if 'supported_resolutions' in values:
            supported_res = values['supported_resolutions']
            for width, height in supported_res:
                res_key = f"{width}x{height}"
                if res_key not in v:
                    logger.warning(f"分辨率 {res_key} 未配置内存块数量，使用默认值100")
                    v[res_key] = 100
        return v
    
    @validator('max_memory_usage_percent')
    def validate_memory_percent(cls, v):
        """验证内存使用百分比"""
        if v > 85.0:
            logger.warning(f"内存使用百分比 {v}% 较高，建议不超过85%")
        return v
    
    def calculate_total_memory_requirement(self) -> int:
        """
        计算总内存需求（字节）
        
        Returns:
            int: 总内存需求（字节）
        """
        total_bytes = 0
        
        for width, height in self.supported_resolutions:
            res_key = f"{width}x{height}"
            block_count = self.blocks_per_resolution.get(res_key, 100)
            
            # 计算单个帧的内存需求（RGB 3通道）
            frame_size = width * height * 3
            
            # 考虑内存对齐
            if self.enable_memory_alignment:
                frame_size = ((frame_size + self.alignment_bytes - 1) 
                             // self.alignment_bytes) * self.alignment_bytes
            
            # 计算该分辨率的总内存需求
            resolution_memory = frame_size * block_count
            total_bytes += resolution_memory
            
            logger.debug(f"分辨率 {res_key}: {block_count}个块, "
                        f"每块{frame_size/1024/1024:.2f}MB, "
                        f"小计{resolution_memory/1024/1024:.2f}MB")
        
        logger.info(f"总内存需求: {total_bytes/1024/1024/1024:.2f}GB")
        return total_bytes
    
    def validate_system_memory(self) -> bool:
        """
        验证系统内存是否满足配置要求
        
        Returns:
            bool: 是否满足要求
        """
        # 获取系统内存信息
        memory_info = psutil.virtual_memory()
        total_memory = memory_info.total
        available_memory = memory_info.available
        
        # 计算可用内存的75%（按配置的百分比）
        max_usable = available_memory * (self.max_memory_usage_percent / 100.0)
        
        # 确保至少保留配置的最小内存
        min_free_bytes = self.min_free_memory_gb * 1024 * 1024 * 1024
        if available_memory - max_usable < min_free_bytes:
            max_usable = available_memory - min_free_bytes
            if max_usable < 0:
                max_usable = 0
        
        # 计算需要的内存
        required_memory = self.calculate_total_memory_requirement()
        
        logger.info(f"系统内存检查:")
        logger.info(f"  总内存: {total_memory/1024/1024/1024:.2f}GB")
        logger.info(f"  可用内存: {available_memory/1024/1024/1024:.2f}GB")
        logger.info(f"  最大可用: {max_usable/1024/1024/1024:.2f}GB ({self.max_memory_usage_percent}%)")
        logger.info(f"  需要内存: {required_memory/1024/1024/1024:.2f}GB")
        logger.info(f"  保留内存: {self.min_free_memory_gb}GB")
        
        if required_memory > max_usable:
            logger.error(f"内存不足: 需要{required_memory/1024/1024/1024:.2f}GB, "
                        f"可用{max_usable/1024/1024/1024:.2f}GB")
            return False
        
        logger.info("内存检查通过")
        return True
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """
        获取内存配置摘要
        
        Returns:
            Dict[str, Any]: 内存配置摘要
        """
        memory_info = psutil.virtual_memory()
        required_memory = self.calculate_total_memory_requirement()
        
        return {
            "config": {
                "enable_memory_pool": self.enable_memory_pool,
                "max_memory_usage_percent": self.max_memory_usage_percent,
                "min_free_memory_gb": self.min_free_memory_gb,
                "supported_resolutions": len(self.supported_resolutions),
                "total_blocks": sum(self.blocks_per_resolution.values()),
            },
            "system": {
                "total_memory_gb": memory_info.total / 1024 / 1024 / 1024,
                "available_memory_gb": memory_info.available / 1024 / 1024 / 1024,
                "memory_usage_percent": memory_info.percent,
            },
            "requirements": {
                "required_memory_gb": required_memory / 1024 / 1024 / 1024,
                "memory_valid": self.validate_system_memory(),
            }
        }


# 默认内存配置实例
default_memory_config = MemoryConfig()
