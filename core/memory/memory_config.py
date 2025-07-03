from typing import Field

class MemoryConfig:
    enable_memory_alignment: bool = Field(
        True,
        description="启用内存对齐优化"
    )

    # ============================================================================
    # 动态扩容配置
    # ============================================================================

    enable_dynamic_expansion: bool = Field(
        True,
        description="当某一分辨率的空闲内存块耗尽时，是否允许内存池自动扩容"
    )

    dynamic_expand_block_count: int = Field(
        20,
        ge=1,
        le=1000,
        description="单次动态扩容时为缺少的分辨率新增的内存块数量"
    )

    alignment_bytes: int = Field(
        64, 
        description="内存对齐字节数（SIMD优化）"
    ) 