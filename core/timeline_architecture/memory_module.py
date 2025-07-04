"""
零拷贝内存块管理模块 - 流水线架构组件
负责高效的内存分配、时间驱动的清理和零拷贝帧存储
"""
import asyncio
import time
import threading
import uuid
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
from concurrent.futures import ThreadPoolExecutor

@dataclass
class MemoryBlockInfo:
    """内存块信息"""
    block_id: str
    frame_id: str
    data: np.ndarray
    timestamp: float
    metadata: Dict[str, Any]
    reference_count: int = 1
    last_access_time: float = 0.0
    size_bytes: int = 0

class MemoryBlock:
    """内存块 - 支持引用计数的零拷贝内存管理"""
    
    def __init__(self, block_id: str, data: np.ndarray, metadata: Dict[str, Any]):
        self.block_id = block_id
        self.data = data
        self.metadata = metadata.copy()
        self.timestamp = time.time()
        self.reference_count = 1
        self.last_access_time = self.timestamp
        self.size_bytes = data.nbytes if data is not None else 0
        self._lock = threading.RLock()
    
    def add_reference(self) -> bool:
        """增加引用计数"""
        with self._lock:
            if self.reference_count <= 0:
                return False
            self.reference_count += 1
            self.last_access_time = time.time()
            return True
    
    def release_reference(self) -> int:
        """释放引用计数，返回剩余引用数"""
        with self._lock:
            self.reference_count = max(0, self.reference_count - 1)
            self.last_access_time = time.time()
            return self.reference_count
    
    def get_data_view(self) -> Optional[np.ndarray]:
        """获取数据视图（零拷贝）"""
        with self._lock:
            if self.reference_count <= 0 or self.data is None:
                return None
            self.last_access_time = time.time()
            return self.data
    
    @property
    def age(self) -> float:
        """内存块年龄（秒）"""
        return time.time() - self.timestamp
    
    @property
    def is_expired(self) -> bool:
        """是否已过期（超过1秒且无引用）"""
        return self.age > 1.0 or (self.reference_count <= 0)

class MemoryModule:
    """零拷贝内存块管理模块"""
    
    def __init__(self, max_memory_mb: int = 1024, cleanup_interval: float = 0.5):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cleanup_interval = cleanup_interval
        
        # 内存池
        self.memory_blocks: Dict[str, MemoryBlock] = {}
        self.blocks_lock = asyncio.Lock()
        
        # 索引
        self.frame_to_block: Dict[str, str] = {}  # frame_id -> block_id
        
        # 统计信息
        self.stats = {
            "total_blocks": 0,
            "active_blocks": 0,
            "total_memory_bytes": 0,
            "allocation_count": 0,
            "deallocation_count": 0
        }
        
        self.running = False
        self.cleanup_task: Optional[asyncio.Task] = None
        
        print(f"[内存模块] 初始化完成 - 最大内存: {max_memory_mb}MB")
    
    async def start(self):
        """启动内存模块"""
        if self.running:
            return
        
        self.running = True
        self.cleanup_task = asyncio.create_task(self._cleanup_expired_blocks())
        print("[内存模块] 启动完成")
    
    async def stop(self):
        """停止内存模块"""
        if not self.running:
            return
        
        self.running = False
        
        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()
        
        await self._cleanup_all_blocks()
        print("[内存模块] 停止完成")
    
    def store_frame(self, frame_id: str, frame_data: np.ndarray, metadata: Dict[str, Any]) -> Optional[str]:
        """存储帧数据到内存块"""
        try:
            # 检查内存限制
            if (self.stats["total_memory_bytes"] + frame_data.nbytes) > self.max_memory_bytes:
                print(f"[内存模块] 内存不足，拒绝存储帧: {frame_id}")
                return None
            
            # 生成块ID
            block_id = f"block_{uuid.uuid4().hex[:12]}"
            
            # 创建内存块
            enhanced_metadata = metadata.copy()
            enhanced_metadata["frame_id"] = frame_id
            
            memory_block = MemoryBlock(block_id, frame_data, enhanced_metadata)
            
            # 异步存储
            asyncio.create_task(self._store_block_async(block_id, memory_block, frame_id))
            
            return block_id
            
        except Exception as e:
            print(f"[内存模块] 存储帧异常: {frame_id}, {e}")
            return None
    
    async def _store_block_async(self, block_id: str, memory_block: MemoryBlock, frame_id: str):
        """异步存储内存块"""
        try:
            async with self.blocks_lock:
                self.memory_blocks[block_id] = memory_block
                self.frame_to_block[frame_id] = block_id
                
                # 更新统计
                self.stats["total_blocks"] += 1
                self.stats["active_blocks"] += 1
                self.stats["total_memory_bytes"] += memory_block.size_bytes
                self.stats["allocation_count"] += 1
            
        except Exception as e:
            print(f"[内存模块] 异步存储异常: {block_id}, {e}")
    
    def get_frame_data(self, frame_id: str) -> Optional[np.ndarray]:
        """获取帧数据（零拷贝）"""
        try:
            if frame_id not in self.frame_to_block:
                return None
            
            block_id = self.frame_to_block[frame_id]
            if block_id not in self.memory_blocks:
                return None
            
            memory_block = self.memory_blocks[block_id]
            return memory_block.get_data_view()
                
        except Exception as e:
            print(f"[内存模块] 获取帧数据异常: {frame_id}, {e}")
            return None
    
    def get_frame_reference(self, frame_id: str) -> Optional[str]:
        """获取帧引用（内存块ID）"""
        try:
            if frame_id not in self.frame_to_block:
                return None
            
            block_id = self.frame_to_block[frame_id]
            if block_id not in self.memory_blocks:
                return None
            
            memory_block = self.memory_blocks[block_id]
            if memory_block.add_reference():
                return block_id
            
            return None
            
        except Exception as e:
            print(f"[内存模块] 获取帧引用异常: {frame_id}, {e}")
            return None
    
    def release_frame_reference(self, block_id: str) -> bool:
        """释放帧引用"""
        try:
            if block_id not in self.memory_blocks:
                return False
            
            memory_block = self.memory_blocks[block_id]
            remaining_refs = memory_block.release_reference()
            
            # 如果没有引用了，标记为可清理
            if remaining_refs <= 0:
                asyncio.create_task(self._schedule_block_cleanup(block_id))
            
            return True
            
        except Exception as e:
            print(f"[内存模块] 释放帧引用异常: {block_id}, {e}")
            return False
    
    async def _schedule_block_cleanup(self, block_id: str):
        """调度内存块清理"""
        try:
            await asyncio.sleep(0.1)  # 等待一小段时间
            
            if block_id in self.memory_blocks:
                memory_block = self.memory_blocks[block_id]
                if memory_block.reference_count <= 0:
                    await self._remove_memory_block(block_id)
                    
        except Exception as e:
            print(f"[内存模块] 调度清理异常: {block_id}, {e}")
    
    async def _cleanup_expired_blocks(self):
        """清理过期内存块的后台任务"""
        while self.running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                expired_blocks = []
                
                async with self.blocks_lock:
                    for block_id, memory_block in list(self.memory_blocks.items()):
                        if memory_block.is_expired:
                            expired_blocks.append(block_id)
                
                # 清理过期块
                for block_id in expired_blocks:
                    await self._remove_memory_block(block_id)
                
                if expired_blocks:
                    print(f"[内存模块] 清理过期块: {len(expired_blocks)}个")
                
            except Exception as e:
                print(f"[内存模块] 清理异常: {e}")
                await asyncio.sleep(1.0)
    
    async def _remove_memory_block(self, block_id: str):
        """移除内存块"""
        try:
            async with self.blocks_lock:
                if block_id not in self.memory_blocks:
                    return
                
                memory_block = self.memory_blocks[block_id]
                
                # 从索引中移除
                frame_id = memory_block.metadata.get("frame_id")
                if frame_id and frame_id in self.frame_to_block:
                    del self.frame_to_block[frame_id]
                
                # 更新统计
                self.stats["active_blocks"] -= 1
                self.stats["total_memory_bytes"] -= memory_block.size_bytes
                self.stats["deallocation_count"] += 1
                
                # 移除内存块
                del self.memory_blocks[block_id]
                
        except Exception as e:
            print(f"[内存模块] 移除内存块异常: {block_id}, {e}")
    
    async def _cleanup_all_blocks(self):
        """清理所有内存块"""
        try:
            async with self.blocks_lock:
                block_ids = list(self.memory_blocks.keys())
                
                for block_id in block_ids:
                    await self._remove_memory_block(block_id)
                
                self.frame_to_block.clear()
                print(f"[内存模块] 清理所有内存块完成: {len(block_ids)}个")
                
        except Exception as e:
            print(f"[内存模块] 清理所有块异常: {e}")
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """获取内存统计信息"""
        memory_usage_mb = self.stats["total_memory_bytes"] / (1024 * 1024)
        memory_limit_mb = self.max_memory_bytes / (1024 * 1024)
        usage_percent = (memory_usage_mb / memory_limit_mb) * 100 if memory_limit_mb > 0 else 0
        
        return {
            "memory_usage_mb": round(memory_usage_mb, 2),
            "memory_limit_mb": round(memory_limit_mb, 2),
            "usage_percent": round(usage_percent, 2),
            **self.stats
        } 