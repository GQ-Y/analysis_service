#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: reference_counter.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 引用计数器

参考timelinetool的引用计数设计，实现自动内存回收机制。

本文件是分析服务项目的一部分。
"""

import threading
import weakref
import logging
from typing import Dict, Any, Optional, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class ReferenceInfo:
    """引用信息"""
    ref_id: str  # 引用ID
    obj_id: str  # 对象ID
    count: int  # 引用计数
    created_at: datetime  # 创建时间
    last_accessed: datetime  # 最后访问时间
    cleanup_callback: Optional[Callable] = None  # 清理回调


class ReferenceCounter:
    """引用计数器"""
    
    def __init__(self, cleanup_interval: float = 60.0):
        """初始化引用计数器
        
        Args:
            cleanup_interval: 清理间隔（秒）
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cleanup_interval = cleanup_interval
        
        # 引用信息存储
        self._references: Dict[str, ReferenceInfo] = {}
        self._object_refs: Dict[str, Set[str]] = {}  # 对象ID到引用ID的映射
        self._weak_refs: Dict[str, weakref.ref] = {}  # 弱引用存储
        
        # 线程安全
        self._lock = threading.RLock()
        
        # 清理相关
        self._cleanup_callbacks: Dict[str, Callable] = {}
        
        self.logger.info("引用计数器初始化完成")
    
    def add_reference(self, obj: Any, ref_id: str = None, cleanup_callback: Callable = None) -> str:
        """添加引用
        
        Args:
            obj: 要引用的对象
            ref_id: 引用ID，如果不提供则自动生成
            cleanup_callback: 清理回调函数
            
        Returns:
            str: 引用ID
        """
        if ref_id is None:
            ref_id = f"ref_{id(obj)}_{datetime.now().timestamp()}"
        
        obj_id = str(id(obj))
        
        with self._lock:
            # 检查是否已存在引用
            if ref_id in self._references:
                # 增加引用计数
                self._references[ref_id].count += 1
                self._references[ref_id].last_accessed = datetime.now()
                self.logger.debug(f"增加引用计数: {ref_id}, 新计数: {self._references[ref_id].count}")
                return ref_id
            
            # 创建新引用
            ref_info = ReferenceInfo(
                ref_id=ref_id,
                obj_id=obj_id,
                count=1,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                cleanup_callback=cleanup_callback
            )
            
            self._references[ref_id] = ref_info
            
            # 更新对象映射
            if obj_id not in self._object_refs:
                self._object_refs[obj_id] = set()
            self._object_refs[obj_id].add(ref_id)
            
            # 创建弱引用
            def cleanup_weak_ref(weak_ref):
                self._cleanup_weak_reference(ref_id, obj_id)
            
            self._weak_refs[ref_id] = weakref.ref(obj, cleanup_weak_ref)
            
            # 注册清理回调
            if cleanup_callback:
                self._cleanup_callbacks[ref_id] = cleanup_callback
            
            self.logger.debug(f"添加引用: {ref_id} -> {obj_id}")
            return ref_id
    
    def remove_reference(self, ref_id: str) -> bool:
        """移除引用
        
        Args:
            ref_id: 引用ID
            
        Returns:
            bool: 是否成功移除
        """
        with self._lock:
            if ref_id not in self._references:
                self.logger.warning(f"引用不存在: {ref_id}")
                return False
            
            ref_info = self._references[ref_id]
            ref_info.count -= 1
            ref_info.last_accessed = datetime.now()
            
            self.logger.debug(f"减少引用计数: {ref_id}, 新计数: {ref_info.count}")
            
            # 如果引用计数为0，清理引用
            if ref_info.count <= 0:
                self._cleanup_reference(ref_id)
                return True
            
            return False
    
    def get_reference_count(self, ref_id: str) -> int:
        """获取引用计数
        
        Args:
            ref_id: 引用ID
            
        Returns:
            int: 引用计数，如果引用不存在则返回0
        """
        with self._lock:
            if ref_id in self._references:
                return self._references[ref_id].count
            return 0
    
    def get_object_reference_count(self, obj: Any) -> int:
        """获取对象的总引用计数
        
        Args:
            obj: 对象
            
        Returns:
            int: 总引用计数
        """
        obj_id = str(id(obj))
        
        with self._lock:
            if obj_id not in self._object_refs:
                return 0
            
            total_count = 0
            for ref_id in self._object_refs[obj_id]:
                if ref_id in self._references:
                    total_count += self._references[ref_id].count
            
            return total_count
    
    def is_referenced(self, ref_id: str) -> bool:
        """检查引用是否存在
        
        Args:
            ref_id: 引用ID
            
        Returns:
            bool: 是否存在引用
        """
        with self._lock:
            return ref_id in self._references and self._references[ref_id].count > 0
    
    def get_reference_info(self, ref_id: str) -> Optional[ReferenceInfo]:
        """获取引用信息
        
        Args:
            ref_id: 引用ID
            
        Returns:
            Optional[ReferenceInfo]: 引用信息
        """
        with self._lock:
            return self._references.get(ref_id)
    
    def get_all_references(self) -> Dict[str, ReferenceInfo]:
        """获取所有引用信息
        
        Returns:
            Dict[str, ReferenceInfo]: 所有引用信息
        """
        with self._lock:
            return self._references.copy()
    
    def cleanup_expired(self, max_age_hours: float = 24.0):
        """清理过期的引用
        
        Args:
            max_age_hours: 最大存活时间（小时）
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        expired_refs = []
        
        with self._lock:
            for ref_id, ref_info in self._references.items():
                if ref_info.last_accessed < cutoff_time:
                    expired_refs.append(ref_id)
        
        for ref_id in expired_refs:
            self._cleanup_reference(ref_id)
            self.logger.info(f"清理过期引用: {ref_id}")
    
    def _cleanup_reference(self, ref_id: str):
        """清理引用
        
        Args:
            ref_id: 引用ID
        """
        if ref_id not in self._references:
            return
        
        ref_info = self._references[ref_id]
        obj_id = ref_info.obj_id
        
        # 执行清理回调
        if ref_id in self._cleanup_callbacks:
            try:
                self._cleanup_callbacks[ref_id]()
            except Exception as e:
                self.logger.error(f"执行清理回调失败 {ref_id}: {e}")
            del self._cleanup_callbacks[ref_id]
        
        # 执行引用信息中的清理回调
        if ref_info.cleanup_callback:
            try:
                ref_info.cleanup_callback()
            except Exception as e:
                self.logger.error(f"执行引用清理回调失败 {ref_id}: {e}")
        
        # 清理引用信息
        del self._references[ref_id]
        
        # 清理对象映射
        if obj_id in self._object_refs:
            self._object_refs[obj_id].discard(ref_id)
            if not self._object_refs[obj_id]:
                del self._object_refs[obj_id]
        
        # 清理弱引用
        if ref_id in self._weak_refs:
            del self._weak_refs[ref_id]
        
        self.logger.debug(f"清理引用完成: {ref_id}")
    
    def _cleanup_weak_reference(self, ref_id: str, obj_id: str):
        """清理弱引用
        
        Args:
            ref_id: 引用ID
            obj_id: 对象ID
        """
        with self._lock:
            self.logger.debug(f"对象被垃圾回收，清理弱引用: {ref_id} -> {obj_id}")
            self._cleanup_reference(ref_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        with self._lock:
            total_references = len(self._references)
            total_objects = len(self._object_refs)
            total_count = sum(ref.count for ref in self._references.values())
            
            # 按对象统计
            object_stats = {}
            for obj_id, ref_ids in self._object_refs.items():
                obj_count = sum(
                    self._references[ref_id].count 
                    for ref_id in ref_ids 
                    if ref_id in self._references
                )
                object_stats[obj_id] = {
                    'reference_count': obj_count,
                    'reference_ids': list(ref_ids)
                }
            
            # 最近活跃的引用
            recent_refs = []
            recent_threshold = datetime.now() - timedelta(minutes=5)
            for ref_id, ref_info in self._references.items():
                if ref_info.last_accessed >= recent_threshold:
                    recent_refs.append(ref_id)
            
            return {
                'total_references': total_references,
                'total_objects': total_objects,
                'total_reference_count': total_count,
                'recent_active_references': len(recent_refs),
                'object_statistics': object_stats,
                'cleanup_callbacks': len(self._cleanup_callbacks)
            }
    
    def clear(self):
        """清理所有引用"""
        with self._lock:
            # 执行所有清理回调
            for ref_id, callback in self._cleanup_callbacks.items():
                try:
                    callback()
                except Exception as e:
                    self.logger.error(f"执行清理回调失败 {ref_id}: {e}")
            
            # 清理所有数据
            self._references.clear()
            self._object_refs.clear()
            self._weak_refs.clear()
            self._cleanup_callbacks.clear()
            
            self.logger.info("清理所有引用完成")
    
    def __len__(self) -> int:
        """返回引用数量"""
        return len(self._references)
    
    def __contains__(self, ref_id: str) -> bool:
        """检查引用是否存在"""
        return ref_id in self._references
    
    def __str__(self) -> str:
        """字符串表示"""
        with self._lock:
            return f"ReferenceCounter({len(self._references)} refs, {len(self._object_refs)} objects)"
