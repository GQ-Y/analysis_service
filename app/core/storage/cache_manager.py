#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: cache_manager.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 缓存管理器

提供文件缓存、内存缓存等缓存功能。

本文件是分析服务项目的一部分。
"""

import json
import pickle
import hashlib
from pathlib import Path
from typing import Any, Optional, Dict, Union
from datetime import datetime, timedelta

from .storage_manager import get_storage_manager


class CacheManager:
    """缓存管理器"""
    
    def __init__(self):
        """初始化缓存管理器"""
        self.storage_manager = get_storage_manager()
        self.cache_path = self.storage_manager.get_storage_path("cache")
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        
    def _get_cache_file_path(self, cache_type: str, key: str) -> Path:
        """获取缓存文件路径
        
        Args:
            cache_type: 缓存类型
            key: 缓存键
            
        Returns:
            缓存文件路径
        """
        # 使用MD5哈希作为文件名，避免特殊字符问题
        key_hash = hashlib.md5(key.encode()).hexdigest()
        cache_dir = self.cache_path / cache_type
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{key_hash}.cache"
    
    def set_file_cache(
        self, 
        cache_type: str, 
        key: str, 
        value: Any, 
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """设置文件缓存
        
        Args:
            cache_type: 缓存类型
            key: 缓存键
            value: 缓存值
            ttl_seconds: 生存时间（秒）
            
        Returns:
            是否设置成功
        """
        try:
            cache_file = self._get_cache_file_path(cache_type, key)
            
            cache_data = {
                "key": key,
                "value": value,
                "created_at": datetime.now().isoformat(),
                "expires_at": None
            }
            
            if ttl_seconds:
                expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
                cache_data["expires_at"] = expires_at.isoformat()
            
            # 使用pickle序列化，支持更多数据类型
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            return True
        except Exception:
            return False
    
    def get_file_cache(self, cache_type: str, key: str) -> Optional[Any]:
        """获取文件缓存
        
        Args:
            cache_type: 缓存类型
            key: 缓存键
            
        Returns:
            缓存值，如果不存在或已过期返回None
        """
        try:
            cache_file = self._get_cache_file_path(cache_type, key)
            
            if not cache_file.exists():
                return None
            
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # 检查是否过期
            if cache_data.get("expires_at"):
                expires_at = datetime.fromisoformat(cache_data["expires_at"])
                if datetime.now() > expires_at:
                    # 删除过期缓存
                    cache_file.unlink()
                    return None
            
            return cache_data["value"]
        except Exception:
            return None
    
    def delete_file_cache(self, cache_type: str, key: str) -> bool:
        """删除文件缓存
        
        Args:
            cache_type: 缓存类型
            key: 缓存键
            
        Returns:
            是否删除成功
        """
        try:
            cache_file = self._get_cache_file_path(cache_type, key)
            if cache_file.exists():
                cache_file.unlink()
                return True
        except Exception:
            pass
        return False
    
    def set_memory_cache(
        self, 
        key: str, 
        value: Any, 
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """设置内存缓存
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl_seconds: 生存时间（秒）
            
        Returns:
            是否设置成功
        """
        try:
            cache_data = {
                "value": value,
                "created_at": datetime.now(),
                "expires_at": None
            }
            
            if ttl_seconds:
                cache_data["expires_at"] = datetime.now() + timedelta(seconds=ttl_seconds)
            
            self._memory_cache[key] = cache_data
            return True
        except Exception:
            return False
    
    def get_memory_cache(self, key: str) -> Optional[Any]:
        """获取内存缓存
        
        Args:
            key: 缓存键
            
        Returns:
            缓存值，如果不存在或已过期返回None
        """
        try:
            if key not in self._memory_cache:
                return None
            
            cache_data = self._memory_cache[key]
            
            # 检查是否过期
            if cache_data.get("expires_at"):
                if datetime.now() > cache_data["expires_at"]:
                    # 删除过期缓存
                    del self._memory_cache[key]
                    return None
            
            return cache_data["value"]
        except Exception:
            return None
    
    def delete_memory_cache(self, key: str) -> bool:
        """删除内存缓存
        
        Args:
            key: 缓存键
            
        Returns:
            是否删除成功
        """
        try:
            if key in self._memory_cache:
                del self._memory_cache[key]
                return True
        except Exception:
            pass
        return False
    
    def clear_cache(self, cache_type: Optional[str] = None):
        """清理缓存
        
        Args:
            cache_type: 缓存类型，None表示清理所有文件缓存
        """
        # 清理内存缓存
        self._memory_cache.clear()
        
        # 清理文件缓存
        if cache_type:
            cache_dir = self.cache_path / cache_type
            if cache_dir.exists():
                import shutil
                shutil.rmtree(cache_dir)
                cache_dir.mkdir(exist_ok=True)
        else:
            # 清理所有文件缓存
            if self.cache_path.exists():
                import shutil
                shutil.rmtree(self.cache_path)
                self.cache_path.mkdir(exist_ok=True)
    
    def clean_expired_cache(self):
        """清理过期缓存"""
        # 清理过期内存缓存
        expired_keys = []
        for key, cache_data in self._memory_cache.items():
            if cache_data.get("expires_at"):
                if datetime.now() > cache_data["expires_at"]:
                    expired_keys.append(key)
        
        for key in expired_keys:
            del self._memory_cache[key]
        
        # 清理过期文件缓存
        for cache_file in self.cache_path.rglob("*.cache"):
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                if cache_data.get("expires_at"):
                    expires_at = datetime.fromisoformat(cache_data["expires_at"])
                    if datetime.now() > expires_at:
                        cache_file.unlink()
            except Exception:
                # 如果文件损坏，也删除它
                try:
                    cache_file.unlink()
                except Exception:
                    pass
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息
        
        Returns:
            缓存统计信息
        """
        stats = {
            "memory_cache": {
                "total_keys": len(self._memory_cache),
                "expired_keys": 0
            },
            "file_cache": {
                "total_files": 0,
                "total_size_bytes": 0,
                "cache_types": {}
            }
        }
        
        # 统计内存缓存
        now = datetime.now()
        for cache_data in self._memory_cache.values():
            if cache_data.get("expires_at") and now > cache_data["expires_at"]:
                stats["memory_cache"]["expired_keys"] += 1
        
        # 统计文件缓存
        for cache_type_dir in self.cache_path.iterdir():
            if cache_type_dir.is_dir():
                cache_files = list(cache_type_dir.glob("*.cache"))
                total_size = sum(f.stat().st_size for f in cache_files)
                
                stats["file_cache"]["cache_types"][cache_type_dir.name] = {
                    "file_count": len(cache_files),
                    "size_bytes": total_size,
                    "size_mb": round(total_size / 1024 / 1024, 2)
                }
                
                stats["file_cache"]["total_files"] += len(cache_files)
                stats["file_cache"]["total_size_bytes"] += total_size
        
        stats["file_cache"]["total_size_mb"] = round(
            stats["file_cache"]["total_size_bytes"] / 1024 / 1024, 2
        )
        
        return stats
