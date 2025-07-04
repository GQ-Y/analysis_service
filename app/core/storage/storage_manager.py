#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: storage_manager.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 存储管理器

统一管理所有存储相关功能，包括文件存储、缓存、日志等。

本文件是分析服务项目的一部分。
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime, timedelta

from config.settings import get_settings


class StorageManager:
    """存储管理器"""
    
    def __init__(self):
        """初始化存储管理器"""
        self.settings = get_settings()
        self.base_path = Path(self.settings.BASE_DIR)
        self.storage_path = self.base_path / "storage"
        self._ensure_directories()
        
    def _ensure_directories(self):
        """确保所有必要的目录存在"""
        directories = [
            self.storage_path / "cache",
            self.storage_path / "logs", 
            self.storage_path / "results",
            self.storage_path / "temp",
            self.storage_path / "uploads",
            self.storage_path / "backups",
            self.storage_path / "exports",
            self.base_path / "resources" / "static",
            self.base_path / "resources" / "templates",
            self.base_path / "resources" / "locales",
            self.base_path / "public" / "assets",
            self.base_path / "public" / "images",
            self.base_path / "public" / "videos"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_storage_path(self, storage_type: str) -> Path:
        """获取存储路径
        
        Args:
            storage_type: 存储类型 (cache, logs, results, temp, uploads, backups, exports)
            
        Returns:
            存储路径
        """
        if storage_type in ["cache", "logs", "results", "temp", "uploads", "backups", "exports"]:
            return self.storage_path / storage_type
        else:
            raise ValueError(f"不支持的存储类型: {storage_type}")
    
    def get_resource_path(self, resource_type: str) -> Path:
        """获取资源路径
        
        Args:
            resource_type: 资源类型 (static, templates, locales)
            
        Returns:
            资源路径
        """
        if resource_type in ["static", "templates", "locales"]:
            return self.base_path / "resources" / resource_type
        else:
            raise ValueError(f"不支持的资源类型: {resource_type}")
    
    def get_public_path(self, public_type: str) -> Path:
        """获取公共文件路径
        
        Args:
            public_type: 公共文件类型 (assets, images, videos)
            
        Returns:
            公共文件路径
        """
        if public_type in ["assets", "images", "videos"]:
            return self.base_path / "public" / public_type
        else:
            raise ValueError(f"不支持的公共文件类型: {public_type}")
    
    def clean_temp_files(self, older_than_hours: int = 24):
        """清理临时文件
        
        Args:
            older_than_hours: 清理多少小时前的文件
        """
        temp_path = self.get_storage_path("temp")
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        
        for file_path in temp_path.rglob("*"):
            if file_path.is_file():
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff_time:
                    try:
                        file_path.unlink()
                    except OSError:
                        pass
    
    def clean_cache(self, cache_type: Optional[str] = None):
        """清理缓存
        
        Args:
            cache_type: 缓存类型，None表示清理所有缓存
        """
        cache_path = self.get_storage_path("cache")
        
        if cache_type:
            cache_dir = cache_path / cache_type
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                cache_dir.mkdir(exist_ok=True)
        else:
            # 清理所有缓存
            if cache_path.exists():
                shutil.rmtree(cache_path)
                cache_path.mkdir(exist_ok=True)
    
    def get_storage_info(self) -> Dict[str, Any]:
        """获取存储信息
        
        Returns:
            存储信息字典
        """
        info = {}
        
        for storage_type in ["cache", "logs", "results", "temp", "uploads", "backups", "exports"]:
            path = self.get_storage_path(storage_type)
            if path.exists():
                size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                file_count = len([f for f in path.rglob("*") if f.is_file()])
                info[storage_type] = {
                    "path": str(path),
                    "size_bytes": size,
                    "size_mb": round(size / 1024 / 1024, 2),
                    "file_count": file_count
                }
            else:
                info[storage_type] = {
                    "path": str(path),
                    "size_bytes": 0,
                    "size_mb": 0,
                    "file_count": 0
                }
        
        return info
    
    def backup_storage(self, backup_name: Optional[str] = None) -> str:
        """备份存储数据
        
        Args:
            backup_name: 备份名称，None则使用时间戳
            
        Returns:
            备份文件路径
        """
        if not backup_name:
            backup_name = f"storage_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_path = self.get_storage_path("backups") / f"{backup_name}.tar.gz"
        
        import tarfile
        with tarfile.open(backup_path, "w:gz") as tar:
            for storage_type in ["results", "uploads"]:  # 只备份重要数据
                path = self.get_storage_path(storage_type)
                if path.exists():
                    tar.add(path, arcname=storage_type)
        
        return str(backup_path)


# 全局存储管理器实例
_storage_manager: Optional[StorageManager] = None


def get_storage_manager() -> StorageManager:
    """获取存储管理器实例
    
    Returns:
        存储管理器实例
    """
    global _storage_manager
    if _storage_manager is None:
        _storage_manager = StorageManager()
    return _storage_manager
