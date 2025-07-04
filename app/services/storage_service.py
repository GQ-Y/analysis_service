#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: storage_service.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 存储服务

提供统一的存储服务接口，整合文件管理、缓存管理、日志管理等功能。

本文件是分析服务项目的一部分。
"""

from typing import Dict, Any, Optional, List, Union, BinaryIO
from datetime import datetime

from app.core.storage import (
    StorageManager, 
    FileManager, 
    CacheManager, 
    LogManager, 
    ResourceManager,
    get_storage_manager
)


class StorageService:
    """存储服务"""
    
    def __init__(self):
        """初始化存储服务"""
        self.storage_manager = get_storage_manager()
        self.file_manager = FileManager()
        self.cache_manager = CacheManager()
        self.log_manager = LogManager()
        self.resource_manager = ResourceManager()
    
    # 文件管理相关方法
    def upload_file(
        self, 
        file_data: Union[bytes, BinaryIO], 
        filename: str,
        subfolder: Optional[str] = None
    ) -> Dict[str, Any]:
        """上传文件
        
        Args:
            file_data: 文件数据
            filename: 文件名
            subfolder: 子文件夹
            
        Returns:
            文件信息
        """
        return self.file_manager.save_upload_file(file_data, filename, subfolder)
    
    def save_analysis_result(
        self, 
        result_data: bytes, 
        filename: str,
        task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """保存分析结果
        
        Args:
            result_data: 结果数据
            filename: 文件名
            task_id: 任务ID
            
        Returns:
            文件信息
        """
        return self.file_manager.save_result_file(result_data, filename, task_id)
    
    def get_file(self, file_path: str) -> Optional[bytes]:
        """获取文件内容
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件内容
        """
        return self.file_manager.get_file(file_path)
    
    def delete_file(self, file_path: str) -> bool:
        """删除文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否删除成功
        """
        return self.file_manager.delete_file(file_path)
    
    def get_file_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """获取文件信息
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件信息
        """
        return self.file_manager.get_file_info(file_path)
    
    def list_files(
        self, 
        storage_type: str, 
        subfolder: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """列出文件
        
        Args:
            storage_type: 存储类型
            subfolder: 子文件夹
            
        Returns:
            文件列表
        """
        return self.file_manager.list_files(storage_type, subfolder)
    
    # 缓存管理相关方法
    def set_cache(
        self, 
        cache_type: str, 
        key: str, 
        value: Any, 
        ttl_seconds: Optional[int] = None,
        use_memory: bool = False
    ) -> bool:
        """设置缓存
        
        Args:
            cache_type: 缓存类型
            key: 缓存键
            value: 缓存值
            ttl_seconds: 生存时间
            use_memory: 是否使用内存缓存
            
        Returns:
            是否设置成功
        """
        if use_memory:
            return self.cache_manager.set_memory_cache(key, value, ttl_seconds)
        else:
            return self.cache_manager.set_file_cache(cache_type, key, value, ttl_seconds)
    
    def get_cache(
        self, 
        cache_type: str, 
        key: str,
        use_memory: bool = False
    ) -> Optional[Any]:
        """获取缓存
        
        Args:
            cache_type: 缓存类型
            key: 缓存键
            use_memory: 是否使用内存缓存
            
        Returns:
            缓存值
        """
        if use_memory:
            return self.cache_manager.get_memory_cache(key)
        else:
            return self.cache_manager.get_file_cache(cache_type, key)
    
    def delete_cache(
        self, 
        cache_type: str, 
        key: str,
        use_memory: bool = False
    ) -> bool:
        """删除缓存
        
        Args:
            cache_type: 缓存类型
            key: 缓存键
            use_memory: 是否使用内存缓存
            
        Returns:
            是否删除成功
        """
        if use_memory:
            return self.cache_manager.delete_memory_cache(key)
        else:
            return self.cache_manager.delete_file_cache(cache_type, key)
    
    def clear_cache(self, cache_type: Optional[str] = None):
        """清理缓存
        
        Args:
            cache_type: 缓存类型，None表示清理所有
        """
        self.cache_manager.clear_cache(cache_type)
    
    # 资源管理相关方法
    def get_static_file(self, file_path: str) -> Optional[bytes]:
        """获取静态文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件内容
        """
        return self.resource_manager.get_static_file(file_path)
    
    def get_template(self, template_name: str) -> Optional[str]:
        """获取模板
        
        Args:
            template_name: 模板名称
            
        Returns:
            模板内容
        """
        return self.resource_manager.get_template(template_name)
    
    def get_locale_data(self, locale: str) -> Optional[Dict[str, Any]]:
        """获取本地化数据
        
        Args:
            locale: 语言代码
            
        Returns:
            本地化数据
        """
        return self.resource_manager.get_locale_data(locale)
    
    # 日志管理相关方法
    def get_log_files(self, log_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取日志文件列表
        
        Args:
            log_type: 日志类型
            
        Returns:
            日志文件列表
        """
        return self.log_manager.get_log_files(log_type)
    
    def read_log_file(
        self, 
        log_file_path: str, 
        lines: Optional[int] = None,
        tail: bool = True
    ) -> List[str]:
        """读取日志文件
        
        Args:
            log_file_path: 日志文件路径
            lines: 读取行数
            tail: 是否从末尾读取
            
        Returns:
            日志行列表
        """
        return self.log_manager.read_log_file(log_file_path, lines, tail)
    
    def search_logs(
        self, 
        keyword: str, 
        log_type: Optional[str] = None,
        max_results: int = 1000
    ) -> List[Dict[str, Any]]:
        """搜索日志
        
        Args:
            keyword: 搜索关键词
            log_type: 日志类型
            max_results: 最大结果数
            
        Returns:
            搜索结果
        """
        return self.log_manager.search_logs(keyword, log_type, max_results=max_results)
    
    # 存储统计和管理
    def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计信息
        
        Returns:
            存储统计信息
        """
        return {
            "storage_info": self.storage_manager.get_storage_info(),
            "cache_stats": self.cache_manager.get_cache_stats(),
            "log_stats": self.log_manager.get_log_stats(),
            "resource_stats": self.resource_manager.get_resource_stats()
        }
    
    def cleanup_storage(self):
        """清理存储"""
        # 清理临时文件
        self.storage_manager.clean_temp_files()
        
        # 清理过期缓存
        self.cache_manager.clean_expired_cache()
        
        # 清理旧日志
        self.log_manager.clean_old_logs()
    
    def backup_storage(self, backup_name: Optional[str] = None) -> str:
        """备份存储
        
        Args:
            backup_name: 备份名称
            
        Returns:
            备份文件路径
        """
        return self.storage_manager.backup_storage(backup_name)


# 全局存储服务实例
_storage_service: Optional[StorageService] = None


def get_storage_service() -> StorageService:
    """获取存储服务实例
    
    Returns:
        存储服务实例
    """
    global _storage_service
    if _storage_service is None:
        _storage_service = StorageService()
    return _storage_service
