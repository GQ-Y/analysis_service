#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: storage.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 存储配置

定义存储相关的配置参数。

本文件是分析服务项目的一部分。
"""

from pathlib import Path
from typing import Dict, Any

# 存储根目录配置
STORAGE_ROOT = "storage"
RESOURCES_ROOT = "resources"
PUBLIC_ROOT = "public"

# 存储目录配置
STORAGE_DIRECTORIES = {
    "cache": "cache",
    "logs": "logs", 
    "results": "results",
    "temp": "temp",
    "uploads": "uploads",
    "backups": "backups",
    "exports": "exports"
}

# 资源目录配置
RESOURCE_DIRECTORIES = {
    "static": "static",
    "templates": "templates",
    "locales": "locales"
}

# 公共文件目录配置
PUBLIC_DIRECTORIES = {
    "assets": "assets",
    "images": "images", 
    "videos": "videos"
}

# 文件上传配置
UPLOAD_CONFIG = {
    "max_file_size": 100 * 1024 * 1024,  # 100MB
    "allowed_extensions": {
        "images": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"],
        "videos": [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"],
        "documents": [".pdf", ".doc", ".docx", ".txt", ".csv", ".xlsx"],
        "archives": [".zip", ".rar", ".7z", ".tar", ".gz"]
    },
    "upload_path": "uploads",
    "temp_path": "temp"
}

# 缓存配置
CACHE_CONFIG = {
    "default_ttl": 3600,  # 1小时
    "max_memory_cache_size": 1000,  # 最大内存缓存条目数
    "file_cache_types": [
        "analysis_results",
        "model_cache", 
        "stream_cache",
        "api_cache"
    ]
}

# 日志配置
LOG_CONFIG = {
    "max_file_size_mb": 100,
    "max_files_per_type": 10,
    "retention_days": 30,
    "log_types": [
        "app",
        "error", 
        "access",
        "analysis",
        "stream",
        "task"
    ],
    "rotation_enabled": True,
    "compression_enabled": True
}

# 清理配置
CLEANUP_CONFIG = {
    "temp_file_ttl_hours": 24,
    "cache_cleanup_interval_hours": 6,
    "log_cleanup_interval_days": 7,
    "auto_cleanup_enabled": True
}

# 备份配置
BACKUP_CONFIG = {
    "auto_backup_enabled": False,
    "backup_interval_days": 7,
    "max_backup_files": 5,
    "backup_types": ["results", "uploads"],
    "compression_enabled": True
}

# 静态文件服务配置
STATIC_CONFIG = {
    "serve_static": True,
    "static_url_prefix": "/static",
    "public_url_prefix": "/public",
    "cache_max_age": 86400,  # 1天
    "etag_enabled": True
}

# 安全配置
SECURITY_CONFIG = {
    "path_traversal_protection": True,
    "file_type_validation": True,
    "virus_scan_enabled": False,
    "max_path_length": 255,
    "forbidden_extensions": [".exe", ".bat", ".cmd", ".scr", ".pif"]
}


def get_storage_config() -> Dict[str, Any]:
    """获取存储配置
    
    Returns:
        存储配置字典
    """
    return {
        "storage_root": STORAGE_ROOT,
        "resources_root": RESOURCES_ROOT,
        "public_root": PUBLIC_ROOT,
        "storage_directories": STORAGE_DIRECTORIES,
        "resource_directories": RESOURCE_DIRECTORIES,
        "public_directories": PUBLIC_DIRECTORIES,
        "upload": UPLOAD_CONFIG,
        "cache": CACHE_CONFIG,
        "log": LOG_CONFIG,
        "cleanup": CLEANUP_CONFIG,
        "backup": BACKUP_CONFIG,
        "static": STATIC_CONFIG,
        "security": SECURITY_CONFIG
    }


def validate_storage_config() -> bool:
    """验证存储配置
    
    Returns:
        配置是否有效
    """
    try:
        config = get_storage_config()
        
        # 检查必要的配置项
        required_keys = [
            "storage_root", "resources_root", "public_root",
            "storage_directories", "resource_directories", "public_directories"
        ]
        
        for key in required_keys:
            if key not in config:
                return False
        
        # 检查目录配置
        if not isinstance(config["storage_directories"], dict):
            return False
        
        if not isinstance(config["resource_directories"], dict):
            return False
        
        if not isinstance(config["public_directories"], dict):
            return False
        
        return True
    except Exception:
        return False


def setup_storage_directories(base_path: Path):
    """设置存储目录
    
    Args:
        base_path: 基础路径
    """
    config = get_storage_config()
    
    # 创建存储目录
    storage_root = base_path / config["storage_root"]
    for dir_name in config["storage_directories"].values():
        (storage_root / dir_name).mkdir(parents=True, exist_ok=True)
    
    # 创建资源目录
    resources_root = base_path / config["resources_root"]
    for dir_name in config["resource_directories"].values():
        (resources_root / dir_name).mkdir(parents=True, exist_ok=True)
    
    # 创建公共文件目录
    public_root = base_path / config["public_root"]
    for dir_name in config["public_directories"].values():
        (public_root / dir_name).mkdir(parents=True, exist_ok=True)


# 导出配置
__all__ = [
    "STORAGE_ROOT",
    "RESOURCES_ROOT", 
    "PUBLIC_ROOT",
    "STORAGE_DIRECTORIES",
    "RESOURCE_DIRECTORIES",
    "PUBLIC_DIRECTORIES",
    "UPLOAD_CONFIG",
    "CACHE_CONFIG",
    "LOG_CONFIG",
    "CLEANUP_CONFIG",
    "BACKUP_CONFIG",
    "STATIC_CONFIG",
    "SECURITY_CONFIG",
    "get_storage_config",
    "validate_storage_config",
    "setup_storage_directories"
]
