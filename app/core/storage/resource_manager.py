#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: resource_manager.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 资源管理器

管理静态资源、模板、本地化文件等资源。

本文件是分析服务项目的一部分。
"""

import json
import mimetypes
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from .storage_manager import get_storage_manager


class ResourceManager:
    """资源管理器"""
    
    def __init__(self):
        """初始化资源管理器"""
        self.storage_manager = get_storage_manager()
        self.base_path = self.storage_manager.base_path
        
    def get_static_file(self, file_path: str) -> Optional[bytes]:
        """获取静态文件内容
        
        Args:
            file_path: 静态文件路径（相对于static目录）
            
        Returns:
            文件内容，如果文件不存在返回None
        """
        static_path = self.storage_manager.get_resource_path("static")
        full_path = static_path / file_path
        
        if full_path.exists() and full_path.is_file():
            # 安全检查：确保文件在static目录内
            try:
                full_path.resolve().relative_to(static_path.resolve())
                return full_path.read_bytes()
            except ValueError:
                # 路径不在static目录内
                return None
        return None
    
    def get_template(self, template_name: str) -> Optional[str]:
        """获取模板内容
        
        Args:
            template_name: 模板名称
            
        Returns:
            模板内容，如果模板不存在返回None
        """
        templates_path = self.storage_manager.get_resource_path("templates")
        template_path = templates_path / template_name
        
        if template_path.exists() and template_path.is_file():
            # 安全检查：确保文件在templates目录内
            try:
                template_path.resolve().relative_to(templates_path.resolve())
                return template_path.read_text(encoding='utf-8')
            except ValueError:
                # 路径不在templates目录内
                return None
        return None
    
    def get_locale_data(self, locale: str) -> Optional[Dict[str, Any]]:
        """获取本地化数据
        
        Args:
            locale: 语言代码（如：zh-CN, en-US）
            
        Returns:
            本地化数据字典，如果不存在返回None
        """
        locales_path = self.storage_manager.get_resource_path("locales")
        locale_file = locales_path / f"{locale}.json"
        
        if locale_file.exists() and locale_file.is_file():
            try:
                content = locale_file.read_text(encoding='utf-8')
                return json.loads(content)
            except (json.JSONDecodeError, UnicodeDecodeError):
                return None
        return None
    
    def save_static_file(self, file_path: str, content: bytes) -> bool:
        """保存静态文件
        
        Args:
            file_path: 文件路径（相对于static目录）
            content: 文件内容
            
        Returns:
            是否保存成功
        """
        try:
            static_path = self.storage_manager.get_resource_path("static")
            full_path = static_path / file_path
            
            # 确保目录存在
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 安全检查：确保文件在static目录内
            full_path.resolve().relative_to(static_path.resolve())
            
            full_path.write_bytes(content)
            return True
        except (ValueError, OSError):
            return False
    
    def save_template(self, template_name: str, content: str) -> bool:
        """保存模板文件
        
        Args:
            template_name: 模板名称
            content: 模板内容
            
        Returns:
            是否保存成功
        """
        try:
            templates_path = self.storage_manager.get_resource_path("templates")
            template_path = templates_path / template_name
            
            # 确保目录存在
            template_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 安全检查：确保文件在templates目录内
            template_path.resolve().relative_to(templates_path.resolve())
            
            template_path.write_text(content, encoding='utf-8')
            return True
        except (ValueError, OSError):
            return False
    
    def save_locale_data(self, locale: str, data: Dict[str, Any]) -> bool:
        """保存本地化数据
        
        Args:
            locale: 语言代码
            data: 本地化数据
            
        Returns:
            是否保存成功
        """
        try:
            locales_path = self.storage_manager.get_resource_path("locales")
            locale_file = locales_path / f"{locale}.json"
            
            # 确保目录存在
            locale_file.parent.mkdir(parents=True, exist_ok=True)
            
            content = json.dumps(data, ensure_ascii=False, indent=2)
            locale_file.write_text(content, encoding='utf-8')
            return True
        except (OSError, TypeError):
            return False
    
    def get_public_file(self, file_path: str, public_type: str = "assets") -> Optional[bytes]:
        """获取公共文件内容
        
        Args:
            file_path: 文件路径（相对于public子目录）
            public_type: 公共文件类型 (assets, images, videos)
            
        Returns:
            文件内容，如果文件不存在返回None
        """
        try:
            public_path = self.storage_manager.get_public_path(public_type)
            full_path = public_path / file_path
            
            if full_path.exists() and full_path.is_file():
                # 安全检查：确保文件在public目录内
                full_path.resolve().relative_to(public_path.resolve())
                return full_path.read_bytes()
        except (ValueError, OSError):
            pass
        return None
    
    def save_public_file(self, file_path: str, content: bytes, public_type: str = "assets") -> bool:
        """保存公共文件
        
        Args:
            file_path: 文件路径（相对于public子目录）
            content: 文件内容
            public_type: 公共文件类型 (assets, images, videos)
            
        Returns:
            是否保存成功
        """
        try:
            public_path = self.storage_manager.get_public_path(public_type)
            full_path = public_path / file_path
            
            # 确保目录存在
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 安全检查：确保文件在public目录内
            full_path.resolve().relative_to(public_path.resolve())
            
            full_path.write_bytes(content)
            return True
        except (ValueError, OSError):
            return False
    
    def list_static_files(self, subfolder: str = "") -> List[Dict[str, Any]]:
        """列出静态文件
        
        Args:
            subfolder: 子文件夹
            
        Returns:
            文件信息列表
        """
        static_path = self.storage_manager.get_resource_path("static")
        if subfolder:
            static_path = static_path / subfolder
        
        files = []
        if static_path.exists():
            for file_path in static_path.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(
                        self.storage_manager.get_resource_path("static")
                    )
                    stat = file_path.stat()
                    mime_type, _ = mimetypes.guess_type(str(file_path))
                    
                    files.append({
                        "filename": file_path.name,
                        "relative_path": str(relative_path),
                        "size_bytes": stat.st_size,
                        "mime_type": mime_type,
                        "modified_time": stat.st_mtime
                    })
        
        return files
    
    def list_templates(self) -> List[str]:
        """列出所有模板
        
        Returns:
            模板名称列表
        """
        templates_path = self.storage_manager.get_resource_path("templates")
        templates = []
        
        if templates_path.exists():
            for template_path in templates_path.rglob("*.html"):
                relative_path = template_path.relative_to(templates_path)
                templates.append(str(relative_path))
        
        return templates
    
    def list_locales(self) -> List[str]:
        """列出所有语言
        
        Returns:
            语言代码列表
        """
        locales_path = self.storage_manager.get_resource_path("locales")
        locales = []
        
        if locales_path.exists():
            for locale_file in locales_path.glob("*.json"):
                locales.append(locale_file.stem)
        
        return locales
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """获取资源统计信息
        
        Returns:
            资源统计信息
        """
        stats = {
            "static_files": {"count": 0, "size_bytes": 0},
            "templates": {"count": 0, "size_bytes": 0},
            "locales": {"count": 0, "size_bytes": 0},
            "public_files": {"count": 0, "size_bytes": 0}
        }
        
        # 统计静态文件
        static_path = self.storage_manager.get_resource_path("static")
        if static_path.exists():
            for file_path in static_path.rglob("*"):
                if file_path.is_file():
                    stats["static_files"]["count"] += 1
                    stats["static_files"]["size_bytes"] += file_path.stat().st_size
        
        # 统计模板文件
        templates_path = self.storage_manager.get_resource_path("templates")
        if templates_path.exists():
            for file_path in templates_path.rglob("*"):
                if file_path.is_file():
                    stats["templates"]["count"] += 1
                    stats["templates"]["size_bytes"] += file_path.stat().st_size
        
        # 统计本地化文件
        locales_path = self.storage_manager.get_resource_path("locales")
        if locales_path.exists():
            for file_path in locales_path.rglob("*"):
                if file_path.is_file():
                    stats["locales"]["count"] += 1
                    stats["locales"]["size_bytes"] += file_path.stat().st_size
        
        # 统计公共文件
        public_base = self.base_path / "public"
        if public_base.exists():
            for file_path in public_base.rglob("*"):
                if file_path.is_file():
                    stats["public_files"]["count"] += 1
                    stats["public_files"]["size_bytes"] += file_path.stat().st_size
        
        # 转换为MB
        for category in stats.values():
            category["size_mb"] = round(category["size_bytes"] / 1024 / 1024, 2)
        
        return stats
