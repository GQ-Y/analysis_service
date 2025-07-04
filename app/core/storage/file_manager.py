#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: file_manager.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 文件管理器

提供文件上传、下载、删除等文件操作功能。

本文件是分析服务项目的一部分。
"""

import os
import uuid
import hashlib
import mimetypes
from pathlib import Path
from typing import Optional, Dict, Any, BinaryIO, Union
from datetime import datetime

from .storage_manager import get_storage_manager


class FileManager:
    """文件管理器"""
    
    def __init__(self):
        """初始化文件管理器"""
        self.storage_manager = get_storage_manager()
        
    def save_upload_file(
        self, 
        file_data: Union[bytes, BinaryIO], 
        filename: str,
        subfolder: Optional[str] = None
    ) -> Dict[str, Any]:
        """保存上传文件
        
        Args:
            file_data: 文件数据
            filename: 原始文件名
            subfolder: 子文件夹
            
        Returns:
            文件信息字典
        """
        # 生成唯一文件名
        file_ext = Path(filename).suffix
        unique_filename = f"{uuid.uuid4().hex}{file_ext}"
        
        # 确定保存路径
        upload_path = self.storage_manager.get_storage_path("uploads")
        if subfolder:
            upload_path = upload_path / subfolder

        # 确保目录存在
        upload_path.mkdir(parents=True, exist_ok=True)

        file_path = upload_path / unique_filename
        
        # 保存文件
        if isinstance(file_data, bytes):
            file_path.write_bytes(file_data)
            file_size = len(file_data)
            file_hash = hashlib.md5(file_data).hexdigest()
        else:
            # BinaryIO对象
            content = file_data.read()
            file_path.write_bytes(content)
            file_size = len(content)
            file_hash = hashlib.md5(content).hexdigest()
        
        # 获取MIME类型
        mime_type, _ = mimetypes.guess_type(filename)
        
        return {
            "original_filename": filename,
            "saved_filename": unique_filename,
            "file_path": str(file_path),
            "relative_path": str(file_path.relative_to(self.storage_manager.base_path)),
            "file_size": file_size,
            "file_hash": file_hash,
            "mime_type": mime_type,
            "upload_time": datetime.now().isoformat(),
            "subfolder": subfolder
        }
    
    def save_result_file(
        self, 
        file_data: bytes, 
        filename: str,
        task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """保存分析结果文件
        
        Args:
            file_data: 文件数据
            filename: 文件名
            task_id: 任务ID
            
        Returns:
            文件信息字典
        """
        # 确定保存路径
        results_path = self.storage_manager.get_storage_path("results")
        if task_id:
            results_path = results_path / task_id
            results_path.mkdir(parents=True, exist_ok=True)
        
        file_path = results_path / filename
        
        # 保存文件
        file_path.write_bytes(file_data)
        file_size = len(file_data)
        file_hash = hashlib.md5(file_data).hexdigest()
        
        return {
            "filename": filename,
            "file_path": str(file_path),
            "relative_path": str(file_path.relative_to(self.storage_manager.base_path)),
            "file_size": file_size,
            "file_hash": file_hash,
            "save_time": datetime.now().isoformat(),
            "task_id": task_id
        }
    
    def save_temp_file(
        self, 
        file_data: bytes, 
        filename: str,
        ttl_hours: int = 24
    ) -> Dict[str, Any]:
        """保存临时文件
        
        Args:
            file_data: 文件数据
            filename: 文件名
            ttl_hours: 生存时间（小时）
            
        Returns:
            文件信息字典
        """
        # 生成唯一文件名
        file_ext = Path(filename).suffix
        unique_filename = f"{uuid.uuid4().hex}{file_ext}"
        
        # 确定保存路径
        temp_path = self.storage_manager.get_storage_path("temp")
        file_path = temp_path / unique_filename
        
        # 保存文件
        file_path.write_bytes(file_data)
        file_size = len(file_data)
        
        return {
            "original_filename": filename,
            "temp_filename": unique_filename,
            "file_path": str(file_path),
            "relative_path": str(file_path.relative_to(self.storage_manager.base_path)),
            "file_size": file_size,
            "save_time": datetime.now().isoformat(),
            "ttl_hours": ttl_hours
        }
    
    def get_file(self, file_path: str) -> Optional[bytes]:
        """获取文件内容
        
        Args:
            file_path: 文件路径（相对或绝对）
            
        Returns:
            文件内容，如果文件不存在返回None
        """
        path = Path(file_path)
        if not path.is_absolute():
            path = self.storage_manager.base_path / file_path
        
        if path.exists() and path.is_file():
            return path.read_bytes()
        return None
    
    def delete_file(self, file_path: str) -> bool:
        """删除文件
        
        Args:
            file_path: 文件路径（相对或绝对）
            
        Returns:
            是否删除成功
        """
        path = Path(file_path)
        if not path.is_absolute():
            path = self.storage_manager.base_path / file_path
        
        try:
            if path.exists() and path.is_file():
                path.unlink()
                return True
        except OSError:
            pass
        return False
    
    def get_file_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """获取文件信息
        
        Args:
            file_path: 文件路径（相对或绝对）
            
        Returns:
            文件信息字典，如果文件不存在返回None
        """
        path = Path(file_path)
        if not path.is_absolute():
            path = self.storage_manager.base_path / file_path
        
        if not (path.exists() and path.is_file()):
            return None
        
        stat = path.stat()
        mime_type, _ = mimetypes.guess_type(str(path))
        
        return {
            "filename": path.name,
            "file_path": str(path),
            "relative_path": str(path.relative_to(self.storage_manager.base_path)),
            "file_size": stat.st_size,
            "mime_type": mime_type,
            "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "accessed_time": datetime.fromtimestamp(stat.st_atime).isoformat()
        }
    
    def list_files(
        self, 
        storage_type: str, 
        subfolder: Optional[str] = None,
        pattern: str = "*"
    ) -> list:
        """列出文件
        
        Args:
            storage_type: 存储类型
            subfolder: 子文件夹
            pattern: 文件模式
            
        Returns:
            文件信息列表
        """
        try:
            base_path = self.storage_manager.get_storage_path(storage_type)
            if subfolder:
                base_path = base_path / subfolder
            
            if not base_path.exists():
                return []
            
            files = []
            for file_path in base_path.glob(pattern):
                if file_path.is_file():
                    info = self.get_file_info(str(file_path))
                    if info:
                        files.append(info)
            
            return files
        except Exception:
            return []
