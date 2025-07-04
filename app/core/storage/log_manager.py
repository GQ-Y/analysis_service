#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: log_manager.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 日志管理器

提供日志文件管理、轮转、清理等功能。

本文件是分析服务项目的一部分。
"""

import gzip
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from .storage_manager import get_storage_manager


class LogManager:
    """日志管理器"""
    
    def __init__(self):
        """初始化日志管理器"""
        self.storage_manager = get_storage_manager()
        self.logs_path = self.storage_manager.get_storage_path("logs")
        
    def get_log_file_path(self, log_type: str, date: Optional[datetime] = None) -> Path:
        """获取日志文件路径
        
        Args:
            log_type: 日志类型 (app, error, access, analysis等)
            date: 日期，None表示当前日期
            
        Returns:
            日志文件路径
        """
        if date is None:
            date = datetime.now()
        
        date_str = date.strftime("%Y-%m-%d")
        return self.logs_path / f"{log_type}_{date_str}.log"
    
    def rotate_log_file(self, log_type: str, max_size_mb: int = 100) -> bool:
        """轮转日志文件
        
        Args:
            log_type: 日志类型
            max_size_mb: 最大文件大小（MB）
            
        Returns:
            是否进行了轮转
        """
        current_log = self.get_log_file_path(log_type)
        
        if not current_log.exists():
            return False
        
        # 检查文件大小
        file_size_mb = current_log.stat().st_size / 1024 / 1024
        if file_size_mb < max_size_mb:
            return False
        
        # 生成轮转文件名
        timestamp = datetime.now().strftime("%H%M%S")
        rotated_name = f"{current_log.stem}_{timestamp}.log"
        rotated_path = current_log.parent / rotated_name
        
        try:
            # 移动当前日志文件
            shutil.move(str(current_log), str(rotated_path))
            
            # 压缩轮转的日志文件
            compressed_path = rotated_path.with_suffix(".log.gz")
            with open(rotated_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # 删除未压缩的文件
            rotated_path.unlink()
            
            return True
        except Exception:
            return False
    
    def clean_old_logs(self, days_to_keep: int = 30):
        """清理旧日志文件
        
        Args:
            days_to_keep: 保留天数
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        for log_file in self.logs_path.glob("*.log*"):
            try:
                # 从文件名提取日期
                file_date = None
                parts = log_file.stem.split("_")
                
                for part in parts:
                    try:
                        file_date = datetime.strptime(part, "%Y-%m-%d")
                        break
                    except ValueError:
                        continue
                
                if file_date and file_date < cutoff_date:
                    log_file.unlink()
            except Exception:
                pass
    
    def get_log_files(self, log_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取日志文件列表
        
        Args:
            log_type: 日志类型，None表示所有类型
            
        Returns:
            日志文件信息列表
        """
        files = []
        pattern = f"{log_type}_*.log*" if log_type else "*.log*"
        
        for log_file in self.logs_path.glob(pattern):
            if log_file.is_file():
                stat = log_file.stat()
                files.append({
                    "filename": log_file.name,
                    "file_path": str(log_file),
                    "size_bytes": stat.st_size,
                    "size_mb": round(stat.st_size / 1024 / 1024, 2),
                    "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "is_compressed": log_file.suffix == ".gz"
                })
        
        # 按修改时间排序
        files.sort(key=lambda x: x["modified_time"], reverse=True)
        return files
    
    def read_log_file(
        self, 
        log_file_path: str, 
        lines: Optional[int] = None,
        tail: bool = True
    ) -> List[str]:
        """读取日志文件内容
        
        Args:
            log_file_path: 日志文件路径
            lines: 读取行数，None表示读取全部
            tail: 是否从末尾开始读取
            
        Returns:
            日志行列表
        """
        file_path = Path(log_file_path)
        if not file_path.exists():
            return []
        
        try:
            # 判断是否为压缩文件
            if file_path.suffix == ".gz":
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    content_lines = f.readlines()
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content_lines = f.readlines()
            
            # 去除换行符
            content_lines = [line.rstrip('\n\r') for line in content_lines]
            
            if lines is None:
                return content_lines
            
            if tail:
                return content_lines[-lines:] if len(content_lines) > lines else content_lines
            else:
                return content_lines[:lines] if len(content_lines) > lines else content_lines
        except Exception:
            return []
    
    def search_logs(
        self, 
        keyword: str, 
        log_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_results: int = 1000
    ) -> List[Dict[str, Any]]:
        """搜索日志内容
        
        Args:
            keyword: 搜索关键词
            log_type: 日志类型
            start_date: 开始日期
            end_date: 结束日期
            max_results: 最大结果数
            
        Returns:
            搜索结果列表
        """
        results = []
        log_files = self.get_log_files(log_type)
        
        for log_file_info in log_files:
            if len(results) >= max_results:
                break
            
            # 检查日期范围
            file_date = datetime.fromisoformat(log_file_info["modified_time"])
            if start_date and file_date < start_date:
                continue
            if end_date and file_date > end_date:
                continue
            
            # 搜索文件内容
            lines = self.read_log_file(log_file_info["file_path"])
            for line_num, line in enumerate(lines, 1):
                if keyword.lower() in line.lower():
                    results.append({
                        "filename": log_file_info["filename"],
                        "line_number": line_num,
                        "content": line,
                        "timestamp": file_date.isoformat()
                    })
                    
                    if len(results) >= max_results:
                        break
        
        return results
    
    def get_log_stats(self) -> Dict[str, Any]:
        """获取日志统计信息
        
        Returns:
            日志统计信息
        """
        stats = {
            "total_files": 0,
            "total_size_bytes": 0,
            "total_size_mb": 0,
            "log_types": {},
            "compressed_files": 0,
            "uncompressed_files": 0
        }
        
        for log_file in self.logs_path.glob("*.log*"):
            if log_file.is_file():
                stats["total_files"] += 1
                file_size = log_file.stat().st_size
                stats["total_size_bytes"] += file_size
                
                if log_file.suffix == ".gz":
                    stats["compressed_files"] += 1
                else:
                    stats["uncompressed_files"] += 1
                
                # 提取日志类型
                log_type = log_file.name.split("_")[0]
                if log_type not in stats["log_types"]:
                    stats["log_types"][log_type] = {
                        "file_count": 0,
                        "size_bytes": 0
                    }
                
                stats["log_types"][log_type]["file_count"] += 1
                stats["log_types"][log_type]["size_bytes"] += file_size
        
        stats["total_size_mb"] = round(stats["total_size_bytes"] / 1024 / 1024, 2)
        
        # 计算每种日志类型的大小（MB）
        for log_type_stats in stats["log_types"].values():
            log_type_stats["size_mb"] = round(log_type_stats["size_bytes"] / 1024 / 1024, 2)
        
        return stats
