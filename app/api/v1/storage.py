#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: storage.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 存储API端点

提供存储管理相关的API接口。

本文件是分析服务项目的一部分。
"""

from typing import Optional, List
from fastapi import APIRouter, HTTPException, UploadFile, File, Query, Depends
from fastapi.responses import Response

from app.services.storage_service import get_storage_service
from app.models.response_model import ResponseModel

router = APIRouter(prefix="/storage", tags=["存储管理"])


@router.get("/stats", response_model=ResponseModel)
async def get_storage_stats():
    """获取存储统计信息"""
    try:
        storage_service = get_storage_service()
        stats = storage_service.get_storage_stats()
        
        return ResponseModel(
            success=True,
            message="存储统计信息获取成功",
            data=stats
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取存储统计信息失败: {str(e)}")


@router.post("/upload", response_model=ResponseModel)
async def upload_file(
    file: UploadFile = File(...),
    subfolder: Optional[str] = Query(None, description="子文件夹")
):
    """上传文件"""
    try:
        storage_service = get_storage_service()
        
        # 读取文件内容
        file_content = await file.read()
        
        # 保存文件
        file_info = storage_service.upload_file(
            file_content, 
            file.filename,
            subfolder
        )
        
        return ResponseModel(
            success=True,
            message="文件上传成功",
            data=file_info
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")


@router.get("/files/{storage_type}", response_model=ResponseModel)
async def list_files(
    storage_type: str,
    subfolder: Optional[str] = Query(None, description="子文件夹")
):
    """列出文件"""
    try:
        storage_service = get_storage_service()
        files = storage_service.list_files(storage_type, subfolder)
        
        return ResponseModel(
            success=True,
            message="文件列表获取成功",
            data={
                "storage_type": storage_type,
                "subfolder": subfolder,
                "files": files,
                "total_count": len(files)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取文件列表失败: {str(e)}")


@router.get("/file/info", response_model=ResponseModel)
async def get_file_info(file_path: str = Query(..., description="文件路径")):
    """获取文件信息"""
    try:
        storage_service = get_storage_service()
        file_info = storage_service.get_file_info(file_path)
        
        if file_info is None:
            raise HTTPException(status_code=404, detail="文件不存在")
        
        return ResponseModel(
            success=True,
            message="文件信息获取成功",
            data=file_info
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取文件信息失败: {str(e)}")


@router.get("/file/download")
async def download_file(file_path: str = Query(..., description="文件路径")):
    """下载文件"""
    try:
        storage_service = get_storage_service()
        file_content = storage_service.get_file(file_path)
        
        if file_content is None:
            raise HTTPException(status_code=404, detail="文件不存在")
        
        # 获取文件信息用于设置响应头
        file_info = storage_service.get_file_info(file_path)
        filename = file_info["filename"] if file_info else "download"
        
        return Response(
            content=file_content,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件下载失败: {str(e)}")


@router.delete("/file", response_model=ResponseModel)
async def delete_file(file_path: str = Query(..., description="文件路径")):
    """删除文件"""
    try:
        storage_service = get_storage_service()
        success = storage_service.delete_file(file_path)
        
        if not success:
            raise HTTPException(status_code=404, detail="文件不存在或删除失败")
        
        return ResponseModel(
            success=True,
            message="文件删除成功",
            data={"file_path": file_path}
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件删除失败: {str(e)}")


@router.get("/cache/stats", response_model=ResponseModel)
async def get_cache_stats():
    """获取缓存统计信息"""
    try:
        storage_service = get_storage_service()
        stats = storage_service.get_storage_stats()
        
        return ResponseModel(
            success=True,
            message="缓存统计信息获取成功",
            data=stats["cache_stats"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取缓存统计信息失败: {str(e)}")


@router.delete("/cache", response_model=ResponseModel)
async def clear_cache(cache_type: Optional[str] = Query(None, description="缓存类型")):
    """清理缓存"""
    try:
        storage_service = get_storage_service()
        storage_service.clear_cache(cache_type)
        
        return ResponseModel(
            success=True,
            message="缓存清理成功",
            data={"cache_type": cache_type or "all"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"缓存清理失败: {str(e)}")


@router.get("/logs", response_model=ResponseModel)
async def get_log_files(log_type: Optional[str] = Query(None, description="日志类型")):
    """获取日志文件列表"""
    try:
        storage_service = get_storage_service()
        log_files = storage_service.get_log_files(log_type)
        
        return ResponseModel(
            success=True,
            message="日志文件列表获取成功",
            data={
                "log_type": log_type,
                "files": log_files,
                "total_count": len(log_files)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取日志文件列表失败: {str(e)}")


@router.get("/logs/read", response_model=ResponseModel)
async def read_log_file(
    log_file_path: str = Query(..., description="日志文件路径"),
    lines: Optional[int] = Query(100, description="读取行数"),
    tail: bool = Query(True, description="是否从末尾读取")
):
    """读取日志文件"""
    try:
        storage_service = get_storage_service()
        log_lines = storage_service.read_log_file(log_file_path, lines, tail)
        
        return ResponseModel(
            success=True,
            message="日志文件读取成功",
            data={
                "file_path": log_file_path,
                "lines": log_lines,
                "line_count": len(log_lines)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取日志文件失败: {str(e)}")


@router.get("/logs/search", response_model=ResponseModel)
async def search_logs(
    keyword: str = Query(..., description="搜索关键词"),
    log_type: Optional[str] = Query(None, description="日志类型"),
    max_results: int = Query(1000, description="最大结果数")
):
    """搜索日志"""
    try:
        storage_service = get_storage_service()
        results = storage_service.search_logs(keyword, log_type, max_results)
        
        return ResponseModel(
            success=True,
            message="日志搜索完成",
            data={
                "keyword": keyword,
                "log_type": log_type,
                "results": results,
                "result_count": len(results)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"日志搜索失败: {str(e)}")


@router.post("/cleanup", response_model=ResponseModel)
async def cleanup_storage():
    """清理存储"""
    try:
        storage_service = get_storage_service()
        storage_service.cleanup_storage()
        
        return ResponseModel(
            success=True,
            message="存储清理完成",
            data={"cleanup_time": "now"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"存储清理失败: {str(e)}")


@router.post("/backup", response_model=ResponseModel)
async def backup_storage(backup_name: Optional[str] = Query(None, description="备份名称")):
    """备份存储"""
    try:
        storage_service = get_storage_service()
        backup_path = storage_service.backup_storage(backup_name)
        
        return ResponseModel(
            success=True,
            message="存储备份完成",
            data={
                "backup_name": backup_name,
                "backup_path": backup_path
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"存储备份失败: {str(e)}")
