#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: __init__.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: API v1版本初始化

管理v1版本的所有API路由。

本文件是分析服务项目的一部分。
"""

from fastapi import APIRouter
from .storage import router as storage_router

# 创建v1 API路由器
api_router = APIRouter()

# 注册各个模块的路由
api_router.include_router(storage_router)

__all__ = ["api_router"]
