#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: __init__.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: API包初始化

统一管理API路由和版本。

本文件是分析服务项目的一部分。
"""

from fastapi import APIRouter
from .v1 import api_router as v1_router

# 创建主API路由器
api_router = APIRouter(prefix="/api")

# 注册v1版本的API
api_router.include_router(v1_router, prefix="/v1")

__all__ = ["api_router"]
