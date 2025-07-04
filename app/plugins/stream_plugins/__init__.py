#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: __init__.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 流媒体插件包

包含各种流媒体处理插件。

本文件是分析服务项目的一部分。
"""

from .rtmp_plugin import RTMPPlugin
from .rtsp_plugin import RTSPPlugin

__all__ = [
    'RTMPPlugin',
    'RTSPPlugin'
]
