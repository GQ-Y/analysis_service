#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: __init__.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 插件系统包

提供插件化架构，支持动态加载和管理各种功能插件。

本文件是分析服务项目的一部分。
"""

from .plugin_manager import PluginManager, get_plugin_manager, initialize_plugin_system
from .base_plugin import BasePlugin
from .plugin_registry import PluginRegistry
from .plugin_loader import PluginLoader

__all__ = [
    'PluginManager',
    'BasePlugin',
    'PluginRegistry',
    'PluginLoader',
    'get_plugin_manager',
    'initialize_plugin_system'
]
