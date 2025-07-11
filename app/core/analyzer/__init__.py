#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
分析器核心模块 - 自动注册分析器
"""

from .analyzer_registry import get_global_registry, ensure_analyzers_registered
from .analyzer_factory import AnalyzerFactory

# 确保分析器在模块导入时自动注册
ensure_analyzers_registered()

# 导出主要接口
__all__ = [
    'get_global_registry',
    'AnalyzerFactory',
    'ensure_analyzers_registered'
]
