#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: start.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 快速启动脚本

提供简单的启动方式，适合开发和测试。

本文件是分析服务项目的一部分。
"""

# 禁用 __pycache__ 生成（必须在所有导入之前设置）
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

# 额外的缓存禁用方法
import sys
sys.dont_write_bytecode = True

from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 设置开发环境
os.environ['ENVIRONMENT'] = 'development'

if __name__ == "__main__":
    import uvicorn
    from main import create_app
    
    print("🚀 快速启动分析服务（开发模式）...")
    print("   - 主机: 127.0.0.1")
    print("   - 端口: 8002")
    print("   - 热重载: 启用")
    print("   - 缓存: 禁用 (__pycache__)")
    print("   - 文档: http://127.0.0.1:8002/docs")
    print()
    
    uvicorn.run(
        "main:create_app",
        factory=True,
        host="127.0.0.1",
        port=8002,
        reload=True,
        log_level="info"
    )
