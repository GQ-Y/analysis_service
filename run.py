#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: run.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 简单启动脚本

不使用热重载的简单启动方式，适合快速测试。

本文件是分析服务项目的一部分。
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 设置开发环境
os.environ['ENVIRONMENT'] = 'development'

if __name__ == "__main__":
    import uvicorn
    from main import create_app
    
    print("🚀 启动分析服务...")
    print("   - 主机: 127.0.0.1")
    print("   - 端口: 8002")
    print("   - 模式: 开发模式")
    print("   - 文档: http://127.0.0.1:8002/docs")
    print("   - 健康检查: http://127.0.0.1:8002/health")
    print()
    
    # 创建应用实例并启动
    app = create_app()
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8002,
        log_level="info"
    )
