#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: main.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 分析服务主启动文件

提供多种启动方式：开发模式、生产模式、Docker模式等。

本文件是分析服务项目的一部分。
"""

import os
import sys
import uvicorn
import argparse
import logging
import logging.config
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.factory import ApplicationFactory
from config.settings import get_settings, get_logging_config


def setup_logging():
    """设置日志配置"""
    try:
        # 确保日志目录存在
        settings = get_settings()
        log_dir = Path(settings.LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取动态日志配置
        logging_config = get_logging_config()
        
        # 初始化日志配置
        logging.config.dictConfig(logging_config)
        
        # 创建根日志记录器
        logger = logging.getLogger(__name__)
        logger.info("📝 日志系统初始化成功")
        logger.info(f"📁 日志目录: {log_dir}")
        
        return True
    except Exception as e:
        print(f"❌ 日志系统初始化失败: {e}")
        return False


def create_app():
    """创建FastAPI应用实例"""
    # 首先初始化日志
    setup_logging()
    
    factory = ApplicationFactory()
    return factory.create_app()


def main():
    """主启动函数"""
    parser = argparse.ArgumentParser(description='分析服务启动器')
    parser.add_argument('--host', default='0.0.0.0', help='服务器主机地址')
    parser.add_argument('--port', type=int, default=8002, help='服务器端口')
    parser.add_argument('--reload', action='store_true', help='启用热重载（开发模式）')
    parser.add_argument('--workers', type=int, default=1, help='工作进程数量')
    parser.add_argument('--log-level', default='info', help='日志级别')
    parser.add_argument('--env', default='development', help='运行环境')
    
    args = parser.parse_args()
    
    # 设置环境变量
    os.environ['ENVIRONMENT'] = args.env
    
    # 获取配置
    settings = get_settings()
    
    print(f"🚀 启动分析服务...")
    print(f"   - 环境: {args.env}")
    print(f"   - 主机: {args.host}")
    print(f"   - 端口: {args.port}")
    print(f"   - 热重载: {'启用' if args.reload else '禁用'}")
    print(f"   - 工作进程: {args.workers}")
    print(f"   - 日志级别: {args.log_level}")
    print()
    
    # 启动服务器
    uvicorn.run(
        "main:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level=args.log_level,
        access_log=True,
        loop="asyncio"
    )


if __name__ == "__main__":
    main()
