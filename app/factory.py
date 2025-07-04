#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: factory.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 应用程序工厂

提供应用程序的创建和配置功能。

本文件是分析服务项目的一部分。
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config.settings import Settings, get_settings
from config.middleware import setup_middleware
from config.routes import setup_routes
from config.dependencies import setup_dependencies
from config.exceptions import setup_exception_handlers

from app.exceptions import global_exception_handler


class ApplicationFactory:
    """应用程序工厂"""
    
    def __init__(self, settings: Optional[Settings] = None):
        """初始化应用程序工厂
        
        Args:
            settings: 应用设置，如果为None则使用默认设置
        """
        self.settings = settings or get_settings()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def create_app(self, **kwargs) -> FastAPI:
        """创建FastAPI应用
        
        Args:
            **kwargs: 额外的FastAPI参数
            
        Returns:
            FastAPI: 配置好的FastAPI应用实例
        """
        self.logger.info("创建FastAPI应用")
        
        # 合并默认参数和传入参数
        app_kwargs = {
            "title": self.settings.APP_NAME,
            "description": self.settings.APP_DESCRIPTION,
            "version": self.settings.APP_VERSION,
            "debug": self.settings.DEBUG,
            "docs_url": "/docs" if self.settings.DEBUG else None,
            "redoc_url": "/redoc" if self.settings.DEBUG else None,
            "openapi_url": "/openapi.json" if self.settings.DEBUG else None,
        }
        app_kwargs.update(kwargs)
        
        # 创建FastAPI实例
        app = FastAPI(**app_kwargs)
        
        # 配置应用
        self._configure_cors(app)
        self._configure_static_files(app)
        self._configure_middleware(app)
        self._configure_exception_handlers(app)
        self._configure_routes(app)
        
        self.logger.info("FastAPI应用创建完成")
        return app
    
    def _configure_cors(self, app: FastAPI):
        """配置CORS
        
        Args:
            app: FastAPI应用实例
        """
        if self.settings.CORS_ORIGINS:
            self.logger.info("配置CORS中间件")
            app.add_middleware(
                CORSMiddleware,
                allow_origins=self.settings.CORS_ORIGINS,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
    
    def _configure_static_files(self, app: FastAPI):
        """配置静态文件服务
        
        Args:
            app: FastAPI应用实例
        """
        if self.settings.STATIC_DIR and os.path.exists(self.settings.STATIC_DIR):
            self.logger.info(f"配置静态文件服务: {self.settings.STATIC_DIR}")
            app.mount("/static", StaticFiles(directory=self.settings.STATIC_DIR), name="static")
    
    def _configure_middleware(self, app: FastAPI):
        """配置中间件
        
        Args:
            app: FastAPI应用实例
        """
        self.logger.info("配置应用中间件")
        setup_middleware(app)
    
    def _configure_exception_handlers(self, app: FastAPI):
        """配置异常处理器
        
        Args:
            app: FastAPI应用实例
        """
        self.logger.info("配置异常处理器")
        
        # 设置自定义异常处理器
        setup_exception_handlers(app)
        
        # 添加全局异常处理器
        app.add_exception_handler(Exception, global_exception_handler)
    
    def _configure_routes(self, app: FastAPI):
        """配置路由

        Args:
            app: FastAPI应用实例
        """
        self.logger.info("配置应用路由")
        setup_routes(app)

        # 注册API路由
        from app.api import api_router
        app.include_router(api_router)
    
    def create_test_app(self, **kwargs) -> FastAPI:
        """创建测试用的FastAPI应用
        
        Args:
            **kwargs: 额外的FastAPI参数
            
        Returns:
            FastAPI: 测试用的FastAPI应用实例
        """
        self.logger.info("创建测试FastAPI应用")
        
        # 测试环境的特殊配置
        test_kwargs = {
            "debug": True,
            "docs_url": "/docs",
            "redoc_url": "/redoc",
            "openapi_url": "/openapi.json",
        }
        test_kwargs.update(kwargs)
        
        return self.create_app(**test_kwargs)
    
    def create_production_app(self, **kwargs) -> FastAPI:
        """创建生产环境的FastAPI应用
        
        Args:
            **kwargs: 额外的FastAPI参数
            
        Returns:
            FastAPI: 生产环境的FastAPI应用实例
        """
        self.logger.info("创建生产环境FastAPI应用")
        
        # 生产环境的特殊配置
        production_kwargs = {
            "debug": False,
            "docs_url": None,
            "redoc_url": None,
            "openapi_url": None,
        }
        production_kwargs.update(kwargs)
        
        return self.create_app(**production_kwargs)


class ApplicationBuilder:
    """应用程序构建器"""
    
    def __init__(self):
        """初始化应用程序构建器"""
        self.settings: Optional[Settings] = None
        self.middleware_configs: list = []
        self.route_configs: list = []
        self.exception_handlers: Dict[Any, Any] = {}
        self.startup_handlers: list = []
        self.shutdown_handlers: list = []
        self.app_kwargs: Dict[str, Any] = {}
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def with_settings(self, settings: Settings) -> "ApplicationBuilder":
        """设置应用配置
        
        Args:
            settings: 应用设置
            
        Returns:
            ApplicationBuilder: 构建器实例
        """
        self.settings = settings
        return self
    
    def with_middleware(self, middleware_class, **kwargs) -> "ApplicationBuilder":
        """添加中间件
        
        Args:
            middleware_class: 中间件类
            **kwargs: 中间件参数
            
        Returns:
            ApplicationBuilder: 构建器实例
        """
        self.middleware_configs.append((middleware_class, kwargs))
        return self
    
    def with_exception_handler(self, exc_class, handler) -> "ApplicationBuilder":
        """添加异常处理器
        
        Args:
            exc_class: 异常类
            handler: 处理器函数
            
        Returns:
            ApplicationBuilder: 构建器实例
        """
        self.exception_handlers[exc_class] = handler
        return self
    
    def with_startup_handler(self, handler) -> "ApplicationBuilder":
        """添加启动处理器
        
        Args:
            handler: 启动处理器函数
            
        Returns:
            ApplicationBuilder: 构建器实例
        """
        self.startup_handlers.append(handler)
        return self
    
    def with_shutdown_handler(self, handler) -> "ApplicationBuilder":
        """添加关闭处理器
        
        Args:
            handler: 关闭处理器函数
            
        Returns:
            ApplicationBuilder: 构建器实例
        """
        self.shutdown_handlers.append(handler)
        return self
    
    def with_app_config(self, **kwargs) -> "ApplicationBuilder":
        """设置应用配置
        
        Args:
            **kwargs: FastAPI应用参数
            
        Returns:
            ApplicationBuilder: 构建器实例
        """
        self.app_kwargs.update(kwargs)
        return self
    
    def build(self) -> FastAPI:
        """构建FastAPI应用
        
        Returns:
            FastAPI: 构建好的FastAPI应用实例
        """
        self.logger.info("开始构建FastAPI应用")
        
        # 使用工厂创建基础应用
        factory = ApplicationFactory(self.settings)
        app = factory.create_app(**self.app_kwargs)
        
        # 添加自定义中间件
        for middleware_class, kwargs in self.middleware_configs:
            self.logger.info(f"添加中间件: {middleware_class.__name__}")
            app.add_middleware(middleware_class, **kwargs)
        
        # 添加自定义异常处理器
        for exc_class, handler in self.exception_handlers.items():
            self.logger.info(f"添加异常处理器: {exc_class.__name__}")
            app.add_exception_handler(exc_class, handler)
        
        # 添加启动处理器
        for handler in self.startup_handlers:
            self.logger.info(f"添加启动处理器: {handler.__name__}")
            app.add_event_handler("startup", handler)
        
        # 添加关闭处理器
        for handler in self.shutdown_handlers:
            self.logger.info(f"添加关闭处理器: {handler.__name__}")
            app.add_event_handler("shutdown", handler)
        
        self.logger.info("FastAPI应用构建完成")
        return app


# 便捷函数
def create_app(settings: Optional[Settings] = None, **kwargs) -> FastAPI:
    """创建FastAPI应用的便捷函数
    
    Args:
        settings: 应用设置
        **kwargs: 额外的FastAPI参数
        
    Returns:
        FastAPI: FastAPI应用实例
    """
    factory = ApplicationFactory(settings)
    return factory.create_app(**kwargs)


def create_test_app(settings: Optional[Settings] = None, **kwargs) -> FastAPI:
    """创建测试用FastAPI应用的便捷函数
    
    Args:
        settings: 应用设置
        **kwargs: 额外的FastAPI参数
        
    Returns:
        FastAPI: 测试用FastAPI应用实例
    """
    factory = ApplicationFactory(settings)
    return factory.create_test_app(**kwargs)


def create_production_app(settings: Optional[Settings] = None, **kwargs) -> FastAPI:
    """创建生产环境FastAPI应用的便捷函数
    
    Args:
        settings: 应用设置
        **kwargs: 额外的FastAPI参数
        
    Returns:
        FastAPI: 生产环境FastAPI应用实例
    """
    factory = ApplicationFactory(settings)
    return factory.create_production_app(**kwargs)


def get_app_builder() -> ApplicationBuilder:
    """获取应用程序构建器的便捷函数
    
    Returns:
        ApplicationBuilder: 应用程序构建器实例
    """
    return ApplicationBuilder()
