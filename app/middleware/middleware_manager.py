#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: middleware_manager.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 中间件管理器

负责加载、配置和管理所有中间件，实现中间件的动态加载和配置。
参考ThinkPHP的中间件管理方式。

本文件是分析服务项目的一部分。
"""

import importlib
from typing import Dict, List, Any, Type, Optional
from fastapi import FastAPI, Request
from .base_middleware import BaseMiddleware, MiddlewarePipeline
from config.middleware import (
    GLOBAL_MIDDLEWARE,
    MIDDLEWARE_ALIAS,
    MIDDLEWARE_GROUPS,
    CONTROLLER_MIDDLEWARE,
    MIDDLEWARE_PARAMS,
    MIDDLEWARE_PRIORITY,
    MIDDLEWARE_ENABLED
)


class MiddlewareManager:
    """中间件管理器"""
    
    def __init__(self, app: FastAPI = None):
        """初始化中间件管理器
        
        Args:
            app: FastAPI应用实例
        """
        self.app = app
        self.middleware_classes: Dict[str, Type[BaseMiddleware]] = {}
        self.middleware_instances: Dict[str, BaseMiddleware] = {}
        self.global_pipeline = MiddlewarePipeline()
        self.route_pipelines: Dict[str, MiddlewarePipeline] = {}
        self.controller_pipelines: Dict[str, MiddlewarePipeline] = {}
        
        # 加载中间件类
        self._load_middleware_classes()
        
        # 初始化全局中间件
        self._init_global_middleware()
        
        # 初始化控制器中间件
        self._init_controller_middleware()
    
    def _load_middleware_classes(self):
        """加载中间件类"""
        # 加载别名中间件
        for alias, class_path in MIDDLEWARE_ALIAS.items():
            try:
                middleware_class = self._import_middleware_class(class_path)
                self.middleware_classes[alias] = middleware_class
            except Exception as e:
                print(f"加载中间件失败 {alias}: {e}")
        
        # 加载全局中间件
        for class_path in GLOBAL_MIDDLEWARE:
            try:
                middleware_class = self._import_middleware_class(class_path)
                # 使用类名作为键
                class_name = class_path.split('.')[-1]
                self.middleware_classes[class_name] = middleware_class
            except Exception as e:
                print(f"加载全局中间件失败 {class_path}: {e}")
    
    def _import_middleware_class(self, class_path: str) -> Type[BaseMiddleware]:
        """导入中间件类
        
        Args:
            class_path: 类路径，格式为 'module.path.ClassName'
            
        Returns:
            Type[BaseMiddleware]: 中间件类
        """
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    
    def _init_global_middleware(self):
        """初始化全局中间件"""
        for class_path in GLOBAL_MIDDLEWARE:
            class_name = class_path.split('.')[-1]
            if class_name in self.middleware_classes:
                # 获取中间件参数
                alias = self._get_middleware_alias(class_name)
                params = MIDDLEWARE_PARAMS.get(alias, {})
                
                # 检查是否启用
                if not MIDDLEWARE_ENABLED.get(alias, True):
                    continue
                
                # 创建中间件实例
                middleware_class = self.middleware_classes[class_name]
                middleware_instance = middleware_class(**params)
                
                # 添加到全局管道
                self.global_pipeline.add_middleware(middleware_instance)
                self.middleware_instances[class_name] = middleware_instance
    
    def _init_controller_middleware(self):
        """初始化控制器中间件"""
        for controller_class, config in CONTROLLER_MIDDLEWARE.items():
            middleware_names = config.get('middleware', [])
            pipeline = MiddlewarePipeline()
            
            # 解析中间件组
            expanded_middleware = self._expand_middleware_groups(middleware_names)
            
            # 创建中间件实例
            for middleware_name in expanded_middleware:
                middleware_instance = self._create_middleware_instance(middleware_name)
                if middleware_instance:
                    pipeline.add_middleware(middleware_instance)
            
            self.controller_pipelines[controller_class] = pipeline
    
    def _expand_middleware_groups(self, middleware_names: List[str]) -> List[str]:
        """展开中间件组
        
        Args:
            middleware_names: 中间件名称列表
            
        Returns:
            List[str]: 展开后的中间件名称列表
        """
        expanded = []
        for name in middleware_names:
            if name in MIDDLEWARE_GROUPS:
                # 递归展开中间件组
                group_middleware = self._expand_middleware_groups(MIDDLEWARE_GROUPS[name])
                expanded.extend(group_middleware)
            else:
                expanded.append(name)
        return expanded
    
    def _create_middleware_instance(self, middleware_name: str) -> Optional[BaseMiddleware]:
        """创建中间件实例
        
        Args:
            middleware_name: 中间件名称
            
        Returns:
            Optional[BaseMiddleware]: 中间件实例
        """
        # 解析中间件名称和参数
        if ':' in middleware_name:
            name, param = middleware_name.split(':', 1)
            params = MIDDLEWARE_PARAMS.get(name, {}).copy()
            params.update({'param': param})
        else:
            name = middleware_name
            params = MIDDLEWARE_PARAMS.get(name, {})
        
        # 检查是否启用
        if not MIDDLEWARE_ENABLED.get(name, True):
            return None
        
        # 获取中间件类
        if name in self.middleware_classes:
            middleware_class = self.middleware_classes[name]
            return middleware_class(**params)
        
        return None
    
    def _get_middleware_alias(self, class_name: str) -> str:
        """获取中间件别名
        
        Args:
            class_name: 类名
            
        Returns:
            str: 别名
        """
        for alias, class_path in MIDDLEWARE_ALIAS.items():
            if class_path.endswith(class_name):
                return alias
        return class_name.lower().replace('middleware', '')
    
    def get_route_middleware(self, route_path: str) -> MiddlewarePipeline:
        """获取路由中间件管道
        
        Args:
            route_path: 路由路径
            
        Returns:
            MiddlewarePipeline: 中间件管道
        """
        if route_path not in self.route_pipelines:
            self.route_pipelines[route_path] = MiddlewarePipeline()
        return self.route_pipelines[route_path]
    
    def get_controller_middleware(self, controller_class: str) -> Optional[MiddlewarePipeline]:
        """获取控制器中间件管道
        
        Args:
            controller_class: 控制器类名
            
        Returns:
            Optional[MiddlewarePipeline]: 中间件管道
        """
        return self.controller_pipelines.get(controller_class)
    
    def add_route_middleware(self, route_path: str, middleware_names: List[str]):
        """为路由添加中间件
        
        Args:
            route_path: 路由路径
            middleware_names: 中间件名称列表
        """
        pipeline = self.get_route_middleware(route_path)
        expanded_middleware = self._expand_middleware_groups(middleware_names)
        
        for middleware_name in expanded_middleware:
            middleware_instance = self._create_middleware_instance(middleware_name)
            if middleware_instance:
                pipeline.add_middleware(middleware_instance)
    
    def register_with_fastapi(self):
        """注册中间件到FastAPI应用"""
        if not self.app:
            return
        
        # 注册全局中间件
        @self.app.middleware("http")
        async def global_middleware_handler(request: Request, call_next):
            return await self.global_pipeline.process(request, call_next)
    
    def get_middleware_info(self) -> Dict[str, Any]:
        """获取中间件信息
        
        Returns:
            Dict[str, Any]: 中间件信息
        """
        return {
            'loaded_classes': list(self.middleware_classes.keys()),
            'global_middleware': [type(m).__name__ for m in self.global_pipeline.middlewares],
            'controller_middleware': {
                controller: [type(m).__name__ for m in pipeline.middlewares]
                for controller, pipeline in self.controller_pipelines.items()
            },
            'route_middleware': {
                route: [type(m).__name__ for m in pipeline.middlewares]
                for route, pipeline in self.route_pipelines.items()
            },
        }


# 全局中间件管理器实例
middleware_manager = MiddlewareManager()
