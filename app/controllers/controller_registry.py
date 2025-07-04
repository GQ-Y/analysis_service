#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: controller_registry.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 控制器注册器

自动发现和注册所有控制器，将装饰器定义的路由注册到FastAPI应用中。

本文件是分析服务项目的一部分。
"""

import inspect
import importlib
from typing import Dict, List, Any, Type
from fastapi import FastAPI, APIRouter
from pathlib import Path

from .base_controller import BaseController
from app.decorators.route import route_registry


class ControllerRegistry:
    """控制器注册器"""
    
    def __init__(self, app: FastAPI = None):
        """初始化控制器注册器
        
        Args:
            app: FastAPI应用实例
        """
        self.app = app
        self.controllers: Dict[str, BaseController] = {}
        self.routers: Dict[str, APIRouter] = {}
        
    def discover_controllers(self, package_path: str = "app.controllers") -> List[Type[BaseController]]:
        """发现控制器类
        
        Args:
            package_path: 控制器包路径
            
        Returns:
            List[Type[BaseController]]: 控制器类列表
        """
        controllers = []
        
        # 获取控制器目录
        controllers_dir = Path(__file__).parent
        
        # 遍历所有Python文件
        for file_path in controllers_dir.glob("*_controller.py"):
            module_name = file_path.stem
            
            try:
                # 导入模块
                module = importlib.import_module(f"{package_path}.{module_name}")
                
                # 查找控制器类
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, BaseController) and 
                        obj != BaseController and 
                        name.endswith('Controller')):
                        controllers.append(obj)
                        
            except ImportError as e:
                print(f"导入控制器模块失败 {module_name}: {e}")
                
        return controllers
    
    def register_controller(self, controller_class: Type[BaseController]) -> APIRouter:
        """注册单个控制器
        
        Args:
            controller_class: 控制器类
            
        Returns:
            APIRouter: 路由器实例
        """
        # 创建控制器实例
        controller_instance = controller_class()
        controller_name = controller_class.__name__
        
        # 存储控制器实例
        self.controllers[controller_name] = controller_instance
        
        # 创建路由器
        router = APIRouter()
        
        # 收集控制器中的路由
        self._collect_routes_from_controller(controller_instance, router)
        
        # 存储路由器
        self.routers[controller_name] = router
        
        return router
    
    def register_all_controllers(self) -> Dict[str, APIRouter]:
        """注册所有控制器
        
        Returns:
            Dict[str, APIRouter]: 控制器名称到路由器的映射
        """
        # 发现所有控制器
        controller_classes = self.discover_controllers()
        
        # 注册每个控制器
        for controller_class in controller_classes:
            self.register_controller(controller_class)
        
        return self.routers
    
    def _collect_routes_from_controller(self, controller: BaseController, router: APIRouter):
        """从控制器收集路由
        
        Args:
            controller: 控制器实例
            router: 路由器实例
        """
        # 遍历控制器的所有方法
        for method_name in dir(controller):
            method = getattr(controller, method_name)
            
            # 检查是否有路由元数据
            if hasattr(method, '_route_meta'):
                self._register_route(method, method._route_meta, router)
            
            # 检查是否有WebSocket元数据
            if hasattr(method, '_websocket_meta'):
                self._register_websocket(method, method._websocket_meta, router)
    
    def _register_route(self, handler, meta: Dict[str, Any], router: APIRouter):
        """注册HTTP路由
        
        Args:
            handler: 处理函数
            meta: 路由元数据
            router: 路由器实例
        """
        path = meta['path']
        methods = meta['methods']
        tags = meta.get('tags', [])
        summary = meta.get('summary', '')
        description = meta.get('description', '')
        response_model = meta.get('response_model')
        status_code = meta.get('status_code', 200)
        
        # 注册路由到FastAPI路由器
        for method in methods:
            router.add_api_route(
                path=path,
                endpoint=handler,
                methods=[method],
                tags=tags,
                summary=summary,
                description=description,
                response_model=response_model,
                status_code=status_code,
                include_in_schema=True
            )
    
    def _register_websocket(self, handler, meta: Dict[str, Any], router: APIRouter):
        """注册WebSocket路由
        
        Args:
            handler: 处理函数
            meta: WebSocket元数据
            router: 路由器实例
        """
        path = meta['path']
        
        # 注册WebSocket路由
        router.add_websocket_route(
            path=path,
            endpoint=handler
        )
    
    def register_with_app(self, app: FastAPI = None):
        """将所有路由器注册到FastAPI应用
        
        Args:
            app: FastAPI应用实例
        """
        if app:
            self.app = app
        
        if not self.app:
            raise ValueError("FastAPI应用实例未设置")
        
        # 注册所有控制器
        self.register_all_controllers()
        
        # 将路由器添加到应用
        for controller_name, router in self.routers.items():
            self.app.include_router(router)
            print(f"已注册控制器: {controller_name}")
    
    def get_controller(self, controller_name: str) -> BaseController:
        """获取控制器实例
        
        Args:
            controller_name: 控制器名称
            
        Returns:
            BaseController: 控制器实例
        """
        return self.controllers.get(controller_name)
    
    def get_router(self, controller_name: str) -> APIRouter:
        """获取路由器实例
        
        Args:
            controller_name: 控制器名称
            
        Returns:
            APIRouter: 路由器实例
        """
        return self.routers.get(controller_name)
    
    def get_routes_info(self) -> Dict[str, Any]:
        """获取路由信息
        
        Returns:
            Dict[str, Any]: 路由信息
        """
        routes_info = {}
        
        for controller_name, router in self.routers.items():
            routes = []
            for route in router.routes:
                if hasattr(route, 'methods') and hasattr(route, 'path'):
                    routes.append({
                        'path': route.path,
                        'methods': list(route.methods),
                        'name': route.name,
                        'tags': getattr(route, 'tags', [])
                    })
            
            routes_info[controller_name] = {
                'controller': controller_name,
                'routes_count': len(routes),
                'routes': routes
            }
        
        return routes_info
    
    def print_routes_summary(self):
        """打印路由摘要"""
        routes_info = self.get_routes_info()
        
        print("\n=== 控制器路由摘要 ===")
        total_routes = 0
        
        for controller_name, info in routes_info.items():
            print(f"\n{controller_name}:")
            print(f"  路由数量: {info['routes_count']}")
            
            for route in info['routes']:
                methods_str = ', '.join(route['methods'])
                print(f"  {methods_str:10} {route['path']}")
            
            total_routes += info['routes_count']
        
        print(f"\n总计: {len(routes_info)} 个控制器, {total_routes} 个路由")


# 全局控制器注册器实例
controller_registry = ControllerRegistry()


def register_controllers(app: FastAPI):
    """注册所有控制器到FastAPI应用
    
    Args:
        app: FastAPI应用实例
    """
    controller_registry.register_with_app(app)
    controller_registry.print_routes_summary()


def get_controller_registry() -> ControllerRegistry:
    """获取控制器注册器实例
    
    Returns:
        ControllerRegistry: 控制器注册器实例
    """
    return controller_registry
