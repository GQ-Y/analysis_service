#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: auth_middleware.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 认证中间件

处理用户身份认证，验证JWT令牌，检查用户权限。

本文件是分析服务项目的一部分。
"""

import jwt
from typing import Callable, Optional, Dict, Any
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from .base_middleware import BaseMiddleware


class AuthMiddleware(BaseMiddleware):
    """认证中间件"""
    
    def __init__(self, **kwargs):
        """初始化认证中间件
        
        Args:
            secret_key: JWT密钥
            algorithm: JWT算法
            token_header: 令牌头名称
            token_prefix: 令牌前缀
            default_role: 默认角色
            exclude_paths: 排除的路径列表
        """
        super().__init__(**kwargs)
        
        self.secret_key = kwargs.get('secret_key', 'your-secret-key')
        self.algorithm = kwargs.get('algorithm', 'HS256')
        self.token_header = kwargs.get('token_header', 'Authorization')
        self.token_prefix = kwargs.get('token_prefix', 'Bearer ')
        self.default_role = kwargs.get('default_role', 'user')
        self.required_role = kwargs.get('param', self.default_role)  # 从参数中获取所需角色
        
        # 默认排除的路径
        default_exclude = ['/docs', '/redoc', '/openapi.json', '/health', '/ping']
        self.exclude_paths.extend(kwargs.get('exclude_paths', default_exclude))
    
    async def __call__(self, request: Request, call_next: Callable):
        """处理认证
        
        Args:
            request: HTTP请求对象
            call_next: 下一个中间件或处理器
            
        Returns:
            Response: HTTP响应对象
        """
        if not self.should_process(request):
            return await call_next(request)
        
        # 获取令牌
        token = self._extract_token(request)
        if not token:
            return self._create_auth_error_response("缺少认证令牌", 401)
        
        # 验证令牌
        try:
            payload = self._verify_token(token)
            user_info = await self._get_user_info(payload)
            
            if not user_info:
                return self._create_auth_error_response("用户不存在", 401)
            
            # 检查角色权限
            if not self._check_role_permission(user_info, self.required_role):
                return self._create_auth_error_response("权限不足", 403)
            
            # 将用户信息存储到请求状态中
            request.state.user = user_info
            request.state.authenticated = True
            
            return await call_next(request)
            
        except jwt.ExpiredSignatureError:
            return self._create_auth_error_response("令牌已过期", 401)
        except jwt.InvalidTokenError:
            return self._create_auth_error_response("无效的令牌", 401)
        except Exception as e:
            return self._create_auth_error_response(f"认证失败: {str(e)}", 401)
    
    def _extract_token(self, request: Request) -> Optional[str]:
        """提取令牌
        
        Args:
            request: HTTP请求对象
            
        Returns:
            Optional[str]: 令牌字符串
        """
        # 从请求头获取令牌
        auth_header = request.headers.get(self.token_header)
        if auth_header and auth_header.startswith(self.token_prefix):
            return auth_header[len(self.token_prefix):].strip()
        
        # 从查询参数获取令牌
        token = request.query_params.get('token')
        if token:
            return token
        
        # 从Cookie获取令牌
        token = request.cookies.get('access_token')
        if token:
            return token
        
        return None
    
    def _verify_token(self, token: str) -> Dict[str, Any]:
        """验证JWT令牌
        
        Args:
            token: JWT令牌
            
        Returns:
            Dict[str, Any]: 令牌载荷
            
        Raises:
            jwt.InvalidTokenError: 令牌无效
        """
        return jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
    
    async def _get_user_info(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """获取用户信息
        
        Args:
            payload: JWT载荷
            
        Returns:
            Optional[Dict[str, Any]]: 用户信息
        """
        user_id = payload.get('user_id')
        if not user_id:
            return None
        
        # 这里应该从数据库或缓存中获取用户信息
        # 暂时返回载荷中的用户信息
        return {
            'user_id': user_id,
            'username': payload.get('username', ''),
            'email': payload.get('email', ''),
            'role': payload.get('role', self.default_role),
            'permissions': payload.get('permissions', []),
            'exp': payload.get('exp'),
            'iat': payload.get('iat'),
        }
    
    def _check_role_permission(self, user_info: Dict[str, Any], required_role: str) -> bool:
        """检查角色权限
        
        Args:
            user_info: 用户信息
            required_role: 所需角色
            
        Returns:
            bool: 是否有权限
        """
        user_role = user_info.get('role', '')
        
        # 定义角色层级
        role_hierarchy = {
            'guest': 0,
            'user': 1,
            'admin': 2,
            'super_admin': 3,
        }
        
        user_level = role_hierarchy.get(user_role, 0)
        required_level = role_hierarchy.get(required_role, 1)
        
        return user_level >= required_level
    
    def _create_auth_error_response(self, message: str, status_code: int) -> JSONResponse:
        """创建认证错误响应
        
        Args:
            message: 错误消息
            status_code: 状态码
            
        Returns:
            JSONResponse: 错误响应
        """
        return JSONResponse(
            status_code=status_code,
            content={
                'error': True,
                'message': message,
                'code': status_code,
                'type': 'AuthenticationError',
            }
        )
    
    async def before_request(self, request: Request) -> None:
        """请求前处理
        
        Args:
            request: HTTP请求对象
        """
        # 初始化认证状态
        request.state.authenticated = False
        request.state.user = None
        request.state.required_role = self.required_role
    
    def should_process(self, request: Request) -> bool:
        """判断是否应该处理此请求
        
        Args:
            request: HTTP请求对象
            
        Returns:
            bool: 是否应该处理
        """
        # 检查排除路径
        if request.url.path in self.exclude_paths:
            return False
        
        # 检查路径前缀
        exclude_prefixes = ['/static/', '/docs', '/redoc']
        for prefix in exclude_prefixes:
            if request.url.path.startswith(prefix):
                return False
        
        return True


class OptionalAuthMiddleware(AuthMiddleware):
    """可选认证中间件
    
    如果有令牌则验证，没有令牌则跳过认证
    """
    
    async def __call__(self, request: Request, call_next: Callable):
        """处理可选认证
        
        Args:
            request: HTTP请求对象
            call_next: 下一个中间件或处理器
            
        Returns:
            Response: HTTP响应对象
        """
        if not self.should_process(request):
            return await call_next(request)
        
        # 获取令牌
        token = self._extract_token(request)
        if not token:
            # 没有令牌，设置为未认证状态
            request.state.authenticated = False
            request.state.user = None
            return await call_next(request)
        
        # 有令牌，进行验证
        return await super().__call__(request, call_next)
