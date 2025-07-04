#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: auth.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 认证装饰器

提供认证和权限控制的装饰器，简化权限验证逻辑。

本文件是分析服务项目的一部分。
"""

from typing import List, Union, Callable, Optional
from functools import wraps
from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt


def require_auth(
    roles: Union[str, List[str]] = None,
    permissions: Union[str, List[str]] = None,
    optional: bool = False
):
    """认证装饰器
    
    Args:
        roles: 所需角色
        permissions: 所需权限
        optional: 是否可选认证
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 从参数中获取request对象
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if not request:
                # 从kwargs中查找request
                request = kwargs.get('request')
            
            if not request:
                if optional:
                    return await func(*args, **kwargs)
                raise HTTPException(status_code=500, detail="无法获取请求对象")
            
            # 验证认证
            user = await _verify_authentication(request, optional)
            
            if user:
                # 验证角色
                if roles and not _check_roles(user, roles):
                    raise HTTPException(status_code=403, detail="角色权限不足")
                
                # 验证权限
                if permissions and not _check_permissions(user, permissions):
                    raise HTTPException(status_code=403, detail="操作权限不足")
                
                # 将用户信息注入到请求中
                request.state.user = user
                request.state.authenticated = True
            else:
                request.state.user = None
                request.state.authenticated = False
            
            return await func(*args, **kwargs)
        
        # 标记函数需要认证
        wrapper._requires_auth = True
        wrapper._auth_roles = roles
        wrapper._auth_permissions = permissions
        wrapper._auth_optional = optional
        
        return wrapper
    
    return decorator


def require_role(role: Union[str, List[str]]):
    """角色装饰器
    
    Args:
        role: 所需角色
    """
    return require_auth(roles=role)


def require_permission(permission: Union[str, List[str]]):
    """权限装饰器
    
    Args:
        permission: 所需权限
    """
    return require_auth(permissions=permission)


def optional_auth():
    """可选认证装饰器"""
    return require_auth(optional=True)


async def _verify_authentication(request: Request, optional: bool = False) -> Optional[dict]:
    """验证认证
    
    Args:
        request: 请求对象
        optional: 是否可选
        
    Returns:
        Optional[dict]: 用户信息
    """
    # 获取认证令牌
    token = _extract_token(request)
    
    if not token:
        if optional:
            return None
        raise HTTPException(status_code=401, detail="缺少认证令牌")
    
    try:
        # 验证JWT令牌
        from config.settings import SECURITY_CONFIG
        payload = jwt.decode(
            token,
            SECURITY_CONFIG['secret_key'],
            algorithms=[SECURITY_CONFIG['algorithm']]
        )
        
        # 获取用户信息
        user_info = {
            'user_id': payload.get('user_id'),
            'username': payload.get('username'),
            'email': payload.get('email'),
            'role': payload.get('role', 'user'),
            'permissions': payload.get('permissions', []),
            'exp': payload.get('exp'),
            'iat': payload.get('iat'),
        }
        
        return user_info
        
    except jwt.ExpiredSignatureError:
        if optional:
            return None
        raise HTTPException(status_code=401, detail="令牌已过期")
    except jwt.InvalidTokenError:
        if optional:
            return None
        raise HTTPException(status_code=401, detail="无效的令牌")


def _extract_token(request: Request) -> Optional[str]:
    """提取认证令牌
    
    Args:
        request: 请求对象
        
    Returns:
        Optional[str]: 令牌
    """
    # 从Authorization头获取
    auth_header = request.headers.get('Authorization')
    if auth_header and auth_header.startswith('Bearer '):
        return auth_header[7:]
    
    # 从查询参数获取
    token = request.query_params.get('token')
    if token:
        return token
    
    # 从Cookie获取
    token = request.cookies.get('access_token')
    if token:
        return token
    
    return None


def _check_roles(user: dict, required_roles: Union[str, List[str]]) -> bool:
    """检查角色权限
    
    Args:
        user: 用户信息
        required_roles: 所需角色
        
    Returns:
        bool: 是否有权限
    """
    user_role = user.get('role', '')
    
    if isinstance(required_roles, str):
        required_roles = [required_roles]
    
    # 定义角色层级
    role_hierarchy = {
        'guest': 0,
        'user': 1,
        'admin': 2,
        'super_admin': 3,
    }
    
    user_level = role_hierarchy.get(user_role, 0)
    
    # 检查是否满足任一所需角色
    for role in required_roles:
        required_level = role_hierarchy.get(role, 1)
        if user_level >= required_level:
            return True
    
    return False


def _check_permissions(user: dict, required_permissions: Union[str, List[str]]) -> bool:
    """检查操作权限
    
    Args:
        user: 用户信息
        required_permissions: 所需权限
        
    Returns:
        bool: 是否有权限
    """
    user_permissions = user.get('permissions', [])
    
    if isinstance(required_permissions, str):
        required_permissions = [required_permissions]
    
    # 检查是否拥有所有所需权限
    for permission in required_permissions:
        if permission not in user_permissions:
            return False
    
    return True


class AuthDecorator:
    """认证装饰器类"""
    
    def __init__(self, secret_key: str, algorithm: str = 'HS256'):
        """初始化认证装饰器
        
        Args:
            secret_key: JWT密钥
            algorithm: JWT算法
        """
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    def require_auth(self, roles: Union[str, List[str]] = None, permissions: Union[str, List[str]] = None):
        """认证装饰器方法
        
        Args:
            roles: 所需角色
            permissions: 所需权限
        """
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # 实现认证逻辑
                return await func(*args, **kwargs)
            return wrapper
        return decorator


# 权限常量
class Permissions:
    """权限常量"""
    
    # 任务权限
    TASK_CREATE = 'task:create'
    TASK_READ = 'task:read'
    TASK_UPDATE = 'task:update'
    TASK_DELETE = 'task:delete'
    TASK_START = 'task:start'
    TASK_STOP = 'task:stop'
    
    # 流权限
    STREAM_CREATE = 'stream:create'
    STREAM_READ = 'stream:read'
    STREAM_UPDATE = 'stream:update'
    STREAM_DELETE = 'stream:delete'
    
    # 系统权限
    SYSTEM_CONFIG = 'system:config'
    SYSTEM_MONITOR = 'system:monitor'
    SYSTEM_LOG = 'system:log'
    
    # 管理权限
    ADMIN_USER = 'admin:user'
    ADMIN_ROLE = 'admin:role'
    ADMIN_PERMISSION = 'admin:permission'


# 角色常量
class Roles:
    """角色常量"""
    
    GUEST = 'guest'
    USER = 'user'
    ADMIN = 'admin'
    SUPER_ADMIN = 'super_admin'
