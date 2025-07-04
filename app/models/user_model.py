#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: user_model.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 用户模型

定义用户相关的数据模型，包括用户实体、角色、权限等。

本文件是分析服务项目的一部分。
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import Field, EmailStr, field_validator
from enum import Enum

from .base_model import BaseEntity


class UserStatusEnum(str, Enum):
    """用户状态枚举"""
    ACTIVE = "active"           # 活跃
    INACTIVE = "inactive"       # 非活跃
    SUSPENDED = "suspended"     # 暂停
    DELETED = "deleted"         # 已删除


class RoleEnum(str, Enum):
    """角色枚举"""
    GUEST = "guest"             # 访客
    USER = "user"               # 普通用户
    ADMIN = "admin"             # 管理员
    SUPER_ADMIN = "super_admin" # 超级管理员


class UserModel(BaseEntity):
    """用户模型"""
    
    username: str = Field(
        description="用户名",
        min_length=3,
        max_length=50,
        pattern="^[a-zA-Z0-9_-]+$",
        example="user001"
    )
    
    email: EmailStr = Field(
        description="邮箱地址",
        example="user@example.com"
    )
    
    password_hash: str = Field(
        description="密码哈希",
        exclude=True  # 序列化时排除
    )
    
    full_name: Optional[str] = Field(
        default=None,
        description="全名",
        max_length=100,
        example="张三"
    )
    
    phone: Optional[str] = Field(
        default=None,
        description="电话号码",
        pattern=r"^[0-9+\-\s()]+$",
        example="+86 138-0013-8000"
    )
    
    role: RoleEnum = Field(
        default=RoleEnum.USER,
        description="用户角色"
    )
    
    status: UserStatusEnum = Field(
        default=UserStatusEnum.ACTIVE,
        description="用户状态"
    )
    
    permissions: List[str] = Field(
        default_factory=list,
        description="用户权限列表"
    )
    
    # 登录信息
    last_login_at: Optional[datetime] = Field(
        default=None,
        description="最后登录时间"
    )
    
    last_login_ip: Optional[str] = Field(
        default=None,
        description="最后登录IP"
    )
    
    login_count: int = Field(
        default=0,
        description="登录次数",
        ge=0
    )
    
    # 账户设置
    timezone: str = Field(
        default="UTC",
        description="时区",
        example="Asia/Shanghai"
    )
    
    language: str = Field(
        default="zh-CN",
        description="语言",
        example="zh-CN"
    )
    
    # 配额限制
    max_tasks: int = Field(
        default=10,
        description="最大任务数",
        ge=0
    )
    
    max_streams: int = Field(
        default=5,
        description="最大流数",
        ge=0
    )
    
    # 元数据
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="用户元数据"
    )
    
    @field_validator('username')
    @classmethod
    def validate_username(cls, v):
        """验证用户名"""
        if v.lower() in ['admin', 'root', 'system', 'api']:
            raise ValueError("用户名不能使用保留字")
        return v
    
    def is_active(self) -> bool:
        """检查用户是否活跃"""
        return self.status == UserStatusEnum.ACTIVE
    
    def is_admin(self) -> bool:
        """检查是否为管理员"""
        return self.role in [RoleEnum.ADMIN, RoleEnum.SUPER_ADMIN]
    
    def has_permission(self, permission: str) -> bool:
        """检查是否有指定权限"""
        return permission in self.permissions
    
    def add_permission(self, permission: str):
        """添加权限"""
        if permission not in self.permissions:
            self.permissions.append(permission)
            self.update_timestamp()
    
    def remove_permission(self, permission: str):
        """移除权限"""
        if permission in self.permissions:
            self.permissions.remove(permission)
            self.update_timestamp()
    
    def update_login_info(self, ip_address: str):
        """更新登录信息"""
        self.last_login_at = datetime.now()
        self.last_login_ip = ip_address
        self.login_count += 1
        self.update_timestamp()
    
    def get_profile(self) -> Dict[str, Any]:
        """获取用户资料"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'full_name': self.full_name,
            'phone': self.phone,
            'role': self.role,
            'status': self.status,
            'permissions': self.permissions,
            'timezone': self.timezone,
            'language': self.language,
            'created_at': self.created_at,
            'last_login_at': self.last_login_at
        }


class UserSession(BaseEntity):
    """用户会话模型"""
    
    user_id: str = Field(
        description="用户ID"
    )
    
    session_token: str = Field(
        description="会话令牌"
    )
    
    refresh_token: Optional[str] = Field(
        default=None,
        description="刷新令牌"
    )
    
    ip_address: str = Field(
        description="IP地址"
    )
    
    user_agent: Optional[str] = Field(
        default=None,
        description="用户代理"
    )
    
    expires_at: datetime = Field(
        description="过期时间"
    )
    
    is_active: bool = Field(
        default=True,
        description="是否活跃"
    )
    
    last_activity_at: datetime = Field(
        default_factory=datetime.now,
        description="最后活动时间"
    )
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        return datetime.now() > self.expires_at
    
    def is_valid(self) -> bool:
        """检查是否有效"""
        return self.is_active and not self.is_expired()
    
    def update_activity(self):
        """更新活动时间"""
        self.last_activity_at = datetime.now()
        self.update_timestamp()
    
    def invalidate(self):
        """使会话无效"""
        self.is_active = False
        self.update_timestamp()


class UserPreferences(BaseEntity):
    """用户偏好设置模型"""
    
    user_id: str = Field(
        description="用户ID"
    )
    
    # 界面设置
    theme: str = Field(
        default="light",
        description="主题",
        pattern="^(light|dark|auto)$"
    )
    
    language: str = Field(
        default="zh-CN",
        description="语言"
    )
    
    timezone: str = Field(
        default="UTC",
        description="时区"
    )
    
    # 通知设置
    email_notifications: bool = Field(
        default=True,
        description="邮件通知"
    )
    
    task_notifications: bool = Field(
        default=True,
        description="任务通知"
    )
    
    system_notifications: bool = Field(
        default=True,
        description="系统通知"
    )
    
    # 显示设置
    items_per_page: int = Field(
        default=20,
        description="每页显示项目数",
        ge=10,
        le=100
    )
    
    auto_refresh: bool = Field(
        default=True,
        description="自动刷新"
    )
    
    auto_refresh_interval: int = Field(
        default=30,
        description="自动刷新间隔（秒）",
        ge=10,
        le=300
    )
    
    # 其他设置
    custom_settings: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="自定义设置"
    )
    
    def update_preference(self, key: str, value: Any):
        """更新偏好设置"""
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            if self.custom_settings is None:
                self.custom_settings = {}
            self.custom_settings[key] = value
        
        self.update_timestamp()
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """获取偏好设置"""
        if hasattr(self, key):
            return getattr(self, key)
        elif self.custom_settings and key in self.custom_settings:
            return self.custom_settings[key]
        else:
            return default


class UserActivity(BaseEntity):
    """用户活动记录模型"""
    
    user_id: str = Field(
        description="用户ID"
    )
    
    action: str = Field(
        description="操作类型",
        example="login"
    )
    
    resource_type: Optional[str] = Field(
        default=None,
        description="资源类型",
        example="task"
    )
    
    resource_id: Optional[str] = Field(
        default=None,
        description="资源ID"
    )
    
    description: str = Field(
        description="操作描述",
        example="用户登录系统"
    )
    
    ip_address: str = Field(
        description="IP地址"
    )
    
    user_agent: Optional[str] = Field(
        default=None,
        description="用户代理"
    )
    
    result: str = Field(
        default="success",
        description="操作结果",
        pattern="^(success|failure|error)$"
    )
    
    error_message: Optional[str] = Field(
        default=None,
        description="错误信息"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="活动元数据"
    )
    
    @classmethod
    def create_activity(
        cls,
        user_id: str,
        action: str,
        description: str,
        ip_address: str,
        resource_type: str = None,
        resource_id: str = None,
        result: str = "success",
        error_message: str = None,
        user_agent: str = None,
        **metadata
    ) -> 'UserActivity':
        """创建活动记录"""
        return cls(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            description=description,
            ip_address=ip_address,
            user_agent=user_agent,
            result=result,
            error_message=error_message,
            metadata=metadata
        )
