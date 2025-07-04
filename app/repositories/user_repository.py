#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: user_repository.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 用户仓储

处理用户相关的数据访问操作，包括用户的增删改查、认证、权限管理等。

本文件是分析服务项目的一部分。
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from .base_repository import InMemoryRepository
from app.models.user_model import (
    UserModel, UserSession, UserPreferences, UserActivity,
    UserStatusEnum, RoleEnum
)
from app.exceptions.business_exception import BusinessException


class UserRepository(InMemoryRepository[UserModel]):
    """用户仓储"""
    
    def __init__(self):
        """初始化用户仓储"""
        super().__init__(UserModel)
    
    async def find_by_username(self, username: str) -> Optional[UserModel]:
        """根据用户名查找用户
        
        Args:
            username: 用户名
            
        Returns:
            Optional[UserModel]: 用户对象
        """
        return await self.find_one_by_field('username', username)
    
    async def find_by_email(self, email: str) -> Optional[UserModel]:
        """根据邮箱查找用户
        
        Args:
            email: 邮箱地址
            
        Returns:
            Optional[UserModel]: 用户对象
        """
        return await self.find_one_by_field('email', email)
    
    async def find_by_role(self, role: RoleEnum) -> List[UserModel]:
        """根据角色查找用户
        
        Args:
            role: 用户角色
            
        Returns:
            List[UserModel]: 用户列表
        """
        return await self.find_by_field('role', role)
    
    async def find_by_status(self, status: UserStatusEnum) -> List[UserModel]:
        """根据状态查找用户
        
        Args:
            status: 用户状态
            
        Returns:
            List[UserModel]: 用户列表
        """
        return await self.find_by_field('status', status)
    
    async def find_active_users(self) -> List[UserModel]:
        """查找活跃用户
        
        Returns:
            List[UserModel]: 活跃用户列表
        """
        return await self.find_by_status(UserStatusEnum.ACTIVE)
    
    async def find_admins(self) -> List[UserModel]:
        """查找管理员用户
        
        Returns:
            List[UserModel]: 管理员用户列表
        """
        admins = []
        admin_roles = [RoleEnum.ADMIN, RoleEnum.SUPER_ADMIN]
        for role in admin_roles:
            admins.extend(await self.find_by_role(role))
        return admins
    
    async def check_username_exists(self, username: str, exclude_user_id: str = None) -> bool:
        """检查用户名是否存在
        
        Args:
            username: 用户名
            exclude_user_id: 排除的用户ID
            
        Returns:
            bool: 是否存在
        """
        user = await self.find_by_username(username)
        if not user:
            return False
        
        if exclude_user_id and user.id == exclude_user_id:
            return False
        
        return True
    
    async def check_email_exists(self, email: str, exclude_user_id: str = None) -> bool:
        """检查邮箱是否存在
        
        Args:
            email: 邮箱地址
            exclude_user_id: 排除的用户ID
            
        Returns:
            bool: 是否存在
        """
        user = await self.find_by_email(email)
        if not user:
            return False
        
        if exclude_user_id and user.id == exclude_user_id:
            return False
        
        return True
    
    async def update_login_info(self, user_id: str, ip_address: str) -> bool:
        """更新登录信息
        
        Args:
            user_id: 用户ID
            ip_address: IP地址
            
        Returns:
            bool: 是否更新成功
        """
        user = await self.get_by_id(user_id)
        if not user:
            return False
        
        user.update_login_info(ip_address)
        await self.update(user)
        return True
    
    async def update_status(self, user_id: str, status: UserStatusEnum) -> bool:
        """更新用户状态
        
        Args:
            user_id: 用户ID
            status: 新状态
            
        Returns:
            bool: 是否更新成功
        """
        user = await self.get_by_id(user_id)
        if not user:
            return False
        
        user.status = status
        user.updated_at = datetime.now()
        await self.update(user)
        return True
    
    async def add_permission(self, user_id: str, permission: str) -> bool:
        """添加用户权限
        
        Args:
            user_id: 用户ID
            permission: 权限
            
        Returns:
            bool: 是否添加成功
        """
        user = await self.get_by_id(user_id)
        if not user:
            return False
        
        user.add_permission(permission)
        await self.update(user)
        return True
    
    async def remove_permission(self, user_id: str, permission: str) -> bool:
        """移除用户权限
        
        Args:
            user_id: 用户ID
            permission: 权限
            
        Returns:
            bool: 是否移除成功
        """
        user = await self.get_by_id(user_id)
        if not user:
            return False
        
        user.remove_permission(permission)
        await self.update(user)
        return True
    
    async def get_user_statistics(self) -> Dict[str, Any]:
        """获取用户统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        all_users = await self.find_all()
        
        # 统计各状态用户数量
        status_counts = {}
        for status in UserStatusEnum:
            status_counts[status.value] = len([user for user in all_users if user.status == status])
        
        # 统计角色分布
        role_counts = {}
        for role in RoleEnum:
            role_counts[role.value] = len([user for user in all_users if user.role == role])
        
        # 统计最近登录用户
        recent_login_threshold = datetime.now() - timedelta(days=30)
        recent_login_users = len([
            user for user in all_users 
            if user.last_login_at and user.last_login_at >= recent_login_threshold
        ])
        
        # 统计新注册用户
        recent_register_threshold = datetime.now() - timedelta(days=7)
        new_users = len([
            user for user in all_users 
            if user.created_at >= recent_register_threshold
        ])
        
        return {
            'total_users': len(all_users),
            'status_distribution': status_counts,
            'role_distribution': role_counts,
            'recent_login_users': recent_login_users,
            'new_users_this_week': new_users,
            'active_rate': status_counts.get(UserStatusEnum.ACTIVE.value, 0) / len(all_users) if all_users else 0
        }
    
    async def _validate_entity(self, entity: UserModel):
        """验证用户实体"""
        await super()._validate_entity(entity)
        
        # 检查用户名唯一性
        if await self.check_username_exists(entity.username, entity.id):
            raise BusinessException("用户名已存在")
        
        # 检查邮箱唯一性
        if await self.check_email_exists(entity.email, entity.id):
            raise BusinessException("邮箱已存在")


class UserSessionRepository(InMemoryRepository[UserSession]):
    """用户会话仓储"""
    
    def __init__(self):
        """初始化用户会话仓储"""
        super().__init__(UserSession)
    
    async def find_by_user_id(self, user_id: str) -> List[UserSession]:
        """根据用户ID查找会话
        
        Args:
            user_id: 用户ID
            
        Returns:
            List[UserSession]: 会话列表
        """
        return await self.find_by_field('user_id', user_id)
    
    async def find_by_token(self, session_token: str) -> Optional[UserSession]:
        """根据会话令牌查找会话
        
        Args:
            session_token: 会话令牌
            
        Returns:
            Optional[UserSession]: 会话对象
        """
        return await self.find_one_by_field('session_token', session_token)
    
    async def find_active_sessions(self, user_id: str = None) -> List[UserSession]:
        """查找活跃会话
        
        Args:
            user_id: 用户ID（可选）
            
        Returns:
            List[UserSession]: 活跃会话列表
        """
        filters = {'is_active': True}
        if user_id:
            filters['user_id'] = user_id
        
        sessions = await self.find_all(filters)
        
        # 过滤未过期的会话
        valid_sessions = []
        for session in sessions:
            if not session.is_expired():
                valid_sessions.append(session)
            else:
                # 自动使过期会话无效
                await self.invalidate_session(session.id)
        
        return valid_sessions
    
    async def invalidate_session(self, session_id: str) -> bool:
        """使会话无效
        
        Args:
            session_id: 会话ID
            
        Returns:
            bool: 是否成功
        """
        session = await self.get_by_id(session_id)
        if not session:
            return False
        
        session.invalidate()
        await self.update(session)
        return True
    
    async def invalidate_user_sessions(self, user_id: str, exclude_session_id: str = None) -> int:
        """使用户的所有会话无效
        
        Args:
            user_id: 用户ID
            exclude_session_id: 排除的会话ID
            
        Returns:
            int: 无效化的会话数量
        """
        sessions = await self.find_by_user_id(user_id)
        count = 0
        
        for session in sessions:
            if exclude_session_id and session.id == exclude_session_id:
                continue
            
            if session.is_active:
                await self.invalidate_session(session.id)
                count += 1
        
        return count
    
    async def cleanup_expired_sessions(self) -> int:
        """清理过期会话
        
        Returns:
            int: 清理的会话数量
        """
        all_sessions = await self.find_all()
        expired_sessions = []
        
        for session in all_sessions:
            if session.is_expired():
                expired_sessions.append(session.id)
        
        # 删除过期会话
        for session_id in expired_sessions:
            await self.delete(session_id)
        
        self.logger.info(f"清理了 {len(expired_sessions)} 个过期会话")
        return len(expired_sessions)
    
    async def update_activity(self, session_id: str) -> bool:
        """更新会话活动时间
        
        Args:
            session_id: 会话ID
            
        Returns:
            bool: 是否更新成功
        """
        session = await self.get_by_id(session_id)
        if not session:
            return False
        
        session.update_activity()
        await self.update(session)
        return True


class UserPreferencesRepository(InMemoryRepository[UserPreferences]):
    """用户偏好仓储"""
    
    def __init__(self):
        """初始化用户偏好仓储"""
        super().__init__(UserPreferences)
    
    async def find_by_user_id(self, user_id: str) -> Optional[UserPreferences]:
        """根据用户ID查找偏好设置
        
        Args:
            user_id: 用户ID
            
        Returns:
            Optional[UserPreferences]: 偏好设置
        """
        return await self.find_one_by_field('user_id', user_id)
    
    async def update_preference(self, user_id: str, key: str, value: Any) -> bool:
        """更新偏好设置
        
        Args:
            user_id: 用户ID
            key: 设置键
            value: 设置值
            
        Returns:
            bool: 是否更新成功
        """
        preferences = await self.find_by_user_id(user_id)
        
        if not preferences:
            # 创建新的偏好设置
            preferences = UserPreferences(user_id=user_id)
            await self.create(preferences)
        
        preferences.update_preference(key, value)
        await self.update(preferences)
        return True


class UserActivityRepository(InMemoryRepository[UserActivity]):
    """用户活动仓储"""
    
    def __init__(self):
        """初始化用户活动仓储"""
        super().__init__(UserActivity)
    
    async def find_by_user_id(self, user_id: str, limit: int = None) -> List[UserActivity]:
        """根据用户ID查找活动记录
        
        Args:
            user_id: 用户ID
            limit: 限制数量
            
        Returns:
            List[UserActivity]: 活动记录列表
        """
        activities = await self.find_by_field('user_id', user_id)
        
        # 按时间倒序排列
        activities.sort(key=lambda x: x.created_at, reverse=True)
        
        if limit:
            activities = activities[:limit]
        
        return activities
    
    async def find_by_action(self, action: str) -> List[UserActivity]:
        """根据操作类型查找活动记录
        
        Args:
            action: 操作类型
            
        Returns:
            List[UserActivity]: 活动记录列表
        """
        return await self.find_by_field('action', action)
    
    async def find_by_date_range(self, start_date: datetime, end_date: datetime, 
                                user_id: str = None) -> List[UserActivity]:
        """根据日期范围查找活动记录
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            user_id: 用户ID（可选）
            
        Returns:
            List[UserActivity]: 活动记录列表
        """
        activities = []
        for activity in self._data.values():
            if start_date <= activity.created_at <= end_date:
                if not user_id or activity.user_id == user_id:
                    activities.append(activity)
        
        # 按时间倒序排列
        activities.sort(key=lambda x: x.created_at, reverse=True)
        return activities
    
    async def log_activity(self, user_id: str, action: str, description: str, 
                          ip_address: str, **kwargs) -> UserActivity:
        """记录用户活动
        
        Args:
            user_id: 用户ID
            action: 操作类型
            description: 操作描述
            ip_address: IP地址
            **kwargs: 其他参数
            
        Returns:
            UserActivity: 活动记录
        """
        activity = UserActivity.create_activity(
            user_id=user_id,
            action=action,
            description=description,
            ip_address=ip_address,
            **kwargs
        )
        
        return await self.create(activity)
    
    async def cleanup_old_activities(self, days: int = 90) -> int:
        """清理旧活动记录
        
        Args:
            days: 保留天数
            
        Returns:
            int: 清理的记录数量
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        old_activities = []
        for activity in self._data.values():
            if activity.created_at < cutoff_date:
                old_activities.append(activity.id)
        
        # 删除旧记录
        for activity_id in old_activities:
            await self.delete(activity_id)
        
        self.logger.info(f"清理了 {len(old_activities)} 个旧活动记录")
        return len(old_activities)
