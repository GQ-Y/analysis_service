#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: base_repository.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 基础仓储类

定义数据访问层的基础功能，提供统一的数据操作接口。
所有具体仓储都应该继承此基类。

本文件是分析服务项目的一部分。
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic
from datetime import datetime

from app.models.base_model import BaseEntity
from app.exceptions.business_exception import BusinessException

# 泛型类型变量
T = TypeVar('T', bound=BaseEntity)


class BaseRepository(Generic[T], ABC):
    """基础仓储类"""
    
    def __init__(self, model_class: Type[T]):
        """初始化基础仓储
        
        Args:
            model_class: 模型类
        """
        self.model_class = model_class
        self.logger = logging.getLogger(self.__class__.__name__)
        self._connection = None
        self._transaction = None
    
    @property
    def connection(self):
        """获取数据库连接"""
        if self._connection is None:
            self._connection = self._get_connection()
        return self._connection
    
    @abstractmethod
    def _get_connection(self):
        """获取数据库连接（子类实现）"""
        pass
    
    async def begin_transaction(self):
        """开始事务"""
        if self._transaction is None:
            self._transaction = await self._begin_transaction()
    
    async def commit_transaction(self):
        """提交事务"""
        if self._transaction:
            await self._commit_transaction()
            self._transaction = None
    
    async def rollback_transaction(self):
        """回滚事务"""
        if self._transaction:
            await self._rollback_transaction()
            self._transaction = None
    
    @abstractmethod
    async def _begin_transaction(self):
        """开始事务（子类实现）"""
        pass
    
    @abstractmethod
    async def _commit_transaction(self):
        """提交事务（子类实现）"""
        pass
    
    @abstractmethod
    async def _rollback_transaction(self):
        """回滚事务（子类实现）"""
        pass
    
    async def create(self, entity: T) -> T:
        """创建实体
        
        Args:
            entity: 实体对象
            
        Returns:
            T: 创建的实体
        """
        try:
            self.logger.info(f"创建{self.model_class.__name__}实体", extra={'entity_id': entity.id})
            
            # 设置创建时间
            if hasattr(entity, 'created_at') and entity.created_at is None:
                entity.created_at = datetime.now()
            if hasattr(entity, 'updated_at'):
                entity.updated_at = datetime.now()
            
            # 验证实体
            await self._validate_entity(entity)
            
            # 执行创建
            created_entity = await self._create_entity(entity)
            
            self.logger.info(f"{self.model_class.__name__}实体创建成功", extra={'entity_id': entity.id})
            return created_entity
            
        except Exception as e:
            self.logger.error(f"创建{self.model_class.__name__}实体失败", exc_info=True, extra={'entity_id': entity.id})
            raise BusinessException(f"创建实体失败: {str(e)}")
    
    async def get_by_id(self, entity_id: str) -> Optional[T]:
        """根据ID获取实体
        
        Args:
            entity_id: 实体ID
            
        Returns:
            Optional[T]: 实体对象
        """
        try:
            self.logger.debug(f"获取{self.model_class.__name__}实体", extra={'entity_id': entity_id})
            
            entity = await self._get_entity_by_id(entity_id)
            
            if entity:
                self.logger.debug(f"{self.model_class.__name__}实体获取成功", extra={'entity_id': entity_id})
            else:
                self.logger.debug(f"{self.model_class.__name__}实体不存在", extra={'entity_id': entity_id})
            
            return entity
            
        except Exception as e:
            self.logger.error(f"获取{self.model_class.__name__}实体失败", exc_info=True, extra={'entity_id': entity_id})
            raise BusinessException(f"获取实体失败: {str(e)}")
    
    async def update(self, entity: T) -> T:
        """更新实体
        
        Args:
            entity: 实体对象
            
        Returns:
            T: 更新的实体
        """
        try:
            self.logger.info(f"更新{self.model_class.__name__}实体", extra={'entity_id': entity.id})
            
            # 设置更新时间
            if hasattr(entity, 'updated_at'):
                entity.updated_at = datetime.now()
            
            # 验证实体
            await self._validate_entity(entity)
            
            # 执行更新
            updated_entity = await self._update_entity(entity)
            
            self.logger.info(f"{self.model_class.__name__}实体更新成功", extra={'entity_id': entity.id})
            return updated_entity
            
        except Exception as e:
            self.logger.error(f"更新{self.model_class.__name__}实体失败", exc_info=True, extra={'entity_id': entity.id})
            raise BusinessException(f"更新实体失败: {str(e)}")
    
    async def delete(self, entity_id: str) -> bool:
        """删除实体
        
        Args:
            entity_id: 实体ID
            
        Returns:
            bool: 是否删除成功
        """
        try:
            self.logger.info(f"删除{self.model_class.__name__}实体", extra={'entity_id': entity_id})
            
            # 检查实体是否存在
            entity = await self.get_by_id(entity_id)
            if not entity:
                raise BusinessException("实体不存在")
            
            # 执行删除
            success = await self._delete_entity(entity_id)
            
            if success:
                self.logger.info(f"{self.model_class.__name__}实体删除成功", extra={'entity_id': entity_id})
            else:
                self.logger.warning(f"{self.model_class.__name__}实体删除失败", extra={'entity_id': entity_id})
            
            return success
            
        except Exception as e:
            self.logger.error(f"删除{self.model_class.__name__}实体失败", exc_info=True, extra={'entity_id': entity_id})
            raise BusinessException(f"删除实体失败: {str(e)}")
    
    async def find_all(self, filters: Dict[str, Any] = None, 
                      order_by: str = None, limit: int = None, offset: int = None) -> List[T]:
        """查找所有实体
        
        Args:
            filters: 过滤条件
            order_by: 排序字段
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            List[T]: 实体列表
        """
        try:
            self.logger.debug(f"查找{self.model_class.__name__}实体列表", extra={
                'filters': filters,
                'order_by': order_by,
                'limit': limit,
                'offset': offset
            })
            
            entities = await self._find_entities(filters, order_by, limit, offset)
            
            self.logger.debug(f"{self.model_class.__name__}实体列表查找成功", extra={'count': len(entities)})
            return entities
            
        except Exception as e:
            self.logger.error(f"查找{self.model_class.__name__}实体列表失败", exc_info=True)
            raise BusinessException(f"查找实体列表失败: {str(e)}")
    
    async def count(self, filters: Dict[str, Any] = None) -> int:
        """统计实体数量
        
        Args:
            filters: 过滤条件
            
        Returns:
            int: 实体数量
        """
        try:
            self.logger.debug(f"统计{self.model_class.__name__}实体数量", extra={'filters': filters})
            
            count = await self._count_entities(filters)
            
            self.logger.debug(f"{self.model_class.__name__}实体数量统计成功", extra={'count': count})
            return count
            
        except Exception as e:
            self.logger.error(f"统计{self.model_class.__name__}实体数量失败", exc_info=True)
            raise BusinessException(f"统计实体数量失败: {str(e)}")
    
    async def exists(self, entity_id: str) -> bool:
        """检查实体是否存在
        
        Args:
            entity_id: 实体ID
            
        Returns:
            bool: 是否存在
        """
        try:
            entity = await self.get_by_id(entity_id)
            return entity is not None
        except Exception:
            return False
    
    async def find_by_field(self, field_name: str, field_value: Any) -> List[T]:
        """根据字段查找实体
        
        Args:
            field_name: 字段名
            field_value: 字段值
            
        Returns:
            List[T]: 实体列表
        """
        filters = {field_name: field_value}
        return await self.find_all(filters)
    
    async def find_one_by_field(self, field_name: str, field_value: Any) -> Optional[T]:
        """根据字段查找单个实体
        
        Args:
            field_name: 字段名
            field_value: 字段值
            
        Returns:
            Optional[T]: 实体对象
        """
        entities = await self.find_by_field(field_name, field_value)
        return entities[0] if entities else None
    
    async def _validate_entity(self, entity: T):
        """验证实体（子类可重写）
        
        Args:
            entity: 实体对象
        """
        # 基础验证
        if not entity.id:
            raise BusinessException("实体ID不能为空")
    
    # 抽象方法，子类必须实现
    @abstractmethod
    async def _create_entity(self, entity: T) -> T:
        """创建实体（子类实现）"""
        pass
    
    @abstractmethod
    async def _get_entity_by_id(self, entity_id: str) -> Optional[T]:
        """根据ID获取实体（子类实现）"""
        pass
    
    @abstractmethod
    async def _update_entity(self, entity: T) -> T:
        """更新实体（子类实现）"""
        pass
    
    @abstractmethod
    async def _delete_entity(self, entity_id: str) -> bool:
        """删除实体（子类实现）"""
        pass
    
    @abstractmethod
    async def _find_entities(self, filters: Dict[str, Any] = None, 
                           order_by: str = None, limit: int = None, offset: int = None) -> List[T]:
        """查找实体（子类实现）"""
        pass
    
    @abstractmethod
    async def _count_entities(self, filters: Dict[str, Any] = None) -> int:
        """统计实体数量（子类实现）"""
        pass


class InMemoryRepository(BaseRepository[T]):
    """内存仓储实现（用于测试和开发）"""
    
    def __init__(self, model_class: Type[T]):
        """初始化内存仓储"""
        super().__init__(model_class)
        self._data: Dict[str, T] = {}
        self._transaction_data: Optional[Dict[str, T]] = None
    
    def _get_connection(self):
        """获取连接（内存实现）"""
        return self._data
    
    async def _begin_transaction(self):
        """开始事务"""
        self._transaction_data = self._data.copy()
        return True
    
    async def _commit_transaction(self):
        """提交事务"""
        self._transaction_data = None
    
    async def _rollback_transaction(self):
        """回滚事务"""
        if self._transaction_data is not None:
            self._data = self._transaction_data
            self._transaction_data = None
    
    async def _create_entity(self, entity: T) -> T:
        """创建实体"""
        self._data[entity.id] = entity
        return entity
    
    async def _get_entity_by_id(self, entity_id: str) -> Optional[T]:
        """根据ID获取实体"""
        return self._data.get(entity_id)
    
    async def _update_entity(self, entity: T) -> T:
        """更新实体"""
        if entity.id not in self._data:
            raise BusinessException("实体不存在")
        self._data[entity.id] = entity
        return entity
    
    async def _delete_entity(self, entity_id: str) -> bool:
        """删除实体"""
        if entity_id in self._data:
            del self._data[entity_id]
            return True
        return False
    
    async def _find_entities(self, filters: Dict[str, Any] = None, 
                           order_by: str = None, limit: int = None, offset: int = None) -> List[T]:
        """查找实体"""
        entities = list(self._data.values())
        
        # 应用过滤器
        if filters:
            filtered_entities = []
            for entity in entities:
                match = True
                for key, value in filters.items():
                    if hasattr(entity, key) and getattr(entity, key) != value:
                        match = False
                        break
                if match:
                    filtered_entities.append(entity)
            entities = filtered_entities
        
        # 排序
        if order_by:
            reverse = order_by.startswith('-')
            field = order_by.lstrip('-')
            entities.sort(key=lambda x: getattr(x, field, None), reverse=reverse)
        
        # 分页
        if offset:
            entities = entities[offset:]
        if limit:
            entities = entities[:limit]
        
        return entities
    
    async def _count_entities(self, filters: Dict[str, Any] = None) -> int:
        """统计实体数量"""
        if not filters:
            return len(self._data)
        
        count = 0
        for entity in self._data.values():
            match = True
            for key, value in filters.items():
                if hasattr(entity, key) and getattr(entity, key) != value:
                    match = False
                    break
            if match:
                count += 1
        
        return count
