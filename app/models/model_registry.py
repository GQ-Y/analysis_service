#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: model_registry.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 模型注册表

统一管理所有数据模型，提供模型发现、验证、序列化等功能。

本文件是分析服务项目的一部分。
"""

import inspect
import importlib
from typing import Dict, Type, Any, List, Optional, Union
from pathlib import Path
from pydantic import BaseModel

from .base_model import BaseEntity


class ModelRegistry:
    """模型注册表"""
    
    def __init__(self):
        """初始化模型注册表"""
        self.models: Dict[str, Type[BaseModel]] = {}
        self.entities: Dict[str, Type[BaseEntity]] = {}
        self._discovered = False
    
    def discover_models(self, package_path: str = "app.models") -> Dict[str, Type[BaseModel]]:
        """发现模型类
        
        Args:
            package_path: 模型包路径
            
        Returns:
            Dict[str, Type[BaseModel]]: 模型类字典
        """
        if self._discovered:
            return self.models
        
        models = {}
        entities = {}
        
        # 获取模型目录
        models_dir = Path(__file__).parent
        
        # 遍历所有Python文件
        for file_path in models_dir.glob("*_model.py"):
            module_name = file_path.stem
            
            # 跳过基础模型文件
            if module_name == 'base_model':
                continue
            
            try:
                # 导入模块
                module = importlib.import_module(f"{package_path}.{module_name}")
                
                # 查找模型类
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, BaseModel) and obj != BaseModel:
                        model_name = self._get_model_name(name)
                        models[model_name] = obj
                        
                        # 如果是实体类，也添加到实体字典
                        if issubclass(obj, BaseEntity) and obj != BaseEntity:
                            entities[model_name] = obj
                        
            except ImportError as e:
                print(f"导入模型模块失败 {module_name}: {e}")
        
        self.models.update(models)
        self.entities.update(entities)
        self._discovered = True
        return models
    
    def register_model(self, name: str, model_class: Type[BaseModel]):
        """注册模型类
        
        Args:
            name: 模型名称
            model_class: 模型类
        """
        self.models[name] = model_class
        
        if issubclass(model_class, BaseEntity):
            self.entities[name] = model_class
    
    def get_model(self, name: str) -> Optional[Type[BaseModel]]:
        """获取模型类
        
        Args:
            name: 模型名称
            
        Returns:
            Optional[Type[BaseModel]]: 模型类
        """
        # 确保已发现模型
        if not self._discovered:
            self.discover_models()
        
        return self.models.get(name)
    
    def get_entity(self, name: str) -> Optional[Type[BaseEntity]]:
        """获取实体类
        
        Args:
            name: 实体名称
            
        Returns:
            Optional[Type[BaseEntity]]: 实体类
        """
        # 确保已发现模型
        if not self._discovered:
            self.discover_models()
        
        return self.entities.get(name)
    
    def create_instance(self, name: str, data: Dict[str, Any]) -> Optional[BaseModel]:
        """创建模型实例
        
        Args:
            name: 模型名称
            data: 数据字典
            
        Returns:
            Optional[BaseModel]: 模型实例
        """
        model_class = self.get_model(name)
        if model_class:
            try:
                return model_class(**data)
            except Exception as e:
                print(f"创建模型实例失败 {name}: {e}")
                return None
        return None
    
    def validate_data(self, name: str, data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """验证数据
        
        Args:
            name: 模型名称
            data: 数据字典
            
        Returns:
            tuple[bool, Optional[str]]: (是否有效, 错误信息)
        """
        model_class = self.get_model(name)
        if not model_class:
            return False, f"模型不存在: {name}"
        
        try:
            model_class(**data)
            return True, None
        except Exception as e:
            return False, str(e)
    
    def serialize_instance(self, instance: BaseModel, **kwargs) -> Dict[str, Any]:
        """序列化模型实例
        
        Args:
            instance: 模型实例
            **kwargs: 序列化参数
            
        Returns:
            Dict[str, Any]: 序列化后的字典
        """
        return instance.model_dump(**kwargs)
    
    def get_model_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """获取模型JSON Schema
        
        Args:
            name: 模型名称
            
        Returns:
            Optional[Dict[str, Any]]: JSON Schema
        """
        model_class = self.get_model(name)
        if model_class:
            return model_class.model_json_schema()
        return None
    
    def get_all_models(self) -> Dict[str, Type[BaseModel]]:
        """获取所有模型
        
        Returns:
            Dict[str, Type[BaseModel]]: 所有模型字典
        """
        # 确保已发现模型
        if not self._discovered:
            self.discover_models()
        
        return self.models.copy()
    
    def get_all_entities(self) -> Dict[str, Type[BaseEntity]]:
        """获取所有实体
        
        Returns:
            Dict[str, Type[BaseEntity]]: 所有实体字典
        """
        # 确保已发现模型
        if not self._discovered:
            self.discover_models()
        
        return self.entities.copy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息
        """
        # 确保已发现模型
        if not self._discovered:
            self.discover_models()
        
        return {
            'total_models': len(self.models),
            'total_entities': len(self.entities),
            'models': list(self.models.keys()),
            'entities': list(self.entities.keys())
        }
    
    def _get_model_name(self, class_name: str) -> str:
        """从类名获取模型名称
        
        Args:
            class_name: 类名
            
        Returns:
            str: 模型名称
        """
        # 移除Model后缀并转换为snake_case
        name = class_name.replace('Model', '')
        
        # 转换为snake_case
        import re
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
        
        return name


class ModelValidator:
    """模型验证器"""
    
    def __init__(self, registry: ModelRegistry):
        """初始化验证器
        
        Args:
            registry: 模型注册表
        """
        self.registry = registry
    
    def validate_model_data(self, model_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """验证模型数据
        
        Args:
            model_name: 模型名称
            data: 数据字典
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        is_valid, error_message = self.registry.validate_data(model_name, data)
        
        result = {
            'model_name': model_name,
            'is_valid': is_valid,
            'error_message': error_message,
            'validated_at': self._get_current_timestamp()
        }
        
        if is_valid:
            # 创建实例以获取验证后的数据
            instance = self.registry.create_instance(model_name, data)
            if instance:
                result['validated_data'] = self.registry.serialize_instance(instance)
        
        return result
    
    def batch_validate(self, validations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量验证
        
        Args:
            validations: 验证列表，每个元素包含model_name和data
            
        Returns:
            List[Dict[str, Any]]: 验证结果列表
        """
        results = []
        
        for validation in validations:
            model_name = validation.get('model_name')
            data = validation.get('data', {})
            
            if not model_name:
                results.append({
                    'error': 'model_name is required',
                    'is_valid': False
                })
                continue
            
            result = self.validate_model_data(model_name, data)
            results.append(result)
        
        return results
    
    def _get_current_timestamp(self) -> str:
        """获取当前时间戳
        
        Returns:
            str: ISO格式时间戳
        """
        from datetime import datetime
        return datetime.now().isoformat()


class ModelSerializer:
    """模型序列化器"""
    
    def __init__(self, registry: ModelRegistry):
        """初始化序列化器
        
        Args:
            registry: 模型注册表
        """
        self.registry = registry
    
    def serialize(self, instance: BaseModel, format: str = 'dict', **kwargs) -> Union[Dict[str, Any], str]:
        """序列化模型实例
        
        Args:
            instance: 模型实例
            format: 序列化格式 ('dict', 'json')
            **kwargs: 序列化参数
            
        Returns:
            Union[Dict[str, Any], str]: 序列化结果
        """
        if format == 'dict':
            return self.registry.serialize_instance(instance, **kwargs)
        elif format == 'json':
            return instance.model_dump_json(**kwargs)
        else:
            raise ValueError(f"不支持的序列化格式: {format}")
    
    def deserialize(self, model_name: str, data: Union[Dict[str, Any], str]) -> Optional[BaseModel]:
        """反序列化数据
        
        Args:
            model_name: 模型名称
            data: 数据（字典或JSON字符串）
            
        Returns:
            Optional[BaseModel]: 模型实例
        """
        if isinstance(data, str):
            import json
            try:
                data = json.loads(data)
            except json.JSONDecodeError as e:
                print(f"JSON解析失败: {e}")
                return None
        
        return self.registry.create_instance(model_name, data)
    
    def batch_serialize(self, instances: List[BaseModel], format: str = 'dict', **kwargs) -> List[Union[Dict[str, Any], str]]:
        """批量序列化
        
        Args:
            instances: 模型实例列表
            format: 序列化格式
            **kwargs: 序列化参数
            
        Returns:
            List[Union[Dict[str, Any], str]]: 序列化结果列表
        """
        return [self.serialize(instance, format, **kwargs) for instance in instances]


# 全局模型注册表实例
_model_registry = ModelRegistry()


def get_model_registry() -> ModelRegistry:
    """获取全局模型注册表实例
    
    Returns:
        ModelRegistry: 模型注册表实例
    """
    return _model_registry


def get_model_validator() -> ModelValidator:
    """获取模型验证器实例
    
    Returns:
        ModelValidator: 模型验证器实例
    """
    return ModelValidator(_model_registry)


def get_model_serializer() -> ModelSerializer:
    """获取模型序列化器实例
    
    Returns:
        ModelSerializer: 模型序列化器实例
    """
    return ModelSerializer(_model_registry)
