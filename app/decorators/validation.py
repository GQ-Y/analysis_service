#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: validation.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 验证装饰器

提供请求和响应验证的装饰器，使用Pydantic进行数据验证。

本文件是分析服务项目的一部分。
"""

from typing import Type, Callable, Any, Optional
from functools import wraps
from pydantic import BaseModel, ValidationError
from fastapi import Request, HTTPException
import json


def validate_request(schema: Type[BaseModel], source: str = 'json'):
    """请求验证装饰器
    
    Args:
        schema: Pydantic模型类
        source: 数据源 ('json', 'form', 'query')
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 获取request对象
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if not request:
                request = kwargs.get('request')
            
            if not request:
                raise HTTPException(status_code=500, detail="无法获取请求对象")
            
            try:
                # 根据数据源获取数据
                if source == 'json':
                    data = await _get_json_data(request)
                elif source == 'form':
                    data = await _get_form_data(request)
                elif source == 'query':
                    data = _get_query_data(request)
                else:
                    raise ValueError(f"不支持的数据源: {source}")
                
                # 验证数据
                validated_data = schema(**data)
                
                # 将验证后的数据注入到kwargs中
                kwargs['validated_data'] = validated_data
                
                return await func(*args, **kwargs)
                
            except ValidationError as e:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "请求数据验证失败",
                        "details": e.errors()
                    }
                )
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"请求数据处理失败: {str(e)}"
                )
        
        # 标记验证信息
        wrapper._validation_schema = schema
        wrapper._validation_source = source
        
        return wrapper
    
    return decorator


def validate_response(schema: Type[BaseModel]):
    """响应验证装饰器
    
    Args:
        schema: Pydantic模型类
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            try:
                # 验证响应数据
                if isinstance(result, dict):
                    validated_result = schema(**result)
                    return validated_result.dict()
                elif isinstance(result, BaseModel):
                    # 如果已经是Pydantic模型，直接验证类型
                    if not isinstance(result, schema):
                        validated_result = schema(**result.dict())
                        return validated_result.dict()
                    return result.dict()
                else:
                    # 尝试转换为字典后验证
                    if hasattr(result, '__dict__'):
                        validated_result = schema(**result.__dict__)
                        return validated_result.dict()
                    else:
                        validated_result = schema(result)
                        return validated_result.dict()
                        
            except ValidationError as e:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "响应数据验证失败",
                        "details": e.errors()
                    }
                )
        
        # 标记响应验证信息
        wrapper._response_schema = schema
        
        return wrapper
    
    return decorator


def validate_params(**param_schemas):
    """参数验证装饰器
    
    Args:
        **param_schemas: 参数名和对应的验证模型
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            validated_params = {}
            
            for param_name, schema in param_schemas.items():
                if param_name in kwargs:
                    try:
                        if isinstance(schema, type) and issubclass(schema, BaseModel):
                            # Pydantic模型验证
                            validated_params[param_name] = schema(**kwargs[param_name])
                        else:
                            # 简单类型验证
                            validated_params[param_name] = schema(kwargs[param_name])
                    except (ValidationError, ValueError, TypeError) as e:
                        raise HTTPException(
                            status_code=422,
                            detail=f"参数 {param_name} 验证失败: {str(e)}"
                        )
            
            # 更新kwargs
            kwargs.update(validated_params)
            
            return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator


async def _get_json_data(request: Request) -> dict:
    """获取JSON数据
    
    Args:
        request: 请求对象
        
    Returns:
        dict: JSON数据
    """
    try:
        body = await request.body()
        if not body:
            return {}
        
        content_type = request.headers.get('content-type', '')
        if 'application/json' not in content_type:
            raise ValueError("请求内容类型不是JSON")
        
        return json.loads(body.decode('utf-8'))
        
    except json.JSONDecodeError:
        raise ValueError("无效的JSON格式")
    except UnicodeDecodeError:
        raise ValueError("无法解码请求体")


async def _get_form_data(request: Request) -> dict:
    """获取表单数据
    
    Args:
        request: 请求对象
        
    Returns:
        dict: 表单数据
    """
    try:
        form = await request.form()
        return dict(form)
    except Exception as e:
        raise ValueError(f"无法解析表单数据: {str(e)}")


def _get_query_data(request: Request) -> dict:
    """获取查询参数数据
    
    Args:
        request: 请求对象
        
    Returns:
        dict: 查询参数数据
    """
    return dict(request.query_params)


class ValidationMixin:
    """验证混入类"""
    
    @staticmethod
    def validate_data(data: dict, schema: Type[BaseModel]) -> BaseModel:
        """验证数据
        
        Args:
            data: 要验证的数据
            schema: 验证模式
            
        Returns:
            BaseModel: 验证后的数据
        """
        try:
            return schema(**data)
        except ValidationError as e:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "数据验证失败",
                    "details": e.errors()
                }
            )
    
    @staticmethod
    def validate_partial_data(data: dict, schema: Type[BaseModel]) -> BaseModel:
        """部分数据验证（用于更新操作）
        
        Args:
            data: 要验证的数据
            schema: 验证模式
            
        Returns:
            BaseModel: 验证后的数据
        """
        try:
            # 创建模型实例，允许部分字段
            return schema.parse_obj(data)
        except ValidationError as e:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "数据验证失败",
                    "details": e.errors()
                }
            )


def create_validator(schema: Type[BaseModel], source: str = 'json'):
    """创建验证器函数
    
    Args:
        schema: Pydantic模型类
        source: 数据源
        
    Returns:
        Callable: 验证器函数
    """
    async def validator(request: Request) -> BaseModel:
        try:
            if source == 'json':
                data = await _get_json_data(request)
            elif source == 'form':
                data = await _get_form_data(request)
            elif source == 'query':
                data = _get_query_data(request)
            else:
                raise ValueError(f"不支持的数据源: {source}")
            
            return schema(**data)
            
        except ValidationError as e:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "请求数据验证失败",
                    "details": e.errors()
                }
            )
    
    return validator
