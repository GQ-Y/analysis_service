#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: base_plugin.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 插件基类

定义插件的基础接口和生命周期管理。

本文件是分析服务项目的一部分。
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class PluginStatus(str, Enum):
    """插件状态枚举"""
    UNLOADED = "unloaded"       # 未加载
    LOADING = "loading"         # 加载中
    LOADED = "loaded"           # 已加载
    INITIALIZING = "initializing"  # 初始化中
    ACTIVE = "active"           # 活跃
    INACTIVE = "inactive"       # 非活跃
    ERROR = "error"             # 错误
    UNLOADING = "unloading"     # 卸载中


class PluginInfo(BaseModel):
    """插件信息"""
    name: str = Field(description="插件名称")
    version: str = Field(description="插件版本")
    description: str = Field(description="插件描述")
    author: str = Field(description="插件作者")
    dependencies: List[str] = Field(default_factory=list, description="依赖插件列表")
    category: str = Field(default="general", description="插件分类")
    priority: int = Field(default=100, description="插件优先级")
    enabled: bool = Field(default=True, description="是否启用")


class BasePlugin(ABC):
    """插件基类"""
    
    def __init__(self):
        """初始化插件"""
        self.logger = logging.getLogger(f"Plugin.{self.__class__.__name__}")
        self._status = PluginStatus.UNLOADED
        self._loaded_at: Optional[datetime] = None
        self._error_message: Optional[str] = None
        self._config: Dict[str, Any] = {}
        
        # 获取插件信息
        self._info = self.get_plugin_info()
        
        self.logger.info(f"插件实例化: {self._info.name} v{self._info.version}")
    
    @abstractmethod
    def get_plugin_info(self) -> PluginInfo:
        """获取插件信息
        
        Returns:
            PluginInfo: 插件信息
        """
        pass
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any] = None) -> bool:
        """初始化插件
        
        Args:
            config: 插件配置
            
        Returns:
            bool: 是否初始化成功
        """
        pass
    
    @abstractmethod
    async def start(self) -> bool:
        """启动插件
        
        Returns:
            bool: 是否启动成功
        """
        pass
    
    @abstractmethod
    async def stop(self) -> bool:
        """停止插件
        
        Returns:
            bool: 是否停止成功
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> bool:
        """清理插件资源
        
        Returns:
            bool: 是否清理成功
        """
        pass
    
    def get_status(self) -> PluginStatus:
        """获取插件状态
        
        Returns:
            PluginStatus: 插件状态
        """
        return self._status
    
    def set_status(self, status: PluginStatus, error_message: str = None):
        """设置插件状态
        
        Args:
            status: 插件状态
            error_message: 错误消息
        """
        old_status = self._status
        self._status = status
        self._error_message = error_message
        
        if status == PluginStatus.LOADED:
            self._loaded_at = datetime.now()
        
        self.logger.info(f"插件状态变更: {old_status} -> {status}")
        
        if error_message:
            self.logger.error(f"插件错误: {error_message}")
    
    def is_active(self) -> bool:
        """检查插件是否活跃
        
        Returns:
            bool: 是否活跃
        """
        return self._status == PluginStatus.ACTIVE
    
    def is_loaded(self) -> bool:
        """检查插件是否已加载
        
        Returns:
            bool: 是否已加载
        """
        return self._status in [PluginStatus.LOADED, PluginStatus.ACTIVE, PluginStatus.INACTIVE]
    
    def get_info(self) -> PluginInfo:
        """获取插件信息
        
        Returns:
            PluginInfo: 插件信息
        """
        return self._info
    
    def get_config(self) -> Dict[str, Any]:
        """获取插件配置
        
        Returns:
            Dict[str, Any]: 插件配置
        """
        return self._config.copy()
    
    def set_config(self, config: Dict[str, Any]):
        """设置插件配置
        
        Args:
            config: 插件配置
        """
        self._config = config.copy() if config else {}
        self.logger.info(f"插件配置已更新: {len(self._config)} 个配置项")
    
    def get_error_message(self) -> Optional[str]:
        """获取错误消息
        
        Returns:
            Optional[str]: 错误消息
        """
        return self._error_message
    
    def get_loaded_time(self) -> Optional[datetime]:
        """获取加载时间
        
        Returns:
            Optional[datetime]: 加载时间
        """
        return self._loaded_at
    
    def get_runtime_info(self) -> Dict[str, Any]:
        """获取运行时信息
        
        Returns:
            Dict[str, Any]: 运行时信息
        """
        runtime_seconds = 0
        if self._loaded_at:
            runtime_seconds = (datetime.now() - self._loaded_at).total_seconds()
        
        return {
            'name': self._info.name,
            'version': self._info.version,
            'status': self._status,
            'loaded_at': self._loaded_at,
            'runtime_seconds': runtime_seconds,
            'error_message': self._error_message,
            'config_items': len(self._config),
            'is_active': self.is_active(),
            'is_loaded': self.is_loaded()
        }
    
    async def reload(self, config: Dict[str, Any] = None) -> bool:
        """重新加载插件
        
        Args:
            config: 新的插件配置
            
        Returns:
            bool: 是否重新加载成功
        """
        self.logger.info(f"重新加载插件: {self._info.name}")
        
        try:
            # 停止插件
            if self.is_active():
                await self.stop()
            
            # 清理资源
            await self.cleanup()
            
            # 重新初始化
            success = await self.initialize(config)
            if success:
                # 重新启动
                success = await self.start()
            
            return success
            
        except Exception as e:
            self.set_status(PluginStatus.ERROR, str(e))
            self.logger.error(f"重新加载插件失败: {e}")
            return False
    
    def validate_dependencies(self, available_plugins: List[str]) -> List[str]:
        """验证插件依赖
        
        Args:
            available_plugins: 可用插件列表
            
        Returns:
            List[str]: 缺失的依赖列表
        """
        missing_deps = []
        for dep in self._info.dependencies:
            if dep not in available_plugins:
                missing_deps.append(dep)
        
        if missing_deps:
            self.logger.warning(f"插件依赖缺失: {missing_deps}")
        
        return missing_deps
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"{self._info.name} v{self._info.version} ({self._status})"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (f"BasePlugin(name='{self._info.name}', "
                f"version='{self._info.version}', "
                f"status='{self._status}', "
                f"category='{self._info.category}')")


class StreamPlugin(BasePlugin):
    """流处理插件基类"""
    
    @abstractmethod
    async def process_stream(self, stream_url: str, config: Dict[str, Any] = None) -> bool:
        """处理流
        
        Args:
            stream_url: 流URL
            config: 处理配置
            
        Returns:
            bool: 是否处理成功
        """
        pass
    
    @abstractmethod
    async def stop_stream(self, stream_url: str) -> bool:
        """停止流处理
        
        Args:
            stream_url: 流URL
            
        Returns:
            bool: 是否停止成功
        """
        pass
    
    @abstractmethod
    def get_stream_status(self, stream_url: str) -> Dict[str, Any]:
        """获取流状态
        
        Args:
            stream_url: 流URL
            
        Returns:
            Dict[str, Any]: 流状态信息
        """
        pass


class AnalysisPlugin(BasePlugin):
    """分析插件基类"""
    
    @abstractmethod
    async def analyze(self, data: Any, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """执行分析
        
        Args:
            data: 分析数据
            config: 分析配置
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """获取支持的数据格式
        
        Returns:
            List[str]: 支持的格式列表
        """
        pass


class StoragePlugin(BasePlugin):
    """存储插件基类"""
    
    @abstractmethod
    async def save(self, key: str, data: Any, metadata: Dict[str, Any] = None) -> bool:
        """保存数据
        
        Args:
            key: 数据键
            data: 数据内容
            metadata: 元数据
            
        Returns:
            bool: 是否保存成功
        """
        pass
    
    @abstractmethod
    async def load(self, key: str) -> Optional[Any]:
        """加载数据
        
        Args:
            key: 数据键
            
        Returns:
            Optional[Any]: 数据内容
        """
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """删除数据
        
        Args:
            key: 数据键
            
        Returns:
            bool: 是否删除成功
        """
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """检查数据是否存在
        
        Args:
            key: 数据键
            
        Returns:
            bool: 是否存在
        """
        pass
