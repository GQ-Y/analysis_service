"""
配置管理模块
负责加载和管理各模块的配置
"""

import os
import json
import configparser
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Set, Type
from pathlib import Path
import threading

class IConfig(ABC):
    """配置接口"""
    
    @abstractmethod
    def load(self) -> bool:
        """加载配置
        
        Returns:
            bool: 是否成功加载
        """
        pass
    
    @abstractmethod
    def save(self) -> bool:
        """保存配置
        
        Returns:
            bool: 是否成功保存
        """
        pass
    
    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项值
        
        Args:
            key: 配置项键
            default: 默认值
            
        Returns:
            Any: 配置项值
        """
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """设置配置项值
        
        Args:
            key: 配置项键
            value: 配置项值
        """
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典
        
        Returns:
            Dict[str, Any]: 配置字典
        """
        pass
    
    @abstractmethod
    def update(self, config_dict: Dict[str, Any]) -> None:
        """从字典更新配置
        
        Args:
            config_dict: 配置字典
        """
        pass

class IConfigFactory(ABC):
    """配置工厂接口"""
    
    @abstractmethod
    def create_config(self, config_type: str, **kwargs) -> IConfig:
        """创建配置实例
        
        Args:
            config_type: 配置类型
            **kwargs: 其他参数
            
        Returns:
            IConfig: 配置实例
        """
        pass

class BaseConfig(IConfig):
    """基础配置类"""
    
    def __init__(self, config_file: Optional[str] = None):
        """初始化基础配置
        
        Args:
            config_file: 配置文件路径
        """
        self._config_file = config_file
        self._config = {}
        self._lock = threading.Lock()
    
    def load(self) -> bool:
        """加载配置
        
        Returns:
            bool: 是否成功加载
        """
        if not self._config_file or not os.path.exists(self._config_file):
            return False
        
        try:
            with open(self._config_file, 'r', encoding='utf-8') as f:
                self._config = json.load(f)
            return True
        except Exception as e:
            print(f"加载配置文件失败: {str(e)}")
            return False
    
    def save(self) -> bool:
        """保存配置
        
        Returns:
            bool: 是否成功保存
        """
        if not self._config_file:
            return False
        
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self._config_file), exist_ok=True)
            
            with open(self._config_file, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"保存配置文件失败: {str(e)}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项值
        
        Args:
            key: 配置项键
            default: 默认值
            
        Returns:
            Any: 配置项值
        """
        with self._lock:
            return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """设置配置项值
        
        Args:
            key: 配置项键
            value: 配置项值
        """
        with self._lock:
            self._config[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典
        
        Returns:
            Dict[str, Any]: 配置字典
        """
        with self._lock:
            return self._config.copy()
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """从字典更新配置
        
        Args:
            config_dict: 配置字典
        """
        with self._lock:
            self._config.update(config_dict)

class JsonConfig(BaseConfig):
    """JSON配置类"""
    pass

class IniConfig(BaseConfig):
    """INI配置类"""
    
    def __init__(self, config_file: Optional[str] = None):
        """初始化INI配置
        
        Args:
            config_file: 配置文件路径
        """
        super().__init__(config_file)
        self._parser = configparser.ConfigParser()
    
    def load(self) -> bool:
        """加载配置
        
        Returns:
            bool: 是否成功加载
        """
        if not self._config_file or not os.path.exists(self._config_file):
            return False
        
        try:
            self._parser.read(self._config_file, encoding='utf-8')
            
            # 转换为字典
            self._config = {}
            for section in self._parser.sections():
                self._config[section] = {}
                for key, value in self._parser[section].items():
                    # 尝试转换为合适的类型
                    try:
                        # 尝试转换为int
                        self._config[section][key] = int(value)
                    except ValueError:
                        try:
                            # 尝试转换为float
                            self._config[section][key] = float(value)
                        except ValueError:
                            # 尝试转换为bool
                            if value.lower() in ('true', 'yes', '1'):
                                self._config[section][key] = True
                            elif value.lower() in ('false', 'no', '0'):
                                self._config[section][key] = False
                            else:
                                # 保持为字符串
                                self._config[section][key] = value
            
            return True
        except Exception as e:
            print(f"加载INI配置文件失败: {str(e)}")
            return False
    
    def save(self) -> bool:
        """保存配置
        
        Returns:
            bool: 是否成功保存
        """
        if not self._config_file:
            return False
        
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self._config_file), exist_ok=True)
            
            # 从字典更新ConfigParser
            self._parser = configparser.ConfigParser()
            for section, items in self._config.items():
                self._parser[section] = {}
                for key, value in items.items():
                    # 将各种类型转换为字符串
                    self._parser[section][key] = str(value)
            
            # 写入文件
            with open(self._config_file, 'w', encoding='utf-8') as f:
                self._parser.write(f)
            
            return True
        except Exception as e:
            print(f"保存INI配置文件失败: {str(e)}")
            return False

class ConfigManager:
    """配置管理器，负责加载和管理各模块配置"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """单例模式"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ConfigManager, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化配置管理器"""
        if self._initialized:
            return
            
        self._config_cache = {}  # 模块名 -> 配置对象
        self._config_path = {}   # 模块名 -> 配置文件路径
        self._config_type = {}   # 模块名 -> 配置类型
        self._initialized = True
    
    def register_config(self, module_name: str, config_path: str, config_type: str = 'json') -> None:
        """注册模块配置路径
        
        Args:
            module_name: 模块名称
            config_path: 配置文件路径
            config_type: 配置类型，支持'json'和'ini'
        """
        self._config_path[module_name] = config_path
        self._config_type[module_name] = config_type
    
    def get_config(self, module_name: str) -> Optional[IConfig]:
        """获取模块配置
        
        Args:
            module_name: 模块名称
            
        Returns:
            Optional[IConfig]: 模块配置，如果不存在则返回None
        """
        # 检查缓存
        if module_name in self._config_cache:
            return self._config_cache[module_name]
        
        # 检查是否注册
        if module_name not in self._config_path:
            return None
        
        # 创建配置对象
        config_path = self._config_path[module_name]
        config_type = self._config_type.get(module_name, 'json')
        
        if config_type == 'json':
            config = JsonConfig(config_path)
        elif config_type == 'ini':
            config = IniConfig(config_path)
        else:
            raise ValueError(f"不支持的配置类型: {config_type}")
        
        # 加载配置
        config.load()
        
        # 缓存配置
        self._config_cache[module_name] = config
        
        return config
    
    def reload_config(self, module_name: str) -> bool:
        """重新加载模块配置
        
        Args:
            module_name: 模块名称
            
        Returns:
            bool: 是否成功重新加载
        """
        # 清除缓存
        if module_name in self._config_cache:
            del self._config_cache[module_name]
        
        # 重新获取配置
        config = self.get_config(module_name)
        return config is not None
    
    def save_config(self, module_name: str) -> bool:
        """保存模块配置
        
        Args:
            module_name: 模块名称
            
        Returns:
            bool: 是否成功保存
        """
        # 获取配置
        config = self.get_config(module_name)
        if not config:
            return False
        
        # 保存配置
        return config.save()
    
    def update_config(self, module_name: str, config_dict: Dict[str, Any]) -> bool:
        """更新模块配置
        
        Args:
            module_name: 模块名称
            config_dict: 配置字典
            
        Returns:
            bool: 是否成功更新
        """
        # 获取配置
        config = self.get_config(module_name)
        if not config:
            return False
        
        # 更新配置
        config.update(config_dict)
        
        # 保存配置
        return config.save()

# 单例实例
config_manager = ConfigManager()
