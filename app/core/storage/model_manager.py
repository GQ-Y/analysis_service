#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模型管理器 - 负责模型的缓存、加载和配置管理
"""

import os
import yaml
import json
import logging
import hashlib
from typing import Dict, Optional, List, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

from app.utils.analysis_type_utils import AnalysisTypeEnum, get_analysis_type_by_number


@dataclass
class ModelInfo:
    """模型信息数据类"""
    code: str                    # 模型代码 (如 yolo11n)
    name: str                    # 模型名称
    description: str             # 模型描述
    analysis_type: int           # 分析类型 (1-6)
    model_path: str              # 模型文件路径
    config_path: str             # 配置文件路径
    version: str                 # 模型版本
    file_size: int               # 文件大小(字节)
    file_hash: str               # 文件哈希值
    classes: Dict[int, str]      # 类别映射
    input_size: tuple            # 输入尺寸 (width, height)
    created_at: datetime         # 创建时间
    updated_at: datetime         # 更新时间
    is_downloaded: bool = False  # 是否已下载


class ModelManager:
    """模型管理器"""
    
    def __init__(self, storage_root: str = "storage"):
        """
        初始化模型管理器
        
        Args:
            storage_root: 存储根目录
        """
        self.storage_root = Path(storage_root)
        self.models_dir = self.storage_root / "models"
        self.cache_dir = self.storage_root / "cache"
        self.downloads_dir = self.storage_root / "downloads"
        
        # 确保目录存在
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.downloads_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self._loaded_models: Dict[str, Any] = {}  # 已加载的模型缓存
        self._model_registry: Dict[str, ModelInfo] = {}  # 模型注册表
        
        # 初始化时扫描已存在的模型
        self._scan_existing_models()
    
    def _scan_existing_models(self):
        """扫描已存在的模型"""
        try:
            for model_dir in self.models_dir.iterdir():
                if model_dir.is_dir():
                    config_file = model_dir / "model_config.yaml"
                    if config_file.exists():
                        try:
                            model_info = self._load_model_config(config_file)
                            self._model_registry[model_info.code] = model_info
                            self.logger.info(f"📦 发现模型: {model_info.code} - {model_info.name}")
                        except Exception as e:
                            self.logger.error(f"❌ 加载模型配置失败 {config_file}: {e}")
        except Exception as e:
            self.logger.error(f"❌ 扫描模型目录失败: {e}")
    
    def _load_model_config(self, config_path: Path) -> ModelInfo:
        """加载模型配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 转换为ModelInfo对象
        return ModelInfo(
            code=config['code'],
            name=config['name'],
            description=config.get('description', ''),
            analysis_type=config['analysis_type'],
            model_path=config['model_path'],
            config_path=str(config_path),
            version=config.get('version', '1.0.0'),
            file_size=config.get('file_size', 0),
            file_hash=config.get('file_hash', ''),
            classes=config.get('classes', {}),
            input_size=tuple(config.get('input_size', [640, 640])),
            created_at=datetime.fromisoformat(config.get('created_at', datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(config.get('updated_at', datetime.now().isoformat())),
            is_downloaded=config.get('is_downloaded', False)
        )
    
    def get_model_info(self, model_code: str) -> Optional[ModelInfo]:
        """获取模型信息"""
        return self._model_registry.get(model_code)
    
    def list_models(self, analysis_type: Optional[int] = None) -> List[ModelInfo]:
        """
        列出模型
        
        Args:
            analysis_type: 可选的分析类型过滤
            
        Returns:
            模型信息列表
        """
        models = list(self._model_registry.values())
        if analysis_type is not None:
            models = [m for m in models if m.analysis_type == analysis_type]
        return models
    
    def is_model_available(self, model_code: str) -> bool:
        """检查模型是否可用（已下载且文件存在）"""
        model_info = self.get_model_info(model_code)
        if not model_info:
            return False
        
        model_file_path = Path(model_info.model_path)
        return model_file_path.exists() and model_info.is_downloaded
    
    def get_model_path(self, model_code: str) -> Optional[str]:
        """获取模型文件路径"""
        model_info = self.get_model_info(model_code)
        if model_info and self.is_model_available(model_code):
            return model_info.model_path
        return None
    
    def register_model(self, model_info: ModelInfo) -> bool:
        """
        注册新模型
        
        Args:
            model_info: 模型信息
            
        Returns:
            bool: 是否注册成功
        """
        try:
            # 创建模型目录
            model_dir = self.models_dir / model_info.code
            model_dir.mkdir(exist_ok=True)
            
            # 保存配置文件
            config_path = model_dir / "model_config.yaml"
            config_data = asdict(model_info)
            
            # 转换datetime为字符串
            config_data['created_at'] = model_info.created_at.isoformat()
            config_data['updated_at'] = model_info.updated_at.isoformat()
            
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, ensure_ascii=False, indent=2)
            
            # 更新注册表
            self._model_registry[model_info.code] = model_info
            
            self.logger.info(f"✅ 模型注册成功: {model_info.code}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 模型注册失败 {model_info.code}: {e}")
            return False
    
    def download_model(self, model_code: str) -> bool:
        """
        下载模型
        
        Args:
            model_code: 模型代码
            
        Returns:
            bool: 是否下载成功
        """
        self.logger.warning(f"⚠️ 模型下载功能暂未实现: {model_code}")
        self.logger.info(f"💡 建议: 手动下载模型到 storage/models/{model_code}/ 目录")
        return False
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """计算文件哈希值"""
        try:
            hasher = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            self.logger.error(f"❌ 计算文件哈希失败 {file_path}: {e}")
            return ""
    
    def verify_model_integrity(self, model_code: str) -> bool:
        """验证模型文件完整性"""
        model_info = self.get_model_info(model_code)
        if not model_info:
            return False
        
        if not os.path.exists(model_info.model_path):
            return False
        
        # 如果有哈希值，验证文件完整性
        if model_info.file_hash:
            actual_hash = self._calculate_file_hash(model_info.model_path)
            return actual_hash == model_info.file_hash
        
        return True
    
    def get_analysis_type_models(self, analysis_type: int) -> List[ModelInfo]:
        """根据分析类型获取模型列表"""
        return [model for model in self._model_registry.values() 
                if model.analysis_type == analysis_type and self.is_model_available(model.code)]
    
    def clear_cache(self):
        """清理模型缓存"""
        self._loaded_models.clear()
        self.logger.info("🧹 模型缓存已清理") 