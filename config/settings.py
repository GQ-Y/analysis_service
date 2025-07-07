#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: settings.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 主配置文件

包含应用程序的主要配置设置，使用Pydantic Settings进行配置管理。

本文件是分析服务项目的一部分。
"""

import os
from enum import Enum
from typing import List, Optional
from pathlib import Path
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Environment(str, Enum):
    """环境枚举"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """应用设置"""

    # 基本信息
    APP_NAME: str = "Analysis Service"
    APP_VERSION: str = "2.0.0"
    APP_DESCRIPTION: str = "基于Python的智能视频分析服务"

    # 环境配置
    ENVIRONMENT: Environment = Environment.DEVELOPMENT
    DEBUG: bool = True

    # 服务器配置
    HOST: str = "0.0.0.0"
    PORT: int = 8002
    WORKERS: int = 1
    AUTO_RELOAD: bool = True

    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = None
    ACCESS_LOG: bool = True

    # 数据库配置
    DATABASE_URL: Optional[str] = None
    REDIS_URL: str = "redis://localhost:6379/0"

    # 安全配置
    SECRET_KEY: str = "your-secret-key-change-in-production"
    CORS_ORIGINS: List[str] = ["*"]

    # 目录配置
    BASE_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    STATIC_DIR: Optional[str] = None
    UPLOAD_DIR: Optional[str] = None
    LOG_DIR: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"  # 忽略额外的环境变量

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # 设置默认目录
        if not self.STATIC_DIR:
            self.STATIC_DIR = str(self.BASE_DIR / "static")

        if not self.UPLOAD_DIR:
            self.UPLOAD_DIR = str(self.BASE_DIR / "uploads")

        if not self.LOG_DIR:
            self.LOG_DIR = str(self.BASE_DIR / "storage" / "logs")

        # 确保目录存在
        for dir_path in [self.STATIC_DIR, self.UPLOAD_DIR, self.LOG_DIR]:
            if dir_path:
                Path(dir_path).mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """获取设置实例（缓存）"""
    return Settings()


# 兼容性：保留原有的配置变量
settings = get_settings()

# 项目根目录
BASE_DIR = settings.BASE_DIR

# 应用基本信息
APP_NAME = settings.APP_NAME
APP_VERSION = settings.APP_VERSION
APP_DESCRIPTION = settings.APP_DESCRIPTION

# 服务器配置
SERVER_CONFIG = {
    "host": settings.HOST,
    "port": settings.PORT,
    "workers": settings.WORKERS,
    "reload": settings.AUTO_RELOAD,
    "debug": settings.DEBUG,
}

# 数据库配置
DATABASE_CONFIG = {
    "redis": {
        "url": settings.REDIS_URL,
    }
}

# 日志配置
def get_logging_config():
    """获取日志配置（动态生成）"""
    settings = get_settings()
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
            },
            "detailed": {
                "format": "[%(asctime)s] %(levelname)s in %(module)s [%(pathname)s:%(lineno)d]: %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "default",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "detailed",
                "filename": str(Path(settings.LOG_DIR) / "app.log"),
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
            },
        },
        "loggers": {
            "": {
                "level": "DEBUG",
                "handlers": ["console", "file"],
            },
        },
    }

# 保持向后兼容性
LOGGING_CONFIG = get_logging_config()

# 缓存配置
CACHE_CONFIG = {
    "default": {
        "backend": "redis",
        "location": settings.REDIS_URL.replace("/0", "/1"),  # 使用数据库1作为缓存
        "timeout": 300,  # 5分钟
    }
}

# 安全配置
SECURITY_CONFIG = {
    "secret_key": os.getenv("SECRET_KEY", "your-secret-key-here"),
    "algorithm": "HS256",
    "access_token_expire_minutes": 30,
    "allowed_hosts": ["*"],
    "cors_origins": ["*"],
}

# 文件存储配置
STORAGE_CONFIG = {
    "base_path": BASE_DIR / "storage",
    "results_path": BASE_DIR / "storage" / "results",
    "uploads_path": BASE_DIR / "storage" / "uploads",
    "temp_path": BASE_DIR / "storage" / "temp",
    "max_file_size": 100 * 1024 * 1024,  # 100MB
}

# API配置
API_CONFIG = {
    "title": APP_NAME,
    "description": APP_DESCRIPTION,
    "version": APP_VERSION,
    "docs_url": "/docs",
    "redoc_url": "/redoc",
    "openapi_url": "/openapi.json",
}

# 环境配置
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
IS_DEVELOPMENT = ENVIRONMENT == "development"
IS_PRODUCTION = ENVIRONMENT == "production"
IS_TESTING = ENVIRONMENT == "testing"
