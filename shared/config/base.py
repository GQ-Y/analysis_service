from pydantic_settings import BaseSettings
from typing import Dict, Any, Optional
import os
import yaml

class ServiceConfig(BaseSettings):
    """服务配置基类"""
    
    # 基础信息
    PROJECT_NAME: str = "MeekYolo Service"
    VERSION: str = "1.0.0"
    
    @classmethod
    def load_config(cls, service_name: Optional[str] = None) -> "ServiceConfig":
        """加载配置"""
        print(f"Starting to load config for service: {service_name}")
        
        # 加载配置
        config = {}
        
        # 1. 加载公共配置
        global_config_path = os.getenv("CONFIG_PATH", "config/config.yaml")
        print(f"Trying to load global config from: {global_config_path}")
        if os.path.exists(global_config_path):
            print(f"Global config file exists")
            with open(global_config_path, "r", encoding="utf-8") as f:
                config.update(yaml.safe_load(f))
                print(f"Loaded global config: {config}")
        
        # 2. 加载服务配置
        if service_name:
            service_config_path = f"{service_name}/config/config.yaml"
            print(f"Trying to load service config from: {service_config_path}")
            if os.path.exists(service_config_path):
                print(f"Service config file exists")
                with open(service_config_path, "r", encoding="utf-8") as f:
                    # 服务配置优先级更高
                    service_config = yaml.safe_load(f)
                    print(f"Loaded service config: {service_config}")
                    config.update(service_config)
        
        # 3. 环境变量覆盖
        print("Checking environment variables")
        if "SERVICES" in config:
            for service, service_config in config["SERVICES"].items():
                env_var = f"{service.upper()}_SERVICE_URL"
                if env_var in os.environ:
                    service_config["url"] = os.environ[env_var]
                    print(f"Updated service URL from env: {env_var}={os.environ[env_var]}")
        
        print(f"Final config: {config}")
        try:
            instance = cls(**config)
            print("Successfully created config instance")
            return instance
        except Exception as e:
            print(f"Failed to create config instance: {str(e)}")
            raise 