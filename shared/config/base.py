from pydantic_settings import BaseSettings

class ServiceConfig(BaseSettings):
    """服务配置基类"""

    # 基础信息
    PROJECT_NAME: str = "N-MeekYolo Analysis Service"
    VERSION: str = "1.0.0"

    class Config:
        env_file = ".env"
        extra = "allow"  # 允许额外字段