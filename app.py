"""
分析服务入口
提供视觉分析和数据处理功能
"""
# 从run模块导入应用创建函数
from run.run import create_app, show_service_banner
from core.config import settings
import os

# 打印环境变量和配置信息
print("\n=== 环境变量检查 ===")
print(f"DEBUG_ENABLED (环境变量): {os.getenv('DEBUG_ENABLED')}")
print(f"DEBUG_ENABLED (settings): {settings.DEBUG_ENABLED}")
print(f"ENVIRONMENT (环境变量): {os.getenv('ENVIRONMENT')}")
print(f"ENVIRONMENT (settings): {settings.ENVIRONMENT}")
print(f"LOG_LEVEL (环境变量): {os.getenv('LOG_LEVEL')}")
print(f"LOG_LEVEL (settings): {settings.LOG_LEVEL}")
print("===================\n")



# 创建应用实例
app = create_app()

if __name__ == "__main__":
    # 从run模块导入main函数并执行
    from run.run import main
    main()