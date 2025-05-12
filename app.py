"""
分析服务入口
提供视觉分析和数据处理功能
"""
# 从run模块导入应用创建函数
from run.run import create_app

# 创建应用实例
app = create_app()

if __name__ == "__main__":
    # 从run模块导入main函数并执行
    from run.run import main
    main()