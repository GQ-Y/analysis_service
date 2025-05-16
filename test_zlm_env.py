#!/usr/bin/env python3
"""
ZLMediaKit环境检测测试脚本
用于测试ZLMediaKit环境检测和自动安装功能
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

def main():
    """主函数"""
    print("=== ZLMediaKit环境检测测试 ===")
    
    try:
        # 配置日志
        from shared.utils.logger import setup_logger
        logger = setup_logger(__name__)
        
        # 导入ZLMChecker
        sys.path.insert(0, str(ROOT_DIR / "run" / "middlewares"))
        from zlm_checker import ZLMChecker
        
        # 检查并安装ZLMediaKit
        result = ZLMChecker.check_and_install()
        
        if result:
            print("\033[92m[成功]\033[0m ZLMediaKit环境检测和安装成功")
            print(f"环境变量 ZLM_LIB_PATH={os.environ.get('ZLM_LIB_PATH', '未设置')}")
            print(f"环境变量 LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '未设置')}")
            
            # 检查库文件是否存在
            lib_path = os.environ.get('ZLM_LIB_PATH', '/usr/local/lib')
            lib_file = os.path.join(lib_path, "libmk_api.dylib")
            if os.path.exists(lib_file):
                print(f"\033[92m[成功]\033[0m 库文件存在: {lib_file}")
            else:
                print(f"\033[91m[失败]\033[0m 库文件不存在: {lib_file}")
        else:
            print("\033[91m[失败]\033[0m ZLMediaKit环境检测和安装失败")
        
        # 检查.env文件
        if os.path.exists(".env"):
            with open(".env", "r") as f:
                content = f.read()
            if "STREAMING__USE_ZLMEDIAKIT=true" in content:
                print("\033[92m[成功]\033[0m .env文件包含ZLMediaKit配置")
            else:
                print("\033[91m[失败]\033[0m .env文件不包含ZLMediaKit配置")
        else:
            print("\033[91m[失败]\033[0m .env文件不存在")
    
    except Exception as e:
        print(f"\033[91m[错误]\033[0m 测试过程中出错: {str(e)}")
    
    print("=== 测试完成 ===")

if __name__ == "__main__":
    main() 