#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: check_startup.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 启动验证脚本

验证项目是否可以正常启动。

本文件是分析服务项目的一部分。
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_dependencies():
    """检查依赖包"""
    print("📦 检查依赖包...")
    
    required_packages = [
        'fastapi',
        'uvicorn',
        'pydantic',
        'dependency_injector'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (缺失)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ 缺失依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    print("✅ 所有依赖包检查通过")
    return True


def check_app_creation():
    """检查应用创建"""
    print("\n🚀 检查应用创建...")
    
    try:
        from main import create_app
        app = create_app()
        
        print(f"   ✅ 应用标题: {app.title}")
        print(f"   ✅ 应用版本: {app.version}")
        print(f"   ✅ 路由数量: {len(app.routes)}")
        
        return True
    except Exception as e:
        print(f"   ❌ 应用创建失败: {e}")
        return False


def check_storage_system():
    """检查存储系统"""
    print("\n💾 检查存储系统...")
    
    try:
        from app.services.storage_service import get_storage_service
        storage_service = get_storage_service()
        stats = storage_service.get_storage_stats()
        
        print(f"   ✅ 存储服务创建成功")
        print(f"   ✅ 存储类型: {len(stats)} 个")
        
        return True
    except Exception as e:
        print(f"   ❌ 存储系统检查失败: {e}")
        return False


def check_directories():
    """检查目录结构"""
    print("\n📁 检查目录结构...")
    
    required_dirs = [
        'app',
        'config',
        'storage',
        'resources',
        'public',
        'tests'
    ]
    
    missing_dirs = []
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"   ✅ {dir_name}/")
        else:
            print(f"   ❌ {dir_name}/ (缺失)")
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"\n❌ 缺失目录: {', '.join(missing_dirs)}")
        return False
    
    print("✅ 目录结构检查通过")
    return True


def main():
    """主检查函数"""
    print("🔍 分析服务启动验证")
    print("=" * 50)
    
    checks = [
        ("依赖包检查", check_dependencies),
        ("目录结构检查", check_directories),
        ("存储系统检查", check_storage_system),
        ("应用创建检查", check_app_creation),
    ]
    
    passed = 0
    total = len(checks)
    
    for name, check_func in checks:
        try:
            if check_func():
                passed += 1
        except Exception as e:
            print(f"❌ {name}失败: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 检查结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有检查通过！项目可以正常启动。")
        print("\n🚀 启动命令:")
        print("   python start.py          # 快速启动")
        print("   python main.py --reload  # 开发模式")
        print("   ./start.sh dev           # 脚本启动")
        return True
    else:
        print("❌ 部分检查失败，请修复后重试。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
