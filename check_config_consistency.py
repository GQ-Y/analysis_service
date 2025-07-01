#!/usr/bin/env python3
"""
配置一致性检查脚本
检查ZLM API secret和端口配置的一致性
"""
import os
import re
import configparser
from typing import Dict, List, Tuple


def read_ini_file(file_path: str) -> Dict[str, str]:
    """读取INI文件中的配置"""
    result = {}
    try:
        # 直接读取文件内容，手动解析以避免configparser的%问题
        with open(file_path, 'r', encoding='utf-8') as f:
            current_section = None
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # 检查是否是section
                if line.startswith('[') and line.endswith(']'):
                    current_section = line[1:-1]
                    continue

                # 检查是否是key=value
                if '=' in line and current_section:
                    key, value = line.split('=', 1)
                    result[f"{current_section}.{key.strip()}"] = value.strip()

        return result
    except Exception as e:
        print(f"❌ 读取配置文件失败 {file_path}: {str(e)}")
        return {}


def extract_python_config(file_path: str, patterns: List[str]) -> Dict[str, str]:
    """从Python文件中提取配置值"""
    result = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        for pattern in patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if isinstance(match, tuple):
                    key, value = match
                    result[key] = value.strip('"\'')
                else:
                    result[pattern] = match.strip('"\'')
    except Exception as e:
        print(f"❌ 读取Python文件失败 {file_path}: {str(e)}")
    
    return result


def check_zlm_secret_consistency():
    """检查ZLM API secret的一致性"""
    print("🔐 检查ZLM API Secret配置一致性")
    print("=" * 50)
    
    secret_configs = {}
    
    # 检查主配置文件
    main_config = read_ini_file("config/zlm/config.ini")
    if "api.secret" in main_config:
        secret_configs["config/zlm/config.ini"] = main_config["api.secret"]
    
    # 检查各平台的ZLM配置文件
    zlm_configs = [
        "zlmos/darwin/config.ini",
        "zlmos/linux/config.ini", 
        "zlmos/windows/config.ini"
    ]
    
    for config_file in zlm_configs:
        if os.path.exists(config_file):
            config = read_ini_file(config_file)
            if "api.secret" in config:
                secret_configs[config_file] = config["api.secret"]
    
    # 检查Python代码中的默认值
    python_patterns = [
        r'zlm_api_secret.*?=.*?["\']([^"\']+)["\']',
        r'api_secret.*?=.*?["\']([^"\']+)["\']',
        r'ZLM_API_SECRET.*?["\']([^"\']+)["\']'
    ]
    
    python_files = [
        "core/config.py",
        "core/media_kit/zlm_config.py"
    ]
    
    for py_file in python_files:
        if os.path.exists(py_file):
            config = extract_python_config(py_file, python_patterns)
            for key, value in config.items():
                secret_configs[f"{py_file}:{key}"] = value
    
    # 分析一致性
    if not secret_configs:
        print("❌ 未找到任何secret配置")
        return False
    
    unique_secrets = set(secret_configs.values())
    
    if len(unique_secrets) == 1:
        secret = list(unique_secrets)[0]
        print(f"✅ 所有secret配置一致: {secret}")
        for file_path, file_secret in secret_configs.items():
            print(f"   📄 {file_path}: {file_secret}")
        return True
    else:
        print(f"❌ 发现 {len(unique_secrets)} 个不同的secret值:")
        for secret in unique_secrets:
            print(f"\n🔑 Secret: {secret}")
            for file_path, file_secret in secret_configs.items():
                if file_secret == secret:
                    print(f"   📄 {file_path}")
        return False


def check_zlm_port_consistency():
    """检查ZLM端口配置的一致性"""
    print("\n🔌 检查ZLM端口配置一致性")
    print("=" * 50)
    
    port_configs = {}
    
    # 检查主配置文件
    main_config = read_ini_file("config/zlm/config.ini")
    if "http.port" in main_config:
        port_configs["config/zlm/config.ini"] = main_config["http.port"]
    
    # 检查各平台的ZLM配置文件
    zlm_configs = [
        "zlmos/darwin/config.ini",
        "zlmos/linux/config.ini",
        "zlmos/windows/config.ini"
    ]
    
    for config_file in zlm_configs:
        if os.path.exists(config_file):
            config = read_ini_file(config_file)
            if "http.port" in config:
                port_configs[config_file] = config["http.port"]
    
    # 检查Python代码中的端口配置
    python_patterns = [
        r'zlm_http_port.*?=.*?(\d+)',
        r'zlm_api_port.*?=.*?(\d+)',
        r'ZLM_HTTP_PORT.*?(\d+)',
        r'ZLM_API_PORT.*?(\d+)'
    ]
    
    python_files = [
        "core/config.py",
        "core/media_kit/zlm_config.py",
        "core/media_kit/protocols/gb28181/config.py",
        "core/media_kit/protocols/webrtc/config.py"
    ]
    
    for py_file in python_files:
        if os.path.exists(py_file):
            config = extract_python_config(py_file, python_patterns)
            for key, value in config.items():
                port_configs[f"{py_file}:{key}"] = value
    
    # 分析一致性
    if not port_configs:
        print("❌ 未找到任何端口配置")
        return False
    
    unique_ports = set(port_configs.values())
    
    if len(unique_ports) == 1:
        port = list(unique_ports)[0]
        print(f"✅ 所有端口配置一致: {port}")
        for file_path, file_port in port_configs.items():
            print(f"   📄 {file_path}: {file_port}")
        return True
    else:
        print(f"❌ 发现 {len(unique_ports)} 个不同的端口值:")
        for port in unique_ports:
            print(f"\n🔌 端口: {port}")
            for file_path, file_port in port_configs.items():
                if file_port == port:
                    print(f"   📄 {file_path}")
        return False


def check_optimization_config():
    """检查优化配置是否可以正常导入"""
    print("\n⚙️ 检查优化配置模块")
    print("=" * 50)
    
    try:
        # 检查优化配置文件是否存在
        if not os.path.exists("core/config/optimization.py"):
            print("❌ 优化配置文件不存在: core/config/optimization.py")
            return False
        
        if not os.path.exists("core/config/__init__.py"):
            print("❌ 配置模块初始化文件不存在: core/config/__init__.py")
            return False
        
        # 尝试导入优化配置
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        from core.config_modules.optimization import optimization_config
        
        print("✅ 优化配置模块导入成功")
        
        # 显示配置摘要
        config_summary = optimization_config.get_config_summary()
        print(f"📊 性能模式: {config_summary['performance_mode']}")
        print(f"🔗 ZLM最大重试次数: {config_summary['zlm_config']['max_retries']}")
        print(f"🎞️ 帧缓冲大小: {config_summary['frame_config']['buffer_size']}")
        print(f"📋 最大并发任务: {config_summary['task_config']['max_concurrent']}")
        
        return True
        
    except ImportError as e:
        print(f"❌ 优化配置模块导入失败: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ 优化配置检查异常: {str(e)}")
        return False


def main():
    """主函数"""
    print("🔍 分析服务配置一致性检查")
    print("=" * 60)
    
    results = []
    
    # 检查secret一致性
    secret_ok = check_zlm_secret_consistency()
    results.append(("ZLM Secret配置", secret_ok))
    
    # 检查端口一致性
    port_ok = check_zlm_port_consistency()
    results.append(("ZLM端口配置", port_ok))
    
    # 检查优化配置
    optimization_ok = check_optimization_config()
    results.append(("优化配置模块", optimization_ok))
    
    # 显示总结
    print("\n📊 检查结果总结")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for check_name, success in results:
        emoji = "✅" if success else "❌"
        status = "通过" if success else "失败"
        print(f"{emoji} {check_name}: {status}")
        if success:
            passed += 1
    
    print(f"\n🎯 总体结果: {passed}/{total} 项检查通过")
    
    if passed == total:
        print("🎉 所有配置检查通过！")
        return 0
    else:
        print("⚠️ 部分配置检查失败，请修复后重试")
        return 1


if __name__ == "__main__":
    import sys
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️ 检查被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 检查执行异常: {str(e)}")
        sys.exit(1)
