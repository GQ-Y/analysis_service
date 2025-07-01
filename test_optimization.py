#!/usr/bin/env python3
"""
优化测试脚本
验证优化配置是否正常工作
"""
import asyncio
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config_modules.optimization import optimization_config
from core.health.checker import (
    health_manager, 
    ZLMHealthChecker, 
    SystemHealthChecker,
    HealthStatus
)
from core.media_kit.zlm_manager import ZLMManager
from shared.utils.logger import get_normal_logger

normal_logger = get_normal_logger(__name__)


async def test_optimization_config():
    """测试优化配置"""
    print("=" * 60)
    print("🔧 优化配置测试")
    print("=" * 60)
    
    # 显示配置摘要
    config_summary = optimization_config.get_config_summary()
    
    print(f"📊 性能模式: {config_summary['performance_mode']}")
    print()
    
    print("🔗 ZLM配置:")
    for key, value in config_summary['zlm_config'].items():
        print(f"  - {key}: {value}")
    print()
    
    print("🎞️ 帧处理配置:")
    for key, value in config_summary['frame_config'].items():
        print(f"  - {key}: {value}")
    print()
    
    print("📋 任务管理配置:")
    for key, value in config_summary['task_config'].items():
        print(f"  - {key}: {value}")
    print()
    
    print("💾 数据库配置:")
    for key, value in config_summary['database_config'].items():
        print(f"  - {key}: {value}")
    print()
    
    print("🔴 Redis配置:")
    for key, value in config_summary['redis_config'].items():
        print(f"  - {key}: {value}")
    print()


async def test_zlm_connection():
    """测试ZLM连接"""
    print("=" * 60)
    print("🔗 ZLM连接测试")
    print("=" * 60)
    
    try:
        # 创建ZLM管理器实例
        zlm_manager = ZLMManager()
        
        # 测试API连接
        print("📡 测试ZLM API连接...")
        result = zlm_manager.call_api("getApiList")
        
        if result.get("code") == 0:
            api_count = len(result.get("data", []))
            print(f"✅ ZLM API连接成功，可用API数量: {api_count}")
        else:
            print(f"❌ ZLM API连接失败，错误码: {result.get('code')}")
            return False
        
        # 测试服务器配置获取
        print("⚙️ 获取ZLM服务器配置...")
        config_result = zlm_manager.call_api("getServerConfig")
        
        if config_result.get("code") == 0:
            server_data = config_result.get("data", {})
            print(f"✅ 服务器配置获取成功")
            print(f"  - 版本: {server_data.get('version', 'unknown')}")
            print(f"  - 运行时间: {server_data.get('uptime', 0)}秒")
        else:
            print(f"❌ 服务器配置获取失败")
        
        return True
        
    except Exception as e:
        print(f"❌ ZLM连接测试失败: {str(e)}")
        return False


async def test_health_check():
    """测试健康检查系统"""
    print("=" * 60)
    print("🏥 健康检查系统测试")
    print("=" * 60)
    
    try:
        # 注册系统健康检查器
        system_checker = SystemHealthChecker()
        health_manager.register_checker(system_checker)
        
        # 如果ZLM连接正常，注册ZLM健康检查器
        try:
            zlm_manager = ZLMManager()
            zlm_checker = ZLMHealthChecker(zlm_manager)
            health_manager.register_checker(zlm_checker)
            print("✅ ZLM健康检查器已注册")
        except Exception as e:
            print(f"⚠️ ZLM健康检查器注册失败: {str(e)}")
        
        # 执行健康检查
        print("🔍 执行健康检查...")
        results = await health_manager.check_all()
        
        for component, result in results.items():
            status_emoji = {
                HealthStatus.HEALTHY: "✅",
                HealthStatus.DEGRADED: "⚠️",
                HealthStatus.UNHEALTHY: "❌",
                HealthStatus.UNKNOWN: "❓"
            }
            
            emoji = status_emoji.get(result.status, "❓")
            print(f"{emoji} {component}: {result.status.value}")
            print(f"   消息: {result.message}")
            print(f"   响应时间: {result.response_time:.3f}s")
            
            if result.details:
                print(f"   详情: {result.details}")
            print()
        
        # 获取整体状态
        overall_status = await health_manager.get_overall_status()
        overall_emoji = {
            HealthStatus.HEALTHY: "✅",
            HealthStatus.DEGRADED: "⚠️",
            HealthStatus.UNHEALTHY: "❌",
            HealthStatus.UNKNOWN: "❓"
        }
        
        print(f"🎯 整体健康状态: {overall_emoji.get(overall_status, '❓')} {overall_status.value}")
        
        return True
        
    except Exception as e:
        print(f"❌ 健康检查测试失败: {str(e)}")
        return False


async def test_retry_mechanism():
    """测试重试机制"""
    print("=" * 60)
    print("🔄 重试机制测试")
    print("=" * 60)
    
    from shared.utils.retry import exponential_backoff
    
    # 测试成功的情况
    @exponential_backoff(max_retries=3, base_delay=0.1)
    def successful_function():
        print("✅ 函数执行成功")
        return "success"
    
    # 测试失败重试的情况
    attempt_count = 0
    
    @exponential_backoff(max_retries=3, base_delay=0.1)
    def failing_function():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            print(f"❌ 第 {attempt_count} 次尝试失败")
            raise Exception(f"模拟失败 {attempt_count}")
        else:
            print(f"✅ 第 {attempt_count} 次尝试成功")
            return "success after retries"
    
    try:
        print("🧪 测试成功函数...")
        result1 = successful_function()
        print(f"结果: {result1}")
        print()
        
        print("🧪 测试重试函数...")
        result2 = failing_function()
        print(f"结果: {result2}")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ 重试机制测试失败: {str(e)}")
        return False


async def main():
    """主测试函数"""
    print("🚀 分析服务优化测试开始")
    print()
    
    test_results = []
    
    # 测试优化配置
    try:
        await test_optimization_config()
        test_results.append(("优化配置", True))
    except Exception as e:
        print(f"❌ 优化配置测试失败: {str(e)}")
        test_results.append(("优化配置", False))
    
    print()
    
    # 测试ZLM连接
    try:
        zlm_success = await test_zlm_connection()
        test_results.append(("ZLM连接", zlm_success))
    except Exception as e:
        print(f"❌ ZLM连接测试异常: {str(e)}")
        test_results.append(("ZLM连接", False))
    
    print()
    
    # 测试健康检查
    try:
        health_success = await test_health_check()
        test_results.append(("健康检查", health_success))
    except Exception as e:
        print(f"❌ 健康检查测试异常: {str(e)}")
        test_results.append(("健康检查", False))
    
    print()
    
    # 测试重试机制
    try:
        retry_success = await test_retry_mechanism()
        test_results.append(("重试机制", retry_success))
    except Exception as e:
        print(f"❌ 重试机制测试异常: {str(e)}")
        test_results.append(("重试机制", False))
    
    # 显示测试结果摘要
    print("=" * 60)
    print("📊 测试结果摘要")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, success in test_results:
        emoji = "✅" if success else "❌"
        status = "通过" if success else "失败"
        print(f"{emoji} {test_name}: {status}")
        if success:
            passed += 1
    
    print()
    print(f"🎯 总体结果: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 所有优化测试通过！")
        return 0
    else:
        print("⚠️ 部分测试失败，请检查配置")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️ 测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 测试执行异常: {str(e)}")
        sys.exit(1)
