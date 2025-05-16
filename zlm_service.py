#!/usr/bin/env python3
"""
ZLMediaKit服务管理脚本
用于手动启动、停止和检查ZLMediaKit服务状态
"""
import os
import sys
import time
import signal
import subprocess
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

# 导入日志模块
try:
    from shared.utils.logger import setup_logger
    logger = setup_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

# 预编译库所在路径
LOCAL_PREBUILT_PATH = "zlmos/darwin"

def check_zlm_process():
    """检查ZLMediaKit进程是否在运行
    
    Returns:
        list: 进程ID列表，如果没有则为空列表
    """
    try:
        # 在macOS上使用pgrep命令检查进程
        process_name = "MediaServer"
        result = subprocess.run(["pgrep", process_name], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE)
        
        if result.returncode == 0:
            pids = result.stdout.decode().strip().split('\n')
            return pids
        else:
            return []
    except Exception as e:
        logger.error(f"检查ZLMediaKit进程时出错: {str(e)}")
        return []

def start_zlm_service(foreground=False):
    """启动ZLMediaKit服务
    
    Args:
        foreground: 是否在前台运行
        
    Returns:
        bool: 是否成功启动
    """
    try:
        # 检查是否已有进程在运行
        pids = check_zlm_process()
        if pids:
            logger.info(f"ZLMediaKit服务已在运行，进程ID: {', '.join(pids)}")
            return True
        
        # 检查MediaServer可执行文件是否存在
        media_server_path = os.path.join(LOCAL_PREBUILT_PATH, "MediaServer")
        if not os.path.exists(media_server_path):
            logger.error(f"找不到ZLMediaKit可执行文件: {media_server_path}")
            return False
        
        # 确保配置目录存在
        config_dir = os.path.join(ROOT_DIR, "config", "zlm")
        os.makedirs(config_dir, exist_ok=True)
        
        # 配置文件路径
        config_file_path = os.path.join(config_dir, "config.ini")
        logger.info(f"使用配置文件: {config_file_path}")
        
        # 复制配置文件
        config_path = os.path.join(LOCAL_PREBUILT_PATH, "config.ini")
        if os.path.exists(config_path):
            dest_file = config_file_path
            if not os.path.exists(dest_file):
                import shutil
                logger.info(f"正在复制配置文件 {config_path} 到 {dest_file}")
                shutil.copy2(config_path, dest_file)
        
        # 启动ZLMediaKit服务
        logger.info(f"正在启动ZLMediaKit服务: {media_server_path}")
        
        if foreground:
            # 前台运行，阻塞当前进程
            print(f"ZLMediaKit服务在前台运行，按Ctrl+C停止")
            subprocess.run([media_server_path, "-c", config_file_path])
            return True
        else:
            # 后台运行
            subprocess.Popen([media_server_path, "-c", config_file_path], 
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            start_new_session=True)
            
            # 等待几秒让进程启动
            time.sleep(2)
            
            # 再次检查进程是否在运行
            pids = check_zlm_process()
            if pids:
                logger.info(f"成功启动ZLMediaKit服务，进程ID: {', '.join(pids)}")
                return True
            else:
                logger.error("ZLMediaKit服务启动失败")
                return False
                
    except Exception as e:
        logger.error(f"启动ZLMediaKit服务时出错: {str(e)}")
        return False

def stop_zlm_service():
    """停止ZLMediaKit服务
    
    Returns:
        bool: 是否成功停止
    """
    try:
        pids = check_zlm_process()
        if not pids:
            logger.info("ZLMediaKit服务未运行")
            return True
        
        # 停止所有服务进程
        for pid in pids:
            try:
                pid = int(pid.strip())
                logger.info(f"正在停止ZLMediaKit进程，PID: {pid}")
                os.kill(pid, signal.SIGTERM)  # 发送终止信号
            except Exception as e:
                logger.error(f"停止进程 {pid} 时出错: {str(e)}")
        
        # 等待进程结束
        time.sleep(2)
        
        # 检查是否已停止
        pids = check_zlm_process()
        if not pids:
            logger.info("所有ZLMediaKit服务已停止")
            return True
        else:
            logger.warning(f"部分ZLMediaKit进程仍在运行: {', '.join(pids)}")
            
            # 尝试强制结束进程
            for pid in pids:
                try:
                    pid = int(pid.strip())
                    logger.info(f"正在强制停止ZLMediaKit进程，PID: {pid}")
                    os.kill(pid, signal.SIGKILL)  # 发送强制终止信号
                except Exception as e:
                    logger.error(f"强制停止进程 {pid} 时出错: {str(e)}")
            
            return False
    except Exception as e:
        logger.error(f"停止ZLMediaKit服务时出错: {str(e)}")
        return False

def status_zlm_service():
    """检查ZLMediaKit服务状态
    
    Returns:
        bool: 服务是否在运行
    """
    pids = check_zlm_process()
    if pids:
        print(f"ZLMediaKit服务正在运行，进程ID: {', '.join(pids)}")
        return True
    else:
        print("ZLMediaKit服务未运行")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="ZLMediaKit服务管理工具")
    
    # 动作参数组
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--start", action="store_true", help="启动ZLMediaKit服务")
    group.add_argument("--stop", action="store_true", help="停止ZLMediaKit服务")
    group.add_argument("--restart", action="store_true", help="重启ZLMediaKit服务")
    group.add_argument("--status", action="store_true", help="查看ZLMediaKit服务状态")
    
    # 其他选项
    parser.add_argument("--foreground", action="store_true", help="在前台运行服务（仅用于--start）")
    
    args = parser.parse_args()
    
    if args.start:
        # 启动服务
        if start_zlm_service(args.foreground):
            print("ZLMediaKit服务已启动")
        else:
            print("启动ZLMediaKit服务失败")
            sys.exit(1)
    
    elif args.stop:
        # 停止服务
        if stop_zlm_service():
            print("ZLMediaKit服务已停止")
        else:
            print("停止ZLMediaKit服务失败")
            sys.exit(1)
    
    elif args.restart:
        # 重启服务
        print("正在重启ZLMediaKit服务...")
        stop_zlm_service()
        time.sleep(1)
        if start_zlm_service(args.foreground):
            print("ZLMediaKit服务已重启")
        else:
            print("重启ZLMediaKit服务失败")
            sys.exit(1)
    
    elif args.status:
        # 查看状态
        status_zlm_service()

if __name__ == "__main__":
    main() 