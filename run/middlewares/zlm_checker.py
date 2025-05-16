"""
ZLMediaKit环境检测模块
在应用启动时自动检查ZLMediaKit库并安装
"""
import os
import sys
import shutil
import platform
import subprocess
import time
from pathlib import Path
from loguru import logger

from shared.utils.logger import setup_logger

logger = setup_logger(__name__)

# 预编译库所在路径
LOCAL_PREBUILT_PATH = "zlmos/darwin"

# 目标库路径 - 使用项目内的lib目录，避免需要管理员权限
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_LIB_PATH = str(PROJECT_ROOT / "lib" / "zlm")

class ZLMChecker:
    """ZLMediaKit环境检测类"""
    
    @staticmethod
    def check_and_install():
        """检查ZLMediaKit环境并自动安装
        
        Returns:
            bool: 安装是否成功
        """
        logger.info("正在检查ZLMediaKit环境...")
        
        # 检查系统类型
        if not ZLMChecker._check_system():
            logger.warning("不支持的系统类型，跳过ZLMediaKit安装")
            return False
            
        # 检查库文件是否已安装
        lib_path = DEFAULT_LIB_PATH
        lib_file = os.path.join(lib_path, "libmk_api.dylib")
        if os.path.exists(lib_file):
            logger.info(f"ZLMediaKit库文件已存在: {lib_file}")
            # 设置环境变量
            ZLMChecker._set_env_now(lib_path)
        else:
            # 安装库文件
            logger.info("ZLMediaKit库文件不存在，开始自动安装...")
            if not ZLMChecker._install_zlm():
                logger.error("安装ZLMediaKit库文件失败")
                return False
                
        # 检查并启动ZLMediaKit服务进程
        if not ZLMChecker.check_zlm_process():
            logger.info("ZLMediaKit服务未运行，尝试启动...")
            if ZLMChecker.start_zlm_service():
                logger.info("成功启动ZLMediaKit服务")
            else:
                logger.warning("无法启动ZLMediaKit服务，部分功能可能不可用")
                # 即使服务无法启动，库文件已经安装，所以仍然返回成功
        else:
            logger.info("ZLMediaKit服务已在运行")
            
        return True
    
    @staticmethod
    def _check_system():
        """检查系统类型
        
        Returns:
            bool: 系统是否支持
        """
        system = platform.system().lower()
        if system != "darwin":
            logger.warning(f"当前只支持macOS系统，当前系统为: {system}")
            return False
        
        return True
    
    @staticmethod
    def _install_zlm():
        """安装ZLMediaKit库
        
        Returns:
            bool: 安装是否成功
        """
        try:
            # 检查预编译库是否存在
            local_lib_path = os.path.join(LOCAL_PREBUILT_PATH, "libmk_api.dylib")
            if not os.path.exists(local_lib_path):
                logger.error(f"找不到预编译库文件: {local_lib_path}")
                return False
            
            # 创建目标目录
            lib_path = DEFAULT_LIB_PATH
            os.makedirs(lib_path, exist_ok=True)
            
            # 复制库文件
            dest_file = os.path.join(lib_path, "libmk_api.dylib")
            logger.info(f"正在复制 {local_lib_path} 到 {dest_file}")
            shutil.copy2(local_lib_path, dest_file)
            
            # 设置库文件权限
            os.chmod(dest_file, 0o755)
            logger.info(f"已设置库文件权限为755")
            
            # 复制配置文件
            ZLMChecker._copy_config_files()
            
            # 更新.env文件
            ZLMChecker._update_env_file(lib_path)
            
            # 设置环境变量
            ZLMChecker._set_env_now(lib_path)
            
            logger.info("ZLMediaKit库安装成功")
            return True
        except Exception as e:
            logger.error(f"安装ZLMediaKit库时出错: {str(e)}")
            return False
    
    @staticmethod
    def _copy_config_files():
        """复制配置文件"""
        # 创建配置目录
        os.makedirs("config/zlm", exist_ok=True)
        
        # 检查配置文件是否存在
        config_path = os.path.join(LOCAL_PREBUILT_PATH, "config.ini")
        if os.path.exists(config_path):
            dest_file = "config/zlm/config.ini"
            try:
                logger.info(f"正在复制配置文件 {config_path} 到 {dest_file}")
                shutil.copy2(config_path, dest_file)
            except Exception as e:
                logger.warning(f"复制配置文件时出错: {str(e)}")
    
    @staticmethod
    def _update_env_file(lib_path):
        """更新.env文件"""
        env_file = ".env"
        
        if not os.path.exists(env_file):
            # 创建新的.env文件
            logger.warning("未找到.env文件，将创建新文件")
            with open(env_file, "w") as f:
                f.write("# ZLMediaKit配置\n")
                f.write("STREAMING__USE_ZLMEDIAKIT=true\n")
                f.write(f"STREAMING__ZLM_LIB_PATH={lib_path}\n")
            logger.info(f"已创建.env文件并添加ZLMediaKit配置")
            return
        
        # 读取现有.env文件
        with open(env_file, "r") as f:
            content = f.read()
        
        # 检查是否需要更新配置
        updated = False
        
        if "STREAMING__USE_ZLMEDIAKIT=" in content:
            # 更新现有配置
            lines = content.split("\n")
            for i, line in enumerate(lines):
                if line.startswith("STREAMING__USE_ZLMEDIAKIT="):
                    lines[i] = "STREAMING__USE_ZLMEDIAKIT=true"
                    updated = True
                elif line.startswith("STREAMING__ZLM_LIB_PATH="):
                    lines[i] = f"STREAMING__ZLM_LIB_PATH={lib_path}"
                    updated = True
            
            if updated:
                # 写回更新后的内容
                with open(env_file, "w") as f:
                    f.write("\n".join(lines))
                logger.info("已更新.env文件中的ZLMediaKit配置")
                return
        
        # 添加新配置
        with open(env_file, "a") as f:
            f.write("\n# ZLMediaKit配置\n")
            f.write("STREAMING__USE_ZLMEDIAKIT=true\n")
            f.write(f"STREAMING__ZLM_LIB_PATH={lib_path}\n")
        logger.info("已向.env文件添加ZLMediaKit配置")
    
    @staticmethod
    def _set_env_now(lib_path):
        """立即设置环境变量（仅对当前会话有效）"""
        os.environ["ZLM_LIB_PATH"] = lib_path
        os.environ["LD_LIBRARY_PATH"] = f"{lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
        logger.info(f"已设置当前会话的环境变量 ZLM_LIB_PATH={lib_path}")

    @staticmethod
    def check_zlm_process():
        """检查ZLMediaKit进程是否在运行
        
        Returns:
            bool: 进程是否在运行
        """
        try:
            # 在macOS上使用pgrep命令检查进程
            process_name = "MediaServer"
            result = subprocess.run(["pgrep", process_name], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE)
            
            if result.returncode == 0:
                pid = result.stdout.decode().strip()
                logger.info(f"检测到ZLMediaKit进程正在运行，PID: {pid}")
                return True
            else:
                logger.warning("未检测到ZLMediaKit进程")
                return False
        except Exception as e:
            logger.error(f"检查ZLMediaKit进程时出错: {str(e)}")
            return False
    
    @staticmethod
    def start_zlm_service():
        """尝试启动ZLMediaKit服务
        
        Returns:
            bool: 是否成功启动
        """
        try:
            # 检查是否已有进程在运行
            if ZLMChecker.check_zlm_process():
                logger.info("ZLMediaKit服务已在运行")
                return True
            
            # 检查MediaServer可执行文件是否存在
            media_server_path = os.path.join(LOCAL_PREBUILT_PATH, "MediaServer")
            if not os.path.exists(media_server_path):
                logger.error(f"找不到ZLMediaKit可执行文件: {media_server_path}")
                return False
            
            # 确保配置目录存在
            os.makedirs("config/zlm", exist_ok=True)
            
            # 复制配置文件
            config_path = os.path.join(LOCAL_PREBUILT_PATH, "config.ini")
            if os.path.exists(config_path):
                dest_file = "config/zlm/config.ini"
                if not os.path.exists(dest_file):
                    logger.info(f"正在复制配置文件 {config_path} 到 {dest_file}")
                    shutil.copy2(config_path, dest_file)
            
            # 启动ZLMediaKit服务
            logger.info(f"正在启动ZLMediaKit服务: {media_server_path}")
            
            # 使用非阻塞方式启动服务，并传递配置文件路径
            config_file_path = os.path.abspath("config/zlm/config.ini")
            subprocess.Popen([media_server_path, "-c", config_file_path], 
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            start_new_session=True)
            
            # 等待几秒让进程启动
            time.sleep(3)
            
            # 再次检查进程是否在运行
            if ZLMChecker.check_zlm_process():
                logger.info("成功启动ZLMediaKit服务")
                
                # 读取ZLMediaKit可能已修改的配置
                try:
                    ZLMChecker._update_zlm_config()
                except Exception as e:
                    logger.warning(f"更新ZLMediaKit配置时出错: {str(e)}")
                
                return True
            else:
                logger.error("ZLMediaKit服务启动失败")
                return False
        except Exception as e:
            logger.error(f"启动ZLMediaKit服务时出错: {str(e)}")
            return False
            
    @staticmethod
    def _update_zlm_config():
        """更新ZLMediaKit配置
        
        ZLMediaKit在启动时可能会修改配置文件，比如生成新的secret
        此方法读取配置文件并更新环境变量
        """
        try:
            config_file = "config/zlm/config.ini"
            if os.path.exists(config_file):
                import configparser
                config = configparser.ConfigParser()
                config.read(config_file)
                
                if 'api' in config and 'secret' in config['api']:
                    new_secret = config['api']['secret']
                    logger.info(f"读取到ZLMediaKit API密钥: {new_secret}")
                    
                    # 更新.env文件中的密钥
                    env_file = ".env"
                    if os.path.exists(env_file):
                        with open(env_file, "r") as f:
                            content = f.read()
                        
                        # 检查是否需要更新密钥
                        if "STREAMING__ZLM_API_SECRET=" in content:
                            lines = content.split("\n")
                            for i, line in enumerate(lines):
                                if line.startswith("STREAMING__ZLM_API_SECRET="):
                                    lines[i] = f"STREAMING__ZLM_API_SECRET={new_secret}"
                            
                            # 写回更新后的内容
                            with open(env_file, "w") as f:
                                f.write("\n".join(lines))
                        else:
                            # 添加新配置
                            with open(env_file, "a") as f:
                                f.write(f"\nSTREAMING__ZLM_API_SECRET={new_secret}\n")
                        
                        logger.info("已更新.env文件中的ZLMediaKit API密钥")
                    
                    # 更新环境变量
                    os.environ["ZLM_API_SECRET"] = new_secret
                    logger.info(f"已设置环境变量 ZLM_API_SECRET={new_secret}")
        except Exception as e:
            logger.error(f"更新ZLMediaKit配置时出错: {str(e)}")
            raise

# 导出类
__all__ = ["ZLMChecker"] 