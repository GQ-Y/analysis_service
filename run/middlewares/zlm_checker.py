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

# 使用新的日志记录器
from shared.utils.logger import get_normal_logger, get_exception_logger

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)

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
        normal_logger.info("正在检查ZLMediaKit环境...")

        # 检查系统类型
        if not ZLMChecker._check_system():
            normal_logger.warning("不支持的系统类型，跳过ZLMediaKit安装")
            return False

        # 检查库文件是否已安装
        lib_path = DEFAULT_LIB_PATH
        # 根据平台选择正确的库文件名
        lib_name = "libmk_api.dylib" if platform.system().lower() == "darwin" else "libmk_api.so"
        lib_file = os.path.join(lib_path, lib_name)
        
        if os.path.exists(lib_file):
            normal_logger.info(f"ZLMediaKit库文件已存在: {lib_file}")
            # 设置环境变量
            ZLMChecker._set_env_now(lib_path)
        else:
            # 安装库文件
            normal_logger.info("ZLMediaKit库文件不存在，开始自动安装...")
            if not ZLMChecker._install_zlm():
                exception_logger.error("安装ZLMediaKit库文件失败")
                return False

        # 库文件检查成功
        return True

    @staticmethod
    def _check_system():
        """检查系统类型

        Returns:
            bool: 系统是否支持
        """
        system = platform.system().lower()
        # 同时支持 macOS (darwin) 和 Linux
        if system not in ["darwin", "linux"]:
            normal_logger.warning(f"当前系统 ({system}) 可能不受官方支持，但将尝试继续。官方支持macOS和Linux。")
            # 仍然返回True，尝试让其在其他类Unix系统上工作，但给出警告
            # 如果确实只想支持darwin, 则取消下一行注释并修改上一行日志
            # return False 
        return True

    @staticmethod
    def _install_zlm():
        """安装ZLMediaKit库

        Returns:
            bool: 安装是否成功
        """
        try:
            lib_path = DEFAULT_LIB_PATH
            os.makedirs(lib_path, exist_ok=True)
            
            # 根据平台选择正确的库文件名和预编译路径
            system = platform.system().lower()
            if system == "darwin":
                lib_name = "libmk_api.dylib"
                local_prebuilt_dir = "zlmos/darwin"
            elif system == "linux":
                lib_name = "libmk_api.so"
                local_prebuilt_dir = "zlmos/linux" # 假设Linux预编译库在这个路径
                if not os.path.exists(local_prebuilt_dir):
                    # 如果Linux预编译库不存在，可以尝试从ZLMediaKit/release目录寻找
                    # 或者提示用户手动编译和放置
                    normal_logger.warning(f"Linux预编译库目录 {local_prebuilt_dir} 不存在，请确保已编译或下载ZLMediaKit库。")
                    # 此处可以添加更复杂的逻辑来查找或下载库
                    # 为简化，我们假设如果zlmos/linux不存在，则安装失败
                    # return False 
            else:
                exception_logger.error(f"不支持的操作系统: {system}，无法确定库文件名和路径。")
                return False

            dest_file = os.path.join(lib_path, lib_name)
            if os.path.exists(dest_file):
                normal_logger.info(f"库文件已存在: {dest_file}")
                os.chmod(dest_file, 0o755)
                normal_logger.info(f"已设置库文件权限为755")
                ZLMChecker._set_env_now(lib_path)
                return True
                
            local_lib_file_path = os.path.join(PROJECT_ROOT, local_prebuilt_dir, lib_name)
            if not os.path.exists(local_lib_file_path):
                exception_logger.error(f"找不到预编译库文件: {local_lib_file_path}")
                # 尝试从 ZLMediaKit/release/linux/libmk_api.so 寻找 (如果适用)
                if system == "linux":
                    alternative_path = PROJECT_ROOT / "ZLMediaKit" / "release" / "linux" / lib_name
                    if alternative_path.exists():
                        local_lib_file_path = str(alternative_path)
                        normal_logger.info(f"在备用路径找到Linux库: {local_lib_file_path}")
                    else:
                        exception_logger.error(f"备用路径 {alternative_path} 也未找到Linux库。")
                        return False
                else:
                     return False # Darwin下如果zlmos/darwin没有，则失败

            normal_logger.info(f"正在复制 {local_lib_file_path} 到 {dest_file}")
            shutil.copy2(local_lib_file_path, dest_file)

            os.chmod(dest_file, 0o755)
            normal_logger.info(f"已设置库文件权限为755")

            ZLMChecker._copy_config_files(local_prebuilt_dir)
            ZLMChecker._update_env_file(lib_path)
            ZLMChecker._set_env_now(lib_path)

            normal_logger.info("ZLMediaKit库安装成功")
            return True
        except Exception as e:
            exception_logger.exception(f"安装ZLMediaKit库时出错")
            return False

    @staticmethod
    def _copy_config_files(prebuilt_dir: str):
        """复制配置文件"""
        os.makedirs("config/zlm", exist_ok=True)
        config_filename = "config.ini"
        config_path_source = os.path.join(PROJECT_ROOT, prebuilt_dir, config_filename)
        
        if os.path.exists(config_path_source):
            dest_file = os.path.join("config", "zlm", config_filename)
            try:
                normal_logger.info(f"正在复制配置文件 {config_path_source} 到 {dest_file}")
                shutil.copy2(config_path_source, dest_file)
            except Exception as e:
                exception_logger.warning(f"复制配置文件时出错: {str(e)}")
        else:
            normal_logger.warning(f"源配置文件 {config_path_source} 不存在，跳过复制。")

    @staticmethod
    def _update_env_file(lib_path):
        """更新.env文件"""
        env_file = PROJECT_ROOT / ".env"
        new_lines = []
        zlm_lib_path_key = "STREAMING__ZLM_LIB_PATH"
        use_zlm_key = "STREAMING__USE_ZLMEDIAKIT"
        lib_path_found = False
        use_zlm_found = False

        if env_file.exists():
            with open(env_file, "r") as f:
                lines = f.readlines()
            
            for line in lines:
                stripped_line = line.strip()
                if stripped_line.startswith(f"{zlm_lib_path_key}="):
                    new_lines.append(f"{zlm_lib_path_key}={lib_path}\n")
                    lib_path_found = True
                elif stripped_line.startswith(f"{use_zlm_key}="):
                    new_lines.append(f"{use_zlm_key}=true\n")
                    use_zlm_found = True
                else:
                    new_lines.append(line)
        else:
            normal_logger.warning("未找到.env文件，将创建新文件并添加ZLMediaKit配置")
            new_lines.append("# ZLMediaKit配置\n")

        if not lib_path_found:
            new_lines.append(f"{zlm_lib_path_key}={lib_path}\n")
        if not use_zlm_found:
            new_lines.append(f"{use_zlm_key}=true\n")
        
        try:
            with open(env_file, "w") as f:
                f.writelines(new_lines)
            normal_logger.info(f"已更新/创建.env文件并设置ZLMediaKit配置: {env_file}")
        except Exception as e:
            exception_logger.error(f"更新.env文件时出错: {str(e)}")

    @staticmethod
    def _set_env_now(lib_path):
        """立即设置环境变量（仅对当前会话有效）"""
        os.environ["ZLM_LIB_PATH"] = lib_path
        env_var_name = "DYLD_LIBRARY_PATH" if platform.system().lower() == "darwin" else "LD_LIBRARY_PATH"
        current_env_path = os.environ.get(env_var_name, "")
        if lib_path not in current_env_path.split(os.pathsep):
            os.environ[env_var_name] = f"{lib_path}{os.pathsep}{current_env_path}"
        
        normal_logger.info(f"已设置当前会话环境变量 {env_var_name}，包含路径: {lib_path}")
        normal_logger.info(f"已设置当前会话环境变量 ZLM_LIB_PATH={lib_path}")
        # 移除print语句，使用日志
        # print(f"设置ZLMediaKit库路径: {lib_path}") 

    @staticmethod
    def check_zlm_process():
        """检查ZLMediaKit进程是否在运行"""
        try:
            process_name = "MediaServer"
            # pgrep对于跨平台可能不是最佳选择，但对于macOS和Linux通常可用
            # 对于Windows，可能需要 tasklist | findstr MediaServer.exe
            if platform.system().lower() in ["darwin", "linux"]:
                result = subprocess.run(["pgrep", "-f", process_name], 
                                        capture_output=True, text=True)
                if result.returncode == 0 and result.stdout.strip():
                    pids = result.stdout.strip().split("\n")
                    normal_logger.info(f"检测到ZLMediaKit ({process_name})进程正在运行，PIDs: {', '.join(pids)}")
                    return True
                else:
                    normal_logger.warning(f"未检测到ZLMediaKit ({process_name})进程。pgrep退出码: {result.returncode}, 输出: {result.stdout.strip()}, 错误: {result.stderr.strip()}")
                    return False
            else:
                normal_logger.warning(f"对于操作系统 {platform.system()}，检查ZLMediaKit进程的逻辑未实现。")
                return False # 假设未运行
        except FileNotFoundError:
            exception_logger.error("检查ZLMediaKit进程时出错: pgrep命令未找到。请确保pgrep已安装。")
            return False
        except Exception as e:
            exception_logger.exception("检查ZLMediaKit进程时发生未知错误")
            return False

    @staticmethod
    def stop_zlm_service():
        """停止ZLMediaKit服务

        Returns:
            bool: 是否成功停止服务
        """
        try:
            # 先检查服务是否在运行
            is_running = ZLMChecker.check_zlm_process()
            if not is_running:
                normal_logger.info("ZLMediaKit服务未运行，无需停止")
                return True

            system_type = platform.system().lower()
            if system_type in ["darwin", "linux"]:
                normal_logger.info("尝试停止ZLMediaKit服务...")
                # 使用killall命令
                try:
                    subprocess.run(["killall", "MediaServer"], 
                                   capture_output=True, 
                                   check=True)
                    normal_logger.info("ZLMediaKit服务已停止")
                    
                    # 等待进程完全退出
                    time.sleep(1)
                    
                    # 验证服务已停止，但不再重复记录日志
                    if not ZLMChecker._check_zlm_process_silent():
                        normal_logger.info("确认ZLMediaKit服务已停止")
                        return True
                    else:
                        normal_logger.warning("ZLMediaKit服务可能未完全停止，尝试强制终止")
                        subprocess.run(["killall", "-9", "MediaServer"], 
                                       capture_output=True)
                        time.sleep(1)
                        return not ZLMChecker._check_zlm_process_silent()
                except subprocess.CalledProcessError:
                    normal_logger.warning("killall命令执行失败，尝试使用pkill")
                    
                    # 使用pkill命令
                    try:
                        subprocess.run(["pkill", "-f", "MediaServer"], 
                                       capture_output=True)
                        time.sleep(1)
                        if not ZLMChecker._check_zlm_process_silent():
                            normal_logger.info("确认ZLMediaKit服务已停止")
                            return True
                        else:
                            normal_logger.warning("ZLMediaKit服务未停止，尝试强制终止")
                            subprocess.run(["pkill", "-9", "-f", "MediaServer"], 
                                           capture_output=True)
                            time.sleep(1)
                            return not ZLMChecker._check_zlm_process_silent()
                    except Exception as e:
                        exception_logger.error(f"使用pkill停止ZLMediaKit服务失败: {str(e)}")
                        return False
            else:
                normal_logger.warning(f"对于操作系统 {system_type}，停止ZLMediaKit服务的逻辑未实现")
                return False
        except Exception as e:
            exception_logger.exception(f"停止ZLMediaKit服务时出错: {str(e)}")
            return False
            
    @staticmethod
    def _check_zlm_process_silent():
        """检查ZLMediaKit进程是否在运行（静默版本，不记录日志）"""
        try:
            process_name = "MediaServer"
            if platform.system().lower() in ["darwin", "linux"]:
                result = subprocess.run(["pgrep", "-f", process_name], 
                                        capture_output=True, text=True)
                return result.returncode == 0 and result.stdout.strip()
            else:
                return False # 假设未运行
        except:
            return False

    @staticmethod
    def start_zlm_service():
        """尝试启动ZLMediaKit服务
            
        Returns:
            bool: 是否成功启动服务
        """
        try:
            # 先尝试杀掉可能存在的进程，不管是否存在都尝试杀掉
            system_type = platform.system().lower()
            if system_type in ["darwin", "linux"]:
                try:
                    # 静默执行，不记录日志
                    subprocess.run(["killall", "-9", "MediaServer"], 
                                  capture_output=True,
                                  check=False)  # 不检查返回码，忽略错误
                    time.sleep(0.5)  # 简短等待，确保进程已关闭
                except:
                    pass  # 忽略任何错误

            # 准备启动服务
            media_server_executable = "MediaServer"
            prebuilt_dir_segment = f"zlmos/{system_type}" 
            media_server_path_candidate1 = PROJECT_ROOT / prebuilt_dir_segment / media_server_executable
            # 备用路径，例如直接在ZLMediaKit编译产物中
            media_server_path_candidate2 = PROJECT_ROOT / "ZLMediaKit" / "release" / system_type / media_server_executable 

            media_server_path = None
            if media_server_path_candidate1.exists():
                media_server_path = media_server_path_candidate1
            elif media_server_path_candidate2.exists():
                 normal_logger.info(f"在主预编译路径 {media_server_path_candidate1} 未找到MediaServer，但在备用路径 {media_server_path_candidate2} 找到。")
                 media_server_path = media_server_path_candidate2
            else:
                exception_logger.error(f"找不到ZLMediaKit可执行文件 ({media_server_executable}) 在尝试的路径: {media_server_path_candidate1} 或 {media_server_path_candidate2}")
                return False

            config_dir = PROJECT_ROOT / "config" / "zlm"
            os.makedirs(config_dir, exist_ok=True)
            config_file_path = config_dir / "config.ini"

            # 尝试从预编译目录复制config.ini (如果它不存在的话)
            if not config_file_path.exists():
                source_config_path = PROJECT_ROOT / prebuilt_dir_segment / "config.ini"
                if source_config_path.exists():
                    try:
                        shutil.copy2(source_config_path, config_file_path)
                        normal_logger.info(f"已从 {source_config_path} 复制配置文件到 {config_file_path}")
                    except Exception as e:
                        exception_logger.warning(f"复制配置文件 {source_config_path} 失败: {str(e)}")
                else:
                    normal_logger.warning(f"源配置文件 {source_config_path} 不存在，ZLMediaKit可能使用默认配置或无法启动。")
            
            # 确保配置文件中的HTTP端口设置正确
            try:
                import configparser
                config = configparser.ConfigParser(interpolation=None)
                config.read(config_file_path)
                
                # 检查HTTP端口是否设置为8088
                if 'http' in config and 'port' in config['http']:
                    port = config['http']['port']
                    if port != '8088':
                        normal_logger.warning(f"配置文件中HTTP端口不是8088，当前值: {port}，将修改为8088")
                        config['http']['port'] = '8088'
                        with open(config_file_path, 'w') as f:
                            config.write(f)
                        normal_logger.info("已修改配置文件中的HTTP端口为8088")
            except Exception as e:
                exception_logger.warning(f"检查/修改配置文件中的HTTP端口设置时出错: {str(e)}")
            
            normal_logger.info(f"正在启动ZLMediaKit服务: {media_server_path} -c {config_file_path}")
            
            # 确保MediaServer可执行
            if not os.access(media_server_path, os.X_OK):
                 os.chmod(media_server_path, 0o755)
                 normal_logger.info(f"已为 {media_server_path} 添加执行权限。")

            process = subprocess.Popen([str(media_server_path), "-c", str(config_file_path)],
                                     stdout=subprocess.PIPE, # 捕获输出以便调试
                                     stderr=subprocess.PIPE,
                                     start_new_session=True,
                                     cwd=media_server_path.parent) # 在MediaServer所在目录运行

            normal_logger.info(f"ZLMediaKit服务启动命令已发送，PID: {process.pid}")
            time.sleep(3) # 给服务一些启动时间

            # 检查服务是否成功启动
            if ZLMChecker.check_zlm_process():
                normal_logger.info("成功启动ZLMediaKit服务。")
                try:
                    ZLMChecker._update_zlm_config()
                except Exception as e:
                    exception_logger.warning(f"更新ZLMediaKit配置时出错: {str(e)}")
                return True
            else:
                # 读取启动过程中的输出，帮助诊断
                stdout, stderr = process.communicate(timeout=2) # 等待子进程结束或超时
                exception_logger.error(f"ZLMediaKit服务启动失败。进程退出码: {process.returncode}")
                if stdout:
                    exception_logger.error(f"ZLMediaKit启动输出 (stdout):\n{stdout.decode(errors='ignore')}")
                if stderr:
                    exception_logger.error(f"ZLMediaKit启动错误 (stderr):\n{stderr.decode(errors='ignore')}")
                return False
        except Exception as e:
            exception_logger.exception("启动ZLMediaKit服务时发生未知错误")
            return False

    @staticmethod
    def _update_zlm_config():
        """更新ZLMediaKit配置"""
        try:
            config_file = PROJECT_ROOT / "config" / "zlm" / "config.ini"
            if config_file.exists():
                import configparser
                config = configparser.ConfigParser(interpolation=None) # 关闭插值以避免 % 问题
                config.read(config_file)

                if 'api' in config and 'secret' in config['api']:
                    new_secret = config['api']['secret']
                    normal_logger.info(f"读取到ZLMediaKit API密钥: {new_secret}")

                    env_file = PROJECT_ROOT / ".env"
                    api_secret_key = "STREAMING__ZLM_API_SECRET"
                    lines = []
                    secret_found = False
                    if env_file.exists():
                        with open(env_file, "r") as f:
                            lines = f.readlines()
                        
                        for i, line in enumerate(lines):
                            if line.strip().startswith(f"{api_secret_key}="):
                                lines[i] = f"{api_secret_key}={new_secret}\n"
                                secret_found = True
                                break
                    
                    if not secret_found:
                        lines.append(f"\n{api_secret_key}={new_secret}\n")
                    
                    with open(env_file, "w") as f:
                        f.writelines(lines)
                    normal_logger.info("已更新.env文件中的ZLMediaKit API密钥")

                    os.environ["ZLM_API_SECRET"] = new_secret
                    normal_logger.info(f"已设置当前会话环境变量 ZLM_API_SECRET={new_secret}")
            else:
                normal_logger.warning(f"ZLMediaKit配置文件 {config_file} 不存在，无法更新API密钥。")
        except Exception as e:
            exception_logger.exception("更新ZLMediaKit配置时出错")
            # 不向上抛出，因为这不是关键失败

# 导出类
__all__ = ["ZLMChecker"]