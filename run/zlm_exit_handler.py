"""
ZLMediaKit 退出处理器
专门处理 ZLMediaKit 库的退出问题，避免析构时的 recursive_mutex 错误
"""
import os
import sys
import atexit
import threading
import time
import ctypes
from shared.utils.logger import get_normal_logger

normal_logger = get_normal_logger(__name__)

class ZLMExitHandler:
    """ZLMediaKit 退出处理器"""
    
    def __init__(self):
        self.zlm_manager = None
        self.exit_registered = False
        self.force_exit = False
        self._exit_lock = threading.Lock()
        
    def register_zlm_manager(self, zlm_manager):
        """注册 ZLMediaKit 管理器"""
        self.zlm_manager = zlm_manager
        
        # 注册退出处理器（只注册一次）
        if not self.exit_registered:
            atexit.register(self._emergency_cleanup)
            self.exit_registered = True
            normal_logger.info("ZLMediaKit 紧急退出处理器已注册")
    
    def _emergency_cleanup(self):
        """紧急清理函数，在进程退出时调用"""
        with self._exit_lock:
            if self.force_exit:
                return
                
            self.force_exit = True
            
            try:
                normal_logger.info("执行 ZLMediaKit 紧急清理...")
                
                if self.zlm_manager:
                    # 立即标记为关闭状态
                    self.zlm_manager._is_shutting_down = True
                    self.zlm_manager._is_running = False
                    
                    # 取消注册所有回调
                    if hasattr(self.zlm_manager, '_lib') and self.zlm_manager._lib:
                        try:
                            # 取消日志回调
                            if hasattr(self.zlm_manager._lib, 'mk_set_log_callback'):
                                self.zlm_manager._lib.mk_set_log_callback(None)
                            
                            # 取消事件回调
                            if hasattr(self.zlm_manager._lib, 'mk_set_event_callback'):
                                self.zlm_manager._lib.mk_set_event_callback(None, None)
                            
                            # 停止所有服务器
                            if hasattr(self.zlm_manager._lib, 'mk_stop_all_server'):
                                self.zlm_manager._lib.mk_stop_all_server()
                                
                        except Exception as e:
                            # 静默处理异常
                            pass
                    
                    # 清理引用
                    self.zlm_manager._lib = None
                    self.zlm_manager._log_callback_ref = None
                    self.zlm_manager._event_callback_ref = None
                
                # 强制垃圾回收
                import gc
                gc.collect()
                
                normal_logger.info("ZLMediaKit 紧急清理完成")
                
            except Exception as e:
                # 完全静默处理
                pass
    
    def force_exit_process(self):
        """强制退出进程，避免析构错误"""
        try:
            normal_logger.info("强制退出进程以避免 ZLMediaKit 析构错误")
            
            # 执行紧急清理
            self._emergency_cleanup()
            
            # 给一点时间让日志输出
            time.sleep(0.1)
            
            # 强制退出，避免 C++ 析构函数执行
            os._exit(0)
            
        except Exception:
            # 如果连强制退出都失败了，使用最原始的方法
            try:
                os.kill(os.getpid(), 9)  # SIGKILL
            except:
                sys.exit(1)

# 全局退出处理器实例
zlm_exit_handler = ZLMExitHandler() 