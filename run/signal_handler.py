"""
信号处理模块
用于处理进程退出时的清理工作，避免 ZLMediaKit 库析构错误
"""
import signal
import sys
import os
import time
import threading
from shared.utils.logger import get_normal_logger

normal_logger = get_normal_logger(__name__)

class SignalHandler:
    """信号处理器类"""
    
    def __init__(self):
        self.shutdown_event = threading.Event()
        self.zlm_manager = None
        self._original_handlers = {}
        
    def setup_signal_handlers(self):
        """设置信号处理器"""
        try:
            # 保存原始信号处理器
            self._original_handlers[signal.SIGTERM] = signal.signal(signal.SIGTERM, self._signal_handler)
            self._original_handlers[signal.SIGINT] = signal.signal(signal.SIGINT, self._signal_handler)
            
            # 在 Unix 系统上设置额外的信号处理器
            if hasattr(signal, 'SIGHUP'):
                self._original_handlers[signal.SIGHUP] = signal.signal(signal.SIGHUP, self._signal_handler)
            if hasattr(signal, 'SIGQUIT'):
                self._original_handlers[signal.SIGQUIT] = signal.signal(signal.SIGQUIT, self._signal_handler)
                
            normal_logger.info("信号处理器已设置")
        except Exception as e:
            normal_logger.error(f"设置信号处理器失败: {str(e)}")
    
    def set_zlm_manager(self, zlm_manager):
        """设置 ZLMediaKit 管理器引用"""
        self.zlm_manager = zlm_manager
    
    def _signal_handler(self, signum, frame):
        """信号处理函数"""
        normal_logger.info(f"接收到信号 {signum}，开始优雅关闭...")
        
        # 设置关闭事件
        self.shutdown_event.set()
        
        # 如果有 ZLMediaKit 管理器，立即标记为关闭状态
        if self.zlm_manager:
            try:
                self.zlm_manager._is_shutting_down = True
                self.zlm_manager._is_running = False
                
                # 取消注册回调函数
                if hasattr(self.zlm_manager, '_lib') and self.zlm_manager._lib:
                    try:
                        if hasattr(self.zlm_manager._lib, 'mk_set_log_callback'):
                            self.zlm_manager._lib.mk_set_log_callback(None)
                        if hasattr(self.zlm_manager._lib, 'mk_set_event_callback'):
                            self.zlm_manager._lib.mk_set_event_callback(None, None)
                    except:
                        pass
                        
                normal_logger.info("ZLMediaKit 管理器已标记为关闭状态")
            except Exception as e:
                normal_logger.error(f"处理 ZLMediaKit 管理器时出错: {str(e)}")
        
        # 尝试停止ZLMediaKit服务进程
        try:
            from run.middlewares import stop_zlm_service
            normal_logger.info("信号处理器正在停止ZLMediaKit服务进程...")
            # 直接调用停止函数，避免再次检查
            stop_result = stop_zlm_service()
            if stop_result:
                normal_logger.info("ZLMediaKit服务进程已被信号处理器成功停止")
            else:
                normal_logger.warning("ZLMediaKit服务进程无法被信号处理器停止")
        except Exception as e:
            normal_logger.error(f"信号处理器停止ZLMediaKit服务时出错: {str(e)}")
        
        # 给一些时间让正常的关闭流程执行
        time.sleep(1.0)
        
        # 强制垃圾回收
        try:
            import gc
            gc.collect()
        except:
            pass
        
        # 如果是 SIGTERM 或 SIGINT，让正常的关闭流程处理
        if signum in (signal.SIGTERM, signal.SIGINT):
            # 恢复原始处理器并重新发送信号
            if signum in self._original_handlers:
                signal.signal(signum, self._original_handlers[signum])
                os.kill(os.getpid(), signum)
        else:
            # 对于其他信号，直接退出
            normal_logger.info("强制退出进程")
            os._exit(0)
    
    def cleanup(self):
        """清理信号处理器"""
        try:
            # 恢复原始信号处理器
            for sig, handler in self._original_handlers.items():
                signal.signal(sig, handler)
            normal_logger.info("信号处理器已清理")
        except Exception as e:
            normal_logger.error(f"清理信号处理器失败: {str(e)}")

# 全局信号处理器实例
signal_handler = SignalHandler() 