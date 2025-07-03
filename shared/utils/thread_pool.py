from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from threading import Lock


class GlobalThreadPool:
    """全局线程池管理器，避免在业务代码中频繁创建临时线程池造成资源浪费。"""

    _instance: Optional["GlobalThreadPool"] = None
    _lock: Lock = Lock()

    # 线程池实例
    _db_pool: Optional[ThreadPoolExecutor] = None
    _io_pool: Optional[ThreadPoolExecutor] = None

    def __new__(cls):
        # 双重锁保证线程安全的单例
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize_pools()
        return cls._instance

    # --------------------- public properties ---------------------

    @property
    def db_executor(self) -> ThreadPoolExecutor:
        """数据库相关 CPU-密集型或阻塞操作线程池"""
        return self._db_pool  # type: ignore

    @property
    def io_executor(self) -> ThreadPoolExecutor:
        """文件 IO / 网络 IO 等操作线程池"""
        return self._io_pool  # type: ignore

    # --------------------- private helpers ---------------------

    def _initialize_pools(self) -> None:
        """实际创建底层 ThreadPoolExecutor"""
        # 创建固定大小线程池，可根据业务需求调整
        self.__class__._db_pool = ThreadPoolExecutor(
            max_workers=5,
            thread_name_prefix="db_worker",
        )
        self.__class__._io_pool = ThreadPoolExecutor(
            max_workers=10,
            thread_name_prefix="io_worker",
        )

    # --------------------- lifecycle ---------------------

    def shutdown(self, wait: bool = True):
        """关闭所有线程池，在应用退出时调用"""
        if self._db_pool:
            self._db_pool.shutdown(wait=wait)
        if self._io_pool:
            self._io_pool.shutdown(wait=wait) 