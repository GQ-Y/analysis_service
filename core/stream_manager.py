# core/stream_manager.py
import asyncio
import cv2
import time
import threading
import gc  # <-- 导入 gc 模块
from typing import Dict, Optional, Tuple, Any
from asyncio import Queue, Lock, Task
import traceback

# 尝试导入共享工具和配置
# 使用 try-except 块或者调整导入路径
try:
    from shared.utils.logger import setup_logger
    from core.config import settings
except ImportError:
    # 如果直接运行此文件或在特定测试环境中，提供备选方案
    print("无法导入共享模块，使用标准日志记录和默认设置替代。")
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    # 提供一个临时的 settings 替代品
    class MockStreamingConfig:
        connection_timeout: int = 10
        read_timeout: int = 15
        reconnect_attempts: int = 5
        reconnect_delay: int = 2
        max_consecutive_errors: int = 10
        frame_buffer_size: int = 5
    class MockSettings:
        STREAMING = MockStreamingConfig()
    settings = MockSettings()


logger = setup_logger(__name__)

class ManagedStream:
    """内部类，用于管理单个流的状态"""
    def __init__(self, url: str):
        self.url: str = url
        self.subscribers: Dict[str, Queue] = {} # subscriber_id -> Queue
        self.ref_count: int = 0
        self.read_task: Optional[Task] = None
        self.capture: Optional[cv2.VideoCapture] = None
        self.lock: Lock = Lock() # 用于保护此特定流状态的锁
        self.is_running: bool = False
        self.last_frame_time: float = 0.0
        self.consecutive_errors: int = 0
        self.error_state: bool = False # 标记流是否处于永久错误状态

    async def stop_reader(self):
        """安全地停止读取器任务并释放资源"""
        # 不再持有锁，让调用者处理锁
        # async with self.lock:
        if self.read_task and not self.read_task.done():
            logger.info(f"[{self.url}] 请求取消读取任务...")
            self.read_task.cancel()
            try:
                # 等待任务实际结束，避免资源未释放
                await asyncio.wait_for(self.read_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(f"[{self.url}] 等待读取任务取消超时")
            except asyncio.CancelledError:
                logger.info(f"[{self.url}] 读取任务已成功取消")
            except Exception as e:
                 logger.error(f"[{self.url}] 等待读取任务取消时发生未知错误: {e}", exc_info=True)

        if self.capture and self.capture.isOpened():
            logger.info(f"[{self.url}] 释放VideoCapture资源")
            try:
                # cap.release() 可能是阻塞的，考虑在executor中运行
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self.capture.release)
            except Exception as e:
                logger.error(f"[{self.url}] 释放VideoCapture时出错: {e}")
        self.capture = None
        self.read_task = None
        self.is_running = False
        logger.info(f"[{self.url}] 读取器已停止")


_stream_manager_instance = None
_stream_manager_lock = threading.Lock() # 使用 threading.Lock 以确保跨线程安全

class StreamManager:
    """
    管理视频流的共享、生命周期和订阅者。
    实现惰性加载和引用计数，确保只有在需要时才拉流。
    """
    def __init__(self):
        # 不需要在 __init__ 中检查单例，让 get_instance 处理
        self._streams: Dict[str, ManagedStream] = {} # url -> ManagedStream
        self._lock: Lock = Lock() # 用于保护 _streams 字典的异步锁
        logger.info("StreamManager 初始化完成")

    @classmethod
    def get_instance(cls) -> "StreamManager":
        """获取 StreamManager 的单例实例"""
        global _stream_manager_instance
        if _stream_manager_instance is None:
             # 使用同步锁保护单例创建过程
            with _stream_manager_lock:
                if _stream_manager_instance is None:
                    logger.info("创建 StreamManager 单例实例")
                    _stream_manager_instance = cls()
        return _stream_manager_instance

    async def subscribe(self, url: str, subscriber_id: str) -> Tuple[bool, Optional[Queue]]:
        """
        订阅一个视频流。

        如果流尚未运行，则启动它。
        如果订阅者已存在，则返回其现有队列。

        Args:
            url: 视频流 URL。
            subscriber_id: 订阅者的唯一标识符。

        Returns:
            Tuple[bool, Optional[Queue]]: (是否成功, 订阅者的帧队列 或 None)
        """
        async with self._lock: # 保护对 _streams 字典的访问
            managed_stream = self._streams.get(url)
            if not managed_stream:
                logger.info(f"[{url}] 第一个订阅者请求，创建新的 ManagedStream")
                managed_stream = ManagedStream(url)
                self._streams[url] = managed_stream
            elif managed_stream.error_state:
                 logger.warning(f"[{url}] 尝试订阅处于永久错误状态的流，订阅者: {subscriber_id}")
                 # 不应返回 True，因为订阅失败了
                 return False, None


        async with managed_stream.lock: # 保护对特定流状态的访问
            if subscriber_id in managed_stream.subscribers:
                logger.warning(f"[{url}] 订阅者 {subscriber_id} 已订阅，返回现有队列")
                return True, managed_stream.subscribers[subscriber_id]

            # 创建新的队列
            queue = Queue(maxsize=settings.STREAMING.frame_buffer_size)
            managed_stream.subscribers[subscriber_id] = queue
            managed_stream.ref_count += 1
            logger.info(f"[{url}] 订阅者 {subscriber_id} 已添加。当前订阅者数: {managed_stream.ref_count}")

            # 如果是第一个订阅者，启动读取循环
            if managed_stream.ref_count == 1:
                # 清理可能存在的旧的、未完成的任务
                if managed_stream.read_task and not managed_stream.read_task.done():
                     logger.warning(f"[{url}] 发现旧的读取任务未完成，尝试强制停止...")
                     await managed_stream.stop_reader() # 调用停止逻辑

                logger.info(f"[{url}] 第一个订阅者加入，启动读取循环...")
                # 创建读取任务
                managed_stream.read_task = asyncio.create_task(
                    self._read_loop(managed_stream), # 传递 ManagedStream 实例
                    name=f"ReadLoop-{url}"
                )
                managed_stream.is_running = True # 标记为尝试运行
                managed_stream.error_state = False # 重置错误状态

            return True, queue

    async def unsubscribe(self, url: str, subscriber_id: str) -> bool:
        """
        取消订阅一个视频流。

        如果这是最后一个订阅者，则停止该流的读取循环。

        Args:
            url: 视频流 URL。
            subscriber_id: 订阅者的唯一标识符。

        Returns:
            bool: 是否成功取消订阅。
        """
        managed_stream = None
        needs_removal = False

        async with self._lock: # 保护对 _streams 字典的访问
            managed_stream = self._streams.get(url)
            if not managed_stream:
                logger.warning(f"[{url}] 尝试取消订阅不存在的流")
                return False

            async with managed_stream.lock: # 保护对特定流状态的访问
                if subscriber_id not in managed_stream.subscribers:
                    logger.warning(f"[{url}] 尝试取消订阅未注册的订阅者: {subscriber_id}")
                    # 即使订阅者不存在，如果引用计数已为0，也可能需要清理管理器条目
                    if managed_stream.ref_count == 0 and not managed_stream.is_running:
                         needs_removal = True # 标记以便后续移除
                    return False # 取消订阅本身是失败的

                # 移除订阅者
                del managed_stream.subscribers[subscriber_id]
                managed_stream.ref_count -= 1
                logger.info(f"[{url}] 订阅者 {subscriber_id} 已移除。当前订阅者数: {managed_stream.ref_count}")

                # 如果是最后一个订阅者，停止读取循环并清理
                if managed_stream.ref_count == 0:
                    logger.info(f"[{url}] 最后一个订阅者离开，停止读取循环并清理...")
                    await managed_stream.stop_reader() # 调用停止逻辑
                    # 标记以便在 self._lock 保护下从管理器中移除这个流
                    needs_removal = True

            # 如果需要移除，在 self._lock 保护下执行
            if needs_removal:
                 logger.info(f"[{url}] 正在从 StreamManager 中删除条目")
                 if url in self._streams: # 再次检查，以防万一
                     # 确保我们删除的是同一个对象，虽然用 URL 通常足够
                     if self._streams[url] is managed_stream:
                          del self._streams[url]
                     else:
                          logger.warning(f"[{url}] 在尝试删除时发现管理器中的对象已更改")
                 else:
                      logger.warning(f"[{url}] 在尝试删除时发现条目已不存在")


        return True

    async def _read_loop(self, stream: ManagedStream):
        """
        后台协程，负责从单个视频流读取帧并分发给订阅者。
        包含连接、读取、重连、超时和错误处理逻辑。

        Args:
            stream: 管理此流状态的 ManagedStream 对象。
        """
        url = stream.url
        logger.info(f"[{url}] 读取循环启动...")
        reconnect_attempts = 0
        # 确保从 settings 加载，如果失败则使用默认值
        try:
             initial_reconnect_delay = settings.STREAMING.reconnect_delay
             max_reconnect_attempts = settings.STREAMING.reconnect_attempts
             read_timeout = settings.STREAMING.read_timeout
             max_consecutive_errors = settings.STREAMING.max_consecutive_errors
        except AttributeError:
             logger.warning(f"[{url}] 无法从settings加载STREAMING配置，使用默认值。")
             initial_reconnect_delay = 2
             max_reconnect_attempts = 5
             read_timeout = 15
             max_consecutive_errors = 10

        current_reconnect_delay = initial_reconnect_delay


        try:
            # 循环条件改为检查 is_running 标志，由 subscribe/unsubscribe 控制
            while stream.is_running:
                # --- 连接阶段 ---
                if stream.capture is None or not stream.capture.isOpened():
                    # 只有当 ref_count > 0 时才尝试重连
                    async with stream.lock:
                         should_reconnect = stream.ref_count > 0
                         is_error = stream.error_state # 检查是否已标记错误

                    if not should_reconnect:
                        logger.info(f"[{url}] 没有订阅者了，退出连接尝试。")
                        stream.is_running = False # 确保退出循环
                        break
                    if is_error:
                         logger.warning(f"[{url}] 流已标记为错误状态，停止重连尝试。")
                         stream.is_running = False # 确保退出循环
                         break

                    if reconnect_attempts >= max_reconnect_attempts:
                        logger.error(f"[{url}] 达到最大重连次数 ({max_reconnect_attempts})，将流标记为永久错误。")
                        async with stream.lock:
                            stream.error_state = True
                            # 通知现有订阅者流已失败
                            for queue in stream.subscribers.values():
                                try:
                                    # 使用 put 而不是 put_nowait，并设置超时，避免无限阻塞
                                    await asyncio.wait_for(queue.put(None), timeout=0.1)
                                except (asyncio.QueueFull, asyncio.TimeoutError):
                                    logger.warning(f"[{url}] 通知订阅者失败时队列已满或超时")
                                except Exception as notify_e:
                                     logger.error(f"[{url}] 通知订阅者失败时发生未知错误: {notify_e}")

                        stream.is_running = False # 退出循环
                        break # 退出循环

                    logger.info(f"[{url}] 尝试连接/重新连接 (尝试 {reconnect_attempts + 1}/{max_reconnect_attempts})...")
                    try:
                        # 运行阻塞的 VideoCapture 在 executor 中
                        loop = asyncio.get_running_loop()
                        logger.debug(f"[{url}] DEBUG: 即将调用 cv2.VideoCapture({url})")
                        stream.capture = await loop.run_in_executor(None, cv2.VideoCapture, url)
                        logger.debug(f"[{url}] DEBUG: cv2.VideoCapture 调用完成, capture is None: {stream.capture is None}")

                        # 检查是否成功打开
                        is_opened = await loop.run_in_executor(None, stream.capture.isOpened) if stream.capture else False
                        logger.debug(f"[{url}] DEBUG: capture.isOpened() 检查完成: {is_opened}")

                        if not is_opened:
                            logger.warning(f"[{url}] 连接失败。将在 {current_reconnect_delay} 秒后重试...")
                            if stream.capture: # 即使打开失败，也尝试释放
                                try:
                                    logger.debug(f"[{url}] DEBUG: 连接失败，即将调用 capture.release()")
                                    await loop.run_in_executor(None, stream.capture.release)
                                    logger.debug(f"[{url}] DEBUG: 连接失败，capture.release() 调用完成")
                                except Exception as release_err:
                                     logger.error(f"[{url}] 连接失败后释放VideoCapture出错: {release_err}")
                            stream.capture = None
                            await asyncio.sleep(current_reconnect_delay)
                            reconnect_attempts += 1
                            current_reconnect_delay = min(current_reconnect_delay * 2, 30) # 指数退避，最大30秒
                            continue # 继续尝试连接
                        else:
                            logger.info(f"[{url}] 连接成功！")
                            reconnect_attempts = 0 # 重置重连计数
                            current_reconnect_delay = initial_reconnect_delay # 重置延迟
                            async with stream.lock:
                                stream.last_frame_time = time.monotonic() # 重置帧时间戳
                                stream.consecutive_errors = 0 # 重置错误计数
                                # stream.error_state = False # 连接成功不应重置错误状态，只有手动重试或新订阅者加入时才重置

                    except Exception as e:
                        logger.error(f"[{url}] 连接过程中发生未预期错误: {e}", exc_info=True)
                        if stream.capture:
                            try:
                                loop = asyncio.get_running_loop()
                                logger.debug(f"[{url}] DEBUG: 连接异常，即将调用 capture.release()")
                                await loop.run_in_executor(None, stream.capture.release)
                                logger.debug(f"[{url}] DEBUG: 连接异常，capture.release() 调用完成")
                            except Exception as release_e:
                                logger.error(f"[{url}] 释放 VideoCapture 时再次出错: {release_e}")
                        stream.capture = None
                        await asyncio.sleep(current_reconnect_delay)
                        reconnect_attempts += 1
                        current_reconnect_delay = min(current_reconnect_delay * 2, 30)
                        continue

                # --- 读取和分发阶段 ---
                try:
                    # 检查超时
                    current_time = time.monotonic()
                    if current_time - stream.last_frame_time > read_timeout:
                        logger.warning(f"[{url}] 读取超时 ({read_timeout}s)，尝试重新连接...")
                        logger.debug(f"[{url}] DEBUG: 进入读取超时处理块，检查 stream.capture 是否存在...")
                        if stream.capture:
                             logger.debug(f"[{url}] DEBUG: stream.capture 存在，将尝试释放")
                             try:
                                 loop = asyncio.get_running_loop()
                                 logger.debug(f"[{url}] DEBUG: 读取超时，即将调用 capture.release()")
                                 # 在后台任务中执行 release 以避免阻塞事件循环
                                 await loop.run_in_executor(None, stream.capture.release)
                                 logger.debug(f"[{url}] DEBUG: capture.release() 调用完成")
                                 time.sleep(0.1) # <-- 在 release 后、None 赋值前，增加短暂同步休眠
                             except Exception as release_exc:
                                 logger.error(f"[{url}] 释放VideoCapture对象时出错: {release_exc}", exc_info=True)
                             finally:
                                 # 确保capture被置为None
                                 stream.capture = None
                                 logger.debug(f"[{stream.url}] DEBUG: stream.capture 已设置为 None")
                        else:
                             logger.debug(f"[{url}] DEBUG: stream.capture 不存在，无需释放")
                        await asyncio.sleep(1) # <-- 在标记为None后、重连前添加更长延迟
                        continue # 返回连接阶段

                    # 读取帧 (在 executor 中运行阻塞操作)
                    loop = asyncio.get_running_loop()
                    ret, frame = await asyncio.wait_for(
                        loop.run_in_executor(None, stream.capture.read),
                        timeout=read_timeout + 1 # 设置略大于读取超时的超时时间
                    )


                    if not ret:
                         async with stream.lock: # 读取失败也需要保护对 consecutive_errors 的访问
                              stream.consecutive_errors += 1
                              consecutive_errors = stream.consecutive_errors
                         logger.warning(f"[{url}] 读取帧失败 (连续 {consecutive_errors}/{max_consecutive_errors})")

                         if consecutive_errors >= max_consecutive_errors:
                              logger.error(f"[{url}] 达到最大连续读取错误次数，将流标记为永久错误。")
                              async with stream.lock:
                                   stream.error_state = True
                                   # 通知现有订阅者
                                   for sub_id, queue in stream.subscribers.items():
                                        try:
                                             await asyncio.wait_for(queue.put(None), timeout=0.1)
                                        except (asyncio.QueueFull, asyncio.TimeoutError):
                                             logger.warning(f"[{url}] 通知订阅者 {sub_id} 失败时队列已满或超时")
                                        except Exception as notify_e:
                                             logger.error(f"[{url}] 通知订阅者 {sub_id} 失败时发生未知错误: {notify_e}")

                              if stream.capture:
                                   try:
                                       await loop.run_in_executor(None, stream.capture.release)
                                   except Exception as release_err:
                                       logger.error(f"[{url}] 连续错误后释放VideoCapture出错: {release_err}")

                              stream.capture = None
                              stream.is_running = False # 退出循环
                              break # 退出循环
                         # 短暂休眠后重试读取，而不是立即重连
                         await asyncio.sleep(0.5)
                         continue
                    else:
                        # 成功读取帧
                         async with stream.lock: # 成功读取也需要保护
                              stream.consecutive_errors = 0
                              stream.last_frame_time = current_time # 使用开始读取的时间更准确

                         # 分发帧给所有订阅者
                         async with stream.lock: # 保护 subscriber 字典
                              # 创建副本以应对迭代时修改, 同时检查订阅者是否存在
                              subscribers_to_notify = list(stream.subscribers.items())

                         if not subscribers_to_notify: # 如果在分发前最后一个订阅者离开了
                              logger.info(f"[{url}] 读取到帧但没有订阅者了，准备退出循环。")
                              stream.is_running = False # 退出循环
                              break

                         # logger.debug(f"[{url}] 读取到帧，分发给 {len(subscribers_to_notify)} 个订阅者")
                         tasks = []
                         for sub_id, queue in subscribers_to_notify:
                              # 使用 create_task 异步放入，避免单个慢速消费者阻塞分发
                              tasks.append(asyncio.create_task(self._put_frame_to_queue(queue, frame, sub_id, url)))

                         # 等待所有放入操作完成（可选，但可以更好地处理队列满的情况）
                         if tasks:
                              done, pending = await asyncio.wait(tasks, timeout=1.0) # 设置超时
                              if pending:
                                   logger.warning(f"[{url}] 分发帧给部分订阅者超时，可能存在阻塞的消费者")
                                   for task in pending:
                                        task.cancel()


                except asyncio.TimeoutError:
                    current_time = time.monotonic()
                    # 检查是否真的超时，因为 wait_for 可能因其他原因（如 CancelledError）退出
                    if current_time - stream.last_frame_time > read_timeout:
                        logger.warning(f"[{url}] 读取帧操作超时（超过 {read_timeout}s），可能流已卡顿，尝试重连...")
                        
                        # --- 不再在此处调用 release() ---
                        logger.debug(f"[{url}] DEBUG: 检测到读取超时，将 capture 设为 None 以触发重连")
                        # if stream.capture:
                        #     logger.debug(f"[{url}] DEBUG: stream.capture 存在，将尝试释放")
                        #     try:
                        #         loop = asyncio.get_running_loop()
                        #         logger.debug(f"[{url}] DEBUG: 读取超时，即将调用 capture.release()")
                        #         # 在后台任务中执行 release 以避免阻塞事件循环
                        #         await loop.run_in_executor(None, stream.capture.release)
                        #         logger.debug(f"[{url}] DEBUG: capture.release() 调用完成")
                        #         # time.sleep(0.1) # <-- 已移除
                        #     except Exception as release_exc:
                        #         logger.error(f"[{url}] 释放VideoCapture对象时出错: {release_exc}", exc_info=True)
                        #     finally:
                        #         pass # finally 不再需要特殊操作
                        stream.capture = None # 直接设置为 None
                        # else:
                        #      logger.debug(f"[{url}] DEBUG: stream.capture 不存在，无需释放")
                        
                        # 增加重连尝试计数
                        reconnect_attempts += 1
                        # 短暂休眠后继续尝试连接
                        await asyncio.sleep(settings.STREAMING.reconnect_delay) 
                        continue # 跳回循环开始处尝试重新连接
                except asyncio.CancelledError:
                     logger.info(f"[{url}] 读取循环被明确取消")
                     stream.is_running = False # 确保退出循环
                     raise # 重新抛出以便 finally 处理
                except Exception as e:
                     logger.error(f"[{url}] 读取循环意外终止: {e}")
                     stream.is_running = False # 确保退出循环
                     raise # 重新抛出以便 finally 处理
                except Exception as e:
                    logger.error(f"""[{url}] 读取循环意外终止: {e}
{traceback.format_exc()}""")
                    async with stream.lock:
                         stream.error_state = True # 标记为错误

                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(f"[{url}] 错误达到上限 ({max_consecutive_errors})，标记为永久错误")
                        async with stream.lock:
                            stream.error_state = True
                             # 通知现有订阅者
                            for sub_id, queue in stream.subscribers.items():
                                try: await asyncio.wait_for(queue.put(None), timeout=0.1)
                                except: pass # 忽略通知错误
                        stream.is_running = False # 退出循环
                        break
                    await asyncio.sleep(1) # 稍作等待后继续

                # 短暂休眠，避免CPU空转，并允许其他任务运行
                await asyncio.sleep(0.001) # 非常短的休眠

        except asyncio.CancelledError:
            logger.info(f"[{url}] 读取循环任务在主循环外被取消")
        except Exception as e:
             logger.error(f"""[{url}] 读取循环意外终止: {e}
{traceback.format_exc()}""")
             async with stream.lock:
                  stream.error_state = True # 标记为错误
                  # 通知现有订阅者
                  for sub_id, queue in stream.subscribers.items():
                       try: await asyncio.wait_for(queue.put(None), timeout=0.1)
                       except: pass # 忽略通知错误
        finally:
            logger.info(f"[{url}] 读取循环结束，执行 finally 清理...")
            # 确保资源被释放
            if stream.capture and stream.capture.isOpened():
                logger.info(f"[{url}] 在 finally 块中释放 VideoCapture")
                try:
                    loop = asyncio.get_running_loop()
                    logger.debug(f"[{url}] DEBUG: finally - 即将调用 run_in_executor(release)")
                    await loop.run_in_executor(None, stream.capture.release)
                    logger.debug(f"[{url}] DEBUG: finally - run_in_executor(release) 调用完成")
                except Exception as e:
                    logger.error(f"[{url}] 在 finally 块中释放 VideoCapture 时出错: {e}")
            stream.capture = None
            async with stream.lock:
                 stream.is_running = False # 确保标记为未运行
                 # 清理可能仍在队列中的订阅者（如果循环异常退出）
                 # for sub_id, queue in list(stream.subscribers.items()):
                 #     logger.warning(f"[{url}] 清理订阅者 {sub_id}，可能由于异常退出")
                 #     del stream.subscribers[sub_id]
                 # stream.ref_count = 0 # 重置计数器


    async def _put_frame_to_queue(self, queue: Queue, frame: Any, subscriber_id: str, url: str):
        """将帧放入单个订阅者的队列，处理队列满的情况"""
        try:
            # 使用 put 而不是 put_nowait，并设置短暂超时
            await asyncio.wait_for(queue.put(frame), timeout=0.5)
        except asyncio.QueueFull:
            logger.warning(f"[{url}] 订阅者 {subscriber_id} 的队列持续已满，丢弃帧")
            # 可以考虑移除慢速消费者，或者增加队列大小
        except asyncio.TimeoutError:
            logger.warning(f"[{url}] 向订阅者 {subscriber_id} 的队列放入帧超时")
        except Exception as e:
            logger.error(f"[{url}] 向订阅者 {subscriber_id} 队列放入帧时发生未知错误: {e}", exc_info=True)


# 在模块级别提供单例实例的访问方式
# 使用 try-except 避免在某些情况下（如直接运行此文件进行测试）出错
try:
    stream_manager = StreamManager.get_instance()
except Exception as e:
    logger.error(f"初始化 StreamManager 单例时出错: {e}")
    stream_manager = None # 或者提供一个假的实例

