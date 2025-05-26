import asyncio
import time
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import httpx # httpx 库已存在
import sys # <--- 新增导入

from shared.utils.logger import get_normal_logger, get_exception_logger

normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)

MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 5  # seconds
DEFAULT_CALLBACK_TIMEOUT = 10 # seconds

@dataclass
class CallbackRequest:
    task_id: str
    url: str # 修改为单个URL
    payload: Dict[str, Any]
    callback_interval_seconds: int = 0 # 任务级别的回调间隔
    retry_count: int = 0

class CallbackService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(CallbackService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, max_queue_size: int = 1000):
        if self._initialized:
            return

        self._queue = asyncio.Queue(maxsize=max_queue_size)
        self._worker_task: Optional[asyncio.Task] = None
        self._active = False
        self._last_task_callback_times: Dict[str, float] = {} # 用于任务级别的回调间隔
        self._initialized = True
        normal_logger.info("CallbackService initialized.")

    async def start(self):
        if self._active:
            normal_logger.warning("CallbackService is already active.")
            return
        self._active = True
        self._worker_task = asyncio.create_task(self._worker())
        normal_logger.info("CallbackService started.")

    async def stop(self):
        if not self._active:
            normal_logger.warning("CallbackService is not active.")
            sys.stdout.flush(); sys.stderr.flush() # <---
            return
        
        normal_logger.info("CallbackService stopping... Setting active=False and putting sentinel in queue.")
        sys.stdout.flush(); sys.stderr.flush() # <--- 
        self._active = False
        try:
            await self._queue.put(None) 
            normal_logger.info("CallbackService: Sentinel put in queue.")
            sys.stdout.flush(); sys.stderr.flush() # <---
        except Exception as e:
            exception_logger.error(f"Error putting sentinel in callback queue: {e}")
            sys.stdout.flush(); sys.stderr.flush() # <--- 

        if self._worker_task:
            normal_logger.info(f"CallbackService: Attempting to wait for worker task (ID: {self._worker_task.get_name()}) to finish...")
            sys.stdout.flush(); sys.stderr.flush() # <--- 
            try:
                await asyncio.wait_for(self._worker_task, timeout=DEFAULT_CALLBACK_TIMEOUT + 5)
                normal_logger.info(f"CallbackService: Worker task (ID: {self._worker_task.get_name()}) finished gracefully.")
                sys.stdout.flush(); sys.stderr.flush() # <--- 
            except asyncio.TimeoutError:
                exception_logger.warning(f"CallbackService: Worker task (ID: {self._worker_task.get_name()}) did not stop gracefully within timeout. Cancelling task.")
                sys.stdout.flush(); sys.stderr.flush() # <--- 
                self._worker_task.cancel()
                try:
                    await self._worker_task
                    normal_logger.info(f"CallbackService: Worker task (ID: {self._worker_task.get_name()}) cancellation awaited.")
                    sys.stdout.flush(); sys.stderr.flush() # <--- 
                except asyncio.CancelledError:
                    normal_logger.info(f"CallbackService: Worker task (ID: {self._worker_task.get_name()}) was cancelled as expected.")
                    sys.stdout.flush(); sys.stderr.flush() # <--- 
                except Exception as e_cancel_await:
                    exception_logger.error(f"Error awaiting cancelled worker task (ID: {self._worker_task.get_name()}): {e_cancel_await}")
                    sys.stdout.flush(); sys.stderr.flush() # <--- 
            except Exception as e:
                exception_logger.error(f"Error waiting for callback worker task (ID: {self._worker_task.get_name()}): {e}")
                sys.stdout.flush(); sys.stderr.flush() # <--- 
        else:
            normal_logger.warning("CallbackService: No worker task found to stop.")
            sys.stdout.flush(); sys.stderr.flush() # <--- 
        
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                if item:
                    normal_logger.warning(f"Discarding callback request during stop for task: {item.task_id} to url: {item.url}")
                    sys.stdout.flush(); sys.stderr.flush() # <--- 
            except asyncio.QueueEmpty:
                break
        normal_logger.info("CallbackService stopped.")
        sys.stdout.flush(); sys.stderr.flush() # <--- 

    async def enqueue_callback(self, task_id: str, url: str, payload: Dict[str, Any], callback_interval_seconds: int = 0):
        if not self._active:
            normal_logger.error("CallbackService is not active. Cannot enqueue callback.")
            return

        if not url:
            normal_logger.warning(f"Task {task_id}: No callback URL provided, skipping enqueue.")
            return

        request = CallbackRequest(
            task_id=task_id,
            url=url, # 使用单个URL
            payload=payload,
            callback_interval_seconds=callback_interval_seconds
        )
        try:
            await self._queue.put(request)
            normal_logger.info(f"Task {task_id}: Callback request enqueued for URL: {url}. Interval: {callback_interval_seconds}s")
        except asyncio.QueueFull:
            exception_logger.error(f"Task {task_id}: Callback queue is full. Dropping request for URL: {url}")
        except Exception as e:
            exception_logger.error(f"Task {task_id}: Failed to enqueue callback request for URL: {url}: {e}")

    async def _worker(self):
        normal_logger.info("CallbackService worker started.")
        async with httpx.AsyncClient(timeout=DEFAULT_CALLBACK_TIMEOUT) as client:
            while self._active:
                try:
                    request: Optional[CallbackRequest] = await self._queue.get()
                    if request is None:
                        normal_logger.info("CallbackService worker received sentinel, shutting down.")
                        self._queue.task_done()
                        break

                    task_id = request.task_id
                    current_time = time.time()

                    if request.callback_interval_seconds > 0:
                        last_sent_time = self._last_task_callback_times.get(task_id, 0)
                        if current_time - last_sent_time < request.callback_interval_seconds:
                            normal_logger.info(f"Task {task_id} to URL {request.url}: Callback interval not met. Skipping for now. Last sent for task: {last_sent_time}, Interval: {request.callback_interval_seconds}s")
                            self._queue.task_done()
                            continue 
                    
                    callback_succeeded = False
                    try:
                        normal_logger.info(f"Task {task_id}: Sending callback to {request.url}. Retry: {request.retry_count}")
                        response = await client.post(request.url, json=request.payload)
                        response.raise_for_status()
                        normal_logger.info(f"Task {task_id}: Callback to {request.url} successful (Status: {response.status_code}).")
                        callback_succeeded = True
                        self._last_task_callback_times[task_id] = current_time 
                    
                    except httpx.HTTPStatusError as e:
                        exception_logger.error(f"Task {task_id}: HTTP Status Error for {request.url} (Retry: {request.retry_count}): {e.response.status_code} - {e.response.text}")
                    except httpx.RequestError as e:
                        exception_logger.error(f"Task {task_id}: Request Error for {request.url} (Retry: {request.retry_count}): {str(e)}")
                    except Exception as e:
                        exception_logger.exception(f"Task {task_id}: Unexpected error sending callback to {request.url} (Retry: {request.retry_count}): {e}")
                    
                    if not callback_succeeded:
                        if request.retry_count < MAX_RETRIES:
                            request.retry_count += 1
                            retry_delay = INITIAL_RETRY_DELAY * (2 ** (request.retry_count - 1))
                            normal_logger.info(f"Task {task_id}: Callback to {request.url} failed. Retrying in {retry_delay}s (Attempt {request.retry_count}/{MAX_RETRIES}).")
                            await asyncio.sleep(retry_delay)
                            try:
                                await self._queue.put(request)
                            except asyncio.QueueFull:
                                exception_logger.error(f"Task {task_id}: Callback queue full while re-enqueueing for retry to {request.url}. Dropping request.")
                            except Exception as e_requeue:
                                exception_logger.error(f"Task {task_id}: Failed to re-enqueue callback request for retry to {request.url}: {e_requeue}")
                        else:
                            exception_logger.error(f"Task {task_id}: Callback to {request.url} failed after {MAX_RETRIES} retries. Giving up. Payload keys: {list(request.payload.keys())}")
                    
                    self._queue.task_done()

                except asyncio.CancelledError:
                    normal_logger.info("CallbackService worker task cancelled.")
                    sys.stdout.flush(); sys.stderr.flush() # <--- 
                    break
                except Exception as e:
                    exception_logger.exception(f"CallbackService worker encountered an unexpected error: {e}")
                    sys.stdout.flush(); sys.stderr.flush() # <--- 
                    await asyncio.sleep(1)
        
        normal_logger.info("CallbackService worker stopped.")
        sys.stdout.flush(); sys.stderr.flush() # <--- 

callback_service = CallbackService() 