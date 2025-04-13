"""
MQTT协议的分析服务
处理MQTT协议的图像分析、视频分析和流分析请求
"""
import os
import json
import uuid
import base64
from typing import Dict, Any, List, Optional, Callable, Union
import asyncio
from asyncio import Queue
from paho.mqtt import client as mqtt_client
import time
import threading
import cv2
import numpy as np
import torch

from shared.utils.logger import setup_logger
from core.task_manager import TaskManager
from core.task_processor import TaskProcessor
from core.config import settings
from core.detection.yolo_detector import YOLODetector
from core.segmentation.yolo_segmentor import YOLOSegmentor
from services.base_analyzer import BaseAnalyzerService
from services.mqtt_client import MQTTClient
from core.stream_manager import stream_manager # 导入 StreamManager 单例

logger = setup_logger(__name__)

class MQTTAnalyzerService(BaseAnalyzerService):
    """MQTT分析服务"""

    def __init__(self, device_id, mqtt_config, model_configs=None, loop=None):
        """
        初始化MQTT分析服务
        
        Args:
            device_id: 设备ID
            mqtt_config: MQTT配置
            model_configs: 模型配置
            loop: 事件循环
        """
        # 调用父类的__init__方法
        super().__init__()
        self.device_id = device_id
        self.mqtt_config = mqtt_config
        self.model_configs = model_configs or {}
        
        # --- 重新添加获取和设置 main_loop 的逻辑 ---
        if loop:
            self.main_loop = loop
            logger.info("MQTTAnalyzerService 使用传入的事件循环")
        else:
            try:
                self.main_loop = asyncio.get_running_loop()
                logger.info("MQTTAnalyzerService 获取到当前正在运行的事件循环")
            except RuntimeError:
                logger.warning("MQTTAnalyzerService 初始化时未找到运行中的循环，尝试获取默认循环")
                # 作为备选方案，尝试获取默认事件循环，但这在非主线程中可能不是期望的循环
                self.main_loop = asyncio.get_event_loop() 
        # --------------------------------------------
        
        # 初始化任务管理器
        self.task_manager = TaskManager()
        
        # 初始化MQTT客户端 (确保传递了循环，如果MQTTClient内部需要)
        self.mqtt_client = MQTTClient(
            device_id=device_id,
            broker_host=mqtt_config.get("host", "localhost"),
            broker_port=mqtt_config.get("port", 1883),
            username=mqtt_config.get("username"),
            password=mqtt_config.get("password"),
            command_topic=mqtt_config.get("command_topic"), # 这些参数可能来自 mqtt_config dict
            response_topic=mqtt_config.get("response_topic"),
            status_topic=mqtt_config.get("status_topic"),
            topic_prefix=mqtt_config.get("topic_prefix", "meek/"), # 从 mqtt_config 获取
            loop=self.main_loop # 明确传递获取到的循环
        )
        
        logger.info(f"MQTT分析服务已初始化: 设备ID={device_id}")
        # 注意：_register_task_handlers() 现在应该在 connect 方法中调用，以确保 mqtt_client 已完全初始化
        # self._register_task_handlers() # 从 __init__ 移到 connect

    async def connect(self):
        """连接到MQTT服务器并注册任务处理器"""
        # 连接前确保注册所有任务处理器
        self._register_task_handlers()
        
        # 启动MQTT客户端
        if not self.mqtt_client.start():
            logger.error("MQTT客户端启动失败")
            return False
            
        logger.info("MQTT客户端启动成功")
        
        # 记录状态信息
        logger.info(f"MQTT客户端节点ID: {self.mqtt_client.node_id}")
        logger.info(f"MQTT客户端MAC地址: {self.mqtt_client.mac_address}")
        request_setting_topic = f"{self.mqtt_client.topic_prefix}{self.mqtt_client.node_id}/request_setting"
        logger.info(f"订阅的请求主题: {request_setting_topic}")
        
        # 所有处理器已注册，连接成功
        logger.info("成功连接到MQTT服务器")
        return True
            
    def _register_task_handlers(self):
        """注册任务处理器"""
        # 注册图像分析任务处理器
        self.mqtt_client.register_task_handler("image", self._handle_image_task)
        logger.info("已注册图像分析任务处理器")
        
        # 注册视频分析任务处理器
        self.mqtt_client.register_task_handler("video", self._handle_video_task)
        logger.info("已注册视频分析任务处理器")
        
        # 注册流分析任务处理器
        self.mqtt_client.register_task_handler("stream", self._handle_stream_task)
        logger.info("已注册流分析任务处理器")
        
        # 注册分割任务处理器
        self.mqtt_client.register_task_handler("segmentation", self._handle_segmentation_task)
        logger.info("已注册分割任务处理器")
        
        # 根据模型配置注册特定类型的处理器
        if self.model_configs:
            for model_code, config in self.model_configs.items():
                analysis_type = config.get("analysis_type")
                if analysis_type and analysis_type not in ["image", "video", "stream", "segmentation"]:
                    # 只为不同于基本类型的分析类型注册处理器
                    self.mqtt_client.register_task_handler(analysis_type, self._handle_detection_task)
                    logger.info(f"已注册{analysis_type}分析任务处理器")
                    
        # 注册通用检测任务处理器作为默认处理器
        self.mqtt_client.register_task_handler("detection", self._handle_detection_task)
        logger.info("已注册通用检测任务处理器")
        
        # 记录所有已注册的处理器
        available_handlers = list(self.mqtt_client.task_handlers.keys())
        logger.info(f"所有已注册的任务处理器: {available_handlers}")
            
    def _handle_stream_task(self, task_id, subtask_id, source, config, result_config, message_id=None, message_uuid=None, confirmation_topic=None):
        """
        处理流分析任务 (使用 StreamManager)
        """
        logger.info(f"处理流分析任务 (v3 - StreamManager): {task_id}/{subtask_id}")

        # --- 参数解析 --- 
        url = source.get("url") or (source.get("urls") and source["urls"][0])
        if not url:
            error_msg = "未指定流URL"
            logger.error(f"[{task_id}/{subtask_id}] {error_msg}")
            if confirmation_topic:
                self.mqtt_client._send_cmd_reply(message_id, message_uuid, confirmation_topic, "error",
                                                data={"error_message": error_msg, "task_id": task_id, "subtask_id": subtask_id})
            return False
        logger.info(f"[{task_id}/{subtask_id}] 流URL: {url}")

        subscriber_id = subtask_id # 使用 subtask_id 作为唯一订阅者ID
        analysis_type = config.get("analysis_type", "detection").lower()

        # --- 异步启动逻辑 --- 
        async def start_analysis_async():
            try:
                # 1. 订阅流
                logger.info(f"[{subscriber_id}] 尝试向 StreamManager 订阅 {url}")
                success, frame_queue = await stream_manager.subscribe(url, subscriber_id)

                if not success or frame_queue is None:
                    error_msg = f"无法订阅流 (可能已失败或队列未创建): {url}"
                    logger.error(f"[{subscriber_id}] {error_msg}")
                    # 发送错误回复（在协程内异步发送）
                    if confirmation_topic:
                        await self._async_send_cmd_reply(message_id, message_uuid, confirmation_topic, "error",
                                                      data={"error_message": error_msg, "task_id": task_id, "subtask_id": subtask_id})
                    return False # 表示协程内部启动失败

                logger.info(f"[{subscriber_id}] 成功订阅流 {url}，获取到帧队列")

                # 2. 创建分析任务 (asyncio.Task)
                logger.info(f"[{subscriber_id}] 创建分析协程任务 (类型: {analysis_type})")
                analysis_task = asyncio.create_task(
                    self._run_stream_analysis_from_queue(
                        subscriber_id, 
                        url, # <--- 将 url 作为 stream_url 参数传递
                        frame_queue, 
                        analysis_type, 
                        config.get("model_code"), 
                        config, 
                        result_config
                    ),
                    name=f"Analysis-{subscriber_id}"
                )

                # 3. 记录活动任务 (将 asyncio.Task 句柄存入)
                with self.mqtt_client.active_tasks_lock:
                    self.mqtt_client.active_tasks[subscriber_id] = {
                        "task_handle": analysis_task,
                        "start_time": time.time(),
                        "source": {"type": "stream", "urls": [url]}, # 重新构造 source 以存储 url
                        "config": config,
                        "result_config": result_config,
                        "stream_url": url # 也直接存储 stream_url
                    }
                logger.info(f"[{subscriber_id}] 分析协程已创建并添加到 active_tasks")
                
                # 确保任务真正启动
                try:
                    # 等待短暂时间，确保任务已开始执行
                    await asyncio.sleep(0.5)
                    
                    # 验证任务是否仍在运行（没有被取消或完成）
                    if analysis_task.done():
                        # 任务已完成，检查是否有异常
                        if analysis_task.exception():
                            error_msg = f"任务启动后立即失败: {analysis_task.exception()}"
                            logger.error(f"[{subscriber_id}] {error_msg}")
                            if confirmation_topic:
                                await self._async_send_cmd_reply(message_id, message_uuid, confirmation_topic, "error",
                                                              data={"error_message": error_msg, "task_id": task_id, "subtask_id": subtask_id, "cmd_type": "start_task"})
                            return False
                except Exception as e:
                    logger.warning(f"[{subscriber_id}] 验证任务启动状态时出错: {e}，继续发送成功响应")

                # 4. 发送成功确认回复 (异步)
                if confirmation_topic:
                    # 获取源类型和任务类型
                    source_type = source.get("type", "stream")
                    task_type = config.get("task_type", analysis_type) or "detection"
                    
                    # 构建完整的响应数据
                    response_data = {
                        "cmd_type": "start_task",
                        "message": "任务已接受并启动",
                        "task_id": task_id,
                        "subtask_id": subtask_id,
                        "task_status": 1,  # 1表示任务成功启动
                        "type": source_type  # 使用源类型（stream/video/image）
                    }
                    
                    # 如果有设置task_type，添加到响应中
                    if task_type and task_type != source_type:
                        response_data["analysis_type"] = task_type
                        
                    await self._async_send_cmd_reply(message_id, message_uuid, confirmation_topic, "success", data=response_data)
                    logger.info(f"[{subscriber_id}] 已发送任务启动成功响应，任务类型: {source_type}, 分析类型: {task_type}")
                return True # 表示协程内部启动成功

            except Exception as e:
                error_msg = f"启动流分析协程时内部出错: {str(e)}"
                logger.error(f"[{subscriber_id}] {error_msg}", exc_info=True)
                # 尝试清理：取消订阅 StreamManager
                try:
                    logger.info(f"[{subscriber_id}] 启动异常，尝试取消订阅 StreamManager")
                    await stream_manager.unsubscribe(url, subscriber_id)
                except Exception as unsub_e:
                    logger.error(f"[{subscriber_id}] 取消 StreamManager 订阅时出错: {unsub_e}", exc_info=True)
                # 清理 active_tasks
                with self.mqtt_client.active_tasks_lock:
                     if subscriber_id in self.mqtt_client.active_tasks:
                          del self.mqtt_client.active_tasks[subscriber_id]
                # 发送错误回复 (异步)
                if confirmation_topic:
                     await self._async_send_cmd_reply(message_id, message_uuid, confirmation_topic, "error",
                                                     data={"error_message": error_msg, "task_id": task_id, "subtask_id": subtask_id})
                return False # 表示协程内部启动失败

        # --- 提交到主事件循环 --- 
        if self.main_loop and self.main_loop.is_running():
            try:
                # 使用 run_coroutine_threadsafe 将协程提交到主循环
                future: asyncio.Future = asyncio.run_coroutine_threadsafe(start_analysis_async(), self.main_loop)
                logger.info(f"[{task_id}/{subtask_id}] 启动流分析请求已安全提交到事件循环")
                # 这里可以选择不等待 future 的结果，让协程在后台独立运行并处理确认回复
                # 返回 True 表示提交成功，后续由协程处理
                return True
            except Exception as e:
                 logger.error(f"[{task_id}/{subtask_id}] 安全提交启动任务到事件循环时发生错误: {e}", exc_info=True)
                 if confirmation_topic:
                      error_msg = f"提交启动任务时发生内部错误: {e}"
                      # 使用同步发送回复，因为这里在回调线程中
                      self.mqtt_client._send_cmd_reply(message_id, message_uuid, confirmation_topic, "error",
                                                       data={"error_message": error_msg, "task_id": task_id, "subtask_id": subtask_id})
                 return False
        else:
            # 事件循环不可用
            logger.error(f"[{task_id}/{subtask_id}] 主事件循环不可用或未运行，无法启动异步任务")
            if confirmation_topic:
                  error_msg = "服务内部错误，事件循环不可用"
                  self.mqtt_client._send_cmd_reply(message_id, message_uuid, confirmation_topic, "error",
                                                   data={"error_message": error_msg, "task_id": task_id, "subtask_id": subtask_id})
            return False

    async def _async_send_cmd_reply(self, message_id, message_uuid, topic, status, message=None, data=None):
        """辅助函数，用于异步发送命令回复
        
        Args:
            message_id: 消息ID
            message_uuid: 消息UUID
            topic: 主题
            status: 状态
            message: 回复消息
            data: 回复数据
        """
        # 注意: 这假设 self.mqtt_client._send_cmd_reply 本身是线程安全的
        # 或者需要一个真正的异步发送方法
        try:
            # 确保data是一个字典并包含task_id和subtask_id（如果data中已有则保留）
            if data is None:
                data = {}
            elif not isinstance(data, dict):
                data = {"original_data": data}
            
            # 添加额外消息
            if message and status == "error" and "message" not in data:
                data["message"] = message
                
            # 确保有cmd_type字段
            if "cmd_type" not in data and "task_id" in data and "subtask_id" in data:
                data["cmd_type"] = data.get("cmd_type", "start_task")
                
            # 添加任务状态字段（如果没有）
            if "task_status" not in data and status == "success" and "task_id" in data:
                data["task_status"] = 1  # 默认为1表示成功启动
            
            loop = asyncio.get_running_loop()
            # 在 executor 中运行同步的发送方法以避免阻塞
            await loop.run_in_executor(None, 
                                      lambda: self.mqtt_client._send_cmd_reply(
                                          message_id, message_uuid, topic, status, None, data))
        except Exception as e:
             logger.error(f"异步发送命令回复到 {topic} 时出错: {e}", exc_info=True)

    async def _run_stream_analysis_from_queue(self, subscriber_id, stream_url, frame_queue, analysis_type, model_code, task_config, result_config):
        """从队列读取帧并进行分析 (修改后包含停止检查和 stream_url 参数)"""
        logger.info(f"[{subscriber_id}] 分析协程启动，处理来自队列的帧 (类型: {analysis_type})")
        analyzer = None
        frame_count = 0
        analysis_interval = task_config.get("analysis_interval", 1)
        save_images = result_config.get("save_images", False)
        result_topic = result_config.get("callback_topic") 
        # stream_url 现在直接从参数获取
        
        # --- 新增：协程启动时检查是否已被请求停止 ---
        was_requested_to_stop = False
        with self.mqtt_client.pending_stop_requests_lock:
            if subscriber_id in self.mqtt_client.pending_stop_requests:
                logger.warning(f"[{subscriber_id}] 协程启动时发现已被请求停止，将直接进入清理。")
                was_requested_to_stop = True
                # 不在此处移除，由 finally 块处理

        if was_requested_to_stop:
             # 直接跳到 finally 块进行清理和状态发送
             # 通过引发一个特定异常或直接返回来触发 finally (直接返回更简洁)
             try:
                 # 发送一个初始的停止状态？或者直接在 finally 发送最终状态
                 pass # 决定在 finally 统一发送
             finally:
                 # --- finally 块 --- 
                 logger.info(f"[{subscriber_id}] 分析协程结束 (因启动时已被请求停止)，执行清理")
                 # 1. 从 pending_stop_requests 移除
                 with self.mqtt_client.pending_stop_requests_lock:
                     if subscriber_id in self.mqtt_client.pending_stop_requests:
                         self.mqtt_client.pending_stop_requests.remove(subscriber_id)
                         logger.debug(f"[{subscriber_id}] 已从待停止集合移除")
                 
                 # 2. 取消订阅 StreamManager (如果需要)
                 if stream_url:
                     logger.info(f"[{subscriber_id}] 取消订阅 StreamManager for {stream_url}")
                     try:
                         await stream_manager.unsubscribe(stream_url, subscriber_id)
                     except Exception as unsub_e:
                         logger.error(f"[{subscriber_id}] 取消 StreamManager 订阅时出错: {unsub_e}")
                 else:
                     logger.warning(f"[{subscriber_id}] 清理时无法获取 stream_url，无法取消订阅 StreamManager")
                 
                 # 3. 从 active_tasks 移除
                 with self.mqtt_client.active_tasks_lock:
                     if subscriber_id in self.mqtt_client.active_tasks:
                         del self.mqtt_client.active_tasks[subscriber_id]
                         logger.info(f"[{subscriber_id}] 已从 active_tasks 移除")
                 
                 # 4. 发送最终状态
                 # 如果任务是因为被取消而结束，发送完成的停止状态(task_status=1)
                 if was_in_pending_stop:
                     try:
                         # 获取原始task_id，如果存在
                         original_task_id = task_info.get('original_task_id', subscriber_id)
                         
                         # 先发送标准的stopped状态消息，使用原始task_id
                         self.mqtt_client._send_task_status(original_task_id, subscriber_id, "stopped")
                         logger.info(f"[{subscriber_id}] 已发送标准停止状态，使用原始task_id={original_task_id}")
                         
                         # 如果有message_id、message_uuid和confirmation_topic，发送带task_status=1的响应
                         if 'message_id' in task_info and 'message_uuid' in task_info and 'confirmation_topic' in task_info:
                             reply_data = {
                                 "cmd_type": "stop_task",
                                 "task_id": original_task_id,  # 使用原始task_id，而不是subscriber_id
                                 "subtask_id": subscriber_id,
                                 "task_status": 1,
                                 "message": "任务已停止",
                                 "timestamp": int(time.time())
                             }
                             self.mqtt_client._send_cmd_reply(
                                 task_info['message_id'], 
                                 task_info['message_uuid'], 
                                 task_info['confirmation_topic'], 
                                 "success", 
                                 data=reply_data
                             )
                             logger.info(f"[{subscriber_id}] 已发送任务已停止响应 (task_status=1)，使用原始task_id={original_task_id}")
                     except Exception as send_e:
                         logger.error(f"[{subscriber_id}] 发送最终状态失败: {send_e}", exc_info=True)
                 
                 # 从active_tasks中移除任务
                 with self.mqtt_client.active_tasks_lock:
                      if subscriber_id in self.mqtt_client.active_tasks:
                          del self.mqtt_client.active_tasks[subscriber_id]
                          logger.info(f"[{subscriber_id}] 已从 active_tasks 移除")
                      else:
                          logger.warning(f"[{subscriber_id}] 清理时未在 active_tasks 找到")
                 
                 logger.info(f"[{subscriber_id}] 清理完成")
             return # 结束协程

        # --- 正常执行逻辑 --- 
        try:
            # --- 初始化 last_analysis_time ---
            last_analysis_time = 0 
            # -------------------------------
            
            # 获取并初始化分析器
            logger.info(f"[{subscriber_id}] 初始化分析器 (类型: {analysis_type}, 模型: {model_code})")
            analyzer = self._get_analyzer(analysis_type)
            if not analyzer:
                 raise Exception(f"不支持的分析类型: {analysis_type}")
            
            # 异步加载模型
            logger.info(f"[{subscriber_id}] 尝试异步加载模型: {model_code}")
            await analyzer.load_model(model_code)
            logger.info(f"[{subscriber_id}] 模型 {model_code} 加载成功")

            # --- 主处理循环 --- 
            while True:
                # --- 添加取消检查点 --- 
                current_task = asyncio.current_task()
                if current_task.cancelled():
                    logger.info(f"[{subscriber_id}] 分析任务被取消，退出循环。")
                    break
                # ------------------------
                
                # 从队列获取帧
                try:
                     # 设置超时以避免无限期阻塞，并允许检查取消
                     frame = await asyncio.wait_for(frame_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                     # 超时后继续循环，允许再次检查取消状态
                     continue
                except asyncio.CancelledError: # 如果在 get() 时被取消
                     logger.info(f"[{subscriber_id}] 在等待帧时任务被取消。")
                     break
                
                if frame is None:  # 检查流结束或错误的信号
                    logger.info(f"[{subscriber_id}] 从队列收到 None 信号，流结束或出错")
                    break # 退出循环
                
                frame_count += 1

                # 分析间隔控制
                current_time = time.time()
                if current_time - last_analysis_time < analysis_interval:
                     await asyncio.sleep(0.001) # 避免完全阻塞，允许其他任务运行
                     continue
                
                last_analysis_time = current_time

                # 执行分析
                analysis_results = await analyzer.detect(frame, config=task_config) # 假设 detect 是通用的
                
                # 处理结果 
                if analysis_results:
                    logger.debug(f"[{subscriber_id}] 帧 {frame_count} 分析完成，找到 {len(analysis_results)} 个结果")
                    # 准备并发送结果
                    result_payload = {
                        "task_id": subscriber_id, # 使用 subscriber_id (subtask_id)
                        "subtask_id": subscriber_id,
                        "timestamp": time.time(),
                        "status": "processing",
                        "frame_count": frame_count,
                        "results": analysis_results,
                        # 如果需要，添加编码后的图像
                        "image_data": None 
                    }
                    
                    # 如果需要保存或编码图像，在这里处理
                    if save_images:
                        # ... (图像保存/编码逻辑) ...
                        # result_payload["image_data"] = encoded_image_data
                        pass # 占位符

                    if result_topic:
                        # 移除 await，因为 _send_task_result 是同步方法
                        self.mqtt_client._send_task_result(
                            task_id=subscriber_id, 
                            subtask_id=subscriber_id, 
                            status="completed",  # 假设分析完成即为 completed 状态
                            result=result_payload # 传递包含结果和其他信息的完整payload
                        )
                else:
                     logger.debug(f"[{subscriber_id}] 帧 {frame_count} 未检测到目标")
                     # 即使没有结果，也可能需要发送状态更新
                     status_payload = {
                        "task_id": subscriber_id,
                        "subtask_id": subscriber_id,
                        "timestamp": time.time(),
                        "status": "processing",
                        "frame_count": frame_count,
                        "results": [], # 空结果
                    }
                     if result_topic:
                          # 移除 await，因为 _send_task_result 是同步方法
                          self.mqtt_client._send_task_result(
                              task_id=subscriber_id, 
                              subtask_id=subscriber_id, 
                              status="processing", # 无结果时仍是 processing 状态
                              result=status_payload # 传递包含空结果和其他信息的完整payload
                          )
                          
                # --- 添加短暂休眠以让出控制权 --- 
                await asyncio.sleep(0) 
                # -------------------------------

        except asyncio.CancelledError:
             logger.info(f"[{subscriber_id}] 分析协程被取消")
        except Exception as e:
            logger.error(f"[{subscriber_id}] 分析协程出错: {e}", exc_info=True)
            # 发生错误时发送错误状态
            try:
                # 调用同步方法
                self.mqtt_client._send_task_status(subscriber_id, subscriber_id, "error", error=str(e))
            except Exception as send_e:
                 logger.error(f"[{subscriber_id}] 发送错误状态失败: {send_e}")
        finally:
            # --- finally 块 (正常结束或异常结束) ---
            logger.info(f"[{subscriber_id}] 分析协程结束，执行清理")
            
            # 1. 从 pending_stop_requests 移除 (确保无论如何都尝试移除)
            was_in_pending_stop = False
            with self.mqtt_client.pending_stop_requests_lock:
                if subscriber_id in self.mqtt_client.pending_stop_requests:
                    was_in_pending_stop = True
                    self.mqtt_client.pending_stop_requests.remove(subscriber_id)
                    logger.debug(f"[{subscriber_id}] 已从待停止集合移除")
            
            # 2. 取消订阅 StreamManager
            if stream_url:
                logger.info(f"[{subscriber_id}] 取消订阅 StreamManager for {stream_url}")
                try:
                    await stream_manager.unsubscribe(stream_url, subscriber_id)
                except Exception as unsub_e:
                    logger.error(f"[{subscriber_id}] 取消 StreamManager 订阅时出错: {unsub_e}")
            else:
                logger.warning(f"[{subscriber_id}] 清理时无法获取 stream_url，无法取消订阅 StreamManager")
                
            # 3. 从 active_tasks 移除
            with self.mqtt_client.active_tasks_lock:
                 if subscriber_id in self.mqtt_client.active_tasks:
                     # 获取task_info以获取更多信息(如message_id, message_uuid, confirmation_topic等)
                     task_info = self.mqtt_client.active_tasks.get(subscriber_id, {})
                     # 删除任务
                     del self.mqtt_client.active_tasks[subscriber_id]
                     logger.info(f"[{subscriber_id}] 已从 active_tasks 移除")

            # 4. 发送最终状态
            # 如果任务是因为被取消而结束，发送完成的停止状态(task_status=1)
            if was_in_pending_stop:
                try:
                    # 获取原始task_id，如果存在
                    original_task_id = task_info.get('original_task_id', subscriber_id)
                    
                    # 先发送标准的stopped状态消息，使用原始task_id
                    self.mqtt_client._send_task_status(original_task_id, subscriber_id, "stopped")
                    logger.info(f"[{subscriber_id}] 已发送标准停止状态，使用原始task_id={original_task_id}")
                    
                    # 如果有message_id、message_uuid和confirmation_topic，发送带task_status=1的响应
                    if 'message_id' in task_info and 'message_uuid' in task_info and 'confirmation_topic' in task_info:
                        reply_data = {
                            "cmd_type": "stop_task",
                            "task_id": original_task_id,  # 使用原始task_id，而不是subscriber_id
                            "subtask_id": subscriber_id,
                            "task_status": 1,
                            "message": "任务已停止",
                            "timestamp": int(time.time())
                        }
                        self.mqtt_client._send_cmd_reply(
                            task_info['message_id'], 
                            task_info['message_uuid'], 
                            task_info['confirmation_topic'], 
                            "success", 
                            data=reply_data
                        )
                        logger.info(f"[{subscriber_id}] 已发送任务已停止响应 (task_status=1)，使用原始task_id={original_task_id}")
                except Exception as send_e:
                    logger.error(f"[{subscriber_id}] 发送最终状态失败: {send_e}", exc_info=True)
            
            # 从active_tasks中移除任务
            with self.mqtt_client.active_tasks_lock:
                 if subscriber_id in self.mqtt_client.active_tasks:
                     del self.mqtt_client.active_tasks[subscriber_id]
                     logger.info(f"[{subscriber_id}] 已从 active_tasks 移除")
                 else:
                     logger.warning(f"[{subscriber_id}] 清理时未在 active_tasks 找到")
            
            logger.info(f"[{subscriber_id}] 清理完成")

    def disconnect(self):
        """断开MQTT连接"""
        if self.mqtt_client and self.connected:
            try:
                self.mqtt_client.disconnect()
                self.mqtt_client.loop_stop()
                self.connected = False
                logger.info("已断开与MQTT代理服务器的连接")
                return True
            except Exception as e:
                logger.error(f"断开MQTT连接时出错: {str(e)}", exc_info=True)
                return False
        return True
        
    def _on_connect(self, client, userdata, flags, rc):
        """MQTT连接回调"""
        if rc == 0:
            self.connected = True
            logger.info(f"已连接到MQTT代理服务器: {self.broker}:{self.port}")
            
            # 订阅命令主题
            client.subscribe(self.command_topic)
            logger.info(f"已订阅命令主题: {self.command_topic}")
        else:
            logger.error(f"连接MQTT代理服务器失败，返回码: {rc}")
            
    def _on_message(self, client, userdata, msg):
        """MQTT消息回调"""
        try:
            # 解析消息
            payload = msg.payload.decode("utf-8")
            logger.debug(f"收到MQTT消息: {payload}")
            
            # 解析JSON命令
            command = json.loads(payload)
            
            # 将命令添加到队列
            asyncio.run_coroutine_threadsafe(
                self.command_queue.put(command),
                asyncio.get_event_loop()
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {str(e)}", exc_info=True)
            self._publish_response({
                "success": False,
                "error": f"无效的JSON格式: {str(e)}"
            })
        except Exception as e:
            logger.error(f"处理MQTT消息时出错: {str(e)}", exc_info=True)
            self._publish_response({
                "success": False,
                "error": f"处理消息时出错: {str(e)}"
            })
            
    def _on_disconnect(self, client, userdata, rc):
        """MQTT断开连接回调"""
        self.connected = False
        if rc != 0:
            logger.warning(f"与MQTT代理服务器的连接意外断开，返回码: {rc}")
            # 尝试重新连接
            asyncio.create_task(self._reconnect())
        else:
            logger.info("已主动断开与MQTT代理服务器的连接")
            
    async def _reconnect(self, max_retries=5, retry_interval=5):
        """重新连接到MQTT代理服务器"""
        retries = 0
        while not self.connected and retries < max_retries:
            logger.info(f"尝试重新连接MQTT代理服务器，第 {retries+1} 次...")
            try:
                self.mqtt_client.reconnect()
                return
            except Exception as e:
                logger.error(f"重新连接失败: {str(e)}", exc_info=True)
                retries += 1
                await asyncio.sleep(retry_interval)
                
        if not self.connected:
            logger.error(f"重新连接MQTT代理服务器失败，已达到最大重试次数: {max_retries}")
            
    def _publish_response(self, response):
        """发布响应到MQTT主题"""
        if not self.connected:
            logger.error("未连接到MQTT代理服务器，无法发布响应")
            return False
            
        try:
            # 将响应对象转换为JSON字符串
            response_json = json.dumps(response)
            
            # 发布响应
            self.mqtt_client.publish(self.response_topic, response_json)
            logger.debug(f"已发布响应到主题 {self.response_topic}")
            return True
            
        except Exception as e:
            logger.error(f"发布MQTT响应时出错: {str(e)}", exc_info=True)
            return False
            
    async def _process_commands(self):
        """处理命令队列中的命令"""
        logger.info("开始处理MQTT命令队列")
        while True:
            try:
                # 从队列获取命令
                command = await self.command_queue.get()
                logger.info(f"正在处理命令: {command.get('action', 'unknown')}")
                
                # 解析命令类型
                action = command.get("action")
                if not action:
                    self._publish_response({
                        "success": False,
                        "command_id": command.get("command_id"),
                        "error": "无效的命令，缺少 'action' 字段"
                    })
                    continue
                    
                # 处理不同类型的命令
                if action == "analyze_image":
                    await self._handle_analyze_image(command)
                elif action == "start_video_analysis":
                    await self._handle_start_video_analysis(command)
                elif action == "start_stream_analysis":
                    await self._handle_start_stream_analysis(command)
                elif action == "stop_task":
                    await self._handle_stop_task(command)
                elif action == "get_task_status":
                    await self._handle_get_task_status(command)
                elif action == "get_tasks":
                    await self._handle_get_tasks(command)
                else:
                    self._publish_response({
                        "success": False,
                        "command_id": command.get("command_id"),
                        "error": f"未知的命令类型: {action}"
                    })
                    
            except Exception as e:
                logger.error(f"处理MQTT命令时出错: {str(e)}", exc_info=True)
                try:
                    self._publish_response({
                        "success": False,
                        "command_id": command.get("command_id") if "command" in locals() else None,
                        "error": f"处理命令时出错: {str(e)}"
                    })
                except Exception as e2:
                    logger.error(f"发送错误响应时出错: {str(e2)}", exc_info=True)
                    
    async def _handle_analyze_image(self, command):
        """处理图像分析命令"""
        try:
            # 获取命令参数
            command_id = command.get("command_id")
            model_code = command.get("model_code")
            image_data = command.get("image_data")  # Base64编码的图像数据
            image_path = command.get("image_path")  # 或者图像路径
            conf_threshold = command.get("conf_threshold", 0.25)
            save_result = command.get("save_result", True)
            include_image = command.get("include_image", False)
            
            # 验证参数
            if not model_code:
                self._publish_response({
                    "success": False,
                    "command_id": command_id,
                    "error": "缺少必需的参数: model_code"
                })
                return
                
            if not image_data and not image_path:
                self._publish_response({
                    "success": False,
                    "command_id": command_id,
                    "error": "缺少必需的参数: image_data 或 image_path"
                })
                return
                
            # 如果提供了图像数据，保存为文件
            if image_data:
                try:
                    # 解码Base64图像数据
                    image_bytes = base64.b64decode(image_data)
                    
                    # 保存为文件
                    filename = f"{uuid.uuid4().hex}.jpg"
                    file_path = f"{settings.OUTPUT.save_dir}/images/{filename}"
                    
                    with open(file_path, "wb") as f:
                        f.write(image_bytes)
                        
                    image_path = file_path
                except Exception as e:
                    self._publish_response({
                        "success": False,
                        "command_id": command_id,
                        "error": f"图像数据解码失败: {str(e)}"
                    })
                    return
                    
            # 创建任务
            task_id = self.task_manager.create_task(
                task_type="image",
                protocol="mqtt",
                params={
                    "image_path": image_path,
                    "model_code": model_code,
                    "conf_threshold": conf_threshold,
                    "save_result": save_result,
                    "include_image": include_image,
                    "command_id": command_id
                }
            )
            
            # 处理图像分析任务
            result = self.task_processor.process_image(task_id)
            
            # 添加命令ID到响应中
            result["command_id"] = command_id
            
            # 发布响应
            self._publish_response(result)
            
        except Exception as e:
            logger.error(f"处理图像分析命令时出错: {str(e)}", exc_info=True)
            self._publish_response({
                "success": False,
                "command_id": command.get("command_id"),
                "error": f"处理图像分析命令时出错: {str(e)}"
            })
            
    async def _handle_start_video_analysis(self, command):
        """处理视频分析命令"""
        try:
            # 获取命令参数
            command_id = command.get("command_id")
            model_code = command.get("model_code")
            video_path = command.get("video_path")
            conf_threshold = command.get("conf_threshold", 0.25)
            save_result = command.get("save_result", True)
            
            # 验证参数
            if not model_code:
                self._publish_response({
                    "success": False,
                    "command_id": command_id,
                    "error": "缺少必需的参数: model_code"
                })
                return
                
            if not video_path:
                self._publish_response({
                    "success": False,
                    "command_id": command_id,
                    "error": "缺少必需的参数: video_path"
                })
                return
                
            # 验证视频文件是否存在
            if not os.path.exists(video_path):
                self._publish_response({
                    "success": False,
                    "command_id": command_id,
                    "error": f"视频文件不存在: {video_path}"
                })
                return
                
            # 创建任务
            task_id = self.task_manager.create_task(
                task_type="video",
                protocol="mqtt",
                params={
                    "video_path": video_path,
                    "model_code": model_code,
                    "conf_threshold": conf_threshold,
                    "save_result": save_result,
                    "command_id": command_id
                }
            )
            
            # 启动视频分析任务
            result = self.task_processor.start_video_analysis(task_id)
            
            # 添加命令ID到响应中
            result["command_id"] = command_id
            
            # 发布响应
            self._publish_response(result)
            
        except Exception as e:
            logger.error(f"处理视频分析命令时出错: {str(e)}", exc_info=True)
            self._publish_response({
                "success": False,
                "command_id": command.get("command_id"),
                "error": f"处理视频分析命令时出错: {str(e)}"
            })
            
    async def _handle_start_stream_analysis(self, command):
        logger.info(f"打印全部命令数据: {command}")
        """处理流分析命令"""
        try:
            # 获取命令参数
            command_id = command.get("command_id")
            model_code = command.get("model_code")
            stream_url = command.get("stream_url")
            conf_threshold = command.get("conf_threshold", 0.25)
            save_interval = command.get("save_interval", 10)
            max_duration = command.get("max_duration", 3600)
            
            # 验证参数
            if not model_code:
                self._publish_response({
                    "success": False,
                    "command_id": command_id,
                    "error": "缺少必需的参数: model_code"
                })
                return
                
            if not stream_url:
                self._publish_response({
                    "success": False,
                    "command_id": command_id,
                    "error": "缺少必需的参数: stream_url"
                })
                return
                
            # 创建任务
            task_id = self.task_manager.create_task(
                task_type="stream",
                protocol="mqtt",
                params={
                    "stream_url": stream_url,
                    "model_code": model_code,
                    "conf_threshold": conf_threshold,
                    "save_interval": save_interval,
                    "max_duration": max_duration,
                    "command_id": command_id
                }
            )
            
            # 启动流分析任务
            result = self.task_processor.start_stream_analysis(task_id)
            
            # 添加命令ID到响应中
            result["command_id"] = command_id
            
            # 发布响应
            self._publish_response(result)
            
        except Exception as e:
            logger.error(f"处理流分析命令时出错: {str(e)}", exc_info=True)
            self._publish_response({
                "success": False,
                "command_id": command.get("command_id"),
                "error": f"处理流分析命令时出错: {str(e)}"
            })
            
    async def _handle_stop_task(self, command):
        """处理停止任务命令"""
        task_id = command.get("task_id")
        subtask_id = command.get("subtask_id") # 通常使用 subtask_id 作为唯一标识
        target_id = subtask_id if subtask_id else task_id

        logger.info(f"收到停止任务请求: {task_id}/{subtask_id}")

        task_info = None
        task_handle = None
        stream_url_to_unsub = None

        with self.mqtt_client.active_tasks_lock:
             task_info = self.mqtt_client.active_tasks.get(target_id)
             if task_info:
                  task_handle = task_info.get("task_handle")
                  stream_url_to_unsub = task_info.get("stream_url") # 获取关联的URL
                  # 存储原始task_id，确保用于回复消息
                  task_info['original_task_id'] = task_id
                  # 存储消息相关信息，用于最终状态回复
                  message_id = command.get('message_id')
                  message_uuid = command.get('message_uuid')
                  confirmation_topic = command.get('confirmation_topic')
                  if message_id and message_uuid and confirmation_topic:
                      task_info['message_id'] = message_id
                      task_info['message_uuid'] = message_uuid
                      task_info['confirmation_topic'] = confirmation_topic
             
        if task_handle and isinstance(task_handle, asyncio.Task) and not task_handle.done():
             logger.info(f"正在取消任务: {target_id}")
             # 先发送task_status=0的响应，表示正在处理停止请求
             if 'message_id' in task_info and 'message_uuid' in task_info and 'confirmation_topic' in task_info:
                 reply_data = {
                     "cmd_type": "stop_task",
                     "task_id": task_id,  # 使用原始task_id
                     "subtask_id": subtask_id,
                     "task_status": 0,  # 0表示正在处理
                     "message": "正在处理结束任务",
                     "timestamp": int(time.time())
                 }
                 self.mqtt_client._send_cmd_reply(
                     task_info['message_id'], 
                     task_info['message_uuid'], 
                     task_info['confirmation_topic'], 
                     "success", 
                     data=reply_data
                 )
                 logger.info(f"已发送正在处理停止请求响应 (task_status=0): {task_id}/{subtask_id}")
             
             # 取消任务
             task_handle.cancel()
             
             # 等待任务实际取消完成 (带超时)
             try:
                 await asyncio.wait_for(task_handle, timeout=5.0)
                 logger.info(f"任务 {target_id} 已成功等待取消完成")
             except asyncio.TimeoutError:
                 logger.warning(f"等待任务 {target_id} 取消超时，但仍继续处理")
             except asyncio.CancelledError:
                 logger.info(f"任务 {target_id} 已成功取消")
             except Exception as wait_e:
                 logger.error(f"等待任务 {target_id} 取消时出错: {wait_e}")
             
             # 手动清理资源：确保释放流资源
             if stream_url_to_unsub:
                 try:
                     from services.streams.stream_manager import StreamManager
                     stream_manager = StreamManager.get_instance()
                     await stream_manager.unsubscribe(stream_url_to_unsub, target_id)
                     logger.info(f"已手动取消订阅流: {stream_url_to_unsub}, 订阅者: {target_id}")
                 except Exception as unsub_e:
                     logger.error(f"手动取消订阅流时出错: {unsub_e}")
             
             # 确保任务从active_tasks中移除
             with self.mqtt_client.active_tasks_lock:
                 if target_id in self.mqtt_client.active_tasks:
                     # 不删除task_info，我们需要其中的消息ID等信息
                     # 只将状态标记为已停止
                     self.mqtt_client.active_tasks[target_id]['status'] = 'stopped'
                     logger.info(f"已将任务 {target_id} 在active_tasks中标记为stopped状态")
             
             # 发送已停止状态
             await self.mqtt_client._send_task_status(task_id, subtask_id, "stopped")
             logger.info(f"已发送任务停止状态: {task_id}/{subtask_id}")
             
        elif task_info:
             logger.warning(f"任务 {target_id} 已完成或句柄无效，无法取消")
             # 即使任务已完成，也尝试发送停止状态，确保使用原始task_id
             await self.mqtt_client._send_task_status(task_id, subtask_id, "stopped")
             
             # 清理相关资源
             with self.mqtt_client.active_tasks_lock:
                 if target_id in self.mqtt_client.active_tasks:
                     self.mqtt_client.active_tasks[target_id]['status'] = 'stopped'
                     logger.info(f"已将已完成任务 {target_id} 在active_tasks中标记为stopped状态")
        else:
             logger.warning(f"未找到活动任务: {target_id}，无法停止")
             # 发送错误状态或任务不存在状态，确保使用原始task_id
             # 我们发送task_status=-1，表示节点上找不到此任务
             if 'message_id' in command and 'message_uuid' in command and 'confirmation_topic' in command:
                 reply_data = {
                     "cmd_type": "stop_task",
                     "task_id": task_id,
                     "subtask_id": subtask_id,
                     "task_status": -1,  # -1表示任务不存在
                     "message": "未找到任务",
                     "timestamp": int(time.time())
                 }
                 self.mqtt_client._send_cmd_reply(
                     command.get('message_id'), 
                     command.get('message_uuid'), 
                     command.get('confirmation_topic'), 
                     "success", 
                     data=reply_data
                 )
                 logger.info(f"已发送任务不存在响应 (task_status=-1): {task_id}/{subtask_id}")

    async def _handle_get_task_status(self, command):
        """处理获取任务状态命令"""
        try:
            # 获取命令参数
            command_id = command.get("command_id")
            task_id = command.get("task_id")
            
            # 验证参数
            if not task_id:
                self._publish_response({
                    "success": False,
                    "command_id": command_id,
                    "error": "缺少必需的参数: task_id"
                })
                return
                
            # 获取任务状态
            result = self.task_processor.get_task_status(task_id)
            
            # 添加命令ID到响应中
            result["command_id"] = command_id
            
            # 发布响应
            self._publish_response(result)
            
        except Exception as e:
            logger.error(f"处理获取任务状态命令时出错: {str(e)}", exc_info=True)
            self._publish_response({
                "success": False,
                "command_id": command.get("command_id"),
                "error": f"处理获取任务状态命令时出错: {str(e)}"
            })
            
    async def _handle_get_tasks(self, command):
        """处理获取任务列表命令"""
        try:
            # 获取命令参数
            command_id = command.get("command_id")
            
            # 获取所有任务
            tasks = self.task_manager.get_all_tasks()
            
            # 过滤出MQTT协议的任务
            mqtt_tasks = {
                task_id: task for task_id, task in tasks.items()
                if task.get("protocol") == "mqtt"
            }
            
            # 构建响应
            result = {
                "success": True,
                "command_id": command_id,
                "tasks": mqtt_tasks
            }
            
            # 发布响应
            self._publish_response(result)
            
        except Exception as e:
            logger.error(f"处理获取任务列表命令时出错: {str(e)}", exc_info=True)
            self._publish_response({
                "success": False,
                "command_id": command.get("command_id"),
                "error": f"处理获取任务列表命令时出错: {str(e)}"
            })

    def _handle_image_task(self, task_id, subtask_id, source, config, result_config, message_id=None, message_uuid=None, confirmation_topic=None):
        """处理图片分析任务 (实际使用流分析处理器流量处理)"""
        logger.info(f"处理图片分析任务: {task_id}/{subtask_id}")
        # 目前图片也使用流分析的方式处理，所以直接委托给_handle_stream_task
        # 但设置特定的type="image"
        success = self._handle_stream_task(task_id, subtask_id, source, config, result_config, message_id, message_uuid, confirmation_topic)
        return success
    
    def _handle_video_task(self, task_id, subtask_id, source, config, result_config, message_id=None, message_uuid=None, confirmation_topic=None):
        """处理视频分析任务 (实际使用流分析处理器流量处理)"""
        logger.info(f"处理视频分析任务: {task_id}/{subtask_id}")
        # 目前视频也使用流分析的方式处理，所以直接委托给_handle_stream_task
        # 但设置特定的type="video"
        success = self._handle_stream_task(task_id, subtask_id, source, config, result_config, message_id, message_uuid, confirmation_topic)
        return success

    def _handle_segmentation_task(self, task_id, subtask_id, source, config, result_config, message_id=None, message_uuid=None, confirmation_topic=None):
        """处理分割任务，可能会使用流处理或图像处理"""
        logger.info(f"处理分割任务: {task_id}/{subtask_id}")
        
        # 分割任务也可能通过流处理，检查 source 类型
        source_type = source.get("type", "unknown")
        
        # 设置任务类型为"segmentation"
        config["task_type"] = "segmentation"
        
        # 将分割任务交给流分析处理器处理，但指定type为segmentation
        if source_type in ["stream", "video", "image"]:
            source["type"] = source_type  # 确保保留原始源类型
            success = self._handle_stream_task(task_id, subtask_id, source, config, result_config, message_id, message_uuid, confirmation_topic)
            return success
        else:
            error_msg = f"分割任务不支持的源类型: {source_type}"
            logger.error(f"[{task_id}/{subtask_id}] {error_msg}")
            if confirmation_topic:
                self.mqtt_client._send_cmd_reply(message_id, message_uuid, confirmation_topic, "error",
                                        data={
                                            "error_message": error_msg, 
                                            "task_id": task_id, 
                                            "subtask_id": subtask_id,
                                            "cmd_type": "start_task",
                                            "type": "segmentation"
                                        })
            return False
    
    def _handle_detection_task(self, task_id, subtask_id, source, config, result_config, message_id=None, message_uuid=None, confirmation_topic=None):
        """处理检测任务，可能会使用流处理或图像处理"""
        logger.info(f"处理检测任务: {task_id}/{subtask_id}")
        
        # 检测任务也可能通过流处理
        source_type = source.get("type", "unknown")
        
        # 设置任务类型为"detection"
        config["task_type"] = "detection"
        
        # 将检测任务交给流分析处理器处理，但指定type为detection
        if source_type in ["stream", "video", "image"]:
            source["type"] = source_type  # 确保保留原始源类型
            success = self._handle_stream_task(task_id, subtask_id, source, config, result_config, message_id, message_uuid, confirmation_topic)
            return success
        else:
            error_msg = f"检测任务不支持的源类型: {source_type}"
            logger.error(f"[{task_id}/{subtask_id}] {error_msg}")
            if confirmation_topic:
                self.mqtt_client._send_cmd_reply(message_id, message_uuid, confirmation_topic, "error",
                                        data={
                                            "error_message": error_msg, 
                                            "task_id": task_id, 
                                            "subtask_id": subtask_id,
                                            "cmd_type": "start_task",
                                            "type": "detection"
                                        })
            return False

    # --- 新增方法 --- 
    def _get_analyzer(self, analysis_type: str) -> Optional[Union[YOLODetector, YOLOSegmentor]]:
        """
        根据分析类型获取或创建分析器实例。
        注意：目前每次调用都创建新实例，未来可优化为缓存实例。
        """
        analysis_type = analysis_type.lower()
        logger.debug(f"获取分析器实例，类型: {analysis_type}")
        if analysis_type == "detection":
            # 这里可以考虑重用实例，但需要处理模型切换
            return YOLODetector()
        elif analysis_type == "segmentation":
            # 同上
            return YOLOSegmentor()
        else:
            logger.error(f"不支持的分析类型: {analysis_type}")
            return None
    # --------------- 

    def disconnect(self):
        """断开MQTT连接"""
        if self.mqtt_client and self.connected:
            try:
                self.mqtt_client.disconnect()
                self.mqtt_client.loop_stop()
                self.connected = False
                logger.info("已断开与MQTT代理服务器的连接")
                return True
            except Exception as e:
                logger.error(f"断开MQTT连接时出错: {str(e)}", exc_info=True)
                return False
        return True 