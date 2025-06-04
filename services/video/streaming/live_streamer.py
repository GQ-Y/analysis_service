"""
直播推流器
提供实时视频流推送到ZLMediaKit服务器功能
"""
from typing import Dict, Any, Optional, List
import os
import time
import asyncio
import threading
import subprocess
import uuid
import cv2
import numpy as np
from datetime import datetime

from core.task_management.utils.status import TaskStatus
from shared.utils.logger import get_normal_logger, get_exception_logger, get_test_logger
from services.video.utils.frame_dropper import SmartFrameDropper
from services.video.utils.ffmpeg_params import FFmpegParamsGenerator
from services.video.utils.frame_renderer import FrameRenderer
from services.video.encoders.base_encoder import BaseEncoder
from shared.utils.app_state import app_state_manager

# 导入优化组件
from services.video.utils.optimized_frame_renderer import optimized_renderer
from services.video.utils.frame_cache import frame_cache
from services.video.utils.smart_frame_skipper import smart_frame_skipper
from services.video.utils.performance_config import performance_config, PerformanceMode

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)
test_logger = get_test_logger()


class LiveStreamer(BaseEncoder):
    """直播推流器 - 用于推送实时视频流到ZLMediaKit服务器"""
    
    def __init__(self):
        """初始化直播推流器"""
        super().__init__()
        self.streaming_tasks = {}  # 存储直播流任务信息
        self.ffmpeg_params = FFmpegParamsGenerator()
        self.frame_renderer = FrameRenderer()
        self.normal_logger = get_normal_logger(__name__)
        self.exception_logger = get_exception_logger(__name__)
        self.frame_queue = None
        self.is_running = False
        self.stop_event = asyncio.Event()
        normal_logger.info("直播推流器初始化完成")
    
    async def start_live_stream(self, task_id: str, task_manager, format: str = "rtmp",
                              quality: int = 80, width: Optional[int] = None,
                              height: Optional[int] = None, fps: int = 15, stream_type: str = "ffmpeg") -> Dict[str, Any]:
        """
        开启直播流编码 - 支持两种推流模式

        Args:
            task_id: 任务ID
            task_manager: 任务管理器实例
            format: 流格式，支持"rtmp"、"hls"、"flv"
            quality: 视频质量(1-100)
            width: 视频宽度，为空则使用原始宽度
            height: 视频高度，为空则使用原始高度
            fps: 视频帧率
            stream_type: 推流类型，"ffmpeg"=直接FFmpeg输出FLV流, "zlm"=使用ZLMediaKit服务器

        Returns:
            Dict[str, Any]: 编码结果，包含流信息和播放地址
        """
        try:
            # 检查任务是否存在
            if not task_manager.has_task(task_id):
                return {
                    "success": False,
                    "message": f"任务不存在: {task_id}",
                    "stream_info": None,
                    "play_urls": None
                }

            # 检查任务是否正在运行或等待中
            task_info = task_manager.get_task(task_id)
            if not task_info:
                return {
                    "success": False,
                    "message": f"任务信息不存在: {task_id}",
                    "stream_info": None,
                    "play_urls": None
                }

            # 支持 WAITING 和 PROCESSING 状态的任务进行视频编码
            allowed_statuses = [TaskStatus.WAITING, TaskStatus.PROCESSING]
            if task_info["status"] not in allowed_statuses:
                status_names = {
                    TaskStatus.WAITING: "等待中",
                    TaskStatus.PROCESSING: "处理中",
                    TaskStatus.COMPLETED: "已完成",
                    TaskStatus.FAILED: "失败",
                    TaskStatus.STOPPED: "已停止"
                }
                current_status_name = status_names.get(task_info["status"], f"未知状态({task_info['status']})")
                return {
                    "success": False,
                    "message": f"任务状态不支持视频编码: {task_id}, 当前状态: {current_status_name}, 支持的状态: 等待中、处理中",
                    "stream_info": None,
                    "play_urls": None
                }

            # 如果已经有编码任务，先停止
            if task_id in self.streaming_tasks:
                stop_result = await self.stop_live_stream(task_id)
                if not stop_result.get("success", False):
                    normal_logger.warning(f"停止旧直播流任务失败: {stop_result.get('message')}")
                # 等待一小段时间，确保旧的编码任务完全停止
                await asyncio.sleep(1)

            # 根据推流类型选择不同的处理方式
            if stream_type.lower() == "ffmpeg":
                return await self._start_ffmpeg_direct_stream(
                    task_id, task_manager, format, quality, width, height, fps
                )
            elif stream_type.lower() == "zlm":
                return await self._start_zlm_server_stream(
                    task_id, task_manager, format, quality, width, height, fps
                )
            else:
                return {
                    "success": False,
                    "message": f"不支持的推流类型: {stream_type}, 仅支持 'ffmpeg' 和 'zlm'",
                    "stream_info": None,
                    "play_urls": None
                }

        except Exception as e:
            exception_logger.exception(f"启动视频直播流失败: {str(e)}")
            return {
                "success": False,
                "message": f"启动视频直播流失败: {str(e)}",
                "stream_info": None,
                "play_urls": None
            }
    
    async def stop_live_stream(self, task_id: str) -> Dict[str, Any]:
        """
        停止视频直播流

        Args:
            task_id: 任务ID

        Returns:
            Dict[str, Any]: 停止结果
        """
        try:
            # 检查任务是否存在
            if task_id not in self.streaming_tasks:
                return {
                    "success": False,
                    "message": f"直播流任务不存在: {task_id}"
                }

            # 获取编码任务信息
            encoding_info = self.streaming_tasks[task_id]
            stream_id = encoding_info.get("stream_id", "")

            # 停止FFmpeg进程
            if task_id in self.ffmpeg_processes:
                process = self.ffmpeg_processes[task_id]
                try:
                    # 尝试正常终止进程
                    if process.poll() is None:  # 如果进程仍在运行
                        process.terminate()
                        # 等待进程终止
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            # 如果超时，强制终止
                            process.kill()

                    normal_logger.info(f"FFmpeg推流进程已停止: {task_id}")
                except Exception as e:
                    exception_logger.exception(f"停止FFmpeg推流进程时出错: {str(e)}")

                # 从字典中移除进程
                del self.ffmpeg_processes[task_id]

            # 从字典中移除编码任务信息
            del self.streaming_tasks[task_id]

            # 清理帧丢弃器
            if task_id in self.frame_droppers:
                del self.frame_droppers[task_id]

            # 清理分析结果缓存
            self.clear_analysis_result_cache(task_id)

            # 记录停止信息
            normal_logger.info(f"直播流任务已停止: {task_id}, 流ID: {stream_id}")

            # 添加测试日志
            test_logger.info("TEST_LOG_MARKER: VIDEO_LIVE_STREAM_STOP_SUCCESS")

            return {
                "success": True,
                "message": f"直播流任务已停止: {task_id}"
            }

        except Exception as e:
            exception_logger.exception(f"停止直播流任务失败: {str(e)}")
            return {
                "success": False,
                "message": f"停止直播流任务失败: {str(e)}"
            }
    
    async def check_stream_status(self, task_id: str) -> Dict[str, Any]:
        """
        检查直播流状态

        Args:
            task_id: 任务ID

        Returns:
            Dict[str, Any]: 状态信息
        """
        try:
            # 检查任务是否存在
            if task_id not in self.streaming_tasks:
                return {
                    "success": False,
                    "message": f"直播流任务不存在: {task_id}",
                    "is_streaming": False
                }

            # 获取编码任务信息
            encoding_info = self.streaming_tasks[task_id]
            stream_id = encoding_info.get("stream_id", "")
            app_name = encoding_info.get("app_name", "live")

            # 检查是否是直播流任务
            if not encoding_info.get("is_live_stream", False):
                return {
                    "success": False,
                    "message": f"任务不是直播流任务: {task_id}",
                    "is_streaming": False
                }

            # 检查FFmpeg进程是否存在
            if task_id not in self.ffmpeg_processes:
                return {
                    "success": False,
                    "message": f"FFmpeg进程不存在: {task_id}",
                    "is_streaming": False
                }

            # 检查FFmpeg进程是否仍在运行
            process = self.ffmpeg_processes[task_id]
            if process.poll() is not None:
                return {
                    "success": False,
                    "message": f"FFmpeg进程已退出，返回码: {process.poll()}",
                    "is_streaming": False
                }

            # 检查ZLMediaKit服务器上的流状态
            try:
                from core.media_kit.zlm_manager import zlm_manager
                # 查询流状态
                stream_info = await zlm_manager.get_stream_info(app_name, stream_id)
                if stream_info.get("code") == 0:
                    # 流存在于ZLMediaKit中
                    return {
                        "success": True,
                        "message": f"直播流正常: {task_id}",
                        "is_streaming": True,
                        "stream_info": stream_info.get("data", {})
                    }
                else:
                    return {
                        "success": False,
                        "message": f"直播流在ZLMediaKit中不存在: {task_id}",
                        "is_streaming": False
                    }
            except Exception as e:
                return {
                    "success": False,
                    "message": f"检查ZLMediaKit流状态时出错: {str(e)}",
                    "is_streaming": False
                }

        except Exception as e:
            exception_logger.exception(f"检查直播流状态失败: {str(e)}")
            return {
                "success": False,
                "message": f"检查直播流状态失败: {str(e)}",
                "is_streaming": False
            }

    async def _start_live_stream_process(self, task_id: str, stream_id: str, app_name: str,
                                        push_url: str, format: str, task_manager, quality: int = 80,
                                        width: Optional[int] = None, height: Optional[int] = None,
                                        fps: int = 15) -> Optional[Dict[str, Any]]:
        """
        启动直播推流进程

        Args:
            task_id: 任务ID
            stream_id: 流ID
            app_name: 应用名称
            push_url: 推流地址
            format: 流格式
            task_manager: 任务管理器
            quality: 视频质量
            width: 视频宽度
            height: 视频高度
            fps: 视频帧率

        Returns:
            Optional[Dict[str, Any]]: 流信息，如果失败则返回None
        """
        try:
            # 创建推流线程
            streaming_thread = threading.Thread(
                target=self._live_streaming_thread,
                args=(task_id, stream_id, app_name, push_url, format, task_manager, quality, width, height, fps),
                daemon=True
            )
            streaming_thread.start()

            # 存储推流线程
            self.encoding_threads[task_id] = streaming_thread

            # 等待一段时间，确保推流进程启动
            normal_logger.info(f"等待推流进程初始化: {task_id}")
            await asyncio.sleep(5)

            # 检查进程是否仍在运行
            if task_id in self.ffmpeg_processes:
                process = self.ffmpeg_processes[task_id]
                if process.poll() is not None:
                    # 进程已退出
                    normal_logger.error(f"FFmpeg推流进程已退出，返回码: {process.poll()}")

                    # 尝试获取错误输出
                    if hasattr(process, 'stderr') and process.stderr:
                        try:
                            stderr_output = process.stderr.read()
                            if stderr_output:
                                normal_logger.error(f"FFmpeg推流错误输出: {stderr_output.decode('utf-8', errors='ignore')}")
                        except Exception as e:
                            normal_logger.error(f"读取FFmpeg推流错误输出失败: {str(e)}")

                    # 返回错误
                    return None

            # 构建流信息
            stream_info = {
                "stream_id": stream_id,
                "app_name": app_name,
                "push_url": push_url,
                "format": format,
                "quality": quality,
                "width": width,
                "height": height,
                "fps": fps,
                "status": "streaming"
            }

            return stream_info

        except Exception as e:
            normal_logger.exception(f"启动推流进程失败: {str(e)}")
            return None

    def _live_streaming_thread(self, task_id: str, stream_id: str, app_name: str,
                              push_url: str, format: str, task_manager=None, quality: int = 80,
                              width: Optional[int] = None, height: Optional[int] = None, fps: int = 15):
        """
        直播推流线程，实时获取任务原始视频流帧并渲染分析目标框后推送到ZLMediaKit

        Args:
            task_id: 任务ID
            stream_id: 流ID
            app_name: 应用名称
            push_url: 推流地址
            format: 流格式
            task_manager: 任务管理器
            quality: 视频质量
            width: 视频宽度
            height: 视频高度
            fps: 视频帧率
        """
        # 记录线程启动
        normal_logger.info(f"直播推流线程启动: {task_id}, 流ID: {stream_id}, 格式: {format}")

        # 在线程中创建新的事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # 运行异步函数
            loop.run_until_complete(self._async_live_streaming_worker(
                task_id, stream_id, app_name, push_url, format, task_manager, quality, width, height, fps
            ))
        except Exception as e:
            normal_logger.error(f"直播推流线程发生错误: {str(e)}")
            import traceback
            normal_logger.error(traceback.format_exc())
        finally:
            loop.close()

    async def _async_live_streaming_worker(self, task_id: str, stream_id: str, app_name: str,
                                         push_url: str, format: str, task_manager=None, quality: int = 80,
                                         width: Optional[int] = None, height: Optional[int] = None, fps: int = 15):
        """
        异步直播推流工作函数

        Args:
            task_id: 任务ID
            stream_id: 流ID
            app_name: 应用名称
            push_url: 推流地址
            format: 流格式
            task_manager: 任务管理器
            quality: 视频质量
            width: 视频宽度
            height: 视频高度
            fps: 视频帧率
        """
        try:
            # 获取任务的原始视频流
            # 从任务管理器获取任务配置
            task_info = task_manager.get_task(task_id) if task_manager else None
            if not task_info:
                normal_logger.error(f"无法获取任务信息: {task_id}")
                return

            # 获取流配置信息
            task_data = task_info.get("data", {})
            task_params = task_data.get("params", {}) if isinstance(task_data, dict) else {}

            # 获取原始视频流ID和URL
            original_stream_id = task_params.get("video_id") or task_params.get("stream_id") or f"stream_{task_id}"
            stream_url = task_params.get("stream_url", "")

            normal_logger.info(f"任务 {task_id}: 准备订阅原始视频流 {original_stream_id}, URL: {stream_url}")

            # 直接订阅原始视频流以获取未处理的帧
            stream_manager = app_state_manager.get_stream_manager()
            if not stream_manager:
                normal_logger.error("流管理器未初始化")
                return

            frame_queue = None
            try:
                # 构建流配置
                stream_config = {
                    "url": stream_url,
                    "rtsp_transport": task_params.get("rtsp_transport", "tcp"),
                    "reconnect_attempts": task_params.get("reconnect_attempts", 3),
                    "reconnect_delay": task_params.get("reconnect_delay", 5),
                    "frame_buffer_size": task_params.get("frame_buffer_size", 100),
                    "task_id": f"{task_id}_encoder",  # 使用不同的任务ID避免冲突
                    "video_id": original_stream_id
                }

                success, frame_queue = await stream_manager.subscribe_stream(
                    original_stream_id,
                    f"{task_id}_encoder",
                    stream_config
                )

                if not success or frame_queue is None:
                    normal_logger.error(f"订阅原始视频流失败: {original_stream_id}")
                    return

                normal_logger.info(f"成功订阅原始视频流: {original_stream_id}")
            except Exception as e:
                normal_logger.error(f"订阅原始视频流时发生异常: {str(e)}")
                return

            # 获取第一帧以确定分辨率
            frame = None
            retry_count = 0
            max_retries = 100

            # 从原始视频流获取第一帧
            normal_logger.info(f"开始从原始流获取帧: {task_id}")
            while frame is None and retry_count < max_retries:
                try:
                    frame_data = await asyncio.wait_for(frame_queue.get(), timeout=1.0)
                    if frame_data and isinstance(frame_data, tuple) and len(frame_data) >= 2:
                        frame, timestamp = frame_data[:2]
                        if frame is not None:
                            normal_logger.info(f"成功获取原始流帧: {task_id}, 帧形状: {frame.shape}")
                            break
                except asyncio.TimeoutError:
                    pass
                except Exception as e:
                    normal_logger.error(f"获取原始流帧失败: {str(e)}")

                retry_count += 1
                if retry_count % 10 == 0:
                    normal_logger.info(f"等待原始流帧中: {task_id}, 已重试 {retry_count} 次")

                # 检查任务是否已停止
                if task_id not in self.streaming_tasks:
                    normal_logger.info(f"编码任务已停止，直播推流线程退出: {task_id}")
                    return

            # 如果仍然没有获取到帧，使用默认帧
            if frame is None:
                normal_logger.warning(f"无法获取原始流帧，使用默认帧: {task_id}")
                frame = self.create_default_frame(640, 480, "等待视频流...")

            # 确定视频分辨率
            frame_height, frame_width = frame.shape[:2]
            final_width = width if width is not None else frame_width
            final_height = height if height is not None else frame_height

            normal_logger.info(f"视频推流参数 - 任务ID: {task_id}, 格式: {format}, 分辨率: {final_width}x{final_height}, 帧率: {fps}, 质量: {quality}")

            # 使用优化的比特率计算
            bitrate = self.ffmpeg_params.calculate_optimal_bitrate(quality, final_width, final_height, fps)
            normal_logger.info(f"计算得出最优推流比特率: {bitrate}k")

            # 使用优化的推流FFmpeg参数
            ffmpeg_cmd = self.ffmpeg_params.get_realtime_streaming_ffmpeg_params(final_width, final_height, fps, bitrate, push_url)

            normal_logger.info(f"FFmpeg命令: {' '.join(ffmpeg_cmd)}")

            # 启动FFmpeg进程
            try:
                process = subprocess.Popen(
                    ffmpeg_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=10*1024*1024
                )

                # 存储进程
                self.ffmpeg_processes[task_id] = process

                # 主循环 - 实时处理原始视频流帧
                frame_interval = 1.0 / fps
                last_frame_time = time.time()
                frame_count = 0
                error_count = 0
                max_errors = 5

                # 记录任务状态并添加到帧上
                task_status = task_info.get("status", "未知")
                task_start_time = datetime.now()

                while task_id in self.streaming_tasks:
                    # 检查FFmpeg进程是否仍在运行
                    if process.poll() is not None:
                        normal_logger.error(f"FFmpeg进程意外退出，返回码: {process.poll()}, 任务ID: {task_id}")
                        break

                    # 控制帧率
                    current_time = time.time()
                    elapsed = current_time - last_frame_time
                    if elapsed < frame_interval:
                        await asyncio.sleep(frame_interval - elapsed)
                    last_frame_time = time.time()

                    # 获取原始视频流帧
                    current_frame = None
                    try:
                        frame_data = await asyncio.wait_for(frame_queue.get(), timeout=0.1)
                        if frame_data and isinstance(frame_data, tuple) and len(frame_data) >= 2:
                            current_frame, timestamp = frame_data[:2]
                    except asyncio.TimeoutError:
                        # 超时，使用上一帧或默认帧
                        pass
                    except Exception as e:
                        normal_logger.error(f"获取原始流帧失败: {str(e)}")

                    # 如果没有获取到有效帧，使用默认帧
                    if current_frame is None:
                        error_count += 1
                        if error_count > max_errors:
                            current_frame = self.create_default_frame(final_width, final_height, "视频流中断...")
                            error_count = 0
                        else:
                            continue
                    else:
                        error_count = 0

                    # 调整帧大小
                    if current_frame.shape[1] != final_width or current_frame.shape[0] != final_height:
                        current_frame = cv2.resize(current_frame, (final_width, final_height))

                    # 获取最新的分析结果并渲染在帧上 - 使用被动缓存机制
                    latest_analysis_result = self.get_cached_analysis_result(task_id)
                    
                    # 使用优化的渲染组件
                    rendered_frame = self._render_frame_optimized(current_frame, latest_analysis_result, task_id)

                    # 计算任务运行时间
                    elapsed_time = (datetime.now() - task_start_time).total_seconds()

                    # 每10秒更新一次任务状态
                    if int(elapsed_time) % 10 == 0:
                        try:
                            # 从任务管理器获取最新状态
                            if task_manager and task_manager.has_task(task_id):
                                updated_task_info = task_manager.get_task(task_id)
                                if updated_task_info:
                                    task_status = updated_task_info.get("status", task_status)
                        except Exception as e:
                            normal_logger.debug(f"更新任务状态时出错: {str(e)}")

                    # 添加状态信息到帧上
                    rendered_frame = optimized_renderer.add_status_info(rendered_frame, task_status, elapsed_time)

                    # 每100帧记录一次日志
                    frame_count += 1
                    if frame_count % 100 == 0:
                        normal_logger.info(f"直播推流线程已处理 {frame_count} 帧: {task_id}")

                    # 写入帧数据到FFmpeg
                    try:
                        if process.poll() is None:  # 确保进程仍在运行
                            process.stdin.write(rendered_frame.tobytes())
                        else:
                            break
                    except BrokenPipeError:
                        normal_logger.error(f"写入帧数据时管道已断开: {task_id}")
                        break
                    except Exception as e:
                        normal_logger.error(f"写入帧数据时出错: {str(e)}")
                        break

                # 关闭FFmpeg输入流
                if process.stdin:
                    try:
                        process.stdin.close()
                    except Exception as e:
                        normal_logger.error(f"关闭FFmpeg输入流时出错: {str(e)}")

                # 等待FFmpeg进程完成
                try:
                    process.wait(timeout=30)
                    normal_logger.info(f"FFmpeg进程已完成: {task_id}, 返回码: {process.returncode}")
                except subprocess.TimeoutExpired:
                    normal_logger.warning(f"等待FFmpeg进程完成超时: {task_id}")
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()

            except Exception as e:
                normal_logger.error(f"FFmpeg进程运行时出错: {str(e)}")
                import traceback
                normal_logger.error(traceback.format_exc())

            finally:
                # 取消订阅原始视频流
                try:
                    await stream_manager.unsubscribe_stream(original_stream_id, f"{task_id}_encoder")
                    normal_logger.info(f"已取消订阅原始视频流: {original_stream_id}")
                except Exception as e:
                    normal_logger.error(f"取消订阅原始视频流失败: {str(e)}")

                # 清理流队列
                if frame_queue:
                    try:
                        # 清空队列中的所有内容
                        while not frame_queue.empty():
                            try:
                                frame_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                break
                        normal_logger.debug(f"已清空帧队列: {task_id}")
                    except Exception as e:
                        normal_logger.error(f"清空帧队列失败: {str(e)}")

                # 通知ZLMediaKit流已结束（如果需要）
                try:
                    from core.media_kit.zlm_manager import zlm_manager
                    # 某些情况下，我们可能需要显式通知ZLMediaKit流已结束
                    await zlm_manager.close_stream(app_name, stream_id)
                    normal_logger.info(f"已通知ZLMediaKit关闭流: {stream_id}")
                except Exception as e:
                    normal_logger.debug(f"通知ZLMediaKit关闭流失败: {str(e)}")

                # 记录线程结束日志
                normal_logger.info(f"直播推流线程清理完成并正常退出: {task_id}")

        except Exception as e:
            normal_logger.error(f"直播推流工作函数发生错误: {str(e)}")
            import traceback
            normal_logger.error(traceback.format_exc())

    async def _start_ffmpeg_direct_stream(self, task_id: str, task_manager, format: str,
                                        quality: int, width: Optional[int], height: Optional[int], fps: int) -> Dict[str, Any]:
        """
        启动FFmpeg直接输出流 - 默认模式，直接输出FLV格式的带分析结果视频流

        Args:
            task_id: 任务ID
            task_manager: 任务管理器
            format: 流格式（FFmpeg模式下强制为FLV）
            quality: 视频质量
            width: 视频宽度
            height: 视频高度
            fps: 视频帧率

        Returns:
            Dict[str, Any]: 推流结果
        """
        try:
            # FFmpeg直接输出模式固定为FLV格式
            actual_format = "flv"
            normal_logger.info(f"FFmpeg直接输出模式启动: {task_id}, 格式: {actual_format}")

            # 创建唯一的流ID，用于HTTP访问
            stream_id = f"ffmpeg_task_{task_id}_{int(time.time())}"
            
            # FFmpeg直接输出的HTTP URL（由FFmpeg内置HTTP服务器提供）
            http_port = 8081 + (hash(task_id) % 1000)  # 为每个任务分配不同端口
            output_url = f"http://127.0.0.1:{http_port}/live.flv"

            # 创建智能帧丢弃器
            self.frame_droppers[task_id] = SmartFrameDropper(target_fps=fps)

            # 存储编码任务信息
            self.streaming_tasks[task_id] = {
                "stream_id": stream_id,
                "stream_type": "ffmpeg",
                "output_url": output_url,
                "http_port": http_port,
                "format": actual_format,
                "quality": quality,
                "width": width,
                "height": height,
                "fps": fps,
                "start_time": datetime.now().isoformat(),
                "is_live_stream": True
            }

            # 启动FFmpeg直接输出进程
            stream_info = await self._start_ffmpeg_direct_process(
                task_id=task_id,
                stream_id=stream_id,
                output_url=output_url,
                http_port=http_port,
                task_manager=task_manager,
                quality=quality,
                width=width,
                height=height,
                fps=fps
            )

            if not stream_info:
                # 如果启动失败，删除编码任务信息
                if task_id in self.streaming_tasks:
                    del self.streaming_tasks[task_id]
                return {
                    "success": False,
                    "message": f"启动FFmpeg直接输出流失败: {task_id}",
                    "stream_info": None,
                    "play_urls": None
                }

            # 构建播放地址 - FFmpeg直接输出只提供HTTP FLV
            play_urls = {
                "http_flv": output_url,
                "description": "FFmpeg直接输出的HTTP FLV流，包含实时分析结果"
            }

            # 添加测试日志
            test_logger.info("TEST_LOG_MARKER: VIDEO_FFMPEG_DIRECT_STREAM_START_SUCCESS")

            return {
                "success": True,
                "message": f"FFmpeg直接输出视频流已启动: {task_id}",
                "stream_info": stream_info,
                "play_urls": play_urls
            }

        except Exception as e:
            exception_logger.exception(f"启动FFmpeg直接输出流失败: {str(e)}")
            return {
                "success": False,
                "message": f"启动FFmpeg直接输出流失败: {str(e)}",
                "stream_info": None,
                "play_urls": None
            }

    async def _start_zlm_server_stream(self, task_id: str, task_manager, format: str,
                                     quality: int, width: Optional[int], height: Optional[int], fps: int) -> Dict[str, Any]:
        """
        启动ZLMediaKit服务器推流 - 传统模式，通过ZLM服务器进行推流

        Args:
            task_id: 任务ID
            task_manager: 任务管理器
            format: 流格式
            quality: 视频质量
            width: 视频宽度
            height: 视频高度
            fps: 视频帧率

        Returns:
            Dict[str, Any]: 推流结果
        """
        try:
            # 验证格式
            if format.lower() not in ["rtmp", "hls", "flv"]:
                return {
                    "success": False,
                    "message": f"不支持的流格式: {format}, 仅支持rtmp、hls和flv",
                    "stream_info": None,
                    "play_urls": None
                }

            normal_logger.info(f"ZLMediaKit服务器推流模式启动: {task_id}, 格式: {format}")

            # 创建唯一的流ID
            stream_id = f"zlm_task_{task_id}_{int(time.time())}"
            app_name = "live"

            # 构建推流地址
            push_url = f"rtmp://127.0.0.1:1935/{app_name}/{stream_id}"

            # 创建智能帧丢弃器
            self.frame_droppers[task_id] = SmartFrameDropper(target_fps=fps)

            # 存储编码任务信息
            self.streaming_tasks[task_id] = {
                "stream_id": stream_id,
                "stream_type": "zlm",
                "app_name": app_name,
                "push_url": push_url,
                "format": format.lower(),
                "quality": quality,
                "width": width,
                "height": height,
                "fps": fps,
                "start_time": datetime.now().isoformat(),
                "is_live_stream": True
            }

            # 启动推流进程（使用现有的方法）
            stream_info = await self._start_live_stream_process(
                task_id=task_id,
                stream_id=stream_id,
                app_name=app_name,
                push_url=push_url,
                format=format.lower(),
                task_manager=task_manager,
                quality=quality,
                width=width,
                height=height,
                fps=fps
            )

            if not stream_info:
                # 如果启动失败，删除编码任务信息
                if task_id in self.streaming_tasks:
                    del self.streaming_tasks[task_id]
                return {
                    "success": False,
                    "message": f"启动ZLM服务器推流失败: {task_id}",
                    "stream_info": None,
                    "play_urls": None
                }

            # 构建播放地址
            play_urls = {
                "rtmp": f"rtmp://127.0.0.1:1935/{app_name}/{stream_id}",
                "hls": f"http://127.0.0.1:8080/{app_name}/{stream_id}.m3u8",
                "flv": f"http://127.0.0.1:8080/{app_name}/{stream_id}.flv",
                "webrtc": f"http://127.0.0.1:8080/index.html?app={app_name}&stream={stream_id}&type=webrtc"
            }

            # 添加测试日志
            test_logger.info("TEST_LOG_MARKER: VIDEO_ZLM_SERVER_STREAM_START_SUCCESS")

            return {
                "success": True,
                "message": f"ZLM服务器视频流已启动: {task_id}",
                "stream_info": stream_info,
                "play_urls": play_urls
            }

        except Exception as e:
            exception_logger.exception(f"启动ZLM服务器推流失败: {str(e)}")
            return {
                "success": False,
                "message": f"启动ZLM服务器推流失败: {str(e)}",
                "stream_info": None,
                "play_urls": None
            }

    async def _start_ffmpeg_direct_process(self, task_id: str, stream_id: str, output_url: str, http_port: int,
                                         task_manager, quality: int = 80, width: Optional[int] = None, 
                                         height: Optional[int] = None, fps: int = 15) -> Optional[Dict[str, Any]]:
        """
        启动FFmpeg直接输出进程

        Args:
            task_id: 任务ID
            stream_id: 流ID
            output_url: 输出URL
            http_port: HTTP端口
            task_manager: 任务管理器
            quality: 视频质量
            width: 视频宽度
            height: 视频高度
            fps: 视频帧率

        Returns:
            Optional[Dict[str, Any]]: 流信息，如果失败则返回None
        """
        try:
            # 创建FFmpeg直接输出线程
            ffmpeg_thread = threading.Thread(
                target=self._ffmpeg_direct_thread,
                args=(task_id, stream_id, output_url, http_port, task_manager, quality, width, height, fps),
                daemon=True
            )
            ffmpeg_thread.start()

            # 存储线程
            self.encoding_threads[task_id] = ffmpeg_thread

            # 等待一段时间，确保FFmpeg进程启动
            normal_logger.info(f"等待FFmpeg直接输出进程初始化: {task_id}")
            await asyncio.sleep(3)

            # 检查进程是否仍在运行
            if task_id in self.ffmpeg_processes:
                process = self.ffmpeg_processes[task_id]
                if process.poll() is not None:
                    # 进程已退出
                    normal_logger.error(f"FFmpeg直接输出进程已退出，返回码: {process.poll()}")
                    return None

            # 构建流信息
            stream_info = {
                "stream_id": stream_id,
                "stream_type": "ffmpeg",
                "output_url": output_url,
                "http_port": http_port,
                "format": "flv",
                "quality": quality,
                "width": width,
                "height": height,
                "fps": fps,
                "status": "streaming"
            }

            return stream_info

        except Exception as e:
            normal_logger.exception(f"启动FFmpeg直接输出进程失败: {str(e)}")
            return None

    def _ffmpeg_direct_thread(self, task_id: str, stream_id: str, output_url: str, http_port: int,
                            task_manager=None, quality: int = 80, width: Optional[int] = None, 
                            height: Optional[int] = None, fps: int = 15):
        """
        FFmpeg直接输出线程，实时获取分析结果并输出HTTP FLV流

        Args:
            task_id: 任务ID
            stream_id: 流ID
            output_url: 输出URL
            http_port: HTTP端口
            task_manager: 任务管理器
            quality: 视频质量
            width: 视频宽度
            height: 视频高度
            fps: 视频帧率
        """
        # 记录线程启动
        normal_logger.info(f"FFmpeg直接输出线程启动: {task_id}, 输出URL: {output_url}")

        # 在线程中创建新的事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # 运行异步函数
            loop.run_until_complete(self._async_ffmpeg_direct_worker(
                task_id, stream_id, output_url, http_port, task_manager, quality, width, height, fps
            ))
        except Exception as e:
            normal_logger.error(f"FFmpeg直接输出线程发生错误: {str(e)}")
            import traceback
            normal_logger.error(traceback.format_exc())
        finally:
            loop.close()

    async def _async_ffmpeg_direct_worker(self, task_id: str, stream_id: str, output_url: str, http_port: int,
                                        task_manager=None, quality: int = 80, width: Optional[int] = None, 
                                        height: Optional[int] = None, fps: int = 15):
        """
        FFmpeg直接输出异步工作函数

        Args:
            task_id: 任务ID
            stream_id: 流ID
            output_url: 输出URL
            http_port: HTTP端口
            task_manager: 任务管理器
            quality: 视频质量
            width: 视频宽度
            height: 视频高度
            fps: 视频帧率
        """
        try:
            # 获取任务的原始视频流
            task_info = task_manager.get_task(task_id) if task_manager else None
            if not task_info:
                normal_logger.error(f"无法获取任务信息: {task_id}")
                return

            # 获取流配置信息
            task_data = task_info.get("data", {})
            task_params = task_data.get("params", {}) if isinstance(task_data, dict) else {}

            # 获取原始视频流ID和URL
            original_stream_id = task_params.get("video_id") or task_params.get("stream_id") or f"stream_{task_id}"
            stream_url = task_params.get("stream_url", "")

            normal_logger.info(f"FFmpeg直接输出任务 {task_id}: 准备订阅原始视频流 {original_stream_id}, URL: {stream_url}")

            # 订阅原始视频流
            stream_manager = app_state_manager.get_stream_manager()
            if not stream_manager:
                normal_logger.error("流管理器未初始化")
                return

            frame_queue = None
            try:
                # 构建流配置
                stream_config = {
                    "url": stream_url,
                    "rtsp_transport": task_params.get("rtsp_transport", "tcp"),
                    "reconnect_attempts": task_params.get("reconnect_attempts", 3),
                    "reconnect_delay": task_params.get("reconnect_delay", 5),
                    "frame_buffer_size": task_params.get("frame_buffer_size", 100),
                    "task_id": f"{task_id}_ffmpeg_direct",  # 使用不同的任务ID避免冲突
                    "video_id": original_stream_id
                }

                success, frame_queue = await stream_manager.subscribe_stream(
                    original_stream_id,
                    f"{task_id}_ffmpeg_direct",
                    stream_config
                )

                if not success or frame_queue is None:
                    normal_logger.error(f"订阅原始视频流失败: {original_stream_id}")
                    return

                normal_logger.info(f"成功订阅原始视频流: {original_stream_id}")
            except Exception as e:
                normal_logger.error(f"订阅原始视频流时发生异常: {str(e)}")
                return

            # 获取第一帧以确定分辨率
            frame = None
            retry_count = 0
            max_retries = 100

            while frame is None and retry_count < max_retries:
                try:
                    frame_data = await asyncio.wait_for(frame_queue.get(), timeout=1.0)
                    if frame_data and isinstance(frame_data, tuple) and len(frame_data) >= 2:
                        frame, timestamp = frame_data[:2]
                        if frame is not None:
                            normal_logger.info(f"FFmpeg直接输出成功获取原始流帧: {task_id}, 帧形状: {frame.shape}")
                            break
                except asyncio.TimeoutError:
                    pass
                except Exception as e:
                    normal_logger.error(f"获取原始流帧失败: {str(e)}")

                retry_count += 1
                if retry_count % 10 == 0:
                    normal_logger.info(f"FFmpeg直接输出等待原始流帧中: {task_id}, 已重试 {retry_count} 次")

                # 检查任务是否已停止
                if task_id not in self.streaming_tasks:
                    normal_logger.info(f"FFmpeg直接输出任务已停止，线程退出: {task_id}")
                    return

            # 如果仍然没有获取到帧，使用默认帧
            if frame is None:
                normal_logger.warning(f"无法获取原始流帧，使用默认帧: {task_id}")
                frame = self.create_default_frame(640, 480, "等待视频流...")

            # 确定视频分辨率
            frame_height, frame_width = frame.shape[:2]
            final_width = width if width is not None else frame_width
            final_height = height if height is not None else frame_height

            normal_logger.info(f"FFmpeg直接输出参数 - 任务ID: {task_id}, 分辨率: {final_width}x{final_height}, 帧率: {fps}, 质量: {quality}, HTTP端口: {http_port}")

            # 计算比特率
            bitrate = self.ffmpeg_params.calculate_optimal_bitrate(quality, final_width, final_height, fps)
            normal_logger.info(f"FFmpeg直接输出计算得出最优比特率: {bitrate}k")

            # 构建FFmpeg命令 - 使用HTTP输出
            ffmpeg_cmd = [
                "ffmpeg",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{final_width}x{final_height}",
                "-r", str(fps),
                "-i", "-",  # 从stdin读取
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-tune", "zerolatency",
                "-b:v", f"{bitrate}k",
                "-maxrate", f"{int(bitrate * 1.2)}k",
                "-bufsize", f"{int(bitrate * 2)}k",
                "-g", str(fps * 2),  # GOP size
                "-pix_fmt", "yuv420p",
                "-f", "flv",
                "-listen", "1",  # 启用HTTP服务器模式
                f"http://0.0.0.0:{http_port}/live.flv"
            ]

            normal_logger.info(f"FFmpeg直接输出命令: {' '.join(ffmpeg_cmd)}")

            # 启动FFmpeg进程
            try:
                process = subprocess.Popen(
                    ffmpeg_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=10*1024*1024
                )

                # 存储进程
                self.ffmpeg_processes[task_id] = process

                # 主循环 - 实时处理原始视频流帧
                frame_interval = 1.0 / fps
                last_frame_time = time.time()
                frame_count = 0
                error_count = 0
                max_errors = 5

                # 记录任务状态并添加到帧上
                task_status = task_info.get("status", "未知")
                task_start_time = datetime.now()

                while task_id in self.streaming_tasks:
                    # 检查FFmpeg进程是否仍在运行
                    if process.poll() is not None:
                        normal_logger.error(f"FFmpeg直接输出进程意外退出，返回码: {process.poll()}, 任务ID: {task_id}")
                        break

                    # 控制帧率
                    current_time = time.time()
                    elapsed = current_time - last_frame_time
                    if elapsed < frame_interval:
                        await asyncio.sleep(frame_interval - elapsed)
                    last_frame_time = time.time()

                    # 获取原始视频流帧
                    current_frame = None
                    try:
                        frame_data = await asyncio.wait_for(frame_queue.get(), timeout=0.1)
                        if frame_data and isinstance(frame_data, tuple) and len(frame_data) >= 2:
                            current_frame, timestamp = frame_data[:2]
                    except asyncio.TimeoutError:
                        # 超时，使用上一帧或默认帧
                        pass
                    except Exception as e:
                        normal_logger.error(f"获取原始流帧失败: {str(e)}")

                    # 如果没有获取到有效帧，使用默认帧
                    if current_frame is None:
                        error_count += 1
                        if error_count > max_errors:
                            current_frame = self.create_default_frame(final_width, final_height, "视频流中断...")
                            error_count = 0
                        else:
                            continue
                    else:
                        error_count = 0

                    # 调整帧大小
                    if current_frame.shape[1] != final_width or current_frame.shape[0] != final_height:
                        current_frame = cv2.resize(current_frame, (final_width, final_height))

                    # 获取最新的分析结果并渲染在帧上
                    latest_analysis_result = self.get_cached_analysis_result(task_id)
                    
                    # 使用优化的渲染组件
                    rendered_frame = self._render_frame_optimized(current_frame, latest_analysis_result, task_id)

                    # 计算任务运行时间
                    elapsed_time = (datetime.now() - task_start_time).total_seconds()

                    # 每10秒更新一次任务状态
                    if int(elapsed_time) % 10 == 0:
                        try:
                            if task_manager and task_manager.has_task(task_id):
                                updated_task_info = task_manager.get_task(task_id)
                                if updated_task_info:
                                    task_status = updated_task_info.get("status", task_status)
                        except Exception as e:
                            normal_logger.debug(f"更新任务状态时出错: {str(e)}")

                    # 添加状态信息到帧上
                    rendered_frame = optimized_renderer.add_status_info(rendered_frame, task_status, elapsed_time)

                    # 每100帧记录一次日志
                    frame_count += 1
                    if frame_count % 100 == 0:
                        normal_logger.info(f"FFmpeg直接输出线程已处理 {frame_count} 帧: {task_id}")

                    # 写入帧数据到FFmpeg
                    try:
                        if process.poll() is None:  # 确保进程仍在运行
                            process.stdin.write(rendered_frame.tobytes())
                        else:
                            break
                    except BrokenPipeError:
                        normal_logger.error(f"FFmpeg直接输出写入帧数据时管道已断开: {task_id}")
                        break
                    except Exception as e:
                        normal_logger.error(f"FFmpeg直接输出写入帧数据时出错: {str(e)}")
                        break

                # 关闭FFmpeg输入流
                if process.stdin:
                    try:
                        process.stdin.close()
                    except Exception as e:
                        normal_logger.error(f"关闭FFmpeg输入流时出错: {str(e)}")

                # 等待FFmpeg进程完成
                try:
                    process.wait(timeout=30)
                    normal_logger.info(f"FFmpeg直接输出进程已完成: {task_id}, 返回码: {process.returncode}")
                except subprocess.TimeoutExpired:
                    normal_logger.warning(f"等待FFmpeg直接输出进程完成超时: {task_id}")
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()

            except Exception as e:
                normal_logger.error(f"FFmpeg直接输出进程运行时出错: {str(e)}")
                import traceback
                normal_logger.error(traceback.format_exc())

            finally:
                # 取消订阅原始视频流
                try:
                    await stream_manager.unsubscribe_stream(original_stream_id, f"{task_id}_ffmpeg_direct")
                    normal_logger.info(f"已取消订阅原始视频流: {original_stream_id}")
                except Exception as e:
                    normal_logger.error(f"取消订阅原始视频流失败: {str(e)}")

                # 清理流队列
                if frame_queue:
                    try:
                        while not frame_queue.empty():
                            try:
                                frame_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                break
                        normal_logger.debug(f"已清空帧队列: {task_id}")
                    except Exception as e:
                        normal_logger.error(f"清空帧队列失败: {str(e)}")

                # 记录线程结束日志
                normal_logger.info(f"FFmpeg直接输出线程清理完成并正常退出: {task_id}")

        except Exception as e:
            normal_logger.error(f"FFmpeg直接输出工作函数发生错误: {str(e)}")
            import traceback
            normal_logger.error(traceback.format_exc())

    def _render_frame_optimized(self, frame: np.ndarray, analysis_result: Optional[Dict[str, Any]], 
                              task_id: str) -> np.ndarray:
        """
        使用优化组件渲染帧
        
        Args:
            frame: 原始帧
            analysis_result: 分析结果
            task_id: 任务ID
            
        Returns:
            np.ndarray: 渲染后的帧
        """
        try:
            # 1. 检查是否应该跳过渲染
            if smart_frame_skipper.should_skip_frame(analysis_result):
                # 如果跳过，尝试从缓存获取上一次的渲染结果
                cached_frame = frame_cache.get(frame, analysis_result)
                if cached_frame is not None:
                    return cached_frame
                # 如果缓存中没有，直接返回原帧
                return frame
            
            # 2. 尝试从缓存获取渲染结果
            cached_frame = frame_cache.get(frame, analysis_result)
            if cached_frame is not None:
                return cached_frame
            
            # 3. 使用优化渲染器渲染
            rendered_frame = optimized_renderer.render_analysis_results_optimized(frame, analysis_result)
            
            # 4. 将结果放入缓存
            frame_cache.put(frame, analysis_result, rendered_frame)
            
            # 5. 定期输出性能统计
            frame_count = getattr(self, '_perf_frame_count', 0) + 1
            setattr(self, '_perf_frame_count', frame_count)
            
            if frame_count % 100 == 0:
                # 输出性能统计
                render_stats = optimized_renderer.get_performance_stats()
                cache_stats = frame_cache.get_stats()
                skip_stats = smart_frame_skipper.get_stats()
                
                normal_logger.info(f"任务 {task_id} 性能统计 (第{frame_count}帧):")
                normal_logger.info(f"  渲染: 平均{render_stats['avg_render_time_ms']:.2f}ms, "
                                 f"最大{render_stats['max_render_time_ms']:.2f}ms")
                normal_logger.info(f"  缓存: 命中率{cache_stats['hit_rate_percent']}%, "
                                 f"大小{cache_stats['cache_size']}/{cache_stats['max_size']}")
                normal_logger.info(f"  跳过: 跳过率{skip_stats['skip_rate_percent']}%, "
                                 f"稳定性{'是' if skip_stats['current_stability'] else '否'}")
            
            return rendered_frame
            
        except Exception as e:
            normal_logger.error(f"优化渲染失败: {str(e)}")
            # 回退到优化渲染器的基本方法
            return optimized_renderer.render_analysis_results(frame, analysis_result)

    def _cleanup_optimized_components(self, task_id: str):
        """
        清理任务相关的优化组件状态
        
        Args:
            task_id: 任务ID
        """
        try:
            # 清理缓存
            frame_cache.clear()
            
            # 重置智能跳过器
            smart_frame_skipper.reset()
            
            # 重置性能计数器
            if hasattr(self, '_perf_frame_count'):
                delattr(self, '_perf_frame_count')
            
            normal_logger.info(f"任务 {task_id} 优化组件状态已清理")
            
        except Exception as e:
            normal_logger.error(f"清理优化组件状态失败: {str(e)}")

    def set_performance_mode(self, mode: str):
        """
        设置性能模式
        
        Args:
            mode: 性能模式 - "high_quality", "balanced", "high_performance"
        """
        try:
            # 转换字符串到枚举
            mode_enum = PerformanceMode(mode)
            success = performance_config.set_performance_mode(mode_enum)
            
            if success:
                normal_logger.info(f"直播流性能模式设置为: {mode}")
            else:
                normal_logger.error(f"设置直播流性能模式失败: {mode}")
                
        except ValueError:
            normal_logger.error(f"无效的性能模式: {mode}, 支持的模式: high_quality, balanced, high_performance")
        except Exception as e:
            normal_logger.error(f"设置性能模式时发生错误: {str(e)}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        获取性能统计信息
        
        Returns:
            Dict[str, Any]: 性能统计信息
        """
        return performance_config.get_performance_stats()

    def reset_performance_stats(self):
        """重置性能统计信息"""
        performance_config.reset_all_stats()

    def create_default_frame(self, width: int, height: int, text: str) -> np.ndarray:
        """
        创建默认帧
        
        Args:
            width: 帧宽度
            height: 帧高度
            text: 显示文本
            
        Returns:
            np.ndarray: 默认帧
        """
        # 创建黑色背景
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 添加文本
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        color = (255, 255, 255)
        thickness = 2
        
        # 计算文本位置（居中）
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        
        # 绘制文本
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)
        
        return frame

    def get_cached_analysis_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        获取缓存的分析结果
        
        Args:
            task_id: 任务ID
            
        Returns:
            Optional[Dict[str, Any]]: 缓存的分析结果
        """
        if not hasattr(self, '_analysis_result_cache'):
            self._analysis_result_cache = {}
        
        return self._analysis_result_cache.get(task_id)

    def clear_analysis_result_cache(self, task_id: str):
        """
        清理分析结果缓存
        
        Args:
            task_id: 任务ID
        """
        if hasattr(self, '_analysis_result_cache'):
            self._analysis_result_cache.pop(task_id, None)
        
        # 清理优化组件状态
        self._cleanup_optimized_components(task_id)

    def update_analysis_result(self, task_id: str, analysis_result: Dict[str, Any]):
        """
        被动接收分析结果的方法（重写基类方法）
        
        Args:
            task_id: 任务ID
            analysis_result: 分析结果字典
        """
        # 更新自己的缓存
        self.update_analysis_result_cache(task_id, analysis_result)
        
        # 同时调用基类方法保持兼容性
        super().update_analysis_result(task_id, analysis_result)

    def update_analysis_result_cache(self, task_id: str, analysis_result: Dict[str, Any]):
        """
        更新分析结果缓存
        
        Args:
            task_id: 任务ID
            analysis_result: 新的分析结果
        """
        if not hasattr(self, '_analysis_result_cache'):
            self._analysis_result_cache = {}
        
        self._analysis_result_cache[task_id] = analysis_result 