"""
任务视频编码路由
提供任务的视频直播流功能，支持RTMP、HLS、FLV格式
"""
from fastapi import APIRouter, Depends, HTTPException, Path, Request, Response, Query
from fastapi.responses import FileResponse, HTMLResponse
import uuid
import os
import asyncio
from typing import Dict, Any, Optional, Literal

from models.requests import VideoEncodingRequest
from models.responses import BaseResponse
from services.http.task_service import TaskService
from services.video.video_service import VideoService
from core.task_management.utils.status import TaskStatus
from core.config import settings
from shared.utils.logger import get_normal_logger, get_exception_logger

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)

# 创建路由
router = APIRouter(
    prefix="/api/v1/tasks",
    tags=["视频直播流"],
    responses={404: {"description": "Not found"}},
)

# 依赖注入
async def get_task_service(request: Request) -> TaskService:
    """获取任务服务实例"""
    if not hasattr(request.app.state, "task_service"):
        raise HTTPException(status_code=500, detail="任务服务未初始化")
    return request.app.state.task_service

async def get_video_service(request: Request) -> VideoService:
    """获取视频服务实例"""
    if not hasattr(request.app.state, "video_service"):
        # 如果视频服务不存在，创建一个
        request.app.state.video_service = VideoService()
    return request.app.state.video_service

@router.post("/video", response_model=BaseResponse, summary="启动实时分析视频直播流")
async def start_task_video_stream(
    encoding_request: VideoEncodingRequest,
    task_service: TaskService = Depends(get_task_service),
    video_service: VideoService = Depends(get_video_service)
) -> BaseResponse:
    """
    启动实时分析视频直播流
    
    使用ffmpeg将分析结果和识别目标合成，然后通过ZLMediaKit构建直播流播放地址。
    系统将实时分析的结果绘制在视频上，并推送到ZLM服务器作为直播流。
    
    支持的参数包括：
    - task_id: 任务ID，必填
    - enable_encoding: 是否开启编码，默认为true
    - format: 视频格式，支持"rtmp"、"hls"、"flv"，默认为"rtmp"
    - quality: 视频质量(1-100)，默认80
    - width: 视频宽度，为空则使用原始宽度
    - height: 视频高度，为空则使用原始高度
    - fps: 视频帧率，默认15
    
    Args:
        encoding_request: 编码请求参数
        
    Returns:
        BaseResponse: 响应结果，包含直播流播放地址
    """
    request_id = str(uuid.uuid4())
    try:
        # 获取任务管理器
        task_manager = task_service.task_manager
        
        # 检查任务是否存在
        if not task_manager.has_task(encoding_request.task_id):
            return BaseResponse(
                requestId=request_id,
                path="/api/v1/tasks/video",
                success=False,
                message=f"任务不存在: {encoding_request.task_id}",
                code=404,
                data=None
            )
        
        # 根据请求开启或关闭编码
        if encoding_request.enable_encoding:
            # 开启直播流编码，支持两种推流模式
            result = await video_service.start_live_stream(
                task_id=encoding_request.task_id,
                task_manager=task_manager,
                format=encoding_request.format or "rtmp",  # 默认使用RTMP格式
                quality=encoding_request.quality,
                width=encoding_request.width,
                height=encoding_request.height,
                fps=encoding_request.fps,
                stream_type=encoding_request.stream_type  # 传递推流类型参数
            )
            
            if not result["success"]:
                return BaseResponse(
                    requestId=request_id,
                    path="/api/v1/tasks/video",
                    success=False,
                    message=result["message"],
                    code=500,
                    data=None
                )
            
            return BaseResponse(
                requestId=request_id,
                path="/api/v1/tasks/video",
                success=True,
                message="视频直播流已启动",
                code=200,
                data={
                    "task_id": encoding_request.task_id,
                    "stream_info": result["stream_info"],
                    "play_urls": result["play_urls"]
                }
            )
        else:
            # 关闭直播流编码
            result = await video_service.stop_live_stream(encoding_request.task_id)
            
            if not result["success"]:
                return BaseResponse(
                    requestId=request_id,
                    path="/api/v1/tasks/video",
                    success=False,
                    message=result["message"],
                    code=400,
                    data=None
                )
            
            return BaseResponse(
                requestId=request_id,
                path="/api/v1/tasks/video",
                success=True,
                message="视频直播流已关闭",
                code=200,
                data=None
            )
    
    except Exception as e:
        exception_logger.exception(f"处理视频直播流请求失败: {str(e)}")
        return BaseResponse(
            requestId=request_id,
            path="/api/v1/tasks/video",
            success=False,
            message=f"处理视频直播流请求失败: {str(e)}",
            code=500,
            data=None
        )

@router.post("/video/file", response_model=BaseResponse, summary="启动实时分析视频文件编码")
async def start_task_video_file(
    encoding_request: VideoEncodingRequest,
    task_service: TaskService = Depends(get_task_service),
    video_service: VideoService = Depends(get_video_service)
) -> BaseResponse:
    """
    启动实时分析视频文件编码
    
    使用ffmpeg将分析结果和识别目标合成，生成MP4或FLV视频文件。
    系统将实时分析的结果绘制在视频上，并编码为视频文件。
    
    支持的参数包括：
    - task_id: 任务ID，必填
    - enable_encoding: 是否开启编码，默认为true
    - format: 视频格式，支持"mp4"、"flv"，默认为"mp4"
    - quality: 视频质量(1-100)，默认80
    - width: 视频宽度，为空则使用原始宽度
    - height: 视频高度，为空则使用原始高度
    - fps: 视频帧率，默认15
    
    Args:
        encoding_request: 编码请求参数
        
    Returns:
        BaseResponse: 响应结果，包含视频文件URL
    """
    request_id = str(uuid.uuid4())
    try:
        # 获取任务管理器
        task_manager = task_service.task_manager
        
        # 检查任务是否存在
        if not task_manager.has_task(encoding_request.task_id):
            return BaseResponse(
                requestId=request_id,
                path="/api/v1/tasks/video/file",
                success=False,
                message=f"任务不存在: {encoding_request.task_id}",
                code=404,
                data=None
            )
        
        # 根据请求开启或关闭编码
        if encoding_request.enable_encoding:
            # 开启文件编码
            result = await video_service.start_encoding(
                task_id=encoding_request.task_id,
                task_manager=task_manager,
                format=encoding_request.format or "mp4",  # 默认使用MP4格式
                quality=encoding_request.quality,
                width=encoding_request.width,
                height=encoding_request.height,
                fps=encoding_request.fps
            )
            
            if not result["success"]:
                return BaseResponse(
                    requestId=request_id,
                    path="/api/v1/tasks/video/file",
                    success=False,
                    message=result["message"],
                    code=500,
                    data=None
                )
            
            return BaseResponse(
                requestId=request_id,
                path="/api/v1/tasks/video/file",
                success=True,
                message="视频文件编码已启动",
                code=200,
                data={
                    "task_id": encoding_request.task_id,
                    "video_url": result["video_url"]
                }
            )
        else:
            # 关闭文件编码
            result = await video_service.stop_encoding(encoding_request.task_id)
            
            if not result["success"]:
                return BaseResponse(
                    requestId=request_id,
                    path="/api/v1/tasks/video/file",
                    success=False,
                    message=result["message"],
                    code=400,
                    data=None
                )
            
            return BaseResponse(
                requestId=request_id,
                path="/api/v1/tasks/video/file",
                success=True,
                message="视频文件编码已关闭",
                code=200,
                data=None
            )
    
    except Exception as e:
        exception_logger.exception(f"处理视频文件编码请求失败: {str(e)}")
        return BaseResponse(
            requestId=request_id,
            path="/api/v1/tasks/video/file",
            success=False,
            message=f"处理视频文件编码请求失败: {str(e)}",
            code=500,
            data=None
        )

@router.post("/video/performance/mode", response_model=BaseResponse, summary="设置直播流性能模式")
async def set_video_performance_mode(
    mode: Literal["high_quality", "balanced", "high_performance"] = Query(
        ...,
        description="性能模式：high_quality=高质量模式（最佳视觉效果，性能较低）, balanced=平衡模式（推荐，质量和性能平衡）, high_performance=高性能模式（最佳性能，质量较低）",
        example="balanced"
    ),
    video_service: VideoService = Depends(get_video_service)
) -> BaseResponse:
    """
    设置直播流性能模式
    
    性能模式说明：
    - **high_quality**: 高质量模式
      - ✅ 启用中文字体渲染
      - ✅ 显示调试信息  
      - ✅ 渲染所有检测目标(最多100个)
      - ⏱️ 缓存TTL: 1秒，跳过比例: 10%
      
    - **balanced**: 平衡模式（默认推荐）
      - ✅ 启用文字渲染(英文字体)
      - ❌ 关闭调试信息
      - ✅ 渲染主要检测目标(最多50个)
      - ⏱️ 缓存TTL: 3秒，跳过比例: 30%
      
    - **high_performance**: 高性能模式
      - ❌ 关闭文字渲染
      - ❌ 关闭调试信息
      - ✅ 仅渲染高置信度目标(最多30个)
      - ⏱️ 缓存TTL: 5秒，跳过比例: 50%
    
    Args:
        mode: 性能模式
        
    Returns:
        BaseResponse: 设置结果
    """
    request_id = str(uuid.uuid4())
    try:
        result = video_service.set_performance_mode(mode)
        
        if result["success"]:
            return BaseResponse(
                requestId=request_id,
                path="/api/v1/tasks/video/performance/mode",
                success=True,
                message=result["message"],
                code=200,
                data={
                    "current_mode": result["current_mode"],
                    "mode_description": {
                        "high_quality": "高质量模式 - 最佳视觉效果，支持中文字体，性能较低",
                        "balanced": "平衡模式 - 推荐使用，质量和性能平衡",
                        "high_performance": "高性能模式 - 最佳性能，基础渲染，质量较低"
                    }.get(mode, "未知模式")
                }
            )
        else:
            return BaseResponse(
                requestId=request_id,
                path="/api/v1/tasks/video/performance/mode",
                success=False,
                message=result["message"],
                code=400,
                data=None
            )
    
    except Exception as e:
        exception_logger.exception(f"设置视频性能模式失败: {str(e)}")
        return BaseResponse(
            requestId=request_id,
            path="/api/v1/tasks/video/performance/mode",
            success=False,
            message=f"设置视频性能模式失败: {str(e)}",
            code=500,
            data=None
        )

@router.get("/video/performance/stats", response_model=BaseResponse, summary="获取直播流性能统计")
async def get_video_performance_stats(
    video_service: VideoService = Depends(get_video_service)
) -> BaseResponse:
    """
    获取直播流性能统计信息
    
    返回的统计信息包括：
    - 当前性能模式
    - 帧缓存统计（命中率、缓存大小等）
    - 智能帧跳过统计（跳过率、稳定性等）
    - 渲染器统计（平均渲染时间、最大渲染时间等）
    
    Returns:
        BaseResponse: 性能统计信息
    """
    request_id = str(uuid.uuid4())
    try:
        result = video_service.get_performance_stats()
        
        if result["success"]:
            # 增强统计信息的可读性
            stats = result["stats"]
            enhanced_stats = {
                "current_mode": stats.get("current_mode", "unknown"),
                "mode_description": {
                    "high_quality": "高质量模式",
                    "balanced": "平衡模式", 
                    "high_performance": "高性能模式"
                }.get(stats.get("current_mode", ""), "未知模式"),
                "components": {},
                "summary": {}
            }
            
            # 处理组件统计
            components = stats.get("components", {})
            
            # 帧缓存统计
            if "frame_cache" in components:
                cache_stats = components["frame_cache"]
                enhanced_stats["components"]["frame_cache"] = {
                    "name": "帧缓存器",
                    "hit_rate_percent": cache_stats.get("hit_rate_percent", 0),
                    "cache_size": cache_stats.get("cache_size", 0),
                    "max_size": cache_stats.get("max_size", 0),
                    "total_requests": cache_stats.get("total_requests", 0),
                    "cache_hits": cache_stats.get("cache_hits", 0),
                    "cache_misses": cache_stats.get("cache_misses", 0)
                }
            
            # 智能帧跳过统计
            if "frame_skipper" in components:
                skipper_stats = components["frame_skipper"]
                enhanced_stats["components"]["frame_skipper"] = {
                    "name": "智能帧跳过器",
                    "skip_rate_percent": skipper_stats.get("skip_rate_percent", 0),
                    "current_stability": skipper_stats.get("current_stability", False),
                    "total_frames": skipper_stats.get("total_frames", 0),
                    "skipped_frames": skipper_stats.get("skipped_frames", 0),
                    "rendered_frames": skipper_stats.get("rendered_frames", 0)
                }
            
            # 渲染器统计
            if "renderer" in components:
                render_stats = components["renderer"]
                enhanced_stats["components"]["renderer"] = {
                    "name": "优化帧渲染器",
                    "avg_render_time_ms": round(render_stats.get("avg_render_time_ms", 0), 2),
                    "max_render_time_ms": round(render_stats.get("max_render_time_ms", 0), 2),
                    "min_render_time_ms": round(render_stats.get("min_render_time_ms", 0), 2),
                    "render_count": render_stats.get("render_count", 0)
                }
            
            # 生成性能总结
            cache_hit_rate = enhanced_stats["components"].get("frame_cache", {}).get("hit_rate_percent", 0)
            skip_rate = enhanced_stats["components"].get("frame_skipper", {}).get("skip_rate_percent", 0)
            avg_render_time = enhanced_stats["components"].get("renderer", {}).get("avg_render_time_ms", 0)
            
            enhanced_stats["summary"] = {
                "performance_level": "优秀" if avg_render_time < 1.0 and cache_hit_rate > 70 else 
                                   "良好" if avg_render_time < 2.0 and cache_hit_rate > 50 else "一般",
                "cache_efficiency": "高效" if cache_hit_rate > 70 else "中等" if cache_hit_rate > 30 else "低效",
                "render_speed": "快速" if avg_render_time < 1.0 else "中等" if avg_render_time < 3.0 else "慢速",
                "optimization_tips": []
            }
            
            # 生成优化建议
            if cache_hit_rate < 50:
                enhanced_stats["summary"]["optimization_tips"].append("缓存命中率较低，建议检查是否存在频繁变化的检测结果")
            if avg_render_time > 3.0:
                enhanced_stats["summary"]["optimization_tips"].append("渲染时间较长，建议切换到高性能模式")
            if skip_rate < 20:
                enhanced_stats["summary"]["optimization_tips"].append("帧跳过率较低，可以考虑调整稳定性阈值")
            
            return BaseResponse(
                requestId=request_id,
                path="/api/v1/tasks/video/performance/stats",
                success=True,
                message="性能统计获取成功",
                code=200,
                data=enhanced_stats
            )
        else:
            return BaseResponse(
                requestId=request_id,
                path="/api/v1/tasks/video/performance/stats",
                success=False,
                message=result["message"],
                code=500,
                data=None
            )
    
    except Exception as e:
        exception_logger.exception(f"获取视频性能统计失败: {str(e)}")
        return BaseResponse(
            requestId=request_id,
            path="/api/v1/tasks/video/performance/stats",
            success=False,
            message=f"获取视频性能统计失败: {str(e)}",
            code=500,
            data=None
        )

@router.post("/video/performance/reset", response_model=BaseResponse, summary="重置直播流性能统计")
async def reset_video_performance_stats(
    video_service: VideoService = Depends(get_video_service)
) -> BaseResponse:
    """
    重置直播流性能统计信息
    
    此操作会清空所有性能统计数据，包括：
    - 帧缓存统计
    - 智能帧跳过统计
    - 渲染器统计
    
    注意：此操作不会影响当前的性能模式设置
    
    Returns:
        BaseResponse: 重置结果
    """
    request_id = str(uuid.uuid4())
    try:
        result = video_service.reset_performance_stats()
        
        if result["success"]:
            return BaseResponse(
                requestId=request_id,
                path="/api/v1/tasks/video/performance/reset",
                success=True,
                message=result["message"],
                code=200,
                data={
                    "reset_time": "性能统计已重置",
                    "note": "性能模式设置保持不变"
                }
            )
        else:
            return BaseResponse(
                requestId=request_id,
                path="/api/v1/tasks/video/performance/reset",
                success=False,
                message=result["message"],
                code=500,
                data=None
            )
    
    except Exception as e:
        exception_logger.exception(f"重置视频性能统计失败: {str(e)}")
        return BaseResponse(
            requestId=request_id,
            path="/api/v1/tasks/video/performance/reset",
            success=False,
            message=f"重置视频性能统计失败: {str(e)}",
            code=500,
            data=None
        )

@router.get("/video/performance/modes", response_model=BaseResponse, summary="获取所有可用的性能模式信息")
async def get_video_performance_modes() -> BaseResponse:
    """
    获取所有可用的直播流性能模式信息
    
    返回详细的性能模式说明，帮助用户选择合适的模式
    
    Returns:
        BaseResponse: 性能模式信息
    """
    request_id = str(uuid.uuid4())
    try:
        modes_info = {
            "available_modes": {
                "high_quality": {
                    "name": "高质量模式",
                    "description": "最佳视觉效果，支持中文字体渲染",
                    "features": {
                        "chinese_fonts": True,
                        "debug_info": True,
                        "max_detections": 100,
                        "text_rendering": True
                    },
                    "performance": {
                        "cache_ttl_seconds": 1,
                        "skip_ratio_percent": 10,
                        "render_quality": "最高",
                        "cpu_usage": "高",
                        "memory_usage": "高"
                    },
                    "use_cases": [
                        "需要高质量视觉效果的场景",
                        "需要显示中文标签的场景",
                        "对性能要求不高的场景",
                        "演示或展示用途"
                    ]
                },
                "balanced": {
                    "name": "平衡模式",
                    "description": "推荐使用，质量和性能的最佳平衡",
                    "features": {
                        "chinese_fonts": False,
                        "debug_info": False,
                        "max_detections": 50,
                        "text_rendering": True
                    },
                    "performance": {
                        "cache_ttl_seconds": 3,
                        "skip_ratio_percent": 30,
                        "render_quality": "高",
                        "cpu_usage": "中等",
                        "memory_usage": "中等"
                    },
                    "use_cases": [
                        "大多数生产环境",
                        "需要标签显示但对中文要求不高",
                        "中等性能要求的场景",
                        "长时间运行的监控系统"
                    ]
                },
                "high_performance": {
                    "name": "高性能模式",
                    "description": "最佳性能，基础渲染功能",
                    "features": {
                        "chinese_fonts": False,
                        "debug_info": False,
                        "max_detections": 30,
                        "text_rendering": False
                    },
                    "performance": {
                        "cache_ttl_seconds": 5,
                        "skip_ratio_percent": 50,
                        "render_quality": "基础",
                        "cpu_usage": "低",
                        "memory_usage": "低"
                    },
                    "use_cases": [
                        "高负载场景",
                        "资源受限的环境",
                        "只需要目标框不需要标签的场景",
                        "实时性要求极高的场景"
                    ]
                }
            },
            "default_mode": "balanced",
            "performance_tips": [
                "大多数情况下推荐使用平衡模式",
                "如果出现卡顿，可切换到高性能模式",
                "如果需要中文标签，使用高质量模式",
                "可以通过性能统计API监控实际效果"
            ]
        }
        
        return BaseResponse(
            requestId=request_id,
            path="/api/v1/tasks/video/performance/modes",
            success=True,
            message="性能模式信息获取成功",
            code=200,
            data=modes_info
        )
    
    except Exception as e:
        exception_logger.exception(f"获取性能模式信息失败: {str(e)}")
        return BaseResponse(
            requestId=request_id,
            path="/api/v1/tasks/video/performance/modes",
            success=False,
            message=f"获取性能模式信息失败: {str(e)}",
            code=500,
            data=None
        )
