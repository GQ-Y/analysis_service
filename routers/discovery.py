"""
设备发现API
提供ONVIF设备发现功能
"""

import asyncio
from typing import List, Dict, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from shared.utils.logger import get_normal_logger
from services.discovery_service import discovery_service

# 初始化日志记录器
logger = get_normal_logger(__name__)

router = APIRouter(
    prefix="/api/v1/discovery",
    tags=["设备发现"],
    responses={404: {"description": "Not found"}},
)

# 请求和响应模型
class ONVIFDiscoveryRequest(BaseModel):
    interface_ip: Optional[str] = Field(None, description="网络接口IP，为空时自动检测")
    timeout: int = Field(10, description="发现超时时间（秒）", ge=5, le=60)

class ONVIFDeviceInfo(BaseModel):
    ip: str = Field(..., description="设备IP地址")
    name: str = Field(..., description="设备名称")
    manufacturer: str = Field(..., description="制造商")
    model: str = Field(..., description="设备型号")
    endpoints: List[str] = Field(..., description="服务端点列表")
    scopes: List[str] = Field(..., description="设备范围信息")

class ONVIFTestRequest(BaseModel):
    ip: str = Field(..., description="设备IP地址")
    username: str = Field("admin", description="用户名")
    password: str = Field("", description="密码")

@router.post("/onvif/discover", response_model=List[ONVIFDeviceInfo])
async def discover_onvif_devices(request: ONVIFDiscoveryRequest):
    """
    发现局域网中的ONVIF设备
    """
    try:
        logger.info(f"开始ONVIF设备发现，接口IP: {request.interface_ip}, 超时: {request.timeout}秒")
        
        # 使用设备发现服务
        devices = await discovery_service.discover_onvif_devices(
            interface_ip=request.interface_ip,
            timeout=request.timeout
        )
        
        logger.info(f"发现 {len(devices)} 个ONVIF设备")
        
        # 转换为响应模型
        result = []
        for device in devices:
            result.append(ONVIFDeviceInfo(
                ip=device['ip'],
                name=device.get('name', 'Unknown Device'),
                manufacturer=device.get('manufacturer', 'Unknown'),
                model=device.get('model', 'Unknown'),
                endpoints=device.get('endpoints', []),
                scopes=device.get('scopes', [])
            ))
        
        return result
        
    except Exception as e:
        logger.error(f"ONVIF设备发现失败: {e}")
        raise HTTPException(status_code=500, detail=f"设备发现失败: {str(e)}")

@router.post("/onvif/test")
async def test_onvif_device(request: ONVIFTestRequest):
    """
    测试ONVIF设备功能
    """
    try:
        logger.info(f"开始测试ONVIF设备: {request.ip}")
        
        # 使用设备发现服务
        result = await discovery_service.test_onvif_device(
            device_ip=request.ip,
            username=request.username,
            password=request.password
        )
        
        logger.info(f"ONVIF设备测试完成: {request.ip}, 成功: {result['onvif_available']}")
        
        return result
        
    except Exception as e:
        logger.error(f"ONVIF设备测试失败: {e}")
        raise HTTPException(status_code=500, detail=f"设备测试失败: {str(e)}") 