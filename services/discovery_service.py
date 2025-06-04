"""
设备发现服务
提供网络设备发现功能，特别是ONVIF设备
"""

import asyncio
import socket
import time
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Any
import re
import struct
import aiohttp
import urllib.parse

from shared.utils.logger import get_normal_logger, get_exception_logger

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)

# 检查ONVIF库可用性
ONVIF_AVAILABLE = False
try:
    from onvif import ONVIFCamera
    ONVIF_AVAILABLE = True
except ImportError:
    normal_logger.warning("onvif-zeep库未安装，ONVIF功能将受限")
    ONVIFCamera = None

class WsDiscoveryService:
    """WS-Discovery服务，用于发现网络设备"""
    
    # WS-Discovery相关常量
    WS_DISCOVERY_MULTICAST_IP = "239.255.255.250"
    WS_DISCOVERY_PORTS = [3702, 80, 3002]  # 支持标准端口和天地伟业的80端口
    ONVIF_DEVICE_TYPE = "dn:NetworkVideoTransmitter"
    
    def __init__(self):
        self.timeout = 10
        self.max_devices = 50
        self.reachable_ips = set()
    
    async def discover_onvif_devices(self, interface_ip: Optional[str] = None, timeout: int = 10) -> List[Dict[str, Any]]:
        """
        发现局域网中的ONVIF设备
        
        Args:
            interface_ip: 网络接口IP，为空时自动检测
            timeout: 发现超时时间（秒）
            
        Returns:
            List[Dict[str, Any]]: 发现的设备列表
        """
        self.timeout = timeout
        devices = []
        seen_devices = set()
        device_responses = {}  # 存储设备响应
        
        try:
            # 构建WS-Discovery Probe消息
            probe_msg = self._create_probe_message()
            normal_logger.info(f"发送的探测消息内容:\n{probe_msg}")
            
            # 创建UDP套接字
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # 设置组播TTL
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 4)
            
            # 允许组播环回
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
            
            # 设置接收超时
            sock.settimeout(1)
            
            # 绑定到所有接口
            sock.bind(('0.0.0.0', 0))
            
            # 加入组播组
            mreq = struct.pack("4s4s", socket.inet_aton(self.WS_DISCOVERY_MULTICAST_IP),
                             socket.inet_aton('0.0.0.0'))
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            
            normal_logger.info(f"开始WS-Discovery设备发现，超时: {timeout}秒")
            
            # 多次发送探测消息以提高成功率
            send_count = 0
            last_send_time = 0
            
            # 收集响应
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                # 每3秒重新发送一次探测消息
                current_time = time.time()
                if current_time - last_send_time >= 3 and send_count < 5:
                    for port in self.WS_DISCOVERY_PORTS:
                        normal_logger.info(f"尝试在端口 {port} 发送探测消息")
                        for _ in range(3):
                            try:
                                sock.sendto(probe_msg.encode('utf-8'), 
                                          (self.WS_DISCOVERY_MULTICAST_IP, port))
                                await asyncio.sleep(0.1)
                            except Exception as e:
                                normal_logger.warning(f"发送探测消息到端口 {port} 失败: {e}")
                    
                    last_send_time = current_time
                    send_count += 1
                    normal_logger.info(f"发送第 {send_count} 轮探测消息")
                
                try:
                    data, addr = sock.recvfrom(8192)
                    device_ip = addr[0]
                    device_port = addr[1]
                    
                    # 存储设备响应
                    if device_ip not in device_responses:
                        response_text = data.decode('utf-8')
                        device_responses[device_ip] = response_text
                        normal_logger.info(f"收到来自 {device_ip}:{device_port} 的响应:\n{response_text}")
                            
                except socket.timeout:
                    continue
                except Exception as e:
                    normal_logger.warning(f"处理WS-Discovery响应失败: {e}")
                    continue
            
            sock.close()
            
            # 处理所有收集到的响应
            for device_ip, response_data in device_responses.items():
                try:
                    # 解析响应
                    device_info = await self._parse_probe_response(response_data, device_ip)
                    if device_info:
                        normal_logger.info(f"解析设备 {device_ip} 的响应结果:\n{device_info}")
                        
                        # 尝试HTTP探测获取更多信息
                        http_info = await self.probe_http_device(device_ip)
                        if http_info:
                            device_info.update(http_info)
                            normal_logger.info(f"HTTP探测 {device_ip} 获取的额外信息:\n{http_info}")
                        
                        devices.append(device_info)
                        seen_devices.add(device_ip)
                        normal_logger.info(f"发现ONVIF设备: {device_ip}")
                except Exception as e:
                    normal_logger.warning(f"解析设备 {device_ip} 响应失败: {e}")
            
            normal_logger.info(f"WS-Discovery完成，发现 {len(devices)} 个设备")
            
        except Exception as e:
            exception_logger.exception(f"WS-Discovery发现失败: {e}")
        
        return devices

    async def probe_http_device(self, device_ip: str) -> Optional[Dict[str, Any]]:
        """通过HTTP探测ONVIF设备获取更多信息"""
        try:
            # 常见的ONVIF设备服务路径
            onvif_paths = [
                '/onvif/device_service',
                '/Device',  # 天地伟业特殊路径
                '/onvif/service',
            ]
            
            device_info = None
            
            async with aiohttp.ClientSession() as session:
                for path in onvif_paths:
                    url = f'http://{device_ip}:80{path}'
                    try:
                        async with session.get(url) as response:
                            if response.status == 200 or response.status == 401:  # 401表示需要认证，说明服务存在
                                device_info = {
                                    'ip': device_ip,
                                    'endpoints': [url],
                                }
                                break  # 找到有效路径后退出
                    except aiohttp.ClientError:
                        continue
            
            # 如果找到HTTP端点，尝试获取更多设备信息
            if device_info and ONVIF_AVAILABLE:
                try:
                    cam = ONVIFCamera(device_ip, 80, 'admin', '')
                    device_service = cam.create_devicemgmt_service()
                    info = device_service.GetDeviceInformation()
                    device_info.update({
                        'manufacturer': info.Manufacturer,
                        'model': info.Model,
                        'name': f"{info.Manufacturer} {info.Model}"
                    })
                except Exception as e:
                    normal_logger.warning(f"获取设备详细信息失败 {device_ip}: {e}")
            
            return device_info
            
        except Exception as e:
            normal_logger.warning(f"HTTP探测失败 {device_ip}: {e}")
            return None

    def _create_probe_message(self) -> str:
        """创建WS-Discovery探测消息"""
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
    xmlns:a="http://schemas.xmlsoap.org/ws/2004/08/addressing"
    xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"
    xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
    <s:Header>
        <a:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</a:Action>
        <a:MessageID>uuid:{self._generate_message_id()}</a:MessageID>
    </s:Header>
    <s:Body>
        <d:Probe>
            <d:Types>dn:NetworkVideoTransmitter</d:Types>
        </d:Probe>
    </s:Body>
</s:Envelope>"""
    
    def _generate_message_id(self) -> str:
        """生成唯一的消息ID"""
        return f"{int(time.time() * 1000)}-{socket.gethostname()}"

    async def _parse_probe_response(self, response_data: str, device_ip: str) -> Optional[Dict[str, Any]]:
        """解析探测响应数据"""
        try:
            # 解析XML响应
            root = ET.fromstring(response_data)
            
            # 处理可能的错误响应
            fault = root.find('.//{http://www.w3.org/2003/05/soap-envelope}Fault')
            if fault is not None:
                normal_logger.warning(f"设备 {device_ip} 返回SOAP错误，尝试使用备用方式获取设备信息")
                # 尝试直接通过HTTP获取设备信息
                return await self._get_device_info_via_http(device_ip)
            
            # 初始化设备信息
            device_info = {
                'ip': device_ip,
                'name': 'Unknown Device',
                'manufacturer': 'Unknown',
                'model': 'Unknown',
                'firmware_version': 'Unknown',
                'serial_number': 'Unknown',
                'hardware_id': 'Unknown',
                'type': 'Unknown',
                'xaddrs': [],
                'scopes': []
            }
            
            # 尝试不同的命名空间获取设备信息
            namespaces = [
                {'d': 'http://schemas.xmlsoap.org/ws/2005/04/discovery'},
                {'ns2': 'http://schemas.xmlsoap.org/ws/2005/04/discovery'},
                {'wsd': 'http://schemas.xmlsoap.org/ws/2005/04/discovery'}
            ]
            
            # 遍历所有可能的命名空间
            for ns in namespaces:
                prefix = list(ns.keys())[0]
                # 获取设备类型
                types_elem = root.find(f'.//{{{ns[prefix]}}}Types')
                if types_elem is not None and types_elem.text:
                    device_info['type'] = types_elem.text.strip()
                
                # 获取XAddrs（设备服务地址）
                xaddrs_elem = root.find(f'.//{{{ns[prefix]}}}XAddrs')
                if xaddrs_elem is not None and xaddrs_elem.text:
                    device_info['xaddrs'] = [addr.strip() for addr in xaddrs_elem.text.split()]
                
                # 获取Scopes
                scopes_elem = root.find(f'.//{{{ns[prefix]}}}Scopes')
                if scopes_elem is not None and scopes_elem.text:
                    scopes = scopes_elem.text.strip().split()
                    device_info['scopes'] = scopes
                    
                    # 从scopes中提取设备信息
                    for scope in scopes:
                        if 'name/' in scope:
                            device_info['name'] = urllib.parse.unquote(scope.split('name/')[-1])
                        elif 'hardware/' in scope:
                            device_info['model'] = scope.split('hardware/')[-1]
                        elif 'location/' in scope:
                            device_info['location'] = scope.split('location/')[-1]
                        elif 'MAC/' in scope:
                            device_info['mac'] = scope.split('MAC/')[-1]
                        elif 'firmware/' in scope:
                            device_info['firmware_version'] = scope.split('firmware/')[-1]
                        elif 'mfr/' in scope:
                            device_info['manufacturer'] = scope.split('mfr/')[-1]
            
            # 尝试获取更多设备信息
            if device_info['xaddrs']:
                additional_info = await self._get_additional_device_info(device_info['xaddrs'][0])
                if additional_info:
                    device_info.update(additional_info)
            
            return device_info
            
        except ET.ParseError as e:
            normal_logger.error(f"解析设备 {device_ip} 的响应XML时出错: {str(e)}")
            return None
        except Exception as e:
            normal_logger.error(f"处理设备 {device_ip} 的响应时出错: {str(e)}")
            return None

    async def _get_device_info_via_http(self, device_ip: str) -> Optional[Dict[str, Any]]:
        """通过HTTP直接获取设备信息"""
        try:
            # 常见的ONVIF服务路径
            service_paths = [
                '/onvif/device_service',
                '/Device',
                '/onvif/services',
                '/onvif'
            ]
            
            device_info = {
                'ip': device_ip,
                'name': 'Unknown Device',
                'manufacturer': 'Unknown',
                'model': 'Unknown',
                'firmware_version': 'Unknown',
                'serial_number': 'Unknown',
                'hardware_id': 'Unknown',
                'type': 'dn:NetworkVideoTransmitter',
                'xaddrs': [],
                'scopes': []
            }
            
            async with aiohttp.ClientSession() as session:
                for path in service_paths:
                    url = f'http://{device_ip}{path}'
                    try:
                        async with session.get(url, timeout=2) as response:
                            if response.status == 200 or response.status == 401:  # 401表示需要认证，说明服务存在
                                device_info['xaddrs'] = [url]
                                # 尝试获取设备名称
                                try:
                                    hostname = socket.gethostbyaddr(device_ip)[0]
                                    device_info['name'] = hostname
                                except:
                                    pass
                                return device_info
                    except:
                        continue
            
            return None
        except Exception as e:
            normal_logger.error(f"HTTP获取设备 {device_ip} 信息失败: {str(e)}")
            return None

    async def _get_additional_device_info(self, device_url: str) -> Optional[Dict[str, Any]]:
        """获取设备的额外信息"""
        try:
            # 构建设备信息请求
            info_msg = """<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">
    <s:Body xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema">
        <GetDeviceInformation xmlns="http://www.onvif.org/ver10/device/wsdl"/>
    </s:Body>
</s:Envelope>"""
            
            headers = {
                'Content-Type': 'application/soap+xml; charset=utf-8',
                'SOAPAction': 'http://www.onvif.org/ver10/device/wsdl/GetDeviceInformation'
            }
            
            # 常用的认证凭据
            credentials = [
                ("admin", ""),
                ("admin", "admin"),
                ("admin", "12345"),
                ("admin", "Admin12345"),
                ("admin", "password"),
                ("root", ""),
                ("root", "root"),
                ("user", ""),
                ("user", "user")
            ]
            
            async with aiohttp.ClientSession() as session:
                # 首先尝试无认证
                try:
                    async with session.post(device_url, data=info_msg, headers=headers, timeout=5) as response:
                        if response.status == 200:
                            response_text = await response.text()
                            return self._parse_device_info_response(response_text)
                except:
                    pass
                
                # 如果需要认证，尝试常用凭据
                if response.status == 401:
                    for username, password in credentials:
                        auth = aiohttp.BasicAuth(username, password)
                        try:
                            async with session.post(device_url, data=info_msg, headers=headers, auth=auth, timeout=5) as response:
                                if response.status == 200:
                                    response_text = await response.text()
                                    return self._parse_device_info_response(response_text)
                        except:
                            continue
            
            normal_logger.warning(f"无法获取设备额外信息: {device_url}")
            return None
                        
        except Exception as e:
            normal_logger.error(f"获取设备额外信息时出错: {str(e)}")
            return None
            
    def _parse_device_info_response(self, response_text: str) -> Dict[str, Any]:
        """解析设备信息响应"""
        try:
            root = ET.fromstring(response_text)
            info = {}
            
            # 提取设备信息
            for tag in ['Manufacturer', 'Model', 'FirmwareVersion', 'SerialNumber', 'HardwareId']:
                elem = root.find(f'.//*{tag}')
                if elem is not None and elem.text:
                    info[tag.lower()] = elem.text.strip()
            
            return info
        except Exception as e:
            normal_logger.error(f"解析设备信息响应失败: {str(e)}")
            return {}

class ONVIFDeviceService:
    """ONVIF设备服务，提供设备测试和管理功能"""
    
    def __init__(self):
        self.default_credentials = [
            ("admin", ""),
            ("admin", "admin"),
            ("admin", "12345"),
            ("admin", "password"),
            ("root", ""),
            ("root", "root"),
            ("user", ""),
            ("user", "user")
        ]
    
    async def test_device(self, device_ip: str, username: str = "admin", password: str = "") -> Dict[str, Any]:
        """
        测试ONVIF设备功能
        
        Args:
            device_ip: 设备IP地址
            username: 用户名
            password: 密码
            
        Returns:
            Dict[str, Any]: 测试结果
        """
        result = {
            'ip': device_ip,
            'onvif_available': False,
            'device_info': None,
            'profiles': [],
            'capabilities': {},
            'error': None
        }
        
        if not ONVIF_AVAILABLE:
            result['error'] = "onvif-zeep库未安装"
            return result
        
        try:
            normal_logger.info(f"测试ONVIF设备: {device_ip}")
            
            # 尝试连接设备
            camera = ONVIFCamera(device_ip, 80, username, password)
            
            # 获取设备信息
            device_service = camera.create_devicemgmt_service()
            device_info_resp = device_service.GetDeviceInformation()
            
            result['onvif_available'] = True
            result['device_info'] = {
                'manufacturer': device_info_resp.Manufacturer,
                'model': device_info_resp.Model,
                'firmware_version': device_info_resp.FirmwareVersion,
                'serial_number': device_info_resp.SerialNumber,
                'hardware_id': device_info_resp.HardwareId
            }
            
            # 获取设备能力
            capabilities = device_service.GetCapabilities()
            result['capabilities'] = {
                'media': bool(capabilities.Media),
                'ptz': bool(getattr(capabilities, 'PTZ', False)),
                'imaging': bool(getattr(capabilities, 'Imaging', False)),
                'events': bool(getattr(capabilities, 'Events', False))
            }
            
            # 获取媒体配置文件
            if capabilities.Media:
                try:
                    media_service = camera.create_media_service()
                    profiles = media_service.GetProfiles()
                    
                    for profile in profiles:
                        profile_info = {
                            'name': profile.Name,
                            'token': profile.token,
                            'resolution': 'Unknown',
                            'video_encoding': 'Unknown',
                            'stream_url': None
                        }
                        
                        # 获取视频编码器配置
                        if hasattr(profile, 'VideoEncoderConfiguration') and profile.VideoEncoderConfiguration:
                            vec = profile.VideoEncoderConfiguration
                            profile_info['video_encoding'] = vec.Encoding
                            if hasattr(vec, 'Resolution'):
                                profile_info['resolution'] = f"{vec.Resolution.Width}x{vec.Resolution.Height}"
                        
                        # 尝试获取流URL
                        try:
                            request = media_service.create_type('GetStreamUri')
                            request.ProfileToken = profile.token
                            request.StreamSetup = {
                                'Stream': 'RTP-Unicast',
                                'Transport': {'Protocol': 'RTSP'}
                            }
                            response = media_service.GetStreamUri(request)
                            profile_info['stream_url'] = response.Uri
                        except:
                            pass
                        
                        result['profiles'].append(profile_info)
                        
                except Exception as e:
                    normal_logger.warning(f"获取媒体配置文件失败: {e}")
            
            normal_logger.info(f"ONVIF设备测试成功: {device_ip}")
            
        except Exception as e:
            result['error'] = str(e)
            normal_logger.warning(f"ONVIF设备测试失败: {device_ip} - {e}")
        
        return result

class DiscoveryService:
    """统一设备发现服务"""
    
    def __init__(self):
        self.ws_discovery = WsDiscoveryService()
        self.onvif_service = ONVIFDeviceService()
    
    async def discover_onvif_devices(self, interface_ip: Optional[str] = None, timeout: int = 10) -> List[Dict[str, Any]]:
        """发现ONVIF设备"""
        return await self.ws_discovery.discover_onvif_devices(interface_ip, timeout)
    
    async def test_onvif_device(self, device_ip: str, username: str = "admin", password: str = "") -> Dict[str, Any]:
        """测试ONVIF设备"""
        return await self.onvif_service.test_device(device_ip, username, password)

# 全局服务实例
discovery_service = DiscoveryService() 