"""
MQTT消息打印工具
提供统一的MQTT消息打印格式
"""
import json
from datetime import datetime
from typing import Any, Dict, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# 创建控制台实例
console = Console()

class MQTTPrinter:
    """
    MQTT消息打印类
    提供统一的MQTT消息打印格式
    """
    
    @staticmethod
    def print_message(topic: str, payload: Any, direction: str = "接收") -> None:
        """
        打印MQTT消息
        
        Args:
            topic: 消息主题
            payload: 消息内容
            direction: 消息方向（接收/发送）
        """
        try:
            # 创建时间戳
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            
            # 创建消息表格
            table = Table(show_header=False, box=None)
            table.add_column("字段", style="cyan")
            table.add_column("值", style="white")
            
            # 添加基本信息
            table.add_row("时间", timestamp)
            table.add_row("方向", direction)
            table.add_row("主题", topic)
            
            # 处理消息内容
            if isinstance(payload, (str, bytes)):
                try:
                    # 尝试解析JSON
                    if isinstance(payload, bytes):
                        payload = payload.decode('utf-8')
                    data = json.loads(payload)
                    # 添加JSON内容
                    table.add_row("内容", json.dumps(data, ensure_ascii=False, indent=2))
                except json.JSONDecodeError:
                    # 如果不是JSON，直接显示原始内容
                    table.add_row("内容", str(payload))
            else:
                # 如果是字典或其他对象，转换为JSON字符串
                table.add_row("内容", json.dumps(payload, ensure_ascii=False, indent=2))
            
            # 创建面板
            panel = Panel(
                table,
                title=f"MQTT消息 {direction}",
                border_style="green" if direction == "接收" else "blue",
                title_align="left"
            )
            
            # 打印消息
            console.print(panel)
            
        except Exception as e:
            console.print(f"[red]打印MQTT消息时出错: {str(e)}[/red]")
            
    @staticmethod
    def print_connection_status(status: str, message: str) -> None:
        """
        打印MQTT连接状态
        
        Args:
            status: 状态（成功/失败）
            message: 状态消息
        """
        try:
            # 创建时间戳
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            
            # 创建状态表格
            table = Table(show_header=False, box=None)
            table.add_column("字段", style="cyan")
            table.add_column("值", style="white")
            
            # 添加状态信息
            table.add_row("时间", timestamp)
            table.add_row("状态", status)
            table.add_row("消息", message)
            
            # 创建面板
            panel = Panel(
                table,
                title="MQTT连接状态",
                border_style="green" if status == "成功" else "red",
                title_align="left"
            )
            
            # 打印状态
            console.print(panel)
            
        except Exception as e:
            console.print(f"[red]打印MQTT连接状态时出错: {str(e)}[/red]")
            
    @staticmethod
    def print_subscription(topic: str, qos: int, action: str = "订阅") -> None:
        """
        打印MQTT订阅信息
        
        Args:
            topic: 主题名称
            qos: 服务质量等级
            action: 动作（订阅/取消订阅）
        """
        try:
            # 创建时间戳
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            
            # 创建订阅表格
            table = Table(show_header=False, box=None)
            table.add_column("字段", style="cyan")
            table.add_column("值", style="white")
            
            # 添加订阅信息
            table.add_row("时间", timestamp)
            table.add_row("动作", action)
            table.add_row("主题", topic)
            table.add_row("QoS", str(qos))
            
            # 创建面板
            panel = Panel(
                table,
                title="MQTT订阅信息",
                border_style="yellow",
                title_align="left"
            )
            
            # 打印订阅信息
            console.print(panel)
            
        except Exception as e:
            console.print(f"[red]打印MQTT订阅信息时出错: {str(e)}[/red]")
            
    @staticmethod
    def print_database_operation(operation: str, table: str, data: Dict[str, Any], status: str = "成功") -> None:
        """
        打印数据库操作信息
        
        Args:
            operation: 操作类型（创建/更新）
            table: 表名
            data: 操作数据
            status: 操作状态（成功/失败）
        """
        try:
            # 创建时间戳
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            
            # 创建操作表格
            table_obj = Table(show_header=False, box=None)
            table_obj.add_column("字段", style="cyan")
            table_obj.add_column("值", style="white")
            
            # 添加操作信息
            table_obj.add_row("时间", timestamp)
            table_obj.add_row("操作", operation)
            table_obj.add_row("表名", table)
            table_obj.add_row("状态", status)
            
            # 添加数据信息
            data_str = json.dumps(data, ensure_ascii=False, indent=2)
            table_obj.add_row("数据", data_str)
            
            # 创建面板
            panel = Panel(
                table_obj,
                title="数据库操作",
                border_style="green" if status == "成功" else "red",
                title_align="left"
            )
            
            # 打印操作信息
            console.print(panel)
            
        except Exception as e:
            console.print(f"[red]打印数据库操作信息时出错: {str(e)}[/red]") 