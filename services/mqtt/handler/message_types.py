"""
MQTT消息类型定义
"""

# 消息类型定义
MESSAGE_TYPE_CONNECTION = 80001  # 连接/上线/遗嘱消息
MESSAGE_TYPE_COMMAND = 80002     # 命令消息
MESSAGE_TYPE_RESULT = 80003      # 分析结果响应
MESSAGE_TYPE_STATUS = 80004      # 状态上报
MESSAGE_TYPE_BROADCAST = 80008   # 系统广播

# 请求设置消息
MESSAGE_TYPE_REQUEST_SETTING = 80002

TOPIC_TYPE_DEVICE_CONFIG_REPLY = 'device_config_reply'

# 命令类型
REQUEST_TYPE_NODE_CMD = "node_cmd"
REQUEST_TYPE_TASK_CMD = "task_cmd"

# 节点命令类型
NODE_CMD_SYNC_TIME = "sync_time"

# 任务命令类型
TASK_CMD_START = "start_task"
TASK_CMD_STOP = "stop_task" 