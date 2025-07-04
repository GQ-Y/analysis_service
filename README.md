# 分析服务 (Analysis Service)

一个基于FastAPI的现代化视频分析服务，提供高性能的视频流处理和智能分析功能。

## 🚀 快速启动

### 方式一：简单启动（推荐新手）

```bash
# 开发模式启动（最简单）
python run.py
```

访问：http://127.0.0.1:8002

### 方式二：热重载启动（推荐开发）

```bash
# 热重载模式（代码修改自动重启）
python start.py
```

### 方式三：使用启动脚本

```bash
# 开发模式（简单启动）
./start.sh dev

# 热重载模式
./start.sh reload

# 生产模式
./start.sh prod

# 测试模式
./start.sh test
```

### 方式四：使用main.py（高级用户）

```bash
# 开发模式（热重载）
python main.py --reload

# 生产模式（多进程）
python main.py --env production --workers 4

# 自定义配置
python main.py --host 0.0.0.0 --port 8080 --log-level debug
```

### 方式四：使用Makefile

```bash
# 开发模式
make dev

# 生产模式
make prod

# Docker模式
make docker
```

## 📋 启动参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | 0.0.0.0 | 服务器主机地址 |
| `--port` | 8002 | 服务器端口 |
| `--reload` | False | 启用热重载（开发模式） |
| `--workers` | 1 | 工作进程数量 |
| `--log-level` | info | 日志级别 |
| `--env` | development | 运行环境 |

## 🔧 环境准备

### 1. Python环境

```bash
# 检查Python版本（需要3.8+）
python --version

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 2. 安装依赖

```bash
# 安装生产依赖
pip install -r requirements.txt

# 安装开发依赖（可选）
pip install -r requirements-dev.txt
```

### 3. 环境变量

创建 `.env` 文件（可选）：

```bash
# 应用配置
ENVIRONMENT=development
DEBUG=true
HOST=127.0.0.1
PORT=8002

# 数据库配置
DATABASE_URL=sqlite:///./analysis.db

# Redis配置
REDIS_URL=redis://localhost:6379/0

# 日志配置
LOG_LEVEL=info
```

## 🌐 访问地址

启动成功后，可以访问以下地址：

- **API文档**: http://127.0.0.1:8002/docs
- **ReDoc文档**: http://127.0.0.1:8002/redoc
- **健康检查**: http://127.0.0.1:8002/health
- **API根路径**: http://127.0.0.1:8002/api/v1/

## 🧪 验证启动

### 1. 健康检查

```bash
curl http://127.0.0.1:8002/health
```

预期响应：
```json
{
  "success": true,
  "message": "服务健康",
  "data": {
    "status": "healthy",
    "service": "analysis-service",
    "version": "2.0.0"
  }
}
```

### 2. 存储统计

```bash
curl http://127.0.0.1:8002/api/v1/storage/stats
```

### 3. API文档

在浏览器中访问：http://127.0.0.1:8002/docs

## 🐳 Docker启动

### 1. 使用Docker Compose（推荐）

```bash
# 启动所有服务
docker-compose up -d

# 查看日志
docker-compose logs -f analysis-service

# 停止服务
docker-compose down
```

### 2. 使用Dockerfile

```bash
# 构建镜像
docker build -t analysis-service .

# 运行容器
docker run -p 8002:8002 analysis-service
```

## 🔍 故障排除

### 常见问题

**1. 端口被占用**
```bash
# 查看端口占用
lsof -i :8002

# 使用其他端口
python main.py --port 8003
```

**2. 依赖安装失败**
```bash
# 升级pip
pip install --upgrade pip

# 清理缓存
pip cache purge

# 重新安装
pip install -r requirements.txt
```

**3. 虚拟环境问题**
```bash
# 删除旧环境
rm -rf venv

# 重新创建
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 日志查看

```bash
# 查看应用日志
tail -f storage/logs/app.log

# 查看错误日志
tail -f storage/logs/error.log
```

## 📊 性能监控

启动后可以通过以下端点监控服务状态：

- **内存使用**: http://127.0.0.1:8002/health/memory
- **插件状态**: http://127.0.0.1:8002/health/plugins
- **存储统计**: http://127.0.0.1:8002/api/v1/storage/stats

## 🛠️ 开发模式

开发模式特性：
- ✅ 热重载：代码修改自动重启
- ✅ 详细日志：debug级别日志输出
- ✅ API文档：自动生成和更新
- ✅ 错误追踪：详细的错误堆栈信息

```bash
# 启动开发模式
python start.py

# 或者
python main.py --reload --log-level debug
```

## 🏭 生产模式

生产模式特性：
- ✅ 多进程：提高并发处理能力
- ✅ 优化日志：减少日志输出
- ✅ 安全配置：禁用调试信息
- ✅ 性能监控：启用性能指标收集

```bash
# 启动生产模式
python main.py --env production --workers 4
```

## 📚 更多信息

- [API文档](docs/api_documentation.md)
- [架构设计](docs/new_architecture_design.md)
- [存储系统](docs/storage_architecture.md)
- [验证报告](docs/architecture_validation_report.md)

## 🎉 启动成功！

如果看到以下信息，说明服务启动成功：

```
🚀 启动分析服务...
   - 环境: development
   - 主机: 127.0.0.1
   - 端口: 8002
   - 热重载: 启用
   - 工作进程: 1
   - 日志级别: info

INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8002
```

现在可以开始使用分析服务了！🎊
