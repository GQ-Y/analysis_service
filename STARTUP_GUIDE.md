# 🚀 启动指南和问题排查

## 📋 启动方式对比

| 启动方式 | 命令 | 热重载 | 适用场景 | 推荐度 |
|---------|------|--------|----------|--------|
| **简单启动** | `python run.py` | ❌ | 新手、快速测试 | ⭐⭐⭐⭐⭐ |
| **热重载启动** | `python start.py` | ✅ | 开发调试 | ⭐⭐⭐⭐ |
| **命令行启动** | `python main.py --reload` | ✅ | 自定义配置 | ⭐⭐⭐ |
| **脚本启动** | `./start.sh dev` | ❌ | 环境管理 | ⭐⭐⭐ |
| **Make启动** | `make dev` | ❌ | 标准化开发 | ⭐⭐⭐ |

## 🎯 推荐启动流程

### 1. 首次启动（新手推荐）

```bash
# 1. 验证环境
python check_startup.py

# 2. 简单启动
python run.py
```

### 2. 开发模式（开发者推荐）

```bash
# 热重载启动（代码修改自动重启）
python start.py
```

### 3. 生产模式

```bash
# 多进程启动
python main.py --env production --workers 4
```

## ❌ 常见启动问题及解决方案

### 问题1：热重载警告

**错误信息**：
```
WARNING: You must pass the application as an import string to enable 'reload' or 'workers'.
```

**原因**：使用热重载时传递了应用实例而不是导入字符串

**解决方案**：
```bash
# ❌ 错误方式
uvicorn.run(app, reload=True)

# ✅ 正确方式
uvicorn.run("main:create_app", factory=True, reload=True)

# 或者使用简单启动（不需要热重载）
python run.py
```

### 问题2：端口被占用

**错误信息**：
```
OSError: [Errno 48] Address already in use
```

**解决方案**：
```bash
# 查看端口占用
lsof -i :8002

# 杀死占用进程
kill -9 <PID>

# 或使用其他端口
python main.py --port 8003
```

### 问题3：模块导入错误

**错误信息**：
```
ModuleNotFoundError: No module named 'app'
```

**解决方案**：
```bash
# 确保在项目根目录
cd /path/to/analysis_service

# 检查目录结构
ls -la

# 应该看到 app/ config/ 等目录
```

### 问题4：依赖包缺失

**错误信息**：
```
ModuleNotFoundError: No module named 'fastapi'
```

**解决方案**：
```bash
# 激活虚拟环境
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 验证安装
python check_startup.py
```

### 问题5：权限问题

**错误信息**：
```
Permission denied: './start.sh'
```

**解决方案**：
```bash
# 添加执行权限
chmod +x start.sh

# 或直接使用Python启动
python run.py
```

## 🔧 启动配置

### 环境变量配置

创建 `.env` 文件（可选）：
```bash
# 应用配置
ENVIRONMENT=development
DEBUG=true
HOST=127.0.0.1
PORT=8002

# 日志配置
LOG_LEVEL=info
```

### 启动参数说明

```bash
python main.py --help
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | 0.0.0.0 | 服务器主机地址 |
| `--port` | 8002 | 服务器端口 |
| `--reload` | False | 启用热重载 |
| `--workers` | 1 | 工作进程数量 |
| `--log-level` | info | 日志级别 |
| `--env` | development | 运行环境 |

## 🧪 启动验证

### 1. 快速验证

```bash
# 检查应用是否可以创建
python -c "from main import create_app; app = create_app(); print('✅ 应用创建成功')"
```

### 2. 完整验证

```bash
# 运行完整的启动检查
python check_startup.py
```

### 3. 健康检查

启动后验证：
```bash
# 基础健康检查
curl http://127.0.0.1:8002/health

# 存储系统检查
curl http://127.0.0.1:8002/api/v1/storage/stats
```

## 🌐 访问地址

启动成功后可以访问：

- **API文档**: http://127.0.0.1:8002/docs
- **ReDoc文档**: http://127.0.0.1:8002/redoc  
- **健康检查**: http://127.0.0.1:8002/health
- **存储API**: http://127.0.0.1:8002/api/v1/storage/stats

## 📊 启动成功标志

看到以下信息说明启动成功：

```
🚀 启动分析服务...
   - 主机: 127.0.0.1
   - 端口: 8002
   - 模式: 开发模式
   - 文档: http://127.0.0.1:8002/docs

INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8002 (Press CTRL+C to quit)
```

## 🎯 最佳实践

### 开发环境

```bash
# 1. 首次启动验证
python check_startup.py

# 2. 简单启动（推荐新手）
python run.py

# 3. 热重载启动（推荐开发）
python start.py
```

### 生产环境

```bash
# 多进程启动
python main.py --env production --workers 4 --host 0.0.0.0
```

### Docker环境

```bash
# Docker Compose启动
docker-compose up -d

# 查看日志
docker-compose logs -f analysis-service
```

## 🆘 获取帮助

如果遇到问题：

1. **查看启动日志**：注意错误信息
2. **运行验证脚本**：`python check_startup.py`
3. **检查环境**：确保Python版本和依赖正确
4. **查看文档**：阅读README.md和相关文档
5. **使用简单启动**：`python run.py`（最稳定）

## 🎉 启动成功！

选择适合你的启动方式：

- **新手用户**: `python run.py`
- **开发调试**: `python start.py`  
- **生产部署**: `python main.py --env production --workers 4`

现在可以开始使用分析服务了！🎊
