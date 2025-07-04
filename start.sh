#!/bin/bash

# 分析服务启动脚本
# 作者: Yanli
# 创建日期: 2025-01-04

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}🚀 分析服务启动脚本${NC}"
echo "=================================================="

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 未安装${NC}"
    exit 1
fi

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}⚠️  虚拟环境不存在，正在创建...${NC}"
    python3 -m venv venv
fi

# 激活虚拟环境
echo -e "${GREEN}📦 激活虚拟环境...${NC}"
source venv/bin/activate

# 安装依赖
if [ ! -f "venv/.deps_installed" ]; then
    echo -e "${GREEN}📥 安装依赖包...${NC}"
    pip install -r requirements.txt
    touch venv/.deps_installed
fi

# 检查启动模式
MODE=${1:-dev}

case $MODE in
    "dev"|"development")
        echo -e "${GREEN}🔧 启动开发模式...${NC}"
        python run.py
        ;;
    "reload")
        echo -e "${GREEN}🔄 启动热重载模式...${NC}"
        python main.py --reload
        ;;
    "prod"|"production")
        echo -e "${GREEN}🏭 启动生产模式...${NC}"
        python main.py --env production --workers 4
        ;;
    "test")
        echo -e "${GREEN}🧪 启动测试模式...${NC}"
        python main.py --env testing --port 8003
        ;;
    *)
        echo -e "${YELLOW}使用方法:${NC}"
        echo "  ./start.sh [dev|reload|prod|test]"
        echo ""
        echo -e "${YELLOW}模式说明:${NC}"
        echo "  dev    - 开发模式 (默认，简单启动)"
        echo "  reload - 热重载模式 (代码修改自动重启)"
        echo "  prod   - 生产模式 (多进程)"
        echo "  test   - 测试模式 (端口8003)"
        exit 1
        ;;
esac
