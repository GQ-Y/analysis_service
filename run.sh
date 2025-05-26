#!/bin/bash

# 定义日志文件
LOG_FILE="./logs/service.log"
ERROR_LOG="./logs/error.log"
ANALYSIS_LOG="./logs/analysis.log"

# 确保日志目录存在
mkdir -p ./logs

# 解析命令行参数
MODE="service" # 默认模式：启动服务
TEST_DURATION=60 # 默认测试时间（秒）
CONFIDENCE=0.01 # 默认置信度阈值
LOGS_ONLY=false # 默认不只分析日志
KEEP_LOGS=false # 默认不保留旧日志

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            MODE="test"
            shift
            ;;
        --logs-only)
            MODE="logs"
            LOGS_ONLY=true
            shift
            ;;
        --duration)
            TEST_DURATION="$2"
            shift
            shift
            ;;
        --confidence)
            CONFIDENCE="$2"
            shift
            shift
            ;;
        --keep-logs)
            KEEP_LOGS=true
            shift
            ;;
        --help)
            echo "用法: ./run.sh [选项]"
            echo "选项:"
            echo "  --test             启动AI测试模式而不是服务"
            echo "  --logs-only        仅分析已有的AI测试日志"
            echo "  --duration N       设置测试持续时间为N秒（默认60）"
            echo "  --confidence N     设置检测置信度阈值为N（默认0.01）"
            echo "  --keep-logs        不清除旧日志文件"
            echo "  --help             显示此帮助信息"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 显示横幅
echo "=========================================="
echo "  MeekYolo 视频分析服务启动脚本"
echo "  版本: 1.0"
echo "  日期: $(date)"
echo "  模式: $MODE"
echo "=========================================="

# 清理旧日志文件
clear_old_logs() {
    if [ "$KEEP_LOGS" = true ]; then
        echo "保留旧日志文件"
        return 0
    fi

    echo "清理旧日志文件..."

    # 定义需要清理的日志文件
    local log_files=(
        "$LOG_FILE"
        "$ERROR_LOG"
        "$ANALYSIS_LOG"
        "./service.out"
    )

    # 清理每个日志文件
    for log_file in "${log_files[@]}"; do
        if [ -f "$log_file" ]; then
            # 创建带时间戳的备份
            timestamp=$(date +"%Y%m%d_%H%M%S")
            backup_file="${log_file}.${timestamp}.bak"

            # 尝试重命名文件
            if mv "$log_file" "$backup_file" 2>/dev/null; then
                echo "已备份旧日志文件: $log_file -> $backup_file"
            else
                # 如果重命名失败，尝试清空文件内容
                if cat /dev/null > "$log_file" 2>/dev/null; then
                    echo "已清空日志文件: $log_file"
                else
                    echo "警告: 无法清理日志文件 $log_file" >&2
                fi
            fi
        fi
    done

    echo "日志清理完成"
}

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3，请确保已安装Python3" | tee -a $ERROR_LOG
    exit 1
fi

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "未找到虚拟环境，正在创建..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "错误: 创建虚拟环境失败" | tee -a $ERROR_LOG
        exit 1
    fi
    echo "虚拟环境创建成功"
fi

# 激活虚拟环境
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "错误: 激活虚拟环境失败" | tee -a $ERROR_LOG
    exit 1
fi

# 检查依赖是否安装
echo "检查依赖..."
if [ ! -f "requirements.lock" ]; then
    echo "未找到依赖锁定文件，将使用requirements.txt"
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "错误: 安装依赖失败" | tee -a $ERROR_LOG
        exit 1
    fi
else
    pip install -r requirements.lock
    if [ $? -ne 0 ]; then
        echo "警告: 使用锁定文件安装依赖失败，尝试使用requirements.txt" | tee -a $ERROR_LOG
        pip install -r requirements.txt
        if [ $? -ne 0 ]; then
            echo "错误: 安装依赖失败" | tee -a $ERROR_LOG
            exit 1
        fi
    fi
fi

# 设置环境变量
export PYTHONDONTWRITEBYTECODE=1
export DEBUG_ENABLED=1

# 函数：启动AI测试
run_test() {
    echo "启动AI测试..."

    # 清理旧日志
    clear_old_logs

    # 检查端口占用
    echo "检查端口8002是否被占用..."
    check_port
    if [ $? -ne 0 ]; then
        echo "错误: 无法释放端口8002，AI测试无法启动" | tee -a $ERROR_LOG
        exit 1
    else
        echo "端口8002检查通过，可以使用"
    fi

    # 创建结果目录
    CURRENT_DATE=$(date +%Y%m%d)
    echo "创建结果保存目录: results/$CURRENT_DATE"
    mkdir -p "./results/$CURRENT_DATE"
    mkdir -p "./results/preview"

    # 设置目录权限
    chmod -R 755 "./results"

    # 列出目录结构
    echo "结果目录结构:"
    ls -la "./results/"

    if [ "$LOGS_ONLY" = true ]; then
        echo "仅分析已有的AI测试日志"
        python ai_test.py --only-logs
    else
        echo "执行AI测试，持续时间: ${TEST_DURATION}秒，置信度阈值: ${CONFIDENCE}"
        echo "图像结果将保存在: ./results/$CURRENT_DATE/"
        python ai_test.py --duration "$TEST_DURATION" --confidence "$CONFIDENCE"
    fi
}

# 函数：启动服务并在崩溃时自动重启
start_service() {
    echo "启动服务..."

    # 清理旧日志
    clear_old_logs

    # 确保ZLMediaKit环境正确
    check_zlmediakit

    echo "服务日志保存在: $LOG_FILE"

    # 设置崩溃计数器
    CRASH_COUNT=0
    SEGFAULT_COUNT=0
    MAX_CRASHES=5

    # 设置环境变量，尝试防止ZLMediaKit段错误
    export ZLM_SAFE_MODE=1
    export OPENCV_VIDEOIO_DEBUG=1

    # 无限循环，服务退出后自动重启
    while true; do
        echo "$(date) - 启动服务实例" >> $LOG_FILE

        # 启动uvicorn，将输出重定向到日志文件，使用warning日志级别以关闭INFO日志
        # 从环境变量获取端口，默认为8002
        PORT=${SERVICES_PORT:-8002}
        HOST=${SERVICES_HOST:-0.0.0.0}
        uvicorn app:app --host $HOST --port $PORT --workers 1 --log-level warning >> $LOG_FILE 2>&1

        # 捕获退出状态
        EXIT_CODE=$?

        # 记录退出状态
        echo "$(date) - 服务退出，状态码: $EXIT_CODE" >> $LOG_FILE

        # 检查是否是段错误
        if [ $EXIT_CODE -eq 139 ] || [ $EXIT_CODE -eq 11 ]; then
            SEGFAULT_COUNT=$((SEGFAULT_COUNT + 1))
            echo "检测到段错误 (SIGSEGV)，这是第 $SEGFAULT_COUNT 次段错误" | tee -a $ERROR_LOG

            # 记录崩溃时的系统状态
            echo "崩溃时系统状态:" >> $ERROR_LOG
            echo "时间: $(date)" >> $ERROR_LOG
            echo "内存使用情况:" >> $ERROR_LOG
            free -h >> $ERROR_LOG 2>&1 || vm_stat >> $ERROR_LOG 2>&1
            echo "进程列表:" >> $ERROR_LOG
            ps aux | grep -E 'python|uvicorn|zlm' >> $ERROR_LOG

            # 如果段错误次数超过3次，尝试在CPU模式下启动
            if [ $SEGFAULT_COUNT -ge 3 ]; then
                echo "段错误次数过多，尝试在CPU模式下启动..." | tee -a $ERROR_LOG
                export FORCE_CPU_MODE=1
                export CUDA_VISIBLE_DEVICES="-1"
            fi
        fi

        # 如果是正常退出(0)或收到中断信号(130)，则不重启
        if [ $EXIT_CODE -eq 0 ] || [ $EXIT_CODE -eq 130 ]; then
            echo "服务正常退出，不再重启" | tee -a $LOG_FILE
            break
        fi

        # 增加崩溃计数
        CRASH_COUNT=$((CRASH_COUNT + 1))

        # 如果崩溃次数超过最大限制，则停止重启
        if [ $CRASH_COUNT -ge $MAX_CRASHES ]; then
            echo "服务已崩溃 $CRASH_COUNT 次，超过最大限制 $MAX_CRASHES，停止重启" | tee -a $ERROR_LOG
            echo "请检查日志文件以获取更多信息: $LOG_FILE, $ERROR_LOG" | tee -a $ERROR_LOG
            exit 1
        fi

        # 根据崩溃次数增加等待时间，避免频繁重启
        WAIT_TIME=$((5 * CRASH_COUNT))
        echo "服务异常退出 (状态码 $EXIT_CODE)，$WAIT_TIME 秒后重启... (第 $CRASH_COUNT 次尝试)" | tee -a $LOG_FILE
        sleep $WAIT_TIME
    done
}

# 检查端口是否被占用
check_port() {
    # 从环境变量获取端口，默认为8002
    PORT=${SERVICES_PORT:-8002}

    if command -v lsof &> /dev/null; then
        if lsof -i :$PORT &> /dev/null; then
            echo "警告: 端口$PORT已被占用，尝试关闭现有进程..." | tee -a $ERROR_LOG
            kill $(lsof -t -i :$PORT) 2>/dev/null
            sleep 2
            if lsof -i :$PORT &> /dev/null; then
                echo "错误: 无法释放端口$PORT，请手动关闭占用该端口的进程" | tee -a $ERROR_LOG
                return 1
            fi
        fi
    fi
    return 0
}

# 清理过程
cleanup() {
    echo "正在清理..."
    # 清理临时文件
    rm -rf ./temp/videos/*
    echo "清理完成"
}

# 检查ZLMediaKit是否正常工作
check_zlmediakit() {
    echo "检查ZLMediaKit环境..."

    # 确保ZLMediaKit库目录存在
    if [ ! -d "lib/zlm" ]; then
        echo "错误: 未找到ZLMediaKit库目录 lib/zlm" | tee -a $ERROR_LOG

        # 尝试创建目录
        mkdir -p lib/zlm

        # 检查zlmos目录是否存在，如果存在则复制库文件
        if [ "$(uname)" == "Darwin" ] && [ -d "zlmos/darwin" ]; then
            if [ -f "zlmos/darwin/libmk_api.dylib" ]; then
                echo "找到zlmos目录中的库文件，复制到lib/zlm目录..." | tee -a $LOG_FILE
                cp zlmos/darwin/libmk_api.dylib lib/zlm/
                chmod 755 lib/zlm/libmk_api.dylib
                echo "库文件已复制到lib/zlm目录" | tee -a $LOG_FILE
            else
                echo "错误: zlmos/darwin目录中未找到libmk_api.dylib文件" | tee -a $ERROR_LOG
                return 1
            fi
        else
            echo "无法找到ZLMediaKit库文件源，请确保lib/zlm目录中包含必要的库文件" | tee -a $ERROR_LOG
            return 1
        fi
    fi

    # 检查库文件
    if [ "$(uname)" == "Darwin" ]; then
        # macOS
        if [ ! -f "lib/zlm/libmk_api.dylib" ]; then
            echo "错误: 未找到ZLMediaKit库文件 lib/zlm/libmk_api.dylib (macOS)" | tee -a $ERROR_LOG
            return 1
        fi
    elif [ "$(uname)" == "Linux" ]; then
        # Linux
        if [ ! -f "lib/zlm/libmk_api.so" ]; then
            echo "错误: 未找到ZLMediaKit库文件 lib/zlm/libmk_api.so (Linux)" | tee -a $ERROR_LOG
            return 1
        fi
    else
        echo "错误: 不支持的操作系统: $(uname)" | tee -a $ERROR_LOG
        return 1
    fi

    # 设置ZLMediaKit库路径
    export ZLM_LIBRARY_PATH="$(pwd)/lib/zlm"
    echo "设置ZLMediaKit库路径: $ZLM_LIBRARY_PATH"

    # 设置LD_LIBRARY_PATH/DYLD_LIBRARY_PATH以确保系统能找到ZLMediaKit库
    if [ "$(uname)" == "Darwin" ]; then
        # macOS使用DYLD_LIBRARY_PATH
        export DYLD_LIBRARY_PATH="$ZLM_LIBRARY_PATH:$DYLD_LIBRARY_PATH"
        echo "设置DYLD_LIBRARY_PATH: $ZLM_LIBRARY_PATH"
    elif [ "$(uname)" == "Linux" ]; then
        # Linux使用LD_LIBRARY_PATH
        export LD_LIBRARY_PATH="$ZLM_LIBRARY_PATH:$LD_LIBRARY_PATH"
        echo "设置LD_LIBRARY_PATH: $ZLM_LIBRARY_PATH"
    fi

    echo "ZLMediaKit环境检查通过"
    return 0
}

# 主执行流程
main() {
    # 清理
    cleanup

    # 检查ZLMediaKit环境
    check_zlmediakit
    if [ $? -ne 0 ]; then
        echo "ZLMediaKit环境检查失败，尝试继续..." | tee -a $ERROR_LOG
    fi

    if [ "$MODE" = "test" ]; then
        # 运行AI测试
        run_test
        exit 0
    elif [ "$MODE" = "logs" ]; then
        # 仅分析日志
        run_test
        exit 0
    else
        # 服务模式
        # 检查端口
        check_port
        if [ $? -ne 0 ]; then
            exit 1
        fi

        # 启动服务
        start_service
    fi
}

# 捕获CTRL+C
trap 'echo "接收到中断信号，正在停止..."; exit 0' INT

# 执行主函数
main

# 退出消息
echo "程序已停止"