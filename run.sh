#!/bin/bash

# 切换到项目根目录
cd $(dirname $0)/..

# 检查NVIDIA GPU
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected, running with GPU support..."
    docker run -d \
        --name analysis-service \
        --network meekyolo-net \
        --gpus all \
        -p 8002:8002 \
        -v $(pwd)/models:/app/models \
        -v $(pwd)/data:/app/data \
        -v $(pwd)/results:/app/results \
        analysis-service
else
    echo "No NVIDIA GPU detected, running CPU version..."
    docker run -d \
        --name analysis-service \
        --network meekyolo-net \
        -p 8002:8002 \
        -v $(pwd)/models:/app/models \
        -v $(pwd)/data:/app/data \
        -v $(pwd)/results:/app/results \
        analysis-service
fi 