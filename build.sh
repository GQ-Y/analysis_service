#!/bin/bash

# 切换到项目根目录
cd $(dirname $0)/..

# 检查NVIDIA GPU
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected, building GPU version..."
    docker build -t analysis-service -f analysis_service/Dockerfile.gpu .
else
    echo "No NVIDIA GPU detected, building CPU version..."
    docker build -t analysis-service -f analysis_service/Dockerfile.cpu .
fi 