#!/bin/bash
# 启动 Qwen API 服务器

echo "Starting Qwen API Server..."
echo "Server will be available at http://localhost:8010"
echo "API documentation: http://localhost:8010/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# 初始化 conda（尝试多个可能的路径）
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
else
    # 如果找不到，尝试从 PATH 中找到 conda
    CONDA_PATH=$(which conda)
    if [ -n "$CONDA_PATH" ]; then
        CONDA_BASE=$(dirname $(dirname $CONDA_PATH))
        if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
            source "$CONDA_BASE/etc/profile.d/conda.sh"
        fi
    fi
fi

# 激活 conda 环境
conda activate flow_grpo

export PYTHONNOUSERSITE=1
# 设置只使用 GPU 5（8B 模型单卡即可）
export CUDA_VISIBLE_DEVICES=1
echo "Using GPU: 5 (visible as 0 to the process)"
echo ""

# 运行 API 服务器
python qwen_api_server_4.py

