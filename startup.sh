#!/bin/bash
# Chirrup API 服务启动脚本
# 使用 uv run 自动管理虚拟环境并启动API服务

# ============================================================================
# 配置区域 - 请根据需要修改以下参数
# ============================================================================

# 模型文件路径（必需）
MODEL_PATH="../models/rwkv7-g0a3-7.2b-20251029-ctx8192.pth"

# 词汇表文件路径（可选，使用默认值可注释掉）
# VOCAB_PATH="./Albatross/rwkv_vocab_v20230424.txt"

# 词汇表大小（可选，使用默认值可注释掉）
# VOCAB_SIZE=65536

# 头大小（可选，使用默认值可注释掉）
# HEAD_SIZE=64

# Worker 数量（可选，使用默认值可注释掉）
# WORKER_NUM=1

# 批处理大小（可选，使用默认值可注释掉）
# BATCH_SIZE=24

# 状态缓存大小（可选，使用默认值可注释掉）
# STATE_CACHE_SIZE=50

# 服务器主机地址（可选，使用默认值可注释掉）
# HOST="127.0.0.1"

# 服务器端口（可选，使用默认值可注释掉）
# PORT=8000

# ============================================================================
# 以下为脚本执行逻辑，无需修改
# ============================================================================

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 检查 uv 是否安装
if ! command -v uv &> /dev/null; then
    echo "错误: 未找到 uv，请先安装 uv"
    echo "安装方法: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# 检查虚拟环境是否存在
if [ ! -d ".venv" ]; then
    echo "错误: 虚拟环境不存在，请先运行 'uv venv'"
    exit 1
fi

# 设置环境变量
export PYTHON_GIL=0

# 构建启动命令（使用 uv run --frozen，与文档保持一致）
START_CMD="uv run --frozen python -m chirrup.web_service.app"

# 添加模型路径（必需参数）
if [ -n "$MODEL_PATH" ]; then
    START_CMD="$START_CMD --model_path \"$MODEL_PATH\""
else
    echo "错误: MODEL_PATH 未设置，请在脚本顶部配置模型路径"
    exit 1
fi

# 添加可选参数
[ -n "$VOCAB_PATH" ] && START_CMD="$START_CMD --vocab_path \"$VOCAB_PATH\""
[ -n "$VOCAB_SIZE" ] && START_CMD="$START_CMD --vocab_size $VOCAB_SIZE"
[ -n "$HEAD_SIZE" ] && START_CMD="$START_CMD --head_size $HEAD_SIZE"
[ -n "$WORKER_NUM" ] && START_CMD="$START_CMD --worker_num $WORKER_NUM"
[ -n "$BATCH_SIZE" ] && START_CMD="$START_CMD --batch_size $BATCH_SIZE"
[ -n "$STATE_CACHE_SIZE" ] && START_CMD="$START_CMD --state_cache_size $STATE_CACHE_SIZE"
[ -n "$HOST" ] && START_CMD="$START_CMD --host \"$HOST\""
[ -n "$PORT" ] && START_CMD="$START_CMD --port $PORT"

echo "正在启动 Chirrup API 服务..."
echo "命令: $START_CMD"
echo ""

# 启动服务
eval $START_CMD
