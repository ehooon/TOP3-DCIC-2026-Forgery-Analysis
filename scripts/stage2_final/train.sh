#!/bin/bash
# 遇到错误立即退出
set -e 

# 自动获取项目根目录 (上两级目录)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

# 切换到项目根目录
cd "$PROJECT_ROOT"

echo "=========================================="
echo "       开始 Stage 2 Final 训练流程       "
echo "=========================================="

echo ">>> [1/3] 正在运行 download_script.sh..."
# 使用 bash 执行，避免文件没有可执行权限(x)的问题
bash models/download_script.sh

echo ">>> [2/3] 正在运行 train_sam.py..."
python src/stage2_final/train_sam.py

echo ">>> [3/3] 正在运行 train_LoRA.py..."
python src/stage2_final/train_LoRA.py

echo "=========================================="
echo "                 训练完成                 "
echo "=========================================="