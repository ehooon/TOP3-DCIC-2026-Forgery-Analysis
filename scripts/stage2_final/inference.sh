#!/bin/bash
# 遇到错误立即退出
set -e 

# 自动获取项目根目录
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

# ==========================================
# ⚙️ 默认路径配置区
# ==========================================
INPUT_PATH="$PROJECT_ROOT/data/ForgeryAnalysis_Stage_2_Test/Image"
OUTPUT_PATH="$PROJECT_ROOT/results_val/final_submission.csv"

# ==========================================
# 🔍 命令行参数解析
# ==========================================
while [[ $# -gt 0 ]]; do
  case $1 in
    --input_path)
      INPUT_PATH="$2"
      shift 2 # 移动两个参数位置 (--input_path 和它的值)
      ;;
    --output_path)
      OUTPUT_PATH="$2"
      shift 2 # 移动两个参数位置 (--output_path 和它的值)
      ;;
    *)
      echo "⚠️ 未知参数: $1"
      echo "用法: bash inference.sh [--input_path <dir>] [--output_path <file.csv>]"
      exit 1
      ;;
  esac
done

# ==========================================
# 🚀 执行流程
# ==========================================
# 切换到项目根目录
cd "$PROJECT_ROOT"

echo "=========================================="
echo "       开始 Stage 2 Final 推理流程       "
echo "=========================================="
echo "📌 输入路径: $INPUT_PATH"
echo "📌 输出路径: $OUTPUT_PATH"
echo "=========================================="
echo ">>> [0/2] 正在运行 download_script.sh..."
# 使用 bash 执行，避免文件没有可执行权限(x)的问题
bash models/download_script.sh

echo ">>> [1/2] 正在运行 connect-lora.py..."
python src/stage2_final/connect-lora.py

echo ">>> [2/2] 正在运行 inference.py..."
# 将设置好的路径作为参数传递给 Python 脚本
python src/stage2_final/inference.py \
    --input_path "$INPUT_PATH" \
    --output_path "$OUTPUT_PATH"

echo "=========================================="
echo "                推理完成                "
echo "=========================================="