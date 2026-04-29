#!/bin/bash
# 遇到错误立即退出
set -e

# ==========================================
# ⚙️ 路径设置 (基于脚本所在位置动态计算)
# ==========================================
# 1. 获取当前脚本所在目录的绝对路径 (即 models/ 目录)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# 2. 定位项目根目录 (向上退一级)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

# 3. 拼接 Qwen 模型保存的绝对路径 (models/Qwen3.5-9B)
QWEN_TARGET_DIR="$SCRIPT_DIR/Qwen3.5-9B"

# 打印路径方便调试核对
echo "📂 脚本所在目录: $SCRIPT_DIR"
echo "📂 项目根目录: $PROJECT_ROOT"
echo "=========================================="

# ==========================================
# ⬇️ 任务 1: 下载 SAM 模型权重 (Meta 官方)
# ==========================================
echo "⬇️ 开始下载 SAM 模型权重 (sam_vit_h_4b8939.pth) ..."
# 使用 wget 下载，-c 支持断点续传，-P 指定直接下载到 models/ 目录
wget -c "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" -P "$SCRIPT_DIR"

# [可选] 如果你需要另外两个版本的 SAM 模型，可以取消下面两行的注释：
# wget -c "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth" -P "$SCRIPT_DIR"
# wget -c "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" -P "$SCRIPT_DIR"

echo "✅ SAM 模型下载完成！"
echo "=========================================="

# ==========================================
# ⬇️ 任务 2: 下载 Qwen 模型权重 (ModelScope)
# ==========================================
echo "👉 Qwen 模型目标路径: $QWEN_TARGET_DIR"
echo "⬇️ 开始下载模型 Qwen/Qwen3.5-9B ..."

# 4. 执行下载命令，使用计算出的绝对路径
modelscope download --model Qwen/Qwen3.5-9B --local_dir "$QWEN_TARGET_DIR"

echo "✅ Qwen 模型下载完成！已保存至: $QWEN_TARGET_DIR"
echo "=========================================="
echo "🎉 所有模型权重下载任务已顺利结束！"