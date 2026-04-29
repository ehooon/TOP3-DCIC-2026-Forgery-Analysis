import os
import argparse
import torch
import shutil
from unsloth import FastVisionModel
from peft import PeftModel

# ==========================================
# ⚙️ 动态路径解析与参数配置
# ==========================================
# 1. 获取当前脚本所在目录的绝对路径 (src/stage2_final)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. 定位项目根目录 (向上退两级)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

# 3. 命令行参数解析 (修正默认路径，并支持外部传参)
parser = argparse.ArgumentParser(description="LoRA 与基础模型物理合并脚本")
parser.add_argument(
    "--lora_path", 
    type=str, 
    default=os.path.join(PROJECT_ROOT, "models", "adapters", "stage2_final", "outputs_qwen3.5_forgery", "checkpoint-189"),
    help="LoRA 权重所在的目录"
)
parser.add_argument(
    "--base_model", 
    type=str, 
    # 🌟 核心修改：将基础模型的默认预期路径移入 models/ 文件夹下，保持根目录整洁
    default=os.path.join(PROJECT_ROOT, "models", "Qwen3.5-9B"), 
    help="纯净基础模型所在的目录"
)
parser.add_argument(
    "--save_path", 
    type=str, 
    default=os.path.join(PROJECT_ROOT, "models", "final_full_model"),
    help="合并后全量模型的保存目录"
)
args = parser.parse_args()

LORA_PATH = args.lora_path
BASE_MODEL = args.base_model
SAVE_PATH = args.save_path

# 打印路径方便调试核对
print(f"📂 项目根目录: {PROJECT_ROOT}")
print(f"👉 LORA路径: {LORA_PATH}")
print(f"👉 基础模型: {BASE_MODEL}")
print(f"👉 保存路径: {SAVE_PATH}\n")

print("🔄 开始底层暴力合并流程...")

# 1. 纯净加载基础模型 (⚠️ 绝对不能开 4bit)
print("📦 1. 加载基础模型 (bfloat16 模式，约需 18GB 内存)...")
model, tokenizer = FastVisionModel.from_pretrained(
    model_name = BASE_MODEL,
    load_in_4bit = False,         # 必须为 False，用高精度读入才能做加法
    torch_dtype = torch.bfloat16,
    local_files_only = True,
)

# 2. 挂载你的 Checkpoint 补丁
print(f"🔗 2. 挂载 Checkpoint: {LORA_PATH}")
model = PeftModel.from_pretrained(model, LORA_PATH)

# 3. 物理合并
print("🔥 3. 正在执行底层参数焊接 (merge_and_unload)...")
# 这行代码会把 LoRA 的参数矩阵直接加到基础模型的矩阵上，并销毁 LoRA 结构
model = model.merge_and_unload() 

# 4. 原生保存
print(f"💾 4. 正在写入全量权重到: {SAVE_PATH} (硬盘在狂转，请耐心等待...)")
os.makedirs(SAVE_PATH, exist_ok=True)
model.save_pretrained(SAVE_PATH, safe_serialization=True)
tokenizer.save_pretrained(SAVE_PATH)

# 5. 补齐 vLLM 需要的视觉配置文件
print("📋 5. 补齐视觉配置文件...")
for f in ["preprocessor_config.json", "generation_config.json", "config.json"]:
    src = os.path.join(BASE_MODEL, f)
    dst = os.path.join(SAVE_PATH, f)
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy(src, dst)

print("\n✅ 物理焊接彻底完成！赶紧去文件夹里看看有没有好几个 GB 的 .safetensors 文件！")