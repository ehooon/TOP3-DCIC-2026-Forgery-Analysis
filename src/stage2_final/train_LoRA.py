import os
import sys
import time
import glob
import shutil
import logging
from transformers import TrainerCallback # 💡 导入回调模块

# =================================================================
# ⚙️ 0. 动态路径解析与环境变量 (必须在 import torch 之前)
# =================================================================
# 1. 获取当前脚本所在目录绝对路径 (src/stage2_final)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. 定位项目根目录 (向上退两级)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

# 3. 将项目根目录加入环境变量，防止跨目录调用 (如 import src...) 时报 ModuleNotFoundError
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["UNSLOTH_USE_MODELSCOPE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
# 统一缓存路径，防止污染全局环境
os.environ["MODELSCOPE_CACHE"] = os.path.join(PROJECT_ROOT, ".cache")
os.environ["HF_HOME"] = os.path.join(PROJECT_ROOT, ".cache")

# =================================================================
# 📝 0.5 全局日志配置 (追加模式)
# =================================================================
# 路径对齐：直接输出到 logs/stage2_final 目录下
LOG_DIR = os.path.join(PROJECT_ROOT, "logs", "stage2_final")
os.makedirs(LOG_DIR, exist_ok=True) # 确保日志目录存在
LOG_FILE = os.path.join(LOG_DIR, "train.log")

# 配置根日志记录器，同时输出到控制台和文件，使用 'a' (append) 模式追加写入
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("="*50)
logger.info("🚀 新的一轮训练任务启动")
logger.info("="*50)

# =================================================================
# 📝 0.6 自定义训练日志回调 (按步记录 Loss)
# =================================================================
class StepLoggingCallback(TrainerCallback):
    def __init__(self, logger):
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        每当 Trainer 触发 log 事件时（由 logging_steps 控制），此函数会被调用。
        """
        if logs is not None and "loss" in logs:
            step = state.global_step
            loss = logs["loss"]
            lr = logs.get("learning_rate", 0.0)
            epoch = logs.get("epoch", 0.0)
            
            # 格式化输出写入 train.log
            self.logger.info(f"[Training] Epoch: {epoch:.4f} | Step: {step} | Loss: {loss:.4f} | LR: {lr:.2e}")

import torch
from PIL import Image as PilImage, ImageFile
from datasets import Dataset, Image as HfImage
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

# 允许加载截断图片
ImageFile.LOAD_TRUNCATED_IMAGES = True
PilImage.MAX_IMAGE_PIXELS = None 

# =================================================================
# 🎛️ 1. 全局配置区 (基于项目根目录动态拼接)
# =================================================================
# [修复] 基础模型路径：原代码直接找根目录，根据目录树已修正为查找 models/ 目录
MODEL_NAME = os.path.join(PROJECT_ROOT, "models", "Qwen3.5-9B")

# 原始训练数据集路径
RAW_DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "ForgeryAnalysis_Stage_1_Train")

# 清洗目录：物理缩放 8K 大图，防止显存直接爆炸 (存放在 data 目录下)
CLEAN_DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "ForgeryAnalysis_Stage_1_Train_Cleaned_1024")

# LoRA 权重输出目录 (对齐项目结构树的 models/adapters 路径)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "adapters", "stage2_final", "outputs_qwen3.5_forgery")

MAX_IMAGE_SIZE = 1024
MAX_SEQ_LENGTH = 4096 

logger.info(f"📂 项目根目录: {PROJECT_ROOT}")
logger.info(f"👉 模型路径: {MODEL_NAME}")
logger.info(f"👉 原始数据: {RAW_DATA_ROOT}")
logger.info(f"👉 清洗数据: {CLEAN_DATA_ROOT}")
logger.info(f"👉 输出目录: {OUTPUT_DIR}\n")

# ==========================================
# 2. 物理缩放与数据清洗 (核心防御)
# ==========================================
def prepare_cleaned_dataset(raw_root, clean_root, max_size):
    if os.path.exists(clean_root):
        logger.info(f"✅ 检测到已清洗目录 {clean_root}，跳过预处理。")
        return
    
    logger.info(f"🧹 正在物理缩放原图至 {max_size}px (针对 8K 图像的降维打击)...")
    valid_count = 0
    for cat in ["Black", "White"]:
        img_raw_dir = os.path.join(raw_root, cat, "Image")
        cap_raw_dir = os.path.join(raw_root, cat, "Caption")
        
        img_clean_dir = os.path.join(clean_root, cat, "Image")
        cap_clean_dir = os.path.join(clean_root, cat, "Caption")
        
        # 宽容模式：若原图文件夹不存在则跳过当前类别
        if not os.path.exists(img_raw_dir): 
            logger.warning(f"⚠️ 找不到路径: {img_raw_dir}，跳过此类别")
            continue

        os.makedirs(img_clean_dir, exist_ok=True)
        os.makedirs(cap_clean_dir, exist_ok=True)

        for img_file in os.listdir(img_raw_dir):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')): continue
            
            raw_path = os.path.join(img_raw_dir, img_file)
            base_name = os.path.splitext(img_file)[0]
            md_path = os.path.join(cap_raw_dir, base_name + ".md")
            
            if not os.path.exists(md_path): continue

            try:
                with PilImage.open(raw_path) as img:
                    if img.mode != "RGB": img = img.convert("RGB")
                    w, h = img.size
                    if max(w, h) > max_size:
                        scale = max_size / max(w, h)
                        img = img.resize((int(w*scale), int(h*scale)), resample=PilImage.Resampling.LANCZOS)
                    img.save(os.path.join(img_clean_dir, base_name + ".jpg"), "JPEG", quality=95)
                
                shutil.copy2(md_path, os.path.join(cap_clean_dir, base_name + ".md"))
                valid_count += 1
            except Exception as e:
                logger.warning(f"⚠️ 跳过异常文件 {img_file}: {e}")
    
    logger.info(f"✅ 数据预处理完成，共生成 {valid_count} 条物理纯净数据！")

# ==========================================
# 3. 构建 Dataset (多进程 & 双重提示词版)
# ==========================================
def load_forgery_dataset(data_root):
    data_list = []
    
    # 💡 重新找回你要求的两种提示词
    PROMPT_BLACK = "请分析这张图片，这是一张被篡改（伪造）过的图片，请指出篡改痕迹并描述目标："
    PROMPT_WHITE = "请分析这张图片，这是一张真实的原始图片，请详细描述图片中的目标内容："

    for category in ["Black", "White"]:
        img_dir = os.path.join(data_root, category, "Image")
        cap_dir = os.path.join(data_root, category, "Caption")
        
        # 根据类别分配对应的 User Prompt
        current_prompt = PROMPT_BLACK if category == "Black" else PROMPT_WHITE
        
        if not os.path.exists(img_dir): continue

        for img_file in os.listdir(img_dir):
            base_name = os.path.splitext(img_file)[0]
            img_path = os.path.abspath(os.path.join(img_dir, img_file))
            md_path = os.path.join(cap_dir, base_name + ".md")

            if not os.path.exists(md_path): continue

            with open(md_path, 'r', encoding='utf-8') as f:
                caption = f.read().strip() 

            data_list.append({
                "image": img_path,
                "user_prompt": current_prompt, 
                "caption": caption
            })

    ds = Dataset.from_list(data_list)
    ds = ds.cast_column("image", HfImage()) # 延迟加载，防止 OOM

    def format_messages(example):
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": example["image"]},
                        {"type": "text", "text": example["user_prompt"]}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": example["caption"]}]
                }
            ]
        }

    # 多进程处理，并安全删除多余列，规避 Arrow Bug
    ds = ds.map(format_messages, num_proc=4)
    cols_to_remove = [c for c in ds.column_names if c != "messages"]
    return ds.remove_columns(cols_to_remove)


# ==========================================
# 4. 主训练流程
# ==========================================
def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # [阶段 1] 物理清洗大图
    if local_rank == 0:
        prepare_cleaned_dataset(RAW_DATA_ROOT, CLEAN_DATA_ROOT, MAX_IMAGE_SIZE)
    
    # 分布式屏障，确保所有进程看到的数据都是洗干净的
    if torch.cuda.device_count() > 1:
        torch.distributed.barrier() if torch.distributed.is_initialized() else time.sleep(5)

    # [阶段 2] 构建数据集
    logger.info(f"📚 进程 {local_rank} 正在加载数据集...")
    dataset = load_forgery_dataset(CLEAN_DATA_ROOT)

    # [阶段 3] 加载模型
    logger.info(f"⏳ 进程 {local_rank} 正在加载模型至显存...")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name = MODEL_NAME,
        load_in_4bit = True,
        use_gradient_checkpointing = "unsloth",
        local_files_only = True,
        device_map = {"": local_rank}
    )

    # 显式开启视觉训练模式
    FastVisionModel.for_training(model)

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers   = True, # 💡 鉴定模型建议微调视觉层以捕捉细微伪影
        finetune_language_layers = True,
        finetune_attention_modules = True,
        finetune_mlp_modules = True,
        r = 16,
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        random_state = 3407,
        target_modules = "all-linear",
    )

    # [阶段 4] 启动炼丹
    logger.info(f"🚀 进程 {local_rank} 配置完成，开始正式训练...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "",
        data_collator = UnslothVisionDataCollator(model, tokenizer),
        callbacks = [StepLoggingCallback(logger)], # 💡 注入自定义回调，记录每步 Loss
        args = SFTConfig(
            per_device_train_batch_size = 4, # 96G 显存 + 1024px 图片，可以开到 4 甚至更高
            gradient_accumulation_steps = 4,
            warmup_steps = 10,
            num_train_epochs = 3,  # 3 轮是微调标配
            learning_rate = 2e-4,
            fp16 = not is_bf16_supported(),
            bf16 = is_bf16_supported(),
            logging_steps = 1, # 💡 保持为 1，确保每步触发回调
            output_dir = OUTPUT_DIR,
            optim = "adamw_8bit",
            seed = 3407,
            remove_unused_columns = False,
            dataset_kwargs = {"skip_prepare_dataset": True}, # 💡 规避 Unsloth 报错的关键参数
            report_to = "none", # 这里的 none 指的是不上传至 wandb/tensorboard，本地 python 日志依然会捕获
        ),
    )

    trainer.train()

    if local_rank == 0:
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        logger.info(f"🎉 任务圆满完成！权重已保存至: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()