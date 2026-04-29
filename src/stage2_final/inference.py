# inference.py

import os
import argparse

# =================================================================
# ⚙️ 0. 动态路径解析与底层环境变量配置
# =================================================================
# 1. 获取当前脚本所在目录绝对路径 (src/stage2_final)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. 定位项目根目录 (向上退两级)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

# 3. 环境变量配置 (必须在 import torch/vllm 之前)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NCCL_P2P_DISABLE"] = "1"         
os.environ["NCCL_IB_DISABLE"] = "1"          
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn" 
os.environ["VLLM_USE_MODELSCOPE"] = "False"  
# 将缓存目录也放到项目根目录下的 .cache 文件夹中，避免污染全局
os.environ["MODELSCOPE_CACHE"] = os.path.join(PROJECT_ROOT, ".cache") 

# =================================================================
# 📦 1. 统一库导入
# =================================================================
import gc
import json
import re
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pycocotools.mask as mask_utils  

# 自定义库与模型导入
from forensics_sam import ForensicsSAM
from segment_anything import sam_model_registry
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


# =================================================================
# 🎛️ 2. 全局统一配置区 (基于项目根目录动态拼接)
# =================================================================
class CFG:
    # --- 阶段一：SAM 模型配置 ---
    # 指向 models/adapters/stage2_final/weight
    WEIGHT_DIR = os.path.join(PROJECT_ROOT, "models", "adapters", "stage2_final", "weight")
    SAM_TYPE = "vit_h"
    # 指向 models/sam_vit_h_4b8939.pth
    SAM_CHECKPOINT = os.path.join(PROJECT_ROOT, "models", "sam_vit_h_4b8939.pth")
    IMAGE_SIZE = 1024
    BATCH_SIZE = 1
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    FULL_MODEL_WEIGHT = 0.20
    MANUAL_THRESHOLD = 0.44

    # --- 阶段二：LLM 模型配置 ---
    # 动态指向 connect-lora.py 生成的新文件夹
    LLM_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "final_full_model")
    MAX_TOKENS = 1024 * 8
    TEMPERATURE = 0.2
    MAX_MODEL_LEN = 1024 * 16
    TENSOR_PARALLEL = 1
    
    # 可调的概率区间 (触发 LLM 判断的阈值)
    PROB_LOWER = 0.48
    PROB_UPPER = 0.55
    
    # 提示词
    PROMPT_CLASSIFY = "请判断这张图片是否被篡改。如果是被篡改的假图，请仅输出一个字“假”；如果是真实的未篡改图片，请仅输出一个字“真”。不要输出任何其他多余的字符或解释。"
    PROMPT_REAL = "这是一张真实的图片（未被篡改）。请分析这张图片，并详细描述图片中的目标内容："
    PROMPT_FAKE = "这是一张被篡改（伪造）过的图片。请分析这张图片，指出篡改痕迹并描述目标："
    
    MAX_IMG_SIZE = 1024


# =================================================================
# 🛠️ 3. 工具类与辅助函数
# =================================================================
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

class TestDataset(Dataset):
    def __init__(self, image_dir: str, target_size: int = 1024):
        self.image_dir = image_dir
        self.target_size = target_size
        self.image_names = [f for f in os.listdir(image_dir) if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]
        if not self.image_names:
            raise ValueError(f"在 {image_dir} 中没有找到任何图片！")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, Tuple[int, int], str]:
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        image = cv2.imread(img_path)
        if image is None: raise ValueError(f"无法读取图像: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        orig_h, orig_w = image.shape[:2]
        image_resized = cv2.resize(image, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
        
        image_tensor = torch.from_numpy(image_resized).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3)
        image_tensor = (image_tensor - mean) / std
        image_tensor = image_tensor.permute(2, 0, 1)
        
        return image_tensor, (orig_h, orig_w), img_name

def get_ensemble_configs(weight_dir: str, full_weight: float) -> list:
    configs = []
    full_path = os.path.join(weight_dir, "training_full_data", "forensics_stage1_last.pth")
    if not os.path.exists(full_path): raise FileNotFoundError(f"找不到全量模型: {full_path}")

    fold_paths = []
    fold_idx = 1
    while True:
        fold_path = os.path.join(weight_dir, f"training_kfold_{fold_idx}", "forensics_stage1_best.pth")
        if os.path.exists(fold_path):
            fold_paths.append(fold_path)
            fold_idx += 1
        else:
            break

    if not fold_paths:
        raise FileNotFoundError(f"未发现 kfold 模型！")

    fold_weight = (1.0 - full_weight) / len(fold_paths)
    for fp in fold_paths:
        configs.append({"path": fp, "weight": fold_weight})
    configs.append({"path": full_path, "weight": full_weight})
    return configs

def mask_array_to_rle(mask: np.ndarray) -> str:
    binary_mask = (mask > 127).astype(np.uint8)
    mask_fortran = np.asfortranarray(binary_mask)
    rle_dict = mask_utils.encode(mask_fortran)
    if isinstance(rle_dict['counts'], bytes):
        rle_dict['counts'] = rle_dict['counts'].decode('utf-8')
    return json.dumps(rle_dict)

def prepare_image_and_bbox(image_path, rle_str, label):
    img = cv2.imread(image_path)
    if img is None: return None
    orig_h, orig_w = img.shape[:2]
    
    if label != 0 and pd.notna(rle_str) and rle_str != "":
        try:
            rle_dict = json.loads(rle_str)
            if isinstance(rle_dict['counts'], str):
                rle_dict['counts'] = rle_dict['counts'].encode('utf-8')
            mask = mask_utils.decode(rle_dict)
            binary_mask = (mask > 0).astype(np.uint8) * 255
            if cv2.countNonZero(binary_mask) > 0:
                x_b, y_b, w_b, h_b = cv2.boundingRect(binary_mask)
                cv2.rectangle(img, (x_b, y_b), (x_b+w_b, y_b+h_b), (0, 0, 255), max(3, int(orig_w*0.005)))
        except Exception:
            pass 

    if max(orig_h, orig_w) > CFG.MAX_IMG_SIZE:
        scale = CFG.MAX_IMG_SIZE / max(orig_h, orig_w)
        img = cv2.resize(img, (int(orig_w * scale), int(orig_h * scale)), interpolation=cv2.INTER_AREA)
        
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# =================================================================
# 🚀 4. 核心流水线阶段定义 (内存流转版本)
# =================================================================
def run_stage1_sam(input_dir: str, mask_output_dir: str) -> pd.DataFrame:
    print("\n" + "="*50)
    print("▶️ 开始阶段一: SAM 模型 Ensemble 推理")
    print("="*50)
    
    device = torch.device(CFG.DEVICE)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model_configs = get_ensemble_configs(CFG.WEIGHT_DIR, CFG.FULL_MODEL_WEIGHT)

    print("⏳ 正在加载基础 SAM Backbone...")
    base_sam, _ = sam_model_registry[CFG.SAM_TYPE](image_size=CFG.IMAGE_SIZE, checkpoint=CFG.SAM_CHECKPOINT)
    model = ForensicsSAM(
        base_sam, r=8, forgery_experts_path=None, adversary_experts_path=None,
        load_pretrained=False, freeze_shared_experts=False, freeze_detector=False, enable_adversary_experts=False,
    ).to(device)

    dataset = TestDataset(image_dir=input_dir, target_size=CFG.IMAGE_SIZE)
    loader = DataLoader(dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=4)

    accumulated_results = {}

    with torch.no_grad():
        for config_idx, config in enumerate(model_configs, 1):
            weight_path, model_weight = config["path"], config["weight"]
            model.load_all_parameters(weight_path)
            model.eval()
            print(f"\n🔄 [{config_idx}/{len(model_configs)}] 推理模型: {os.path.basename(os.path.dirname(weight_path))} | 权重: {model_weight:.4f}")
            
            for images, (orig_hs, orig_ws), img_names in tqdm(loader, desc="Inference", leave=False):
                images = images.to(device, non_blocking=True)
                activate_adv = torch.zeros(images.size(0), device=device, dtype=torch.long)
                
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    mask_logits, cls_logits = model(images, activate_adv)
                
                cls_probs = torch.sigmoid(cls_logits.float()).cpu().numpy()
                mask_probs = torch.sigmoid(mask_logits.float()).cpu().half()

                for i in range(images.size(0)):
                    img_name = img_names[i]
                    if img_name not in accumulated_results:
                        accumulated_results[img_name] = {
                            "cls_prob": 0.0,
                            "mask_prob": torch.zeros_like(mask_probs[i]),
                            "orig_h": orig_hs[i].item(),
                            "orig_w": orig_ws[i].item()
                        }
                    accumulated_results[img_name]["cls_prob"] += float(cls_probs[i][0]) * model_weight
                    accumulated_results[img_name]["mask_prob"] += mask_probs[i] * model_weight

    print(f"\n✨ 模型遍历完成！生成初步推理逻辑...")
    logic_data_rows = []

    for img_name, data in tqdm(accumulated_results.items(), desc="Processing Results"):
        final_cls_prob = data["cls_prob"]
        pred_label = 1 if final_cls_prob > CFG.MANUAL_THRESHOLD else 0
        case_id = os.path.splitext(img_name)[0]

        mask_prob = data["mask_prob"].float().unsqueeze(0) 
        orig_h, orig_w = data["orig_h"], data["orig_w"]
        
        mask_prob_resized = F.interpolate(
            mask_prob, size=(orig_h, orig_w), mode='bilinear', align_corners=False
        )[0, 0] 
        
        mask_binary = (mask_prob_resized > CFG.MANUAL_THRESHOLD).byte().numpy() * 255
        mask_active_pixels = int(np.sum(mask_binary > 127))

        # 仍保留掩码图片保存，便于查验，若不需要可注释掉下面两行
        mask_save_path = os.path.join(mask_output_dir, case_id + ".png")
        cv2.imwrite(mask_save_path, mask_binary)

        if pred_label == 0 or mask_active_pixels == 0:
            location = json.dumps({"size": [orig_h, orig_w], "counts": ""})
        else:
            location = mask_array_to_rle(mask_binary)

        logic_data_rows.append({
            "image_name": img_name,
            "label": pred_label,
            "location": location,
            "explanation": "",
            "cls_prob": round(final_cls_prob, 6)
        })

    # ================= 🧹 显存回收核心区域 =================
    print("🧹 释放 SAM 模型显存...")
    del model
    del base_sam
    torch.cuda.empty_cache()
    gc.collect()
    # =======================================================

    print("✅ 阶段一完成！数据已驻留内存中。")
    return pd.DataFrame(logic_data_rows)


def run_stage2_llm(df: pd.DataFrame, input_dir: str) -> pd.DataFrame:
    print("\n" + "="*50)
    print("▶️ 开始阶段二: vLLM 多模态验证与解释生成")
    print("="*50)

    # 再次确保显存干净
    torch.cuda.empty_cache()

    if 'label' not in df.columns:
        df['label'] = (df['cls_prob'] >= 0.5).astype(int) if 'cls_prob' in df.columns else 0

    print("🚀 初始化 vLLM 引擎...")
    processor = AutoProcessor.from_pretrained(CFG.LLM_MODEL_PATH, trust_remote_code=True, local_files_only=True)
    llm = LLM(
        model=CFG.LLM_MODEL_PATH, 
        trust_remote_code=True, 
        dtype="bfloat16", 
        tensor_parallel_size=CFG.TENSOR_PARALLEL,
        max_model_len=CFG.MAX_MODEL_LEN,      
        gpu_memory_utilization=0.9, 
        enforce_eager=True,
        limit_mm_per_prompt={"image": 1} 
    )
    
    # --- 1. 模糊数据再判断 ---
    print(f"🔍 寻找概率在 {CFG.PROB_LOWER}~{CFG.PROB_UPPER} 之间的模糊样本...")
    uncertain_indices, uncertain_inputs = [], []
    for idx in df.index:
        cls_prob = float(df.loc[idx, 'cls_prob'])
        if CFG.PROB_LOWER <= cls_prob <= CFG.PROB_UPPER:
            img_path = os.path.join(input_dir, df.loc[idx, 'image_name'])
            pil_image = prepare_image_and_bbox(img_path, "", 0) 
            if pil_image is None: continue
                
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": CFG.PROMPT_CLASSIFY}]}]
            formatted_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            uncertain_inputs.append({"prompt": formatted_prompt, "multi_modal_data": {"image": pil_image}})
            uncertain_indices.append(idx)

    if uncertain_inputs:
        print(f"🧠 开始进行真假判别 (共 {len(uncertain_inputs)} 条)...")
        sp_cls = SamplingParams(temperature=0.0, max_tokens=CFG.MAX_TOKENS, stop_token_ids=[processor.tokenizer.eos_token_id])
        cls_outputs = llm.generate(uncertain_inputs, sampling_params=sp_cls, use_tqdm=True)
        
        for i, idx in enumerate(uncertain_indices):
            raw_ans = cls_outputs[i].outputs[0].text.strip()
            clean_ans = re.sub(r'(?is)<think>.*?(</think>|$)', '', raw_ans).strip()
            if '假' in clean_ans: df.loc[idx, 'label'] = 1
            elif '真' in clean_ans: df.loc[idx, 'label'] = 0
            else: df.loc[idx, 'label'] = 1 if float(df.loc[idx, 'cls_prob']) >= 0.5 else 0

    # --- 2. 批量生成最终解释 ---
    print("🔍 构建最终解释生成队列...")
    vllm_inputs, valid_indices = [], []
    for idx in df.index:
        img_path = os.path.join(input_dir, df.loc[idx, 'image_name'])
        label = int(df.loc[idx, 'label'])
        rle_str = df.loc[idx, 'location']

        pil_image = prepare_image_and_bbox(img_path, rle_str, label)
        if pil_image is None: continue
            
        selected_prompt = CFG.PROMPT_REAL if label == 0 else CFG.PROMPT_FAKE
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": selected_prompt}]}]
        formatted_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        vllm_inputs.append({"prompt": formatted_prompt, "multi_modal_data": {"image": pil_image}})
        valid_indices.append(idx)

    print(f"🧠 开始批量生成最终解释 (共 {len(vllm_inputs)} 条)...")
    sp_final = SamplingParams(temperature=CFG.TEMPERATURE, max_tokens=CFG.MAX_TOKENS, stop_token_ids=[processor.tokenizer.eos_token_id])
    outputs = llm.generate(vllm_inputs, sampling_params=sp_final, use_tqdm=True)
        
    df['explanation'] = "" 
    for i, idx in enumerate(valid_indices):
        raw_text = outputs[i].outputs[0].text.strip()
        clean_text = re.sub(r'(?is)<think>.*?(</think>|$)', '', raw_text)
        clean_text = clean_text.replace('**', '').replace('"', '”')
        df.loc[idx, 'explanation'] = re.sub(r'[\r\n\t]+', ' ', clean_text).strip()

    # ================= 🧹 显存回收核心区域 =================
    print("🧹 释放 vLLM 模型显存...")
    # vllm 分布式状态销毁（防止残留）
    from vllm.distributed.parallel_state import destroy_model_parallel
    destroy_model_parallel()
    del llm
    del processor
    torch.cuda.empty_cache()
    gc.collect()
    # =======================================================

    print(f"✅ 阶段二完成！数据解释已追加完毕。")
    return df


def run_stage3_postprocess(df: pd.DataFrame, output_path: str):
    print("\n" + "="*50)
    print("▶️ 开始阶段三: 最终格式清洗与打包落地")
    print("="*50)

    def process_location(location_str):
        if pd.isna(location_str) or not location_str: return location_str
        try:
            location = json.loads(location_str)
            location['counts'] = '' # 清空 counts
            return json.dumps(location, ensure_ascii=False)
        except:
            return location_str

    df['location'] = df['location'].apply(process_location)
    result_df = df[['image_name', 'label', 'location', 'explanation']]
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"🎉 全部处理完成！最终提交文件已生成并保存至:\n👉 {output_path}")


# =================================================================
# 🎯 5. 启动入口与命令行解析
# =================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="端到端 SAM + LLM 图像篡改分析流水线")
    # 输入与输出同样根据 PROJECT_ROOT 进行动态拼接
    parser.add_argument(
        "--input_path", 
        type=str, 
        default=os.path.join(PROJECT_ROOT, "data/ForgeryAnalysis_Stage_2_Test/Image"), 
        help="输入数据集路径 (存放测试图片的目录)"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default=os.path.join(PROJECT_ROOT, "results_val/final_submission.csv"), 
        help="最终推理结果输出的 CSV 文件路径"
    )
    args = parser.parse_args()

    # 如果有需要，可以单独为 masks 指定一个文件夹
    mask_output_dir = os.path.join(os.path.dirname(args.output_path), "masks")
    os.makedirs(mask_output_dir, exist_ok=True)

    print(f"📌 数据集输入目录: {os.path.abspath(args.input_path)}")
    print(f"📌 最终输出文件: {os.path.abspath(args.output_path)}")

    # ==================================
    # 流水线数据在内存中直接传递 (df)
    # ==================================
    
    # 阶段一：获取初始推理 DataFrame
    df_stage1 = run_stage1_sam(args.input_path, mask_output_dir)
    
    # 阶段二：传入 DataFrame，由 vLLM 进行多模态核验并添加解释
    df_stage2 = run_stage2_llm(df_stage1, args.input_path)
    
    # 阶段三：对 DataFrame 进行最终清洗，并写出到指定的 output_path
    run_stage3_postprocess(df_stage2, args.output_path)