import json
import os
import argparse
import datetime  # 新增：用于在日志中添加时间戳
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
import csv
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.model_selection import KFold

from forensics_sam import ForensicsSAM
from mini_dataloader import (
    BasicDataloader,
)
from segment_anything import sam_model_registry

# 获取当前文件所在目录、src目录以及项目根目录的绝对路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # 当前: src/stage2_final
SRC_DIR = os.path.dirname(CURRENT_DIR)                    # 上级: src
ROOT_DIR = os.path.dirname(SRC_DIR)                       # 根目录: 提交/

# ==============================================================================
# ============================ 核心参数配置区域 ============================
# ==============================================================================
class Config:
    # ---------------- 数据与路径配置 ----------------
    data_root = os.path.join(ROOT_DIR, "data", "ForgeryAnalysis_Stage_1_Train")            # 训练数据集的根目录
    output_dir = os.path.join(ROOT_DIR, "models", "adapters", "stage2_final", "weight")    # 模型权重及训练日志的输出保存路径
    
    # 【新增】全局日志路径配置
    global_log_dir = os.path.join(ROOT_DIR, "logs", "stage2_final")
    global_log_path = os.path.join(global_log_dir, "train.log")
    
    best_model_name = "forensics_stage1_best.pth"                    # 在验证集上取得最佳指标的模型保存名称
    last_model_name = "forensics_stage1_last.pth"                    # 最后一轮训练结束后的模型保存名称

    # ---------------- 模型与权重配置 ----------------
    sam_type = "vit_h"                                               # SAM 主干网络的模型规模类型 (vit_b, vit_l, vit_h)
    sam_checkpoint = os.path.join(ROOT_DIR, "models", "sam_vit_h_4b8939.pth")                  # SAM 官方预训练权重的加载路径
    rank = 8                                                         # LoRA / Adapter 等微调结构的秩 (Rank) 大小
    init_forensics_weights = os.path.join(ROOT_DIR, "models", "forgery_experts.pth")           # 初始化伪造检测专家模型的权重路径
    resume = ""                                                      # 恢复训练的断点权重路径（留空表示从头训练）

    # ---------------- 训练超参数配置 ----------------
    k_folds = 5                                                      # K折交叉验证的折数
    epochs = 10                                                      # 每次 K折交叉验证中，单折的训练轮数
    full_epochs = 5                                                  # 使用全量数据集进行最终训练的轮数
    batch_size = 2                                                   # 每次迭代的批处理大小 (Batch Size)
    lr = 1e-4                                                        # 初始学习率 (Learning Rate)
    weight_decay = 0                                                 # 优化器的权重衰减惩罚系数 (Weight Decay)
    num_workers = 8                                                  # DataLoader 加载数据的子进程数量
    seed = 42                                                        # 全局随机种子，用于确保实验的可复现性
    device = "cuda" if torch.cuda.is_available() else "cpu"          # 训练使用的硬件设备 (GPU 或 CPU)

    # ---------------- 图像与处理配置 ----------------
    image_size = 1024                                                # 网络输入图像的分辨率大小
    normalize_type = 2                                               # 图像归一化方式的策略枚举值

    # ---------------- 评估与阈值配置 ----------------
    threshold = 0.5                                                  # 默认的二分类预测概率判定阈值
    search_threshold_on_val = True                                   # 是否开启在验证集上动态搜索最佳 F1 分数阈值
    threshold_search_min = 0.01                                      # 动态搜索分类阈值的下界
    threshold_search_max = 0.99                                      # 动态搜索分类阈值的上界
    threshold_search_step = 0.01                                     # 动态搜索分类阈值的步长

    # ---------------- 数据增强配置 (Augmentation) ----------------
    augment_prob = 0.0                                               # 触发数据增强的整体概率 (0.0 表示完全关闭数据增强)
    enable_aug_types = ""                                            # 启用的特定数据增强类型的编号/名称集合 (以逗号分隔)
    rates = "0.8"                                                    # 增强操作参数：下采样/缩放等操作的比例率
    qfs = "75"                                                       # 增强操作参数：JPEG 压缩的质量因子 (Quality Factor)
    sds = "9"                                                        # 增强操作参数：高斯模糊等操作的标准差 (Standard Deviation)
    ksizes = "9"                                                     # 增强操作参数：模糊滤波等操作的卷积核大小 (Kernel Size)

# 实例化配置对象
args = Config()

# 针对 data_root 增加命令行参数解析
parser = argparse.ArgumentParser(description="Forensics SAM Training")
parser.add_argument("--data_root", type=str, default=args.data_root, help="指定训练数据集的根目录")
cmd_args, _ = parser.parse_known_args()
args.data_root = cmd_args.data_root
# ==============================================================================


MODEL_TYPES = ["vit_b", "vit_l", "vit_h"]
# 同样修复备用的 SAM 路径
DEFAULT_SAM_CHECKPOINTS = {
    "vit_b": os.path.join(ROOT_DIR, "models", "sam_vit_b_01ec64.pth"),
    "vit_l": os.path.join(ROOT_DIR, "models", "sam_vit_l_0b3195.pth"),
    "vit_h": os.path.join(ROOT_DIR, "models", "sam_vit_h_4b8939.pth"),
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class DummyRNGDataset(Dataset):
    """
    用于消耗 PyTorch 随机数生成器 (RNG) 状态的空数据集，仅用于对齐随机状态，不加载实际图像数据。
    """
    def __init__(self, total_samples):
        self.total_samples = total_samples

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        return 0 


def _safe_join(root: str, *parts: str) -> str:
    root_abs = os.path.abspath(root)
    target = os.path.abspath(os.path.join(root_abs, *parts))
    if os.path.commonpath([root_abs, target]) != root_abs:
        raise ValueError(f"Path escapes root directory: {target}")
    return target

def _resolve_mask_path(mask_dir: str, image_name: str, strict: bool) -> str:
    stem, ext = os.path.splitext(image_name)
    candidates = [
        _safe_join(mask_dir, image_name),
        _safe_join(mask_dir, f"{stem}.png"),
        _safe_join(mask_dir, f"{stem}{ext}"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    if strict:
        raise FileNotFoundError(f"Mask file not found for {image_name} in {mask_dir}")
    return candidates[1]

def build_train_records(data_root: str, adv_label: int = 0, strict: bool = True) -> list:
    black_image_dir = _safe_join(data_root, "Black", "Image")
    black_mask_dir = _safe_join(data_root, "Black", "Mask")
    white_image_dir = _safe_join(data_root, "White", "Image")

    if not os.path.isdir(black_image_dir):
        raise FileNotFoundError(f"Black image directory not found: {black_image_dir}")
    if not os.path.isdir(white_image_dir):
        raise FileNotFoundError(f"White image directory not found: {white_image_dir}")

    records = []
    
    for image_name in os.listdir(black_image_dir):
        ext = os.path.splitext(image_name)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            continue
        forgery_path = _safe_join(black_image_dir, image_name)
        gt_mask_path = _resolve_mask_path(black_mask_dir, image_name, strict=strict)
        records.append({
            "forgery_path": forgery_path,
            "gt_mask_path": gt_mask_path,
            "forged_label": 1,
            "adv_label": adv_label,
            "image_name": image_name,
        })
    
    for image_name in os.listdir(white_image_dir):
        ext = os.path.splitext(image_name)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            continue
        forgery_path = _safe_join(white_image_dir, image_name)
        records.append({
            "forgery_path": forgery_path,
            "gt_mask_path": "",
            "forged_label": 0,
            "adv_label": adv_label,
            "image_name": image_name,
        })

    if strict and len(records) == 0:
        raise ValueError(f"No training images found in {black_image_dir} or {white_image_dir}")
    
    print(f"自动扫描训练集完成：共{len(records)}张图像，正样本{sum(1 for r in records if r['forged_label'] == 1)}张，负样本{sum(1 for r in records if r['forged_label'] == 0)}张")
    return records


class AverageMeter:
    def __init__(self) -> None:
        self.reset()
    def reset(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.avg = 0.0
    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_int_list(value: str) -> List[int]:
    value = value.strip()
    return [int(item.strip()) for item in value.split(",") if item.strip()] if value else []

def parse_float_list(value: str) -> List[float]:
    value = value.strip()
    return [float(item.strip()) for item in value.split(",") if item.strip()] if value else []

def resolve_sam_checkpoint(sam_type: str, sam_checkpoint: str) -> str:
    if sam_checkpoint:
        return sam_checkpoint
    return DEFAULT_SAM_CHECKPOINTS[sam_type]

def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(dim=1)
    denominator = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()

def binary_cls_metrics_from_probs(probs: np.ndarray, labels: np.ndarray, threshold: float, eps: float = 1e-8) -> Dict[str, float]:
    if probs.shape[0] != labels.shape[0]:
        raise ValueError("probs and labels must have the same length")
    if probs.size == 0:
        return {"acc": 0.0, "pos_precision": 0.0, "pos_recall": 0.0, "pos_f1": 0.0,
                "neg_precision": 0.0, "neg_recall": 0.0, "neg_f1": 0.0, 
                "macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0,
                "tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0}

    pred = (probs > threshold).astype(np.float32)
    target = labels.astype(np.float32)

    acc = float(np.mean(pred == target))
    tp = float(np.sum(pred * target))
    fp = float(np.sum(pred * (1.0 - target)))
    fn = float(np.sum((1.0 - pred) * target))
    tn = float(np.sum((1.0 - pred) * (1.0 - target)))
    
    pos_precision = tp / (tp + fp + eps)
    pos_recall = tp / (tp + fn + eps)
    pos_f1 = float((2.0 * pos_precision * pos_recall) / (pos_precision + pos_recall + eps))
    
    neg_precision = tn / (tn + fn + eps)
    neg_recall = tn / (tn + fp + eps)
    neg_f1 = float((2.0 * neg_precision * neg_recall) / (neg_precision + neg_recall + eps))
    
    macro_precision = (pos_precision + neg_precision) / 2.0
    macro_recall = (pos_recall + neg_recall) / 2.0
    macro_f1 = float((2.0 * macro_precision * macro_recall) / (macro_precision + macro_recall + eps))
    
    return {
        "acc": acc, 
        "pos_precision": pos_precision, "pos_recall": pos_recall, "pos_f1": pos_f1,
        "neg_precision": neg_precision, "neg_recall": neg_recall, "neg_f1": neg_f1,
        "macro_precision": macro_precision, "macro_recall": macro_recall, "macro_f1": macro_f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }

def search_best_cls_threshold(probs: np.ndarray, labels: np.ndarray, threshold_min: float, threshold_max: float, threshold_step: float) -> Dict[str, float]:
    if probs.size == 0: raise ValueError("Cannot search threshold: no validation samples.")
    
    thresholds = np.arange(threshold_min, threshold_max + threshold_step * 0.5, threshold_step, dtype=np.float32)
    
    best_threshold = float(thresholds[0])
    best_metrics = binary_cls_metrics_from_probs(probs, labels, best_threshold)
    best_score = best_metrics["macro_f1"] 

    for threshold in thresholds[1:]:
        threshold = float(threshold)
        metrics = binary_cls_metrics_from_probs(probs, labels, threshold)
        score = metrics["macro_f1"]
        
        score_improved = score > best_score + 1e-12
        same_score_better_tie = abs(score - best_score) <= 1e-12 and abs(threshold - 0.5) < abs(best_threshold - 0.5)
        
        if score_improved or same_score_better_tie:
            best_threshold = threshold
            best_score = score
            best_metrics = metrics

    return {
        "threshold": best_threshold,
        **best_metrics
    }

def save_threshold_metadata(path: str, threshold: float, epoch: int, cls_acc: float, cls_f1: float, source: str) -> None:
    payload = {"threshold": float(threshold), "epoch": int(epoch), "cls_acc": float(cls_acc), "cls_f1": float(cls_f1), "source": source}
    output_dir = os.path.dirname(path)
    if output_dir: os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f: json.dump(payload, f, indent=2)

def build_model(args: Config, device: torch.device, resume_path: str = None) -> ForensicsSAM:
    sam_checkpoint = resolve_sam_checkpoint(args.sam_type, args.sam_checkpoint)
    sam, _ = sam_model_registry[args.sam_type](image_size=args.image_size, checkpoint=sam_checkpoint)

    init_forensics = args.init_forensics_weights if args.init_forensics_weights else None
    model = ForensicsSAM(
        sam, r=args.rank, forgery_experts_path=init_forensics, adversary_experts_path=None,
        load_pretrained=bool(init_forensics), freeze_shared_experts=False, freeze_detector=False, enable_adversary_experts=False,
    )
    
    load_path = resume_path if resume_path else args.resume
    if load_path:
        model.load_all_parameters(load_path)
    
    model.to(device)
    return model

def evaluate(model: ForensicsSAM, loader: DataLoader, device: torch.device, args: Config) -> Dict[str, object]:
    bce_criterion = torch.nn.BCEWithLogitsLoss()
    loss_meter = AverageMeter()
    cls_probs: List[float] = []
    cls_targets: List[float] = []

    model.eval()
    with torch.no_grad():
        for images, _, forged_label, _ in loader:
            images = images.to(device, non_blocking=True)
            forged_label = forged_label.float().to(device, non_blocking=True).unsqueeze(1)
            activate_adv = torch.zeros(images.size(0), device=device, dtype=torch.long)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                _, cls_logits = model(images, activate_adv)
                cls_bce = bce_criterion(cls_logits, forged_label)
                loss = cls_bce

            cls_prob = torch.sigmoid(cls_logits.float())
            cls_probs.extend(cls_prob.view(-1).detach().cpu().tolist())
            cls_targets.extend(forged_label.view(-1).detach().cpu().tolist())

            batch_size = images.size(0)
            loss_meter.update(loss.item(), batch_size)

    probs_arr = np.asarray(cls_probs, dtype=np.float32)
    targets_arr = np.asarray(cls_targets, dtype=np.float32)

    if args.search_threshold_on_val:
        best_metrics = search_best_cls_threshold(
            probs=probs_arr, labels=targets_arr, 
            threshold_min=args.threshold_search_min, threshold_max=args.threshold_search_max, threshold_step=args.threshold_search_step
        )
    else:
        best_metrics = binary_cls_metrics_from_probs(probs_arr, targets_arr, args.threshold)
        best_metrics["threshold"] = args.threshold

    return {"loss": loss_meter.avg, **best_metrics, "cls_probs": cls_probs, "cls_targets": cls_targets}


def run_training_session(args: Config, device: torch.device, 
                         train_records: list, val_records: list, 
                         session_dir: str, session_name: str,
                         custom_epochs: int = None,
                         skip_eval_but_consume_rng: bool = False) -> int:
    total_epochs = custom_epochs if custom_epochs is not None else args.epochs

    # 确保局部权重目录和全局日志目录存在
    os.makedirs(session_dir, exist_ok=True)
    os.makedirs(args.global_log_dir, exist_ok=True)
    
    # 1. 局部日志文件 (覆盖模式)
    log_file_path = os.path.join(session_dir, f"training_log_{session_name}.txt")
    log_file = open(log_file_path, "w", encoding="utf-8")
    
    # 2. 全局日志文件 (追加模式)
    global_log_file = open(args.global_log_path, "a", encoding="utf-8")

    def print_and_log(message: str):
        print(message)
        # 写入局部日志
        log_file.write(message + "\n")
        log_file.flush() 
        # 写入全局日志并刷新缓存
        global_log_file.write(message + "\n")
        global_log_file.flush()

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print_and_log(f"\n{'='*80}\n🚀 [{current_time}] 开始训练 Session: {session_name}\n数据分布: Train={len(train_records)} | Val={len(val_records) if val_records else 0}\n保存路径: {session_dir}\n设定Epochs: {total_epochs}\n局部日志: {log_file_path}\n全局日志: {args.global_log_path}\n{'='*80}")
    
    enable_aug_types = parse_int_list(args.enable_aug_types)
    intensity = {
        "rates": parse_float_list(args.rates), "qfs": [int(v) for v in parse_float_list(args.qfs)],
        "sds": parse_float_list(args.sds), "ksizes": [int(v) for v in parse_float_list(args.ksizes)],
    }

    train_dataset = BasicDataloader(sample_records=train_records, input_size=args.image_size, normalize_type=args.normalize_type, augment_prob=args.augment_prob, enable_aug_types=enable_aug_types, intensity=intensity, mode="train")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    val_loader = None
    if len(val_records) > 0:
        val_dataset = BasicDataloader(sample_records=val_records, input_size=args.image_size, normalize_type=args.normalize_type, augment_prob=0.0, enable_aug_types=[], intensity=intensity, mode="val")
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    model = build_model(args, device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay, fused=True)
    bce_criterion = torch.nn.BCEWithLogitsLoss()

    best_path = os.path.join(session_dir, args.best_model_name)
    last_path = os.path.join(session_dir, args.last_model_name)
    best_threshold_path = f"{os.path.splitext(best_path)[0]}_threshold.json"
    last_threshold_path = f"{os.path.splitext(last_path)[0]}_threshold.json"

    best_score = float("-inf")
    best_epoch = 1

    for epoch in range(1, total_epochs + 1):
        model.train()
        train_loss_meter = AverageMeter()
        train_cls_acc_meter = AverageMeter()

        pbar = tqdm(train_loader, desc=f'[{session_name}] Epoch {epoch}/{total_epochs} [Train]')
        
        for step, (images, gt_masks, forged_label, _) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            gt_masks = gt_masks.to(device, non_blocking=True)
            forged_label = forged_label.float().to(device, non_blocking=True).unsqueeze(1)
            activate_adv = torch.zeros(images.size(0), device=device, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                mask_logits, cls_logits = model(images, activate_adv)
                seg_bce = bce_criterion(mask_logits, gt_masks)
                seg_dice = dice_loss_from_logits(mask_logits, gt_masks)
                cls_bce = bce_criterion(cls_logits, forged_label)
                loss = seg_bce + seg_dice + cls_bce

            loss.backward()
            optimizer.step()

            current_lr = optimizer.param_groups[0]['lr']

            cls_prob = torch.sigmoid(cls_logits.float())
            cls_pred = (cls_prob > args.threshold).float()
            cls_acc = (cls_pred == forged_label).float().mean().item()

            batch_size = images.size(0)
            train_loss_meter.update(loss.item(), batch_size)
            train_cls_acc_meter.update(cls_acc, batch_size)
            
            step_log = f"Epoch: {epoch:03d}/{total_epochs:03d} | Step: {step+1:04d}/{len(train_loader):04d} | Loss: {loss.item():.6f} | LR: {current_lr:.6e}"
            log_file.write(step_log + "\n")
            global_log_file.write(step_log + "\n")  # 追加到全局日志

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{train_loss_meter.avg:.4f}', 'acc': f'{train_cls_acc_meter.avg:.4f}', 'lr': f'{current_lr:.2e}'})

        # 保证当前 epoch 的日志立即落盘
        log_file.flush() 
        global_log_file.flush() 

        model.save_all_parameters(last_path)

        if val_loader is not None and not skip_eval_but_consume_rng:
            metrics = evaluate(model, val_loader, device, args)
            selected_threshold = float(metrics["threshold"])
            selected_cls_acc = float(metrics["acc"])
            selected_cls_f1 = float(metrics["pos_f1"])
            threshold_source = "searched_on_val" if args.search_threshold_on_val else "fixed"
            score_for_selection = float(metrics.get("macro_f1", selected_cls_acc)) 
            
            save_threshold_metadata(last_threshold_path, threshold=selected_threshold, epoch=epoch, cls_acc=selected_cls_acc, cls_f1=selected_cls_f1, source=threshold_source)

            if score_for_selection > best_score:
                best_score = score_for_selection
                best_epoch = epoch
                model.save_all_parameters(best_path)
                save_threshold_metadata(best_threshold_path, threshold=selected_threshold, epoch=epoch, cls_acc=selected_cls_acc, cls_f1=selected_cls_f1, source=threshold_source)

            print_and_log(f'\nEpoch {epoch:02d}/{total_epochs} | Train Loss: {train_loss_meter.avg:.4f} | Train Acc: {train_cls_acc_meter.avg:.4f}')
            print_and_log(f'🔹 Val 验证报告（自动搜索最优阈值={selected_threshold:.3f}，来源={threshold_source}）：')
            print_and_log(f'  Val Loss  : {metrics["loss"]:.4f}')
            print_and_log(f'  [混淆矩阵] :')
            print_and_log(f'              预测:未伪造(0)   预测:伪造(1)')
            print_and_log(f'  真实:未伪造(0)      {int(metrics["tn"]):<12} {int(metrics["fp"]):<12}')
            print_and_log(f'  真实:伪造(1)        {int(metrics["fn"]):<12} {int(metrics["tp"]):<12}')
            print_and_log(f'  ------------------------------------------------------')
            print_and_log(f'  准确率(AC): {selected_cls_acc:.4f}')
            print_and_log(f'  未伪造(0) : Recall={metrics["neg_recall"]:.4f} | F1={metrics["neg_f1"]:.4f}')
            print_and_log(f'  伪造(1)   : Recall={metrics["pos_recall"]:.4f} | F1={metrics["pos_f1"]:.4f}')
            print_and_log(f'  宏平均    : Recall={metrics.get("macro_recall", 0):.4f} | F1={metrics.get("macro_f1", 0):.4f}')
            print_and_log(f'💡 当前最高得分(Macro F1/Acc): {best_score:.4f} (出现在 Epoch {best_epoch})\n')

        elif skip_eval_but_consume_rng:
            dummy_dataset = DummyRNGDataset(total_samples=700)
            dummy_loader = DataLoader(
                dummy_dataset, 
                batch_size=args.batch_size, 
                shuffle=False, 
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda")
            )

            for _ in dummy_loader:
                pass 
                
            best_epoch = total_epochs 
            print_and_log(f'\nEpoch {epoch:02d}/{total_epochs} | Train Loss: {train_loss_meter.avg:.4f} | Train Acc: {train_cls_acc_meter.avg:.4f}')
            print_and_log(f'🔸 全量数据训练中，已使用空数据集 (700样本) 对齐打乱 DataLoader 随机数状态。当前已完成 {epoch}/{total_epochs} 轮。\n')
        
        else:
            best_epoch = total_epochs 
            print_and_log(f'\nEpoch {epoch:02d}/{total_epochs} | Train Loss: {train_loss_meter.avg:.4f} | Train Acc: {train_cls_acc_meter.avg:.4f}')
            print_and_log(f'🔸 训练中，不进行评估也不消耗多余随机数。当前已完成 {epoch}/{total_epochs} 轮。\n')

    print_and_log(f"\n[{session_name}] 训练完成! {'最优 Epoch: ' + str(best_epoch) if (val_loader is not None and not skip_eval_but_consume_rng) else '已跑完指定的 ' + str(total_epochs) + ' 轮'}，模型保存在: {session_dir}")
    
    log_file.close()
    global_log_file.close()  # 关闭全局日志流
    
    return best_epoch


def main() -> None:
    seed_everything(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available.")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_records = build_train_records(data_root=args.data_root, strict=True)
    if len(train_records) == 0:
        raise ValueError("No training records available.")
    
    # ================= 阶段 1：进行 K 折训练 =================
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    
    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(train_records)):
        fold_name = f"Fold_{fold_idx + 1}"
        session_dir = os.path.join(args.output_dir, f"training_kfold_{fold_idx + 1}")
        
        fold_train_records = [train_records[i] for i in train_indices]
        fold_val_records = [train_records[i] for i in val_indices]
        
        run_training_session(
            args=args, device=device,
            train_records=fold_train_records, val_records=fold_val_records,
            session_dir=session_dir, session_name=fold_name
        )
        
    # ================= 阶段 2：全量数据训练 =================
    full_session_dir = os.path.join(args.output_dir, "training_full_data")
    run_training_session(
        args=args, device=device,
        train_records=train_records, 
        val_records=[], 
        session_dir=full_session_dir, session_name="Full_Data",
        custom_epochs=args.full_epochs, 
        skip_eval_but_consume_rng=True 
    )
    
    print("\n🎉 K折模型及全量数据集模型训练全部结束！")

if __name__ == "__main__":
    main()