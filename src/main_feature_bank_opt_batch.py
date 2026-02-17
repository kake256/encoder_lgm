# =========================
# src/main_feature_bank_batch.py
# =========================
import os
import sys
import argparse
import json
import re
import random
import numpy as np
import torch
import pandas as pd
from PIL import ImageDraw, ImageFont
from torchvision.utils import save_image
from torchvision.transforms.functional import to_tensor, to_pil_image
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from torchvision import models
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import easyocr

from model_utils_bank_opt_batch import (
    EncoderClassifier, MultiModelGM, manage_model_allocation,
)

class Evaluator:
    def __init__(self, device, use_ocr=True):
        self.device = device
        self.use_ocr = use_ocr
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)
        weights = models.ResNet50_Weights.DEFAULT
        self.resnet = models.resnet50(weights=weights).to(device)
        self.resnet.eval()
        self.resnet_transform = weights.transforms()
        self.reader = easyocr.Reader(["en"], gpu=(device.type == "cuda"), verbose=False) if use_ocr else None

    def preprocess_tensor(self, img_tensor): return F.interpolate(img_tensor, size=(224, 224), mode="bilinear", align_corners=False)

    def calc_visual_score(self, img_tensor, target_class_idx):
        img_tensor = self.preprocess_tensor(img_tensor)
        with torch.no_grad():
            probs = F.softmax(self.resnet(self.resnet_transform(img_tensor)), dim=1)
            score = probs[0, target_class_idx].item() if target_class_idx < 1000 else 0.0
            top1_prob, top1_idx = torch.max(probs, dim=1)
        return score, top1_idx.item(), top1_prob.item()

    def calc_text_score(self, img_tensor, target_text):
        if not self.use_ocr or not target_text: return 0.0, ""
        results = self.reader.readtext(np.array(to_pil_image(img_tensor.detach().cpu().squeeze(0))))
        max_score, detected_text, target_clean = 0.0, "", target_text.lower().strip()
        for _bbox, text, conf in results:
            if target_clean in text.lower().strip() and conf > max_score:
                max_score, detected_text = conf, text
        return max_score, detected_text

class Logger:
    def __init__(self, log_path=None):
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            self.file = open(log_path, "w", encoding="utf-8")
        else: self.file = None
    def log(self, message):
        tqdm.write(message)
        if self.file: self.file.write(message + "\n"); self.file.flush()
    def close(self):
        if self.file: self.file.close()

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

def sanitize_dirname(name): return re.sub(r"[^a-zA-Z0-9]", "_", name.split(",")[0].strip()).strip("_")

def add_text_overlay(pil_image, text, font_scale=0.15, color="red"):
    if not text: return pil_image
    img = pil_image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", int(h * font_scale))
    except: font = ImageFont.load_default()
    try: left, top, right, bottom = draw.textbbox((0, 0), text, font=font); text_w, text_h = right - left, bottom - top
    except: text_w, text_h = draw.textsize(text, font=font)
    draw.text(((w - text_w) // 2, (h - text_h) // 2), text, fill=color, font=font)
    return img

def collect_target_images(dataset, target_class_ids, images_per_class, max_scan_global, image_size, logger):
    collected_images = {cls_id: [] for cls_id in target_class_ids}
    needs_collection = {cls_id: True for cls_id in target_class_ids}
    actual_scan_limit = min(len(dataset) if hasattr(dataset, "__len__") else max_scan_global, max_scan_global)
    
    with tqdm(total=actual_scan_limit, desc="Scanning & Caching PIL Images", unit="img") as pbar:
        for i, item in enumerate(dataset):
            if i >= actual_scan_limit: break
            pbar.update(1)
            try: label_idx = int(item["label"])
            except: continue
            if label_idx in collected_images and needs_collection[label_idx]:
                collected_images[label_idx].append(item["image"].convert("RGB").resize((image_size, image_size)))
                if len(collected_images[label_idx]) >= images_per_class: needs_collection[label_idx] = False
            if not any(needs_collection.values()): break
    return collected_images

def append_completion_log(output_dir, class_id, class_name, exp_name, gen_idx):
    import datetime
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "completion_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w", encoding="utf-8") as f: f.write("Timestamp,Class_ID,Class_Name,Experiment,Gen_Idx,Status\n")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{class_id},{class_name},{exp_name},{gen_idx},DONE\n")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_names", type=str, nargs="+", required=True)
    parser.add_argument("--projection_dims", type=int, nargs="+", default=[2048])
    parser.add_argument("--experiments", type=str, nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, default="./lgm_results")
    parser.add_argument("--target_classes", type=str, nargs="+", default=[])
    parser.add_argument("--seed_noise_level", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_file", type=str, default="training_log.txt")
    parser.add_argument("--num_iterations", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--batch_opt_size", type=int, default=10, help="Number of classes to optimize simultaneously")
    parser.add_argument("--pyramid_start_res", type=int, default=16)
    parser.add_argument("--pyramid_grow_interval", type=int, default=400)
    parser.add_argument("--weight_grad", type=float, default=1.0)
    parser.add_argument("--weight_tv", type=float, default=0.00025)
    parser.add_argument("--use_real", type=int, default=0)
    parser.add_argument("--max_scan_limit", type=int, default=1300000)
    parser.add_argument("--syn_aug", type=int, default=32)
    parser.add_argument("--num_generations", type=int, default=1)
    parser.add_argument("--overlay_text", type=str, default="")
    parser.add_argument("--text_color", type=str, default="red")
    parser.add_argument("--font_scale", type=float, default=0.15)
    parser.add_argument("--disable_ocr", action="store_true")
    parser.add_argument("--dataset_type", type=str, default="imagenet-1k")
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--min_scale", type=float, default=0.08)
    parser.add_argument("--max_scale", type=float, default=1.0)
    parser.add_argument("--noise_prob", type=float, default=0.5)
    parser.add_argument("--noise_std", type=float, default=0.2)
    parser.add_argument("--no_evaluation", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = Logger(args.log_file)
    
    target_ids_to_run = [int(t) for t in args.target_classes] if args.target_classes else []
    
    should_load_dataset = not args.no_evaluation
    dataset_class_names, collected_data_map = [], {}

    if should_load_dataset:
        dataset = load_dataset(args.dataset_type, split=args.dataset_split, streaming=False, trust_remote_code=True, cache_dir=args.data_root)
        try: dataset_class_names = dataset.features["label"].names
        except: pass
        
        images_per_class = args.use_real if args.use_real > 0 else 50
        collected_data_map = collect_target_images(dataset, target_ids_to_run, images_per_class, args.max_scan_limit, args.image_size, logger)

    if not dataset_class_names: dataset_class_names = [str(i) for i in range(1000)]
    
    active_model_indices = set()
    for exp_str in args.experiments:
        if ":" in exp_str:
            weights_str = exp_str.split(":")[1]
            weights = [float(w) for w in weights_str.split(",")]
            for idx, w in enumerate(weights):
                if w > 0: active_model_indices.add(idx)

    models_list = [None] * len(args.encoder_names)
    proj_dims = args.projection_dims * len(args.encoder_names) if len(args.projection_dims) == 1 else args.projection_dims
    
    for i in range(len(args.encoder_names)):
        if i in active_model_indices:
            name = args.encoder_names[i]
            p_dim = proj_dims[i]
            logger.log(f"Loading active model: {name}")
            m = EncoderClassifier(name, freeze_encoder=True, num_classes=len(dataset_class_names), projection_dim=p_dim).to(device).eval()
            models_list[i] = m

    evaluator = Evaluator(device, use_ocr=(not args.disable_ocr))
    evaluation_results = []

    total_steps = len(target_ids_to_run) * len(args.experiments) * args.num_generations * (args.num_iterations // args.batch_opt_size + 1)
    global_pbar = tqdm(total=total_steps, unit="step", desc="Total Progress")

    chunk_size = args.batch_opt_size
    for chunk_start in range(0, len(target_ids_to_run), chunk_size):
        batch_classes = target_ids_to_run[chunk_start:chunk_start+chunk_size]
        if not batch_classes: continue

        logger.log(f"--- Processing Batch: Classes {batch_classes} ---")
        
        # ★ 超高速化の要：ループ中のPIL処理を無くすため、全画像を事前にCPUテンソルに変換
        logger.log("Pre-converting real images to Tensors for maximum throughput...")
        real_images_tensor_list = []
        for tgt_cls in batch_classes:
            imgs = collected_data_map.get(tgt_cls, [])
            tensors = []
            for img_pil in imgs:
                if args.overlay_text:
                    img_pil = add_text_overlay(img_pil, args.overlay_text, args.font_scale, args.text_color)
                tensors.append(to_tensor(img_pil))
            if tensors:
                # [N, 3, H, W] の形でCPUメモリに保持
                real_images_tensor_list.append(torch.stack(tensors))
            else:
                real_images_tensor_list.append(torch.randn(10, 3, args.image_size, args.image_size))
                
        for exp_str in args.experiments:
            exp_name, weights_str = exp_str.split(":")
            weights = [float(w) for w in weights_str.split(",")]
            manage_model_allocation(models_list, weights, device)
            
            for gen_idx in range(args.num_generations):
                set_seed(args.seed + gen_idx)
                
                # ★ 途中経過保存用のディレクトリを事前に作成
                save_dirs = []
                for b_idx, target_cls in enumerate(batch_classes):
                    class_name = dataset_class_names[target_cls] if target_cls < len(dataset_class_names) else str(target_cls)
                    safe_cls_name = sanitize_dirname(class_name)
                    save_dir = os.path.join(args.output_dir, exp_name, f"{target_cls}_{safe_cls_name}")
                    gen_subdir = os.path.join(save_dir, f"gen_{gen_idx:02d}")
                    os.makedirs(gen_subdir, exist_ok=True)
                    save_dirs.append(gen_subdir)
                
                # LGMの実行 (PILリストではなく Tensorリストを渡す, save_dirsを追加)
                lgm = MultiModelGM(models_list, weights, batch_classes, args, device)
                final_batch_img, best_batch_img, metrics = lgm.run(real_images_tensor_list, global_pbar, gen_idx, save_dirs)
                
                target_img_batch = best_batch_img if best_batch_img is not None else final_batch_img
                target_img_batch = target_img_batch.to(device)
                
                # 最終画像の保存と評価
                for b_idx, target_cls in enumerate(batch_classes):
                    img_tensor = target_img_batch[b_idx].unsqueeze(0)
                    gen_subdir = save_dirs[b_idx]
                    save_dir = os.path.dirname(gen_subdir)
                    
                    save_image(img_tensor, os.path.join(gen_subdir, "result.png"))
                    save_image(img_tensor, os.path.join(save_dir, f"result_gen{gen_idx:02d}.png"))
                    
                    if not args.no_evaluation:
                        vis_score, _, _ = evaluator.calc_visual_score(img_tensor, target_cls)
                        text_score, _ = evaluator.calc_text_score(img_tensor, args.overlay_text)
                        evaluation_results.append({
                            "exp_name": exp_name, "class_id": target_cls, "class_name": dataset_class_names[target_cls] if target_cls < len(dataset_class_names) else str(target_cls),
                            "gen_idx": gen_idx, "result_visual_score": vis_score, "result_text_score": text_score
                        })
                    append_completion_log(args.output_dir, target_cls, dataset_class_names[target_cls] if target_cls < len(dataset_class_names) else str(target_cls), exp_name, gen_idx)

            if torch.cuda.is_available(): torch.cuda.empty_cache()

    global_pbar.close()
    if evaluation_results:
        pd.DataFrame(evaluation_results).to_csv(os.path.join(args.output_dir, "final_evaluation.csv"), index=False)
    logger.close()