# =========================
# main_feature_bank.py
# (Priority A: Download Mode + HF Cache Save)
# =========================
import os
import sys
import argparse
import json
import re
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision.utils import save_image
from torchvision.transforms.functional import to_tensor, to_pil_image
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
import pandas as pd

# 評価用ライブラリ
from torchvision import models
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import easyocr

from model_utils_bank import (
    EncoderClassifier,
    MultiModelGM,
    FeatureBankSystem,
    manage_model_allocation,
)

# ... (Evaluator, Logger, set_seed, sanitize_dirname, add_text_overlay は変更なし) ...
# (前回のコードの該当部分をそのまま使用してください)
class Evaluator:
    def __init__(self, device, use_ocr=True):
        self.device = device
        self.use_ocr = use_ocr
        # SSIM / LPIPS
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)
        
        # ResNet50
        weights = models.ResNet50_Weights.DEFAULT
        self.resnet = models.resnet50(weights=weights).to(device)
        self.resnet.eval()
        self.resnet_transform = weights.transforms()
        
        # EasyOCR
        if self.use_ocr:
            use_gpu_ocr = (device.type == "cuda")
            self.reader = easyocr.Reader(["en"], gpu=use_gpu_ocr, verbose=False)
        else:
            self.reader = None

    def preprocess_tensor(self, img_tensor):
        return F.interpolate(img_tensor, size=(224, 224), mode="bilinear", align_corners=False)

    def calc_visual_score(self, img_tensor, target_class_idx):
        img_tensor = self.preprocess_tensor(img_tensor)
        with torch.no_grad():
            processed = self.resnet_transform(img_tensor)
            logits = self.resnet(processed)
            probs = F.softmax(logits, dim=1)
            score = probs[0, target_class_idx].item()
            top1_prob, top1_idx = torch.max(probs, dim=1)
        return score, top1_idx.item(), top1_prob.item()

    def calc_text_score(self, img_tensor, target_text):
        if not self.use_ocr or not target_text:
            return 0.0, ""
        img_cpu = img_tensor.detach().cpu().squeeze(0)
        img_pil = to_pil_image(img_cpu)
        img_np = np.array(img_pil)
        
        results = self.reader.readtext(img_np)
        max_score = 0.0
        detected_text = ""
        target_clean = target_text.lower().strip()
        
        for _bbox, text, conf in results:
            text_clean = text.lower().strip()
            if target_clean in text_clean or text_clean in target_clean:
                if conf > max_score:
                    max_score = conf
                    detected_text = text
        return max_score, detected_text

    def calc_similarity(self, gen_tensor, ref_tensor):
        if ref_tensor.device != self.device:
            ref_tensor = ref_tensor.to(self.device)
        gen_resized = self.preprocess_tensor(gen_tensor)
        ref_resized = self.preprocess_tensor(ref_tensor)
        with torch.no_grad():
            ssim = self.ssim_metric(gen_resized, ref_resized).item()
            lpips = self.lpips_metric(gen_resized, ref_resized).item()
        return ssim, lpips

class Logger:
    def __init__(self, log_path=None):
        self.log_path = log_path
        if log_path:
            log_dir = os.path.dirname(log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            self.file = open(log_path, "w", encoding="utf-8")
        else:
            self.file = None
    def log(self, message):
        tqdm.write(message)
        if self.file:
            self.file.write(message + "\n")
            self.file.flush()
    def close(self):
        if self.file:
            self.file.close()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

def sanitize_dirname(name):
    safe_name = re.sub(r"[^a-zA-Z0-9]", "_", name)
    return safe_name.strip("_")

def add_text_overlay(pil_image, text, font_scale=0.15, color="red"):
    if text is None or str(text).strip() == "":
        return pil_image
    img = pil_image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    font_size = int(h * font_scale)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_w = right - left
        text_h = bottom - top
    except Exception:
        text_w, text_h = draw.textsize(text, font=font)
    x = (w - text_w) // 2
    y = (h - text_h) // 2
    draw.text((x, y), text, fill=color, font=font)
    return img

# ==============================================================================
# 2. Data Collection Utility (Modified for Downloaded Dataset)
# ==============================================================================
def collect_target_images(dataset, target_class_ids, images_per_class, max_scan_global, image_size, logger):
    """
    データセットを走査し、必要なクラスの画像をPIL形式で収集する。
    streaming=Falseの場合は全データがダウンロード済みなので、len(dataset)を使ってプログレスバーを表示。
    """
    collected_images = {cls_id: [] for cls_id in target_class_ids}
    needs_collection = {cls_id: True for cls_id in target_class_ids}
    
    scanned_count = 0
    found_total = 0
    
    # データセット全体の長さを取得（streaming=Falseなら取得可能）
    total_samples = len(dataset) if hasattr(dataset, "__len__") else max_scan_global
    # スキャン上限は「データセット全体」か「指定された上限」の小さい方
    actual_scan_limit = min(total_samples, max_scan_global)

    logger.log(f"Starting Dataset Scan (Total: {total_samples} images)...")
    
    with tqdm(total=actual_scan_limit, desc="Scanning", unit="img") as pbar:
        for i, item in enumerate(dataset):
            if i >= actual_scan_limit:
                break

            scanned_count += 1
            pbar.update(1)
            
            try:
                label_idx = int(item["label"])
            except:
                continue

            if label_idx in collected_images and needs_collection[label_idx]:
                img = item["image"].convert("RGB").resize((image_size, image_size))
                collected_images[label_idx].append(img)
                found_total += 1
                
                if len(collected_images[label_idx]) >= images_per_class:
                    needs_collection[label_idx] = False
                    
            if not any(needs_collection.values()):
                logger.log("All target classes collected.")
                break
                
    return collected_images


# ==============================================================================
# 3. Args Parse (Modified defaults)
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Model LGM with Feature Bank (Download Mode)")

    parser.add_argument("--encoder_names", type=str, nargs="+", required=True)
    parser.add_argument("--projection_dims", type=int, nargs="+", default=[2048])
    parser.add_argument("--experiments", type=str, nargs="+", required=True)

    parser.add_argument("--output_dir", type=str, default="./lgm_results")
    parser.add_argument("--target_classes", type=str, nargs="+", default=[])
    parser.add_argument("--initial_image_path", type=str, default=None)
    parser.add_argument("--seed_noise_level", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_file", type=str, default="training_log.txt")

    parser.add_argument("--num_iterations", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--pyramid_start_res", type=int, default=16)
    parser.add_argument("--pyramid_grow_interval", type=int, default=400)
    parser.add_argument("--weight_grad", type=float, default=1.0)
    parser.add_argument("--weight_tv", type=float, default=0.00025)

    parser.add_argument("--use_real", type=int, default=0)
    parser.add_argument("--max_scan_limit", type=int, default=1300000, help="Limit for scanning (default set to cover full ImageNet-1k)")
    parser.add_argument("--real_aug", type=int, default=10)
    parser.add_argument("--syn_aug", type=int, default=32)
    parser.add_argument("--cache_dir", type=str, default="./feature_cache")

    parser.add_argument("--num_generations", type=int, default=1)
    parser.add_argument("--overlay_text", type=str, default="ipod")
    parser.add_argument("--text_color", type=str, default="red")
    parser.add_argument("--font_scale", type=float, default=0.15)
    parser.add_argument("--disable_ocr", action="store_true")

    parser.add_argument("--dataset_type", type=str, default="imagenet-1k")
    # data_root を HFキャッシュ先として利用します
    parser.add_argument("--data_root", type=str, default="", help="Path to save/load HF dataset cache")
    parser.add_argument("--dataset_split", type=str, default="train")

    parser.add_argument("--min_scale", type=float, default=0.08)
    parser.add_argument("--max_scale", type=float, default=1.0)
    parser.add_argument("--noise_prob", type=float, default=0.5)
    parser.add_argument("--noise_std", type=float, default=0.2)

    # streaming フラグを削除 (常に False として扱うため)
    # 必要なら残して default=False にしても良いですが、今回は明確化のため削除または無視します

    return parser.parse_args()


# ==============================================================================
# 4. Main Logic
# ==============================================================================
if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = Logger(args.log_file)
    logger.log(f"Using GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "Using CPU")
    logger.log(f"Config: RealImgs={args.use_real}, RealAug={args.real_aug}, SynAug={args.syn_aug}")
    
    bank_system = FeatureBankSystem(args, device, args.cache_dir)

    # --- Dataset Loading (Streaming=False) ---
    logger.log("Loading Dataset (Download mode)...")
    
    # data_root があればそれをキャッシュディレクトリとして使う
    hf_cache_dir = args.data_root if args.data_root else None
    if hf_cache_dir:
        logger.log(f"Using cache directory: {hf_cache_dir}")
        os.makedirs(hf_cache_dir, exist_ok=True)

    try:
        # streaming=False を明示的に指定
        dataset = load_dataset(
            args.dataset_type,
            split=args.dataset_split,
            streaming=False,  # ここをFalseに変更
            trust_remote_code=True,
            cache_dir=hf_cache_dir 
        )
        try:
            dataset_class_names = dataset.features["label"].names
        except Exception:
            dataset_class_names = [str(i) for i in range(1000)]
            
        logger.log(f"Dataset loaded. Total samples: {len(dataset)}")
        
    except Exception as e:
        logger.log(f"Error loading dataset: {e}")
        logger.log("Please check if you have access to the dataset (e.g., ImageNet requires Gated Access on HF).")
        sys.exit(1)

    # --- Target Preparation ---
    target_ids_to_run = []
    if not args.target_classes:
        logger.log("No target classes specified.")
        sys.exit(0)

    for t in args.target_classes:
        try:
            target_ids_to_run.append(int(t))
        except:
            pass
            
    # --- Image Collection ---
    images_per_class = args.use_real if args.use_real > 0 else 50
    if args.use_real == 0:
        logger.log(f"Note: --use_real 0 specified. Defaulting to collecting {images_per_class} images per class.")

    # 収集処理の呼び出し
    collected_data_map = collect_target_images(
        dataset, 
        target_ids_to_run, 
        images_per_class, 
        args.max_scan_limit, 
        args.image_size,
        logger
    )

    # --- 以下、モデル初期化〜実験ループは前回と同じ ---
    models_list = []
    proj_dims = args.projection_dims
    if len(proj_dims) == 1:
        proj_dims = proj_dims * len(args.encoder_names)

    logger.log("Initializing Models...")
    for name, p_dim in zip(args.encoder_names, proj_dims):
        m = EncoderClassifier(name, freeze_encoder=True, num_classes=1000, projection_dim=p_dim)
        m.to(device)
        m.eval()
        models_list.append(m)

    evaluator = Evaluator(device, use_ocr=(not args.disable_ocr))

    total_steps = len(target_ids_to_run) * len(args.experiments) * args.num_generations * args.num_iterations
    global_pbar = tqdm(total=total_steps, unit="step", desc="Total Progress", position=0, dynamic_ncols=True)
    evaluation_results = []

    for target_cls in target_ids_to_run:
        class_name = dataset_class_names[target_cls] if target_cls < len(dataset_class_names) else str(target_cls)
        logger.log(f"--- Processing Class {target_cls}: {class_name} ---")

        pil_images_list = collected_data_map.get(target_cls, [])
        if not pil_images_list:
            logger.log(f"Skipping class {target_cls} (No images found).")
            continue
            
        logger.log(f"Using {len(pil_images_list)} real images for feature bank.")
        
        raw_images_tensor_list = []
        ref_clean_tensor_cpu = to_tensor(pil_images_list[0]).unsqueeze(0) 

        for p_img in pil_images_list:
            img_with_text = add_text_overlay(p_img, args.overlay_text, args.font_scale, args.text_color)
            raw_images_tensor_list.append(to_tensor(img_with_text))
            
        real_images_pool = torch.stack(raw_images_tensor_list)
        
        feature_bank_list = bank_system.create_or_load_bank(models_list, real_images_pool, target_cls, logger)
        
        del real_images_pool, raw_images_tensor_list
        import gc; gc.collect()

        safe_cls_name = sanitize_dirname(class_name)

        for exp_str in args.experiments:
            exp_name, weights_str = exp_str.split(":")
            weights = [float(w) for w in weights_str.split(",")]

            manage_model_allocation(models_list, weights, device)

            save_dir = os.path.join(args.output_dir, exp_name, f"{target_cls}_{safe_cls_name}")
            os.makedirs(save_dir, exist_ok=True)

            for gen_idx in range(args.num_generations):
                current_seed = args.seed + gen_idx
                set_seed(current_seed)
                
                gen_subdir = os.path.join(save_dir, f"gen_{gen_idx:02d}")
                os.makedirs(gen_subdir, exist_ok=True)

                lgm = MultiModelGM(models_list, weights, target_cls, args, device)
                final_img, best_img, metrics = lgm.run(
                    feature_bank_list,
                    gen_subdir,
                    dataset_class_names,
                    logger,
                    global_pbar,
                    gen_idx,
                )

                target_img_tensor = best_img if best_img is not None else final_img
                target_img_tensor = target_img_tensor.to(device)

                save_image(target_img_tensor, os.path.join(gen_subdir, f"result_gen{gen_idx:02d}.png"))
                save_image(target_img_tensor, os.path.join(save_dir, f"result_gen{gen_idx:02d}.png"))
                
                with open(os.path.join(gen_subdir, f"metrics_gen{gen_idx:02d}.json"), "w") as f:
                    json.dump({"args": vars(args), "metrics": metrics}, f, indent=2)

                vis_score, _, _ = evaluator.calc_visual_score(target_img_tensor, target_cls)
                text_score, _ = evaluator.calc_text_score(target_img_tensor, args.overlay_text)
                ssim_val, lpips_val = evaluator.calc_similarity(target_img_tensor, ref_clean_tensor_cpu)

                result_entry = {
                    "exp_name": exp_name,
                    "class_id": target_cls,
                    "class_name": class_name,
                    "gen_idx": gen_idx,
                    "overlay_text": args.overlay_text,
                    "result_visual_score": vis_score,
                    "result_text_score": text_score,
                    "ssim": ssim_val,
                    "lpips": lpips_val,
                }
                evaluation_results.append(result_entry)
                logger.log(f"   [Eval] Visual:{vis_score:.3f} Text:{text_score:.3f} SSIM:{ssim_val:.3f}")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    global_pbar.close()
    
    df = pd.DataFrame(evaluation_results)
    csv_path = os.path.join(args.output_dir, "final_evaluation.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(csv_path, index=False)
    
    logger.log(f"All experiments finished. Evaluation saved to {csv_path}")
    logger.close()