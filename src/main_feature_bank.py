# =========================
# main_feature_bank.py
# (Feature Bank 対応メイン: streaming走査, cache_only, skip_cache)
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

from model_utils_bank import (
    EncoderClassifier,
    MultiModelGM,
    FeatureBankSystem,
    manage_model_allocation,
)

# ==============================================================================
# Logger
# ==============================================================================
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
        # TF32設定は警告が出るが致命的ではない. 必要なら新APIへ移行.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def sanitize_dirname(name):
    safe_name = re.sub(r"[^a-zA-Z0-9]", "_", str(name))
    return safe_name.strip("_")


def add_text_overlay(pil_image, text, font_scale=0.15, color=(255, 0, 0)):
    if text is None or str(text).strip() == "":
        return pil_image

    img = pil_image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    font_size = int(h * float(font_scale))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    try:
        left, top, right, bottom = draw.textbbox((0, 0), str(text), font=font)
        text_w = right - left
        text_h = bottom - top
    except Exception:
        text_w, text_h = draw.textsize(str(text), font=font)

    x = (w - text_w) // 2
    y = (h - text_h) // 2
    draw.text((x, y), str(text), fill=color, font=font)
    return img


# ==============================================================================
# Args
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Model LGM with Feature Bank")

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
    parser.add_argument("--pyramid_start_res", type=int, default=16)
    parser.add_argument("--pyramid_grow_interval", type=int, default=400)
    parser.add_argument("--weight_grad", type=float, default=1.0)
    parser.add_argument("--weight_tv", type=float, default=0.00025)

    parser.add_argument("--use_real", type=int, default=0)
    parser.add_argument("--real_aug", type=int, default=10)
    parser.add_argument("--syn_aug", type=int, default=32)
    parser.add_argument("--cache_dir", type=str, default="./feature_cache")

    parser.add_argument("--num_generations", type=int, default=1)

    parser.add_argument("--overlay_text", type=str, default="ipod")
    parser.add_argument("--text_color", type=str, default="red")
    parser.add_argument("--font_scale", type=float, default=0.15)

    parser.add_argument("--dataset_type", type=str, default="imagenet")
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--dataset_split", type=str, default="train")

    parser.add_argument("--min_scale", type=float, default=0.08)
    parser.add_argument("--max_scale", type=float, default=1.0)
    parser.add_argument("--noise_prob", type=float, default=0.5)
    parser.add_argument("--noise_std", type=float, default=0.2)

    parser.add_argument("--streaming", action="store_true")

    # 2段運用
    parser.add_argument("--cache_only", action="store_true", help="Build/load feature cache only, then exit.")
    parser.add_argument("--skip_cache", action="store_true", help="Do not compute cache. Error if cache missing.")

    return parser.parse_args()


# ==============================================================================
# 評価器(遅延import)
# ==============================================================================
def build_evaluator(device):
    """
    cache_only のときは呼ばれない想定.
    重い拡張モジュールを import するので, 生成フェーズでのみ呼ぶ.
    """
    from torchvision import models
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    import easyocr

    class Evaluator:
        def __init__(self, device):
            self.device = device
            self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
            self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)

            weights = models.ResNet50_Weights.DEFAULT
            self.resnet = models.resnet50(weights=weights).to(device)
            self.resnet.eval()
            self.resnet_transform = weights.transforms()

            use_gpu_ocr = (device.type == "cuda")
            self.reader = easyocr.Reader(["en"], gpu=use_gpu_ocr, verbose=False)

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
            if not target_text:
                return 0.0, ""
            img_cpu = img_tensor.detach().cpu().squeeze(0)
            img_pil = to_pil_image(img_cpu)
            img_np = np.array(img_pil)

            results = self.reader.readtext(img_np)
            max_score = 0.0
            detected_text = ""
            target_clean = str(target_text).lower().strip()

            for _bbox, text, conf in results:
                text_clean = str(text).lower().strip()
                if target_clean in text_clean or text_clean in target_clean:
                    if conf > max_score:
                        max_score = conf
                        detected_text = text
            return max_score, detected_text

        def calc_similarity(self, gen_tensor, ref_tensor):
            gen_resized = self.preprocess_tensor(gen_tensor)
            ref_resized = self.preprocess_tensor(ref_tensor)
            with torch.no_grad():
                ssim = self.ssim_metric(gen_resized, ref_resized).item()
                lpips = self.lpips_metric(gen_resized, ref_resized).item()
            return ssim, lpips

    return Evaluator(device)


# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = Logger(args.log_file)
    logger.log(f"Using GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "Using CPU")
    logger.log(f"Dataset: {args.dataset_type} split={args.dataset_split} streaming={args.streaming}")
    logger.log(f"Cache Dir: {args.cache_dir}")
    logger.log(f"Mode: cache_only={args.cache_only}, skip_cache={args.skip_cache}")

    if args.cache_only and args.skip_cache:
        logger.log("Error: --cache_only and --skip_cache cannot be used together.")
        sys.exit(1)

    bank_system = FeatureBankSystem(args, device, args.cache_dir)

    # trust_remote_code は削除(非対応になったため)
    try:
        dataset = load_dataset(
            "imagenet-1k",
            split=args.dataset_split,
            streaming=args.streaming,
        )
        try:
            dataset_class_names = dataset.features["label"].names
        except Exception:
            dataset_class_names = [str(i) for i in range(1000)]
    except Exception as e:
        logger.log(f"Error: Failed to load ImageNet. Reason: {e}")
        sys.exit(1)

    target_ids_to_run = []
    if not args.target_classes:
        logger.log("No target classes specified. Exiting.")
        sys.exit(0)

    for t in args.target_classes:
        try:
            target_ids_to_run.append(int(t))
        except Exception:
            pass

    # モデル初期化
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

    # progress bar は生成時のみ
    if not args.cache_only:
        total_steps = len(target_ids_to_run) * len(args.experiments) * args.num_generations * args.num_iterations
        global_pbar = tqdm(total=total_steps, unit="step", desc="Total Progress", position=0, dynamic_ncols=True)
    else:
        global_pbar = None

    evaluation_results = []

    for target_cls in target_ids_to_run:
        class_name = dataset_class_names[target_cls] if target_cls < len(dataset_class_names) else str(target_cls)
        safe_cls_name = sanitize_dirname(class_name)
        logger.log(f"--- Processing Class {target_cls}: {class_name} ---")

        # skip_cache: 走査なしでキャッシュを読む(この関数は model_utils_bank.py 側に実装が必要)
        if args.skip_cache:
            feature_bank_list = bank_system.load_bank_only(models_list, target_cls, logger)
            ref_clean_tensor = None
            if args.cache_only:
                logger.log(f"[cache_only] Finished cache check/load for class {target_cls}.")
                continue
        else:
            # 走査して real_images_pool を構築
            max_need = args.use_real if args.use_real > 0 else None
            raw_images_tensor_list = []
            ref_clean_tensor = None

            found = 0
            scanned_count = 0
            logger.log(f"Scanning dataset stream for class {target_cls}...")

            for item in dataset:
                scanned_count += 1
                if scanned_count % 5000 == 0:
                    sys.stdout.write(
                        f"\r[Data Scan] Scanned: {scanned_count} images... (Found: {found} target images)"
                    )
                    sys.stdout.flush()

                if int(item["label"]) != int(target_cls):
                    continue

                img = item["image"].convert("RGB").resize((args.image_size, args.image_size))

                if ref_clean_tensor is None:
                    ref_clean_tensor = to_tensor(img).unsqueeze(0).to(device)

                img_with_text = add_text_overlay(img, args.overlay_text, args.font_scale, args.text_color)
                raw_images_tensor_list.append(to_tensor(img_with_text))

                found += 1
                if max_need is not None and found >= max_need:
                    break

            print("")

            if not raw_images_tensor_list:
                logger.log(f"Warning: No images found for class {target_cls}. Skipping.")
                continue

            logger.log(f"Collected {len(raw_images_tensor_list)} images for class {target_cls}.")
            real_images_pool = torch.stack(raw_images_tensor_list)

            feature_bank_list = bank_system.create_or_load_bank(
                models_list,
                real_images_pool,
                target_cls,
                logger,
                require_cache=False,
            )

            if args.cache_only:
                logger.log(f"[cache_only] Finished cache for class {target_cls}.")
                continue

        # 生成フェーズ
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

                evaluator = build_evaluator(device)
                vis_score, _, _ = evaluator.calc_visual_score(target_img_tensor, target_cls)
                text_score, _detected = evaluator.calc_text_score(target_img_tensor, args.overlay_text)

                if ref_clean_tensor is not None:
                    ssim_val, lpips_val = evaluator.calc_similarity(target_img_tensor, ref_clean_tensor)
                else:
                    ssim_val, lpips_val = None, None

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                evaluation_results.append(
                    {
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
                )

                if ssim_val is None:
                    logger.log(f"   [Eval] Visual:{vis_score:.3f} Text:{text_score:.3f} SSIM/LPIPS: skipped")
                else:
                    logger.log(f"   [Eval] Visual:{vis_score:.3f} Text:{text_score:.3f} SSIM:{ssim_val:.3f}")

    if global_pbar is not None:
        global_pbar.close()

    # CSV保存は生成時のみ, pandas も生成時のみ import する(終了クラッシュ回避に効く)
    if not args.cache_only:
        import pandas as pd

        df = pd.DataFrame(evaluation_results)
        csv_path = os.path.join(args.output_dir, "final_evaluation.csv")
        os.makedirs(args.output_dir, exist_ok=True)
        df.to_csv(csv_path, index=False)
        logger.log(f"All experiments finished. Evaluation saved to {csv_path}")

    logger.close()
