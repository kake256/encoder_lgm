# =========================
# main_feature_bank.py
# (Feature Bank 対応メインスクリプト: 進捗表示追加版)
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

# model_utils_bank.py から必要なクラスをインポート
from model_utils_bank import (
    EncoderClassifier,
    MultiModelGM,
    FeatureBankSystem,
    manage_model_allocation,
)

# ==============================================================================
# 1. 評価用クラス (Evaluator)
# ==============================================================================
class Evaluator:
    def __init__(self, device):
        self.device = device
        # SSIM / LPIPS (類似度評価)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)
        
        # ResNet50 (Visual Score評価用: 生成画像が正しくクラス分類されるか)
        weights = models.ResNet50_Weights.DEFAULT
        self.resnet = models.resnet50(weights=weights).to(device)
        self.resnet.eval()
        self.resnet_transform = weights.transforms()
        
        # EasyOCR (文字認識評価用)
        use_gpu_ocr = (device.type == "cuda")
        self.reader = easyocr.Reader(["en"], gpu=use_gpu_ocr, verbose=False)

    def preprocess_tensor(self, img_tensor):
        # 評価用に224x224にリサイズ
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
        # OCRはCPU/NumPyで行う
        img_cpu = img_tensor.detach().cpu().squeeze(0)
        img_pil = to_pil_image(img_cpu)
        img_np = np.array(img_pil)
        
        results = self.reader.readtext(img_np)
        max_score = 0.0
        detected_text = ""
        target_clean = target_text.lower().strip()
        
        for _bbox, text, conf in results:
            text_clean = text.lower().strip()
            # 部分一致判定
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
        # tqdm使用時の表示崩れを防ぐため tqdm.write を使用
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


def add_text_overlay(pil_image, text, font_scale=0.15, color=(255, 0, 0)):
    """実画像にテキストを合成する (Overlay Text攻撃の再現用)"""
    if text is None or str(text).strip() == "":
        return pil_image

    img = pil_image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    font_size = int(h * font_scale)
    try:
        # 一般的なLinux環境のフォントパス
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
# 2. Args Parse
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Model LGM with Feature Bank")

    # モデル設定
    parser.add_argument("--encoder_names", type=str, nargs="+", required=True)
    parser.add_argument("--projection_dims", type=int, nargs="+", default=[2048])
    parser.add_argument("--experiments", type=str, nargs="+", required=True)

    # 出力・データ設定
    parser.add_argument("--output_dir", type=str, default="./lgm_results")
    parser.add_argument("--target_classes", type=str, nargs="+", default=[])
    parser.add_argument("--initial_image_path", type=str, default=None)
    parser.add_argument("--seed_noise_level", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_file", type=str, default="training_log.txt")

    # 学習パラメータ
    parser.add_argument("--num_iterations", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--pyramid_start_res", type=int, default=16)
    parser.add_argument("--pyramid_grow_interval", type=int, default=400)
    parser.add_argument("--weight_grad", type=float, default=1.0)
    parser.add_argument("--weight_tv", type=float, default=0.00025)

    # Feature Bank & Dataset Scale Params
    parser.add_argument("--use_real", type=int, default=0, help="Number of real images to use (0 = ALL from split)")
    parser.add_argument("--real_aug", type=int, default=10, help="Augmentations per real image for bank creation")
    parser.add_argument("--syn_aug", type=int, default=32, help="Augmentations per synthetic step (Batch size)")
    parser.add_argument("--cache_dir", type=str, default="./feature_cache")

    parser.add_argument("--num_generations", type=int, default=1)

    # 文字攻撃設定
    parser.add_argument("--overlay_text", type=str, default="ipod")
    parser.add_argument("--text_color", type=str, default="red")
    parser.add_argument("--font_scale", type=float, default=0.15)

    # ImageNet用
    parser.add_argument("--dataset_type", type=str, default="imagenet")
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--dataset_split", type=str, default="train")

    # Augmentation制御
    parser.add_argument("--min_scale", type=float, default=0.08)
    parser.add_argument("--max_scale", type=float, default=1.0)
    parser.add_argument("--noise_prob", type=float, default=0.5)
    parser.add_argument("--noise_std", type=float, default=0.2)

    # Streaming Mode (ダウンロードしながら読み込む)
    parser.add_argument("--streaming", action="store_true", help="Use streaming dataset mode (recommended for large datasets)")

    return parser.parse_args()


# ==============================================================================
# 3. Main Logic
# ==============================================================================
if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = Logger(args.log_file)
    logger.log(f"Using GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "Using CPU")
    logger.log(f"Config: Use Real={args.use_real} (0=All), Real Augs={args.real_aug}, Syn Augs={args.syn_aug}")
    logger.log(f"Cache Dir: {args.cache_dir}")
    logger.log(f"Dataset: {args.dataset_type} split={args.dataset_split} streaming={args.streaming}")

    # Feature Bank System の初期化
    bank_system = FeatureBankSystem(args, device, args.cache_dir)

    # データセットロード
    # streaming=True の場合、Downloadせずネットワーク越しに読み込む
    try:
        dataset = load_dataset(
            "imagenet-1k",
            split=args.dataset_split,
            streaming=args.streaming,
            trust_remote_code=True
        )
        # クラス名リストの取得 (Streamingだとfeaturesメタデータが取れない場合があるので安全策)
        try:
            dataset_class_names = dataset.features["label"].names
        except Exception:
            # 取得できない場合は数字で代用
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

    # モデルの準備 (Feature Bank作成用 & 最適化用)
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

    total_steps = len(target_ids_to_run) * len(args.experiments) * args.num_generations * args.num_iterations
    global_pbar = tqdm(total=total_steps, unit="step", desc="Total Progress", position=0, dynamic_ncols=True)

    evaluation_results = []

    # --- クラスごとのループ ---
    for target_cls in target_ids_to_run:
        class_name = dataset_class_names[target_cls] if target_cls < len(dataset_class_names) else str(target_cls)
        logger.log(f"--- Processing Class {target_cls}: {class_name} ---")

        # use_real=0 なら全量、それ以外なら指定枚数でストップ
        max_need = args.use_real if args.use_real > 0 else None

        raw_images_tensor_list = []
        ref_clean_tensor = None
        safe_cls_name = sanitize_dirname(class_name)

        found = 0
        scanned_count = 0  # スキャン済み枚数のカウンタ

        # === データセット走査 (進捗表示付き) ===
        # streaming=True の場合、目的の画像が見つかるまで数万枚スキップすることもあるため進捗を表示
        logger.log(f"Scanning dataset stream for class {target_cls}...")
        
        for item in dataset:
            scanned_count += 1
            
            # 5000枚ごとに進捗を上書き表示 (\r で行頭に戻る)
            if scanned_count % 5000 == 0:
                sys.stdout.write(f"\r[Data Scan] Scanned: {scanned_count} images... (Found: {found} target images)")
                sys.stdout.flush()

            if int(item["label"]) != int(target_cls):
                continue

            # ターゲットクラスの画像を発見
            img = item["image"].convert("RGB").resize((args.image_size, args.image_size))
            
            # 評価用(SSIM計算用)にクリーンな画像を1枚キープ
            if ref_clean_tensor is None:
                ref_clean_tensor = to_tensor(img).unsqueeze(0).to(device)

            # 文字攻撃 (Overlay Text) を適用
            img_with_text = add_text_overlay(img, args.overlay_text, args.font_scale, args.text_color)
            raw_images_tensor_list.append(to_tensor(img_with_text))
            
            found += 1

            # 指定枚数に達したら終了
            if max_need is not None and found >= max_need:
                break
        
        print("") # 進捗表示の行を確定させるために改行
        
        if not raw_images_tensor_list:
            logger.log(f"Warning: No images found for class {target_cls}. Skipping.")
            continue
        
        logger.log(f"Collected {len(raw_images_tensor_list)} images for class {target_cls}.")

        # 画像プールを作成 (Feature Bank作成用, 一旦CPU保持)
        real_images_pool = torch.stack(raw_images_tensor_list)

        # === Feature Bank の作成 または ロード ===
        # (ここで重い計算が走るが、Utils側のtqdmで進捗が出る)
        feature_bank_list = bank_system.create_or_load_bank(models_list, real_images_pool, target_cls, logger)

        # --- 実験設定ごとのループ ---
        for exp_str in args.experiments:
            exp_name, weights_str = exp_str.split(":")
            weights = [float(w) for w in weights_str.split(",")]

            # モデルのGPU割り当て (重み0のモデルはCPUへ退避してメモリ節約)
            manage_model_allocation(models_list, weights, device)

            save_dir = os.path.join(args.output_dir, exp_name, f"{target_cls}_{safe_cls_name}")
            os.makedirs(save_dir, exist_ok=True)

            # 生成数のループ (デフォルト1回)
            for gen_idx in range(args.num_generations):
                current_seed = args.seed + gen_idx
                set_seed(current_seed)
                
                gen_subdir = os.path.join(save_dir, f"gen_{gen_idx:02d}")
                os.makedirs(gen_subdir, exist_ok=True)

                # 最適化の実行
                lgm = MultiModelGM(models_list, weights, target_cls, args, device)
                final_img, best_img, metrics = lgm.run(
                    feature_bank_list,
                    gen_subdir,
                    dataset_class_names,
                    logger,
                    global_pbar,
                    gen_idx,
                )

                # 結果の保存
                target_img_tensor = best_img if best_img is not None else final_img
                target_img_tensor = target_img_tensor.to(device)

                save_image(target_img_tensor, os.path.join(gen_subdir, f"result_gen{gen_idx:02d}.png"))
                save_image(target_img_tensor, os.path.join(save_dir, f"result_gen{gen_idx:02d}.png"))
                
                with open(os.path.join(gen_subdir, f"metrics_gen{gen_idx:02d}.json"), "w") as f:
                    json.dump({"args": vars(args), "metrics": metrics}, f, indent=2)

                # === 評価 (Evaluation) ===
                temp_evaluator = Evaluator(device)
                
                # 1. Visual Score (ResNet50による認識スコア)
                vis_score, _, _ = temp_evaluator.calc_visual_score(target_img_tensor, target_cls)
                
                # 2. Text Score (OCRによる文字認識スコア)
                text_score, _detected = temp_evaluator.calc_text_score(target_img_tensor, args.overlay_text)
                
                # 3. Similarity (元画像との類似度)
                ssim_val, lpips_val = temp_evaluator.calc_similarity(target_img_tensor, ref_clean_tensor)
                
                # メモリ解放
                del temp_evaluator
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # 結果集計
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

    global_pbar.close()

    # 最終結果をCSV保存
    df = pd.DataFrame(evaluation_results)
    csv_path = os.path.join(args.output_dir, "final_evaluation.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(csv_path, index=False)
    
    logger.log(f"All experiments finished. Evaluation saved to {csv_path}")
    logger.close()