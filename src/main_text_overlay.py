import os
import sys
import argparse
import copy
import json
import re
import glob
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast
import torchvision.transforms as T
from torchvision.utils import save_image
from torchvision.transforms.functional import resize, to_tensor, to_pil_image
from tqdm import tqdm
from datasets import load_dataset

# 評価用ライブラリ
from torchvision import models
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import easyocr

# 既存のユーティリティ
from multi_model_utils import EncoderClassifier, PyramidGenerator, TVLoss

# ==============================================================================
# 1. 評価用クラス (Evaluator)
# ==============================================================================
class Evaluator:
    def __init__(self, device):
        self.device = device
        
        # 1. 構造・知覚評価用 (SSIM, LPIPS)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device)

        # 2. Visual Score用 (ResNet-50)
        weights = models.ResNet50_Weights.DEFAULT
        self.resnet = models.resnet50(weights=weights).to(device)
        self.resnet.eval()
        self.resnet_transform = weights.transforms()

        # 3. Text Score用 (EasyOCR)
        use_gpu_ocr = (device.type == 'cuda')
        self.reader = easyocr.Reader(['en'], gpu=use_gpu_ocr, verbose=False)

    def preprocess_tensor(self, img_tensor):
        """生成されたTensor(0-1, cuda)を評価用にリサイズ"""
        return F.interpolate(img_tensor, size=(224, 224), mode='bilinear', align_corners=False)

    def calc_visual_score(self, img_tensor, target_class_idx):
        """ResNet-50によるVisual Score"""
        img_tensor = self.preprocess_tensor(img_tensor)
        with torch.no_grad():
            processed = self.resnet_transform(img_tensor)
            logits = self.resnet(processed)
            probs = F.softmax(logits, dim=1)
            score = probs[0, target_class_idx].item()
            top1_prob, top1_idx = torch.max(probs, dim=1)
        return score, top1_idx.item(), top1_prob.item()

    def calc_text_score(self, img_tensor, target_text):
        """EasyOCRによるText Score"""
        if not target_text:
            return 0.0, ""
        
        # Tensor -> Numpy (H, W, C) uint8
        img_cpu = img_tensor.detach().cpu().squeeze(0) # [C, H, W]
        img_pil = to_pil_image(img_cpu)
        img_np = np.array(img_pil)

        results = self.reader.readtext(img_np)
        max_score = 0.0
        detected_text = ""
        target_clean = target_text.lower().strip()
        
        for bbox, text, conf in results:
            text_clean = text.lower().strip()
            # 簡易的な包含判定
            if target_clean in text_clean or text_clean in target_clean:
                if conf > max_score:
                    max_score = conf
                    detected_text = text
        
        return max_score, detected_text

    def calc_similarity(self, gen_tensor, ref_tensor):
        """SSIMとLPIPS (ref_tensorはClean画像)"""
        gen_resized = self.preprocess_tensor(gen_tensor)
        ref_resized = self.preprocess_tensor(ref_tensor)
        
        with torch.no_grad():
            ssim = self.ssim_metric(gen_resized, ref_resized).item()
            lpips = self.lpips_metric(gen_resized, ref_resized).item()
        return ssim, lpips

# ==============================================================================
# 2. Logger Class
# ==============================================================================
class Logger:
    def __init__(self, log_path=None):
        self.log_path = log_path
        if log_path:
            log_dir = os.path.dirname(log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            self.file = open(log_path, 'w', encoding='utf-8')
        else:
            self.file = None

    def log(self, message):
        tqdm.write(message)
        if self.file:
            self.file.write(message + '\n')
            self.file.flush()

    def close(self):
        if self.file:
            self.file.close()

# ==============================================================================
# 3. Helper Functions & 文字合成
# ==============================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def sanitize_dirname(name):
    safe_name = re.sub(r'[^a-zA-Z0-9]', '_', name)
    return safe_name.strip('_')

def add_text_overlay(pil_image, text, font_scale=0.15, color=(255, 0, 0)):
    """画像の中央付近に文字を書き込む"""
    img = pil_image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    
    font_size = int(h * font_scale)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()
        
    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_w = right - left
        text_h = bottom - top
    except:
        text_w, text_h = draw.textsize(text, font=font)
        
    x = (w - text_w) // 2
    y = (h - text_h) // 2
    
    draw.text((x, y), text, fill=color, font=font)
    return img

# ==============================================================================
# 4. MultiModelGM クラス
# ==============================================================================

class MultiModelGM:
    def __init__(self, models, model_weights, target_class, args, device, initial_image=None):
        self.models = models
        self.model_weights = model_weights
        self.target_class = target_class
        self.args = args
        self.device = device
        
        self.generator = PyramidGenerator(
            target_size=args.image_size,
            start_size=args.pyramid_start_res,
            activation='sigmoid',
            initial_image=initial_image,
            noise_level=args.seed_noise_level
        ).to(device)
        
        self.optimizer = self._init_optimizer()
        
        self.augmentor = T.Compose([
            T.RandomResizedCrop(args.image_size, scale=(0.8, 1.0), antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            T.ColorJitter(0.1, 0.1, 0.1, 0.05),
        ])
        
        self.scaler = torch.cuda.amp.GradScaler()
        self.tv_loss_fn = TVLoss().to(device)

    def _init_optimizer(self):
        return optim.Adam(self.generator.parameters(), lr=self.args.lr)

    def preprocess(self, img):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        return (img - mean) / std

    def get_grads(self, model, inputs, create_graph=False):
        params = list(model.classifier.parameters())
        logits = model(inputs)
        targets = torch.tensor([self.target_class] * inputs.size(0), device=self.device)
        loss = F.cross_entropy(logits, targets)
        grads = torch.autograd.grad(loss, params, create_graph=create_graph)
        flat_grad = torch.cat([g.view(-1) for g in grads])
        return flat_grad

    def optimize_step(self, real_images_pool):
        self.optimizer.zero_grad()
        syn_image = self.generator()
        
        with torch.no_grad():
            indices = torch.randperm(len(real_images_pool))[:self.args.num_ref_images]
            real_batch = real_images_pool[indices].detach()
            aug_real = self.augmentor(real_batch)
            inp_real = self.preprocess(aug_real)

        syn_batch_list = []
        for _ in range(self.args.augs_per_step):
            syn_batch_list.append(self.augmentor(syn_image))
        syn_batch = torch.cat(syn_batch_list, dim=0)
        inp_syn = self.preprocess(syn_batch)

        total_grad_loss = 0.0
        per_model_sims = {}
        
        for i, model in enumerate(self.models):
            weight = self.model_weights[i]
            if weight == 0: continue

            model.reset_classifier()
            
            with autocast():
                target_grad = self.get_grads(model, inp_real, create_graph=False)
            with autocast():
                syn_grad = self.get_grads(model, inp_syn, create_graph=True)
            
            sim = F.cosine_similarity(target_grad.unsqueeze(0).detach(), syn_grad.unsqueeze(0)).mean()
            loss_k = 1.0 - sim
            total_grad_loss += loss_k * weight
            
            model_name = self.args.encoder_names[i] if i < len(self.args.encoder_names) else f"model_{i}"
            per_model_sims[model_name] = sim.item()

        loss_tv = self.tv_loss_fn(syn_image)
        total_loss = (total_grad_loss * self.args.weight_grad) + (loss_tv * self.args.weight_tv)
        
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return total_grad_loss.item(), loss_tv.item(), per_model_sims, total_loss.item()

    def run(self, real_images_pool, save_dir, class_names, logger, global_pbar):
        logger.log(f"[{self.target_class}] Optimization Start. Pool size: {len(real_images_pool)}")
        loss_history = []
        
        best_loss = float('inf')
        best_img_tensor = None

        local_pbar = tqdm(range(self.args.num_iterations), desc=f"Exp {self.target_class}", leave=False, position=1, dynamic_ncols=True)
        
        for i in local_pbar:
            if i > 0 and i % self.args.pyramid_grow_interval == 0:
                if self.generator.extend():
                    self.optimizer = self._init_optimizer()
            
            l_grad, l_tv, model_sims, current_total_loss = self.optimize_step(real_images_pool)
            
            if current_total_loss < best_loss:
                best_loss = current_total_loss
                best_img_tensor = self.generator().detach().cpu()

            if i % 100 == 0:
                if logger.file:
                    logger.file.write(f"__PROGRESS__ {i}/{self.args.num_iterations} {l_grad:.4f}\n")

            step_metrics = {"loss_grad": l_grad, "loss_tv": l_tv, "total_loss": current_total_loss}
            for m_name, m_sim in model_sims.items():
                step_metrics[f"sim_{m_name}"] = m_sim
            loss_history.append(step_metrics)
            
            local_pbar.set_description(f"LossG: {l_grad:.3f} Res: {self.generator.levels[-1].shape[-1]}")
            global_pbar.update(1)

            if i % 500 == 0:
                with torch.no_grad():
                    save_image(self.generator().detach().cpu(), os.path.join(save_dir, f"step_{i:04d}.png"))

        final_img = self.generator().detach().cpu()
        return final_img, best_img_tensor, {"loss_history": loss_history}


# ==============================================================================
# 5. Main (文字合成と評価ループ)
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Model LGM with Text Overlay")
    
    # モデル設定
    parser.add_argument("--encoder_names", type=str, nargs='+', required=True, help="List of encoder models")
    parser.add_argument("--projection_dims", type=int, nargs='+', default=[2048], help="Projection dims")
    parser.add_argument("--experiments", type=str, nargs='+', required=True, help="List of experiments (name:w1,w2,w3)")
    
    # 出力・データ設定
    parser.add_argument("--output_dir", type=str, default="./lgm_text_attack_results")
    parser.add_argument("--target_classes", type=str, nargs='+', default=[], help="Target class IDs (ImageNet)")
    parser.add_argument("--initial_image_path", type=str, default=None)
    parser.add_argument("--seed_noise_level", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_file", type=str, default="training_log.txt")

    # 学習パラメータ
    parser.add_argument("--num_ref_images", type=int, default=10)
    parser.add_argument("--augs_per_step", type=int, default=32)
    parser.add_argument("--num_iterations", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--pyramid_start_res", type=int, default=16)
    parser.add_argument("--pyramid_grow_interval", type=int, default=400)
    parser.add_argument("--weight_grad", type=float, default=1.0)
    parser.add_argument("--weight_tv", type=float, default=0.00025)
    
    # 文字攻撃設定
    parser.add_argument("--overlay_text", type=str, default="ipod", help="Text to write on image (Conflict)")
    parser.add_argument("--text_color", type=str, default="red", help="Color of the text")
    parser.add_argument("--font_scale", type=float, default=0.15, help="Size of text relative to image")
    
    # ImageNet用 (SHスクリプトの互換性のため)
    parser.add_argument("--dataset_type", type=str, default="imagenet") 
    parser.add_argument("--data_root", type=str, default="")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger = Logger(args.log_file)
    logger.log(f"Using GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "Using CPU")

    # 評価器の初期化
    evaluator = Evaluator(device)

    # ImageNetデータのロード
    logger.log("Loading ImageNet validation set...")
    try:
        dataset = load_dataset("imagenet-1k", split="validation", streaming=False)
        dataset_class_names = dataset.features['label'].names
    except:
        logger.log("Error: Failed to load ImageNet. Ensure 'datasets' library is installed and logged in.")
        sys.exit(1)

    # ターゲットクラスIDのパース
    target_ids_to_run = []
    if not args.target_classes:
        logger.log("No target classes specified. Exiting.")
        sys.exit(0)
    
    for t in args.target_classes:
        try:
            target_ids_to_run.append(int(t))
        except:
            pass

    # モデルリスト初期化
    models_list = []
    proj_dims = args.projection_dims
    if len(proj_dims) == 1: proj_dims = proj_dims * len(args.encoder_names)
    
    logger.log("Initializing Models...")
    for name, p_dim in zip(args.encoder_names, proj_dims):
        m = EncoderClassifier(name, freeze_encoder=True, num_classes=1000, projection_dim=p_dim)
        m.to(device)
        m.eval()
        models_list.append(m)

    # プログレスバー
    total_steps = len(target_ids_to_run) * len(args.experiments) * args.num_iterations
    global_pbar = tqdm(total=total_steps, unit="step", desc="Total Progress", position=0, dynamic_ncols=True)

    # 実験結果をまとめるリスト
    evaluation_results = []

    # --- クラスループ ---
    for target_cls in target_ids_to_run:
        class_name = dataset_class_names[target_cls]
        safe_cls_name = sanitize_dirname(class_name)
        
        # 1. 参照画像の収集 (Clean)
        clean_images_pil = []
        MAX_POOL = 50
        for item in dataset:
            if item['label'] == target_cls:
                img = item['image'].convert('RGB').resize((args.image_size, args.image_size))
                clean_images_pil.append(img)
                if len(clean_images_pil) >= MAX_POOL: break
        
        if not clean_images_pil: continue
        
        # 2. 文字合成 (Text Overlay)
        overlay_text = args.overlay_text
        attacked_images_tensor = []
        
        # クリーン画像のTensor化（評価のReference用）
        ref_clean_tensor = to_tensor(clean_images_pil[0]).unsqueeze(0).to(device)

        # [新規追加] Clean画像の事前評価 (Baseline: 上限値)
        clean_vis_score, _, _ = evaluator.calc_visual_score(ref_clean_tensor, target_cls)
        clean_text_score, _ = evaluator.calc_text_score(ref_clean_tensor, overlay_text)

        # 攻撃画像作成ループ
        for img in clean_images_pil[:args.num_ref_images]: 
            img_with_text = add_text_overlay(img, overlay_text, args.font_scale, args.text_color)
            attacked_images_tensor.append(to_tensor(img_with_text))
            
            # [Debug] 最初の1枚だけ保存して確認
            if len(attacked_images_tensor) == 1:
                debug_dir = os.path.join(args.output_dir, "debug_inputs")
                os.makedirs(debug_dir, exist_ok=True)
                img_with_text.save(os.path.join(debug_dir, f"{target_cls}_{safe_cls_name}_attacked.png"))
        
        real_images_pool = torch.stack(attacked_images_tensor).to(device)

        # [新規追加] Attacked画像(入力)の事前評価 (Input Baseline: 攻撃成功度)
        # 代表として最初の1枚だけ評価
        ref_attacked_tensor = real_images_pool[0].unsqueeze(0)
        attacked_vis_score, _, _ = evaluator.calc_visual_score(ref_attacked_tensor, target_cls)
        attacked_text_score, _ = evaluator.calc_text_score(ref_attacked_tensor, overlay_text)

        # --- 実験ループ ---
        for exp_str in args.experiments:
            exp_name, weights_str = exp_str.split(':')
            weights = [float(w) for w in weights_str.split(',')]
            
            save_dir = os.path.join(args.output_dir, exp_name, f"{target_cls}_{safe_cls_name}")
            os.makedirs(save_dir, exist_ok=True)
            
            # 参照画像(攻撃済み)の保存
            save_image(real_images_pool[:16], os.path.join(save_dir, "ref_pool_attacked.png"), nrow=4)

            # 最適化実行
            lgm = MultiModelGM(models_list, weights, target_cls, args, device)
            final_img, best_img, metrics = lgm.run(real_images_pool, save_dir, dataset_class_names, logger, global_pbar)
            
            # 保存
            target_img_tensor = best_img if best_img is not None else final_img
            target_img_tensor = target_img_tensor.to(device) # 評価用にGPUへ
            save_image(target_img_tensor, os.path.join(save_dir, "result.png"))
            
            with open(os.path.join(save_dir, "metrics.json"), 'w') as f:
                json.dump({"args": vars(args), "metrics": metrics}, f, indent=2)

            # --- 即時評価 (On-the-fly Evaluation) ---
            # 1. Visual Score (ResNet) - Result
            vis_score, top1_idx, top1_prob = evaluator.calc_visual_score(target_img_tensor, target_cls)
            
            # 2. Text Score (OCR) - Result
            text_score, detected_text = evaluator.calc_text_score(target_img_tensor, overlay_text)
            
            # 3. SSIM / LPIPS (元画像=Clean画像と比較)
            ssim_val, lpips_val = evaluator.calc_similarity(target_img_tensor, ref_clean_tensor)
            
            # [更新] 結果記録 (Clean/Attacked/Result の比較が可能に)
            result_entry = {
                "exp_name": exp_name,
                "class_id": target_cls,
                "class_name": class_name,
                "overlay_text": overlay_text,
                
                # Baseline (Clean)
                "clean_visual_score": clean_vis_score,
                "clean_text_score": clean_text_score,

                # Input (Attacked)
                "attacked_visual_score": attacked_vis_score,
                "attacked_text_score": attacked_text_score,
                
                # Output (Result)
                "result_visual_score": vis_score,
                "result_text_score": text_score,
                
                # Similarity
                "ssim": ssim_val,
                "lpips": lpips_val,
                "detected_text": detected_text
            }
            evaluation_results.append(result_entry)
            
            logger.log(f"   [Eval] Visual: {vis_score:.4f} (Base:{clean_vis_score:.2f} In:{attacked_vis_score:.2f}), Text: {text_score:.4f} (In:{attacked_text_score:.2f})")

    global_pbar.close()
    
    # 評価結果の保存
    import pandas as pd
    df = pd.DataFrame(evaluation_results)
    csv_path = os.path.join(args.output_dir, "final_evaluation.csv")
    df.to_csv(csv_path, index=False)
    logger.log(f"All experiments finished. Evaluation saved to {csv_path}")
    logger.close()