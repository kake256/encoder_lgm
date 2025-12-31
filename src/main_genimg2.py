import os
import sys
import argparse
import copy
import json
import re
import glob
import random
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import torchvision.transforms as T
from torchvision.utils import save_image
from torchvision.transforms.functional import resize, to_tensor, to_pil_image
from tqdm import tqdm
from datasets import load_dataset

# 評価用
from torchvision import models
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import easyocr

# 最新モデル用 (DINOv3等)
try:
    import timm
except ImportError:
    timm = None

# ==============================================================================
# 1. Utility Functions
# ==============================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 高速化設定
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

def sanitize_dirname(name):
    safe_name = re.sub(r'[^a-zA-Z0-9]', '_', name)
    return safe_name.strip('_')

def add_text_overlay(pil_image, text, font_scale=0.15, color=(255, 0, 0)):
    if not text: return pil_image
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

class RandomGaussianNoise(nn.Module):
    def __init__(self, mean=0.0, std=1.0, p=0.5):
        super().__init__()
        self.mean = mean
        self.std = std
        self.p = p
    def forward(self, images):
        if torch.rand(1).item() < self.p: return images
        noise = torch.randn_like(images) * self.std + self.mean
        return images + noise

class Logger:
    def __init__(self, log_path=None, rank=0):
        self.log_path = log_path
        self.rank = rank
        if log_path:
            if "rank" not in log_path:
                base, ext = os.path.splitext(log_path)
                actual_path = f"{base}_gpu{rank}{ext}"
            else:
                actual_path = log_path
            log_dir = os.path.dirname(actual_path)
            if log_dir: os.makedirs(log_dir, exist_ok=True)
            self.file = open(actual_path, 'w', encoding='utf-8')
        else:
            self.file = None
    def log(self, message):
        prefix = f"[GPU {self.rank}] "
        tqdm.write(prefix + message)
        if self.file:
            self.file.write(message + '\n')
            self.file.flush()
    def close(self):
        if self.file: self.file.close()

# ==============================================================================
# 2. Models (Encoder & Evaluator)
# ==============================================================================

class EncoderClassifier(nn.Module):
    def __init__(self, encoder_model="openai/clip-vit-base-patch16", encoder=None, freeze_encoder=True, num_classes=1000, feature_source="pooler", projection_dim=2048):
        super().__init__()
        self.encoder_model_name = encoder_model
        self.use_timm_encoder = False 
        
        # --- エンコーダーロード ---
        if encoder is not None:
            self.encoder = encoder
            if not hasattr(encoder, 'config'): self.use_timm_encoder = True
        else:
            from transformers import CLIPVisionModel, Dinov2Model, ViTModel, SiglipVisionModel
            
            if "siglip" in self.encoder_model_name:
                self.encoder = SiglipVisionModel.from_pretrained(encoder_model)
            elif self.encoder_model_name.startswith("openai/clip") or "laion" in self.encoder_model_name:
                self.encoder = CLIPVisionModel.from_pretrained(encoder_model)
            elif "dinov2" in self.encoder_model_name and "timm" not in self.encoder_model_name:
                self.encoder = Dinov2Model.from_pretrained(encoder_model)
            elif "dino" in self.encoder_model_name and "v3" not in self.encoder_model_name and "timm" not in self.encoder_model_name:
                self.encoder = ViTModel.from_pretrained(encoder_model)
            elif "dinov3" in self.encoder_model_name or self.encoder_model_name.startswith("timm/"):
                if timm is None: raise ImportError("timm required for DINOv3/timm models")
                model_name = encoder_model.replace("timm/", "")
                self.encoder = timm.create_model(model_name, pretrained=True, num_classes=0)
                self.use_timm_encoder = True
            else:
                try:
                    self.encoder = ViTModel.from_pretrained(encoder_model)
                except:
                    raise ValueError(f"Unsupported model: {encoder_model}")

        # --- 特徴量ソース設定 ---
        if self.use_timm_encoder:
            self.embed_dim = self.encoder.num_features
            self.feature_source = feature_source 
        else:
            if "siglip" in self.encoder_model_name:
                self.feature_source = 'mean' if feature_source == 'pooler' else feature_source
            elif self.encoder_model_name.startswith("openai/clip"):
                self.feature_source = 'pooler' if feature_source not in ['cls', 'mean'] else feature_source
            elif "dinov2" in self.encoder_model_name:
                self.feature_source = 'cls' if feature_source == 'pooler' else feature_source
            elif "dino" in self.encoder_model_name:
                if feature_source == 'pooler' and not hasattr(self.encoder.config, 'pooler_type'):
                     self.feature_source = 'cls'
                else: self.feature_source = feature_source
            else:
                self.feature_source = feature_source
            cfg = self.encoder.config
            self.embed_dim = cfg.hidden_size

        if freeze_encoder:
            for p in self.encoder.parameters(): p.requires_grad = False

        if projection_dim > 0:
            self.projector = nn.Sequential(nn.Linear(self.embed_dim, projection_dim), nn.ReLU(inplace=True))
            classifier_in_dim = projection_dim
        else:
            self.projector = nn.Identity()
            classifier_in_dim = self.embed_dim

        self.classifier = nn.Linear(classifier_in_dim, num_classes)
        self.reset_classifier()

    def reset_classifier(self):
        self.classifier.weight.data.normal_(mean=0.0, std=0.01)
        self.classifier.bias.data.zero_()

    def extract_features(self, outputs):
        if isinstance(outputs, torch.Tensor): return outputs
        if "siglip" in self.encoder_model_name:
            if self.feature_source == 'pooler' and hasattr(outputs, 'pooler_output'): return outputs.pooler_output
            elif self.feature_source == 'mean': return outputs.last_hidden_state.mean(dim=1)
            else: return outputs.last_hidden_state[:, 0, :]
        elif self.encoder_model_name.startswith("openai/clip"):
            if self.feature_source == 'pooler': return outputs.pooler_output
            elif self.feature_source == 'cls': return outputs.last_hidden_state[:, 0, :]
            elif self.feature_source == 'mean': return outputs.last_hidden_state[:, 1:, :].mean(dim=1)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            return outputs.pooler_output
        if hasattr(outputs, 'last_hidden_state'):
             return outputs.last_hidden_state[:, 0, :]
        return outputs

    def forward_encoder(self, x):
        """最適化ループ用: Encoder出力だけを取り出す"""
        if self.use_timm_encoder: outputs = self.encoder(x)
        else: outputs = self.encoder(pixel_values=x, output_hidden_states=False)
        return self.extract_features(outputs)

    def forward(self, x):
        features = self.forward_encoder(x)
        projected = self.projector(features)
        return self.classifier(projected)

class Evaluator:
    def __init__(self, device):
        self.device = device
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device)
        weights = models.ResNet50_Weights.DEFAULT
        self.resnet = models.resnet50(weights=weights).to(device).eval()
        self.resnet_transform = weights.transforms()
        self.reader = easyocr.Reader(['en'], gpu=(device.type == 'cuda'), verbose=False)
    
    def preprocess_tensor(self, img_tensor):
        return F.interpolate(img_tensor, size=(224, 224), mode='bilinear', align_corners=False)

    def calc_visual_score(self, img_tensor, target_class_idx):
        img_tensor = self.preprocess_tensor(img_tensor)
        with torch.no_grad():
            processed = self.resnet_transform(img_tensor)
            logits = self.resnet(processed)
            probs = F.softmax(logits, dim=1)
            if img_tensor.shape[0] > 1:
                if isinstance(target_class_idx, int): scores = probs[:, target_class_idx]
                else: scores = probs[range(len(target_class_idx)), target_class_idx]
                return scores.cpu().tolist()
            score = probs[0, target_class_idx].item()
            return score, 0, 0 # 簡易化

    def calc_similarity(self, gen_tensor, ref_tensor):
        gen_resized = self.preprocess_tensor(gen_tensor)
        ref_resized = self.preprocess_tensor(ref_tensor)
        if ref_resized.shape[0] != gen_resized.shape[0]:
            ref_resized = ref_resized.repeat(gen_resized.shape[0], 1, 1, 1)
        with torch.no_grad():
            ssim = self.ssim_metric(gen_resized, ref_resized).item()
            lpips = self.lpips_metric(gen_resized, ref_resized).item()
        return ssim, lpips

def manage_model_allocation(models, weights, device):
    for i, model in enumerate(models):
        if weights[i] > 0: model.to(device)
        else: model.cpu()
    torch.cuda.empty_cache()

# ==============================================================================
# 3. Batched Generation & Optimization Logic
# ==============================================================================

class TVLoss(nn.Module):
    def forward(self, img):
        b, c, h, w = img.size()
        h_tv = torch.pow((img[:, :, 1:, :] - img[:, :, :h-1, :]), 2).mean()
        w_tv = torch.pow((img[:, :, :, 1:] - img[:, :, :, :w-1]), 2).mean()
        return h_tv + w_tv

class BatchedPyramidGenerator(nn.Module):
    def __init__(self, batch_size=1, target_size=224, start_size=16, activation='sigmoid', initial_images=None):
        super().__init__()
        self.batch_size = batch_size
        self.target_size = target_size
        self.activation = activation
        
        color_correlation_svd_sqrt = torch.tensor([[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]])
        self.register_buffer('color_correlation', color_correlation_svd_sqrt)
        max_norm = torch.max(torch.linalg.norm(color_correlation_svd_sqrt, dim=0))
        self.register_buffer('max_norm', max_norm)
        
        normalized_matrix = color_correlation_svd_sqrt / max_norm
        try: inverse_matrix = torch.linalg.inv(normalized_matrix)
        except RuntimeError: inverse_matrix = torch.linalg.pinv(normalized_matrix)
        self.register_buffer('inverse_color_correlation', inverse_matrix)

        self.levels = nn.ParameterList([
            nn.Parameter(torch.randn(batch_size, 3, start_size, start_size) * 0.1)
        ])

    def extend(self):
        current_res = max([p.shape[-1] for p in self.levels])
        if current_res >= self.target_size: return False
        new_res = min(current_res * 2, self.target_size)
        old_len = len(self.levels)
        new_len = old_len + 1
        with torch.no_grad():
            for p in self.levels: p.mul_(old_len / new_len)
        device = self.levels[0].device
        new_level = nn.Parameter(torch.randn(self.batch_size, 3, new_res, new_res).to(device) * (1.0 / new_len))
        self.levels.append(new_level)
        return True

    def linear_decorrelate_color(self, t):
        t_permute = t.permute(0, 2, 3, 1)
        t_matched = torch.matmul(t_permute, self.color_correlation.T)
        t_matched = t_matched / self.max_norm
        return t_matched.permute(0, 3, 1, 2)

    def inverse_linear_decorrelate_color(self, t):
        t_permute = t.permute(0, 2, 3, 1)
        t_inverted = torch.matmul(t_permute, self.inverse_color_correlation.T)
        return t_inverted.permute(0, 3, 1, 2)

    def forward(self):
        device = self.levels[0].device
        image = torch.zeros(self.batch_size, 3, self.target_size, self.target_size).to(device)
        for level_tensor in self.levels:
            upsampled = F.interpolate(level_tensor, size=(self.target_size, self.target_size), mode='bilinear', align_corners=False, antialias=True)
            image = image + upsampled
        image = self.linear_decorrelate_color(image)
        if self.activation == 'sigmoid': return torch.sigmoid(2 * image)
        return image

def compute_explicit_gradients(features, logits, target_indices):
    probs = F.softmax(logits, dim=1)
    one_hot = torch.zeros_like(logits)
    one_hot.scatter_(1, target_indices.view(-1, 1), 1.0)
    delta = probs - one_hot
    grads = torch.bmm(delta.unsqueeze(2), features.unsqueeze(1))
    return grads

class BatchedMultiModelGM:
    def __init__(self, models, model_weights, target_classes_list, args, device, initial_images=None):
        self.models = models
        self.model_weights = model_weights
        self.target_classes = torch.tensor(target_classes_list, device=device)
        self.batch_size = len(target_classes_list)
        self.args = args
        self.device = device
        
        self.generator = BatchedPyramidGenerator(
            batch_size=self.batch_size,
            target_size=args.image_size,
            start_size=args.pyramid_start_res,
            activation='sigmoid',
            initial_images=initial_images
        ).to(device)
        
        use_fused = (device.type == 'cuda')
        self.optimizer = optim.Adam(self.generator.parameters(), lr=args.lr, fused=use_fused)
        
        self.augmentor = T.Compose([
            T.RandomResizedCrop(args.image_size, scale=(args.min_scale, args.max_scale), antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            T.ColorJitter(0.1, 0.1, 0.1, 0.05),
            RandomGaussianNoise(mean=0.0, std=args.noise_std, p=args.noise_prob), 
        ])
        
        self.scaler = torch.amp.GradScaler('cuda')
        self.tv_loss_fn = TVLoss().to(device)

    def preprocess(self, img):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        return (img - mean) / std

    def optimize_step(self, ref_images_batch):
        self.optimizer.zero_grad()
        syn_images = self.generator()
        B = self.batch_size
        
        refs_selected = []
        for i in range(B):
            pool_size = ref_images_batch.shape[1]
            idx = torch.randint(0, pool_size, (self.args.augs_per_step,))
            refs_selected.append(ref_images_batch[i][idx])
        real_batch = torch.cat(refs_selected, dim=0).to(self.device)
        aug_real = self.augmentor(real_batch)
        inp_real = self.preprocess(aug_real)

        syn_batch = syn_images.repeat_interleave(self.args.augs_per_step, dim=0)
        aug_syn = self.augmentor(syn_batch)
        inp_syn = self.preprocess(aug_syn)
        batch_targets = self.target_classes.repeat_interleave(self.args.augs_per_step)

        total_grad_loss = 0.0
        for i, model in enumerate(self.models):
            weight = self.model_weights[i]
            if weight == 0: continue
            model.reset_classifier()
            
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.amp.autocast('cuda', dtype=dtype):
                # 高速化修正: 重いEncoder計算を1回にして再利用
                
                # --- Real Images ---
                real_feat_raw = model.forward_encoder(inp_real)
                features_real = model.projector(real_feat_raw)
                logits_real = model.classifier(features_real)
                
                # --- Synthetic Images ---
                syn_feat_raw = model.forward_encoder(inp_syn)
                features_syn = model.projector(syn_feat_raw)
                logits_syn = model.classifier(features_syn)
            
            # 勾配計算
            grad_real = compute_explicit_gradients(features_real.detach(), logits_real.detach(), batch_targets)
            grad_syn = compute_explicit_gradients(features_syn, logits_syn, batch_targets)
            
            g_real_flat = grad_real.view(grad_real.shape[0], -1)
            g_syn_flat = grad_syn.view(grad_syn.shape[0], -1)
            sim = F.cosine_similarity(g_real_flat, g_syn_flat, dim=1).mean()
            total_grad_loss += (1.0 - sim) * weight

        loss_tv = self.tv_loss_fn(syn_images) / B
        total_loss = (total_grad_loss * self.args.weight_grad) + (loss_tv * self.args.weight_tv)
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return total_grad_loss.item(), loss_tv.item(), total_loss.item()

    def run(self, ref_images_batch, save_dir, pbar_desc, rank=0):
        best_loss = float('inf')
        local_pbar = tqdm(range(self.args.num_iterations), desc=pbar_desc, leave=False, dynamic_ncols=True, position=rank)
        
        for i in local_pbar:
            if i > 0 and i % self.args.pyramid_grow_interval == 0:
                if self.generator.extend():
                    use_fused = (self.device.type == 'cuda')
                    self.optimizer = optim.Adam(self.generator.parameters(), lr=self.args.lr, fused=use_fused)
            l_grad, l_tv, total = self.optimize_step(ref_images_batch)
            if total < best_loss: best_loss = total
            local_pbar.set_description(f"{pbar_desc} L:{l_grad:.3f}")
            
            # [修正] 指定インターバルで全画像をグリッド保存
            if i % self.args.save_interval == 0:
                 with torch.no_grad():
                    imgs = self.generator().detach().cpu()
                    # バッチ内の全画像をグリッド状にして保存
                    # nrowはグリッドの列数（正方形に近くなるように計算）
                    grid_nrow = int(len(imgs)**0.5)
                    if grid_nrow < 1: grid_nrow = 1
                    save_image(imgs, os.path.join(save_dir, f"prog_step{i:04d}_rank{rank}.png"), nrow=grid_nrow)

        return self.generator().detach().cpu()

# ==============================================================================
# 4. Main Worker
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Batched Multi-Model LGM")
    parser.add_argument("--encoder_names", type=str, nargs='+', required=True)
    parser.add_argument("--projection_dims", type=int, nargs='+', default=[2048])
    parser.add_argument("--experiments", type=str, nargs='+', required=True)
    parser.add_argument("--gpus", type=str, default="0", help="Comma separated GPU IDs")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--target_classes", type=str, nargs='+', default=[])
    parser.add_argument("--batch_size_gen", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_file", type=str, default="log.txt")
    parser.add_argument("--num_ref_images", type=int, default=10)
    parser.add_argument("--augs_per_step", type=int, default=64)
    parser.add_argument("--num_iterations", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--pyramid_start_res", type=int, default=4)
    parser.add_argument("--pyramid_grow_interval", type=int, default=400)
    parser.add_argument("--weight_grad", type=float, default=1.0)
    parser.add_argument("--weight_tv", type=float, default=0.00025)
    parser.add_argument("--min_scale", type=float, default=0.08)
    parser.add_argument("--max_scale", type=float, default=1.0)
    parser.add_argument("--noise_std", type=float, default=0.2)
    parser.add_argument("--noise_prob", type=float, default=0.5)
    parser.add_argument("--overlay_text", type=str, default="")
    parser.add_argument("--text_color", type=str, default="red")
    parser.add_argument("--font_scale", type=float, default=0.15)
    parser.add_argument("--dataset_type", type=str, default="imagenet") 
    parser.add_argument("--data_root", type=str, default="")
    # [追加] 保存インターバル設定 (デフォルト500)
    parser.add_argument("--save_interval", type=int, default=500, help="Interval for saving intermediate images")
    
    # Bash互換用ダミー
    parser.add_argument("--num_generations", type=int, default=1)
    return parser.parse_args()

def main_worker(rank, world_size, gpu_ids, args, target_ids_all):
    gpu_id = gpu_ids[rank]
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)
    set_seed(args.seed + rank)
    
    logger = Logger(args.log_file, rank=gpu_id)
    logger.log(f"Worker {rank} on GPU {gpu_id} started. Targets: {len(target_ids_all)} total.")

    # 分散担当割り当て (Stride)
    my_target_ids = target_ids_all[rank::world_size]
    if not my_target_ids: return

    # データセットロード
    try:
        dataset = load_dataset("imagenet-1k", split="validation", streaming=False)
        class_names = dataset.features['label'].names
    except:
        logger.log("Dataset load failed.")
        return

    # モデル準備
    models_list = []
    p_dims = args.projection_dims
    if len(p_dims) == 1: p_dims = p_dims * len(args.encoder_names)
    
    for name, pd in zip(args.encoder_names, p_dims):
        m = EncoderClassifier(name, freeze_encoder=True, num_classes=1000, projection_dim=pd)
        m.to(device).eval()
        # コンパイル
        if torch.cuda.is_available():
            try: m.encoder = torch.compile(m.encoder)
            except: pass
        models_list.append(m)

    evaluator = Evaluator(device)

    # 実験ループ
    batch_size = args.batch_size_gen
    target_chunks = [my_target_ids[i:i + batch_size] for i in range(0, len(my_target_ids), batch_size)]

    for exp_str in args.experiments:
        exp_name, w_str = exp_str.split(':')
        weights = [float(w) for w in w_str.split(',')]
        
        manage_model_allocation(models_list, weights, device)
        exp_dir = os.path.join(args.output_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        for c_idx, chunk in enumerate(target_chunks):
            # 参照画像準備
            ref_pool_list = []
            ref_clean_list = []
            for cid in chunk:
                imgs = []
                cnt = 0
                for item in dataset:
                    if item['label'] == cid:
                        imgs.append(item['image'].convert('RGB').resize((args.image_size, args.image_size)))
                        cnt += 1
                        if cnt >= args.num_ref_images: break
                if not imgs: imgs = [Image.new('RGB', (args.image_size, args.image_size))]
                
                ref_clean_list.append(to_tensor(imgs[0]).unsqueeze(0))
                
                atk_tensors = []
                for img in imgs:
                    if args.overlay_text:
                        img = add_text_overlay(img, args.overlay_text, args.font_scale, args.text_color)
                    atk_tensors.append(to_tensor(img))
                while len(atk_tensors) < args.num_ref_images: atk_tensors.append(atk_tensors[0])
                ref_pool_list.append(torch.stack(atk_tensors))
            
            ref_pool_batch = torch.stack(ref_pool_list).to(device)
            ref_clean_batch = torch.cat(ref_clean_list).to(device)

            # 最適化実行
            lgm = BatchedMultiModelGM(models_list, weights, chunk, args, device)
            final_imgs = lgm.run(ref_pool_batch, exp_dir, f"[GPU{gpu_id}] {exp_name}", rank=rank)
            final_imgs = final_imgs.to(device)

            # 評価と保存
            for i, cid in enumerate(chunk):
                cname = sanitize_dirname(class_names[cid])
                s_dir = os.path.join(exp_dir, f"{cid}_{cname}")
                os.makedirs(s_dir, exist_ok=True)
                save_image(final_imgs[i], os.path.join(s_dir, "result.png"))
                
                v_score, _, _ = evaluator.calc_visual_score(final_imgs[i].unsqueeze(0), cid)
                ssim, lpips = evaluator.calc_similarity(final_imgs[i].unsqueeze(0), ref_clean_batch[i].unsqueeze(0))
                
                with open(os.path.join(s_dir, "metrics.json"), 'w') as f:
                    json.dump({"visual": v_score, "ssim": ssim, "lpips": lpips}, f, indent=2)

    logger.log("Finished.")
    logger.close()

def main():
    args = parse_args()
    target_ids = []
    if args.target_classes:
        for t in args.target_classes:
            try: target_ids.append(int(t))
            except: pass
    if not target_ids: return

    gpu_ids = [int(x) for x in args.gpus.split(',')]
    world_size = len(gpu_ids)
    
    if world_size > 1:
        mp.spawn(main_worker, nprocs=world_size, args=(world_size, gpu_ids, args, target_ids))
    else:
        main_worker(0, 1, gpu_ids, args, target_ids)

if __name__ == "__main__":
    main()