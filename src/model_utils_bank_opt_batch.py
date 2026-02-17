# =========================
# src/model_utils_bank_opt_batch.py (VRAM安全・超高速版)
# =========================
import os
import re
import hashlib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.amp import autocast as amp_autocast
from torchvision.transforms import v2 as T
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image
from transformers import CLIPVisionModel, Dinov2Model, ViTModel, SiglipVisionModel
from tqdm import tqdm
from PIL import ImageDraw, ImageFont, Image

try:
    import timm
except ImportError:
    timm = None

def manage_model_allocation(models, weights, device):
    for i, model in enumerate(models):
        if model is None: continue
        if weights[i] > 0:
            model.to(device)
        else:
            model.cpu()
    # モデルの移動後に確実にVRAMを空ける
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _add_text_overlay_internal(pil_image, text, font_scale=0.15, color="red"):
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

class RandomGaussianNoise(nn.Module):
    def __init__(self, mean=0.0, std=1.0, p=0.5):
        super().__init__()
        self.mean, self.std, self.p = mean, std, p
    def forward(self, images):
        if self.p == 0 or torch.rand(1).item() >= self.p: return images
        return images + torch.randn_like(images) * self.std + self.mean

class TVLoss(nn.Module):
    def forward(self, img):
        _b, _c, h, w = img.size()
        h_tv = torch.pow((img[:, :, 1:, :] - img[:, :, : h - 1, :]), 2).mean()
        w_tv = torch.pow((img[:, :, :, 1:] - img[:, :, :, : w - 1]), 2).mean()
        return h_tv + w_tv

class EncoderClassifier(nn.Module):
    def __init__(self, encoder_model="openai/clip-vit-base-patch16", encoder=None, freeze_encoder=True, num_classes=1000, feature_source="pooler", projection_dim=2048):
        super().__init__()
        self.encoder_model_name = encoder_model
        self.use_timm_encoder = False
        if encoder is not None:
            self.encoder = encoder
            if not hasattr(encoder, "config"): self.use_timm_encoder = True
        else:
            if "siglip" in self.encoder_model_name: self.encoder = SiglipVisionModel.from_pretrained(encoder_model)
            elif self.encoder_model_name.startswith("openai/clip") or "laion" in self.encoder_model_name: self.encoder = CLIPVisionModel.from_pretrained(encoder_model)
            elif "dinov2" in self.encoder_model_name and "timm" not in self.encoder_model_name: self.encoder = Dinov2Model.from_pretrained(encoder_model)
            elif "dinov3" in self.encoder_model_name or self.encoder_model_name.startswith("timm/"):
                self.encoder = timm.create_model(encoder_model.replace("timm/", ""), pretrained=True, num_classes=0); self.use_timm_encoder = True
            else: self.encoder = ViTModel.from_pretrained(encoder_model)

        if self.use_timm_encoder: self.embed_dim = self.encoder.num_features; self.feature_source = feature_source
        else: self.embed_dim = self.encoder.config.hidden_size; self.feature_source = "cls"
        if freeze_encoder:
            for p in self.encoder.parameters(): p.requires_grad = False
        self.projector = nn.Sequential(nn.Linear(self.embed_dim, projection_dim), nn.ReLU(inplace=True)) if projection_dim > 0 else nn.Identity()
        self.classifier = nn.Linear(projection_dim if projection_dim > 0 else self.embed_dim, num_classes)

    def forward_features(self, x):
        if self.use_timm_encoder: return self.encoder(x)
        outputs = self.encoder(pixel_values=x, output_hidden_states=False)
        return outputs.pooler_output if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None else outputs.last_hidden_state[:, 0, :]

    def forward(self, x): return self.classifier(self.projector(self.forward_features(x)))

class PyramidGenerator(nn.Module):
    def __init__(self, batch_size=1, target_size=224, start_size=16, activation="sigmoid"):
        super().__init__()
        self.target_size, self.activation, self.batch_size = target_size, activation, batch_size
        color_correlation_svd_sqrt = torch.tensor([[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]])
        self.register_buffer("color_correlation", color_correlation_svd_sqrt)
        self.register_buffer("max_norm", torch.max(torch.linalg.norm(color_correlation_svd_sqrt, dim=0)))
        init_tensor = torch.randn(self.batch_size, 3, start_size, start_size) * 0.1
        self.levels = nn.ParameterList([nn.Parameter(init_tensor)])
    def extend(self):
        current_res = max([p.shape[-1] for p in self.levels])
        if current_res >= self.target_size: return False
        new_res = min(current_res * 2, self.target_size); old_len = len(self.levels); new_len = old_len + 1
        with torch.no_grad():
            for p in self.levels: p.mul_(old_len / new_len)
        self.levels.append(nn.Parameter(torch.randn(self.batch_size, 3, new_res, new_res, device=next(self.parameters()).device) * (1.0 / new_len)))
        return True
    def forward(self):
        image = torch.zeros(self.batch_size, 3, self.target_size, self.target_size, device=next(self.parameters()).device)
        for level in self.levels:
            image += F.interpolate(level, size=(self.target_size, self.target_size), mode="bilinear", align_corners=False, antialias=True)
        t_permute = image.permute(0, 2, 3, 1)
        image = (torch.matmul(t_permute, self.color_correlation.T) / self.max_norm).permute(0, 3, 1, 2)
        return torch.sigmoid(2 * image) if self.activation == "sigmoid" else image

class MultiModelGM:
    def __init__(self, models, model_weights, target_classes, args, device):
        self.models, self.model_weights, self.target_classes, self.batch_size = models, model_weights, target_classes, len(target_classes)
        self.args, self.device, self.syn_aug_num = args, device, int(args.syn_aug)
        self.generator = PyramidGenerator(batch_size=self.batch_size, target_size=args.image_size, start_size=args.pyramid_start_res).to(device)
        self.optimizer = optim.Adam(self.generator.parameters(), lr=args.lr)
        aug_list = [T.RandomResizedCrop(args.image_size, scale=(args.min_scale, args.max_scale), antialias=True), T.RandomHorizontalFlip(p=0.5), RandomGaussianNoise(mean=0.0, std=args.noise_std, p=args.noise_prob)]
        self.syn_augmentor, self.real_augmentor = T.Compose(aug_list), T.Compose(aug_list)
        self.tv_loss_fn = TVLoss().to(device)
        self.norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    def preprocess(self, img): return (img - self.norm_mean) / self.norm_std

    # Analytical Gradient (数学的手計算による勾配)
    # これにより create_graph=True が不要になり、メモリ爆発を防ぐ
    def get_fc_grads(self, features, labels, fc):
        logits = fc(features) 
        probs = F.softmax(logits, dim=1)
        labels_onehot = F.one_hot(labels, num_classes=logits.size(1)).to(logits.dtype)
        
        d_logits = (probs - labels_onehot) / features.size(0)
        grad_weight = torch.matmul(d_logits.t(), features)
        grad_bias = d_logits.sum(dim=0)
        
        return torch.cat([grad_weight.flatten(), grad_bias.flatten()])

    def optimize_step(self, real_images_tensor_list):
        AMP_SCALE = 1024.0
        self.optimizer.zero_grad()
        
        total_grad_loss_val = 0.0
        per_model_sims = {self.args.encoder_names[i]: 0.0 for i in range(len(self.models)) if self.models[i] is not None}

        # ★ ここが最大のポイント ★
        # 120GBのVRAMでも、4つのモデルを同時ロードして100クラスを一括処理するには、
        # syn_aug (拡張枚数) をチャンクで刻む必要があります。
        # 2枚ずつ (100クラス×2枚 = 200枚バッチ) でループを回します。
        inner_batch_size = 2 
        num_chunks = (self.syn_aug_num + inner_batch_size - 1) // inner_batch_size

        for chunk_idx in range(num_chunks):
            current_chunk_size = min(inner_batch_size, self.syn_aug_num - chunk_idx * inner_batch_size)

            sampled_real = []
            for class_tensor in real_images_tensor_list:
                idx = torch.randint(0, class_tensor.size(0), (current_chunk_size,))
                sampled_real.append(class_tensor[idx])
            
            sampled_real_tensor = torch.stack(sampled_real).transpose(0, 1).reshape(-1, 3, self.args.image_size, self.args.image_size)
            inp_real = self.preprocess(self.real_augmentor(sampled_real_tensor.to(self.device, non_blocking=True)))
            chunk_labels = torch.arange(self.batch_size, device=self.device).repeat(current_chunk_size)

            for i, model in enumerate(self.models):
                if model is None or self.model_weights[i] == 0: continue

                fc = nn.Linear(model.embed_dim, self.batch_size).to(self.device)
                fc.weight.data.normal_(0, 0.01); fc.bias.data.zero_()

                # フレッシュなグラフをチャンクごとに生成 (retain_graph不要化)
                syn_images_base = self.generator()
                syn_batch_chunk = syn_images_base.repeat(current_chunk_size, 1, 1, 1)
                inp_syn = self.preprocess(self.syn_augmentor(syn_batch_chunk))

                with amp_autocast('cuda', dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16):
                    with torch.no_grad():
                        real_feats = model.forward_features(inp_real).detach()
                    
                    # 数式ベースの勾配計算 (create_graph不要)
                    grad_real = self.get_fc_grads(real_feats, chunk_labels, fc).detach()

                    # 特徴量抽出 (1次微分のみ)
                    syn_feats = model.forward_features(inp_syn)
                    grad_syn = self.get_fc_grads(syn_feats, chunk_labels, fc)

                    sim = F.cosine_similarity(grad_real.unsqueeze(0), grad_syn.unsqueeze(0)).mean()
                    loss_matching = (1.0 - sim) * (self.model_weights[i] / num_chunks)

                    # retain_graph=False で完全にメモリ解放
                    (loss_matching * AMP_SCALE).backward()

                total_grad_loss_val += loss_matching.item()
                per_model_sims[self.args.encoder_names[i]] += sim.item() / num_chunks
                
                # ゴミ掃除
                del fc, grad_real, grad_syn, syn_feats, real_feats, syn_images_base, inp_syn

        loss_tv = self.tv_loss_fn(self.generator())
        (loss_tv * self.args.weight_tv * AMP_SCALE).backward()

        for param in self.generator.parameters():
            if param.grad is not None: param.grad /= AMP_SCALE
        self.optimizer.step()

        return total_grad_loss_val, loss_tv.item(), per_model_sims, total_grad_loss_val + (loss_tv.item() * self.args.weight_tv)

    def run(self, real_images_tensor_list, global_pbar, gen_idx=0, save_dirs=None):
        loss_history = []
        best_loss, best_img_tensor = float("inf"), None
        local_pbar = tqdm(range(int(self.args.num_iterations)), desc=f"G{gen_idx}", leave=False)
        for i in local_pbar:
            if i > 0 and i % int(self.args.pyramid_grow_interval) == 0 and self.generator.extend():
                self.optimizer = optim.Adam(self.generator.parameters(), lr=self.args.lr)
            
            l_grad, l_tv, model_sims, current_total_loss = self.optimize_step(real_images_tensor_list)
            if current_total_loss < best_loss:
                best_loss, best_img_tensor = current_total_loss, self.generator().detach().cpu()
            
            if i > 0 and i % 500 == 0 and save_dirs is not None:
                current_imgs = self.generator().detach().cpu()
                for b_idx, s_dir in enumerate(save_dirs):
                    save_image(current_imgs[b_idx].unsqueeze(0), os.path.join(s_dir, f"iter_{i:04d}.png"))

            step_metrics = {"loss_grad": l_grad, "loss_tv": l_tv, "total_loss": current_total_loss}
            step_metrics.update({f"sim_{k}": v for k, v in model_sims.items()})
            loss_history.append(step_metrics)
            local_pbar.set_description(f"G{gen_idx} L:{l_grad:.3f}")
            global_pbar.update(1)
            
        return self.generator().detach().cpu(), best_img_tensor, {"loss_history": loss_history}