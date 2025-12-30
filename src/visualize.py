import os
import argparse
import copy
import json
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torchvision.utils import save_image
from torchvision.transforms.functional import resize, to_tensor
import torchvision.transforms as T
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from transformers import CLIPVisionModel, Dinov2Model, ViTModel

# model.py から読み込み
from model import EncoderClassifier

# ==============================================================================
# Helper Classes (TVLoss, PyramidGenerator, SingleClassGM)
# ==============================================================================
# ※ 以前のコードと同じため省略しませんが、長くなるのでクラス定義はそのまま使います
# ------------------------------------------------------------------------------

class TVLoss(nn.Module):
    def forward(self, img):
        b, c, h, w = img.size()
        h_tv = torch.pow((img[:, :, 1:, :] - img[:, :, :h-1, :]), 2).mean()
        w_tv = torch.pow((img[:, :, :, 1:] - img[:, :, :, :w-1]), 2).mean()
        return h_tv + w_tv

class PyramidGenerator(nn.Module):
    def __init__(self, target_size=224, start_size=16, activation='sigmoid', initial_image=None, noise_level=0.0):
        super().__init__()
        self.target_size = target_size
        self.activation = activation
        
        color_correlation_svd_sqrt = torch.tensor([
            [0.26, 0.09, 0.02],
            [0.27, 0.00, -0.05],
            [0.27, -0.09, 0.03]
        ])
        self.register_buffer('color_correlation', color_correlation_svd_sqrt)
        max_norm = torch.max(torch.linalg.norm(color_correlation_svd_sqrt, dim=0))
        self.register_buffer('max_norm', max_norm)
        
        normalized_matrix = color_correlation_svd_sqrt / max_norm
        try:
            inverse_matrix = torch.linalg.inv(normalized_matrix)
        except RuntimeError:
            inverse_matrix = torch.linalg.pinv(normalized_matrix)
        self.register_buffer('inverse_color_correlation', inverse_matrix)

        if initial_image is not None:
            init_low_res = F.interpolate(initial_image, size=(start_size, start_size), mode='bilinear', align_corners=False, antialias=True)
            eps = 1e-6
            if activation == 'sigmoid':
                init_low_res = init_low_res.clamp(eps, 1 - eps)
                init_val = torch.logit(init_low_res) / 2
            else:
                init_val = init_low_res
            init_val = self.inverse_linear_decorrelate_color(init_val)
            if noise_level > 0:
                noise = torch.randn_like(init_val) * 0.1
                init_val = init_val * (1 - noise_level) + noise * noise_level
            self.levels = nn.ParameterList([nn.Parameter(init_val)])
        else:
            self.levels = nn.ParameterList([
                nn.Parameter(torch.randn(1, 3, start_size, start_size) * 0.1)
            ])

    def extend(self):
        current_res = max([p.shape[-1] for p in self.levels])
        if current_res >= self.target_size: return False 
        new_res = min(current_res * 2, self.target_size)
        old_len = len(self.levels)
        new_len = old_len + 1
        with torch.no_grad():
            for p in self.levels: p.mul_(old_len / new_len)
        device = next(self.parameters()).device
        new_level = nn.Parameter(torch.randn(1, 3, new_res, new_res).to(device) * (1.0 / new_len))
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
        device = next(self.parameters()).device
        image = torch.zeros(1, 3, self.target_size, self.target_size).to(device)
        for level_tensor in self.levels:
            upsampled = F.interpolate(level_tensor, size=(self.target_size, self.target_size), mode='bilinear', align_corners=False, antialias=True)
            image = image + upsampled
        image = self.linear_decorrelate_color(image)
        if self.activation == 'sigmoid': return torch.sigmoid(2 * image) 
        return image

class SingleClassGM:
    def __init__(self, model, target_class, args, device, initial_image=None):
        self.model = model
        self.target_class = target_class
        self.args = args
        self.device = device
        self.target_modules = [self.model.classifier]
        
        self.generator = PyramidGenerator(
            target_size=args.image_size,
            start_size=args.pyramid_start_res,
            activation='sigmoid',
            initial_image=initial_image,
            noise_level=args.seed_noise_level
        ).to(device)
        
        self.optimizer = optim.Adam(self.generator.parameters(), lr=self.args.lr)
        
        self.augmentor = T.Compose([
            T.RandomResizedCrop(args.image_size, scale=(0.8, 1.0), antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            T.ColorJitter(0.1, 0.1, 0.1, 0.05),
        ])
        
        self.scaler = torch.cuda.amp.GradScaler()
        self.tv_loss_fn = TVLoss().to(device)

    def preprocess(self, img):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        return (img - mean) / std

    def get_grads(self, inputs, create_graph=False):
        params = []
        for module in self.target_modules:
            for p in module.parameters():
                p.requires_grad = True
                params.append(p)
        logits = self.model(inputs)
        targets = torch.tensor([self.target_class] * inputs.size(0), device=self.device)
        loss = F.cross_entropy(logits, targets)
        grads = torch.autograd.grad(loss, params, create_graph=create_graph)
        flat_grad = torch.cat([g.view(-1) for g in grads])
        return flat_grad, loss

    def optimize_step(self, real_images_subset):
        self.optimizer.zero_grad()
        self.model.reset_classifier()
        
        # Real
        real_subset = real_images_subset.detach()
        with autocast():
            aug_real = self.augmentor(real_subset)
            inp_real = self.preprocess(aug_real)
            target_grad, _ = self.get_grads(inp_real, create_graph=False)
        
        # Syn
        syn_image = self.generator()
        with autocast():
            syn_batch = []
            for _ in range(self.args.augs_per_step):
                syn_batch.append(self.augmentor(syn_image))
            syn_batch = torch.cat(syn_batch, dim=0)
            inp_syn = self.preprocess(syn_batch)
            syn_grad, loss_class = self.get_grads(inp_syn, create_graph=True)
        
        loss_grad = 1.0 - F.cosine_similarity(target_grad.unsqueeze(0).detach(), syn_grad.unsqueeze(0)).mean()
        loss_tv = self.tv_loss_fn(syn_image)
        
        total_loss = (loss_grad * self.args.weight_grad) + (loss_class * self.args.weight_class) + (loss_tv * self.args.weight_tv)
        
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss_grad.item(), loss_class.item(), loss_tv.item()

    def run(self, real_images_pool, save_dir, class_names):
        pbar = tqdm(range(self.args.num_iterations), desc=f"Cls {self.target_class}", leave=False)
        batch_size = min(len(real_images_pool), self.args.num_ref_images)
        loss_history = []
        
        for i in pbar:
            if i > 0 and i % self.args.pyramid_grow_interval == 0:
                if self.generator.extend():
                    self.optimizer = optim.Adam(self.generator.parameters(), lr=self.args.lr)
            
            indices = torch.randperm(len(real_images_pool))[:batch_size]
            l_grad, l_class, l_tv = self.optimize_step(real_images_pool[indices])
            loss_history.append({"l_grad": l_grad, "l_tv": l_tv})
            pbar.set_description(f"G:{l_grad:.3f} TV:{l_tv:.4f} Sz:{self.generator.levels[-1].shape[-1]}")
            
            if i % 500 == 0:
                with torch.no_grad():
                    save_image(self.generator().detach().cpu(), os.path.join(save_dir, f"step_{i:04d}.png"))
        
        final = self.generator().detach().cpu()
        return final, {"history": loss_history}

def sanitize_dirname(name):
    return re.sub(r'[^a-zA-Z0-9]', '_', name).strip('_')

# ==============================================================================
# Main (Loop Optimized)
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    # 複数受け取れるように変更
    parser.add_argument("--projection_dims", type=int, nargs='+', default=[2048, 1024, 0], help="List of dim")
    
    parser.add_argument("--encoder_name", type=str, default="facebook/dino-vitb16")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--target_classes", type=int, nargs='+', default=[9])
    parser.add_argument("--num_iterations", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_ref_images", type=int, default=10)
    parser.add_argument("--augs_per_step", type=int, default=32)
    parser.add_argument("--pyramid_start_res", type=int, default=16)
    parser.add_argument("--pyramid_grow_interval", type=int, default=400)
    parser.add_argument("--weight_grad", type=float, default=1.0)
    parser.add_argument("--weight_class", type=float, default=0.0)
    parser.add_argument("--weight_tv", type=float, default=0.00025)
    parser.add_argument("--seed_noise_level", type=float, default=0.0)
    parser.add_argument("--initial_image_path", type=str, default=None)
    parser.add_argument("--force_random_classifier", action="store_true") # 互換性のため残す
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    # 1. データセットのロード (1回だけ)
    print("Loading Dataset...")
    try:
        dataset = load_dataset("imagenet-1k", split="validation", streaming=False)
        class_names = dataset.features['label'].names
    except:
        print("Dataset load failed. Using dummy.")
        dataset = []
        class_names = [f"class_{i}" for i in range(1000)]

    # 2. クラスごとの画像プール作成 (1回だけ)
    real_images_dict = {}
    for t_cls in args.target_classes:
        pool = []
        count = 0
        for item in dataset:
            if item['label'] == t_cls:
                img = item['image'].convert('RGB')
                img = resize(img, [args.image_size, args.image_size])
                pool.append(to_tensor(img))
                count += 1
                if count >= 200: break
        if pool:
            real_images_dict[t_cls] = torch.stack(pool).to(device)
    
    # 3. エンコーダーのロード (1回だけ！)
    print(f"Loading Base Encoder: {args.encoder_name} ...")
    if "openai/clip" in args.encoder_name:
        base_encoder = CLIPVisionModel.from_pretrained(args.encoder_name)
    elif "facebook/dinov2" in args.encoder_name:
        base_encoder = Dinov2Model.from_pretrained(args.encoder_name)
    elif "facebook/dino" in args.encoder_name:
        base_encoder = ViTModel.from_pretrained(args.encoder_name)
    else:
        # Fallback
        base_encoder = ViTModel.from_pretrained(args.encoder_name)
    
    base_encoder.to(device)
    # 勾配計算しないので固定
    for p in base_encoder.parameters():
        p.requires_grad = False

    # 4. ループ実行 (射影次元 x クラス)
    for p_dim in args.projection_dims:
        print(f"\n>>> Running with Projection Dim: {p_dim} <<<")
        
        # ここで base_encoder を渡して使い回す (ロード時間ほぼゼロ)
        model = EncoderClassifier(
            encoder_model=args.encoder_name,
            encoder=base_encoder, # ★ここがポイント
            num_classes=1000,
            projection_dim=p_dim
        )
        model.to(device)
        model.eval()
        
        for t_cls in args.target_classes:
            if t_cls not in real_images_dict: continue
            
            c_name = sanitize_dirname(class_names[t_cls])
            save_dir = os.path.join(args.output_dir, f"Proj{p_dim}", f"{t_cls}_{c_name}")
            os.makedirs(save_dir, exist_ok=True)
            
            # 実験開始
            lgm = SingleClassGM(model, t_cls, args, device)
            final_img, metrics = lgm.run(real_images_dict[t_cls], save_dir, class_names)
            
            save_image(final_img, os.path.join(save_dir, "final.png"))
            with open(os.path.join(save_dir, "metrics.json"), 'w') as f:
                json.dump({"args": vars(args), "metrics": metrics}, f, indent=2)

    print("All Done.")