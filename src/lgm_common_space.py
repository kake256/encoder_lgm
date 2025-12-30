import os
import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torchvision.utils import save_image
from torchvision import transforms
from tqdm import tqdm
from datasets import load_dataset

# 共通モデル定義のインポート
try:
    from src.common_models import ExpandedProjector, FixedProjector, DummyBackbone, load_backbone
except ImportError:
    from common_models import ExpandedProjector, FixedProjector, DummyBackbone, load_backbone

# ==============================================================================
# データロード・前処理 (提示されたコードを参考)
# ==============================================================================
def normalize(x):
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std

def load_real_images_from_hf(dataset_name, target_class_idx, num_images=32, image_size=224, device='cuda'):
    """
    提示されたコードを参考に、Hugging Face Datasetから特定クラスの画像を収集する関数
    """
    print(f"Loading dataset '{dataset_name}' to find class {target_class_idx}...")
    
    # 前処理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])

    try:
        # ストリーミングモードでロード (ダウンロード待ち時間を回避)
        # split="train" を使用しますが、クラスが見つかりやすいようにシャッフルします
        dataset = load_dataset(dataset_name, split="train", streaming=True, trust_remote_code=True)
        dataset = dataset.shuffle(seed=42, buffer_size=10000)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return torch.randn(num_images, 3, image_size, image_size).to(device)

    images = []
    count = 0
    max_scan = 10000 # 無限ループ防止用のスキャン上限
    scanned = 0

    print(f"Scanning dataset for class {target_class_idx}...")
    
    # 提示コードと同様のループ処理
    for item in dataset:
        scanned += 1
        # ラベルカラム名の揺らぎに対応 ('label' or 'labels')
        label = item.get('label', item.get('labels'))
        
        if label == target_class_idx:
            try:
                img = item['image'].convert('RGB')
                img_t = transform(img)
                images.append(img_t)
                count += 1
                if count >= num_images:
                    break
            except Exception as e:
                print(f"Error processing image: {e}")
                continue
        
        if scanned >= max_scan:
            print(f"Timeout: Scanned {max_scan} items but only found {len(images)} images.")
            break

    if len(images) == 0:
        print(f"!!! WARNING: No images found for class {target_class_idx}. Falling back to RANDOM NOISE. !!!")
        return torch.randn(num_images, 3, image_size, image_size).to(device)

    # 画像が足りない場合は複製して埋める
    if len(images) < num_images:
        print(f"Warning: Only found {len(images)} images. Duplicating to fill batch.")
        while len(images) < num_images:
            images.append(images[len(images) % len(images)])

    print(f"Successfully loaded {len(images)} real images.")
    return torch.stack(images).to(device)

# ==============================================================================
# Augmentation & Generator
# ==============================================================================
class DifferentiableAugmentor(nn.Module):
    def __init__(self, size):
        super().__init__()
    def forward(self, x):
        if torch.rand(1) < 0.5: x = torch.flip(x, dims=[3])
        return x

class PyramidGenerator(nn.Module):
    def __init__(self, target_size=224, start_size=16):
        super().__init__()
        self.target_size = target_size
        self.levels = nn.ParameterList([nn.Parameter(torch.randn(1, 3, start_size, start_size) * 0.05)])
        self.register_buffer('color_corr', torch.tensor([[0.26, 0.09, 0.02],[0.27, 0.00, -0.05],[0.27, -0.09, 0.03]]))
    def extend(self):
        curr = self.levels[-1].shape[-1]
        if curr >= self.target_size: return False
        new_res = min(curr * 2, self.target_size)
        old_len = len(self.levels)
        with torch.no_grad():
            for p in self.levels: p.mul_(old_len / (old_len + 1))
        self.levels.append(nn.Parameter(torch.randn(1, 3, new_res, new_res).to(self.levels[0].device) * (0.05/(old_len+1))))
        return True
    def forward(self):
        img = torch.zeros(1, 3, self.target_size, self.target_size).to(self.levels[0].device)
        for p in self.levels:
            img = img + F.interpolate(p, size=(self.target_size, self.target_size), mode='bilinear', align_corners=False)
        t = img.permute(0, 2, 3, 1)
        max_norm = torch.max(torch.linalg.norm(self.color_corr, dim=0))
        t = torch.matmul(t, self.color_corr.T) / max_norm
        return torch.sigmoid(2.0 * t.permute(0, 3, 1, 2))

# ==============================================================================
# Common Space LGM (ハイブリッド損失版)
# ==============================================================================
class CommonSpaceLGM:
    def __init__(self, models, target_class, args, device):
        self.models = models
        self.target_class = target_class
        self.args = args
        self.device = device
        self.gen = PyramidGenerator(args.image_size, args.pyramid_start_res).to(device)
        self.opt = optim.Adam(self.gen.parameters(), lr=args.lr)
        self.aug = DifferentiableAugmentor(args.image_size).to(device)
        self.scaler = torch.cuda.amp.GradScaler()

    def run(self, real_imgs, weights):
        # 1. 実画像の特徴量平均を計算 (Feature Matching用ターゲット)
        real_norm = normalize(real_imgs)
        target_features = {}
        grad_real_accum = 0
        
        # 共通空間での特徴量と、初期の勾配ターゲットを計算
        with torch.no_grad():
            # 勾配計算用の初期分類器 (Feature Matchingが効くまでのガイド)
            classifier_fixed = nn.Linear(self.args.common_dim, 1000).to(self.device)
            nn.init.normal_(classifier_fixed.weight, std=0.01)

            for name, w in weights.items():
                if w == 0: continue
                backbone, proj = self.models[name]
                feat = backbone(real_norm)
                if isinstance(feat, tuple): feat = feat[0]
                if feat.dim() == 4: feat = feat.mean([2,3])
                feat = feat.view(feat.size(0), -1)
                z, _ = proj(feat)
                
                # 特徴量平均を保存
                target_features[name] = z.mean(dim=0, keepdim=True)
                
                # 固定分類器での勾配計算 (create_graph=False)
                with torch.enable_grad():
                    logits = classifier_fixed(z)
                    loss = F.cross_entropy(logits, torch.tensor([self.target_class]*len(z), device=self.device))
                    grads = torch.autograd.grad(loss, classifier_fixed.parameters())
                    grad_real_accum += w * torch.cat([g.view(-1) for g in grads])

        # 最適化ループ
        pbar = tqdm(range(self.args.num_iterations), leave=False)
        for i in pbar:
            if i > 0 and i % self.args.pyramid_grow_interval == 0:
                if self.gen.extend(): 
                    self.opt = optim.Adam(self.gen.parameters(), lr=self.args.lr)
            
            self.opt.zero_grad()
            
            # --- Source Gradient & Feature Matching の計算 ---
            syn = self.gen()
            syn_batch = torch.cat([self.aug(syn) for _ in range(self.args.augs_per_step)])
            syn_norm = normalize(syn_batch)
            
            grad_syn_accum = 0
            loss_feature_match = 0
            
            for name, w in weights.items():
                if w == 0: continue
                backbone, proj = self.models[name]
                
                feat = backbone(syn_norm)
                if isinstance(feat, tuple): feat = feat[0]
                if feat.dim() == 4: feat = feat.mean([2,3])
                feat = feat.view(feat.size(0), -1)
                z, _ = proj(feat)
                
                # A. Feature Matching Loss (特徴量を実画像の平均に近づける)
                if name in target_features:
                    loss_feature_match += F.mse_loss(z, target_features[name].expand_as(z))

                # B. Gradient Calculation (固定分類器を使用)
                # ※ 本来のLGMはランダム分類器ですが、黒画像対策として一時的に固定分類器の勾配を使用
                with autocast():
                    logits = classifier_fixed(z)
                    loss = F.cross_entropy(logits, torch.tensor([self.target_class]*len(z), device=self.device))
                
                grads = torch.autograd.grad(loss, classifier_fixed.parameters(), create_graph=True)
                grad_syn_accum += w * torch.cat([g.view(-1) for g in grads])
            
            # C. Gradient Matching Loss
            loss_gm = 1.0 - F.cosine_similarity(grad_real_accum.unsqueeze(0), grad_syn_accum.unsqueeze(0)).mean()
            
            # ★ ハイブリッド損失: LGM + 特徴量マッチング
            # 特徴量ロス(Feature Matching)の重みを大きくして、まず「似た特徴」を出させる
            total_loss = loss_gm + (100.0 * loss_feature_match)

            if torch.isnan(total_loss):
                print("Loss NaN detected! Reducing LR or Skipping.")
                break

            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.gen.parameters(), max_norm=1.0)
            self.scaler.step(self.opt)
            self.scaler.update()
            
            if i % 100 == 0: 
                pbar.set_description(f"GM: {loss_gm.item():.4f} | Feat: {loss_feature_match.item():.4f}")

        return self.gen().detach()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="imagenet-1k")
    # fraction削除
    parser.add_argument("--target_classes", type=int, nargs='+', default=[9])
    parser.add_argument("--common_dim", type=int, default=2048)
    parser.add_argument("--num_iterations", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--pyramid_start_res", type=int, default=16)
    parser.add_argument("--pyramid_grow_interval", type=int, default=400)
    parser.add_argument("--augs_per_step", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 実モデルロード
    bb_dino = load_backbone("dino_vitb16", device)
    bb_clip = load_backbone("clip", device)
    
    proj_dino = ExpandedProjector(768, args.common_dim).to(device).eval()
    proj_clip = FixedProjector(512, args.common_dim).to(device).eval()
    
    if os.path.exists(args.weights_path):
        ckpt = torch.load(args.weights_path, map_location=device)
        proj_dino.load_state_dict(ckpt['dino'])
        proj_clip.load_state_dict(ckpt['clip'])
    
    models = {'dino': (bb_dino, proj_dino), 'clip': (bb_clip, proj_clip)}
    
    for cls in args.target_classes:
        print(f"\n>> Processing Class {cls}")
        save_base = os.path.join(args.output_dir, str(cls))
        os.makedirs(save_base, exist_ok=True)
        
        real_imgs = load_real_images_from_hf(
            args.dataset_name, cls, num_images=32, device=device
        )
        save_image(real_imgs, f"{save_base}/ref_real.png")
        
        lgm = CommonSpaceLGM(models, cls, args, device)

        print("   [Mode: Single] Generating...")
        img = lgm.run(real_imgs, {'dino': 1.0, 'clip': 0.0})
        save_image(img, f"{save_base}/single.png")

        print("   [Mode: Intersection] Generating...")
        img = lgm.run(real_imgs, {'dino': 1.0, 'clip': 1.0})
        save_image(img, f"{save_base}/intersection.png")

        print("   [Mode: Difference] Generating...")
        img = lgm.run(real_imgs, {'dino': 1.0, 'clip': -0.5})
        save_image(img, f"{save_base}/difference.png")