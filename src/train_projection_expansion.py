import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from datasets import load_dataset

# ★修正: src.models ではなく src.common_models からロード
from src.common_models import ExpandedProjector, FixedProjector, load_backbone

def collate_fn(batch, transform):
    images = []
    for item in batch:
        img = item['image'].convert("RGB")
        if transform:
            img = transform(img)
        images.append(img)
    return torch.stack(images)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"--- Training Projection Layer ---")
    print(f"Dataset: {args.dataset_name}")

    # 1. バックボーン準備 (実モデルをロード)
    backbone_dino = load_backbone("dino_vitb16", device)
    backbone_clip = load_backbone("clip", device)
    
    # パラメータ固定
    for p in backbone_dino.parameters(): p.requires_grad = False
    for p in backbone_clip.parameters(): p.requires_grad = False

    # 2. 射影モデル準備
    proj_dino = ExpandedProjector(768, args.common_dim).to(device)
    proj_clip = FixedProjector(512, args.common_dim).to(device)

    # 3. データセット準備
    print(f"Loading dataset '{args.dataset_name}' (Streaming + Shuffle)...")
    try:
        dataset = load_dataset(
            args.dataset_name, 
            split="train", 
            streaming=True, 
            trust_remote_code=True
        )
        dataset = dataset.shuffle(seed=42, buffer_size=10000)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        collate_fn=lambda x: collate_fn(x, transform),
        num_workers=4
    )

    optimizer = optim.Adam(list(proj_dino.parameters()) + list(proj_clip.parameters()), lr=args.lr)
    
    proj_dino.train()
    proj_clip.train()

    print("Start training...")
    steps_per_epoch = 500 # 高速化のためステップ数を調整
    data_iter = iter(dataloader)

    for epoch in range(args.epochs):
        total_loss = 0
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for _ in pbar:
            try:
                images = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                images = next(data_iter)
                
            images = images.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                feat_dino = backbone_dino(images)
                # DINOv1/v2の出力形式対応
                if isinstance(feat_dino, tuple): feat_dino = feat_dino[0]
                if feat_dino.dim() == 4: feat_dino = feat_dino.mean([2,3]) # GAP
                
                feat_clip = backbone_clip(images)
                if isinstance(feat_clip, tuple): feat_clip = feat_clip[0]
                if feat_clip.dim() == 4: feat_clip = feat_clip.mean([2,3])

            z_dino, rec_dino = proj_dino(feat_dino)
            z_clip, _ = proj_clip(feat_clip)

            z_dino_n = F.normalize(z_dino, dim=1)
            z_clip_n = F.normalize(z_clip, dim=1)
            logits = torch.matmul(z_dino_n, z_clip_n.T) / 0.07
            labels = torch.arange(logits.size(0), device=device)
            loss_align = F.cross_entropy(logits, labels)

            loss_recon = F.mse_loss(rec_dino, feat_dino)

            loss = loss_align + (args.recon_weight * loss_recon)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'L_align': loss_align.item(), 'L_recon': loss_recon.item()})

    save_path = os.path.join(args.output_dir, "projection_weights.pth")
    torch.save({
        'dino': proj_dino.state_dict(),
        'clip': proj_clip.state_dict()
    }, save_path)
    print(f"Saved weights to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="imagenet-1k")
    parser.add_argument("--dataset_fraction", type=float, default=1.0) 
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--common_dim", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--recon_weight", type=float, default=50.0)
    args = parser.parse_args()
    train(args)