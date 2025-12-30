import torch
import torch.nn as nn
import os

# --- 射影モデル ---
class ExpandedProjector(nn.Module):
    def __init__(self, input_dim, common_dim=2048):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, common_dim),
            nn.LayerNorm(common_dim),
            nn.GELU()
        )
        self.decoder = nn.Linear(common_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon

class FixedProjector(nn.Module):
    def __init__(self, input_dim, common_dim=2048):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, common_dim, bias=False),
            nn.LayerNorm(common_dim)
        )

    def forward(self, x):
        return self.projection(x), None

# --- ダミーバックボーン (フォールバック用) ---
class DummyBackbone(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.conv = nn.Conv2d(3, out_dim, 16, 16)
    def forward(self, x): return self.conv(x).mean([2,3])

# --- ★実モデルローダー (ここを追加) ---
def load_backbone(name, device):
    print(f"Loading backbone model: {name}...")
    
    # 1. DINO (v1/v2)
    if "dino" in name.lower():
        try:
            # DINOv2 (ViT-B/14)
            if "v2" in name.lower():
                model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
                # DINOv2は出力が(Batch, 768)のTensor
                # バックボーンとして使うため、forwardのみ調整が必要な場合があるが、
                # torch.hubのdinov2はそのまま特徴量を返す
            else:
                # DINOv1 (ViT-B/16)
                model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
            
            model.to(device)
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading DINO from torch.hub: {e}")
            print("Using Dummy DINO (768 dim).")
            return DummyBackbone(768).to(device)

    # 2. CLIP
    elif "clip" in name.lower():
        try:
            import clip
            # ViT-B/16 (Output: 512)
            model, _ = clip.load("ViT-B/16", device=device)
            return model.visual.eval() # 画像エンコーダのみ返す
        except ImportError:
            print("Error: 'clip' library not installed. (pip install git+https://github.com/openai/CLIP.git)")
            print("Using Dummy CLIP (512 dim).")
            return DummyBackbone(512).to(device)
        except Exception as e:
            print(f"Error loading CLIP: {e}")
            return DummyBackbone(512).to(device)

    else:
        print(f"Unknown model name: {name}. Using Dummy.")
        return DummyBackbone(768).to(device)