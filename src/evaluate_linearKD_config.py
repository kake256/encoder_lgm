# ファイル名: evaluate_linearKD_config.py
# 内容: 実験設定, KD用パラメータ, 全モデルリストの一元管理定義ファイル

import os
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
IMG_SIZE = 224

BATCH_SIZE = 1024
NUM_WORKERS = min(32, os.cpu_count() if os.cpu_count() else 4)

# ========================================================
# 2. Knowledge Distillation (KD) Settings
# ========================================================
KD_ALPHA = 0.5 
KD_TEMPERATURE = 2.0 

# ========================================================
# 3. Model & Evaluator Lists (完全版)
# ========================================================
AVAILABLE_EVAL_MODELS = {
    # --- Giant Models (Teachers / for Linear Probe & Full FT) ---
    "ResNet50": "microsoft/resnet-50",
    "DINOv1":   "facebook/dino-vitb16",
    "DINOv2":   "facebook/dinov2-base",
    "CLIP":     "openai/clip-vit-base-patch16",
    "SigLIP":   "google/siglip-base-patch16-224",
    
    # ↓↓↓ 今回スキップされていたモデルを追加 ↓↓↓
    "MAE":      "facebook/vit-mae-base",
    "SwAV":     "swav_resnet50",
    "OpenCLIP_ViT_B32": "openclip:ViT-B-32:laion2b_s34b_b79k",
    "OpenCLIP_RN50":    "openclip:RN50:openai",
    "OpenCLIP_ConvNeXt": "openclip:convnext_base_w:laion2b_s13b_b82k",

    # --- Lightweight Models (Targets / for Scratch Training) ---
    "MobileNetV2_050":   "timm:mobilenetv2_050",
    "MobileNetV3_S":     "timm:mobilenetv3_small_100",
    "GhostNet_100":      "timm:ghostnet_100",
    "EfficientNet_B0":   "timm:efficientnet_b0",
    "ConvNeXt_Atto":     "timm:convnext_atto",
    "ResNet10t":         "timm:resnet10t",
    "ResNet18":          "timm:resnet18",
}

GENERATOR_SOURCE_MAP = {
    "Only_V1":          ["DINOv1"],
    "Only_V2":          ["DINOv2"],
    "Only_CLIP":        ["CLIP"],
    "Only_SigLIP":      ["SigLIP"],
    "Hybrid_V1_CLIP":   ["DINOv1", "CLIP"],
    "Hybrid_V1_SigLIP": ["DINOv1", "SigLIP"],
    "Hybrid_V2_CLIP":   ["DINOv2", "CLIP"],
    "Hybrid_V2_SigLIP": ["DINOv2", "SigLIP"],
    "Hybrid_V1_V2":     ["DINOv1", "DINOv2"],
    "Hybrid_CLIP_SigLIP":["CLIP", "SigLIP"],
}

# ========================================================
# 4. Training Protocols
# ========================================================
TRAIN_CONFIGS = {
    "linear_torch": {
        "optimizer": "Adam", "lr": 0.001, "weight_decay": 1e-4, "epochs": 1000, "patience": 50,
    },
    "scratch": {
        "optimizer": "AdamW", "lr_backbone": 1e-3, "lr_head": 1e-3, "weight_decay": 1e-2, "epochs": 60, "patience": 10,
    },
    "full_ft": {
        "optimizer": "AdamW", "lr_backbone": 1e-5, "lr_head": 1e-3, "weight_decay": 1e-4, "epochs": 30, "patience": 10,
    },
}