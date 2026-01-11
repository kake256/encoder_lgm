# ファイル名: evaluate_linear2_config.py
# 内容: 実験設定, 定数, モデルリストの一元管理定義ファイル

import os
import torch

# ========================================================
# 1. System & Hardware Constants
# ========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
IMG_SIZE = 224

# [SPEEDUP] High VRAM Settings
BATCH_SIZE = 1024
NUM_WORKERS = min(32, os.cpu_count() if os.cpu_count() else 4)

# ========================================================
# 2. Dataset Constants
# ========================================================
MAX_REAL_TRAIN = 50
MAX_REAL_TEST = 50

# ========================================================
# 3. Model & Evaluator Lists
# ========================================================
AVAILABLE_EVAL_MODELS = {
    "ResNet50": "microsoft/resnet-50",
    "DINOv1":   "facebook/dino-vitb16",
    "DINOv2":   "facebook/dinov2-base",
    "CLIP":     "openai/clip-vit-base-patch16",
    "SigLIP":   "google/siglip-base-patch16-224",
    "MAE":      "facebook/vit-mae-base",
    "SwAV":     "swav_resnet50",
    "OpenCLIP_ViT_B32": "openclip:ViT-B-32:laion2b_s34b_b79k",
    "OpenCLIP_RN50":    "openclip:RN50:openai",
    "OpenCLIP_ConvNeXt": "openclip:convnext_base_w:laion400m_s13b_b51k",
}

GENERATOR_SOURCE_MAP = {
    "Only_V1":     ["DINOv1"],
    "Only_V2":     ["DINOv2"],
    "Only_CLIP":   ["CLIP"],
    "Only_SigLIP": ["SigLIP"],
    "Hybrid_V1_CLIP":   ["DINOv1", "CLIP"],
    "Hybrid_V1_SigLIP": ["DINOv1", "SigLIP"],
    "Hybrid_V2_CLIP":   ["DINOv2", "CLIP"],
    "Hybrid_V2_SigLIP": ["DINOv2", "SigLIP"],
    "Family_DINO": ["DINOv1", "DINOv2"],
    "Family_CLIP": ["CLIP", "SigLIP"],
}

STANDARD_EVALUATOR = "ResNet50"

# ========================================================
# 4. Training Protocols (Hyperparameters)
# ========================================================
TRAIN_CONFIGS = {
    "linear_lbfgs": {
        "optimizer": "LBFGS",
        "solver": "lbfgs",
        "max_iter": 1000,
        "C": 1.0,
    },
    "linear_torch": {
        "optimizer": "Adam",
        "lr": 0.001,
        "weight_decay": 1e-4,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "epochs": 1000,
        "patience": 50,
        "scheduler": "cosine",
        "val_interval": 50,
        # LR scaling cap. 元コード互換を意識した上限.
        "lr_cap": 0.005,
        "lr_ref_bs": 256.0,
    },
    "partial_ft": {
        "optimizer": "AdamW",
        "lr_backbone": 1e-5,
        "lr_head": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 30,
        "batch_size": 16,
        "scheduler": "cosine",
        "val_interval": 5,
    },
    "lora": {
        "optimizer": "AdamW",
        "lr_lora": 1e-4,
        "lr_head": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 30,
        "batch_size": 16,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "scheduler": "cosine",
        "val_interval": 5,
        # LoRAの探索候補. 存在しないモジュールは自動除外.
        "lora_candidate_modules": [
            "q_proj", "k_proj", "v_proj", "out_proj",
            "query", "key", "value",
            "c_fc", "c_proj",
        ],
    },
    "full_ft": {
        "optimizer": "AdamW",
        "lr_backbone": 1e-5,
        "lr_head": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 30,
        "batch_size": 16,
        "scheduler": "cosine",
        "val_interval": 5,
    },
}
