# ファイル名: evaluate_linearKD_models.py
# 内容: FeatureExtractorとTrainableModelの定義 (ロジットKD対応版)

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm
from torch.cuda.amp import autocast
from transformers import AutoModel, AutoProcessor, ResNetModel, AutoConfig

from evaluate_linearKD_config import DEVICE, IMG_SIZE

try:
    import timm
    from timm.data.transforms_factory import create_transform
    from timm.data import resolve_data_config
except ImportError:
    timm = None

try:
    import open_clip
except ImportError:
    open_clip = None

class FeatureExtractor:
    def __init__(self, model_identifier: str, pretrained: bool = True):
        self.model_identifier = model_identifier
        self.type = "base"
        self.processor, self.core, self.preprocess = None, None, None
        self.embed_dim = 512
        
        print(f"Loading backbone: {model_identifier} (Pretrained: {pretrained})")

        if model_identifier.startswith("timm:"):
            if timm is None: raise ImportError("Please install timm")
            model_name = model_identifier.split(":", 1)[1]
            self.core = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
            self.type = "timm"
            self.data_config = resolve_data_config({}, model=self.core)
            self.core.to(DEVICE)
            self.core.eval()
            with torch.no_grad():
                dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
                self.embed_dim = int(self.core(dummy).flatten(1).shape[1])
            return

        if model_identifier.startswith("openclip:"):
            if open_clip is None: raise ImportError("Please install open_clip_torch")
            parts = model_identifier.split(":")
            pt_tag = parts[2] if (len(parts) > 2 and pretrained) else None
            self.core, _, self.preprocess = open_clip.create_model_and_transforms(parts[1], pretrained=pt_tag)
            self.type = "open_clip"
            self.embed_dim = getattr(self.core.visual, "output_dim", 512)
            self.core.to(DEVICE)
            self.core.eval()
            return

        self.type = "hf_model"
        self.processor = AutoProcessor.from_pretrained(model_identifier)
        if pretrained:
            self.core = AutoModel.from_pretrained(model_identifier)
        else:
            config = AutoConfig.from_pretrained(model_identifier)
            self.core = AutoModel.from_config(config)
            
        cfg = self.core.config
        
        # --- [修正箇所] ResNetなど hidden_sizes (配列) を持つモデルに対応 ---
        if hasattr(cfg, "projection_dim"): 
            self.embed_dim = cfg.projection_dim
        elif hasattr(cfg, "hidden_sizes"):
            self.embed_dim = cfg.hidden_sizes[-1]
        elif hasattr(cfg, "hidden_size"): 
            self.embed_dim = cfg.hidden_size
        elif hasattr(cfg, "vision_config") and hasattr(cfg.vision_config, "hidden_size"): 
            self.embed_dim = cfg.vision_config.hidden_size
        else:
            self.embed_dim = 768
        # -----------------------------------------------------------------
            
        self.core.to(DEVICE)
        self.core.eval()

    def get_transform(self, augment: bool = False, mode: str = "linear_torch"):
        if self.type == "timm":
            if augment and mode in ["scratch", "full_ft"]:
                return create_transform(
                    input_size=IMG_SIZE, is_training=True,
                    auto_augment="rand-m9-mstd0.5-inc1", interpolation='bicubic'
                )
            return create_transform(**self.data_config, is_training=augment)
        
        if self.type == "open_clip":
            if augment:
                mean, std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
                return T.Compose([
                    T.RandomResizedCrop(224, scale=(0.8, 1.0), antialias=True),
                    T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean=mean, std=std)
                ])
            return self.preprocess

        def hf_process_wrapper(img):
            if augment:
                aug = T.Compose([T.RandomResizedCrop(224, scale=(0.8, 1.0), antialias=True), T.RandomHorizontalFlip()])
                img = aug(img)
            return self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
            
        return hf_process_wrapper

class TrainableModel(nn.Module):
    def __init__(self, extractor: FeatureExtractor, num_classes: int):
        super().__init__()
        self.model_type = extractor.type
        self.embed_dim = extractor.embed_dim
        self.core = extractor.core
        self.dropout = nn.Dropout(p=0.5)
        self.head = nn.Linear(self.embed_dim, num_classes)
        
        self.head.weight.data.normal_(mean=0.0, std=0.01)
        self.head.bias.data.zero_()

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.model_type == "timm":
            f = self.core(x)
        elif self.model_type == "open_clip":
            f = self.core.encode_image(x)
        else:
            if hasattr(self.core, "get_image_features"):
                f = self.core.get_image_features(pixel_values=x)
            else:
                out = self.core(pixel_values=x)
                if hasattr(out, "image_embeds") and out.image_embeds is not None:
                    f = out.image_embeds
                elif hasattr(out, "pooler_output") and out.pooler_output is not None:
                    f = out.pooler_output
                elif hasattr(out, "last_hidden_state"):
                    f = out.last_hidden_state[:, 0]
                else:
                    f = out[0]
                
        if len(f.shape) > 2: f = f.flatten(1)
        f = f / (f.norm(dim=-1, keepdim=True) + 1e-6)
        return f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.get_features(x)
        logits = self.head(self.dropout(feats))
        return logits

    def set_mode(self, mode: str):
        if mode == "scratch": self.train()
        elif mode in ["full_ft"]:
            self.train()
            for m in self.modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm): m.eval()
        else:
            self.eval()
            self.head.train()