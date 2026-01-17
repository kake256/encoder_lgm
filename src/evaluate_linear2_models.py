# ファイル名: evaluate_linear2_models.py
# 内容: FeatureExtractor (次元確定版), TrainableModel (BN制御修正版), 学習対象選択

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm
from torch.cuda.amp import autocast
from transformers import AutoModel, AutoProcessor, ResNetModel, AutoConfig

from evaluate_linear2_config import DEVICE, IMG_SIZE

# --- Optional Imports ---
try:
    import open_clip
except ImportError:
    open_clip = None

try:
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
except ImportError:
    timm = None
# ------------------------

class FeatureExtractor:
    def __init__(self, model_identifier: str, pretrained: bool = True):
        self.model_identifier = model_identifier
        self.type = "base"
        self.processor, self.core, self.preprocess = None, None, None
        self.embed_dim = 512
        
        init_state = "Pre-trained" if pretrained else "Random Init (Scratch)"
        print(f"Loading backbone: {model_identifier} [{init_state}] ...")

        # =========================================================
        # [1] timm Models
        # =========================================================
        if model_identifier.startswith("timm:"):
            if timm is None:
                raise ImportError("Please install timm: `pip install timm`")
            
            model_name = model_identifier.split(":", 1)[1]
            
            self.core = timm.create_model(
                model_name, 
                pretrained=pretrained, 
                num_classes=0 
            )
            self.type = "timm"
            self.data_config = resolve_data_config({}, model=self.core)
            self.core.to(DEVICE)
            self.core.eval() # デフォルトはeval

            # embed_dim 確定
            with torch.no_grad():
                dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
                out = self.core(dummy)
                if out.ndim > 2:
                    out = out.flatten(1)
                self.embed_dim = int(out.shape[1])
                print(f"   -> [Debug] Detected embed_dim: {self.embed_dim}")
            return

        # =========================================================
        # [2] SwAV
        # =========================================================
        if model_identifier == "swav_resnet50":
            self.core = torch.hub.load('facebookresearch/swav:main', 'resnet50', pretrained=pretrained)
            self.core.fc = nn.Identity()
            self.type = "swav"
            self.embed_dim = 2048
            self.core.to(DEVICE)
            self.core.eval()
            return 

        # =========================================================
        # [3] OpenCLIP
        # =========================================================
        if model_identifier.startswith("openclip:"):
            if open_clip is None:
                raise ImportError("Please install open_clip: `pip install open_clip_torch`")
                
            parts = model_identifier.split(":")
            pretrained_tag = parts[2] if (len(parts) > 2 and pretrained) else None
            
            self.core, _, self.preprocess = open_clip.create_model_and_transforms(
                parts[1], 
                pretrained=pretrained_tag
            )
            self.type = "open_clip"
            self.embed_dim = getattr(self.core.visual, "output_dim", 512)

        # =========================================================
        # [4] Hugging Face Models
        # =========================================================
        else:
            self.type = "hf_model"
            if "resnet" in model_identifier.lower(): 
                self.type = "resnet_hf"
                self.processor = AutoProcessor.from_pretrained(model_identifier)
                if pretrained:
                    self.core = ResNetModel.from_pretrained(model_identifier)
                else:
                    config = AutoConfig.from_pretrained(model_identifier)
                    self.core = ResNetModel(config)
                self.embed_dim = self.core.config.hidden_sizes[-1]
            else:
                self.processor = AutoProcessor.from_pretrained(model_identifier)
                if pretrained:
                    self.core = AutoModel.from_pretrained(model_identifier)
                else:
                    config = AutoConfig.from_pretrained(model_identifier)
                    self.core = AutoModel.from_config(config)
                
                cfg = self.core.config
                if hasattr(cfg, "projection_dim"): 
                    self.embed_dim = cfg.projection_dim
                elif hasattr(cfg, "hidden_size"): 
                    self.embed_dim = cfg.hidden_size
                elif hasattr(cfg, "vision_config") and hasattr(cfg.vision_config, "hidden_size"): 
                    self.embed_dim = cfg.vision_config.hidden_size
                else:
                    print(f"Warning: Could not detect embed_dim for {model_identifier}. Using 768.")
                    self.embed_dim = 768

        self.core.to(DEVICE)
        self.core.eval()

    def get_transform(self, augment: bool = False):
        if self.type == "timm":
            return create_transform(**self.data_config, is_training=augment)
        if self.type == "swav":
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
            if augment:
                return T.Compose([
                    T.RandomResizedCrop(224, scale=(0.8, 1.0), antialias=True),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(mean=mean, std=std)
                ])
            else:
                return T.Compose([
                    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(mean=mean, std=std)
                ])
        if self.type == "open_clip":
            if augment:
                mean = (0.48145466, 0.4578275, 0.40821073)
                std = (0.26862954, 0.26130258, 0.27577711)
                return T.Compose([
                    T.RandomResizedCrop(224, scale=(0.8, 1.0), antialias=True),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(mean=mean, std=std)
                ])
            else:
                return self.preprocess

        def hf_process_wrapper(img):
            if augment:
                aug = T.Compose([
                    T.RandomResizedCrop(224, scale=(0.8, 1.0), antialias=True),
                    T.RandomHorizontalFlip()
                ])
                img = aug(img)
            return self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
        return hf_process_wrapper

    def extract_features(self, dataloader):
        self.core.eval()
        feats, lbls = [], []
        with torch.no_grad(), autocast():
            for imgs, labels in tqdm(dataloader, desc="Extracting", leave=False):
                imgs = imgs.to(DEVICE)
                if self.type == "timm" or self.type == "swav":
                    f = self.core(imgs)
                elif self.type == "open_clip": 
                    f = self.core.encode_image(imgs)
                else: 
                    if hasattr(self.core, "get_image_features"):
                        f = self.core.get_image_features(pixel_values=imgs)
                    else:
                        out = self.core(pixel_values=imgs)
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
                feats.append(f.float().cpu().numpy())
                lbls.append(labels.numpy())
        if not feats: return np.array([]), np.array([])
        return np.concatenate(feats), np.concatenate(lbls)

class TrainableModel(nn.Module):
    def __init__(self, extractor: FeatureExtractor, num_classes: int):
        super().__init__()
        self.model_type = extractor.type
        self.embed_dim = extractor.embed_dim
        self.full_core = None
        self.visual_core = None
        self.core = extractor.core

        if self.model_type == "open_clip":
            self.full_core = extractor.core
            self.visual_core = extractor.core.visual

        self.head = nn.Linear(self.embed_dim, num_classes)
        self.head.weight.data.normal_(mean=0.0, std=0.01)
        self.head.bias.data.zero_()

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.model_type == "timm" or self.model_type == "swav":
            f = self.core(x)
        elif self.model_type == "open_clip":
            f = self.full_core.encode_image(x)
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
        if len(f.shape) > 2:
            f = f.flatten(1)
        f = f / (f.norm(dim=-1, keepdim=True) + 1e-6)
        return f

    def forward_head(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.get_features(x)
        return self.forward_head(feats)
    
    # 【修正】BNの挙動をScratchとFullFTで分離
    def set_mode(self, mode: str):
        if mode == "scratch":
            # Scratch: 統計量がないのでBNも学習する (trainモード)
            self.train()
            
        elif mode in ["full_ft", "full_ft_long"]:
            # Full FT: 重みは学習したいが、統計量は壊したくない
            self.train() # 全体は学習モード
            # BN層だけを探して固定(eval)モードにする
            for m in self.modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    m.eval()
                    
        elif mode == "partial_ft":
            self.eval()  
            self.head.train()
        else:
            self.eval()

def _named_module_leaf_keys(module: nn.Module):
    keys = set()
    for name, _m in module.named_modules():
        if not name:
            continue
        leaf = name.split(".")[-1]
        keys.add(leaf)
    return keys

def select_trainable_params(model: TrainableModel, mode: str, config: dict, train_batch_size: int):
    # 初期化：一旦すべて固定
    for p in model.parameters():
        p.requires_grad = False
    
    # Headは常に学習
    for p in model.head.parameters():
        p.requires_grad = True

    params_to_optimize = []
    did_fallback = False
    policy_name = "none"
    lora_targets = []

    # -----------------------------------------------------------
    # Linear Probing
    # -----------------------------------------------------------
    if mode == "linear_torch":
        base_lr = float(config["lr"])
        ref_bs = float(config.get("lr_ref_bs", 256.0))
        lr_cap = float(config.get("lr_cap", 0.005))
        bs = max(1, int(train_batch_size))
        scaled_lr = min(base_lr * (bs / ref_bs), lr_cap)
        policy_name = f"head_only_scaled_lr(bs={bs})"
        params_to_optimize = [{"params": model.head.parameters(), "lr": scaled_lr}]
        
        # Check
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   -> [Debug] Policy: {policy_name}, Trainable Params: {n_p}")
        return params_to_optimize, [], did_fallback, policy_name

    # -----------------------------------------------------------
    # Scratch / Full FT
    # -----------------------------------------------------------
    if mode == "full_ft" or mode == "full_ft_long" or mode == "scratch":
        policy_name = "full_backbone_plus_head"
        
        # 【重要】Backbone全体を解凍
        for p in model.core.parameters():
            p.requires_grad = True
            
        if model.model_type == "open_clip":
             for p in model.full_core.parameters():
                 p.requires_grad = True

        params_to_optimize = [
            {"params": model.core.parameters(), "lr": float(config["lr_backbone"])},
            {"params": model.head.parameters(), "lr": float(config["lr_head"])},
        ]
        
        # Check
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   -> [Debug] Policy: {policy_name}, Trainable Params: {n_p}")
        if n_p < 100000:
            print("   -> [WARNING] Trainable params seem too low for Scratch/Full-FT! Check Freeze logic.")
            
        return params_to_optimize, [], did_fallback, policy_name

    # -----------------------------------------------------------
    # Partial FT
    # -----------------------------------------------------------
    if mode == "partial_ft":
        trainable_backbone = None
        
        if model.model_type == "swav":
            if hasattr(model.core, "layer4"):
                policy_name = "swav_resnet_layer4"
                trainable_backbone = list(model.core.layer4.parameters())

        if model.model_type == "resnet_hf" and hasattr(model.core, "encoder") and hasattr(model.core.encoder, "stages"):
            policy_name = "resnet_encoder_stages_last"
            trainable_backbone = list(model.core.encoder.stages[-1].parameters())

        elif hasattr(model.core, "encoder") and hasattr(model.core.encoder, "layers"):
            policy_name = "transformer_encoder_layers_last"
            trainable_backbone = list(model.core.encoder.layers[-1].parameters())

        elif model.model_type == "open_clip" and model.visual_core is not None:
            if hasattr(model.visual_core, "transformer") and hasattr(model.visual_core.transformer, "resblocks"):
                policy_name = "openclip_visual_resblocks_last"
                trainable_backbone = list(model.visual_core.transformer.resblocks[-1].parameters())

        if trainable_backbone and len(trainable_backbone) > 0:
            for p in trainable_backbone:
                p.requires_grad = True
            params_to_optimize = [
                {"params": trainable_backbone, "lr": float(config["lr_backbone"])},
                {"params": model.head.parameters(), "lr": float(config["lr_head"])},
            ]
        else:
            did_fallback = True
            policy_name = f"partial_ft_fallback_to_head_only(model_type={model.model_type})"
            params_to_optimize = [{"params": model.head.parameters(), "lr": float(config.get("lr_head", 1e-3))}]
        
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   -> [Debug] Policy: {policy_name}, Trainable Params: {n_p}")
        return params_to_optimize, [], did_fallback, policy_name

    # -----------------------------------------------------------
    # LoRA
    # -----------------------------------------------------------
    if mode == "lora":
        from peft import LoraConfig, get_peft_model
        candidates = list(config.get("lora_candidate_modules", []))
        if not candidates:
            candidates = ["q_proj", "v_proj", "query", "value", "c_fc", "c_proj"]
        leaf_keys = _named_module_leaf_keys(model.core)
        chosen = [c for c in candidates if c in leaf_keys]

        if not chosen:
            raise RuntimeError(f"[LoRA] No target_modules matched. {model.model_type}")

        peft_config = LoraConfig(
            r=int(config["lora_rank"]),
            lora_alpha=int(config["lora_alpha"]),
            target_modules=chosen,
            lora_dropout=float(config["lora_dropout"]),
            bias="none",
        )
        model.core = get_peft_model(model.core, peft_config)
        policy_name = f"lora_targets={chosen}"
        lora_targets = chosen
        params_to_optimize = [
            {"params": model.core.parameters(), "lr": float(config["lr_lora"])},
            {"params": model.head.parameters(), "lr": float(config["lr_head"])},
        ]
        
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   -> [Debug] Policy: {policy_name}, Trainable Params: {n_p}")
        return params_to_optimize, lora_targets, did_fallback, policy_name

    return params_to_optimize, lora_targets, did_fallback, policy_name