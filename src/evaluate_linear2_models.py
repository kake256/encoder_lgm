# ファイル名: evaluate_linear2_models.py
# 内容: FeatureExtractor, TrainableModel, および学習対象パラメータ選択ロジック

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm
from torch.cuda.amp import autocast
from transformers import AutoModel, AutoProcessor, ResNetModel

from evaluate_linear2_config import DEVICE, IMG_SIZE

try:
    import open_clip
except ImportError:
    open_clip = None


class FeatureExtractor:
    def __init__(self, model_identifier: str):
        self.model_identifier = model_identifier
        self.type = "base"
        self.processor, self.core, self.preprocess = None, None, None
        self.embed_dim = 512
        print(f"Loading backbone: {model_identifier} ...")

        # --- [SwAV] ロード処理 ---
        if model_identifier == "swav_resnet50":
            # PyTorch HubからSwAV学習済みのResNet50をロード
            self.core = torch.hub.load('facebookresearch/swav:main', 'resnet50')
            # 最終層(fc)を無効化して特徴量(2048次元)を直接取り出す
            self.core.fc = nn.Identity()
            self.type = "swav"
            self.embed_dim = 2048
            self.core.to(DEVICE)
            self.core.eval()
            return # 初期化終了
        # ------------------------

        if model_identifier.startswith("openclip:"):
            import open_clip
            parts = model_identifier.split(":")
            self.core, _, self.preprocess = open_clip.create_model_and_transforms(parts[1], pretrained=parts[2] if len(parts)>2 else "openai")
            self.type = "open_clip"
            self.embed_dim = getattr(self.core.visual, "output_dim", 512)
        elif "resnet" in model_identifier.lower(): # HF ResNet
            self.core = ResNetModel.from_pretrained(model_identifier)
            self.processor = AutoProcessor.from_pretrained(model_identifier)
            self.type = "resnet_hf"
            self.embed_dim = self.core.config.hidden_sizes[-1]
        else: # HF Transformer (MAE, SigLIP, etc.)
            self.core = AutoModel.from_pretrained(model_identifier)
            self.processor = AutoProcessor.from_pretrained(model_identifier)
            self.type = "hf_model"
            self.embed_dim = self.core.config.hidden_size

        self.core.to(DEVICE)
        self.core.eval()

    def get_transform(self, augment: bool = False):
        # --- [SwAV] 前処理 ---
        if self.type == "swav":
            # ImageNet標準の正規化
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
        # ---------------------

        # --- [OpenCLIP] 前処理 (修正: pass を削除しロジックを復元) ---
        if self.type == "open_clip":
            if augment:
                # CLIP用の Augmentation (Mean/Std は近似値)
                mean = (0.48145466, 0.4578275, 0.40821073)
                std = (0.26862954, 0.26130258, 0.27577711)
                return T.Compose([
                    T.RandomResizedCrop(224, scale=(0.8, 1.0), antialias=True),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(mean=mean, std=std)
                ])
            else:
                # Augmentationなしの場合は既存のpreprocessを使う
                return self.preprocess
        # ------------------------------------------------

        # --- [Hugging Face Models] 前処理 (修正: pass を削除しロジックを復元) ---
        def hf_process_wrapper(img):
            # Augmentationの適用
            if augment:
                aug = T.Compose([
                    T.RandomResizedCrop(224, scale=(0.8, 1.0), antialias=True),
                    T.RandomHorizontalFlip()
                ])
                img = aug(img)
            
            # HFプロセッサに通す
            return self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
        
        return hf_process_wrapper
        # ------------------------------------------------------------

    def extract_features(self, dataloader):
        self.core.eval()
        feats, lbls = [], []
        with torch.no_grad(), autocast():
            for imgs, labels in tqdm(dataloader, desc="Extracting", leave=False):
                imgs = imgs.to(DEVICE)
                
                # --- [SwAV] 推論 ---
                if self.type == "swav":
                    # fc=Identity() にしているので特徴量が返る
                    f = self.core(imgs)
                # -------------------------
                elif self.type == "open_clip": 
                    f = self.core.encode_image(imgs)
                else: 
                    # HF models
                    out = self.core(pixel_values=imgs)
                    if hasattr(out, "pooler_output") and out.pooler_output is not None: f = out.pooler_output
                    elif hasattr(out, "last_hidden_state"): f = out.last_hidden_state[:, 0]
                    else: f = out[0]
                
                if len(f.shape) > 2: f = f.flatten(1)
                f = f / (f.norm(dim=-1, keepdim=True) + 1e-6)
                feats.append(f.float().cpu().numpy())
                lbls.append(labels.numpy())
        if not feats: return np.array([]), np.array([])
        return np.concatenate(feats), np.concatenate(lbls)

class TrainableModel(nn.Module):
    """
    open_clipのとき, .visual だけに差し替えると encode_image が消えるため,
    full_core と visual_core を分けて保持する.
    """
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
        if self.model_type == "open_clip":
            f = self.full_core.encode_image(x)
        elif self.model_type == "swav": # SwAV対応
            f = self.core(x)
        else:
            out = self.core(pixel_values=x)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
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

def _named_module_leaf_keys(module: nn.Module):
    keys = set()
    for name, _m in module.named_modules():
        if not name:
            continue
        leaf = name.split(".")[-1]
        keys.add(leaf)
    return keys

def select_trainable_params(model: TrainableModel, mode: str, config: dict, train_batch_size: int):
    """
    点検の要: partial_ft, lora の対象パラメータをモデル構造に応じて明示的に決定する.
    return: (params_to_optimize, lora_targets, did_fallback, policy_name)
    """
    for p in model.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True

    params_to_optimize = []
    did_fallback = False
    policy_name = "none"
    lora_targets = []

    if mode == "linear_torch":
        base_lr = float(config["lr"])
        ref_bs = float(config.get("lr_ref_bs", 256.0))
        lr_cap = float(config.get("lr_cap", 0.005))
        bs = max(1, int(train_batch_size))
        scaled_lr = min(base_lr * (bs / ref_bs), lr_cap)
        policy_name = f"head_only_scaled_lr(bs={bs})"
        params_to_optimize = [{"params": model.head.parameters(), "lr": scaled_lr}]
        return params_to_optimize, [], did_fallback, policy_name

    if mode == "full_ft":
        policy_name = "full_backbone_plus_head"
        for p in model.core.parameters():
            p.requires_grad = True
        params_to_optimize = [
            {"params": model.core.parameters(), "lr": float(config["lr_backbone"])},
            {"params": model.head.parameters(), "lr": float(config["lr_head"])},
        ]
        return params_to_optimize, [], did_fallback, policy_name

    if mode == "partial_ft":
        trainable_backbone = None

        # --- [追加] SwAV (Standard ResNet structure) ---
        if model.model_type == "swav":
            # PyTorch標準のResNet構造なので layer4 が最終ステージ
            if hasattr(model.core, "layer4"):
                policy_name = "swav_resnet_layer4"
                trainable_backbone = list(model.core.layer4.parameters())
        # -----------------------------------------------

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

        return params_to_optimize, [], did_fallback, policy_name

    if mode == "lora":
        from peft import LoraConfig, get_peft_model

        candidates = list(config.get("lora_candidate_modules", []))
        if not candidates:
            candidates = ["q_proj", "v_proj", "query", "value", "c_fc", "c_proj"]

        # 対象モジュールが存在するかを leaf key でチェックする.
        leaf_keys = _named_module_leaf_keys(model.core)
        chosen = [c for c in candidates if c in leaf_keys]

        if not chosen:
            # 妥当性の観点で, 何も当たらないLoRAは実験として成立しないため止める.
            raise RuntimeError(
                f"[LoRA] No target_modules matched. model_type={model.model_type}, "
                f"candidates={candidates}, leaf_keys(sample)={sorted(list(leaf_keys))[:30]}"
            )

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
        return params_to_optimize, lora_targets, did_fallback, policy_name

    return params_to_optimize, lora_targets, did_fallback, policy_name