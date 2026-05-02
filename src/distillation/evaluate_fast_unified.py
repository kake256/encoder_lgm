# ファイル名: src/distillation/evaluate_fast_unified_es.py
# 内容: 全学習モード・KD・自動ロジット・キャッシュ・CSV・実行順序・PyTorch高速化 ＋ 【大容量データ対応（ハイブリッド読み込み）版】

import os
import sys
import argparse
import json
import re
import random
import time
import copy
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms as T
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoProcessor, ResNetModel, AutoConfig
from datasets import load_dataset

warnings.filterwarnings("ignore", category=FutureWarning)

# Optional imports
try:
    import requests
except ImportError:
    requests = None

try:
    import timm
    from timm.data import resolve_data_config
except ImportError:
    timm = None

try:
    import open_clip
except ImportError:
    open_clip = None

# ========================================================
# 1. Configuration & Constants
# ========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 1024
NUM_WORKERS = min(32, os.cpu_count() if os.cpu_count() else 4)

KD_ALPHA = 0.5 
KD_TEMPERATURE = 4.0 

# [新規追加] メモリに一括で載せる最大サンプル数（これを超えると遅延読み込みに自動切替）
MAX_MEMORY_SAMPLES = 5000 

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
    "OpenCLIP_ConvNeXt": "openclip:convnext_base_w:laion2b_s13b_b82k",
    "MobileNetV2_050":   "timm:mobilenetv2_050",
    "MobileNetV3_S":     "timm:mobilenetv3_small_100",
    "GhostNet_100":      "timm:ghostnet_100",
    "EfficientNet_B0":   "timm:efficientnet_b0",
    "ConvNeXt_Atto":     "timm:convnext_atto",
    "ResNet10t":         "timm:resnet10t",
    "ResNet18":          "timm:resnet18",
}

GENERATOR_SOURCE_MAP = {
    "Hybrid_V1_CLIP":   ["DINOv1", "CLIP"],
    "Hybrid_V1_SigLIP": ["DINOv1", "SigLIP"],
    "Hybrid_V2_CLIP":   ["DINOv2", "CLIP"],
    "Hybrid_V2_SigLIP": ["DINOv2", "SigLIP"],
    "Hybrid_V1_V2":     ["DINOv1", "DINOv2"],
    "Hybrid_CLIP_SigLIP":["CLIP", "SigLIP"],
    "Only_V1":          ["DINOv1"],
    "Only_V2":          ["DINOv2"],
    "Only_CLIP":        ["CLIP"],
    "Only_SigLIP":      ["SigLIP"],
    "DINO-v1":          ["DINOv1"],
    "DINO-v2":          ["DINOv2"],
    "CLIP":             ["CLIP"],
    "SigLIP":           ["SigLIP"],
}

TRAIN_CONFIGS = {
    "linear": {
        "optimizer": "Adam", "lr": 0.001, "weight_decay": 0, "epochs": 1000, "val_interval": 20, "patience": 5, "batch_size": 128,
    },
    "scratch": {
        "optimizer": "AdamW", "lr_backbone": 1e-3, "lr_head": 1e-3, "weight_decay": 1e-2, "epochs": 60, "val_interval": 5, "patience": 10, "batch_size": 64,
    },
    "full_ft": {
        "optimizer": "AdamW", "lr_backbone": 1e-5, "lr_head": 1e-3, "weight_decay": 1e-4, "epochs": 50, "val_interval": 5, "patience": 5, "batch_size": 32,
    },
}

# ========================================================
# 2. Utilities & Feature Extraction for KD
# ========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def norm_name(s: str) -> str:
    s = s.lower()
    s = s.replace(" ", "").replace("_", "").replace(".", "").replace("-", "")
    return re.sub(r"_+", "_", s).strip("_")

def get_fewshot_subset(imgs, lbls, shots_per_class):
    subset_imgs, subset_lbls = [], []
    counts = {}
    for img, lbl in zip(imgs, lbls):
        counts.setdefault(lbl, 0)
        if counts[lbl] < shots_per_class:
            subset_imgs.append(img)
            subset_lbls.append(lbl)
            counts[lbl] += 1
    return subset_imgs, subset_lbls

def get_teachers_from_exp_name(exp_name):
    for key, teachers in GENERATOR_SOURCE_MAP.items():
        if key == exp_name or key in exp_name:
            return teachers
    teachers = []
    lower_name = exp_name.lower()
    if "dino-v1" in lower_name or "dinov1" in lower_name or "v1" in lower_name: teachers.append("DINOv1")
    if "dino-v2" in lower_name or "dinov2" in lower_name or "v2" in lower_name: teachers.append("DINOv2")
    if "clip" in lower_name: teachers.append("CLIP")
    if "siglip" in lower_name: teachers.append("SigLIP")
    return teachers if teachers else None

def ensure_logits_extracted(exp_name, image_paths, labels):
    missing_paths = [p for p in image_paths if not os.path.exists(os.path.splitext(p)[0] + '.pt')]
    if not missing_paths:
        return

    print(f"\n  [KD Auto-Extract] Missing {len(missing_paths)} logits for '{exp_name}'. Generating now...")
    
    teachers_to_use = get_teachers_from_exp_name(exp_name)
            
    if not teachers_to_use:
        print(f"  [Warning] Unknown generator source for '{exp_name}'. Cannot auto-extract logits.")
        return

    print(f"  -> Target Teachers: {teachers_to_use}")
    y_train = np.array(labels)
    all_teachers_logits = []

    for t_name in teachers_to_use:
        if t_name not in AVAILABLE_EVAL_MODELS: continue
        print(f"  -> Extracting features using {t_name}...")
        
        ext = FeatureExtractor(AVAILABLE_EVAL_MODELS[t_name], pretrained=True)
        temp_model = TrainableModel(ext, num_classes=1000).to(DEVICE)
        temp_model.eval()
        
        features = []
        with torch.no_grad(), autocast():
            for img_path in tqdm(image_paths, desc=f"Extract ({t_name})", leave=False):
                img = Image.open(img_path).convert("RGB")
                tensor_img = ext.get_transform(augment=False)(img).unsqueeze(0).to(DEVICE)
                feat = temp_model.get_features(tensor_img)
                features.append(feat.cpu().numpy()[0])
                
        del temp_model
        del ext
        torch.cuda.empty_cache()

        X_train = np.array(features)
        print(f"  -> Training Linear Probe for {t_name}...")
        clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0, multi_class="auto", n_jobs=-1)
        clf.fit(X_train, y_train)

        logits = np.dot(X_train, clf.coef_.T) + clf.intercept_
        all_teachers_logits.append(logits)

    if not all_teachers_logits: return

    print("  -> Ensembling and saving logits (.pt files)...")
    mean_logits = np.mean(all_teachers_logits, axis=0)
    
    for i, img_path in enumerate(image_paths):
        save_path = os.path.splitext(img_path)[0] + '.pt'
        logit_tensor = torch.tensor(mean_logits[i], dtype=torch.float32)
        torch.save(logit_tensor, save_path)
        
    print(f"  [KD Auto-Extract] Finished generating logits for '{exp_name}'.\n")

# ========================================================
# 3. Data Loading (Fast Memory Load & Lazy Load)
# ========================================================
class KDImageDataset(Dataset):
    def __init__(self, data_source, labels, transform):
        self.data_source = data_source
        self.labels = labels
        self.transform = transform
    def __len__(self): return len(self.data_source)
    def __getitem__(self, idx):
        item = self.data_source[idx]
        img = item
        if isinstance(item, str):
            try: img = Image.open(item).convert("RGB")
            except: img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
        if self.transform: img = self.transform(img)
        return img, self.labels[idx]

# [新規追加] 大容量データ用の遅延読み込みデータセット
class LazyTrainDataset(Dataset):
    def __init__(self, data_source, labels, use_kd, img_size=224):
        self.data_source = data_source
        self.labels = labels
        self.use_kd = use_kd
        self.transform = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])
        
    def __len__(self): 
        return len(self.data_source)
        
    def __getitem__(self, idx):
        item = self.data_source[idx]
        if isinstance(item, str):
            try: img = Image.open(item).convert("RGB")
            except: img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
        else:
            img = item 
            
        tensor_img = self.transform(img)
        label = self.labels[idx]
        
        logit = torch.zeros(1)
        if self.use_kd and isinstance(item, str):
            pt_path = os.path.splitext(item)[0] + '.pt'
            if os.path.exists(pt_path):
                try:
                    loaded = torch.load(pt_path, map_location='cpu')
                    if loaded.dim() == 2: loaded = loaded.squeeze(0)
                    logit = loaded
                except: pass
                
        return tensor_img, label, logit

def load_syn_data_to_memory(image_paths, labels, load_logits=False, img_size=224):
    transform = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])
    all_images, all_logits = [], []
    
    for img_path in tqdm(image_paths, desc="Loading Syn Data (Memory)", leave=False):
        try:
            img = Image.open(img_path).convert("RGB")
            tensor_img = transform(img)
        except Exception:
            tensor_img = torch.zeros(3, img_size, img_size)
        all_images.append(tensor_img)
        
        if load_logits:
            pt_path = os.path.splitext(img_path)[0] + '.pt'
            if os.path.exists(pt_path):
                try:
                    logit = torch.load(pt_path, map_location='cpu')
                    if logit.dim() == 2: logit = logit.squeeze(0)
                    all_logits.append(logit)
                except:
                    all_logits.append(None)
            else:
                all_logits.append(None)

    images_tensor = torch.stack(all_images)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    logits_tensor = None
    if load_logits and any(l is not None for l in all_logits):
        valid_l = next(l for l in all_logits if l is not None)
        dim = valid_l.shape[0]
        logits_tensor = torch.stack([l if l is not None else torch.zeros(dim) for l in all_logits])
        
    return images_tensor, labels_tensor, logits_tensor

def convert_pil_to_tensor_memory(images, labels, img_size=224):
    transform = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])
    tensor_list = []
    for img in tqdm(images, desc="Processing Real Data (Memory)", leave=False):
        try:
            tensor_img = transform(img)
        except Exception:
            tensor_img = torch.zeros(3, img_size, img_size)
        tensor_list.append(tensor_img)
    images_tensor = torch.stack(tensor_list)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return images_tensor, labels_tensor, None

# --- HuggingFace Helper Functions ---
def fetch_imagenet_wnid_map():
    if requests is None: return {}
    try:
        resp = requests.get("https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json", timeout=5)
        return {wnid: int(idx_str) for idx_str, (wnid, _) in resp.json().items()}
    except Exception: return {}

def _infer_label_id_map_from_hf(ds, synthetic_class_map):
    label_col = "label"
    if label_col not in ds.features:
        for c in ["fine_label", "coarse_label", "labels"]:
            if c in ds.features:
                label_col = c
                break
    
    hf_names = list(ds.features[label_col].names) if hasattr(ds.features[label_col], "names") else None
    syn_samples = list(synthetic_class_map.keys())[:3]
    hf_norm_to_id = {}
    if hf_names:
        for i, n in enumerate(hf_names):
            hf_norm_to_id[norm_name(n)] = i
            for p in n.split(","):
                hf_norm_to_id[norm_name(p)] = i

    syn_internal_to_hf_label = {}
    wnid_map = fetch_imagenet_wnid_map() if any(re.match(r"^n\d{8}$", k) for k in syn_samples) else None

    for syn_class_name, internal_idx in synthetic_class_map.items():
        hf_id = None
        name_part = syn_class_name
        m_prefix = re.match(r"^(\d+)[._](.+)$", syn_class_name)
        if m_prefix: name_part = m_prefix.group(2)
        if hf_id is None and wnid_map and syn_class_name in wnid_map: hf_id = wnid_map[syn_class_name]
        if hf_id is None:
            key = norm_name(name_part)
            if key in hf_norm_to_id: hf_id = hf_norm_to_id[key]
            if hf_id is None and "__" in name_part:
                k2 = norm_name(name_part.split("__")[0])
                if k2 in hf_norm_to_id: hf_id = hf_norm_to_id[k2]
            if hf_id is None and norm_name(syn_class_name) in hf_norm_to_id:
                hf_id = hf_norm_to_id[norm_name(syn_class_name)]
        if hf_id is not None:
            syn_internal_to_hf_label[internal_idx] = hf_id
    return syn_internal_to_hf_label, label_col

def load_real_data_hf_generic(hf_dataset, split, synthetic_class_map, max_per_class, seed=42):
    try: ds = load_dataset(hf_dataset, split=split, keep_in_memory=True)
    except Exception: ds = load_dataset(hf_dataset, split=split, trust_remote_code=True, keep_in_memory=True)

    internal_to_hf, label_col = _infer_label_id_map_from_hf(ds, synthetic_class_map)
    hf_to_internal = {v: k for k, v in internal_to_hf.items()}
    
    all_labels = np.array(ds[label_col])
    selected_indices = []
    
    if max_per_class <= 0:
        valid_hf_ids = set(internal_to_hf.values())
        mask = np.isin(all_labels, list(valid_hf_ids))
        selected_indices = np.where(mask)[0].tolist()
    else:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(ds))
        counts = {k: 0 for k in internal_to_hf.keys()}
        needed = len(internal_to_hf) * max_per_class
        collected = 0
        for idx in perm:
            hf_lbl = all_labels[idx]
            internal = hf_to_internal.get(hf_lbl)
            if internal is not None:
                if counts[internal] < max_per_class:
                    selected_indices.append(int(idx))
                    counts[internal] += 1
                    collected += 1
                    if collected >= needed: break

    subset = ds.select(selected_indices)
    images = [item["image"].convert("RGB") for item in subset]
    labels = [hf_to_internal[item[label_col]] for item in subset]
    return images, labels, len(internal_to_hf)

# ========================================================
# 4. Models & Feature Extractor
# ========================================================
class FeatureExtractor:
    def __init__(self, model_identifier: str, pretrained: bool = True):
        self.model_identifier = model_identifier
        self.type = "base"
        self.processor, self.core, self.preprocess = None, None, None
        self.embed_dim = 512
        
        if model_identifier.startswith("timm:"):
            if timm is None: raise ImportError("Please install timm")
            model_name = model_identifier.split(":", 1)[1]
            self.core = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
            self.type = "timm"
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
        if pretrained: self.core = AutoModel.from_pretrained(model_identifier)
        else:
            config = AutoConfig.from_pretrained(model_identifier)
            self.core = AutoModel.from_config(config)
            
        cfg = self.core.config
        if hasattr(cfg, "projection_dim"): self.embed_dim = cfg.projection_dim
        elif hasattr(cfg, "hidden_sizes"): self.embed_dim = cfg.hidden_sizes[-1]
        elif hasattr(cfg, "hidden_size"): self.embed_dim = cfg.hidden_size
        elif hasattr(cfg, "vision_config") and hasattr(cfg.vision_config, "hidden_size"): self.embed_dim = cfg.vision_config.hidden_size
        else: self.embed_dim = 768
            
        self.core.to(DEVICE)
        self.core.eval()

    def get_transform(self, augment: bool = False):
        if self.type == "timm":
            data_config = resolve_data_config({}, model=self.core)
            return create_transform(**data_config, is_training=False)
        if self.type == "open_clip": return self.preprocess
        def hf_process_wrapper(img):
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
        if self.model_type == "timm": f = self.core(x)
        elif self.model_type == "open_clip": f = self.core.encode_image(x)
        else:
            if hasattr(self.core, "get_image_features"): f = self.core.get_image_features(pixel_values=x)
            else:
                out = self.core(pixel_values=x)
                if hasattr(out, "image_embeds") and out.image_embeds is not None: f = out.image_embeds
                elif hasattr(out, "pooler_output") and out.pooler_output is not None: f = out.pooler_output
                elif hasattr(out, "last_hidden_state"): f = out.last_hidden_state[:, 0]
                else: f = out[0]
        if len(f.shape) > 2: f = f.flatten(1)
        f = f / (f.norm(dim=-1, keepdim=True) + 1e-6)
        return f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.get_features(x)
        logits = self.head(self.dropout(feats))
        return logits

    def set_mode(self, mode: str):
        if mode == "scratch": self.train()
        elif mode == "full_ft":
            self.train()
            for m in self.modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm): m.eval()
        else:
            self.eval()
            self.head.train()

# ========================================================
# 5. Fast Training Engine (Hybrid Enabled)
# ========================================================
gpu_augmentor = nn.Sequential(
    T.RandomResizedCrop(224, scale=(0.08, 1.0), antialias=True),
    T.RandomHorizontalFlip(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
).to(DEVICE)

def manual_kd_loss(student_logits, teacher_logits, temperature):
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
    return loss * (temperature ** 2)

# [変更] train_data に DataLoader(遅延読み込み) もしくは Tuple(オンメモリ) の両方を受け取れるように改修
def train_fast_unified(model, train_data, test_loader, mode, config, use_kd=False, cached_test_data=None):
    is_loader = isinstance(train_data, DataLoader)
    
    if not is_loader:
        # 1. 少ないデータの場合（オンメモリ高速モード）
        images, labels, soft_targets = train_data
        images = images.pin_memory()
        labels = labels.pin_memory()
        
        do_kd = False
        if use_kd and soft_targets is not None:
            if soft_targets.abs().sum() > 0: 
                soft_targets = soft_targets.pin_memory()
                do_kd = True
                
        num_samples = len(images)
        batch_size = min(num_samples, int(config.get("batch_size", 128)))
        num_batches = (num_samples + batch_size - 1) // batch_size
    else:
        # 2. 多いデータの場合（DataLoader遅延読み込みモード）
        train_loader = train_data
        do_kd = use_kd
        num_samples = len(train_loader.dataset)
        batch_size = train_loader.batch_size
        num_batches = len(train_loader)
    
    params = []
    if mode == "linear": params = [{"params": model.head.parameters()}]
    else: params = [{"params": model.core.parameters(), "lr": float(config.get("lr_backbone", 1e-4))},
                    {"params": model.head.parameters(), "lr": float(config.get("lr_head", 1e-3))}]
    
    opt_name = config.get("optimizer", "Adam")
    lr, wd = float(config.get("lr", 1e-3)), float(config.get("weight_decay", 0.0))
    if opt_name == "AdamW": optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    else: optimizer = torch.optim.Adam(params, lr=lr, weight_decay=wd)
    
    epochs = int(config.get("epochs", 100))
    val_interval = int(config.get("val_interval", 20))
    patience = int(config.get("patience", 5))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()
    criterion_ce = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    patience_counter = 0

    model.set_mode(mode)
    
    for epoch in range(epochs):
        if not is_loader:
            indices = torch.randperm(num_samples)
            
        batch_iter = train_loader if is_loader else range(num_batches)
        
        for step_data in batch_iter:
            if is_loader:
                # DataLoaderからバッチ取得
                x_batch, y_batch, t_logits = step_data
                x_batch = x_batch.to(DEVICE, non_blocking=True)
                y_batch = y_batch.to(DEVICE, non_blocking=True)
                if do_kd:
                    t_logits = t_logits.to(DEVICE, non_blocking=True)
            else:
                # メモリ上のTensorからバッチ取得
                i = step_data
                batch_idx = indices[i*batch_size : min((i+1)*batch_size, num_samples)]
                x_batch = images[batch_idx].to(DEVICE, non_blocking=True)
                y_batch = labels[batch_idx].to(DEVICE, non_blocking=True)
                if do_kd:
                    t_logits = soft_targets[batch_idx].to(DEVICE, non_blocking=True)
            
            with torch.no_grad(): x_batch = gpu_augmentor(x_batch)
                
            optimizer.zero_grad(set_to_none=True)
            
            with autocast():
                student_logits = model(x_batch)
                loss_ce = criterion_ce(student_logits, y_batch)
                loss = loss_ce
                
                if do_kd:
                    stu_dim = student_logits.shape[1]
                    tea_dim = t_logits.shape[1]
                    
                    if stu_dim != tea_dim:
                        if tea_dim > stu_dim:
                            t_logits_aligned = t_logits[:, :stu_dim]
                        else:
                            pad = torch.zeros((t_logits.shape[0], stu_dim - tea_dim), device=DEVICE)
                            t_logits_aligned = torch.cat([t_logits, pad], dim=1)
                    else:
                        t_logits_aligned = t_logits

                    loss_kd = manual_kd_loss(student_logits, t_logits_aligned, KD_TEMPERATURE)
                    loss = (1.0 - KD_ALPHA) * loss_ce + KD_ALPHA * loss_kd
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        scheduler.step()
        
        if (epoch + 1) % val_interval == 0 or (epoch + 1) == epochs:
            model.eval()
            with torch.no_grad(), autocast():
                if cached_test_data is not None:
                    test_feats, test_lbls = cached_test_data
                    test_logits = model.head(model.dropout(test_feats))
                    preds = test_logits.argmax(dim=1)
                    acc = (preds == test_lbls).float().mean().item()
                else:
                    correct, total = 0, 0
                    for tx, ty in test_loader:
                        tx = tx.to(DEVICE, non_blocking=True)
                        ty = ty.to(DEVICE, non_blocking=True)
                        preds = model(tx).argmax(dim=1)
                        correct += (preds == ty).sum().item()
                        total += ty.size(0)
                    acc = correct / total
            
            model.set_mode(mode)
            
            if acc > best_acc:
                best_acc = acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience > 0 and patience_counter >= patience:
                    break

    model.eval()
    with torch.no_grad(), autocast():
        mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1,3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1,3,1,1)
        correct, total = 0, 0
        
        if is_loader:
            for x_b, y_b, _ in train_loader:
                x_b = x_b.to(DEVICE, non_blocking=True)
                y_b = y_b.to(DEVICE, non_blocking=True)
                x_b = (x_b - mean) / std
                preds = model(x_b).argmax(dim=1)
                correct += (preds == y_b).sum().item()
                total += y_b.size(0)
        else:
            for i in range(num_batches):
                s = i * batch_size
                e = min(s + batch_size, num_samples)
                x_b = images[s:e].to(DEVICE, non_blocking=True)
                y_b = labels[s:e].to(DEVICE, non_blocking=True)
                x_b = (x_b - mean) / std
                preds = model(x_b).argmax(dim=1)
                correct += (preds == y_b).sum().item()
                total += y_b.size(0)
            
        train_acc = correct / total

    return float(best_acc), float(train_acc)

# ========================================================
# 6. Main Execution
# ========================================================
def get_sort_weight(exp_name):
    name = exp_name.lower()
    if "real" in name or "baseline" in name or "prototype" in name: return 10
    if "only_v1" in name or "dino-v1" in name: return 20
    if "only_v2" in name or "dino-v2" in name: return 30
    if "only_clip" in name or "clip" in name and "hybrid" not in name: return 40
    if "only_siglip" in name or "siglip" in name and "hybrid" not in name: return 50
    if "hybrid_v1_clip" in name: return 60
    if "hybrid_v1_siglip" in name: return 70
    if "hybrid_v2_clip" in name: return 80
    if "hybrid_v2_siglip" in name: return 90
    return 100

def main():
    parser = argparse.ArgumentParser(description="Unified Fast Evaluation for Synthetic Data.")
    parser.add_argument("dataset_dir", type=str)
    parser.add_argument("--output_dir", type=str, default="evaluation_results_fast")
    parser.add_argument("--mode", type=str, default="linear", choices=["linear", "scratch", "full_ft"])
    parser.add_argument("--kd_mode", type=str, default="none", choices=["logits", "none"])
    parser.add_argument("--num_trials", type=int, default=1)
    
    parser.add_argument("--dataset_type", type=str, default="imagenet", choices=["imagenet", "cub", "food101", "imagenet100"])
    parser.add_argument("--max_real_test", type=int, default=50)
    parser.add_argument("--syn_counts", type=int, nargs="+", default=[])
    parser.add_argument("--evaluators", type=str, nargs="+", default=["ResNet50"])
    parser.add_argument("--real_baseline_counts", type=int, nargs="+", default=[])

    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--val_interval", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)

    parser.add_argument("--mix_json", type=str, default=None)
    parser.add_argument("--mix_sources", type=str, nargs="+", default=[])

    parser.add_argument("--imagenet_hf_dataset", type=str, default="imagenet-1k")
    parser.add_argument("--imagenet_hf_test_split", type=str, default="validation")
    parser.add_argument("--food_hf_dataset", type=str, default="ethz/food101")
    parser.add_argument("--food_hf_test_split", type=str, default="validation")
    parser.add_argument("--cub_hf_dataset", type=str, default="Donghyun99/CUB-200-2011")
    parser.add_argument("--cub_hf_test_split", type=str, default="test")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(SEED)
    
    csv_path = os.path.join(args.output_dir, f"results_{args.mode}_{args.kd_mode}.csv")
    
    def save_result_realtime(result_dict):
        df_new = pd.DataFrame([result_dict])
        if not os.path.exists(csv_path):
            df_new.to_csv(csv_path, index=False)
        else:
            df_new.to_csv(csv_path, mode='a', header=False, index=False)

    train_config = TRAIN_CONFIGS[args.mode].copy()
    if args.patience is not None: train_config["patience"] = args.patience
    if args.val_interval is not None: train_config["val_interval"] = args.val_interval
    if args.epochs is not None: train_config["epochs"] = args.epochs

    use_kd = (args.kd_mode == "logits")

    root = Path(args.dataset_dir)
    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    class_map = {d.name: i for i, d in enumerate(class_dirs)}
    num_classes = len(class_map)

    experiments = {}
    for c_name, c_idx in class_map.items():
        for m_dir in (root / c_name).iterdir():
            if m_dir.is_dir():
                exp_name = m_dir.name
                experiments.setdefault(exp_name, {"paths": [], "labels": []})
                for img in m_dir.iterdir():
                    if img.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                        experiments[exp_name]["paths"].append(str(img))
                        experiments[exp_name]["labels"].append(c_idx)

    if use_kd:
        print("\n[Step] Verifying and Extracting KD Soft Targets (.pt files)...")
        for exp_name, data in experiments.items():
            if data["paths"]:
                ensure_logits_extracted(exp_name, data["paths"], data["labels"])

    mix_strategies = {}
    if args.mix_json and os.path.exists(args.mix_json):
        try: mix_strategies = json.load(open(args.mix_json))
        except: pass
    if args.mix_sources: mix_strategies["Hybrid_Custom"] = args.mix_sources
    for mix_name, sources in list(mix_strategies.items()):
        mix_strategies[mix_name] = [s for s in sources if s in experiments]

    target_syn_counts = sorted(list(set(args.syn_counts)))
    target_real_baseline_counts = sorted(list(set(args.real_baseline_counts)))

    if args.dataset_type in ["imagenet", "imagenet100"]:
        hf_name, te_split, tr_split = args.imagenet_hf_dataset, args.imagenet_hf_test_split, "train"
    elif args.dataset_type == "food101":
        hf_name, te_split, tr_split = args.food_hf_dataset, args.food_hf_test_split, "train"
    else:
        hf_name, te_split, tr_split = args.cub_hf_dataset, args.cub_hf_test_split, "train"

    real_test_imgs, real_test_lbls, _ = load_real_data_hf_generic(hf_name, te_split, class_map, args.max_real_test, seed=SEED)
    if not real_test_imgs: return

    real_train_imgs, real_train_lbls = [], []
    if target_real_baseline_counts:
        max_train_needed = max(target_real_baseline_counts)
        real_train_imgs, real_train_lbls, _ = load_real_data_hf_generic(hf_name, tr_split, class_map, max_train_needed, seed=SEED)

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    cache_dir = PROJECT_ROOT / ".feature_cache"
    if args.mode == "linear":
        cache_dir.mkdir(exist_ok=True, parents=True)

    for eval_name in args.evaluators:
        if eval_name not in AVAILABLE_EVAL_MODELS: continue
        print(f"\n========== Evaluator: {eval_name} ==========")
        extractor = FeatureExtractor(AVAILABLE_EVAL_MODELS[eval_name], pretrained=(args.mode != "scratch"))
        test_ds = KDImageDataset(real_test_imgs, real_test_lbls, extractor.get_transform(augment=False))
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        cached_test_data = None
        if args.mode == "linear":
            safe_eval_name = eval_name.replace("/", "_").replace(":", "_")
            cache_file = cache_dir / f"linear_probe_test_feats_{args.dataset_type}_{safe_eval_name}_max{args.max_real_test}.pt"
            
            if cache_file.exists():
                print(f"  [Cache] Loading pre-computed test features from {cache_file}...")
                cached_feats, cached_lbls = torch.load(cache_file, map_location=DEVICE)
                cached_test_data = (cached_feats, cached_lbls)
            else:
                print(f"  [Cache] Extracting test features for {eval_name} (First time only)...")
                temp_model = TrainableModel(extractor, num_classes).to(DEVICE)
                temp_model.eval()
                f_list, l_list = [], []
                with torch.no_grad(), autocast():
                    for tx, ty in tqdm(test_loader, desc="Caching Test Feats", leave=False):
                        f_list.append(temp_model.get_features(tx.to(DEVICE)).cpu())
                        l_list.append(ty.cpu())
                
                cached_feats = torch.cat(f_list)
                cached_lbls = torch.cat(l_list)
                
                print(f"  [Cache] Saving extracted features to {cache_file}...")
                torch.save((cached_feats, cached_lbls), cache_file)
                
                cached_test_data = (cached_feats.to(DEVICE), cached_lbls.to(DEVICE))

        def run_exp(name, imgs, lbls, is_real=False):
            if not imgs: return
            trial_accs, trial_t_accs = [], []
            
            # [変更] 画像の枚数によって読み込み方式を分岐
            is_large_dataset = len(imgs) > MAX_MEMORY_SAMPLES
            
            if is_large_dataset:
                print(f"  [Info] Data size ({len(imgs)}) exceeds threshold. Using Lazy DataLoader to prevent OOM.")
                current_use_kd = use_kd if not is_real else False
                lazy_ds = LazyTrainDataset(imgs, lbls, use_kd=current_use_kd)
                train_data = DataLoader(lazy_ds, batch_size=int(train_config.get("batch_size", 128)), 
                                        shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
            else:
                if is_real:
                    train_data = convert_pil_to_tensor_memory(imgs, lbls)
                    current_use_kd = False
                else:
                    train_data = load_syn_data_to_memory(imgs, lbls, load_logits=use_kd)
                    current_use_kd = use_kd
            
            for t in range(args.num_trials):
                set_seed(SEED + t)
                trainable_model = TrainableModel(extractor, num_classes).to(DEVICE)
                acc, t_acc = train_fast_unified(trainable_model, train_data, test_loader, args.mode, train_config, use_kd=current_use_kd, cached_test_data=cached_test_data)
                trial_accs.append(acc)
                trial_t_accs.append(t_acc)

            mean_acc, std_acc = float(np.mean(trial_accs)), float(np.std(trial_accs))
            print(f"  > {name:<30}: {mean_acc:.4f} ± {std_acc:.4f} (n={len(imgs)})")
            
            result_row = {
                "Evaluator": eval_name, "Generator": name, "Mode": args.mode, "KD": "none" if is_real else args.kd_mode,
                "Samples": len(imgs), "Accuracy": mean_acc, "Acc_Std": std_acc, "TrainAcc": float(np.mean(trial_t_accs))
            }
            save_result_realtime(result_row)

        task_queue = []

        if target_real_baseline_counts and real_train_imgs:
            for count in target_real_baseline_counts:
                r_imgs, r_lbls = get_fewshot_subset(real_train_imgs, real_train_lbls, count)
                task_name = f"Real_Baseline_{count}shot"
                task_queue.append((get_sort_weight(task_name), task_name, r_imgs, r_lbls, True))

        if experiments:
            for exp_name, data in experiments.items():
                if target_syn_counts:
                    for count in target_syn_counts:
                        s_imgs, s_lbls = get_fewshot_subset(data["paths"], data["labels"], count)
                        task_name = f"{exp_name}_{count}shot"
                        task_queue.append((get_sort_weight(task_name), task_name, s_imgs, s_lbls, False))
                else:
                    task_queue.append((get_sort_weight(exp_name), exp_name, data["paths"], data["labels"], False))

        if mix_strategies and target_syn_counts:
            for mix_name, source_list in mix_strategies.items():
                if not source_list: continue
                for count in target_syn_counts:
                    mix_imgs, mix_lbls = [], []
                    for src in source_list:
                        s_imgs, s_lbls = get_fewshot_subset(experiments[src]["paths"], experiments[src]["labels"], count)
                        mix_imgs.extend(s_imgs)
                        mix_lbls.extend(s_lbls)
                    task_name = f"{mix_name}_{count}shot_each"
                    task_queue.append((get_sort_weight(task_name), task_name, mix_imgs, mix_lbls, False))

        task_queue.sort(key=lambda x: x[0])
        
        for weight, name, imgs, lbls, is_real in task_queue:
            run_exp(name, imgs, lbls, is_real=is_real)

    print(f"\nDone! Saved to: {csv_path}")

if __name__ == "__main__":
    main()