import os
import argparse
import random
import json
import copy
import re
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T, models

from sklearn.metrics import accuracy_score
from datasets import load_dataset
from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor, ResNetModel

# ========================================================
# 1. Config & Constants
# ========================================================
EVAL_MODELS = {
    "ResNet50": "microsoft/resnet-50",
    # "DINOv2":   "facebook/dinov2-base",
    # "CLIP":     "openai/clip-vit-base-patch16",
}

FT_CONFIG = {
    "pretrain_epochs": 10,
    "finetune_epochs": 30,
    "batch_size": 16,
    "lr_backbone": 1e-5,
    "lr_head": 1e-3,
    "weight_decay": 1e-4,
    "patience": 5
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
IMG_SIZE = 224
MAX_REAL_TRAIN = 50
MAX_REAL_TEST = 50

DISTRACTOR_IDS = {
    0: "Distractor_Tench",
    895: "Distractor_Warplane",
}

# ========================================================
# 2. Utilities & Dataset
# ========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_config(config_path):
    global FT_CONFIG
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            FT_CONFIG.update(json.load(f))
    print(f"FT Config: {FT_CONFIG}")

def norm_name(s):
    if not isinstance(s, str): return str(s)
    s = s.lower()
    s = s.replace(" ", "_").replace(".", "_")
    s = re.sub(r"[^a-z0-9_]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

class SimpleImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_input = self.images[idx]
        label = self.labels[idx]
        try:
            if isinstance(img_input, str):
                img = Image.open(img_input).convert("RGB")
            else:
                img = img_input.convert("RGB")
            
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception:
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), label

# ========================================================
# 3. Model Definition (Backbone + Head)
# ========================================================
class UnifiedClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.model_name = model_name
        print(f"Initializing UnifiedClassifier with {model_name}")
        
        if "resnet" in model_name:
            self.backbone = ResNetModel.from_pretrained(model_name)
            self.feature_dim = 2048
            self.model_type = "resnet"
        elif "clip" in model_name:
            clip = CLIPModel.from_pretrained(model_name)
            self.backbone = clip.vision_model
            self.feature_dim = clip.vision_model.config.hidden_size
            self.model_type = "clip"
        elif "dino" in model_name or "vit" in model_name:
            self.backbone = AutoModel.from_pretrained(model_name)
            self.feature_dim = self.backbone.config.hidden_size
            self.model_type = "vit"
        else:
            raise ValueError(f"Unknown model type: {model_name}")

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.fc = nn.Linear(self.feature_dim, num_classes)

    def get_transform(self, augment=False):
        def transform(image):
            if augment:
                aug_pil = T.Compose([
                    T.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0), antialias=True),
                    T.RandomHorizontalFlip(),
                    T.ColorJitter(0.2, 0.2, 0.2, 0.1),
                    T.RandomAffine(degrees=10, translate=(0.1, 0.1))
                ])
                image = aug_pil(image)
            inputs = self.processor(images=image, return_tensors="pt")
            return inputs["pixel_values"].squeeze(0)
        return transform

    def forward(self, x):
        if self.model_type == "resnet":
            out = self.backbone(x).pooler_output.flatten(1)
        elif self.model_type == "clip":
            out = self.backbone(x).pooler_output
        else:
            out = self.backbone(x).last_hidden_state[:, 0]
        return self.fc(out)

# ========================================================
# 4. Training Engine
# ========================================================
def train_model(model, train_loader, val_loader, epochs, lr_backbone, lr_head, device):
    criterion = nn.CrossEntropyLoss()
    params = [
        {'params': model.backbone.parameters(), 'lr': lr_backbone},
        {'params': model.fc.parameters(),       'lr': lr_head}
    ]
    optimizer = optim.SGD(params, momentum=0.9, weight_decay=FT_CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    loss_history = []
    
    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
        
        scheduler.step()
        epoch_loss = running_loss / total_samples if total_samples > 0 else 0
        loss_history.append(epoch_loss)
        
        if val_loader:
            acc = evaluate_model(model, val_loader, device)
            if acc > best_acc:
                best_acc = acc
                best_model_wts = copy.deepcopy(model.state_dict())
        else:
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    return model, loss_history, best_acc

def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    if len(all_labels) == 0: return 0.0
    return accuracy_score(all_labels, all_preds)

# ========================================================
# 5. Data Loading Logic (Integrated from evaluate_linear)
# ========================================================
def _infer_label_id_map_from_hf(ds, synthetic_class_map, dataset_name_tag="Dataset"):
    label_col = "label"
    if label_col not in ds.features:
        for c in ["fine_label", "coarse_label", "labels"]:
            if c in ds.features: label_col = c; break
    if label_col not in ds.features:
        raise ValueError(f"HF dataset has no label feature.")

    label_feature = ds.features[label_col]
    hf_names = list(label_feature.names) if hasattr(label_feature, "names") else None
    num_labels = len(hf_names) if hf_names else int(ds[label_col].max()) + 1
    hf_norm_to_id = {norm_name(n): i for i, n in enumerate(hf_names)} if hf_names else {}

    syn_internal_to_hf_label = {}
    for syn_class_name, internal_idx in synthetic_class_map.items():
        hf_id = None
        m = re.match(r"^(\d+)", syn_class_name)
        if m and 0 <= int(m.group(1))-1 < num_labels:
            hf_id = int(m.group(1))-1
        if hf_id is None and hf_norm_to_id:
            name_part = syn_class_name
            if "_" in syn_class_name and m: name_part = re.split(r"[._]", syn_class_name, 1)[1]
            elif "." in syn_class_name and m: name_part = syn_class_name.split(".", 1)[1]
            hf_id = hf_norm_to_id.get(norm_name(name_part))
        if hf_id is not None:
            syn_internal_to_hf_label[internal_idx] = hf_id
    return syn_internal_to_hf_label, label_col

def collect_imagenet_subset(dataset_name, split_name, target_ids, max_per_class, distractors=None):
    print(f"Connecting to {dataset_name} [{split_name}] stream...")
    try: ds = load_dataset(dataset_name, split=split_name, streaming=True, trust_remote_code=True)
    except: return [], [], [], []
    collected_imgs, collected_lbls, dist_imgs, dist_lbls = [], [], [], []
    counts = {k: 0 for k in target_ids.keys()}
    dist_counts = {k: 0 for k in (distractors.keys() if distractors else {})}
    for item in ds:
        lbl = item["label"]
        if lbl in target_ids and counts[lbl] < max_per_class:
            collected_imgs.append(item["image"].convert("RGB"))
            collected_lbls.append(target_ids[lbl])
            counts[lbl] += 1
        if distractors and lbl in distractors and dist_counts[lbl] < max_per_class:
            dist_imgs.append(item["image"].convert("RGB"))
            dist_lbls.append(lbl)
            dist_counts[lbl] += 1
        if all(c >= max_per_class for c in counts.values()) and \
           (not distractors or all(c >= max_per_class for c in dist_counts.values())): break
    return collected_imgs, collected_lbls, dist_imgs, dist_lbls

def load_cub_real_data_local(data_root, target_class_map, split="train", max_per_class=50):
    root = Path(data_root)
    images_dir = root / "images"
    if not images_dir.exists(): return [], []
    real_folders = sorted([d.name for d in images_dir.iterdir() if d.is_dir()])
    real_id_to_folder = {}
    real_norm_to_folder = {}
    for rf in real_folders:
        m = re.match(r"^(\d+)\.(.+)$", rf)
        if m:
            real_id_to_folder[int(m.group(1))] = rf
            real_norm_to_folder[norm_name(m.group(2))] = rf
        else: real_norm_to_folder[norm_name(rf)] = rf
    
    collected_imgs, collected_lbls = [], []
    print(f"Collecting CUB images (LOCAL) for [{split}]...")
    for class_name, class_idx in tqdm(target_class_map.items(), desc="Loading Classes"):
        target_real_folder = None
        m = re.match(r"^(\d+)", class_name)
        if m: target_real_folder = real_id_to_folder.get(int(m.group(1)))
        if not target_real_folder:
            name_part = class_name
            if "_" in class_name: name_part = class_name.split("_", 1)[1]
            elif "." in class_name: name_part = class_name.split(".", 1)[1]
            target_real_folder = real_norm_to_folder.get(norm_name(name_part))
        if not target_real_folder: continue
        
        class_dir = images_dir / target_real_folder
        img_files = sorted(list(class_dir.glob("*.jpg")))
        count = 0
        for img_path in img_files:
            if count >= max_per_class: break
            threshold = int(len(img_files) * 0.8)
            idx = img_files.index(img_path)
            is_target = (split == "train" and idx < threshold) or (split != "train" and idx >= threshold)
            if is_target:
                try:
                    img = Image.open(img_path).convert("RGB")
                    collected_imgs.append(img)
                    collected_lbls.append(class_idx)
                    count += 1
                except: pass
    return collected_imgs, collected_lbls

def load_real_data_hf_generic(hf_dataset, split, synthetic_class_map, max_per_class, seed=42, dataset_name_tag="Dataset"):
    print(f"Loading {dataset_name_tag} from HF...")
    try: ds = load_dataset(hf_dataset, split=split)
    except: ds = load_dataset(hf_dataset, split=split, trust_remote_code=True)
    internal_to_hf, label_col = _infer_label_id_map_from_hf(ds, synthetic_class_map, dataset_name_tag)
    hf_to_internal = {v: k for k, v in internal_to_hf.items()}
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    counts = {k: 0 for k in internal_to_hf.keys()}
    images, labels = [], []
    for idx in indices:
        item = ds[idx]
        internal = hf_to_internal.get(int(item[label_col]))
        if internal is None or counts[internal] >= max_per_class: continue
        img = item["image"]
        if not isinstance(img, Image.Image): img = Image.fromarray(np.array(img)).convert("RGB")
        else: img = img.convert("RGB")
        images.append(img)
        labels.append(internal)
        counts[internal] += 1
        if all(c >= max_per_class for c in counts.values()): break
    return images, labels

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

# ========================================================
# 6. Main
# ========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=str)
    parser.add_argument("--output_dir", type=str, default="ft_results")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset_type", type=str, default="imagenet", choices=["imagenet", "cub", "food101"])
    parser.add_argument("--cub_source", type=str, default="hf")
    parser.add_argument("--cub_hf_dataset", type=str, default="cassiekang/cub200_dataset")
    parser.add_argument("--cub_hf_train_split", type=str, default="train")
    parser.add_argument("--cub_hf_test_split", type=str, default="test")
    parser.add_argument("--food_hf_dataset", type=str, default="ethz/food101")
    parser.add_argument("--food_hf_train_split", type=str, default="train")
    parser.add_argument("--food_hf_test_split", type=str, default="validation")
    parser.add_argument("--imagenet_hf_dataset", type=str, default="imagenet-1k")
    parser.add_argument("--imagenet_hf_train_split", type=str, default="train")
    parser.add_argument("--imagenet_hf_test_split", type=str, default="validation")
    parser.add_argument("--real_data_dir", type=str, default=None)
    args = parser.parse_args()
    
    set_seed(SEED)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.config: load_config(args.config)

    # 1. Synthetic Data
    root = Path(args.dataset_dir)
    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    class_map = {d.name: i for i, d in enumerate(class_dirs)}
    num_classes = len(class_map)
    print(f"Classes: {num_classes}")

    experiments = {}
    for c_name, c_idx in class_map.items():
        for m_dir in (root / c_name).iterdir():
            if m_dir.is_dir():
                exp_name = m_dir.name
                experiments.setdefault(exp_name, {"paths": [], "labels": []})
                for img in m_dir.glob("*.png"):
                    experiments[exp_name]["paths"].append(str(img))
                    experiments[exp_name]["labels"].append(c_idx)
    
    if experiments:
        print(f"Synthetic Shots: {len(list(experiments.values())[0]['paths']) // num_classes}")
    else:
        print("No synthetic data found.")
        return

    # 2. Real Data
    real_train_imgs, real_train_lbls = [], []
    real_test_imgs, real_test_lbls = [], []
    
    if args.dataset_type == "imagenet":
        target_ids = {}
        for k, v in class_map.items():
            try: target_ids[int(k.split("_")[0])] = v
            except: pass
        real_test_imgs, real_test_lbls, _, _ = collect_imagenet_subset(
            args.imagenet_hf_dataset, args.imagenet_hf_test_split, target_ids, MAX_REAL_TEST)
        real_train_imgs, real_train_lbls, _, _ = collect_imagenet_subset(
            args.imagenet_hf_dataset, args.imagenet_hf_train_split, target_ids, MAX_REAL_TRAIN)
            
    elif args.dataset_type == "cub":
        if args.cub_source == "hf":
            real_train_imgs, real_train_lbls = load_real_data_hf_generic(
                args.cub_hf_dataset, args.cub_hf_train_split, class_map, MAX_REAL_TRAIN, SEED, "CUB")
            real_test_imgs, real_test_lbls = load_real_data_hf_generic(
                args.cub_hf_dataset, args.cub_hf_test_split, class_map, MAX_REAL_TEST, SEED+1, "CUB")
        else:
            real_train_imgs, real_train_lbls = load_cub_real_data_local(args.real_data_dir, class_map, "train", MAX_REAL_TRAIN)
            real_test_imgs, real_test_lbls = load_cub_real_data_local(args.real_data_dir, class_map, "validation", MAX_REAL_TEST)
            
    elif args.dataset_type == "food101":
        real_train_imgs, real_train_lbls = load_real_data_hf_generic(
            args.food_hf_dataset, args.food_hf_train_split, class_map, MAX_REAL_TRAIN, SEED, "Food101")
        real_test_imgs, real_test_lbls = load_real_data_hf_generic(
            args.food_hf_dataset, args.food_hf_test_split, class_map, MAX_REAL_TEST, SEED+1, "Food101")

    real_fewshot_imgs, real_fewshot_lbls = get_fewshot_subset(real_train_imgs, real_train_lbls, 5)
    print(f"Real Few-Shot (5-shot) Size: {len(real_fewshot_imgs)}")

    if len(real_fewshot_imgs) == 0:
        raise ValueError("Real Few-Shot data is empty. Check data loading paths or splits.")

    # 3. Experiment
    results = []
    
    for eval_name, model_id in EVAL_MODELS.items():
        print(f"\n====== Evaluator: {eval_name} ======")
        model_wrapper = UnifiedClassifier(model_id, num_classes)
        transform_eval = model_wrapper.get_transform(augment=False)
        transform_train = model_wrapper.get_transform(augment=True)
        
        val_ds = SimpleImageDataset(real_test_imgs, real_test_lbls, transform=transform_eval)
        val_loader = DataLoader(val_ds, batch_size=FT_CONFIG["batch_size"], shuffle=False)
        
        # Baseline
        print("\n--- Training Real Baseline (Few-Shot FT) ---")
        model = UnifiedClassifier(model_id, num_classes)
        train_ds = SimpleImageDataset(real_fewshot_imgs, real_fewshot_lbls, transform=transform_train)
        train_loader = DataLoader(train_ds, batch_size=FT_CONFIG["batch_size"], shuffle=True)
        
        model, _, acc = train_model(model, train_loader, val_loader, 
                                    FT_CONFIG["finetune_epochs"], FT_CONFIG["lr_backbone"], FT_CONFIG["lr_head"], DEVICE)
        print(f"  Result: {acc:.4f}")
        results.append({"Evaluator": eval_name, "Generator": "Real_Baseline", "Accuracy": acc})
        
        # Synthetic Transfer
        for exp_name, data in experiments.items():
            print(f"\n--- Training {exp_name} (Syn-Pretrain -> Real-FT) ---")
            
            # Phase 1
            model = UnifiedClassifier(model_id, num_classes)
            syn_ds = SimpleImageDataset(data["paths"], data["labels"], transform=transform_train)
            syn_loader = DataLoader(syn_ds, batch_size=FT_CONFIG["batch_size"], shuffle=True)
            
            print("  Phase 1: Pre-training on Synthetic...")
            model, _, syn_acc = train_model(model, syn_loader, val_loader, 
                                            FT_CONFIG["pretrain_epochs"], FT_CONFIG["lr_backbone"], FT_CONFIG["lr_head"], DEVICE)
            print(f"    Phase 1 Acc (Syn Only): {syn_acc:.4f}")
            results.append({"Evaluator": eval_name, "Generator": f"{exp_name} (SynOnly)", "Accuracy": syn_acc})
            
            # Phase 2
            print("  Phase 2: Fine-tuning on Real Few-Shot...")
            real_loader = DataLoader(train_ds, batch_size=FT_CONFIG["batch_size"], shuffle=True)
            model, _, ft_acc = train_model(model, real_loader, val_loader, 
                                           FT_CONFIG["finetune_epochs"], FT_CONFIG["lr_backbone"]*0.1, FT_CONFIG["lr_head"]*0.1, DEVICE)
            print(f"    Phase 2 Acc (Syn->Real): {ft_acc:.4f}")
            results.append({"Evaluator": eval_name, "Generator": f"{exp_name} (Syn->Real)", "Accuracy": ft_acc})

            del model
            torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, "ft_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")

if __name__ == "__main__":
    main()