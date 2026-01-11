# ファイル名: evaluate_linear2_data.py
# 内容: データセットの定義, HuggingFaceからのデータロード, 前処理ユーティリティ

import re
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from datasets import load_dataset

try:
    import requests
except ImportError:
    requests = None

# Config Import
from evaluate_linear2_config import IMG_SIZE

# ========================================================
# Utilities
# ========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def norm_name(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9_]+", "", s.replace(" ", "_").replace(".", "_"))
    return re.sub(r"_+", "_", s).strip("_")

class ImageDataset(Dataset):
    def __init__(self, data_source, labels, transform):
        self.data_source = data_source
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]
        lbl = self.labels[idx]
        img = item
        if isinstance(item, str):
            try:
                img = Image.open(item).convert("RGB")
            except Exception:
                img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
        if self.transform:
            img = self.transform(img)
        return img, lbl

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
# Data Loading Logic
# ========================================================
def fetch_imagenet_wnid_map():
    if requests is None:
        return {}
    url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    try:
        resp = requests.get(url, timeout=5)
        data = resp.json()
        wnid_to_id = {}
        for idx_str, (wnid, class_name) in data.items():
            wnid_to_id[wnid] = int(idx_str)
        return wnid_to_id
    except Exception:
        return {}

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
    wnid_map = None
    if any(re.match(r"^n\d{8}$", k) for k in syn_samples):
        wnid_map = fetch_imagenet_wnid_map()

    for syn_class_name, internal_idx in synthetic_class_map.items():
        hf_id = None
        name_part = syn_class_name
        m_prefix = re.match(r"^(\d+)[._](.+)$", syn_class_name)
        if m_prefix:
            name_part = m_prefix.group(2)

        if hf_id is None and wnid_map and syn_class_name in wnid_map:
            hf_id = wnid_map[syn_class_name]

        if hf_id is None:
            key = norm_name(name_part)
            if key in hf_norm_to_id:
                hf_id = hf_norm_to_id[key]
            if hf_id is None and "__" in name_part:
                k2 = norm_name(name_part.split("__")[0])
                if k2 in hf_norm_to_id:
                    hf_id = hf_norm_to_id[k2]
            if hf_id is None and norm_name(syn_class_name) in hf_norm_to_id:
                hf_id = hf_norm_to_id[norm_name(syn_class_name)]

        if hf_id is not None:
            syn_internal_to_hf_label[internal_idx] = hf_id

    print(f"  [Data] Matched {len(syn_internal_to_hf_label)} / {len(synthetic_class_map)} classes from HF.")
    return syn_internal_to_hf_label, label_col

def load_real_data_hf_generic(hf_dataset, split, synthetic_class_map, max_per_class, seed=42):
    print(f"Loading HF: {hf_dataset} [{split}]")
    try:
        ds = load_dataset(hf_dataset, split=split)
    except Exception:
        ds = load_dataset(hf_dataset, split=split, trust_remote_code=True)

    internal_to_hf, label_col = _infer_label_id_map_from_hf(ds, synthetic_class_map)
    hf_to_internal = {v: k for k, v in internal_to_hf.items()}

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    counts = {k: 0 for k in internal_to_hf.keys()}
    images, labels = [], []

    for idx in indices:
        try:
            lbl = int(ds[idx][label_col])
            internal = hf_to_internal.get(lbl)
            if internal is None or counts[internal] >= max_per_class:
                continue

            img = ds[idx]["image"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.array(img)).convert("RGB")
            else:
                img = img.convert("RGB")

            images.append(img)
            labels.append(internal)
            counts[internal] += 1
            if all(c >= max_per_class for c in counts.values()):
                break
        except Exception:
            continue

    return images, labels, len(internal_to_hf)
