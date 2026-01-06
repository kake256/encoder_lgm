import os
import argparse
import random
import re
import numpy as np
import pandas as pd
import matplotlib

# Server execution, headless backend.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm
from PIL import Image
from scipy.stats import spearmanr

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

from datasets import load_dataset
from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor, ResNetModel

# ========================================================
# 1. Config, constants.
# ========================================================
EVAL_MODELS = {
    "ResNet50": "microsoft/resnet-50",
    "DINOv1":   "facebook/dino-vitb16",
    "DINOv2":   "facebook/dinov2-base",
    "CLIP":     "openai/clip-vit-base-patch16",
    "SigLIP":   "google/siglip-base-patch16-224",
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

DISTRACTOR_IDS = {
    0: "Distractor_Tench",
    895: "Distractor_Warplane",
}

BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
IMG_SIZE = 224

MAX_REAL_TRAIN = 50
MAX_REAL_TEST = 50

# ========================================================
# 2. Utilities.
# ========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def norm_name(s: str) -> str:
    s = s.lower()
    s = s.replace(" ", "_").replace(".", "_")
    s = re.sub(r"[^a-z0-9_]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

class AddGaussianNoise(object):
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img), self.labels[idx]
        except Exception:
            return torch.zeros((3, IMG_SIZE, IMG_SIZE)), self.labels[idx]

class PilImageDataset(Dataset):
    def __init__(self, pil_images, labels, transform):
        self.images = pil_images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.transform(self.images[idx]), self.labels[idx]

# ========================================================
# 3. Real data loaders.
# ========================================================
def collect_imagenet_subset(dataset_name, split_name, target_ids, max_per_class, distractors=None):
    print(f"Connecting to {dataset_name} [{split_name}] stream...")
    try:
        ds = load_dataset(dataset_name, split=split_name, streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load {dataset_name}. Error: {e}")
        return [], [], [], []

    collected_imgs, collected_lbls = [], []
    dist_imgs, dist_lbls = [], []

    counts = {k: 0 for k in target_ids.keys()}
    dist_counts = {k: 0 for k in (distractors.keys() if distractors else {})}

    print(f"Collecting images from {split_name} (Max {max_per_class}/class)...")

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

        targets_done = all(c >= max_per_class for c in counts.values())
        dists_done = True
        if distractors:
            dists_done = all(c >= max_per_class for c in dist_counts.values())
        if targets_done and dists_done:
            break

    return collected_imgs, collected_lbls, dist_imgs, dist_lbls

def load_cub_real_data_local(data_root, target_class_map, split="train", max_per_class=50):
    root = Path(data_root)
    images_dir = root / "images"

    if not images_dir.exists():
        print(f"[Error] 'images' directory not found in {data_root}")
        return [], []

    real_folders = sorted([d.name for d in images_dir.iterdir() if d.is_dir()])
    real_id_to_folder = {}
    real_norm_to_folder = {}
    for rf in real_folders:
        m = re.match(r"^(\d+)\.(.+)$", rf)
        if m:
            rid = int(m.group(1))
            real_id_to_folder[rid] = rf
            real_norm_to_folder[norm_name(m.group(2))] = rf
        else:
            real_norm_to_folder[norm_name(rf)] = rf

    split_map = {}
    split_file = root / "train_test_split.txt"
    images_file = root / "images.txt"
    use_official_split = split_file.exists() and images_file.exists()

    path_to_id = {}
    if use_official_split:
        print(f"Loading CUB official split from {split_file}...")
        with open(images_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    path_to_id[parts[1]] = parts[0]
        with open(split_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    split_map[parts[0]] = int(parts[1])

    collected_imgs, collected_lbls = [], []
    print(f"Collecting CUB images (LOCAL) for [{split}] (Max {max_per_class}/class)...")

    matched_count, total_loaded = 0, 0
    debug_misses = []

    for class_name, class_idx in tqdm(target_class_map.items(), desc="Loading Classes"):
        target_real_folder = None
        m = re.match(r"^(\d+)", class_name)
        if m:
            cid = int(m.group(1))
            target_real_folder = real_id_to_folder.get(cid)
        if not target_real_folder:
            name_part = class_name
            if "_" in class_name: name_part = class_name.split("_", 1)[1]
            elif "." in class_name: name_part = class_name.split(".", 1)[1]
            key = norm_name(name_part)
            target_real_folder = real_norm_to_folder.get(key)

        if not target_real_folder:
            debug_misses.append(class_name)
            continue

        matched_count += 1
        class_dir = images_dir / target_real_folder
        img_files = sorted(list(class_dir.glob("*.jpg")))
        if len(img_files) == 0: continue

        count = 0
        for img_path in img_files:
            if count >= max_per_class: break
            is_target_split = True
            if use_official_split:
                rel_path = f"{target_real_folder}/{img_path.name}"
                if rel_path in path_to_id:
                    img_id = path_to_id[rel_path]
                    if img_id in split_map:
                        is_train_flag = (split_map[img_id] == 1)
                        if split == "train" and not is_train_flag: is_target_split = False
                        if split in ["validation", "test"] and is_train_flag: is_target_split = False
                    else: is_target_split = False
                else: is_target_split = False
            else:
                threshold = int(len(img_files) * 0.8)
                idx = img_files.index(img_path)
                if split == "train" and idx >= threshold: is_target_split = False
                if split in ["validation", "test"] and idx < threshold: is_target_split = False

            if is_target_split:
                try:
                    img = Image.open(img_path).convert("RGB")
                    collected_imgs.append(img)
                    collected_lbls.append(class_idx)
                    count += 1
                except Exception as e:
                    print(f"Error loading {img_path}. Error: {e}")
        total_loaded += count

    if matched_count == 0:
        print("\n[CRITICAL ERROR] No matching class folders found in LOCAL loader.")
        if debug_misses: print(f"Miss examples: {debug_misses[:10]}")
    elif total_loaded == 0:
        print(f"\n[WARNING] Classes matched ({matched_count}) but 0 images loaded.")
    else:
        print(f"[CUB LOCAL] Matched classes: {matched_count}, Loaded images: {total_loaded}")
    return collected_imgs, collected_lbls

def _infer_label_id_map_from_hf(ds, synthetic_class_map, dataset_name_tag="Dataset"):
    label_col = "label"
    if label_col not in ds.features:
        for candidate in ["fine_label", "coarse_label", "labels"]:
            if candidate in ds.features:
                label_col = candidate
                break
    if label_col not in ds.features:
        raise ValueError(f"HF dataset has no label feature. Available: {list(ds.features.keys())}")

    label_feature = ds.features[label_col]
    hf_names = None
    if hasattr(label_feature, "names") and label_feature.names is not None:
        hf_names = list(label_feature.names)

    num_labels = None
    if hf_names is not None:
        num_labels = len(hf_names)
    else:
        try: num_labels = int(ds[label_col].max()) + 1
        except Exception: num_labels = None

    hf_norm_to_id = {}
    if hf_names is not None:
        for i, n in enumerate(hf_names):
            hf_norm_to_id[norm_name(n)] = i

    syn_internal_to_hf_label = {}
    misses = []
    for syn_class_name, internal_idx in synthetic_class_map.items():
        hf_id = None
        m = re.match(r"^(\d+)", syn_class_name)
        if m and num_labels is not None:
            cid = int(m.group(1))
            candidate = cid - 1
            if 0 <= candidate < num_labels:
                hf_id = candidate
        if hf_id is None and hf_norm_to_id:
            name_part = syn_class_name
            if "_" in syn_class_name and m:
                parts = re.split(r"[._]", syn_class_name, 1)
                if len(parts) > 1: name_part = parts[1]
            elif "." in syn_class_name and m:
                parts = syn_class_name.split(".", 1)
                if len(parts) > 1: name_part = parts[1]
            key = norm_name(name_part)
            hf_id = hf_norm_to_id.get(key)
        
        if hf_id is None:
            misses.append(syn_class_name)
            continue
        syn_internal_to_hf_label[internal_idx] = hf_id

    if len(syn_internal_to_hf_label) == 0:
        raise RuntimeError(f"Failed to map any synthetic classes. Example: {list(synthetic_class_map.keys())[:5]}")
    if misses:
        print(f"[{dataset_name_tag} HF] Unmapped synthetic classes (examples): {misses[:10]}")
    return syn_internal_to_hf_label, label_col

def load_real_data_hf_generic(hf_dataset, split, synthetic_class_map, max_per_class, seed=42, dataset_name_tag="Dataset"):
    print(f"Loading {dataset_name_tag} from HF. dataset={hf_dataset}, split={split}")
    try:
        ds = load_dataset(hf_dataset, split=split)
    except Exception as e:
        print(f"Standard load failed: {e}. Retrying with trust_remote_code=True...")
        try: ds = load_dataset(hf_dataset, split=split, trust_remote_code=True)
        except Exception as e2: raise RuntimeError(f"Failed to load {hf_dataset}: {e2}")

    if "image" not in ds.features:
        raise ValueError(f"HF dataset has no 'image' feature.")

    internal_to_hf, label_col = _infer_label_id_map_from_hf(ds, synthetic_class_map, dataset_name_tag)
    hf_to_internal = {v: k for k, v in internal_to_hf.items()}

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    counts = {internal_idx: 0 for internal_idx in internal_to_hf.keys()}
    images, labels = [], []

    for idx in indices:
        item = ds[idx]
        hf_lbl = int(item[label_col])
        internal = hf_to_internal.get(hf_lbl, None)
        if internal is None: continue
        if counts[internal] >= max_per_class: continue

        img = item["image"]
        if not isinstance(img, Image.Image):
            try: img = Image.fromarray(np.array(img)).convert("RGB")
            except Exception: continue
        else: img = img.convert("RGB")
        images.append(img)
        labels.append(internal)
        counts[internal] += 1
        if all(c >= max_per_class for c in counts.values()): break

    total = len(images)
    matched_classes = sum(1 for c in counts.values() if c > 0)
    print(f"[{dataset_name_tag} HF] Matched classes: {matched_classes}/{len(counts)}, Loaded images: {total}")
    if total == 0: raise RuntimeError(f"Loaded 0 images from HF ({dataset_name_tag}).")
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
# 4. Feature extractor.
# ========================================================
class FeatureExtractor(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        print(f"Loading model: {model_name} ...")
        if "clip" in model_name and "siglip" not in model_name:
            self.core = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.type = "clip"
        elif "siglip" in model_name:
            self.core = AutoModel.from_pretrained(model_name)
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.type = "siglip"
        elif "resnet" in model_name:
            self.core = ResNetModel.from_pretrained(model_name)
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.type = "resnet"
        else:
            self.core = AutoModel.from_pretrained(model_name)
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.type = "base_vit"
        self.core.to(DEVICE)
        self.core.eval()

    def get_transform(self, augment=False):
        def transform(image):
            if augment:
                aug_pil = T.Compose([
                    T.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), antialias=True),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomAffine(degrees=10, translate=(0.1, 0.1)),
                    T.ColorJitter(0.1, 0.1, 0.1, 0.05),
                ])
                image = aug_pil(image)
                inputs = self.processor(images=image, return_tensors="pt")
                pixel_values = inputs["pixel_values"].squeeze(0)
                pixel_values = AddGaussianNoise(mean=0.0, std=0.05)(pixel_values)
                return pixel_values
            inputs = self.processor(images=image, return_tensors="pt")
            return inputs["pixel_values"].squeeze(0)
        return transform

    def forward(self, pixel_values):
        with torch.no_grad():
            pixel_values = pixel_values.to(DEVICE)
            if self.type in ["clip", "siglip"]:
                outputs = self.core.get_image_features(pixel_values=pixel_values)
            elif self.type == "resnet":
                out = self.core(pixel_values=pixel_values)
                outputs = out.pooler_output.flatten(1)
            else:
                outputs = self.core(pixel_values=pixel_values).last_hidden_state[:, 0]
            outputs = outputs / outputs.norm(dim=-1, keepdim=True)
            return outputs

def extract_features_and_labels(model, dataloader):
    feats, lbls = [], []
    for imgs, labels in tqdm(dataloader, desc="Extracting", leave=False):
        f = model(imgs)
        feats.append(f.cpu().numpy())
        lbls.append(labels.numpy())
    if not feats: return None, None
    return np.concatenate(feats), np.concatenate(lbls)

# ========================================================
# 5. Visualization, evaluation.
# ========================================================
def visualize_tsne(features_dict, labels_dict, title, save_path, class_names_map):
    print(f"Generating t-SNE plot for {title}...")
    all_feats, all_labels, all_types = [], [], []
    all_feats.append(features_dict["Real"])
    all_labels.append(labels_dict["Real"])
    all_types.append(np.zeros(len(labels_dict["Real"])))

    all_feats.append(features_dict["Train"])
    all_labels.append(labels_dict["Train"])
    all_types.append(np.ones(len(labels_dict["Train"])))

    if "Distractor" in features_dict and features_dict["Distractor"] is not None and len(features_dict["Distractor"]) > 0:
        all_feats.append(features_dict["Distractor"])
        all_labels.append(np.full(len(features_dict["Distractor"]), -1))
        all_types.append(np.full(len(features_dict["Distractor"]), 2))

    X = np.concatenate(all_feats, axis=0)
    y = np.concatenate(all_labels, axis=0)
    types = np.concatenate(all_types, axis=0)
    n_samples = X.shape[0]
    perp = min(30, n_samples - 1) if n_samples > 1 else 1
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp, init="pca", learning_rate="auto")
    X_embedded = tsne.fit_transform(X)

    plt.figure(figsize=(12, 10))
    unique_labels = np.unique(y)
    mask_real = (types == 0)
    for lbl in unique_labels:
        if lbl == -1: continue
        idx = (y == lbl) & mask_real
        if np.any(idx):
            c_names = [k for k, v in class_names_map.items() if v == lbl]
            c_name = c_names[0] if c_names else str(lbl)
            plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], marker="o", alpha=0.3, label=f"Test Real {c_name}", s=30)
    mask_syn = (types == 1)
    for lbl in unique_labels:
        if lbl == -1: continue
        idx = (y == lbl) & mask_syn
        if np.any(idx):
            plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], marker="*", c="black", s=150, edgecolors="white", label="_nolegend_", zorder=10)
    mask_dist = (types == 2)
    if np.any(mask_dist):
        plt.scatter(X_embedded[mask_dist, 0], X_embedded[mask_dist, 1], marker="^", c="red", s=50, alpha=0.5, label="Distractors")

    plt.title(f"t-SNE: {title}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def evaluate_per_class(clf, X_test, y_test, id_to_class_name, generator_name, evaluator_name, output_dir):
    y_pred = clf.predict(X_test)
    overall_acc = clf.score(X_test, y_test)

    # CM Save
    cm_dir = os.path.join(output_dir, "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)
    unique_classes_all = sorted(list(id_to_class_name.keys()))
    cm = confusion_matrix(y_test, y_pred, labels=unique_classes_all)
    
    cm_cols = [id_to_class_name.get(i, str(i)) for i in unique_classes_all]
    cm_df = pd.DataFrame(cm, index=cm_cols, columns=cm_cols)
    cm_path = os.path.join(cm_dir, f"{evaluator_name}_{generator_name}.csv")
    cm_df.to_csv(cm_path)

    results = []
    unique_classes = sorted(np.unique(y_test))
    acc_map = {}
    for cls_id in unique_classes:
        mask = (y_test == cls_id)
        if np.sum(mask) == 0:
            acc_map[cls_id] = 0.0
            continue
        cls_acc = (y_pred[mask] == y_test[mask]).mean()
        acc_map[cls_id] = float(cls_acc)
        cls_name = id_to_class_name.get(int(cls_id), str(cls_id))
        results.append({
            "Generator": generator_name,
            "Evaluator": evaluator_name,
            "Class": cls_name,
            "Accuracy": float(cls_acc),
        })

    acc_vector = [acc_map[cid] for cid in unique_classes]
    return float(overall_acc), results, acc_vector, cm

def calculate_complex_correlations(acc_store, experiments, output_dir):
    results = []
    if "Real_Baseline" not in acc_store.get(STANDARD_EVALUATOR, {}):
        print(f"[Warning] {STANDARD_EVALUATOR} evaluated on Real_Baseline is missing. Skipping Policy A.")
    for gen_name in experiments.keys():
        source_models = GENERATOR_SOURCE_MAP.get(gen_name, [])
        if not source_models: continue
        for source_model in source_models:
            if source_model not in acc_store: continue
            vec_resnet_syn = acc_store[STANDARD_EVALUATOR].get(gen_name)
            vec_resnet_real = acc_store[STANDARD_EVALUATOR].get("Real_Baseline")
            vec_source_syn = acc_store[source_model].get(gen_name)
            vec_source_real = acc_store[source_model].get("Real_Baseline")

            if vec_resnet_syn is None: continue
            row = {"Generator": gen_name, "Target_Source_Model": source_model}
            if vec_resnet_real is not None:
                corr, _ = spearmanr(vec_resnet_syn, vec_resnet_real)
                row["Policy_A (ResNet_Syn vs ResNet_Real)"] = float(corr)
            else: row["Policy_A (ResNet_Syn vs ResNet_Real)"] = None
            if vec_source_syn is not None and vec_source_real is not None:
                corr, _ = spearmanr(vec_source_syn, vec_source_real)
                row["Policy_D (Source_Syn vs Source_Real)"] = float(corr)
            else: row["Policy_D (Source_Syn vs Source_Real)"] = None
            results.append(row)
    if results:
        df = pd.DataFrame(results)
        cols = ["Generator", "Target_Source_Model", "Policy_A (ResNet_Syn vs ResNet_Real)", "Policy_D (Source_Syn vs Source_Real)"]
        df = df[[c for c in cols if c in df.columns]]
        save_path = os.path.join(output_dir, "complex_correlation_policies.csv")
        df.to_csv(save_path, index=False)
        print(f"\n[Success] Policy 1 (Difficulty Alignment) saved to: {save_path}")

def calculate_self_preference(final_results, output_dir):
    df = pd.DataFrame(final_results)
    if 'Accuracy' not in df.columns: return
    piv = df.pivot(index="Generator", columns="Evaluator", values="Accuracy")
    preference_results = []
    for gen_name in piv.index:
        source_models = GENERATOR_SOURCE_MAP.get(gen_name, [])
        if not source_models: continue
        for source_model in source_models:
            if source_model not in piv.columns: continue
            self_score = piv.loc[gen_name, source_model]
            other_cols = [c for c in piv.columns if c != source_model]
            if not other_cols: continue
            avg_other_score = piv.loc[gen_name, other_cols].mean()
            gap = self_score - avg_other_score
            preference_results.append({
                "Generator": gen_name,
                "Source_Model": source_model,
                "Self_Score": self_score,
                "Avg_Other_Score": avg_other_score,
                "Self_Preference_Gap": gap
            })
    if preference_results:
        pref_df = pd.DataFrame(preference_results)
        save_path = os.path.join(output_dir, "self_preference_policy.csv")
        pref_df.to_csv(save_path, index=False)
        print(f"\n[Success] Policy 2 (Self-Preference Gap) saved to: {save_path}")

def calculate_confusion_matrix_similarity(cm_store, experiments, output_dir):
    results = []
    for eval_name, gen_cms in cm_store.items():
        if "Real_Baseline" not in gen_cms: continue
        cm_real_baseline = gen_cms["Real_Baseline"].flatten()
        
        for gen_name in experiments.keys():
            if gen_name not in gen_cms: continue
            cm_syn = gen_cms[gen_name].flatten()
            if np.linalg.norm(cm_syn) > 0 and np.linalg.norm(cm_real_baseline) > 0:
                sim_baseline = np.dot(cm_syn, cm_real_baseline) / (np.linalg.norm(cm_syn) * np.linalg.norm(cm_real_baseline))
            else:
                sim_baseline = 0.0
            results.append({
                "Evaluator": eval_name,
                "Generator": gen_name,
                "Comparison": "vs_Real_Baseline",
                "Cosine_Similarity": sim_baseline
            })
            
    policy_results = []
    for gen_name in experiments.keys():
        source_models = GENERATOR_SOURCE_MAP.get(gen_name, [])
        for source_model in source_models:
            if source_model not in cm_store: continue
            cm_source_syn = cm_store[source_model].get(gen_name)
            cm_source_real = cm_store[source_model].get("Real_Baseline")
            if cm_source_syn is not None and cm_source_real is not None:
                f1 = cm_source_syn.flatten()
                f2 = cm_source_real.flatten()
                if np.linalg.norm(f1) > 0 and np.linalg.norm(f2) > 0:
                    sim = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))
                    policy_results.append({
                        "Generator": gen_name,
                        "Target_Source_Model": source_model,
                        "Policy_3 (Error_Consistency_Source_View)": sim
                    })

    if results:
        df_sim = pd.DataFrame(results)
        df_pivot = df_sim.pivot(index="Generator", columns="Evaluator", values="Cosine_Similarity")
        df_pivot.to_csv(os.path.join(output_dir, "confusion_matrix_similarity_baseline.csv"))
        print(f"\n[Success] General CM Similarity saved to: confusion_matrix_similarity_baseline.csv")
    if policy_results:
        df_policy = pd.DataFrame(policy_results)
        df_policy.to_csv(os.path.join(output_dir, "confusion_matrix_policy_consistency.csv"), index=False)
        print(f"\n[Success] Policy 3 (Error Consistency) saved to: confusion_matrix_policy_consistency.csv")

# ========================================================
# 6. Main.
# ========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=str, help="Path to synthetic dataset")
    parser.add_argument("--output_dir", type=str, default="evaluation_results_critique")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")
    parser.add_argument("--dataset_type", type=str, default="imagenet", choices=["imagenet", "cub", "food101"])
    parser.add_argument("--cub_source", type=str, default="hf", choices=["hf", "local"])
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

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(SEED)
    print(f"Dataset Type: {args.dataset_type}, Augmentation: {args.augment}")
    print(f"Scanning synthetic dataset: {args.dataset_dir}")
    root = Path(args.dataset_dir)
    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    class_map = {d.name: i for i, d in enumerate(class_dirs)}
    id_to_class_name = {v: k for k, v in class_map.items()}

    experiments = {}
    syn_shots_per_class = 0
    for c_name, c_idx in class_map.items():
        for m_dir in (root / c_name).iterdir():
            if m_dir.is_dir():
                exp_name = m_dir.name
                experiments.setdefault(exp_name, {"paths": [], "labels": []})
                for img in m_dir.glob("*.png"):
                    experiments[exp_name]["paths"].append(str(img))
                    experiments[exp_name]["labels"].append(c_idx)
    if len(experiments) > 0:
        first_exp = list(experiments.values())[0]
        if len(class_map) > 0: syn_shots_per_class = len(first_exp["paths"]) // len(class_map)
        print(f"Detected synthetic shots per class: {syn_shots_per_class}")
    else:
        print("No experiments found."); return

    real_test_imgs, real_test_lbls = [], []
    real_train_imgs, real_train_lbls = [], []
    dist_imgs, dist_lbls = [], []

    if args.dataset_type == "imagenet":
        target_ids = {}
        for k, v in class_map.items():
            try: target_ids[int(k.split("_")[0])] = v
            except Exception: pass
        real_test_imgs, real_test_lbls, dist_imgs, dist_lbls = collect_imagenet_subset(
            args.imagenet_hf_dataset, args.imagenet_hf_test_split, target_ids, MAX_REAL_TEST, DISTRACTOR_IDS)
        real_train_imgs, real_train_lbls, _, _ = collect_imagenet_subset(
            args.imagenet_hf_dataset, args.imagenet_hf_train_split, target_ids, MAX_REAL_TRAIN, None)

    elif args.dataset_type == "cub":
        if args.cub_source == "hf":
            real_train_imgs, real_train_lbls = load_real_data_hf_generic(
                args.cub_hf_dataset, args.cub_hf_train_split, class_map, MAX_REAL_TRAIN, SEED, "CUB")
            real_test_imgs, real_test_lbls = load_real_data_hf_generic(
                args.cub_hf_dataset, args.cub_hf_test_split, class_map, MAX_REAL_TEST, SEED+1, "CUB")
        else:
            if args.real_data_dir is None: raise ValueError("--real_data_dir required")
            real_train_imgs, real_train_lbls = load_cub_real_data_local(args.real_data_dir, class_map, "train", MAX_REAL_TRAIN)
            real_test_imgs, real_test_lbls = load_cub_real_data_local(args.real_data_dir, class_map, "validation", MAX_REAL_TEST)

    elif args.dataset_type == "food101":
        real_train_imgs, real_train_lbls = load_real_data_hf_generic(
            args.food_hf_dataset, args.food_hf_train_split, class_map, MAX_REAL_TRAIN, SEED, "Food101")
        real_test_imgs, real_test_lbls = load_real_data_hf_generic(
            args.food_hf_dataset, args.food_hf_test_split, class_map, MAX_REAL_TEST, SEED+1, "Food101")

    real_fewshot_imgs, real_fewshot_lbls = get_fewshot_subset(real_train_imgs, real_train_lbls, syn_shots_per_class)
    print(f"Created Real_FewShot dataset with {len(real_fewshot_imgs)} images.")

    final_results = []
    class_wise_results = []
    acc_store = {model_name: {} for model_name in EVAL_MODELS.keys()}
    cm_store = {model_name: {} for model_name in EVAL_MODELS.keys()} # CM Store

    for eval_name, model_id in EVAL_MODELS.items():
        print(f"\n--- Judge: {eval_name} ---")
        extractor = FeatureExtractor(model_id)
        transform_eval = extractor.get_transform(augment=False)
        transform_train = extractor.get_transform(augment=args.augment)
        if len(real_test_imgs) == 0: continue

        test_ds = PilImageDataset(real_test_imgs, real_test_lbls, transform_eval)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
        X_test, y_test = extract_features_and_labels(extractor, test_loader)
        
        X_dist = None
        if dist_imgs:
            dist_ds = PilImageDataset(dist_imgs, dist_lbls, transform_eval)
            dist_loader = DataLoader(dist_ds, batch_size=BATCH_SIZE)
            X_dist, _ = extract_features_and_labels(extractor, dist_loader)

        if len(real_train_imgs) > 0:
            train_real_ds = PilImageDataset(real_train_imgs, real_train_lbls, transform_train)
            train_real_loader = DataLoader(train_real_ds, batch_size=BATCH_SIZE, shuffle=True)
            X_train_real, y_train_real = extract_features_and_labels(extractor, train_real_loader)
            clf = LogisticRegression(max_iter=2000, solver="lbfgs")
            clf.fit(X_train_real, y_train_real)
            acc, cls_res, acc_vec, cm = evaluate_per_class(
                clf, X_test, y_test, id_to_class_name, "Real_Baseline", eval_name, args.output_dir
            )
            class_wise_results.extend(cls_res)
            acc_store[eval_name]["Real_Baseline"] = acc_vec
            cm_store[eval_name]["Real_Baseline"] = cm # Store CM
            final_results.append({"Generator": "Real_Baseline", "Evaluator": eval_name, "Accuracy": acc})
            
            tsne_dir = os.path.join(args.output_dir, "tsne_plots", eval_name)
            feats_dict = {"Real": X_test, "Train": X_train_real}
            lbls_dict = {"Real": y_test, "Train": y_train_real}
            if X_dist is not None: feats_dict["Distractor"] = X_dist
            visualize_tsne(feats_dict, lbls_dict, f"Real Baseline ({eval_name})", os.path.join(tsne_dir, "Baseline.png"), class_map)

        if len(real_fewshot_imgs) > 0:
            train_few_ds = PilImageDataset(real_fewshot_imgs, real_fewshot_lbls, transform_train)
            train_few_loader = DataLoader(train_few_ds, batch_size=BATCH_SIZE, shuffle=True)
            X_train_few, y_train_few = extract_features_and_labels(extractor, train_few_loader)
            clf = LogisticRegression(max_iter=2000, solver="lbfgs")
            clf.fit(X_train_few, y_train_few)
            acc, cls_res, acc_vec, cm = evaluate_per_class(
                clf, X_test, y_test, id_to_class_name, "Real_FewShot", eval_name, args.output_dir
            )
            class_wise_results.extend(cls_res)
            acc_store[eval_name]["Real_FewShot"] = acc_vec
            cm_store[eval_name]["Real_FewShot"] = cm # Store CM
            final_results.append({"Generator": "Real_FewShot", "Evaluator": eval_name, "Accuracy": acc})

        for exp_name, data in experiments.items():
            train_ds = CustomImageDataset(data["paths"], data["labels"], transform_train)
            if len(train_ds) < len(class_map): continue
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            X_syn, y_syn = extract_features_and_labels(extractor, train_loader)
            if X_syn is None: continue
            clf = LogisticRegression(max_iter=2000, solver="lbfgs")
            clf.fit(X_syn, y_syn)
            acc, cls_res, acc_vec, cm = evaluate_per_class(
                clf, X_test, y_test, id_to_class_name, exp_name, eval_name, args.output_dir
            )
            class_wise_results.extend(cls_res)
            acc_store[eval_name][exp_name] = acc_vec
            cm_store[eval_name][exp_name] = cm # Store CM
            final_results.append({"Generator": exp_name, "Evaluator": eval_name, "Accuracy": acc})
            
            tsne_path = os.path.join(args.output_dir, "tsne_plots", eval_name, f"{exp_name}.png")
            feats_dict = {"Real": X_test, "Train": X_syn}
            lbls_dict = {"Real": y_test, "Train": y_syn}
            if X_dist is not None: feats_dict["Distractor"] = X_dist
            visualize_tsne(feats_dict, lbls_dict, f"{exp_name} ({eval_name})", tsne_path, class_map)

        del extractor
        torch.cuda.empty_cache()

    if final_results:
        df = pd.DataFrame(final_results)
        piv = df.pivot(index="Generator", columns="Evaluator", values="Accuracy")
        cols = [c for c in EVAL_MODELS.keys() if c in piv.columns]
        piv = piv[cols]
        csv_path = os.path.join(args.output_dir, "final_results.csv")
        piv.to_csv(csv_path)
        print(f"\nSaved overall results to {csv_path}")
        calculate_self_preference(final_results, args.output_dir)

    if class_wise_results:
        df_cls = pd.DataFrame(class_wise_results)
        df_cls = df_cls[["Generator", "Evaluator", "Class", "Accuracy"]]
        cls_csv_path = os.path.join(args.output_dir, "class_wise_accuracy.csv")
        df_cls.to_csv(cls_csv_path, index=False)
        print(f"\nSaved class-wise accuracy to {cls_csv_path}")

    print("\n--- Calculating Policy Correlations ---")
    calculate_complex_correlations(acc_store, experiments, args.output_dir)
    print("\n--- Calculating Confusion Matrix Similarities ---")
    calculate_confusion_matrix_similarity(cm_store, experiments, args.output_dir)

if __name__ == "__main__":
    main()