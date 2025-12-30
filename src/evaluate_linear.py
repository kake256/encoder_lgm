import os
import argparse
import random
import numpy as np
import pandas as pd
import matplotlib
# サーバー実行用にGUIなしバックエンドを指定
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from datasets import load_dataset
from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor, ResNetModel

# ========================================================
# 1. 設定 & 定数
# ========================================================
EVAL_MODELS = {
    "ResNet50": "microsoft/resnet-50",
    "DINOv1":   "facebook/dino-vitb16",
    "DINOv2":   "facebook/dinov2-base",
    "CLIP":     "openai/clip-vit-base-patch16",
    "SigLIP":   "google/siglip-base-patch16-224"
}

DISTRACTOR_IDS = {
    0: "Distractor_Tench",
    895: "Distractor_Warplane"
}

BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
IMG_SIZE = 224

MAX_REAL_TRAIN = 50
MAX_REAL_TEST = 50

# ========================================================
# 2. ユーティリティ
# ========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert('RGB')
            return self.transform(img), self.labels[idx]
        except:
            return torch.zeros((3, 224, 224)), self.labels[idx]

class PilImageDataset(Dataset):
    def __init__(self, pil_images, labels, transform):
        self.images = pil_images
        self.labels = labels
        self.transform = transform
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        return self.transform(self.images[idx]), self.labels[idx]

def collect_imagenet_subset(split_name, target_ids, max_per_class, distractors=None):
    print(f"Connecting to ImageNet-1k [{split_name}] stream...")
    try:
        ds = load_dataset("imagenet-1k", split=split_name, streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load ImageNet {split_name}. Error: {e}")
        return [], [], [], []

    collected_imgs = []
    collected_lbls = []
    dist_imgs = []
    dist_lbls = []
    
    counts = {k:0 for k in target_ids.keys()}
    dist_counts = {k:0 for k in (distractors.keys() if distractors else {})}

    print(f"Collecting images from {split_name} (Max {max_per_class}/class)...")
    
    for item in ds:
        lbl = item['label']
        if lbl in target_ids and counts[lbl] < max_per_class:
            collected_imgs.append(item['image'].convert('RGB'))
            collected_lbls.append(target_ids[lbl])
            counts[lbl] += 1
            
        if distractors and lbl in distractors and dist_counts[lbl] < max_per_class:
            dist_imgs.append(item['image'].convert('RGB'))
            dist_lbls.append(lbl)
            dist_counts[lbl] += 1
        
        targets_done = all(c >= max_per_class for c in counts.values())
        dists_done = True
        if distractors:
            dists_done = all(c >= max_per_class for c in dist_counts.values())
            
        if targets_done and dists_done:
            break
            
    return collected_imgs, collected_lbls, dist_imgs, dist_lbls

def get_fewshot_subset(imgs, lbls, shots_per_class):
    subset_imgs = []
    subset_lbls = []
    counts = {}
    for img, lbl in zip(imgs, lbls):
        if lbl not in counts: counts[lbl] = 0
        if counts[lbl] < shots_per_class:
            subset_imgs.append(img)
            subset_lbls.append(lbl)
            counts[lbl] += 1
    return subset_imgs, subset_lbls

# ========================================================
# 3. 特徴抽出器
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
            # 1. データ拡張 (Augmentation)
            if augment:
                aug_pil = T.Compose([
                    T.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), antialias=True),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomAffine(degrees=10, translate=(0.1, 0.1)),
                    T.ColorJitter(0.1, 0.1, 0.1, 0.05),
                ])
                image = aug_pil(image)
                
                inputs = self.processor(images=image, return_tensors="pt")
                pixel_values = inputs['pixel_values'].squeeze(0)

                noise_adder = AddGaussianNoise(mean=0.0, std=0.05)
                pixel_values = noise_adder(pixel_values)
                return pixel_values

            else:
                # 2. 評価用
                inputs = self.processor(images=image, return_tensors="pt")
                return inputs['pixel_values'].squeeze(0)
            
        return transform

    def forward(self, pixel_values):
        with torch.no_grad():
            pixel_values = pixel_values.to(DEVICE)
            
            if self.type == "clip":
                outputs = self.core.get_image_features(pixel_values=pixel_values)
            elif self.type == "siglip":
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
# 4. 可視化 (t-SNE)
# ========================================================
def visualize_tsne(features_dict, labels_dict, title, save_path, class_names_map):
    print(f"Generating t-SNE plot for {title}...")
    
    all_feats, all_labels, all_types = [], [], []
    all_feats.append(features_dict['Real'])
    all_labels.append(labels_dict['Real'])
    all_types.append(np.zeros(len(labels_dict['Real'])))
    
    all_feats.append(features_dict['Train'])
    all_labels.append(labels_dict['Train'])
    all_types.append(np.ones(len(labels_dict['Train'])))
    
    if 'Distractor' in features_dict and len(features_dict['Distractor']) > 0:
        all_feats.append(features_dict['Distractor'])
        all_labels.append(np.full(len(features_dict['Distractor']), -1)) 
        all_types.append(np.full(len(features_dict['Distractor']), 2))
    
    X = np.concatenate(all_feats, axis=0)
    y = np.concatenate(all_labels, axis=0)
    types = np.concatenate(all_types, axis=0)
    
    n_samples = X.shape[0]
    perp = min(30, n_samples - 1) if n_samples > 1 else 1
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp, init='pca', learning_rate='auto')
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
            plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], marker='o', alpha=0.3, label=f"Test Real {c_name}", s=30)

    mask_syn = (types == 1)
    for lbl in unique_labels:
        if lbl == -1: continue
        idx = (y == lbl) & mask_syn
        if np.any(idx):
            plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], marker='*', c='black', s=150, edgecolors='white', label='_nolegend_', zorder=10)

    mask_dist = (types == 2)
    if np.any(mask_dist):
        plt.scatter(X_embedded[mask_dist, 0], X_embedded[mask_dist, 1], marker='^', c='red', s=50, alpha=0.5, label='Distractors')

    plt.title(f"t-SNE: {title}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ========================================================
# 5. ヘルパー関数: クラスごとの精度計算
# ========================================================
def evaluate_per_class(clf, X_test, y_test, id_to_class_name, generator_name, evaluator_name):
    """
    分類器の予測を行い、クラスごとの精度を計算して辞書リストを返す
    """
    y_pred = clf.predict(X_test)
    overall_acc = clf.score(X_test, y_test)
    
    results = []
    unique_classes = np.unique(y_test)
    
    for cls_id in unique_classes:
        # このクラスに該当するマスク
        mask = (y_test == cls_id)
        if np.sum(mask) == 0: continue
        
        # このクラスだけでの正解率
        cls_acc = (y_pred[mask] == y_test[mask]).mean()
        cls_name = id_to_class_name.get(cls_id, str(cls_id))
        
        results.append({
            "Generator": generator_name,
            "Evaluator": evaluator_name,
            "Class": cls_name,
            "Accuracy": cls_acc
        })
        
    return overall_acc, results

# ========================================================
# 6. メイン処理
# ========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=str)
    parser.add_argument("--output_dir", type=str, default="evaluation_results_critique")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation for training sets")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(SEED)

    print(f"Data Augmentation: {'ENABLED (Strong)' if args.augment else 'DISABLED'}")

    # --- Dataset解析 ---
    print(f"Scanning synthetic dataset: {args.dataset_dir}")
    root = Path(args.dataset_dir)
    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    class_map = {d.name: i for i, d in enumerate(class_dirs)} 
    # IDからクラス名への逆引きマップ
    id_to_class_name = {v: k for k, v in class_map.items()}
    
    experiments = {}
    syn_shots_per_class = 0
    
    for c_name, c_idx in class_map.items():
        for m_dir in (root / c_name).iterdir():
            if m_dir.is_dir():
                exp_name = m_dir.name
                if exp_name not in experiments: experiments[exp_name] = {"paths":[], "labels":[]}
                for img in m_dir.glob("*.png"):
                    experiments[exp_name]["paths"].append(str(img))
                    experiments[exp_name]["labels"].append(c_idx)

    if len(experiments) > 0:
        first_exp = list(experiments.values())[0]
        syn_shots_per_class = len(first_exp["paths"]) // len(class_map)
        print(f"Detected synthetic shots per class: {syn_shots_per_class}")
    else:
        print("No experiments found.")
        return

    target_ids = {}
    for k, v in class_map.items():
        try: target_ids[int(k.split('_')[0])] = v
        except: pass

    print("\n--- Loading REAL TEST Images (Validation) ---")
    real_test_imgs, real_test_lbls, dist_imgs, dist_lbls = collect_imagenet_subset(
        split_name="validation", target_ids=target_ids, max_per_class=MAX_REAL_TEST, distractors=DISTRACTOR_IDS
    )
    
    print("\n--- Loading REAL TRAIN Images (Train) for Baseline ---")
    real_train_imgs, real_train_lbls, _, _ = collect_imagenet_subset(
        split_name="train", target_ids=target_ids, max_per_class=MAX_REAL_TRAIN, distractors=None
    )
    
    real_fewshot_imgs, real_fewshot_lbls = get_fewshot_subset(
        real_train_imgs, real_train_lbls, shots_per_class=syn_shots_per_class
    )
    print(f"Created Real_FewShot dataset with {len(real_fewshot_imgs)} images ({syn_shots_per_class} per class).")

    # 全体の結果リスト
    final_results = []
    # クラスごとの詳細結果リスト
    class_wise_results = []

    for eval_name, model_id in EVAL_MODELS.items():
        print(f"\n--- Judge: {eval_name} ---")
        extractor = FeatureExtractor(model_id)
        
        transform_eval = extractor.get_transform(augment=False)
        transform_train = extractor.get_transform(augment=args.augment)
        
        # 1. Real TEST (Eval Transform)
        test_ds = PilImageDataset(real_test_imgs, real_test_lbls, transform_eval)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
        X_test, y_test = extract_features_and_labels(extractor, test_loader)
        
        # 2. Distractor (Eval Transform) - 可視化用
        X_dist = None
        if dist_imgs:
            dist_ds = PilImageDataset(dist_imgs, dist_lbls, transform_eval)
            dist_loader = DataLoader(dist_ds, batch_size=BATCH_SIZE)
            X_dist, _ = extract_features_and_labels(extractor, dist_loader)

        # 3. Real Baseline (Train Transform)
        if len(real_train_imgs) > 0:
            train_real_ds = PilImageDataset(real_train_imgs, real_train_lbls, transform_train)
            train_real_loader = DataLoader(train_real_ds, batch_size=BATCH_SIZE, shuffle=True)
            X_train_real, y_train_real = extract_features_and_labels(extractor, train_real_loader)
            
            clf = LogisticRegression(max_iter=1000, solver='lbfgs')
            clf.fit(X_train_real, y_train_real)
            
            # クラスごとの精度計算
            acc, cls_res = evaluate_per_class(clf, X_test, y_test, id_to_class_name, "Real_Baseline", eval_name)
            class_wise_results.extend(cls_res)
            
            print(f"  {'Real_Baseline (50)':<20} Acc: {acc:.4f} (Upper Bound)")
            final_results.append({
                "Generator": "Real_Baseline", "Evaluator": eval_name, "Accuracy": acc
            })
            
            tsne_dir = os.path.join(args.output_dir, "tsne_plots", eval_name)
            os.makedirs(tsne_dir, exist_ok=True)
            feats_dict = {'Real': X_test, 'Train': X_train_real}
            lbls_dict = {'Real': y_test, 'Train': y_train_real}
            if X_dist is not None: feats_dict['Distractor'] = X_dist
            visualize_tsne(feats_dict, lbls_dict, f"Real Baseline 50-shot ({eval_name})", os.path.join(tsne_dir, "Baseline_50shot.png"), class_map)

        # 4. Real Few-shot (Train Transform)
        if len(real_fewshot_imgs) > 0:
            train_few_ds = PilImageDataset(real_fewshot_imgs, real_fewshot_lbls, transform_train)
            train_few_loader = DataLoader(train_few_ds, batch_size=BATCH_SIZE, shuffle=True)
            X_train_few, y_train_few = extract_features_and_labels(extractor, train_few_loader)
            
            clf = LogisticRegression(max_iter=1000, solver='lbfgs')
            clf.fit(X_train_few, y_train_few)
            
            acc, cls_res = evaluate_per_class(clf, X_test, y_test, id_to_class_name, "Real_FewShot", eval_name)
            class_wise_results.extend(cls_res)
            
            print(f"  {f'Real_FewShot ({syn_shots_per_class})':<20} Acc: {acc:.4f}")
            final_results.append({
                "Generator": "Real_FewShot", "Evaluator": eval_name, "Accuracy": acc
            })
            
            feats_dict = {'Real': X_test, 'Train': X_train_few}
            lbls_dict = {'Real': y_test, 'Train': y_train_few}
            if X_dist is not None: feats_dict['Distractor'] = X_dist
            visualize_tsne(feats_dict, lbls_dict, f"Real Few-shot ({eval_name})", os.path.join(tsne_dir, "Baseline_FewShot.png"), class_map)

        # 5. Generator (Train Transform)
        for exp_name, data in experiments.items():
            train_ds = CustomImageDataset(data["paths"], data["labels"], transform_train)
            if len(train_ds) < len(class_map): continue
            
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            X_syn, y_syn = extract_features_and_labels(extractor, train_loader)
            if X_syn is None: continue

            clf = LogisticRegression(max_iter=1000, solver='lbfgs')
            clf.fit(X_syn, y_syn)
            
            acc, cls_res = evaluate_per_class(clf, X_test, y_test, id_to_class_name, exp_name, eval_name)
            class_wise_results.extend(cls_res)
            
            print(f"  {exp_name:<20} Acc: {acc:.4f}")
            final_results.append({
                "Generator": exp_name, "Evaluator": eval_name, "Accuracy": acc
            })

            tsne_path = os.path.join(args.output_dir, "tsne_plots", eval_name, f"{exp_name}.png")
            feats_dict = {'Real': X_test, 'Train': X_syn}
            lbls_dict = {'Real': y_test, 'Train': y_syn}
            if X_dist is not None: feats_dict['Distractor'] = X_dist
            visualize_tsne(feats_dict, lbls_dict, f"{exp_name} ({eval_name})", tsne_path, class_map)

        del extractor
        torch.cuda.empty_cache()

    # --- 保存処理 ---
    
    # 1. 全体精度の保存 (Pivot形式)
    if final_results:
        df = pd.DataFrame(final_results)
        piv = df.pivot(index="Generator", columns="Evaluator", values="Accuracy")
        cols = [c for c in EVAL_MODELS.keys() if c in piv.columns]
        piv = piv[cols]
        
        csv_path = os.path.join(args.output_dir, "final_critique_results_simple.csv")
        piv.to_csv(csv_path)
        print(f"\nSaved overall results to {csv_path}")
        print(piv)

    # 2. クラスごとの精度の保存 (CSV形式)
    if class_wise_results:
        df_cls = pd.DataFrame(class_wise_results)
        # カラム順を整理
        df_cls = df_cls[["Generator", "Evaluator", "Class", "Accuracy"]]
        
        cls_csv_path = os.path.join(args.output_dir, "class_wise_accuracy.csv")
        df_cls.to_csv(cls_csv_path, index=False)
        print(f"\nSaved class-wise accuracy to {cls_csv_path}")

if __name__ == "__main__":
    main()