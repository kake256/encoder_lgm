# ファイル名: src/distillation/extract_nearest_real.py
# 内容: 合成画像に最も近い実画像 ＆ クラス中心(セントロイド)に最も近い実画像をモデルごとに抽出するスクリプト

import os
import argparse
import re
import warnings
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoProcessor, AutoConfig
from datasets import load_dataset
from torchvision import transforms as T

warnings.filterwarnings("ignore", category=FutureWarning)

try: import requests
except ImportError: requests = None
try:
    import timm
    from timm.data import resolve_data_config, create_transform
except ImportError: timm = None
try: import open_clip
except ImportError: open_clip = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

AVAILABLE_MODELS = {
    "ResNet50": "microsoft/resnet-50",
    "DINOv1":   "facebook/dino-vitb16",
    "DINOv2":   "facebook/dinov2-base",
    "CLIP":     "openai/clip-vit-base-patch16",
    "SigLIP":   "google/siglip-base-patch16-224",
    "MAE":      "facebook/vit-mae-base",
    "OpenCLIP_RN50": "openclip:RN50:openai",
}

# ==========================================
# Utils & Feature Extractor (評価コードと共通)
# ==========================================
def norm_name(s: str) -> str:
    s = s.lower().replace(" ", "").replace("_", "").replace(".", "").replace("-", "")
    return re.sub(r"_+", "_", s).strip("_")

class FeatureExtractor:
    def __init__(self, model_identifier: str):
        self.type = "base"
        self.processor, self.core, self.preprocess = None, None, None
        
        if model_identifier.startswith("timm:"):
            model_name = model_identifier.split(":", 1)[1]
            self.core = timm.create_model(model_name, pretrained=True, num_classes=0)
            self.type = "timm"
        elif model_identifier.startswith("openclip:"):
            parts = model_identifier.split(":")
            pt_tag = parts[2] if len(parts) > 2 else None
            self.core, _, self.preprocess = open_clip.create_model_and_transforms(parts[1], pretrained=pt_tag)
            self.type = "open_clip"
        else:
            self.type = "hf_model"
            self.processor = AutoProcessor.from_pretrained(model_identifier)
            self.core = AutoModel.from_pretrained(model_identifier)
            
        self.core.to(DEVICE)
        self.core.eval()

    def get_transform(self):
        if self.type == "timm":
            data_config = resolve_data_config({}, model=self.core)
            return create_transform(**data_config, is_training=False)
        if self.type == "open_clip": return self.preprocess
        def hf_process_wrapper(img):
            return self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
        return hf_process_wrapper

class ExtractorWrapper(nn.Module):
    def __init__(self, extractor):
        super().__init__()
        self.model_type = extractor.type
        self.core = extractor.core

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

# HFラベルの推論
def fetch_imagenet_wnid_map():
    if requests is None: return {}
    try:
        resp = requests.get("https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json", timeout=5)
        return {wnid: int(idx_str) for idx_str, (wnid, _) in resp.json().items()}
    except Exception: return {}

def infer_hf_labels(ds, local_class_names):
    label_col = "label"
    if label_col not in ds.features:
        for c in ["fine_label", "coarse_label", "labels"]:
            if c in ds.features: label_col = c; break
    
    mapping = {}
    
    # パターン1: フォルダ名が "0_bonnet" のように "数字_名前" の形式になっている場合
    # 最初の数字部分を直接HFのラベルIDとして採用する (最も確実)
    all_numeric = True
    for c_name in local_class_names:
        match = re.match(r"^(\d+)_", c_name)
        if match:
            mapping[c_name] = int(match.group(1))
        else:
            all_numeric = False
            break
            
    if all_numeric and len(mapping) == len(local_class_names):
        print(f"[*] Successfully matched classes using prefix numbers (e.g., '0_bonnet' -> 0)")
        return mapping, label_col

    # パターン2: 従来の名前やWNIDベースのマッチング
    hf_names = list(ds.features[label_col].names) if hasattr(ds.features[label_col], "names") else None
    hf_norm_to_id = {}
    if hf_names:
        for i, n in enumerate(hf_names):
            hf_norm_to_id[norm_name(n)] = i
            for p in n.split(","): hf_norm_to_id[norm_name(p)] = i

    wnid_map = fetch_imagenet_wnid_map()
    
    matched_count = 0
    for c_name in local_class_names:
        hf_id = None
        # WordNet ID (例: n01440764) の確認
        if wnid_map and c_name in wnid_map: 
            hf_id = wnid_map[c_name]
        
        if hf_id is None:
            # フォルダ名の正規化マッチング
            norm_c = norm_name(c_name)
            if norm_c in hf_norm_to_id:
                hf_id = hf_norm_to_id[norm_c]
            else:
                # "0_bonnet" などの場合、"bonnet" 部分だけで再検索
                parts = c_name.split("_", 1)
                if len(parts) > 1 and norm_name(parts[1]) in hf_norm_to_id:
                    hf_id = hf_norm_to_id[norm_name(parts[1])]
                    
        if hf_id is not None: 
            mapping[c_name] = hf_id
            matched_count += 1
            
    print(f"[*] Matched {matched_count}/{len(local_class_names)} classes with HF dataset.")
    return mapping, label_col

# ==========================================
# Main KNN Logic
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("syn_dataset_dir", type=str, help="クエリとなる合成画像が入ったディレクトリ")
    parser.add_argument("--output_base_dir", type=str, default="makeData", help="出力先のベースディレクトリ")
    parser.add_argument("--models", type=str, nargs="+", default=["ResNet50", "MAE", "CLIP"], help="KNNを計算するモデル")
    parser.add_argument("--top_k", type=int, default=5, help="抽出する近傍画像の数")
    parser.add_argument("--hf_dataset", type=str, default="clane9/imagenet-100", help="実画像の取得元")
    parser.add_argument("--hf_split", type=str, default="train", help="検索対象のデータスプリット(通常はtrain)")
    args = parser.parse_args()

    syn_root = Path(args.syn_dataset_dir)
    class_dirs = sorted([d for d in syn_root.iterdir() if d.is_dir()])
    local_classes = [d.name for d in class_dirs]
    print(f"[*] Found {len(local_classes)} classes in synthetic dataset.")

    print(f"[*] Loading real dataset from HuggingFace: {args.hf_dataset} [{args.hf_split}]")
    try: ds = load_dataset(args.hf_dataset, split=args.hf_split, keep_in_memory=True)
    except: ds = load_dataset(args.hf_dataset, split=args.hf_split, trust_remote_code=True, keep_in_memory=True)

    class_to_hf_id, label_col = infer_hf_labels(ds, local_classes)
    all_labels = np.array(ds[label_col])

    for model_name in args.models:
        if model_name not in AVAILABLE_MODELS:
            print(f"[!] Warning: Model '{model_name}' not found. Skipping.")
            continue
            
        print(f"\n=========================================")
        print(f" Processing KNN Extraction for: {model_name}")
        print(f"=========================================")
        
        out_dir = Path(args.output_base_dir) / f"dataset_nearest_{model_name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        extractor = FeatureExtractor(AVAILABLE_MODELS[model_name])
        wrapper = ExtractorWrapper(extractor).to(DEVICE)
        wrapper.eval()
        transform = extractor.get_transform()

        for c_name in tqdm(local_classes, desc=f"Classes ({model_name})"):
            if c_name not in class_to_hf_id: continue
            
            hf_id = class_to_hf_id[c_name]
            class_out_dir = out_dir / c_name
            class_out_dir.mkdir(exist_ok=True)

            # そのクラスの実画像をすべて取得
            real_indices = np.where(all_labels == hf_id)[0]
            real_subset = ds.select(real_indices)
            
            # 実画像の特徴量を一括抽出
            real_feats = []
            real_pil_images = []
            with torch.no_grad():
                for item in real_subset:
                    img = item["image"].convert("RGB")
                    real_pil_images.append(img)
                    tensor_img = transform(img).unsqueeze(0).to(DEVICE)
                    feat = wrapper.get_features(tensor_img)
                    real_feats.append(feat.cpu())
            real_feats_tensor = torch.cat(real_feats) # [N, D]

            # ==============================================================
            # 【追加】クラスタ自体(クラス全体)のプロトタイプ(重心)を計算・保存
            # ==============================================================
            # 1. クラス全体の平均特徴量(重心)を計算し、L2正規化
            centroid_feat = real_feats_tensor.mean(dim=0, keepdim=True)
            centroid_feat = centroid_feat / (centroid_feat.norm(dim=-1, keepdim=True) + 1e-6)
            
            # 2. 重心と各実画像のコサイン類似度を計算
            centroid_sims = F.cosine_similarity(centroid_feat, real_feats_tensor)
            
            k = min(args.top_k, len(centroid_sims))
            c_topk_vals, c_topk_indices = torch.topk(centroid_sims, k)
            
            # 3. 保存 (評価スクリプトで1つの手法として認識されるようにフォルダ分け)
            c_save_dir = class_out_dir / "Real_Class_Prototype"
            c_save_dir.mkdir(exist_ok=True)
            for rank, idx in enumerate(c_topk_indices.tolist()):
                c_save_path = c_save_dir / f"top{rank+1}_sim{c_topk_vals[rank]:.3f}.png"
                real_pil_images[idx].save(c_save_path)
            # ==============================================================


            # 各ジェネレータの合成画像をクエリとしてNNを探す
            class_path = syn_root / c_name
            for gen_dir in class_path.iterdir():
                if not gen_dir.is_dir(): continue
                gen_name = gen_dir.name
                
                syn_images = list(gen_dir.glob("*.png")) + list(gen_dir.glob("*.jpg"))
                if not syn_images: continue
                
                # フォルダ内の全ての合成画像を読み込み、特徴量の平均（重心）を計算する
                q_feats = []
                with torch.no_grad():
                    for img_path in syn_images:
                        q_img = Image.open(str(img_path)).convert("RGB")
                        q_tensor = transform(q_img).unsqueeze(0).to(DEVICE)
                        feat = wrapper.get_features(q_tensor).cpu()
                        q_feats.append(feat)
                
                # 特徴量の平均をとり、L2正規化して中心（セントロイド）を求める
                q_feat_mean = torch.cat(q_feats).mean(dim=0, keepdim=True)
                q_feat_mean = q_feat_mean / (q_feat_mean.norm(dim=-1, keepdim=True) + 1e-6)
                
                # 合成画像の中心特徴量と、実画像群の類似度を計算
                similarities = F.cosine_similarity(q_feat_mean, real_feats_tensor)
                
                """
                syn_images = list(gen_dir.glob("*.png")) + list(gen_dir.glob("*.jpg"))
                if not syn_images: continue
                
                query_path = str(syn_images[0]) 
                
                with torch.no_grad():
                    q_img = Image.open(query_path).convert("RGB")
                    q_tensor = transform(q_img).unsqueeze(0).to(DEVICE)
                    q_feat = wrapper.get_features(q_tensor).cpu()
                
                similarities = F.cosine_similarity(q_feat, real_feats_tensor)
                """
                
                k = min(args.top_k, len(similarities))
                topk_vals, topk_indices = torch.topk(similarities, k)
                
                save_dir = class_out_dir / f"Nearest_{gen_name}"
                save_dir.mkdir(exist_ok=True)
                
                for rank, idx in enumerate(topk_indices.tolist()):
                    save_path = save_dir / f"top{rank+1}_sim{topk_vals[rank]:.3f}.png"
                    real_pil_images[idx].save(save_path)

        del wrapper
        del extractor
        torch.cuda.empty_cache()

    print("\n[✔] All models completed successfully!")

if __name__ == "__main__":
    main()