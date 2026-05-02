# ファイル名: evaluate_linearKD_extract.py
# 内容: 合成画像上で教師モデルごとのLinear層を学習させ、ロジットのアンサンブル(平均)を保存する

import os
import argparse
import warnings
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

# scikit-learnの警告を非表示にする
warnings.filterwarnings("ignore", category=FutureWarning)

from evaluate_linearKD_config import DEVICE, GENERATOR_SOURCE_MAP, AVAILABLE_EVAL_MODELS
from evaluate_linearKD_models import FeatureExtractor
from evaluate_linearKD_data import set_seed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=str, help="Path to synthetic datasets")
    args = parser.parse_args()

    set_seed(42)
    root = Path(args.dataset_dir)
    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    
    # フォルダ(クラス)とラベルIDのマッピング
    class_map = {d.name: i for i, d in enumerate(class_dirs)}

    # 教師モデルの事前ロード
    required_teachers = set()
    for c_dir in class_dirs:
        for g_dir in c_dir.iterdir():
            if g_dir.is_dir() and g_dir.name in GENERATOR_SOURCE_MAP:
                required_teachers.update(GENERATOR_SOURCE_MAP[g_dir.name])

    teachers = {}
    for t_name in required_teachers:
        if t_name in AVAILABLE_EVAL_MODELS:
            print(f"Loading Teacher: {t_name}")
            teachers[t_name] = FeatureExtractor(AVAILABLE_EVAL_MODELS[t_name], pretrained=True)

    # 生成元(Generator)ごとに処理
    generators = set(g.name for c in class_dirs for g in c.iterdir() if g.is_dir())
    
    for gen_name in generators:
        if gen_name not in GENERATOR_SOURCE_MAP:
            continue
            
        print(f"\n--- Processing Generator: {gen_name} ---")
        t_names = GENERATOR_SOURCE_MAP[gen_name]
        
        # 1. 全画像のパスとラベルを収集
        image_paths, labels = [], []
        for c_dir in class_dirs:
            g_dir = c_dir / gen_name
            if g_dir.exists():
                for img_path in list(g_dir.glob("*.png")) + list(g_dir.glob("*.jpg")):
                    image_paths.append(img_path)
                    labels.append(class_map[c_dir.name])
        
        if not image_paths:
            continue

        y_train = np.array(labels)
        all_teachers_logits = []

        # 2. 各教師モデルで「独立して」特徴抽出とLinear学習を行う
        for t_name in t_names:
            print(f"  -> Processing Teacher: {t_name}")
            ext = teachers[t_name]
            features = []
            
            with torch.no_grad():
                for img_path in tqdm(image_paths, desc=f"Extracting ({t_name})", leave=False):
                    img = Image.open(img_path).convert("RGB")
                    tensor_img = ext.get_transform(augment=False)(img).unsqueeze(0).to(DEVICE)
                    
                    if ext.type == "timm" or ext.type == "swav":
                        feat = ext.core(tensor_img)
                    elif ext.type == "open_clip":
                        feat = ext.core.encode_image(tensor_img)
                    else:
                        if hasattr(ext.core, "get_image_features"):
                            feat = ext.core.get_image_features(pixel_values=tensor_img)
                        else:
                            out = ext.core(pixel_values=tensor_img)
                            if hasattr(out, "image_embeds") and out.image_embeds is not None:
                                feat = out.image_embeds
                            elif hasattr(out, "pooler_output") and out.pooler_output is not None:
                                feat = out.pooler_output
                            elif hasattr(out, "last_hidden_state"):
                                feat = out.last_hidden_state[:, 0]
                            else:
                                feat = out[0]

                    if len(feat.shape) > 2: 
                        feat = feat.flatten(1)
                    feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-6)
                    features.append(feat.cpu().numpy()[0])
            
            X_train = np.array(features)

            # Linear Probe の学習
            clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0, multi_class="auto", n_jobs=-1)
            clf.fit(X_train, y_train)

            # ロジットの計算
            logits = np.dot(X_train, clf.coef_.T) + clf.intercept_
            all_teachers_logits.append(logits)

        # 3. 複数の教師がいる場合（Hybrid）、ロジットを平均化（Ensemble）
        print("  -> Calculating and Saving Ensemble Logits...")
        mean_logits = np.mean(all_teachers_logits, axis=0)
        
        for i, img_path in enumerate(image_paths):
            save_path = img_path.with_suffix('.pt')
            logit_tensor = torch.tensor(mean_logits[i], dtype=torch.float32)
            torch.save(logit_tensor, save_path)

if __name__ == "__main__":
    main()