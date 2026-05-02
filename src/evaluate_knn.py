# src/evaluate_knn.py
import argparse
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
import umap
import plotly.express as px
import plotly.graph_objects as go

# 既存モジュールの再利用
from evaluate_linear2_config import AVAILABLE_EVAL_MODELS, DEVICE, SEED, BATCH_SIZE, NUM_WORKERS, MAX_REAL_TEST
from evaluate_linear2_data import load_real_data_hf_generic, ImageDataset, set_seed
from evaluate_linear2_models import FeatureExtractor

def predict_knn_labels(distances, indices, ref_labels, k=1, temperature=0.07, num_classes=1000):
    """
    DINOの実装に準拠した、温度パラメータ付きの重み付き投票。
    """
    # 1. すべての入力を Tensor に変換し、デバイスを統一する
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dists = torch.from_numpy(distances).float().to(device)
    sims = 1.0 - dists
    idx = torch.from_numpy(indices).long().to(device)
    
    # ref_labels が numpy array や list の場合を考慮して確実に Tensor 化
    if not isinstance(ref_labels, torch.Tensor):
        ref_labels = torch.tensor(ref_labels).long().to(device)
    else:
        ref_labels = ref_labels.long().to(device)
    
    batch_size = sims.shape[0]
    
    # 2. 重みの計算: exp(similarity / T)
    weights = (sims / temperature).exp()
    
    # 3. 近傍点のラベルを取得
    # [1, N_ref] -> [batch_size, N_ref]
    candidates = ref_labels.view(1, -1).expand(batch_size, -1)
    # 各バッチの近傍インデックスに対応するラベルを抽出
    retrieved_neighbors = torch.gather(candidates, 1, idx) # [batch_size, k]
    
    # 4. 重み付き投票の集計
    # [batch_size * k, num_classes] の one-hot を作成
    retrieval_one_hot = torch.zeros(batch_size * k, num_classes).to(device)
    retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
    
    # 重みを掛けてクラスごとに合計: [batch_size, num_classes]
    probs = torch.sum(
        retrieval_one_hot.view(batch_size, k, num_classes) * \
        weights.view(batch_size, k, 1),
        dim=1
    )
    
    # 5. 最大スコアのクラスを予測
    pred_labels = probs.argmax(dim=1)
    
    return pred_labels.cpu().numpy()
    
def calculate_intra_class_distance(X, y):
    unique_classes = np.unique(y)
    class_dists = []
    for cls in unique_classes:
        feats = X[y == cls]
        if len(feats) > 1:
            dists = cosine_distances(feats)
            upper_tri_indices = np.triu_indices_from(dists, k=1)
            mean_dist = np.mean(dists[upper_tri_indices])
            class_dists.append(mean_dist)
    return np.mean(class_dists) if class_dists else 0.0

def calculate_nn_coverage(idxs_v, y_syn):
    unique_classes = np.unique(y_syn)
    coverages = []
    for cls in unique_classes:
        cls_indices = np.where(y_syn == cls)[0]
        hit_real_indices = [idxs_v[i][0] for i in cls_indices]
        num_syn_imgs = len(cls_indices)
        if num_syn_imgs > 0:
            coverage = len(set(hit_real_indices)) / num_syn_imgs
            coverages.append(coverage)
    return np.mean(coverages) if coverages else 0.0

def calculate_centroids(X, y):
    unique_classes = np.unique(y)
    centroids = {}
    for cls in unique_classes:
        centroids[cls] = np.mean(X[y == cls], axis=0)
    return centroids

def calculate_centroid_proximity(X_syn, y_syn, real_centroids):
    sims = []
    for i in range(len(X_syn)):
        syn_feat = X_syn[i]
        syn_label = y_syn[i]
        
        if syn_label in real_centroids:
            target_centroid = real_centroids[syn_label]
            sim = cosine_similarity(syn_feat.reshape(1, -1), target_centroid.reshape(1, -1))[0][0]
            sims.append(sim)
    return np.mean(sims) if sims else 0.0

def save_neighbor_viz_split(syn_paths, real_imgs, indices, distances, class_names, syn_lbls, real_lbls, output_dir, model_name, gen_name):
    base_dir = os.path.join(output_dir, "viz", gen_name, model_name)
    correct_dir = os.path.join(base_dir, "correct")
    incorrect_dir = os.path.join(base_dir, "incorrect")
    os.makedirs(correct_dir, exist_ok=True)
    os.makedirs(incorrect_dir, exist_ok=True)
    
    count_correct, count_incorrect = 0, 0
    limit = 50 
    
    for i, syn_idx_list in enumerate(indices):
        neighbor_real_idx = syn_idx_list[0]
        dist = distances[i][0]
        syn_label = syn_lbls[i]
        real_label = real_lbls[neighbor_real_idx]
        is_correct = (syn_label == real_label)
        
        if is_correct:
            if count_correct >= limit: continue
        else:
            if count_incorrect >= limit: continue

        syn_data = syn_paths[i]
        real_data = real_imgs[neighbor_real_idx]
        cls_name_syn = class_names[syn_label] if syn_label < len(class_names) else str(syn_label)
        cls_name_real = class_names[real_label] if real_label < len(class_names) else str(real_label)
        
        try:
            img_syn = Image.open(syn_data).convert("RGB") if isinstance(syn_data, str) else syn_data.convert("RGB")
            img_real = Image.open(real_data).convert("RGB") if isinstance(real_data, str) else real_data.convert("RGB")
            
            w, h = img_syn.size
            img_real_resized = img_real.resize((w, h))
            combined = Image.new("RGB", (w * 2, h))
            combined.paste(img_syn, (0, 0))
            combined.paste(img_real_resized, (w, 0))
            
            fname = f"{i:04d}_S[{cls_name_syn}]_R[{cls_name_real}]_d{dist:.2f}.png"
            if is_correct:
                combined.save(os.path.join(correct_dir, fname))
                count_correct += 1
            else:
                combined.save(os.path.join(incorrect_dir, fname))
                count_incorrect += 1
        except Exception as e:
            pass

def create_umap_visualization(X_real_test, y_real_test, X_real_train_baseline, y_real_train_baseline, X_syn_dict, y_syn_dict, class_names, output_dir, model_name, dataset_type, baseline_shots):
    print(f"    -> Generating UMAP visualization for {model_name}...")
    umap_dir = os.path.join(output_dir, "umap")
    os.makedirs(umap_dir, exist_ok=True)

    baseline_label = f"Real Train ({baseline_shots}-shot)"

    all_X = [X_real_test, X_real_train_baseline]
    all_y = [y_real_test, y_real_train_baseline]
    all_types = [["Real Test"] * len(X_real_test), [baseline_label] * len(X_real_train_baseline)]
    
    for gen_name, X_s in X_syn_dict.items():
        all_X.append(X_s)
        all_y.append(y_syn_dict[gen_name])
        all_types.append([f"Syn: {gen_name}"] * len(X_s))
        
    X_combined = np.vstack(all_X)
    y_combined = np.concatenate(all_y)
    type_combined = np.concatenate(all_types)
    
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=SEED)
    embedding = reducer.fit_transform(X_combined)
    
    df_umap = pd.DataFrame({
        'UMAP_1': embedding[:, 0],
        'UMAP_2': embedding[:, 1],
        'Label_ID': y_combined,
        'Class_Name': [class_names[lbl] if lbl < len(class_names) else str(lbl) for lbl in y_combined],
        'Type': type_combined
    })

    # A. Plotly (HTML)
    fig = px.scatter(
        df_umap, x='UMAP_1', y='UMAP_2', 
        color='Class_Name', symbol='Type',
        hover_data=['Type', 'Class_Name'],
        title=f"UMAP Feature Space ({model_name} on {dataset_type})",
        opacity=0.8,
        template="plotly_white"
    )
    fig.update_traces(marker=dict(size=4), selector=dict(name="Real Test"))
    fig.update_traces(marker=dict(size=8, symbol='x'), selector=dict(name=baseline_label))
    
    html_path = os.path.join(umap_dir, f"umap_{model_name}.html")
    fig.write_html(html_path)
    
    # B. Matplotlib (PDF)
    gen_names = [baseline_label] + list(X_syn_dict.keys())
    num_plots = len(gen_names)
    
    cols = 2 if num_plots > 1 else 1
    rows = (num_plots + cols - 1) // cols
    fig_plt, axes = plt.subplots(rows, cols, figsize=(8*cols, 6*rows), squeeze=False)
    axes = axes.flatten()
    
    unique_classes = np.unique(y_real_test)
    cmap = plt.get_cmap('tab20')
    colors = {cls: cmap(i % 20) for i, cls in enumerate(unique_classes)}
    
    for idx, group_name in enumerate(gen_names):
        ax = axes[idx]
        
        real_mask = df_umap['Type'] == "Real Test"
        ax.scatter(df_umap.loc[real_mask, 'UMAP_1'], df_umap.loc[real_mask, 'UMAP_2'], 
                   c='lightgray', alpha=0.3, s=10, label='Real Test (Bg)')
        
        if group_name == baseline_label:
            target_mask = df_umap['Type'] == baseline_label
            marker_style = 'X' 
            s_size = 50
            title_str = f"Baseline: {baseline_label}"
        else:
            target_mask = df_umap['Type'] == f"Syn: {group_name}"
            marker_style = '*' 
            s_size = 80
            title_str = f"Generator: {group_name}"
            
        target_data = df_umap[target_mask]
        
        for cls_id in unique_classes:
            cls_mask = target_data['Label_ID'] == cls_id
            if cls_mask.any():
                ax.scatter(target_data.loc[cls_mask, 'UMAP_1'], target_data.loc[cls_mask, 'UMAP_2'], 
                           color=colors[cls_id], edgecolor='black', linewidth=0.5, s=s_size, marker=marker_style,
                           label=class_names[cls_id])
        
        ax.set_title(title_str, fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        if len(unique_classes) <= 20:
            ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize='small', title="Classes")
        
    for idx in range(num_plots, len(axes)):
        axes[idx].axis('off')
        
    plt.tight_layout()
    pdf_path = os.path.join(umap_dir, f"umap_comparison_{model_name}.pdf")
    plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate synthetic datasets using KNN.")
    parser.add_argument("dataset_dir", type=str, help="Path to synthetic dataset.")
    parser.add_argument("--output_dir", type=str, default="evaluation_results_knn")
    parser.add_argument("--dataset_type", type=str, required=True, choices=["imagenet", "cub", "food101", "imagenet100"])
    parser.add_argument("--hf_dataset", type=str, default=None)
    parser.add_argument("--hf_split", type=str, default=None)
    parser.add_argument("--evaluators", nargs="+", default=None)
    parser.add_argument("--k_neighbors", type=int, default=1, help="K for KNN (default: 1)")
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature for weighted KNN")
    parser.add_argument("--save_analysis", action="store_true", help="Save viz images.")
    parser.add_argument("--umap", action="store_true", help="Generate UMAP visualization (PDF & HTML).")
    
    parser.add_argument("--baseline_shots", type=str, default="5", help="Comma-separated list of shots for baseline (e.g., '5,200').")
    parser.add_argument("--max_train_samples", type=int, default=50, help="Max real images to load per class for centroids and baseline.")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(SEED)
    
    try:
        baseline_shots_list = [int(s.strip()) for s in args.baseline_shots.split(",")]
    except ValueError:
        print("Error: --baseline_shots must be a comma-separated list of integers.")
        return
        
    max_needed_shots = max(baseline_shots_list)
    actual_max_train_samples = max(args.max_train_samples, max_needed_shots)

    print(f"==================================================")
    print(f" KNN Evaluation (k={args.k_neighbors}, T={args.temperature})")
    print(f" Target         : {args.dataset_dir}")
    print(f" Baseline Shots : {baseline_shots_list}")
    print(f" Max Train Load : {actual_max_train_samples}")
    print(f" Mode           : {'Analysis/UMAP' if args.save_analysis or args.umap else 'Accuracy Only'}")
    print(f"==================================================")

    root = Path(args.dataset_dir)
    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    class_map = {d.name: i for i, d in enumerate(class_dirs)}
    id_to_class = {v: k for k, v in class_map.items()}
    class_names_list = [id_to_class[i] for i in range(len(class_map))]
    num_classes = len(class_map)
    
    experiments = {}
    total_imgs = 0
    for c_name, c_idx in class_map.items():
        sub_dirs = [d for d in (root / c_name).iterdir() if d.is_dir()]
        if sub_dirs:
            for m_dir in sub_dirs:
                exp_name = m_dir.name
                if exp_name not in experiments: experiments[exp_name] = {"paths": [], "labels": []}
                for img_path in m_dir.glob("*.png"):
                    experiments[exp_name]["paths"].append(str(img_path))
                    experiments[exp_name]["labels"].append(c_idx)
                    total_imgs += 1
        else:
            default_exp = "default"
            if default_exp not in experiments: experiments[default_exp] = {"paths": [], "labels": []}
            for img_path in (root / c_name).glob("*.png"):
                experiments[default_exp]["paths"].append(str(img_path))
                experiments[default_exp]["labels"].append(c_idx)
                total_imgs += 1

    print(f" [Data] Syn Images: {total_imgs}, Gens: {list(experiments.keys())}")
    if total_imgs == 0: return

    print(" [Data] Loading Real Test Data (for evaluation)...")
    if args.dataset_type == "imagenet":
        hf_name = args.hf_dataset if args.hf_dataset else "imagenet-1k"
        hf_split_test = args.hf_split if args.hf_split else "validation"
        hf_split_train = "train"
    elif args.dataset_type == "food101":
        hf_name = args.hf_dataset if args.hf_dataset else "ethz/food101"
        hf_split_test = args.hf_split if args.hf_split else "validation"
        hf_split_train = "train"
    elif args.dataset_type == "cub":
        hf_name = args.hf_dataset if args.hf_dataset else "Donghyun99/CUB-200-2011"
        hf_split_test = args.hf_split if args.hf_split else "test"
        hf_split_train = "train"
    elif args.dataset_type == "imagenet100":
        hf_name = args.hf_dataset if args.hf_dataset else "clane9/imagenet-100"
        hf_split_test = args.hf_split if args.hf_split else "validation"
        hf_split_train = "train"
    
    real_imgs_test, real_lbls_test, _ = load_real_data_hf_generic(hf_name, hf_split_test, class_map, MAX_REAL_TEST, SEED)
    if len(real_imgs_test) == 0: return

    print(f" [Data] Loading Real Train Data (Max {actual_max_train_samples}/class for baselines & Centroids)...")
    tmp_train_imgs, tmp_train_lbls, _ = load_real_data_hf_generic(hf_name, hf_split_train, class_map, actual_max_train_samples, SEED)
    
    from collections import defaultdict
    class_dict = defaultdict(list)
    for img, lbl in zip(tmp_train_imgs, tmp_train_lbls):
        class_dict[lbl].append(img)
        
    baseline_data_dict = {}
    for shots in baseline_shots_list:
        imgs_list = []
        lbls_list = []
        for lbl, img_list in class_dict.items():
            np.random.seed(SEED + int(lbl))
            num_samples = min(shots, len(img_list))
            selected_indices = np.random.choice(len(img_list), num_samples, replace=False)
            for idx in selected_indices:
                imgs_list.append(img_list[idx])
                lbls_list.append(lbl)
        baseline_data_dict[shots] = {"imgs": imgs_list, "lbls": lbls_list}

    target_evaluators = args.evaluators if args.evaluators else AVAILABLE_EVAL_MODELS.keys()
    
    summary_results = []
    detailed_results = []
    
    for model_name in target_evaluators:
        if model_name not in AVAILABLE_EVAL_MODELS: continue
        print(f"\n >> Evaluator: {model_name} ...")
        
        try:
            extractor = FeatureExtractor(AVAILABLE_EVAL_MODELS[model_name], pretrained=True)
            transform = extractor.get_transform(augment=False, mode="eval")
            
            # Real Test Data Features
            real_loader_test = DataLoader(ImageDataset(real_imgs_test, real_lbls_test, transform), 
                                     batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
            X_real_test, y_real_test = extractor.extract_features(real_loader_test)
            
            # 実画像(Train)の重心
            print("      -> Calculating Real Centroids...")
            real_loader_train_all = DataLoader(ImageDataset(tmp_train_imgs, tmp_train_lbls, transform), 
                                     batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
            X_real_train_all, y_real_train_all = extractor.extract_features(real_loader_train_all)
            real_centroids = calculate_centroids(X_real_train_all, y_real_train_all)
            
            # --- Baseline Eval ---
            min_shots = min(baseline_shots_list)
            X_real_train_min_base, y_real_train_min_base = None, None

            for shots, base_data in baseline_data_dict.items():
                baseline_name = f"Baseline_Real_{shots}shot"
                imgs = base_data["imgs"]
                lbls = base_data["lbls"]
                
                if len(imgs) == 0: continue
                
                real_loader_train_base = DataLoader(ImageDataset(imgs, lbls, transform), 
                                         batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
                X_real_train_base, y_real_train_base = extractor.extract_features(real_loader_train_base)

                if shots == min_shots:
                    X_real_train_min_base = X_real_train_base
                    y_real_train_min_base = y_real_train_base

                print(f"    [Gen: {baseline_name}] Processing {len(imgs)} imgs...")
                
                # Accuracy (Weighted Voting)
                nbrs_acc_base = NearestNeighbors(n_neighbors=args.k_neighbors, metric="cosine").fit(X_real_train_base)
                dist_acc_base, idx_acc_base = nbrs_acc_base.kneighbors(X_real_test)
                # 変更: 重み付き投票を使用
                pred_labels_acc_base = predict_knn_labels(
                    dist_acc_base, idx_acc_base, y_real_train_base, 
                    k=args.k_neighbors, temperature=args.temperature, num_classes=num_classes
                )
                acc_base = (pred_labels_acc_base == y_real_test).mean()
                
                # Consistency (Weighted Voting)
                nbrs_viz_base = NearestNeighbors(n_neighbors=args.k_neighbors, metric="cosine").fit(X_real_test)
                dists_v_base, idxs_v_base = nbrs_viz_base.kneighbors(X_real_train_base)
                # 変更: 重み付き投票を使用
                pred_real_labels_base = predict_knn_labels(
                    dists_v_base, idxs_v_base, y_real_test, 
                    k=args.k_neighbors, temperature=args.temperature, num_classes=num_classes
                )
                cons_base = (np.array(y_real_train_base) == pred_real_labels_base).mean()
                
                intra_dist_base = calculate_intra_class_distance(X_real_train_base, y_real_train_base)
                coverage_base = calculate_nn_coverage(idxs_v_base, y_real_train_base)
                centroid_prox_base = calculate_centroid_proximity(X_real_train_base, y_real_train_base, real_centroids)
                
                print(f"      -> KNN Accuracy    : {acc_base:.4f}")
                print(f"      -> Consistency     : {cons_base:.4f}")

                summary_results.append({
                    "Evaluator": model_name,
                    "Generator": baseline_name,
                    "KNN_Accuracy": acc_base,
                    "Label_Consistency": cons_base,
                    "Intra_Class_Dist": intra_dist_base,
                    "NN_Coverage": coverage_base,
                    "Centroid_Proximity": centroid_prox_base
                })

                for i, (dist_list, idx_list) in enumerate(zip(dists_v_base, idxs_v_base)):
                    nearest_real_idx = idx_list[0]
                    nearest_dist = float(dist_list[0])
                    syn_label = y_real_train_base[i]
                    pred_label = pred_real_labels_base[i]
                    
                    detailed_results.append({
                        "Evaluator": model_name,
                        "Generator": baseline_name,
                        "Syn_Image_Path": f"RealTrain_{shots}shot_Class{syn_label}_{i}", 
                        "Syn_Label_ID": syn_label,
                        "Syn_Class_Name": class_names_list[syn_label],
                        "Pred_Real_Class_Name": class_names_list[pred_label],
                        "Nearest_Real_Dist": nearest_dist,
                        "Is_Consistent": (syn_label == pred_label)
                    })

            # --- Syn Eval ---
            X_syn_dict = {}
            y_syn_dict = {}
            
            for gen_name, data in experiments.items():
                syn_paths = data["paths"]
                syn_lbls_list = data["labels"]
                if not syn_paths: continue
                
                print(f"    [Gen: {gen_name}] Processing {len(syn_paths)} imgs...")
                syn_loader = DataLoader(ImageDataset(syn_paths, syn_lbls_list, transform), 
                                        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
                X_syn, y_syn = extractor.extract_features(syn_loader)
                
                X_syn_dict[gen_name] = X_syn
                y_syn_dict[gen_name] = y_syn
                
                # Accuracy (Weighted Voting)
                nbrs_acc = NearestNeighbors(n_neighbors=args.k_neighbors, metric="cosine").fit(X_syn)
                dist_acc, idx_acc = nbrs_acc.kneighbors(X_real_test)
                # 変更: 重み付き投票
                pred_labels_acc = predict_knn_labels(
                    dist_acc, idx_acc, y_syn, 
                    k=args.k_neighbors, temperature=args.temperature, num_classes=num_classes
                )
                acc = (pred_labels_acc == y_real_test).mean()

                # Consistency (Weighted Voting)
                nbrs_viz = NearestNeighbors(n_neighbors=args.k_neighbors, metric="cosine").fit(X_real_test)
                dists_v, idxs_v = nbrs_viz.kneighbors(X_syn)
                # 変更: 重み付き投票
                pred_real_labels = predict_knn_labels(
                    dists_v, idxs_v, y_real_test, 
                    k=args.k_neighbors, temperature=args.temperature, num_classes=num_classes
                )
                cons = (np.array(syn_lbls_list) == pred_real_labels).mean()
                
                intra_dist = calculate_intra_class_distance(X_syn, y_syn)
                coverage = calculate_nn_coverage(idxs_v, y_syn)
                centroid_prox = calculate_centroid_proximity(X_syn, y_syn, real_centroids)
                
                print(f"      -> KNN Accuracy    : {acc:.4f}")
                print(f"      -> Consistency     : {cons:.4f}")

                for i, (dist_list, idx_list) in enumerate(zip(dists_v, idxs_v)):
                    detailed_results.append({
                        "Evaluator": model_name,
                        "Generator": gen_name,
                        "Syn_Image_Path": syn_paths[i],
                        "Syn_Label_ID": syn_lbls_list[i],
                        "Syn_Class_Name": class_names_list[syn_lbls_list[i]],
                        "Pred_Real_Class_Name": class_names_list[pred_real_labels[i]],
                        "Nearest_Real_Dist": float(dist_list[0]),
                        "Is_Consistent": (syn_lbls_list[i] == pred_real_labels[i])
                    })
                
                summary_results.append({
                    "Evaluator": model_name,
                    "Generator": gen_name,
                    "KNN_Accuracy": acc,
                    "Label_Consistency": cons,
                    "Intra_Class_Dist": intra_dist,
                    "NN_Coverage": coverage,
                    "Centroid_Proximity": centroid_prox
                })
                
                if args.save_analysis:
                    save_neighbor_viz_split(syn_paths, real_imgs_test, idxs_v, dists_v, class_names_list, syn_lbls_list, real_lbls_test, 
                                            args.output_dir, model_name, gen_name)

            if args.umap and len(X_syn_dict) > 0 and X_real_train_min_base is not None:
                create_umap_visualization(X_real_test, y_real_test, X_real_train_min_base, y_real_train_min_base, X_syn_dict, y_syn_dict, class_names_list, args.output_dir, model_name, args.dataset_type, min_shots)

            del extractor, X_real_test, X_real_train_all
            torch.cuda.empty_cache()
            
        except Exception as e:
            import traceback
            traceback.print_exc()

    if summary_results:
        df_sum = pd.DataFrame(summary_results).sort_values(by=["Evaluator", "Generator"])
        df_sum.to_csv(os.path.join(args.output_dir, f"knn_summary_k{args.k_neighbors}.csv"), index=False)
        
    if detailed_results:
        df_det = pd.DataFrame(detailed_results)
        cols = ["Evaluator", "Generator", "Syn_Class_Name", "Pred_Real_Class_Name", "Is_Consistent", "Nearest_Real_Dist", "Syn_Image_Path"]
        final_cols = [c for c in cols if c in df_det.columns] + [c for c in df_det.columns if c not in cols]
        df_det[final_cols].to_csv(os.path.join(args.output_dir, f"knn_details_k{args.k_neighbors}.csv"), index=False)

if __name__ == "__main__":
    main()