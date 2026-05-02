# ファイル名: evaluate_linearKD.py
# 内容: KD対応の合成データセット評価メインスクリプト (機能完全復活版)

import os
import argparse
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

from evaluate_linearKD_config import TRAIN_CONFIGS, AVAILABLE_EVAL_MODELS, BATCH_SIZE, NUM_WORKERS
from evaluate_linearKD_data import set_seed, KDImageDataset, get_fewshot_subset, load_real_data_hf_generic
from evaluate_linearKD_models import FeatureExtractor
from evaluate_linearKD_engine import train_pytorch_pipeline_kd

def main():
    parser = argparse.ArgumentParser(description="Evaluate synthetic datasets with Logit-based Knowledge Distillation.")
    
    # --- 基本設定 ---
    parser.add_argument("dataset_dir", type=str, help="Path to synthetic datasets.")
    parser.add_argument("--output_dir", type=str, default="evaluation_results_kd")
    parser.add_argument("--mode", type=str, default="scratch", choices=list(TRAIN_CONFIGS.keys()))
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--no_tsne", action="store_true") 
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--evaluators", type=str, nargs="+", default=["ResNet50", "MAE", "OpenCLIP_RN50"])
    
    # --- データ拡張戦略 ---
    parser.add_argument("--aug_strategy", type=str, default="on_the_fly", choices=["default", "none", "precompute", "on_the_fly"])
    parser.add_argument("--aug_expansion", type=int, default=20)
    parser.add_argument("--num_trials", type=int, default=1)

    # --- データセットと枚数制御 ---
    parser.add_argument("--dataset_type", type=str, default="imagenet", choices=["imagenet", "cub", "food101"])
    parser.add_argument("--real_counts", type=int, nargs="+", default=[50])
    parser.add_argument("--max_real_test", type=int, default=50)
    parser.add_argument("--syn_counts", type=int, nargs="+", default=[])
    
    # --- Mix設定 ---
    parser.add_argument("--mix_sources", type=str, nargs="+", default=[])
    parser.add_argument("--mix_json", type=str, default=None)

    # --- HF Dataset Configs ---
    parser.add_argument("--cub_hf_dataset", type=str, default="Donghyun99/CUB-200-2011")
    parser.add_argument("--cub_hf_test_split", type=str, default="test")
    parser.add_argument("--food_hf_dataset", type=str, default="ethz/food101")
    parser.add_argument("--food_hf_test_split", type=str, default="validation")
    parser.add_argument("--imagenet_hf_dataset", type=str, default="imagenet-1k")
    parser.add_argument("--imagenet_hf_test_split", type=str, default="validation")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(42)

    # --- Mix JSON Parsing ---
    mix_strategies = {}
    if args.mix_json:
        try:
            mix_strategies = json.loads(args.mix_json) if not os.path.exists(args.mix_json) else json.load(open(args.mix_json))
        except Exception as e:
            print(f"[Error] Failed to parse mix_json: {e}")
    if args.mix_sources:
        mix_strategies["Hybrid_Custom"] = args.mix_sources

    target_real_counts = sorted(list(set(args.real_counts)))
    target_syn_counts = sorted(list(set(args.syn_counts)))

    # --- Directory Parsing ---
    root = Path(args.dataset_dir)
    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    class_map = {d.name: i for i, d in enumerate(class_dirs)}
    num_classes = len(class_map)
    print(f"Detected {num_classes} classes in {args.dataset_dir}")

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
                        
    # Filter Mix Strategies to valid sources
    for mix_name, sources in list(mix_strategies.items()):
        mix_strategies[mix_name] = [s for s in sources if s in experiments]

    # --- Load Real Test Data ---
    print("\n[Step] Loading Real Test Data (HuggingFace)...")
    if args.dataset_type == "imagenet":
        hf_name, te_split = args.imagenet_hf_dataset, args.imagenet_hf_test_split
    elif args.dataset_type == "food101":
        hf_name, te_split = args.food_hf_dataset, args.food_hf_test_split
    else:
        hf_name, te_split = args.cub_hf_dataset, args.cub_hf_test_split

    real_test_imgs, real_test_lbls, matched_classes = load_real_data_hf_generic(
        hf_name, te_split, class_map, args.max_real_test, seed=42
    )
    
    if not real_test_imgs:
        print("[Warning] No real test data loaded. Evaluation will be skipped or fail.")
        return

    # --- Evaluation Loop ---
    final_results = []
    
    for eval_name in args.evaluators:
        if eval_name not in AVAILABLE_EVAL_MODELS:
            print(f"[Skip] {eval_name} not in AVAILABLE_EVAL_MODELS.")
            continue
            
        print(f"\n========== Evaluator: {eval_name} ==========")
        extractor = FeatureExtractor(AVAILABLE_EVAL_MODELS[eval_name], pretrained=(args.mode != "scratch"))
        
        test_ds = KDImageDataset(real_test_imgs, real_test_lbls, extractor.get_transform(augment=False, mode=args.mode))
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        def run_exp(name, imgs, lbls):
            if not imgs: return
            trial_accs, trial_t_accs = [], []
            
            for t in range(args.num_trials):
                set_seed(42 + t)
                use_aug = (args.aug_strategy in ["on_the_fly", "precompute"]) or args.augment
                train_ds = KDImageDataset(imgs, lbls, extractor.get_transform(augment=use_aug, mode=args.mode))
                bs = min(int(TRAIN_CONFIGS[args.mode].get("batch_size", 16)), len(imgs))
                train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=NUM_WORKERS)

                acc, t_acc, _, _ = train_pytorch_pipeline_kd(
                    extractor, train_loader, test_loader, num_classes, args.mode, TRAIN_CONFIGS[args.mode]
                )
                trial_accs.append(acc)
                trial_t_accs.append(t_acc)
                
            mean_acc, std_acc = float(np.mean(trial_accs)), float(np.std(trial_accs))
            mean_t_acc = float(np.mean(trial_t_accs))
            print(f"  > {name:<30}: {mean_acc:.4f} ± {std_acc:.4f} (n={len(imgs)})")
            
            final_results.append({
                "Evaluator": eval_name, "Generator": name, "Mode": args.mode, 
                "Samples": len(imgs), "Accuracy": mean_acc, "Acc_Std": std_acc, "TrainAcc": mean_t_acc
            })

        # 1. Synthetic (Individual)
        for exp_name, data in experiments.items():
            if target_syn_counts:
                for count in target_syn_counts:
                    s_imgs, s_lbls = get_fewshot_subset(data["paths"], data["labels"], count)
                    run_exp(f"{exp_name}_{count}shot", s_imgs, s_lbls)
            else:
                run_exp(exp_name, data["paths"], data["labels"])

        # 2. Synthetic (Mix)
        if mix_strategies and target_syn_counts:
            for mix_name, source_list in mix_strategies.items():
                if not source_list: continue
                for count in target_syn_counts:
                    mix_imgs, mix_lbls = [], []
                    for src in source_list:
                        s_imgs, s_lbls = get_fewshot_subset(experiments[src]["paths"], experiments[src]["labels"], count)
                        mix_imgs.extend(s_imgs)
                        mix_lbls.extend(s_lbls)
                    run_exp(f"{mix_name}_{count}shot_each", mix_imgs, mix_lbls)

    # Save Results
    if final_results:
        df = pd.DataFrame(final_results)
        csv_path = os.path.join(args.output_dir, f"kd_results_multi_{args.mode}.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nDone! Saved to: {csv_path}")

if __name__ == "__main__":
    main()