# ファイル名: evaluate_linear2.py
# 内容: Augmentation戦略(none/precompute/on_the_fly), 複数回試行, Multi-Mix対応, 空Mixスキップ修正済み

import os
import math
import argparse
import re
import json
import numpy as np
import pandas as pd
import matplotlib
from pathlib import Path
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import torch

# 設定ファイルやモジュールのインポート
from evaluate_linear2_config import (
    TRAIN_CONFIGS, AVAILABLE_EVAL_MODELS, DEVICE, SEED,
    BATCH_SIZE, NUM_WORKERS
)

from evaluate_linear2_data import (
    set_seed, load_real_data_hf_generic, get_fewshot_subset, ImageDataset
)

from evaluate_linear2_models import FeatureExtractor
from evaluate_linear2_engine import train_sklearn_lbfgs, train_pytorch_pipeline

from evaluate_linear2_metrics import (
    calculate_self_preference, calculate_complex_correlations, calculate_confusion_matrix_similarity
)

def main():
    parser = argparse.ArgumentParser(description="Evaluate synthetic datasets with advanced augmentation strategies.")
    
    # --- 基本設定 ---
    parser.add_argument("dataset_dir", type=str, help="Path to synthetic datasets.")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Directory to save the results.")
    parser.add_argument("--mode", type=str, default="linear_lbfgs", choices=list(TRAIN_CONFIGS.keys()))
    parser.add_argument("--augment", action="store_true", help="Enable augmentation (default behavior depends on strategy).")
    parser.add_argument("--no_tsne", action="store_true") 
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--evaluators", type=str, nargs="+", default=None)

    # --- [追加] データ拡張戦略 ---
    parser.add_argument("--aug_strategy", type=str, default="default", 
                        choices=["default", "none", "precompute", "on_the_fly"],
                        help="Augmentation strategy. 'precompute' caches N variations. 'on_the_fly' is slow but infinite.")
    parser.add_argument("--aug_expansion", type=int, default=20, 
                        help="Expansion factor for pre-computed augmentation (e.g. 20 means 20 variations per image).")

    # --- [追加] 試行回数 (平均・分散の計算用) ---
    parser.add_argument("--num_trials", type=int, default=1, 
                        help="Number of trials to calculate Mean/Std (default: 1).")

    # --- データセット設定 ---
    parser.add_argument("--dataset_type", type=str, default="imagenet", choices=["imagenet", "cub", "food101", "imagenet100"])
    parser.add_argument("--real_data_dir", type=str, default=None)

    # --- 実画像の枚数制御 ---
    parser.add_argument("--real_counts", type=int, nargs="+", default=[50, 500, 1300])
    parser.add_argument("--max_real_test", type=int, default=50)

    # --- 合成画像の段階的検証設定 ---
    parser.add_argument("--syn_counts", type=int, nargs="+", default=[])
    
    # --- Mix設定 (JSON対応) ---
    parser.add_argument("--mix_sources", type=str, nargs="+", default=[], help="Legacy: List of folders.")
    parser.add_argument("--mix_json", type=str, default=None, help="JSON string or file path defining multiple mix strategies.")

    # --- HuggingFace Dataset Configs ---
    parser.add_argument("--cub_hf_dataset", type=str, default="Donghyun99/CUB-200-2011")
    parser.add_argument("--cub_hf_train_split", type=str, default="train")
    parser.add_argument("--cub_hf_test_split", type=str, default="test")
    parser.add_argument("--cub_source", type=str, default="hf")

    parser.add_argument("--food_hf_dataset", type=str, default="ethz/food101")
    parser.add_argument("--food_hf_train_split", type=str, default="train")
    parser.add_argument("--food_hf_test_split", type=str, default="validation")

    parser.add_argument("--imagenet100_hf_dataset", type=str, default="clane9/imagenet-100")
    parser.add_argument("--imagenet_hf_train_split", type=str, default="train")
    parser.add_argument("--imagenet_hf_test_split", type=str, default="validation")

    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    log_save_dir = os.path.join(args.output_dir, "Log")
    os.makedirs(log_save_dir, exist_ok=True)

    set_seed(SEED)

    mode = args.mode
    config = dict(TRAIN_CONFIGS[mode])
    
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            user_cfg = json.load(f)
            config.update(user_cfg)

    # Mix設定の構築
    mix_strategies = {}
    if args.mix_json:
        if os.path.exists(args.mix_json):
            with open(args.mix_json, 'r') as f:
                mix_strategies = json.load(f)
        else:
            try:
                mix_strategies = json.loads(args.mix_json)
            except json.JSONDecodeError:
                print("[Error] Failed to parse --mix_json string.")
    
    if args.mix_sources:
        short_names = [s.replace("Only_", "") for s in args.mix_sources]
        name = "Hybrid_" + "_".join(short_names)
        mix_strategies[name] = args.mix_sources

    if not args.real_counts: args.real_counts = [50]
    max_train_needed = max(args.real_counts)
    target_real_counts = sorted(list(set(args.real_counts)))
    target_syn_counts = sorted(list(set(args.syn_counts)))

    print("\n" + "=" * 60)
    print(" EXPERIMENT: AUGMENTATION STRATEGY & MULTI-MIX")
    print("=" * 60)
    print(f" [Settings] Mode       : {mode}")
    print(f" [Settings] Trials     : {args.num_trials} (Mean/Std)")
    print(f" [Settings] Augment    : {args.augment} (Flag)")
    print(f" [Settings] Strategy   : {args.aug_strategy}")
    if args.aug_strategy == "precompute" or (args.aug_strategy=="default" and args.augment):
        print(f" [Settings] Expansion  : {args.aug_expansion}x (if precompute used)")
    print(f" [Data]     Target     : {args.dataset_type}")
    print(f" [Real]     Counts     : {target_real_counts}")
    print(f" [Syn]      Counts     : {target_syn_counts if target_syn_counts else 'Full'}")
    print(f" [Mix]      Strategies : {list(mix_strategies.keys()) if mix_strategies else 'None'}")
    print("-" * 60)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    root = Path(args.dataset_dir)
    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    class_map = {d.name: i for i, d in enumerate(class_dirs)}
    id_to_class_name = {v: k for k, v in class_map.items()}
    num_classes = len(class_map)

    print(f" [Data]     Syn Classes : {num_classes} classes detected")

    experiments = {}
    for c_name, c_idx in class_map.items():
        for m_dir in (root / c_name).iterdir():
            if m_dir.is_dir():
                exp_name = m_dir.name
                experiments.setdefault(exp_name, {"paths": [], "labels": []})
                for img in m_dir.glob("*.png"):
                    experiments[exp_name]["paths"].append(str(img))
                    experiments[exp_name]["labels"].append(c_idx)
    
    for mix_name, sources in mix_strategies.items():
        mix_strategies[mix_name] = [s for s in sources if s in experiments]

    print("\n [Step] Loading Real Data (HuggingFace)...")
    if args.dataset_type == "imagenet":
        hf_name, tr_split, te_split = args.imagenet_hf_dataset, args.imagenet_hf_train_split, args.imagenet_hf_test_split
    elif args.dataset_type == "imagenet100":
        hf_name, tr_split, te_split = args.imagenet100_hf_dataset, args.imagenet_hf_train_split, args.imagenet_hf_test_split
    elif args.dataset_type == "food101":
        hf_name, tr_split, te_split = args.food_hf_dataset, args.food_hf_train_split, args.food_hf_test_split
    else:
        hf_name, tr_split, te_split = args.cub_hf_dataset, args.cub_hf_train_split, args.cub_hf_test_split

    real_test_imgs, real_test_lbls, matched_classes = load_real_data_hf_generic(
        hf_name, te_split, class_map, args.max_real_test, SEED
    )
    all_real_train_imgs, all_real_train_lbls, _ = load_real_data_hf_generic(
        hf_name, tr_split, class_map, max_train_needed, SEED
    )

    if not real_test_imgs:
        print("Fatal: No real data loaded.")
        return

    eval_models = AVAILABLE_EVAL_MODELS
    if args.evaluators:
        eval_models = {k: v for k, v in eval_models.items() if k in args.evaluators}

    final_results = []
    acc_store = {model_name: {} for model_name in eval_models.keys()}
    cm_store = {model_name: {} for model_name in eval_models.keys()}

    # --- 評価ループ ---
    for eval_name, model_id in eval_models.items():
        print(f"\n[{eval_name}] Init...")
        try:
            is_pretrained = (mode != "scratch")
            extractor = FeatureExtractor(model_id, pretrained=is_pretrained)
        except Exception as e:
            print(f"Skip {eval_name}: {e}")
            continue

        # --- [修正] Augmentation戦略の決定 ---
        effective_strategy = args.aug_strategy
        if effective_strategy == "default":
            effective_strategy = "on_the_fly" if args.augment else "none"
        
        # Loader上でAugmentationを有効にするかどうか
        use_aug_in_transform = (effective_strategy in ["on_the_fly", "precompute"])

        train_transform = extractor.get_transform(augment=use_aug_in_transform, mode=mode)
        test_transform = extractor.get_transform(augment=False, mode=mode)

        test_ds = ImageDataset(real_test_imgs, real_test_lbls, test_transform)
        test_loader = DataLoader(
            test_ds, batch_size=BATCH_SIZE, shuffle=False, 
            num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
        )

        X_test, y_test = None, None
        if mode == "linear_lbfgs":
            X_test, y_test = extractor.extract_features(test_loader)
            if X_test.size == 0: continue

        def run_exp(name, imgs, lbls):
            trial_accs = []
            trial_t_accs = []
            last_yt, last_yp = [], []
            last_t_p = 0

            for t in range(args.num_trials):
                current_seed = SEED + t
                set_seed(current_seed)

                ds = ImageDataset(imgs, lbls, train_transform)
                bs = BATCH_SIZE if mode == "linear_torch" else int(config.get("batch_size", 16))
                ldr = DataLoader(
                    ds, batch_size=bs, shuffle=True, 
                    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
                )

                if mode == "linear_lbfgs":
                    acc, t_acc, t_p, tot_p, yt, yp = train_sklearn_lbfgs(extractor, ldr, X_test, y_test, config)
                else:
                    run_tag = f"{name}__{eval_name}_trial{t}"
                    acc, t_acc, t_p, tot_p, _, yt, yp = train_pytorch_pipeline(
                        extractor, ldr, test_loader, num_classes, mode, config,
                        log_dir=log_save_dir, run_name=run_tag,
                        aug_strategy=effective_strategy,
                        aug_expansion=args.aug_expansion
                    )
                
                trial_accs.append(acc)
                trial_t_accs.append(t_acc)
                last_yt, last_yp = yt, yp
                last_t_p = t_p
            
            mean_acc = float(np.mean(trial_accs))
            std_acc = float(np.std(trial_accs))
            max_acc = float(np.max(trial_accs))
            min_acc = float(np.min(trial_accs))
            mean_t_acc = float(np.mean(trial_t_accs))

            final_results.append({
                "Generator": name,
                "Evaluator": eval_name,
                "Mode": mode,
                "Accuracy": mean_acc,
                "Acc_Std": std_acc,
                "Acc_Min": min_acc,
                "Acc_Max": max_acc,
                "TrainAcc": mean_t_acc,
                "Samples": len(imgs),
                "Matched": matched_classes,
                "Trainable": last_t_p,
                "Trials": args.num_trials,
                "Aug_Strategy": effective_strategy
            })
            
            std_str = f"±{std_acc:.3f}" if args.num_trials > 1 else ""
            print(f"  > {name:<30}: {mean_acc:.4f} {std_str} (n={len(imgs)})")

            unique_classes = sorted(np.unique(last_yt))
            acc_map = {}
            for cls_id in unique_classes:
                mask = (last_yt == cls_id)
                cls_acc = (last_yp[mask] == last_yt[mask]).mean() if mask.sum() > 0 else 0.0
                acc_map[cls_id] = float(cls_acc)
            acc_vector = [acc_map.get(cid, 0.0) for cid in sorted(list(id_to_class_name.keys()))]
            acc_store[eval_name][name] = acc_vector

            all_labels = sorted(list(id_to_class_name.keys()))
            cm = confusion_matrix(last_yt, last_yp, labels=all_labels)
            cm_store[eval_name][name] = cm

        # 1. Real Baseline
        if all_real_train_imgs:
            for count in target_real_counts:
                sub_imgs, sub_lbls = get_fewshot_subset(all_real_train_imgs, all_real_train_lbls, count)
                run_exp(f"Real_Ref_{count}", sub_imgs, sub_lbls)
            
            max_cnt = max(target_real_counts)
            if f"Real_Ref_{max_cnt}" in acc_store[eval_name]:
                 acc_store[eval_name]["Real_Baseline"] = acc_store[eval_name][f"Real_Ref_{max_cnt}"]

        # 2. Synthetic (Individual)
        for exp_name, data in experiments.items():
            if target_syn_counts:
                for count in target_syn_counts:
                    sub_imgs, sub_lbls = get_fewshot_subset(data["paths"], data["labels"], count)
                    run_exp(f"{exp_name}_{count}shot", sub_imgs, sub_lbls)
            else:
                run_exp(exp_name, data["paths"], data["labels"])

        # 3. Synthetic (Mix)
        if mix_strategies and target_syn_counts:
            for mix_name, source_list in mix_strategies.items():
                
                # 【修正1】ソースリストが空（0個）ならスキップ
                if not source_list:
                    print(f"  [Skip] Mix Strategy '{mix_name}' has no valid sources (Sources: 0).")
                    continue

                print(f"  [Mix] Strategy: {mix_name} (Sources: {len(source_list)})")
                
                for count in target_syn_counts:
                    mixed_imgs, mixed_lbls = [], []
                    for source_name in source_list:
                        s_data = experiments[source_name]
                        s_imgs, s_lbls = get_fewshot_subset(s_data["paths"], s_data["labels"], count)
                        mixed_imgs.extend(s_imgs)
                        mixed_lbls.extend(s_lbls)
                    
                    # 【修正2】画像が1枚もない場合はスキップ (エラー回避)
                    if len(mixed_imgs) == 0:
                        print(f"    [Skip] No images generated for {mix_name} ({count}shot).")
                        continue

                    run_exp(f"{mix_name}_{count}shot_each", mixed_imgs, mixed_lbls)

        del extractor
        torch.cuda.empty_cache()

    if final_results:
        df = pd.DataFrame(final_results)
        eval_order = list(eval_models.keys())
        existing_evals = df["Evaluator"].unique().tolist()
        final_eval_order = [e for e in eval_order if e in existing_evals]
        for e in existing_evals:
            if e not in final_eval_order: final_eval_order.append(e)
        df["Evaluator"] = pd.Categorical(df["Evaluator"], categories=final_eval_order, ordered=True)

        existing_gens = df["Generator"].unique().tolist()
        real_gens = [g for g in existing_gens if g.startswith("Real_Ref_")]
        def get_cnt(name):
            m = re.search(r"(\d+)", name)
            return int(m.group(1)) if m else 0
        real_gens.sort(key=get_cnt)
        
        syn_gens = [g for g in existing_gens if not g.startswith("Real_Ref_")]
        desired_syn_order = [
            "Only_V1", "Only_V2", "Only_CLIP", "Only_SigLIP",
            "Hybrid_V1_V2", "Hybrid_CLIP_SigLIP", "Hybrid_V1_CLIP",
            "Hybrid_V1_SigLIP", "Hybrid_V2_CLIP", "Hybrid_V2_SigLIP"
        ]
        def syn_sort_key(name):
            base = name.split("_")[0]
            shot_m = re.search(r"(\d+)shot", name)
            shot = int(shot_m.group(1)) if shot_m else 9999
            idx = 999
            for i, order_name in enumerate(desired_syn_order):
                if name.startswith(order_name):
                    idx = i
                    break
            return (idx, base, shot, name)
        
        syn_gens.sort(key=syn_sort_key)
        final_gen_order = real_gens + syn_gens
        df["Generator"] = pd.Categorical(df["Generator"], categories=final_gen_order, ordered=True)
        df.sort_values(by=["Evaluator", "Generator"], inplace=True)

        csv_path = os.path.join(args.output_dir, f"results_multi_{mode}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")
        
        calculate_self_preference(final_results, args.output_dir)

    print("\n--- Saving Class-wise Accuracy ---")
    class_names = [id_to_class_name[i] for i in sorted(id_to_class_name.keys())]
    for eval_name, gen_data in acc_store.items():
        if not gen_data: continue
        row_list = []
        for gen_name, acc_vec in gen_data.items():
            row = {"Generator": gen_name}
            for i, val in enumerate(acc_vec):
                c_name = class_names[i] if i < len(class_names) else f"Class_{i}"
                row[c_name] = val
            row_list.append(row)
        if not row_list: continue
        df_class_acc = pd.DataFrame(row_list)
        if "Generator" in df_class_acc.columns:
            df_class_acc["Generator"] = pd.Categorical(df_class_acc["Generator"], categories=final_gen_order, ordered=True)
            df_class_acc.sort_values("Generator", inplace=True)
        safe_eval_name = eval_name.replace("/", "_")
        df_class_acc.to_csv(os.path.join(args.output_dir, f"class_accuracy_{safe_eval_name}_{mode}.csv"), index=False)
    
    print("\n--- Calculating Correlation & Similarity ---")
    targets = list(eval_models.keys())
    for target in targets:
        if target not in acc_store or len(acc_store[target]) == 0: continue
        calculate_complex_correlations(acc_store, experiments, args.output_dir, target_evaluator=target)
        calculate_confusion_matrix_similarity(cm_store, experiments, args.output_dir, target_evaluator=target)

if __name__ == "__main__":
    main()