# ファイル名: evaluate_linear2.py
# 内容: エントリーポイント. 実験のセットアップ, ループ実行, 結果の保存を担当

import os
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib
from pathlib import Path
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import torch

from evaluate_linear2_config import (
    TRAIN_CONFIGS, AVAILABLE_EVAL_MODELS, DEVICE, SEED,
    BATCH_SIZE, NUM_WORKERS, MAX_REAL_TRAIN, MAX_REAL_TEST
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
    parser = argparse.ArgumentParser(description="Evaluate synthetic datasets with check-logs for validity.")
    parser.add_argument("dataset_dir", type=str)
    parser.add_argument("--output_dir", type=str, default="evaluation_results")
    parser.add_argument("--mode", type=str, default="linear_lbfgs", choices=list(TRAIN_CONFIGS.keys()))
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--dataset_type", type=str, default="imagenet", choices=["imagenet", "cub", "food101"])
    parser.add_argument("--real_data_dir", type=str, default=None)

    parser.add_argument("--cub_hf_dataset", type=str, default="Donghyun99/CUB-200-2011")
    parser.add_argument("--cub_hf_train_split", type=str, default="train")
    parser.add_argument("--cub_hf_test_split", type=str, default="test")
    parser.add_argument("--cub_source", type=str, default="hf")

    parser.add_argument("--food_hf_dataset", type=str, default="ethz/food101")
    parser.add_argument("--food_hf_train_split", type=str, default="train")
    parser.add_argument("--food_hf_test_split", type=str, default="validation")

    parser.add_argument("--imagenet_hf_dataset", type=str, default="imagenet-1k")
    parser.add_argument("--imagenet_hf_train_split", type=str, default="train")
    parser.add_argument("--imagenet_hf_test_split", type=str, default="validation")

    parser.add_argument("--evaluators", type=str, nargs="+", default=None)
    parser.add_argument("--no_tsne", action="store_true")
    parser.add_argument("--config", type=str, default=None)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 【追加修正】Logフォルダのパスを作成し、実際にフォルダを作る
    log_save_dir = os.path.join(args.output_dir, "Log")
    os.makedirs(log_save_dir, exist_ok=True)

    set_seed(SEED)

    mode = args.mode
    config = dict(TRAIN_CONFIGS[mode])

    print("\n" + "=" * 60)
    print(" EXPERIMENT CHECK LOG")
    print("=" * 60)
    print(f" [Settings] Mode      : {mode}")
    print(f" [Settings] Augment   : {args.augment}")
    print(f" [Settings] Seed      : {SEED}")
    print(f" [Settings] Device    : {DEVICE}")
    print(f" [Settings] Config    : {config}")
    print(f" [Data]     Target    : {args.dataset_type} (HF)")
    print(f" [Output]   Log Dir   : {log_save_dir}") # 確認用表示
    print("-" * 60)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    root = Path(args.dataset_dir)
    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    class_map = {d.name: i for i, d in enumerate(class_dirs)}
    id_to_class_name = {v: k for k, v in class_map.items()}
    num_classes = len(class_map)

    print(f" [Data]     Syn Classes : {num_classes} classes detected in {root}")

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
        if num_classes > 0:
            # 切り上げ計算 (前回の修正を維持)
            raw_shots = len(first_exp["paths"]) / num_classes
            syn_shots_per_class = max(1, math.ceil(raw_shots))
            
        print(f" [Data]     Syn Shots   : ~{syn_shots_per_class} img/class (Calculated from {len(first_exp['paths'])} imgs)")

    print("\n [Step] Loading Real Data (HuggingFace)...")
    real_train_imgs, real_train_lbls = [], []
    real_test_imgs, real_test_lbls, matched_classes = [], [], 0

    if args.dataset_type == "imagenet":
        real_test_imgs, real_test_lbls, matched_classes = load_real_data_hf_generic(
            args.imagenet_hf_dataset, args.imagenet_hf_test_split, class_map, MAX_REAL_TEST, SEED
        )
        real_train_imgs, real_train_lbls, _ = load_real_data_hf_generic(
            args.imagenet_hf_dataset, args.imagenet_hf_train_split, class_map, MAX_REAL_TRAIN, SEED
        )

    elif args.dataset_type == "food101":
        real_test_imgs, real_test_lbls, matched_classes = load_real_data_hf_generic(
            args.food_hf_dataset, args.food_hf_test_split, class_map, MAX_REAL_TEST, SEED
        )
        real_train_imgs, real_train_lbls, _ = load_real_data_hf_generic(
            args.food_hf_dataset, args.food_hf_train_split, class_map, MAX_REAL_TRAIN, SEED
        )

    else:
        real_test_imgs, real_test_lbls, matched_classes = load_real_data_hf_generic(
            args.cub_hf_dataset, args.cub_hf_test_split, class_map, MAX_REAL_TEST, SEED
        )
        real_train_imgs, real_train_lbls, _ = load_real_data_hf_generic(
            args.cub_hf_dataset, args.cub_hf_train_split, class_map, MAX_REAL_TRAIN, SEED
        )

    print(f" [Check]    Matched Classes: {matched_classes} / {num_classes}")
    if matched_classes < num_classes:
        print(" [Warning]  Partial class match. Evaluation may be biased to matched subset.")

    if not real_test_imgs:
        print("Fatal: No real data loaded.")
        return

    real_fewshot_imgs, real_fewshot_lbls = get_fewshot_subset(real_train_imgs, real_train_lbls, syn_shots_per_class)
    print(f" [Data]     Real FewShot: {len(real_fewshot_imgs)} images (Target: {num_classes * syn_shots_per_class})")

    eval_models = AVAILABLE_EVAL_MODELS
    if args.evaluators:
        eval_models = {k: v for k, v in eval_models.items() if k in args.evaluators}

    final_results = []
    acc_store = {model_name: {} for model_name in eval_models.keys()}
    cm_store = {model_name: {} for model_name in eval_models.keys()}

    for eval_name, model_id in eval_models.items():
        print(f"\n[{eval_name}] Init...")
        try:
            extractor = FeatureExtractor(model_id)
        except Exception as e:
            print(f"Skip {eval_name}: {e}")
            continue

        train_transform = extractor.get_transform(augment=args.augment)
        test_transform = extractor.get_transform(augment=False)

        test_ds = ImageDataset(real_test_imgs, real_test_lbls, test_transform)
        test_loader = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True,
        )

        X_test, y_test = None, None
        if mode == "linear_lbfgs":
            X_test, y_test = extractor.extract_features(test_loader)
            if X_test.size == 0:
                continue

        def run_exp(name, imgs, lbls):
            ds = ImageDataset(imgs, lbls, train_transform)
            bs = BATCH_SIZE if mode == "linear_torch" else int(config.get("batch_size", 16))
            ldr = DataLoader(
                ds,
                batch_size=bs,
                shuffle=True,
                num_workers=NUM_WORKERS,
                pin_memory=True,
                persistent_workers=True,
            )

            if mode == "linear_lbfgs":
                acc, t_acc, t_p, tot_p, yt, yp = train_sklearn_lbfgs(extractor, ldr, X_test, y_test, config)
            else:
                run_tag = f"{name}__{eval_name}"
                # 【修正】log_dir 引数に、作成した log_save_dir を渡す
                acc, t_acc, t_p, tot_p, lora_t, yt, yp = train_pytorch_pipeline(
                    extractor, ldr, test_loader, num_classes, mode, config,
                    log_dir=log_save_dir,
                    run_name=run_tag
                )
                if mode == "partial_ft" and t_p < 100000:
                    print(f"    [Check] {name}: Trainable params={t_p}. Too low for partial_ft. fallback is possible.")

            final_results.append({
                "Generator": name,
                "Evaluator": eval_name,
                "Mode": mode,
                "Accuracy": acc,
                "TrainAcc": t_acc,
                "Samples": len(imgs),
                "Matched": matched_classes,
                "Trainable": t_p,
            })
            print(f"  > {name}: {acc:.4f} (Train: {t_acc:.4f})")

            key_name = name
            unique_classes = sorted(np.unique(yt))
            acc_map = {}
            for cls_id in unique_classes:
                mask = (yt == cls_id)
                cls_acc = (yp[mask] == yt[mask]).mean() if mask.sum() > 0 else 0.0
                acc_map[cls_id] = float(cls_acc)
            acc_vector = [acc_map.get(cid, 0.0) for cid in sorted(list(id_to_class_name.keys()))]
            acc_store[eval_name][key_name] = acc_vector

            all_labels = sorted(list(id_to_class_name.keys()))
            cm = confusion_matrix(yt, yp, labels=all_labels)
            cm_store[eval_name][key_name] = cm

        if real_train_imgs:
            run_exp("Real_Baseline", real_train_imgs, real_train_lbls)
        if real_fewshot_imgs:
            run_exp("Real_FewShot", real_fewshot_imgs, real_fewshot_lbls)
        for exp_name, data in experiments.items():
            run_exp(exp_name, data["paths"], data["labels"])

        del extractor
        torch.cuda.empty_cache()

    if final_results:
        df = pd.DataFrame(final_results)
        csv_path = os.path.join(args.output_dir, f"results_fast_{mode}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")
        calculate_self_preference(final_results, args.output_dir)


    print("\n--- Calculating Correlation & Similarity ---")
    
    # 実行された評価モデルのリストを取得 (eval_models のキー)
    targets = list(eval_models.keys())

    for target in targets:
        # データが十分にない場合はスキップ
        if target not in acc_store or len(acc_store[target]) == 0:
            continue

        print(f" >> Analyzing correlations based on target: {target} ...")
        
        calculate_complex_correlations(
            acc_store, experiments, args.output_dir, target_evaluator=target
        )
        calculate_confusion_matrix_similarity(
            cm_store, experiments, args.output_dir, target_evaluator=target
        )

if __name__ == "__main__":
    main()