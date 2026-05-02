import sys
import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score

# リポジトリ内のモジュールパス設定
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from augmentation import AugBasic
from models import get_model, get_fc
from data.dataloaders import get_dataset

# ==============================================================================
# 0. Utils
# ==============================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_balanced_subset_indices(dataset, ipc, seed=None):
    """
    データセットからクラスごとにipc枚ずつ抽出したインデックスを返す。
    seedを指定すると、抽出前にインデックスをシャッフルする（ランダムサンプリング）。
    """
    if seed is not None:
        # ローカルなRandomStateを使ってグローバルなseedへの影響を避ける
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState(42) # 固定

    if isinstance(dataset, Subset):
        full_targets = np.array(dataset.dataset.targets)
        subset_indices = dataset.indices
        targets = full_targets[subset_indices]
        original_indices = np.array(subset_indices)
    else:
        targets = np.array(dataset.targets)
        original_indices = np.arange(len(dataset))

    selected_indices = []
    unique_classes = np.unique(targets)
    
    for cls in unique_classes:
        cls_locs = np.where(targets == cls)[0]
        
        # ランダムシャッフル
        rng.shuffle(cls_locs)
        
        if len(cls_locs) > ipc:
            cls_locs = cls_locs[:ipc]
        selected_indices.extend(original_indices[cls_locs])
            
    return selected_indices

# ==============================================================================
# 1. Evaluator Class
# ==============================================================================
class ComprehensiveEvaluator:
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        num_feats: int,
        lr: float = 0.001,
        epochs: int = 1000,
        device: str = "cuda"
    ):
        self.device = device
        self.model = model.to(device)
        self.model.eval() # Backbone frozen
        self.num_classes = num_classes
        self.epochs = epochs
        
        # Headの初期化
        self.fc = get_fc(num_feats, num_classes, False).to(device)
        
        # Optimizerの初期化
        self.optimizer = torch.optim.Adam(self.fc.parameters(), lr=lr, weight_decay=0)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, epochs, eta_min=0)
        self.scaler = GradScaler()
        self.augmentor = AugBasic(crop_res=224).to(device)
        self.normalize = None 

    def set_normalization(self, normalize_fn):
        self.normalize = normalize_fn

    def train_linear_probe(self, train_loader_or_tensor, disable_gpu_aug=False):
        is_tensor_dataset = isinstance(train_loader_or_tensor, (list, tuple))
        
        if is_tensor_dataset:
            syn_images, syn_labels = train_loader_or_tensor
            ds = TensorDataset(syn_images, syn_labels)
            loader = DataLoader(ds, batch_size=min(100, len(syn_images)), shuffle=True)
        else:
            loader = train_loader_or_tensor

        # Quiet training loop
        for epoch in range(self.epochs):
            for x, y in loader:
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                
                with autocast():
                    with torch.no_grad():
                        if not disable_gpu_aug:
                            x = self.augmentor(x)
                        if self.normalize:
                            x = self.normalize(x)
                        feats = self.model(x)

                    out = self.fc(feats)
                    loss = nn.functional.cross_entropy(out, y)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            self.scheduler.step()

    @torch.no_grad()
    def evaluate_detailed(self, test_loader, class_names=None):
        self.fc.eval()
        all_preds, all_targets = [], []
        
        for x, y in test_loader:
            x, y = x.to(self.device), y.to(self.device)
            if self.normalize:
                x = self.normalize(x)
            
            feats = self.model(x)
            logits = self.fc(feats)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        acc = accuracy_score(all_targets, all_preds)
        return acc

# ==============================================================================
# 2. PNG Loader
# ==============================================================================
def load_png_dataset_to_memory(root_dir, image_size=256, crop_size=224, ipc=0):
    print(f"Loading Syn PNGs from {root_dir} (IPC={ipc if ipc > 0 else 'All'})...")
    transform = transforms.Compose([
        transforms.Resize((crop_size, crop_size)), 
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(root=root_dir, transform=transform)
    
    if ipc > 0:
        count_per_class = {}
        new_samples = []
        for path, target in dataset.samples:
            count_per_class.setdefault(target, 0)
            if count_per_class[target] < ipc:
                new_samples.append((path, target))
                count_per_class[target] += 1
        dataset.samples = new_samples
        dataset.targets = [s[1] for s in new_samples] 
    
    loader = DataLoader(dataset, batch_size=256, num_workers=8, shuffle=False)
    all_imgs, all_lbls = [], []
    for imgs, lbls in loader:
        all_imgs.append(imgs)
        all_lbls.append(lbls)
    
    return torch.cat(all_imgs), torch.cat(all_lbls), dataset.classes

# ==============================================================================
# 3. Main Script
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="syn", choices=["syn", "real"])
    parser.add_argument("--syn_data_dir", type=str, default=None)
    parser.add_argument("--ipc", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="imagenet")
    parser.add_argument("--data_root", type=str, default="data/datasets")
    parser.add_argument("--model", type=str, default="ResNet50")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=100) 
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--exp_name", type=str, default="experiment")
    parser.add_argument("--test_ipc", type=int, default=0)
    
    # [追加] 試行回数とシード
    parser.add_argument("--num_trials", type=int, default=1, help="Number of trials for mean/std")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")

    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Data (One-time load)
    print(f"Loading Real Dataset ({args.dataset})...")
    train_dataset_real, test_dataset_real = get_dataset(
        name=args.dataset, res=256, crop_res=224, train_crop_mode="random", data_root=args.data_root
    )

    # Test Set Preparation
    if args.test_ipc > 0:
        subset_indices = get_balanced_subset_indices(test_dataset_real, args.test_ipc, seed=42) # Test set is fixed
        test_dataset_real = Subset(test_dataset_real, subset_indices)
    test_loader = DataLoader(test_dataset_real, shuffle=False, num_workers=8, batch_size=100)
    
    # Syn Data Loading (Memory)
    syn_images, syn_labels = None, None
    if args.mode == "syn":
        if not args.syn_data_dir: raise ValueError("For mode='syn', --syn_data_dir is required.")
        syn_images, syn_labels, _ = load_png_dataset_to_memory(args.syn_data_dir, crop_size=224, ipc=args.ipc)
        if torch.cuda.is_available():
            syn_images = syn_images.cuda()
            syn_labels = syn_labels.cuda()

    # Model Backbone (Load once)
    print(f"Loading Backbone: {args.model}...")
    backbone, num_feats = get_model(args.model, distributed=False)

    # =========================================================
    # MULTI-TRIAL LOOP
    # =========================================================
    acc_history = []
    
    print(f"\nStarting Evaluation: {args.num_trials} trials")
    
    for trial in range(args.num_trials):
        current_seed = args.seed + trial
        set_seed(current_seed)
        
        # Evaluator (Initialize Head & Optimizer per trial)
        evaluator = ComprehensiveEvaluator(
            model=backbone,
            num_classes=len(train_dataset_real.classes) if hasattr(train_dataset_real, "classes") else 1000,
            num_feats=num_feats,
            lr=args.lr,
            epochs=args.epochs,
            device=device
        )
        evaluator.set_normalization(train_dataset_real.normalize)

        # Train
        if args.mode == "syn":
            # Synthetic: Data is fixed, optimization varies by seed
            real_bs = min(args.batch_size, len(syn_images))
            scaled_lr = args.lr * (real_bs / 256.0)
            for pg in evaluator.optimizer.param_groups: pg['lr'] = scaled_lr
            
            evaluator.train_linear_probe((syn_images, syn_labels), disable_gpu_aug=False)
            
        elif args.mode == "real":
            # Real Baseline: Resample data subset per trial for variance
            train_source = train_dataset_real
            if args.ipc > 0:
                indices = get_balanced_subset_indices(train_source, args.ipc, seed=current_seed)
                train_source = Subset(train_source, indices)
            
            real_loader = DataLoader(train_source, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
            
            scaled_lr = args.lr * (args.batch_size / 256.0)
            for pg in evaluator.optimizer.param_groups: pg['lr'] = scaled_lr

            evaluator.train_linear_probe(real_loader, disable_gpu_aug=True)

        # Evaluate
        acc = evaluator.evaluate_detailed(test_loader)
        acc_history.append(acc)
        print(f"  [Trial {trial+1}/{args.num_trials}] Acc: {acc*100:.2f}% (Seed: {current_seed})")

    # =========================================================
    # SUMMARY
    # =========================================================
    mean_acc = np.mean(acc_history) * 100
    std_acc = np.std(acc_history) * 100
    
    print(f"\nFinal Result ({args.num_trials} trials): {mean_acc:.2f}% ± {std_acc:.2f}%")

    file_prefix = f"{args.exp_name}_{args.mode}_ipc{args.ipc}"
    
    # Save Summary
    with open(os.path.join(args.output_dir, f"{file_prefix}_summary.txt"), "w") as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Train IPC: {args.ipc}\n")
        f.write(f"Test IPC: {args.test_ipc}\n")
        f.write(f"Trials: {args.num_trials}\n")
        f.write(f"Mean Acc: {mean_acc:.4f}\n")
        f.write(f"Std Acc: {std_acc:.4f}\n")
        f.write(f"Raw Accs: {acc_history}\n")
        
    # Save CSV for all trials
    df = pd.DataFrame({"Trial": range(1, args.num_trials+1), "Accuracy": acc_history})
    df.to_csv(os.path.join(args.output_dir, f"{file_prefix}_trials.csv"), index=False)