# ファイル名: evaluate_linear2_engine.py
# 内容: データ拡張戦略(3パターン)に対応した高速化エンジン

import os
import re
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset

from evaluate_linear2_config import DEVICE
from evaluate_linear2_models import TrainableModel, select_trainable_params

def train_sklearn_lbfgs(extractor, train_loader, X_test, y_test, config):
    X_train, y_train = extractor.extract_features(train_loader)
    if X_train.size == 0:
        return 0.0, 0, 0, 0, np.array([]), np.array([])

    clf = LogisticRegression(
        max_iter=int(config["max_iter"]),
        solver="lbfgs",
        C=float(config["C"]),
        penalty="l2",
        multi_class="auto",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    return float(acc), float(train_acc), 0, 0, y_test, y_pred

def train_pytorch_pipeline(extractor, train_loader, test_loader, num_classes, mode, config, 
                           log_dir=None, run_name=None, 
                           aug_strategy="none", aug_expansion=20):
    
    # -------------------------------------------------------
    # [Speedup] Feature Caching / Pre-computation Logic
    # -------------------------------------------------------
    cached_features = False
    
    if mode == "linear_torch":
        # --- Pattern 1: None (Augなし, Cacheあり) ---
        if aug_strategy == "none":
            print("   [Mode] No Augmentation. Standard Caching (1x).")
            feats, lbls = extractor.extract_features(train_loader)
            if len(feats) > 0:
                train_ds = TensorDataset(torch.from_numpy(feats).float(), torch.from_numpy(lbls).long())
                cached_features = True
        
        # --- Pattern 2: Precompute (Augあり, Cacheあり, Nx膨張) ---
        elif aug_strategy == "precompute":
            print(f"   [Mode] Pre-computed Augmentation. Caching with {aug_expansion}x expansion...")
            all_feats_list = []
            all_lbls_list = []
            try:
                for i in range(aug_expansion):
                    if i % 10 == 0: print(f"     -> Expansion step {i+1}/{aug_expansion}...")
                    f, l = extractor.extract_features(train_loader)
                    if len(f) > 0:
                        all_feats_list.append(f)
                        all_lbls_list.append(l)
                
                if len(all_feats_list) > 0:
                    final_feats = np.concatenate(all_feats_list, axis=0)
                    final_lbls = np.concatenate(all_lbls_list, axis=0)
                    train_ds = TensorDataset(
                        torch.from_numpy(final_feats).float(), 
                        torch.from_numpy(final_lbls).long()
                    )
                    cached_features = True
                    print(f"   [Done] Expanded Dataset: {len(train_loader.dataset)} -> {len(train_ds)} samples.")
            except Exception as e:
                print(f"   [Error] Pre-computation failed: {e}. Fallback to on_the_fly.")

        # --- Pattern 3: On-the-fly (Augあり, Cacheなし) ---
        elif aug_strategy == "on_the_fly":
            print("   [Mode] On-the-fly Augmentation. No Caching (Slow but Infinite diversity).")
            cached_features = False

    # キャッシュ成功時: Loaderを差し替えてオンメモリ爆速化
    if cached_features:
        bs = train_loader.batch_size
        if bs is None or bs < 256: bs = 1024
        
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0, pin_memory=True)
        
        test_feats, test_lbls = extractor.extract_features(test_loader)
        if len(test_feats) > 0:
            test_ds = TensorDataset(torch.from_numpy(test_feats).float(), torch.from_numpy(test_lbls).long())
            test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)

    # -------------------------------------------------------
    # Model Setup
    # -------------------------------------------------------
    model = TrainableModel(extractor, num_classes).to(DEVICE)

    params_to_optimize, lora_targets, did_fallback, policy_name = select_trainable_params(
        model, mode, config, train_batch_size=int(getattr(train_loader, "batch_size", 1) or 1)
    )

    t_cnt = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(
            params_to_optimize,
            weight_decay=float(config.get("weight_decay", 0.0)),
            betas=tuple(config.get("betas", (0.9, 0.999))),
            eps=float(config.get("eps", 1e-8)),
        )
    else:
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            weight_decay=float(config.get("weight_decay", 0.0))
        )

    scaler = GradScaler()
    scheduler = None
    if config.get("scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config["epochs"]))

    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    patience = int(config.get("patience", 100))
    patience_counter = 0
    max_epochs = int(config["epochs"])
    val_interval = int(config.get("val_interval", 5))

    history = []

    for epoch in range(max_epochs):
        model.set_mode(mode)

        running_loss = 0.0
        total_samples = 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            with autocast():
                # [Speedup] キャッシュ済みならForwardHeadのみ
                if cached_features:
                    out = model.forward_head(x)
                else:
                    if mode == "linear_torch":
                        with torch.no_grad():
                            feats = model.get_features(x)
                        out = model.forward_head(feats)
                    else:
                        out = model(x)
                loss = criterion(out, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size_curr = x.size(0)
            running_loss += loss.item() * batch_size_curr
            total_samples += batch_size_curr

        if scheduler:
            scheduler.step()

        epoch_train_loss = running_loss / total_samples if total_samples > 0 else 0.0
        
        val_loss = None
        val_acc = None
        
        if (epoch + 1) % val_interval == 0 or epoch == max_epochs - 1:
            val_loss, val_acc, _, _ = evaluate_metrics(model, test_loader, criterion, is_features=cached_features)
            
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += val_interval
                if patience_counter >= patience:
                    history.append({
                        "epoch": epoch + 1,
                        "mode": mode,
                        "train_loss": epoch_train_loss,
                        "val_loss": val_loss,
                        "val_acc": val_acc
                    })
                    break
        
        history.append({
            "epoch": epoch + 1,
            "mode": mode,
            "train_loss": epoch_train_loss,
            "val_loss": val_loss, 
            "val_acc": val_acc
        })

    if log_dir:
        if not run_name:
            run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name)[:150]
        csv_filename = f"{safe_name}_log.csv"
        csv_path = os.path.join(log_dir, csv_filename)
        try:
            df_history = pd.DataFrame(history)
            df_history.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"[Warning] Failed to save training log: {e}")

    _, final_acc, y_true, y_pred = evaluate_metrics(model, test_loader, criterion, is_features=cached_features)
    _, train_acc, _, _ = evaluate_metrics(model, train_loader, criterion, is_features=cached_features)

    tot_cnt = sum(p.numel() for p in model.parameters())
    return float(final_acc), float(train_acc), t_cnt, tot_cnt, (lora_targets or []), y_true, y_pred

def evaluate_metrics(model, loader, criterion, is_features=False):
    model.eval()
    preds, targets = [], []
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad(), autocast():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            if is_features:
                out = model.forward_head(x)
            else:
                out = model(x)
            
            loss = criterion(out, y)
            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            preds.extend(out.argmax(1).cpu().numpy())
            targets.extend(y.cpu().numpy())
            
    y_true = np.array(targets)
    y_pred = np.array(preds)
    acc = accuracy_score(y_true, y_pred)
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    
    return avg_loss, acc, y_true, y_pred