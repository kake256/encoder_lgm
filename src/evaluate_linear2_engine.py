# ファイル名: evaluate_linear2_engine.py
# 内容: PyTorchおよびSklearnの学習ループ, 評価関数 (BN修正版)

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

from evaluate_linear2_config import DEVICE
from evaluate_linear2_models import TrainableModel, select_trainable_params

def train_sklearn_lbfgs(extractor, train_loader, X_test, y_test, config):
    """
    Sklearnを使用したロジスティック回帰
    """
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

def train_pytorch_pipeline(extractor, train_loader, test_loader, num_classes, mode, config, log_dir=None, run_name=None):
    """
    PyTorch学習パイプライン
    修正: ループ内で model.set_mode(mode) を使用し、BNの挙動を正しく制御するように変更
    """
    model = TrainableModel(extractor, num_classes).to(DEVICE)

    params_to_optimize, lora_targets, did_fallback, policy_name = select_trainable_params(
        model, mode, config, train_batch_size=int(getattr(train_loader, "batch_size", 1) or 1)
    )

    # ログ用にパラメータ数を計算
    t_cnt = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if did_fallback:
        print(f"[Warning] select_trainable_params fallback occurred. policy={policy_name}")

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

    # 記録用リスト
    history = []

    for epoch in range(max_epochs):
        # =========================================================
        # 【修正】ここを model.set_mode(mode) に一任する
        # これにより、scratch時はBN学習(train)、full_ft時はBN固定(eval)が正しく適用される
        # =========================================================
        model.set_mode(mode)

        running_loss = 0.0
        total_samples = 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            with autocast():
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
        
        # Validation Check
        if (epoch + 1) % val_interval == 0 or epoch == max_epochs - 1:
            val_loss, val_acc, _, _ = evaluate_metrics(model, test_loader, criterion)
            
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += val_interval
                if patience_counter >= patience:
                    # Early stopping logging
                    history.append({
                        "epoch": epoch + 1,
                        "mode": mode,
                        "policy": policy_name,
                        "trainable_params": t_cnt,
                        "train_loss": epoch_train_loss,
                        "val_loss": val_loss,
                        "val_acc": val_acc
                    })
                    break
        
        # 履歴追加
        history.append({
            "epoch": epoch + 1,
            "mode": mode,
            "policy": policy_name,
            "trainable_params": t_cnt,
            "train_loss": epoch_train_loss,
            "val_loss": val_loss, 
            "val_acc": val_acc
        })
        
        if (epoch + 1) % val_interval == 0:
            # 進捗確認用 (必要に応じてコメントアウト解除)
            pass

    # CSV保存ロジック (サニタイズ + 安全策)
    if log_dir:
        # run_name が無い場合のフォールバック
        if not run_name:
            run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        # ファイル名サニタイズ: 英数字, ., -, _ 以外を _ に置換し、長さを制限
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name)[:150]
        csv_filename = f"{safe_name}_log.csv"
        csv_path = os.path.join(log_dir, csv_filename)
        
        try:
            df_history = pd.DataFrame(history)
            df_history.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"[Warning] Failed to save training log to {csv_path}: {e}")

    # 最終評価
    _, final_acc, y_true, y_pred = evaluate_metrics(model, test_loader, criterion)
    _, train_acc, _, _ = evaluate_metrics(model, train_loader, criterion)

    tot_cnt = sum(p.numel() for p in model.parameters())
    return float(final_acc), float(train_acc), t_cnt, tot_cnt, (lora_targets or []), y_true, y_pred

def evaluate_metrics(model, loader, criterion):
    """
    AccuracyだけでなくLossも計算して返す評価関数
    """
    model.eval()
    preds, targets = [], []
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad(), autocast():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            
            # Loss計算
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