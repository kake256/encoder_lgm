# ファイル名: evaluate_linearKD_engine.py
# 内容: KLダイバージェンスを用いたロジットベースの知識蒸留 (Logit-based KD) 学習ループ

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score
from torch.cuda.amp import autocast, GradScaler
from evaluate_linearKD_config import DEVICE, KD_ALPHA, KD_TEMPERATURE

def train_pytorch_pipeline_kd(extractor, train_loader, test_loader, num_classes, mode, config):
    from evaluate_linearKD_models import TrainableModel
    
    model = TrainableModel(extractor, num_classes).to(DEVICE)
    
    # Optimizerの設定 (ScratchかLinear Probeかで対象を変える)
    params_to_optimize = []
    if mode in ["scratch", "full_ft"]:
        for p in model.core.parameters():
            p.requires_grad = True
        params_to_optimize.append({"params": model.core.parameters(), "lr": float(config.get("lr_backbone", 1e-4))})
    else:
        for p in model.core.parameters():
            p.requires_grad = False
            
    # Headは常に学習
    params_to_optimize.append({"params": model.head.parameters(), "lr": float(config.get("lr_head", config.get("lr", 1e-3)))})
    
    optimizer = torch.optim.AdamW(params_to_optimize, weight_decay=float(config.get("weight_decay", 1e-4)))
    scaler = GradScaler()
    
    # 損失関数の定義
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kd = nn.KLDivLoss(reduction="batchmean")
    
    best_acc = 0.0
    epochs = int(config.get("epochs", 100))
    patience = int(config.get("patience", 10))
    patience_counter = 0

    for epoch in range(epochs):
        model.set_mode(mode)
        running_loss = 0.0
        total_samples = 0

        # dataローダーから x(画像), y(正解ラベル), soft_target(教師ロジット) を受け取る
        for x, y, soft_target in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            with autocast():
                # 生徒モデルのロジット出力
                student_logits = model(x)
                
                # 1. ハードラベルによる通常のクロスエントロピー誤差
                loss_ce = criterion_ce(student_logits, y)
                loss = loss_ce
                
                # 2. 教師のロジットが存在する場合、Knowledge Distillationを適用
                if soft_target is not None and soft_target.numel() > 0:
                    teacher_logits = soft_target.to(DEVICE) # shape: (B, num_classes)
                    
                    # KL Divergenceの要件: 生徒はlog_softmax, 教師はsoftmax
                    student_log_probs = F.log_softmax(student_logits / KD_TEMPERATURE, dim=1)
                    teacher_probs = F.softmax(teacher_logits / KD_TEMPERATURE, dim=1)
                    
                    # 温度Tの2乗でスケールを調整
                    loss_kd = criterion_kd(student_log_probs, teacher_probs) * (KD_TEMPERATURE ** 2)
                    
                    # ハードラベルとソフトラベルの損失を KD_ALPHA でブレンド
                    loss = (1.0 - KD_ALPHA) * loss_ce + KD_ALPHA * loss_kd

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size_curr = x.size(0)
            running_loss += loss.item() * batch_size_curr
            total_samples += batch_size_curr

        # Validation チェック (5エポック毎、または最終エポック)
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            val_loss, val_acc, _, _ = evaluate_metrics(model, test_loader, criterion_ce)
            
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 5
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break

    # 最終結果の評価
    _, final_acc, y_true, y_pred = evaluate_metrics(model, test_loader, criterion_ce)
    _, train_acc, _, _ = evaluate_metrics(model, train_loader, criterion_ce)
    
    return float(final_acc), float(train_acc), y_true, y_pred


def evaluate_metrics(model, loader, criterion):
    """モデルの性能を評価し、精度と予測結果を返す"""
    model.eval()
    preds = []
    targets = []
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad(), autocast():
        # 評価時は soft_target は不要なので _ で受ける
        for x, y, _ in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            logits = model(x)
            loss = criterion(logits, y)
            
            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            preds.extend(logits.argmax(1).cpu().numpy())
            targets.extend(y.cpu().numpy())
            
    y_true = np.array(targets)
    y_pred = np.array(preds)
    acc = accuracy_score(y_true, y_pred)
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    
    return avg_loss, acc, y_true, y_pred