# ファイル名: evaluate_linear2_engine.py
# 内容: PyTorchおよびSklearnの学習ループ, 評価関数

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from torch.cuda.amp import autocast, GradScaler

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

def train_pytorch_pipeline(extractor, train_loader, test_loader, num_classes, mode, config):
    model = TrainableModel(extractor, num_classes).to(DEVICE)

    params_to_optimize, lora_targets, did_fallback, policy_name = select_trainable_params(
        model, mode, config, train_batch_size=int(getattr(train_loader, "batch_size", 1) or 1)
    )

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

    for epoch in range(max_epochs):
        if mode == "linear_torch":
            model.core.eval()
            model.head.train()
        else:
            model.train()
            # ----------------------------------------------------------------
            # [CRITICAL FIX] Freeze BatchNorm Statistics during Fine-Tuning
            # 合成データの統計量でBN層が汚染されるのを防ぐため、BN層は常にevalモードにする
            # これを行わないとResNet等のCNNモデルは精度が崩壊する
            # ----------------------------------------------------------------
            for m in model.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()

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

        if scheduler:
            scheduler.step()

        if (epoch + 1) % val_interval == 0 or epoch == max_epochs - 1:
            val_acc, _, _ = evaluate_full_preds(model, test_loader)
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += val_interval
                if patience_counter >= patience:
                    break

    final_acc, y_true, y_pred = evaluate_full_preds(model, test_loader)
    train_acc, _, _ = evaluate_full_preds(model, train_loader)

    t_cnt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot_cnt = sum(p.numel() for p in model.parameters())
    return float(final_acc), float(train_acc), t_cnt, tot_cnt, (lora_targets or []), y_true, y_pred

def evaluate_full_preds(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad(), autocast():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            preds.extend(out.argmax(1).cpu().numpy())
            targets.extend(y.cpu().numpy())
    y_true = np.array(targets)
    y_pred = np.array(preds)
    acc = accuracy_score(y_true, y_pred)
    return acc, y_true, y_pred