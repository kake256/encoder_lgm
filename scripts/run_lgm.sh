#!/bin/bash
set -e

# =========================================================
# ディレクトリ設定
# =========================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$PROJECT_DIR/src"

# 1. 重み保存用ディレクトリ
WEIGHTS_DIR="$PROJECT_DIR/weights"
mkdir -p "$WEIGHTS_DIR"

# 2. 画像保存用ディレクトリ
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
VIS_DIR="$PROJECT_DIR/visualizations/${TIMESTAMP}_LGM"
mkdir -p "$VIS_DIR"

export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

# ========================================================
# ★ 設定パラメータ
# ========================================================
TARGET_CLASSES="1 9 404 950"
COMMON_DIM=2048
EPOCHS=5
BATCH_SIZE=32
ITERATIONS=1500
LR=0.002
IMAGE_SIZE=224
AUGS_PER_STEP=8
PYRAMID_GROW=400

# Hugging Face Dataset
HF_DATASET_NAME="imagenet-1k"
DATASET_FRACTION=0.1

# ========================================================
# 実行フロー
# ========================================================
echo "=========================================="
echo "Starting LGM Pipeline"
echo "Weights Dir:  $WEIGHTS_DIR"
echo "Results Dir:  $VIS_DIR"
echo "=========================================="

# --------------------------------------------------------
# Phase 1: 重みの確認 & 学習 (Training)
# --------------------------------------------------------
echo ""
echo ">> Checking for existing weights..."

LATEST_WEIGHT=$(ls -t "$WEIGHTS_DIR"/*.pth 2>/dev/null | head -n 1)

if [ -n "$LATEST_WEIGHT" ]; then
    PROJ_WEIGHTS="$LATEST_WEIGHT"
    echo "Found existing weights: $PROJ_WEIGHTS"
    echo "Skipping training phase."
else
    echo "No weights found in $WEIGHTS_DIR. Starting training..."
    
    python3 "$SRC_DIR/train_projection_expansion.py" \
        --dataset_name "$HF_DATASET_NAME" \
        --dataset_fraction "$DATASET_FRACTION" \
        --output_dir "$WEIGHTS_DIR" \
        --common_dim $COMMON_DIM \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr 0.0001

    GENERATED_WEIGHT="$WEIGHTS_DIR/projection_weights.pth"
    NEW_WEIGHT_NAME="$WEIGHTS_DIR/weights_${TIMESTAMP}.pth"
    
    if [ -f "$GENERATED_WEIGHT" ]; then
        mv "$GENERATED_WEIGHT" "$NEW_WEIGHT_NAME"
        PROJ_WEIGHTS="$NEW_WEIGHT_NAME"
        echo "Training finished. Weights saved to: $PROJ_WEIGHTS"
    else
        echo "Error: Training failed (Weight file not created)."
        exit 1
    fi
fi

# --------------------------------------------------------
# Phase 2: 画像生成 (Visualization)
# --------------------------------------------------------
echo ""
echo ">> [Phase 2] Running Visualization using: $(basename "$PROJ_WEIGHTS")"

# ★修正: ここから --dataset_fraction を削除しました
python3 "$SRC_DIR/lgm_common_space.py" \
    --weights_path "$PROJ_WEIGHTS" \
    --dataset_name "$HF_DATASET_NAME" \
    --output_dir "$VIS_DIR" \
    --target_classes $TARGET_CLASSES \
    --common_dim $COMMON_DIM \
    --num_iterations $ITERATIONS \
    --lr $LR \
    --image_size $IMAGE_SIZE \
    --pyramid_grow_interval $PYRAMID_GROW \
    --augs_per_step $AUGS_PER_STEP

echo "=========================================="
echo "All Completed!"
echo "Check images in: $VIS_DIR"
echo "=========================================="