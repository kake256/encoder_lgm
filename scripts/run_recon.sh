#!/bin/bash
set -e

# ========================================================
# ディレクトリ設定
# ========================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$PROJECT_DIR/src"
RESULTS_BASE_DIR="$PROJECT_DIR/classification_results"
EXPERIMENT_ROOT_DIR="$RESULTS_BASE_DIR/visualization_experiments"

mkdir -p "$EXPERIMENT_ROOT_DIR"
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

# ========================================================
# パラメータ設定
# ========================================================

# 比較したいモデルのリスト
ENCODER_LIST=(
    "openai/clip-vit-large-patch14"
    "facebook/dinov2-base"
    "facebook/dino-vitb16"
)

# [重要] Python側でループさせる射影次元リスト (スペース区切り)
PROJ_DIMS="0 1024 2048"

# ターゲットクラス
TARGET_CLASSES="1 9 950"

# 固定パラメータ
ITERATIONS=8000  # 高速化のため少し減らす推奨
LR=0.002
IMAGE_SIZE=224
NUM_REF_IMAGES=10
AUGS_PER_STEP=32
WEIGHT_TV=0.00025

# ========================================================
# 実行 (並列化対応)
# ========================================================
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
BATCH_DIR="$EXPERIMENT_ROOT_DIR/$TIMESTAMP"
mkdir -p "$BATCH_DIR"

echo "Starting Fast-Loop Experiments"
echo "Projections to test: $PROJ_DIMS"

MAX_JOBS=2
PIDS=()

# モデルごとのループ (ここは並列化のためShellに残す)
for ENCODER_NAME in "${ENCODER_LIST[@]}"; do
    SAFE_MODEL_NAME=$(echo "$ENCODER_NAME" | tr '/' '-')
    GPU_ID=$((${#PIDS[@]} % MAX_JOBS))
    
    # 出力ディレクトリ
    CURRENT_EXP_DIR="$BATCH_DIR/$SAFE_MODEL_NAME"
    mkdir -p "$CURRENT_EXP_DIR"

    echo "Dispatching $ENCODER_NAME to GPU $GPU_ID"

    # コマンド構築
    # --projection_dims にリストをそのまま渡す
    CMD="cd $SRC_DIR && CUDA_VISIBLE_DEVICES=$GPU_ID python3 visualize.py \
        --encoder_name \"$ENCODER_NAME\" \
        --projection_dims $PROJ_DIMS \
        --target_classes $TARGET_CLASSES \
        --output_dir \"$CURRENT_EXP_DIR\" \
        --num_iterations $ITERATIONS \
        --lr $LR \
        --image_size $IMAGE_SIZE \
        --num_ref_images $NUM_REF_IMAGES \
        --augs_per_step $AUGS_PER_STEP \
        --weight_tv $WEIGHT_TV"

    # 実行
    eval "$CMD" > "$CURRENT_EXP_DIR/run.log" 2>&1 &
    PIDS+=($!)

    if [ ${#PIDS[@]} -ge $MAX_JOBS ]; then
        wait -n
        wait
        PIDS=()
    fi
done

wait
echo "All Done. Results at: $BATCH_DIR"