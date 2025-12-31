#!/bin/bash
set -e

# ========================================================
# 1. 環境設定
# ========================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$PROJECT_DIR/src"
RESULTS_BASE_DIR="$PROJECT_DIR/classification_results"

# 出力先のディレクトリ
EXPERIMENT_ROOT_DIR="$RESULTS_BASE_DIR/gene_experiment_4models_comparison"

mkdir -p "$EXPERIMENT_ROOT_DIR"
export PYTHONPATH="$SRC_DIR:$PROJECT_DIR:$PYTHONPATH"

# ========================================================
# 2. データセット設定 (ここを切り替えて使用)
# ========================================================

# ★★★ 設定A: ImageNet (デフォルト: 元の設定) ★★★
DATASET_TYPE="imagenet"
DATA_ROOT="" 
# 元のターゲットクラスリスト
TARGET_CLASSES="1 9 130 250 340 404 483 598 682 776 805 815 850 920 937 948 973 980 993 999"
# テスト用短縮版（必要に応じてコメント解除）
# TARGET_CLASSES="9 130"

# --- 以下、他のデータセット用設定（使用時は上のImageNetをコメントアウトし、こちらを解除） ---

# ★★★ 設定B: Food-101 (料理) ★★★
# DATASET_TYPE="food101"
# DATA_ROOT="./data"
# # 73:Pizza, 76:Ramen, 92:Sushi, 100:Waffles, 49:Gyoza
# TARGET_CLASSES="73 76 92 100 49"

# ★★★ 設定C: CUB-200-2011 (鳥類) ★★★
# DATASET_TYPE="cub200"
# DATA_ROOT="./data"
# # 1:Laysan Albatross, 11:Black_footed_Albatross, 36:Northern_Flicker
# TARGET_CLASSES="1 11 36"


# ========================================================
# 3. 実験パラメータ
# ========================================================

ENCODER_NAMES=(
    "facebook/dino-vitb16"
    "facebook/dinov2-base"
    "openai/clip-vit-base-patch16"
    "google/siglip-base-patch16-224"
)

PROJ_DIM_LIST=("0 0 0 0")

EXPERIMENTS=(
    "Only_V1:1.0,0.0,0.0,0.0"
    "Only_V2:0.0,1.0,0.0,0.0"
    "Only_CLIP:0.0,0.0,1.0,0.0"
    "Only_SigLIP:0.0,0.0,0.0,1.0"
    "Hybrid_V1_CLIP:1.0,0.0,1.0,0.0"
    "Hybrid_V2_CLIP:0.0,1.0,1.0,0.0"
)

# --- 生成パラメータ ---
#ITERATIONS=2000
ITERATIONS=4000

#AUGS_PER_STEP=16
AUGS_PER_STEP=64

NUM_GENERATIONS=1

LR=0.002
IMAGE_SIZE=224
NUM_REF_IMAGES=10

# --- 品質制御パラメータ ---
MIN_SCALE=0.08
MAX_SCALE=1.0
NOISE_STD=0.2
NOISE_PROB=0.5
WEIGHT_TV=0.00025
PYRAMID_START_RES=4

# 文字設定
OVERLAY_TEXT=""
TEXT_COLOR="red"
FONT_SCALE=0.5

# 並列実行設定
MAX_JOBS=2
NUM_GPUS=1

# ========================================================
# 4. 実行ループ
# ========================================================
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
BASE_OUTPUT_DIR="$EXPERIMENT_ROOT_DIR/$TIMESTAMP"
mkdir -p "$BASE_OUTPUT_DIR"

echo "=========================================="
echo "Dataset: $DATASET_TYPE"
echo "Classes: $TARGET_CLASSES"
echo "Output: $BASE_OUTPUT_DIR"
echo "=========================================="

for PROJ_DIMS in "${PROJ_DIM_LIST[@]}"; do
    BATCH_DIR="$BASE_OUTPUT_DIR/gen_data"
    mkdir -p "$BATCH_DIR"

    if [ -z "$TARGET_CLASSES" ]; then
        CLASS_LOOP=("ALL")
    else
        CLASS_LOOP=($TARGET_CLASSES)
    fi

    PIDS=()
    JOB_COUNTER=0

    for CLASS_ID in "${CLASS_LOOP[@]}"; do
        GPU_ID=$((JOB_COUNTER % NUM_GPUS))
        JOB_COUNTER=$((JOB_COUNTER + 1))
        
        LOG_FILENAME="class_${CLASS_ID}.log"
        LOG_FILE_PATH="$BATCH_DIR/$LOG_FILENAME"

        echo "   [GPU $GPU_ID] Starting Class: $CLASS_ID"

        CMD="cd \"$SRC_DIR\" && CUDA_VISIBLE_DEVICES=$GPU_ID python3 main_genimg.py \
            --encoder_names ${ENCODER_NAMES[@]} \
            --projection_dims $PROJ_DIMS \
            --experiments ${EXPERIMENTS[@]} \
            --target_classes $CLASS_ID \
            --dataset_type \"$DATASET_TYPE\" \
            --data_root \"$DATA_ROOT\" \
            --output_dir \"$BATCH_DIR\" \
            --num_iterations $ITERATIONS \
            --lr $LR \
            --image_size $IMAGE_SIZE \
            --num_ref_images $NUM_REF_IMAGES \
            --augs_per_step $AUGS_PER_STEP \
            --weight_tv $WEIGHT_TV \
            --num_generations $NUM_GENERATIONS \
            --overlay_text \"$OVERLAY_TEXT\" \
            --text_color \"$TEXT_COLOR\" \
            --font_scale $FONT_SCALE \
            --pyramid_start_res $PYRAMID_START_RES \
            --min_scale $MIN_SCALE \
            --max_scale $MAX_SCALE \
            --noise_std $NOISE_STD \
            --noise_prob $NOISE_PROB \
            --log_file \"$LOG_FILE_PATH\" > /dev/null 2>&1 &"

        eval "$CMD"
        PIDS+=($!)

        if [ ${#PIDS[@]} -ge $MAX_JOBS ]; then
            for pid in "${PIDS[@]}"; do
                wait "$pid"
            done
            PIDS=()
        fi
    done

    if [ ${#PIDS[@]} -gt 0 ]; then
        for pid in "${PIDS[@]}"; do
            wait "$pid"
        done
        PIDS=()
    fi
done

echo "Done."