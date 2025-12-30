#!/bin/bash
set -e

# ========================================================
# 1. 環境設定
# ========================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$PROJECT_DIR/src"
RESULTS_BASE_DIR="$PROJECT_DIR/classification_results"
EXPERIMENT_ROOT_DIR="$RESULTS_BASE_DIR/multi_model_experiments"

mkdir -p "$EXPERIMENT_ROOT_DIR"
export PYTHONPATH="$SRC_DIR:$PROJECT_DIR:$PYTHONPATH"

# ========================================================
# 2. データセット設定 (ここを切り替えて使用)
# ========================================================

# --- 設定A: ImageNet (デフォルト) ---
DATASET_TYPE="imagenet"
RTA_MODE="label"                 # ImageNetでは無視されます
DATA_ROOT=""                     # ImageNetでは無視されます
# ターゲットクラス (生物5種 + 無機物4種)
# 9:ダチョウ, 950:イチゴ, 1:金魚, 108:クラゲ, 404:旅客機 ...
TARGET_CLASSES="9 950 1 108 404"

# --- 設定B: RTA100 (使う場合はコメント解除) ---
# DATASET_TYPE="rta100"
# RTA_MODE="label"               # "label"(物体) or "text"(文字)
# DATA_ROOT="$PROJECT_DIR/Data/Rta100"
# # 特定のクラスだけやる場合はIDを記述、全部やる場合は空欄("")にする
# TARGET_CLASSES="" 

# ========================================================
# 3. 実験パラメータ
# ========================================================

# モデルリスト (Index 0: v1, Index 1: v2, Index 2: CLIP)
ENCODER_NAMES=(
    "facebook/dino-vitb16"
    "facebook/dinov2-base"
    "openai/clip-vit-base-patch16"
)

# 射影層の次元バリエーション (ループ実行)
PROJ_DIM_LIST=(
    "0 0 0"             # 射影なし
    #"1024 1024 1024"    # 中間
    "2048 2048 2048"    # 高次元
)

# 実験設定リスト
EXPERIMENTS=(
    # --- 1. Single Model ---
    "Only_v1:1.0,0.0,0.0"
    "Only_v2:0.0,1.0,0.0"
    "Only_CLIP:0.0,0.0,1.0"

    # --- 2. Two Models AND ---
    "AND_v1_CLIP:1.0,0.0,1.0"
    "AND_v2_CLIP:0.0,1.0,1.0"
    "AND_v1_v2:1.0,1.0,0.0"

    # --- 3. Three Models AND ---
    "AND_All_Three:1.0,1.0,1.0"

    # --- 4. VS Mode ---
    #"VS_v1_minus_CLIP:1.0,0.0,-0.3"
    #"VS_v2_minus_CLIP:0.0,1.0,-0.3"
)

# 固定パラメータ
ITERATIONS=4000
LR=0.002
IMAGE_SIZE=224
NUM_REF_IMAGES=10
AUGS_PER_STEP=32
WEIGHT_TV=0.00025
MAX_JOBS=2  # 並列数

# ========================================================
# 4. 実行ループ (次元 -> クラス)
# ========================================================
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
BASE_OUTPUT_DIR="$EXPERIMENT_ROOT_DIR/$TIMESTAMP"
mkdir -p "$BASE_OUTPUT_DIR"

echo "=========================================="
echo "Starting Experiments"
echo "Dataset: $DATASET_TYPE (Mode: $RTA_MODE)"
echo "Output: $BASE_OUTPUT_DIR"
echo "=========================================="

# --- 次元設定のループ ---
for PROJ_DIMS in "${PROJ_DIM_LIST[@]}"; do
    DIM_DIR_NAME="dim_${PROJ_DIMS// /_}"
    BATCH_DIR="$BASE_OUTPUT_DIR/$DIM_DIR_NAME"
    mkdir -p "$BATCH_DIR"

    echo ""
    echo "------------------------------------------"
    echo ">>> Running with Projection Dims: [ $PROJ_DIMS ]"
    echo "------------------------------------------"

    PIDS=()
    WATCH_LOG=""

    # ターゲットクラスの処理
    # 空欄の場合は "ALL" というフラグを立てて1回だけ実行する(RTA全件処理用)
    if [ -z "$TARGET_CLASSES" ]; then
        CLASS_LOOP=("ALL")
    else
        CLASS_LOOP=($TARGET_CLASSES)
    fi

    # --- クラスのループ ---
    for CLASS_ID in "${CLASS_LOOP[@]}"; do
        GPU_ID=$((${#PIDS[@]} % MAX_JOBS))
        
        # ID指定の有無でコマンドを切り替え
        if [ "$CLASS_ID" == "ALL" ]; then
            echo "   Dispatching [ALL CLASSES] to GPU $GPU_ID"
            TARGET_ARG="" # 引数なし＝全件
            LOG_FILE="$BATCH_DIR/run_all_classes.log"
        else
            echo "   Dispatching Class $CLASS_ID to GPU $GPU_ID"
            TARGET_ARG="--target_classes $CLASS_ID"
            LOG_FILE="$BATCH_DIR/class_${CLASS_ID}_run.log"
        fi

        # Python実行 ( -u オプション付き )
        CMD="cd $SRC_DIR && CUDA_VISIBLE_DEVICES=$GPU_ID python3 -u multi_model_main.py \
            --encoder_names ${ENCODER_NAMES[@]} \
            --projection_dims $PROJ_DIMS \
            --experiments ${EXPERIMENTS[@]} \
            $TARGET_ARG \
            --dataset_type $DATASET_TYPE \
            --rta_mode $RTA_MODE \
            --data_root \"$DATA_ROOT\" \
            --output_dir \"$BATCH_DIR\" \
            --num_iterations $ITERATIONS \
            --lr $LR \
            --image_size $IMAGE_SIZE \
            --num_ref_images $NUM_REF_IMAGES \
            --augs_per_step $AUGS_PER_STEP \
            --weight_tv $WEIGHT_TV"

        eval "$CMD" > "$LOG_FILE" 2>&1 &
        PIDS+=($!)

        # GPU 0 を監視対象に設定
        if [ "$GPU_ID" -eq 0 ] && [ -z "$WATCH_LOG" ]; then
            WATCH_LOG="$LOG_FILE"
        fi

        # バッチ同期
        if [ ${#PIDS[@]} -ge $MAX_JOBS ]; then
            if [ -n "$WATCH_LOG" ]; then
                python3 "$SRC_DIR/watch_log.py" "$WATCH_LOG" $ITERATIONS || true
                WATCH_LOG="" 
            fi
            echo "   >>> Batch full. Waiting..."
            wait
            PIDS=()
        fi
    done

    # 次元ループ内の端数処理
    if [ ${#PIDS[@]} -gt 0 ]; then
        echo "   >>> Waiting for final remaining jobs..."
        if [ -n "$WATCH_LOG" ]; then
            python3 "$SRC_DIR/watch_log.py" "$WATCH_LOG" $ITERATIONS || true
        fi
        wait
    fi
    
    echo ">>> Completed Projection Dims: $PROJ_DIMS"
done

echo "=========================================="
echo "All Experiments & All Dimensions Completed."
echo "Results located at: $BASE_OUTPUT_DIR"
echo "=========================================="