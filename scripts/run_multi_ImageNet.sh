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
# 2. データセット設定
# ========================================================

# --- 設定A: ImageNet ---
DATASET_TYPE="imagenet"
RTA_MODE="label"
DATA_ROOT=""
# ターゲットクラス (リスト形式で記述)
TARGET_CLASSES="9 950 1 108 404"

# --- 設定B: RTA100 (コメントアウト中) ---
# DATASET_TYPE="rta100"
# RTA_MODE="label" # "label" or "text"
# DATA_ROOT="$PROJECT_DIR/Data/rta100"
# # 並列実行する場合、ここにクラス名を列挙する必要があります。
# # 空欄("")の場合はPython側で全件取得するため、1つのGPUで順次実行になります。
# TARGET_CLASSES="" 

# ========================================================
# 3. 実験パラメータ
# ========================================================

# モデルリスト
ENCODER_NAMES=(
    "facebook/dino-vitb16"
    "facebook/dinov2-base"
    "openai/clip-vit-base-patch16"
)

# 射影層の次元バリエーション
PROJ_DIM_LIST=(
    "0 0 0"             # 射影なし
    "2048 2048 2048"    # 高次元
)

# 実験設定リスト
EXPERIMENTS=(
    "Only_v1:1.0,0.0,0.0"
    "Only_v2:0.0,1.0,0.0"
    "Only_CLIP:0.0,0.0,1.0"
    "AND_v1_CLIP:1.0,0.0,1.0"
    "AND_v2_CLIP:0.0,1.0,1.0"
    "AND_v1_v2:1.0,1.0,0.0"
    "AND_All_Three:1.0,1.0,1.0"
)

# 固定パラメータ
ITERATIONS=4000
LR=0.002
IMAGE_SIZE=224
NUM_REF_IMAGES=10
AUGS_PER_STEP=32
WEIGHT_TV=0.00025

# 並列設定
MAX_JOBS=2

# ========================================================
# 4. 実行ループ
# ========================================================
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
BASE_OUTPUT_DIR="$EXPERIMENT_ROOT_DIR/$TIMESTAMP"
mkdir -p "$BASE_OUTPUT_DIR"

echo "=========================================="
echo "Starting Parallel Experiments (Max Jobs: $MAX_JOBS)"
echo "Dataset: $DATASET_TYPE"
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

    # ターゲットクラスのリスト化
    # TARGET_CLASSESが空の場合は "ALL" というダミーを入れて1回だけ回す(Python側で全件処理)
    if [ -z "$TARGET_CLASSES" ]; then
        CLASS_LOOP=("ALL")
        echo "   Target Classes is empty. Running all classes on single GPU sequentially."
    else
        CLASS_LOOP=($TARGET_CLASSES)
    fi

    PIDS=()
    
    # --- クラスごとのループ (並列処理) ---
    for CLASS_ID in "${CLASS_LOOP[@]}"; do
        
        # GPU IDの割り当て (0 または 1)
        # 現在実行中のPID数を使って割り振る
        GPU_ID=$((${#PIDS[@]} % MAX_JOBS))
        
        # 引数とログファイルの設定
        if [ "$CLASS_ID" == "ALL" ]; then
            TARGET_ARG=""
            LOG_FILENAME="run_all_classes.log"
        else
            TARGET_ARG="--target_classes $CLASS_ID"
            LOG_FILENAME="run_class_${CLASS_ID}.log"
        fi
        
        LOG_FILE_PATH="$BATCH_DIR/$LOG_FILENAME"

        # 【重要】出力制御のロジック
        # GPU 0 の場合は画面に出す (REDIRECTなし)
        # GPU 1 の場合は画面に出さない (REDIRECTあり)
        if [ "$GPU_ID" -eq 0 ]; then
            REDIRECT=""
            echo "   [GPU $GPU_ID] Class: $CLASS_ID (Log visible)"
        else
            REDIRECT="> /dev/null 2>&1"
            echo "   [GPU $GPU_ID] Class: $CLASS_ID (Log hidden, saved to file)"
        fi

        # Python実行
        # バックグラウンド実行(&)にするため、evalを使う
        CMD="cd \"$SRC_DIR\" && CUDA_VISIBLE_DEVICES=$GPU_ID python3 multi_model_main.py \
            --encoder_names ${ENCODER_NAMES[@]} \
            --projection_dims $PROJ_DIMS \
            --experiments ${EXPERIMENTS[@]} \
            $TARGET_ARG \
            --dataset_type \"$DATASET_TYPE\" \
            --rta_mode \"$RTA_MODE\" \
            --data_root \"$DATA_ROOT\" \
            --output_dir \"$BATCH_DIR\" \
            --num_iterations $ITERATIONS \
            --lr $LR \
            --image_size $IMAGE_SIZE \
            --num_ref_images $NUM_REF_IMAGES \
            --augs_per_step $AUGS_PER_STEP \
            --weight_tv $WEIGHT_TV \
            --log_file \"$LOG_FILE_PATH\" $REDIRECT &"

        eval "$CMD"
        PIDS+=($!)

        # 指定した並列数(MAX_JOBS)に達したら完了を待つ
        if [ ${#PIDS[@]} -ge $MAX_JOBS ]; then
            echo "   >>> Waiting for batch completion..."
            for pid in "${PIDS[@]}"; do
                wait "$pid"
            done
            PIDS=() # PIDリストをリセット
            echo "   >>> Batch finished. Proceeding."
        fi

    done

    # ループ終了時に残っているジョブがあれば待つ
    if [ ${#PIDS[@]} -gt 0 ]; then
        echo "   >>> Waiting for remaining jobs..."
        for pid in "${PIDS[@]}"; do
            wait "$pid"
        done
        PIDS=()
    fi

    echo ">>> Completed Projection Dims: $PROJ_DIMS"
done

echo "=========================================="
echo "All Experiments & All Dimensions Completed."
echo "Results located at: $BASE_OUTPUT_DIR"
echo "=========================================="