#!/bin/bash
# run_debug.sh
set -euo pipefail

# ========================================================
# 1. Path & Config
# ========================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ========================================================
# 2. Dataset Settings
# ========================================================
# 実行したい USE_DATA をここに並べる
USE_DATA_LIST=(
    #"ImageNet100_layer-last_32x1_s1_fulluse_grow400_it3600"
    # /workspace/encoder_lgm/makeData/dataset_clean_imagenet100_FULL_prot
    "imagenet100_FULL_LORAtest"
)

DATASET_TYPE="imagenet100"

# Pythonスクリプトパス
PYTHON_SCRIPT="$PROJECT_ROOT/src/distillation/evaluate_fast_unified_es.py"

# GPU設定
GPU_ID=0

# --- 評価モード設定 ---
EVAL_MODE="linear"
KD_MODE="none"   # "logits" or "none"

# --- キャッシュ設定 ---
# 1 にすると Python 側へ --disable_cache を渡す
# 0 にすると従来通りキャッシュを利用する
DISABLE_CACHE=1

# --- Early Stopping設定 ---
PATIENCE=5
VAL_INTERVAL=20
MAX_EPOCHS=1000

# --- 実験設定 ---
SELECTED_EVALUATORS="ResNet50 MAE OpenCLIP_RN50"
NUM_TRIALS=5
SYN_COUNTS="1"   # "1 5" なども可

# 空文字 "" にすると実画像ベースラインをスキップ
REAL_BASELINE_COUNTS="1"

# HFデータセット引数
HF_ARGS="--imagenet_hf_dataset clane9/imagenet-100 --imagenet_hf_test_split validation"

# ========================================================
# 3. Notification Setup
# ========================================================
notification_exit() {
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        python3 "$NOTIFICATION_SCRIPT" "success" "$CURRENT_OUTPUT_DIR"
    else
        python3 "$NOTIFICATION_SCRIPT" "failed" "$CURRENT_OUTPUT_DIR"
    fi
}
# trap notification_exit EXIT   # 必要なら有効化

# ========================================================
# 4. Common Args Builder
# ========================================================
REAL_BASELINE_ARG=""
if [ -n "$REAL_BASELINE_COUNTS" ]; then
    REAL_BASELINE_ARG="--real_baseline_counts $REAL_BASELINE_COUNTS"
    ECHO_BASELINE_STATUS="ENABLED (Counts: $REAL_BASELINE_COUNTS)"
else
    ECHO_BASELINE_STATUS="SKIPPED (Empty variable)"
fi

DISABLE_CACHE_ARG=""
if [ "$DISABLE_CACHE" -eq 1 ]; then
    DISABLE_CACHE_ARG="--disable_cache"
    ECHO_CACHE_STATUS="DISABLED"
else
    ECHO_CACHE_STATUS="ENABLED"
fi

# ========================================================
# 5. Main Loop
# ========================================================
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Script not found at $PYTHON_SCRIPT"
    exit 1
fi

for USE_DATA in "${USE_DATA_LIST[@]}"; do
    DATASET_DIR="$PROJECT_ROOT/makeData/dataset_clean_$USE_DATA"

    if [ ! -d "$DATASET_DIR" ]; then
        echo "Warning: Dataset directory not found, skipping: $DATASET_DIR"
        continue
    fi

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="$PROJECT_ROOT/evaluation_results_fast/${USE_DATA}/${EVAL_MODE}_${KD_MODE}_${TIMESTAMP}"
    mkdir -p "$OUTPUT_DIR"
    CURRENT_OUTPUT_DIR="$OUTPUT_DIR"

    echo "========================================================"
    echo " Starting Fast Evaluation Pipeline (DEBUG MODE)"
    echo "========================================================"
    echo " Script       : $PYTHON_SCRIPT"
    echo " USE_DATA     : $USE_DATA"
    echo " Dataset      : $DATASET_DIR"
    echo " GPU ID       : $GPU_ID"
    echo " Config       : Patience=$PATIENCE, Interval=$VAL_INTERVAL, MaxEpochs=$MAX_EPOCHS"
    echo " Baseline     : $ECHO_BASELINE_STATUS"
    echo " Cache        : $ECHO_CACHE_STATUS"
    echo " Output       : $OUTPUT_DIR"
    echo "========================================================"

    echo ">>> Executing Python command..."
    set -x

    CUDA_VISIBLE_DEVICES=$GPU_ID python3 -u "$PYTHON_SCRIPT" "$DATASET_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --mode "$EVAL_MODE" \
        --kd_mode "$KD_MODE" \
        --dataset_type "$DATASET_TYPE" \
        --evaluators $SELECTED_EVALUATORS \
        --num_trials $NUM_TRIALS \
        --syn_counts $SYN_COUNTS \
        $REAL_BASELINE_ARG \
        $DISABLE_CACHE_ARG \
        --max_real_test 50 \
        --patience $PATIENCE \
        --val_interval $VAL_INTERVAL \
        --epochs $MAX_EPOCHS \
        $HF_ARGS

    RET=$?
    set +x

    if [ $RET -eq 0 ]; then
        echo ">>> Evaluation Complete for $USE_DATA."
    else
        echo ">>> Evaluation Failed for $USE_DATA with exit code $RET."
        exit $RET
    fi

    echo
done