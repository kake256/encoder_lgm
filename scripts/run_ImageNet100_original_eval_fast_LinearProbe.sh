#!/bin/bash
# run_debug.sh
set -euo pipefail

# ========================================================
# 1. Path & Config
# ========================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# プロジェクトルートの判定（このスクリプトの配置場所に応じて適宜調整してください）
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# データ設定
USE_DATA="imagenet100_orignal" #"imagenet100_orignal" #"imagenet100_FULL" #"imagenet100_FULL_batch"
DATASET_DIR="$PROJECT_ROOT/makeData/dataset_clean_$USE_DATA"
DATASET_TYPE="imagenet100"

# Pythonスクリプトパス
PYTHON_SCRIPT="$PROJECT_ROOT/src/distillation/evaluate_fast_unified_es.py"

# GPU設定
GPU_ID=0

# --- 評価モード設定 ---
EVAL_MODE="linear"
KD_MODE="logits" #"logits" #"none"

# --- Early Stopping設定 ---
PATIENCE=5
VAL_INTERVAL=20
MAX_EPOCHS=1000

# --- 実験設定 ---
#SELECTED_EVALUATORS="OpenCLIP_RN50 MAE ResNet50" #"ResNet50 MAE OpenCLIP" #"ResNet50 MAE OpenCLIP_RN50"
SELECTED_EVALUATORS="DINOv2 CLIP ResNet50 OpenCLIP_RN50 MAE"
NUM_TRIALS=5 #5

# [追加] 事前にキャッシュするデータ拡張のパターン数（例: 5）
NUM_AUG_CACHES=20

SYN_COUNTS="1" #"1 5"

# [変更] メモリ問題を解決したので、FULLデータセットで回すなら1400に戻してOKです
# 例: REAL_BASELINE_COUNTS="1 3 5" で有効化、REAL_BASELINE_COUNTS="" でスキップ
REAL_BASELINE_COUNTS="" #1400" #"1400" 

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$PROJECT_ROOT/evaluation_results_fast/${USE_DATA}/${EVAL_MODE}_${KD_MODE}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

# ========================================================
# 1.5. Notification Setup
# ========================================================
notification_exit() {
    # 最後に実行されたコマンドの終了コードを取得
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        python3 "$NOTIFICATION_SCRIPT" "success" "$OUTPUT_DIR"
    else
        python3 "$NOTIFICATION_SCRIPT" "failed" "$OUTPUT_DIR"
    fi
}
# スクリプト終了時（正常・エラー問わず）に notification_exit を実行
# trap notification_exit EXIT # ※動作確認時はコメントアウト推奨

# ========================================================
# 2. Execution (Debug Mode)
# ========================================================

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Script not found at $PYTHON_SCRIPT"
    exit 1
fi

# --- 実画像ベースライン引数の動的構築 ---
REAL_BASELINE_ARG=""
if [ -n "$REAL_BASELINE_COUNTS" ]; then
    REAL_BASELINE_ARG="--real_baseline_counts $REAL_BASELINE_COUNTS"
    ECHO_BASELINE_STATUS="ENABLED (Counts: $REAL_BASELINE_COUNTS)"
else
    ECHO_BASELINE_STATUS="SKIPPED (Empty variable)"
fi

echo "========================================================"
echo " Starting Fast Evaluation Pipeline (DEBUG MODE)"
echo "========================================================"
echo " Script       : $PYTHON_SCRIPT"
echo " Dataset      : $DATASET_DIR"
echo " GPU ID       : $GPU_ID"
echo " Config       : Patience=$PATIENCE, Interval=$VAL_INTERVAL, MaxEpochs=$MAX_EPOCHS"
echo " Aug Caches   : $NUM_AUG_CACHES patterns" # ログに追加
echo " Baseline     : $ECHO_BASELINE_STATUS"
echo "========================================================"

# HFデータセット引数
HF_ARGS="--imagenet_hf_dataset clane9/imagenet-100 --imagenet_hf_test_split validation"

echo ">>> Executing Python command..."
set -x  # 実行コマンドをコンソールに表示

# [追加] --num_aug_caches $NUM_AUG_CACHES を引数に追加
CUDA_VISIBLE_DEVICES=$GPU_ID python3 -u "$PYTHON_SCRIPT" "$DATASET_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --mode "$EVAL_MODE" \
    --kd_mode "$KD_MODE" \
    --num_aug_caches "$NUM_AUG_CACHES" \
    --dataset_type "$DATASET_TYPE" \
    --evaluators $SELECTED_EVALUATORS \
    --num_trials $NUM_TRIALS \
    --syn_counts $SYN_COUNTS \
    $REAL_BASELINE_ARG \
    --max_real_test 50 \
    --patience $PATIENCE \
    --val_interval $VAL_INTERVAL \
    --epochs $MAX_EPOCHS \
    $HF_ARGS

RET=$?
set +x  # 表示を戻す

if [ $RET -eq 0 ]; then
    echo ">>> Evaluation Complete."
else
    echo ">>> Evaluation Failed with exit code $RET."
    exit $RET
fi