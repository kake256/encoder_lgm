#!/bin/bash
# run_debug.sh
set -euo pipefail

# ========================================================
# 1. Path & Config
# ========================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# プロジェクトルートの判定（このスクリプトの配置場所に応じて適宜調整してください）
# 例: スクリプトがルート直下にある場合は PROJECT_ROOT="$SCRIPT_DIR"
# 例: スクリプトが scripts/ にある場合は PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# データ設定
DATASET_DIR="$PROJECT_ROOT/makeData/dataset_clean_food101_FULL"
DATASET_TYPE="food101"

# Pythonスクリプトパス
# ※前回作成したファイル名が *_es.py の場合はここを書き換えてください
PYTHON_SCRIPT="$PROJECT_ROOT/src/distillation/evaluate_fast_unified.py"

# GPU設定
GPU_ID=0

# --- 評価モード設定 ---
EVAL_MODE="linear"
# "logits" or "none"
KD_MODE="none"

# --- Early Stopping設定 (新規追加) ---
# 精度が向上しない状態が何回続いたら止めるか
PATIENCE=5
# 何エポックごとに評価を行うか (頻繁にチェックすることで早期終了を機能させる)
VAL_INTERVAL=20
# 最大エポック数 (早期終了しなければここまで回る)
MAX_EPOCHS=1000

# --- 実験設定 ---
SELECTED_EVALUATORS="ResNet50 MAE OpenCLIP_RN50"
NUM_TRIALS=1

SYN_COUNTS="1" #"1 5"
REAL_BASELINE_COUNTS="1 5"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$PROJECT_ROOT/evaluation_results_fast/$DATASET_TYPE/${EVAL_MODE}_${KD_MODE}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

# ========================================================
# 2. Execution (Debug Mode)
# ========================================================

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Script not found at $PYTHON_SCRIPT"
    exit 1
fi

echo "========================================================"
echo " Starting Fast Evaluation Pipeline (DEBUG MODE)"
echo "========================================================"
echo " Script    : $PYTHON_SCRIPT"
echo " Dataset   : $DATASET_DIR"
echo " GPU ID    : $GPU_ID"
echo " Config    : Patience=$PATIENCE, Interval=$VAL_INTERVAL, MaxEpochs=$MAX_EPOCHS"
echo "========================================================"

# HFデータセット引数
HF_ARGS="--food_hf_dataset ethz/food101 --food_hf_test_split validation"

# --- [変更点] 実行コマンドを表示し、Pythonのバッファリングを無効化 (-u) ---
echo ">>> Executing Python command..."
set -x  # 実行コマンドをコンソールに表示

CUDA_VISIBLE_DEVICES=$GPU_ID python3 -u "$PYTHON_SCRIPT" "$DATASET_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --mode "$EVAL_MODE" \
    --kd_mode "$KD_MODE" \
    --dataset_type "$DATASET_TYPE" \
    --evaluators $SELECTED_EVALUATORS \
    --num_trials $NUM_TRIALS \
    --syn_counts $SYN_COUNTS \
    --real_baseline_counts $REAL_BASELINE_COUNTS \
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