#!/bin/bash
# run_eval_kd_pipeline.sh
set -euo pipefail

# ========================================================
# 1. Common Configuration (共通設定)
# ========================================================

# 評価に使用するモデル
#SELECTED_EVALUATORS="ResNet50 MAE OpenCLIP_RN50"
SELECTED_EVALUATORS="MAE OpenCLIP_RN50"

# フラグ設定
ENABLE_AUGMENTATION="true"
ENABLE_TSNE="false"

# Data Augmentation Strategy
AUG_STRATEGY="on_the_fly"
AUG_EXPANSION=20

# 試行回数と枚数設定
#NUM_TRIALS=5
NUM_TRIALS=1
REAL_IMAGE_COUNTS="5" 
REAL_TEST_COUNT=50
SYN_COUNTS="5"
#SYN_COUNTS="1 3 5"

# Mixソースの定義
MIX_SET_SINGLE=""
MIX_SET_MULTI=""
MIX_SET_ALL=""

# パス設定
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_ROOT/src"

# スクリプトの定義
EXTRACT_SCRIPT="$SRC_DIR/evaluate_linearKD_extract.py"
PYTHON_SCRIPT="$SRC_DIR/evaluate_linearKD.py"
NOTIFY_SCRIPT="$SRC_DIR/send_notification.py"

# 前提チェック
if [ ! -f "$EXTRACT_SCRIPT" ]; then echo "Error: Script not found at $EXTRACT_SCRIPT"; exit 1; fi
if [ ! -f "$PYTHON_SCRIPT" ]; then echo "Error: Script not found at $PYTHON_SCRIPT"; exit 1; fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ========================================================
# 通知処理用トラップ (Discord送信用)
# ========================================================
# 最後に処理したディレクトリを保持する変数
CURRENT_OUTPUT_DIR="$PROJECT_ROOT/evaluation_results_kd"

notification_exit() {
    EXIT_CODE=$?
    if [ -f "$NOTIFY_SCRIPT" ]; then
        if [ $EXIT_CODE -eq 0 ]; then
            echo ">>> Pipeline finished successfully. Sending notification..."
            python3 "$NOTIFY_SCRIPT" "success" "$CURRENT_OUTPUT_DIR"
        else
            echo ">>> Pipeline failed with exit code $EXIT_CODE. Sending notification..."
            python3 "$NOTIFY_SCRIPT" "failed" "$CURRENT_OUTPUT_DIR"
        fi
    else
        echo "Warning: Notification script not found at $NOTIFY_SCRIPT"
    fi
}
# スクリプトの終了時(正常・異常問わず)に notification_exit を実行
trap notification_exit EXIT

# ========================================================
# 2. Execution Function (実行用関数)
# ========================================================

run_dataset_cycle() {
    local DS_NAME="$1"       
    local DS_TYPE="$2"       
    local HF_DS="$3"         
    local HF_SPLIT="$4"      

    local DATASET_DIR="$PROJECT_ROOT/makeData/dataset_clean_${DS_NAME}"

    if [ ! -d "$DATASET_DIR" ]; then
        echo "--------------------------------------------------------"
        echo " [SKIP] Directory not found: $DATASET_DIR"
        echo "--------------------------------------------------------"
        return
    fi

    echo "********************************************************"
    echo " STARTING DATASET: $DS_NAME ($DS_TYPE)"
    echo "********************************************************"

    # [Step 1] 教師モデルのLinear学習とロジット抽出
    echo " [Step 1] Extracting Teacher Logits (Linear Probe)..."
    python3 "$EXTRACT_SCRIPT" "$DATASET_DIR"

    # データセットごとの引数分岐
    local EXTRA_ARGS=""
    if [ "$DS_TYPE" == "imagenet" ]; then
        EXTRA_ARGS="--imagenet_hf_dataset $HF_DS --imagenet_hf_test_split $HF_SPLIT"
    elif [ "$DS_TYPE" == "food101" ]; then
        EXTRA_ARGS="--food_hf_dataset $HF_DS --food_hf_test_split $HF_SPLIT"
    elif [ "$DS_TYPE" == "cub" ]; then
        EXTRA_ARGS="--cub_hf_dataset $HF_DS --cub_hf_test_split $HF_SPLIT"
    fi

    local LIST_SINGLE=$(echo $MIX_SET_SINGLE | sed 's/ /", "/g')
    local LIST_MULTI=$(echo $MIX_SET_MULTI | sed 's/ /", "/g')
    local LIST_ALL=$(echo $MIX_SET_ALL | sed 's/ /", "/g')
    local MIX_JSON_STR="{\"Mix_Single\": [\"$LIST_SINGLE\"], \"Mix_Multi\": [\"$LIST_MULTI\"], \"Mix_All\": [\"$LIST_ALL\"]}"

    # 生徒モデルを Linear Probing で評価するモードを指定
    local EVAL_MODE="linear_torch"

    local OUTPUT_DIR="$PROJECT_ROOT/evaluation_results_kd/${DS_NAME}/${TIMESTAMP}_${EVAL_MODE}_Combined_${AUG_STRATEGY}"
    mkdir -p "$OUTPUT_DIR"
    
    # 通知用にグローバル変数を更新
    CURRENT_OUTPUT_DIR="$OUTPUT_DIR"

    echo "--------------------------------------------------------"
    echo " [Step 2] Run KD Training (Linear Probing): $DS_NAME | Trials: $NUM_TRIALS"
    echo " Output: $OUTPUT_DIR"
    echo "--------------------------------------------------------"
    
    # [Step 2] 生徒モデルのKD学習
    python3 "$PYTHON_SCRIPT" "$DATASET_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --mode "$EVAL_MODE" \
        --real_counts $REAL_IMAGE_COUNTS \
        --max_real_test $REAL_TEST_COUNT \
        --syn_counts $SYN_COUNTS \
        --mix_json "$MIX_JSON_STR" \
        --num_trials $NUM_TRIALS \
        --augment \
        --aug_strategy "$AUG_STRATEGY" \
        --aug_expansion $AUG_EXPANSION \
        --no_tsne \
        --evaluators $SELECTED_EVALUATORS \
        --dataset_type "$DS_TYPE" \
        $EXTRA_ARGS

    echo "Finished KD cycle for $DS_NAME"
    echo ""
}

# ========================================================
# 3. Main Execution Sequence (実行順序)
# ========================================================

run_dataset_cycle "ImageNet1k_200" "imagenet" "imagenet-1k" "validation"
#run_dataset_cycle "food101" "food101" "ethz/food101" "validation"
#run_dataset_cycle "CUB" "cub" "Donghyun99/CUB-200-2011" "test"

echo "========================================================"
echo " All KD Evaluation Cycles Completed."
echo "========================================================"