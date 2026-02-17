#!/bin/bash
# run_1gpu_imagenet100_multi_test.sh
# 動作テスト用: 1GPU (GPU 0) での全実験直列実行
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$PROJECT_DIR/src"
RESULTS_BASE_DIR="$PROJECT_DIR/classification_results"

EXPERIMENT_ROOT_DIR="$RESULTS_BASE_DIR/gene_experiment_simple_imagenet100_TEST_1GPU"
mkdir -p "$EXPERIMENT_ROOT_DIR"
export PYTHONPATH="$SRC_DIR:$PROJECT_DIR:$PYTHONPATH"

# ----------------------------
# パラメータ設定
# ----------------------------
DATASET_TYPE="clane9/imagenet-100"
DATASET_SPLIT="train"
DATA_ROOT="$PROJECT_DIR/hf_datasets_cache"
mkdir -p "$DATA_ROOT"

# ========================================================
# ImageNet-100の全クラス(0~99)を対象にし、バッチサイズを合わせる
# ========================================================
TARGET_CLASSES=$(seq -s ' ' 0 99)
BATCH_OPT_SIZE=100
USE_REAL=1500
ITERATIONS=5000

SYN_AUG=2
SEED=42 
NUM_GENERATIONS=1 
MIN_SCALE=0.08
MAX_SCALE=1.0
NOISE_STD=0.2
NOISE_PROB=0.5
WEIGHT_TV=0.00025
PYRAMID_START_RES=1
PYRAMID_GROW_INTERVAL=200

OVERLAY_TEXT=""
TEXT_COLOR="red"
FONT_SCALE=0.15

ENCODER_NAMES=(
  "facebook/dino-vitb16"
  "facebook/dinov2-base"
  "openai/clip-vit-base-patch16"
  "google/siglip-base-patch16-224"
)
PROJ_DIM_LIST=("0 0 0 0")

# ========================================================
# 全ての実験を定義 (1つのGPUで順番に実行されます)
# ========================================================
EXPERIMENTS=(
  "Only_V1:1.0,0.0,0.0,0.0"
  "Hybrid_V1_CLIP:1.0,0.0,1.0,0.0"
  "Only_V2:0.0,1.0,0.0,0.0"
  "Hybrid_V1_SigLIP:1.0,0.0,0.0,1.0"
  "Only_CLIP:0.0,0.0,1.0,0.0"
  "Hybrid_V2_CLIP:0.0,1.0,1.0,0.0"
  "Only_SigLIP:0.0,0.0,0.0,1.0"
  "Hybrid_V2_SigLIP:0.0,1.0,0.0,1.0"
)

TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
BASE_OUTPUT_DIR="$EXPERIMENT_ROOT_DIR/$TIMESTAMP"
mkdir -p "$BASE_OUTPUT_DIR"
export BATCH_DIR="$BASE_OUTPUT_DIR/gen_data"
mkdir -p "$BATCH_DIR"

# ========================================================
# 通知処理用トラップ (メール送信用)
# ========================================================
notification_exit() {
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        python3 "$SRC_DIR/send_notification.py" "success" "$BASE_OUTPUT_DIR"
    else
        python3 "$SRC_DIR/send_notification.py" "failed" "$BASE_OUTPUT_DIR"
    fi
}
trap notification_exit EXIT

# ========================================================
# 1. データセットの事前ダウンロード
# ========================================================
echo ">>> Pre-downloading and caching dataset (Single Process)..."
python3 -c "
from datasets import load_dataset
print('Verifying dataset cache...')
load_dataset('$DATASET_TYPE', split='$DATASET_SPLIT', cache_dir='$DATA_ROOT', trust_remote_code=True)
print('Dataset cached successfully.')
"

echo ">>> Proceeding to Optimization on GPU 0..."

# ========================================================
# 2. GPU 0 で全実験を直列実行
# ========================================================
LOG_FILE="$BATCH_DIR/log_gpu_0.txt"

# EXPERIMENTS配列をスペース区切りの文字列に変換
ALL_EXP_LIST="${EXPERIMENTS[*]}"

echo "Starting GPU 0 with all experiments: ${ALL_EXP_LIST}"
CUDA_VISIBLE_DEVICES=0 python3 "$SRC_DIR/main_feature_bank_opt_batch.py" \
  --encoder_names ${ENCODER_NAMES[@]} --projection_dims ${PROJ_DIM_LIST[0]} \
  --experiments $ALL_EXP_LIST \
  --target_classes $TARGET_CLASSES \
  --dataset_type "$DATASET_TYPE" --dataset_split "$DATASET_SPLIT" --data_root "$DATA_ROOT" \
  --output_dir "$BATCH_DIR" --num_iterations $ITERATIONS --num_generations $NUM_GENERATIONS \
  --use_real $USE_REAL --syn_aug $SYN_AUG --weight_tv $WEIGHT_TV \
  --overlay_text "$OVERLAY_TEXT" --text_color "$TEXT_COLOR" --font_scale $FONT_SCALE \
  --pyramid_start_res $PYRAMID_START_RES --pyramid_grow_interval $PYRAMID_GROW_INTERVAL \
  --min_scale $MIN_SCALE --max_scale $MAX_SCALE --noise_std $NOISE_STD --noise_prob $NOISE_PROB \
  --seed $SEED --batch_opt_size $BATCH_OPT_SIZE \
  --log_file "$LOG_FILE"

echo "All phases completed on GPU 0!"