#!/bin/bash
# run_1gpu_imagenet100_prototype_layer.sh
# 1x GPU向け: ImageNet-100 を同一GPU上の複数ワーカーで並列最適化 (Prototype alignment)
set -e

# ========================================================
# 設定エリア
# ========================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SRC_DIR="$PROJECT_DIR/src"
RESULTS_BASE_DIR="$PROJECT_DIR/classification_results"

# 出力先
EXPERIMENT_ROOT_DIR="$RESULTS_BASE_DIR/gene_experiment_imagenet100_PROTOTYPE"
mkdir -p "$EXPERIMENT_ROOT_DIR"

export PYTHONPATH="$SRC_DIR:$PROJECT_DIR:$PYTHONPATH"

# ========================================================
# データセット設定: ImageNet-100
# ========================================================
DATASET_TYPE="clane9/imagenet-100"
DATASET_SPLIT="train"
DATA_ROOT="$PROJECT_DIR/hf_datasets_cache"
mkdir -p "$DATA_ROOT"

# 全100クラス
TARGET_CLASSES="$(seq -s " " 0 99)"

# ========================================================
# パラメータ
# ========================================================
USE_REAL=1500
SYN_AUG=32
ITERATIONS=2500 #3600 #2500
SEED=42
NUM_GENERATIONS=1
MIN_SCALE=0.08
MAX_SCALE=1.0
NOISE_STD=0.2
NOISE_PROB=0.5
WEIGHT_TV=0.00025
WEIGHT_PROTO=0.1
PYRAMID_START_RES=1 #16
PYRAMID_GROW_INTERVAL=200 #400 #400

# 1GPU 内で同時に最適化するクラス数
BATCH_OPT_SIZE=10

# 1GPU 環境向け並列設定
GPU_ID_FOR_ALL_WORKERS=0
NUM_PARALLEL_WORKERS=1  # 旧4GPU並列の約半分

# ========================================================
# 実画像サンプリング設定
#   single: 1枚DA
#   multi : 複数枚DA
# ========================================================
REAL_SAMPLING_MODE="multi"
REAL_IMAGES_PER_STEP=1
REAL_AUG_PER_IMAGE=32

#REAL_SAMPLING_MODE="single"
#REAL_IMAGES_PER_STEP=1
#REAL_AUG_PER_IMAGE=32


# ========================================================
# 特徴抽出設定
#   -1: 最終層
#   -2, -3 ...: 中間層
# ========================================================
FEATURE_LAYER=-1
FEATURE_SOURCE="cls"

# ========================================================
# 文字攻撃設定
# ========================================================
OVERLAY_TEXT=""
TEXT_COLOR="red"
FONT_SCALE=0.15
DISABLE_OCR_FLAG="--disable_ocr"

# ========================================================
# モデル設定
# ========================================================
# 変更前（4モデル構成）:
# ENCODER_NAMES=(
#   "facebook/dino-vitb16"
#   "facebook/dinov2-base"
#   "openai/clip-vit-base-patch16"
#   "google/siglip-base-patch16-224"
# )
#
# PROJ_DIM_LIST=("0 0 0 0")
#
# EXPERIMENTS=(
#   "Only_V1:1.0,0.0,0.0,0.0"
#   "Only_V2:0.0,1.0,0.0,0.0"
#   "Only_CLIP:0.0,0.0,1.0,0.0"
#   #"Only_SigLIP:0.0,0.0,0.0,1.0"
#   "Hybrid_V1_CLIP:1.0,0.0,1.0,0.0"
#   "Hybrid_V2_CLIP:0.0,1.0,1.0,0.0"
#   #"Hybrid_V2_SigLIP:0.0,1.0,0.0,1.0"
# )
#
# DINOv3シングルモデル（動作確認用）
# ========================================================
# モデル設定
# ========================================================
ENCODER_NAMES=(
  "facebook/dino-vitb16"
  "facebook/dinov2-base"
  "openai/clip-vit-base-patch16"
  "timm/vit_small_patch16_dinov3"
)

PROJ_DIM_LIST=("0 0 0 0")

EXPERIMENTS=(
  # ---- Single ----
  "Only_V1:1.0,0.0,0.0,0.0"
  "Only_V2:0.0,1.0,0.0,0.0"
  "Only_CLIP:0.0,0.0,1.0,0.0"
  "Only_V3:0.0,0.0,0.0,1.0"

  # ---- Pair ----
  #"V1_V2:1.0,1.0,0.0,0.0"
  #"V1_CLIP:1.0,0.0,1.0,0.0"
  #"V1_V3:1.0,0.0,0.0,1.0"
  "V2_CLIP:0.0,1.0,1.0,0.0"
  "V2_V3:0.0,1.0,0.0,1.0"
  #"V3_CLIP:0.0,0.0,1.0,1.0"

  # ---- Triple ----
  #"V1_V2_CLIP:1.0,1.0,1.0,0.0"
  #"V1_V2_V3:1.0,1.0,0.0,1.0"
  #"V1_CLIP_V3:1.0,0.0,1.0,1.0"
  #"V2_CLIP_V3:0.0,1.0,1.0,1.0"

  # ---- All ----
  #"ALL_4:1.0,1.0,1.0,1.0"
)

# ========================================================
# 出力ディレクトリ設定
# ========================================================
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
BASE_OUTPUT_DIR="$EXPERIMENT_ROOT_DIR/$TIMESTAMP"
mkdir -p "$BASE_OUTPUT_DIR"

# ========================================================
# 通知処理用トラップ
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
# 環境変数を export
# ========================================================
export SRC_DIR PROJECT_DIR
export ENCODER_NAMES_STR="${ENCODER_NAMES[*]}"
export EXPERIMENTS_STR="${EXPERIMENTS[*]}"
export DATASET_TYPE DATASET_SPLIT DATA_ROOT ITERATIONS NUM_GENERATIONS
export USE_REAL SYN_AUG WEIGHT_TV OVERLAY_TEXT TEXT_COLOR FONT_SCALE SEED
export WEIGHT_PROTO
export PYRAMID_START_RES PYRAMID_GROW_INTERVAL MIN_SCALE MAX_SCALE NOISE_STD NOISE_PROB DISABLE_OCR_FLAG
export BATCH_OPT_SIZE
export REAL_SAMPLING_MODE REAL_IMAGES_PER_STEP REAL_AUG_PER_IMAGE
export FEATURE_LAYER FEATURE_SOURCE

run_generation_worker() {
    local WORKER_ID=$1
    local CLS_LIST=$2
    local PROJ_DIMS="$3"
    local MODE=$4
    local LOG_FILE="$BATCH_DIR/log_gen_worker_${WORKER_ID}.txt"

    local BASE_CMD="CUDA_VISIBLE_DEVICES=$GPU_ID_FOR_ALL_WORKERS python3 \"$SRC_DIR/select_layer/main_feature_prototype_layer.py\" \
      --encoder_names $ENCODER_NAMES_STR \
      --projection_dims $PROJ_DIMS \
      --experiments $EXPERIMENTS_STR \
      --target_classes $CLS_LIST \
      --batch_opt_size $BATCH_OPT_SIZE \
      --dataset_type \"$DATASET_TYPE\" \
      --dataset_split \"$DATASET_SPLIT\" \
      --data_root \"$DATA_ROOT\" \
      --output_dir \"$BATCH_DIR\" \
      --num_iterations $ITERATIONS \
      --num_generations $NUM_GENERATIONS \
      --use_real $USE_REAL \
      --syn_aug $SYN_AUG \
      --weight_proto $WEIGHT_PROTO \
      --weight_tv $WEIGHT_TV \
      --overlay_text \"$OVERLAY_TEXT\" \
      --text_color \"$TEXT_COLOR\" \
      --font_scale $FONT_SCALE \
      --pyramid_start_res $PYRAMID_START_RES \
      --pyramid_grow_interval $PYRAMID_GROW_INTERVAL \
      --min_scale $MIN_SCALE \
      --max_scale $MAX_SCALE \
      --noise_std $NOISE_STD \
      --noise_prob $NOISE_PROB \
      --seed $SEED \
      --real_sampling_mode $REAL_SAMPLING_MODE \
      --real_images_per_step $REAL_IMAGES_PER_STEP \
      --real_aug_per_image $REAL_AUG_PER_IMAGE \
      --feature_layer $FEATURE_LAYER \
      --feature_source $FEATURE_SOURCE \
      $DISABLE_OCR_FLAG \
      --no_evaluation \
      --log_file \"$LOG_FILE\""

    if [ "$MODE" == "verbose" ]; then
        eval "$BASE_CMD"
    else
        eval "$BASE_CMD > /dev/null 2>&1"
    fi
}
export -f run_generation_worker

# ========================================================
# 全100クラスを 1GPU 上の複数ワーカーへ順番に割り振る
# ========================================================
CLASS_ARRAY=($TARGET_CLASSES)
declare -a WORKER_CLASS_LISTS

for ((w=0; w<NUM_PARALLEL_WORKERS; w++)); do
  WORKER_CLASS_LISTS[$w]=""
done

for i in "${!CLASS_ARRAY[@]}"; do
  WORKER_ID=$((i % NUM_PARALLEL_WORKERS))
  WORKER_CLASS_LISTS[$WORKER_ID]="${WORKER_CLASS_LISTS[$WORKER_ID]} ${CLASS_ARRAY[$i]}"
done

echo ">>> Assigning Tasks to 1 GPU with ${NUM_PARALLEL_WORKERS} parallel workers"
echo " GPU ID: $GPU_ID_FOR_ALL_WORKERS"
for ((w=0; w<NUM_PARALLEL_WORKERS; w++)); do
  COUNT=$(echo "${WORKER_CLASS_LISTS[$w]}" | wc -w)
  FIRST=$(echo "${WORKER_CLASS_LISTS[$w]}" | awk '{print $1}')
  echo " Worker $w: ${COUNT} classes, Starts with ${FIRST:-None}"
done
echo " feature_layer=$FEATURE_LAYER, feature_source=$FEATURE_SOURCE, weight_proto=$WEIGHT_PROTO"
echo " real_sampling_mode=$REAL_SAMPLING_MODE"

for PROJ_DIMS_VAL in "${PROJ_DIM_LIST[@]}"; do
  export BATCH_DIR="$BASE_OUTPUT_DIR/gen_data"
  mkdir -p "$BATCH_DIR"

  PIDS=""

  for ((w=NUM_PARALLEL_WORKERS-1; w>=0; w--)); do
      CLS_LIST="${WORKER_CLASS_LISTS[$w]}"
      if [ -z "${CLS_LIST// /}" ]; then
          continue
      fi

      MODE="silent"
      if [ $w -eq 0 ]; then
          MODE="verbose"
      fi

      run_generation_worker "$w" "$CLS_LIST" "$PROJ_DIMS_VAL" "$MODE" &
      PIDS="$PIDS $!"
  done

  if [ -n "$PIDS" ]; then
      wait $PIDS
  fi
done

echo "All phases completed. Results at: $BASE_OUTPUT_DIR"
