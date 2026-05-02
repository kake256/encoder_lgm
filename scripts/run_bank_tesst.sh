#!/bin/bash
# run_bank_1phase_single_gpu.sh
# 1x GPU向け: [Phase 1: GPU0で準備] -> [Phase 2: GPU0ですべて処理]
set -e

# ========================================================
# 設定エリア
# ========================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$PROJECT_DIR/src"
RESULTS_BASE_DIR="$PROJECT_DIR/classification_results"

EXPERIMENT_ROOT_DIR="$RESULTS_BASE_DIR/gene_experiment_graduation"
mkdir -p "$EXPERIMENT_ROOT_DIR"
export PYTHONPATH="$SRC_DIR:$PROJECT_DIR:$PYTHONPATH"

# 並列数 (GPU 1枚あたりのプロセス数)
# GPUが1枚になったため、VRAM容量と相談して調整してください。
# 4プロセス同時実行の設定です。
MAX_JOBS_PER_GPU=4

# データセット設定
DATASET_TYPE="imagenet-1k"
DATASET_SPLIT="train"
DATA_ROOT="$PROJECT_DIR/hf_datasets_cache"
mkdir -p "$DATA_ROOT"

# ターゲットクラス
# クラス総数を設定 (ImageNet-100なら100, 1kなら1000)
NUM_CLASSES=1000

# ターゲット設定
# 例: "950" (オレンジ)
TARGET_CLASSES="483" #"437 888 948 950"
# 全クラス回す場合は以下をコメントイン
# TARGET_CLASSES=$(seq -s " " 0 $((NUM_CLASSES - 1)))

# パラメータ
USE_REAL=20      # 200
REAL_AUG=30
SYN_AUG=32
ITERATIONS=1250  # 2500

SEED=42
NUM_GENERATIONS=1 

MIN_SCALE=0.08
MAX_SCALE=1.0
NOISE_STD=0.2
NOISE_PROB=0.5
WEIGHT_TV=0.00025
PYRAMID_START_RES=16
PYRAMID_GROW_INTERVAL=200 # 400

# 文字攻撃設定
OVERLAY_TEXT=""
TEXT_COLOR="red"
FONT_SCALE=0.15
DISABLE_OCR_FLAG=""
if [ -z "$OVERLAY_TEXT" ]; then DISABLE_OCR_FLAG="--disable_ocr"; fi

# モデル設定
ENCODER_NAMES=(
  "facebook/dino-vitb16"
  "facebook/dinov2-base"
  "openai/clip-vit-base-patch16"
  "google/siglip-base-patch16-224"
)
PROJ_DIM_LIST=("0 0 0 0")

EXPERIMENTS=(
  # --- 単体モデル (基準用) ---
  "Only_V1:1.0,0.0,0.0,0.0"
  "Only_V2:0.0,1.0,0.0,0.0"
  "Only_CLIP:0.0,0.0,1.0,0.0"
  #"Only_SigLIP:0.0,0.0,0.0,1.0"

  # --- マルチモデル (Family) ---
  #"Hybrid_V1_V2:1.0,1.0,0.0,0.0"
  #"Hybrid_CLIP_SigLIP:0.0,0.0,1.0,1.0"
  
  # --- マルチモデル ---
  "Hybrid_V1_CLIP:1.0,0.0,1.0,0.0"
  #"Hybrid_V1_SigLIP:1.0,0.0,0.0,1.0"
  "Hybrid_V2_CLIP:0.0,1.0,1.0,0.0"
  #"Hybrid_V2_SigLIP:0.0,1.0,0.0,1.0"
)

# ========================================================
# Phase 1: Feature Bank キャッシュの作成 (準備)
# ========================================================
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
BASE_OUTPUT_DIR="$EXPERIMENT_ROOT_DIR/$TIMESTAMP"
CACHE_DIR="$EXPERIMENT_ROOT_DIR/feature_cache"
mkdir -p "$BASE_OUTPUT_DIR"
mkdir -p "$CACHE_DIR"

echo "=========================================="
echo "Phase 1: Checking Feature Banks (Using GPU 0)"
echo "=========================================="

CMD_PHASE1="cd \"$SRC_DIR\" && CUDA_VISIBLE_DEVICES=0 python3 main_feature_bank.py \
  --encoder_names ${ENCODER_NAMES[@]} \
  --projection_dims ${PROJ_DIM_LIST[0]} \
  --experiments ${EXPERIMENTS[@]} \
  --target_classes $TARGET_CLASSES \
  --dataset_type \"$DATASET_TYPE\" \
  --dataset_split \"$DATASET_SPLIT\" \
  --data_root \"$DATA_ROOT\" \
  --output_dir \"$BASE_OUTPUT_DIR\" \
  --cache_dir \"$CACHE_DIR\" \
  --use_real $USE_REAL \
  --real_aug $REAL_AUG \
  --min_scale $MIN_SCALE \
  --max_scale $MAX_SCALE \
  --noise_std $NOISE_STD \
  --noise_prob $NOISE_PROB \
  --overlay_text \"$OVERLAY_TEXT\" \
  --text_color \"$TEXT_COLOR\" \
  --font_scale $FONT_SCALE \
  --seed $SEED \
  --only_bank_creation \
  --log_file \"$BASE_OUTPUT_DIR/phase1_cache_log.txt\""

eval "$CMD_PHASE1"
echo "Phase 1 Completed."

# ========================================================
# Phase 2: Single-GPU 並列画像生成 (本番)
# ========================================================
echo "=========================================="
echo "Phase 2: High-Performance Generation (Single GPU)"
echo "  - GPU 0: Monitor + All Background Jobs"
echo "=========================================="

export ENCODER_NAMES_STR="${ENCODER_NAMES[*]}"
export EXPERIMENTS_STR="${EXPERIMENTS[*]}"
export SRC_DIR DATASET_TYPE DATASET_SPLIT DATA_ROOT CACHE_DIR ITERATIONS NUM_GENERATIONS
export USE_REAL REAL_AUG SYN_AUG WEIGHT_TV OVERLAY_TEXT TEXT_COLOR FONT_SCALE SEED
export PYRAMID_START_RES PYRAMID_GROW_INTERVAL MIN_SCALE MAX_SCALE NOISE_STD NOISE_PROB DISABLE_OCR_FLAG

# --- 実行関数 ---
run_generation_worker() {
    local GPU_ID=$1
    local CLS_ID=$2
    local MODE=$3 # "silent" or "verbose"
    
    local LOG_FILE="$BATCH_DIR/log_gen_class_${CLS_ID}.txt"
    
    if [ "$MODE" == "verbose" ]; then
        echo ">>> [GPU $GPU_ID] Monitoring Progress for Class $CLS_ID <<<"
        CUDA_VISIBLE_DEVICES=$GPU_ID python3 main_feature_bank.py \
          --encoder_names $ENCODER_NAMES_STR \
          --projection_dims $PROJ_DIMS \
          --experiments $EXPERIMENTS_STR \
          --target_classes $CLS_ID \
          --dataset_type "$DATASET_TYPE" \
          --dataset_split "$DATASET_SPLIT" \
          --data_root "$DATA_ROOT" \
          --output_dir "$BATCH_DIR" \
          --cache_dir "$CACHE_DIR" \
          --num_iterations $ITERATIONS \
          --num_generations $NUM_GENERATIONS \
          --use_real $USE_REAL \
          --real_aug $REAL_AUG \
          --syn_aug $SYN_AUG \
          --weight_tv $WEIGHT_TV \
          --overlay_text "$OVERLAY_TEXT" \
          --text_color "$TEXT_COLOR" \
          --font_scale $FONT_SCALE \
          --pyramid_start_res $PYRAMID_START_RES \
          --pyramid_grow_interval $PYRAMID_GROW_INTERVAL \
          --min_scale $MIN_SCALE \
          --max_scale $MAX_SCALE \
          --noise_std $NOISE_STD \
          --noise_prob $NOISE_PROB \
          --seed $SEED \
          $DISABLE_OCR_FLAG \
          --no_evaluation \
          --log_file "$LOG_FILE"
    else
        # Silent mode
        CUDA_VISIBLE_DEVICES=$GPU_ID python3 main_feature_bank.py \
          --encoder_names $ENCODER_NAMES_STR \
          --projection_dims $PROJ_DIMS \
          --experiments $EXPERIMENTS_STR \
          --target_classes $CLS_ID \
          --dataset_type "$DATASET_TYPE" \
          --dataset_split "$DATASET_SPLIT" \
          --data_root "$DATA_ROOT" \
          --output_dir "$BATCH_DIR" \
          --cache_dir "$CACHE_DIR" \
          --num_iterations $ITERATIONS \
          --num_generations $NUM_GENERATIONS \
          --use_real $USE_REAL \
          --real_aug $REAL_AUG \
          --syn_aug $SYN_AUG \
          --weight_tv $WEIGHT_TV \
          --overlay_text "$OVERLAY_TEXT" \
          --text_color "$TEXT_COLOR" \
          --font_scale $FONT_SCALE \
          --pyramid_start_res $PYRAMID_START_RES \
          --pyramid_grow_interval $PYRAMID_GROW_INTERVAL \
          --min_scale $MIN_SCALE \
          --max_scale $MAX_SCALE \
          --noise_std $NOISE_STD \
          --noise_prob $NOISE_PROB \
          --seed $SEED \
          $DISABLE_OCR_FLAG \
          --no_evaluation \
          --log_file "$LOG_FILE" > /dev/null 2>&1
    fi
}
export -f run_generation_worker

# --- クラスリストの分割ロジック (Single GPU用に変更) ---
CLASS_ARRAY=($TARGET_CLASSES)

# すべてのクラスを GPU 0 リストに入れる
GPU0_LIST=("${CLASS_ARRAY[@]}")

# GPU0用の先頭クラス（画面表示用）と残り
GPU0_FIRST=${GPU0_LIST[0]}
GPU0_REST=("${GPU0_LIST[@]:1}")

echo ">>> Total Classes: ${#CLASS_ARRAY[@]}"
echo ">>> All classes assigned to GPU 0"

for PROJ_DIMS_VAL in "${PROJ_DIM_LIST[@]}"; do
  export PROJ_DIMS=$PROJ_DIMS_VAL
  export BATCH_DIR="$BASE_OUTPUT_DIR/gen_data"
  mkdir -p "$BATCH_DIR"
  
  PIDS=""

  # ----------------------------------------
  # GPU 0 のタスク実行 (Single GPU)
  # ----------------------------------------
  
  # 1. 残りのクラスをバックグラウンド実行 (MAX_JOBS_PER_GPU 並列)
  if [ ${#GPU0_REST[@]} -gt 0 ]; then
      echo "${GPU0_REST[*]}" | tr ' ' '\n' | xargs -P $MAX_JOBS_PER_GPU -I {} bash -c "run_generation_worker 0 {} silent" &
      PIDS="$PIDS $!"
  fi

  # 2. 先頭クラスをフォアグラウンド実行（進捗バー表示）
  # もしクラスが1つもなければ実行しない
  if [ -n "$GPU0_FIRST" ]; then
      run_generation_worker 0 $GPU0_FIRST verbose
  fi
  
  # ----------------------------------------
  # 完了待ち
  # ----------------------------------------
  if [ -n "$PIDS" ]; then
      echo ">>> Monitor finished. Waiting for background jobs to finish..."
      wait $PIDS
  fi

done

echo "All phases completed. Results at: $BASE_OUTPUT_DIR"